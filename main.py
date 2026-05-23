"""Entry point for the wideband BPSK nonlinear amplifier simulation."""

import math
import sys
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim.config import load_config
from sim.modulation import bits_per_symbol
from sim.theory import ebn0_for_ber
from sim.plots import (plot_wideband_results, plot_nl_tables, plot_channel_response,
                       print_metrics_table, plot_carrier_detector, write_report)
from sim.sweep import parameter_sweep

_ProgressCB = Callable[[float, str], None] | None


def main(config_path: str = "simulation.toml",
         progress_callback: _ProgressCB = None) -> None:
    """Load config, run the parameter sweep, and save plots/reports.

    The sweep is the sole simulation driver: every (IBO, noise) point runs the
    full chunk pipeline (including per-carrier demod for `sweep_demod` carriers).
    The first sweep point's wideband result feeds the PSD plot; subsequent
    points contribute rows to the sweep table and detector_results.
    """
    plt.close("all")

    def _prog(frac: float, msg: str) -> None:
        pct = min(100, int(round(frac * 100.0)))
        print(f"[{pct:3d}%] {msg}", flush=True)
        if progress_callback is not None:
            progress_callback(frac, msg)

    _prog(0.00, "Loading configuration...")
    cfg = load_config(config_path)

    active_carriers = [c for c in cfg["carrier"] if c.get("enabled", True)]
    sweep_cfg = cfg["sweep"]
    amp = cfg["amplifier"]
    ola = cfg["ola"]
    sim = cfg["simulation"]
    out = cfg["output"]

    out_dir = Path(out.get("output_dir", "."))
    out_dir.mkdir(exist_ok=True)
    plots_enabled = out.get("plots", True)

    def plot_path(name: str) -> str | None:
        return str(out_dir / name) if plots_enabled else None

    def carrier_slug(name: str) -> str:
        return name.replace(" ", "_")

    sample_rate = sweep_cfg["sample_rate"]
    ibo_sweep   = sweep_cfg["ibo_db"]
    noise_sweep = sweep_cfg["noise_density_dbfs"]
    if not ibo_sweep or not noise_sweep:
        raise ValueError("[sweep].ibo_db and [sweep].noise_density_dbfs must each "
                         "contain at least one value.")

    max_block_size_samples = int(sim["max_block_size_samples"])
    target_ci_half_width   = float(sim["target_ci_half_width"])
    confidence             = float(sim["confidence"])
    min_errors             = int(sim["min_errors"])
    max_iterations         = int(sim["max_iterations"])

    n_sweep   = len(ibo_sweep) * len(noise_sweep)
    demod_names = {c["name"] for c in active_carriers if c.get("sweep_demod", False)}

    # ── Progress fractions ────────────────────────────────────────────────────
    _SWEEP_START = 0.05
    _PLOT_FRAC   = 0.05
    _unit        = (0.95 - _SWEEP_START - _PLOT_FRAC) / n_sweep
    _P_sweep_end = _SWEEP_START + _unit * n_sweep

    def _chunk_print(msg: str) -> None:
        print(f"        {msg}", flush=True)

    _prog(_SWEEP_START, f"Running parameter sweep: {len(ibo_sweep)} IBO x "
          f"{len(noise_sweep)} noise = {n_sweep} points "
          f"({len(active_carriers)} carriers, {len(demod_names)} demodulated)...")

    def _sweep_pt_cb(done: int, total: int) -> None:
        _prog(_SWEEP_START + _unit * done,
              f"Sweep: {done}/{total} points complete")

    first_sim, sweep_results = parameter_sweep(
        carriers                  = active_carriers,
        sample_rate               = sample_rate,
        am_am_cfg                 = amp["am_am"],
        am_pm_cfg                 = amp["am_pm"],
        ibo_db_values             = ibo_sweep,
        noise_density_dbfs_values = noise_sweep,
        max_block_size_samples    = max_block_size_samples,
        target_ci_half_width      = target_ci_half_width,
        confidence                = confidence,
        min_errors                = min_errors,
        max_iterations            = max_iterations,
        ola_filter_span           = ola["filter_span"],
        ola_block_size            = ola["block_size"],
        seed                      = sim["seed"],
        chunk_print               = _chunk_print,
        point_cb                  = _sweep_pt_cb,
    )

    _prog(_P_sweep_end, "Sweep complete.")
    print_metrics_table(first_sim["carriers"])

    # ── Plots / reports ───────────────────────────────────────────────────────
    if plots_enabled:
        _prog(_P_sweep_end + _PLOT_FRAC * 0.2, "Saving wideband PSD plot "
              f"(IBO={ibo_sweep[0]:.1f} dB, noise={noise_sweep[0]:.1f} dBFS/Hz)...")
        plot_wideband_results(first_sim, sample_rate=sample_rate,
                              save_path=plot_path("wideband.png"))

        plot_nl_tables(amp["am_am"], amp["am_pm"],
                       input_backoff_db=ibo_sweep[0],
                       save_path=plot_path("amplifier.png"))

        for carr in active_carriers:
            ch_cfg = carr.get("channel")
            if ch_cfg and ch_cfg.get("enabled", True):
                native_rate = carr["sps"] * carr["symbol_rate"]
                signal_bw   = (1 + carr["rolloff"]) * carr["symbol_rate"]
                plot_channel_response(
                    native_rate, signal_bw, ch_cfg,
                    title=f"{carr['name']}  ({carr['symbol_rate']/1e6:.3g} MHz sym/s)",
                    save_path=plot_path(f"{carrier_slug(carr['name'])}_channel.png"),
                )

        for cname in demod_names:
            plot_carrier_detector(
                sweep_results, cname,
                save_path=plot_path(f"{carrier_slug(cname)}_detector.png"),
            )

    # ── Per-point report rows ─────────────────────────────────────────────────
    # Each (ibo, noise) point becomes a row per demodulated carrier; effective
    # Eb/N0 and theoretical Eb/N0 are derived from the measured CNIR and BER.
    report_rows: list[dict] = []
    for r in sweep_results:
        ibo  = r["ibo_db"]
        nd   = r["noise_density_dbfs"]
        iters = r.get("iterations")
        conv  = r.get("converged")
        for carr in active_carriers:
            if not carr.get("sweep_demod", False):
                continue
            cr = next((c for c in r["carriers"] if c["name"] == carr["name"]), None)
            if cr is None:  # pragma: no cover  — sweep always returns every demod carrier
                continue
            mod = carr.get("modulation", "BPSK").upper()
            bps = bits_per_symbol(mod)
            cnir_db  = cr["cnir_db"]
            # CNIR is already in symbol-rate (matched-filter) bandwidth, so CNIR = Es/N0.
            # Eb/N0 = Es/N0 - 10*log10(bps).  For BPSK (bps=1) Eb/N0 == CNIR.
            eff_ebn0 = cnir_db - 10.0 * math.log10(bps)
            ber_val  = cr.get("ber")
            theory   = ebn0_for_ber(mod, ber_val) if (ber_val is not None and ber_val > 0) else None
            impl_loss = (eff_ebn0 - theory) if theory is not None else None
            report_rows.append(dict(
                name=carr["name"],
                ibo_db=ibo,
                noise_density_dbfs=nd,
                ber=ber_val,
                ber_upper_95=cr.get("ber_upper_95"),
                ci_half_width=cr.get("ci_half_width"),
                n_bits=cr.get("n_bits"),
                n_errors=cr.get("n_errors"),
                iterations=iters,
                converged=conv,
                effective_ebn0_db=eff_ebn0,
                theory_ebn0_db=theory,
                implementation_loss_db=impl_loss,
                cnr_db=cr["cnr_db"],
                cir_db=cr["cir_db"],
                cnir_db=cnir_db,
                evm_rms=cr.get("evm_rms"),
            ))

    if report_rows:
        report_path = str(out_dir / "report.md")
        write_report(report_rows, report_path)
        _prog(0.99, f"Report written -> {report_path}")

    _prog(1.00, "Done.")


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1] if len(sys.argv) > 1 else "simulation.toml")
