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
from sim.simulation import wideband_bpsk_simulation
from sim.theory import ebn0_for_ber
from sim.plots import (plot_wideband_results, plot_nl_tables, plot_channel_response,
                       print_metrics_table, plot_sweep_results, write_sweep_report,
                       write_detector_results)
from sim.sweep import parameter_sweep
from sim.targeter import seek_all_carriers

_ProgressCB = Callable[[float, str], None] | None


def main(config_path: str = "simulation.toml",
         progress_callback: _ProgressCB = None) -> None:
    """Load config, run wideband simulation, and save results."""
    plt.close("all")

    def _prog(frac: float, msg: str) -> None:
        pct = min(100, int(round(frac * 100.0)))
        print(f"[{pct:3d}%] {msg}", flush=True)
        if progress_callback is not None:
            progress_callback(frac, msg)

    _prog(0.00, "Loading configuration...")
    cfg = load_config(config_path)

    active_carriers = [c for c in cfg["carrier"] if c.get("enabled", True)]
    wb  = cfg["wideband"]
    amp = cfg["amplifier"]
    ola = cfg["ola"]
    sim = cfg["simulation"]
    out = cfg["output"]

    out_dir = Path(out.get("output_dir", "."))
    out_dir.mkdir(exist_ok=True)

    def out_path(name: str | None) -> str | None:
        return str(out_dir / name) if name else None

    # Only carriers with sweep_demod=True are downsampled and analyzed.
    # Others contribute to the composite NL environment but skip the costly demod step.
    demod_names = {c["name"] for c in active_carriers if c.get("sweep_demod", False)}

    # ── Progress fractions ────────────────────────────────────────────────────
    # Assumption: each sweep point costs ~1 wideband simulation.
    # Budget from SIM_START to the seeker boundary (or near-end if no seeker).
    sweep_cfg   = cfg.get("sweep", {})
    ibo_sweep   = sweep_cfg.get("ibo_db", [])
    noise_sweep = sweep_cfg.get("noise_density_dbfs", [])
    n_sweep     = len(ibo_sweep) * len(noise_sweep) if (ibo_sweep and noise_sweep) else 0

    has_seeker  = any(c.get("enabled", True) and c.get("sweep_demod", False)
                      and c.get("use_seeker", False) for c in active_carriers)

    _SIM_START    = 0.05
    _PLOT_FRAC    = 0.03   # fixed budget for saving plots
    _SEEKER_START = 0.25   # seeker section always begins here when present
    _budget_end   = _SEEKER_START if has_seeker else 0.94
    _unit         = (_budget_end - _SIM_START - _PLOT_FRAC) / (1 + n_sweep)
    _P_sim_done   = _SIM_START + _unit
    _P_plots_done = _P_sim_done + _PLOT_FRAC

    def _chunk_print(msg: str) -> None:
        print(f"        {msg}", flush=True)

    _prog(0.05, f"Running wideband simulation ({len(active_carriers)} carriers, "
          f"{len(demod_names)} demodulated)...")
    results = wideband_bpsk_simulation(
        carriers           = active_carriers,
        sample_rate        = wb["sample_rate"],
        am_am_cfg          = amp["am_am"],
        am_pm_cfg          = amp["am_pm"],
        input_backoff_db   = amp["input_backoff_db"],
        noise_density_dbfs = wb.get("noise_density_dbfs"),
        ola_filter_span    = ola["filter_span"],
        ola_block_size     = ola["block_size"],
        seed               = sim["seed"],
        demod_carriers     = demod_names if demod_names else None,
        chunk_print        = _chunk_print,
    )

    _prog(_P_sim_done, "Wideband simulation complete.")
    print_metrics_table(results["carriers"])

    _prog(_P_sim_done + _PLOT_FRAC * 0.4, "Saving wideband PSD plot...")
    plot_wideband_results(results, sample_rate=wb["sample_rate"],
                          save_path=out_path(out.get("wideband")))

    plot_nl_tables(amp["am_am"], amp["am_pm"],
                   input_backoff_db=amp["input_backoff_db"],
                   save_path=out_path(out.get("nl_tables")))

    for carr in active_carriers:
        ch_cfg = carr.get("channel")
        if ch_cfg and ch_cfg.get("enabled", True):
            native_rate = carr["sps"] * carr["symbol_rate"]
            signal_bw   = (1 + carr["rolloff"]) * carr["symbol_rate"]
            plot_channel_response(
                native_rate, signal_bw, ch_cfg,
                title=f"{carr['name']}  ({carr['symbol_rate']/1e6:.3g} MHz sym/s)",
                save_path=out_path(ch_cfg.get("plot")),
            )

    _prog(_P_plots_done, "Plots saved.")

    if ibo_sweep and noise_sweep:
        n_pts = len(ibo_sweep) * len(noise_sweep)
        _prog(_P_plots_done, f"Running parameter sweep: {len(ibo_sweep)} IBO × "
              f"{len(noise_sweep)} noise = {n_pts} points...")

        def _sweep_pt_cb(done: int, total: int) -> None:
            _prog(_P_plots_done + _unit * done,
                  f"Sweep: {done}/{total} points complete")

        sweep_results = parameter_sweep(
            carriers                  = active_carriers,
            sample_rate               = wb["sample_rate"],
            am_am_cfg                 = amp["am_am"],
            am_pm_cfg                 = amp["am_pm"],
            ibo_db_values             = ibo_sweep,
            noise_density_dbfs_values = noise_sweep,
            ola_filter_span           = ola["filter_span"],
            ola_block_size            = ola["block_size"],
            seed                      = sim["seed"],
            chunk_print               = _chunk_print,
            point_cb                  = _sweep_pt_cb,
        )
        plot_sweep_results(sweep_results, save_path=out_path(out.get("sweep")))
        write_sweep_report(sweep_results, cfg=cfg,
                           save_path=out_path(out.get("sweep_table")))
        _prog(_P_plots_done + _unit * n_sweep, "Sweep complete.")

    # ── Fixed-noise demod carriers ──────────────────────────────────────────
    # Carriers with sweep_demod=True and use_seeker=False were already
    # demodulated in the main wideband run.  Extract their stats and compute
    # effective Eb/N0 from the CNIR measurement.
    noise_dbfs = wb.get("noise_density_dbfs")
    fixed_demod = [c for c in active_carriers
                   if c.get("sweep_demod", False) and not c.get("use_seeker", False)]
    fixed_results: dict[str, dict] = {}
    for carr in fixed_demod:
        cr = next((r for r in results["carriers"] if r["name"] == carr["name"]), None)
        if cr is None:
            continue
        mod = carr.get("modulation", "BPSK").upper()
        bps = bits_per_symbol(mod)
        sps = int(carr.get("sps", 4))
        cnir_db = cr["cnir_db"]
        eff_ebn0 = cnir_db + 10.0 * math.log10(sps / bps)
        ber_val  = cr.get("ber")
        theory   = ebn0_for_ber(mod, ber_val) if (ber_val is not None and ber_val > 0) else None
        impl_loss = (eff_ebn0 - theory) if theory is not None else None
        fixed_results[carr["name"]] = dict(
            mode="fixed",
            noise_density_dbfs=noise_dbfs,
            ber=ber_val,
            ber_ci_lo=None,
            ber_ci_hi=None,
            effective_ebn0_db=eff_ebn0,
            theory_ebn0_db=theory,
            implementation_loss_db=impl_loss,
            cnr_db=cr["cnr_db"],
            cir_db=cr["cir_db"],
            cnir_db=cnir_db,
            evm_rms=cr.get("evm_rms"),
            n_bits_total=None,
            n_iter=None,
        )

    # ── BER seeker for seekable carriers ────────────────────────────────────
    seekable = [c for c in active_carriers
                if c.get("enabled", True)
                and c.get("sweep_demod", False)
                and c.get("use_seeker", False)]
    seeker_results: dict[str, dict] = {}
    if seekable:
        _prog(_SEEKER_START, f"Starting BER seeker for {len(seekable)} carrier(s)...")

        def _seeker_cb(frac: float, msg: str) -> None:
            _prog(_SEEKER_START + frac * (0.99 - _SEEKER_START), msg)

        raw = seek_all_carriers(
            carriers          = active_carriers,
            sample_rate       = wb["sample_rate"],
            am_am_cfg         = amp["am_am"],
            am_pm_cfg         = amp["am_pm"],
            input_backoff_db  = amp["input_backoff_db"],
            ola_filter_span   = ola["filter_span"],
            ola_block_size    = ola["block_size"],
            seed              = sim["seed"],
            progress_callback = _seeker_cb,
            chunk_print       = _chunk_print,
        )
        for name, r in raw.items():
            seeker_results[name] = dict(r, mode="seeker")

    # ── Write detector results ───────────────────────────────────────────────
    all_detector: dict[str, dict] = {**fixed_results, **seeker_results}
    if all_detector:
        det_path = out_path(out.get("detector_results", "detector_results.md"))
        write_detector_results(all_detector, det_path)
        _prog(0.99, f"Detector results written → {det_path}")

    _prog(1.00, "Done.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "simulation.toml")
