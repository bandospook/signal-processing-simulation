from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def psd_db(sig: np.ndarray, fs: float,
           nfft: int = 16384) -> tuple[np.ndarray, np.ndarray]:
    """Hann-windowed FFT magnitude spectrum in dB.

    Caps at nfft points (centre segment) so large wideband signals don't
    produce multi-megapixel figures that are slow to save.
    """
    n = min(len(sig), nfft)
    start = (len(sig) - n) // 2          # centre segment avoids transients
    s = sig[start : start + n]
    w = np.hanning(n)
    S = np.fft.fftshift(np.fft.fft(s * w))
    f = np.fft.fftshift(np.fft.fftfreq(n, 1.0 / fs))
    return f, 20 * np.log10(np.abs(S) / np.sum(w) + 1e-12)


def plot_nl_tables(am_am_cfg: dict, am_pm_cfg: dict,
                   input_backoff_db: float = 0.0,
                   save_path: str | None = None) -> None:
    """
    Plot AM-AM and AM-PM curves from config lookup tables.
    Marks the peak-signal operating point derived from input_backoff_db.
    """
    am_in  = np.asarray(am_am_cfg["input"])
    am_out = np.asarray(am_am_cfg["output"])
    pm_in  = np.asarray(am_pm_cfg["input"])
    pm_deg = np.asarray(am_pm_cfg["phase_deg"])

    drive      = 10 ** (-input_backoff_db / 20)
    op_am_out  = float(np.interp(drive, am_in, am_out))
    op_pm_deg  = float(np.interp(drive, pm_in, pm_deg))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(
        f"Amplifier Transfer Characteristic  "
        f"(Input Backoff = {input_backoff_db:.1f} dB)")

    # AM-AM
    ax1.plot(am_in, am_out, color="tab:blue", marker="o", ms=4, label="AM-AM")
    ax1.plot([0, 1], [0, 1], "--", color="gray", lw=0.8, label="Linear (ideal)")
    ax1.axvline(drive, color="tab:red", lw=1.0, ls="--")
    ax1.axhline(op_am_out, color="tab:red", lw=1.0, ls="--")
    ax1.plot(drive, op_am_out, "o", color="tab:red", ms=8,
             label=f"Operating point\n(A_in={drive:.3f}, A_out={op_am_out:.3f})")
    ax1.set_xlabel("Input Amplitude")
    ax1.set_ylabel("Output Amplitude")
    ax1.set_title("AM-AM")
    ax1.legend(fontsize=8)
    ax1.grid(True)

    # AM-PM
    ax2.plot(pm_in, pm_deg, color="tab:orange", marker="o", ms=4, label="AM-PM")
    ax2.axvline(drive, color="tab:red", lw=1.0, ls="--")
    ax2.axhline(op_pm_deg, color="tab:red", lw=1.0, ls="--")
    ax2.plot(drive, op_pm_deg, "o", color="tab:red", ms=8,
             label=f"Operating point\n(A_in={drive:.3f}, φ={op_pm_deg:.2f}°)")
    ax2.set_xlabel("Input Amplitude")
    ax2.set_ylabel("Phase Shift (degrees)")
    ax2.set_title("AM-PM")
    ax2.legend(fontsize=8)
    ax2.grid(True)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)


def plot_channel_response(sample_rate: float, signal_bw: float,
                           channel_cfg: dict, title: str = "",
                           save_path: str | None = None) -> None:
    """Plot amplitude and phase response of a channel impairment config."""
    N = 4096
    freqs = np.fft.fftshift(np.fft.fftfreq(N, 1.0 / sample_rate))

    half_bw = signal_bw / 2.0
    in_band = np.abs(freqs) <= half_bw
    f_norm  = np.where(in_band, freqs / half_bw, 0.0)

    ripple_db     = channel_cfg.get("ripple_db", 0.0)
    ripple_cycles = channel_cfg.get("ripple_cycles", 1.0)
    max_phase_deg = channel_cfg.get("max_phase_dev_deg", 0.0)
    poly_order    = channel_cfg.get("phase_poly_order", 2)

    r = (10 ** (ripple_db / 20) - 1) / (10 ** (ripple_db / 20) + 1)
    ampl_lin   = np.where(in_band, 1.0 + r * np.cos(np.pi * ripple_cycles * f_norm), np.nan)
    ampl_db    = 20 * np.log10(np.abs(ampl_lin) + 1e-12)
    phase_plot = np.where(in_band,
                          max_phase_deg * np.abs(f_norm) ** poly_order
                          * np.sign(f_norm) ** (poly_order % 2),
                          np.nan)

    f_mhz = freqs / 1e6
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(f"Channel Impairment Response{' — ' + title if title else ''}")

    ax1.plot(f_mhz, ampl_db, color="tab:blue", lw=0.9)
    ax1.set_xlabel("Frequency (MHz)")
    ax1.set_ylabel("Amplitude (dB)")
    ax1.set_title(f"Passband Ripple  (±{ripple_db/2:.2f} dB pk)")
    ax1.grid(True)

    ax2.plot(f_mhz, phase_plot, color="tab:orange", lw=0.9)
    ax2.set_xlabel("Frequency (MHz)")
    ax2.set_ylabel("Phase (degrees)")
    ax2.set_title(f"Phase Nonlinearity  (±{max_phase_deg:.1f}° pk, order {poly_order})")
    ax2.grid(True)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)


def plot_phase_noise_response(pn_cfg: dict, native_rate: float,
                              title: str = "",
                              save_path: str | None = None) -> None:
    """Plot a phase-noise mask and the cumulative RMS jitter it implies.

    Left panel: L(f) (dBc/Hz) as a function of offset frequency on a log
    x-axis.  The dashed curve is the log-log interpolation that
    ``sim.phase_noise`` uses; the dots are the user's anchor points.

    Right panel: cumulative RMS phase jitter from 0 to f, in degrees.
    σ_φ(f) = sqrt(2 · ∫_0^f 10^(L/10) df).  The asymptote at f = fs/2 is
    the total in-band RMS phase noise the carrier actually sees.
    """
    from .phase_noise import interp_dbc_mask    # local import → no cycle

    offsets = np.asarray(pn_cfg["offset_hz"], dtype=float)
    dbc_anchor = np.asarray(pn_cfg["dbc_per_hz"], dtype=float)
    if offsets.size == 0:
        return

    # Sample the mask densely across [min(offset)/3, fs/2] for a smooth curve.
    f_max = max(float(native_rate) / 2.0, float(offsets.max()))
    f_min = float(offsets.min()) / 3.0
    f_grid = np.logspace(np.log10(f_min), np.log10(f_max), 400)
    dbc_grid = interp_dbc_mask(f_grid, offsets, dbc_anchor)

    # Cumulative variance via trapezoid on linear S_phi over the same grid.
    s_phi = 2.0 * 10.0 ** (dbc_grid / 10.0)              # rad²/Hz
    cum_var = np.zeros_like(f_grid)
    cum_var[1:] = np.cumsum(0.5 * (s_phi[1:] + s_phi[:-1]) * np.diff(f_grid))
    rms_deg = np.sqrt(cum_var) * (180.0 / np.pi)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(f"Phase Noise{' — ' + title if title else ''}")

    ax1.semilogx(f_grid, dbc_grid, color="tab:blue", lw=1.0,
                 label="interpolated mask")
    ax1.semilogx(offsets, dbc_anchor, "o", color="tab:red", ms=5,
                 label="anchors")
    ax1.set_xlabel("Offset (Hz)")
    ax1.set_ylabel("L(f) (dBc/Hz)")
    ax1.set_title("Phase noise mask")
    ax1.grid(True, which="both", alpha=0.4)
    ax1.legend(fontsize=8)

    ax2.semilogx(f_grid, rms_deg, color="tab:orange", lw=1.0)
    ax2.axvline(native_rate / 2.0, color="gray", ls=":", lw=0.8,
                label=f"fs/2 = {native_rate / 2.0:.2g} Hz")
    ax2.set_xlabel("Upper integration limit (Hz)")
    ax2.set_ylabel("Cumulative RMS phase (°)")
    ax2.set_title(f"Total RMS at fs/2: {float(np.interp(native_rate/2.0, f_grid, rms_deg)):.3f}°")
    ax2.grid(True, which="both", alpha=0.4)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)


def print_metrics_table(carriers: list[dict]) -> None:
    """Print per-carrier CNR / CIR / CNIR / EVM / BER to stdout."""
    def _db(v: float) -> str:
        return f"{v:>8.1f}" if np.isfinite(v) else "     inf"

    header = (f"{'Carrier':<10}  {'CNR (dB)':>8}  {'CIR (dB)':>8}  "
              f"{'CNIR (dB)':>9}  {'EVM (%)':>7}  {'BER':>11}")
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for cr in carriers:
        ber = cr.get("ber")
        ber_str = f"{ber:.3e}" if (ber is not None and ber > 0) else "         0"
        print(f"{cr['name']:<10}  {_db(cr['cnr_db'])}  {_db(cr['cir_db'])}  "
              f"{_db(cr['cnir_db']):>9}  {cr['evm_rms']:>7.2f}  {ber_str:>11}")
    print(sep)


def _enabled_carrier_names(sweep_results: list[dict]) -> list[str]:
    """Carriers that have at least one finite metric value across the sweep grid."""
    all_names = [cr["name"] for cr in sweep_results[0]["carriers"]]
    def has_data(name):
        for r in sweep_results:
            for cr in r["carriers"]:
                if cr["name"] == name and np.isfinite(cr.get("cnir_db", float("nan"))):
                    return True
        return False
    return [n for n in all_names if has_data(n)]


def _ibo_pts(sweep_results: list[dict], noise: float,
             carrier_name: str) -> tuple[list[float], list[dict]]:
    """Return (ibo_xs, per-carrier-dicts) for one noise slice, sorted by IBO."""
    pts = sorted(
        [r for r in sweep_results if r["noise_density_dbfs"] == noise],
        key=lambda r: r["ibo_db"])
    ibos = [p["ibo_db"] for p in pts]
    cdata = [next(cr for cr in p["carriers"] if cr["name"] == carrier_name)
             for p in pts]
    return ibos, cdata


def _cnr_pts(sweep_results: list[dict], ibo: float,
             carrier_name: str) -> tuple[list[float], list[dict]]:
    """Return (cnr_xs, per-carrier-dicts) for one IBO slice, sorted by CNR."""
    pts = [r for r in sweep_results if r["ibo_db"] == ibo]
    cdata = [next(cr for cr in p["carriers"] if cr["name"] == carrier_name)
             for p in pts]
    cnrs_x = [cd["cnr_db"] if np.isfinite(cd["cnr_db"]) else np.nan
              for cd in cdata]
    order = sorted(range(len(pts)),
                   key=lambda k: (np.inf if np.isnan(cnrs_x[k]) else cnrs_x[k]))
    return [cnrs_x[k] for k in order], [cdata[k] for k in order]


def _ber_or_nan(cd: dict) -> float:
    """Replace zero / None BERs with NaN so semilogy skips them cleanly."""
    b = cd["ber"]
    return b if (b is not None and b > 0) else float("nan")


def _ber_zero_arrows_vs_ibo(ax, sweep_results: list[dict], carrier_name: str,
                            noise_vals: list[float], noise_colours) -> None:
    """Annotate zero-BER points on a BER-vs-IBO axis with downward arrows."""
    for ni, noise in enumerate(noise_vals):
        for p in [r for r in sweep_results if r["noise_density_dbfs"] == noise]:
            cd = next(cr for cr in p["carriers"] if cr["name"] == carrier_name)
            if cd["ber"] == 0 or cd["ber"] is None:
                ax.annotate("", xy=(p["ibo_db"], ax.get_ylim()[0]),
                            xytext=(p["ibo_db"], ax.get_ylim()[0] * 3),
                            arrowprops=dict(arrowstyle="->",
                                            color=noise_colours[ni], lw=1.2))


def _ber_zero_arrows_vs_cnr(ax, sweep_results: list[dict], carrier_name: str,
                            ibo_vals: list[float], ibo_colours) -> None:
    """Annotate zero-BER points on a BER-vs-CNR axis (skip non-finite CNR)."""
    for ii, ibo in enumerate(ibo_vals):
        for p in [r for r in sweep_results if r["ibo_db"] == ibo]:
            cd = next(cr for cr in p["carriers"] if cr["name"] == carrier_name)
            if not np.isfinite(cd["cnr_db"]):
                continue
            if cd["ber"] == 0 or cd["ber"] is None:
                ax.annotate("", xy=(cd["cnr_db"], ax.get_ylim()[0]),
                            xytext=(cd["cnr_db"], ax.get_ylim()[0] * 3),
                            arrowprops=dict(arrowstyle="->",
                                            color=ibo_colours[ii], lw=1.2))


def _finalize_ibo_axis(ax, ibo_vals: list[float]) -> None:
    ax.set_xlabel("IBO (dB)")
    if min(ibo_vals) < max(ibo_vals):
        ax.set_xlim(min(ibo_vals), max(ibo_vals))
    ax.grid(True, which="both", alpha=0.4)
    ax.legend(fontsize=7)


def _finalize_cnr_axis(ax) -> None:
    ax.set_xlabel("CNR (dB)")
    ax.grid(True, which="both", alpha=0.4)
    ax.legend(fontsize=7)


def _panel_ber_vs_ibo(ax, sweep_results: list[dict], carrier_name: str,
                      noise_vals: list[float], ibo_vals: list[float],
                      noise_colours) -> None:
    for ni, noise in enumerate(noise_vals):
        ibos, cdata = _ibo_pts(sweep_results, noise, carrier_name)
        ax.semilogy(ibos, [_ber_or_nan(cd) for cd in cdata],
                    label=f"{noise:.0f} dBFS/Hz",
                    marker="o", ms=4, color=noise_colours[ni])
    ax.set_title("BER vs IBO")
    ax.set_ylabel("BER")
    _finalize_ibo_axis(ax, ibo_vals)
    _ber_zero_arrows_vs_ibo(ax, sweep_results, carrier_name,
                            noise_vals, noise_colours)


def _panel_evm_vs_ibo(ax, sweep_results: list[dict], carrier_name: str,
                      noise_vals: list[float], ibo_vals: list[float],
                      noise_colours) -> None:
    for ni, noise in enumerate(noise_vals):
        ibos, cdata = _ibo_pts(sweep_results, noise, carrier_name)
        ax.plot(ibos, [cd["evm_rms"] for cd in cdata],
                label=f"{noise:.0f} dBFS/Hz",
                marker="o", ms=4, color=noise_colours[ni])
    ax.set_title("EVM (%) vs IBO")
    ax.set_ylabel("EVM (%)")
    _finalize_ibo_axis(ax, ibo_vals)


def _panel_db_vs_ibo(ax, sweep_results: list[dict], carrier_name: str,
                     noise_vals: list[float], ibo_vals: list[float],
                     noise_colours) -> None:
    for ni, noise in enumerate(noise_vals):
        ibos, cdata = _ibo_pts(sweep_results, noise, carrier_name)
        col   = noise_colours[ni]
        label = f"{noise:.0f} dBFS/Hz"
        cnrs  = [cd["cnr_db"]  if np.isfinite(cd["cnr_db"])  else np.nan for cd in cdata]
        cirs  = [cd["cir_db"]  if np.isfinite(cd["cir_db"])  else np.nan for cd in cdata]
        cnirs = [cd["cnir_db"] if np.isfinite(cd["cnir_db"]) else np.nan for cd in cdata]
        ax.plot(ibos, cnrs,  ls="-",  label=f"CNR  {label}",
                marker="o", ms=4, color=col)
        ax.plot(ibos, cirs,  ls="--", label=f"CIR  {label}",
                marker="s", ms=4, color=col)
        ax.plot(ibos, cnirs, ls=":",  label=f"CNIR {label}",
                marker="^", ms=4, color=col)
    ax.set_title("CNR / CIR / CNIR (dB) vs IBO")
    ax.set_ylabel("dB")
    _finalize_ibo_axis(ax, ibo_vals)


def _panel_ber_vs_cnr(ax, sweep_results: list[dict], carrier_name: str,
                      noise_vals: list[float], ibo_vals: list[float],
                      ibo_colours) -> None:
    _ = noise_vals   # not used; kept for uniform panel signatures
    for ii, ibo in enumerate(ibo_vals):
        cnr_s, cdata = _cnr_pts(sweep_results, ibo, carrier_name)
        ax.semilogy(cnr_s, [_ber_or_nan(cd) for cd in cdata],
                    label=f"IBO {ibo:.1f} dB",
                    marker="o", ms=4, color=ibo_colours[ii])
    ax.set_title("BER vs CNR")
    ax.set_ylabel("BER")
    _finalize_cnr_axis(ax)
    _ber_zero_arrows_vs_cnr(ax, sweep_results, carrier_name,
                            ibo_vals, ibo_colours)


def _panel_evm_vs_cnr(ax, sweep_results: list[dict], carrier_name: str,
                      noise_vals: list[float], ibo_vals: list[float],
                      ibo_colours) -> None:
    _ = noise_vals
    for ii, ibo in enumerate(ibo_vals):
        cnr_s, cdata = _cnr_pts(sweep_results, ibo, carrier_name)
        ax.plot(cnr_s, [cd["evm_rms"] for cd in cdata],
                label=f"IBO {ibo:.1f} dB",
                marker="o", ms=4, color=ibo_colours[ii])
    ax.set_title("EVM (%) vs CNR")
    ax.set_ylabel("EVM (%)")
    _finalize_cnr_axis(ax)


def _panel_db_vs_cnr(ax, sweep_results: list[dict], carrier_name: str,
                     noise_vals: list[float], ibo_vals: list[float],
                     ibo_colours) -> None:
    # CIR is fixed per IBO; CNR-vs-CNR is trivially diagonal, so plot CIR
    # and CNIR against CNR only.
    _ = noise_vals
    for ii, ibo in enumerate(ibo_vals):
        cnr_s, cdata = _cnr_pts(sweep_results, ibo, carrier_name)
        col   = ibo_colours[ii]
        label = f"IBO {ibo:.1f} dB"
        cirs  = [cd["cir_db"]  if np.isfinite(cd["cir_db"])  else np.nan for cd in cdata]
        cnirs = [cd["cnir_db"] if np.isfinite(cd["cnir_db"]) else np.nan for cd in cdata]
        ax.plot(cnr_s, cirs,  ls="--", label=f"CIR  {label}",
                marker="s", ms=4, color=col)
        ax.plot(cnr_s, cnirs, ls=":",  label=f"CNIR {label}",
                marker="^", ms=4, color=col)
    ax.set_title("CIR / CNIR (dB) vs CNR")
    ax.set_ylabel("dB")
    _finalize_cnr_axis(ax)


# Order is the same as the 2×3 layout: row 1 vs IBO, row 2 vs CNR.
_DETECTOR_PANELS = (
    ("ber_vs_ibo", _panel_ber_vs_ibo),
    ("evm_vs_ibo", _panel_evm_vs_ibo),
    ("db_vs_ibo",  _panel_db_vs_ibo),
    ("ber_vs_cnr", _panel_ber_vs_cnr),
    ("evm_vs_cnr", _panel_evm_vs_cnr),
    ("db_vs_cnr",  _panel_db_vs_cnr),
)


def plot_carrier_detector(sweep_results: list[dict],
                          carrier_name: str,
                          save_path: str | None = None) -> None:
    """
    Plot BER, EVM, and CNR/CIR/CNIR for a single carrier across the sweep.

    Produces a combined 2×3 figure at ``save_path``:
        Row 1 — x-axis = IBO (dB); one line per noise level.
        Row 2 — x-axis = CNR (dB); one line per IBO (CNR varies via the noise axis).
                CNR is reported in the symbol-rate (matched-filter) bandwidth,
                so for BPSK it equals Eb/N0 directly.

    When ``save_path`` is provided, also writes each panel as a standalone
    PNG alongside the combined figure, named ``<stem>_<panel>.png`` (panels:
    ``ber_vs_ibo``, ``evm_vs_ibo``, ``db_vs_ibo``, ``ber_vs_cnr``,
    ``evm_vs_cnr``, ``db_vs_cnr``).

    Returns silently if the named carrier has no finite metrics in the sweep
    (e.g. sweep_demod=False).
    """
    if carrier_name not in _enabled_carrier_names(sweep_results):
        return
    ibo_vals   = sorted(set(r["ibo_db"] for r in sweep_results))
    noise_vals = sorted(set(r["noise_density_dbfs"] for r in sweep_results))
    n_noise    = len(noise_vals)
    n_ibo      = len(ibo_vals)

    noise_colours = plt.colormaps["viridis"](np.linspace(0.15, 0.85, n_noise))
    ibo_colours   = plt.colormaps["plasma"](np.linspace(0.15, 0.85, n_ibo))

    # Bound the per-panel render: the first three take noise_colours, the
    # second three take ibo_colours.
    panel_args = (
        (noise_colours,) * 3 + (ibo_colours,) * 3
    )

    # ── Combined 2×3 figure ──────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f"Detector Sweep: {carrier_name} — vs IBO (top) and vs CNR (bottom)")
    flat_axes = list(axes[0]) + list(axes[1])
    for ax, (_, panel_fn), extra in zip(flat_axes, _DETECTOR_PANELS, panel_args):
        panel_fn(ax, sweep_results, carrier_name,
                 noise_vals, ibo_vals, extra)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)

        # ── Standalone per-panel figures (same data, fresh figure each) ──
        base = Path(save_path)
        for (key, panel_fn), extra in zip(_DETECTOR_PANELS, panel_args):
            f2, ax2 = plt.subplots(figsize=(6, 4.5))
            panel_fn(ax2, sweep_results, carrier_name,
                     noise_vals, ibo_vals, extra)
            f2.tight_layout()
            f2.savefig(base.with_stem(f"{base.stem}_{key}"), dpi=150)
            plt.close(f2)


def plot_wideband_results(results: dict,
                           sample_rate: float,
                           save_path: str | None = None) -> None:
    """
    Composite wideband PSD: pre-NL, post-NL, and (if noise present) post-noise.
    Accepts the chunk-pipeline return format: results["psd_pre_nl"] etc. are
    (f_array, psd_db_array) tuples from the Welch accumulator.
    """
    carriers = results["carriers"]
    carrier_labels = "  |  ".join(
        f"{cr['name']}: {cr['symbol_rate']/1e6:.3g} MHz sym/s" for cr in carriers)

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.suptitle(
        f"Wideband NL Simulation — {sample_rate/1e9:.3g} GHz\n{carrier_labels}")

    f, p = results["psd_pre_nl"]
    ax.plot(f, p, lw=0.8, color="tab:blue", label="Pre-NL")
    f, p = results["psd_post_nl"]
    ax.plot(f, p, lw=0.8, color="tab:orange", alpha=0.85, label="Post-NL")
    if results.get("has_noise", False):
        f, p = results["psd_noisy"]
        ax.plot(f, p, lw=0.6, color="tab:green", alpha=0.7, label="Post-NL + noise")

    ax.set_title("Wideband PSD (Welch average)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("dB")
    ax.set_ylim(bottom=-100)
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)


def _fmt_metric(key: str, val) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    if key == "ber":
        return "0" if val == 0 else f"{val:.2e}"
    if key == "evm_rms":
        return f"{val:.2f}"
    return "∞" if not np.isfinite(val) else f"{val:.1f}"


def write_report(
    rows: list[dict],
    save_path: str | None,
    append: bool = False,
) -> None:
    """
    Write a single flat Markdown table of per-(carrier, IBO, noise) results.

    Each entry in `rows` is a dict for one (carrier, ibo_db, noise_density_dbfs)
    measurement and contains (all fields optional except name):
        name, ibo_db, noise_density_dbfs
        ber                     — aggregated BER across iterations
        ber_upper_95            — rule-of-three upper bound when n_errors == 0
        ci_half_width           — Wilson half-width at the chosen confidence
        n_bits, n_errors        — cumulative counts across iterations
        iterations              — iterations actually run at this point
        converged               — bool: True if CI target met before iteration cap
        effective_ebn0_db, theory_ebn0_db, implementation_loss_db
        cnr_db, cir_db, cnir_db, evm_rms

    A BER of 0 with a known n_bits is rendered as "< x.xe-y" using the
    rule-of-three upper bound, not as "0".  Points that exited at the
    iteration cap are flagged with a "*" suffix on the iteration count.

    append=True opens the file for appending rather than overwriting.
    """
    if not rows or save_path is None:
        return

    from pathlib import Path

    def _f(v, fmt=".2f") -> str:
        if v is None:
            return "—"
        if isinstance(v, float) and (np.isnan(v) or not np.isfinite(v)):
            return "∞" if v > 0 else "—"
        return format(v, fmt)

    def _ber_cell(r: dict) -> str:
        ber = r.get("ber")
        if ber is None:
            return "—"
        if ber == 0:
            upper = r.get("ber_upper_95")
            return f"< {upper:.2e}" if upper is not None else "0"
        return f"{ber:.2e}"

    def _iters_cell(r: dict) -> str:
        n = r.get("iterations")
        if n is None:
            return "—"
        return f"{n}*" if r.get("converged") is False else f"{n}"

    L = []
    def ln(s=""): L.append(s)

    ln("## Results")
    ln()
    ln("Iteration counts marked `*` exited at the iteration cap without meeting "
       "the Wilson CI target. BER of `< x.xe-y` means zero errors observed; the "
       "value is the rule-of-three upper 95% bound.")
    ln()
    ln("| Carrier | IBO (dB) | Noise (dBFS/Hz) | Iters | n_bits | n_err "
       "| BER | CI ± | Eff Eb/N0 (dB) | Theory Eb/N0 (dB) | Impl Loss (dB) "
       "| CNR (dB) | CIR (dB) | CNIR (dB) | EVM (%) |")
    ln("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for r in rows:
        n_bits = r.get("n_bits")
        n_bits_s = f"{int(n_bits):,}" if n_bits is not None else "—"
        n_err = r.get("n_errors")
        n_err_s = f"{int(n_err):,}" if n_err is not None else "—"

        ln(
            f"| {r.get('name', '—')}"
            f" | {_f(r.get('ibo_db'), '.1f')}"
            f" | {_f(r.get('noise_density_dbfs'), '.1f')}"
            f" | {_iters_cell(r)}"
            f" | {n_bits_s}"
            f" | {n_err_s}"
            f" | {_ber_cell(r)}"
            f" | {_f(r.get('ci_half_width'), '.2e')}"
            f" | {_f(r.get('effective_ebn0_db'), '.2f')}"
            f" | {_f(r.get('theory_ebn0_db'), '.2f')}"
            f" | {_f(r.get('implementation_loss_db'), '.2f')}"
            f" | {_f(r.get('cnr_db'), '.1f')}"
            f" | {_f(r.get('cir_db'), '.1f')}"
            f" | {_f(r.get('cnir_db'), '.1f')}"
            f" | {_f(r.get('evm_rms'), '.2f')} |"
        )

    ln()

    text = "\n".join(L)
    p = Path(save_path)
    if append and p.exists():
        with p.open("a", encoding="utf-8") as f:
            f.write("\n" + text)
    else:
        p.write_text(text, encoding="utf-8")
