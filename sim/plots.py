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


def plot_carrier_detector(sweep_results: list[dict],
                          carrier_name: str,
                          save_path: str | None = None) -> None:
    """
    Plot BER, EVM, and CNR/CIR/CNIR for a single carrier across the sweep.

    Layout is a 2×3 grid:
        Row 1 — x-axis = IBO (dB); one line per noise level.
        Row 2 — x-axis = CNR (dB); one line per IBO (CNR varies via the noise axis).
                CNR is reported in the symbol-rate (matched-filter) bandwidth,
                so for BPSK it equals Eb/N0 directly.

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

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    (ax_ber_i, ax_evm_i, ax_db_i) = axes[0]
    (ax_ber_e, ax_evm_e, ax_db_e) = axes[1]
    fig.suptitle(f"Detector Sweep: {carrier_name} — vs IBO (top) and vs CNR (bottom)")
    ax_ber_i.set_title("BER vs IBO")
    ax_evm_i.set_title("EVM (%) vs IBO")
    ax_db_i.set_title("CNR / CIR / CNIR (dB) vs IBO")
    ax_ber_e.set_title("BER vs CNR")
    ax_evm_e.set_title("EVM (%) vs CNR")
    ax_db_e.set_title("CIR / CNIR (dB) vs CNR")

    # ── Row 1: x-axis = IBO, one line per noise level ────────────────────────
    for ni, noise in enumerate(noise_vals):
        pts = sorted(
            [r for r in sweep_results if r["noise_density_dbfs"] == noise],
            key=lambda r: r["ibo_db"])
        ibos = [p["ibo_db"] for p in pts]
        cdata = [[cr for cr in p["carriers"] if cr["name"] == carrier_name][0]
                 for p in pts]

        # BER — use NaN for zero so semilogy skips those points cleanly
        bers = [cd["ber"] if (cd["ber"] is not None and cd["ber"] > 0)
                else np.nan for cd in cdata]

        evms  = [cd["evm_rms"] for cd in cdata]
        cnrs  = [cd["cnr_db"]  if np.isfinite(cd["cnr_db"])  else np.nan for cd in cdata]
        cirs  = [cd["cir_db"]  if np.isfinite(cd["cir_db"])  else np.nan for cd in cdata]
        cnirs = [cd["cnir_db"] if np.isfinite(cd["cnir_db"]) else np.nan for cd in cdata]

        col   = noise_colours[ni]
        label = f"{noise:.0f} dBFS/Hz"
        kw    = dict(marker="o", ms=4, color=col)

        ax_ber_i.semilogy(ibos, bers, label=label, **kw)
        ax_evm_i.plot(ibos, evms, label=label, **kw)
        ax_db_i.plot(ibos, cnrs,  ls="-",  label=f"CNR  {label}", **kw)
        ax_db_i.plot(ibos, cirs,  ls="--", label=f"CIR  {label}",
                     marker="s", ms=4, color=col)
        ax_db_i.plot(ibos, cnirs, ls=":",  label=f"CNIR {label}",
                     marker="^", ms=4, color=col)

    # ── Row 2: x-axis = CNR (dB), one line per IBO ───────────────────────────
    # CNR varies with noise; CIR is fixed per IBO. The CNR panel drops CNR-vs-CNR
    # (trivially diagonal) and plots only CIR and CNIR against CNR.
    for ii, ibo in enumerate(ibo_vals):
        pts = [r for r in sweep_results if r["ibo_db"] == ibo]
        cdata = [[cr for cr in p["carriers"] if cr["name"] == carrier_name][0]
                 for p in pts]
        cnrs_x = [cd["cnr_db"] if np.isfinite(cd["cnr_db"]) else np.nan for cd in cdata]
        order = sorted(range(len(pts)), key=lambda k: (np.inf if np.isnan(cnrs_x[k])
                                                       else cnrs_x[k]))
        cnr_s   = [cnrs_x[k] for k in order]
        cdata_s = [cdata[k] for k in order]

        bers = [cd["ber"] if (cd["ber"] is not None and cd["ber"] > 0)
                else np.nan for cd in cdata_s]
        evms  = [cd["evm_rms"] for cd in cdata_s]
        cirs  = [cd["cir_db"]  if np.isfinite(cd["cir_db"])  else np.nan for cd in cdata_s]
        cnirs = [cd["cnir_db"] if np.isfinite(cd["cnir_db"]) else np.nan for cd in cdata_s]

        col   = ibo_colours[ii]
        label = f"IBO {ibo:.1f} dB"
        kw    = dict(marker="o", ms=4, color=col)

        ax_ber_e.semilogy(cnr_s, bers, label=label, **kw)
        ax_evm_e.plot(cnr_s, evms, label=label, **kw)
        ax_db_e.plot(cnr_s, cirs,  ls="--", label=f"CIR  {label}",
                     marker="s", ms=4, color=col)
        ax_db_e.plot(cnr_s, cnirs, ls=":",  label=f"CNIR {label}",
                     marker="^", ms=4, color=col)

    for ax in (ax_ber_i, ax_ber_e):
        ax.set_ylabel("BER")
    for ax in (ax_evm_i, ax_evm_e):
        ax.set_ylabel("EVM (%)")
    for ax in (ax_db_i, ax_db_e):
        ax.set_ylabel("dB")

    for ax in (ax_ber_i, ax_evm_i, ax_db_i):
        ax.set_xlabel("IBO (dB)")
        if min(ibo_vals) < max(ibo_vals):
            ax.set_xlim(min(ibo_vals), max(ibo_vals))
        ax.grid(True, which="both", alpha=0.4)
        ax.legend(fontsize=7)
    for ax in (ax_ber_e, ax_evm_e, ax_db_e):
        ax.set_xlabel("CNR (dB)")
        ax.grid(True, which="both", alpha=0.4)
        ax.legend(fontsize=7)

    # Annotate zero-BER points on the IBO panel with downward arrows
    for ni, noise in enumerate(noise_vals):
        pts = sorted(
            [r for r in sweep_results if r["noise_density_dbfs"] == noise],
            key=lambda r: r["ibo_db"])
        for p in pts:
            cdata_pt = next(cr for cr in p["carriers"] if cr["name"] == carrier_name)
            if cdata_pt["ber"] == 0 or cdata_pt["ber"] is None:
                ax_ber_i.annotate("", xy=(p["ibo_db"], ax_ber_i.get_ylim()[0]),
                                  xytext=(p["ibo_db"], ax_ber_i.get_ylim()[0] * 3),
                                  arrowprops=dict(arrowstyle="->",
                                                  color=noise_colours[ni], lw=1.2))

    # Same on the CNR panel (annotate at each zero-BER point's CNR)
    for ii, ibo in enumerate(ibo_vals):
        pts = [r for r in sweep_results if r["ibo_db"] == ibo]
        for p in pts:
            cdata_pt = next(cr for cr in p["carriers"] if cr["name"] == carrier_name)
            if not np.isfinite(cdata_pt["cnr_db"]):
                continue
            if cdata_pt["ber"] == 0 or cdata_pt["ber"] is None:
                cnr = cdata_pt["cnr_db"]
                ax_ber_e.annotate("", xy=(cnr, ax_ber_e.get_ylim()[0]),
                                  xytext=(cnr, ax_ber_e.get_ylim()[0] * 3),
                                  arrowprops=dict(arrowstyle="->",
                                                  color=ibo_colours[ii], lw=1.2))

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)


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
