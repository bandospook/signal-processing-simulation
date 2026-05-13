import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def psd_db(sig: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Hann-windowed FFT magnitude spectrum in dB."""
    w = np.hanning(len(sig))
    S = np.fft.fftshift(np.fft.fft(sig * w))
    f = np.fft.fftshift(np.fft.fftfreq(len(sig), 1.0 / fs))
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


def plot_sweep_results(sweep_results: list[dict],
                       save_path: str | None = None) -> None:
    """
    Plot BER, EVM, and CNR/CIR/CNIR vs IBO for each carrier.

    Sweep results are a flat list of {ibo_db, noise_density_dbfs, carriers}.
    Noise density is used as a colour parameter; line style distinguishes
    CNR (solid), CIR (dashed), and CNIR (dotted) on the third panel.
    """
    ibo_vals   = sorted(set(r["ibo_db"] for r in sweep_results))
    noise_vals = sorted(set(r["noise_density_dbfs"] for r in sweep_results))
    carrier_names = [cr["name"] for cr in sweep_results[0]["carriers"]]
    n_carriers    = len(carrier_names)
    n_noise       = len(noise_vals)

    colours = cm.viridis(np.linspace(0.15, 0.85, n_noise))

    fig, axes = plt.subplots(n_carriers, 3,
                             figsize=(15, 4.5 * n_carriers),
                             squeeze=False)
    fig.suptitle("Parameter Sweep: Performance vs Input Backoff (IBO)")

    for row, cname in enumerate(carrier_names):
        ax_ber, ax_evm, ax_db = axes[row]
        ax_ber.set_title(f"{cname}  —  BER")
        ax_evm.set_title(f"{cname}  —  EVM (%)")
        ax_db.set_title(f"{cname}  —  CNR / CIR / CNIR (dB)")

        for ni, noise in enumerate(noise_vals):
            pts = sorted(
                [r for r in sweep_results if r["noise_density_dbfs"] == noise],
                key=lambda r: r["ibo_db"])
            ibos = [p["ibo_db"] for p in pts]
            cdata = [[cr for cr in p["carriers"] if cr["name"] == cname][0]
                     for p in pts]

            # BER — use NaN for zero so semilogy skips those points cleanly
            bers = [cd["ber"] if (cd["ber"] is not None and cd["ber"] > 0)
                    else np.nan for cd in cdata]

            evms  = [cd["evm_rms"] for cd in cdata]
            cnrs  = [cd["cnr_db"]  if np.isfinite(cd["cnr_db"])  else np.nan for cd in cdata]
            cirs  = [cd["cir_db"]  if np.isfinite(cd["cir_db"])  else np.nan for cd in cdata]
            cnirs = [cd["cnir_db"] if np.isfinite(cd["cnir_db"]) else np.nan for cd in cdata]

            col   = colours[ni]
            label = f"{noise:.0f} dBFS/Hz"
            kw    = dict(marker="o", ms=4, color=col)

            ax_ber.semilogy(ibos, bers, label=label, **kw)
            ax_evm.plot(ibos, evms, label=label, **kw)
            ax_db.plot(ibos, cnrs,  ls="-",  label=f"CNR  {label}", **kw)
            ax_db.plot(ibos, cirs,  ls="--", label=f"CIR  {label}",
                       marker="s", ms=4, color=col)
            ax_db.plot(ibos, cnirs, ls=":",  label=f"CNIR {label}",
                       marker="^", ms=4, color=col)

        ax_ber.set_ylabel("BER")
        ax_evm.set_ylabel("EVM (%)")
        ax_db.set_ylabel("dB")
        for ax in (ax_ber, ax_evm, ax_db):
            ax.set_xlabel("IBO (dB)")
            ax.grid(True, which="both", alpha=0.4)
            ax.legend(fontsize=7)

        # Annotate zero-BER points with a downward arrow at the plot bottom
        for ni, noise in enumerate(noise_vals):
            pts = sorted(
                [r for r in sweep_results if r["noise_density_dbfs"] == noise],
                key=lambda r: r["ibo_db"])
            for p in pts:
                cdata_pt = next(cr for cr in p["carriers"] if cr["name"] == cname)
                if cdata_pt["ber"] == 0 or cdata_pt["ber"] is None:
                    ax_ber.annotate("", xy=(p["ibo_db"], ax_ber.get_ylim()[0]),
                                    xytext=(p["ibo_db"], ax_ber.get_ylim()[0] * 3),
                                    arrowprops=dict(arrowstyle="->",
                                                    color=colours[ni], lw=1.2))

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)


def plot_wideband_results(results: dict,
                           sample_rate: float,
                           save_path: str | None = None) -> None:
    """
    Two-row figure:
      top    — wideband PSD: pre-NL, post-NL, and (if present) post-noise
      bottom — one panel per carrier: baseband PSD pre-NL vs post-NL
    """
    from matplotlib.gridspec import GridSpec

    carriers = results["carriers"]
    n_carriers = len(carriers)
    fig_w = max(13, 5 * n_carriers)
    fig = plt.figure(figsize=(fig_w, 7))

    carrier_labels = "  |  ".join(
        f"{cr['name']}: {cr['symbol_rate']/1e6:.3g} MHz sym/s" for cr in carriers)
    fig.suptitle(
        f"Wideband NL Simulation — {sample_rate/1e9:.3g} GHz\n{carrier_labels}")

    gs = GridSpec(2, n_carriers, figure=fig, hspace=0.45, wspace=0.35)

    ax_wb = fig.add_subplot(gs[0, :])
    f, p = psd_db(results["wideband"], sample_rate)
    ax_wb.plot(f, p, lw=0.8, color="tab:blue", label="Pre-NL")
    f, p = psd_db(results["wideband_nl"], sample_rate)
    ax_wb.plot(f, p, lw=0.8, color="tab:orange", alpha=0.85, label="Post-NL")
    if results.get("wideband_noisy") is not results["wideband_nl"]:
        f, p = psd_db(results["wideband_noisy"], sample_rate)
        ax_wb.plot(f, p, lw=0.6, color="tab:green", alpha=0.7, label="Post-NL + noise")
    ax_wb.set_title("Wideband PSD")
    ax_wb.set_xlabel("Frequency (Hz)")
    ax_wb.set_ylabel("dB")
    ax_wb.set_ylim(bottom=-100)
    ax_wb.legend()
    ax_wb.grid(True)

    for col, cr in enumerate(carriers):
        ax = fig.add_subplot(gs[1, col])
        f, p = psd_db(cr["bb"], cr["native_rate"])
        ax.plot(f, p, lw=0.8, color="tab:blue", label="Pre-NL")
        f, p = psd_db(cr["nl"], cr["native_rate"])
        ax.plot(f, p, lw=0.8, color="tab:orange", alpha=0.85, label="Post-NL")
        ax.set_title(f"{cr['name']}  ({cr['symbol_rate']/1e6:.3g} MHz sym/s)")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("dB")
        ax.set_ylim(bottom=-100)
        ax.legend(fontsize=8)
        ax.grid(True)

    if save_path is not None:
        fig.savefig(save_path)
