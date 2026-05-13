import numpy as np
import matplotlib.pyplot as plt


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


def plot_wideband_results(results: dict,
                           sample_rate: float,
                           native_rate_slow: float,
                           native_rate_fast: float,
                           symbol_rate_slow: float,
                           symbol_rate_fast: float,
                           save_path: str | None = None) -> None:
    """
    Three-panel figure:
      top          — wideband PSD pre-NL vs post-NL
      bottom-left  — slow carrier PSD pre-NL vs post-NL at native rate
      bottom-right — fast carrier PSD pre-NL vs post-NL at native rate
    """
    fig = plt.figure(figsize=(13, 7))
    fig.suptitle(
        f"Wideband NL Simulation\n"
        f"Slow: {symbol_rate_slow/1e6:.3g} MHz sym/s  |  "
        f"Fast: {symbol_rate_fast/1e6:.3g} MHz sym/s  |  "
        f"Wideband: {sample_rate/1e9:.3g} GHz")

    ax_wb = fig.add_subplot(2, 1, 1)
    f, p = psd_db(results['wideband'], sample_rate)
    ax_wb.plot(f, p, lw=0.8, color='tab:blue', label='Pre-NL')
    f, p = psd_db(results['wideband_nl'], sample_rate)
    ax_wb.plot(f, p, lw=0.8, color='tab:orange', alpha=0.85, label='Post-NL')
    ax_wb.set_title("Wideband PSD")
    ax_wb.set_xlabel("Frequency (Hz)")
    ax_wb.set_ylabel("dB")
    ax_wb.set_ylim(bottom=-100)
    ax_wb.legend()
    ax_wb.grid(True)

    ax_slow = fig.add_subplot(2, 2, 3)
    f, p = psd_db(results['slow_bb'], native_rate_slow)
    ax_slow.plot(f, p, lw=0.8, color='tab:blue', label='Pre-NL')
    f, p = psd_db(results['slow_nl'], native_rate_slow)
    ax_slow.plot(f, p, lw=0.8, color='tab:orange', alpha=0.85, label='Post-NL')
    ax_slow.set_title(f"Slow carrier  ({symbol_rate_slow/1e6:.3g} MHz sym/s)")
    ax_slow.set_xlabel("Frequency (Hz)")
    ax_slow.set_ylabel("dB")
    ax_slow.set_ylim(bottom=-100)
    ax_slow.legend()
    ax_slow.grid(True)

    ax_fast = fig.add_subplot(2, 2, 4)
    f, p = psd_db(results['fast_bb'], native_rate_fast)
    ax_fast.plot(f, p, lw=0.8, color='tab:blue', label='Pre-NL')
    f, p = psd_db(results['fast_nl'], native_rate_fast)
    ax_fast.plot(f, p, lw=0.8, color='tab:orange', alpha=0.85, label='Post-NL')
    ax_fast.set_title(f"Fast carrier  ({symbol_rate_fast/1e6:.3g} MHz sym/s)")
    ax_fast.set_xlabel("Frequency (Hz)")
    ax_fast.set_ylabel("dB")
    ax_fast.set_ylim(bottom=-100)
    ax_fast.legend()
    ax_fast.grid(True)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
