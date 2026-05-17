import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


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


def plot_sweep_results(sweep_results: list[dict],
                       save_path: str | None = None) -> None:
    """
    Plot BER, EVM, and CNR/CIR/CNIR vs IBO for each carrier.

    Sweep results are a flat list of {ibo_db, noise_density_dbfs, carriers}.
    Noise density is used as a colour parameter; line style distinguishes
    CNR (solid), CIR (dashed), and CNIR (dotted) on the third panel.
    """
    ibo_vals      = sorted(set(r["ibo_db"] for r in sweep_results))
    noise_vals    = sorted(set(r["noise_density_dbfs"] for r in sweep_results))
    carrier_names = _enabled_carrier_names(sweep_results)
    if not carrier_names:
        return
    n_carriers = len(carrier_names)
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
            ax.set_xlim(min(ibo_vals), max(ibo_vals))
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
    Composite wideband PSD: pre-NL, post-NL, and (if noise present) post-noise.
    Per-carrier downsampled views are omitted — only the composite spectrum is shown.
    """
    carriers = results["carriers"]
    carrier_labels = "  |  ".join(
        f"{cr['name']}: {cr['symbol_rate']/1e6:.3g} MHz sym/s" for cr in carriers)

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.suptitle(
        f"Wideband NL Simulation — {sample_rate/1e9:.3g} GHz\n{carrier_labels}")

    f, p = psd_db(results["wideband"], sample_rate)
    ax.plot(f, p, lw=0.8, color="tab:blue", label="Pre-NL")
    f, p = psd_db(results["wideband_nl"], sample_rate)
    ax.plot(f, p, lw=0.8, color="tab:orange", alpha=0.85, label="Post-NL")
    if results.get("wideband_noisy") is not results["wideband_nl"]:
        f, p = psd_db(results["wideband_noisy"], sample_rate)
        ax.plot(f, p, lw=0.6, color="tab:green", alpha=0.7, label="Post-NL + noise")

    ax.set_title("Wideband PSD")
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


def write_sweep_report(sweep_results: list[dict], cfg: dict,
                       save_path: str | None = None) -> None:
    """
    Write sweep results as a Markdown report containing a config summary,
    a per-carrier performance summary, and a full IBO × noise results table.
    Nothing is written if save_path is None.
    """
    if not sweep_results or save_path is None:
        return

    from pathlib import Path

    ibo_vals      = sorted(set(r["ibo_db"] for r in sweep_results))
    noise_vals    = sorted(set(r["noise_density_dbfs"] for r in sweep_results))
    carrier_names = _enabled_carrier_names(sweep_results)

    L = []
    def ln(s=""): L.append(s)

    # ── Config summary ────────────────────────────────────────────────────────
    ln("# Simulation Sweep Report")
    ln()
    ln("## Configuration")
    ln()

    wb  = cfg.get("wideband",   {})
    amp = cfg.get("amplifier",  {})
    ola = cfg.get("ola",        {})
    sim = cfg.get("simulation", {})

    ln("| Parameter | Value |")
    ln("|---|---|")
    ln(f"| Seed | {sim.get('seed', '—')} |")
    sr = wb.get("sample_rate", 0)
    ln(f"| Sample Rate | {sr / 1e9:.4g} GHz |")
    nd = wb.get("noise_density_dbfs")
    ln(f"| Noise Density | {nd:.1f} dBFS/Hz |" if nd is not None else "| Noise Density | disabled |")
    ln(f"| Input Backoff | {amp.get('input_backoff_db', '—')} dB |")
    ln(f"| OLA Filter Span | {ola.get('filter_span', '—')} |")
    ln(f"| OLA Block Size | {ola.get('block_size', '—')} |")
    ln()

    ln("### Carriers")
    ln()
    ln("| Name | Symbol Rate | SPS | Num Symbols | Sweep Demod |")
    ln("|---|---|---|---|---|")
    for c in cfg.get("carrier", []):
        sym_rate = c.get("symbol_rate", 0)
        sym_str  = f"{sym_rate / 1e6:.4g} MHz"
        demod    = "Yes" if c.get("sweep_demod", True) else "No"
        ln(f"| {c['name']} | {sym_str} | {c.get('sps', '—')} | {c.get('num_symbols', '—')} | {demod} |")
    ln()

    ln("### Sweep Grid")
    ln()
    ln(f"- **IBO values (dB):** {', '.join(f'{x:g}' for x in ibo_vals)}")
    ln(f"- **Noise values (dBFS/Hz):** {', '.join(f'{x:g}' for x in noise_vals)}")
    ln(f"- **Total points:** {len(ibo_vals) * len(noise_vals)}")
    ln()

    # ── Performance summary ───────────────────────────────────────────────────
    if carrier_names:
        ln("## Performance Summary")
        ln()
        SUMMARY_METRICS = [
            ("cnr_db",  "CNR",  "dB"),
            ("cir_db",  "CIR",  "dB"),
            ("cnir_db", "CNIR", "dB"),
            ("evm_rms", "EVM",  "%"),
            ("ber",     "BER",  ""),
        ]
        for cname in carrier_names:
            ln(f"### {cname}")
            ln()
            for key, label, unit in SUMMARY_METRICS:
                vals = []
                for r in sweep_results:
                    cr = next((c for c in r["carriers"] if c["name"] == cname), None)
                    if cr is None:
                        continue
                    v = cr.get(key)
                    if v is not None and isinstance(v, (int, float)):
                        if key == "ber" and v == 0:
                            vals.append(0.0)
                        elif np.isfinite(float(v)):
                            vals.append(float(v))
                if not vals:
                    ln(f"- **{label}:** no data")
                elif key == "ber":
                    lo = "0" if min(vals) == 0 else f"{min(vals):.2e}"
                    hi = "0" if max(vals) == 0 else f"{max(vals):.2e}"
                    ln(f"- **{label}:** {lo} – {hi}")
                else:
                    suffix = f" {unit}" if unit else ""
                    ln(f"- **{label}:** {min(vals):.2f} – {max(vals):.2f}{suffix}")
            ln()

    # ── Sweep results tables ──────────────────────────────────────────────────
    ln("## Sweep Results")
    ln()

    sorted_results = sorted(sweep_results,
                            key=lambda r: (r["ibo_db"], r["noise_density_dbfs"]))

    for cname in carrier_names:
        if len(carrier_names) > 1:
            ln(f"### {cname}")
            ln()

        ln("| IBO (dB) | Noise (dBFS/Hz) | BER | EVM (%) | CNR (dB) | CIR (dB) | CNIR (dB) |")
        ln("|---:|---:|---:|---:|---:|---:|---:|")

        for r in sorted_results:
            cr = next((c for c in r["carriers"] if c["name"] == cname), None)
            if cr is None:
                continue
            row = (
                f"| {r['ibo_db']:g}"
                f" | {r['noise_density_dbfs']:g}"
                f" | {_fmt_metric('ber',     cr.get('ber'))}"
                f" | {_fmt_metric('evm_rms', cr.get('evm_rms'))}"
                f" | {_fmt_metric('cnr_db',  cr.get('cnr_db'))}"
                f" | {_fmt_metric('cir_db',  cr.get('cir_db'))}"
                f" | {_fmt_metric('cnir_db', cr.get('cnir_db'))} |"
            )
            ln(row)
        ln()

    Path(save_path).write_text("\n".join(L), encoding="utf-8")


def write_detector_results(
    results: dict[str, dict],
    save_path: str | None,
    append: bool = False,
) -> None:
    """
    Write a Markdown table of detector-model results to save_path.

    Each entry in results is keyed by carrier name and contains:
        mode              — "seeker" or "fixed"
        noise_density_dbfs — noise level used (dBFS/Hz)
        ber               — measured BER
        ber_ci_lo/hi      — 95% CI bounds (None for fixed-noise runs)
        effective_ebn0_db — C/(N+I)*sps/bps in dB (None if not computed)
        theory_ebn0_db    — theory Eb/N0 at measured BER (None if no formula)
        implementation_loss_db — effective − theory (None if not available)
        cnr_db, cir_db, cnir_db
        evm_rms           — EVM % (None if not computed)
        n_bits_total      — bits used in final measurement (None for fixed)
        n_iter            — bisection iterations (None for fixed)

    append=True opens the file for appending rather than overwriting.
    """
    if not results or save_path is None:
        return

    from pathlib import Path

    def _f(v, fmt=".2f") -> str:
        if v is None:
            return "—"
        if isinstance(v, float) and (np.isnan(v) or not np.isfinite(v)):
            return "∞" if v > 0 else "—"
        return format(v, fmt)

    L = []
    def ln(s=""): L.append(s)

    ln("## Detector Results")
    ln()
    ln("| Carrier | Mode | Noise (dBFS/Hz) | BER | BER CI | Eff Eb/N0 (dB) "
       "| Theory Eb/N0 (dB) | Impl Loss (dB) | CNR (dB) | CIR (dB) | CNIR (dB) "
       "| EVM (%) | Bits |")
    ln("|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    for name, r in results.items():
        ber = r.get("ber")
        ber_str = "0" if ber == 0 else (_f(ber, ".2e") if ber is not None else "—")

        ci_lo = r.get("ber_ci_lo")
        ci_hi = r.get("ber_ci_hi")
        if ci_lo is not None and ci_hi is not None:
            ci_str = f"[{_f(ci_lo, '.2e')}, {_f(ci_hi, '.2e')}]"
        else:
            ci_str = "—"

        n_bits = r.get("n_bits_total")
        bits_str = f"{n_bits:,}" if n_bits is not None else "—"

        ln(
            f"| {name}"
            f" | {r.get('mode', '—')}"
            f" | {_f(r.get('noise_density_dbfs'), '.1f')}"
            f" | {ber_str}"
            f" | {ci_str}"
            f" | {_f(r.get('effective_ebn0_db'), '.2f')}"
            f" | {_f(r.get('theory_ebn0_db'), '.2f')}"
            f" | {_f(r.get('implementation_loss_db'), '.2f')}"
            f" | {_f(r.get('cnr_db'), '.1f')}"
            f" | {_f(r.get('cir_db'), '.1f')}"
            f" | {_f(r.get('cnir_db'), '.1f')}"
            f" | {_f(r.get('evm_rms'), '.2f')}"
            f" | {bits_str} |"
        )

    ln()

    text = "\n".join(L)
    p = Path(save_path)
    if append and p.exists():
        with p.open("a", encoding="utf-8") as f:
            f.write("\n" + text)
    else:
        p.write_text(text, encoding="utf-8")
