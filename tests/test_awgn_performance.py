"""AWGN performance tests: BER monotonicity, theory match, and diagnostic plots."""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from sim.baseband import rrc_baseband
from sim.modulation import bits_per_symbol
from sim.receiver import matched_filter, receive

# ── Constants ─────────────────────────────────────────────────────────────────

SPS = 8
ROLLOFF = 0.35
FILTER_SPAN = 10
PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots", "performance")

ALL_MODS = ["BPSK", "DBPSK", "QPSK", "OQPSK", "8PSK", "16QAM", "16APSK", "32APSK"]

# Bits per simulation run for the performance plot.
# n_sym is derived per modulation as _N_BITS_PLOT // bps so all modulations
# see the same number of bits and therefore the same sigma-band width.
# For rigorous stats use 1_000_000 (≈960k needed for ±0.001 at 95 % CI, worst
# case p=0.5); 10_000 is the sniff-test level — fast but visibly uncertain.
_N_BITS_PLOT = 10_000

# ── Helpers ───────────────────────────────────────────────────────────────────

def _q(x: float) -> float:
    return 0.5 * math.erfc(float(x) / math.sqrt(2))


def ber_theory(mod: str, EsN0_dB: float) -> float | None:
    """Closed-form BER vs Eb/N0. Returns None when no formula is available."""
    bps = bits_per_symbol(mod.upper())
    EbN0 = 10.0 ** (EsN0_dB / 10.0) / bps
    m = mod.upper()
    if m in ("BPSK", "QPSK", "OQPSK"):
        return 0.5 * math.erfc(math.sqrt(EbN0))
    if m == "DBPSK":
        # Two DBPSK variants exist with different theory curves:
        #
        #   1. Differentially-coherent detection (no carrier reference):
        #      BER = 0.5 * exp(-Eb/N0)
        #      Decisions are made by comparing the phase of each received symbol
        #      directly against the previous received symbol — no matched filter.
        #      ~3 dB worse than coherent BPSK at moderate SNR.
        #
        #   2. Coherent detection + differential decoding (this implementation):
        #      BER = 2 * p * (1 - p),  p = 0.5 * erfc(sqrt(Eb/N0))
        #      A coherent matched filter recovers symbols; consecutive hard decisions
        #      are XOR-ed to recover bits.  A single symbol error flips two decoded
        #      bits, so BER ≈ 2p at low error rates — about 1 dB worse than plain
        #      coherent BPSK but ~2 dB better than differentially-coherent detection.
        p = 0.5 * math.erfc(math.sqrt(EbN0))
        return 2.0 * p * (1.0 - p)
    if m == "8PSK":
        return (1.0 / 3.0) * math.erfc(math.sqrt(3.0 * EbN0) * math.sin(math.pi / 8))
    if m == "16QAM":
        return (3.0 / 8.0) * math.erfc(math.sqrt(2.0 * EbN0 / 5.0))
    return None  # 16APSK, 32APSK


def simulate_awgn(mod: str, EsN0_dB: float, n_sym: int = 2000,
                  sps: int = SPS, rolloff: float = ROLLOFF,
                  filter_span: int = FILTER_SPAN, seed: int = 42) -> dict:
    """RRC baseband → AWGN channel → receive(). Returns the receive() result dict."""
    EsN0_linear = 10.0 ** (EsN0_dB / 10.0)
    bb, _, bits, _ = rrc_baseband(
        mod, n_sym, symbol_rate=1.0, sample_rate=float(sps),
        rolloff=rolloff, filter_span=filter_span, seed=seed,
    )
    rng = np.random.default_rng(seed + 1)
    sigma_c = np.sqrt(sps / (2.0 * EsN0_linear))
    N = len(bb)
    noise = sigma_c * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
    return receive(bb + noise, mod, rolloff, filter_span, sps, reference_bits=bits)


def _interp_ebn0_at_ber(ebn0: np.ndarray, ber: np.ndarray, target: float) -> float | None:
    """Find Eb/N0 where BER crosses target via log-linear inverse interpolation."""
    lb = np.log10(np.maximum(ber, 1e-15))
    lt = math.log10(target)
    if not (lb[-1] <= lt <= lb[0]):
        return None
    return float(np.interp(lt, lb[::-1], ebn0[::-1]))


def _interp_ber_at_ebn0(ebn0: np.ndarray, ber: np.ndarray, ebn0_target: float) -> float | None:
    """Find BER at a specific Eb/N0 via log-linear interpolation."""
    if not (ebn0[0] <= ebn0_target <= ebn0[-1]):
        return None
    lb = np.log10(np.maximum(ber, 1e-15))
    return float(10.0 ** np.interp(ebn0_target, ebn0, lb))


# ── Assertion tests ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("mod", ALL_MODS)
def test_ber_monotone(mod):
    """BER must strictly decrease as Eb/N0 increases over 5 evenly-spaced points."""
    bps = bits_per_symbol(mod.upper())
    esn0_dB = 10.0 * np.log10(bps) + np.linspace(0, 10, 5)
    bers = [simulate_awgn(mod, es, n_sym=1000, seed=7)["ber"] for es in esn0_dB]
    for i in range(1, len(bers)):
        assert bers[i] < bers[i - 1], (
            f"{mod}: BER not decreasing at Eb/N0 step {i}  "
            f"({bers[i]:.4f} >= {bers[i-1]:.4f})"
        )


# EsN0_dB chosen per modulation so theory BER ≈ 4–6 %
_THEORY_POINTS = [
    ("BPSK",   2.0),
    ("QPSK",   5.0),
    ("OQPSK",  5.0),
    ("8PSK",   9.5),
    ("16QAM", 11.0),
]


@pytest.mark.parametrize("mod,EsN0_dB", _THEORY_POINTS)
def test_ber_matches_theory(mod, EsN0_dB):
    """Measured BER must be within a factor of 2 of the theoretical value."""
    result = simulate_awgn(mod, EsN0_dB, n_sym=5_000, seed=42)
    measured = result["ber"]
    theory = ber_theory(mod, EsN0_dB)
    assert measured > 0, f"{mod}: zero bit errors at Es/N0={EsN0_dB} dB — SNR too high"
    ratio = measured / theory
    assert 0.5 <= ratio <= 2.0, (
        f"{mod}: measured BER {measured:.4f} is {ratio:.2f}× theory {theory:.4f}"
    )


# ── Theory comparison table ───────────────────────────────────────────────────

_TARGET_BERS = [0.10, 0.05, 0.02, 0.01, 0.005]
_THEORY_MODS = ["BPSK", "DBPSK", "QPSK", "OQPSK", "8PSK", "16QAM"]


def test_ber_theory_table():
    """Full-range BER vs theory comparison; writes theory_comparison.md."""
    os.makedirs(PLOT_DIR, exist_ok=True)

    ebn0_arr = np.linspace(-2, 18, 41)
    rows: list[tuple] = []
    failures: list[str] = []

    for mod in _THEORY_MODS:
        bps = bits_per_symbol(mod.upper())
        esn0_arr = ebn0_arr + 10.0 * np.log10(bps)

        n_sym = _N_BITS_PLOT // bps
        theory_bers = np.array([ber_theory(mod, es) for es in esn0_arr], dtype=float)
        meas_bers = np.array([
            simulate_awgn(mod, es, n_sym=n_sym, seed=42)["ber"] for es in esn0_arr
        ])

        for target in _TARGET_BERS:
            th_ebn0 = _interp_ebn0_at_ber(ebn0_arr, theory_bers, target)
            me_ebn0 = _interp_ebn0_at_ber(ebn0_arr, meas_bers, target)

            if th_ebn0 is None or me_ebn0 is None:
                rows.append((mod, f"{target:.3f}", "—", "—", "—", "—"))
                continue

            delta = me_ebn0 - th_ebn0
            meas_ber_at_th = _interp_ber_at_ebn0(ebn0_arr, meas_bers, th_ebn0)
            ratio = meas_ber_at_th / target if meas_ber_at_th is not None else None

            rows.append((
                mod,
                f"{target:.3f}",
                f"{th_ebn0:.2f}",
                f"{me_ebn0:.2f}",
                f"{delta:+.2f}",
                f"{ratio:.2f}" if ratio is not None else "—",
            ))

            if abs(delta) > 1.5:
                failures.append(
                    f"{mod} @ BER={target:.3f}: dEb/N0={delta:+.2f} dB (limit +-1.5 dB)"
                )
            if ratio is not None and not (0.3 <= ratio <= 3.0):
                failures.append(
                    f"{mod} @ BER={target:.3f}: BER ratio={ratio:.2f} (outside [0.3, 3.0])"
                )

    header = (
        "| Mod | Target BER | Theory Eb/N0 (dB) | Measured Eb/N0 (dB) "
        "| dEb/N0 (dB) | BER ratio |\n"
        "|-----|:----------:|:-----------------:|:-------------------:"
        "|:-----------:|:---------:|\n"
    )
    body = "\n".join(
        f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} |"
        for r in rows
    )
    md = f"# BER vs Theory Comparison\n\n{header}{body}\n"
    if failures:
        md += "\n## Assertion Failures\n\n" + "\n".join(f"- {f}" for f in failures) + "\n"

    out_path = os.path.join(PLOT_DIR, "theory_comparison.md")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(md)
    print(f"\n{md}")

    assert not failures, "BER-vs-theory failures:\n" + "\n".join(failures)


# ── Plot generation ───────────────────────────────────────────────────────────

def test_generate_performance_plots():
    """Sweep Eb/N0 for all modulations; save BER, EVM, and eye-diagram plots."""
    os.makedirs(PLOT_DIR, exist_ok=True)

    ebn0_dB_arr = np.linspace(-2, 14, 20)

    sweep: dict[str, dict] = {}
    for mod in ALL_MODS:
        bps = bits_per_symbol(mod.upper())
        n_sym = _N_BITS_PLOT // bps
        esn0_arr = ebn0_dB_arr + 10.0 * np.log10(bps)
        bers, evms = [], []
        for esn0 in esn0_arr:
            r = simulate_awgn(mod, esn0, n_sym=n_sym, seed=0)
            bers.append(max(r["ber"], 1e-6))
            evms.append(r["evm_rms"])
        sweep[mod] = dict(ebn0=ebn0_dB_arr, bers=bers, evms=evms,
                          n_bits=_N_BITS_PLOT)

    _plot_ber(sweep)
    _plot_evm(sweep)
    for mod in ("BPSK", "QPSK", "16QAM"):
        _plot_eye(mod)


_BPSK_EQUIV = {"BPSK", "QPSK", "OQPSK"}  # share the same theory formula


def _plot_ber(sweep: dict) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    bpsk_theory_drawn = False
    for i, (mod, d) in enumerate(sweep.items()):
        color = f"C{i}"
        bps = bits_per_symbol(mod.upper())
        esn0_arr = d["ebn0"] + 10.0 * np.log10(bps)
        bers = np.array(d["bers"])
        ebn0 = np.array(d["ebn0"])
        n_bits = d.get("n_bits", len(bers) * bps)

        # 1/2/3-sigma uncertainty bands for the measured curve.
        # sigma = sqrt(p*(1-p)/N) — binomial standard deviation of the BER estimate.
        sigma = np.sqrt(bers * (1.0 - bers) / n_bits)
        for k_sig, alpha in ((3, 0.07), (2, 0.13), (1, 0.22)):
            y_lo = np.maximum(bers - k_sig * sigma, 1e-6)
            y_hi = np.minimum(bers + k_sig * sigma, 1.0)
            ax.fill_between(ebn0, y_lo, y_hi, color=color, alpha=alpha, linewidth=0)

        # Theory curve (dashed).  BPSK/QPSK/OQPSK share one formula so draw it
        # only once, in BPSK's colour (C0), to avoid three overlapping dashed lines
        # blending into an unintended colour.
        in_bpsk_family = mod.upper() in _BPSK_EQUIV
        if in_bpsk_family and bpsk_theory_drawn:
            pass  # already drawn for this family
        else:
            theory_color = "C0" if in_bpsk_family else color
            pairs = [(e, ber_theory(mod, es)) for e, es in zip(ebn0, esn0_arr)
                     if ber_theory(mod, es) is not None]
            if pairs:
                tx, ty = zip(*pairs)
                ax.semilogy(tx, ty, "--", color=theory_color, alpha=0.55, linewidth=1)
            if in_bpsk_family:
                bpsk_theory_drawn = True

        # Measured curve on top of bands
        ax.semilogy(ebn0, bers, "-o", color=color, markersize=3, label=mod)

    ax.set_xlabel("Eb/N0 (dB)")
    ax.set_ylabel("BER")
    ax.set_title("BER vs Eb/N0  (theory dashed, measured solid, shading = 1/2/3σ)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(1e-3, 1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "ber_vs_ebn0.png"), dpi=120)
    plt.close(fig)


def _plot_evm(sweep: dict) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, (mod, d) in enumerate(sweep.items()):
        ax.plot(d["ebn0"], d["evms"], f"-o", color=f"C{i}", markersize=3, label=mod)
    ax.set_xlabel("Eb/N0 (dB)")
    ax.set_ylabel("EVM RMS (%)")
    ax.set_title("EVM vs Eb/N0")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "evm_vs_ebn0.png"), dpi=120)
    plt.close(fig)


def _plot_eye(mod: str) -> None:
    bps = bits_per_symbol(mod.upper())
    EsN0_dB = 10.0 + 10.0 * np.log10(bps)   # Eb/N0 = 10 dB for all mods
    EsN0_linear = 10.0 ** (EsN0_dB / 10.0)

    bb, _, _, _ = rrc_baseband(
        mod, 500, symbol_rate=1.0, sample_rate=float(SPS),
        rolloff=ROLLOFF, filter_span=FILTER_SPAN, seed=1,
    )
    rng = np.random.default_rng(99)
    sigma_c = np.sqrt(SPS / (2.0 * EsN0_linear))
    N = len(bb)
    noise = sigma_c * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
    mf = matched_filter(bb + noise, ROLLOFF, FILTER_SPAN, SPS)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    half = SPS // 2
    n_traces = min(200, (len(mf) // SPS) - 2)
    for ax, rail, label in zip(axes, (np.real(mf), np.imag(mf)), ("I", "Q")):
        for i in range(1, n_traces + 1):
            start = i * SPS - half
            seg = rail[start : start + 2 * SPS]
            if len(seg) == 2 * SPS:
                ax.plot(seg, color="steelblue", alpha=0.15, linewidth=0.7)
        ax.axvline(half,         color="crimson", linestyle="--", linewidth=0.9)
        ax.axvline(half + SPS,   color="crimson", linestyle="--", linewidth=0.9)
        ax.set_title(f"{mod} {label}-rail eye  (Eb/N0 = 10 dB)")
        ax.set_xlabel("Sample offset")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, f"eye_diagram_{mod.lower()}.png"), dpi=120)
    plt.close(fig)
