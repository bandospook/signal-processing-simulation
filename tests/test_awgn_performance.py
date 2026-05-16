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
PLOT_DIR = os.path.join(os.path.dirname(__file__), "..", "plots", "performance")

ALL_MODS = ["BPSK", "DBPSK", "QPSK", "OQPSK", "8PSK", "16QAM", "16APSK", "32APSK"]

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
        return 0.5 * math.exp(-EbN0)
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
    result = simulate_awgn(mod, EsN0_dB, n_sym=5000, seed=42)
    measured = result["ber"]
    theory = ber_theory(mod, EsN0_dB)
    assert measured > 0, f"{mod}: zero bit errors at Es/N0={EsN0_dB} dB — SNR too high"
    ratio = measured / theory
    assert 0.5 <= ratio <= 2.0, (
        f"{mod}: measured BER {measured:.4f} is {ratio:.2f}× theory {theory:.4f}"
    )


# ── Plot generation ───────────────────────────────────────────────────────────

def test_generate_performance_plots():
    """Sweep Eb/N0 for all modulations; save BER, EVM, and eye-diagram plots."""
    os.makedirs(PLOT_DIR, exist_ok=True)

    ebn0_dB_arr = np.linspace(-2, 14, 20)

    sweep: dict[str, dict] = {}
    for mod in ALL_MODS:
        bps = bits_per_symbol(mod.upper())
        esn0_arr = ebn0_dB_arr + 10.0 * np.log10(bps)
        bers, evms = [], []
        for esn0 in esn0_arr:
            r = simulate_awgn(mod, esn0, n_sym=2000, seed=0)
            bers.append(max(r["ber"], 1e-6))
            evms.append(r["evm_rms"])
        sweep[mod] = dict(ebn0=ebn0_dB_arr, bers=bers, evms=evms)

    _plot_ber(sweep)
    _plot_evm(sweep)
    for mod in ("BPSK", "QPSK", "16QAM"):
        _plot_eye(mod)


def _plot_ber(sweep: dict) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, (mod, d) in enumerate(sweep.items()):
        color = f"C{i}"
        bps = bits_per_symbol(mod.upper())
        esn0_arr = d["ebn0"] + 10.0 * np.log10(bps)
        pairs = [(e, ber_theory(mod, es)) for e, es in zip(d["ebn0"], esn0_arr)
                 if ber_theory(mod, es) is not None]
        if pairs:
            tx, ty = zip(*pairs)
            ax.semilogy(tx, ty, "--", color=color, alpha=0.55, linewidth=1)
        ax.semilogy(d["ebn0"], d["bers"], "-o", color=color, markersize=3, label=mod)
    ax.set_xlabel("Eb/N0 (dB)")
    ax.set_ylabel("BER")
    ax.set_title("BER vs Eb/N0  (theory dashed, measured solid)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(1e-5, 1.0)
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
    for ax, rail, label in zip(axes, (mf.real, mf.imag), ("I", "Q")):
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
