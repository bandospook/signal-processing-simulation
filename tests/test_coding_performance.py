"""Coding-gain waterfall: BER vs Eb/N0 for BPSK across the four FEC schemes.

Mirrors test_awgn_performance.py — measures BER curves and saves a diagnostic
plot.  _N_FRAMES is kept small so the suite stays fast; raise it for a thorough
deep-BER run (see docs/coding_design.md, "Testing strategy").
"""
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sim.coding import ConcatenatedCode, LDPCCode, TurboCode
from sim.receiver import soft_demap

PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots", "performance")
_LDPC_ALIST = Path(__file__).resolve().parent.parent / "data" / "ldpc" / "mackay_13298.alist"
_EBN0_DB = np.arange(0.0, 11.0, 1.0)
_N_FRAMES = 3                       # raise for a thorough deep-BER run


def _bpsk_theory(ebn0_db: float) -> float:
    return 0.5 * math.erfc(math.sqrt(10.0 ** (ebn0_db / 10.0)))


def _sweep_uncoded(rng: np.random.Generator) -> list[float]:
    """Measured uncoded BPSK BER at each Eb/N0."""
    bers = []
    for ebn0_db in _EBN0_DB:
        sigma = math.sqrt(1.0 / (2.0 * 10.0 ** (ebn0_db / 10.0)))
        data = rng.integers(0, 2, _N_FRAMES * 5000)
        rx = (1.0 - 2.0 * data) + sigma * rng.standard_normal(len(data))
        bers.append(max(float(np.mean((rx < 0) != data)), 1e-7))
    return bers


def _sweep_blockcode(code: ConcatenatedCode | TurboCode, k: int,
                     rng: np.random.Generator) -> list[float]:
    """Measured coded BER for a code that encodes random data blocks of k bits."""
    bers = []
    for ebn0_db in _EBN0_DB:
        sigma = math.sqrt(1.0 / (2.0 * 10.0 ** (ebn0_db / 10.0) * code.rate))
        errs = 0
        for _ in range(_N_FRAMES):
            data = rng.integers(0, 2, k)
            tx = 1.0 - 2.0 * code.encode(data)
            rx = tx + sigma * (rng.standard_normal(len(tx)) + 1j * rng.standard_normal(len(tx)))
            llrs = soft_demap(rx, "BPSK", noise_var=2.0 * sigma ** 2)
            errs += int(np.sum(code.decode(llrs) != data))
        bers.append(max(errs / (_N_FRAMES * k), 1e-7))
    return bers


def _sweep_ldpc(code: LDPCCode, rng: np.random.Generator) -> list[float]:
    """Measured LDPC BER using the all-zero codeword (valid since the code is linear)."""
    bers = []
    for ebn0_db in _EBN0_DB:
        sigma = math.sqrt(1.0 / (2.0 * 10.0 ** (ebn0_db / 10.0) * code.design_rate))
        errs = 0
        for _ in range(_N_FRAMES):
            rx = 1.0 + sigma * (rng.standard_normal(code.n) + 1j * rng.standard_normal(code.n))
            llrs = soft_demap(rx, "BPSK", noise_var=2.0 * sigma ** 2)
            errs += int(np.sum(code.decode(llrs) != 0))
        bers.append(max(errs / (_N_FRAMES * code.n), 1e-7))
    return bers


def test_coding_gain_waterfall():
    """BER-vs-Eb/N0 waterfalls for uncoded, concatenated, LDPC and turbo (BPSK)."""
    os.makedirs(PLOT_DIR, exist_ok=True)
    rng = np.random.default_rng(2024)

    uncoded = _sweep_uncoded(rng)
    turbo_code = TurboCode(2000)
    turbo = _sweep_blockcode(turbo_code, turbo_code.k, rng)
    concat_code = ConcatenatedCode()
    concat = _sweep_blockcode(concat_code, concat_code.k_data_bits, rng)
    ldpc = _sweep_ldpc(LDPCCode(_LDPC_ALIST), rng)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.semilogy(_EBN0_DB, [_bpsk_theory(e) for e in _EBN0_DB], "k--",
                alpha=0.5, label="Uncoded BPSK theory")
    ax.semilogy(_EBN0_DB, uncoded, "-o", label="Uncoded BPSK")
    ax.semilogy(_EBN0_DB, concat, "-s", label="Concatenated (RS + conv)")
    ax.semilogy(_EBN0_DB, ldpc, "-^", label="LDPC")
    ax.semilogy(_EBN0_DB, turbo, "-d", label="Turbo")
    ax.set_xlabel("Eb/N0 (dB)")
    ax.set_ylabel("BER")
    ax.set_title("Coding gain — BPSK")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(1e-6, 1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "coding_gain_bpsk.png"), dpi=120)
    plt.close(fig)

    # At a mid-high Eb/N0 every coded scheme must beat uncoded BPSK.
    idx = int(np.argmin(np.abs(_EBN0_DB - 6.0)))
    assert concat[idx] < uncoded[idx], "concatenated shows no gain at 6 dB"
    assert ldpc[idx] < uncoded[idx], "LDPC shows no gain at 6 dB"
    assert turbo[idx] < uncoded[idx], "turbo shows no gain at 6 dB"
