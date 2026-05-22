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

from sim.coding import ConcatenatedCode, ConvolutionalCode, LDPCCode, TurboCode
from sim.receiver import soft_demap

PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots", "performance")
_LDPC_ALIST = Path(__file__).resolve().parent.parent / "data" / "ldpc" / "mackay_13298.alist"
_EBN0_DB = np.arange(0.0, 11.0, 1.0)
_N_FRAMES = 3                       # waterfall plot — raise for a thorough run
_VALIDATION_FRAMES = 400            # validation test — raise for a deeper run


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


def _block_ber(code, k: int, ebn0_db: float, n_frames: int,
               rng: np.random.Generator) -> float:
    """Coded BER for a block code at one Eb/N0 over n_frames random data frames."""
    sigma = math.sqrt(1.0 / (2.0 * 10.0 ** (ebn0_db / 10.0) * code.rate))
    errs = 0
    for _ in range(n_frames):
        data = rng.integers(0, 2, k)
        tx = 1.0 - 2.0 * code.encode(data)
        rx = tx + sigma * (rng.standard_normal(len(tx)) + 1j * rng.standard_normal(len(tx)))
        llrs = soft_demap(rx, "BPSK", noise_var=2.0 * sigma ** 2)
        errs += int(np.sum(code.decode(llrs) != data))
    return errs / (n_frames * k)


def _ldpc_ber(code: LDPCCode, ebn0_db: float, n_frames: int,
              rng: np.random.Generator) -> float:
    """LDPC BER from the all-zero codeword at one Eb/N0 over n_frames frames."""
    sigma = math.sqrt(1.0 / (2.0 * 10.0 ** (ebn0_db / 10.0) * code.design_rate))
    errs = 0
    for _ in range(n_frames):
        rx = 1.0 + sigma * (rng.standard_normal(code.n) + 1j * rng.standard_normal(code.n))
        llrs = soft_demap(rx, "BPSK", noise_var=2.0 * sigma ** 2)
        errs += int(np.sum(code.decode(llrs) != 0))
    return errs / (n_frames * code.n)


def _sweep_blockcode(code: ConcatenatedCode | TurboCode, k: int,
                     rng: np.random.Generator) -> list[float]:
    """Measured coded BER across the Eb/N0 grid for a random-data block code."""
    return [max(_block_ber(code, k, e, _N_FRAMES, rng), 1e-7) for e in _EBN0_DB]


def _sweep_ldpc(code: LDPCCode, rng: np.random.Generator) -> list[float]:
    """Measured LDPC BER (all-zero codeword) across the Eb/N0 grid."""
    return [max(_ldpc_ber(code, e, _N_FRAMES, rng), 1e-7) for e in _EBN0_DB]


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


def test_coding_validation():
    """Validate that the codes reach their *expected* performance, not just
    "beats uncoded": convolutional against its Viterbi union bound, and
    turbo / LDPC / concatenated against reference operating points.

    Raise _VALIDATION_FRAMES for a deeper run.
    """
    os.makedirs(PLOT_DIR, exist_ok=True)
    rng = np.random.default_rng(7777)

    # Convolutional: measured BER must track the Viterbi union bound (an upper
    # bound, tight at moderate-to-high SNR).
    conv = ConvolutionalCode()
    conv_ebn0 = [3.5, 4.0]
    measured = [_block_ber(conv, 4000, e, _VALIDATION_FRAMES, rng) for e in conv_ebn0]
    bound = [conv.union_bound_ber(e, d_max=30) for e in conv_ebn0]
    for e, me, bd in zip(conv_ebn0, measured, bound):
        assert me <= bd * 2.0, f"conv BER {me:.2e} exceeds union bound {bd:.2e} at {e} dB"
        assert me >= bd * 0.15, f"conv BER {me:.2e} far below union bound {bd:.2e} at {e} dB"
    assert measured[0] > measured[1], "convolutional BER not decreasing with Eb/N0"

    # Turbo / LDPC / concatenated: BER must be low where the code should work.
    turbo = TurboCode(2000)
    turbo_ber = _block_ber(turbo, turbo.k, 2.5, 6, rng)
    assert turbo_ber < 1e-2, f"turbo BER {turbo_ber:.2e} too high at 2.5 dB"

    concat = ConcatenatedCode()
    concat_ber = _block_ber(concat, concat.k_data_bits, 4.0, 6, rng)
    assert concat_ber < 1e-2, f"concatenated BER {concat_ber:.2e} too high at 4.0 dB"

    ldpc = LDPCCode(_LDPC_ALIST)
    ldpc_ber = _ldpc_ber(ldpc, 3.5, 6, rng)
    assert ldpc_ber < 1e-2, f"LDPC BER {ldpc_ber:.2e} too high at 3.5 dB"

    # Plot the convolutional measured BER against the union bound.
    grid = np.linspace(3.0, 5.5, 26)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(grid, [conv.union_bound_ber(e, d_max=30) for e in grid], "k--",
                label="Viterbi union bound")
    ax.semilogy(conv_ebn0, measured, "o", markersize=8, label="Measured")
    ax.set_xlabel("Eb/N0 (dB)")
    ax.set_ylabel("BER")
    ax.set_title("Convolutional code — measured BER vs Viterbi union bound")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(1e-7, 1e-1)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "coding_validation_conv.png"), dpi=120)
    plt.close(fig)
