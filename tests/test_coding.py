"""Tests for the FEC coding subsystem (sim/coding)."""
from pathlib import Path

import numpy as np

from sim.coding import ConvolutionalCode, LDPCCode, TurboCode
from sim.receiver import soft_demap

_LDPC_ALIST = Path(__file__).resolve().parent.parent / "data" / "ldpc" / "mackay_13298.alist"


# ── Convolutional code ───────────────────────────────────────────────────────

def test_conv_encode_length():
    """Encoded length = (n_data + K - 1) * n  (tail-terminated, rate 1/n)."""
    code = ConvolutionalCode()
    coded = code.encode(np.zeros(100, dtype=int))
    assert len(coded) == (100 + code.K - 1) * code.n


def test_conv_roundtrip_noiseless():
    """Noiseless high-confidence LLRs decode back to the original data exactly."""
    rng = np.random.default_rng(2)
    code = ConvolutionalCode()
    data = rng.integers(0, 2, 500)
    coded = code.encode(data)
    llrs = np.where(coded == 0, 30.0, -30.0)
    decoded = code.decode(llrs)
    assert np.array_equal(decoded, data)


def test_conv_coding_gain():
    """Soft-decision Viterbi beats uncoded BPSK at the same information Eb/N0."""
    rng = np.random.default_rng(0)
    code = ConvolutionalCode()
    n_data = 5000
    data = rng.integers(0, 2, n_data)
    ebn0 = 10.0 ** (5.0 / 10.0)

    # Coded path — rate 1/2, so each coded symbol carries half the info-bit energy.
    coded = code.encode(data)
    tx = 1.0 - 2.0 * coded
    esn0 = ebn0 * code.rate
    sigma = np.sqrt(1.0 / (2.0 * esn0))
    rx = tx + sigma * (rng.standard_normal(len(tx)) + 1j * rng.standard_normal(len(tx)))
    llrs = soft_demap(rx, "BPSK", noise_var=2.0 * sigma ** 2)
    coded_ber = float(np.mean(code.decode(llrs) != data))

    # Uncoded BPSK at the same information Eb/N0.
    sigma_u = np.sqrt(1.0 / (2.0 * ebn0))
    rx_u = (1.0 - 2.0 * data) + sigma_u * rng.standard_normal(n_data)
    uncoded_ber = float(np.mean((rx_u < 0) != data))

    assert coded_ber < uncoded_ber, f"no coding gain: {coded_ber} vs {uncoded_ber}"
    assert coded_ber < 0.01, f"coded BER too high: {coded_ber}"


# ── LDPC code ────────────────────────────────────────────────────────────────

def test_ldpc_parses():
    """The alist parity-check matrix loads with the expected dimensions."""
    code = LDPCCode(_LDPC_ALIST)
    assert code.n == 13298
    assert code.m == 10002
    assert 0.2 < code.design_rate < 0.3


def test_ldpc_decodes_clean():
    """A near-noiseless all-zero codeword decodes to all zeros."""
    code = LDPCCode(_LDPC_ALIST)
    rng = np.random.default_rng(0)
    sigma = 0.05
    rx = 1.0 + sigma * (rng.standard_normal(code.n) + 1j * rng.standard_normal(code.n))
    llrs = soft_demap(rx, "BPSK", noise_var=2.0 * sigma ** 2)
    assert np.all(code.decode(llrs) == 0)


def test_ldpc_coding_gain():
    """All-zero codeword over AWGN: BP decoding beats uncoded BPSK at same Eb/N0.

    The code is linear, so BER is independent of the transmitted codeword; the
    all-zero codeword is the conventional choice for LDPC BER simulation.
    """
    code = LDPCCode(_LDPC_ALIST)
    rng = np.random.default_rng(1)
    ebn0 = 10.0 ** (4.0 / 10.0)
    n_frames = 5

    # Coded path — each coded symbol carries design_rate info-bits of energy.
    esn0 = ebn0 * code.design_rate
    sigma = np.sqrt(1.0 / (2.0 * esn0))
    coded_errs = 0
    for _ in range(n_frames):
        rx = 1.0 + sigma * (rng.standard_normal(code.n) + 1j * rng.standard_normal(code.n))
        llrs = soft_demap(rx, "BPSK", noise_var=2.0 * sigma ** 2)
        coded_errs += int(np.sum(code.decode(llrs) != 0))
    coded_ber = coded_errs / (n_frames * code.n)

    # Uncoded BPSK at the same information Eb/N0.
    sigma_u = np.sqrt(1.0 / (2.0 * ebn0))
    rx_u = 1.0 + sigma_u * rng.standard_normal(n_frames * code.n)
    uncoded_ber = float(np.mean(rx_u < 0))

    assert coded_ber < uncoded_ber, f"no coding gain: {coded_ber} vs {uncoded_ber}"
    assert coded_ber < 0.005, f"coded BER too high: {coded_ber}"


# ── Turbo code ───────────────────────────────────────────────────────────────

def test_turbo_encode_length():
    """Rate-1/3 turbo: encoded length is 3 * k (systematic + two parity streams)."""
    code = TurboCode(500)
    assert len(code.encode(np.zeros(500, dtype=int))) == 3 * 500


def test_turbo_roundtrip_noiseless():
    """Noiseless high-confidence LLRs decode back to the original data exactly."""
    rng = np.random.default_rng(4)
    code = TurboCode(800)
    data = rng.integers(0, 2, 800)
    coded = code.encode(data)
    llrs = np.where(coded == 0, 30.0, -30.0)
    decoded = code.decode(llrs, iterations=4)
    assert np.array_equal(decoded, data)


def test_turbo_coding_gain():
    """Iterative BCJR turbo decoding beats uncoded BPSK at the same Eb/N0."""
    rng = np.random.default_rng(5)
    code = TurboCode(2000)
    data = rng.integers(0, 2, code.k)
    ebn0 = 10.0 ** (2.5 / 10.0)

    coded = code.encode(data)
    tx = 1.0 - 2.0 * coded
    esn0 = ebn0 * code.rate
    sigma = np.sqrt(1.0 / (2.0 * esn0))
    rx = tx + sigma * (rng.standard_normal(len(tx)) + 1j * rng.standard_normal(len(tx)))
    llrs = soft_demap(rx, "BPSK", noise_var=2.0 * sigma ** 2)
    coded_ber = float(np.mean(code.decode(llrs, iterations=8) != data))

    sigma_u = np.sqrt(1.0 / (2.0 * ebn0))
    rx_u = (1.0 - 2.0 * data) + sigma_u * rng.standard_normal(code.k)
    uncoded_ber = float(np.mean((rx_u < 0) != data))

    assert coded_ber < uncoded_ber, f"no coding gain: {coded_ber} vs {uncoded_ber}"
    assert coded_ber < 0.01, f"coded BER too high: {coded_ber}"
