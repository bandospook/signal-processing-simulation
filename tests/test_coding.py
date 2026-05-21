"""Tests for the FEC coding subsystem (sim/coding)."""
import numpy as np

from sim.coding import ConvolutionalCode
from sim.receiver import soft_demap


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
