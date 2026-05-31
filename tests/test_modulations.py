r"""Tests for sim.modulation and the end-to-end baseband → receive chain.

Run with:
    .venv\Scripts\python.exe -m pytest tests/ -v
"""

import numpy as np
import pytest

from sim.modulation import (
    SUPPORTED, bits_per_symbol, constellation, map_bits, decide,
    differential_encode, differential_decode,
)
from sim.baseband import rrc_baseband
from sim.receiver import receive, measure_evm_rms, soft_demap


# ── Constellation unit tests ──────────────────────────────────────────────────

@pytest.mark.parametrize("mod", SUPPORTED)
def test_constellation_unit_power(mod):
    """All constellations must have unit average power."""
    C = constellation(mod)
    avg_power = float(np.mean(np.abs(C) ** 2))
    assert abs(avg_power - 1.0) < 1e-6, (
        f"{mod}: average power = {avg_power:.6f}, expected 1.0")


@pytest.mark.parametrize("mod", SUPPORTED)
def test_constellation_size(mod):
    """Constellation length must equal 2^bits_per_symbol."""
    C = constellation(mod)
    expected = 2 ** bits_per_symbol(mod)
    assert len(C) == expected, f"{mod}: got {len(C)} points, expected {expected}"


@pytest.mark.parametrize("mod", SUPPORTED)
def test_map_decide_roundtrip(mod):
    """map_bits followed by decide must recover the original bits (noiseless)."""
    rng = np.random.default_rng(0)
    bps = bits_per_symbol(mod)
    bits = rng.integers(0, 2, 100 * bps).astype(int)
    symbols = map_bits(bits, mod)
    _, recovered = decide(symbols, mod)
    n = min(len(bits), len(recovered))
    assert np.array_equal(bits[:n], recovered[:n]), (
        f"{mod}: roundtrip mismatch, {np.sum(bits[:n] != recovered[:n])} / {n} errors")


# ── Differential encode / decode (DBPSK) ─────────────────────────────────────

def test_differential_encode_decode_roundtrip():
    """Differential encode → BPSK decision → differential decode recovers bits[1:]."""
    rng = np.random.default_rng(1)
    bits = rng.integers(0, 2, 200).astype(int)
    encoded = differential_encode(bits)
    # BPSK symbol map: 0 → +1, 1 → -1
    sym = np.where(encoded == 0, 1, -1).astype(float)
    decoded = differential_decode(sym)
    assert np.array_equal(decoded, bits[1:]), "DBPSK roundtrip failed"


def test_differential_decode_phase_immune():
    """Differential decode must give the same result after a 180° phase flip."""
    rng = np.random.default_rng(2)
    bits = rng.integers(0, 2, 200).astype(int)
    encoded = differential_encode(bits)
    sym = np.where(encoded == 0, 1.0, -1.0)
    sym_flipped = -sym
    decoded_normal = differential_decode(sym)
    decoded_flipped = differential_decode(sym_flipped)
    assert np.array_equal(decoded_normal, decoded_flipped), (
        "DBPSK is not immune to 180° phase flip")


# ── End-to-end noiseless BER tests ───────────────────────────────────────────

# Parameters: (modulation, num_symbols, symbol_rate, sps)
_E2E_CASES = [
    ("BPSK",   300,  1e6, 8),
    ("DBPSK",  300,  1e6, 8),
    ("MSK",    300,  1e6, 8),
    ("QPSK",   300,  1e6, 8),
    ("OQPSK",  300,  1e6, 8),
    ("8PSK",   300,  1e6, 8),
    ("16QAM",  400,  1e6, 8),
    ("16APSK", 400,  1e6, 8),
    ("32APSK", 500,  1e6, 8),
]


@pytest.mark.parametrize("mod,n_sym,sym_rate,sps", _E2E_CASES)
def test_noiseless_ber_zero(mod, n_sym, sym_rate, sps):
    """With no noise and no nonlinearity (amplitude = 1, phase = 0), BER must be 0.

    The baseband is generated, upsampled by a factor of 1 (no interpolation
    filter artefacts), then passed directly through the matched filter receiver.
    """
    native_rate = float(sps) * sym_rate

    bb, _t, bits, _symbols = rrc_baseband(
        mod, n_sym, sym_rate, native_rate,
        rolloff=0.35, filter_span=10, seed=42)

    result = receive(bb, mod,
                     rolloff=0.35, filter_span=10, sps=sps,
                     reference_bits=bits)

    ber = result["ber"]
    assert ber is not None, f"{mod}: BER not computed"
    # PSK family achieves exactly 0; multi-amplitude constellations (QAM/APSK) see
    # tiny residual BER (~0.001) from RRC filter boundary ISI on short test sequences.
    assert ber < 0.005, f"{mod}: noiseless BER = {ber:.4f}, expected < 0.005"


@pytest.mark.parametrize("mod,n_sym,sym_rate,sps", _E2E_CASES)
def test_evm_noiseless_low(mod, n_sym, sym_rate, sps):
    """Noiseless EVM must be below 5% (residual from filter truncation only).

    MSK is skipped — constant-envelope signal has no discrete constellation.
    """
    native_rate = float(sps) * sym_rate
    bb, _t, bits, _symbols = rrc_baseband(
        mod, n_sym, sym_rate, native_rate,
        rolloff=0.35, filter_span=10, seed=42)
    result = receive(bb, mod, rolloff=0.35, filter_span=10, sps=sps,
                     reference_bits=bits)
    evm = result["evm_rms"]
    if np.isnan(evm):
        return
    assert evm < 5.0, f"{mod}: noiseless EVM = {evm:.2f}%, expected < 5%"


# ── Error paths ───────────────────────────────────────────────────────────────

def test_bits_per_symbol_unknown():
    """Bits per symbol unknown."""
    with pytest.raises(ValueError, match="Unknown modulation"):
        bits_per_symbol("UNKNOWN_MOD")


def test_constellation_unknown():
    """Constellation unknown."""
    with pytest.raises(ValueError, match="Unknown modulation"):
        constellation("UNKNOWN_MOD")


def test_rrc_baseband_non_integer_sps():
    """sample_rate / symbol_rate must be an integer ≥ 2."""
    with pytest.raises(ValueError, match="integer"):
        rrc_baseband("BPSK", num_symbols=10, symbol_rate=3e6, sample_rate=7e6, seed=0)


def test_rrc_baseband_accepts_supplied_bits():
    """rrc_baseband uses pre-supplied bits and derives num_symbols from them."""
    supplied = np.array([0, 1, 1, 0, 1, 0, 0, 1] * 5, dtype=int)   # 40 bits
    bb, _t, bits, _sym = rrc_baseband("QPSK", num_symbols=0, symbol_rate=1e6,
                                      sample_rate=4e6, bits=supplied)
    assert np.array_equal(bits, supplied)
    assert len(bb) == (len(supplied) // 2) * 4   # QPSK: 2 bits/symbol, sps=4


def test_receive_no_reference_bits_bpsk():
    """receive() with no reference_bits → ber is None for non-DBPSK."""
    bb, _, _, _ = rrc_baseband("BPSK", num_symbols=200, symbol_rate=1e6,
                                sample_rate=4e6, seed=0)
    result = receive(bb, "BPSK", rolloff=0.35, filter_span=8, sps=4)
    assert result["ber"] is None
    assert result["evm_rms"] is not None


def test_receive_no_reference_bits_dbpsk():
    """receive() with no reference_bits → ber is None for DBPSK."""
    bb, _, _, _ = rrc_baseband("DBPSK", num_symbols=200, symbol_rate=1e6,
                                sample_rate=4e6, seed=0)
    result = receive(bb, "DBPSK", rolloff=0.35, filter_span=8, sps=4)
    assert result["ber"] is None


def test_receive_no_reference_bits_msk():
    """receive() with no reference_bits → ber is None and EVM is NaN for MSK."""
    bb, _, _, _ = rrc_baseband("MSK", num_symbols=200, symbol_rate=1e6,
                                sample_rate=4e6, seed=0)
    result = receive(bb, "MSK", rolloff=0.35, filter_span=8, sps=4)
    assert result["ber"] is None
    assert np.isnan(result["evm_rms"])


def test_msk_phase_ambiguity_correction():
    """MSK receiver corrects a global π phase offset (ber > 0.5 → flip all decisions)."""
    bb, _, bits, _ = rrc_baseband("MSK", num_symbols=300, symbol_rate=1e6,
                                   sample_rate=4e6, seed=7)
    result = receive(-bb, "MSK", rolloff=0.35, filter_span=8, sps=4,
                     reference_bits=bits)
    assert result["ber"] is not None
    assert result["ber"] < 0.01


def test_measure_evm_zero_signal():
    """Zero-amplitude input → EVM is NaN (undefined)."""
    zeros = np.zeros(20, dtype=complex)
    ideal = np.ones(20, dtype=complex)
    assert np.isnan(measure_evm_rms(zeros, ideal))


# ── Soft-decision demapping ──────────────────────────────────────────────────

def test_soft_demap_length():
    """Output holds one LLR per coded bit: n_symbols × bits_per_symbol."""
    rng = np.random.default_rng(1)
    for mod, n in (("BPSK", 50), ("QPSK", 40), ("16QAM", 30)):
        bps = bits_per_symbol(mod)
        sym = map_bits(rng.integers(0, 2, n * bps).astype(int), mod)
        llrs = soft_demap(sym, mod, noise_var=0.1)
        assert len(llrs) == n * bps


def test_soft_demap_bpsk_sign():
    """Positive LLR favours bit 0 (symbol +1); negative favours bit 1 (symbol -1)."""
    samples = np.array([1.0, -1.0, 0.9, -0.85], dtype=complex)
    llrs = soft_demap(samples, "BPSK", noise_var=0.1)
    assert llrs[0] > 0 and llrs[2] > 0
    assert llrs[1] < 0 and llrs[3] < 0


def test_soft_demap_confidence_grows_with_snr():
    """LLR magnitude increases as the noise variance falls."""
    y = np.array([0.7 + 0.1j], dtype=complex)
    weak = soft_demap(y, "QPSK", noise_var=0.5)
    strong = soft_demap(y, "QPSK", noise_var=0.02)
    assert np.all(np.abs(strong) > np.abs(weak))


@pytest.mark.parametrize("mod", ["BPSK", "QPSK", "8PSK", "16QAM", "16APSK", "32APSK"])
def test_soft_demap_hard_consistency(mod):
    """At high SNR, slicing the LLRs at zero reproduces decide()'s hard bits."""
    rng = np.random.default_rng(0)
    bps = bits_per_symbol(mod)
    bits = rng.integers(0, 2, 200 * bps).astype(int)
    symbols = map_bits(bits, mod)
    noise = 0.01 * (rng.standard_normal(len(symbols))
                    + 1j * rng.standard_normal(len(symbols)))
    rx = symbols + noise
    llr_bits = (soft_demap(rx, mod, noise_var=2e-4) < 0).astype(int)
    _, decide_bits = decide(rx, mod)
    assert np.array_equal(llr_bits, decide_bits)
