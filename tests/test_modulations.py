"""
Tests for sim.modulation and the end-to-end baseband → receive chain.

Run with:
    .venv\\Scripts\\python.exe -m pytest tests/ -v
"""

import numpy as np
import pytest

from sim.modulation import (
    SUPPORTED, bits_per_symbol, constellation, map_bits, decide,
    differential_encode, differential_decode, rotational_symmetry,
)
from sim.baseband import rrc_baseband
from sim.receiver import receive


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
    ("QPSK",   300,  1e6, 8),
    ("OQPSK",  300,  1e6, 8),
    ("8PSK",   300,  1e6, 8),
    ("16QAM",  400,  1e6, 8),
    ("16APSK", 400,  1e6, 8),
    ("32APSK", 500,  1e6, 8),
]


@pytest.mark.parametrize("mod,n_sym,sym_rate,sps", _E2E_CASES)
def test_noiseless_ber_zero(mod, n_sym, sym_rate, sps):
    """
    With no noise and no nonlinearity (amplitude = 1, phase = 0), BER must be 0.

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
    """Noiseless EVM must be below 5% (residual from filter truncation only)."""
    native_rate = float(sps) * sym_rate
    bb, _t, bits, _symbols = rrc_baseband(
        mod, n_sym, sym_rate, native_rate,
        rolloff=0.35, filter_span=10, seed=42)
    result = receive(bb, mod, rolloff=0.35, filter_span=10, sps=sps,
                     reference_bits=bits)
    evm = result["evm_rms"]
    assert evm < 5.0, f"{mod}: noiseless EVM = {evm:.2f}%, expected < 5%"
