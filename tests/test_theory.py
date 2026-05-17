"""Unit tests for sim.theory: ber_awgn and ebn0_for_ber."""
import math

import pytest

from sim.theory import ber_awgn, ebn0_for_ber
from sim.modulation import bits_per_symbol


# ── ber_awgn ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("mod", ["BPSK", "QPSK", "OQPSK"])
def test_ber_awgn_bpsk_family(mod):
    val = ber_awgn(mod, 0.0)
    assert val is not None
    assert 0.0 < val < 0.5


def test_ber_awgn_dbpsk_worse_than_bpsk():
    esn0 = 3.0
    dbpsk = ber_awgn("DBPSK", esn0)
    bpsk  = ber_awgn("BPSK",  esn0)
    assert dbpsk is not None and 0.0 < dbpsk < 0.5
    assert dbpsk > bpsk


def test_ber_awgn_8psk():
    val = ber_awgn("8PSK", 8.0)
    assert val is not None
    assert 0.0 < val < 0.5


def test_ber_awgn_16qam():
    val = ber_awgn("16QAM", 10.0)
    assert val is not None
    assert 0.0 < val < 0.5


@pytest.mark.parametrize("mod", ["16APSK", "32APSK"])
def test_ber_awgn_apsk_returns_none(mod):
    assert ber_awgn(mod, 10.0) is None


@pytest.mark.parametrize("mod", ["BPSK", "DBPSK", "8PSK", "16QAM"])
def test_ber_awgn_monotone_decreasing(mod):
    vals = [ber_awgn(mod, esn0) for esn0 in range(-2, 20, 2)]
    for a, b in zip(vals, vals[1:]):
        assert a > b, f"{mod}: BER not decreasing"


# ── ebn0_for_ber ─────────────────────────────────────────────────────────────

def test_ebn0_for_ber_inverts_ber_awgn():
    target = 1e-3
    ebn0_db = ebn0_for_ber("BPSK", target)
    assert ebn0_db is not None
    bps    = bits_per_symbol("BPSK")
    esn0   = ebn0_db + 10.0 * math.log10(bps)
    recovered = ber_awgn("BPSK", esn0)
    assert abs(recovered - target) / target < 0.01


def test_ebn0_for_ber_no_formula():
    assert ebn0_for_ber("16APSK", 1e-3) is None


def test_ebn0_for_ber_target_too_high():
    # BER=0.999 is higher than BPSK can produce within the default [-5, 25] dB range
    assert ebn0_for_ber("BPSK", 0.999) is None


def test_ebn0_for_ber_target_too_low():
    # Narrow range ebn0_hi=1 dB → BER_hi ≈ 0.05; target 0.001 < 0.05 → out of range
    assert ebn0_for_ber("BPSK", 0.001, ebn0_lo_db=-2.0, ebn0_hi_db=1.0) is None
