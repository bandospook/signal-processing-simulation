"""Unit tests for sim.theory: ber_awgn and ebn0_for_ber."""
import math

import numpy as np
import pytest

from sim import theory
from sim.theory import ber_awgn, ebn0_for_ber
from sim.modulation import bits_per_symbol


@pytest.fixture(autouse=True)
def _clear_theory_cache():
    """Reset the npz cache between tests so monkeypatched dirs take effect."""
    theory._TABLE_CACHE.clear()
    yield
    theory._TABLE_CACHE.clear()


# ── ber_awgn ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("mod", ["BPSK", "QPSK", "OQPSK", "MSK"])
def test_ber_awgn_bpsk_family(mod):
    """Ber awgn bpsk family."""
    val = ber_awgn(mod, 0.0)
    assert val is not None
    assert 0.0 < val < 0.5


def test_ber_awgn_dbpsk_worse_than_bpsk():
    """Ber awgn dbpsk worse than bpsk."""
    esn0 = 3.0
    dbpsk = ber_awgn("DBPSK", esn0)
    bpsk  = ber_awgn("BPSK",  esn0)
    assert dbpsk is not None and bpsk is not None and 0.0 < dbpsk < 0.5
    assert dbpsk > bpsk


def test_ber_awgn_8psk():
    """Ber awgn 8psk."""
    val = ber_awgn("8PSK", 8.0)
    assert val is not None
    assert 0.0 < val < 0.5


def test_ber_awgn_16qam():
    """Ber awgn 16qam."""
    val = ber_awgn("16QAM", 10.0)
    assert val is not None
    assert 0.0 < val < 0.5


@pytest.mark.parametrize("mod", ["BPSK", "DBPSK", "8PSK", "16QAM"])
def test_ber_awgn_monotone_decreasing(mod):
    """Ber awgn monotone decreasing."""
    vals = [ber_awgn(mod, esn0) for esn0 in range(-2, 20, 2)]
    for a, b in zip(vals, vals[1:]):
        assert a is not None and b is not None, f"{mod}: unexpected None from ber_awgn"
        assert a > b, f"{mod}: BER not decreasing"


# ── APSK table lookup ────────────────────────────────────────────────────────

def _write_table(tmp_path, mod, ebn0_grid, ber_grid, **gammas):
    """Write a synthetic ber_awgn_<MOD>.npz under tmp_path/data/theory."""
    dst = tmp_path / "data" / "theory"
    dst.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "ebn0_db":       np.asarray(ebn0_grid, dtype=np.float64),
        "ber":           np.asarray(ber_grid,  dtype=np.float64),
        "n_bits":        np.full(len(ebn0_grid), 1_000_000, dtype=np.int64),
        "n_errors":      (np.asarray(ber_grid) * 1_000_000).astype(np.int64),
        "ci_half_width": np.full(len(ebn0_grid), 1e-4, dtype=np.float64),
        "modulation":    np.array(mod),
        "confidence":    np.array(0.95),
        "target_ci_rel": np.array(0.05),
        "generated_at":  np.array("2026-05-24T00:00:00+00:00"),
    }
    for k, v in gammas.items():
        arrays[k] = np.array(v)
    # numpy's stub for savez incorrectly types the **kwds spread as allow_pickle.
    np.savez(dst / f"ber_awgn_{mod}.npz", **arrays)  # type: ignore[arg-type]
    return dst


def test_ber_awgn_apsk_uses_table(tmp_path, monkeypatch):
    """ber_awgn looks up the npz table for APSK and returns the table value."""
    _write_table(tmp_path, "16APSK",
                 ebn0_grid=[6.0, 8.0, 10.0, 12.0],
                 ber_grid=[1e-2, 1e-3, 1e-4, 1e-5],
                 apsk_gamma=2.57)
    monkeypatch.setattr(theory, "_TABLE_DIR", tmp_path / "data" / "theory")
    # Es/N0 = Eb/N0 + 10·log10(4) at the 10 dB Eb/N0 sample (table BER = 1e-4).
    val = ber_awgn("16APSK", 10.0 + 10.0 * math.log10(4))
    assert val is not None
    assert math.isclose(val, 1e-4, rel_tol=1e-9)


def test_ber_awgn_apsk_no_table_returns_none(tmp_path, monkeypatch):
    """No npz on disk → APSK lookup returns None (same as before the feature)."""
    monkeypatch.setattr(theory, "_TABLE_DIR", tmp_path / "data" / "theory_empty")
    assert ber_awgn("16APSK", 10.0) is None


def test_ber_awgn_apsk_non_default_gamma_returns_none(tmp_path, monkeypatch):
    """User-overridden gammas suppress the table lookup."""
    _write_table(tmp_path, "16APSK",
                 ebn0_grid=[6.0, 8.0, 10.0],
                 ber_grid=[1e-2, 1e-3, 1e-4],
                 apsk_gamma=2.57)
    monkeypatch.setattr(theory, "_TABLE_DIR", tmp_path / "data" / "theory")
    assert ber_awgn("16APSK", 10.0,
                    mod_kwargs={"apsk_gamma": 3.0}) is None


def test_ber_awgn_apsk_extrapolates_below_table(tmp_path, monkeypatch):
    """Below the table edge, extrapolation in log10(BER) extends the slope."""
    _write_table(tmp_path, "16APSK",
                 ebn0_grid=[6.0, 8.0, 10.0, 12.0],
                 ber_grid=[1e-2, 1e-3, 1e-4, 1e-5],
                 apsk_gamma=2.57)
    monkeypatch.setattr(theory, "_TABLE_DIR", tmp_path / "data" / "theory")
    # Slope is -1/2 in log10(BER) per dB (one decade per 2 dB).  At 4 dB
    # (2 dB below the table edge of 6 dB) we extrapolate log10(BER) by +1,
    # i.e. from -2 to -1 → BER = 0.1.
    bps = bits_per_symbol("16APSK")
    val = ber_awgn("16APSK", 4.0 + 10.0 * math.log10(bps))
    assert val is not None
    assert math.isclose(val, 0.1, rel_tol=1e-6)


def test_ber_awgn_apsk_extrapolates_above_table(tmp_path, monkeypatch):
    """Beyond the table edge, extrapolation extends the slope (deeper BER)."""
    _write_table(tmp_path, "16APSK",
                 ebn0_grid=[6.0, 8.0, 10.0, 12.0],
                 ber_grid=[1e-2, 1e-3, 1e-4, 1e-5],
                 apsk_gamma=2.57)
    monkeypatch.setattr(theory, "_TABLE_DIR", tmp_path / "data" / "theory")
    bps = bits_per_symbol("16APSK")
    # At 14 dB (2 dB above table) extrapolation gives log10(BER) = -6 → 1e-6.
    val = ber_awgn("16APSK", 14.0 + 10.0 * math.log10(bps))
    assert val is not None
    assert math.isclose(val, 1e-6, rel_tol=1e-6)


# ── ebn0_for_ber ─────────────────────────────────────────────────────────────

def test_ebn0_for_ber_inverts_ber_awgn():
    """Ebn0 for ber inverts ber awgn."""
    target = 1e-3
    ebn0_db = ebn0_for_ber("BPSK", target)
    assert ebn0_db is not None
    bps    = bits_per_symbol("BPSK")
    esn0   = ebn0_db + 10.0 * math.log10(bps)
    recovered = ber_awgn("BPSK", esn0)
    assert recovered is not None
    assert abs(recovered - target) / target < 0.01


def test_ebn0_for_ber_no_formula_no_table(tmp_path, monkeypatch):
    """APSK without a npz table → None (pre-feature behavior preserved)."""
    monkeypatch.setattr(theory, "_TABLE_DIR", tmp_path / "data" / "theory_empty")
    assert ebn0_for_ber("16APSK", 1e-3) is None


def test_ebn0_for_ber_inverts_apsk_table(tmp_path, monkeypatch):
    """The table-based inversion round-trips with ber_awgn."""
    _write_table(tmp_path, "16APSK",
                 ebn0_grid=[6.0, 8.0, 10.0, 12.0],
                 ber_grid=[1e-2, 1e-3, 1e-4, 1e-5],
                 apsk_gamma=2.57)
    monkeypatch.setattr(theory, "_TABLE_DIR", tmp_path / "data" / "theory")
    bps = bits_per_symbol("16APSK")
    target = 1e-3
    ebn0_db = ebn0_for_ber("16APSK", target)
    assert ebn0_db is not None
    recovered = ber_awgn("16APSK", ebn0_db + 10.0 * math.log10(bps))
    assert recovered is not None
    assert math.isclose(recovered, target, rel_tol=1e-6)


def test_ebn0_for_ber_apsk_extrapolates_below_table(tmp_path, monkeypatch):
    """Targets deeper than the table extrapolate linearly in log10(BER)."""
    _write_table(tmp_path, "16APSK",
                 ebn0_grid=[6.0, 8.0, 10.0, 12.0],
                 ber_grid=[1e-2, 1e-3, 1e-4, 1e-5],
                 apsk_gamma=2.57)
    monkeypatch.setattr(theory, "_TABLE_DIR", tmp_path / "data" / "theory")
    # Slope is +2 dB per decade of BER; target 1e-6 sits one decade below
    # the deepest table point (1e-5 at Eb/N0=12) → Eb/N0 = 14 dB.
    ebn0_db = ebn0_for_ber("16APSK", 1e-6)
    assert ebn0_db is not None
    assert math.isclose(ebn0_db, 14.0, abs_tol=1e-6)


def test_ebn0_for_ber_apsk_non_default_gamma_returns_none(tmp_path, monkeypatch):
    """Non-default gammas suppress the table-based inversion too."""
    _write_table(tmp_path, "16APSK",
                 ebn0_grid=[6.0, 8.0, 10.0],
                 ber_grid=[1e-2, 1e-3, 1e-4],
                 apsk_gamma=2.57)
    monkeypatch.setattr(theory, "_TABLE_DIR", tmp_path / "data" / "theory")
    assert ebn0_for_ber("16APSK", 1e-3,
                        mod_kwargs={"apsk_gamma": 3.0}) is None


def test_table_cache_hits_after_first_load(tmp_path, monkeypatch):
    """Loader caches both the positive and the negative result."""
    monkeypatch.setattr(theory, "_TABLE_DIR", tmp_path / "missing_dir")
    assert theory._load_table("16APSK") is None
    # Now point at a different dir; the cached None should still be returned.
    monkeypatch.setattr(theory, "_TABLE_DIR", tmp_path / "another_missing")
    assert theory._load_table("16APSK") is None


def test_ebn0_for_ber_target_too_high():
    """Ebn0 for ber target too high."""
    # BER=0.999 is higher than BPSK can produce within the default [-5, 25] dB range
    assert ebn0_for_ber("BPSK", 0.999) is None


def test_ebn0_for_ber_target_too_low():
    """Ebn0 for ber target too low."""
    # Narrow range ebn0_hi=1 dB → BER_hi ≈ 0.05; target 0.001 < 0.05 → out of range
    assert ebn0_for_ber("BPSK", 0.001, ebn0_lo_db=-2.0, ebn0_hi_db=1.0) is None
