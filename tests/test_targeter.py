"""Tests for sim/targeter.py: BER seeker and implementation-loss reporting."""

from typing import Any

import pytest

from sim.targeter import (
    _n_bits_for_ci, _erfinv, _simulate_ber_at_noise,
    seek_ber_noise_level, seek_all_carriers,
)

# ----------------------------------------------------------------------------
# Shared fixtures
# ----------------------------------------------------------------------------

# Linear (pass-through) amplifier: output = input, no phase rotation.
_LINEAR_AM_AM = {"input": [0.0, 2.0], "output": [0.0, 2.0]}
_LINEAR_AM_PM = {"input": [0.0, 2.0], "phase_deg": [0.0, 0.0]}

# Single BPSK carrier at 1 MHz, 4 sps.  sample_rate = native_rate so L = 1
# (no OLA upsampling overhead), keeping simulation cost low.
_SAMPLE_RATE = 4_000_000
_CARRIER = dict(
    name="tgt",
    symbol_rate=1_000_000,
    sps=4,
    rolloff=0.35,
    filter_span=8,
    num_symbols=100,   # overridden by targeter per run
    power_db=0.0,
    freq=0.0,
    modulation="BPSK",
    sweep_demod=True,
)

# Fast seek parameters: high BER target (lots of errors/run), loose accuracy,
# short OLA blocks, few seeds.
_TARGET_BER = 0.10
_CONFIDENCE = 0.90
_ACCURACY   = 0.03    # loose -> small n_bits_final ~270 bits
_MAX_ITER   = 14
_N_SEEDS    = 2
_OLA_SPAN   = 4
_OLA_BLOCK  = 512

# Noise bracket calibrated for the test carrier.  With sample_rate = native_rate
# (L=1) and a peak-normalised BPSK signal, the operating noise level for BER=0.1
# is approximately -69 dBFS/Hz.  -90 dBFS is essentially silent; -50 dBFS is
# very loud (BER -> 0.5).
_NOISE_LO = -90.0
_NOISE_HI = -50.0


# ----------------------------------------------------------------------------
# Unit test: _n_bits_for_ci
# ----------------------------------------------------------------------------

def test_n_bits_for_ci_sanity():
    # BER=0.1, 95% CI, +-0.01 absolute accuracy
    # z ~ 1.960, n = ceil((1.960/0.01)^2 * 0.1 * 0.9) ~ 3457
    n = _n_bits_for_ci(0.1, confidence=0.95, accuracy=0.01)
    assert isinstance(n, int)
    assert 3000 < n < 4000

    # Higher confidence → more bits
    assert _n_bits_for_ci(0.1, 0.99, 0.01) > _n_bits_for_ci(0.1, 0.95, 0.01)

    # Finer accuracy → more bits (quadratic: (1/accuracy)^2)
    assert _n_bits_for_ci(0.1, 0.95, 0.005) > _n_bits_for_ci(0.1, 0.95, 0.01)

    # Higher BER (up to 0.5) needs more bits than very low BER at the same
    # absolute accuracy (because p*(1-p) is larger near p=0.1 than near p=0.01)
    assert _n_bits_for_ci(0.1, 0.95, 0.01) > _n_bits_for_ci(0.01, 0.95, 0.01)


# ----------------------------------------------------------------------------
# Integration test: seek converges to target BER
# ----------------------------------------------------------------------------

def test_seek_ber_convergence():
    """Bisection reaches a noise level where measured BER is near target_ber."""
    result = seek_ber_noise_level(
        target_ber=_TARGET_BER,
        confidence=_CONFIDENCE,
        ber_accuracy=_ACCURACY,
        carrier_name="tgt",
        carriers=[_CARRIER],
        sample_rate=_SAMPLE_RATE,
        am_am_cfg=_LINEAR_AM_AM,
        am_pm_cfg=_LINEAR_AM_PM,
        noise_lo_dbfs=_NOISE_LO,
        noise_hi_dbfs=_NOISE_HI,
        ola_filter_span=_OLA_SPAN,
        ola_block_size=_OLA_BLOCK,
        max_iter=_MAX_ITER,
        n_final_seeds=_N_SEEDS,
        seed=0,
    )

    # Result dict has all expected keys
    expected_keys = {
        "noise_density_dbfs", "ber", "ber_ci_lo", "ber_ci_hi",
        "effective_ebn0_db", "theory_ebn0_db", "implementation_loss_db",
        "cnr_db", "cir_db", "cnir_db", "n_bits_total", "n_iter",
    }
    assert expected_keys.issubset(result.keys())

    # BER should be within ±0.08 of target (loose — small bit count, statistical scatter)
    assert abs(result["ber"] - _TARGET_BER) < 0.08, (
        f"BER {result['ber']:.4f} too far from target {_TARGET_BER}"
    )

    # Confidence interval is well-ordered
    assert result["ber_ci_lo"] <= result["ber"] <= result["ber_ci_hi"]

    # Bisection stayed within the allowed step count
    assert result["n_iter"] <= _MAX_ITER

    # Bits used is positive
    assert result["n_bits_total"] > 0


# ----------------------------------------------------------------------------
# Integration test: linear amplifier → near-zero implementation loss
# ----------------------------------------------------------------------------

def test_linear_amplifier_zero_implementation_loss():
    """
    With a pass-through amplifier (no distortion) and a single carrier,
    implementation_loss_db should be near zero: effective Eb/N0 from the
    C/(N+I) power projection should match the theory Eb/N0 for the same BER.
    """
    result = seek_ber_noise_level(
        target_ber=_TARGET_BER,
        confidence=_CONFIDENCE,
        ber_accuracy=_ACCURACY,
        carrier_name="tgt",
        carriers=[_CARRIER],
        sample_rate=_SAMPLE_RATE,
        am_am_cfg=_LINEAR_AM_AM,
        am_pm_cfg=_LINEAR_AM_PM,
        noise_lo_dbfs=_NOISE_LO,
        noise_hi_dbfs=_NOISE_HI,
        ola_filter_span=_OLA_SPAN,
        ola_block_size=_OLA_BLOCK,
        max_iter=_MAX_ITER,
        n_final_seeds=_N_SEEDS,
        seed=1,
    )

    assert result["implementation_loss_db"] is not None, (
        "BPSK should have a theory formula; implementation_loss_db must not be None"
    )

    # 3 dB tolerance: accounts for statistical noise from the small bit budget
    # and any power-level calibration offset in the CNIR computation.
    assert abs(result["implementation_loss_db"]) < 3.0, (
        f"Implementation loss {result['implementation_loss_db']:.2f} dB is too large "
        f"for a linear, single-carrier setup"
    )


# ----------------------------------------------------------------------------
# Error test: invalid bracket raises ValueError
# ----------------------------------------------------------------------------

def test_invalid_bracket_lo_too_noisy():
    """
    If even the quietest bracket endpoint gives BER > target, raise ValueError.
    Both endpoints are in the very-high-noise region for the test carrier
    (~-55 dBFS gives BER ~0.4 >> target=0.001).
    """
    with pytest.raises(ValueError, match="noise_lo_dbfs"):
        seek_ber_noise_level(
            target_ber=0.001,
            confidence=0.90,
            ber_accuracy=0.001,
            carrier_name="tgt",
            carriers=[_CARRIER],
            sample_rate=_SAMPLE_RATE,
            am_am_cfg=_LINEAR_AM_AM,
            am_pm_cfg=_LINEAR_AM_PM,
            noise_lo_dbfs=-55.0,   # very loud for this carrier -> BER ~0.4 >> 0.001
            noise_hi_dbfs=-50.0,
            ola_filter_span=_OLA_SPAN,
            ola_block_size=_OLA_BLOCK,
            max_iter=4,
            n_final_seeds=1,
            seed=0,
        )


def test_invalid_bracket_hi_too_quiet():
    """
    If even the loudest bracket endpoint gives BER < target, raise ValueError.
    Both endpoints here are very quiet (low noise_dbfs), so BER << target.
    """
    with pytest.raises(ValueError, match="noise_hi_dbfs"):
        seek_ber_noise_level(
            target_ber=0.40,
            confidence=0.90,
            ber_accuracy=0.05,
            carrier_name="tgt",
            carriers=[_CARRIER],
            sample_rate=_SAMPLE_RATE,
            am_am_cfg=_LINEAR_AM_AM,
            am_pm_cfg=_LINEAR_AM_PM,
            noise_lo_dbfs=-200.0,  # essentially silent → BER ≈ 0
            noise_hi_dbfs=-200.0,  # also silent → BER << 0.40
            ola_filter_span=_OLA_SPAN,
            ola_block_size=_OLA_BLOCK,
            max_iter=4,
            n_final_seeds=1,
            seed=0,
        )


# ----------------------------------------------------------------------------
# Unit tests: _erfinv edge cases
# ----------------------------------------------------------------------------

def test_erfinv_zero():
    assert _erfinv(0.0) == 0.0


def test_erfinv_negative_symmetric():
    import math
    pos = _erfinv(0.5)
    neg = _erfinv(-0.5)
    assert abs(neg + pos) < 1e-9
    assert abs(math.erf(pos) - 0.5) < 1e-9


# ----------------------------------------------------------------------------
# Error paths: unknown carrier name
# ----------------------------------------------------------------------------

def test_simulate_ber_at_noise_bad_carrier():
    with pytest.raises(ValueError, match="not found"):
        _simulate_ber_at_noise(
            noise_dbfs=-70.0,
            carrier_name="missing",
            carriers=[_CARRIER],
            sample_rate=_SAMPLE_RATE,
            am_am_cfg=_LINEAR_AM_AM,
            am_pm_cfg=_LINEAR_AM_PM,
            input_backoff_db=0.0,
            ola_filter_span=_OLA_SPAN,
            ola_block_size=_OLA_BLOCK,
            n_bits=100,
            seeds=[0],
        )


def test_seek_ber_bad_carrier():
    with pytest.raises(ValueError, match="not found"):
        seek_ber_noise_level(
            target_ber=_TARGET_BER,
            confidence=_CONFIDENCE,
            ber_accuracy=_ACCURACY,
            carrier_name="missing",
            carriers=[_CARRIER],
            sample_rate=_SAMPLE_RATE,
            am_am_cfg=_LINEAR_AM_AM,
            am_pm_cfg=_LINEAR_AM_PM,
            noise_lo_dbfs=_NOISE_LO,
            noise_hi_dbfs=_NOISE_HI,
            ola_filter_span=_OLA_SPAN,
            ola_block_size=_OLA_BLOCK,
            max_iter=4,
            n_final_seeds=1,
            seed=0,
        )


# ----------------------------------------------------------------------------
# Progress callback is invoked
# ----------------------------------------------------------------------------

def test_seek_ber_progress_callback():
    calls: list[float] = []
    seek_ber_noise_level(
        target_ber=_TARGET_BER,
        confidence=_CONFIDENCE,
        ber_accuracy=_ACCURACY,
        carrier_name="tgt",
        carriers=[_CARRIER],
        sample_rate=_SAMPLE_RATE,
        am_am_cfg=_LINEAR_AM_AM,
        am_pm_cfg=_LINEAR_AM_PM,
        noise_lo_dbfs=_NOISE_LO,
        noise_hi_dbfs=_NOISE_HI,
        ola_filter_span=_OLA_SPAN,
        ola_block_size=_OLA_BLOCK,
        max_iter=_MAX_ITER,
        n_final_seeds=_N_SEEDS,
        seed=2,
        progress_callback=lambda frac, _msg: calls.append(frac),
    )
    assert len(calls) > 0
    assert all(0.0 <= f <= 1.0 for f in calls)


# ----------------------------------------------------------------------------
# seek_all_carriers: filtering and result shape
# ----------------------------------------------------------------------------

_SEEK_CARRIER = dict(_CARRIER, use_seeker=True)

_SEEK_KWARGS: dict[str, Any] = dict(
    sample_rate=_SAMPLE_RATE,
    am_am_cfg=_LINEAR_AM_AM,
    am_pm_cfg=_LINEAR_AM_PM,
    ola_filter_span=_OLA_SPAN,
    ola_block_size=_OLA_BLOCK,
    noise_lo_dbfs=_NOISE_LO,
    noise_hi_dbfs=_NOISE_HI,
    max_iter=_MAX_ITER,
    n_final_seeds=_N_SEEDS,
    seed=3,
    target_ber=_TARGET_BER,
    confidence=_CONFIDENCE,
    ber_accuracy=_ACCURACY,
)


def test_seek_all_carriers_empty_list():
    assert seek_all_carriers(carriers=[], **_SEEK_KWARGS) == {}


def test_seek_all_carriers_skips_non_seekable():
    non_seekable = [
        dict(_CARRIER, name="c1", sweep_demod=False, use_seeker=True),
        dict(_CARRIER, name="c2", sweep_demod=True,  use_seeker=False),
        dict(_CARRIER, name="c3", enabled=False,     sweep_demod=True, use_seeker=True),
    ]
    assert seek_all_carriers(carriers=non_seekable, **_SEEK_KWARGS) == {}


def test_seek_all_carriers_single_seekable():
    cb_calls: list[float] = []
    result = seek_all_carriers(
        carriers=[_SEEK_CARRIER],
        progress_callback=lambda f, _m: cb_calls.append(f),
        **_SEEK_KWARGS,
    )
    assert set(result.keys()) == {"tgt"}
    assert "ber" in result["tgt"]
    assert len(cb_calls) > 0
