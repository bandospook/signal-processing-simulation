"""Unit tests for sim.phase_noise: mask interpolation, PSD shape, BER impact."""
import math

import numpy as np
import pytest

from sim.phase_noise import (
    apply_phase_noise,
    generate_phase_noise,
    interp_dbc_mask,
)


# ── interp_dbc_mask ──────────────────────────────────────────────────────────

def test_interp_dbc_mask_at_anchor_points():
    """Mask evaluated at its own anchor offsets returns those anchor values."""
    offsets = [1e2, 1e4, 1e6]
    dbc     = [-60, -100, -140]
    got = interp_dbc_mask(np.array(offsets), offsets, dbc)
    np.testing.assert_allclose(got, dbc)


def test_interp_dbc_mask_log_log_interpolation():
    """Halfway between two log-spaced anchors gives the dB average."""
    # Anchors one decade apart: midpoint in log10 is geometric mean of offsets.
    got = float(interp_dbc_mask(np.array([1e3]), [1e2, 1e4], [-60, -100])[0])
    assert math.isclose(got, -80.0, abs_tol=1e-12)


def test_interp_dbc_mask_flat_extrapolation():
    """Beyond either end of the anchor grid the value clamps to the edge."""
    offsets = [1e3, 1e5]
    dbc     = [-70, -120]
    above = float(interp_dbc_mask(np.array([1e7]), offsets, dbc)[0])
    below = float(interp_dbc_mask(np.array([1e0]), offsets, dbc)[0])
    assert above == dbc[-1]
    assert below == dbc[0]


def test_interp_dbc_mask_accepts_unsorted_offsets():
    """Anchors can be passed in any order; results match the sorted form."""
    sorted_val   = interp_dbc_mask(np.array([1e3]), [1e2, 1e4], [-60, -100])
    unsorted_val = interp_dbc_mask(np.array([1e3]), [1e4, 1e2], [-100, -60])
    np.testing.assert_allclose(unsorted_val, sorted_val)


def test_interp_dbc_mask_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        interp_dbc_mask(np.array([1e3]), [1e2, 1e4], [-60])


def test_interp_dbc_mask_rejects_nonpositive_offsets():
    with pytest.raises(ValueError):
        interp_dbc_mask(np.array([1e3]), [0.0, 1e4], [-60, -100])


# ── generate_phase_noise ─────────────────────────────────────────────────────

def test_generate_phase_noise_variance_matches_integrated_psd():
    """The realized phase noise variance is within a few % of the analytical
    σ²_φ = ∫ S_φ(f) df = 2·∫_0^{fs/2} S_φ(f) df."""
    fs = 1e6
    n  = 1 << 17                        # 131072 samples for tight sample-variance CI
    # Flat mask: L(f) = -60 dBc/Hz over [10 Hz, fs/2 + slack].  S_φ = 2e-6 rad²/Hz.
    offsets = [1.0, fs]
    dbc     = [-60.0, -60.0]
    rng = np.random.default_rng(0)
    phi = generate_phase_noise(n, fs, offsets, dbc, rng)
    expected_var = 2.0 * 10 ** (-60.0 / 10.0) * (fs / 2.0) * 2.0  # 2·∫_0^{fs/2}
    measured_var = float(np.var(phi))
    # Spectral coloring + finite sample → expect within ~5% on 131072 samples.
    assert math.isclose(measured_var, expected_var, rel_tol=0.05)


def test_generate_phase_noise_rejects_short_arrays():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        generate_phase_noise(1, 1e6, [1e3], [-100.0], rng)


def test_generate_phase_noise_is_reproducible():
    """Same RNG seed → identical phase noise samples."""
    fs = 1e6
    n  = 4096
    p1 = generate_phase_noise(n, fs, [1e3, 1e5], [-80, -120],
                              np.random.default_rng(7))
    p2 = generate_phase_noise(n, fs, [1e3, 1e5], [-80, -120],
                              np.random.default_rng(7))
    np.testing.assert_array_equal(p1, p2)


# ── apply_phase_noise ────────────────────────────────────────────────────────

def test_apply_phase_noise_preserves_envelope():
    """exp(j·φ) is unit magnitude → |signal| is unchanged sample-by-sample."""
    fs = 1e6
    sig = np.exp(2j * np.pi * 1e4 * np.arange(2048) / fs)   # unit-mag tone
    out = apply_phase_noise(sig, fs, [1e2, 1e5], [-60, -120],
                            np.random.default_rng(0))
    np.testing.assert_allclose(np.abs(out), np.abs(sig), atol=1e-12)


def test_apply_phase_noise_rotates_phase():
    """Quiet mask → near-identity; loud mask → meaningful phase departure."""
    fs = 1e6
    sig = np.ones(4096, dtype=complex)
    quiet = apply_phase_noise(sig, fs, [1e2, 1e5], [-160, -180],
                              np.random.default_rng(0))
    loud  = apply_phase_noise(sig, fs, [1e2, 1e5], [-40, -60],
                              np.random.default_rng(0))
    assert float(np.std(np.angle(quiet))) < 0.001
    assert float(np.std(np.angle(loud)))  > 0.05
