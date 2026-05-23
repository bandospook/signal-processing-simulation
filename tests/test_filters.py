"""Tests for OLA convolution, upsampling, downsampling, and channel impairment."""
import numpy as np
from scipy.signal import resample_poly
from sim.filters import (ola_convolve, fft_ola_upsample, fft_ola_downsample,
                          apply_channel_impairment, rrc_coeffs)


# ── ola_convolve ──────────────────────────────────────────────────────────────

def test_ola_convolve_matches_numpy():
    """OLA convolution must produce the same result as np.convolve (linear conv)."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(200) + 1j * rng.standard_normal(200)
    h = rng.standard_normal(17).astype(float)
    assert np.allclose(ola_convolve(x, h, block_size=32), np.convolve(x, h), atol=1e-10)


# ── fft_ola_upsample ──────────────────────────────────────────────────────────

def test_upsample_factor_1_is_identity():
    x = np.array([1+0j, 2+0j, 3+0j, 4+0j])
    assert np.allclose(fft_ola_upsample(x, 1), x)


def test_upsample_output_length():
    assert len(fft_ola_upsample(np.ones(64, dtype=complex), 4)) == 256


def test_upsample_image_rejection():
    """
    After upsampling by L, spectral images at f0 + k*fs_orig (k≥1) must be
    attenuated by at least 40 dB relative to the signal.
    The N-point FFT of x_up has signal at bin round(f0*N) and the first image
    at bin round((f0+1)*N); the filter should suppress the image.
    """
    N, L, f0 = 1024, 4, 0.10
    x = np.exp(1j * 2*np.pi*f0 * np.arange(N)).astype(complex)
    X = np.abs(np.fft.fft(fft_ola_upsample(x, L, filter_span=16)))
    sig_bin   = int(round(f0 * N))
    image_bin = sig_bin + N          # first image at f0 + fs_orig (in N*L-bin FFT)
    assert X[image_bin]**2 / (X[sig_bin]**2 + 1e-30) < 1e-4


# ── fft_ola_downsample ────────────────────────────────────────────────────────

def test_downsample_factor_1_is_identity():
    x = np.array([1+0j, 2+0j, 3+0j, 4+0j])
    assert np.allclose(fft_ola_downsample(x, 1), x)


def test_downsample_output_length():
    assert len(fft_ola_downsample(np.ones(64, dtype=complex), 4)) == 16


def test_downsample_alias_rejection():
    """
    A tone above the new Nyquist (fs_orig / (2*L)) must be attenuated >35 dB
    after downsampling.
    """
    N, L = 1024, 4          # new Nyquist = 0.5/4 = 0.125 (normalised)
    f_alias = 0.30           # well above new Nyquist, below original Nyquist
    x = np.exp(1j * 2*np.pi*f_alias * np.arange(N)).astype(complex)
    out = fft_ola_downsample(x, L, filter_span=16)
    assert np.mean(np.abs(out)**2) / np.mean(np.abs(x)**2) < 3e-4


def test_upsample_downsample_roundtrip():
    """
    Upsample by L then downsample by L must recover a band-limited signal.

    A broadband random signal contains content near the Nyquist where the
    Kaiser-sinc filter rolls off, so we use a signal limited to the bottom
    12.5% of the spectrum where the filter is flat.
    """
    rng = np.random.default_rng(7)
    N, L = 256, 4
    X = np.zeros(N, dtype=complex)
    n_keep = N // 8
    X[:n_keep] = rng.standard_normal(n_keep) + 1j * rng.standard_normal(n_keep)
    x = np.fft.ifft(X)
    x_rec = fft_ola_downsample(fft_ola_upsample(x, L, filter_span=8), L, filter_span=8)
    m = N // 8
    assert np.allclose(x_rec[m:-m], x[m:-m], atol=1e-3)


# ── apply_channel_impairment ──────────────────────────────────────────────────

def test_channel_disabled_is_identity():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(256) + 1j * rng.standard_normal(256)
    out = apply_channel_impairment(x, sample_rate=1e6, signal_bw=0.4e6,
                                   channel_cfg={"enabled": False})
    assert np.allclose(out, x, atol=1e-12)


def test_zero_ripple_zero_phase_is_identity():
    """With H(f)=1 everywhere the output must equal the input."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal(512) + 1j * rng.standard_normal(512)
    cfg = dict(enabled=True, ripple_db=0.0, ripple_cycles=1.0,
               max_phase_dev_deg=0.0, phase_poly_order=2)
    out = apply_channel_impairment(x, sample_rate=1e6, signal_bw=0.4e6, channel_cfg=cfg)
    assert np.allclose(out, x, atol=1e-10)


def test_ripple_changes_in_band_signal():
    """Ripple channel must modify the in-band signal (output ≠ input)."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal(1024) + 1j * rng.standard_normal(1024)
    cfg = dict(enabled=True, ripple_db=3.0, ripple_cycles=2.0,
               max_phase_dev_deg=0.0, phase_poly_order=2)
    out = apply_channel_impairment(x, sample_rate=1e6, signal_bw=0.4e6, channel_cfg=cfg)
    diff_power = np.mean(np.abs(out - x)**2)
    signal_power = np.mean(np.abs(x)**2)
    assert diff_power / signal_power > 0.001


def test_pure_phase_preserves_amplitude():
    """Pure phase channel must not change the amplitude of the signal."""
    N = 1024
    fs, bw = 1e6, 0.4e6
    f_in_band = 0.05e6
    x = np.exp(1j * 2*np.pi * f_in_band/fs * np.arange(N)).astype(complex)
    cfg = dict(enabled=True, ripple_db=0.0, ripple_cycles=0.0,
               max_phase_dev_deg=15.0, phase_poly_order=2)
    out = apply_channel_impairment(x, fs, bw, cfg)
    m = N // 8
    assert np.allclose(np.abs(out[m:-m]), 1.0, atol=0.05)
    assert not np.allclose(np.angle(out[m:-m]), np.angle(x[m:-m]), atol=0.01)


def test_pure_phase_does_not_change_out_of_band():
    """
    Out-of-band content is left untouched (H=1 outside the signal bandwidth).
    Use a tone well above bw/2; its power should survive downsampling unchanged.
    """
    N = 512
    fs, bw = 1e6, 0.2e6
    f_oob = 0.4e6   # outside bw/2=0.1 MHz, inside fs/2=0.5 MHz
    x = np.exp(1j * 2*np.pi * f_oob/fs * np.arange(N)).astype(complex)
    cfg = dict(enabled=True, ripple_db=2.0, ripple_cycles=2.0,
               max_phase_dev_deg=10.0, phase_poly_order=2)
    out = apply_channel_impairment(x, fs, bw, cfg)
    # Amplitude should be 1 everywhere (H=1 out of band)
    assert np.allclose(np.abs(out[N//4:-N//4]), 1.0, atol=0.01)


# ── resample_poly (scipy) ─────────────────────────────────────────────────────

def test_resample_poly_identity():
    """P == Q must return the input unchanged."""
    x = np.array([1+0j, 2+1j, 3-1j, 4+0j])
    assert np.allclose(resample_poly(x, 4, 4), x)


def test_resample_poly_upsample_2_samples():
    """After upsampling by 2, even-indexed output samples must match the original."""
    N = 256
    f = 0.10   # normalised frequency, well below anti-alias cutoff
    x = np.exp(1j * 2*np.pi*f * np.arange(N)).astype(complex)
    y = resample_poly(x, 2, 1)
    m = 16   # skip filter transient
    assert np.allclose(y[2*m : 2*(m + 100) : 2], x[m : m + 100], atol=1e-3)


def test_resample_poly_roundtrip_3_2():
    """
    Roundtrip by 3/2 then 2/3 must recover a band-limited signal.
    """
    rng = np.random.default_rng(42)
    N = 512
    X = np.zeros(N, dtype=complex)
    X[:N // 8] = rng.standard_normal(N // 8) + 1j * rng.standard_normal(N // 8)
    x = np.fft.ifft(X)
    y  = resample_poly(x, 3, 2)
    xr = resample_poly(y, 2, 3).astype(complex)[:N]
    m = N // 8
    assert np.allclose(xr[m:-m], x[m:-m], atol=1e-3)


def test_rrc_singularity_rolloff_025():
    """rolloff=0.25 puts the Nyquist singularity at t=1.0; with sps=4 that sample
    is always present, exercising the special-case branch in rrc_coeffs."""
    h = rrc_coeffs(filter_span=2, rolloff=0.25, sps=4)
    assert len(h) > 0
    assert np.all(np.isfinite(h))


def test_ola_convolve_chunk_cb():
    """chunk_cb is invoked at least once and reports done == total on the last call."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(500) + 1j * rng.standard_normal(500)
    h = rng.standard_normal(17).astype(float)
    calls: list[tuple[int, int]] = []
    ola_convolve(x, h, block_size=64,
                 chunk_cb=lambda done, total: calls.append((done, total)))
    assert len(calls) > 0
    assert calls[-1][0] == calls[-1][1]


def test_rational_resample_no_fractional_L_error():
    """
    A wideband_bpsk_simulation with L_float=62.5 must not raise ValueError.

    sample_rate=500e6, symbol_rate=2e6, sps=4 → L_float=62.5.
    """
    from sim.simulation import wideband_bpsk_simulation
    _am_am = {"input":  [0.0, 0.5, 1.0], "output": [0.0, 0.45, 0.85]}
    _am_pm = {"input":  [0.0, 0.5, 1.0], "phase_deg": [0.0, 1.0, 3.0]}
    carrier = dict(
        name="fast", symbol_rate=2e6, sps=4, rolloff=0.35,
        filter_span=8, power_db=0.0, freq=0.0,
        modulation="BPSK",
    )
    result = wideband_bpsk_simulation(
        carriers=[carrier],
        sample_rate=500e6,
        am_am_cfg=_am_am,
        am_pm_cfg=_am_pm,
        max_block_size_samples=1200,    # 300 symbols × sps=4
        input_backoff_db=6.0,
        ola_filter_span=8,
        seed=0,
    )
    cr = result["carriers"][0]
    assert cr["ber"] is not None
    assert 0.0 <= cr["ber"] <= 1.0
