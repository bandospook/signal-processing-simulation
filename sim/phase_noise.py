"""Per-carrier phase noise model.

The user specifies the phase-noise spec as a point-by-point mask of
single-sideband phase noise power, L(f), in dBc/Hz at a list of frequency
offsets from the carrier.  This module interpolates that mask in log-log
space (linear in log10(offset_Hz), linear in dBc/Hz), generates a real-valued
phase fluctuation sequence φ[n] at the per-carrier native sample rate whose
PSD matches the mask, and multiplies the complex baseband by exp(j·φ[n]).

The mask is flat-extrapolated past either end (last value held), per the
phase-noise datasheet convention.  Phase noise is applied at the per-carrier
native sample rate, immediately after the channel-impairment filter and
before the OLA upsample / wideband composite stage, so the spectral shape is
defined in the carrier's own baseband bandwidth.
"""

import numpy as np


def interp_dbc_mask(freq_hz, offsets_hz, dbc_per_hz) -> np.ndarray:
    """Log-log interpolate a phase-noise mask at the requested frequencies.

    Parameters
    ----------
    freq_hz : array-like or float
        Frequency offsets (Hz, > 0) where the mask should be evaluated.
    offsets_hz : array-like
        Mask anchor offsets (Hz, > 0).
    dbc_per_hz : array-like
        L(f) values at each anchor offset, in dBc/Hz.

    Returns
    -------
    np.ndarray
        L(f) in dBc/Hz at each requested frequency.  Flat extrapolation
        beyond either end of the anchor grid (last value held).

    """
    o = np.asarray(offsets_hz, dtype=float)
    d = np.asarray(dbc_per_hz, dtype=float)
    if o.size < 1 or d.size != o.size:
        raise ValueError(
            f"offset_hz ({o.size}) and dbc_per_hz ({d.size}) must have "
            f"the same length >= 1")
    if np.any(o <= 0):
        raise ValueError("All offset_hz values must be > 0 (log-log scale).")
    # Sort by offset so np.interp's flat-edge behavior matches the spec.
    order = np.argsort(o)
    o_s = o[order]
    d_s = d[order]
    f = np.asarray(freq_hz, dtype=float)
    # Clamp to a tiny positive value so log10 is finite at DC (DC bin then
    # gets the flat-low value, matching the "flat past the lowest anchor"
    # convention).
    log_f = np.log10(np.maximum(f, np.finfo(float).tiny))
    return np.interp(log_f, np.log10(o_s), d_s)


def generate_phase_noise(n: int, sample_rate: float,
                          offsets_hz, dbc_per_hz,
                          rng: np.random.Generator) -> np.ndarray:
    """Generate `n` real-valued phase-noise samples at `sample_rate`.

    The output's two-sided PSD ≈ S_φ(f) = 2 · 10^(L(f)/10) [rad²/Hz], where
    L(f) is the user's mask (single-sideband, dBc/Hz, IEEE convention).

    Implementation: shape white Gaussian noise in the frequency domain.
    Discrete-time white noise of variance σ² has flat PSD = σ²/f_s; after
    multiplying by |H(f)| = sqrt(S_φ(|f|) · f_s) the output PSD is S_φ(f).
    """
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")
    fs = float(sample_rate)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)               # length n//2 + 1
    # L(f) evaluated at each positive-frequency bin; the DC bin clamps to
    # the lowest anchor value via interp_dbc_mask's epsilon floor.
    dbc = interp_dbc_mask(freqs, offsets_hz, dbc_per_hz)
    s_phi = 2.0 * 10.0 ** (dbc / 10.0)                   # rad²/Hz
    h = np.sqrt(s_phi * fs)
    w = rng.standard_normal(n)
    return np.fft.irfft(np.fft.rfft(w) * h, n=n)


def apply_phase_noise(signal: np.ndarray, sample_rate: float,
                      offsets_hz, dbc_per_hz,
                      rng: np.random.Generator) -> np.ndarray:
    """Multiply `signal` by exp(j·φ[n]) where φ has the spec'd phase-noise PSD.

    Returns a new complex array of the same length as `signal`.
    """
    phi = generate_phase_noise(len(signal), sample_rate,
                               offsets_hz, dbc_per_hz, rng)
    return signal * np.exp(1j * phi)


__all__ = ["interp_dbc_mask", "generate_phase_noise", "apply_phase_noise"]
