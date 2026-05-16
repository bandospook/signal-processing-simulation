"""Tests for sim.nonlinear_amplifier: AM-AM, AM-PM, and IMD vs drive level."""
import numpy as np
import pytest
from sim.nonlinear_amplifier import nonlinear_amplifier

# Production curves from simulation.toml
_AM_AM = {
    "input":  [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "output": [0.000, 0.119, 0.238, 0.356, 0.473, 0.586, 0.692, 0.788, 0.873, 0.944, 1.000],
}
_AM_PM = {
    "input":     [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "phase_deg": [0.000, 0.050, 0.200, 0.450, 0.800, 1.250, 1.800, 2.450, 3.200, 4.050, 5.000],
}


def _apply(x):
    return nonlinear_amplifier(x, _AM_AM, _AM_PM)


def test_zero_input_gives_zero_output():
    assert np.allclose(_apply(np.zeros(10, dtype=complex)), 0)


def test_amam_table_points_matched():
    """At each LUT knot, output amplitude must match the table exactly."""
    for a_in, a_out in zip(_AM_AM["input"][1:], _AM_AM["output"][1:]):
        result = _apply(np.array([complex(a_in)]))
        assert abs(abs(result[0]) - a_out) < 1e-9, f"AM-AM mismatch at input={a_in}"


def test_ampm_table_points_matched():
    """At each LUT knot, added phase must match the table exactly (pure-real input)."""
    for a_in, ph in zip(_AM_PM["input"][1:], _AM_PM["phase_deg"][1:]):
        result = _apply(np.array([complex(a_in)]))
        assert abs(np.degrees(np.angle(result[0])) - ph) < 1e-6, \
            f"AM-PM mismatch at input={a_in}"


def test_phase_rotation_preserved():
    """Output phase = input phase + AM-PM(amplitude) for any input phase."""
    amp, phi_in = 0.7, np.pi / 3
    expected_phase = phi_in + np.radians(
        np.interp(amp, _AM_PM["input"], _AM_PM["phase_deg"]))
    result = _apply(np.array([amp * np.exp(1j * phi_in)]))
    assert abs(np.angle(result[0]) - expected_phase) < 1e-9


def test_identity_for_linear_amplifier():
    """Linear AM-AM (slope 1) and zero AM-PM → output equals input exactly."""
    lin_am = {"input": [0.0, 1.0], "output": [0.0, 1.0]}
    lin_pm = {"input": [0.0, 1.0], "phase_deg": [0.0, 0.0]}
    rng = np.random.default_rng(0)
    x = rng.standard_normal(100) + 1j * rng.standard_normal(100)
    x /= np.max(np.abs(x))
    assert np.allclose(nonlinear_amplifier(x, lin_am, lin_pm), x, atol=1e-12)


def test_two_tone_imd_increases_with_drive():
    """
    Third-order IM product power relative to carrier must rise monotonically as
    drive level increases toward saturation.

    Two-tone test: x(t) = cos(2π f1 t) + cos(2π f2 t), normalised to peak=drive.
    IM3 products appear at 2f1-f2 and 2f2-f1; their ratio to carrier grows with drive.
    """
    N = 4096
    f1, f2 = 0.10, 0.13
    t = np.arange(N, dtype=float)

    def _imd_ratio(drive: float) -> float:
        tone = np.cos(2*np.pi*f1*t) + np.cos(2*np.pi*f2*t)
        tone = (tone * drive / np.max(np.abs(tone))).astype(complex)
        X = np.abs(np.fft.fft(_apply(tone))) / N
        def _p(f): return X[int(round(f * N))]**2
        carrier = _p(f1) + _p(f2)
        im3     = _p(2*f1 - f2) + _p(2*f2 - f1)
        return im3 / (carrier + 1e-30)

    drives = [0.3, 0.5, 0.7, 0.9]
    ratios = [_imd_ratio(d) for d in drives]
    for i in range(1, len(ratios)):
        assert ratios[i] > ratios[i-1], (
            f"IMD ratio should increase with drive: "
            f"drive={drives[i]:.1f} → {ratios[i]:.2e} not > "
            f"drive={drives[i-1]:.1f} → {ratios[i-1]:.2e}"
        )
