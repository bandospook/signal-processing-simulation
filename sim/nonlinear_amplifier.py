"""Memoryless nonlinear amplifier model with AM-AM and AM-PM distortion."""

import numpy as np


def nonlinear_amplifier(signal_in: np.ndarray,
                        am_am_cfg: dict,
                        am_pm_cfg: dict) -> np.ndarray:
    """
    Apply a memoryless nonlinear amplifier using lookup-table AM-AM and AM-PM curves.

    Parameters:
        signal_in:  Complex baseband input signal (normalised so peak amplitude = 1)
        am_am_cfg:  Dict with keys 'input' and 'output' (amplitude → amplitude)
        am_pm_cfg:  Dict with keys 'input' and 'phase_deg' (amplitude → degrees)

    Returns:
        Complex baseband output signal
    """
    am_in  = np.asarray(am_am_cfg["input"],    dtype=float)
    am_out = np.asarray(am_am_cfg["output"],   dtype=float)
    pm_in  = np.asarray(am_pm_cfg["input"],    dtype=float)
    pm_deg = np.asarray(am_pm_cfg["phase_deg"], dtype=float)

    amplitude_in  = np.abs(signal_in)
    phase_in      = np.angle(signal_in)

    amplitude_out = np.interp(amplitude_in, am_in, am_out)
    delta_phase   = np.radians(np.interp(amplitude_in, pm_in, pm_deg))

    return amplitude_out * np.exp(1j * (phase_in + delta_phase))


if __name__ == "__main__":
    from sim.plots import plot_nl_tables

    _am_am = {
        "input":  [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "output": [0.0, 0.125, 0.248, 0.367, 0.481, 0.589, 0.688, 0.780, 0.862, 0.936, 1.000],
    }
    _am_pm = {
        "input":     [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "phase_deg": [0.000, 0.029, 0.115, 0.258, 0.459, 0.718, 1.034, 1.407, 1.838, 2.326, 2.872],
    }
    plot_nl_tables(_am_am, _am_pm)
