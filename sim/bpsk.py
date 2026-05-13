import numpy as np
from .filters import rrc_coeffs


def rrc_bpsk_baseband(num_symbols: int, symbol_rate: float, sample_rate: float,
                      rolloff: float = 0.35, filter_span: int = 10,
                      seed: int | None = None,
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a complex baseband RRC-filtered BPSK signal centred at DC.

    Output signal is normalised to unit RMS power.

    Parameters:
        num_symbols:  Number of BPSK symbols
        symbol_rate:  Symbol rate in Hz
        sample_rate:  Sample rate in Hz; must be an integer multiple of symbol_rate
        rolloff:      RRC rolloff factor (0 to 1)
        filter_span:  RRC filter length in symbols
        seed:         Random seed

    Returns:
        signal   Complex baseband signal, unit RMS power (complex ndarray)
        t        Time axis in seconds (ndarray)
        symbols  Transmitted BPSK symbols: +1 or -1 (int ndarray)
    """
    sps = sample_rate / symbol_rate
    if abs(sps - round(sps)) > 1e-9 or sps < 2:
        raise ValueError(
            f"sample_rate / symbol_rate must be an integer >= 2, got {sps:.3f}")
    sps = int(round(sps))

    rng = np.random.default_rng(seed)
    symbols = (2 * rng.integers(0, 2, num_symbols) - 1)

    upsampled = np.zeros(num_symbols * sps, dtype=complex)
    upsampled[::sps] = symbols.astype(complex)

    baseband = np.convolve(upsampled, rrc_coeffs(filter_span, rolloff, sps), mode='same')

    rms = np.sqrt(np.mean(np.abs(baseband) ** 2))
    if rms > 0:
        baseband /= rms

    t = np.arange(len(baseband)) / sample_rate
    return baseband, t, symbols
