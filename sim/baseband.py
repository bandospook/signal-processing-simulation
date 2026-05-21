"""Baseband signal generation for all supported modulations."""
import numpy as np
from .filters import rrc_coeffs
from .modulation import bits_per_symbol, map_bits, differential_encode


def rrc_baseband(modulation: str,
                 num_symbols: int,
                 symbol_rate: float,
                 sample_rate: float,
                 rolloff: float = 0.35,
                 filter_span: int = 10,
                 seed: int | None = None,
                 **mod_kwargs,
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a complex baseband RRC-filtered signal for any supported modulation.

    For OQPSK the Q rail is delayed by T/2 (sps//2 samples) relative to I.
    For DBPSK the bits are differentially encoded before BPSK mapping; the
    returned `bits` array holds the original (pre-encoding) data bits.

    Parameters
    ----------
    modulation   : str    Modulation name (see sim.modulation.SUPPORTED)
    num_symbols  : int    Number of symbols to generate
    symbol_rate  : float  Symbol rate in Hz
    sample_rate  : float  Sample rate in Hz (must be integer multiple of symbol_rate)
    rolloff      : float  RRC rolloff factor
    filter_span  : int    RRC filter half-span in symbols
    seed         : int    Random seed
    **mod_kwargs          Passed to modulation helpers (e.g. apsk_gamma)

    Returns
    -------
    bb      Complex baseband signal, unit RMS power
    t       Time axis in seconds
    bits    Transmitted data bits (flat, 0/1 int array, length = num_symbols × bps)
    symbols Complex constellation points that were transmitted (length = num_symbols)
    """
    mod = modulation.upper()
    sps = sample_rate / symbol_rate
    if abs(sps - round(sps)) > 1e-9 or sps < 2:
        raise ValueError(
            f"sample_rate / symbol_rate must be an integer ≥ 2, got {sps:.3f}")
    sps = int(round(sps))

    bps = bits_per_symbol(mod)
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, num_symbols * bps).astype(int)

    if mod == "MSK":
        symbols = (1 - 2 * bits).astype(complex)
        bb = _msk_baseband(bits, sps)
    elif mod == "DBPSK":
        encoded_bits = differential_encode(bits)
        symbols = np.where(encoded_bits == 0, 1.0 + 0j, -1.0 + 0j)
        h = rrc_coeffs(filter_span, rolloff, sps)
        upsampled = np.zeros(num_symbols * sps, dtype=complex)
        upsampled[::sps] = symbols
        bb = np.convolve(upsampled, h, mode='same').astype(complex)
    else:
        symbols = map_bits(bits, mod, **mod_kwargs)
        h = rrc_coeffs(filter_span, rolloff, sps)
        if mod == "OQPSK":
            bb = _oqpsk_baseband(symbols, sps, h)
        else:
            upsampled = np.zeros(num_symbols * sps, dtype=complex)
            upsampled[::sps] = symbols
            bb = np.convolve(upsampled, h, mode='same').astype(complex)

    rms = float(np.sqrt(np.mean(np.abs(bb) ** 2)))
    if rms > 0:
        bb /= rms

    t = np.arange(len(bb)) / sample_rate
    return bb, t, bits, symbols


def _msk_baseband(bits: np.ndarray, sps: int) -> np.ndarray:
    """
    MSK baseband, built as offset-QPSK with half-sine pulse shaping.

    MSK is exactly offset-QPSK whose pulse is a half-sine 2*sps samples wide:
    even-indexed bits drive the in-phase rail, odd-indexed bits the quadrature
    rail, and the Q rail is delayed by sps.  The sum is the constant-envelope,
    continuous-phase MSK waveform.  Because each rail is then an independent
    antipodal channel, a per-rail matched filter attains the BPSK error rate.

    Returns a complex array of length n_sym * sps.
    """
    n_sym = len(bits)
    total = n_sym * sps
    d = 1.0 - 2.0 * bits.astype(float)                  # bit 0 -> +1, bit 1 -> -1
    pulse = np.sin(np.pi * np.arange(2 * sps) / (2.0 * sps))

    i_rail = np.repeat(d[0::2], 2 * sps) * np.tile(pulse, (n_sym + 1) // 2)
    q_rail = np.repeat(d[1::2], 2 * sps) * np.tile(pulse, n_sym // 2)

    bb = np.zeros(total, dtype=complex)
    bb += i_rail[:total]
    bb[sps:] += 1j * q_rail[:total - sps]
    return bb


def _oqpsk_baseband(symbols: np.ndarray, sps: int, h: np.ndarray) -> np.ndarray:
    """
    OQPSK baseband: RRC-filter I and Q rails separately, then delay Q by T/2.

    The I rail carries the real part of each QPSK symbol; the Q rail carries
    the imaginary part.  Q is delayed by sps//2 samples so that only one rail
    changes at each symbol boundary, reducing envelope variation.
    """
    n_sym = len(symbols)

    I_up = np.zeros(n_sym * sps)
    Q_up = np.zeros(n_sym * sps)
    I_up[::sps] = np.real(symbols)
    Q_up[::sps] = np.imag(symbols)

    I_filt = np.convolve(I_up, h, mode='same')
    Q_filt = np.convolve(Q_up, h, mode='same')

    half = sps // 2
    Q_delayed = np.zeros_like(Q_filt)
    Q_delayed[half:] = Q_filt[:-half]

    return (I_filt + 1j * Q_delayed).astype(complex)
