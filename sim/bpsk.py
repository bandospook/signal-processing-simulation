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

    if mod == "DBPSK":
        encoded_bits = differential_encode(bits)
        # Map encoded bits to BPSK symbols (0→+1, 1→-1)
        symbols = np.where(encoded_bits == 0, 1.0 + 0j, -1.0 + 0j)
    else:
        symbols = map_bits(bits, mod, **mod_kwargs)

    h = rrc_coeffs(filter_span, rolloff, sps)

    if mod == "OQPSK":
        bb = _oqpsk_baseband(bits, symbols, sps, h)
    else:
        upsampled = np.zeros(num_symbols * sps, dtype=complex)
        upsampled[::sps] = symbols
        bb = np.convolve(upsampled, h, mode='same').astype(complex)

    rms = float(np.sqrt(np.mean(np.abs(bb) ** 2)))
    if rms > 0:
        bb /= rms

    t = np.arange(len(bb)) / sample_rate
    return bb, t, bits, symbols


def _oqpsk_baseband(bits: np.ndarray, symbols: np.ndarray,
                    sps: int, h: np.ndarray) -> np.ndarray:
    """
    OQPSK baseband: RRC-filter I and Q rails separately, then delay Q by T/2.

    The I rail carries the real part of each QPSK symbol; the Q rail carries
    the imaginary part.  Q is delayed by sps//2 samples so that only one rail
    changes at each symbol boundary, reducing envelope variation.
    """
    n_sym = len(symbols)
    I_syms = symbols.real
    Q_syms = symbols.imag

    I_up = np.zeros(n_sym * sps)
    Q_up = np.zeros(n_sym * sps)
    I_up[::sps] = I_syms
    Q_up[::sps] = Q_syms

    I_filt = np.convolve(I_up, h, mode='same')
    Q_filt = np.convolve(Q_up, h, mode='same')

    # Delay Q rail by half a symbol period
    half = sps // 2
    Q_delayed = np.zeros_like(Q_filt)
    Q_delayed[half:] = Q_filt[:-half]

    return (I_filt + 1j * Q_delayed).astype(complex)


# ── Backward-compatible alias ─────────────────────────────────────────────────

def rrc_bpsk_baseband(num_symbols: int, symbol_rate: float, sample_rate: float,
                      rolloff: float = 0.35, filter_span: int = 10,
                      seed: int | None = None,
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Legacy wrapper — returns (bb, t, symbols) for BPSK only."""
    bb, t, bits, symbols = rrc_baseband(
        "BPSK", num_symbols, symbol_rate, sample_rate, rolloff, filter_span, seed)
    # Convert ±1 complex back to ±1 int for callers that expect the old signature
    return bb, t, (symbols.real).astype(int)
