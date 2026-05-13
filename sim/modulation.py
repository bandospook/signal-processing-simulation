"""
Constellation definitions and symbol-level utilities for all supported modulations.

Supported modulations
---------------------
BPSK    – Binary PSK (1 bit/symbol)
DBPSK   – Differential BPSK (1 bit/symbol, differential encoding removes 180° ambiguity)
QPSK    – Quadrature PSK (2 bits/symbol)
OQPSK   – Offset QPSK (2 bits/symbol, Q rail delayed T/2 relative to I)
8PSK    – 8-ary PSK (3 bits/symbol)
16QAM   – 16-point QAM (4 bits/symbol)
16APSK  – 16-point APSK on 2 rings (4 bits/symbol, DVB-S2 style)
32APSK  – 32-point APSK on 3 rings (5 bits/symbol, DVB-S2 style)

All constellations are normalised to unit average power.
Constellation arrays are indexed by the integer value of the bit pattern (MSB first),
so constellation[0b0110] is the symbol for the bit sequence [0,1,1,0].
"""

import numpy as np

# ── Supported modulations ─────────────────────────────────────────────────────

_BITS_PER_SYMBOL: dict[str, int] = {
    "BPSK": 1, "DBPSK": 1,
    "QPSK": 2, "OQPSK": 2,
    "8PSK": 3,
    "16QAM": 4, "16APSK": 4,
    "32APSK": 5,
}

SUPPORTED: list[str] = list(_BITS_PER_SYMBOL)

# Rotational symmetry order used to resolve phase ambiguity in BER measurement.
# DBPSK = 1 because differential decoding resolves the 180° flip automatically.
_ROTATIONAL_SYMMETRY: dict[str, int] = {
    "BPSK": 2, "DBPSK": 1,
    "QPSK": 4, "OQPSK": 4,
    "8PSK": 8,
    "16QAM": 4, "16APSK": 4, "32APSK": 4,
}


def bits_per_symbol(mod: str) -> int:
    try:
        return _BITS_PER_SYMBOL[mod.upper()]
    except KeyError:
        raise ValueError(f"Unknown modulation '{mod}'. Supported: {SUPPORTED}")


def rotational_symmetry(mod: str) -> int:
    """N-fold rotational symmetry of the constellation (for BER ambiguity resolution)."""
    return _ROTATIONAL_SYMMETRY[mod.upper()]


# ── Gray code utilities ───────────────────────────────────────────────────────

def _gray_decode(g: int) -> int:
    """Decode a Gray-coded integer to its natural binary equivalent."""
    n, mask = g, g >> 1
    while mask:
        n ^= mask
        mask >>= 1
    return n


# ── Constellation builders ────────────────────────────────────────────────────

def _psk_constellation(M: int, offset: float = 0.0) -> np.ndarray:
    """
    Gray-coded M-PSK indexed by bit pattern 0..M-1.
    Symbol for bit pattern p is at angle: gray_decode(p) * 2π/M + offset.
    Unit average power (all symbols on the unit circle).
    """
    angles = np.array([_gray_decode(p) for p in range(M)]) * (2 * np.pi / M) + offset
    return np.exp(1j * angles)


def _qam16_constellation() -> np.ndarray:
    """
    Standard Gray-coded 16QAM indexed by bit pattern 0..15.
    Bits split as [b3 b2] → I axis, [b1 b0] → Q axis, each 2-bit Gray coded.
    Normalised to unit average power (divisor √10).
    """
    vals = np.array([-3.0, -1.0, 1.0, 3.0]) / np.sqrt(10.0)
    C = np.empty(16, dtype=complex)
    for p in range(16):
        C[p] = vals[_gray_decode((p >> 2) & 3)] + 1j * vals[_gray_decode(p & 3)]
    return C


def _apsk16_constellation(gamma: float = 2.57) -> np.ndarray:
    """
    16APSK on 2 rings (4 inner + 12 outer), unit average power.
    gamma = r2/r1 (outer/inner radius ratio, default 2.57 per DVB-S2).

    Bit mapping: patterns 0–3  → inner ring, Gray-coded by angle (4 = 2²).
                 patterns 4–15 → outer ring, natural angular order.

    Note: 12-point rings cannot use Gray coding directly because 12 is not
    a power of 2.  _gray_decode on indices 0–11 produces values up to 14,
    causing angles to exceed 330° and wrap to duplicates.  Natural order
    gives 12 unique points and is consistent between TX and RX.
    """
    r1 = np.sqrt(16.0 / (4.0 + 12.0 * gamma ** 2))
    r2 = gamma * r1
    inner = r1 * np.exp(1j * (np.pi / 4 + np.array([_gray_decode(k) for k in range(4)]) * np.pi / 2))
    outer = r2 * np.exp(1j * (np.arange(12) * np.pi / 6))
    return np.concatenate([inner, outer])


def _apsk32_constellation(gamma1: float = 2.84, gamma2: float = 5.27) -> np.ndarray:
    """
    32APSK on 3 rings (4 inner + 12 mid + 16 outer), unit average power.
    gamma1 = r2/r1, gamma2 = r3/r1 (defaults per DVB-S2).

    Bit mapping: patterns  0–3  → ring 1 (4 pts, Gray-coded, 4 = 2²).
                 patterns  4–15 → ring 2 (12 pts, natural angular order).
                 patterns 16–31 → ring 3 (16 pts, Gray-coded, 16 = 2⁴).
    """
    denom = 4.0 + 12.0 * gamma1 ** 2 + 16.0 * gamma2 ** 2
    r1 = np.sqrt(32.0 / denom)
    r2, r3 = gamma1 * r1, gamma2 * r1
    inner = r1 * np.exp(1j * (np.pi / 4 + np.array([_gray_decode(k) for k in range(4)]) * np.pi / 2))
    mid   = r2 * np.exp(1j * (np.arange(12) * np.pi / 6))
    outer = r3 * np.exp(1j * (np.array([_gray_decode(k) for k in range(16)]) * np.pi / 8))
    return np.concatenate([inner, mid, outer])


# ── Public API ────────────────────────────────────────────────────────────────

def constellation(mod: str, **kwargs) -> np.ndarray:
    """
    Return the complex symbol alphabet for the given modulation, indexed by bit pattern.

    kwargs for APSK:
        apsk_gamma        float  r2/r1 for 16APSK (default 2.57)
        apsk_gamma1       float  r2/r1 for 32APSK (default 2.84)
        apsk_gamma2       float  r3/r1 for 32APSK (default 5.27)
    """
    mod = mod.upper()
    if mod in ("BPSK", "DBPSK"):
        return _psk_constellation(2, offset=0.0)
    elif mod in ("QPSK", "OQPSK"):
        return _psk_constellation(4, offset=np.pi / 4)
    elif mod == "8PSK":
        return _psk_constellation(8, offset=0.0)
    elif mod == "16QAM":
        return _qam16_constellation()
    elif mod == "16APSK":
        return _apsk16_constellation(kwargs.get("apsk_gamma", 2.57))
    elif mod == "32APSK":
        return _apsk32_constellation(kwargs.get("apsk_gamma1", 2.84),
                                     kwargs.get("apsk_gamma2", 5.27))
    raise ValueError(f"Unknown modulation '{mod}'. Supported: {SUPPORTED}")


def map_bits(bits: np.ndarray, mod: str, **kwargs) -> np.ndarray:
    """
    Map a flat integer bit array (values 0/1, length = N × bits_per_symbol) to
    N complex symbols.  Bits are packed MSB-first within each symbol.
    """
    bps = bits_per_symbol(mod)
    C = constellation(mod, **kwargs)
    n_sym = len(bits) // bps
    b = np.asarray(bits[:n_sym * bps], dtype=int).reshape(n_sym, bps)
    powers = 1 << np.arange(bps - 1, -1, -1)   # [2^(bps-1), …, 2, 1]
    indices = b @ powers                          # integer index per symbol
    return C[indices]


def decide(samples: np.ndarray, mod: str, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """
    Nearest-neighbour hard decision for the given modulation.

    Returns
    -------
    symbol_decisions : complex ndarray  — nearest constellation point per sample
    bit_decisions    : int ndarray      — decoded bits (flat, 0/1, length N×bps)
    """
    C = constellation(mod, **kwargs)
    bps = bits_per_symbol(mod)
    diffs = samples[:, np.newaxis] - C[np.newaxis, :]
    indices = np.argmin(np.abs(diffs) ** 2, axis=1)
    symbol_decisions = C[indices]
    powers = 1 << np.arange(bps - 1, -1, -1)
    bits = ((indices[:, np.newaxis] & powers[np.newaxis, :]) > 0).astype(int).ravel()
    return symbol_decisions, bits


# ── Differential encode / decode (DBPSK) ─────────────────────────────────────

def differential_encode(bits: np.ndarray) -> np.ndarray:
    """
    DBPSK differential encoder.

    d[n] = d[n-1] XOR b[n],  d[-1] = 0.
    Equivalently: d[n] = XOR of b[0..n] = cumulative XOR.
    """
    return np.bitwise_xor.accumulate(np.asarray(bits, dtype=int))


def differential_decode(sym_decisions: np.ndarray) -> np.ndarray:
    """
    DBPSK differential decoder.

    Input:  N BPSK symbol decisions ∈ {−1, +1}.
    Output: N−1 decoded bits ∈ {0, 1}.

    b_hat[n] = 1  if  d_hat[n] and d_hat[n+1] have opposite sign, else 0.
    Returns bits aligned to reference_bits[1:] of the transmitter.
    """
    return ((sym_decisions[:-1] * sym_decisions[1:]) < 0).astype(int)
