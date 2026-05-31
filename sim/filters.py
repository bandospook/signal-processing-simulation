"""Digital-filter primitives for the simulator.

RRC pulse shaping, FFT-based overlap-add convolution and resampling, and
per-carrier channel impairments (amplitude ripple + phase nonlinearity
applied in the frequency domain).
"""
import math
from collections.abc import Callable

import numpy as np

_ChunkCB = Callable[[int, int], None] | None
_CHUNK_REPORT = 64


class OLAState:
    """Stateful overlap-add convolution processor.

    Each call to process() feeds one block of input samples and returns
    block_size output samples, maintaining the overlap tail between calls.
    Correct for any filter length M relative to block_size (including M > B).
    """

    def __init__(self, h: np.ndarray, block_size: int) -> None:
        """Cache the FFT of `h` and pre-zero the overlap tail."""
        M = len(h)
        self._N_fft = 2 ** int(np.ceil(np.log2(block_size + M - 1)))
        self._H    = np.fft.fft(h.astype(complex), self._N_fft)
        self._M    = M
        self._B    = block_size
        self._tail = np.zeros(M - 1, dtype=complex)

    def process(self, block: np.ndarray) -> np.ndarray:
        """Return block_size filtered samples; pads the last (partial) block with zeros."""
        B, M = self._B, self._M
        block_pad = np.zeros(B, dtype=complex)
        block_pad[:len(block)] = block
        y = np.fft.ifft(np.fft.fft(block_pad, self._N_fft) * self._H)
        combined = y[:B + M - 1].copy()
        combined[:M - 1] += self._tail
        self._tail = combined[B:].copy()
        return combined[:B]

    @staticmethod
    def for_upsample(L: int, filter_span: int, block_size: int) -> "OLAState":
        """Kaiser-sinc upsample filter for the given integer ratio L."""
        n_half = filter_span * L
        n = np.arange(-n_half, n_half + 1)
        h = np.sinc(n / L) * np.kaiser(2 * n_half + 1, beta=8.0)
        return OLAState(h, block_size)

    @staticmethod
    def for_downsample(L: int, filter_span: int, block_size: int) -> "OLAState":
        """Kaiser-sinc downsample anti-alias filter (gain=1/L) for integer ratio L."""
        n_half = filter_span * L
        n = np.arange(-n_half, n_half + 1)
        h = np.sinc(n / L) * np.kaiser(2 * n_half + 1, beta=8.0) / L
        return OLAState(h, block_size)


def x_up_block(bb: np.ndarray, L: int, block_start: int, block_size: int) -> np.ndarray:
    """One block of the zero-inserted upsampled signal without allocating the full array.

    Equivalent to x_up[block_start : block_start + block_size] where
    x_up = zeros(len(bb) * L) with x_up[::L] = bb, but only the block is built.
    """
    out = np.zeros(block_size, dtype=complex)
    n_lo = math.ceil(block_start / L)
    n_hi = math.ceil((block_start + block_size) / L)
    ns   = np.arange(n_lo, min(n_hi, len(bb)))
    if len(ns):
        pos = ns * L - block_start
        out[pos] = bb[ns]
    return out


def rrc_coeffs(filter_span: int, rolloff: float, sps: int) -> np.ndarray:
    """Root raised cosine filter coefficients, normalised to unit energy."""
    num_taps = filter_span * sps + 1
    t = np.arange(-(num_taps // 2), num_taps // 2 + 1) / sps

    # Evaluate the general formula everywhere, guarding the two singularities
    # (t=0 and |t|=1/(4r)) against division by zero, then overwrite with the
    # correct closed-form limits via np.where.
    t_safe  = np.where(np.abs(t) < 1e-10, 1.0, t)
    denom_f = 1.0 - (4.0 * rolloff * t_safe) ** 2
    denom_f = np.where(np.abs(denom_f) < 1e-10, 1.0, denom_f)
    num = (np.sin(np.pi * t_safe * (1.0 - rolloff))
           + 4.0 * rolloff * t_safe * np.cos(np.pi * t_safe * (1.0 + rolloff)))
    h_gen = num / (np.pi * t_safe * denom_f)

    h_t0   = 1.0 - rolloff + 4.0 * rolloff / np.pi
    h_sing = (rolloff / np.sqrt(2.0)) * (
        (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * rolloff))
        + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * rolloff))
    )
    is_t0   = np.abs(t) < 1e-10
    is_sing = np.abs(np.abs(t) - 1.0 / (4.0 * rolloff)) < 1e-10
    h = np.where(is_t0, h_t0, np.where(is_sing, h_sing, h_gen))

    return h / np.sqrt(np.sum(h ** 2))


def ola_convolve(x: np.ndarray, h: np.ndarray, block_size: int = 4096,
                 chunk_cb: _ChunkCB = None) -> np.ndarray:
    """Linear convolution of x with h using FFT overlap-and-add.

    Output length: len(x) + len(h) - 1.
    chunk_cb(done, total) is called every _CHUNK_REPORT blocks if provided.
    """
    M = len(h)
    N = len(x)
    N_fft = 2 ** int(np.ceil(np.log2(block_size + M - 1)))
    H = np.fft.fft(h.astype(complex), N_fft)
    n_blocks = int(np.ceil(N / block_size))

    y = np.zeros(N + M - 1, dtype=complex)
    for i in range(n_blocks):
        start = i * block_size
        block = x[start : start + block_size]
        block_out_len = len(block) + M - 1
        y_block = np.fft.ifft(np.fft.fft(block, N_fft) * H)
        y[start : start + block_out_len] += y_block[:block_out_len]
        if chunk_cb is not None and (i % _CHUNK_REPORT == _CHUNK_REPORT - 1
                                     or i == n_blocks - 1):
            chunk_cb(i + 1, n_blocks)

    return y


def fft_ola_upsample(x: np.ndarray, L: int,
                     filter_span: int = 16, block_size: int = 4096,
                     chunk_cb: _ChunkCB = None) -> np.ndarray:
    """Upsample x by integer L via zero-insertion + Kaiser-windowed sinc, OLA-convolved.

    Filter satisfies the Nyquist criterion at original sample locations:
      h[n] = sinc(n/L) * kaiser(n, β=8)
    """
    if L == 1:
        return x.astype(complex)

    n_half = filter_span * L
    n = np.arange(-n_half, n_half + 1)
    h = np.sinc(n / L) * np.kaiser(2 * n_half + 1, beta=8.0)

    x_up = np.zeros(len(x) * L, dtype=complex)
    x_up[::L] = x

    y_full = ola_convolve(x_up, h, block_size, chunk_cb=chunk_cb)
    return y_full[n_half : n_half + len(x) * L]


def fft_ola_downsample(x: np.ndarray, L: int,
                       filter_span: int = 16, block_size: int = 4096,
                       chunk_cb: _ChunkCB = None) -> np.ndarray:
    """Downsample x by integer L via anti-alias Kaiser-windowed sinc LPF.

    Cutoff is at the new Nyquist; OLA-convolved, then decimated.
    Filter has unity passband gain:
      h[n] = sinc(n/L) * kaiser(n, β=8) / L
    """
    if L == 1:
        return x.astype(complex)

    n_half = filter_span * L
    n = np.arange(-n_half, n_half + 1)
    h = np.sinc(n / L) * np.kaiser(2 * n_half + 1, beta=8.0) / L

    y_full = ola_convolve(x, h, block_size, chunk_cb=chunk_cb)
    y = y_full[n_half : n_half + len(x)]
    return y[::L]


def apply_channel_impairment(signal: np.ndarray, sample_rate: float,
                              signal_bw: float, channel_cfg: dict) -> np.ndarray:
    """Apply passband amplitude ripple and phase nonlinearity in the frequency domain.

    H(f) = A(f) * exp(j * phi(f))
      A(f)   = 1 + r * cos(pi * ripple_cycles * f_norm)   [in-band only]
      phi(f) = max_phase_dev_rad * f_norm^phase_poly_order [in-band only]

    where f_norm = f / (signal_bw / 2) ∈ [-1, +1] at band edges
    and r = (10^(ripple_db/20) - 1) / (10^(ripple_db/20) + 1)

    The DFT is zero-padded so that multiplication by H(f) is equivalent to linear
    (not circular) convolution.  The amplitude ripple term creates an implicit delay
    tap at ±ripple_cycles/signal_bw seconds; padding by that extent keeps circular
    wrap-around in the zero region.
    """
    if not channel_cfg.get("enabled", True):
        return signal

    N = len(signal)
    ripple_db      = channel_cfg.get("ripple_db", 0.0)
    ripple_cycles  = channel_cfg.get("ripple_cycles", 1.0)
    max_phase_deg  = channel_cfg.get("max_phase_dev_deg", 0.0)
    poly_order     = channel_cfg.get("phase_poly_order", 2)

    # Delay tap of the cosine ripple is at ±ripple_cycles/signal_bw seconds.
    # Pad by that many samples (+ 8 for sinc sidelobe decay) so wrap-around is harmless.
    pad = int(np.ceil(ripple_cycles * sample_rate / signal_bw)) + 8
    N_fft = 2 ** int(np.ceil(np.log2(N + pad)))

    freqs = np.fft.fftfreq(N_fft, 1.0 / sample_rate)

    half_bw = signal_bw / 2.0
    in_band = np.abs(freqs) <= half_bw
    f_norm = np.where(in_band, freqs / half_bw, 0.0)

    r = (10 ** (ripple_db / 20) - 1) / (10 ** (ripple_db / 20) + 1)
    ampl = np.where(in_band, 1.0 + r * np.cos(np.pi * ripple_cycles * f_norm), 1.0)

    max_phase_rad = np.radians(max_phase_deg)
    phase = np.where(in_band, max_phase_rad * np.abs(f_norm) ** poly_order
                               * np.sign(f_norm) ** (poly_order % 2), 0.0)

    H = ampl * np.exp(1j * phase)

    padded = np.zeros(N_fft, dtype=complex)
    padded[:N] = signal
    return np.fft.ifft(np.fft.fft(padded) * H)[:N]
