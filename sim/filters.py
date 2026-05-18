import math
from collections.abc import Callable

import numpy as np

_ChunkCB = Callable[[int, int], None] | None
_CHUNK_REPORT = 64


class OLAState:
    """
    Stateful overlap-add convolution processor.

    Each call to process() feeds one block of input samples and returns
    block_size output samples, maintaining the overlap tail between calls.
    Correct for any filter length M relative to block_size (including M > B).
    """

    def __init__(self, h: np.ndarray, block_size: int) -> None:
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
    """
    One block of the zero-inserted upsampled signal without allocating the full array.

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


def rational_resample(x: np.ndarray, P: int, Q: int, filter_span: int = 8) -> np.ndarray:
    """
    Resample x by rational factor P/Q using a polyphase Kaiser-sinc filter.

    The filter cutoff sits at 1/max(P,Q) of the upsampled rate, which equals
    min(fs_in, fs_out)/2 — correct for both upsampling and downsampling.
    Filter coefficients are scaled so the passband gain is unity.

    Output length: ceil(len(x) * P / Q).
    Returns x.astype(complex) unchanged when P == Q after gcd reduction.
    """
    g = math.gcd(P, Q)
    P //= g; Q //= g
    if P == Q:
        return x.astype(complex)

    M = max(P, Q)
    n_half = filter_span * M
    n_arr  = np.arange(-n_half, n_half + 1)
    # (P/M) * sinc(n/M): sum(sinc(n/M)) ≈ M, so sum(h) ≈ P, exactly cancelling the
    # 1/P DC loss from P-fold zero-insertion → unity passband gain for any P/Q.
    h_full = (float(P) / M) * np.sinc(n_arr / M) * np.kaiser(2 * n_half + 1, beta=8.0)
    h_len  = len(h_full)   # 2 * n_half + 1

    # Polyphase decomposition: branch k gets taps h_full[k], h_full[k+P], ...
    # Each branch has L_poly taps (padded with zeros if needed).
    L_poly = math.ceil(h_len / P)
    h_branches: list[np.ndarray] = []
    for k in range(P):
        b = h_full[k : h_len : P]
        if len(b) < L_poly:
            b = np.append(b, np.zeros(L_poly - len(b)))
        h_branches.append(b)

    N     = len(x)
    N_out = math.ceil(N * P / Q)

    # Zero-pad x so boundary windows are always in-bounds.
    pad   = L_poly
    x_pad = np.zeros(N + 2 * pad, dtype=complex)
    x_pad[pad : pad + N] = x

    y = np.zeros(N_out, dtype=complex)

    # For output sample m:
    #   t     = m*Q + n_half           (time index in the P-upsampled domain)
    #   phase = t % P                  (which polyphase branch)
    #   n_top = t // P                 (topmost input sample contributing)
    #   y[m]  = dot(h_branches[phase], x[n_top], x[n_top-1], ..., x[n_top-L_poly+1])
    #
    # Outputs sharing the same phase k appear every P samples: m = m0, m0+P, m0+2P, ...
    # For those outputs, n_top increases by Q per step.  We vectorise per phase.
    Q_inv = pow(int(Q), -1, int(P))   # modular inverse: Q * Q_inv ≡ 1 (mod P)

    for k in range(P):
        # Smallest m >= 0 where (m*Q + n_half) % P == k
        target    = (k - int(n_half) % P + P) % P
        m0        = (target * Q_inv) % P
        m_indices = np.arange(m0, N_out, P, dtype=np.int64)
        if len(m_indices) == 0:
            continue

        n0        = (int(m0) * Q + n_half) // P   # n_top for the first output in this phase
        n_tops    = n0 + np.arange(len(m_indices), dtype=np.int64) * Q

        # Build input matrix: rows are reversed L_poly-sample windows.
        # row j: x_pad[n_tops[j]+pad], x_pad[n_tops[j]+pad-1], ..., x_pad[n_tops[j]+pad-L_poly+1]
        row_starts = n_tops + pad - (L_poly - 1)                     # leftmost index per row
        col_idx    = np.arange(L_poly - 1, -1, -1, dtype=np.int64)  # reversed column offsets
        idx_matrix = row_starts[:, np.newaxis] + col_idx[np.newaxis, :]
        idx_matrix = np.clip(idx_matrix, 0, len(x_pad) - 1)

        y[m_indices] = x_pad[idx_matrix] @ h_branches[k]

    return y


def rrc_coeffs(filter_span: int, rolloff: float, sps: int) -> np.ndarray:
    """Root raised cosine filter coefficients, normalised to unit energy."""
    num_taps = filter_span * sps + 1
    t = np.arange(-(num_taps // 2), num_taps // 2 + 1) / sps

    h = np.zeros(num_taps)
    for i, ti in enumerate(t):
        if ti == 0.0:
            h[i] = 1.0 - rolloff + 4 * rolloff / np.pi
        elif abs(abs(ti) - 1.0 / (4.0 * rolloff)) < 1e-10:
            h[i] = (rolloff / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * rolloff))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * rolloff))
            )
        else:
            num = (np.sin(np.pi * ti * (1 - rolloff))
                   + 4 * rolloff * ti * np.cos(np.pi * ti * (1 + rolloff)))
            den = np.pi * ti * (1 - (4 * rolloff * ti) ** 2)
            h[i] = num / den

    return h / np.sqrt(np.sum(h ** 2))


def ola_convolve(x: np.ndarray, h: np.ndarray, block_size: int = 4096,
                 chunk_cb: _ChunkCB = None) -> np.ndarray:
    """
    Linear convolution of x with h using FFT overlap-and-add.
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
    """
    Upsample x by integer L via zero-insertion and Kaiser-windowed sinc filter
    applied with OLA convolution.

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
    """
    Downsample x by integer L via anti-alias Kaiser-windowed sinc LPF
    (cutoff at the new Nyquist) applied with OLA, then decimate.

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
    """
    Apply passband amplitude ripple and phase nonlinearity in the frequency domain.

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
