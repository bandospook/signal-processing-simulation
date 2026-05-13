import numpy as np


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


def ola_convolve(x: np.ndarray, h: np.ndarray, block_size: int = 4096) -> np.ndarray:
    """
    Linear convolution of x with h using FFT overlap-and-add.
    Output length: len(x) + len(h) - 1.
    """
    M = len(h)
    N = len(x)
    N_fft = 2 ** int(np.ceil(np.log2(block_size + M - 1)))
    H = np.fft.fft(h.astype(complex), N_fft)

    y = np.zeros(N + M - 1, dtype=complex)
    for i in range(int(np.ceil(N / block_size))):
        start = i * block_size
        block = x[start : start + block_size]
        block_out_len = len(block) + M - 1
        y_block = np.fft.ifft(np.fft.fft(block, N_fft) * H)
        y[start : start + block_out_len] += y_block[:block_out_len]

    return y


def fft_ola_upsample(x: np.ndarray, L: int,
                     filter_span: int = 16, block_size: int = 4096) -> np.ndarray:
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

    y_full = ola_convolve(x_up, h, block_size)
    return y_full[n_half : n_half + len(x) * L]


def fft_ola_downsample(x: np.ndarray, L: int,
                       filter_span: int = 16, block_size: int = 4096) -> np.ndarray:
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

    y_full = ola_convolve(x, h, block_size)
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
    """
    if not channel_cfg.get("enabled", True):
        return signal

    N = len(signal)
    freqs = np.fft.fftfreq(N, 1.0 / sample_rate)

    half_bw = signal_bw / 2.0
    in_band = np.abs(freqs) <= half_bw
    f_norm = np.where(in_band, freqs / half_bw, 0.0)

    ripple_db      = channel_cfg.get("ripple_db", 0.0)
    ripple_cycles  = channel_cfg.get("ripple_cycles", 1.0)
    max_phase_deg  = channel_cfg.get("max_phase_dev_deg", 0.0)
    poly_order     = channel_cfg.get("phase_poly_order", 2)

    r = (10 ** (ripple_db / 20) - 1) / (10 ** (ripple_db / 20) + 1)
    ampl = np.where(in_band, 1.0 + r * np.cos(np.pi * ripple_cycles * f_norm), 1.0)

    max_phase_rad = np.radians(max_phase_deg)
    phase = np.where(in_band, max_phase_rad * np.abs(f_norm) ** poly_order
                               * np.sign(f_norm) ** (poly_order % 2), 0.0)

    H = ampl * np.exp(1j * phase)
    return np.fft.ifft(np.fft.fft(signal) * H)
