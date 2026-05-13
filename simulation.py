import numpy as np
from bpsk import rrc_bpsk_baseband
from filters import fft_ola_upsample, fft_ola_downsample, apply_channel_impairment
from nonlinear_amplifier import nonlinear_amplifier


def wideband_bpsk_simulation(num_symbols_slow: int,
                              symbol_rate_slow: float,
                              sample_rate: float,
                              freq_slow: float,
                              freq_fast: float,
                              sps: int = 10,
                              rolloff: float = 0.35,
                              filter_span: int = 10,
                              input_backoff_db: float = 0.0,
                              am_am_cfg: dict | None = None,
                              am_pm_cfg: dict | None = None,
                              channel_slow_cfg: dict | None = None,
                              channel_fast_cfg: dict | None = None,
                              rate_ratio: int = 20,
                              ola_filter_span: int = 16,
                              ola_block_size: int = 4096,
                              seed: int | None = None) -> dict:
    """
    Wideband two-carrier BPSK simulation with a single shared nonlinear amplifier.

    Each carrier is generated at its own native rate (sps × symbol_rate), optionally
    passed through a per-signal channel impairment (passband ripple + phase nonlinearity),
    then upsampled to the common wideband rate via FFT overlap-and-add.  The two signals
    are frequency-shifted, summed, and passed through the nonlinear amplifier as a
    composite wideband signal.  Each carrier is then extracted back to its native
    bandwidth by downconversion and OLA downsampling.

    Returns a dict with keys:
        wideband      combined signal before NL, at wideband rate
        wideband_nl   combined signal after NL,  at wideband rate
        t_wb          wideband time axis
        slow_bb       slow carrier pre-NL  at native rate
        slow_nl       slow carrier post-NL extracted to native rate
        t_slow        time axis for slow signal at native rate
        fast_bb       fast carrier pre-NL  at native rate
        fast_nl       fast carrier post-NL extracted to native rate
        t_fast        time axis for fast signal at native rate
    """
    symbol_rate_fast = float(rate_ratio) * symbol_rate_slow
    native_rate_slow = float(sps * symbol_rate_slow)
    native_rate_fast = float(sps * symbol_rate_fast)

    L_slow = sample_rate / native_rate_slow
    L_fast = sample_rate / native_rate_fast
    for name, L in (('slow', L_slow), ('fast', L_fast)):
        if abs(L - round(L)) > 1e-9 or L < 1:
            raise ValueError(
                f"{name} upsample factor must be an integer >= 1, got {L:.4f}")
    L_slow, L_fast = int(round(L_slow)), int(round(L_fast))

    rng = np.random.default_rng(seed)
    seed_slow, seed_fast = rng.integers(0, 2 ** 31, 2)

    # Generate each signal at its own native sample rate
    slow_bb, t_slow = rrc_bpsk_baseband(
        num_symbols_slow, symbol_rate_slow, native_rate_slow,
        rolloff, filter_span, seed=int(seed_slow))

    fast_bb, t_fast = rrc_bpsk_baseband(
        rate_ratio * num_symbols_slow, symbol_rate_fast, native_rate_fast,
        rolloff, filter_span, seed=int(seed_fast))

    # Apply per-signal channel impairments at native rate
    signal_bw_slow = (1 + rolloff) * symbol_rate_slow
    signal_bw_fast = (1 + rolloff) * symbol_rate_fast

    slow_ch = slow_bb
    if channel_slow_cfg is not None:
        slow_ch = apply_channel_impairment(slow_bb, native_rate_slow,
                                           signal_bw_slow, channel_slow_cfg)

    fast_ch = fast_bb
    if channel_fast_cfg is not None:
        fast_ch = apply_channel_impairment(fast_bb, native_rate_fast,
                                           signal_bw_fast, channel_fast_cfg)

    # OLA upsample each signal to the common wideband rate
    slow_up = fft_ola_upsample(slow_ch, L_slow, ola_filter_span, ola_block_size)
    fast_up = fft_ola_upsample(fast_ch, L_fast, ola_filter_span, ola_block_size)

    # Frequency-shift and aggregate into wideband composite
    N = min(len(slow_up), len(fast_up))
    t_wb = np.arange(N) / sample_rate

    wideband = (slow_up[:N] * np.exp(1j * 2 * np.pi * freq_slow * t_wb)
                + fast_up[:N] * np.exp(1j * 2 * np.pi * freq_fast * t_wb))

    # Apply nonlinear amplifier to the composite.
    # Normalise to unit peak then scale by the drive level (input backoff).
    _am_am_cfg = am_am_cfg or {
        "input":  [0.0, 0.5, 1.0],
        "output": [0.0, 0.5, 1.0],
    }
    _am_pm_cfg = am_pm_cfg or {
        "input":     [0.0, 1.0],
        "phase_deg": [0.0, 0.0],
    }
    drive = 10 ** (-input_backoff_db / 20)
    wideband_nl = nonlinear_amplifier(
        wideband * drive / np.max(np.abs(wideband)), _am_am_cfg, _am_pm_cfg)

    # Extract each carrier: downconvert then OLA downsample
    slow_nl = fft_ola_downsample(
        wideband_nl * np.exp(-1j * 2 * np.pi * freq_slow * t_wb),
        L_slow, ola_filter_span, ola_block_size)

    fast_nl = fft_ola_downsample(
        wideband_nl * np.exp(-1j * 2 * np.pi * freq_fast * t_wb),
        L_fast, ola_filter_span, ola_block_size)

    return dict(wideband=wideband, wideband_nl=wideband_nl, t_wb=t_wb,
                slow_bb=slow_bb, slow_nl=slow_nl, t_slow=t_slow,
                fast_bb=fast_bb, fast_nl=fast_nl, t_fast=t_fast)
