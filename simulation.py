import numpy as np
from bpsk import rrc_bpsk_baseband
from filters import fft_ola_upsample, fft_ola_downsample, apply_channel_impairment
from nonlinear_amplifier import nonlinear_amplifier


def wideband_bpsk_simulation(carriers: list[dict],
                              sample_rate: float,
                              am_am_cfg: dict,
                              am_pm_cfg: dict,
                              input_backoff_db: float = 0.0,
                              noise_density_dbfs: float | None = None,
                              ola_filter_span: int = 16,
                              ola_block_size: int = 4096,
                              seed: int | None = None) -> dict:
    """
    Wideband N-carrier BPSK simulation with a single shared nonlinear amplifier.

    Each carrier is generated at its own native rate (sps × symbol_rate), optionally
    passed through a per-carrier channel impairment, then upsampled to the common
    wideband rate via FFT overlap-and-add.  The carriers are frequency-shifted, scaled
    by their individual power_db, summed, and passed through the nonlinear amplifier.
    Optional AWGN is added after the amplifier.  Each carrier is then extracted back to
    its native rate by downconversion and OLA downsampling.

    Each element of `carriers` is a dict with keys:
        name        str    identifier
        symbol_rate float  Hz
        sps         int    samples per symbol at native rate
        rolloff     float  RRC rolloff factor
        filter_span int    RRC filter length in symbols
        num_symbols int    number of BPSK symbols to generate
        power_db    float  power relative to 0 dB reference (amplitude scale = 10^(p/20))
        freq        float  Hz  centre frequency in the wideband spectrum
        channel     dict   optional; keys: enabled, ripple_db, ripple_cycles,
                           max_phase_dev_deg, phase_poly_order

    Returns a dict:
        wideband        combined signal before NL, at wideband rate
        wideband_nl     combined signal after NL, before noise
        wideband_noisy  combined signal after NL + noise (== wideband_nl if no noise)
        t_wb            wideband time axis
        carriers        list of per-carrier result dicts, each containing:
                          name, bb, nl, symbols, t, symbol_rate, native_rate
    """
    rng = np.random.default_rng(seed)
    per_carrier_seeds = rng.integers(0, 2 ** 31, len(carriers))

    carrier_state = []
    upsampled_signals = []

    for i, carr in enumerate(carriers):
        symbol_rate = float(carr["symbol_rate"])
        sps         = int(carr["sps"])
        rolloff     = float(carr["rolloff"])
        filter_span = int(carr["filter_span"])
        num_symbols = int(carr["num_symbols"])
        power_db    = float(carr.get("power_db", 0.0))
        freq        = float(carr["freq"])
        channel_cfg = carr.get("channel")

        native_rate = float(sps) * symbol_rate
        L = sample_rate / native_rate
        if abs(L - round(L)) > 1e-9 or L < 1:
            raise ValueError(
                f"Carrier '{carr['name']}': upsample factor {L:.4f} is not an integer >= 1")
        L = int(round(L))

        bb, t, symbols = rrc_bpsk_baseband(
            num_symbols, symbol_rate, native_rate,
            rolloff, filter_span, seed=int(per_carrier_seeds[i]))

        signal_bw = (1 + rolloff) * symbol_rate
        bb_ch = (apply_channel_impairment(bb, native_rate, signal_bw, channel_cfg)
                 if channel_cfg is not None else bb)

        bb_up = fft_ola_upsample(bb_ch, L, ola_filter_span, ola_block_size)

        amplitude_scale = 10 ** (power_db / 20)
        upsampled_signals.append((bb_up, freq, amplitude_scale))

        carrier_state.append(dict(
            name=carr["name"], bb=bb, symbols=symbols, t=t,
            symbol_rate=symbol_rate, native_rate=native_rate, freq=freq, L=L))

    # Trim all carriers to the same wideband length and form the composite
    N = min(len(u) for u, _, _ in upsampled_signals)
    t_wb = np.arange(N) / sample_rate

    wideband = np.zeros(N, dtype=complex)
    for bb_up, freq, amp_scale in upsampled_signals:
        wideband += amp_scale * bb_up[:N] * np.exp(1j * 2 * np.pi * freq * t_wb)

    # Normalise composite to unit peak then apply drive level (input backoff)
    drive = 10 ** (-input_backoff_db / 20)
    wideband_nl = nonlinear_amplifier(
        wideband * drive / np.max(np.abs(wideband)), am_am_cfg, am_pm_cfg)

    # Add wideband AWGN after the amplifier
    if noise_density_dbfs is not None:
        noise_power = 10 ** (noise_density_dbfs / 10) * sample_rate
        noise = np.sqrt(noise_power / 2) * (
            rng.standard_normal(N) + 1j * rng.standard_normal(N))
        wideband_noisy = wideband_nl + noise
    else:
        wideband_noisy = wideband_nl

    # Extract each carrier: downconvert → OLA downsample
    for cr in carrier_state:
        nl_down = fft_ola_downsample(
            wideband_noisy * np.exp(-1j * 2 * np.pi * cr["freq"] * t_wb),
            cr["L"], ola_filter_span, ola_block_size)
        cr["nl"] = nl_down

    return dict(wideband=wideband, wideband_nl=wideband_nl,
                wideband_noisy=wideband_noisy, t_wb=t_wb,
                carriers=carrier_state)
