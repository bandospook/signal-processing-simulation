import math
from collections.abc import Callable

import numpy as np

from .baseband import rrc_baseband
from scipy.signal import resample_poly

from .coding import (build_code, decode_frames, encode_frames,
                     ConcatenatedCode, ConvolutionalCode, LDPCCode, TurboCode)
from .filters import OLAState, x_up_block, apply_channel_impairment
from .modulation import bits_per_symbol
from .phase_noise import apply_phase_noise
from .nonlinear_amplifier import nonlinear_amplifier
from .receiver import receive, soft_demap

_AnyCode = ConvolutionalCode | ConcatenatedCode | LDPCCode | TurboCode

_PrintCB = Callable[[str], None] | None

# Report OLA chunk progress every this many wideband chunks.
_CHUNK_REPORT = 64

# Welch PSD segment length (samples at wideband rate).  Accumulated across all
# chunks; segments shorter than this (last partial segment) are discarded.
_WELCH_NFFT = 16384


# ── Welch PSD accumulator ────────────────────────────────────────────────────

class _WelchState:
    """Incremental Welch periodogram.  Feed chunks; call result() when done."""

    def __init__(self, nfft: int) -> None:
        self._nfft   = nfft
        self._w      = np.hanning(nfft)
        self._w_sq   = float(np.sum(self._w) ** 2)
        self._accum: np.ndarray | None = None
        self._count  = 0
        self._buf    = np.zeros(0, dtype=complex)

    def add(self, chunk: np.ndarray) -> None:
        self._buf = np.concatenate((self._buf, chunk))
        while len(self._buf) >= self._nfft:
            seg = self._buf[:self._nfft]
            self._buf = self._buf[self._nfft:]
            P = np.abs(np.fft.fft(seg * self._w)) ** 2
            self._accum = P if self._accum is None else self._accum + P
            self._count += 1

    def result(self, sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
        nfft = self._nfft
        f = np.fft.fftshift(np.fft.fftfreq(nfft, 1.0 / sample_rate))
        if self._count == 0 or self._accum is None:
            return f, np.full(nfft, -100.0)
        avg = np.fft.fftshift(self._accum / self._count)
        psd = 10.0 * np.log10(avg / self._w_sq + 1e-24)
        return f, psd


# ── Decimation helper ────────────────────────────────────────────────────────

def _decimate(filtered: np.ndarray, L: int,
              offset: int) -> tuple[np.ndarray, int]:
    """
    Decimate filtered by L, starting at sample index `offset` within the block.
    Returns (decimated_samples, next_offset).  Offset carries between blocks so
    that decimation is phase-coherent across chunk boundaries.
    """
    if offset >= len(filtered):
        return np.empty(0, dtype=complex), offset - len(filtered)
    indices = np.arange(offset, len(filtered), L)
    new_offset = int(indices[-1]) + L - len(filtered)
    return filtered[indices], new_offset


# ── Simulation ───────────────────────────────────────────────────────────────

def _derive_block_counts(carr: dict, sps: int, bps: int,
                         budget_samples: int) -> tuple[int, int, _AnyCode | None]:
    """Return (num_symbols, n_frames, code) sized to fit budget_samples.

    For uncoded carriers, n_frames == 0 and code is None.
    For coded carriers, n_frames is the integer number of FEC frames whose
    total native-rate sample length fits within budget_samples (≥ 1), and
    num_symbols is the corresponding symbol count.
    """
    coding_cfg = carr.get("coding")
    if coding_cfg is None:
        return max(1, budget_samples // sps), 0, None
    code = build_code(coding_cfg)
    if isinstance(code, LDPCCode):
        code.build_generator()
    syms_per_frame = math.ceil(code.coded_bits / bps)
    native_per_frame = max(1, syms_per_frame * sps)
    n_frames = max(1, budget_samples // native_per_frame)
    return syms_per_frame * n_frames, n_frames, code


def simulate(carriers: list[dict],
                              sample_rate: float,
                              am_am_cfg: dict,
                              am_pm_cfg: dict,
                              max_block_size_samples: int,
                              input_backoff_db: float = 0.0,
                              noise_density_dbfs: float | None = None,
                              ola_filter_span: int = 16,
                              ola_block_size: int = 4096,
                              seed: int | None = None,
                              demod_carriers: set[str] | None = None,
                              chunk_print: _PrintCB = None) -> dict:
    """
    Wideband N-carrier simulation processed chunk-by-chunk (O(1) wideband RAM).

    The wideband composite is never materialised in full.  Each OLA block of
    ola_block_size wideband samples is: formed from per-carrier upsampled
    contributions → normalised → nonlinear amp → optional AWGN → fed to per-
    carrier OLA downsamplers.  A Welch PSD accumulator collects the spectrum of
    all three wideband stages (pre-NL, post-NL, post-noise).

    NLA input normalisation uses the analytical composite RMS derived from
    carrier power_db values (not the empirical signal peak), ensuring a
    deterministic, seed-independent drive level.  See memory/technical_notes.md
    § "NLA input normalization".

    Returns a dict:
        psd_pre_nl   (f, psd_db)  Welch PSD of composite before NL amp
        psd_post_nl  (f, psd_db)  Welch PSD after NL amp
        psd_noisy    (f, psd_db)  Welch PSD after NL + noise
        has_noise    bool
        carriers     list of per-carrier result dicts:
                       name, bb, bits, symbols, t, symbol_rate, native_rate,
                       freq, L, rolloff, filter_span, sps, modulation,
                       mod_kwargs, nl, cnr_db, cir_db, cnir_db,
                       samples, decisions, ber, evm_rms
    """
    rng = np.random.default_rng(seed)
    per_carrier_seeds = rng.integers(0, 2 ** 31, len(carriers))

    # ── Analytical RMS normalization factor ──────────────────────────────────
    # Sum of linear carrier powers (reference: 0 dB = unit power).
    composite_rms = math.sqrt(sum(10 ** (float(c.get("power_db", 0.0)) / 10)
                                  for c in carriers))
    drive        = 10 ** (-input_backoff_db / 20)
    norm_factor  = drive / composite_rms   # scalar applied to each composite chunk

    # ── Generate native-rate baseband for each carrier ───────────────────────
    carrier_state: list[dict] = []
    bb_ch_list:    list[np.ndarray] = []

    for i, carr in enumerate(carriers):
        symbol_rate = float(carr["symbol_rate"])
        sps         = int(carr["sps"])
        rolloff     = float(carr["rolloff"])
        filter_span = int(carr["filter_span"])
        power_db    = float(carr.get("power_db", 0.0))
        freq        = float(carr["freq"])
        channel_cfg = carr.get("channel")

        native_rate = float(sps) * symbol_rate
        L_float = sample_rate / native_rate
        if L_float < 1:
            raise ValueError(
                f"Carrier '{carr['name']}': sample_rate / native_rate = {L_float:.4f} < 1")
        # Round to nearest integer upsample factor; resample_poly corrects the remainder.
        L = max(1, int(math.floor(L_float + 0.5)))
        sr_num = int(round(sample_rate))
        sr_den = int(round(L * native_rate))
        _g = math.gcd(sr_num, sr_den)
        P_rs, Q_rs = sr_num // _g, sr_den // _g

        modulation = carr.get("modulation", "BPSK").upper()
        mod_kwargs = {k: carr[k] for k in ("apsk_gamma", "apsk_gamma1", "apsk_gamma2")
                      if k in carr}

        # Derive symbol / frame counts from the memory budget.  The budget is
        # interpreted per-carrier on the native-rate buffer (num_symbols × sps).
        bps = bits_per_symbol(modulation)
        num_symbols, n_frames, code = _derive_block_counts(
            carr, sps, bps, max_block_size_samples)

        # FEC-coded carrier: encode random data frames and feed the coded bits
        # to the modulator.  Uncoded carrier: rrc_baseband generates the bits.
        if code is not None:
            data_bits, coded_bits = encode_frames(
                code, n_frames, np.random.default_rng(int(per_carrier_seeds[i])))
            bb, t, bits, symbols = rrc_baseband(
                modulation, 0, symbol_rate, native_rate,
                rolloff, filter_span, bits=coded_bits, **mod_kwargs)
        else:
            data_bits = None
            bb, t, bits, symbols = rrc_baseband(
                modulation, num_symbols, symbol_rate, native_rate,
                rolloff, filter_span, seed=int(per_carrier_seeds[i]), **mod_kwargs)

        signal_bw = (1 + rolloff) * symbol_rate
        bb_ch = (apply_channel_impairment(bb, native_rate, signal_bw, channel_cfg)
                 if channel_cfg is not None else bb)

        # Phase noise: per-carrier oscillator phase fluctuation, applied at
        # the carrier's own native bandwidth right after the channel filter
        # and before the OLA upsample / wideband stage.  Reproducible via a
        # per-carrier-derived RNG so reruns of the same seed see the same φ(t).
        # The spec is part of the carrier config — each carrier can have its
        # own oscillator characteristic.
        pn_cfg = carr.get("phase_noise")
        if pn_cfg is not None and pn_cfg.get("enabled", True):
            pn_rng = np.random.default_rng(int(per_carrier_seeds[i]) ^ 0x5A5A_5A5A)
            bb_ch = apply_phase_noise(
                bb_ch, native_rate,
                pn_cfg["offset_hz"], pn_cfg["dbc_per_hz"],
                pn_rng,
            )

        # Rational-resample from native_rate to sample_rate/L (effective native rate)
        # when L_float was non-integer.  P_rs/Q_rs == 1/1 for integer-L carriers.
        n_bb_orig = len(bb_ch)
        if P_rs != Q_rs:
            bb_ch = resample_poly(bb_ch, P_rs, Q_rs).astype(complex)

        bb_ch_list.append(bb_ch)
        carrier_state.append(dict(
            name=carr["name"], bb=bb, bits=bits, symbols=symbols, t=t,
            modulation=modulation, mod_kwargs=mod_kwargs,
            symbol_rate=symbol_rate, native_rate=native_rate, freq=freq, L=L,
            rolloff=rolloff, filter_span=filter_span, sps=sps,
            amp_scale=10 ** (power_db / 20),
            P_rs=P_rs, Q_rs=Q_rs, n_bb_orig=n_bb_orig,
            code=code, data_bits=data_bits, n_frames=n_frames,
        ))

    # ── Wideband extent (trim to shortest carrier, as in the non-chunk path) ─
    N_wb     = min(len(bb_ch) * cr["L"] for bb_ch, cr in zip(bb_ch_list, carrier_state))
    N_chunks = math.ceil(N_wb / ola_block_size)

    # ── Stateful OLA processors ───────────────────────────────────────────────
    # One upsampler per carrier; three downsamplers per demod carrier
    # (pre-NL reference, post-NL noiseless, post-NL+noise).
    up_state   = [OLAState.for_upsample(cr["L"], ola_filter_span, ola_block_size)
                  for cr in carrier_state]

    dn_ref   = [OLAState.for_downsample(cr["L"], ola_filter_span, ola_block_size)
                for cr in carrier_state]
    dn_nl    = [OLAState.for_downsample(cr["L"], ola_filter_span, ola_block_size)
                for cr in carrier_state]
    dn_noisy = [OLAState.for_downsample(cr["L"], ola_filter_span, ola_block_size)
                for cr in carrier_state]

    # Decimation phase offsets (carry between chunks)
    off_ref   = [0] * len(carrier_state)
    off_nl    = [0] * len(carrier_state)
    off_noisy = [0] * len(carrier_state)

    # Native-rate output buffers (small; scale with num_symbols, not N_wb)
    buf_ref   = [[] for _ in carrier_state]
    buf_nl    = [[] for _ in carrier_state]
    buf_noisy = [[] for _ in carrier_state]

    # ── Welch PSD accumulators ────────────────────────────────────────────────
    nfft_welch = min(_WELCH_NFFT, N_wb)
    w_pre  = _WelchState(nfft_welch)
    w_nl   = _WelchState(nfft_welch)
    w_noisy = _WelchState(nfft_welch)

    has_noise   = noise_density_dbfs is not None
    noise_power = (10 ** (noise_density_dbfs / 10) * sample_rate
                   if noise_density_dbfs is not None else 0.0)

    # ── Chunk loop ────────────────────────────────────────────────────────────
    B = ola_block_size

    for k in range(N_chunks):
        chunk_start  = k * B
        actual_size  = min(B, N_wb - chunk_start)

        # Absolute time axis for this chunk (needed for phase-coherent mixing)
        t_chunk = np.arange(chunk_start, chunk_start + actual_size) / sample_rate

        # Form the composite wideband chunk
        composite = np.zeros(actual_size, dtype=complex)
        for i, cr in enumerate(carrier_state):
            x_blk     = x_up_block(bb_ch_list[i], cr["L"], chunk_start, B)
            up_out    = up_state[i].process(x_blk)[:actual_size]
            composite += cr["amp_scale"] * up_out * np.exp(
                1j * 2 * np.pi * cr["freq"] * t_chunk)

        # Normalise → NL amp
        composite_normed = composite * norm_factor
        composite_nl     = nonlinear_amplifier(composite_normed, am_am_cfg, am_pm_cfg)

        # AWGN after amplifier
        if has_noise:
            noise_chunk = np.sqrt(noise_power / 2) * (
                rng.standard_normal(actual_size)
                + 1j * rng.standard_normal(actual_size))
            composite_noisy = composite_nl + noise_chunk
        else:
            composite_noisy = composite_nl

        # Accumulate Welch PSD
        w_pre.add(composite)
        w_nl.add(composite_nl)
        w_noisy.add(composite_noisy)

        # Downsample each demod carrier
        for i, cr in enumerate(carrier_state):
            if demod_carriers is not None and cr["name"] not in demod_carriers:
                continue

            shift = np.exp(-1j * 2 * np.pi * cr["freq"] * t_chunk)
            L     = cr["L"]

            filt_ref   = dn_ref[i].process(composite_normed * shift)
            filt_nl    = dn_nl[i].process(composite_nl    * shift)
            filt_noisy = dn_noisy[i].process(composite_noisy * shift)

            dec_ref,   off_ref[i]   = _decimate(filt_ref,   L, off_ref[i])
            dec_nl,    off_nl[i]    = _decimate(filt_nl,    L, off_nl[i])
            dec_noisy, off_noisy[i] = _decimate(filt_noisy, L, off_noisy[i])

            buf_ref[i].append(dec_ref)
            buf_nl[i].append(dec_nl)
            buf_noisy[i].append(dec_noisy)

        if chunk_print is not None and (
                k % _CHUNK_REPORT == _CHUNK_REPORT - 1 or k == N_chunks - 1):
            chunk_print(f"chunk {k + 1:>{len(str(N_chunks))}}/{N_chunks}")

    # ── Per-carrier demod and metrics ─────────────────────────────────────────
    for i, cr in enumerate(carrier_state):
        if demod_carriers is not None and cr["name"] not in demod_carriers:
            cr.update(nl=None, cnr_db=float("nan"), cir_db=float("nan"),
                      cnir_db=float("nan"), ber=None, evm_rms=float("nan"),
                      n_bits=0, n_errors=0,
                      uncoded_n_bits=0, uncoded_n_errors=0)
            continue

        raw_ref   = np.concatenate(buf_ref[i])   if buf_ref[i]   else np.zeros(0, dtype=complex)
        raw_nl    = np.concatenate(buf_nl[i])    if buf_nl[i]    else np.zeros(0, dtype=complex)
        raw_noisy = np.concatenate(buf_noisy[i]) if buf_noisy[i] else np.zeros(0, dtype=complex)

        # Strip the filter transient: each OLA stage (upsample + downsample) introduces
        # a group delay of ola_filter_span native-rate samples (n_half/L = filter_span).
        # Trimming 2*ola_filter_span from the start restores alignment with reference bits.
        trim        = 2 * ola_filter_span
        n_native_rs = len(bb_ch_list[i])   # length at effective native rate
        bb_rx   = raw_ref  [trim : trim + n_native_rs]
        nl_pure = raw_nl   [trim : trim + n_native_rs]
        nl_down = raw_noisy[trim : trim + n_native_rs]

        # Reverse the rational resample to restore original native rate and integer sps.
        # P_rs/Q_rs == 1/1 for integer-L carriers; the branch is a no-op in that case.
        p_rs_i = cr["P_rs"]; q_rs_i = cr["Q_rs"]; n_orig = cr["n_bb_orig"]
        if p_rs_i != q_rs_i:
            bb_rx   = resample_poly(bb_rx,   q_rs_i, p_rs_i).astype(complex)[:n_orig]
            nl_pure = resample_poly(nl_pure, q_rs_i, p_rs_i).astype(complex)[:n_orig]
            nl_down = resample_poly(nl_down, q_rs_i, p_rs_i).astype(complex)[:n_orig]

        # Project nl_pure onto bb_rx to separate linear gain from true IM distortion.
        alpha      = np.vdot(bb_rx, nl_pure) / (np.vdot(bb_rx, bb_rx) + 1e-30)
        sig        = alpha * bb_rx
        distortion = nl_pure - sig

        p_sig   = float(np.mean(np.abs(sig) ** 2))
        p_dist  = float(np.mean(np.abs(distortion) ** 2))
        p_noise = (float(np.mean(np.abs(nl_down - nl_pure) ** 2))
                   if has_noise else 0.0)

        # Normalise noise to the symbol-rate bandwidth (÷ sps) so that CNR and CNIR
        # match the link-budget convention: noise bandwidth ≈ symbol_rate for RRC.
        sps_f   = float(cr["sps"])
        p_noise_bw = p_noise / sps_f
        eps     = 1e-30
        cir_db  = 10.0 * np.log10(p_sig / (p_dist + eps))
        cnr_db  = (10.0 * np.log10(p_sig / p_noise_bw) if p_noise > 0 else float("inf"))
        cnir_db = 10.0 * np.log10(p_sig / (p_dist + p_noise_bw + eps))

        cr["nl"]      = nl_down
        cr["cnr_db"]  = cnr_db
        cr["cir_db"]  = cir_db
        cr["cnir_db"] = cnir_db
        rx = receive(
            nl_down,
            modulation=cr["modulation"],
            rolloff=cr["rolloff"],
            filter_span=cr["filter_span"],
            sps=cr["sps"],
            reference_bits=cr["bits"],
            **cr["mod_kwargs"],
        )
        # Channel-bit counts (denominator/numerator for the pre-FEC BER).  For
        # uncoded carriers these double as the primary n_bits/n_errors below.
        n_channel = rx.get("n_bits", 0)
        e_channel = rx.get("n_errors", 0)

        # For a coded carrier, soft-demap the symbol samples, FEC-decode, and
        # report the post-decoder BER as `ber` (the channel BER becomes uncoded_ber).
        if cr["code"] is not None:
            evm = rx["evm_rms"]
            noise_var = (evm / 100.0) ** 2 if (not math.isnan(evm) and evm > 0) else 1.0
            llrs = soft_demap(rx["samples"], cr["modulation"], noise_var,
                              **cr["mod_kwargs"])
            decoded = decode_frames(cr["code"], llrs, cr["n_frames"])
            data = cr["data_bits"]
            n = min(len(decoded), len(data))
            rx["uncoded_ber"]      = rx["ber"]
            rx["uncoded_n_bits"]   = n_channel
            rx["uncoded_n_errors"] = e_channel
            n_post = int(n)
            e_post = int(np.sum(decoded[:n] != data[:n]))
            rx["ber"]      = (e_post / n_post) if n_post > 0 else None
            rx["n_bits"]   = n_post
            rx["n_errors"] = e_post
        else:
            rx["uncoded_n_bits"]   = 0
            rx["uncoded_n_errors"] = 0
        cr.update(rx)

    # ── Finalise Welch PSDs ───────────────────────────────────────────────────
    psd_pre_nl  = w_pre.result(sample_rate)
    psd_post_nl = w_nl.result(sample_rate)
    psd_noisy   = w_noisy.result(sample_rate) if has_noise else psd_post_nl

    return dict(
        psd_pre_nl=psd_pre_nl,
        psd_post_nl=psd_post_nl,
        psd_noisy=psd_noisy,
        has_noise=has_noise,
        carriers=carrier_state,
    )
