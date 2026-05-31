"""Receive chain: matched filter, sampling, decisions, BER, EVM, and soft LLRs."""
import numpy as np
from .filters import rrc_coeffs, ola_convolve
from .modulation import bits_per_symbol, constellation, decide, differential_decode, rotational_symmetry


def matched_filter(signal: np.ndarray, rolloff: float,
                   filter_span: int, sps: int) -> np.ndarray:
    """Apply receive-side RRC matched filter via OLA convolution.

    Strips the filter group delay so that symbol centres align with
    the same indices they occupied in the transmit baseband (i.e. 0, sps, 2*sps …).
    Returns an array the same length as the input.
    """
    h = rrc_coeffs(filter_span, rolloff, sps)
    delay = len(h) // 2
    y_full = ola_convolve(signal, h)
    return y_full[delay : delay + len(signal)]


def measure_evm_rms(samples: np.ndarray, ideal: np.ndarray) -> float:
    """RMS EVM as a percentage of the RMS constellation radius.

    samples : complex received samples (one per symbol)
    ideal   : complex ideal constellation points (nearest decision)
    """
    n = min(len(samples), len(ideal))
    s = samples[:n]
    d = np.asarray(ideal[:n], dtype=complex)
    rms_rx = float(np.sqrt(np.mean(np.abs(s) ** 2)))
    if rms_rx < 1e-30:
        return float("nan")
    norm = s / rms_rx
    # Normalise ideal by its own RMS so EVM is reference-independent
    rms_ref = float(np.sqrt(np.mean(np.abs(d) ** 2)))
    d_norm = d / rms_ref if rms_ref > 1e-30 else d
    return 100.0 * float(np.sqrt(np.mean(np.abs(norm - d_norm) ** 2)))


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    """Numerically stable log(sum(exp(values))) reduced along an axis."""
    peak = np.max(values, axis=axis, keepdims=True)
    summed = np.sum(np.exp(values - peak), axis=axis, keepdims=True)
    return np.squeeze(peak + np.log(summed), axis=axis)


def soft_demap(samples: np.ndarray, modulation: str, noise_var: float,
               **mod_kwargs) -> np.ndarray:
    """Exact per-bit log-likelihood ratios (LLRs) for constellation-mapped symbols.

    Returns one LLR per coded bit, MSB-first within each symbol (matching
    map_bits / decide), flattened to length n_symbols * bits_per_symbol.

    Sign convention: LLR = log P(bit=0 | y) - log P(bit=1 | y), so a positive
    LLR favours bit 0.  The exact value marginalises over the constellation:

        LLR = logsumexp_{s: bit=0}(-|y - s|^2 / noise_var)
            - logsumexp_{s: bit=1}(-|y - s|^2 / noise_var)

    noise_var is the total noise power per complex sample, on the same scale as
    the unit-average-power constellation, and must be positive.
    """
    mod = modulation.upper()
    C = constellation(mod, **mod_kwargs)
    bps = bits_per_symbol(mod)
    y = np.asarray(samples, dtype=complex)

    metric = -np.abs(y[:, np.newaxis] - C[np.newaxis, :]) ** 2 / noise_var
    sym_idx = np.arange(len(C))
    llrs = np.empty((len(y), bps))
    for i in range(bps):
        is_zero = ((sym_idx >> (bps - 1 - i)) & 1) == 0
        llrs[:, i] = (_logsumexp(metric[:, is_zero], axis=1)
                      - _logsumexp(metric[:, ~is_zero], axis=1))
    return llrs.ravel()


def receive(signal: np.ndarray,
            modulation: str,
            rolloff: float,
            filter_span: int,
            sps: int,
            reference_bits: np.ndarray | None = None,
            **mod_kwargs) -> dict:
    """Full receive chain for any supported modulation.

    Pipeline: RRC matched filter (group-delay compensated) → symbol sampling
    (I at [0::sps]; for OQPSK also Q at [sps//2::sps]) → nearest-neighbour
    hard decision → for DBPSK, differential decode → phase-ambiguity-resolved
    BER (tries all rotationally symmetric equivalents) → RMS EVM.

    Inputs: `signal` is the complex baseband at native sample rate;
    `modulation` is the modulation name; `rolloff` and `filter_span`
    configure the RRC matched filter; `sps` is samples per symbol;
    `reference_bits` is the transmitted data bits used for BER (None
    skips BER); `**mod_kwargs` are forwarded to the constellation /
    decide helpers (e.g. `apsk_gamma`).

    Returns a dict with keys ``samples``, ``decisions``, ``ber``,
    ``evm_rms``, ``n_bits``, ``n_errors``.
    """
    mod = modulation.upper()
    if mod == "MSK":
        return _msk_receive(signal, sps, reference_bits)
    mf = matched_filter(signal, rolloff, filter_span, sps)

    if mod == "OQPSK":
        # I rail peaks at [0::sps], Q rail (delayed T/2 at TX) peaks at [sps//2::sps]
        I_samp = np.real(mf)[0::sps]
        Q_samp = np.imag(mf)[sps // 2::sps]
        n = min(len(I_samp), len(Q_samp))
        samples = (I_samp[:n] + 1j * Q_samp[:n]).astype(complex)
    else:
        samples = mf[::sps]

    # Normalise to the unit-average-power constellation before nearest-neighbour
    # decision.  The baseband generator normalises signal RMS, not symbol amplitude,
    # so the received symbol values are scaled by a factor k ≠ 1 for multi-amplitude
    # constellations (QAM, APSK).  Dividing by the sample RMS recovers the correct
    # scale for distance comparisons.
    rms_s = float(np.sqrt(np.mean(np.abs(samples) ** 2)))
    samples_norm = samples / rms_s if rms_s > 1e-30 else samples

    sym_decisions, bit_decisions = decide(samples_norm, mod, **mod_kwargs)

    n_bits = 0
    n_errors = 0
    if mod == "DBPSK":
        # Differential decode: N decisions → N-1 bits; compare with reference[1:]
        bit_decisions = differential_decode(sym_decisions)
        if reference_bits is not None:
            ref = np.asarray(reference_bits, dtype=int)
            n = min(len(bit_decisions), len(ref) - 1)
            n_bits = int(n)
            n_errors = int(np.sum(bit_decisions[:n] != ref[1 : n + 1]))
            ber = (n_errors / n_bits) if n_bits > 0 else None
        else:
            ber = None
    elif reference_bits is not None:
        ber, n_bits, n_errors = _ber_with_ambiguity(
            samples_norm, reference_bits, mod, **mod_kwargs)
    else:
        ber = None

    evm = measure_evm_rms(samples_norm, sym_decisions)
    return dict(samples=samples_norm, decisions=bit_decisions, ber=ber, evm_rms=evm,
                n_bits=n_bits, n_errors=n_errors)


def _msk_receive(signal: np.ndarray, sps: int,
                 reference_bits: np.ndarray | None) -> dict:
    """Coherent MSK receiver via the offset-QPSK / half-sine matched filter.

    MSK is offset-QPSK with a half-sine pulse (see sim.baseband._msk_baseband):
    the in-phase rail carries even-indexed bits, the quadrature rail (delayed
    by sps) carries odd-indexed bits.  Each rail is matched-filtered with the
    same 2*sps half-sine pulse, giving two independent antipodal decisions and
    hence the BPSK error rate.  A residual 180 degree ambiguity (BER > 0.5) is
    corrected by inverting every decision.

    EVM is not defined for a constant-envelope signal, so evm_rms is NaN.
    """
    n_sym = len(signal) // sps
    total = n_sym * sps
    n_i = (n_sym + 1) // 2
    n_q = n_sym // 2
    pulse = np.sin(np.pi * np.arange(2 * sps) / (2.0 * sps))

    # In-phase rail: matched filter over non-overlapping 2*sps windows.
    i_buf = np.zeros(n_i * 2 * sps)
    i_buf[:total] = np.real(signal[:total])
    i_dec = (i_buf.reshape(n_i, 2 * sps) @ pulse <= 0.0).astype(int)

    # Quadrature rail: same, but shifted by sps (the offset-QPSK delay).
    q_buf = np.zeros(n_q * 2 * sps)
    q_buf[:total - sps] = np.imag(signal[sps:total])
    q_dec = (q_buf.reshape(n_q, 2 * sps) @ pulse <= 0.0).astype(int)

    bit_decisions = np.empty(n_sym, dtype=int)
    bit_decisions[0::2] = i_dec
    bit_decisions[1::2] = q_dec

    ber: float | None = None
    n_bits = 0
    n_errors = 0
    if reference_bits is not None:
        ref = np.asarray(reference_bits[:n_sym], dtype=int)
        n = min(len(bit_decisions), len(ref))
        n_bits = int(n)
        n_errors = int(np.sum(bit_decisions[:n] != ref[:n]))
        if n_errors * 2 > n_bits:
            n_errors = n_bits - n_errors
            bit_decisions = 1 - bit_decisions
        ber = (n_errors / n_bits) if n_bits > 0 else None

    return dict(samples=signal[:total:sps], decisions=bit_decisions,
                ber=ber, evm_rms=float("nan"),
                n_bits=n_bits, n_errors=n_errors)


def _ber_with_ambiguity(samples: np.ndarray, reference_bits: np.ndarray,
                        mod: str, **mod_kwargs) -> tuple[float, int, int]:
    """BER with phase-ambiguity resolution.

    Tries all N rotationally equivalent orientations of the received samples
    (where N = rotational_symmetry(mod)) and returns (best_ber, n_bits, best_errors)
    for the orientation that minimises the error count.  This handles the
    systematic phase offset introduced by AM-PM without requiring explicit
    carrier phase recovery.
    """
    ref = np.asarray(reference_bits, dtype=int)
    n_rot = rotational_symmetry(mod)
    best_n_bits = 0
    best_errors = 0
    best_ber = 1.0
    for k in range(n_rot):
        angle = k * 2 * np.pi / n_rot
        rotated = samples * np.exp(1j * angle)
        _, bit_dec = decide(rotated, mod, **mod_kwargs)
        n = min(len(bit_dec), len(ref))
        errs_k = int(np.sum(bit_dec[:n] != ref[:n]))
        ber_k = (errs_k / n) if n > 0 else 1.0
        if ber_k < best_ber:
            best_ber = ber_k
            best_n_bits = int(n)
            best_errors = errs_k
    return best_ber, best_n_bits, best_errors
