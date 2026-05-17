"""Receive chain: matched filter, symbol sampling, decisions, BER, and EVM."""
import numpy as np
from .filters import rrc_coeffs, ola_convolve
from .modulation import decide, differential_decode, rotational_symmetry


def matched_filter(signal: np.ndarray, rolloff: float,
                   filter_span: int, sps: int) -> np.ndarray:
    """
    Apply receive-side RRC matched filter via OLA convolution.

    Strips the filter group delay so that symbol centres align with
    the same indices they occupied in the transmit baseband (i.e. 0, sps, 2*sps …).
    Returns an array the same length as the input.
    """
    h = rrc_coeffs(filter_span, rolloff, sps)
    delay = len(h) // 2
    y_full = ola_convolve(signal, h)
    return y_full[delay : delay + len(signal)]


def measure_evm_rms(samples: np.ndarray, ideal: np.ndarray) -> float:
    """
    RMS EVM as a percentage of the RMS constellation radius.

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


def receive(signal: np.ndarray,
            modulation: str,
            rolloff: float,
            filter_span: int,
            sps: int,
            reference_bits: np.ndarray | None = None,
            **mod_kwargs) -> dict:
    """
    Full receive chain for any supported modulation.

    Steps
    -----
    1. RRC matched filter (group-delay compensated)
    2. Symbol sampling  — I at [0::sps]; for OQPSK also Q at [sps//2::sps]
    3. Nearest-neighbour hard decision
    4. For DBPSK: differential decode
    5. Phase-ambiguity-resolved BER (tries all rotationally symmetric equivalents)
    6. RMS EVM

    Parameters
    ----------
    signal         : complex baseband at native sample rate
    modulation     : modulation name string
    rolloff        : RRC rolloff factor
    filter_span    : RRC filter half-span in symbols
    sps            : samples per symbol
    reference_bits : transmitted data bits for BER (None → BER not computed)
    **mod_kwargs   : passed to constellation/decide (e.g. apsk_gamma)

    Returns
    -------
    dict with keys: samples, decisions, ber, evm_rms
    """
    mod = modulation.upper()
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

    if mod == "DBPSK":
        # Differential decode: N decisions → N-1 bits; compare with reference[1:]
        bit_decisions = differential_decode(sym_decisions)
        if reference_bits is not None:
            ref = np.asarray(reference_bits, dtype=int)
            n = min(len(bit_decisions), len(ref) - 1)
            ber = float(np.mean(bit_decisions[:n] != ref[1 : n + 1]))
        else:
            ber = None
    elif reference_bits is not None:
        ber = _ber_with_ambiguity(samples_norm, reference_bits, mod, **mod_kwargs)
    else:
        ber = None

    evm = measure_evm_rms(samples_norm, sym_decisions)
    return dict(samples=samples_norm, decisions=bit_decisions, ber=ber, evm_rms=evm)


def _ber_with_ambiguity(samples: np.ndarray, reference_bits: np.ndarray,
                        mod: str, **mod_kwargs) -> float:
    """
    BER with phase-ambiguity resolution.

    Tries all N rotationally equivalent orientations of the received samples
    (where N = rotational_symmetry(mod)) and returns the minimum BER.
    This handles the systematic phase offset introduced by AM-PM without
    requiring explicit carrier phase recovery.
    """
    ref = np.asarray(reference_bits, dtype=int)
    n_rot = rotational_symmetry(mod)
    best = 1.0
    for k in range(n_rot):
        angle = k * 2 * np.pi / n_rot
        rotated = samples * np.exp(1j * angle)
        _, bit_dec = decide(rotated, mod, **mod_kwargs)
        n = min(len(bit_dec), len(ref))
        ber_k = float(np.mean(bit_dec[:n] != ref[:n]))
        if ber_k < best:
            best = ber_k
    return best
