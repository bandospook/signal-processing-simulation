import numpy as np
from filters import rrc_coeffs, ola_convolve


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


def symbol_sample(filtered: np.ndarray, sps: int) -> np.ndarray:
    """Decimate matched-filtered signal to one sample per symbol."""
    return filtered[::sps]


def bpsk_decide(samples: np.ndarray) -> np.ndarray:
    """Hard BPSK decisions on the real part; returns +1 / -1 array."""
    return np.where(samples.real >= 0, 1, -1)


def measure_ber(decisions: np.ndarray, reference: np.ndarray) -> float:
    """
    BER against reference symbols (+1/-1).

    Checks both polarities and returns the lower error rate, resolving
    the inherent BPSK 0/π phase ambiguity.
    """
    n = min(len(decisions), len(reference))
    dec = decisions[:n]
    ref = np.asarray(reference[:n], dtype=int)
    errors = min(np.sum(dec != ref), np.sum(dec != -ref))
    return float(errors) / n


def measure_evm_rms(samples: np.ndarray, decisions: np.ndarray) -> float:
    """
    RMS EVM as a percentage of the unit constellation radius.

    Samples are normalised to unit RMS power before comparison against
    the ideal ±1 BPSK constellation points.
    """
    n = min(len(samples), len(decisions))
    s = samples[:n]
    d = decisions[:n].astype(complex)
    rms = float(np.sqrt(np.mean(np.abs(s) ** 2)))
    if rms < 1e-30:
        return float("nan")
    norm = s / rms
    return 100.0 * float(np.sqrt(np.mean(np.abs(norm - d) ** 2)))


def bpsk_receive(signal: np.ndarray, rolloff: float,
                 filter_span: int, sps: int,
                 reference_symbols: np.ndarray | None = None) -> dict:
    """
    Full BPSK receive chain: matched filter → symbol sample → decide → metrics.

    Returns:
        samples    complex samples at 1 sample/symbol
        decisions  hard decisions (+1 / -1)
        ber        bit error rate (None if no reference provided)
        evm_rms    RMS EVM in percent
    """
    mf = matched_filter(signal, rolloff, filter_span, sps)
    samples = symbol_sample(mf, sps)
    decisions = bpsk_decide(samples)

    ber = measure_ber(decisions, reference_symbols) if reference_symbols is not None else None
    evm = measure_evm_rms(samples, decisions)

    return dict(samples=samples, decisions=decisions, ber=ber, evm_rms=evm)
