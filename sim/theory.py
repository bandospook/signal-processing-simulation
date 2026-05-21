"""Closed-form AWGN BER curves and their numerical inverses."""

import math
from scipy.optimize import brentq
from .modulation import bits_per_symbol


def ber_awgn(mod: str, EsN0_dB: float) -> float | None:
    """
    Theoretical BER in pure AWGN for the given modulation at Es/N0 (dB).

    Returns None for modulations that have no closed-form formula (16APSK, 32APSK).

    DBPSK uses the coherent-detection + differential-decoding formula
    (BER = 2p(1-p), p = 0.5·erfc(√Eb/N0)) rather than the differentially-coherent
    formula (0.5·exp(-Eb/N0)), because the simulation uses an RRC matched filter
    followed by differential decoding of hard symbol decisions.
    """
    bps = bits_per_symbol(mod.upper())
    EbN0 = 10.0 ** (EsN0_dB / 10.0) / bps
    m = mod.upper()
    if m in ("BPSK", "QPSK", "OQPSK", "MSK"):
        return 0.5 * math.erfc(math.sqrt(EbN0))
    if m == "DBPSK":
        p = 0.5 * math.erfc(math.sqrt(EbN0))
        return 2.0 * p * (1.0 - p)
    if m == "8PSK":
        return (1.0 / 3.0) * math.erfc(math.sqrt(3.0 * EbN0) * math.sin(math.pi / 8))
    if m == "16QAM":
        return (3.0 / 8.0) * math.erfc(math.sqrt(2.0 * EbN0 / 5.0))
    return None


def ebn0_for_ber(mod: str, target_ber: float,
                 ebn0_lo_db: float = -5.0,
                 ebn0_hi_db: float = 25.0,
                 tol_db: float = 1e-4) -> float | None:
    """
    Numerically invert ber_awgn(): find Eb/N0 (dB) such that BER = target_ber.

    Returns None if no closed-form formula is available for this modulation, or if
    target_ber lies outside the BER range achievable within [ebn0_lo_db, ebn0_hi_db].
    """
    bps = bits_per_symbol(mod.upper())

    def theory_ber(ebn0_db: float) -> float:
        esn0_db = ebn0_db + 10.0 * math.log10(bps)
        v = ber_awgn(mod, esn0_db)
        return v if v is not None else float("nan")

    ber_lo = theory_ber(ebn0_lo_db)
    ber_hi = theory_ber(ebn0_hi_db)

    if math.isnan(ber_lo):
        return None  # no formula for this modulation

    # BER decreases as Eb/N0 increases, so ber_lo > ber_hi.
    if target_ber > ber_lo or target_ber < ber_hi:
        return None  # out of range

    return float(brentq(lambda e: theory_ber(e) - target_ber,  # type: ignore[arg-type]
                        ebn0_lo_db, ebn0_hi_db, xtol=tol_db))
