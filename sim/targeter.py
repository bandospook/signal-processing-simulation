"""
Adaptive BER seeker: find noise_density_dbfs that achieves a target BER for a
specified carrier, then report implementation loss vs AWGN theory.

The bisection variable is noise_density_dbfs.  Higher noise → higher BER, so
noise_lo_dbfs (most negative, quietest) gives the lowest BER and noise_hi_dbfs
(least negative, loudest) gives the highest.  A lossless, linear amplifier
should produce implementation_loss_db ≈ 0 because CNIR = CNR in that case.
"""

import math
from typing import Callable
import numpy as np

from .modulation import bits_per_symbol
from .simulation import wideband_bpsk_simulation
from .theory import ebn0_for_ber

_ProgressCB = Callable[[float, str], None] | None


def _erfinv(p: float) -> float:
    """
    Inverse error function via bisection on math.erf.
    Works for p ∈ (-1, 1).  Used only for computing normal-distribution
    quantiles from a confidence level, so the small bisection overhead is fine.
    """
    if p < 0:
        return -_erfinv(-p)
    if p == 0.0:
        return 0.0
    lo, hi = 0.0, 6.0  # erf(6) ≈ 1 - 2e-17, safe upper bound
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if math.erf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _n_bits_for_ci(target_ber: float, confidence: float = 0.95,
                   accuracy: float = 0.001) -> int:
    """
    Minimum bit count so that the normal-approximation CI is ±accuracy at
    target_ber with the given confidence.

    Derivation: sigma = sqrt(p*(1-p)/N) ≤ accuracy / z  →  N ≥ (z/accuracy)² * p*(1-p)
    where z = sqrt(2) * erfinv(confidence) (the standard normal quantile for
    a two-sided interval at the given confidence).
    """
    z = math.sqrt(2.0) * _erfinv(confidence)
    return math.ceil((z / accuracy) ** 2 * target_ber * (1.0 - target_ber))


def _simulate_ber_at_noise(
    noise_dbfs: float,
    carrier_name: str,
    carriers: list[dict],
    sample_rate: float,
    am_am_cfg: dict,
    am_pm_cfg: dict,
    input_backoff_db: float,
    ola_filter_span: int,
    ola_block_size: int,
    n_bits: int,
    seeds: list[int],
) -> tuple[float, float, float, float]:
    """
    Run the wideband simulation at noise_dbfs and return
    (ber, cnr_db, cir_db, cnir_db) pooled/averaged across seeds.

    Only the target carrier is demodulated; others are included in the
    composite for correct nonlinear loading but skip the expensive decode step.
    n_bits is split equally across seeds so total bits ≈ n_bits.
    """
    target_idx = next(
        (i for i, c in enumerate(carriers) if c["name"] == carrier_name), None)
    if target_idx is None:
        raise ValueError(f"Carrier '{carrier_name}' not found in carrier list")

    modulation = carriers[target_idx].get("modulation", "BPSK").upper()
    bps = bits_per_symbol(modulation)
    n_sym_per_seed = max(1, n_bits // (bps * len(seeds)))

    bers: list[float] = []
    cnrs: list[float] = []
    cirs: list[float] = []
    cnirs: list[float] = []

    for s in seeds:
        carriers_run = [dict(c) for c in carriers]
        carriers_run[target_idx] = dict(carriers[target_idx],
                                        num_symbols=n_sym_per_seed)

        result = wideband_bpsk_simulation(
            carriers_run, sample_rate, am_am_cfg, am_pm_cfg,
            input_backoff_db=input_backoff_db,
            noise_density_dbfs=noise_dbfs,
            ola_filter_span=ola_filter_span,
            ola_block_size=ola_block_size,
            seed=s,
            demod_carriers={carrier_name},
        )

        for cr in result["carriers"]:
            if cr["name"] == carrier_name:
                bers.append(cr["ber"] if cr["ber"] is not None else 0.0)
                cnrs.append(cr["cnr_db"])
                cirs.append(cr["cir_db"])
                cnirs.append(cr["cnir_db"])
                break

    return (
        float(np.mean(bers)),
        float(np.mean(cnrs)),
        float(np.mean(cirs)),
        float(np.mean(cnirs)),
    )


def seek_ber_noise_level(
    target_ber: float,
    confidence: float,
    ber_accuracy: float,
    carrier_name: str,
    carriers: list[dict],
    sample_rate: float,
    am_am_cfg: dict,
    am_pm_cfg: dict,
    input_backoff_db: float = 0.0,
    ola_filter_span: int = 16,
    ola_block_size: int = 4096,
    noise_lo_dbfs: float = -160.0,
    noise_hi_dbfs: float = -80.0,
    max_iter: int = 20,
    n_final_seeds: int = 5,
    seed: int = 42,
    progress_callback: _ProgressCB = None,
) -> dict:
    """
    Adaptive bisection to find the noise_density_dbfs that achieves target_ber
    for carrier_name, then measure implementation loss vs AWGN theory.

    Bracket convention: noise_lo_dbfs (quietest) → lowest BER; noise_hi_dbfs
    (loudest) → highest BER.  The bracket must straddle target_ber or a
    ValueError is raised.

    N_bits per bisection step doubles every two iterations (coarse → fine),
    capped at the n_bits_final computed from confidence/ber_accuracy.  The
    final measurement pools n_final_seeds independent seeds for that full
    bit budget.

    progress_callback(frac, msg): optional callable called at key steps.
    frac is in [0, 1]; msg is a human-readable status string.

    Returns a dict:
        noise_density_dbfs    — converged noise level (dBFS/Hz)
        ber                   — measured BER at convergence
        ber_ci_lo / ber_ci_hi — normal-approximation confidence interval
        effective_ebn0_db     — C/(N+I) per bit in dB
        theory_ebn0_db        — theory Eb/N0 for measured BER (None if no formula)
        implementation_loss_db — effective - theory (None if no formula)
        cnr_db, cir_db, cnir_db
        n_bits_total          — total bits used in final measurement
        n_iter                — bisection steps taken
    """
    def _cb(frac: float, msg: str) -> None:
        if progress_callback is not None:
            progress_callback(frac, msg)

    rng = np.random.default_rng(seed)

    n_bits_final = _n_bits_for_ci(target_ber, confidence, ber_accuracy)
    # Early bisection steps use fewer bits for speed; doubles every 2 steps.
    n_bits_initial = max(500, n_bits_final // 32)

    target_idx = next(
        (i for i, c in enumerate(carriers) if c["name"] == carrier_name), None)
    if target_idx is None:
        raise ValueError(f"Carrier '{carrier_name}' not found")

    modulation = carriers[target_idx].get("modulation", "BPSK").upper()
    bps = bits_per_symbol(modulation)
    sps = int(carriers[target_idx].get("sps", 4))

    bisect_seed = [int(rng.integers(0, 2 ** 31))]

    _cb(0.03, f"[seeker] '{carrier_name}' — bracket check (lo: {noise_lo_dbfs:.1f} dBFS)...")
    ber_at_lo, *_ = _simulate_ber_at_noise(
        noise_lo_dbfs, carrier_name, carriers, sample_rate,
        am_am_cfg, am_pm_cfg, input_backoff_db,
        ola_filter_span, ola_block_size, n_bits_initial, bisect_seed)

    _cb(0.08, f"[seeker] '{carrier_name}' — bracket check (hi: {noise_hi_dbfs:.1f} dBFS)...")
    ber_at_hi, *_ = _simulate_ber_at_noise(
        noise_hi_dbfs, carrier_name, carriers, sample_rate,
        am_am_cfg, am_pm_cfg, input_backoff_db,
        ola_filter_span, ola_block_size, n_bits_initial, bisect_seed)

    if ber_at_lo > target_ber:
        raise ValueError(
            f"BER at noise_lo_dbfs={noise_lo_dbfs:.1f} dBFS is {ber_at_lo:.4f} "
            f"> target {target_ber:.4f}; lower noise_lo_dbfs (more negative) "
            f"to widen the bracket"
        )
    if ber_at_hi < target_ber:
        raise ValueError(
            f"BER at noise_hi_dbfs={noise_hi_dbfs:.1f} dBFS is {ber_at_hi:.4f} "
            f"< target {target_ber:.4f}; raise noise_hi_dbfs (less negative) "
            f"to widen the bracket"
        )

    lo, hi = noise_lo_dbfs, noise_hi_dbfs
    n_iter = 0

    for k in range(max_iter):
        n_bits_step = min(n_bits_final, n_bits_initial * (1 << (k // 2)))
        mid = (lo + hi) / 2.0
        ber_mid, *_ = _simulate_ber_at_noise(
            mid, carrier_name, carriers, sample_rate,
            am_am_cfg, am_pm_cfg, input_backoff_db,
            ola_filter_span, ola_block_size, n_bits_step, bisect_seed)
        n_iter += 1

        step_frac = 0.10 + 0.75 * (k / max_iter)
        _cb(step_frac,
            f"[seeker] '{carrier_name}' — step {k + 1}/{max_iter}  "
            f"BER {ber_mid:.2e} → target {target_ber:.2e}  "
            f"noise {mid:.2f} dBFS")

        # Higher noise → higher BER.  If mid is too noisy (BER > target), pull hi down.
        if ber_mid > target_ber:
            hi = mid
        else:
            lo = mid

        if hi - lo < 0.05:
            break

    converged_noise = (lo + hi) / 2.0

    _cb(0.88, f"[seeker] '{carrier_name}' — final measurement ({n_final_seeds} seeds)...")
    # Final pooled measurement at full bit budget
    final_seeds = [int(x) for x in rng.integers(0, 2 ** 31, n_final_seeds)]
    ber_final, cnr_db, cir_db, cnir_db = _simulate_ber_at_noise(
        converged_noise, carrier_name, carriers, sample_rate,
        am_am_cfg, am_pm_cfg, input_backoff_db,
        ola_filter_span, ola_block_size, n_bits_final, final_seeds)

    n_sym_final = max(1, n_bits_final // (bps * n_final_seeds))
    n_bits_total = n_sym_final * bps * n_final_seeds

    z_ci = math.sqrt(2.0) * _erfinv(confidence)
    sigma = math.sqrt(max(ber_final, 1e-10) * (1.0 - ber_final) / n_bits_total)
    ber_ci_lo = max(0.0, ber_final - z_ci * sigma)
    ber_ci_hi = min(1.0, ber_final + z_ci * sigma)

    # CNIR is measured at the native sample rate (sps samples/symbol), so it is
    # sps × Eb/N0 per bit.  Dividing by bps and multiplying by sps converts from
    # the per-sample power ratio to Eb/N0.  For linear, distortion-free: IL = 0.
    effective_ebn0_db = cnir_db + 10.0 * math.log10(sps / bps)
    theory_ebn0_db = ebn0_for_ber(modulation, ber_final)
    implementation_loss_db = (
        effective_ebn0_db - theory_ebn0_db
        if theory_ebn0_db is not None else None
    )

    _cb(1.0, f"[seeker] '{carrier_name}' — done.  "
        f"BER={ber_final:.3e}  IL={implementation_loss_db:.2f} dB"
        if implementation_loss_db is not None else
        f"[seeker] '{carrier_name}' — done.  BER={ber_final:.3e}")

    return dict(
        noise_density_dbfs=converged_noise,
        ber=ber_final,
        ber_ci_lo=ber_ci_lo,
        ber_ci_hi=ber_ci_hi,
        effective_ebn0_db=effective_ebn0_db,
        theory_ebn0_db=theory_ebn0_db,
        implementation_loss_db=implementation_loss_db,
        cnr_db=cnr_db,
        cir_db=cir_db,
        cnir_db=cnir_db,
        n_bits_total=n_bits_total,
        n_iter=n_iter,
    )


def seek_all_carriers(
    carriers: list[dict],
    sample_rate: float,
    am_am_cfg: dict,
    am_pm_cfg: dict,
    target_ber: float = 0.01,
    confidence: float = 0.95,
    ber_accuracy: float = 0.005,
    input_backoff_db: float = 0.0,
    ola_filter_span: int = 16,
    ola_block_size: int = 4096,
    noise_lo_dbfs: float = -160.0,
    noise_hi_dbfs: float = -80.0,
    max_iter: int = 20,
    n_final_seeds: int = 5,
    seed: int = 42,
    progress_callback: _ProgressCB = None,
) -> dict[str, dict]:
    """
    Run seek_ber_noise_level for every carrier with enabled=True,
    sweep_demod=True, and use_seeker=True.

    Per-carrier seeker parameters are read from carr["seeker"] sub-dict:
        target_ber, confidence, ber_accuracy, noise_lo_dbfs, noise_hi_dbfs

    Falls back to the global parameters for any key not present in the
    per-carrier sub-dict.

    progress_callback(frac, msg) receives overall progress across all seekable
    carriers, with each carrier allocated an equal fraction of [0, 1].

    Returns a dict keyed by carrier name.
    """
    def _cb(frac: float, msg: str) -> None:
        if progress_callback is not None:
            progress_callback(frac, msg)

    rng = np.random.default_rng(seed)

    seekable = [
        c for c in carriers
        if c.get("enabled", True)
        and c.get("sweep_demod", False)
        and c.get("use_seeker", False)
    ]

    results: dict[str, dict] = {}
    n = len(seekable)

    for i, carr in enumerate(seekable):
        lo_frac = i / n if n > 0 else 0.0
        hi_frac = (i + 1) / n if n > 0 else 1.0

        def _carrier_cb(frac: float, msg: str,
                        lo: float = lo_frac, hi: float = hi_frac) -> None:
            _cb(lo + frac * (hi - lo), msg)

        sk = carr.get("seeker", {})
        carr_seed = int(rng.integers(0, 2 ** 31))

        results[carr["name"]] = seek_ber_noise_level(
            target_ber=sk.get("target_ber", target_ber),
            confidence=sk.get("confidence", confidence),
            ber_accuracy=sk.get("ber_accuracy", ber_accuracy),
            carrier_name=carr["name"],
            carriers=carriers,
            sample_rate=sample_rate,
            am_am_cfg=am_am_cfg,
            am_pm_cfg=am_pm_cfg,
            input_backoff_db=input_backoff_db,
            ola_filter_span=ola_filter_span,
            ola_block_size=ola_block_size,
            noise_lo_dbfs=sk.get("noise_lo_dbfs", noise_lo_dbfs),
            noise_hi_dbfs=sk.get("noise_hi_dbfs", noise_hi_dbfs),
            max_iter=max_iter,
            n_final_seeds=n_final_seeds,
            seed=carr_seed,
            progress_callback=_carrier_cb,
        )

    return results
