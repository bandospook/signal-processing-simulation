"""Adaptive 2D parameter sweep with Wilson-CI BER convergence per point."""
import math
from collections.abc import Callable

from .simulation import wideband_bpsk_simulation
from .stats import rule_of_three_upper, wilson_half_width

_PrintCB = Callable[[str], None] | None


_PointCB = Callable[[int, int], None] | None
_IterCB  = Callable[[int, int, int], None] | None   # (iter_idx, ibo_idx, noise_idx)


# Large prime spacing so per-iteration seeds (base + i * _SEED_STRIDE) don't
# collide with the per-(point) seeds and stay well-separated.
_SEED_STRIDE = 2_147_483_587


class _ErrorAccumulator:
    """Cumulative (n_bits, n_errors) tracker with Wilson-CI convergence test."""

    def __init__(self) -> None:
        self.n_bits           = 0
        self.n_errors         = 0
        self.uncoded_n_bits   = 0
        self.uncoded_n_errors = 0

    def add(self, n_bits: int, n_errors: int,
            uncoded_n_bits: int = 0, uncoded_n_errors: int = 0) -> None:
        self.n_bits           += int(n_bits)
        self.n_errors         += int(n_errors)
        self.uncoded_n_bits   += int(uncoded_n_bits)
        self.uncoded_n_errors += int(uncoded_n_errors)

    @property
    def ber(self) -> float | None:
        if self.n_bits <= 0:
            return None
        return self.n_errors / self.n_bits

    @property
    def uncoded_ber(self) -> float | None:
        if self.uncoded_n_bits <= 0:
            return None
        return self.uncoded_n_errors / self.uncoded_n_bits

    def half_width(self, confidence: float) -> float:
        return wilson_half_width(self.n_errors, self.n_bits, confidence)

    def converged(self, target: float, confidence: float, min_errors: int) -> bool:
        if self.n_bits <= 0:
            return False
        if self.n_errors < min_errors:
            return False
        return self.half_width(confidence) <= target


def parameter_sweep(carriers: list[dict],
                    sample_rate: float,
                    am_am_cfg: dict,
                    am_pm_cfg: dict,
                    ibo_db_values: list[float],
                    noise_density_dbfs_values: list[float],
                    max_block_size_samples: int,
                    target_ci_half_width: float,
                    confidence: float = 0.95,
                    min_errors: int = 50,
                    max_iterations: int = 100,
                    ola_filter_span: int = 16,
                    ola_block_size: int = 4096,
                    seed: int | None = None,
                    chunk_print: _PrintCB = None,
                    point_cb: _PointCB = None,
                    iter_cb: _IterCB = None) -> tuple[dict, list[dict]]:
    """
    Run the simulation on a 2-D grid of IBO × noise density values, with
    adaptive iteration at each grid point until every demodulated carrier
    meets the Wilson-CI half-width target (or the iteration cap is hit).

    Returns (first_sim, results) where `first_sim` is the first sim run's
    full wideband_bpsk_simulation return dict (used to draw the wideband PSD),
    and `results` is a list of compact per-point dicts:
        ibo_db              float
        noise_density_dbfs  float
        iterations          int     iterations actually run at this point
        converged           bool    all demod carriers met the CI target
        carriers            list of per-carrier aggregated dicts
    """
    n_total = len(ibo_db_values) * len(noise_density_dbfs_values)
    n_done  = 0
    results: list[dict] = []
    first_sim: dict | None = None

    # Carriers with sweep_demod=False contribute to the wideband composite but
    # their per-carrier demod (BER/EVM/CNR/CIR/CNIR) is skipped each grid point.
    demod_carriers = {c["name"] for c in carriers if c.get("sweep_demod", False)}
    base_seed = 0 if seed is None else int(seed)

    for ibo_i, ibo in enumerate(ibo_db_values):
        for noise_i, noise in enumerate(noise_density_dbfs_values):
            accs = {n: _ErrorAccumulator() for n in demod_carriers}
            # Non-BER metrics are averaged across iterations (they're nearly
            # deterministic across seeds; EVM has slight jitter).
            sums: dict[str, dict[str, float]] = {
                n: {"cnr_db": 0.0, "cir_db": 0.0, "cnir_db": 0.0, "evm_rms": 0.0,
                    "cnr_n": 0, "cir_n": 0, "cnir_n": 0, "evm_n": 0}
                for n in demod_carriers
            }

            it = 0
            converged_all = True
            last_sim: dict | None = None
            for it in range(max_iterations):
                iter_num = it + 1
                point_seed = base_seed + (ibo_i * len(noise_density_dbfs_values) + noise_i) \
                             * _SEED_STRIDE + it

                # Wrap chunk_print to prefix every chunk line with the iteration
                # count so the user sees both baseband-consumption progress AND
                # the iteration count incrementing when a point needs more stats.
                inner_print: _PrintCB = None
                if chunk_print is not None:
                    def _iter_chunk_print(msg: str, _n: int = iter_num) -> None:
                        chunk_print(f"iter {_n}/{max_iterations}: {msg}")
                    inner_print = _iter_chunk_print

                sim = wideband_bpsk_simulation(
                    carriers=carriers,
                    sample_rate=sample_rate,
                    am_am_cfg=am_am_cfg,
                    am_pm_cfg=am_pm_cfg,
                    max_block_size_samples=max_block_size_samples,
                    input_backoff_db=ibo,
                    noise_density_dbfs=noise,
                    ola_filter_span=ola_filter_span,
                    ola_block_size=ola_block_size,
                    seed=point_seed,
                    demod_carriers=demod_carriers,
                    chunk_print=inner_print,
                )
                if first_sim is None:
                    first_sim = sim
                last_sim = sim

                for cr in sim["carriers"]:
                    name = cr["name"]
                    if name not in accs:
                        continue
                    accs[name].add(
                        n_bits=cr.get("n_bits", 0),
                        n_errors=cr.get("n_errors", 0),
                        uncoded_n_bits=cr.get("uncoded_n_bits", 0),
                        uncoded_n_errors=cr.get("uncoded_n_errors", 0),
                    )
                    s = sums[name]
                    for key in ("cnr_db", "cir_db", "cnir_db", "evm_rms"):
                        val = cr.get(key, float("nan"))
                        if val is not None and math.isfinite(val):
                            s[key] += float(val)
                            s[key.split("_")[0] + "_n"] += 1

                # Cumulative running tally per demod carrier — one line per iter.
                if chunk_print is not None and accs:
                    for name in sorted(accs):
                        a = accs[name]
                        hw = a.half_width(confidence)
                        ber_s = ("BER=0 (no errors)" if a.n_errors == 0
                                 else f"BER={a.ber:.2e}")
                        target_met = a.converged(
                            target_ci_half_width, confidence, min_errors)
                        suffix = "  (target met)" if target_met else ""
                        chunk_print(
                            f"iter {iter_num}/{max_iterations} done: {name} "
                            f"bits={a.n_bits} errors={a.n_errors} {ber_s} "
                            f"CI±={hw:.1e}{suffix}")

                if iter_cb is not None:
                    iter_cb(it + 1, ibo_i, noise_i)

                if accs and all(
                        a.converged(target_ci_half_width, confidence, min_errors)
                        for a in accs.values()):
                    break
            else:
                converged_all = False

            iterations_run = it + 1

            # Build aggregated per-carrier dicts (preserving order from last sim)
            assert last_sim is not None
            agg_carriers = []
            for cr in last_sim["carriers"]:
                name = cr["name"]
                if name in accs:
                    acc = accs[name]
                    s = sums[name]
                    def _avg(s: dict, key: str, count_key: str) -> float:
                        c = s[count_key]
                        return s[key] / c if c > 0 else float("nan")
                    agg_carriers.append({
                        "name":             name,
                        "cnr_db":           _avg(s, "cnr_db",  "cnr_n"),
                        "cir_db":           _avg(s, "cir_db",  "cir_n"),
                        "cnir_db":          _avg(s, "cnir_db", "cnir_n"),
                        "evm_rms":          _avg(s, "evm_rms", "evm_n"),
                        "ber":              acc.ber,
                        "n_bits":           acc.n_bits,
                        "n_errors":         acc.n_errors,
                        "uncoded_ber":      acc.uncoded_ber,
                        "uncoded_n_bits":   acc.uncoded_n_bits,
                        "uncoded_n_errors": acc.uncoded_n_errors,
                        "ci_half_width":    acc.half_width(confidence),
                        "ber_upper_95":     (rule_of_three_upper(acc.n_bits, confidence)
                                              if acc.n_errors == 0 else None),
                    })
                else:
                    agg_carriers.append({
                        "name":     name,
                        "cnr_db":   cr["cnr_db"],
                        "cir_db":   cr["cir_db"],
                        "cnir_db":  cr["cnir_db"],
                        "evm_rms":  cr["evm_rms"],
                        "ber":      cr["ber"],
                        "n_bits":   0,
                        "n_errors": 0,
                    })

            results.append({
                "ibo_db":             ibo,
                "noise_density_dbfs": noise,
                "iterations":         iterations_run,
                "converged":          converged_all,
                "carriers":           agg_carriers,
            })
            n_done += 1
            if point_cb is not None:
                point_cb(n_done, n_total)
            tag = "converged" if converged_all else f"CAPPED at {max_iterations}"
            print(f"  [{n_done:>{len(str(n_total))}}/{n_total}] "
                  f"IBO={ibo:.1f} dB  noise={noise:.1f} dBFS/Hz  "
                  f"iters={iterations_run} ({tag})")

    assert first_sim is not None
    return first_sim, results


__all__ = ["parameter_sweep", "_ErrorAccumulator"]
