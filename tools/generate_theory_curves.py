r"""Generate AWGN-only BER reference curves for APSK modulations.

Uses the same direct-AWGN chain as ``tests/test_awgn_performance.py``: generate
RRC-shaped baseband at the carrier's native sample rate, add complex Gaussian
noise of variance ``sps / Es/N0_linear``, run the matched-filter receiver,
count errors.  No OLA up/downsample, no NLA, no projection-based metrics —
this is the *theoretical* AWGN reference, and the Eb/N0 axis it produces is
the analytical value derived from the noise sigma.

The npz tables are loaded by ``sim.theory`` so ``ebn0_for_ber("16APSK", ...)``
returns the matched Eb/N0 for any measured BER, and ``main.py`` reports
implementation loss on APSK carriers exactly the same way it does for the
closed-form modulations.

Iterations are accumulated adaptively at each Eb/N0 point until the Wilson
half-width on the running BER estimate is ≤ ``target_ci_relative`` of the BER
itself (default 5 %, 95 % CI).  Cheap per-iteration cost (no chunk pipeline)
means deep BER points (~1e-6) finish in tens of seconds rather than minutes.

Run from the repo root:

    .venv\Scripts\python.exe tools/generate_theory_curves.py                 # both
    .venv\Scripts\python.exe tools/generate_theory_curves.py --modulation 16APSK
    .venv\Scripts\python.exe tools/generate_theory_curves.py --smoke         # 2 pts each
"""
import argparse
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Force UTF-8 on stdout/stderr so any Unicode in dependencies doesn't crash on
# Windows consoles that default to cp1252.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

import numpy as np  # noqa: E402

from sim.baseband import rrc_baseband  # noqa: E402
from sim.modulation import bits_per_symbol  # noqa: E402
from sim.receiver import receive  # noqa: E402
from sim.stats import wilson_half_width  # noqa: E402


# Receiver geometry — matches sim/baseband.py defaults and the production sim.
_SPS         = 4
_ROLLOFF     = 0.35
_FILTER_SPAN = 8

# Convergence target (user spec): ≤ 5 % relative half-width at 95 % CI down
# to ~1e-6 BER.  min_errors floor prevents premature stops on tiny samples.
_TARGET_CI_REL  = 0.05
_CONFIDENCE     = 0.95
_MIN_ERRORS     = 50
_MAX_ITER       = 200
_N_SYM_PER_ITER = 4_000_000        # 16M complex samples per iter at sps=4

_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "theory"


# Analytical Eb/N0 grid (0.5 dB spacing) per modulation.  Ranges chosen so
# the deepest point reaches ~1e-6 BER for the default DVB-S2 gammas.
_GRIDS: dict[str, tuple[np.ndarray, dict]] = {
    "16APSK": (np.round(np.arange(3.0, 16.001, 0.5), 2),
               {"apsk_gamma": 2.57}),
    "32APSK": (np.round(np.arange(4.0, 19.001, 0.5), 2),
               {"apsk_gamma1": 2.84, "apsk_gamma2": 5.27}),
}


def _measure_iter(modulation: str, ebn0_db: float, mod_kwargs: dict,
                  rng: np.random.Generator) -> tuple[int, int]:
    """One direct-AWGN iteration; returns (n_bits, n_errors)."""
    bps = bits_per_symbol(modulation)
    esn0_lin = 10.0 ** ((ebn0_db + 10.0 * math.log10(bps)) / 10.0)
    bb, _, bits, _ = rrc_baseband(
        modulation, _N_SYM_PER_ITER,
        symbol_rate=1.0, sample_rate=float(_SPS),
        rolloff=_ROLLOFF, filter_span=_FILTER_SPAN,
        seed=int(rng.integers(0, 2 ** 31)),
        **mod_kwargs,
    )
    sigma_c = math.sqrt(_SPS / (2.0 * esn0_lin))
    n_samples = len(bb)
    noise = sigma_c * (rng.standard_normal(n_samples)
                       + 1j * rng.standard_normal(n_samples))
    result = receive(bb + noise, modulation, _ROLLOFF, _FILTER_SPAN, _SPS,
                     reference_bits=bits, **mod_kwargs)
    return int(result["n_bits"]), int(result["n_errors"])


def _measure_point(modulation: str, ebn0_db: float, mod_kwargs: dict,
                   seed: int) -> dict:
    """Accumulate iterations until the running BER's Wilson half-width is.

    within ``_TARGET_CI_REL`` of the BER, or until ``_MAX_ITER`` is hit.
    """
    rng = np.random.default_rng(seed)
    n_bits = 0
    n_errors = 0
    converged = False
    iters = 0
    for iters in range(1, _MAX_ITER + 1):
        nb, ne = _measure_iter(modulation, ebn0_db, mod_kwargs, rng)
        n_bits += nb
        n_errors += ne
        if n_errors < _MIN_ERRORS:
            continue
        hw = wilson_half_width(n_errors, n_bits, _CONFIDENCE)
        ber = n_errors / n_bits
        if ber > 0 and hw / ber <= _TARGET_CI_REL:
            converged = True
            break
    return dict(
        ebn0_db       = float(ebn0_db),
        ber           = float(n_errors / n_bits) if n_bits else 0.0,
        n_bits        = n_bits,
        n_errors      = n_errors,
        ci_half_width = (wilson_half_width(n_errors, n_bits, _CONFIDENCE)
                         if n_bits else float("inf")),
        iterations    = iters,
        converged     = converged,
    )


def generate(modulation: str, seed: int = 0, smoke: bool = False) -> Path:
    """Run the full Eb/N0 sweep for one modulation and save the npz."""
    if modulation not in _GRIDS:
        raise ValueError(f"Unsupported modulation: {modulation}")
    grid, mod_kwargs = _GRIDS[modulation]
    if smoke:
        # Two points: one cheap (high BER) and one deeper (mid-range).
        grid = np.array([grid[0], grid[len(grid) // 2]])

    rows: list[dict] = []
    print(f"\n=== {modulation} ({len(grid)} Eb/N0 points) ===", flush=True)
    t0 = time.perf_counter()
    for i, ebn0 in enumerate(grid):
        pt0 = time.perf_counter()
        row = _measure_point(modulation, float(ebn0), mod_kwargs,
                             seed=seed * 10_000 + i)
        rows.append(row)
        elapsed = time.perf_counter() - pt0
        status = "OK" if row["converged"] else "CAP"
        print(f"  Eb/N0={row['ebn0_db']:5.2f} dB  BER={row['ber']:.3e}  "
              f"CI={row['ci_half_width']:.2e}  iters={row['iterations']:3d} "
              f"({status})  bits={row['n_bits']:>13,}  "
              f"errs={row['n_errors']:>7,}  ({elapsed:6.1f} s)",
              flush=True)
    print(f"  total {(time.perf_counter() - t0) / 60.0:.1f} min", flush=True)

    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if smoke else ""
    out_path = _DATA_DIR / f"ber_awgn_{modulation}{suffix}.npz"
    rows.sort(key=lambda r: r["ebn0_db"])
    save: dict = dict(
        ebn0_db       = np.asarray([r["ebn0_db"]       for r in rows], dtype=np.float64),
        ber           = np.asarray([r["ber"]           for r in rows], dtype=np.float64),
        n_bits        = np.asarray([r["n_bits"]        for r in rows], dtype=np.int64),
        n_errors      = np.asarray([r["n_errors"]      for r in rows], dtype=np.int64),
        ci_half_width = np.asarray([r["ci_half_width"] for r in rows], dtype=np.float64),
        modulation    = np.array(modulation),
        confidence    = np.array(_CONFIDENCE),
        target_ci_rel = np.array(_TARGET_CI_REL),
        generated_at  = np.array(datetime.now(timezone.utc).isoformat(timespec="seconds")),
    )
    for k, v in mod_kwargs.items():
        save[k] = np.array(v)
    np.savez(out_path, **save)
    print(f"  -> wrote {out_path}", flush=True)
    return out_path


def main() -> None:
    """Main."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--modulation", choices=("all", "16APSK", "32APSK"),
                   default="all")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true",
                   help="Two-point dry run that writes *_smoke.npz; verifies "
                        "wiring without paying the full cost.")
    args = p.parse_args()
    targets = ["16APSK", "32APSK"] if args.modulation == "all" else [args.modulation]
    for mod in targets:
        generate(mod, seed=args.seed, smoke=args.smoke)


if __name__ == "__main__":
    main()
