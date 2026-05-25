"""Generate AWGN-only BER reference curves for APSK modulations.

The simulator pipeline is driven through `parameter_sweep` with the nonlinear
amplifier bypassed (identity AM-AM table, zero AM-PM) at a grid of Eb/N0
points.  The adaptive Wilson-CI loop accumulates iterations until the BER
estimate sits inside the configured relative half-width.

The resulting (Eb/N0, BER, n_bits, n_errors, CI) tables are written to
``data/theory/ber_awgn_<MOD>.npz`` and committed to the repo.  ``sim.theory``
loads these tables to enable implementation-loss reporting on APSK carriers,
which have no closed-form BER formula.

Run from the repo root:

    .venv\\Scripts\\python.exe tools/generate_theory_curves.py                 # both
    .venv\\Scripts\\python.exe tools/generate_theory_curves.py --modulation 16APSK
    .venv\\Scripts\\python.exe tools/generate_theory_curves.py --smoke         # 2 pts each
"""
import argparse
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Force UTF-8 on stdout/stderr so sweep.py's CI±= prints don't crash on
# Windows consoles that default to cp1252.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

import numpy as np  # noqa: E402

from sim.modulation import bits_per_symbol  # noqa: E402
from sim.sweep import parameter_sweep  # noqa: E402


# Identity NLA — AM-AM passes amplitude through unchanged, AM-PM adds no phase.
_AM_AM_IDENTITY = {"input": [0.0, 1.0], "output": [0.0, 1.0]}
_AM_PM_ZERO     = {"input": [0.0, 1.0], "phase_deg": [0.0, 0.0]}

# Carrier geometry.  L = sample_rate / native_rate = 1 → no upsampling work.
_SYMBOL_RATE = 1_000_000.0
_SPS         = 4
_SAMPLE_RATE = _SYMBOL_RATE * _SPS

# Convergence target — user spec: <5% relative half-width at 95% CI down to ~1e-6.
_TARGET_CI_REL = 0.05
_CONFIDENCE    = 0.95
_MIN_ERRORS    = 50
_MAX_ITER      = 120
_MAX_BLOCK     = 16_777_216           # ~256 MB peak per iter

_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "theory"


# Eb/N0 target grid (0.5 dB spacing) per modulation.  The simulator reports
# Eb/N0 via cnir_db - 10·log10(bps), which sits a few dB below the target Eb/N0
# we set via noise PSD — the offset depends on modulation (~3 dB for 16APSK,
# ~5 dB for 32APSK).  That offset cancels in IL because both the table and
# user runs use the same cnir-derived convention, but the target grid is
# extended upward here so the *measured* Eb/N0 reaches deep enough to cover
# BER ~ 1e-6.  Default gammas are the DVB-S2 references.
_GRIDS: dict[str, tuple[np.ndarray, dict]] = {
    "16APSK": (np.round(np.arange(3.0, 17.001, 0.5), 2),
               {"apsk_gamma": 2.57}),
    "32APSK": (np.round(np.arange(5.0, 21.001, 0.5), 2),
               {"apsk_gamma1": 2.84, "apsk_gamma2": 5.27}),
}


def _noise_density_dbfs(ebn0_db: float, bps: int) -> float:
    """Eb/N0 (dB) → noise PSD (dBFS/Hz) for a unit-power carrier at _SYMBOL_RATE.

    Derivation: CNIR is reported in the symbol-rate (matched-filter) bandwidth,
    so for a unit-power carrier, Es/N0 = 1 / (N0 · symbol_rate) where N0 is the
    one-sided PSD in linear units.  Then noise_density_dbfs = 10·log10(N0)
    = -Es/N0_dB - 10·log10(symbol_rate) = -(Eb/N0_dB + 10·log10(bps))
                                            - 10·log10(symbol_rate).
    """
    esn0_db = ebn0_db + 10.0 * math.log10(bps)
    return -esn0_db - 10.0 * math.log10(_SYMBOL_RATE)


def _measure_point(modulation: str, ebn0_db: float, mod_kwargs: dict,
                   seed: int) -> dict:
    """One adaptive parameter_sweep run at one Eb/N0 point."""
    bps = bits_per_symbol(modulation)
    nd_dbfs = _noise_density_dbfs(ebn0_db, bps)
    carrier = dict(
        name="ref", modulation=modulation,
        symbol_rate=_SYMBOL_RATE, sps=_SPS, rolloff=0.35, filter_span=8,
        power_db=0.0, freq=0.0, sweep_demod=True,
        **mod_kwargs,
    )
    _, results = parameter_sweep(
        carriers=[carrier], sample_rate=_SAMPLE_RATE,
        am_am_cfg=_AM_AM_IDENTITY, am_pm_cfg=_AM_PM_ZERO,
        ibo_db_values=[0.0], noise_density_dbfs_values=[nd_dbfs],
        max_block_size_samples=_MAX_BLOCK,
        target_ci_half_width=1e-12,           # absolute unreachable → relative governs
        target_ci_relative=_TARGET_CI_REL,
        confidence=_CONFIDENCE,
        min_errors=_MIN_ERRORS,
        max_iterations=_MAX_ITER,
        ola_filter_span=16, ola_block_size=8192,
        seed=seed,
    )
    cr = results[0]["carriers"][0]
    # Use the simulator's REPORTED Eb/N0 (= cnir_db - 10·log10(bps)) as the
    # table x-axis, not the value we asked for.  The chain has a small
    # constant offset between target Eb/N0 (set via noise PSD) and the value
    # reported through cnir_db; that offset cancels in IL calculations because
    # main.py computes eff_ebn0 the same way.  The user-supplied target is
    # preserved separately so the table stays self-describing.
    cnir_db          = float(cr["cnir_db"])
    measured_ebn0_db = cnir_db - 10.0 * math.log10(bps)
    return dict(
        ebn0_db        = measured_ebn0_db,
        ebn0_db_target = float(ebn0_db),
        cnir_db        = cnir_db,
        ber            = float(cr["ber"]) if cr["ber"] is not None else 0.0,
        n_bits         = int(cr["n_bits"]),
        n_errors       = int(cr["n_errors"]),
        ci_half_width  = float(cr["ci_half_width"]),
        iterations     = int(results[0]["iterations"]),
        converged      = bool(results[0]["converged"]),
    )


def generate(modulation: str, seed: int = 0, smoke: bool = False) -> Path:
    """Generate the full Eb/N0 sweep for one modulation and save the npz."""
    if modulation not in _GRIDS:
        raise ValueError(f"Unsupported modulation: {modulation}")
    grid, mod_kwargs = _GRIDS[modulation]
    if smoke:
        # Two points: one at the low-Eb/N0 (cheap) end and one mid-range.
        grid = np.array([grid[0], grid[len(grid) // 2]])

    rows: list[dict] = []
    print(f"\n=== {modulation} ({len(grid)} Eb/N0 points) ===", flush=True)
    t0 = time.perf_counter()
    for i, ebn0 in enumerate(grid):
        pt0 = time.perf_counter()
        row = _measure_point(modulation, float(ebn0), mod_kwargs, seed=seed + i)
        rows.append(row)
        elapsed = time.perf_counter() - pt0
        status = "OK" if row["converged"] else "CAP"
        print(f"  Eb/N0 target={row['ebn0_db_target']:5.2f} -> "
              f"measured={row['ebn0_db']:5.2f} dB  "
              f"BER={row['ber']:.3e}  CI={row['ci_half_width']:.2e}  "
              f"iters={row['iterations']:3d} ({status})  "
              f"bits={row['n_bits']:>13,}  errs={row['n_errors']:>7,}  "
              f"({elapsed:6.1f} s)",
              flush=True)
    print(f"  total {(time.perf_counter() - t0) / 60.0:.1f} min", flush=True)

    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if smoke else ""
    out_path = _DATA_DIR / f"ber_awgn_{modulation}{suffix}.npz"
    # Sort by measured Eb/N0 ascending so the loader can interpolate without
    # re-sorting on every read.
    rows.sort(key=lambda r: r["ebn0_db"])
    save: dict = dict(
        ebn0_db        = np.asarray([r["ebn0_db"]        for r in rows], dtype=np.float64),
        ebn0_db_target = np.asarray([r["ebn0_db_target"] for r in rows], dtype=np.float64),
        cnir_db        = np.asarray([r["cnir_db"]        for r in rows], dtype=np.float64),
        ber            = np.asarray([r["ber"]            for r in rows], dtype=np.float64),
        n_bits         = np.asarray([r["n_bits"]         for r in rows], dtype=np.int64),
        n_errors       = np.asarray([r["n_errors"]       for r in rows], dtype=np.int64),
        ci_half_width  = np.asarray([r["ci_half_width"]  for r in rows], dtype=np.float64),
        modulation     = np.array(modulation),
        confidence     = np.array(_CONFIDENCE),
        target_ci_rel  = np.array(_TARGET_CI_REL),
        generated_at   = np.array(datetime.now(timezone.utc).isoformat(timespec="seconds")),
    )
    for k, v in mod_kwargs.items():
        save[k] = np.array(v)
    np.savez(out_path, **save)
    print(f"  -> wrote {out_path}", flush=True)
    return out_path


def main() -> None:
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
