# Sweep-Level Parallelism — Design Proposal

**Status:** design proposal — not yet implemented. The pseudocode below was
drafted before the adaptive Wilson-CI iteration refactor; signatures and the
worker body have been refreshed accordingly. The parallelism unit and the
load-balancing story have changed as a result — see *Load balancing under
adaptive iteration* below.

## Goal

Cut wall-clock time on multi-point sweeps by running independent grid points
in parallel processes. The unit of parallelism is one `(IBO, noise)` point —
the full adaptive-iteration loop for that point, not a single
`simulate` call. Grid points are independent and share no
mutable state.

## Why processes, not Numba

The uncoded chain is already NumPy / SciPy bound at every hot spot:

- OLA up/downsamplers use `np.fft.fft` (already calls the threaded native FFT).
- NLA is a vectorised `np.interp` over the chunk.
- The receiver/BER work is vectorised array math.

There is no interpreted message-passing loop to JIT-compile, so a Numba pass
over `simulation.py` would likely buy < 5 %. The real lever is **process-level
parallelism across the sweep grid**: a 16-point sweep on an 8-core desktop
can land at ~6–7× speedup with no inner-loop changes.

This is the inverse of the FEC story (see `docs/coding_design.md`), where the
inner trellis / belief-propagation recursions *are* interpreted loops and
`@njit` is a 100×+ win.

## What changes

Only `sim/sweep.py`. `simulate` is already a pure function
of its arguments; nothing in `simulation.py`, `filters.py`, or `receiver.py`
needs to change. `main.py` gains one config-read for `n_workers`.

The first grid point keeps running in-process so the full sim dict is
available for `plot_wideband_results` and the console metrics table; only
points 2..N are dispatched to workers.

### 1. Worker function (top-level, picklable)

The worker now runs the full adaptive-iteration loop for one grid point, not
a single `simulate` call. The natural shape is to extract the
per-point body of `parameter_sweep` into a helper (`_sweep_point`) that takes
the grid-point parameters plus all the adaptive-iteration knobs, returns the
aggregated per-carrier dict for that point, and lets the parent process
dispatch one such call per grid point.

```python
# sim/sweep.py
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

def _sweep_point_worker(args: tuple) -> dict:
    """Run one (ibo, noise) point's full adaptive-iteration loop in a worker."""
    (ibo_i, noise_i, ibo, noise, carriers, sample_rate, am_am_cfg, am_pm_cfg,
     max_block_size_samples, target_ci_half_width, confidence,
     min_errors, max_iterations,
     ola_filter_span, ola_block_size, base_seed, demod_carriers,
     n_noise_axis) = args
    # Single-thread BLAS inside each worker to avoid N×M oversubscription
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    # Calls the same per-point accumulation logic that parameter_sweep uses
    # serially, returning the aggregated dict (ibo_db, noise_density_dbfs,
    # iterations, converged, carriers[…]).
    return _accumulate_point(
        ibo_i=ibo_i, noise_i=noise_i, ibo=ibo, noise=noise,
        carriers=carriers, sample_rate=sample_rate,
        am_am_cfg=am_am_cfg, am_pm_cfg=am_pm_cfg,
        max_block_size_samples=max_block_size_samples,
        target_ci_half_width=target_ci_half_width, confidence=confidence,
        min_errors=min_errors, max_iterations=max_iterations,
        ola_filter_span=ola_filter_span, ola_block_size=ola_block_size,
        base_seed=base_seed, demod_carriers=demod_carriers,
        n_noise_axis=n_noise_axis,
        chunk_print=None,          # callbacks aren't safe across processes
        iter_cb=None,
    )
```

### 2. Restructured `parameter_sweep`

`parameter_sweep` becomes a dispatch shell around `_accumulate_point`. The
first point still runs in-process so the parent has its first iteration's
full sim dict for the PSD plot; points 2..N go to the pool.

```python
def parameter_sweep(..., n_workers: int = 1) -> tuple[dict, list[dict]]:
    grid = [(ibo_i, noise_i, ibo, noise)
            for ibo_i, ibo in enumerate(ibo_db_values)
            for noise_i, noise in enumerate(noise_density_dbfs_values)]
    demod = {c["name"] for c in carriers if c.get("sweep_demod", False)}
    n_noise_axis = len(noise_density_dbfs_values)

    def args_for(ibo_i, noise_i, ibo, noise):
        return (ibo_i, noise_i, ibo, noise, carriers, sample_rate,
                am_am_cfg, am_pm_cfg, max_block_size_samples,
                target_ci_half_width, confidence, min_errors, max_iterations,
                ola_filter_span, ola_block_size, base_seed, demod, n_noise_axis)

    # First point runs in-process; the worker also returns the first iteration's
    # full sim dict alongside the aggregated point dict for use by plot_wideband.
    first_sim, results0 = _accumulate_point_with_first_sim(*args_for(*grid[0]))
    results: list[dict | None] = [None] * len(grid)
    results[0] = results0

    rest = list(enumerate(grid[1:], start=1))
    if n_workers <= 1 or not rest:
        for i, args in rest:
            results[i] = _sweep_point_worker(args_for(*args))
            if point_cb: point_cb(i + 1, len(grid))
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(_sweep_point_worker, args_for(*args)): i
                    for i, args in rest}
            done = 1
            for fut in as_completed(futs):
                results[futs[fut]] = fut.result()
                done += 1
                if point_cb: point_cb(done, len(grid))

    return first_sim, [r for r in results if r is not None]
```

The shape of the returned per-point dict (`iterations`, `converged`,
aggregated `carriers[…]`) is identical to the serial path, so downstream
report and plot code is unchanged.

### 3. Config knob

```toml
[sweep]
sample_rate        = 16
ibo_db             = [0, 3, 6]
noise_density_dbfs = [-160, -150, -140]
n_workers          = 8     # NEW: 1 (default) = serial; >1 = ProcessPoolExecutor
```

`main.py`:

```python
n_workers = sweep_cfg.get("n_workers", 1)
# ... pass to parameter_sweep(..., n_workers=n_workers)
```

## Tradeoffs

| Today (serial) | After (parallel, opt-in) |
|---|---|
| Chunk-level progress (`chunk 12/24`) per iteration, per point | Point-level progress only when `n_workers > 1` (worker stdout is suppressed — interleaving would be unreadable) |
| Ctrl-C cleanly stops the run | Ctrl-C still works, but in-flight worker processes have to be killed by the pool — slightly messier shutdown |
| Single Python process; one memory footprint | N workers × per-iteration arrays. Memory scales with `n_workers × max_block_size_samples`, not grid size — each worker holds only its own point's iteration buffers |
| `chunk_print` and `iter_cb` callbacks work as-is | Workers can't call the GUI callbacks; falls back to point-level updates only |

## Load balancing under adaptive iteration

Before adaptive iteration, every grid point did the same amount of work and a
naive `as_completed` over the grid gave near-perfect load balance. After the
refactor, points are no longer equal: a low-BER point may need 10× the
iterations of a high-BER point to satisfy `target_ci_half_width`. The
`as_completed` pattern still works (the pool returns whichever workers finish
first), but two practical consequences follow:

- **Wall-clock is set by the slowest point**, not the average. A 12-point
  sweep with 11 fast points and one stubborn low-BER point will not finish
  noticeably faster than running just the slow point — the other 11 finish
  early and their workers idle.
- **Per-worker memory is steady**, because each worker reuses the same
  iteration buffers across iterations within a point. There is no need to
  shard a single hard point across multiple workers.

If load balancing becomes a problem, the right lever is to cap each point's
work via `max_iterations` rather than to split a point across workers.

## Caveats

- **Windows process spawn is ~250 ms/worker** (uses `spawn`, not `fork`).
  For very small sweeps (< ~8 points) parallelism actively hurts. The
  `n_workers` knob lets the user opt in only when worth it.

- **BLAS oversubscription is a real trap.** Without `OMP_NUM_THREADS=1`
  per worker, NumPy's FFTs will each spawn N threads and you get N² total
  threads fighting for cores. The env-var hack above must run *inside* the
  worker before NumPy is imported. For absolute safety, set it before
  `ProcessPoolExecutor` spawns too — either via the `initializer=` argument
  or by exporting it in `main.py` when `n_workers > 1`.

- **Determinism preserved.** Each iteration is seeded by
  `base_seed + grid_index × stride + iter_index` (same scheme as the serial
  path), so per-point bit streams and noise realisations are bit-identical
  between serial and parallel runs.

- **Tests.** Existing `test_main_runs` and friends pass unchanged at the
  default `n_workers=1`. A new test sets `n_workers=2` and asserts the
  parallel results match the serial baseline.

## Related future work

- **Batch of TOMLs.** A coarser-grained layer above the sweep — run a list
  of `.toml` configs in parallel, each its own output directory, no GUI.
  See [batch_runs.md](batch_runs.md). The two layers do not nest: when the
  batch layer is active, it forces sweep-level `n_workers = 1` inside each
  scenario.
