# Sweep-Level Parallelism — Design Proposal

**Status:** design proposal — not yet implemented.

## Goal

Cut wall-clock time on multi-point sweeps by running independent grid points
in parallel processes. The unit of parallelism is one `(IBO, noise)` point;
each is a self-contained call into `wideband_bpsk_simulation` and shares no
mutable state with any other point.

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

Only `sim/sweep.py`. `wideband_bpsk_simulation` is already a pure function
of its arguments; nothing in `simulation.py`, `filters.py`, or `receiver.py`
needs to change. `main.py` gains one config-read for `n_workers`.

The first grid point keeps running in-process so the full sim dict is
available for `plot_wideband_results` and the console metrics table; only
points 2..N are dispatched to workers.

### 1. Worker function (top-level, picklable)

```python
# sim/sweep.py
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

def _sweep_one(args: tuple) -> tuple[float, float, dict]:
    """Run one (ibo, noise) grid point in a worker process."""
    (ibo, noise, carriers, sample_rate, am_am_cfg, am_pm_cfg,
     ola_filter_span, ola_block_size, seed, demod_carriers) = args
    # Single-thread BLAS inside each worker to avoid N×M oversubscription
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    sim = wideband_bpsk_simulation(
        carriers=carriers, sample_rate=sample_rate,
        am_am_cfg=am_am_cfg, am_pm_cfg=am_pm_cfg,
        input_backoff_db=ibo, noise_density_dbfs=noise,
        ola_filter_span=ola_filter_span, ola_block_size=ola_block_size,
        seed=seed, demod_carriers=demod_carriers,
        chunk_print=None,         # callbacks aren't safe across processes
    )
    return ibo, noise, sim
```

### 2. Restructured `parameter_sweep`

```python
def parameter_sweep(..., n_workers: int = 1) -> tuple[dict, list[dict]]:
    points = [(ibo, noise)
              for ibo in ibo_db_values
              for noise in noise_density_dbfs_values]
    demod = {c["name"] for c in carriers if c.get("sweep_demod", False)}

    def args_for(ibo, noise):
        return (ibo, noise, carriers, sample_rate, am_am_cfg, am_pm_cfg,
                ola_filter_span, ola_block_size, seed, demod)

    # First point runs in-process so we keep the full sim dict for the PSD plot.
    ibo0, noise0 = points[0]
    _, _, first_sim = _sweep_one(args_for(ibo0, noise0))
    results: list[dict | None] = [None] * len(points)
    results[0] = _compact(ibo0, noise0, first_sim)

    rest = list(enumerate(points[1:], start=1))
    if n_workers <= 1 or not rest:
        for i, (ibo, noise) in rest:
            _, _, sim = _sweep_one(args_for(ibo, noise))
            results[i] = _compact(ibo, noise, sim)
            if point_cb: point_cb(i + 1, len(points))
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(_sweep_one, args_for(ibo, noise)): i
                    for i, (ibo, noise) in rest}
            done = 1
            for fut in as_completed(futs):
                i = futs[fut]
                ibo, noise, sim = fut.result()
                results[i] = _compact(ibo, noise, sim)
                done += 1
                if point_cb: point_cb(done, len(points))

    return first_sim, [r for r in results if r is not None]


def _compact(ibo, noise, sim) -> dict:
    return dict(ibo_db=ibo, noise_density_dbfs=noise,
                carriers=[dict(name=cr["name"],
                               cnr_db=cr["cnr_db"], cir_db=cr["cir_db"],
                               cnir_db=cr["cnir_db"], evm_rms=cr["evm_rms"],
                               ber=cr["ber"])
                          for cr in sim["carriers"]])
```

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
| Chunk-level progress (`chunk 12/24`) per point | Point-level progress only when `n_workers > 1` (worker stdout is suppressed — interleaving would be unreadable) |
| Ctrl-C cleanly stops the run | Ctrl-C still works, but in-flight worker processes have to be killed by the pool — slightly messier shutdown |
| Single Python process; one memory footprint | N workers × per-point arrays. Memory scales with `n_workers`, not grid size — each worker holds only its own point |
| `chunk_print` callable from GUI works as-is | Workers can't call the GUI callback; falls back to point-level updates only |

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

- **Determinism preserved.** Each point is seeded with the same global seed
  (which is what serial does), so per-point results are bit-identical
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
