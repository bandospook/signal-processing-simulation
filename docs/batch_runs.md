# Batch Runs — Design Proposal

**Status:** design proposal — not yet implemented. Decisions captured here are
the project owner's; open questions are listed at the end.

This document predates the adaptive-iteration sweep refactor but the batch
proposal is unaffected: each scenario is still one full `main.main()` call,
and the sweep layer inside it now iterates to a Wilson-CI target (see
[GUIDE.md §8](GUIDE.md)). Runtime per scenario is variable instead of
predictable, which has consequences noted under *Concurrency shape*.

## Goal

Run a group of independent `.toml` configurations together — overnight studies,
parameter sweeps across modulation schemes, regression comparisons against a
baseline, etc. No GUI; this is a headless `batch.py` CLI sibling to `main.py`.

Each config in the batch is a fully-formed SO-WAT simulation as defined in
`docs/simulation_overview.md`. The batch layer adds nothing per-run; its job
is to dispatch, isolate outputs, and aggregate pass/fail.

## What this is, and isn't

**Is:**
- A coarse-grained outer loop above the sweep — one TOML = one worker process.
- Decoupled from the GUI; suitable for cron, CI, remote machines.

**Isn't:**
- A replacement for `[sweep]`. The sweep stays inside one TOML; the batch is
  *across* TOMLs.
- A scenario-generation framework. Parameterised studies are produced either
  by manifest-mode overrides (see below) or by a separate small script that
  emits per-scenario TOMLs which CLI-mode then runs.

## Two input modes

Both are supported. The mode is selected by how `batch.py` is invoked.

### Mode A — CLI glob / list

```
python batch.py configs/*.toml --out batch_out/ --workers 4
python batch.py a.toml b.toml c.toml --out batch_out/
```

- Each positional argument is a path to a `.toml`.
- Output subdirectory per TOML, named from the TOML stem:
  `batch_out/qpsk_high_ibo/`, `batch_out/8psk_low_ibo/`, …
- Stem collisions (two TOMLs with the same filename in different dirs) are
  resolved by appending `__1`, `__2`, … to the later ones.

### Mode B — Manifest TOML

```
python batch.py --manifest study.toml
```

Manifest format (minimal version):

```toml
[batch]
output_dir = "batch_out"
workers    = 4          # parallel scenarios; default 1
fail_fast  = false      # default false (continue-and-report)

[[scenario]]
config = "configs/base.toml"
name   = "qpsk_high_ibo"     # optional; metadata only — outdir is numbered

[[scenario]]
config = "configs/base.toml"
name   = "8psk_low_ibo"
```

Output layout for manifest mode: **numbered subdirectories** in scenario order,
with the optional `name` appended for readability:

```
batch_out/
  000_qpsk_high_ibo/
  001_8psk_low_ibo/
  002/                      # name omitted → just the index
  batch_summary.md
```

Numbering rationale: scenarios that share a base TOML can't be uniquely
identified by stem. Numbers also preserve manifest order in the directory
listing, which matters when manifests are generated.

#### Manifest extension — overrides (deferred)

A second iteration of the manifest could deep-merge per-scenario overrides
into the base config, removing the need to pre-generate TOMLs for
parameterised studies:

```toml
[[scenario]]
config = "configs/base.toml"
name   = "qpsk_ibo_6"
[scenario.override.sweep]
ibo_db = [6]
[[scenario.override.carrier]]   # matched by name
name       = "main"
modulation = "QPSK"
```

This raises questions (deep vs. shallow merge, how to address list items,
how overrides interact with the carrier array) — deferred to a follow-up;
the minimal manifest above is enough for the common case.

## Failure mode

Configurable. Default is continue-and-report.

```
python batch.py configs/*.toml --fail-fast            # cancel remaining on first error
python batch.py configs/*.toml                        # continue, summarise at end
```

In manifest mode, `fail_fast` is also settable under `[batch]`; the CLI flag
takes precedence.

A `batch_summary.md` is written to `output_dir` after every run (pass and
fail alike). Schema:

```markdown
# Batch Summary

| # | Name | Config | Status | Duration | Output Dir |
|---|------|--------|--------|---------:|------------|
| 000 | qpsk_high_ibo | configs/base.toml | OK    | 42s      | 000_qpsk_high_ibo |
| 001 | 8psk_low_ibo  | configs/base.toml | FAIL  | 8s       | 001_8psk_low_ibo |

## Failures
### 001 — 8psk_low_ibo
<traceback or last 20 lines of simulation.log>
```

## Concurrency shape

One worker process per scenario via `ProcessPoolExecutor`. Each worker calls
`main.main(config_path, progress_callback=None)` after rewriting the loaded
config's `output.output_dir` to its assigned subdirectory.

Note on per-scenario runtime variability: because each scenario's sweep now
iterates adaptively until its CI target is met, scenario duration is no longer
predictable from the config alone. Two scenarios with identical sweep grids
but different operating points can take very different times (low-BER points
need more iterations to satisfy `target_ci_half_width`). The batch layer
should therefore tolerate large variance in completion times — `as_completed`
is the right primitive — and report each scenario's actual iteration counts
in `batch_summary.md` (e.g. read them out of the scenario's `report.md`).

Pseudocode:

```python
# batch.py
def run_scenario(slot: int, scenario: dict, batch_out: Path) -> dict:
    subdir = batch_out / scenario["subdir_name"]
    subdir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(scenario["config"])
    cfg["output"]["output_dir"] = str(subdir)
    if "override" in scenario:
        deep_merge(cfg, scenario["override"])     # manifest mode only
    # Write the resolved config alongside outputs for reproducibility
    (subdir / "resolved.toml").write_text(serialise(cfg))
    # Run.  Returns nothing useful; the value is the on-disk artefacts.
    try:
        run_with_resolved_cfg(cfg)                # see "Open questions" below
        return {"slot": slot, "status": "OK", ...}
    except Exception as e:
        return {"slot": slot, "status": "FAIL", "error": format_exc(e), ...}

with ProcessPoolExecutor(max_workers=workers) as ex:
    futs = [ex.submit(run_scenario, i, s, batch_out)
            for i, s in enumerate(scenarios)]
    for fut in as_completed(futs):
        result = fut.result()
        update_summary(result)
        if fail_fast and result["status"] == "FAIL":
            for f in futs: f.cancel()
            break
```

## Interaction with sweep-level parallelism

Two layers of `ProcessPoolExecutor` (batch × sweep) is a footgun. If a batch
worker is itself a sweep with `n_workers > 1`, total live workers is
`batch_workers × sweep_workers` — easy to oversubscribe a desktop and trash
BLAS thread scheduling.

**Rule:** when the batch layer is active, it forces `n_workers = 1` inside
each scenario before running. Sweep-level parallelism only applies when a
TOML is run directly via `main.py`, not via `batch.py`. This is the
simplest non-surprising policy. Users who want fine control can run a
single-scenario "batch" of one TOML to get the inner-loop parallelism.

The CLI summary lists which level of parallelism was applied:

```
Batch: 8 scenarios, 4 workers (sweep-level parallelism disabled)
```

## Reproducibility

For every scenario, `resolved.toml` (the fully merged config) is written into
the scenario's output directory alongside `wideband.png` etc. This makes any
single run rerunnable: `python main.py path/to/scenario_dir/resolved.toml`
reproduces it exactly, no manifest needed.

## Implementation sketch

New files:
- `batch.py` (repo root) — CLI entry point. Mirrors `main.py`'s shape.

Touches:
- Nothing in `sim/` needs to change.
- `main.py` is unaffected; `batch.py` imports and reuses `main.main()` (or a
  factored-out `main.run_with_cfg(cfg)` if `main.main` keeps loading config
  from a path).

Suggested split inside `main.py`:

```python
def main(config_path: str = "simulation.toml", progress_callback=None) -> None:
    cfg = load_config(config_path)
    run_with_cfg(cfg, progress_callback=progress_callback)

def run_with_cfg(cfg: dict, progress_callback=None) -> None:
    # Everything currently in main() after load_config(...)
    ...
```

`batch.py` then calls `run_with_cfg(cfg)` after applying any per-scenario
overrides. The GUI keeps calling `main.main(path)` as today.

## Tests

- A batch of two trivially small TOMLs (1×1 sweep each) runs and produces
  two subdirectories with the expected files.
- `--fail-fast` cancels the second scenario when the first fails.
- Default mode continues past a failure and writes `batch_summary.md` with
  one OK and one FAIL row.
- Stem-collision rename works (two same-stem inputs from different paths).
- Numbered subdir naming matches manifest order.

## Open questions

- **Deep-merge semantics for manifest overrides.** Deferred to the override
  iteration. The minimal manifest (no overrides) ships first.
- **Per-scenario logs.** Should `simulation.log` (the GUI's per-run log)
  also be written by `batch.py`? Probably yes; cheap and useful for
  post-mortem when the summary says FAIL. Default on.
- **Progress reporting.** Coarse only — one line per scenario completion.
  No per-chunk or per-iteration progress (workers are silent). Trade-off
  acceptable for overnight use.
- **Shared random seed across the batch.** Each TOML carries its own seed;
  no cross-scenario coordination needed. If a manifest study wants varied
  seeds, that goes in the override block. Within a scenario the sweep layer
  derives per-iteration seeds as `base_seed + grid_index × stride + iter_index`,
  so scenarios with the same base seed and same grid produce identical
  iteration streams.
- **Adaptive iteration is sweep-internal.** No flag is needed on the batch
  layer to enable or tune it; each scenario inherits its own `[simulation]`
  settings (`max_iterations`, `target_ci_half_width`, etc.) from its TOML.
