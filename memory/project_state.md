# Project State

Last updated: 2026-05-17, commit 5bfd607

---

## Current code quality

- **Tests:** 112 passing, 0 failing
- **Pyright:** 0 errors
- **Ruff:** 0 errors (E701/E702 suppressed in pyproject.toml — intentional compact GUI style)
- **Coverage:** 86% overall; sim/ core 95–100%; plots.py 66% (rendering paths, expected low)
- **Branch:** master, up to date with origin/master

---

## What is complete (all committed and pushed)

- Multi-modulation baseband: BPSK, DBPSK, QPSK, OQPSK, 8PSK, 16QAM, 16APSK, 32APSK
- AWGN performance test suite: BER monotonicity, theory comparison, BER/EVM/eye-diagram plots
- N-carrier wideband simulation with shared NLA, per-carrier channel impairments, CNR/CIR/CNIR
- OLA upsample/downsample with chunk-progress callbacks (for GUI progress reporting)
- 2D IBO × noise density sweep
- Adaptive BER seeker (bisects noise_density_dbfs to hit a target BER per carrier)
- GUI: tkinter TOML editor, SO-WAT branding, per-carrier enable/seeker controls, progress log
- BER theory module + numerical inverse (ber_awgn, ebn0_for_ber) for all non-APSK modulations
- Detector results markdown table (write_detector_results in plots.py)
- Docs: GUIDE.md, simulation_overview.md, memory_scaling.md, filter_analysis.md (all Mermaid diagrams)

---

## Active design decision — chunk pipeline refactor (NOT yet implemented)

The wideband simulation currently materialises the full composite signal in RAM before
processing. The plan is a chunk-wise pipeline where the wideband signal is never fully
held in memory. Two design choices have been agreed:

**NLA input normalization:** use analytical RMS, not empirical peak.
  See technical_notes.md § "NLA input normalization" for full rationale.
  Current code in sim/simulation.py still uses np.max() — this is the thing to change.

**PSD estimate:** Welch averaging over all samples (not a single chunk).
  Build PSD incrementally as chunks arrive; average periodograms across all segments.
  This was chosen over a single large-chunk estimate for lower variance.

The chunk pipeline itself has not been started. When implementing:
- Upsample pass produces chunks at wideband rate → accumulate composite → NL → AWGN
- Per-carrier downsample reads the same wideband chunks
- Welch PSD accumulates segment FFTs as chunks pass through
- RMS normalization computed analytically before any chunks are processed

---

## Open work items

1. **Chunk pipeline refactor** — described above; design decided, no code written yet
2. **Carrier plan visualisation** — frequency-domain spectrum view of all carriers with
   click-to-select/edit. Deferred, no code started.

---

## Toolchain / setup

- Package manager: `uv`; venv at `.venv/`
- Python: `.venv/Scripts/python.exe` (Windows) or `.venv/bin/python` (Linux/Mac)
- Run tests: `python -m pytest tests/ -v`
- Run tests with coverage: `python -m pytest tests/ --cov=sim --cov=main --cov-report=term-missing`
- Type check: `python -m pyright gui.py main.py sim/ tests/`
- Lint: `python -m ruff check gui.py main.py sim/ tests/`
- Git remote: https://github.com/bandospook/signal-processing-simulation.git, branch master
