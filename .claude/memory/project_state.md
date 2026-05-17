# Project State

---
name: project-state
description: Current feature set, architecture, open work items, and code quality state
metadata:
  type: project
---

## What exists (as of 2026-05-17, commit bf42d1b)

**Core simulation (`sim/`):**

- `baseband.py` — RRC pulse shaping, multi-modulation symbol generation (BPSK, DBPSK,
  QPSK, OQPSK, 8PSK, 16QAM, 16APSK, 32APSK)
- `filters.py` — RRC coefficients, FFT OLA upsample/downsample, channel impairments;
  OLA chunk callbacks (_ChunkCB, _CHUNK_REPORT=64) for progress reporting
- `nonlinear_amplifier.py` — AM-AM / AM-PM lookup table amplifier
- `simulation.py` — wideband N-carrier simulation; AWGN added AFTER amplifier;
  per-carrier demod controlled by `demod_carriers` set; returns CNR, CIR, CNIR per carrier;
  chunk_print parameter threads OLA block progress to GUI log
- `sweep.py` — 2D parameter sweep (IBO × noise density); honours sweep_demod flag;
  chunk_print and point_cb parameters
- `receiver.py` — matched filter, hard decisions, BER (phase ambiguity resolution), EVM
- `modulation.py` — constellation definitions, Gray coding, APSK ring ratios
- `plots.py` — wideband PSD, NL tables, channel response, sweep results, markdown reports,
  write_detector_results(); colormaps via plt.colormaps["viridis"] (not cm.viridis)
- `config.py` — tomllib config loader
- `theory.py` — closed-form BER (ber_awgn) and inverse (ebn0_for_ber) for
  BPSK/DBPSK/QPSK/OQPSK/8PSK/16QAM; APSK returns None (no formula)
- `targeter.py` — adaptive BER seeker; seek_ber_noise_level / seek_all_carriers;
  bisects noise_density_dbfs; optional progress_callback and chunk_print;
  per-carrier seeker params from carr["seeker"] sub-dict;
  returns noise level, BER+CI, effective_ebn0_db, theory_ebn0_db, implementation_loss_db

**Entry points:**

- `main.py` — CLI; filters enabled carriers; dynamic progress fractions computed from
  sweep size; fixed-noise detector stats from main run; BER seeker for use_seeker=True
  carriers; [NNN%] progress to stdout; writes detector_results.md; chunk_print wired
  through sim, sweep, and seeker
- `gui.py` — tkinter TOML editor (4 tabs: General, Amplifier, Sweep/Output, Carriers);
  SO-WAT branding header (dark navy band, title centred, icon top-right);
  hover tooltips on all parameter fields; Stop button; 30-second silence warning in log;
  per-carrier: single "Include in wideband" enable, "Enable detector model" checkbox,
  mode radio (Fixed noise / BER seeker), seeker param fields; channel impairments
  controlled by a SINGLE checkbox (collapsed = absent from config; no redundant inner
  "Enabled" checkbox); subprocess monitoring via daemon thread + queue + root.after(100)

**Tests (`tests/`):** 112 tests, all passing

- `test_theory.py` — ber_awgn all branches, ebn0_for_ber inversion, out-of-range and no-formula None returns
- `test_targeter.py` — _erfinv edge cases, bad-carrier ValueError paths,
  progress callback invocation, seek_all_carriers filtering + single-seekable run
- `test_main.py` — fixed-demod path, progress_callback, detector_results write
- Others unchanged

**Code quality:**

- Pyright: 0 errors
- Ruff: 0 errors (E701/E702 suppressed in pyproject.toml — intentional GUI style)
- Line length: 110 chars
- Coverage: 86% overall; sim/ core modules 95–100%; plots.py 66% (rendering, expected)

**Assets:**

- `misc/gen_icon.py` — regenerates the SO-WAT icon PNG and patches gui.py in-place;
  `--preview` flag saves misc/icon_preview.png without touching gui.py

**Documentation (`docs/`):**

- `GUIDE.md` — main reference; §13–15 blurb sections link to the three deeper docs
- `simulation_overview.md` — full execution flow, all three optional paths, output files
- `memory_scaling.md` — OLA memory analysis; block buffer vs persistent wideband arrays
- `filter_analysis.md` — filter size justification; documents the circular-conv bug fix
- All flow diagrams are Mermaid (`flowchart LR` + `direction TB` subgraphs + colours)

## Open work items

- **Carrier plan visualisation (deferred):** frequency-domain picture showing all
  carriers on a spectrum view; click to select/edit

**Why:** See [[technical-notes]] for AWGN placement, seeker algorithm, and CNIR→Eb/N0.
