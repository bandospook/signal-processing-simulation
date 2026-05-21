# Project State

Last updated: 2026-05-17, commit ec936f3

---

## Current code quality

- **Tests:** 162 passing, 0 failing
- **Pyright:** 0 errors
- **Ruff:** 0 errors (E701/E702 suppressed in pyproject.toml — intentional compact GUI style)
- **Coverage:** 100% across all modules (1091 statements, 0 missed)
- **Branch:** master, up to date with origin/master

---

## What is complete (all committed and pushed)

- Multi-modulation baseband: BPSK, DBPSK, MSK, QPSK, OQPSK, 8PSK, 16QAM, 16APSK, 32APSK
  (MSK built as offset-QPSK/half-sine — see technical_notes.md)
- AWGN performance test suite: BER monotonicity, theory comparison, BER/EVM/eye-diagram plots
- N-carrier wideband simulation with shared NLA, per-carrier channel impairments, CNR/CIR/CNIR
- OLA upsample/downsample with chunk-progress callbacks (for GUI progress reporting)
- 2D IBO × noise density sweep
- Adaptive BER seeker (bisects noise_density_dbfs to hit a target BER per carrier)
- GUI: tkinter TOML editor, SO-WAT branding, per-carrier enable/seeker controls, progress log
- GUI: app icon generated at runtime (misc.gen_icon.build_icon() + Pillow); no base64 blob in gui.py
- GUI: simulation stdout teed to `simulation.log` in the configured output directory each run
- BER theory module + numerical inverse (ber_awgn, ebn0_for_ber) for all non-APSK modulations
- Detector results markdown table (write_detector_results in plots.py)
- Docs: GUIDE.md, simulation_overview.md, memory_scaling.md, filter_analysis.md (all Mermaid diagrams)

---

## Recent major change: chunk pipeline refactor (complete, commit e33778e)

wideband_bpsk_simulation now processes the composite signal in OLA blocks of
ola_block_size wideband samples. No full-length wideband arrays are ever held in RAM.

Key implementation details (sim/simulation.py):
- OLAState (sim/filters.py) — stateful overlap-add convolution; process() returns
  block_size filtered samples per call, maintaining the overlap tail between calls
- x_up_block() — zero-inserted upsample block without allocating the full N_wb array
- _WelchState — Welch PSD accumulator; feeds chunks until nfft=16384 samples, averages
- _decimate() — phase-coherent decimation; carries offset across chunk boundaries
- Analytical RMS normalization: norm = drive / sqrt(sum(10^(power_db/10))) per carrier
  (replaces seed-dependent np.max() — see technical_notes.md § "NLA input normalization")
- Three downsampler paths per carrier: pre-NL reference, post-NL noiseless, post-NL+noise
- Transient trim: 2 * ola_filter_span native samples stripped from collected buffers
  before BER/EVM computation (restores symbol alignment across the two OLA filter stages)
- Return dict keys: psd_pre_nl, psd_post_nl, psd_noisy, has_noise, carriers

---

## Open work items

1. **Carrier plan visualisation** — frequency-domain spectrum view of all carriers with
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
