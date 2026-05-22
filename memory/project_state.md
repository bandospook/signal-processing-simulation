# Project State

Last updated: 2026-05-22, commit d7ae2b8

---

## Current code quality

- **Tests:** 180 passing, 0 failing
- **Pyright:** 0 errors
- **Ruff:** 0 errors (E701/E702 suppressed in pyproject.toml — intentional compact GUI style)
- **Coverage:** 100% across all modules (1253 statements, 0 missed)
- **Branch:** master, up to date with origin/master

---

## What is complete (all committed and pushed)

- Multi-modulation baseband: BPSK, DBPSK, MSK, QPSK, OQPSK, 8PSK, 16QAM, 16APSK, 32APSK
  (MSK built as offset-QPSK/half-sine — see technical_notes.md)
- AWGN performance test suite: BER monotonicity, theory comparison, BER/EVM/eye-diagram plots
- N-carrier wideband simulation with shared NLA, per-carrier channel impairments, CNR/CIR/CNIR
- OLA upsample/downsample with chunk-progress callbacks (for GUI progress reporting)
- 2D IBO × noise density sweep
- FEC coding: convolutional (K=7, rate-1/2, soft Viterbi), concatenated (RS+convolutional),
  turbo (rate-1/3 PCCC, max-log-MAP BCJR), LDPC (normalized min-sum BP)
  — fully wired into simulation TX/RX chain via [carrier.coding] config block
- GUI: tkinter TOML editor, SO-WAT branding, per-carrier enable/FEC coding controls, progress log
- GUI: app icon generated at runtime (misc.gen_icon.build_icon() + Pillow); no base64 blob in gui.py
- GUI: simulation stdout teed to `simulation.log` in the configured output directory each run
- BER theory module + numerical inverse (ber_awgn, ebn0_for_ber) for all non-APSK modulations
- Detector results markdown table (write_detector_results in plots.py)
- Docs: GUIDE.md, simulation_overview.md, memory_scaling.md, filter_analysis.md,
  channel_impairment.md, msk_modulation.md, synchronization.md, coding_design.md

---

## FEC coding implementation (complete, commit d7ae2b8)

Four codec classes in sim/coding/:
- ConvolutionalCode: (171,133)-octal K=7, soft Viterbi, Numba @njit(parallel=True) batch decode
- ConcatenatedCode: RS(255,223) outer (galois) + ConvolutionalCode inner + random interleaver
- TurboCode: rate-1/3 PCCC, two RSC encoders, max-log-MAP BCJR, Numba @njit(parallel=True)
- LDPCCode: normalized min-sum belief propagation, GF(2) systematic generator, Numba @njit(parallel=True)

Wiring in simulation.py:
- TX: if carrier has [carrier.coding], encode_frames() → FEC coded bits → rrc_baseband(bits=...)
- RX: soft_demap() → decode_frames() → post-decoder BER; uncoded_ber also stored

Config: [carrier.coding] block with scheme, block_length (conv/turbo), matrix (ldpc), num_frames

BER seeker (sim/targeter.py) removed. Implementation loss now computed from single fixed-noise run.

---

## Open work items

1. **Carrier plan visualisation** — frequency-domain spectrum view of all carriers with
   click-to-select/edit. Deferred, no code started.

---

## Toolchain / setup

- Package manager: `uv`; venv at `.venv/`
- Python: `.venv\Scripts\python.exe` (Windows) or `.venv/bin/python` (Linux/Mac)
- Run tests: `.venv\Scripts\python.exe -m pytest tests/ -v --cov=sim --cov=main --cov-report=term-missing`
- Type check: `.venv\Scripts\pyright.exe gui.py main.py sim/ tests/`
- Lint: `.venv\Scripts\ruff.exe check gui.py main.py sim/ tests/`
- Git remote: https://github.com/bandospook/signal-processing-simulation.git, branch master
