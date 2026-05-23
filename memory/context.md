# SO-WAT — Session Context
**Last updated: 2026-05-22**

---

## What this project is

SO-WAT (Simulation Orchestrator – Waveform Analysis Tool) is a Python satellite
communications link simulator. It models a wideband downlink: multiple carriers through a
shared nonlinear amplifier (AM-AM/AM-PM), OLA-based resampling, optional per-carrier
channel impairments, AWGN (added after the amp — see TECHNICAL_NOTES.md), and optional
FEC coding (convolutional, concatenated, turbo, LDPC).

---

## Repository layout

```
sim/            Core simulation library
  baseband.py       RRC pulse shaping, modulation symbol generation
  filters.py        FFT OLA up/downsample, channel impairments
  nonlinear_amplifier.py  AM-AM / AM-PM memoryless model
  simulation.py     N-carrier wideband simulation (returns CNR/CIR/CNIR)
  sweep.py          2D IBO × noise density sweep
  receiver.py       Matched filter, hard decisions, BER, EVM, soft_demap
  modulation.py     Constellations, Gray coding, APSK ratios
  plots.py          All figure generation + markdown reports
  config.py         TOML loader (tomllib)
  theory.py         Closed-form BER curves + numerical inverse
  coding/           FEC: ConvolutionalCode, ConcatenatedCode, TurboCode, LDPCCode
                    build_code(), encode_frames(), decode_frames()

main.py         CLI entry point — runs full sim + optional sweep + writes outputs
gui.py          Standalone tkinter TOML editor and sim launcher (SO-WAT GUI)
simulation.toml Example / default configuration
tests/          180 tests, all passing, 100% coverage
misc/
  __init__.py   Makes misc a package (importable by gui.py)
  gen_icon.py   build_icon() generates the app icon PNG; imported by gui.py at startup.
                --preview saves icon_preview.png for visual inspection.
memory/
  context.md    This file
  technical_notes.md  Key implementation decisions
```

---

## Current code quality state (as of last commit)

- **Pyright:** 0 errors
- **Ruff:** 0 errors (E701/E702 suppressed — intentional compact GUI style)
- **Tests:** 180 passing, 0 failing
- **Coverage:** 100% (1253 statements, 0 missed)
- Tools installed in venv: `pyright`, `ruff`, `pytest-cov`
- Run quality checks: `.venv\Scripts\pyright.exe gui.py main.py sim/ tests/`
  and `.venv\Scripts\ruff.exe check gui.py main.py sim/ tests/`
- Run tests with coverage: `.venv\Scripts\python.exe -m pytest tests/ --cov=sim --cov=main --cov-report=term-missing`

---

## GUI notes

- **Header band:** dark navy strip at top; "SO-WAT" centred in Consolas bold cyan;
  icon (sine wave dissolving into binary) pinned top-right; icon is generated at
  runtime via `misc.gen_icon.build_icon()` — no embedded blob in gui.py. Run
  `python misc/gen_icon.py --preview` to save icon_preview.png for inspection.
- **Channel impairments:** single checkbox only — checked = include channel dict in
  config; unchecked = omit entirely. No inner "Enabled" checkbox.
- **Carrier enable vs detector enable:** two separate checkboxes —
  "Include in wideband" and "Enable detector model" (sweep_demod)
- **FEC coding:** "FEC coding" checkbox expands a parameters panel with
  scheme (combobox), num_frames, block_length, and LDPC matrix path.

---

## Key config fields

```toml
[[carrier]]
name = "c1"
modulation = "BPSK"          # BPSK DBPSK MSK QPSK OQPSK 8PSK 16QAM 16APSK 32APSK
symbol_rate = 1_000_000
sps = 4
rolloff = 0.35
filter_span = 8              # half-span in symbols
num_symbols = 10000          # set to 0 when using FEC (derived from num_frames)
power_db = 0.0
freq = -3_000_000
enabled = true               # include in wideband composite
sweep_demod = true           # demodulate and measure BER
num_frames = 5               # for FEC-coded carriers (optional)

[carrier.coding]             # optional — omit section for uncoded
scheme = "convolutional"     # convolutional | concatenated | turbo | ldpc
block_length = 1024          # data bits/frame (conv/turbo)
# matrix = "data/ldpc/mackay_13298.alist"  # ldpc only

[carrier.channel]            # optional — omit section to disable
ripple_db = 0.5
ripple_cycles = 2.0
max_phase_dev_deg = 5.0
phase_poly_order = 2
```

---

## Open work items

1. **Carrier plan visualisation** — frequency-domain spectrum view of all carriers;
   click to select/edit. Deferred, no code started.

---

*See also: `memory/technical_notes.md` for AWGN placement rationale, DBPSK formula,
CNIR→Eb/N0 conversion, and NLA normalization decision.*
