# SO-WAT — Session Context
**Last updated: 2026-05-17**

---

## What this project is

SO-WAT (Simulation Orchestrator – Waveform Analysis Tool) is a Python satellite
communications link simulator. It models a wideband downlink: multiple carriers through a
shared nonlinear amplifier (AM-AM/AM-PM), OLA-based resampling, optional per-carrier
channel impairments, AWGN (added after the amp — see TECHNICAL_NOTES.md), and a BER
seeker that bisects noise_density_dbfs to find the operating point for a target BER.

---

## Repository layout

```
sim/            Core simulation library
  baseband.py       RRC pulse shaping, modulation symbol generation
  filters.py        FFT OLA up/downsample, channel impairments
  nonlinear_amplifier.py  AM-AM / AM-PM memoryless model
  simulation.py     N-carrier wideband simulation (returns CNR/CIR/CNIR)
  sweep.py          2D IBO × noise density sweep
  receiver.py       Matched filter, hard decisions, BER, EVM
  modulation.py     Constellations, Gray coding, APSK ratios
  plots.py          All figure generation + markdown reports
  config.py         TOML loader (tomllib)
  theory.py         Closed-form BER curves + numerical inverse
  targeter.py       Adaptive BER seeker (bisection on noise_density_dbfs)

main.py         CLI entry point — runs full sim + seeker + writes outputs
gui.py          Standalone tkinter TOML editor and sim launcher (SO-WAT GUI)
simulation.toml Example / default configuration
tests/          112 tests, all passing, 86% coverage
misc/
  gen_icon.py   Regenerates the app icon PNG and patches gui.py in-place
memory/
  context.md    This file
  technical_notes.md  Key implementation decisions
```

---

## Current code quality state (as of last commit)

- **Pyright:** 0 errors
- **Ruff:** 0 errors (E701/E702 suppressed — intentional compact GUI style)
- **Tests:** 112 passing, 0 failing
- **Coverage:** 86% overall; sim/ core modules 95–100%
- Tools installed in venv: `pyright`, `ruff`, `pytest-cov`
- Run quality checks: `python -m pyright gui.py main.py sim/ tests/`
  and `python -m ruff check gui.py main.py sim/ tests/`
- Run tests with coverage: `python -m pytest tests/ --cov=sim --cov=main --cov-report=term-missing`

---

## GUI notes

- **Header band:** dark navy strip at top; "SO-WAT" centred in Consolas bold cyan;
  icon (sine wave dissolving into binary) pinned top-right; `_ICON_B64` constant
  in gui.py holds the embedded PNG; regenerate with `python misc/gen_icon.py`
- **Channel impairments:** single checkbox only — checked = include channel dict in
  config; unchecked = omit entirely. There is NO inner "Enabled" checkbox anymore
  (removed as redundant; sim treats absent channel key same as enabled=false)
- **Carrier enable vs detector enable:** two separate checkboxes —
  "Include in wideband" and "Enable detector model" (sweep_demod)
- **Seeker mode:** shown only when both enables are on; radio between
  "Fixed noise level" and "BER seeker"; seeker params hidden unless seeker selected

---

## Key config fields

```toml
[[carrier]]
name = "c1"
modulation = "BPSK"          # BPSK DBPSK QPSK OQPSK 8PSK 16QAM 16APSK 32APSK
symbol_rate = 1_000_000
sps = 4
rolloff = 0.35
filter_span = 8              # half-span in symbols
num_symbols = 10000
power_db = 0.0
freq = -3_000_000
enabled = true               # include in wideband composite
sweep_demod = true           # demodulate and measure BER
use_seeker = false           # false = fixed noise level; true = BER seeker

[carrier.seeker]             # only used when use_seeker = true
target_ber = 0.01
confidence = 0.95
ber_accuracy = 0.005
noise_lo_dbfs = -160.0
noise_hi_dbfs = -80.0

[carrier.channel]            # optional — omit section to disable
ripple_db = 0.5
ripple_cycles = 2.0
max_phase_dev_deg = 5.0
phase_poly_order = 2
```

---

## Open work items

1. **Chunk pipeline refactor** — highest priority active item. The wideband signal
   is currently materialized in RAM in full before any processing. Plan agreed:
   - NLA normalization: analytical RMS (`sqrt(sum of 10^(power_db/10))`), not empirical peak
   - PSD estimate: Welch averaging over all chunks, not a single snapshot
   - Current `plots.psd_db()` takes a single centre segment (up to 16384 samples, Hann window);
     this will be replaced by incremental segment accumulation in the chunk pipeline
   - Current `sim/simulation.py` line 121 still uses `np.max()` — this is the thing to change
   - See `memory/technical_notes.md` § "NLA input normalization" for full rationale

2. **Carrier plan visualisation** — frequency-domain spectrum view of all carriers;
   click to select/edit. Deferred, no code started.

---

*See also: `memory/technical_notes.md` for AWGN placement rationale, DBPSK formula,
CNIR→Eb/N0 conversion, NLA normalization decision, and seeker algorithm details.*
