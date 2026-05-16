---
name: project-state
description: Current implementation status, what's done, and the exact next task to resume
metadata:
  type: project
---

## What was just completed (committed and pushed to master)

Commit `e0d771f`: Multi-modulation support — DBPSK, QPSK, OQPSK, 8PSK, 16QAM, 16APSK, 32APSK

Files added/changed:
- `sim/modulation.py` — NEW: constellations, map_bits, decide, differential encode/decode, rotational_symmetry
- `sim/bpsk.py` — `rrc_baseband()` now handles all supported modulations; OQPSK Q-rail delay built in
- `sim/receiver.py` — `receive()` is now general; RMS normalisation before decision (critical for QAM/APSK); `_ber_with_ambiguity()` tries all rotationally equivalent orientations
- `sim/simulation.py` — reads `modulation` and APSK gamma keys from carrier config
- `tests/test_modulations.py` — 42 tests: constellation properties, map/decide roundtrip, DBPSK encoding, noiseless end-to-end BER/EVM for all 8 modulations (all pass)

All 42 tests pass. Branch is clean and up to date with origin/master.

## Immediately next task (NOT started yet)

Write `tests/test_awgn_performance.py` — AWGN noise sweep tests with plots.

The user explicitly requested:
> "If you could do that over a noise sweep for a single carrier that would be pretty awesome. It'd be good to capture a few plots too. An EVM plot, maybe an eye diagram, a BER curve Es/No showing theory and actual."

### Planned content for test_awgn_performance.py

**Helper functions**:
- `simulate_awgn(mod, EsN0_dB, n_sym, sps, rolloff, filter_span, seed)` → calls `rrc_baseband` + adds noise + calls `receive`
- `ber_theory(mod, EsN0_dB)` → theoretical BER using erfc (scipy NOT available — must use `math.erfc` or `numpy` erfc approximation); returns None for 16APSK/32APSK

**Note: scipy is not in dependencies.** Use `math.erfc` or implement Q-function via `numpy`:
```python
from math import erfc
Q = lambda x: 0.5 * erfc(x / np.sqrt(2))
```

**Noise formula** (derived, verified):
For baseband normalized to unit RMS, symbol_rate=1, sample_rate=sps:
```
sigma_c = sqrt(sps / (2 * EsN0_linear))   # per-component (real or imaginary) noise std
noise = sigma_c * rng.standard_normal(len(bb)) + 1j * sigma_c * rng.standard_normal(len(bb))
```
This gives exactly the correct Es/N0 at the decision point after the matched filter pair. Derivation verified:
- Signal power = 1 (normalized), symbol amplitude at MF output ≈ sqrt(sps)
- Noise per component after MF: sigma_c * sqrt(||h_rrc||^2) = sigma_c (unit-energy filter)
- BER_BPSK = Q(sqrt(sps)/sigma_c) = Q(sqrt(2*EsN0)) = Q(sqrt(2*EbN0)) ✓

**Theoretical BER formulas** (all in terms of EbN0 = EsN0_linear / bps):
- BPSK:   `0.5 * erfc(sqrt(EbN0))`
- DBPSK:  `0.5 * exp(-EbN0)`
- QPSK:   `0.5 * erfc(sqrt(EbN0))`   (same as BPSK per bit)
- OQPSK:  `0.5 * erfc(sqrt(EbN0))`   (same as QPSK per bit)
- 8PSK:   `(1/3) * erfc(sqrt(3*EbN0) * sin(pi/8))`
- 16QAM:  `(3/8) * erfc(sqrt(2*EbN0/5))`
- 16APSK: None
- 32APSK: None

**Tests**:
- `test_ber_monotone[mod]` — BER strictly decreases over 5 SNR points; 1000 symbols/point, fast
- `test_ber_matches_theory[mod]` — at one moderate SNR (~5% BER), measured BER within factor 2 of theory; 5000 symbols; parametrized for BPSK/QPSK/OQPSK/8PSK/16QAM (not DBPSK/16APSK/32APSK — no simple theory)

**Plots** (always saved to `plots/performance/` which is created if missing):
- `ber_vs_ebn0.png` — BER vs Eb/N0 all mods, theory dashed, measured solid
- `evm_vs_ebn0.png` — EVM% vs Eb/N0 all mods
- `eye_diagram_{mod}.png` — BPSK, QPSK, 16QAM at 10 dB Eb/N0

**Eye diagram implementation**:
```python
from sim.receiver import matched_filter
mf_out = matched_filter(bb_noisy, rolloff, filter_span, sps)
# Overlay 200 traces of 2*sps samples each, triggered at symbol boundaries
```

## Deferred (agreed with user to do later)

- MSK (Tier 3)
- OFDM (Tier 3)

## Package manager / toolchain

- uv: `uv add scipy` if scipy needed (currently not in deps — use math.erfc)
- Python: `.venv/Scripts/python.exe`
- Tests: `.venv/Scripts/python.exe -m pytest tests/ -v`
- Git remote: https://github.com/bandospook/signal-processing-simulation.git, branch master
