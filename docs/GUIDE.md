# Simulation Guide

## Contents

1. [What this tool does](#1-what-this-tool-does)
2. [Signal chain](#2-signal-chain)
3. [Getting started](#3-getting-started)
4. [Source files](#4-source-files)
5. [Configuration reference](#5-configuration-reference)
6. [Output files](#6-output-files)
7. [Example results](#7-example-results)
8. [Sweep mode](#8-sweep-mode)
9. [Adding or modifying carriers](#9-adding-or-modifying-carriers)
10. [GUI](#10-gui)
11. [Test suite](#11-test-suite)
12. [Memory scaling](memory_scaling.md) — filter cost, FFT buffer sizing, OLA efficiency vs symbol rate ratio
13. [Filter analysis](filter_analysis.md) — filter size justification, upsampling fidelity, IMD rejection adequacy

---

## 1. What this tool does

This simulation models a satellite or RF ground-station scenario where multiple
carriers — potentially at very different symbol rates and modulations — share a single
power amplifier. It lets you answer questions like:

- How much intermodulation distortion (IMD) does the amplifier inject into each
  carrier at a given input backoff?
- What is the carrier-to-interference ratio (CIR) and how does it compare to the
  carrier-to-noise ratio (CNR)?
- At what IBO does BER become unacceptable for each carrier?
- How does a passband channel impairment (amplitude ripple, phase nonlinearity)
  interact with the NL distortion?
- How do EVM and BER track across a 2-D sweep of IBO and noise density?

The simulation is fully deterministic (fixed seed), TOML-configured, and produces
both console tables and PNG plots. No code changes are needed to explore new
operating points.

---

## 2. Signal chain

```
 Per carrier (native rate = sps × symbol_rate)
 ─────────────────────────────────────────────
  Random symbols — modulation set per carrier
  (BPSK / DBPSK / QPSK / OQPSK / 8PSK / 16QAM / 16APSK / 32APSK)
        │
        ▼  RRC transmit filter  (baseband.py)
        │
        ▼  Channel impairment — optional  (filters.py)
        │    amplitude ripple: cosine across passband
        │    phase nonlinearity: polynomial vs frequency
        │
        ▼  OLA upsample → wideband sample rate  (filters.py)
        │
        ▼  Frequency shift to carrier centre freq
        │
        ▼  Scale by carrier power_db

 Composite wideband signal (wideband sample rate)
 ─────────────────────────────────────────────────
        Σ  Sum all carriers
        │
        ▼  Normalise to unit peak, apply input backoff
        │
        ▼  Nonlinear amplifier — AM-AM + AM-PM  (nonlinear_amplifier.py)
        │
        ▼  Add wideband AWGN — models receiver thermal noise
        │  (AWGN is added AFTER the amp: satellite downlink model where
        │   uplink noise is a separate link-budget item)

 Per carrier — extraction and receive  (if sweep_demod = true)
 ─────────────────────────────────────
        ▼  Downconvert (multiply by exp(−j 2π f_c t))
        │
        ▼  OLA downsample → native rate  (filters.py)
        │     (anti-alias Kaiser sinc: ≈80 dB stopband)
        │
        ▼  RRC matched filter  (receiver.py)
        │
        ▼  Symbol sampling (1 sample / symbol)
        │
        ▼  Hard decisions + phase-ambiguity resolution  →  BER
        │
        ▼  Metrics: EVM, CNR, CIR, CNIR  (simulation.py / receiver.py)
```

### AWGN placement

AWGN is added after the nonlinear amplifier. This models a single-hop satellite
downlink where thermal noise is primarily at the receiver. The uplink noise
contribution (retransmitted by the transponder) is handled as a separate link budget
item using the reciprocal sum:

```
1/(C/N)_total = 1/(C/N)_UL + 1/(C/N)_DL + 1/(C/N)_IM
```

Placing noise before the amp would couple the noise level to the IM products,
making the noise-vs-distortion trade-off analysis ill-conditioned.

### CNR / CIR / CNIR computation

Three OLA downsampling extractions per carrier: pre-NL reference, post-NL noiseless,
post-NL+noise. A complex projection separates the deterministic AM-AM/AM-PM effect
from residual in-band IM distortion:

```
α          = ⟨bb_rx, nl_pure⟩ / ⟨bb_rx, bb_rx⟩   (complex gain of desired component)
signal     = α · bb_rx                              (desired part of nl_pure)
distortion = nl_pure − signal                       (true IM products)
noise      = nl_noisy − nl_pure                     (additive noise)

CIR  (dB)  = 10 log₁₀( P_signal / P_distortion )
CNR  (dB)  = 10 log₁₀( P_signal / P_noise )
CNIR (dB)  = 10 log₁₀( P_signal / (P_distortion + P_noise) )
```

This correctly attributes AM-AM compression as a gain change rather than distortion,
and is independent of absolute amplitude scaling.

---

## 3. Getting started

### Prerequisites

- [uv](https://github.com/astral-sh/uv) installed (handles Python version and packages)
- Git (to clone the repo)

### Step 1 — Clone and install

```powershell
git clone https://github.com/bandospook/signal-processing-simulation.git
cd signal-processing-simulation
uv sync          # creates .venv and installs numpy + matplotlib
```

### Step 2 — Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 3 — Run with the default configuration

```powershell
python main.py
```

This will:
1. Print a metrics table to the console.
2. Open interactive matplotlib windows showing the wideband PSD, amplifier curves,
   and channel responses.
3. Save PNG files into the `output/` directory.
4. If `[sweep]` is present, run the full IBO × noise grid and save
   `sweep_results.png` and `sweep_table.md`.

### Step 4 — Use the GUI

```powershell
python gui.py
```

Load any `.toml` config, edit all parameters in a tabbed interface, save, and launch
`main.py` with that config directly from the GUI. See [§10 GUI](#10-gui).

---

## 4. Source files

```
signal-processing-simulation/
├── sim/                      ← simulation package
│   ├── baseband.py           ← multi-modulation RRC baseband generation
│   ├── modulation.py         ← constellation definitions (all 8 modulations)
│   ├── config.py             ← TOML loader
│   ├── filters.py            ← RRC, OLA convolution, upsample/downsample, channel impairment
│   ├── nonlinear_amplifier.py← memoryless AM-AM + AM-PM model
│   ├── plots.py              ← all visualisation and sweep markdown report
│   ├── receiver.py           ← matched filter, decisions, BER (phase-ambiguity resolved), EVM
│   ├── simulation.py         ← full wideband signal chain, per-carrier metric extraction
│   └── sweep.py              ← 2-D IBO × noise sweep
├── tests/
│   ├── test_awgn_performance.py  ← BER vs theory (see §11)
│   ├── test_filters.py           ← RRC and OLA correctness
│   ├── test_nonlinear_amplifier.py
│   ├── test_main.py              ← end-to-end smoke test
│   └── test_wideband.py          ← wideband simulation integration
├── docs/
│   ├── GUIDE.md              ← this file
│   ├── memory_scaling.md
│   └── filter_analysis.md
├── output/                   ← generated files (git-ignored)
├── gui.py                    ← standalone TOML editor + launcher
├── main.py                   ← CLI entry point
├── simulation.toml           ← configuration
└── pyproject.toml
```

| File | Role |
|---|---|
| `sim/baseband.py` | Generates RRC-filtered complex baseband signal for any supported modulation at native sample rate. Normalised to unit RMS power. |
| `sim/modulation.py` | Constellation definitions, Gray coding, APSK ring ratios, `bits_per_symbol()`. All constellations normalised to unit average power. |
| `sim/filters.py` | RRC coefficients, OLA convolution, OLA upsample/downsample (anti-alias Kaiser sinc), per-carrier channel impairments. |
| `sim/nonlinear_amplifier.py` | Memoryless AM-AM + AM-PM model; piecewise linear interpolation of user-supplied lookup tables. |
| `sim/simulation.py` | Orchestrates the full signal chain. AWGN added after amp. Per-carrier demod controlled by `demod_carriers` set. Returns CNR/CIR/CNIR per carrier via projection method. |
| `sim/receiver.py` | `matched_filter`, `receive` (chains filter → sampling → decisions → BER with rotational ambiguity resolution → EVM). Uses `np.real()`/`np.imag()` throughout (Pylance compatible). |
| `sim/sweep.py` | 2-D sweep over IBO × noise; honours `sweep_demod` per carrier. |
| `sim/plots.py` | Wideband PSD (capped at 16384-point FFT), amplifier curves, channel response, sweep plots, `write_sweep_report` (markdown). |

---

## 5. Configuration reference

All parameters live in `simulation.toml`. Large integers may use underscores
(`2_000_000_000`); `tomllib` and the GUI serialiser both preserve this convention.

### `[simulation]`

| Key | Type | Description |
|---|---|---|
| `seed` | int | Global random seed for reproducible symbol sequences and noise. |

### `[wideband]`

| Key | Type | Description |
|---|---|---|
| `sample_rate` | int (Hz) | Common sample rate for the composite signal. Must be an integer multiple of every carrier's native rate (`sps × symbol_rate`). |
| `noise_density_dbfs` | float (dBFS/Hz) | One-sided AWGN PSD added **after** the amplifier. Total noise power = 10^(N₀/10) × sample_rate. Remove to disable noise. |

### `[amplifier]`

| Key | Type | Description |
|---|---|---|
| `input_backoff_db` | float (dB) | Peak drive level relative to saturation. 0 dB = full saturation; 3 dB = peak at 0.71 of saturation (typical). Higher = more linear, less efficient. |

### `[amplifier.am_am]` and `[amplifier.am_pm]`

| Key | Type | Description |
|---|---|---|
| `input` | float list | Normalised input amplitude breakpoints (0–1). |
| `output` | float list | (am_am) Output amplitude at each breakpoint. |
| `phase_deg` | float list | (am_pm) Phase shift in degrees at each amplitude. |

### `[ola]`

| Key | Default | Description |
|---|---|---|
| `filter_span` | 16 | Half-span of the Kaiser-sinc interpolation filter in symbols. Larger = better stopband, slower. |
| `block_size` | 4096 | FFT block size for OLA convolution. Powers of two are most efficient. |

### `[output]`

| Key | Description |
|---|---|
| `output_dir` | Directory for all output files. Created automatically. |
| `wideband` | Filename for the wideband PSD figure. |
| `nl_tables` | Filename for the AM-AM/AM-PM plot. |
| `sweep` | Filename for the sweep results PNG. |
| `sweep_table` | Filename for the sweep markdown report. |

### `[sweep]`

Remove this section to disable the sweep. Both keys required to trigger a run.

| Key | Description |
|---|---|
| `ibo_db` | List of IBO values (dB) to sweep. |
| `noise_density_dbfs` | List of noise densities (dBFS/Hz) to sweep. |

### `[[carrier]]` (repeated block, one per carrier)

| Key | Type | Description |
|---|---|---|
| `name` | string | Label used in plots and console. |
| `modulation` | string | One of: `BPSK`, `DBPSK`, `QPSK`, `OQPSK`, `8PSK`, `16QAM`, `16APSK`, `32APSK`. Defaults to `BPSK` if omitted. |
| `symbol_rate` | int (Hz) | Symbol rate. Native sample rate = `sps × symbol_rate`. |
| `sps` | int | Samples per symbol at native rate. |
| `rolloff` | float | RRC rolloff factor (0–1). Occupied BW ≈ `(1+rolloff) × symbol_rate`. |
| `filter_span` | int | RRC filter half-span in symbols (TX and RX use the same value). Total taps = `filter_span × sps + 1`. |
| `num_symbols` | int | Symbols to generate. More = better BER statistics, more memory. |
| `power_db` | float (dB) | Carrier power relative to the 0 dB reference. Controls inter-carrier ratio before the amp; the composite is peak-normalised before the NL. |
| `freq` | int (Hz) | Centre frequency in the wideband spectrum. Carriers must not overlap. |
| `sweep_demod` | bool | If `false`, skip demodulation and metrics for this carrier during sweep (default `true`). The carrier still appears in the composite and contributes to IM products. |

### `[carrier.channel]` (optional, per carrier)

| Key | Description |
|---|---|
| `enabled` | Master enable. `false` = bypass all impairments. |
| `ripple_db` | Peak-to-peak amplitude ripple across passband (dB). |
| `ripple_cycles` | Number of complete ripple cycles across the signal bandwidth. |
| `max_phase_dev_deg` | Peak phase deviation from linear phase at band edge (°). |
| `phase_poly_order` | Polynomial order of the phase shape (2 = quadratic). |
| `plot` | Filename for the channel impairment response plot. |

---

## 6. Output files

| File | Contents |
|---|---|
| `output/wideband.png` | Wideband PSD (pre-NL, post-NL, post-NL+noise) plus per-carrier baseband PSD panels. Spectral regrowth from the NL amplifier is visible as raised floor between carriers. |
| `output/amplifier_nl.png` | AM-AM and AM-PM curves with peak operating point (red marker). |
| `output/channel_<name>.png` | Amplitude ripple and phase nonlinearity across each carrier's passband. |
| `output/sweep_results.png` | Per-carrier rows: BER vs IBO, EVM vs IBO, CNR/CIR/CNIR vs IBO. Multiple noise levels colour-coded. |
| `output/sweep_table.md` | Markdown report: configuration summary, performance summary (min/max ranges), full IBO × noise grid table. |

---

## 7. Example results

### Console metrics table

```
Carrier     CNR (dB)  CIR (dB)  CNIR (dB)  EVM (%)  BER
slow            78.9      48.1       48.1     4.84    0.000
fast            65.8      40.8       40.8     3.33    0.000
```

**Interpreting these numbers:**

- **CNR 79/66 dB** — Noise density of −160 dBFS/Hz is far below the carrier power;
  thermal noise is negligible. Raise `noise_density_dbfs` toward −140 dBFS/Hz to
  bring CNR into the picture.
- **CIR 48/41 dB** — At 3 dB IBO the amp is moderately backed off; measurable but
  not severe IM distortion. Reduce IBO to 1 dB to drop CIR sharply.
- **CNIR ≈ CIR** — Distortion-limited regime; noise is not yet a factor.
- **EVM 4.8/3.3%** — Includes NL distortion and channel impairments. The slow
  carrier has larger ripple (0.5 dB vs 0.3 dB) hence slightly higher EVM.
- **BER = 0** — No errors in the simulated symbols. Drive harder (IBO = 0–1 dB)
  or increase noise to push into error.

---

## 8. Sweep mode

The sweep runs the full simulation on every (IBO, noise) pair in the Cartesian
product of the two lists in `[sweep]`. Only carriers with `sweep_demod = true`
have demodulation performed at each sweep point; others contribute to the wideband
IM environment but their BER/EVM are not computed, saving significant time.

**Markdown report** (`sweep_table.md`) contains:
- Configuration summary (IBO range, noise range, carrier list)
- Performance summary (min/max BER, EVM, CNIR across the grid)
- Full table of every IBO × noise combination

---

## 9. Adding or modifying carriers

**Constraints:**

1. **Integer upsample factor** — `sample_rate` must be an exact integer multiple of
   `sps × symbol_rate`. E.g., at 2 GHz, `symbol_rate=5_000_000` with `sps=10`
   gives native rate 50 MHz and factor 40 (valid). `symbol_rate=3_000_000` gives
   factor 66.67 (invalid → `ValueError`).

2. **No spectral overlap** — each carrier occupies roughly
   `[freq − (1+rolloff)·sr/2, freq + (1+rolloff)·sr/2]`. Check all pairs.

3. **Within wideband bandwidth** — all carriers must fit within
   `[−sample_rate/2, +sample_rate/2]`.

### Example: adding a 16QAM carrier

```toml
[[carrier]]
name        = "medium"
modulation  = "16QAM"
symbol_rate = 5_000_000
sps         = 8
rolloff     = 0.25
filter_span = 10
num_symbols = 5_000
power_db    = 0
freq        = -500_000_000
sweep_demod = true

[carrier.channel]
enabled           = false
```

---

## 10. GUI

`gui.py` is a standalone tkinter application. It does not import any `sim/` modules —
it reads and writes `simulation.toml` directly.

**Tabs:**

| Tab | Contents |
|---|---|
| General | Wideband sample rate, noise density, OLA parameters, simulation seed |
| Amplifier | IBO, AM-AM table (input/output columns), AM-PM table (input/phase_deg columns) |
| Sweep & Output | IBO sweep list, noise sweep list, output directory and filenames |
| Carriers | One expandable frame per carrier; all per-carrier parameters including `sweep_demod` checkbox and per-carrier channel impairment config |

**Launching the simulation:**

The "Run Simulation" button saves the current config to the loaded TOML path and
launches `main.py` with that path as a command-line argument, so the simulation
always uses the values shown in the GUI.

---

## 11. Test suite

Run all tests:

```powershell
python -m pytest
```

Run with coverage:

```powershell
python -m pytest --cov=sim --cov-report=term-missing
```

### `tests/test_awgn_performance.py`

The primary validation suite for the modulation and receiver chain. All tests operate
in an isolated AWGN channel (no nonlinear amplifier) to verify the baseband and
receiver modules independently.

| Test | What it checks | Why it matters |
|---|---|---|
| `test_ber_monotone[MOD]` | BER strictly decreases as Eb/N0 increases over 5 points | Catches sign errors, inverted noise, or sampling timing bugs |
| `test_ber_matches_theory[MOD]` | Measured BER is within 2× of theory at one mid-range SNR point | Confirms the noise model and symbol count are calibrated |
| `test_ber_theory_table` | Interpolation-based Eb/N0 vs BER comparison across 5 target BER levels for 6 modulations; writes `tests/plots/performance/theory_comparison.md` | Quantifies implementation loss across the full BER range; catches systematic offsets |
| `test_generate_performance_plots` | Generates BER-vs-Eb/N0 and EVM-vs-Eb/N0 plots with 1/2/3σ uncertainty bands; writes PNGs to `tests/plots/performance/` | Visual regression reference; sigma bands show where statistical confidence is meaningful |

**Theory formulas used:**

| Modulation | BER formula | Notes |
|---|---|---|
| BPSK, QPSK, OQPSK | `0.5 · erfc(√(Eb/N0))` | Same formula; curves are identical |
| DBPSK | `2p(1−p)`, `p = 0.5·erfc(√(Eb/N0))` | Coherent detection + differential decoding; NOT the differentially-coherent `0.5·exp(−Eb/N0)` |
| 8PSK | `(1/3)·erfc(√(3·Eb/N0)·sin(π/8))` | Approximate for Gray-coded 8PSK |
| 16QAM | `(3/8)·erfc(√(2·Eb/N0/5))` | Standard rectangular 16QAM |
| 16APSK, 32APSK | No closed form | Only monotonicity tested |

**Symbol count and confidence:**

Both `test_ber_theory_table` and `test_generate_performance_plots` use the same
`_N_BITS_PLOT` constant with `n_sym = _N_BITS_PLOT // bps` per modulation, ensuring
equal statistical confidence across all modulations. Current default: `10_000` bits
(≈4 s total). For a rigorous run set `_N_BITS_PLOT = 1_000_000`; this gives ±0.001
BER accuracy at 95% confidence (worst-case p=0.5 derivation:
N = (1.96/0.001)² × 0.25 = 960,400 → rounded to 1,000,000).

### `tests/test_filters.py`

Verifies RRC filter properties (Nyquist criterion: zero ISI at symbol samples),
OLA convolution accuracy (result matches `np.convolve`), and channel impairment
transfer functions.

### `tests/test_nonlinear_amplifier.py`

Verifies AM-AM and AM-PM table lookup, interpolation at breakpoints, saturation
behaviour, and that a linear AM-AM table produces no phase distortion.

### `tests/test_main.py`

End-to-end smoke test: mocks `load_config` with a minimal two-carrier config, runs
`main()`, and asserts the expected output PNGs are written. Catches import errors,
config-loading regressions, and broken output paths.

### `tests/test_wideband.py`

Integration tests on the full `wideband_bpsk_simulation` function: checks that CNR
varies correctly with noise density, CIR varies with IBO, and that disabling
`demod_carriers` returns NaN placeholders without affecting the wideband signal.
