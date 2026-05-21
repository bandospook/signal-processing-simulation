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
9. [BER seeker and implementation loss](#9-ber-seeker-and-implementation-loss)
10. [Adding or modifying carriers](#10-adding-or-modifying-carriers)
11. [GUI](#11-gui)
12. [Test suite](#12-test-suite)
13. [Simulation overview](simulation_overview.md) — full execution flow, all three optional paths, and output files produced by each
14. [Memory scaling](memory_scaling.md) — filter cost, FFT buffer sizing, OLA efficiency vs symbol rate ratio
15. [Filter analysis](filter_analysis.md) — filter size justification, upsampling fidelity, IMD rejection adequacy
16. [Toolchain](toolchain.md) — correct invocations for pytest/pyright/ruff, Windows Store Python stub, pyrightconfig.json
17. [Channel impairment model](channel_impairment.md) — transfer function, band-edge behaviour, baseband-equivalent representation, cross-carrier limitation
18. [MSK modulation](msk_modulation.md) — why MSK uses a matched-filter (offset-QPSK) demodulator rather than a Viterbi decoder, and why it attains the BPSK error rate

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
- **What noise level achieves a specific target BER for each carrier, and how much
  implementation loss does the nonlinear environment impose relative to AWGN theory?**

The simulation is fully deterministic (fixed seed), TOML-configured, and produces
both console tables and PNG/markdown outputs. No code changes are needed to explore new
operating points.

---

## 2. Signal chain

```mermaid
flowchart LR
    subgraph TX["Per carrier  —  native rate = sps × symbol_rate"]
        direction TB
        S["Random symbols<br/>BPSK · DBPSK · MSK · QPSK · OQPSK<br/>8PSK · 16QAM · 16APSK · 32APSK"]
        RRC_TX["RRC transmit filter<br/>baseband.py"]
        CHIMP["Channel impairment — optional<br/>filters.py<br/>amplitude ripple · phase nonlinearity"]
        UPSMPL["OLA upsample to wideband rate<br/>filters.py"]
        FSHIFT["Frequency-shift to carrier centre freq<br/>Scale by carrier power_db"]
        S --> RRC_TX --> CHIMP --> UPSMPL --> FSHIFT
    end

    subgraph WB["Composite  —  wideband sample rate"]
        direction TB
        SUM["Σ  Sum all enabled carriers"]
        NORM["Normalise to unit peak<br/>apply input backoff"]
        NLA["Nonlinear amplifier<br/>AM-AM + AM-PM<br/>nonlinear_amplifier.py"]
        AWGN["Add wideband AWGN<br/>after amp · satellite downlink model"]
        SUM --> NORM --> NLA --> AWGN
    end

    subgraph RX["Per carrier  —  extraction and receive  (if sweep_demod = true)"]
        direction TB
        DCNV["Downconvert<br/>exp(-j 2π f_c t)"]
        DNSMPL["OLA downsample to native rate<br/>filters.py · Kaiser sinc ≈80 dB"]
        MF["RRC matched filter<br/>receiver.py"]
        SSAMP["Symbol sampling<br/>1 sample per symbol"]
        HDEC["Hard decisions<br/>phase-ambiguity resolution → BER"]
        MTRX["Metrics: EVM · CNR · CIR · CNIR<br/>simulation.py / receiver.py"]
        DCNV --> DNSMPL --> MF --> SSAMP --> HDEC --> MTRX
    end

    FSHIFT --> SUM
    AWGN --> DCNV

    classDef tx fill:#dbeafe,stroke:#3b82f6,color:#1e3a8a
    classDef wb fill:#fef3c7,stroke:#f59e0b,color:#78350f
    classDef rx fill:#dcfce7,stroke:#22c55e,color:#14532d

    class S,RRC_TX,CHIMP,UPSMPL,FSHIFT tx
    class SUM,NORM,NLA,AWGN wb
    class DCNV,DNSMPL,MF,SSAMP,HDEC,MTRX rx
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
1. Print a per-carrier metrics table to the console, with progress indicators.
2. Save PNG files into the `output/` directory (wideband PSD, amplifier curves,
   channel responses).
3. If `[sweep]` is configured, run the full IBO × noise grid and save
   `sweep_results.png` and `sweep_table.md`.
4. If any carriers have `sweep_demod = true`, write a `detector_results.md` with
   BER, Eb/N0, and implementation loss. If `use_seeker = true`, the noise level is
   found automatically by bisection; otherwise the globally configured noise level is
   used.

### Step 4 — Use the GUI

```powershell
python gui.py
```

Load any `.toml` config, edit all parameters in a tabbed interface, save, and launch
`main.py` with that config directly from the GUI. Progress and log output are shown
live in the GUI. See [§11 GUI](#11-gui).

---

## 4. Source files

```
signal-processing-simulation/
├── sim/                      ← simulation package
│   ├── baseband.py           ← multi-modulation RRC baseband generation
│   ├── modulation.py         ← constellation definitions (all 9 modulations)
│   ├── config.py             ← TOML loader
│   ├── filters.py            ← RRC, OLA convolution, upsample/downsample, channel impairment
│   ├── nonlinear_amplifier.py← memoryless AM-AM + AM-PM model
│   ├── plots.py              ← all visualisation, sweep report, detector results table
│   ├── receiver.py           ← matched filter, decisions, BER (phase-ambiguity resolved), EVM
│   ├── simulation.py         ← full wideband signal chain, per-carrier metric extraction
│   ├── sweep.py              ← 2-D IBO × noise sweep
│   ├── theory.py             ← closed-form BER curves and numerical Eb/N0 inverse
│   └── targeter.py           ← adaptive BER seeker (bisection over noise_density_dbfs)
├── tests/
│   ├── test_awgn_performance.py  ← BER vs theory (see §12)
│   ├── test_modulations.py       ← constellation + baseband/receive round-trip
│   ├── test_theory.py            ← closed-form BER and Eb/N0 inverse
│   ├── test_filters.py           ← RRC and OLA correctness
│   ├── test_nonlinear_amplifier.py
│   ├── test_main.py              ← end-to-end smoke test
│   ├── test_simulation.py        ← wideband simulation integration
│   └── test_targeter.py          ← BER seeker: unit, convergence, implementation loss
├── docs/
│   ├── GUIDE.md              ← this file
│   ├── simulation_overview.md← execution paths and output files (§13)
│   ├── memory_scaling.md     ← OLA memory analysis (§14)
│   ├── filter_analysis.md    ← filter size justification (§15)
│   ├── toolchain.md          ← toolchain invocations and Windows quirks (§16)
│   ├── channel_impairment.md ← channel transfer function model (§17)
│   └── msk_modulation.md     ← MSK matched-filter implementation (§18)
├── output/                   ← generated files (git-ignored)
├── gui.py                    ← standalone TOML editor + launcher with live progress
├── main.py                   ← CLI entry point
├── simulation.toml           ← configuration
└── pyproject.toml
```

| File | Role |
|---|---|
| `sim/baseband.py` | Generates the complex baseband signal for any supported modulation at native sample rate — RRC pulse shaping, except MSK which uses offset-QPSK half-sine shaping (see §18). Normalised to unit RMS power. |
| `sim/modulation.py` | Constellation definitions, Gray coding, APSK ring ratios, `bits_per_symbol()`. All constellations normalised to unit average power. |
| `sim/filters.py` | RRC coefficients, OLA convolution, OLA upsample/downsample (anti-alias Kaiser sinc), per-carrier channel impairments. |
| `sim/nonlinear_amplifier.py` | Memoryless AM-AM + AM-PM model; piecewise linear interpolation of user-supplied lookup tables. |
| `sim/simulation.py` | Orchestrates the full signal chain. AWGN added after amp. Per-carrier demod controlled by `demod_carriers` set (carriers not in the set contribute to the IM environment but skip the expensive receiver chain). Returns CNR/CIR/CNIR per carrier via projection method. |
| `sim/receiver.py` | `matched_filter`, `receive` (chains filter → sampling → decisions → BER with rotational ambiguity resolution → EVM). Uses `np.real()`/`np.imag()` throughout (Pylance compatible). |
| `sim/sweep.py` | 2-D sweep over IBO × noise; honours `sweep_demod` per carrier. |
| `sim/theory.py` | `ber_awgn(mod, EsN0_dB)` — closed-form BER for BPSK/DBPSK/MSK/QPSK/OQPSK/8PSK/16QAM (returns `None` for APSK). `ebn0_for_ber(mod, target_ber)` — numerical inverse by bisection. |
| `sim/targeter.py` | `seek_ber_noise_level` — adaptive bisection finding `noise_density_dbfs` that achieves a target BER, then computes implementation loss. `seek_all_carriers` — runs the seeker for each carrier that has `enabled=True`, `sweep_demod=True`, `use_seeker=True`. Both accept an optional `progress_callback(frac, msg)`. |
| `sim/plots.py` | Wideband PSD (capped at 16384-point FFT), amplifier curves, channel response, sweep plots, `write_sweep_report` (markdown), `write_detector_results` (markdown table of BER/Eb/N0/implementation loss per carrier). |

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
| `noise_density_dbfs` | float (dBFS/Hz) | One-sided AWGN PSD added **after** the amplifier. Total noise power = 10^(N₀/10) × sample_rate. Remove to disable noise. Used as the fixed noise level for carriers with `sweep_demod=true` and `use_seeker=false`. |

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
| `detector_results` | Filename for the detector-model results table (BER seeker and/or fixed-noise stats). Defaults to `detector_results.md` if omitted. |

### `[sweep]`

Remove this section to disable the sweep. Both keys required to trigger a run.

| Key | Description |
|---|---|
| `ibo_db` | List of IBO values (dB) to sweep. |
| `noise_density_dbfs` | List of noise densities (dBFS/Hz) to sweep. |

### `[[carrier]]` (repeated block, one per carrier)

| Key | Type | Default | Description |
|---|---|---|---|
| `name` | string | — | Label used in plots and console output. |
| `modulation` | string | `"BPSK"` | One of: `BPSK`, `DBPSK`, `MSK`, `QPSK`, `OQPSK`, `8PSK`, `16QAM`, `16APSK`, `32APSK`. |
| `symbol_rate` | int (Hz) | — | Symbol rate. Native sample rate = `sps × symbol_rate`. |
| `sps` | int | — | Samples per symbol at native rate. |
| `rolloff` | float | — | RRC rolloff factor (0–1). Occupied BW ≈ `(1+rolloff) × symbol_rate`. |
| `filter_span` | int | — | RRC filter half-span in symbols (TX and RX share the same value). Total taps = `filter_span × sps + 1`. |
| `num_symbols` | int | — | Symbols to generate. Controls BER statistics and memory. |
| `power_db` | float (dB) | — | Carrier power relative to 0 dB. Controls inter-carrier ratio before the amp; the composite is peak-normalised before the NL. |
| `freq` | int (Hz) | — | Centre frequency in the wideband spectrum. Carriers must not overlap. |
| `enabled` | bool | `true` | If `false`, the carrier is excluded from the wideband composite entirely (no signal, no IM contribution). Useful for quickly disabling a carrier without removing it from the config. |
| `sweep_demod` | bool | `false` | If `true`, this carrier is downsampled, demodulated, and included in the detector-results table. If `false`, it contributes to the IM environment but its BER/EVM are not computed. |
| `use_seeker` | bool | `false` | If `true` (and `sweep_demod=true` and `enabled=true`), run the adaptive BER seeker to find the noise level that achieves the target BER. If `false`, use the global `noise_density_dbfs` and report statistics at that operating point. |

### `[carrier.seeker]` (optional, per carrier — only relevant when `use_seeker = true`)

| Key | Type | Default | Description |
|---|---|---|---|
| `target_ber` | float | `0.01` | Target bit-error rate the seeker bisects toward. |
| `confidence` | float | `0.95` | Confidence level for the normal-approximation CI on the final BER estimate (e.g., 0.95 = 95% CI). |
| `ber_accuracy` | float | `0.005` | Half-width of the CI in absolute BER. Determines the final bit count: `N ≥ (z/accuracy)² × p(1−p)`. |
| `noise_lo_dbfs` | float (dBFS/Hz) | `−160.0` | Quietest bracket endpoint. BER at this noise level must be below `target_ber`. |
| `noise_hi_dbfs` | float (dBFS/Hz) | `−80.0` | Loudest bracket endpoint. BER at this noise level must be above `target_ber`. |

If any `[carrier.seeker]` key is absent, the global default (shown above) is used.
If the entire `[carrier.seeker]` table is absent, all global defaults apply.

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
| `output/wideband.png` | Composite wideband PSD (pre-NL, post-NL, post-NL+noise). Spectral regrowth from the NL amplifier is visible as a raised floor between carriers. |
| `output/amplifier_nl.png` | AM-AM and AM-PM curves with peak operating point (red marker). |
| `output/channel_<name>.png` | Amplitude ripple and phase nonlinearity across each carrier's passband. |
| `output/sweep_results.png` | Per-carrier rows: BER vs IBO, EVM vs IBO, CNR/CIR/CNIR vs IBO. Multiple noise levels colour-coded. |
| `output/sweep_table.md` | Markdown report: configuration summary, performance summary (min/max ranges), full IBO × noise grid table. |
| `output/detector_results.md` | Per-carrier detector-model results: BER, 95% CI, effective Eb/N0, theory Eb/N0, implementation loss, CNR/CIR/CNIR, EVM, mode (seeker or fixed), and total bits used. Written whenever any carrier has `sweep_demod = true`. |

---

## 7. Example results

### Console output (with progress indicators)

```
[  0%] Loading configuration...
[  5%] Running wideband simulation (2 carriers, 2 demodulated)...
[ 15%] Wideband simulation complete.
---------------------------------------------------------------
Carrier     CNR (dB)  CIR (dB)  CNIR (dB)  EVM (%)          BER
---------------------------------------------------------------
slow            78.9      48.1       48.1     4.84            0
fast            65.8      40.8       40.8     3.33            0
---------------------------------------------------------------
[ 17%] Saving wideband PSD plot...
[ 20%] Plots saved.
[100%] Done.
```

Each line beginning with `[NNN%]` marks a significant milestone. The GUI uses these
percentages to drive the progress bar; a terminal user sees them as plain-text status.

**Interpreting the metrics table:**

- **CNR 79/66 dB** — Noise density of −160 dBFS/Hz is far below the carrier power;
  thermal noise is negligible. Raise `noise_density_dbfs` toward −140 dBFS/Hz to
  bring CNR into the picture.
- **CIR 48/41 dB** — At 3 dB IBO the amp is moderately backed off; measurable but
  not severe IM distortion. Reduce IBO to 1 dB to drop CIR sharply.
- **CNIR ≈ CIR** — Distortion-limited regime; noise is not yet a factor.
- **EVM 4.8/3.3%** — Includes NL distortion and channel impairments.
- **BER = 0** — No errors in the simulated symbols. Drive harder or increase noise.

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

## 9. BER seeker and implementation loss

The BER seeker answers the question: *at a given IBO and carrier plan, what noise
level produces a specified target BER for a given carrier?* Once that noise level is
found, the tool computes implementation loss — the gap between the effective Eb/N0
measured in the simulation and the theoretical Eb/N0 required for the same BER in a
pure AWGN channel. Implementation loss captures all system impairments combined:
nonlinear distortion, inter-carrier IM products, channel ripple, and non-ideal
receiver filtering.

### How the seeker works

The bisection variable is `noise_density_dbfs`. Higher noise density → higher BER.
The seeker:

1. **Bracket check** — evaluates BER at `noise_lo_dbfs` (quiet, low BER) and
   `noise_hi_dbfs` (loud, high BER) to confirm the target BER lies between them.
   Raises `ValueError` if the bracket is invalid; adjust the bounds accordingly.

2. **Adaptive bisection** — halves the bracket repeatedly. The bit count used per
   step starts at `max(500, n_bits_final // 32)` and doubles every two iterations,
   giving coarse-but-fast early steps and fine-but-accurate later steps. Stops when
   the bracket width is less than 0.05 dB (much finer than any physically meaningful
   distinction).

3. **Final pooled measurement** — runs `n_final_seeds` independent seeds at the
   converged noise level, pooling their bit counts to produce a statistically
   significant BER estimate and a normal-approximation confidence interval.
   The bit count is:
   ```
   N ≥ (z / ber_accuracy)² × target_ber × (1 − target_ber)
   ```
   where `z = √2 × erfinv(confidence)`. For `target_ber=0.01`, `confidence=0.95`,
   `ber_accuracy=0.005` this yields approximately N = 1,500 bits.

### Effective Eb/N0 and implementation loss

CNIR is measured at the native sample rate (sps samples per symbol), *before* the
matched filter. The matched filter provides sps-fold processing gain, so:

```
Effective Eb/N0 (dB) = CNIR_dB + 10 · log₁₀(sps / bps)
```

where `bps` is bits per symbol for the carrier's modulation. For BPSK with `sps=4`
this adds +6 dB; for 16QAM (bps=4) with `sps=4` it adds 0 dB.

Implementation loss is then:

```
Implementation loss (dB) = Effective Eb/N0 − Theory Eb/N0(at measured BER)
```

A linear amplifier with a single carrier and no channel impairments should produce
implementation loss near 0 dB. A nonlinear operating point or multi-carrier loading
will produce positive implementation loss — the system needs more signal power than
theory predicts to achieve the same BER.

Implementation loss is `None` for 16APSK and 32APSK because no closed-form BER
formula exists for those modulations.

### Configuration example

```toml
[[carrier]]
name        = "slow"
modulation  = "BPSK"
symbol_rate = 100_000
sps         = 4
rolloff     = 0.35
filter_span = 8
num_symbols = 5_000
power_db    = 0.0
freq        = -500_000
enabled     = true
sweep_demod = true
use_seeker  = true

[carrier.seeker]
target_ber    = 0.001
confidence    = 0.95
ber_accuracy  = 0.0005
noise_lo_dbfs = -120.0
noise_hi_dbfs = -80.0
```

### Progress output during a seeker run

```
[ 25%] Starting BER seeker for 1 carrier(s)...
[ 28%] [seeker] 'slow' — bracket check (lo: -120.0 dBFS)...
[ 30%] [seeker] 'slow' — bracket check (hi: -80.0 dBFS)...
[ 34%] [seeker] 'slow' — step 1/20  BER 2.14e-01 → target 1.00e-03  noise -100.00 dBFS
[ 39%] [seeker] 'slow' — step 2/20  BER 4.32e-04 → target 1.00e-03  noise -110.00 dBFS
...
[ 88%] [seeker] 'slow' — final measurement (5 seeds)...
[ 99%] [seeker] 'slow' — done.  BER=9.87e-04  IL=0.12 dB
[100%] Done.
```

### Detector results table

Results are written to `output/detector_results.md`:

| Carrier | Mode | Noise (dBFS/Hz) | BER | BER CI | Eff Eb/N0 (dB) | Theory Eb/N0 (dB) | Impl Loss (dB) | CNR (dB) | CIR (dB) | CNIR (dB) | EVM (%) | Bits |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| slow | seeker | -104.32 | 9.87e-04 | [8.1e-04, 1.16e-03] | 9.81 | 9.69 | 0.12 | 78.4 | 48.1 | 46.7 | 4.8 | 7,500 |
| fast | fixed | -160.00 | 0 | — | 62.3 | — | — | 65.8 | 40.8 | 40.8 | 3.3 | — |

- **Mode = seeker** — noise level was found by bisection; CI and bit count are reported.
- **Mode = fixed** — the global `noise_density_dbfs` was used; statistics come from the
  main wideband simulation run.
- **Impl Loss = —** — BER was zero or the modulation has no closed-form theory.

### Noise bracket calibration

The bracket `[noise_lo_dbfs, noise_hi_dbfs]` must straddle the target BER:
BER at `noise_lo` must be *below* the target; BER at `noise_hi` must be *above* it.
A bracket check is always run first, and a `ValueError` is raised with a descriptive
message if either bound is wrong.

To estimate appropriate bounds: the target BER point is typically within 20–40 dB of
the noise floor where CNR → ∞. Start with `noise_lo = −160` and `noise_hi = −80`; if
the check fails, read the error message — it tells you which endpoint is wrong and in
which direction to move it. The GUI exposes both fields per carrier in the seeker
parameters panel.

---

## 10. Adding or modifying carriers

**Constraints:**

1. **Integer upsample factor** — `sample_rate` must be an exact integer multiple of
   `sps × symbol_rate`. E.g., at 2 GHz, `symbol_rate=5_000_000` with `sps=10`
   gives native rate 50 MHz and factor 40 (valid). `symbol_rate=3_000_000` gives
   factor 66.67 (invalid → `ValueError`).

2. **No spectral overlap** — each carrier occupies roughly
   `[freq − (1+rolloff)·sr/2, freq + (1+rolloff)·sr/2]`. Check all pairs.

3. **Within wideband bandwidth** — all carriers must fit within
   `[−sample_rate/2, +sample_rate/2]`.

### Example: adding a 16QAM carrier with the BER seeker

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
enabled     = true
sweep_demod = true
use_seeker  = true

[carrier.seeker]
target_ber    = 0.001
confidence    = 0.95
ber_accuracy  = 0.0005
noise_lo_dbfs = -120.0
noise_hi_dbfs = -80.0

[carrier.channel]
enabled = false
```

---

## 11. GUI

`gui.py` is a standalone tkinter application. It does not import any `sim/` modules —
it reads and writes `.toml` files directly and launches `main.py` as a subprocess.

### Layout

```
┌──────────────────────────────────────────────────────────────┐
│ File: [path/to/config.toml]  [Open…] [Save] [Save As…]  |  [▶ Launch Simulation] │
├──────────────────────────────────────────────────────────────┤
│  [General]  [Amplifier]  [Sweep & Output]  [Carriers]        │ ← Notebook tabs
│                                                              │
│  (tab contents scroll independently)                         │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│ ████████████████████░░░░░░░░░░░░░░░░░░░░  (progress bar)    │
│ [  25%] Starting BER seeker for 2 carrier(s)...             │ ← scrolling log
│ [  30%] [seeker] 'slow' — bracket check (lo: -120.0 dBFS)  │   (4 lines, dark)
│ [  35%] [seeker] 'slow' — step 1/20  BER 2.1e-01 → target  │
├──────────────────────────────────────────────────────────────┤
│ Running simulation...                         (status bar)  │
└──────────────────────────────────────────────────────────────┘
```

### Tabs

| Tab | Contents |
|---|---|
| **General** | Simulation seed, wideband sample rate and noise density, OLA filter span and block size |
| **Amplifier** | Input backoff (dB), AM-AM table (input/output amplitude columns), AM-PM table (input/phase columns) |
| **Sweep & Output** | IBO sweep list, noise sweep list, output directory (with Browse button), filenames for all output files including `detector_results` |
| **Carriers** | One scrollable labeled frame per carrier (see below); view-filter dropdown at the top |

### Carriers tab controls

**View dropdown** — selects which carriers are shown. Choose "All" to see every
carrier at once, or pick a specific carrier name to show only that frame. Use this
when you have many carriers and want to focus on one without scrolling. The dropdown
updates automatically when carriers are added or removed.

**Per-carrier frame** — each carrier has:

- **Name, Modulation, Symbol Rate, SPS, Roll-off, Filter Span, Num Symbols, Power (dB),
  Freq (Hz)** — the basic carrier parameters, arranged in a two-column grid.

- **Include in wideband** checkbox (`enabled`) — when unchecked, the carrier is
  excluded from the simulation entirely. It does not appear in the composite signal and
  contributes no IM products.

- **Enable detector model** checkbox (`sweep_demod`) — when checked, the carrier is
  downsampled, demodulated, and included in the detector-results table after the run.
  When unchecked, the carrier contributes to the wideband IM environment but its
  BER/EVM are not computed.

- **Mode radio buttons** — visible only when *both* enables are checked:
  - **Fixed noise level** — use the global `noise_density_dbfs` from the General tab.
    The carrier is demodulated in the main wideband run, and all statistics (BER, CNR,
    CIR, CNIR, effective Eb/N0, implementation loss) are reported at that operating point.
  - **BER seeker** — the adaptive bisection algorithm finds the noise level that
    achieves the target BER. Seeker parameter fields appear below the radio buttons.

- **BER Seeker Parameters** panel — visible only in seeker mode:
  - Target BER, Confidence, BER Accuracy
  - Noise Lo (dBFS/Hz), Noise Hi (dBFS/Hz)

- **Channel impairments** checkbox — when checked, expands fields for amplitude
  ripple, ripple cycles, phase nonlinearity, poly order, and an optional plot filename.

- **Remove** button — removes the carrier from the config.

### Progress bar and log

When the simulation is running:

- The **▶ Launch Simulation** button is disabled to prevent concurrent runs.
- The **progress bar** advances as `main.py` emits `[NNN%]` lines to stdout.
  The GUI reads these in real time and sets the bar's position accordingly.
- The **scrolling log** (4-line dark panel) shows the last few output lines from
  `main.py`, auto-scrolling to the newest entry. All output — metrics tables, seeker
  step updates, error messages — appears here.
- When the run finishes, the button re-enables and the status bar shows either
  "Simulation complete." or the exit code if it failed.

The subprocess is launched with Python's `-u` flag (unbuffered stdout) and
`stderr=STDOUT` so all output flows through the same pipe. A daemon background
thread reads lines continuously into a queue; the main thread drains the queue
every 100 ms via `root.after()`, ensuring the GUI stays responsive.

### Saving and launching

The **Save** button (and **▶ Launch Simulation**) serialize all GUI fields to TOML
and write them to the currently loaded file path before launching `main.py`. This
guarantees that what you see in the GUI is exactly what `main.py` receives — no
separate "apply" step is needed.

---

## 12. Test suite

Run tests with coverage (Windows PowerShell):

```powershell
.venv\Scripts\python.exe -m pytest tests/ -v --cov=sim --cov=main --cov-report=term-missing
```

Coverage is reported inline with the test run — no separate invocation is needed.
Use the explicit `.venv\Scripts\` path; unqualified `python` hits the Windows Store
stub and breaks the run. See [toolchain.md](toolchain.md) for Linux/macOS paths and
the full explanation.

### `tests/test_awgn_performance.py`

The primary validation suite for the modulation and receiver chain. All tests operate
in an isolated AWGN channel (no nonlinear amplifier) to verify the baseband and
receiver modules independently.

| Test | What it checks | Why it matters |
|---|---|---|
| `test_ber_monotone[MOD]` | BER strictly decreases as Eb/N0 increases over 5 points | Catches sign errors, inverted noise, or sampling timing bugs |
| `test_ber_matches_theory[MOD]` | Measured BER is within 2× of theory at one mid-range SNR point | Confirms the noise model and symbol count are calibrated |
| `test_ber_theory_table` | Interpolation-based Eb/N0 vs BER comparison across 5 target BER levels for 7 modulations; writes `tests/plots/performance/theory_comparison.md` | Quantifies implementation loss across the full BER range; catches systematic offsets |
| `test_generate_performance_plots` | Generates BER-vs-Eb/N0 and EVM-vs-Eb/N0 plots with 1/2/3σ uncertainty bands; writes PNGs to `tests/plots/performance/` | Visual regression reference; sigma bands show where statistical confidence is meaningful |

**Theory formulas used:**

| Modulation | BER formula | Notes |
|---|---|---|
| BPSK, QPSK, OQPSK, MSK | `0.5 · erfc(√(Eb/N0))` | Same formula; curves are identical. MSK detail in §18 |
| DBPSK | `2p(1−p)`, `p = 0.5·erfc(√(Eb/N0))` | Coherent detection + differential decoding; NOT the differentially-coherent `0.5·exp(−Eb/N0)` |
| 8PSK | `(1/3)·erfc(√(3·Eb/N0)·sin(π/8))` | Approximate for Gray-coded 8PSK |
| 16QAM | `(3/8)·erfc(√(2·Eb/N0/5))` | Standard rectangular 16QAM |
| 16APSK, 32APSK | No closed form | Only monotonicity tested |

**Symbol count and confidence:**

Both `test_ber_theory_table` and `test_generate_performance_plots` use the same
`_N_BITS_PLOT` constant with `n_sym = _N_BITS_PLOT // bps` per modulation, ensuring
equal statistical confidence across all modulations. Current default: `10_000` bits
(≈4 s total). For a rigorous run set `_N_BITS_PLOT = 1_000_000`; this gives ±0.001
BER accuracy at 95% confidence.

### `tests/test_modulations.py`

Unit and round-trip tests for `sim/modulation.py`, `sim/baseband.py`, and
`sim/receiver.py`. Confirms every constellation has unit average power and
`2^bits_per_symbol` points; that `map_bits` → `decide` recovers the original bits
with no noise; and that the full baseband → receive chain yields zero BER and low
EVM in the absence of noise and nonlinearity. Also covers DBPSK differential
encode/decode (including 180° phase immunity), MSK phase-ambiguity correction,
the `ber is None` path when no reference bits are supplied, and error paths
(unknown modulation, non-integer `sps`).

### `tests/test_theory.py`

Unit tests for the closed-form BER module `sim/theory.py`. Checks that `ber_awgn`
returns sensible in-range values for each modulation family, is monotonically
decreasing in Es/N0, ranks DBPSK above coherent BPSK, and returns `None` for the
APSK formats that have no closed form. Verifies that `ebn0_for_ber` numerically
inverts `ber_awgn`, and returns `None` when the target BER is unreachable within
the search bracket.

### `tests/test_targeter.py`

Validates the BER seeker in `sim/targeter.py`.

| Test | What it checks |
|---|---|
| `test_n_bits_for_ci_sanity` | `_n_bits_for_ci()` formula: correct order for increasing confidence, accuracy, and BER |
| `test_seek_ber_convergence` | Bisection reaches a noise level where measured BER is within ±0.08 of target; all result keys present; CI is well-ordered |
| `test_linear_amplifier_zero_implementation_loss` | With a pass-through (linear) amplifier and single carrier, implementation loss is within ±3 dB of zero |
| `test_invalid_bracket_lo_too_noisy` | Raises `ValueError` mentioning `noise_lo_dbfs` when the quiet endpoint already exceeds target BER |
| `test_invalid_bracket_hi_too_quiet` | Raises `ValueError` mentioning `noise_hi_dbfs` when even the loud endpoint is below target BER |

### `tests/test_filters.py`

Verifies RRC filter properties (Nyquist criterion: zero ISI at symbol samples),
OLA convolution accuracy (result matches `np.convolve`), and channel impairment
transfer functions.

### `tests/test_nonlinear_amplifier.py`

Verifies AM-AM and AM-PM table lookup, interpolation at breakpoints, saturation
behaviour, and that a linear AM-AM table produces no phase distortion.

### `tests/test_main.py`

End-to-end smoke test: mocks `load_config` with a minimal two-carrier config, runs
`main()`, and asserts the expected output PNGs are written.

### `tests/test_simulation.py`

Integration tests on the full `wideband_bpsk_simulation` function: checks that CNR
varies correctly with noise density, CIR varies with IBO, and that disabling
`demod_carriers` returns NaN placeholders without affecting the wideband signal.

---

## 13. Simulation overview

**→ [simulation_overview.md](simulation_overview.md)**

A top-down reference for understanding what the simulator does on every run, which
code paths activate under which conditions, and exactly what output files each path
produces.

The document opens with an ASCII architecture diagram showing the always-on core and
the three independent optional paths branching from it. It then walks through the
wideband signal chain step by step — bit generation, modulation, pulse shaping,
optional channel impairment, OLA upsampling, frequency shifting, composite formation,
nonlinear amplification, noise injection, and per-carrier extraction — before
explaining how the C/N/I decomposition separates distortion from noise.

The three execution paths are each documented in their own section:

- **Path A — fixed-noise demodulation** (`sweep_demod = true`, `use_seeker = false`):
  demodulates the target carrier at the globally configured noise level from the main
  wideband run, then computes effective Eb/N0 and implementation loss. Activates with
  no extra configuration beyond enabling demodulation on a carrier.

- **Path B — parameter sweep** (`[sweep]` section present with both `ibo_db` and
  `noise_density_dbfs` arrays): re-runs the full wideband simulation at every point on
  the IBO × noise grid and collects per-carrier BER, EVM, CNR, CIR, and CNIR.
  Completely independent of the seeker path.

- **Path C — BER seeker** (`sweep_demod = true`, `use_seeker = true`): bisects over
  noise density to find the operating point that achieves a user-specified target BER,
  then reports a confidence-interval-bounded BER estimate and implementation loss.
  Completely independent of the sweep path.

The document closes with a concise output-file table cross-referencing each file to
the path that generates it, and worked example TOML snippets covering five common
scenarios from a minimal wideband-PSD-only run through a combined sweep + seeker
configuration.

---

## 14. Memory scaling

**→ [memory_scaling.md](memory_scaling.md)**

Analyses where memory goes in the simulation and how it scales with the two
configuration dimensions that matter most: the upsample factor L (ratio of wideband
sample rate to carrier native rate) and the simulation duration T (governed by the
longest carrier's `num_symbols / symbol_rate`).

The key result is that OLA chunk processing decouples FFT working memory from signal
length. The per-block FFT buffer — the dominant working allocation — is sized by the
filter length, which scales with L but is reused for every block regardless of how
many symbols are simulated. By contrast, the persistent wideband arrays (`wideband`,
`wideband_nl`, `wideband_noisy`, and intermediate OLA outputs) all scale with
`T × sample_rate` and are the true memory cost of a long simulation.

The document includes a worked example for the default configuration (~11 MB peak),
a scaling table across seven orders of magnitude of carrier symbol rate, and a
concrete demonstration of what happens to OLA efficiency as L grows (from 50% at
L = 10 to 0.05% at L = 200,000). For very narrowband carriers the document shows
that raising `block_size` is more effective than any other tuning lever — it recovers
OLA efficiency without changing the filter or the output.

---

## 15. Filter analysis

**→ [filter_analysis.md](filter_analysis.md)**

Justifies every filter size in the signal chain and verifies that none is undersized
for the default carrier geometry.

Three filters are examined:

- **RRC pulse-shaping filter** (`filter_span × sps + 1` taps, applied at native rate
  at both TX and RX): at `filter_span = 10`, `sps = 10` this gives ±5 symbol periods,
  comfortably above the ±4T practical minimum for `rolloff = 0.35`. The document also
  confirms that the Kaiser sinc passband is 7.4× wider than the RRC signal bandwidth,
  so upsampling leaves the pulse shape intact.

- **Channel impairment filter** (full-block frequency-domain multiplication at native
  rate): no conventional tap count — the response is defined analytically. The
  document identifies and documents the fix for a circular-convolution wrap-around
  bug: the amplitude ripple cosine has delay taps at `±ripple_cycles / signal_bw`
  seconds; without zero-padding those taps corrupt ~1.5 symbols at each signal edge.
  The fix zero-pads by the delay extent plus 8 samples of sidelobe margin, making the
  DFT multiplication exactly equivalent to linear convolution.

- **Kaiser-windowed sinc upsampling/downsampling filter** (`2 × ola_filter_span × L + 1`
  taps, applied at wideband rate inside the OLA engine): the document derives the
  minimum tap count for 80 dB stopband attenuation (~10L taps) and shows the default
  `ola_filter_span = 16` gives 32L + 1 taps — 3.2× the minimum, yielding 120–140 dB
  realised stopband. It also confirms that the nearest IMD products after downconversion
  are 30 × f_s into the stopband, far beyond where even the minimum filter would matter.

The document concludes with guidance on when filter sizes would need to increase:
lower RRC rolloff (below ~0.25), very close carrier spacing (sidebands within one
`f_s/2` of each other), or extreme ripple-cycle counts in the channel impairment model.

---

## 16. Toolchain

**→ [toolchain.md](toolchain.md)**

Documents the correct way to invoke pytest, pyright, and ruff on each platform, and
explains the Windows-specific complications discovered during development.

The key points: pyright and ruff are compiled binary executables installed into
`.venv/Scripts/` (Windows) or `.venv/bin/` (Linux/macOS) by `uv sync`. They cannot
be invoked as `python -m pyright` or `python -m ruff`. On Windows, the unqualified
`python` command is intercepted by the Windows Store App Execution Alias when no
global Python is installed, which breaks any tool that calls `python` internally to
discover the active environment.

`pyrightconfig.json` at the repo root works around this by pointing pyright directly
at the `.venv/` directory, making it independent of PATH. It is required on Windows
and harmless elsewhere.

The document also covers how to add or remove dev tools via `pyproject.toml` and
`uv sync`, and lists the platform-specific binary paths for all three tools.

---

## 17. Channel impairment model

**→ [channel_impairment.md](channel_impairment.md)**

Documents `apply_channel_impairment` in `sim/filters.py`: the transfer function
H(f) it constructs, why unity gain outside `signal_bw` is the correct choice
(not a limitation), the baseband-equivalent representation that makes the model
valid for any LTI passband filter, and the two genuine constraints of the
current implementation.

The key points:

- H(f) is defined as amplitude ripple × phase nonlinearity inside
  `|f| ≤ signal_bw / 2`, and unity elsewhere. The in-band region is normalised
  to `f_norm ∈ [−1, +1]` so that `ripple_cycles` and `phase_poly_order` are
  independent of the carrier symbol rate.

- Unity outside `signal_bw` is intentional: the RRC transmit filter already
  suppresses out-of-band power by 40–60 dB, and real transponder filters do not
  distort outside their allocated slot.

- The hard step at `|f| = signal_bw / 2` creates sinc ringing in the time
  domain (magnitude ≈ `r ≈ 0.028` for 0.5 dB ripple, or about −31 dBc). This
  is negligible for any practically configured ripple depth.

- Each carrier's impairment is applied independently at its own baseband rate.
  A wideband filter that couples multiple carriers is not modelled.
