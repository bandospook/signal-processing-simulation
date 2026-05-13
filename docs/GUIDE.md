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

---

## 1. What this tool does

This simulation models a satellite or RF ground-station scenario where multiple BPSK carriers — potentially at very different symbol rates — share a single power amplifier. It lets you answer questions like:

- How much intermodulation distortion (IMD) does the amplifier inject into each carrier at a given input backoff?
- What is the carrier-to-interference ratio (CIR) and how does it compare to the carrier-to-noise ratio (CNR)?
- At what IBO does BER become unacceptable for each carrier?
- How does a passband channel impairment (amplitude ripple, phase nonlinearity) interact with the NL distortion?

The simulation is fully deterministic (fixed seed), TOML-configured, and produces both console tables and PNG plots. No code changes are needed to explore new operating points.

---

## 2. Signal chain

```
 Per carrier (native rate = sps × symbol_rate)
 ─────────────────────────────────────────────
  Random BPSK symbols (+1 / −1)
        │
        ▼  RRC transmit filter  (bpsk.py)
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
        ▼  Add wideband AWGN  (optional)

 Per carrier — extraction and receive
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
        ▼  Hard BPSK decisions  →  BER
        │
        ▼  Metrics: EVM, CNR, CIR, CNIR  (simulation.py / receiver.py)
```

**CNR / CIR / CNIR computation**

Three separate extractions are performed per carrier: from the NL input, the noiseless NL output, and the noisy NL output. A complex projection separates the deterministic AM-AM/AM-PM effect from the residual in-band IM distortion:

```
α          = ⟨bb_rx, nl_pure⟩ / ⟨bb_rx, bb_rx⟩   (complex gain of desired component)
signal     = α · bb_rx                              (desired part of nl_pure)
distortion = nl_pure − signal                       (true IM products)
noise      = nl_noisy − nl_pure                     (additive noise)

CIR  (dB)  = 10 log₁₀( P_signal / P_distortion )
CNR  (dB)  = 10 log₁₀( P_signal / P_noise )
CNIR (dB)  = 10 log₁₀( P_signal / (P_distortion + P_noise) )
```

This correctly attributes AM-AM compression as a gain change rather than distortion, and is independent of absolute amplitude scaling.

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

Your prompt will change to show `(signal-processing-simulation)`.

### Step 3 — Run with the default configuration

```powershell
python main.py
```

This will:
1. Print a metrics table to the console.
2. Open interactive matplotlib windows showing the wideband PSD, amplifier curves, and channel responses.
3. Save PNG files into the `output/` directory (created automatically; configurable via `output_dir` in `[output]`).
4. If `[sweep]` is present in `simulation.toml`, run the full IBO × noise grid and save `sweep_results.png`.

### Step 4 — Adjust the operating point

Open `simulation.toml` and change `input_backoff_db` under `[amplifier]` to a lower value (e.g., `1.0`) to drive the amplifier harder and observe increased distortion. Re-run `python main.py`.

### Step 5 — Disable the sweep for faster iteration

Comment out or remove the `[sweep]` block in `simulation.toml`. The single-point run completes in a few seconds; the full 5 × 4 sweep takes a couple of minutes.

---

## 4. Source files

```
signal-processing-simulation/
├── sim/                      ← all simulation modules (Python package)
│   ├── bpsk.py
│   ├── config.py
│   ├── filters.py
│   ├── nonlinear_amplifier.py
│   ├── plots.py
│   ├── receiver.py
│   ├── simulation.py
│   └── sweep.py
├── docs/
│   └── GUIDE.md              ← this file
├── output/                   ← generated PNG files (created on first run)
├── main.py                   ← entry point
├── simulation.toml           ← configuration
└── pyproject.toml
```

| File | Role |
|---|---|
| `simulation.toml` | All simulation parameters — edit this to change scenarios |
| `main.py` | Entry point: loads config, runs simulation, calls all plots |
| `sim/config.py` | TOML loader (thin wrapper around `tomllib`) |
| `sim/bpsk.py` | Generates RRC-filtered BPSK baseband signal at native rate |
| `sim/filters.py` | RRC coefficients, OLA convolution, OLA upsample/downsample, channel impairment |
| `sim/nonlinear_amplifier.py` | Memoryless AM-AM + AM-PM model using interpolated lookup tables |
| `sim/simulation.py` | Orchestrates the full wideband signal chain and per-carrier metric extraction |
| `sim/receiver.py` | Matched filter, symbol sampling, BPSK decisions, BER, EVM |
| `sim/sweep.py` | 2-D parameter sweep over IBO × noise; calls `wideband_bpsk_simulation` repeatedly |
| `sim/plots.py` | All visualisation: wideband PSD, amplifier curves, channel response, metrics table, sweep plots |

### `sim/bpsk.py`

Generates a complex baseband BPSK signal filtered through a root raised-cosine (RRC) transmit filter. Output is normalised to unit RMS power. Returns the signal, a time axis, and the underlying symbol sequence (used later for BER comparison).

### `sim/filters.py`

Four public functions:

- **`rrc_coeffs`** — computes RRC filter taps for a given rolloff factor and samples-per-symbol. Both transmitter and receiver use this with the same parameters, so the combined response is a raised cosine with controlled ISI-free sample points.
- **`ola_convolve`** — linear convolution via the FFT overlap-and-add algorithm. Used internally for efficiency when the signal is much longer than the filter.
- **`fft_ola_upsample`** — inserts zeros then applies a Kaiser-windowed sinc anti-imaging filter. Upsamples from native to wideband rate.
- **`fft_ola_downsample`** — applies a Kaiser-windowed sinc anti-alias filter (cutoff at new Nyquist, ≈80 dB stopband) then decimates. Prevents aliasing when extracting a narrowband carrier from the wideband composite.
- **`apply_channel_impairment`** — applies per-carrier amplitude ripple and phase nonlinearity in the frequency domain before upsampling.

### `sim/nonlinear_amplifier.py`

A memoryless (no memory, instantaneous) nonlinear model. The input complex envelope is split into amplitude and phase. Amplitude is mapped through the AM-AM table (piecewise linear interpolation). A phase shift read from the AM-PM table is added. The model is parameterised entirely by the two lookup tables in `simulation.toml`.

### `sim/simulation.py`

The main computation loop. Calls all the above modules in order, then performs three OLA downsampling extractions per carrier (pre-NL reference, post-NL noiseless, post-NL+noise) and computes CNR/CIR/CNIR via the projection method described in §2. Returns a results dict consumed by `main.py`.

### `sim/receiver.py`

Implements the digital receive chain independently of the simulation waveform generation:

- **`matched_filter`** — RRC filter (same coefficients as transmitter), group delay stripped.
- **`symbol_sample`** — decimates to 1 sample per symbol at the correct timing phase.
- **`bpsk_decide`** — hard decision on real part; returns ±1 array.
- **`measure_ber`** — compares decisions against reference symbols, resolving the inherent BPSK 0/π phase ambiguity by testing both polarities.
- **`measure_evm_rms`** — normalises received samples to unit RMS power, then computes the RMS distance from ideal ±1 constellation points, expressed as a percentage.
- **`bpsk_receive`** — chains all of the above; returns `samples`, `decisions`, `ber`, `evm_rms`.

### `sim/sweep.py`

Runs `wideband_bpsk_simulation` on every point in the Cartesian product of `ibo_db_values × noise_density_dbfs_values`. Prints progress to the console. Returns a flat list of result dicts for `plot_sweep_results`.

---

## 5. Configuration reference

All parameters live in `simulation.toml`. Sections and keys:

### `[simulation]`

| Key | Type | Description |
|---|---|---|
| `seed` | int | Random seed for reproducible symbol sequences and noise. |

### `[wideband]`

| Key | Type | Description |
|---|---|---|
| `sample_rate` | int (Hz) | Common sample rate for the composite signal. Must be an integer multiple of every carrier's native rate (`sps × symbol_rate`). |
| `noise_density_dbfs` | float (dBFS/Hz) | One-sided AWGN spectral density added after the amplifier. Total noise power = 10^(N₀/10) × sample_rate. Remove this key to disable noise entirely. |

### `[amplifier]`

| Key | Type | Description |
|---|---|---|
| `input_backoff_db` | float (dB) | Peak drive level relative to the saturation knee. 0 dB = peak signal reaches amplitude 1.0 on the AM-AM curve (full saturation). 3 dB = peak reaches 0.71 (typical backed-off operation). Higher values = more linear but less efficient. |

### `[amplifier.am_am]`

| Key | Type | Description |
|---|---|---|
| `input` | float list | Input amplitude breakpoints (normalised, 0–1). |
| `output` | float list | Corresponding output amplitudes. Must be the same length as `input`. Piecewise linear interpolation is used between breakpoints. |

### `[amplifier.am_pm]`

| Key | Type | Description |
|---|---|---|
| `input` | float list | Input amplitude breakpoints (normalised, 0–1). |
| `phase_deg` | float list | Phase shift in degrees at each amplitude. Positive = phase lead. |

### `[ola]`

| Key | Type | Default | Description |
|---|---|---|---|
| `filter_span` | int | 16 | Half-span of the Kaiser-sinc interpolation filter in symbols. Larger values give better stopband rejection at the cost of longer computation. |
| `block_size` | int | 4096 | FFT block size for the overlap-and-add convolution. Tune for memory/speed trade-off; must be a power of two for efficiency. |

### `[output]`

| Key | Type | Description |
|---|---|---|
| `output_dir` | string | Directory where all PNG files are written. Created automatically on first run. Defaults to `"output"`. |
| `wideband` | string | Filename (inside `output_dir`) for the wideband PSD + per-carrier baseband PSD figure. |
| `nl_tables` | string | Filename for the AM-AM / AM-PM characteristic plot. |
| `sweep` | string | Filename for the sweep results figure. |

### `[sweep]`

Remove this section entirely to disable the sweep. Both keys must be present and non-empty to trigger a sweep run.

| Key | Type | Description |
|---|---|---|
| `ibo_db` | float list | IBO values (dB) to sweep. |
| `noise_density_dbfs` | float list | Noise density values (dBFS/Hz) to sweep. Every combination is run. |

### `[[carrier]]` (repeat for each carrier)

| Key | Type | Description |
|---|---|---|
| `name` | string | Label used in plots and the console table. |
| `symbol_rate` | int (Hz) | Symbol rate. The carrier's native sample rate = `sps × symbol_rate`. |
| `sps` | int | Samples per symbol at native rate. Must divide evenly into `sample_rate / symbol_rate`. |
| `rolloff` | float | RRC rolloff factor (0–1). Occupied bandwidth ≈ `(1 + rolloff) × symbol_rate`. |
| `filter_span` | int | RRC filter length in symbols (both TX and RX use this). |
| `num_symbols` | int | Number of BPSK symbols to generate. Longer sequences give more stable BER estimates but increase memory. |
| `power_db` | float (dB) | Carrier power relative to the 0 dB reference carrier. Used to set the carrier-to-carrier power ratio before the amplifier. The composite is peak-normalised before the NL, so this controls the *ratio* between carriers, not the absolute level. |
| `freq` | int (Hz) | Centre frequency in the wideband spectrum. Carriers must not overlap: check that `|f_a − f_b| > ((1+rolloff_a)×sr_a + (1+rolloff_b)×sr_b) / 2` for each pair. |

### `[carrier.channel]` (optional, per carrier)

Remove this subsection or set `enabled = false` to bypass channel impairments for that carrier.

| Key | Type | Description |
|---|---|---|
| `enabled` | bool | Master enable. Set to `false` to bypass all impairments without removing the block. |
| `ripple_db` | float (dB) | Peak-to-peak amplitude ripple across the passband. Realised as a cosine: `A(f) = 1 + r·cos(π·cycles·f_norm)`. |
| `ripple_cycles` | float | Number of complete ripple cycles across the signal bandwidth (from −BW/2 to +BW/2). 1.0 = one full cycle, 2.0 = two cycles, etc. |
| `max_phase_dev_deg` | float (°) | Peak phase deviation from linear phase at the band edge. |
| `phase_poly_order` | int | Polynomial order of the phase shape vs normalised frequency. 2 = quadratic (even symmetry). |
| `plot` | string | Filename for the channel impairment response plot for this carrier. |

---

## 6. Output files

Running `python main.py` with the default configuration writes all PNG files into the `output/` directory (configured via `output_dir` in `[output]`; created automatically if it does not exist).

| File | Contents |
|---|---|
| `output/wideband_bpsk.png` | **Top row**: wideband PSD (pre-NL, post-NL, post-NL+noise). Spectral regrowth from the NL amplifier is visible as a raised noise floor between and around the carriers. **Bottom row**: one panel per carrier showing the baseband PSD before and after the NL, so in-band distortion is directly visible. |
| `output/amplifier_nl.png` | AM-AM and AM-PM curves with the peak operating point (determined by IBO) marked in red. A large gap between the AM-AM curve and the ideal linear line indicates significant compression. |
| `output/channel_slow.png` | Amplitude ripple (dB vs MHz) and phase nonlinearity (° vs MHz) across the slow carrier's passband. |
| `output/channel_fast.png` | Same as above for the fast carrier. |
| `output/sweep_results.png` | Three columns per carrier row: BER vs IBO, EVM vs IBO, and CNR/CIR/CNIR vs IBO. Multiple curves (one per noise level) are colour-coded. Points where BER = 0 (no errors observed) are omitted from the log-scale BER plot. |

---

## 7. Example results

The following output is produced by `python main.py` with the default `simulation.toml` (IBO = 3.0 dB, noise density = −160 dBFS/Hz, two carriers).

### Console metrics table

```
---------------------------------------------------------------
Carrier     CNR (dB)  CIR (dB)  CNIR (dB)  EVM (%)          BER
---------------------------------------------------------------
slow            78.9      48.1       48.1     4.84            0
fast            65.8      40.8       40.8     3.33            0
---------------------------------------------------------------
```

**Interpreting these numbers:**

- **CNR 79 / 66 dB** — The noise density of −160 dBFS/Hz is very low relative to the carrier power, so thermal noise is negligible. If you raise `noise_density_dbfs` toward −140 dBFS/Hz, CNR will drop and eventually dominate CNIR.
- **CIR 48 / 41 dB** — At 3 dB IBO the amplifier is not saturating heavily, but in-band IM distortion is already measurable. Reducing IBO to 1 dB will drop CIR noticeably; increasing it to 6 dB will raise it.
- **CNIR ≈ CIR** — Because CNR >> CIR here, distortion is the dominant impairment. CNIR will only diverge significantly from CIR when noise and distortion are of comparable power.
- **EVM 4.8 / 3.3%** — Includes both amplitude/phase distortion from the NL and the channel impairments. The slow carrier sees slightly higher EVM because its channel has larger ripple and phase deviation (0.5 dB / 5°) versus the fast carrier (0.3 dB / 3°).
- **BER = 0** — At this operating point no bit errors occur in 500 / 10 000 symbols. Drive the amplifier harder (IBO = 0–1 dB) or increase noise to push the system into error.

### What to check if results look wrong

| Symptom | Likely cause |
|---|---|
| BER = 0.5 for all IBO values | Timing or phase error — check `sps` and `filter_span` are consistent with symbol generation |
| CIR does not change with IBO | AM-AM table is linear (output ≈ input); add compression by reducing high-amplitude output values |
| CNR unchanged when `noise_density_dbfs` is varied | Noise may be disabled (key absent) or the noise level is far below CIR — look at the raw power difference |
| `ValueError: upsample factor ... is not an integer` | `sample_rate` is not an integer multiple of `sps × symbol_rate` for one of the carriers |
| Wideband PSD shows carrier overlap | Carrier centre frequencies are too close; increase separation or reduce rolloff |

---

## 8. Sweep mode

The sweep runs the full simulation on every (IBO, noise) pair in the Cartesian product of the two lists in `[sweep]`. Progress is printed to the console:

```
Running sweep: 5 IBO × 4 noise = 20 points …
  [ 1/20] IBO=0.0 dB  noise=-180.0 dBFS/Hz  done
  [ 2/20] IBO=0.0 dB  noise=-165.0 dBFS/Hz  done
  ...
  [20/20] IBO=6.0 dB  noise=-140.0 dBFS/Hz  done
```

**Sweep plot layout** (`sweep_results.png`):

- One row of subplots per carrier.
- Column 1: BER vs IBO (log scale). Points where BER = 0 are omitted.
- Column 2: EVM (%) vs IBO.
- Column 3: CNR (solid), CIR (dashed), and CNIR (dotted) vs IBO. Each noise level gets a distinct colour from the viridis colourmap.

**Typical sweep behaviour to expect:**

- As IBO decreases toward 0 dB: CIR drops steeply, EVM rises, BER eventually appears. CNR is unchanged (it depends only on noise level and signal power, not on IBO directly).
- As noise density increases (less negative): CNR drops. At high noise levels, CNIR tracks CNR rather than CIR, meaning the system becomes noise-limited rather than distortion-limited.
- The crossover point where CNR ≈ CIR is the most sensitive operating region — small changes in either IBO or noise level produce large changes in BER.

**To disable the sweep**, remove the `[sweep]` section from `simulation.toml` or delete both keys. The single-point run (which still computes all metrics) completes in a few seconds.

---

## 9. Adding or modifying carriers

Each `[[carrier]]` block in `simulation.toml` adds one signal to the composite. The constraints are:

1. **Integer upsample factor**: `sample_rate` must be an exact integer multiple of `sps × symbol_rate`. For example, with `sample_rate = 2_000_000_000`, a carrier with `symbol_rate = 5_000_000` and `sps = 10` has native rate 50 MHz and upsample factor 40 — valid. A `symbol_rate = 3_000_000` with `sps = 10` gives native rate 30 MHz and factor 66.67 — invalid.

2. **No spectral overlap**: carriers must not overlap in the wideband spectrum. A carrier at frequency `f` occupies roughly `[f − (1+rolloff)·sr/2, f + (1+rolloff)·sr/2]`. Check all pairs.

3. **Wideband bandwidth**: all carriers must fit within `[−sample_rate/2, +sample_rate/2]`. At 2 GHz, that is ±1 GHz.

### Example: adding a third carrier

```toml
[[carrier]]
name        = "medium"
symbol_rate = 5_000_000        # 5 Msym/s
sps         = 8                # native rate = 40 MHz; upsample factor = 50
rolloff     = 0.25
filter_span = 10
num_symbols = 2500
power_db    = -3.0             # 3 dB below the reference carriers
freq        = -500_000_000     # −500 MHz centre frequency

[carrier.channel]
enabled           = true
ripple_db         = 0.2
ripple_cycles     = 1.0
max_phase_dev_deg = 2.0
phase_poly_order  = 2
plot              = "channel_medium.png"
```

Re-run `python main.py`; the new carrier will automatically appear in the wideband PSD, the metrics table, and the sweep plots.
