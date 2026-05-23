# Memory Scaling Analysis

## Overview

This report analyses how memory consumption scales with the symbol-rate ratio
and per-iteration buffer budget in the wideband simulation. It verifies the
claim that chunk (OLA) processing makes per-block FFT memory independent of
total simulation length, and quantifies where real scaling costs come from.

Per-iteration symbol or frame counts are no longer user-set: they are derived
from `[simulation].max_block_size_samples` so the largest per-carrier
native-rate buffer stays within budget. Total simulation length scales with
`max_iterations × max_block_size_samples`, not with a hard-coded `num_symbols`.

---

## Notation

| Symbol | Meaning |
|--------|---------|
| SR | Wideband sample rate (configured under `[sweep].sample_rate`) |
| f_s | Carrier native sample rate = `sps × symbol_rate` |
| L | Upsampling ratio = SR / f_s |
| B_native | Per-carrier native-rate buffer per iteration = derived from `max_block_size_samples` |
| N_iter | Wideband samples processed per iteration = B_native × L |
| F | OLA filter half-span (`ola_filter_span`, default 16) |
| B | OLA block size (`ola_block_size`, default 4096) |

---

## Memory Components

### 1. Anti-alias filter (Kaiser-windowed sinc)

The filter used for up/downsampling has length:

```
filter_taps = 2 × F × L + 1
```

This is a one-off allocation of `filter_taps × 16 bytes` (complex128). It scales **linearly with L** — the dominant cost for narrowband carriers.

### 2. OLA FFT working buffer

Each block convolution requires an FFT of size:

```
N_fft = next_power_of_2(B + filter_taps − 1)
      ≈ next_power_of_2(B + 2 × F × L)
```

The per-block working buffer (input block + FFT output) is:

```
FFT buffer = N_fft × 16 bytes
```

This is **reused for every block** — it does not grow with `num_symbols` or `N_wb`. This is the key benefit of OLA chunk processing.

### 3. Persistent wideband signal arrays

The wideband arrays allocated once per simulation:

| Array | Size | Purpose |
|-------|------|---------|
| `wideband_normed` | N_wb × 16 B | Normalised composite signal |
| `wideband_nl` | N_wb × 16 B | After nonlinear amplifier |
| `wideband_noisy` | N_wb × 16 B | After AWGN |
| `x_up` (per carrier) | N_wb × 16 B | Upsampled carrier (in `fft_ola_upsample`) |
| `y_full` (OLA output) | ≈ N_wb × 16 B | Full convolution output (in `ola_convolve`) |

These scale with **N_wb = T × SR**, not with `num_symbols` directly.

### 4. OLA output array (`ola_convolve`)

```python
y = np.zeros(N + M - 1, dtype=complex)   # N = len(x), M = filter_taps
```

For wideband input (N = N_wb), this is ≈ N_wb × 16 bytes.

**Important**: `fft_ola_downsample` returns a **strided view** (`y[::L]`) into `y_full`, which means the full N_wb array stays alive in memory until the view goes out of scope.

---

## Current Configuration (Worked Example)

Default `simulation.toml`: SR = 800 MHz, F = 16, B = 4096,
`max_block_size_samples = 4_194_304` (≈ 4M samples per native-rate buffer).
One BPSK carrier at `symbol_rate = 100 kHz`, `sps = 4`.

| Parameter | Value |
|-----------|------|
| symbol_rate | 100 kHz |
| sps | 4 |
| native rate f_s | 400 kHz |
| L = SR / f_s | **2,000** |
| B_native (derived) | `max_block_size_samples = 4,194,304` samples |
| num_symbols per iteration | B_native / sps = **1,048,576** |
| filter_taps = 2×16×L+1 | **64,001** |
| N_fft = next_pow2(4096 + taps) | 131,072 |
| FFT buffer | **2 MB** (reused per chunk) |
| OLA efficiency = B / N_fft | 3.1% |

Under the chunk pipeline, the wideband signal is never materialised in full.
Per-iteration peak memory is bounded by the OLA state arrays and the
native-rate carrier output (see "Post-refactor" section below for the full
breakdown). With the configuration above, peak working memory is around 12 MB
regardless of the iteration count.

The dominant variable cost is the **upsampling ratio L**, which sets the OLA
filter length. Doubling `max_block_size_samples` doubles the per-carrier
native-rate output buffer but does not change L-driven costs.

---

## Scaling with Symbol Rate Ratio (L)

The table below holds SR = 2 GHz, F = 16, B = 4096 fixed and varies the carrier symbol rate.

| Symbol rate | L = SR/f_s | filter_taps | N_fft | FFT buffer | OLA efficiency |
|------------|-----------|-------------|-------|-----------|---------------|
| 20 MHz | 10 | 321 | 8,192 | 128 KB | 50% |
| 5 MHz | 40 | 1,281 | 8,192 | 128 KB | 50% |
| 1 MHz | 200 | 6,401 | 16,384 | 256 KB | 25% |
| 500 kHz | 400 | 12,801 | 16,384 | 256 KB | 25% |
| 100 kHz | 2,000 | 64,001 | 131,072 | 2 MB | 3.1% |
| 10 kHz | 20,000 | 640,001 | 1,048,576 | 16 MB | 0.4% |
| 1 kHz | 200,000 | 6,400,001 | 8,388,608 | 128 MB | 0.05% |

**Takeaway**: the per-block FFT buffer grows with L (via filter length), but it is always bounded and reused. The OLA efficiency collapses as L grows, meaning more FFT operations per output sample — but memory per operation stays fixed.

---

## Verification of the Claim

> "chunk processing makes memory independent of the number of symbols (except at the native sample rate for the narrowband signal)"

**Verdict: TRUE for FFT working memory and (post-refactor) wideband memory; the
binding scaling constraint is the per-carrier native-rate output buffer.**

| Memory type | Scales with B_native? | Scales with L? |
|-------------|----------------------|----------------|
| FFT working buffer (`N_fft × 16 B`) | **No** — reused per chunk | Yes (via filter_taps) |
| Filter coefficients (`filter_taps × 16 B`) | No | Yes (linearly) |
| Wideband composite (post-refactor) | **No** — chunked, never materialised | No |
| Per-carrier native-rate output (`B_native × 16 B`) | **Yes** — linear | No |

The most precise statement:

> The per-block FFT buffer depends only on the filter length (i.e. L) and is
> reused per chunk. The wideband composite is never held in memory in full —
> only one chunk at a time plus OLA state. The per-carrier native-rate output
> arrays (downsampled, used for BER/EVM) do scale with `B_native`, but at the
> narrow native rate.

`[simulation].max_block_size_samples` directly caps `B_native` for every carrier.
Iterations are independent of one another — the buffers are reused across
iterations within a sweep point — so total simulation length scales with
**iteration count**, not buffer size.

### Concrete check: doubling max_block_size_samples

Doubling `max_block_size_samples` from 4,194,304 → 8,388,608:

- B_native doubles for every carrier.
- Per-carrier native-rate output buffer doubles (16 → 32 MB per carrier for
  the example BPSK case above).
- FFT buffer: **unchanged** (still 2 MB, sized by L and the filter length).
- Wideband chunk size: **unchanged** (set by `[ola].block_size`).

To grow total simulation length without growing memory, raise `max_iterations`
(or rely on the CI-driven convergence to do that automatically when needed).

---

## Cost of a Very Narrowband Carrier

**Example: 100 kHz carrier in a 2 GHz wideband system**

```
L         = 2,000,000,000 / (10 × 100,000) = 2,000
filter     = 2 × 16 × 2,000 + 1 = 64,001 taps  →  1 MB coefficient array
N_fft      = next_pow2(4096 + 64,001) = 131,072
FFT buffer = 131,072 × 16 = 2 MB  (reused)
OLA eff.   = 4096 / 131,072 = 3.1%
```

The carrier is cheap in memory — 2 MB working buffer, 1 MB filter. But it pays a **32× overhead per output sample** in FFT operations versus a well-matched carrier.

For a carrier at 1 kHz: N_fft exceeds 8 M points and the coefficient array is 96 MB. At that extreme, **increasing `block_size`** is more effective than adding more symbols — it directly raises OLA efficiency (more useful output per FFT).

---

## Practical Guidance

| Scenario | Recommendation |
|---------|---------------|
| Many carriers, all within 10× of wideband rate | Default `block_size = 4096` is fine |
| One carrier ≥ 100× narrower than wideband | Increase `block_size` to 65536 or more to recover OLA efficiency |
| Need longer total simulation at fixed memory | Raise `[simulation].max_iterations`; each iteration reuses the same buffers |
| Need more bits per iteration | Raise `max_block_size_samples`; native-rate output buffer grows linearly with it |
| Extremely narrowband carrier (L > 10,000) | Consider a staged decimation (intermediate sample rate) rather than direct L-fold decimation |

### Effect of increasing `block_size`

For the 100 kHz carrier example:

| block_size | N_fft | FFT buffer | OLA efficiency |
|-----------|-------|-----------|---------------|
| 4,096 | 131,072 | 2 MB | 3.1% |
| 65,536 | 131,072 | 2 MB | 50% |
| 131,072 | 262,144 | 4 MB | 50% |

Raising `block_size` to 65,536 recovers 50% efficiency with no increase in N_fft for this case — a free win.

---

## Chunk Pipeline Memory Model (current)

The persistent wideband arrays described in §3 were the motivation for the
chunk pipeline refactor. They are now eliminated:

**What goes away:**

| Eliminated array | Old size |
|-----------------|---------|
| `wideband_normed` | N_wb × 16 B |
| `wideband_nl` | N_wb × 16 B |
| `wideband_noisy` | N_wb × 16 B |
| `x_up` (per carrier, in upsample) | N_wb × 16 B |
| `y_full` (OLA output, in `ola_convolve`) | N_wb × 16 B |

These are replaced by chunk-local buffers that are allocated once and reused:

**What replaces them:**

| Buffer | Size | Notes |
|--------|------|-------|
| Wideband chunk (composite) | `chunk_size × 16 B` | One chunk, reused each iteration |
| Per-carrier upsample state | `N_fft × 16 B` per carrier | OLA overlap tail, one per carrier |
| Per-carrier downsample state × 3 | `N_fft × 16 B` per carrier | One per (carrier × 3 passes) |
| Welch accumulator | `N_fft_welch × 8 B` × 3 | One per wideband signal (pre/post-NL/noisy) |
| Per-carrier native-rate output | `B_native × 16 B` | BER/EVM; at native rate. Capped by `max_block_size_samples`. |

**Peak memory after refactor** (example: 100 kHz carrier, L=2000, `block_size=65536`):

```
Chunk buffer:             65,536 × 16 =    1.0 MB   (wideband, reused)
Upsample OLA state:      131,072 × 16 =    2.0 MB   (per carrier)
Downsample OLA state × 3: 131,072 × 16 × 3 = 6.0 MB (per carrier)
Welch accumulators:      131,072 × 8  × 3 =  3.0 MB  (all 3 wideband PSDs)
Native-rate output:      B_native × 16              (capped by max_block_size_samples)
```

Total wideband overhead ≈ 12 MB regardless of iteration count or total bits
processed. The native-rate per-carrier arrays are the only thing that scales
with `max_block_size_samples`, and they do so at the native (narrow) rate.

**The critical case** — 100k symbols (one iteration's worth at sps=4 with
`max_block_size_samples = 400_000`) of a 100 kHz carrier in an 800 MHz
wideband system:

| Approach | Wideband memory |
|----------|----------------|
| Pre-refactor (full arrays) | 100,000 × 2,000 × 16 × 5 arrays ≈ **16 GB** |
| Post-refactor (chunk pipeline) | ≈ **12 MB** regardless of B_native |

**Return values change:** The simulation no longer returns `wideband`, `wideband_nl`,
or `wideband_noisy` as arrays. It returns their Welch PSD estimates instead —
`(f, psd_pre_nl)`, `(f, psd_post_nl)`, `(f, psd_noisy)` — which is all the
downstream plot code ever needed from them.

**Strided view issue fixed:** The old `fft_ola_downsample` returned `y[::L]`, a strided
view into the full N_wb allocation, keeping it alive until the view was dropped. The
stateful chunk-pipeline downsamplers emit decimated output directly, with no full-length
intermediate array.
