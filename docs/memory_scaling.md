# Memory Scaling Analysis

## Overview

This report analyses how memory consumption scales with the symbol rate ratio and number of symbols in the wideband simulation. It verifies the claim that chunk (OLA) processing makes per-block FFT memory independent of the number of symbols, and quantifies where real scaling costs come from.

---

## Notation

| Symbol | Meaning |
|--------|---------|
| SR | Wideband sample rate (2 GHz default) |
| f_s | Carrier native sample rate = `sps × symbol_rate` |
| L | Upsampling ratio = SR / f_s |
| T | Simulation duration = num_symbols / symbol_rate |
| N_wb | Total wideband samples = T × SR = num_symbols × L |
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

Default config: SR = 2 GHz, F = 16, B = 4096, T ≈ 0.5 ms (slow carrier governs)

| Parameter | Slow carrier | Fast carrier |
|-----------|-------------|-------------|
| symbol_rate | 1 MHz | 20 MHz |
| sps | 10 | 10 |
| native rate f_s | 10 MHz | 200 MHz |
| L = SR / f_s | **200** | **10** |
| num_symbols | 500 | 10,000 |
| N_wb = num_symbols × L | 100,000 | 100,000 |
| filter_taps = 2×16×L+1 | **6,401** | **321** |
| N_fft = next_pow2(4096 + taps) | 16,384 | 8,192 |
| FFT buffer | **256 KB** | **128 KB** |
| OLA efficiency = B / N_fft | 25% | 50% |

Both carriers produce the same N_wb = 100,000 samples because they share the same simulation duration T.

**Peak persistent memory** (all wideband arrays simultaneously live):

```
≈ 7 × N_wb × 16 bytes
= 7 × 100,000 × 16
= 11.2 MB
```

This is very modest. The dominant cost is the upsampling ratio L, not num_symbols.

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

**Verdict: TRUE for FFT working memory; requires a nuance for persistent arrays.**

| Memory type | Scales with num_symbols? | Scales with L? |
|-------------|--------------------------|----------------|
| FFT working buffer (`N_fft × 16 B`) | **No** — reused per block | Yes (via filter_taps) |
| Filter coefficients (`filter_taps × 16 B`) | No | Yes (linearly) |
| Wideband persistent arrays (N_wb) | Only if T changes | No — fixed by T × SR |
| Native-rate baseband arrays (N_wb / L) | **Yes** — proportional to num_symbols | No |

The most precise statement:

> For a **fixed simulation duration T**, all wideband memory is determined by `T × SR` regardless of individual carrier parameters. The per-block FFT buffer is independent of both T and num_symbols — it depends only on the filter length (i.e. L). The native-rate carrier arrays (downsampled output) do scale with num_symbols, but at native rate: `num_symbols × sps × 16 bytes`, which is always ≤ N_wb / L.

### Concrete check: doubling num_symbols vs doubling T

**Case A**: double `num_symbols` from 500 → 1000 for the slow carrier, keeping fast carrier the same.

- Both carriers now have different T (0.5 ms vs 0.05 ms). The simulation must use the longer one.
- N_wb grows from 100,000 → 200,000 — persistent arrays double.
- FFT buffer: unchanged (still 256 KB for slow, 128 KB for fast).

**Case B**: double the fast carrier's num_symbols from 10,000 → 20,000 (T_fast = 1 ms).

- T is now governed by the fast carrier: N_wb = 1 ms × 2 GHz = 2,000,000.
- All wideband persistent arrays grow 20×.
- FFT buffer: unchanged.

In both cases the OLA working memory (FFT buffer) is unaffected; only the persistent arrays reflect the new simulation duration.

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
| Very long simulation (large num_symbols) | Memory grows with T × SR; consider reducing `num_symbols` or SR |
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

## Post-refactor: Chunk Pipeline Memory Model

The persistent wideband arrays in §3 are the motivation for the chunk pipeline refactor.
After the refactor, they are eliminated:

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
| Per-carrier native-rate output | `num_symbols × sps × 16 B` | Still needed for BER/EVM; at native rate |

**Peak memory after refactor** (example: 100 kHz carrier, L=2000, `block_size=65536`):

```
Chunk buffer:             65,536 × 16 =    1.0 MB   (wideband, reused)
Upsample OLA state:      131,072 × 16 =    2.0 MB   (per carrier)
Downsample OLA state × 3: 131,072 × 16 × 3 = 6.0 MB (per carrier)
Welch accumulators:      131,072 × 8  × 3 =  3.0 MB  (all 3 wideband PSDs)
Native-rate output:      num_symbols × sps × 16       (scales with symbols)
```

Total wideband overhead ≈ 12 MB regardless of simulation duration or N_wb.
The native-rate per-carrier arrays are the only thing that still scales with
`num_symbols`, and they do so at the native (narrow) rate — at most N_wb / L bytes.

**The critical case** — 100k symbols of a 100 kHz carrier in an 800 MHz wideband system:

| Approach | Wideband memory |
|----------|----------------|
| Pre-refactor (full arrays) | 100,000 × 2,000 × 16 × 5 arrays ≈ **16 GB** |
| Post-refactor (chunk pipeline) | ≈ **12 MB** regardless of num_symbols |

**Return values change:** The simulation no longer returns `wideband`, `wideband_nl`,
or `wideband_noisy` as arrays. It returns their Welch PSD estimates instead —
`(f, psd_pre_nl)`, `(f, psd_post_nl)`, `(f, psd_noisy)` — which is all the
downstream plot code ever needed from them.

**Strided view issue fixed:** The old `fft_ola_downsample` returned `y[::L]`, a strided
view into the full N_wb allocation, keeping it alive until the view was dropped. The
stateful chunk-pipeline downsamplers emit decimated output directly, with no full-length
intermediate array.
