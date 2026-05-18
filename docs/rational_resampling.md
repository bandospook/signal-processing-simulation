# Rational Resampling for Non-Integer Upsample Factors

## The Problem

The wideband simulation requires each carrier's native-rate baseband signal to be
upsampled by an integer factor L to reach the wideband sample rate:

```
L = sample_rate / (sps × symbol_rate)
```

When L is not an integer — e.g. `500 MHz / (4 × 2 MHz) = 62.5` — the simulation
previously raised `ValueError`.  The fix is to round L to the nearest integer and
use rational resampling to bridge the gap.

**Approach:** round L to `L_int`, compute the rational correction factor as the
reduced fraction `P/Q = sample_rate / (L_int × native_rate)`, then:

1. Resample the native-rate baseband by P/Q before the OLA chunk pipeline
2. Run the OLA pipeline with integer `L_int` (unchanged)
3. Resample the downsampled result back by Q/P to recover native rate and integer sps
4. Pass to the receiver unchanged

For the 62.5 example: `P/Q = 500 / (62 × 8) = 125/124`.

---

## Implementation Trade-off: Custom vs. scipy

### What both approaches do

Both implement the standard P/Q rational resampler:

1. Zero-insert by P (conceptually — the polyphase form never builds the full array)
2. Apply a Kaiser-windowed sinc LP filter at cutoff `1/max(P,Q)` of the upsampled rate
3. Decimate by Q

Algorithmic complexity is identical: **O(N × filter_span)** multiply-adds, where
`filter_span ≈ 2 × n_half / P` taps per polyphase branch.

---

### Custom implementation (committed as of 2026-05-17, later replaced)

`rational_resample(x, P, Q, filter_span)` in `sim/filters.py`:

- Reduces P/Q by gcd, returns identity when P==Q
- Builds polyphase branches: `h_full[k::P]` for k=0..P-1
- For each branch, finds its output indices via modular inverse (`pow(Q,-1,P)`) and
  vectorises the windowed dot products with numpy fancy indexing + matmul
- Filter: `h = (P/M) * sinc(n/M) * kaiser(n, β=8)` where `M = max(P,Q)`
  — the `P/M` scale makes `sum(h) ≈ P`, cancelling the 1/P DC loss from zero-insertion

**Execution complexity:** correct O(N × L_poly).

**Performance:** ~20–100× slower than scipy for the same inputs.  The P-iteration
Python loop (125 iterations for the 62.5 case) allocates and GC's a `(N/P, L_poly)`
complex128 array per branch, with non-sequential x_pad accesses that hurt cache
efficiency.  Estimated wall time for N=400,000: 1–3 s vs. ~10–50 ms for scipy.

**Correctness risk:** non-trivial index arithmetic.  One bug was introduced and
caught by tests during development: the filter gain was `P × sinc` instead of
`(P/M) × sinc`, producing a passband gain of M rather than 1.  Other subtle pieces
include the phase-offset formula `(k - n_half%P + P) % P` and the modular inverse.

**Maintenance cost:** ~60 lines of custom polyphase code that a future reader must
re-derive to audit.

---

### scipy.signal.resample_poly (current implementation)

```python
from scipy.signal import resample_poly
bb_ch = resample_poly(bb_ch, P_rs, Q_rs).astype(complex)
```

**Execution complexity:** same O(N × filter_span) algorithm.

**Performance:** `upfirdn` is a C extension with SIMD-friendly sequential access.
It processes all P polyphase branches in a single C-level pass with no Python
overhead.  ~20–100× faster than the custom version for large N.

**Correctness:** battle-tested across thousands of scipy users; edge cases (P=1,
Q=1, very short signals, half-sample phasing) are handled and regression-tested
upstream.

**Maintenance cost:** 2 lines of code.  The tradeoff logic lives here in this
document, not in the implementation.

**Dependency cost:** scipy is not otherwise required by this project.  It is a
~20 MB package but is standard in scientific Python environments.  Added to
`pyproject.toml` dependencies.

---

### Summary

| Dimension              | Custom                        | scipy.signal.resample_poly  |
|------------------------|-------------------------------|-----------------------------|
| Algorithm              | Polyphase Kaiser-sinc (same)  | Polyphase via upfirdn (same)|
| Big-O                  | O(N · filter_span)            | O(N · filter_span)          |
| Constant factor        | ~20–100× slower (Python loop) | C-speed, SIMD-friendly      |
| Memory                 | P temporary arrays per call   | One C-allocated working buf |
| Code size              | ~60 lines, subtle index math  | 2 lines                     |
| Bug surface            | Moderate (1 bug found in dev) | Minimal                     |
| New dependency         | None                          | scipy                       |

The custom implementation was written when scipy was not a project dependency.
Once the trade-off was documented, the project switched to `resample_poly` and
scipy was added as a dependency.

---

### When rational resampling is invoked

`rational_resample` (or `resample_poly`) is called **twice per non-integer-L
carrier per simulation run**: once before the OLA chunk pipeline and once after.
It is never called inside the hot chunk loop.  The performance difference (seconds
vs. milliseconds) is only material for very long records (millions of symbols) or
when the seeker calls the simulation dozens of times.
