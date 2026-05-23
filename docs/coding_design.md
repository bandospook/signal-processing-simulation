# Forward Error Correction — Design Proposal

**Status:** design proposal — not yet implemented. Open decisions for the
project owner are listed at the end; nothing here is committed until those are
resolved.

## Goal

Add three forward-error-correction (FEC) families to the per-carrier signal
chain and measure coded BER / frame-error-rate (FER) and coding gain against the
existing uncoded theory curves:

- **Concatenated** — convolutional inner code + Reed-Solomon outer code (classic
  CCSDS-style), with an interleaver between them.
- **Turbo** — parallel-concatenated convolutional codes, iteratively decoded.
- **LDPC** — DVB-S2X parity-check codes, decoded with belief propagation.

---

## Where coding sits in the chain

```mermaid
flowchart LR
    DATA["data bits"] --> ENC["FEC encode"]
    ENC --> ILV["interleave"]
    ILV --> MAP["map_bits → baseband"]
    MAP --> CH["channel · NLA · AWGN"]
    CH --> DEMAP["soft demap → LLRs"]
    DEMAP --> DILV["deinterleave"]
    DILV --> DEC["FEC decode"]
    DEC --> OUT["decoded data bits"]
```

BER is then measured as **decoded data bits vs. the original data bits**
(post-decoder), in addition to the existing raw channel BER.

---

## The keystone: soft-decision demapping

The current receiver makes **hard decisions** only (`decide()` → nearest
constellation point → bits). Turbo and LDPC decoders need **soft information** —
a per-coded-bit log-likelihood ratio (LLR). Feeding them hard bits forfeits
roughly 2 dB and largely defeats the purpose.

So the first deliverable, independent of which codes are built, is a soft
demapper in `receiver.py`:

```
exact:    LLR(b) = log( Σ_{s: b=0} exp(-|y-s|²/σ²) ) − log( Σ_{s: b=1} exp(-|y-s|²/σ²) )
max-log:  LLR(b) ≈ ( min_{s: b=1} |y-s|²  −  min_{s: b=0} |y-s|² ) / σ²
```

`y` is the received symbol, `s` ranges over constellation points, `σ²` is the
noise variance (already known — the simulator sets the noise density). **The
exact LLR is the chosen form**; max-log (~0.1–0.5 dB worse) stays available as a
cheaper fallback. Either way the demapper is a single vectorized pass — **not** a
performance hot spot — so the exact form's extra cost is immaterial.

---

## Module structure

A new `sim/coding/` subpackage, each codec exposing a uniform interface
(`encode(data_bits) -> coded_bits`, `decode(llrs) -> data_bits`):

```
sim/coding/
  __init__.py        scheme registry + factory
  interleaver.py     block / random / turbo interleavers
  convolutional.py   convolutional encoder + soft-decision Viterbi
  reed_solomon.py    Reed-Solomon outer code (see speed section: galois-backed)
  turbo.py           PCCC encoder + iterative BCJR/max-log-MAP decoder
  ldpc.py            parity-check matrices + belief-propagation decoder
```

A new optional `[carrier.coding]` config block: `scheme`, `code_rate`,
`block_length`, decoder `iterations`. Absent block ⇒ uncoded (current behaviour).

---

## The three schemes

| Scheme | Components | Decoder | Needs soft input | Relative effort |
|---|---|---|---|---|
| Concatenated | conv. inner + RS outer + interleaver | soft Viterbi + algebraic RS | inner: yes; outer: no | medium |
| Turbo | 2× conv. + turbo interleaver | iterative BCJR (SISO) | yes | high |
| LDPC | sparse parity-check matrix H | belief propagation (min-sum) | yes | medium |

---

## Execution speed: the library question

This is the decision that most affects the result, so it gets its own analysis.

### What is actually expensive

Encoders and the soft demapper are cheap single-pass operations. The cost is
concentrated in the **iterative and sequential decoders**: LDPC belief
propagation (tens of iterations of message passing over the Tanner graph), turbo
BCJR (sequential forward-backward recursion, repeated each iteration), and the
Viterbi trellis. To resolve a coded BER near 1e-6 you need on the order of
1e7–1e8 channel bits, so decoder throughput sets the whole simulation budget.

### The speed axis is *compiled vs. interpreted*, not *library vs. our code*

This is the key point. A **pure-Python** library (e.g. `commpy`, `pyldpc`) runs
the same interpreted message-passing loops our own code would — it is **not
faster than code we write ourselves**. The libraries that *are* fast
(`aff3ct`, `sionna`) are fast because they have a **compiled backend** (C++,
or TensorFlow/GPU) — and those are heavyweight, install-fragile dependencies
(the `aff3ct` Python bindings are particularly painful on Windows; `sionna`
pulls in all of TensorFlow and targets GPUs).

| Approach | Speed vs. compiled C | Notes |
|---|---|---|
| Pure-Python loops (our code **or** a pure-Python library) | ~100–1000× slower | Unusable for low-BER sweeps — minutes-to-hours per SNR point |
| NumPy-vectorized | ~5–20× slower | Works for LDPC BP (edge-parallel); poor fit for the sequential turbo/Viterbi recursions |
| Our code + Numba `@njit` | ~1.5–3× slower | Near-compiled speed; code stays readable Python |
| C/C++ library (`aff3ct`) | baseline | Fastest, but heavy and fragile to install |

The practical consequence: a pure-Python library buys us **no speed** while
costing inspectability and integration. We do **not** have to choose between
"fast" and "our own code" — JIT-compiling our own decoders with **Numba** gets
us within ~2–3× of the fastest C libraries.

### Recommended dependency strategy

- **`numba`** — JIT-compile the decoder hot loops (`@njit`). Lightweight,
  pip/uv-installable, ships Windows wheels. One-time ~1 s compile on first call.
- **`galois`** — Numba-backed GF(2ᵐ) arithmetic and a validated Reed-Solomon
  implementation. Use it for the RS layer: hand-rolling GF(2⁸) arithmetic and a
  Berlekamp-Massey decoder is error-prone, and `galois` is both fast and
  correct. This is the one place a library is a clear win on speed *and*
  quality.
- **Ship standard code definitions as data** — published DVB-S2 / CCSDS LDPC
  parity-check matrices, turbo polynomials, and interleaver tables. Using
  standard-compliant codes is the real quality win, and it needs no decoder
  library — only our own decoder run against the standard matrix.
- **Avoid** `aff3ct`/`sionna` (heavy, fragile, black-box) and pure-Python comms
  libraries (no speed gain, integration cost).

Net: **our own decoders, JIT-compiled with Numba; `galois` for the RS layer;
standard matrices shipped as data.** Near-library speed, fully inspectable and
documentable code, two lightweight new dependencies.

### Python version — verified compatible (checked 2026-05-20)

Earlier drafts flagged a risk that Numba might trail CPython and not yet support
this project's Python 3.14. **That risk has been checked and does not apply.**

- Numba 0.65.1 (released 2026-04-24) officially supports Python 3.10–3.14 and
  ships `cp314` wheels (standard and free-threaded).
- galois 0.4.11 (released 2026-05-02) supports Python 3.7–3.14.
- `uv pip install --dry-run numba galois` against this machine's Python 3.14.4
  resolves cleanly: `numba==0.65.1`, `llvmlite==0.47.0`, `galois==0.4.11`.

No change to the project's Python version is required.

---

## Knock-on effects

- **`theory.py`** — turbo and LDPC have no closed-form BER. Comparison shifts to
  the uncoded curve plus a coding-gain annotation, or to published reference
  curves; only the convolutional inner code has tractable bounds.
- **Eb/N0 bookkeeping** — Eb becomes energy per *information* bit, so the
  Es/N0 ↔ Eb/N0 conversion must include the code rate.
- **Runtime budget** — coded low-BER sweeps are a significant new runtime
  pressure point; see *Runtime budget and BER floor* below.

---

## Runtime budget and BER floor

BER measurement is a counting experiment: to estimate an error rate `p`, transmit
`N` units and count errors. Accuracy depends on the *error count*, not `N`:

```
relative standard error  ≈  1 / √(errors counted)
    100 errors → ~10%      400 errors → ~5%
N  ≈  (errors wanted) / p
```

So **each decade lower in target BER costs ~10× the bits and ~10× the runtime.**

Coded specifics:

- Decoding works on **frames**, not bits. DVB-S2X FECFRAME is 64,800 bits
  (normal) or 16,200 (short). The natural rare event is a **frame error**; report
  FER alongside coded BER.
- Coded curves fall in a steep **waterfall** (many decades per dB), sometimes
  followed by an **error floor** at very low BER (trapping sets). DVB-S2X codes
  are designed for floors below ~1e-9, so the routine job is to characterize the
  *waterfall*, not the floor.

Illustrative runtime — actual decoder throughput must be measured, but the shape
holds (100 frame-errors per SNR point):

```
runtime ≈ (frames needed) / (decoder throughput) × (SNR points) × (MODCODs)
```

| Target FER | Frames needed | @ 5000 frame/s | @ 500 frame/s |
|---|---|---|---|
| 1e-3 | 1e5 | ~20 s/pt | ~3 min/pt |
| 1e-4 | 1e6 | ~3 min/pt | ~33 min/pt |
| 1e-5 | 1e7 | ~33 min/pt | ~5.5 h/pt |
| 1e-6 | 1e8 | ~5.5 h/pt | ~2 days/pt |

A 64,800-bit LDPC frame under iterative belief propagation is heavy; throughput
depends on code rate, frame size, and iteration count. Three techniques keep it
tractable and should be built in from the start:

- **Early termination** — stop belief propagation once the parity check is
  satisfied. At high SNR most frames decode in a few iterations rather than the
  50-iteration cap. Biggest single win, and it accelerates exactly the low-BER
  region.
- **Adaptive stop-on-errors** — end an SNR point once enough frame errors are
  collected, instead of running a fixed `N`.
- **Multicore** — frames are independent; Numba `prange` parallelises across
  cores (~8–16× on a typical desktop).

Recommended policy: a routine sweep floor of **1e-5** (overnight-feasible), an
opt-in long-run mode for **1e-6**, and sub-1e-7 floor characterization treated as
a separate dedicated campaign rather than a sweep point.

---

## Testing strategy

Coded-decoder tests follow the existing `_N_BITS_PLOT` pattern in
`test_awgn_performance.py`: a module-level frame-count constant kept small enough
that the suite still runs in seconds, raised by hand for a thorough validation
run.

- **Default (routine `pytest`)** — a few hundred to a few thousand frames per
  point, in the *shallow* waterfall (FER ~1e-1…1e-2). Confirms the decoder runs,
  converges, and lands on roughly the right curve. Adds seconds to the suite,
  not minutes.
- **Thorough (on demand)** — raise the constant to push down to FER 1e-5–1e-6
  and compare against published reference curves. This is the deep validation
  gate from *Runtime budget and BER floor* — run deliberately, not on every
  invocation.

The fast default catches gross breakage — a decoder that fails to converge, or
sits at FER ≈ 1. Subtle correctness bugs (message-passing scaling,
early-termination, interleaver off-by-one) hide in the deep region, so the
thorough run against reference curves is the real correctness gate.

---

## Recommended staging

Each stage is independently testable against known reference curves:

1. **Soft demapper** — the enabler; self-contained, validated on its own.
2. **Convolutional + soft Viterbi** — exercises the full
   encode → modulate → channel → soft-demap → decode loop with a code that has
   well-known curves.
3. **LDPC + min-sum belief propagation** — highest value/effort ratio; LDPC is
   what modern satcom (DVB-S2) actually uses.
4. **Turbo + BCJR** — most intricate; last.

Concatenated coding then falls out as plumbing: convolutional (stage 2) +
Reed-Solomon (`galois`) + interleaver, chained.

---

## Decisions

All settled with the project owner:

- **Dependencies / approach** — `numba` and `galois` approved; decoders written
  from scratch and JIT-compiled with Numba, rather than a compiled library
  (Python-3.14 compatibility verified above).
- **Codes** — DVB-S2X LDPC; plus turbo and convolutional codes as selectable
  options. The specific turbo/convolutional parameterisation (CCSDS is the
  natural satcom choice) to be fixed at implementation time.
- **Soft demapper** — exact LLR, not the max-log approximation.
- **Test runtime** — coded tests default to a small frame count so the suite
  still runs in seconds, with a constant to raise for thorough deep validation;
  see *Testing strategy*.
- **Seeker** — extended to coded carriers, with a feasibility guard that reports
  an infeasible (too-low) target rather than running indefinitely; see the
  seeker bullet under *Knock-on effects*.
