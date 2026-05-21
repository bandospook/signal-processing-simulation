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
- **LDPC** — sparse parity-check codes, decoded with belief propagation.

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
noise variance (already known — the simulator sets the noise density). The
max-log form is ~0.1–0.5 dB from exact, vectorizes cleanly in NumPy, and is the
recommended default. The demapper is a single pass — **not** a performance
hot spot.

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

### Open risk — Numba and the Python version

Numba typically trails new CPython releases by a few months. This project
currently runs on Python 3.14, which may be ahead of Numba's supported range.
**Verify current Numba 3.14 support before committing to this plan.** If Numba
does not yet support 3.14, the options are: (a) a NumPy-vectorized LDPC decoder
— acceptable (~5–20× off compiled), though weak for the sequential turbo/Viterbi
recursions; (b) Cython, which supports any Python but adds a build-time C
compiler requirement; or (c) hold the turbo decoder until Numba catches up.
`galois` depends on Numba and inherits the same constraint.

---

## Knock-on effects

- **BER seeker (`targeter.py`)** — would bisect on *post-decoder* BER or FER.
  Coded curves have a steep waterfall, so convergence is fast but the bracket
  and accuracy logic near the cliff need review.
- **`theory.py`** — turbo and LDPC have no closed-form BER. Comparison shifts to
  the uncoded curve plus a coding-gain annotation, or to published reference
  curves; only the convolutional inner code has tractable bounds.
- **Eb/N0 bookkeeping** — Eb becomes energy per *information* bit, so the
  Es/N0 ↔ Eb/N0 conversion must include the code rate.
- **Runtime budget** — the existing OLA chunk pipeline already exists for
  performance reasons (see `memory_scaling.md`); coded low-BER sweeps are the
  next runtime pressure point. The target BER floor for sweeps should be chosen
  deliberately.

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

## Open decisions

1. Approve adding `numba` and `galois` as dependencies (pending the Python-3.14
   support check above)?
2. Confirm the from-scratch-decoders + Numba approach over a compiled library?
3. Which standards to target — DVB-S2 LDPC? CCSDS turbo / convolutional? Which
   code rates?
4. Target BER floor for coded sweeps — this sets the runtime budget.
5. Soft demapper: max-log (recommended) vs. exact LLR.
