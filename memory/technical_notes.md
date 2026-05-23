# Technical Notes

Key implementation decisions -- the non-obvious stuff that isn't in the code comments.

---

## AWGN placement: noise AFTER the amplifier

sim/simulation.py adds AWGN after the nonlinear amplifier. This models a single-hop
satellite downlink where thermal noise is primarily at the receiver.

Why it matters: If noise is placed before the amp, C/I shifts with noise level.
Noise after amp means C/I is determined solely by IBO and signal statistics.

---

## DBPSK theory formula -- coherent + differential, NOT differentially-coherent

    BER = 2 * p * (1 - p),   p = 0.5 * erfc(sqrt(Eb/N0))

NOT 0.5 * exp(-Eb/N0) (the differentially-coherent formula).

Why: The simulation uses a coherent RRC matched filter followed by differential
decoding of hard symbol decisions. A single symbol error flips two decoded bits -> BER ~= 2p.
The exp(-Eb/N0) formula applies to a receiver that makes decisions purely from phase
differences (no matched filter) -- about 3 dB worse. Wrong formula = 3 dB offset in IL.

---

## CNIR -> Eb/N0 conversion

CNIR (and CNR) in simulation.py are reported in the **symbol-rate (matched-filter)
bandwidth**: sim/simulation.py divides the native-rate noise power by sps before
forming the ratios.  In this convention CNIR = Es / N0 directly.  Converting to
Eb/N0 per information bit:

    effective_ebn0_db = cnir_db - 10 * log10(bps)

For BPSK (bps = 1) the conversion is a no-op: CNIR = Es/N0 = Eb/N0.
For QPSK -3 dB, 8PSK -4.77 dB, 16QAM -6 dB.

Historical note: pre-2026-05-18 (commit 5915eda), CNIR was reported in the
native-rate bandwidth and the conversion was `cnir + 10*log10(sps/bps)`.
Targeter.py was updated at the same time, but when targeter.py was removed with
the seeker, main.py was carried forward with the stale formula — re-fixed when
the discrepancy was caught.  If any code is computing eff_ebn0 with the old
`+sps` term, it is double-counting the matched-filter gain.

---

## Implementation loss definition

    IL_dB = effective_Eb/N0_dB - theory_Eb/N0_dB(at_measured_BER)

Uses C/(N+I) not C/N so that with no nonlinearity (I=0) the loss is zero.

---

## BPSK / QPSK / OQPSK / MSK share the same theory BER curve

All four: BER = 0.5 * erfc(sqrt(Eb/N0)). The BER plot draws the dashed theory line
once via the _BPSK_EQUIV set to avoid duplicate legend entries.

---

## MSK is implemented as offset-QPSK, NOT as CPFSK

MSK is generated and detected as offset-QPSK with a half-sine pulse 2*sps samples
wide (even bits -> I rail, odd bits -> Q rail delayed by sps), not as a
continuous-phase FSK phase accumulator.

Why it matters: a symbol-by-symbol correlator on the CPFSK form is ~3 dB worse than
BPSK, because over a single bit period the two MSK waveforms (phase ramping up vs.
down) are nearly orthogonal, not antipodal. Recovering MSK's defining BPSK-equal
BER requires a 2-bit-period observation. The OQPSK/half-sine view makes each rail an
independent antipodal channel, so a plain per-rail matched filter is optimal -- no
Viterbi/MLSE needed. The half-sine matched-filter energy is exactly sps, so there is
no discretization loss; measured BER tracks the BPSK theory curve within ~0.05 dB.

A consequence: the at-most-one rail symbol whose 2*sps window runs off the signal
end is detected from a clipped (half-energy) window. Negligible for the test sizes
(n_sym >= 300) and harmless noiseless. EVM is NaN for MSK (constant envelope, no
discrete constellation).

---

## Filter span convention

filter_span in carrier config = RRC filter half-span in symbols.
Total filter length = filter_span * sps + 1 taps.

---

## matplotlib colormaps

Use plt.colormaps["viridis"](...) not cm.viridis(...). Newer matplotlib type stubs
don't expose colormaps as direct attributes of matplotlib.cm.

---

## Why the chunk pipeline matters: symbol rate ratio explosion

When carriers differ greatly in symbol rate, N_wb = num_symbols × L grows explosively
for the narrowband carrier. Example: 100,000 symbols of a 100 kHz carrier at
L = 800MHz / (4 × 100kHz) = 2000 gives 200M wideband samples. At 16 bytes each,
the five persistent wideband arrays (wideband, wideband_normed, wideband_nl,
wideband_noisy, x_up) consume ~16 GB.

The chunk pipeline eliminates all five of those arrays. Peak wideband memory becomes
~12 MB regardless of simulation duration — only the per-carrier native-rate arrays
(BER/EVM output) scale with num_symbols, and they do so at the narrow native rate.

This is documented in detail in docs/memory_scaling.md § "Post-refactor: Chunk Pipeline".

---

## NLA input normalization: RMS-based, not empirical-peak-based

**Decision (2026-05-17):** The wideband composite is normalized to unit RMS before
driving the nonlinear amplifier, using the *analytical* RMS — not np.max() of a
single realization.

Analytical RMS:
    rms = sqrt( sum of 10^(power_db/10) for each carrier )
    wideband_normed = wideband * drive / rms
    drive = 10^(-input_backoff_db / 20)

Why not empirical peak (np.max(np.abs(wideband))):
  - np.max() is a random variable — it shifts ~0.3–0.5 dB between seeds, causing
    run-to-run variation in effective drive level and thus in CIR/CNIR.
  - A two-pass approach (one pass to find the peak, one to process) would cost ~25%
    extra runtime: upsample is 1 of 4 OLA stages per carrier (1 up + 3 down), so
    doubling the upsample adds 1/4 extra work.
  - RMS-referenced IBO is the standard physical definition for satellite amplifier
    characterisation: IBO = (single-carrier saturated input power) / (actual RMS input
    power). Results are directly comparable to manufacturer specs and link budgets.
  - The PAPR of the composite (peak/RMS) is the physical quantity that tells you how
    hard the amp clips on peaks — it doesn't need to be baked into the normalization.

What IBO means under this convention:
    At IBO = 0 dB the RMS composite drives the amp at its reference (saturation) level.
    At IBO = 3 dB the RMS is 3 dB below that. Peak excursions above saturation still
    occur if PAPR > IBO; that is expected and physically correct.

This is implemented in sim/simulation.py (commit e33778e).

---

## np.real() / np.imag() -- use function form

Use np.real(x) not x.real. Pylance's numpy stubs have broken overloads for the
property form on ndarray[Any, dtype[Any]].

---

## Implementation loss is measured, not approximated

**Decision (2026-05-23):** Implementation loss is reported only as the
*measured* gap between effective Eb/N0 and the AWGN theory curve at the
observed BER:

    IL_dB = effective_Eb/N0_dB - theory_Eb/N0_dB(at_measured_BER)

We considered also reporting a cheap predictor based on the Gaussian-
distortion approximation:

    IL_approx_dB = (C/N)_dB - (C/(N+I))_dB

That formula treats the NL-induced distortion `I` as if it were additional
AWGN of power equal to the measured CIR's `I`.  It's accurate when the
distortion really is Gaussian-like — many independent carriers through a
memoryless NL at moderate IBO, radially symmetric constellations
(BPSK/QPSK/MSK/OQPSK), CLT-friendly statistics.

The approximation breaks down in regimes we actually care about:

- **Single carrier, low IBO** — distortion is a deterministic per-symbol warp,
  not random.  BPSK can sit at BER ≈ 0 even with finite CIR; IL_approx
  predicts several dB of loss that doesn't exist.
- **Multi-amplitude constellations** (16QAM, 16APSK, 32APSK) — AM-AM and
  AM-PM act differently on inner vs. outer rings.  Distortion is
  symbol-correlated and asymmetric; Gaussian-equivalence is off by 0.5–2 dB
  in either direction depending on regime.
- **AM-PM-dominated regimes** — pure phase distortion looks like rotation,
  not noise.  Phase-ambiguity resolution removes the systematic component;
  the amplitude-dependent residual still warps multi-ring constellations.
- **Coded carriers** — the decoder assumes Gaussian-noise LLRs.  Non-Gaussian
  distortion mis-scales LLR magnitudes and shifts coding gain by an amount
  no AWGN-theory curve can predict.  IL_approx is essentially a guess here.
- **Error floors at high CNR** — IL_approx predicts BER keeps falling on the
  AWGN curve translated by a constant; reality is a floor.  Gap grows
  without bound.

Adding IL_approx as a second column was considered.  Decision: don't.  It
encourages reading the predicted value when the measured one is available
and authoritative.  If we ever want the gap as a diagnostic ("is distortion
Gaussian-like at this operating point?"), compute it offline from the
existing CNR/CIR/CNIR columns — no need to bake it into the report.

---

## Adaptive iteration: CI-bounded BER measurement at a fixed memory budget

**Decision (2026-05-23):** The sim runs each (IBO, noise) sweep point as an
adaptive accumulation of independent iterations rather than a single fixed-
size shot.  Per-carrier `num_symbols` / `num_frames` are derived from a
memory budget; total bits are grown by re-running with fresh seeds until a
Wilson-CI half-width target on BER is met.

### Why

A fixed `num_symbols` forces an awkward trade-off: too small → BER estimates
have huge variance at low error rates (a single error swings the estimate by
orders of magnitude); too large → the narrowband carrier's native-rate buffer
blows the memory budget that the chunk pipeline was built to respect.
Adaptive iteration breaks the trade-off: each iteration stays within budget,
and the count of iterations scales automatically with the difficulty of the
operating point (more iters needed at low BER, fewer at high BER).

### Configuration

New `[simulation]` keys (also displayed on the GUI's General tab):

    max_block_size_samples    int    per-carrier native-rate buffer cap (samples)
    target_ci_half_width      float  absolute half-width on BER (e.g. 2e-3)
    confidence                float  two-sided level for the Wilson CI (e.g. 0.95)
    min_errors                int    minimum cumulative errors before convergence
                                     can be declared (default 50)
    max_iterations            int    safety cap; iteration runs that hit this
                                     without converging are flagged `CAPPED`

`num_symbols` and `num_frames` are no longer user-settable on carriers.  They
are derived per-carrier as:

    uncoded:  num_symbols   = max(1, max_block_size_samples // sps)
    coded:    syms_per_frame = ceil(code.coded_bits / bps)
              n_frames       = max(1, max_block_size_samples // (syms_per_frame * sps))

### Stopping rule (per carrier, per sweep point)

Stop when:

    (wilson_half_width(k, n, confidence) <= target_ci_half_width  AND  k >= min_errors)
    OR  iterations >= max_iterations

The Wilson score interval is used rather than the Wald (normal-approx)
interval because Wald collapses to zero half-width at k=0 and gives
non-sensical bounds at small p; Wilson behaves correctly across the full
range.  The `min_errors` floor prevents premature stops when the Wilson
half-width is small but jittery from too few errors.

### Zero-errors reporting

If a point exits the loop with k=0, BER is reported as an upper bound rather
than zero.  Using the rule of three (95%):

    BER < -ln(1 - confidence) / n_bits

This shows up in `report.md` as e.g. `< 3.2e-9` so the absence of evidence
isn't mistaken for evidence of absence.  Implementation loss is recorded as
`—` in this case (no theory inversion possible from a zero BER).

### Aggregation of non-BER metrics

CNR / CIR / CNIR / EVM are averaged across iterations (arithmetic mean of
the per-iteration values).  CNR/CIR/CNIR are nearly deterministic across
seeds anyway — both depend on signal statistics and analytical noise power,
not on noise realisation — so the mean is essentially identical to any
single iteration's value.  EVM has slight jitter; the mean is the right
reduction.

### Per-iteration seeding

Iterations use `point_seed = base_seed + grid_index * STRIDE + iter_index`
with `STRIDE = 2_147_483_587` (large prime).  This guarantees independent
bit streams and AWGN realisations across iterations while remaining
reproducible from the user-set base seed.

### Per-carrier convergence (strict policy)

All `sweep_demod` carriers at a point must individually converge before the
point is considered done.  A single low-BER carrier can dominate iteration
count for an entire sweep — accepted as the price of statistical honesty.

### Cost shape

Per-iteration cost is the full chunk-pipeline run at the budget size.  Worst
case is `max_iterations × grid_size` runs (e.g. 100 × 12 = 1200 sim runs);
typical case converges in 1–10 iterations per point depending on operating
point and target.  The user controls the worst case via `max_iterations`.
