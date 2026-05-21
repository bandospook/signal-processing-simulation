# Synchronization — Genie-Aided by Design

## Decision

The simulator does **not** implement blind timing recovery or blind carrier
(phase / frequency) recovery. Symbol timing is known by construction, and carrier
phase ambiguity is resolved against the transmitted reference bits. This is a
deliberate design choice, recorded here so it is not mistaken for an oversight.

This document explains what the receiver assumes, why genie-aided synchronization
is the right choice for *this* simulator, and what the blind alternatives would
cost if the decision is ever revisited.

---

## What the receiver assumes today

**Timing.** The transmitter and receiver share the same `sps` sample grid, so
symbol centres lie at known integer sample positions. The receiver samples those
positions directly (`mf[::sps]` for the linear modulations, the `2j·sps` window
starts for MSK) — it never *searches* for timing. There is no timing offset to
find.

**Carrier phase.** `_ber_with_ambiguity` in `sim/receiver.py` rotates the
received samples through the N rotationally-symmetric orientations of the
constellation and keeps the one that best matches the reference bits. This is
reference-aided, not blind. DBPSK is the single genuinely blind case:
differential decoding removes the 180° ambiguity with no reference at all.

**No offsets are injected.** Nowhere in the signal chain — transmitter, channel
impairment, nonlinear amplifier, or AWGN stage — is a carrier frequency offset, a
random carrier phase, or a fractional-sample timing error introduced.

---

## Why genie-aided synchronization is correct here

**1. There is nothing to recover.** Because no timing, phase, or frequency offset
is injected, blind recovery has nothing to estimate. Adding it would first
require *adding* those offsets to the transmitter or channel — recovery is
meaningless without an impairment to track.

**2. It isolates the quantity of interest.** This simulator exists to measure
nonlinear-amplifier distortion, channel impairment, and implementation loss
relative to AWGN theory (see GUIDE §9). Genie-aided synchronization removes
synchronization loss as a confounding variable, so a measured BER reflects the
modulation and channel alone. Isolating the impairment under study from sync
losses is standard practice for link-budget and impairment simulators.

**3. Blind synchronization has its own cost.** Every recovery loop introduces a
small implementation loss of its own — timing jitter and phase-estimator noise
each cost a few tenths of a dB. That would blur the BER-vs-theory comparison that
is currently clean to ~0.05 dB (see [msk_modulation.md](msk_modulation.md) and
`tests/plots/performance/theory_comparison.md`). The clean comparison is a
feature, not an accident.

---

## Options considered

### Blind carrier phase recovery

A Viterbi & Viterbi M-th-power feedforward estimator: raising the samples to the
M-th power cancels the modulation, and the residual angle divided by M is the
phase estimate. About 30 lines, feedforward, no loop. Limitation: it leaves an
M-fold phase ambiguity, so differential encoding or a known preamble would still
be needed to fix the absolute bit mapping. The gain over the existing
reference-aided rotation search is modest.

### Blind symbol timing recovery

A fractional-delay interpolator (Farrow structure), a timing-error detector
(Gardner — decision-independent, two samples per symbol), and a second-order loop
filter. Roughly 150 lines plus careful testing. OQPSK and MSK are harder than the
linear PSK/QAM formats here: the offset rails and continuous phase require
sync-aware detector variants rather than the textbook PSK detectors.

### Blind carrier frequency-offset estimation

Relevant only if oscillator drift is modelled. It needs a frequency estimator or
frequency-locked loop ahead of the phase loop. Not currently applicable — no
frequency offset exists in the signal chain.

---

## If this is revisited

Stage the work so each piece is independently testable:

1. Inject the offsets (fractional timing delay, random carrier phase, optional
   frequency offset) into the transmitter or channel — so there is something to
   recover.
2. Add a blind static-phase estimator.
3. Add blind timing recovery last.

Expect each recovery loop to introduce its own small, characterizable
implementation loss — which is precisely the loss the current genie-aided design
deliberately excludes.
