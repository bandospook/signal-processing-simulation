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

## CNIR -> Eb/N0 conversion (includes sps factor -- easy to get wrong)

CNIR in simulation.py is measured at the native sample rate (sps samples/symbol),
before matched filtering. Converting to Eb/N0 per bit:

    effective_ebn0_db = cnir_db + 10 * log10(sps / bps)

Why sps matters: noise power at native rate = `noise_density * sps * symbol_rate`.
Theory `Eb/N0 = signal_power / (bps * symbol_rate * noise_density)`.
Ratio = `CNIR * sps / bps`. For BPSK sps=4: +6 dB correction.
Forgetting this produces a systematic -6 dB implementation loss on a linear amplifier.

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
