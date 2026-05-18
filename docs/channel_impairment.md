# Channel Impairment Model

## What it does

`apply_channel_impairment` in `sim/filters.py` models the distortion that a
transponder's passband filter imparts on a single carrier. It multiplies the
carrier's baseband signal by a frequency-domain transfer function H(f) that has
two components:

```
H(f) = A(f) · exp(j·φ(f))
```

**Amplitude ripple:**

```
A(f) = 1 + r · cos(π · ripple_cycles · f_norm)    |f| ≤ signal_bw / 2
A(f) = 1                                            |f| > signal_bw / 2
```

**Phase nonlinearity:**

```
φ(f) = max_phase_dev_rad · |f_norm|^poly_order · sign(f_norm)^(poly_order % 2)
                                                   |f| ≤ signal_bw / 2
φ(f) = 0                                           |f| > signal_bw / 2
```

where `f_norm = f / (signal_bw / 2)` runs from −1 to +1 across the allocated
bandwidth, and `r = (10^(ripple_db/20) − 1) / (10^(ripple_db/20) + 1)`.

The DFT is zero-padded so that the multiplication is equivalent to linear (not
circular) convolution. The cosine ripple has implicit delay taps at
`±ripple_cycles / signal_bw` seconds; the padding covers those taps plus 8
samples of sidelobe margin, ensuring the wrap-around region stays in the zero
padding.

---

## Operation at native rate

The impairment is applied **at the carrier's native sample rate**
(`sps × symbol_rate`) *before* the OLA upsampler, operating on the full-length
baseband signal in one FFT multiply. This is the natural point to insert a
baseband-equivalent passband filter: the signal occupies roughly
`[−signal_bw/2, +signal_bw/2]` Hz, and H(f) matches that span.

---

## Why H(f) = 1 outside signal_bw is correct

For a carrier with `symbol_rate = 100 kHz` and `rolloff = 0.35`, the RRC
mainlobe spans `±67.5 kHz` (`signal_bw/2 = symbol_rate × (1 + rolloff) / 2`).
H(f) is unity outside that band.

This is not an approximation or an omission — it is a deliberate model choice:

1. **The RRC transmit filter already suppresses out-of-band energy by 40–60 dB.**
   There is negligible signal power outside `signal_bw` to distort.

2. **Real transponder filters roll off to unity gain (matched to the allocating
   input filter) outside the allocated slot.** The in-band distortion is what
   matters for link budget and BER. Applying arbitrary distortion in the
   transition region or beyond would model a physically implausible filter.

3. **Baseband-equivalent representation.** Any LTI filter centred on the carrier
   frequency can be described exactly by its complex baseband equivalent. H(f)
   here *is* that equivalent — defined only over the carrier's occupied bandwidth.
   The upsampler and frequency shift that follow preserve this equivalence.

---

## Genuine limitation: hard band edge

H(f) steps discontinuously from the in-band response to unity gain at exactly
`|f| = signal_bw / 2`. In the time domain this rectangular window on `H(f) − 1`
creates sinc-like ringing that extends a few symbol periods beyond the nominal
band edge.

For the ripple amplitudes this model targets (0.5 dB corresponds to
`r ≈ 0.028`), the ringing is small enough to be negligible: the impulsive
contribution is about 0.028 of the signal amplitude, or roughly −31 dBc. A real
transponder filter tapers the ripple smoothly toward the band edges, but the
difference in BER impact is negligible for any `ripple_db` value likely to be
configured.

The transition-band discontinuity matters only for very large ripple amplitudes
(e.g., `ripple_db > 3 dB`) or extremely high cycle counts where the cosine
sidelobes themselves become significant. In those cases the model is a reasonable
approximation, not an exact replica of any physical filter.

---

## Cross-carrier limitation

Each carrier's channel impairment is applied independently at its own baseband
before upsampling. There is no shared wideband filter that couples carriers: a
transponder filter spanning multiple carriers would affect all of them
coherently, potentially introducing inter-carrier crosstalk and correlated phase
shifts.

This model approximates the case where each carrier occupies its own slot in a
multi-carrier transponder and the per-slot filter dominates the impairment.
If a single wideband ripple spans multiple carriers — for example, a reflective
multipath that creates standing waves across the full transponder bandwidth — the
current model cannot capture the correlated distortion between carriers.
