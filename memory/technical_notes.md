---
name: technical-notes
description: Non-obvious implementation decisions, bug fixes, and key formulas
metadata:
  type: project
---

## RMS normalisation before nearest-neighbour decision

`receive()` in `sim/receiver.py` normalises received symbol samples to unit RMS before calling `decide()`. This is **required** for multi-amplitude constellations (16QAM, 16APSK, 32APSK). The baseband generator normalises the full signal's RMS (not individual symbol amplitudes), so after the matched filter the samples are scaled by a factor k ≠ 1 relative to the unit-power constellation. Without normalisation, decision boundaries are misplaced and BER is catastrophic.

PSK constellations (BPSK, QPSK, 8PSK) are immune because all symbols are equidistant from the origin — scale doesn't shift which quadrant a symbol lands in.

## APSK 12-point rings must use natural angular order

`_gray_decode(k)` for k in 0..11 produces values in 0..14, not 0..11. Values ≥ 12 wrap (e.g., gray_decode(8)=12 → 12×π/6=2π=0°, a duplicate). This creates only 8 unique outer ring points instead of 12, causing 4 bit patterns to be unreachable from nearest-neighbour decisions → catastrophic BER.

Fix: use `np.arange(12) * np.pi / 6` (natural angular order) for all 12-point rings (16APSK outer, 32APSK mid). Gray coding only applies to power-of-2 ring sizes.

## Circular convolution in apply_channel_impairment

The DFT multiplication in `apply_channel_impairment` (sim/filters.py) implements circular convolution. A cosine ripple in frequency = a delay tap in time; without zero-padding, that tap wraps around. Fixed by zero-padding to length `ceil(ripple_cycles * sample_rate / signal_bw) + 8` before the DFT.

## RRC filter normalisation and the Es/N0 noise formula

The RRC filter from `rrc_coeffs` has unit energy: `sum(h**2) = 1`.

For baseband normalised to unit RMS power, with symbol_rate = 1, sample_rate = sps:
- Es = signal_power / symbol_rate = 1
- N0 = 2*sigma_c^2 / sample_rate = 2*sigma_c^2 / sps
- Es/N0 = sps / (2*sigma_c^2)

To add AWGN at a desired Es/N0:
```python
sigma_c = np.sqrt(sps / (2.0 * EsN0_linear))
noise = sigma_c * rng.standard_normal(N) + 1j * sigma_c * rng.standard_normal(N)
```

After the matched filter pair (RRC ⊗ RRC = raised cosine), symbol amplitude at decision = sqrt(sps). BER_BPSK = Q(sqrt(sps)/sigma_c) = Q(sqrt(2*EsN0)) = Q(sqrt(2*EbN0)) ✓.

The receiver's RMS normalisation doesn't affect SNR (it scales both signal and noise equally).

## Phase ambiguity resolution

`_ber_with_ambiguity` (sim/receiver.py) tries all N rotationally equivalent orientations of received samples and returns the minimum BER. This handles systematic AM-PM phase shift without carrier recovery.

`rotational_symmetry` values: BPSK=2, DBPSK=1, QPSK=4, OQPSK=4, 8PSK=8, 16QAM=4, 16APSK=4, 32APSK=4.

DBPSK=1 because differential decoding already removes the 180° flip.

## OQPSK baseband and reception

TX: I and Q rails are pulse-shaped separately; Q is delayed by sps//2 samples before combining.
RX: Sample I at `mf.real[0::sps]` and Q at `mf.imag[sps//2::sps]`, then combine into complex.

## scipy is NOT a dependency

Use `math.erfc` or `numpy`-only Q-function: `Q = lambda x: 0.5 * erfc(x / sqrt(2))`.
```python
from math import erfc
import numpy as np
def _q(x): return 0.5 * erfc(float(x) / np.sqrt(2))
```
For array inputs, use a vectorized version.
