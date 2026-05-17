# Technical Notes

---
name: technical-notes
description: Signal chain decisions, theory formulas, and key implementation details
metadata:
  type: project
---

## AWGN placement — noise after the amplifier

AWGN is added AFTER the nonlinear amplifier in `sim/simulation.py`.
This models a single-hop satellite downlink where thermal noise is primarily at the
receiver. The uplink noise contribution is handled as a separate link budget item
(reciprocal-sum C/N: 1/(C/N)_total = 1/(C/N)_UL + 1/(C/N)_DL + 1/(C/N)_IM).

**Why:** If noise is placed before the amp, C/I shifts with noise level, breaking the
"C/I constant" assumption needed for the BER seeker. Noise after amp means C/I is
determined solely by IBO and signal statistics — exactly what we want.

## DBPSK theory formula — coherent detection + differential decoding

`ber_theory("DBPSK", EsN0_dB)` uses: BER = 2 * p * (1 - p), p = 0.5 * erfc(sqrt(Eb/N0))

NOT the differentially-coherent formula `0.5 * exp(-Eb/N0)`.

**Why:** The simulation uses a coherent RRC matched filter followed by differential
decoding of hard symbol decisions. A single symbol error flips two decoded bits, giving
BER ≈ 2p. The `exp(-Eb/N0)` formula applies to a receiver that makes decisions purely
from phase differences between consecutive symbols (no matched filter) — about 3 dB worse.

## CNIR-to-Eb/N0 conversion in targeter.py (IMPORTANT: includes sps factor)

CNIR in `simulation.py` is measured at the **native sample rate** (sps samples/symbol),
BEFORE matched filtering. Converting to Eb/N0 per bit requires:

```
Eb/N0 = CNIR * sps / bps
effective_ebn0_db = cnir_db + 10*log10(sps / bps)
```

**Why:** p_noise at native rate = noise_density * native_rate = noise_density * sps * symbol_rate.
But theory Eb/N0 = signal_power / (bps * symbol_rate * noise_density).
Dividing: Eb/N0 = CNIR * sps / bps. For BPSK sps=4: +6 dB correction.
Forgetting the sps factor causes systematic -6 dB implementation loss on a linear amplifier.

## Implementation loss definition

Effective Eb/N0 = C/(N+I) * sps/bps (see CNIR conversion above).
Implementation loss = effective_Eb/N0_dB - theory_Eb/N0_dB(at measured BER).

**Why C/(N+I):** With no nonlinearity (I=0), effective Eb/N0 = C/N*sps/bps and BER matches
theory → loss = 0. Residual loss beyond N+I captures non-Gaussian character of IM.

## BER seeker — adaptive bisection (sim/targeter.py)

Bisects noise_density_dbfs to achieve target BER for a named carrier.
Higher noise_dbfs → higher BER. Bracket: noise_lo (quiet, BER < target) to
noise_hi (loud, BER > target). ValueError if bracket is invalid.

Algorithm:

- Bracket check: 1 seed, n_bits_initial = max(500, n_bits_final//32)
- Bisection: N_bits doubles every 2 steps; stop when hi-lo < 0.05 dB
- Final: pool n_final_seeds seeds at converged noise; report normal-approx CI
- N_bits formula: N = ceil((z/accuracy)^2 * p*(1-p)), z = sqrt(2)*erfinv(confidence)
  Note: math.erfinv not in Python stdlib; _erfinv() bisects math.erf instead.

Per-carrier: each carrier with sweep_demod=True gets its own independent seek.
Carriers with sweep_demod=False still contribute to the wideband IM environment.

## BPSK / QPSK / OQPSK share the same theory curve

All three use BER = 0.5 * erfc(sqrt(Eb/N0)). The BER plot deduplicates the dashed
theory line — draws it once in BPSK's colour (C0) via `_BPSK_EQUIV` set.

## Filter span convention

`filter_span` in carrier config = RRC filter HALF-span in symbols.
Total filter length = filter_span * sps + 1 taps.

## Test symbol-count strategy

`_N_BITS_PLOT` (in `tests/test_awgn_performance.py`) is the single knob:

- 10_000 → ~4 s sniff test
- 1_000_000 → ~2 min rigorous run (±0.001 BER at 95% CI, worst-case p=0.5)

Both `test_ber_theory_table` and `test_generate_performance_plots` use it via
`n_sym = _N_BITS_PLOT // bps` so all modulations see equal bits.

## np.real() / np.imag() instead of .real / .imag

Use function form throughout the codebase — Pylance's numpy stubs have broken
overloads for the property form on `ndarray[Any, dtype[Any]]`.
