# Wideband BPSK Nonlinear Amplifier Simulation

A Python simulation for analysing digital communications channels that experience nonlinear amplifier distortion. Multiple carriers of varying symbol rates and bandwidths share a single power amplifier, allowing you to study the carrier-to-noise ratio (CNR), carrier-to-interference ratio (CIR), combined CNIR, EVM, and bit-error rate (BER) as functions of amplifier drive level and noise floor.

## Key capabilities

- **N-carrier composite**: any number of BPSK carriers, each with its own symbol rate, centre frequency, power level, and optional passband channel impairment (amplitude ripple + phase nonlinearity).
- **Memoryless nonlinear amplifier**: user-supplied AM-AM and AM-PM lookup tables; input backoff (IBO) sets the peak drive level relative to saturation.
- **Full receiver chain**: RRC matched filter, symbol sampling, hard BPSK decisions, BER, and RMS EVM.
- **Rigorous metrics**: CNR, CIR, and CNIR computed per carrier via a complex projection that separates true in-band IM distortion from AM-AM compression and AM-PM rotation.
- **2-D parameter sweep**: automated grid over IBO × noise density with performance plots.
- **TOML configuration**: all simulation parameters live in `simulation.toml`; no code changes needed to explore new scenarios.

## Quick start

```powershell
# 1. Activate the project virtual environment
.\.venv\Scripts\Activate.ps1

# 2. Run the simulation (single operating point + optional sweep)
python main.py

# 3. Inspect the console metrics table and the output PNGs
```

See [GUIDE.md](docs/GUIDE.md) for a full walkthrough, configuration reference, and annotated example results.

## Requirements

- Python ≥ 3.14 (managed by [uv](https://github.com/astral-sh/uv))
- numpy ≥ 2.4, matplotlib ≥ 3.10
