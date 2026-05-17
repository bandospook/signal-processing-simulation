# Wideband Nonlinear Amplifier Simulation

A Python simulation for analysing digital communications channels that experience
nonlinear amplifier distortion. Multiple carriers of varying modulations, symbol rates,
and bandwidths share a single power amplifier, allowing you to study carrier-to-noise
ratio (CNR), carrier-to-interference ratio (CIR), combined CNIR, EVM, and bit-error
rate (BER) as functions of amplifier drive level and noise floor.

## Key capabilities

- **Multi-modulation, N-carrier composite** — BPSK, DBPSK, QPSK, OQPSK, 8PSK,
  16QAM, 16APSK, 32APSK; each carrier independently configured with its own symbol
  rate, centre frequency, power level, and modulation.
- **Memoryless nonlinear amplifier** — user-supplied AM-AM and AM-PM lookup tables;
  input backoff (IBO) sets the peak drive level relative to saturation.
- **Realistic satellite link model** — AWGN added after the amplifier (receiver
  thermal noise dominant); uplink noise handled as a separate link-budget item.
- **Full receiver chain** — RRC matched filter, symbol sampling, hard decisions,
  phase-ambiguity-resolved BER, and RMS EVM.
- **Rigorous per-carrier metrics** — CNR, CIR, and CNIR computed via complex
  projection that correctly separates AM-AM compression from true IM distortion.
- **2-D parameter sweep** — automated grid over IBO × noise density; per-carrier
  results in a markdown report and a multi-panel PNG.
- **Selective demodulation** — `sweep_demod` flag per carrier skips the expensive
  demod step for carriers not under test while still including them in the composite.
- **TOML configuration** — all parameters in `simulation.toml`; no code changes
  needed to explore new scenarios.
- **GUI** — standalone `gui.py` (tkinter): load/edit/save `simulation.toml` and
  launch `main.py` from inside the editor.

## Quick start

```powershell
# 1. Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# 2. Run the simulation
python main.py

# 3. Or open the GUI
python gui.py
```

See [docs/GUIDE.md](docs/GUIDE.md) for a full walkthrough, configuration reference,
output descriptions, and test-suite documentation.

## Requirements

- Python ≥ 3.11 (managed by [uv](https://github.com/astral-sh/uv))
- numpy, matplotlib (installed automatically via `uv sync`)
