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

## Setup

The project uses [uv](https://docs.astral.sh/uv/) to manage the Python version
and virtual environment.  First-time setup takes about a minute.

**1. Install uv** (if you don't have it already)

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**2. Clone and sync**

```bash
git clone https://github.com/bandospook/signal-processing-simulation.git
cd signal-processing-simulation
uv sync          # creates .venv/, downloads Python 3.14, installs all dependencies
```

`uv sync` reads `pyproject.toml` and installs numpy, matplotlib, pillow, scipy,
pytest, and pytest-cov into an isolated `.venv/` directory.  Nothing is installed
globally.

**3. Activate the virtual environment**

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```
```bash
# macOS / Linux
source .venv/bin/activate
```

Your shell prompt will change to show `(signal-processing-simulation)`.

**4. Run**

```bash
python main.py        # headless simulation; results written to output/
python gui.py         # GUI — load, edit, and run simulation.toml interactively
```

**5. Verify (optional)**

```bash
python -m pytest tests/ -v     # 117 tests, all should pass
python -m pyright sim/ main.py # 0 type errors expected (pre-existing unrelated)
python -m ruff check sim/ main.py gui.py   # 0 lint errors
```

> **IDE note (VS Code / PyCharm):** point your editor's Python interpreter at
> `.venv/Scripts/python.exe` (Windows) or `.venv/bin/python` (macOS/Linux) so
> that imports resolve correctly and the integrated test runner uses the right
> environment.

See [docs/GUIDE.md](docs/GUIDE.md) for a full walkthrough, configuration reference,
output descriptions, and test-suite documentation.

## Requirements

- Python 3.14 (downloaded automatically by uv; no manual install needed)
- numpy, matplotlib, pillow, scipy — all installed automatically by `uv sync`
