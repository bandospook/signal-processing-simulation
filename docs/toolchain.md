# Toolchain and Development Environment

## Overview

This project uses [uv](https://docs.astral.sh/uv/) to manage the Python version and
virtual environment. On most systems — Linux, macOS, or a clean Windows install —
`uv sync` followed by activating `.venv` is all that is needed. This document records
the exceptions, explains why they exist, and documents the configuration files that
work around them.

---

## Standard setup (all platforms)

```bash
uv sync        # installs Python 3.14 + all dependencies into .venv/
```

`uv sync` reads `pyproject.toml` and installs both the runtime dependencies
(numpy, matplotlib, pillow, scipy) and the dev group (pytest, pytest-cov, pyright,
ruff) into an isolated `.venv/` directory. Nothing is installed globally.

---

## Correct invocations for each tool

Pyright and ruff are compiled binary tools. They are **not** Python modules and
cannot be invoked as `python -m pyright` or `python -m ruff`. After `uv sync` they
are available as executables in the venv's binary directory.

### Linux / macOS

| Task | Command |
|---|---|
| Tests | `.venv/bin/python -m pytest tests/ -v` |
| Type check | `.venv/bin/pyright gui.py main.py sim/ tests/` |
| Lint | `.venv/bin/ruff check gui.py main.py sim/ tests/` |

### Windows (PowerShell)

| Task | Command |
|---|---|
| Tests | `.venv\Scripts\python.exe -m pytest tests/ -v` |
| Type check | `.venv\Scripts\pyright.exe gui.py main.py sim/ tests/` |
| Lint | `.venv\Scripts\ruff.exe check gui.py main.py sim/ tests/` |

Use the explicit `.venv` path rather than relying on the activated environment so
that the correct interpreter is used regardless of shell state. This is particularly
important in automated contexts (CI, editors, scripted tool invocations) where
activation may not have occurred.

---

## Windows-specific complications

### The Windows Store Python stub

Windows ships an App Execution Alias that intercepts the unqualified `python`
command and redirects it to the Microsoft Store with the message
"Python was not found; run without arguments to install from the Microsoft Store."
This happens when no Python is installed globally — which is the intended state when
using `uv`, since `uv` manages its own Python copies and the project Python lives
only inside `.venv/`.

**Consequence:** any tool that internally calls unqualified `python` to discover the
active environment will fail or behave unexpectedly. Pyright does this at startup.

**Fix:** use explicit `.venv\Scripts\` paths (table above). `pyrightconfig.json` at
the repo root also tells pyright where the venv is, so it does not need to invoke
`python` to find it.

### Pyright venv discovery on Windows

Even when pyright is invoked via `.venv\Scripts\pyright.exe`, it attempts to locate
the Python interpreter by calling `python` on PATH. On Windows with the Store stub
active, this prints a spurious "Python was not found" warning to stderr but does
**not** prevent pyright from running. The warning is cosmetic — pyright falls through
to the `venvPath`/`venv` settings in `pyrightconfig.json` and finds the packages
correctly.

The warning can be suppressed entirely by adding the `.venv\Scripts\` directory to
PATH, but this is not required for correct operation.

### Shell choice on Windows

Claude Code (and other automated tools) run commands in a shell. On Windows,
Git Bash is available but behaves inconsistently with Windows paths and `.exe`
resolution. Use PowerShell for all project commands to avoid ambiguity.

---

## pyrightconfig.json

`pyrightconfig.json` at the repo root contains:

```json
{
  "venvPath": ".",
  "venv": ".venv"
}
```

This tells pyright that the virtual environment is at `.venv/` relative to the
project root, regardless of which Python is on PATH. It is required on Windows (Store
stub problem) and is harmless on Linux and macOS where pyright can usually discover
the venv automatically.

---

## Linux / macOS: what changes

On Linux and macOS:

- The unqualified `python3` command is a real interpreter (either the system Python or
  a uv-managed one). There is no Store stub.
- Pyright discovers the venv via `python3` without needing `pyrightconfig.json`,
  though the file does no harm.
- The binary directory is `.venv/bin/` instead of `.venv\Scripts\`.
- The executables have no `.exe` extension: `.venv/bin/pyright`, `.venv/bin/ruff`.
- Any shell (bash, zsh) works correctly.

All other toolchain behaviour — `uv sync`, `pyproject.toml` structure, `pyrightconfig.json`
— is identical across platforms.

---

## Adding or removing dev tools

Dev tools (pyright, ruff, pytest, pytest-cov) are declared in `pyproject.toml` under
`[dependency-groups] dev`. To add a tool:

```toml
[dependency-groups]
dev = [
    "pytest>=9.0.3",
    "pytest-cov>=7.1.0",
    "pyright>=1.1",
    "ruff>=0.9",
    "new-tool>=x.y",   # add here
]
```

Then run `uv sync` to install it. The tool will appear as an executable in
`.venv/Scripts/` (Windows) or `.venv/bin/` (Linux/macOS). Do not install dev tools
with `pip install` — they will not be tracked in `pyproject.toml` and will be
removed by the next `uv sync`.

---

## Code quality gates

The quality gates that must pass before every commit are:

```
pyright: 0 errors
ruff:    0 errors
pytest:  all tests passing
```

Run them with the commands in the table above for your platform. CLAUDE.md contains
the Windows PowerShell form as the reference for automated use.
