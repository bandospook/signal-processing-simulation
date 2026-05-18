---
name: feedback
description: User preferences and corrections about how to approach work
metadata:
  type: feedback
---

## Commit and push after each meaningful chunk

**Why:** User works in chunks and wants progress preserved at each natural boundary.
**How to apply:** After completing any self-contained feature (modulation tier, test suite, doc, bug fix), commit and push without waiting to be asked. Confirm first if the scope is ambiguous.

## Tests are highly valued

"I am really happy with your suggestion to add tests. The project is getting more complicated, so we need to be more aware of limitations of simple inspection."

**Why:** The project is growing in complexity; tests catch integration failures that inspection misses (e.g., APSK constellation bug, RMS normalisation requirement).
**How to apply:** Proactively suggest tests when adding new functionality. Tests should cover both unit correctness (zero BER noiseless) and physical correctness (BER matches AWGN theory).

## No Claude/Anthropic attribution in commits

Never include `Co-Authored-By: Claude` or any Anthropic attribution in git commit messages, across all projects.
**Why:** User explicitly rejected it.
**How to apply:** Omit the Co-Authored-By line from every commit message.

## Toolchain: use PowerShell and explicit venv paths on Windows

Never use the Bash tool for running project commands (python, pytest, pyright, ruff) on this Windows machine.
**Why:** The Bash tool runs Git Bash. Unqualified `python` hits the Windows Store stub ("Python was not found") and breaks tools. Pyright is Node-based — its internal Python discovery calls unqualified `python`, so even `python.exe -m pyright` fails unless pyright can find the venv independently via pyrightconfig.json.
**How to apply:**
- Use the PowerShell tool for all project commands.
- Run Python as `.venv\Scripts\python.exe -m <module>` (e.g., `-m pytest`).
- **pyright and ruff are compiled binary tools** — they install as `.exe` files in `.venv\Scripts\`, NOT as Python modules. Use `.venv\Scripts\pyright.exe` and `.venv\Scripts\ruff.exe` directly. Do NOT use `python.exe -m pyright` or `python.exe -m ruff`.
- Before running any tool, verify its `.exe` is present in `.venv\Scripts\` with: `ls .venv/Scripts/` via Bash. If missing, add it to `[dependency-groups] dev` in `pyproject.toml` and run `uv sync`.
- `pyrightconfig.json` must exist at repo root with `venvPath`/`venv` set so pyright finds packages when invoked any way.
- The README's "activate then `python -m pyright`" pattern is correct for a human interactive shell but does NOT work for Claude Code since venv activation doesn't persist between tool calls.

## Plots alongside tests

User wants BER vs Es/N0 with theory, EVM plots, and eye diagrams as part of the test suite — not just correctness assertions, but visual validation that the physics is right.
**How to apply:** When writing performance tests, always save plots to `plots/performance/` in addition to making assertions.
