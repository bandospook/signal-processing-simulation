# SO-WAT — Claude Code project instructions

## Memory location

All project memory lives in `memory/` in this repo root. It is committed to git
and is the single source of truth across all machines.

**At the start of every session, read these four files before doing anything else:**

1. `memory/context.md` — what this project is, repo layout, GUI notes, key config fields
2. `memory/technical_notes.md` — non-obvious implementation decisions; read this before touching sim/ code
3. `memory/feedback.md` — user preferences and things to avoid repeating
4. `memory/project_state.md` — current state, what is done, and the immediate next task

**When saving memory,** write to `memory/` and commit. Do not write to `~/.claude/` or
`.claude/memory/` — those are machine-local and will be lost on a new machine.

---

## Toolchain

- Package manager: `uv`; venv at `.venv/`; run `uv sync` after editing pyproject.toml
- Python (Windows): `.venv\Scripts\python.exe`
- Python (Linux/Mac): `.venv/bin/python`
- Tests: `.venv\Scripts\python.exe -m pytest tests/ -v`
- Type check: `.venv\Scripts\pyright.exe gui.py main.py sim/ tests/`
- Lint: `.venv\Scripts\ruff.exe check gui.py main.py sim/ tests/`
- Line length: 110 chars (configured in pyproject.toml)

**Shell: always use PowerShell (not Bash) for project commands on Windows.**
Unqualified `python` hits the Windows Store stub on this machine, breaking both pytest and
pyright's internal venv discovery. Always use full `.venv\Scripts\` paths.
`pyrightconfig.json` at repo root points pyright at the venv so it finds packages.

## Code quality gates (run before every commit)

Pyright 0 errors, Ruff 0 errors, all tests passing.

## Commit style

Commit and push after each self-contained chunk of work. The user does not want to
ask — just do it at natural completion points.

**Never include `Co-Authored-By: Claude` or any Anthropic attribution in commit messages.**
This applies to every commit in this project, no exceptions.
