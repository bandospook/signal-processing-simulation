# Memory lives in the project repo

All project memory is committed to git at `memory/` in the project root.
It travels with the repo and survives machine changes.

At the start of every session, read these files:

- `memory/context.md` — project overview, repo layout, GUI notes, config reference, open work items
- `memory/technical_notes.md` — non-obvious implementation decisions (AWGN placement, formulas, NLA normalization, seeker algorithm)
- `memory/feedback.md` — user preferences and corrections
- `memory/project_state.md` — current implementation state and next task

Do NOT write memory to this folder or to the user-level ~/.claude/projects/.../memory/.
Write updates to `memory/` in the project root and commit them.
