# User Profile

---
name: user-profile
description: Who the user is, their background, and how to collaborate effectively with them
metadata:
  type: user
---

Signal processing engineer with satellite communications background. Working in Python
but comes from C++ — occasionally surprised by Python syntax (e.g. `//` is floor division
not a comment, `tomllib` is read-only).

Thinks carefully about system-level correctness (link budgets, noise placement, confidence
intervals) before implementation. Prefers interactive design discussions ("tell me where
you have concerns") before committing to a plan. Values statistical rigour and wants
metrics to be well-defined.

Has some GUI experience but not an experienced GUI designer — open to suggestions on UX.

Prefers:

- Short, direct answers
- No trailing summaries after diffs (can read the diff)
- Code comments only for non-obvious WHY, not what
- Sniff-test level by default, with a single constant to flip for rigorous runs
- sim/ code first, GUI later — don't implement features twice
