# Feedback

---
name: feedback
description: Coding preferences and approaches the user has confirmed or corrected
metadata:
  type: feedback
---

## No trailing summaries after making code changes

**Why:** User can read the diff.
**How to apply:** End responses with one or two sentences max after edits.

## sim/ first, GUI later

**Why:** User doesn't want to implement features twice — get the logic right in sim/
with tests, then wire into the GUI.
**How to apply:** When adding new simulation features, don't touch gui.py until the
sim-layer code and tests are solid and confirmed by the user.

## Single constant to toggle sniff-test vs rigorous run

**Why:** Tests should be fast by default; user wants to be able to flip one number
to go rigorous without hunting through multiple test functions.
**How to apply:** Use a module-level constant (like `_N_BITS_PLOT`) that all related
tests reference.

## Equal bits across modulations (not equal symbols)

**Why:** Higher-order mods have more bits per symbol, so equal n_sym gives unequal
statistical confidence. Equal n_bits = equal confidence.
**How to apply:** n_sym = n_bits // bps for each modulation.

## Interactive design discussion before implementation

**Why:** User thinks carefully about system-level correctness and wants to catch
inconsistencies before writing code.
**How to apply:** For non-trivial features, present the plan, raise concerns, ask
clarifying questions, and wait for confirmation before coding.

## No Co-Authored-By or AI attribution in commits

**Why:** User wants to avoid ownership disputes and doesn't want to explain AI involvement.
**How to apply:** Never include "Co-Authored-By: Claude..." or any Anthropic/AI attribution
in commit messages. Strip it if it appears.

## Comments in code: WHY only, not WHAT

**Why:** Well-named identifiers explain what. Comments should explain non-obvious
constraints, workarounds, or subtle invariants.
**How to apply:** Don't narrate the code. Do explain things like the DBPSK formula
choice or the noise-placement rationale.

## Mermaid over ASCII for flow diagrams in docs

**Why:** User confirmed preference; cleaner, rendered natively on GitHub.
**How to apply:** Use `flowchart LR` with `direction TB` inside subgraphs (not `TD`)
so pipelines fill page width as horizontal columns. Use `<br/>` not `\n` for label
line breaks. Apply `classDef` colour blocks — blue/amber/green is the established
palette for TX/composite/RX stages. Exception: 2-D spatial UI mockups (like the
GUI layout in §11) stay as ASCII since Mermaid has no spatial-layout primitive.

## Documentation: blurb sections for reference docs

**Why:** User wants a single entry point (GUIDE.md) with enough context to know
whether to open a deeper reference doc, not just a bare link.
**How to apply:** When adding a new doc to docs/, add a numbered section in GUIDE.md
with 2–4 paragraphs summarising key findings, trade-offs, and when to read it.

## Memory files belong in the project repo

**Why:** User works across physical machines; user-level `.claude/projects/` is machine-local.
**How to apply:** Keep memory files in `.claude/memory/` inside the project root so they
travel with the git repo and are accessible from any machine. The user-level auto-memory
path (`~/.claude/projects/.../memory/`) is a secondary copy; the project-level files
are the source of truth.
