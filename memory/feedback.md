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

## Plots alongside tests

User wants BER vs Es/N0 with theory, EVM plots, and eye diagrams as part of the test suite — not just correctness assertions, but visual validation that the physics is right.
**How to apply:** When writing performance tests, always save plots to `plots/performance/` in addition to making assertions.
