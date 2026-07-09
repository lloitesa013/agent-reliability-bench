# START HERE → VSI-0

This repo is **VSI-0 — Verified Self-Improvement (v0)**: a self-verifying agent that gates its own
{prompt, rule, memory, tool} changes through a hidden-test + regression verifier.

**Read in this order:**
1. **`README.md`** — what VSI-0 is + headline results.
2. **`STUDY.md`** — the front-page narrative (thesis, both directions, embodied, honest limits).
3. **`RESULTS.md`** — all results R1–R21 with exact numbers and the script that produced each.
4. **`NOTE.md`** — the technical note / mini-paper.

**Legacy (historical context, not the current front door):** this project began as an agent-reliability
watcher (the "B300"/sealed-benchmark phase). `bench/` (sealed spec), `agent.py`, `diagnose.py`,
`guardrail.py`, `DATA_REPORT.md` and the earlier reliability scripts are that substrate — the watcher VSI-0
stands on (`RESULTS.md` R1–R7). They are kept for provenance; they are not the current thesis.
