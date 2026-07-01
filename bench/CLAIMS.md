# CLAIMS / NON-CLAIMS (SEALED 2026-06-27)

## What this benchmark can claim (if the numbers hold)
- On a **300-trace, 3-domain, 3-case-type** synthetic-but-labeled benchmark, adding a
  calibrated **watcher / gate / evidence** layer on the SAME base LLM changes the system's
  **effective reliability** (unsafe-pass vs overblock trade-off) by a measured amount.
- Specifically whether: E > A/B/C/D on effective_reliability; a *trained* watcher (D) beats a
  *rule* (C) on reasoning; value-mismatch + evidence (E) beats D on distractor.

## NON-claims (read these)
- **Not** a claim about base-model capability — the base LLM is held fixed; we only measure the
  reliability *layer*.
- **Not** production performance — the data is **synthetic** (LLM-generated + templated + value-controlled),
  single base model, one compute node. Real regulated-domain data would differ.
- **Not** general SOTA — this is a scoped, self-defined benchmark, not the Who&When leaderboard.
- **Not** a safety guarantee — a passing answer is not certified correct; escalation reduces, not
  eliminates, risk. Human review remains in the loop.
- Numbers are mean ± std over 5 splits on a small dataset; treat gaps smaller than the std as noise.

## Discipline
- The spec, metrics and win conditions were **sealed before results were seen** (`BENCH_SPEC.md`).
- Results are reported whether or not they support the thesis. A miss is a finding.
