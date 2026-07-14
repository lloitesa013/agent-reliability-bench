# START HERE

One system, two layers, one question: **when should an AI agent's output — or its own
self-modification — be trusted?**

## Layer 1 — the agent-reliability watcher (runtime; the production story)
Watches a RAG/LLM agent at inference time: trust scoring (groundedness × relevance),
**ABSTAIN** when unsure, **ESCALATE** to a human with a reason, audit log for every decision.
- Try it offline, no key, ~60 seconds: `python demo.py`
- Evidence: `RESULTS.md` **R1–R7** — zero-shot reading judge **0.946±0.014 vs 0.789** (tuned
  groundedness rule) on the sealed 300-trace benchmark with **0 unsafe passes**; calibrated
  abstention **0.84 → 0.91**; honest HaluBench field placement.
- Pieces: `guardrail.py` (the gate), `agent.py` / `diagnose.py` / `evaluate.py` (traced agent +
  failure-cause attribution with abstain), `bench/` (sealed spec: BENCH_SPEC / METRICS / CLAIMS),
  `traces_bench.jsonl` (300 records), `judge_zeroshot.py`, `calibrate2.py`, `halubench_eval.py`.

## Layer 2 — VSI-0, Verified Self-Improvement (the research frontier)
The same verification discipline pointed at the agent's OWN changes: every self-modification to
{prompt, rule, memory, tool} is UNTRUSTED until it clears a hidden test + regression check.
- Core artifact: `vsi_gate.py` (pure stdlib; `python vsi_gate.py --demo`)
- Evidence: `RESULTS.md` **R8–R31** — reward-hacks are the norm (R17: naive ships a broken tool 59%
  vs verifier 0%), replicated on real code (MBPP/HumanEval, R22/R24/R26), and closed embodied
  (CARLA: gate REJECTS a harmful plausible fix R25, ACCEPTS a real one R27–R31, p=6e-8).

**Read in this order:**
1. **`README.md`** — both layers + headline results.
2. **`STUDY.md`** — the front-page narrative (thesis, both directions, embodied, honest limits).
3. **`RESULTS.md`** — all results R1–R31 with exact numbers and the script that produced each.
4. **`NOTE.md`** / **`PAPER.md`** — technical note / workshop-format draft.

Provenance note: the project began as the Layer-1 watcher (the "B300"/sealed-benchmark phase);
VSI-0 grew out of it and stands on it. The two layers are one thesis at two levels of autonomy —
watch the agent's outputs, then gate the agent's self-modifications. `REWRITE_NOTE.md` documents a
2026-07-15 commit-metadata normalization (file trees unchanged; sealed-baseline SHA remapped there).
