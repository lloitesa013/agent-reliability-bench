# START HERE — B300 agent entry point

**Read `bench/B300_AGENT_BRIEF.md` first.** It is your task brief (Weeks 2-4).

## What's in this repo
- `bench/` — **SEALED** benchmark spec. `BENCH_SPEC.md`, `METRICS.md`, `CLAIMS.md` and
  `run_bench.py` are the official definition — **do NOT edit them, do NOT re-derive them.**
  `sample_bench.jsonl` is a 6-record smoke sample. `B300_AGENT_BRIEF.md` is your brief.
- `agent.py`, `diagnose.py`, `guardrail.py`, `hard_cases.py`, `evaluate.py` — the validated
  **prototype** pipeline (RAG agent + trace + groundedness guardrail + calibrated diagnose/abstain +
  crafted hard cases). It runs with a MOCK LLM offline; on B300 you swap in a real model.
  Reuse its patterns (labeling, guardrail decision, hard-case construction) — these encode
  bug-fixes already paid for. Do not lose them.
- `traces/`, `audit_log.json` — sample outputs from the prototype.

## Not yet in this repo (5090 box was offline at push time)
The real Qwen+bge data engine + watcher training from the RTX 5090
(`data_engine_v5.py`, `real_rag.py`, `make_balanced.py`, `train_watcher.py`, `train_eval_hard.py`)
will be added in a later commit. Until then, build the Week-2 data engine from the brief's
documented lessons: corrupted-value twins for distractors, strict token-overlap correctness
(NOT semantic similarity), an expanded DECLINE/abstain list, and fault injection. The SEALED
spec/metrics/win-conditions are fully present in `bench/` — nothing sealed is missing.

## Operating rules (from the operator)
- Everything runs **inside Docker**, base image NGC PyTorch `nvcr.io/nvidia/pytorch:25.xx`.
- **GPU 0 is off-limits.** Use GPUs 1-7 only.
- Scope for now: do env setup + Week-2 data generation, then STOP and report data quality
  (case balance 100/100/100, label audit, gold validation, groundedness distribution)
  BEFORE any fine-tuning. Look at the data before spending GPU on training.
- Report honestly per `bench/CLAIMS.md`. A miss is a finding. Never change the sealed spec.
