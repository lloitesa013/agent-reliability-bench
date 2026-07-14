# Agent Reliability Watcher × VSI-0 (Verified Self-Improvement)

One system, two layers, one question: **when should an AI agent's output — or its own
self-modification — be trusted?**

**Layer 1 — the watcher (runtime reliability; the production story).** Watches a RAG/LLM agent at
inference time, scores trust, **ABSTAINS** when unsure, **ESCALATES** to a human with a reason, and
writes an audit log. Headline (sealed 300-trace benchmark, fact-group split, `RESULTS.md` R1–R7): a
zero-shot *reading* judge beats a tuned groundedness rule — **effective reliability 0.946±0.014 vs
0.789, 0 unsafe passes** — and calibrated abstention lifts accuracy **0.84 → 0.91** by routing the
least-confident 30% to humans. Try it offline in 60 seconds: `python demo.py`.
> Lesson kept on purpose: an early version of the watcher's own demo shipped a grounded-but-irrelevant
> answer at full trust in its own audit log. Lexical grounding alone is not a safety signal — that
> failure motivated the relevance gate in `guardrail.py` and the benchmark's distractor class.

**Layer 2 — VSI-0 (the research frontier).** The same verification discipline pointed at the agent's
OWN changes: a *self-verifying agent* treats every self-modification to its {prompt, rule, memory,
tool} as UNTRUSTED and adopts it only if it clears a HIDDEN TEST (transfers to held-out cases) AND a
REGRESSION check (breaks no existing capability). Built and validated on one RTX 5090, in text and in
an embodied driving domain (CARLA).

**Why it matters.** Self-improving-agent methods (Reflexion, ExpeL, …) report gains measured on the same
tasks they learned from — and in April 2026 UC Berkeley showed all 8 major agent benchmarks are
reward-hackable to ~100%. A self-improvement loop *without a trustworthy verifier* converges to a confident
reward-hacker, not to improvement. **The verifier — the thing that separates real improvement from fake — is
the missing, load-bearing piece.** VSI-0 builds it and stress-tests it.

## Headline results (see `RESULTS.md` for all, R1–R31)

| what | result |
|---|---|
| **Tool/code self-mods, statistical (R17)** | Across 24 tool-rewrite decisions a reward-hack existed in **100%**; selecting by seen-score (naive) deploys a broken tool **59%** of the time, the executing verifier **0%** — while still improving in **88%**. |
| **REAL code, MBPP + HumanEval (R22/R24/R26)** — closes the toy→real gap, replicated | Across two real benchmarks (MBPP n=100, HumanEval n=80) a genuine overfit appears in **29% / 30%** of problems; where it exists, deploying by the shown test (naive) ships it **19/29 and 21/24 times — the verifier 0 on both**. On solvable problems naive ships a held-out-failing solution **12% on both**, the executing verifier **0%**. Near-identical rates on independent benchmarks: not a dataset artifact. |
| **Flagship self-modification (R13)** | An agent fixes a new task (0→0.83) while keeping its old one, **rejecting even the edit that fixes the new task best because it regresses the old** → capability **0.50 → 0.92** (naive 0.67). |
| **Regression caught (R12)** | Naively fine-tuning to fix one source silently forgets another by **−0.16**; a 1-D "did the target improve?" check misses it, the 2-D verifier catches it. |
| **Embodied (R19–R21, CARLA)** | Re-running real routes N times: route 11755 fails **7/12 = 58%** (genuinely flaky); routes labeled FAILURE by the single-run Bench2Drive taxonomy **pass 100% on re-run** → **single-run failure labels are unreliable**; verification must be a failure RATE. |
| **ACCEPTED embodied fix (R27–R31)** — the loop closes on real driving | Student fails 11755 at 58% → expert demos collected → 10-epoch retention-DAgger fine-tune → pre-registered gate: **0/19 fresh rollouts fail (p=6e-8), retention routes 0/12** → **ACCEPT**. Same harness earlier **REJECTED** a harmful plausible fix (R25). Honest scope: same-route repair + 3-route regression set, not cross-route generalization. |

## What is robust vs not (honest — this is the point)
- **Robust:** the **single adoption decision**. The verifier reliably accepts real transfer and rejects the
  overfit fine-tune, the −0.16 regression, the best-fix-that-regresses, and lookup-hacks that all fool the
  naive agent (R11–R13, R15, R17) — and this holds on **real code** (MBPP, R24: of the 29% of problems with a
  genuine overfit, naive ships it 19/29 vs verifier 0; 12% vs 0% on solvable problems), not just toy
  conventions. Covers all four of {prompt, rule, memory, tool}.
- **Not robust (honest):** the **multi-round cumulative** advantage is small/noisy at scale (R14→R16, final
  mean Δ≈+0.03) — the per-decision guarantee does not (yet) compound into a large cumulative gap on toy tasks.
- **Open hard core:** the **embodied self-improvement loop** — a fix that provably lowers the failure rate —
  is not solved; failures are flaky (need statistical verification) and a real fix needs retention-DAgger
  (a multi-week build). VSI-0 delivers the embodied *verifier*, not the embodied *fix*.

## Repo map
- **`STUDY.md`** — start here: the front-page narrative (thesis, both directions, embodied, honest limits).
- **`NOTE.md`** — technical note / mini-paper (problem, method, results, related work, limitations).
- **`RESULTS.md`** — authoritative results, R1–R31, each with the exact numbers and the script that made them.
- Text self-verifying agent: `self_verify_agent.py` (flagship), `self_improve_loop.py` /
  `self_improve_loop6.py` (multi-round), `self_verify_tool.py` + `stat_robustness.py` (tool/code + statistics),
  `candidate_ft.py` (fine-tuning level), `verified_integration.py` (regression), `fresh_vs_reused.py`
  (why the hidden test must be fresh), `proof_*.py` (the original accept/reject/autonomous proofs).
- **Layer 1 — the watcher (runtime reliability):** `demo.py` (60-second offline tour), `guardrail.py`
  (PASS/SAFE/ESCALATE gate: groundedness × relevance + audit log), `agent.py`/`diagnose.py`/`evaluate.py`
  (traced agent + failure-cause attribution with abstain), `judge_zeroshot.py`, `calibrate2.py`,
  `halubench_eval.py`, `cross_source*.py`, `bench/` (sealed benchmark) + `traces_bench.jsonl` (300 records).
- Embodied: `run_multi.ps1` (statistical failure-rate verifier on CARLA/Bench2Drive), `carla/` (watcher/judge).

## Use it — `vsi_gate.py` (the verifier as a library)
Pure stdlib, no dependencies. Wrap ANY candidate self-modifications (prompts, rules, memory entries, tool
code, checkpoints) with your own evaluators; the gate ACCEPTs only what clears held-out gain + no
regression, else it abstains. `gate_rate()` is the statistical form for flaky/stochastic domains (verify by
failure RATE — a single run is noise). Deterministic demo, no LLM needed:
```bash
python vsi_gate.py --demo
```

## Reproduce
Text results need a 7B instruct model on one GPU (`real_rag.py` loads `Qwen/Qwen2.5-7B-Instruct`) plus
`transformers`, `peft`, `datasets`. Each `RESULTS.md` entry names its script; run e.g.
`python self_verify_agent.py`, `python stat_robustness.py`. Embodied results need the LEAD/CARLA stack
(`run_multi.ps1`, native Windows). All numbers here were produced on a single RTX 5090.

## Claims / non-claims (honest)
- **We claim:** a hidden-test + regression verifier reliably wins the single self-modification decision
  across {prompt, rule, memory, tool}, in text and (as a statistical failure-rate verifier) in embodied
  driving; and that fake/overfit/hack self-modifications are the *norm*, not the exception.
- **We do NOT claim:** SOTA raw detection, large multi-round cumulative gains, a solved embodied
  self-improvement loop, or results beyond a 7B model + controlled/benchmark substrates. The honest
  negatives are reported as first-class results.

MIT.
