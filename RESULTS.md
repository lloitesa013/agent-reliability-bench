# RESULTS — first honest A-E score (conflict-fix data, 5090, 2026-07-01)

Sealed benchmark (`bench/`), regenerated data (conflict retrieval + `group` field), scored with the
SEALED `bench/run_bench.py`. Honest report — a miss is a finding.

## Setup
- Base LLM (fixed for A-E): **Qwen2.5-7B-Instruct** (fits one RTX 5090; 14B/32B are a later "best-result" run).
- Retriever: bge-small-en-v1.5. Data: 300 traces, `direct/reasoning/distractor` 100/100/100, 3 domains.
- Distractor = **conflict retrieval** (both correct + corrupt doc retrieved, len 2) → wrong-but-grounded
  is honestly detectable and no longer degenerate (correct/wrong = 33/67, not 0/100).
- Protocol: **fact-group 5-split** (split by `group`, NOT per-record → anti-leak per old DATA_REPORT §8-4);
  thresholds chosen on train; metrics on eval; mean±std over 5 seeds.

## Result 1 — Tier-1 embedding watcher: NULL (ties the rule)
`train_watcher_w3.py`: bge(query+evidence+answer) + groundedness → logistic → P(escalate).
```
sys | eff_rel        unsafe_pass  overblock
C   | 0.789±0.037   0.148        0.357     (tuned groundedness rule)
D   | 0.787±0.017   0.128        0.411     (embedding watcher)
```
**D does NOT beat C** (0.787 vs 0.789; D reasoning 0.700 < C 0.726). A compressed trace embedding
cannot extract the residual signal (verify arithmetic; detect conflict) → it just replicates the rule.

## Result 2 — Tier-2-lite zero-shot READING judge: BEATS the rule
`judge_zeroshot.py`: the 7B READS each trace and decides PASS/ESCALATE (no training).
```
sys | eff_rel        unsafe_pass  overblock    dec_acc
C   | 0.789±0.037   0.148        0.357        0.692    (rule)
D   | 0.946±0.014   0.000        0.180        0.867    (zero-shot judge)  <- beats C by +0.157
per case_type eff_rel:   direct / reasoning / distractor
  C   1.000 / 0.726 / 0.882
  D   1.000 / 0.918 / 0.880
```
**D beats C, +0.157 (≈4× std). The win is on REASONING (0.918 vs 0.726):** the judge recomputes the
arithmetic → catches wrong answers AND passes correct-but-ungrounded ones the rule over-escalates.
Distractor is a tie (both handle it). The embedding watcher couldn't do this; the *reading* judge can.

## Verification (`verify_judge.py` — good numbers → look at the data)
```
confusion (caught_wrong / UNSAFE_pass / overblock / clean_pass)
ALL        : 74 / 0 / 48 / 178
direct     : 0 / 0 / 0 / 100      distractor: 67 / 0 / 17 / 16      reasoning: 7 / 0 / 31 / 62
UNSAFE PASSES: 0   (judge escalated EVERY wrong answer)
```
Judge rationale is sound & case-appropriate: direct→pass, reasoning→verify-arithmetic, distractor→
detect-conflict. Sample decisions confirm it (e.g. "evidence conflicting values → ESCALATE").

## Honest caveats (read before quoting any number)
1. **Zero-shot self-judge** — the judge is the same 7B as the agent. A separate/larger judge could
   differ. "trained watcher" here is actually a *reading judge with no fine-tune* (cheaper; a plus).
2. **unsafe_pass=0.000 is clean partly because the data is synthetic** — wrong answers are declines /
   wrong-values / wrong-arithmetic, which reading catches. Real-world confident-plausible errors are
   harder. This is a benchmark number, **not a production guarantee** (`CLAIMS.md`).
3. **Part of the 0.180 overblock is appropriate caution, not error** — escalating an answer drawn from
   self-contradicting evidence is a reasonable call even when the answer happens to be right.
4. **Distractor "win" is crude conflict-abstention** (escalate on conflict) — ties C, not a clear win.
5. Synthetic/templated, one base model, one node — not production, not base-model capability, not SOTA.

## Bottom line
**First honest positive: a reading judge (zero-shot) beats a tuned groundedness rule on the sealed
benchmark (0.946 vs 0.789), verified by inspection, driven by arithmetic verification on reasoning.**
The embedding watcher tied — so *reading the trace* is the load-bearing ingredient, as predicted.

## Next
- **Calibration / abstention** (our differentiator): make the judge emit confidence → measure ECE +
  risk-coverage + an abstain band. Not yet done (current judge is binary).
- Fine-tuned judge vs zero-shot (is training worth it?). Public benchmarks (RAGTruth/HaluBench) for field-level level.
