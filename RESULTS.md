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

## Result 3 — calibrated abstention (the differentiator): DEMONSTRATED
`self_consistency.py` (K=5 sampling) FAILED as an uncertainty signal — the judge is 98% self-
consistent (confidently wrong on its errors) → abstain captures 2%, no gain. But `calibrate2.py`
(first-token logit margin z = logit(ESCALATE)−logit(PASS)) works:
```
verdict acc 0.840 | z-sign acc 0.843 | verdict==sign(z) 0.997   (z is a clean single signal)  ECE 0.155
risk-coverage (abstain least-confident):  cov 1.00 → err 0.160 ;  cov 0.60 → err 0.072
abstain band:  abstain smallest 30% → accuracy 0.906 on the rest  (vs 0.840 overall)
```
**The logit margin is a real uncertainty signal: abstaining on low-margin cases more than halves the
error.** This is "a watcher that knows when it is unsure and abstains" — shown with numbers.
(Temperature scaling did NOT improve ECE, T≈6.4, 0.131→0.141 — raw ECE already ~0.13-0.16 on small,
noisy data; the load-bearing part is the *ranking* / risk-coverage, not the magnitude.)

## Two pillars now established
1. Watcher (reading judge) beats the tuned rule: **D 0.949 vs C 0.789**.
2. Watcher is self-aware (calibrated abstention): **abstain 30% → accuracy 0.84 → 0.91**.
Both on the sealed benchmark, fact-group split, honest caveats above. Self-consistency was a null;
logit-margin was the win. Abstention = sending N% to humans (a real cost), valuable in high-stakes.

## Result 4 — PUBLIC benchmark (HaluBench): first field-level number
`halubench_eval.py`: our 7B zero-shot reading judge on PatronusAI/HaluBench (2400, 6 sources balanced),
task = faithful(PASS) vs hallucinated(FAIL=ESCALATE).
```
overall: accuracy 0.688  F1 0.685
per source: pubmedQA 0.807 | halueval 0.787 | RAGTruth 0.745(F1 0.386) | covidQA 0.688
            DROP 0.565 | FinanceBench 0.532   <- near-chance on finance + multi-step reasoning
risk-coverage (abstain least-confident by |logit margin|): cov 1.00 -> 0.688 ; cov 0.50 -> 0.838
```
Approx published on HaluBench (context): Lynx-8B (fine-tuned) ~0.85, Lynx-70B ~0.88, GPT-4o ~0.86,
GPT-3.5 ~0.70, RAGAS ~0.66.
**Honest placement: raw 0.688 ≈ GPT-3.5-tier zero-shot, above RAGAS, well BELOW fine-tuned SOTA (0.85+),
and near-chance on FinanceBench (our own target domain) + DROP.** BUT the abstention differentiator
transfers to real data: abstain the 50% least-confident → **0.838 on the committed half** (near
fine-tuned-SOTA accuracy, at 50% coverage). Raw detection = modest; calibrated abstention = the edge.

## Honest current level (field-anchored)
- Raw hallucination detection: **modest zero-shot (0.69), NOT SOTA, weak on finance/multi-step.**
- Differentiator (calibrated abstention): **real and transfers (0.69→0.84 by abstaining 50%).**
- Levers to raise raw: fine-tune the judge (how Lynx reached 0.85), larger/separate judge, finance focus.

## Result 5 — FINE-TUNED LoRA judge (is training worth it? YES)
`train_judge_lora.py`: LoRA (r=16, 0.13% params) on 3000 HaluBench train examples, 2 epochs, 7B bf16
on one 5090; evaluated on the SAME 2400 test as the zero-shot judge.
```
overall: accuracy 0.815  F1 0.796     (zero-shot 0.688 -> +0.127)
per source: halueval 0.960 | pubmedQA 0.877 | RAGTruth 0.875 | DROP 0.833 | covidQA 0.728
            FinanceBench 0.620   <- still the weakest (near-chance 0.53 -> 0.62)
risk-coverage: cov 1.0 -> 0.816 ; cov 0.6 -> 0.964 ; cov 0.5 -> 0.973
```
Published context: Lynx-8B ~0.85, Lynx-70B ~0.88, GPT-4o ~0.86, GPT-3.5 ~0.70.
**Fine-tuning lifts us into the competitive range: 0.815 (approaching Lynx-8B 0.85, well above GPT-3.5),
and with calibrated abstention 0.60 coverage -> 0.964.** Training IS worth it (+0.13).

### Honest caveats (do not overclaim "Lynx-level")
- **IN-DISTRIBUTION**: trained on HaluBench sources, tested on held-out HaluBench examples of the SAME
  sources. This is not a cross-domain generalization claim; a held-out-SOURCE test is needed for that.
- **FinanceBench still weakest (0.62)** — our target regulated domain remains hard even after fine-tune.
- halueval (0.96, easy) inflates the overall; covidQA 0.73 / finance 0.62 are the honest indicators.
- Small train (3000), 2 epochs — a quick fine-tune; more data/tuning could go higher.

## Current level (field-anchored, updated)
- Zero-shot 7B: 0.69 (GPT-3.5-tier). **Fine-tuned 7B LoRA: 0.815 in-distribution (competitive range,
  near Lynx-8B).** Calibrated abstention stacks on top (0.60 coverage -> 0.96).
- Honest boundary: in-distribution, finance weak. NOT yet a cross-domain / SOTA-beating claim.

## Next
- Cross-SOURCE generalization (train on 5 sources, test on the held-out one) — the real "does it
  generalize" test. Improve FinanceBench (our domain). Larger/separate judge. Package + Sionic outreach.
