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

## Result 6 — CROSS-SOURCE generalization (leave-one-source-out): WEAK (honest negative)
`cross_source.py`: for each source, train a fresh LoRA on the OTHER 5, test on the held-out source.
```
held-out     | cross-source | zero-shot | in-dist     verdict
DROP         | 0.555        | 0.565     | 0.833       no generalization (≈ zero-shot)
FinanceBench | 0.588        | 0.532     | 0.620       marginal (+0.06), far below in-dist
RAGTruth     | 0.795 (F1 0.109) | 0.745 | 0.875       degenerate (predicts mostly PASS)
covidQA      | 0.760        | 0.688     | 0.728       generalizes (medical transfer) — the only clean win
halueval     | 0.730        | 0.787     | 0.960       NEGATIVE transfer (worse than zero-shot)
pubmedQA     | 0.785        | 0.807     | 0.877       slight negative transfer
```
**The fine-tuning gains were largely IN-DISTRIBUTION. On a held-out source the judge falls back to
~zero-shot (or worse). Generalization is weak and inconsistent** (only covidQA clearly transfers;
halueval/pubmed show negative transfer; FinanceBench stays weak whatever we do).

## Honest final level (do not overstate)
- **Defensible *generalizing* level ≈ zero-shot 0.69.** The 0.815 holds only WITH in-domain training data.
- Robust cross-domain generalization: NO (with our modest fine-tune; Lynx's 0.88 came from large diverse data we don't have).
- FinanceBench (our target domain): weak (0.53-0.62) regardless.
- The calibrated-abstention differentiator MATTERS MORE given this: if detection doesn't generalize,
  knowing when it is unreliable (abstain) is the value. (Re-verify abstention holds cross-source — TODO.)

## Result 7 — does ABSTENTION transfer cross-source? PARTIALLY (the differentiator survives)
`cross_source_abstain.py`: leave-one-source-out, at eval capture the logit-margin risk-coverage on the
UNSEEN source.
```
held-out     | full-cov | @0.6  | @0.5     transfer
covidQA      | 0.740    | 0.879 | 0.915    strong (+0.18)
RAGTruth     | 0.792    | 0.908 | 0.915    strong (+0.12)
pubmedQA     | 0.782    | 0.875 | 0.885    yes (+0.10)
halueval     | 0.690    | 0.754 | 0.800    yes (+0.11)
FinanceBench | 0.573    | 0.625 | 0.640    fails (+0.07)  <- near-chance: no signal to be uncertain about
DROP         | 0.585    | 0.629 | 0.640    fails (+0.06)
```
**Abstention transfers on 4/6 UNSEEN sources** — abstaining on low-|z| lifts accuracy to ~0.88-0.92 on
domains never trained on. So the "knows when it is unsure" property GENERALIZES where there is partial
signal, even though raw detection does not. **Boundary: on near-chance domains (FinanceBench, DROP) the
uncertainty also fails** — with no detection signal, the judge does not know it is wrong. Finance is a
genuine wall (both detection and abstention fail there).

## Honest bottom line (whole study)
- Raw hallucination detection: modest zero-shot (0.69), fine-tune helps IN-DISTRIBUTION (0.82) but does
  not robustly generalize cross-domain; FinanceBench (our target) is a hard wall.
- **Differentiator — calibrated abstention — is more robust than the detector: it transfers to 4/6 unseen
  domains, lifting reliability to ~0.9 on the committed half.** "The watcher knows its own limits in
  unseen domains (where any signal exists) and abstains." The honest exception is signal-less domains.
- What this proves: capability to build + honestly evaluate a reliability watcher, and a real,
  transferable self-monitoring edge — NOT SOTA raw detection. Entry-ticket + a differentiated angle.

## Next
- Package the HONEST story (positives + negatives + the abstention-transfer edge) — the discipline IS
  the differentiator. Sionic OSS PR + outreach. FinanceBench remains the open hard problem (in-domain
  data / better judge). Larger diverse training data if chasing raw generalization (the Lynx recipe).

## Result 8 — verified self-improvement PROVEN (both directions of the verifier)
`proof_loop.py`: hidden-convention task (overtime after 6h; the model assumes the usual 8h -> fails,
baseline held-out 0.00). Three candidate self-improvements, each verified on HELD-OUT disjoint values:
```
candidate                 | seen Δ | held-out Δ | verifier | naive-seen
correct (general rule)    | +0.80  | +1.00      | ACCEPT   | accept
overfit (train lookup)    | +0.80  | +0.00      | REJECT   | accept   <- naive FOOLED
spurious (wrong rule)     | -0.20  | +0.00      | REJECT   | reject
```
The verifier ACCEPTS the real transferring improvement and REJECTS the overfit one that the naive
"measure on the tasks you learned from" baseline wrongly accepts (the reward-hacking / illusion-of-
progress gap, caught by held-out verification). **This completes the thesis:** the REJECT direction was
shown earlier (loop_v2 / faith_loop on real-gap data); ACCEPT + discrimination + naive-fooled contrast
are now shown. Verified self-improvement = the verifier separates real from fake self-improvement.

### Result 8 robustness (`proof_robust.py`) — verifier holds across 4 hidden-rule tasks
Repeated the proof on 4 distinct hidden-convention tasks (overtime/week/dozen/century). **The verifier
is correct on 4/4** — it ACCEPTS the general rule that transfers to held-out values (held-out +1.00) and
REJECTS the overfit + spurious rules on every task. The naive seen-only baseline is fooled by the overfit
on 2/4 (the tasks where the lookup actually raised the seen score) — i.e. wherever a fake creates a
seen-gain, naive accepts it and the verifier catches it. Verified self-improvement is robust, not cherry-picked.

## Result 9 — autonomous rule-induction FAILS (prior-anchoring); verifier still robust
`proof_autonomous.py`: the model induces rules from its OWN failures (not given). On both hidden-rule
tasks it FAILED to induce the true rule — it stayed **anchored to its prior** ("overtime after 8h",
"dozen = 12") despite failure examples showing 6h / ×10. All induced rules were wrong → 0 transfer, and
the verifier correctly **adopted none**. Finding: prompt-rule self-improvement cannot override the model's
strong priors from a few examples (a confirmation-bias failure mode) → autonomous self-improvement via
induction is unreliable — *which is exactly why the verifier is needed*. The control-based ACCEPT proof
(R8) shows the verifier CAN recognize a real improvement when one exists; autonomy is blocked by the
*inducer*, not the verifier.

## Result 10 — AUTONOMOUS verified self-improvement CLOSED (the full loop)
`proof_autonomous2.py`: after R9 showed naive induction is prior-anchored, a PATTERN-FITTING inducer
(told to ignore priors and fit output=f(input) to the failure pairs) produces candidate rules that
INCLUDE the true one. On both hidden-rule tasks the verifier ADOPTED the transferring induced rule
(overtime "minus 6" held-out +1.00; dozen "×10" +0.83) and rejected the prior-anchored/spurious
candidates. **2/2 — the full loop closes: fail → induce several candidates from its OWN failures →
VERIFY each on held-out → adopt only the one that transfers.** The thesis fully realized: the model's
self-improvement candidates are unreliable (some prior-anchored/wrong), and the VERIFIER is the
essential mechanism that selects the genuinely-transferring one. (Text domain; the embodied/CARLA
version stays hard — flaky failures + retention-DAgger is multi-week.)

## Result 11 — verified self-improvement at the FINE-TUNING level (verifier as a model-selection gate)
`candidate_ft.py`: instead of prompt-rules, we now generate three candidate LoRA fine-tunes of the 7B
judge with DIFFERENT training-data compositions and ask whether HELD-OUT selection beats IN-DIST
selection. Held-out sources (never trained): FinanceBench, pubmedQA. Candidates & results:

| candidate (training data) | verified (held-out finance/pubmed) | in-dist (own-domain held-out) |
|---|---|---|
| broad4 (250 ea × 4 sources) | **0.578** ← verifier picks | 0.800 |
| single (halueval ×1000)     | 0.507 | **0.993** ← naive picks |
| pair (DROP+RAGTruth ×500)   | 0.547 | 0.847 |

`single` trained on one source, drove training loss to **0.000** (memorized, in-dist **0.993**) yet
generalizes **worst** (held-out **0.507**) — the reward-hacking/overfit signature at the fine-tuning
level. **NAIVE selection (best in-dist) deploys `single` = the worst generalizer; the VERIFIER (best
held-out) deploys `broad4` = +0.071 better held-out.** The two policies pick DIFFERENT models, and the
verifier's pick generalizes better. Confirms the thesis scales from prompt-rules to fine-tunes: seen/
in-dist score is fool's gold; held-out transfer is the correct selection signal.
(Caveat: absolute held-out is modest (0.51–0.58) — cross-source faithfulness transfer is intrinsically
hard; brick 4 measures base-vs-candidate per source to separate "which generalizes best" from "does any
beat base". The selection result — verified ≠ naive, verified better — stands regardless.)
