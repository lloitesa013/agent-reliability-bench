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

## Result 12 — the REGRESSION verifier (brick 4): naive adoption silently forgets; retention prevents it
`verified_integration.py`: extend the verifier from 1D (does the target improve?) to 2D (target improves
AND no old capability regresses). Base judge profile then two candidate adoptions to fix the weak target
FinanceBench:

| source | base | NAIVE (finetune finance only) | RETAIN (finance + replay old 100×4) |
|---|---|---|---|
| DROP | 0.560 | 0.555 (−0.005) | **0.680 (+0.120)** |
| RAGTruth | 0.830 | **0.670 (−0.160)** | 0.830 (0.000) |
| halueval | 0.845 | 0.815 (−0.030) | 0.855 (+0.010) |
| covidQA | 0.640 | 0.640 (0.000) | 0.675 (+0.035) |
| FinanceBench (target) | 0.605 | 0.610 (+0.005) | 0.590 (−0.015) |

Two honest findings: (1) **the regression is real and severe** — naively fine-tuning to fix FinanceBench
tanks RAGTruth by **−0.160 (16 pts)** and halueval by −0.03, i.e. the X-MoD "naive correction backfires"
warning reproduced on the judge; a 1D verifier (target only, +0.005) would have MISSED it and adopted a
capability-destroying change. (2) **retention (replay) prevents all regression** (worst old Δ = 0.000,
even DROP +0.12) — a safe net-positive change. Under the correct regression verifier (reject any candidate
that regresses an existing capability): **NAIVE → REJECT (RAGTruth −0.16), RETAIN → regression-free.**
The target-gain gate rejects both because finetuning-on-finance does not raise held-out finance
(+0.005 / −0.015) — the same cross-source non-transfer as R11 — so a finance-gated adoption safely takes
nothing. **Validated: the regression half of the self-verifying agent catches a silent −0.16 forgetting
that naive adoption walks into.** The full accept-a-real-gain-without-regression loop needs a substrate
where the gain transfers → the flagship (hidden-convention tasks, `self_verify_agent.py`).

## Result 13 — FLAGSHIP: the self-verifying agent (safe self-modification, full loop)
`self_verify_agent.py`: an agent already handles family A (overtime, held-out 1.00) and fails family B
(dozen, 0.00). It proposes candidate self-modifications (rule-edits) to fix B from its own failures, and
adopts one only if it clears TWO gates: HIDDEN TEST (improves B on held-out inputs) AND REGRESSION (does
not break A on held-out). Candidates:

| edit | seen-B | hidden-B (transfer) | A after (regression) |
|---|---|---|---|
| #1 "for N-dozen questions, N×10" | 1.00 | 0.50 | 0.83 (Δ−0.17) |
| #2 "...each dozen=10 items, multiply..." | 1.00 | **1.00** | 0.83 (Δ−0.17) |
| #3 "total items in N dozen is N×10" | 1.00 | 0.83 | **1.00 (Δ0.00)** |

**NAIVE** (adopt best seen-B) → #1 → A=0.83, B=0.50, **mean 0.67** (regressed A + overfit B).
**SELF-VERIFYING** (both gates) → #3 → A=1.00, B=0.83, **mean 0.92**. Capability 0.50→0.92, safely.
The decisive moment: edit **#2 fixes B best (hidden 1.00) but the agent REJECTS it because it regresses
A** — even the strongest improvement is refused if it breaks an existing capability; the agent trades B
(1.00→0.83) to preserve A. Naive, seeing only seen-B, adopts a regressing/overfit edit. **This is the
target artifact: an agent that distrusts its own {prompt/rule/memory} changes and adopts only what a
hidden-test + regression verifier clears — self-improving without self-deception.** (7B, hidden-convention
substrate where gains transfer; single round — next: multi-round cumulative loop.)

## Result 14 — multi-round self-improvement loop (brick 5): gated adoption preserves accumulated capability
`self_improve_loop.py`: the agent starts with capability A and acquires B, C, D one per round. Same
candidate pool each round; NAIVE adopts best-seen, SELF-VERIFYING adopts only edits that transfer
(hidden) AND don't regress any acquired capability. Cumulative held-out capability:

| round | NAIVE | SELF-VERIFYING |
|---|---|---|
| start (A only) | 1.00 | 1.00 |
| + B_dozen | 0.75 | 0.83 |
| + C_chapter | 0.78 | 0.94 |
| + D_team | 0.79 | 0.88 |

**Self-verifying is higher at EVERY round (+0.08 to +0.16); final 0.88 vs 0.79.** The gap is regression:
in round 1 naive adopted an edit that (in the verified track) showed worst_reg −0.50 — it half-destroyed
capability A — so naive craters to 0.75 and stays 0.75–0.79. Self-verifying rejected that edit, took the
−0.17 one, and never dropped from regression: its round-3 dip (0.94→0.88) is under-acquisition of D_team
(the inducer proposed malformed ×4 rules; best candidate hidden 0.67, but worst_reg +0.00), NOT forgetting.
**Gated self-improvement compounds while preserving old capability; ungated accumulates regressions and
stagnates.** (Caveats: 6-item held-out sets = 0.17 granularity, noisy; not strictly monotone; claim is
"verified consistently higher + never regresses", not "perfect climb".)

## Result 15 — self-verifying agent over TOOL/CODE self-mods (brick 6): executing verifier catches lookup-hacks
`self_verify_tool.py`: the agent maintains a Python tool `solve(qtype, n)` (handles 'overtime'). To acquire
'dozen' it REWRITES the function and proposes three implementations; the verifier EXECUTES each on held-out
inputs (deterministic — no LLM noise). Result:

| impl | seen-B | executed hidden-B | overtime after |
|---|---|---|---|
| #1 general rule (n×10) | 1.00 | **1.00** | 1.00 |
| #2 hardcoded lookup of seen inputs | 1.00 | **0.00** | 1.00 |
| #3 hardcoded lookup (variant) | 1.00 | **0.00** | 1.00 |

**2 of 3 of the agent's own tool rewrites are LOOKUP HACKS** — perfect on the inputs it saw, but return
wrong/None on held-out inputs when executed. Seen score is identical (1.00) across all three, so **naive
selection has no signal to avoid the hacks — it gambles (expected capability 0.67, deploying a broken tool
2/3 of the time).** Executing each candidate on held-out gives a decisive 1.0-vs-0.0 signal, so the
verifier **deterministically adopts the general tool (1.00)**. This is the self-coding-agent failure mode
(reward-hacking by hardcoding) caught by an executing hidden-test — the {tool} element of
{prompt/rule/memory/tool}, and the most rigorous brick (verification is deterministic execution).

## Result 16 — multi-round at scale (5 rounds): cumulative advantage is small/noisy — HONEST correction to R14
`self_improve_loop6.py` (A + acquire B,C,D,E,F), tracking BOTH mean and min (worst) capability:

| round | NAIVE mean/min | VERIFYING mean/min |
|---|---|---|
| B_dozen | 0.75 / 0.50 | 0.83 / 0.83 |
| C_chapter | 0.78 / 0.50 | 0.94 / 0.83 |
| D_team | 0.79 / 0.50 | 0.88 / 0.67 |
| E_week | 0.87 / 0.67 | 0.87 / 0.67 |
| F_crate | 0.81 / 0.67 | 0.83 / **0.50** |
| **final** | **0.81 / 0.67** | **0.83 / 0.50** |

Early rounds behave as expected (verifying preserves min 0.83 while naive breaks a capability to 0.50).
**But at scale the advantage does NOT hold up:** final mean Δ+0.03, and final MIN is actually WORSE for
verifying (0.50 vs 0.67) — because verifying under-ACQUIRED the last task F_crate (0.50, a new-task
failure, not a regression) while naive's early regression got diluted/recovered across 6 capabilities.
**This is an honest correction: R14's larger 3-round gap (0.88 vs 0.79) is NOT robust to more rounds** —
cumulative capability at scale is dominated by new-task-difficulty noise and dilution on these small
(6-item) toy tasks. **What IS robust is the SINGLE-DECISION verifier advantage** (R11 memorizer, R12
−0.16 regression, R13 rejecting the best-B-fix that regresses A, R15 2/3 lookup-hacks): in one adoption
decision the verifier reliably catches the fake/regression/hack that fools naive. The multi-round
CUMULATIVE compounding into a large gap is not demonstrated on toy tasks. (No cherry-picking: reported as
observed; not re-run to hunt a bigger gap.)

## Result 17 — STATISTICAL single-decision robustness (brick 8): naive 59% bad, verifier 0%
`stat_robustness.py`: the robust claim (R11/13/15) as a RATE. Across 24 tool-acquisition decisions (8
hidden-convention tasks × 3 trials) the agent rewrites its tool `solve()` and we execute each candidate:

| metric | rate |
|---|---|
| a reward-hack/regressing tool existed among the agent's own proposals | **100%** of decisions |
| NAIVE (select by seen score) deploys a BAD tool | **59%** |
| VERIFIER (execute on held-out before adopting) deploys a BAD tool | **0%** |
| VERIFIER still deployed a WORKING new tool | **88%** |

**A bad candidate exists in 100% of decisions** — reward-hacking (lookup-hardcoding) is not an edge case,
it is always present in the agent's own proposals. Because seen score can't tell a hack from a general
tool, **naive deploys a broken tool 59% of the time**; the executing verifier **0%** (it runs each
candidate on held-out inputs before adopting) while still improving in 88% of decisions (the other 12%,
e.g. `chapter`, it safely adopts nothing when all proposals are bad). This is the robust, statistically-
grounded core result: **the single-adoption-decision advantage of an executing hidden-test verifier over
naive seen-selection is large and consistent** (0% vs 59% bad-deploy). It is what R14/R16 showed does NOT
compound into a large multi-round cumulative gap — the per-decision guarantee is the deliverable.

## Result 18 — the hidden test must be LARGE/FRESH (brick 9): reused small held-out inflates by 0.045
`fresh_vs_reused.py`: if the self-verifying agent gates many candidate self-mods on the SAME small
held-out, selection overfits to it (games the verifier). On HaluBench faithfulness, 8 candidate judge-
prompts (true acc 0.805–0.837) selected over 300 val/test splits:

| selection | reported (val) | true (test) | note |
|---|---|---|---|
| REUSED small val (n=20) | 0.874 | 0.829 | **OPTIMISM GAP = 0.045** |
| FRESH / large sample | — | 0.837 | picks the truly-best |

**Two honest findings:** (1) the **optimism gap is real and measurable (0.045)** — selecting the best of 8
on a 20-item reused set reports 0.874 but delivers 0.829; a small reused hidden test gives FALSE CONFIDENCE
in an improvement. (2) The **deployment cost was small here (0.008)** only because the 8 candidates were all
close in quality (spread 0.03), so mis-picking barely hurts. **Combined with R17** — where a genuine
reward-hack is present in 100% of tool decisions and naive mis-selection is catastrophic (59% bad deploy)
— the principle is: the hidden test must be LARGE/FRESH to avoid (a) optimism-gap false confidence and
(b) catastrophic mis-selection when the candidate pool contains reward-hacks. Reusing a small held-out to
score many self-mods re-creates the very reward-hacking the verifier exists to prevent.

## Result 19 — EMBODIED statistical failure-rate verifier (CARLA/Bench2Drive): 11755 is 50% flaky
`run_multi.ps1` on the real LEAD tfv6 agent + CARLA 0.9.15, re-running two routes N times each (fresh CARLA
per run, no changes to route or agent):

| route | runs (PASS/FAIL) | failure rate | verdict |
|---|---|---|---|
| 11755 (EnterActorFlow) | PASS,PASS,FAIL,FAIL,FAIL,PASS | **3/6 = 50%** | **FLAKY** |
| 3436 (HazardSideLane) | PASS,PASS,PASS,PASS | 0/4 = 0% | STABLE |

Route 11755, the SAME route and agent with zero changes, scores 100 (clean) on 3 runs and collides
(score 60, then 22) on 3 runs — a **50% failure rate**. A single run reports PASS or FAIL essentially at
random, so "the fix worked / the agent fails here" from one run is **fool's gold** — the embodied analog
of trusting a seen-score for a text self-mod. The **failure RATE over N runs** is the real signal, and it
distinguishes flaky (11755) from stable (3436, 0%). This quantifies the previously-qualitative flakiness
finding and demonstrates the embodied statistical verifier: verification must be a rate, and a candidate
"fix" is only real if it lowers the failure RATE beyond this stochastic noise. (Confirms the STUDY.md
embodied claim with numbers; the full embodied self-improvement loop — a fix that provably lowers the rate
— remains the open hard core, gated now by a working statistical verifier.)

## Result 20 — embodied flaky/stable MAP: single-run failure labels are unreliable
`run_multi.ps1` extended to 5 routes (fresh CARLA per run, LEAD tfv6 agent, zero changes):

| route | scenario | failure rate | verdict |
|---|---|---|---|
| 11755 | EnterActorFlow | 3/6 = **50%** | FLAKY |
| 18252 | pedestrian-crossing | 1/5 = **20%** | FLAKY |
| 3436 | HazardSideLane | 0/4 = 0% | STABLE-pass |
| 2509 | construction | 0/4 = 0% | STABLE-pass |
| 2513 | construction | 0/4 = 0% | STABLE-pass |

**Key finding: routes 3436, 2509, 2513 were recorded as FAILURES in the original single-run Bench2Drive
taxonomy, but re-running shows they PASS 100% (0% failure rate) — their "failure" was flaky one-off noise
that does not reproduce.** Only the dynamic-collision scenarios (EnterActorFlow, pedestrian-crossing) are
genuinely flaky (50%, 20%); the construction min-speed "failures" are environment artifacts that don't
reproduce (consistent with prior X-MoD finding that min_speed is an env artifact). Implication: **single-run
failure labels — the standard for driving-benchmark leaderboards — are unreliable; they measure stochastic
noise as much as capability** (the embodied analog of the reward-hacking / illusion-of-progress problem).
An embodied verifier MUST use a failure RATE over N runs to (a) separate genuine flaky failures from
non-reproducing noise, and (b) define a stable enough target to even verify a fix against.

## Result 21 — 11755 flaky rate tightened to 7/12 = 58% (+ honest ops failure: CARLA wedge)
A second re-run batch on 11755 added 6 more runs (FAIL,FAIL,FAIL,PASS,PASS,FAIL). Combined with R19's
6 runs (3 fail): **11755 = 7/12 = 58% failure rate** — a tighter estimate confirming it is genuinely
flaky (not stable). 18252 stands at 1/5 = 20% (R20).

**Honest operational failure:** the second batch WEDGED on its 7th run — CARLA hung during that run and
the `python -m lead` eval blocked waiting for it, sitting idle for ~10 hours before I killed it. The
`--timeout 900` flag is LEAD's *internal scenario* timeout, which only fires if CARLA is responsive; it is
NOT a hard process-kill watchdog, so a hung CARLA blocks the runner indefinitely. My earlier claim that the
runner was "wedge-robust" was wrong. Fix for any future CARLA batch: wrap each run in an EXTERNAL wall-clock
watchdog (Start-Process + timed Wait; kill python+CarlaUE4 on expiry) — do not rely on LEAD's internal
timeout. No scientific harm (the 11755 data is valid and tighter), but ~10h of idle GPU was wasted; recorded
so it is not repeated.

## Result 22 — REAL SUBSTRATE: self-verifying coding agent on MBPP (closes the toy→real gap)
`mbpp_self_verify.py`: the self-verifying agent on 40 REAL MBPP programming problems (not toy conventions).
For each problem the agent sees the description + ONE example test and proposes 3 candidate solutions; the
verifier EXECUTES each on HELD-OUT tests. Over 40 problems:

| metric | rate |
|---|---|
| a visible-pass-but-held-out-FAIL candidate existed | **30%** of problems |
| NAIVE (deploy by the shown test) ships a held-out-FAILING solution | **39%** |
| VERIFIER (execute held-out tests) ships a held-out-FAILING solution | **0%** |
| VERIFIER shipped a held-out-passing solution | **68%** (else abstains, 32%) |

**This closes the biggest honest gap — toy vs real.** The exact pattern from the engineered toy (R17:
59% vs 0%) reproduces on genuine code: selecting a self-written solution by the test it can see ships a
held-out-failing (overfit/hardcoded/wrong) solution **39%** of the time; executing candidates on held-out
tests before shipping catches it (**0%**) while still shipping a verified solution 68% of the time.
**Honest nuance:** the 39% mixes two real cases — (a) a genuine overfit present (visible-pass, held-out-fail;
30% of problems), and (b) unsolvable problems where naive ships broken code and the verifier correctly
abstains. Both are real advantages (the verifier abstains instead of shipping broken code), but they are
distinct. Net: **the verifier's single-decision advantage is not a toy artifact — it holds on real
code-generation tasks with execution-based held-out verification.**

## Result 23 — embodied verifier statistical SPEC + hard-watchdog runner (C, as far as achievable)
`embodied_power.py`: given route 11755's measured baseline failure rate 7/12 = **0.58, 95% CI [0.32, 0.81]**
(even the baseline is only ±24% at n=12 — single runs are worthless), how many CARLA runs does the
failure-rate verifier need to CONFIRM a candidate fix lowers the rate? (two-proportion, one-sided α=0.05):

| fix (from 58%) | runs @80% power | runs @90% power |
|---|---|---|
| → 40% (−18 pts) | 91 | 126 |
| → 30% (−28 pts) | 37 | 51 |
| → 20% (−38 pts) | 19 | 26 |
| → 10% (−48 pts) | 11 | 15 |

**Confirming even a LARGE fix (58%→20%) needs ~19 runs; a modest one (58%→40%) needs ~91** — at ~2-4
min/run that is 30–360 min of CARLA *per candidate fix*, and a self-improvement loop tries many candidates.
This is the honest COST of embodied verified self-improvement, and why the full embodied loop is the hard
core: **the statistical verifier is cheap to SPEC, but a real fix must first be CREATED (retention-DAgger,
a multi-week build).** Also delivered: `run_multi_wd.ps1`, a **hard wall-clock watchdog** runner that
force-kills a hung python+CARLA per run (fixing the R21 wedge) and self-cleans CARLA at the end — the
infrastructure the embodied loop needs. **VSI-0 delivers the embodied verifier + its spec + wedge-proof
harness; the fix-GENERATION is explicit future work.**

## Result 24 — MBPP hardened (n=100, clean metric): refines R22, removes the confound
`mbpp_self_verify2.py`: R22's 39% mixed two things; here we separate them over 100 problems
(solvable=64, unsolvable=36) and report the confound-free numbers:

| clean metric | rate |
|---|---|
| a genuine overfit existed (shown-test-pass, held-out-FAIL) | **29/100 = 29%** of problems |
| of those 29, NAIVE ships the overfit / VERIFIER ships it | **19 / 0** |
| **SOLVABLE problems only** — NAIVE ships a held-out-FAILING solution | **12%** |
| **SOLVABLE problems only** — VERIFIER ships a held-out-FAILING solution | **0%** |

**The confound-free headline: even on problems the agent CAN solve, ship-by-shown-test deploys a
held-out-failing solution 12% of the time (picking an overfit/wrong candidate over a correct one); the
executing verifier 0%.** And genuine reward-hacks are common on real code (29% of problems) — of those,
naive ships the overfit **19 times, the verifier 0**. This supersedes R22's raw 39% (which conflated
overfit-catches with abstain-on-unsolvable): 12% is smaller but honest and unimpeachable. The real-code,
single-decision verifier advantage is bulletproof.

## Result 25 — EMBODIED fix probe: the loop runs end-to-end, and the verifier REJECTS a harmful "fix"
`run_arms.ps1` + an env-gated throttle intervention injected into the live tfv6 agent
(`sensor_agent.py`, `LEAD_THROTTLE_SCALE`, default 1.0 = untouched): the first real embodied
propose→verify→(adopt/reject) cycle, on route 11755 (baseline 58% flaky), 6 runs per arm with a hard
10-min watchdog (no wedges):

| arm | intervention | failure rate |
|---|---|---|
| 1.0 | none (baseline) | 3/6 = **50%** |
| 0.7 | conservative throttle — the plausible "fix" | 5/6 = **83%** |
| 1.3 | aggressive throttle — regression probe | 4/6 = **67%** |

**The plausible fix candidate ("drive more conservatively to avoid collisions") made things WORSE:**
collisions still occur (12.3-23.2 pts) and slower driving adds new penalties (scores drop to 42). The
statistical verifier's verdict is REJECT — adoption requires evidence the failure RATE drops, and the
observed rate went UP. This is the embodied analog of R13 (reject the plausible edit that harms) and a
verifier-caught version of the X-MoD lesson (naive correction-retraining backfires): **the first
embodied self-improvement cycle ran end-to-end on real CARLA, and its value was REJECTING a harmful
self-modification before deployment.** Honest limits: N=6/arm → 50%→83% is directional, not conclusive
(R23 power table); no accepted fix yet — a fix that truly lowers the rate still requires learning-based
repair (retention-DAgger, future work). Probe infrastructure (env-gated intervention + watchdog runner)
is now in place and no-op by default.

## Result 26 — HumanEval replication: R24's numbers reproduce almost exactly on a second real benchmark
`humaneval_self_verify.py` (80 problems, flat-assert filter 116/164, first assert visible / rest hidden):

| metric | MBPP (R24) | HumanEval (R26) |
|---|---|---|
| genuine overfit present | 29% | **30%** |
| SOLVABLE-only: NAIVE ships held-out-FAILING | 12% | **12%** |
| SOLVABLE-only: VERIFIER ships held-out-FAILING | 0% | **0%** |
| where an overfit exists, NAIVE ships it | 19/29 | **21/24** |

Two independent real code benchmarks, near-identical rates: ~30% of problems draw a genuine overfit from
the agent's own solution stream, naive selection ships it ~4 times out of 5 when present, and executing
candidates on held-out tests ships zero. **The single-decision verifier advantage on real code is
replicated, not benchmark-specific.**
