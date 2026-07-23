# Paper 3 seeds — Verified Repair Synthesis (recorded 2026-07-23, relay session)

Source: external AI roadmap ("performance problems after the gate"), cross-checked against our
actual state. NOT August work — nothing here before 8/29. Post-9/30 (after paper 2 draft).

## The trilogy frame (for research statements / KAIST interview)
- Paper 1 (The Verifier Is the Loop): the gate BLOCKS bad self-modifications.  [done]
- Paper 2 (The Price of the Gate): the gate can be STREAMED cheaply.          [draft 9/30]
- Paper 3 (working title: Passing the Gate): LEARNING to synthesize repairs
  that pass both gates — "고칠 능력에는 크게, 보존할 능력에는 거의 안 변하는 업데이트".

## Overlap audit — what the external map doesn't know we already have
| Their item | Our state |
|---|---|
| 6. gate reuse / alpha-spending / anytime-valid | R46 (naive SPRT loses), R47 (no reuse decay), limitation ¶ added 07-23; PACE cited |
| 7. verification cost (SPRT, early stop, paired eval) | R45 curtailment 26% verdict-identical = paper 2's core |
| 8. auto regression-scope discovery | retrodiction-benchmark seed (REVIEW_NOTES Q&A section): predict K1's 12 hidden regressions from pre-deployment info — dataset already in repo |
| 3. structured verifier feedback (partial) | prescriber emits next-move; feedback abstraction NOT done |

## Reprioritized for one RTX 5090 + solo (my ordering, not theirs)
P1 — **Constraint-aware updates on the K1/A2 failure** (their #1): gradient projection /
  orthogonal update / trust-region / adapter-routing at head+LoRA scale. Feasible on 5090
  (full-FT variants already ran in ~2.5h). The experiment: does a constraint-aware recipe pass
  BOTH pre-registered gates where all 10 naive recipes failed? If yes → the bounded negative is
  shown to be a statement about naive recipes, and paper 3 has its flagship. Reuses the entire
  existing harness (panels, gate, seals). NOTE: S1 bounded negative stays sealed as published —
  this is a NEW pre-registered search, new seal.
P2 — **Information-limited structured feedback** (their #3): verifier returns abstracted failure
  structure ("collision regressions concentrated in ordinary-intersection family") WITHOUT
  revealing hidden cases; measure proposer improvement vs leak-induced overfit. Cheap to
  prototype on the MBPP/HumanEval harness FIRST (text/code substrate = fast iterations),
  embodied confirmation later. This is an interface-design + measurement contribution — our
  signature shape, not a compute war.
P3 — **Diagnosis-lite** (their #2): we own a unique asset for this — the privileged expert.
  Expert–student disagreement localization + counterfactual replay in sim are cheap; skip
  influence functions / full gradient attribution (lab-scale).
DEFER — meta-prescriber learning (their #5): needs 10-100x more trial history than our 12
  verdicts; revisit after P1/P2 generate data. DEFER heavy causal attribution stacks.

## Identity guard
Do NOT drift into proposer-lab territory (AlphaEvolve/DGM turf — compute-dominated). Paper 3
stays verification-shaped: repair synthesis UNDER a fixed pre-registered gate with budgeted
verification. Success metrics keep the gate sovereign: pass-candidate discovery rate per compute
budget, trials-to-pass vs grid search, deployment regression rate (must stay 0), audit-set gate
integrity. The gate never becomes the training loss verbatim (leak rule).

## One-line next-paper question (theirs, adopted)
"Can a self-improving agent learn to synthesize repairs that satisfy both target improvement and
deployment-scale capability retention?"
