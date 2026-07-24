# Paper 3 seeds — Verified Repair Synthesis (recorded 2026-07-23, relay session)

Source: external AI roadmap ("performance problems after the gate"), cross-checked against our
actual state. NOT August work — nothing here before 8/29. Post-9/30 (after paper 2 draft).

## The trilogy frame (for research statements / interviews)
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

## SEQUENCED PIPELINE (REVISED 2026-07-23: GPU lane starts IMMEDIATELY — user order.
Scheduling is per-resource-lane, not calendar: GPU never idles; session writing bandwidth
belongs to the paper reframe until 8/29; experiments queue by engineering-readiness.)
Ordering principle: multipliers before adders; zero-new-code work fills GPU gaps.
GPU queue: E1 (now, seal first) → E1-ext variance atlas (10-15 routes, zero new code) ∥
baseline bank (fixed-seed baseline rollouts on the 13 panel routes — reusable paired arms
that cheapen every future panel) → Sept: P1 arms by ascending engineering cost (b EWC →
a projection → c adapter) ∥ E2 harness when coded.
- E1 (first, ~2-4 wks, sealed prereg): **stochasticity source tracing** — localize the injection
  points of route-level nondeterminism (traffic seed / physics substep / inference nondeterminism)
  by fixing layers one at a time and measuring failure-rate variance collapse on 11755 + 2 control
  routes. Success → verification cost collapses for ALL later experiments (E2, P1 need hundreds of
  rollouts). "Mundane cause" outcome still ships as a deterministic-verification protocol note.
  Connects to the variance literature (TransFuser PAMI; B2D single-seed practice).
- E2 (second): **causal failure attribution** — same-seed counterfactual replay, intervene on one
  actor; upgrades attribution from correlational to causal; becomes paper 3's diagnosis stage.
- P1 (third, flagship): constraint-aware repair search (as above) — cheaper if E1 succeeded.
- Anchor typology: written as the FRAME section of the constitution paper, not a standalone
  flagship. Independence experiment (same-family judges, text substrate): fellowship-project
  scope (12-week), keep in reserve.
- Checkpoint-zoo mechanistic study: preserved asset, NOT pursued now (different subfield/tooling).

## FULL DISPOSITION of the 4-pile mine (2026-07-24 — every item gets a home)
ZERO-GPU items (writing-lane, fold into paper 2 drafting in Sept):
- Verification-cost scaling law: formalize the price curve (effect size × flakiness × scope →
  required rollouts) from EXISTING ledger data points (19/91/~450 runs, 169-screen). Paper 2
  economics section. No new runs.
- Paired sequential test design: successor to the naive SPRT that R46 killed — design + Monte
  Carlo validation on RECORDED counts (CPU only); later live-validate on E1's paired-seed data.
  Paper 2 methods contribution or standalone stats note.
INSTRUMENTATION items (free byproducts — add logging to P1 arms, no dedicated runs):
- Paralysis-collapse dynamics (0.4 ratio: drives at 3ep, freezes at 5ep): log per-epoch action
  distributions + loss decomposition during ALL P1 training runs → the phenomenon gets its
  data as a side effect.
- Fractal dilution cause (A3 same-family conflict): tag P1 training batches by route family,
  log gradient-conflict stats between family members. Same trick.
COLLABORATION-CURRENCY item:
- Checkpoint zoo (13 same-base differently-broken/repaired models): do NOT dissect ourselves —
  offer as a dataset to an interpretability collaborator (December venue conversations).
  "Mechanistic anatomy of forgetting" is a strong joint-paper hook; we bring the zoo, they
  bring the microscope.
Already queued elsewhere: stochasticity=E1 (running), causal attribution=E2, repair
routing=P1 arm (c), anchor typology=constitution frame section, independence
experiment=fellowship-scoped reserve.

## DESIGN SKETCHES (relay 2026-07-23 — session writes the actual PREREG before any run)
E1 variance-source localization: determinism ladder L0(as-is)→L1(+TM seed)→L2(+physics/sync)
→L3(+inference determinism)→L4(all). 3 routes (11755 + 1 stable-pass + 1 flaky control) × 12
runs/level, early-stop when variance→0. Metrics: failure-rate CI width, trajectory divergence
time. H1 TM-seed dominates; H2 L4 deterministic; H3 paired-seed verification concordance ≥95%
vs rate-gate at ≤50% rollouts. Budget ≈180 rollouts, 1-2 wks fleet. Both outcomes ship
(protocol note vs irreducible-sensitivity result strengthening rate-gate necessity).
E2 counterfactual attribution (DEPENDS on E1 determinism): same-seed replay, single
interventions (remove/delay one actor; splice expert action at divergence point). Attribution =
minimal outcome-flipping intervention. 12 failure cases × 10-20 CF runs ≈ 150-250 rollouts.
Side payoff: causally re-labels the B2D failure taxonomy (v2 of the taxonomy repo).
E2 MUST-CITE prior art (from study-gap research 2026-07-24):
  - Buesing et al., "Woulda Coulda Shoulda: Counterfactually-Guided Policy Search" (CF-GPS,
    ICLR 2019, arXiv:1811.06272) — counterfactual replay in a simulator via an SCM, fixing the
    exogenous noise to de-bias. This IS E2's method, done at DeepMind. Cite + state the delta.
  - Halpern-Pearl actual causality: "minimal outcome-flipping intervention" = actual cause /
    degree of responsibility (responsibility = 1/|minimal contingency set|). Gives E2's metric a
    name + citation (Halpern, "Actual Causality," MIT Press, open access).
  - Frame the sim explicitly as an SCM: seed = exogenous U, dynamics = structural equations,
    intervene-on-one-actor = do(x); same-seed replay = the abduction step that makes the
    Layer-3 counterfactual identifiable (data-only counterfactuals are NOT — a defensible
    privileged-access point vs a reviewer).
P1 constraint-aware repair, 3 pre-registered arms + A6 baseline: (a) gradient projection
(fix-grad ⊥ retention-grad subspace, head-scale), (b) EWC/trust-region penalty on planning
head (Fisher from retention data), (c) LoRA adapter + scenario-gated routing (architecture
arm). SAME S1 gates (fix ≤2/8; retention ≤30%; deployment gate for passers), new registry
epoch, timeboxed. Budget ≈300-450 rollouts (less if E1 lands). Any pass → flagship; all fail →
second bounded negative with mechanism map.
Assembly: paper 3 = E2 diagnosis + P1 repair + E1 economics. Order: E1 → (E2 ∥ P1).
