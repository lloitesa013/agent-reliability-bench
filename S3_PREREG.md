# S3 Pre-registration — Risk-Weighted Deployment Verification, FULL (SEALED 2026-07-17, before any S3 code)

**Operator decision (2026-07-17): "가자." Full S3 as the NEXT paper's spine, run in parallel with the
sealed SEA submission.**

## Rails (restated, non-negotiable)

- **SEA is untouchable.** No S3 result — however good — enters the SEA submission (`paper_sea/`).
  Only prose/figure polish is permitted there. SEA submits 2026-08-29 AoE regardless of S3's state.
- **Registry discipline.** Every S3 stage is a trial in the S2 registry machinery (`vsi_registry.py`,
  new ledger `s3_trials.jsonl`): lines registered before data, verdicts computed from raw counts,
  append-only. The gate remains the only verdict authority.
- **Hard timebox: 2026-09-30 AoE** for W1–W3 (clock pauses only for operator-ordered box pauses).
  Max **3** new candidate models. Misses → bounded negative, reported first-class in paper 2.

## The question

The deployment-scope gate is the loop's most expensive stage (169-route screen + interleaved
confirms ≈ 176 runs / ~12 h per candidate — R33/R37). S3 asks: can a **risk-ranked top-30 screen**,
whose scorer never sees candidate outcomes, buy most of the full screen's regression-catching power
at ~5.6× lower screen cost — and does that ranking **generalize across candidates** (W2/W3) and
survive **held-out failure families** (W4)?

## Feature contract (sealed)

The scorer may use ONLY candidate-independent route properties:
route/scenario metadata (scenario type & family, town, length, #scenarios), BASELINE-arm behavior
(flakiness/variance from taxonomy & multi-run records, baseline infraction profile from R33/R37
baseline arms), taxonomy cluster. **Excluded: any outcome of any candidate model.** Model class is
free during W1 (simple/interpretable preferred); the scorer is FROZEN (committed) at W1's end and
may not change during W2/W3.

## Stages and sealed lines

- **W1 — retrospective, GPU-free (go/no-go for all GPU spend).** Ground truth: the 18 confirmed
  regressions (K1: 12, A2: 6) over the 169 baseline-clean routes. Test: leave-one-candidate-out —
  scored without K1's outcomes, rank K1's 12; scored without A2's outcomes, rank A2's 6.
  **Line: pooled LOCO recall@30 ≥ 12/18.** (Random expectation ≈ 3.2/18.) Honesty note, sealed:
  W1 involves model development on 2 folds — mild adaptive fitting is possible; the REAL tests are
  W2/W3 with the frozen scorer. W1 below the line → S3's risk-screen direction closes as a bounded
  negative; no GPU is spent.
- **W2 — prospective on A6.** A6 (`vsi_a6`) never received a deployment gate. ORDER SEALED: the
  frozen scorer's top-30 for A6 is committed BEFORE the A6 screen produces any result. Then: full
  169 screen (ground truth) + R33-rule confirms (flag → 4+4 interleaved; confirmed = cand ≥3/4 AND
  base ≤1/4). **Line: frozen top-30 contains ≥ 2/3 of A6's confirmed regressions.**
  Degenerate branch (sealed): if A6 yields < 3 confirmed regressions, W2 = UNDECIDED-degenerate
  (reported, not spun) and weight shifts to W3.
- **W3 — generalization on NEW candidates (≥2, ≤3).**
  C1 = route-18252 repair produced through the AUTOMATED pipeline (prescriber proposes the recipe
  from the registry; embodied continuation of S2's W2). C2 = LoRA/adapter candidate (the sealed
  re-open axis) with a **3-day integration timebox**; fallback if the timebox blows: a second
  scenario-family repair target through known machinery. Optional C3 within budget.
  For EACH candidate, the frozen scorer's top-30 is committed before its screen runs.
  **Line: pooled over new candidates' confirmed regressions, frozen-scorer recall@30 ≥ 2/3**, with
  the same degenerate branch (pooled confirmed < 3 → extend to next candidate or report bounded).
  Cost accounting is a deliverable: runs used by top-30-screen vs full-screen, per candidate.
- **W4 — held-out families + gate-reuse decay (measurement, not pass/fail; sealed as such).**
  The 51 non-clean routes (reserved by R37, never used in recipe iteration) as fresh probes on the
  most-studied candidate(s): compare regression-flag rates on 169-screen routes vs the 51 fresh
  probes → the embodied gate-reuse-decay number. Completion = W4 done.

## Non-goals (sealed)

No reopening of the closed 11755 repair search (R41 stands). No safety-guarantee claim — S3's claim
is verification-COST reduction with a MEASURED miss rate; missed regressions are a first-class
number. No SEA edits. New-candidate repairs do NOT need to pass their own panels for S3 to succeed —
S3 evaluates the SCREEN's ranking power over whatever regressions the candidates actually cause.
