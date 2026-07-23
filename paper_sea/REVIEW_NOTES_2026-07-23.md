# External review findings — verified 2026-07-23 (relay session)

Source: independent AI review of `main.pdf` (2026-07-18 build). Every checkable claim below was
verified against `main.tex`/`references.bib` and by recomputation. Fold into the August
Verify-Agents reframe. Priority order as listed.

## CRITICAL 1 — template/footer year
- Paper still built on `neurips_2025.sty` (stand-in; acknowledged in main.tex:2 comment) → footer
  reads "NeurIPS 2025" while body says "Literature scan of July 2026".
- ACTION: swap to the Who-Verifies-the-Agents / NeurIPS 2026 workshop template the moment it is
  published (check verify-agents-workshop.github.io); until then at minimum fix the footer year.
  MUST be resolved before submission, not camera-ready.

## CRITICAL 2 — references.bib placeholders (VERIFIED REAL, data below)
Both flagged entries are real papers (NOT hallucinated). Replace placeholders + delete both
`note = {Re-verify...}` lines:

- GRACE (references.bib ~line 121):
  - Exact title: **"No Loss, No Gain: Gated Refinement and Adaptive Compression for Prompt Optimization"**
    (bib is missing the "No Loss, No Gain:" prefix)
  - Authors: Wenhang Shi, Yiren Chen, Shuqing Bian, Xinyi Zhang, Kai Tang, Pengfei Hu, Zhe Zhao,
    Wei Lu, Xiaoyong Du
  - arXiv:2509.23387; accepted NeurIPS 2025 (poster 118743; code github.com/Eric8932/GRACE)
- GRASP (references.bib ~line 129):
  - Title as-is is correct: "GRASP: Gated Regression-Aware Skill Proposer for Self-Improving LLM Agents"
  - Authors: Johannes Moll, Jean-Philippe Corbeil, Jiazhen Pan, Martin Hadamitzky, Daniel Rueckert,
    Lisa Adams, Keno Bressem
  - arXiv:2605.29668; preprint (no venue as of 2026-07-23) — cite as arXiv preprint
- Also sweep the whole .bib for any other `note`/placeholder fields.

## STATS — recomputed 2026-07-23 (reviewer's critique CONFIRMED numerically)
Current tests treat the observed baseline rate as a fixed known constant:
- R31 embodied accept: paper's (5/12)^19 = 6.0e-8 reproduced. Two-sample Fisher exact
  (7/12 vs 0/19): **p = 3.0e-4** (one-sided; observed table is the extreme one).
- R33 deployment reject: paper's fixed-baseline binomial 8.5e-45 reproduced. Two-sample Fisher
  (5/76 vs 53/76): **p = 6.3e-17**.
- Conclusions survive decisively under the honest test; the extreme p-values do not.
- ACTION: switch headline tests to two-sample Fisher exact (or Barnard / beta-binomial); update ALL
  occurrences — main.tex lines ~55-56, 62, 114, 117, 129, 320, 343, 350, 373, 388, 535, 552 —
  and the two figure annotations. MBPP p=2.8e-8 (41%→8.5%) has the same structure: recompute
  from raw counts in RESULTS.md with Fisher.
- gate_rate() in the vsi package uses fixed-baseline exact binomial: either add the "baseline as
  pre-declared reference probability" justification where the paper describes it (§ ~636-648), or
  align the paper's reported tests with Fisher and note the package's operational form separately.
- Adaptive multiple testing: 10 recipes tried sequentially against the same panel → pre-registration
  alone doesn't cover it. Add a concrete mitigation to the protocol/limitations: alpha-spending or
  per-candidate fresh holdout or e-process; can cross-reference the R46 sequential-testing material.

## PHRASING (reviewer-bait; all locations verified)
- main.tex:45 + §heading :200 "fake self-improvement is the norm, not the exception" →
  "frequent and systematic" (29-30% + 24/24 does not literally support "norm" across all surfaces).
- "100%" claims: already scoped at :46 ("of tool-rewrite decision sets") and :226 ("of 24
  decisions") — keep the scoping in every instance; check :101.
- :541/:554 "the outer loop needs no human" → "can execute without human intervention within its
  encoded state space" (keep the adjacent unmapped-state→operator honesty sentence).

## STRUCTURE (aligns with the planned 3-pillar 4-9pp reframe — no extra work)
- Center of the paper = the scope demonstration: same candidate, target-scope ACCEPT vs
  deployment-scope REJECT (this is the memorable result; reviewer agrees).
- Text/code experiments compress to (1) motivation: fake proposals are frequent, (2) controlled
  validation of the two-gate verifier, (3) embodied deployment-scope result completes the claim.
- Contributions: currently 7 numbered → cut to 3-4. Abstract: compress ~25-35%.
- Novelty positioning: NOT "hidden tests are new" but (a) error-controlled gate for a stochastic
  embodied loop, (b) verdict-scope vs deployment-scope mismatch demonstrated, (c) two opposite,
  both-correct verdicts on the same candidate, (d) repair search that never shipped a regression.

## RELATED-WORK REFRESH — scoop-risk scan verified 2026-07-23
Adjacent papers now publishing at ~monthly cadence. All verified real on arXiv. Two are already
cited (grace2025→check year tag, pace2026, grasp); TWO ARE NEW and must be added + differentiated:

- **NEW — Self-Harness: Harnesses That Improve Themselves** (Hangfan Zhang et al.,
  arXiv:2606.09498, June 2026): weakness mining → harness proposal → regression-tested acceptance
  on held-out tasks. Differentiate: text-harness domain, score-threshold acceptance, no statistical
  error control, no deployment-scope split.
- **NEW — Governed Capability Evolution for Embodied Agents** (Xue Qin et al., arXiv:2604.08059,
  April 2026, v4 retitled "Lifecycle-Time Compatibility Checking and Rollback..."): embodied
  upgrades as governed deployment candidates — sandbox, shadow deployment, gated activation,
  rollback; same candidate can be allowed/refused per deployment profile. CLOSEST to our
  deployment-scope theme → MUST cite and differentiate: their gates are systems/compatibility
  checks (interface/policy/safety/recovery), not statistical hypothesis tests with error control;
  no same-candidate two-scope opposite-verdict experiment; no bounded-negative repair search.
- Also worth citing for field velocity (both July 2026): "Self-Evolving Agents with Anytime-Valid
  Certificates" (arXiv:2607.00871, PACE-adjacent) and "Rethinking the Evaluation of Harness
  Evolution for Agents" (arXiv:2607.12227).

Positioning consequence (matches review #1): fix **contribution #1 = the scope demonstration**
(same candidate: target-scope statistical ACCEPT → deployment-scope REJECT via catastrophic
forgetting). "Self-improvement needs an external verifier" is now common ground — never lead with
it; lead with what nobody else has shown. Related-work paragraph should explicitly place
GRASP/PACE/Self-Harness (text domains, no embodiment) and Governed-Capability (embodied, but
systems checks not statistical verdicts) and state the delta in one sentence each.

Schedule note: this scan is the argument for holding the 8/29 date — the paper is finished while
adjacent groups would still need months of embodied experiments; public repo commits (R1–R51,
dated) already timestamp priority.
