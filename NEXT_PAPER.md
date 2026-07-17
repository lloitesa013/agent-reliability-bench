# Next-paper material (parked 2026-07-17 — design only, NOTHING here has been run)

Operator decision: S3+ is the NEXT paper's spine, not a bolt-on to the SEA submission. This file
parks the design and the assets so nothing rots. No pre-registration exists for any of this yet;
seal one (S2_PREREG style, before any code) whenever a line item is activated.

## Working title direction
"The price of the gate": making deployment-scope verification affordable — risk-weighted screening
+ verification-cost scaling laws for self-improving embodied agents.

## ⚠️ 2026-07-17 UPDATE — W1 CLOSED NEGATIVE (R44): static risk ranking is dead; pivot
W1 retrospective failed its sealed line (LOCO 3/18 ≈ random; in-sample ceiling 4/16): deployment
regressions are CANDIDATE-DEPENDENT — no candidate-independent route prior can compress the screen.
Zero GPU was spent. Paper-2 spine pivots to the surviving directions:
  1. **Sequential/adaptive verification** (the new core) — FIRST NUMBER IN (R45): pure curtailment
     replay of the R33/R37 confirm records saves **26% of runs (338→249) with per-route-identical
     verdicts, zero assumptions** (s3_sprt_replay.py). Next rungs: Wald-SPRT / anytime-valid
     (controlled error, earlier stops on strong effects — validate on the same records first),
     adaptive allocation across flagged routes, candidate-AWARE screen ordering (training-recipe vs
     route-family distance — permitted, uses no candidate outcomes). Paper-2 thesis now two-legged:
     "you cannot pre-compress the gate (R44); you can stream it (R45)."
  2. **W4 gate-reuse decay** (still live, scorer-independent, sealed as measurement).
  3. The candidate-dependence finding itself (R44) as a headline negative: "your deployment screen
     cannot be pre-compressed; verify at scope or don't know."

## S3 — risk-weighted deployment screen (ORIGINAL PLAN — closed by R44, kept for the record)
Claim shape: the 169-route deployment screen (176 runs / ~12 h) can be cut to a top-K risk-ranked
screen at a measured recall of confirmed regressions.
- **Training data ALREADY ON DISK** (produced by S1, no GPU needed to start):
  - K1 full 169-route screen + 176-run interleaved confirm → 12 confirmed regressions (R33)
  - A2 fleet confirm → 6 confirmed regressions + 9 near-misses (R37)
  - 220-route failure taxonomy + scenario attribution (watcher/judge runs)
  - per-route features usable WITHOUT peeking at candidate outcomes: scenario family, town,
    route length, baseline flakiness/variance, taxonomy cluster, infraction-type history
- **Retrospective experiment (GPU-free, ~1 day):** build risk scorer on features only → measure
  recall@30 of the 18 confirmed regressions. Leave-one-family-out to probe generalization.
- **Prospective card (fleet, ~2 days): A6 has NEVER received a deployment gate** (it failed the
  panel, so the deployment stage never ran). Run top-30 risk-weighted screen vs full screen on the
  A6 checkpoint (`outputs/checkpoints/vsi_a6`) — a true prospective test of the scorer.
- Honest boundary from day one: this is verification-COST reduction with measured recall, not a
  safety guarantee; a missed-regression rate is a first-class number.

## Held-out failure families (S3's second half)
The 51 non-clean routes were never used during recipe iteration (R37's sealed caution reserved
them). They are the fresh-probe pool for any future finally-accepted candidate — and the substrate
for measuring GATE-REUSE DECAY at system level (embodied R18): how much does iterating against a
fixed screen inflate apparent safety?

## Re-open axes for the repair problem (explicitly OUT of the closed S1 search)
The R41 closure claim is bounded to "full/head-only fine-tuning at this budget". Untested axes:
  - LoRA/adapter dedicated capacity (R11 analog at driving scale)
  - larger fix-side data collection (only ~1.6k frames were ever used)
  - action-space parametrization (the X-MoD re-open axis — see xmod-b2d-brick memory)
  - embodied W2 continuation: route 18252 (R32's canary) through the AUTOMATED pipeline —
    a new target, not a reopening of 11755
## S4 sketch (defense line of the ladder)
AutoML/random-search comparison arm (is the prescriber better than blind search per GPU-hour?) +
full repro pack. Depends on S3's cheap gate to be affordable — sequencing is S3 → S4.

## Assets inventory (where things live)
5090: checkpoints vsi_a1..a6; expert_targeted collection; quarantined 13-route dirs
(C:\lead\data\vsi_quarantine); panel outputs outputs/local_evaluation_win/{a3_,a5_,a6p_}*;
screen/confirm CSVs+JSONs from R33/R37. Local repo: results_w2/, s2_replay.py, registry/prescriber
(reusable as-is for S3 trial bookkeeping).
