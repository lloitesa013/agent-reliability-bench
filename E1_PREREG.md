# E1 PREREG — SEALED 2026-07-24 (this commit = the seal; L0 launched immediately after)

Experiment: variance-source localization (determinism ladder) on CARLA/Bench2Drive.
Status: SEALED by relay 2026-07-24 on user's order (5090 idle; paper work is writing-bound).
L0 runner: run_e1_l0.ps1 on the 5090 box (derived from run_multi.ps1 2026-07-09; loop/env
identical, only plan+naming differ). L1+ toggles: session implements; decision lines below
are FIXED as of this commit.

## Question
Where does route-level run-to-run nondeterminism enter? (R19-R21 observed 11755 = 7/12 fail
under identical config; three FAILURE-labeled routes pass 100% on rerun.)

## Ladder (cumulative levels)
- L0: current harness as-is (baseline replication arm)
- L1: L0 + TrafficManager seed fixed
- L2: L1 + synchronous mode / fixed physics substepping (document exact settings)
- L3: L2 + inference determinism (torch deterministic algs, fixed seeds, cudnn benchmark off)
- L4: L3 + any remaining sources found (sensor noise seeds, scenario trigger timing) — enumerate
  in the final seal; if none remain, L4 = L3 (state so).

## Routes & runs
- Routes (FIXED, chosen from R19-R21 recorded data before any E1 run): **11755**
  (EnterActorFlow, 7/12=58% — high variance), **18252** (pedestrian crossing, 1/5=20% — mid
  variance), **3436** (HazardSideLane, 0/4 stable-pass AND originally mislabeled FAILURE in the
  single-run taxonomy — zero-variance control). Spans the variance spectrum.
- 12 runs per (level, route). EARLY-STOP rule: if a level shows 12/12 identical outcomes AND
  trajectory divergence < threshold (define metric in seal) on all 3 routes, higher levels run
  6 runs (confirmation only).
- Budget cap: 220 rollouts total. Timebox: 14 days from first run. On cap/timebox: report as-is.

## Metrics
- Per (level, route): failure rate + 95% CI width; outcome entropy; trajectory divergence time
  (first frame where ego position diverges > X m between paired runs — fix X in seal).

## Pre-registered hypotheses & decision lines
- H1: L1 (TM seed) removes the majority of outcome variance on 11755 (divergence begins at
  first traffic interaction). SUPPORTED iff CI width shrinks ≥50% vs L0.
- H2: L4 is outcome-deterministic (12/12 identical on all 3 routes).
- H3 (payoff): paired-seed verification concordance — using K fixed seeds × 1 run/arm
  (K defined in seal, ≤10), verdicts match the S1 rate-gate verdicts on recorded cases
  ≥95% at ≤50% rollout cost. Evaluate by replaying recorded K1/A2 decisions where data allows;
  else state not-evaluable and defer to E2 setup.
- Honest-negative branch: if L4 variance remains (H2 fails), report irreducible-sensitivity
  result — strengthens rate-gate necessity claim; equally publishable. No line may be moved
  after seal.

## Firewall (hard)
- NOTHING from E1 enters paper 1 (Verify-Agents submission) before the 8/29 deadline,
  regardless of outcome. Paper 1's evidence base is closed as of 2026-07-23.
- E1 must not consume paper-reframe writing bandwidth: launch, babysit logs, defer analysis
  writing until reframe milestones are met. If fleet ops ever conflict with reframe work,
  reframe wins.
- S1 seals untouched; E1 gets its own registry epoch.

## Remaining parameters (fixed BEFORE the analyses they affect, per clause below)
- Trajectory-divergence threshold X and paired-seed count K (≤10): do not affect L0 data
  collection; MUST be fixed and committed before any L1+ analysis or H3 evaluation begins.
- Exact L2/L3 toggle settings: session enumerates and commits before L1 runs.

## Session TODO on pickup
1. Read L0 results in C:\lead\outputs\e1_l0\ (master log: outputs\e1_l0_master.log),
2. fix X/K + L2/L3 settings in a follow-up commit, 3. implement L1 toggles, 4. record this
seal's commit hash in RESULTS.md when opening the E1 entry.
