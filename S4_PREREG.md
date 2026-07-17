# S4 Pre-registration — The Defense Line (SEALED 2026-07-18, before any S4 code or sampling)

**Operator: "시작하자" (2026-07-18). S4 = three defenses that make the S0–S3 claims attack-proof.
Rails carried over: SEA untouchable (8/29); one GPU executor at a time; registry discipline for any
verdict-bearing experiment; hard timebox for S4 = 2026-09-30 AoE (S4 closes with the paper-2 draft).**

## S4-① Repro pack (engineering; completion-based)
Anyone with this repository must be able to recompute every verdict-bearing number without us.
DONE means: (a) `REPRO.md` — a reproduction matrix mapping each headline result (R17, R22/24/26,
R31, R33, R37, R41, R42, R43, R44, R45, R46, R47) to its script, its raw data location, and its
expected output; (b) the GPU-free replays (`s2_replay.py`, `s3_w1.py`, `s3_sprt_replay.py`,
`s3_sprt2.py`, `vsi_gate.py --demo`) verified fresh on a clean checkout, outputs recorded in
REPRO.md; (c) environment specs pinned (local Python >=3.8 stdlib-only for replays; WSL `lead` env
for 7B experiments; `lead-win` + CARLA 0.9.15 for embodied). No new claims; packaging only.

## S4-② Held-back validation protocol (writing; completion-based)
Formalize, as a short spec (`PROTOCOL.md`), the rule the evidence chain R18 → R37-caution → R47
supports: any gate surface that is REUSED across candidate iterations must be paired, before any
final adoption, with held-back fresh probes never touched during iteration; report both surfaces'
uplifts side by side (W4 format: iterated-surface uplift vs fresh-surface uplift). Includes the
honest boundary: R47 found NO decay for A6 at 2 runs/arm — the protocol is justified as insurance
priced by R45's curtailment savings, not by an observed exploit.

## S4-③ Prescriber vs blind search (verdict-bearing; the hard defense)
**Question:** did S1's structured outer loop (literature-prior prescriber) buy anything over blind
search, per GPU-hour, under the identical gate?
- **Recipe space (sealed now):** the cartesian space the S1 search moved in —
  epochs ∈ {3, 5, 10} × data ∈ {narrow, broad} × fix_ratio ∈ {none, 0.4, 0.6} ×
  fix_bucket ∈ {route, family} × targeted_ret ∈ {no, yes} × freeze ∈ {no, yes}.
  Invalid combos (fix_bucket=family with data=narrow; fix_ratio=none with data=narrow) excluded
  mechanically before sampling.
- **Blind arm:** N = 6 recipes (budget cap 8) sampled uniformly WITHOUT replacement from the space
  minus the 10 already-run recipes, seed fixed at registration time in the registry note. Each
  sampled recipe: train (LEAD native, same budget as S1 recipes) + the SAME pre-registered panel
  (fix ≤2/8; reg12 pooled ≤30%) run by the fleet, verdicts computed by the registry. No peeking, no
  resampling, no early abandonment except the standing 30-min watchdog.
- **Primary metric:** panel-passes discovered per GPU-hour (prescriber trajectory: 1 pass — A2 — in
  its first 5 recipes ≈ 30 GPU-h; blind: measured). **Secondary:** frontier coverage (does the blind
  sample bracket the repair↔retention trade-off as informatively?).
- **Sealed branches:** (a) blind finds 0 passes → prescriber advantage stands, CI reported honestly
  (N=6 is crude — stated up front); (b) blind finds ≥1 pass at comparable cost → the S2 claim
  NARROWS to "the gate makes any outer-loop search safe; structure buys convergence diagnosis, not
  discovery speed" — reported first-class; (c) any blind recipe passing the panel goes to the
  deployment gate before any adoption claim (scope rule, always).
- **Schedule:** runs AFTER the ④ prospective experiment (own prereg, early August), target window
  2026-08-20 → 09-15. GPU: ~6 × 6h serial on the 5090.

## Non-goals (sealed)
No reopening of S1's closed repair search (blind-arm recipes are for METHOD comparison; any
panel-passer still faces the deployment gate before any claim). No SEA edits. No claim that N=6
settles the AutoML question — S4-③ bounds it honestly at this budget.
