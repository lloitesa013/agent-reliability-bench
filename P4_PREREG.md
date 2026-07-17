# ④ Pre-registration — The Streamed Gate, Live (SEALED 2026-07-18, before any probe beyond W4's k1-2)

**Purpose.** Paper 2's headline experiment: run ONE new embodied repair cycle end-to-end where
(a) the outer loop is the AUTOMATED machinery (registry + prescriber propose; gate decides), and
(b) every confirm batch runs under CURTAILMENT (R45's shippable rule) with live run-count
accounting vs the fixed-N counterfactual. Success = the streamed gate runs live with measured
savings and correct verdicts. An adopted repair is a bonus, not the bar.

## Target selection (mechanical, sealed before the probe batch)
Candidate pool from committed W4 data — fresh-surface routes with baseline 2/2 fail BY COLLISION,
excluding the closed 11755 and all EnterActorFlow-family routes: **{17655, 26401, 27018, 28330}.**
Probe batch: baseline x6 more runs each (k3-8; k1-2 exist) + privileged-expert x2 each.
SELECT the route with the highest confirmed baseline fail rate over k1-8 among those where the
expert is clean 2/2 (teacher exists); ties -> smallest route id. If NO route has expert 2/2 clean
-> the cycle stops there and reports "no teacher available" (an honest terminal state).

## The automated cycle (zero human verdicts, human = launcher/ops only)
1. Register target trial; prescriber proposes from its ladder (empty target-history -> its rules
   decide; it may propose control-probe first — that is the ladder, let it).
2. Any training recipe: LEAD native, retention per PROTOCOL.md (>=20% of evaluation surface held
   back from iteration from day one).
3. Panel: fix arm n=8 (pass <=2/8), retention reg12-style pooled <=30%, ALL confirms curtailed.
4. Panel pass -> deployment screen (169-clean) + curtailed interleaved confirms (R33 rules) +
   held-back fresh probes (PROTOCOL.md) before any adoption claim.
5. Budget: max 2 recipes for this target (this is a demonstration cycle, not a new repair search);
   every verdict computed by the registry; run-count ledger (streamed vs fixed) is a deliverable.

## Sealed lines
- Cycle validity: >=1 recipe reaches a panel verdict with zero human verdict intervention.
- Streaming claim: curtailed confirms reach verdicts identical to their fixed-N counterfactual on
  the same runs (identity by construction; the ledger reports runs saved).
- Honest branches: expert unavailable -> report; both recipes REJECT -> report (the gate protected
  the floor again); panel pass + deployment REJECT -> report (scope rule demonstrated live, again).
Timebox: complete by 2026-08-15 AoE. GPU: probe ~28 runs, then per-recipe ~6h train + panels.
