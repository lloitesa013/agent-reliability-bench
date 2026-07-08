# CARLA verified self-improvement loop — progress

Goal (Phase-1): the higher-level embodied version of verified self-improvement, combining the
operator's unique assets (CARLA+LEAD env, B2D failure taxonomy, X-MoD interpretability) with the
watcher/judge/verification built in the reliability study. Loop: drive -> WATCHER observes failures
-> JUDGE attributes -> MEMORY induces transferable rules -> policy changes -> VERIFY transfer to
unseen routes (not overfit/reward-hacked).

## Step 1 — WATCHER (DONE, on real B2D-220 data)
LEAD tfv6 baseline eval over ~203 route results (`carla_failure_taxonomy.json`, from
`build_pdmlite_failure_taxonomy.py` via `run_taxonomy.ps1`). 201 non-clean routes:
- **min_speed pervasive: 199/201** — the agent is systematically over-conservative (slow), but these
  still score ~100 (soft penalty). A behavioral trait, not a task failure.
- **10 hard failures (score<60)**; excluding 3 infra crashes (sim-crash / TickRuntime), **7 real
  driving failures**: vehicle collisions x3 (3436, 11755, 2201), pedestrian collisions x2 (18252,
  24224), agent-blocked (3905, 24041), highway off-route (24041).

## Step 2 — JUDGE (DONE, scenario attribution)
Each hard failure attributed to its Bench2Drive scenario (`get_scenarios.ps1`):
```
route  town     scenario                              failure
11755  Town12   EnterActorFlow                        vehicle collision + route_dev
2201   Town12   EnterActorFlow                        vehicle collision + red_light
18252  Town12   ParkingCrossingPedestrian             pedestrian collision
24224  Town02   DynamicObjectCrossing                 pedestrian collision
3436   Town13   HazardAtSideLaneTwoWays               vehicle collision x3
3905   Town13   VanillaNonSignalizedTurnEncounterStopsign  blocked
24041  Town13   HighwayExit                           layout + blocked + off-route
```
**Failure clusters:** (a) **merging into traffic (EnterActorFlow) -> vehicle collisions**; (b)
**pedestrian-crossing scenarios -> pedestrian collisions** (safety-critical); (c) side-lane hazard
-> collisions; (d) stop-sign turns -> blocked; highway exit -> off-route.

## Next
- **MEMORY (analyzable next):** induce candidate transferable rules from the clusters
  (e.g. "in EnterActorFlow, yield/slow before merging"; "in pedestrian-crossing scenarios, cap speed
  and prepare to stop"). Verification-as-selection over rules (as in loop_v2).
- **POLICY-CHANGE + VERIFY (the slow hard core):** applying a rule to LEAD tfv6 (an end-to-end model)
  needs a rule-based override layer or retraining; verifying transfer needs RE-RUNNING held-out
  routes in CARLA (minutes/route). This is the genuinely hard, slow part — consistent with the
  text-loop finding that TRANSFER is the frontier (the verifier is the proven, working piece).

## Step 3 — VERIFY pipeline CONFIRMED + a critical finding (2026-07-09)
Re-ran route 11755 (EnterActorFlow, baseline: collision + route_dev, score 29.6 = FAILED) with the
unchanged tfv6 agent (`run_lead_route.ps1`, CARLA up in 3s, ~2min/route).
- **RE-RUN: status Completed, score 100.0, only min_speed — NO collision, NO route_dev = PASSED.**
- **CARLA driving failures are FLAKY/stochastic**: the same route+agent, zero changes, flips
  FAILED -> PASSED across runs (traffic/timing stochasticity).
- **Implication:** single-run "the rule fixed it" verification is meaningless — the failure may not
  reproduce anyway. Verified self-improvement in CARLA REQUIRES multi-run failure-RATE estimation
  (statistical), comparing rates before/after a change. A flaky non-reproduction masquerading as a
  "fix" is the embodied analog of the fake-self-improvement (seen-gain) problem from the text loops.
- **This unifies the whole study:** verified self-improvement needs rigorous verification (held-out
  transfer / statistical failure-rate); naive improvement is fake. The verifier is the value — the
  field's exact gap (reward-hacking / illusion of progress), now shown in BOTH text and driving.

## Next (the honest hard core, unchanged)
- Multi-run baseline: run each failure route N times -> per-route failure RATE (stable vs flaky).
- Policy-override layer on tfv6 (watcher-as-enforcer: e.g. speed cap / yield in the failing scenario).
- Verify: does the override lower the failure RATE on held-out same-scenario routes (real transfer)?
