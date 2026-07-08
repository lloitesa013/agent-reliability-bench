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
