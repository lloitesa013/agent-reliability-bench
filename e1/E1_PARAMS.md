# E1 PARAMS — remaining-parameter commit (fixed BEFORE any L1+ analysis, per E1_PREREG.md e46c94d)

The seal deferred exactly three things: the trajectory-divergence threshold **X**, the paired-seed
count **K**, and the exact **L2/L3 toggle settings**. All are fixed here, before any L1+ run is
launched and before any L1+ or H3 analysis. Nothing in the seal is modified.

## Configuration audit of L0 (source-verified, zero GPU cost)

Before implementing the rungs, the harness was read to establish what L0 already locks:

| rung | intended toggle | ACTUAL state during all L0 runs | source |
|---|---|---|---|
| L1 | TrafficManager seed fixed | **already fixed at 0** (CLI default; our runner never overrode it) | `leaderboard_evaluator.py:518` (`--traffic-manager-seed`, default=0) applied at `:282` `set_random_device_seed` |
| L2 | synchronous mode / fixed physics step | **already on**: `synchronous_mode=True`, `fixed_delta_seconds=1/frame_rate=0.05` | `leaderboard_evaluator.py:73,197,198,220` |
| L3 | inference determinism | **NOT locked**: `cudnn.benchmark=True`, `cudnn.deterministic=False`, `allow_tf32=True` | `lead/inference/sensor_agent.py:47-49` |

**Consequence, stated before the runs (not after):** the L0 variance measured in R52 survived a
fixed TM seed and a fixed synchronous physics step. The mechanism H1 names (TM seed) is therefore
excluded by construction, and L1's delta over L0 is empty. Per the seal's no-line-moving rule, H1
is still judged by its stated criterion on the L1 data; the audit predicts NOT SUPPORTED.

Because L1 is configuration-identical to L0, its 12x3 runs are run as a **replication arm**: they
test whether the outcome distribution is stable across collection sessions and machine states —
the exact confound recorded as R52's honest caveat. This control is a prerequisite for attributing
any later change to the L2/L3 toggles rather than to session drift. From L1 on, GPU memory in use
is logged with every run.

## Fixed parameters

- **X (trajectory-divergence threshold) = 0.50 m.** Justification is physical, not data-derived:
  numerical/bitwise divergence manifests at mm scale, a lane is ~3.5 m wide, so 0.5 m is well above
  noise and well below the scale at which a maneuver decision has visibly changed. The full
  divergence-vs-time curve is reported for every paired comparison; X only fixes the headline
  "divergence time" number so it cannot be chosen after seeing outcomes.
- **K (paired-seed count for H3) = 5.** Within the seal's K<=10 and the 220-rollout cap
  (5 seeds x 2 arms x 3 routes = 30 rollouts).

## L2 settings (exact, to be applied on top of L1)

Synchronous mode and `fixed_delta_seconds = 0.05` are already active at L0, so L2's remaining delta
is explicit, non-default physics substepping (CARLA otherwise uses defaults that permit a
dt-dependent substep count):

```
settings.substepping            = True
settings.max_substep_delta_time = 0.0125
settings.max_substeps           = 4          # 0.0125 * 4 == 0.05 exactly -> substep count fixed
settings.synchronous_mode       = True       # already on; re-asserted explicitly
settings.fixed_delta_seconds    = 0.05       # already on; re-asserted explicitly
```

## L3 settings (exact, to be applied on top of L2)

```
torch.backends.cudnn.benchmark        = False   # was True: algorithm chosen by wall-clock timing
torch.backends.cudnn.deterministic    = True    # was False
torch.backends.cudnn.allow_tf32       = False   # was True
torch.backends.cuda.matmul.allow_tf32 = False
torch.use_deterministic_algorithms(True, warn_only=True)
torch.manual_seed(0); torch.cuda.manual_seed_all(0); numpy.random.seed(0); random.seed(0)
env CUBLAS_WORKSPACE_CONFIG=":4096:8"          # required by deterministic CuBLAS kernels
```

Applied via an env-gated block in `lead/inference/sensor_agent.py` (`E1_DETERMINISM=1`), so the
default harness behaviour is untouched and the toggle is auditable in the run log.

## Diagnostic arm (exploratory, NOT part of the sealed ladder, labelled as such)

`11755 x 12 with TM seeds 1..12` (seed VARIED instead of fixed). The ladder only ever locks
sources; this arm probes the same hypothesis from the opposite direction and costs 12 rollouts. If
varying the seed leaves the outcome distribution and the branch values unchanged, the TM seed is
irrelevant to branch selection — a positive-control complement to the audit above. Reported as
exploratory; no sealed decision line depends on it.

## Budget ledger (cap 220)

L0 36 (spent) + L1 replication 36 + seed-vary diagnostic 12 + L2 36 + L3 36 = 156, leaving 64 for
H3 paired-seed (30) and slack.
