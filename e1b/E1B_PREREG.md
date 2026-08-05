# E1-B PREREG — paired divergence localization (SEALED before any E1-B rollout)

Separate accounting from E1's sealed 220-rollout budget. Budget: **<= 8 rollouts, 5 days**.
On cap/timebox: report as-is. Paper-1 firewall (nothing from E1/E1-B enters the 8/29 submission)
and the S1 seal remain untouched. E1-B uses its own registry epoch.

## Why this instead of more rate comparison

E1's ladder closed as-is at 70/220. The reason is arithmetic, not impatience: exact two-sided
Fisher at n=12/arm has **power 0.20** against a halving of the failure rate (E1_CORRECTIONS C5),
and the observed between-session movement puts the measurement floor near +/-0.35-0.40 in rate —
larger than any plausible L3 effect short of total determinism. More rollouts spent on rates would
buy uninterpretable numbers. This experiment measures **trajectories**, not rates, and localizes
the entry point of nondeterminism with 4 rollouts.

## Question

When two runs with identical seed and identical configuration diverge, does the divergence enter
(a) server-side (CARLA sim/physics/traffic), (b) inference-side (same input, different output),
(c) control-stack-side (PID/shield/post-processors), or (d) not at all under `E1_DETERMINISM=1`?

## Instrumentation (C7, required before the first rollout)

Per tick, one appended row, full precision (no rounding):
`step, sim_time, obs_hash, net_out_hash, ctrl_hash, x, y, yaw, speed, steer, throttle, brake`

- `obs_hash`    = SHA256 over the exact tensor bytes of the dict passed to
  `closed_loop_inference.forward(...)`, taken immediately before the call (post-preprocessing;
  raw tensor bytes, never re-encoded imagery).
- `net_out_hash`= SHA256 over the raw `ClosedLoopPrediction` fields, taken immediately after the
  forward call and **before** any post-processor (`force_move`, `stop_sign`, throttle scaling).
- `ctrl_hash`   = SHA256 over the final applied `carla.VehicleControl` (steer, throttle, brake),
  after post-processors, shield, and the initial-frames override.

Three separate streams are what make server / inference / control separable. Logging is env-gated
(`E1B_TICKLOG`) and wrapped so it cannot alter or abort a run.

**C7-b (strict determinism probe).** One run with `torch.use_deterministic_algorithms(True)` in
**strict** mode (`E1B_STRICT=1`, i.e. `warn_only=False`). If it raises, the named op has no
deterministic implementation and is itself the localization. If it does not raise, every op used
has one. The outcome is recorded either way.

## Design (4 rollouts; optional replication +4)

Route **11755** (highest observed variance), CARLA default physics (**L2 substepping is NOT
applied** — per E1_CORRECTIONS C6 the 0.0125x4 spec changes the integrator from 5 substeps to 4,
i.e. a perturbation, not a lock):

- **det=0 pair**: 2 rollouts, `E1_DETERMINISM=0`, everything else identical (TM seed 0).
- **det=1 pair**: 2 rollouts, `E1_DETERMINISM=1`, everything else identical.

Optional replication on route 18252 with the same 4-cell design.

## Decision rules (fixed here; no line may move after the first rollout)

Within a pair, compare tick-by-tick and record the first differing tick per stream:
`t_obs`, `t_net`, `t_ctrl` (absent stream => no divergence observed in that stream).

- **HB1** — det=1 pair is bit-identical on all three streams for the whole episode
  => the source is L3 (cudnn/kernel selection). Conclusion: *determinism is purchasable.*
- **HB2** — det=1 still diverges and `t_obs < t_net`
  => server-side source (sim/physics/traffic).
- **HB3** — det=1 diverges with identical `obs_hash` but differing `net_out_hash`
  (`t_net <= t_obs`) => residual inference nondeterminism the flags did not cover.
- **HB4** — `obs_hash` and `net_out_hash` identical but `ctrl_hash` differs
  => control-stack source (post-processors / PID / shield).

Only the **first** divergence is interpreted; everything after it is feedback-contaminated.
The **magnitude** of the first difference is reported alongside its tick, to distinguish
rounding-level from behaviour-level divergence. Ambiguous or unreachable verdicts are reported
as such, not resolved by choice.

## Deliverables

`e1b/E1B_PREREG.md` (this file, committed before the first rollout) - `e1b/data/*_ticks.csv` -
`e1b/compare_pair.py` (stdlib) - `RESULTS.md` R53 - push to origin research-sea only.
