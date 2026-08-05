# E1-B2 PREREG — SEALED 2026-08-05, after the E1-B instrument defect (new seal, not a moved line)

E1-B ran its sealed plan (9 attempts / 5 valid completions, at its <=8-rollout cap) and returned
**one valid result and one instrument defect**. Per E1B_PREREG's "ambiguous or unreachable verdicts
are reported as such", no HB verdict is claimed from it. This is a NEW seal with its own budget;
E1-B's lines are untouched and its data stands as recorded.

## What E1-B established (kept)

**C7-b: strict determinism does NOT raise.** With `torch.use_deterministic_algorithms(True)` in
strict mode (`warn_only=False`), route 11755 ran to completion (score 100). Every operator the
model uses therefore has a deterministic implementation available; nondeterminism cannot be
attributed to an op that simply lacks one. (Independent of the defect below: this result needs no
hash stream.)

## What E1-B could NOT establish (the defect)

`net_out_hash` was the constant string `"ERR"` on **every tick of every run** (351/351, 1682/1682,
...). Cause: the v1 byte image called `tensor.numpy()`, which raises `TypeError` on half/bfloat16
tensors — the model's prediction heads. A broad `except` turned that into a constant. Consequently
"the network stream is identical across the pair" was **an artifact of a constant error string, not
evidence**, and HB2 vs HB3 (server-side vs inference-side) was never decidable from that data.
The obs stream did vary per tick and differed between pair members from tick 1 — but with the
physical state (x, y, yaw, speed, steer, throttle, brake) *bit-identical* at that tick, the cause
(sensor-render nondeterminism vs a run-varying metadata field inside the observation dict) is
unresolved without per-field attribution, which v1 did not record.

## Instrument v2 (deployed before this seal was committed)

- dtype-safe byte image: tensors are hashed through `float64` (exact for f16/bf16/f32) or `int64`,
  with dtype+shape mixed in; no dtype can silently degrade a stream to a constant again.
- **per-key hashes** for both streams (`obs_keys`, `net_keys` columns: `key:hash|key:hash|...`),
  so a divergence is attributed to a specific input field or prediction head.
- failures are named, never swallowed: a per-key hash that raises is written as `ERR<ExcType>`.

## Instrument validation gate (must pass BEFORE rollouts are spent on the science)

Run 1 of this seal is admissible only if, within that single run, **both** hold:
1. `net_out_hash` contains no `ERR` and takes **>1 distinct value** across ticks;
2. `obs_hash` contains no `ERR` and takes **>1 distinct value** across ticks.
If either fails, the plan aborts before further rollouts and the failure is reported as-is.
(This is the lesson of the defect, made mechanical: verify the instrument before believing it.)

## Design (4 rollouts, route 11755, CARLA default physics, TM seed 0)

- det=1 pair: 2 rollouts with `E1_DETERMINISM=1`
- det=0 pair: 2 rollouts with `E1_DETERMINISM=0`
(`E1_L2` is never set — per E1_CORRECTIONS C6 the substep spec is a physics change, not a lock.)

## Decision rules (unchanged from E1B_PREREG, restated so this seal stands alone)

Per pair, first differing tick per stream: `t_obs`, `t_net`, `t_ctrl`; sealed pose threshold
X = 0.50 m.
- **HB1** det=1 pair bit-identical on all three streams => source is L3 kernel selection.
- **HB2** `t_obs < t_net` => server-side (sim/physics/traffic/sensor render).
- **HB3** obs identical, `net_out` differs (`t_net <= t_obs`) => residual inference nondeterminism.
- **HB4** obs and net_out identical, control differs => control stack (post-processors/shield).
Only the first divergence is interpreted; magnitudes are reported alongside ticks. With per-key
hashes, HB2 additionally names the diverging field, and a divergence confined to a metadata field
is reported as such rather than being read as sensor nondeterminism.

## Budget

**<= 5 rollouts** (1 validation + 4 science; retries after infrastructure kills count as
attempts and are ledgered), timebox 3 days. On cap: report as-is.
