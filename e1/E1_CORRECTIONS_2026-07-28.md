# E1 CORRECTIONS — adversarial panel review, 2026-07-28 (relay session)

Source: 5-lens independent review + adversarial verification of each verdict (11 agents).
4/5 verifications held up; the framing-strategist's "publish a priority note on branch
quantization" recommendation was REFUTED (it rested on the metric error below) and is withdrawn.
Every number below was re-derived locally before being written here.

## C1 — FACTUAL ERROR IN RESULTS.md (fix first, blocks everything)

`RESULTS.md` R52 Finding 2 states: *"Identical collision magnitude means the same impact geometry
and speed."* **This is false.** CARLA leaderboard `global_record.infractions` fields are
**events per kilometre driven**, not magnitudes. Verified arithmetically:

```
11755  col_vehicle 12.325 x 0.081137 km = 1.00001 events
11755  col_vehicle 24.650 x 0.081137 km = 2.00003 events   (L1 run 2)
18252  col_pedestrian 7.510 x 0.133153 km = 0.99998 events
3436   col_vehicle  7.572 x 0.132071 km = 1.00004 events
11755  col_vehicle 23.215 x 0.043076 km = 1.00000 events   (L0 run 7, RC=53.09%)
min_speed fields normalize to exactly 17 / 20 / 18 / 8 events
```

Correspondingly `score_composed = route_completion x prod(penalty coefficients over INTEGER
infraction counts)` — leaderboard-1.0-style multiplicative penalties (0.6 vehicle, 0.5 pedestrian;
100 x 0.6^2 = 36 verified). So the "quantization" is **the definition of the driving score**, not a
discovered property of the simulator. Everything downstream ("the simulator behaves
deterministically WITHIN a branch") is unsupported — and is contradicted by our own Finding 3 and
by L1 11755 run 6 (same outcome value 60/12.325 at 46.6 game-seconds vs 16.1-20.3 s on that branch).

**Required rewrite.** Multiply every infraction field by km_driven; report integer event counts.
Delete the impact-geometry sentence. Correct framing: *the outcome space is discrete by
construction; the object of study is the branch probability p, not the branch values.*
Caveat to avoid a new error: route_completion is continuous, so two values sit OFF the
100x0.6^k lattice — L0 11755 run 7 (31.854, RC=53.09) and L1 11755 run 5 (58.480415, continuous
outside_lane term). Cite the leaderboard-1.0 multiplicative formula, not the current 2.1 docs page.

## C2 — Finding 1 is refuted by our own L1 arm
"Reproduces its historical rate exactly (7/12 -> 7/12); flakiness is a stable property, not drift."
The configuration-identical replication returned **8/11**, and all three routes moved UP
(7/12->8/11, 6/12->7/11, 1/12->2/12; pooled 14/36 -> 17/34, Fisher p=0.47; sign test 3/3 up,
one-sided p=0.125). Nothing is significant — and that IS the point: exact Clopper-Pearson
half-width at n=12 is +/-0.29, so "reproduces exactly" was a coincidence the CI predicted.
**Rewrite Finding 1 as:** *the point estimate replicated, but per-route rates are not stably
estimable at n=12 (CI half-width +/-0.286; replication arm 8/11).*

## C3 — Run-count error + informative censoring
Current-state summary says "L1 replication arm ran (36 runs)". The committed CSV has **34**
(11755 n=11 run8 absent; 18252 n=11 run1 absent; 3436 n=12) — verified. `e1_l1_machine.csv` shows
duplicate rows exactly 21:00 apart => the 1200 s watchdog fired, retried, failed twice.
Hangs correlate with the failure mode being measured (3436 failing branch 527-625 s wall vs ~82 s
for passes), and L0 lost zero runs while L1 lost two => **non-random 6% dropout in one arm only.**
Actions: add a `censored` row type; log every attempt with termination reason; report
attempted/valid/censored per cell with Manski best-case/worst-case bounds; raise
`$perRunTimeoutSec` 1200 -> >=2400; STOP `Remove-Item -Recurse -Force $out` before retries (it
destroys evidence). Do NOT assert the bias direction — state it as unresolved.

## C4 — The sealed H1 line is arithmetically degenerate (report this, do not hide it)
Minimum attainable exact CP width at n=12 is **0.2646** (only at k=0 or k=12). H1 thresholds are
0.2858 / 0.2891 / 0.1913 => route 3436's acceptance region is the **EMPTY SET for any outcome**;
the other two are satisfiable only under total determinism (which is H2, not H1). At realized
n=11, 0/11 gives width 0.2849 vs threshold 0.2858 — margin **0.0009**, i.e. the verdict turns on
which two runs the watchdog killed.
**Do:** publish the CP-width-vs-k table, report H1 NOT SUPPORTED per the no-line-moving rule, then
pre-register a REPLACEMENT variance functional in a new dated commit (p(1-p) ratio, or
first-divergence-tick distribution). Self-diagnosing an infeasible pre-registered rule is a
publishable methodological artifact; being caught on it is fatal.

## C5 — Power: the rate comparison cannot answer the question
Exact two-sided Fisher at n=12/arm: 0.583 vs 0.29 -> **power 0.200**; 0.583 vs 0.25 -> **0.262**.
Need ~55/arm (0.83) and ~40/arm (0.82). A toggle that genuinely HALVED the failure rate would be
reported as "no effect". Plus the L1 between-session movement puts the measurement floor at
roughly +/-0.35-0.40 in failure rate — larger than any plausible L3 effect short of total
determinism. **Whatever L3 returns is uninterpretable as a rate comparison.** Internalize before
spending the remaining ~120 rollouts of the sealed 220 budget.

## C6 — L2 as specified is a physics CHANGE, not a determinism lock
`E1_PARAMS.md` claims CARLA "otherwise uses defaults that permit a dt-dependent substep count."
Wrong: defaults are `substepping=True, max_substep_delta_time=0.01, max_substeps=10`, and
`fixed_delta_seconds=0.05` was already locked at L0 => substep count is already a fixed 5 x 0.01.
Setting **0.0125 x 4 changes the integrator from 5 substeps to 4** — a perturbation, not a lock.
Either set **0.01 x 5** (true no-op lock, reported as a null rung) or report L2 explicitly as a
physics change whose effect is unattributable. FIX BEFORE spending 36 rollouts on L2.

## C7 — Nothing logged can produce the pre-registered X=0.50 m divergence metric
Both CSVs contain end-of-route aggregates only: no pose, no per-tick data, no recorder.
The paper would have to report a pre-registered metric it never computed.
**Add per-tick logging now (before the next rollout): (frame, x, y, yaw, v, steer, throttle,
brake) + scenario trigger frame.** Highest-leverage instrumentation change in the project.

## C8 — The 2-rollout experiment that beats 220 rollouts of rate comparison
With `E1_DETERMINISM=1`, run 11755 twice and hash BOTH the observation stream and the control
stream tick-by-tick. Decision rule: **observation diverges before control => server-side;
control diverges given an identical observation => inference-side.** Two runs answer the
localization question that rate comparison at this n provably cannot.

## C9 — Free result already in the data (verified independently, panel under-exploited it)
Route **18252**, pooling all 23 configuration-identical L0+L1 runs: `duration_game` **separates
the branches perfectly** — all 13 FAIL in [21.45, 22.15] s, all 10 PASS in [22.30, 25.80] s, zero
overlap. Under exchangeability that is 1/C(23,13) = **8.74e-7**. All 23 have route_completion=100,
so it is not a truncation artifact. Caveat: duration is DOWNSTREAM of the yield/no-yield decision,
so this shows a clean binary behavioural mode, not an arrival-time race — separating those needs
the divergence-time instrumentation of C7. Directly Paper-2-relevant as a **curtailment signal**
(cheap early predictor of the branch).

## C10 — Prior art to cite BEFORE a reviewer finds it
- **Osikowicz, Shin & McMinn, Flaky Tests Workshop @ ICSE 2025** — CLOSEST prior work: CARLA
  Leaderboard 1.0 official benchmark, TransFuser++ (same model family), 10 repeats/scenario, and
  its flakiness instrument is literally *the number of unique infraction-count behaviours per
  scenario* = our "distinct outcome VALUES". 31.3% (40/128) flaky. Lead with the delta or it reads
  as rediscovery.
- Nagarajan, Warnell & Stone (2018) — nondeterminism ablation methodology in deep RL (same design).
- "Non-Determinism in TensorFlow ResNets" (arXiv:2001.11396) — attributes ~74% of accuracy sd to
  exactly the cudnn/kernel-selection layer that is our L3.
- Colas et al. (arXiv:1806.08295), Agarwal et al. rliable (NeurIPS 2021) — the "use rate-based
  statistics" conclusion already exists in deep-RL evaluation.

## VERDICT (unanimous across 5 lenses, undisturbed by verification)
**Not a standalone paper. Section 3 of Paper 2 ("The Price of the Gate"), ~1-1.5 pages + appendix.**
It buys the Bernoulli noise model the cost law rests on and pre-empts the most predictable reviewer
objection: *"if the environment is flaky, just make it deterministic and your gate is unnecessary."*
Hard cap: past 1.5 pages it dilutes the economics thesis and becomes net negative.
Standalone fallback ONLY after fixes: Flaky Tests Workshop @ ICSE (2-page abstract) — but that is
where the closest prior work was published, so it must lead with the delta.
NOT viable: NeurIPS/ICML/ICLR/CoRL/CVPR main track; ICSE/FSE/ISSTA main track (their n is 1000x ours).
**Aug 29 firewall: HOLD.** L0's 7/12 would have looked like it strengthened Paper 1; the L1 arm's
8/11 shows the honest read is "unresolved to +/-0.29", which would have weakened Paper 1's framing.
