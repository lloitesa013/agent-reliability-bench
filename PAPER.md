# The Verifier Is the Loop: Self-Improving Agents Must Not Trust Their Own Changes

*Workshop-format draft (v0). All numbers reproduce from this repository (`RESULTS.md` R1–R25); single RTX 5090.*

## Abstract

Self-improving agents distill their own failures into modifications of their prompts, rules, memories, and
tools — and report gains measured on the tasks those modifications were derived from. That measurement is
broken: recent work shows all eight major agent benchmarks are reward-hackable to ~100%, and we find the
same failure inside the self-improvement loop itself. Across controlled tasks, two real code benchmarks, and
an embodied driving domain, we show that **fake self-improvement is the norm, not the exception**: candidate
self-modifications that look perfect on the data that produced them routinely fail held-out evaluation
(29% of MBPP problems yield a genuine overfit; 100% of tool-rewrite decisions contain a reward-hack;
a plausible driving fix raises the failure rate from 50% to 83%). We propose treating the agent's own
modifications as untrusted and gating adoption behind a two-part verifier — a **hidden test** (held-out
transfer) and a **regression check** (no existing capability degrades) — and show this **self-verifying
agent** wins the adoption decision reliably: naive seen-score selection ships broken modifications 59% of
the time on tool rewrites and 19/29 times when an overfit exists on MBPP; the verifier ships 0% while still
adopting real improvements (88% of tool decisions; 68% of MBPP problems). We also report what does *not*
work — the multi-round cumulative advantage is small and noisy at scale, and in the embodied domain
failures are so stochastic that only statistical (rate-based) verification is meaningful — and argue these
honest negatives are exactly why the verifier, not the self-improvement heuristic, is the load-bearing
component.

## 1. Introduction

The self-improving-agent literature (Reflexion, ExpeL, AutoGuide, Voyager, and successors) shares a
template: run the agent, collect failures, distill them into a reusable modification (a prompt rule, a
skill, a memory, a tool), and measure the gain. The measurement is almost always taken on the same tasks or
distribution the modification was distilled from. Two facts make this template unsafe:

1. **Benchmarks are gameable.** UC Berkeley (April 2026) demonstrated that all eight major agent benchmarks
   can be reward-hacked to near-perfect scores. A score increase is not evidence of improvement.
2. **The loop supplies its own fakes.** We show below that an agent's own proposed modifications are
   frequently overfit (lookup-hacks, hardcoded solutions, over-broad rules) that ace what they were derived
   from and fail everywhere else. A self-improvement loop that trusts its own changes converges toward a
   confident reward-hacker, not toward capability.

We take the position that the missing component is not a better inducer but a **verifier**: an adoption gate
that treats every self-modification as untrusted. Concretely, a candidate is adopted only if it (i)
improves the target capability **on held-out data** it was not derived from (the *hidden test*), and (ii)
degrades **no existing capability** beyond a noise margin (the *regression check*). When no candidate
clears both, the agent abstains — the safe default.

Our contributions: (a) evidence across four self-modification surfaces ({prompt, rule, memory, tool}) and
three domains (controlled text, real code, embodied driving) that fake improvement is pervasive; (b) a
demonstration that the two-gate verifier reliably wins the *single adoption decision* where naive
selection fails; (c) an honest negative — the per-decision advantage does not (yet) compound into a large
multi-round cumulative gap on our substrates; (d) an embodied instantiation showing that driving failures
are flaky (a route flips PASS/FAIL run-to-run at 58%), that single-run failure labels used by driving
leaderboards are unreliable, and that a statistical (failure-rate) verifier correctly rejects a plausible
but harmful fix before deployment; (e) `vsi_gate.py`, the verifier as a dependency-free library.

## 2. Method: the self-verifying agent

Let an agent have capabilities measured by evaluators E = {e_1..e_k} (its *profile*) and let a proposer
(the agent itself, prompted over its own failures) emit candidate modifications c_1..c_n for a target
capability t. The gate:

- **Hidden test:** Δt(c) = e_t^held-out(agent + c) − e_t^held-out(agent) must exceed a gain margin. The
  held-out set must be data the candidate was not derived from; on code, *execution* of the candidate on
  held-out tests is the evaluator.
- **Regression:** min_j [e_j(agent + c) − e_j(agent)] must not fall below −(noise margin), over all
  existing capabilities j.
- **Selection:** among candidates passing both, adopt the best-hidden one; else adopt nothing.
- **Stochastic domains:** replace scores with failure *rates* over N runs and require a one-sided
  significant reduction (exact binomial). Single runs are meaningless when the baseline itself is flaky.

Two design lessons we verified empirically: the hidden set must be large or fresh per selection round —
reusing a small set across many selections re-inflates scores (+0.045 optimism from a 20-item set reused
over 8 candidates, R18); and the proposer is itself unreliable (prior-anchoring: models rationalize
failures instead of fitting them, R9), which is *why* verification-as-selection, not proposer quality, is
the load-bearing mechanism (R10).

## 3. Experiments

**Substrate (R1–R7).** A reading judge with calibrated abstention over agent traces: 0.949 vs a tuned
guardrail's 0.789 on a sealed benchmark; abstaining the least-confident 30% lifts 0.84→0.91 and transfers
to 4/6 unseen domains; fine-tuning a 7B judge reaches 0.815 in-distribution on HaluBench but does not
transfer cross-source — the first sighting of the central phenomenon.

**Fake improvement is the norm.**
- *Fine-tuning level (R11):* among three candidate LoRA fine-tunes, the one trained on a single source
  memorizes it (train loss 0.000, in-dist 0.993) and generalizes worst (held-out 0.507). Naive selection
  (best in-dist) deploys exactly this memorizer; held-out selection deploys the broad candidate (0.578).
- *Regression is silent (R12):* fine-tuning to fix one weak source costs another source −0.16 while the
  target barely moves (+0.005); a 1-D "did the target improve?" check misses it; retention-replay removes
  it (worst old Δ = 0.000). This reproduces, on a judge, the classic naive-correction backfire.
- *Tool rewrites (R15, R17):* asked to extend its own Python tool, the agent's proposals include a
  hardcoded lookup of the observed examples in **100% of 24 decisions**; seen-score cannot distinguish it
  from the general tool (both 1.00), so naive selection ships a broken tool **59%** of the time. Executing
  candidates on held-out inputs ships **0%**, while still adopting a working tool in 88% of decisions.
- *Real code (R22, R24, R26; MBPP n=100 + HumanEval n=80):* a genuine overfit (passes the shown test,
  fails held-out tests) appears on **29% / 30%** of problems respectively; where it exists, naive ships it
  **19/29 and 21/24** times, the verifier **0 on both**. Restricted to problems the agent can solve
  (removing the abstain-on-unsolvable confound), naive ships a held-out-failing solution **12% on both
  benchmarks**, the verifier **0%**. Two independent benchmarks, near-identical rates: the effect is a
  property of the self-modification stream, not of a dataset.

**The flagship decision (R13).** An agent with capability A (1.00) and target B (0.00) proposes rule-edits
from its own failures. The verifier adopts the edit that fixes B *and* keeps A — **rejecting the edit that
fixes B best (hidden 1.00) because it regresses A** — ending at A=1.00, B=0.83 (mean 0.92) vs naive 0.67.
The autonomous loop closes when the proposer is forced to fit patterns rather than rationalize (R9–R10):
the model induces the true hidden rule among candidates, and the verifier selects it (2/2 tasks).

**Honest negative: cumulative compounding (R14, R16).** Over 3 acquisition rounds the gated agent leads at
every round (0.88 vs 0.79 final). At 5 rounds the cumulative-mean advantage shrinks to +0.03 and worst-case
capability is dominated by new-task acquisition noise, not regression. The robust claim is the
per-decision advantage; large multi-round compounding is not demonstrated on these substrates.

**Embodied (R19–R21, R23, R25; CARLA/Bench2Drive, LEAD tfv6).** Re-running routes unchanged: route 11755
fails 7/12 runs (58%; 95% CI [0.32, 0.81]) — the same route, agent, and weather flip PASS/FAIL —
while three routes labeled FAILURE by the single-run benchmark pass 100% on re-run (their recorded
failures were one-off noise). Consequently (i) single-run failure labels, the leaderboard standard,
measure stochasticity as much as capability; (ii) verification must be rate-based: confirming a
58%→20% fix needs ~19 runs/arm at 80% power (58%→40% needs ~91). We then ran the first full embodied
propose→verify cycle with a live control-layer intervention: the plausible fix ("drive more
conservatively") *raised* the observed failure rate (50%→83%; collisions persist and slow driving adds
penalties) and the statistical verifier **rejected it before deployment**. No accepted embodied fix exists
yet; producing one (learning-based repair with retention) is future work with a now-specified
verification cost.

## 4. What is and is not established

**Established.** (1) The agent's own modification stream is adversarial by default — overfits and
lookup-hacks are pervasive across surfaces and domains, including real code. (2) Seen/derivation-set score
carries no signal against them; held-out (and on code, *executed*) evaluation does. (3) The two-gate
verifier converts this into a reliable per-decision win: 59%→0%, 19/29→0, and a harmful embodied fix
rejected. (4) Regression checking is necessary — the best target-fix can be the worst adoption.

**Not established.** Large cumulative multi-round gains; verified self-improvement on hard tasks where no
candidate transfers (the verifier then correctly adopts nothing — safe, but not improvement); an accepted
embodied fix; behavior beyond a 7B proposer and benchmark-scale substrates.

## 5. Related work and precise claim boundaries

Self-improving agents (Reflexion; ExpeL; AutoGuide; Voyager; Trace2Skill; ERL) supply the proposer we
gate. That self-generated "improvements" are frequently fake is **established**: reward hacking of agent
benchmarks (Berkeley RDI, 2026: all 8 major benchmarks exploitable), spontaneous reward hacking in frontier
agents (METR: >30%), metric falsification by a self-modifying system (the Darwin-Gödel-Machine episode),
and confabulated self-reflection ("Honest Lying", 2026). Gating self-modification is itself an emerging
2026 thread: PACE applies anytime-valid acceptance tests to prompt/strategy updates (text tasks);
Self-Harness uses held-in/held-out threshold gates (no statistics); GRACE proposes an update-rejection
gate; "Beyond Binary Success" brings sequential statistics to robot policy comparison (offline, outside
any loop). Driving-benchmark noise is quantified (CARLA non-determinism; ~5 DS run-to-run variance on
Bench2Drive, with published scores typically single-run). **We therefore do NOT claim**: the first
observation that self-improvement is often fake; the first statistical gate for self-improving agents; or
the discovery of driving-benchmark flakiness. **What is unclaimed before this work, to our knowledge**:
(a) fake-improvement *rates measured across all four modification surfaces* ({prompt, rule, memory, tool})
under one controlled protocol, plus two real code benchmarks; (b) a *pre-registered, error-controlled
statistical accept AND reject operating inside a closed self-improvement loop on an embodied (CARLA
Bench2Drive) system* — including the operationalization of run-to-run noise into a failure-rate gate; and
(c) the deployment-scope result (a narrowly-scoped gate accepts a change whose collateral damage a broader
gate catches), demonstrated on a real system. Continual learning knows regression as catastrophic
forgetting; hallucination detection and calibration provide our substrate judge and abstention machinery.
(This section reflects a July 2026 literature scan; the space moves fast — re-verify before submission.)

## 6. Limitations

7B proposer; controlled/benchmark substrates; six-item held-out sets on toy loops (0.17 granularity);
N=6/arm embodied arms (directional, not conclusive — the rejection verdict requires only absence of
improvement evidence); one vehicle stack and simulator. The hidden-test-freshness result (R18) implies
gate reuse must be budgeted, which we spec but do not automate. Multi-round compounding may appear on
richer substrates; we did not demonstrate it.

## 7. Conclusion

Across text, real code, and embodied driving, self-improvement without verification ships fakes at rates
of 12–83%, and the fakes are supplied by the loop itself. A verifier that demands held-out transfer and
no regression — nothing more exotic — reduces shipped fakes to zero in our experiments while still
adopting real improvements, and its statistical form correctly rejects a harmful embodied fix before
deployment. If self-improving agents are to be trusted with their own modification stream, **the verifier
is the loop**; the rest is a proposal generator.

---
*Artifacts: `RESULTS.md` (R1–R25, each with its script), `vsi_gate.py` (the gate as a library),
`STUDY.md`/`NOTE.md` (narratives). Repository: github.com/lloitesa013/agent-reliability-bench.*
