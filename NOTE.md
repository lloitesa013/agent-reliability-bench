# Verified Self-Improvement: the verifier is the selection mechanism

*Technical note. Single RTX 5090, Qwen2.5-7B, synthetic + public (HaluBench) data, and an embodied
driving domain (CARLA / Bench2Drive-220). All code and numbers in this repo (`RESULTS.md` R1–10).*

## Abstract
Self-improving LLM agents learn from their own failures, but the field cannot tell whether an
"improvement" is real: in April 2026 UC Berkeley showed all eight major agent benchmarks are
reward-hackable to ~100%. We argue the missing, valuable component is a **verifier** that confirms an
improvement *transfers to held-out cases* (and, in stochastic domains, lowers a *failure rate*, not a
single run). We show, across synthetic tasks and a public faithfulness benchmark and an embodied driving
domain, that (i) fake improvement is the norm — prompt-rule induction and cross-source fine-tuning do not
transfer, (ii) a naive "measure on what you learned from" reading accepts these fakes, (iii) a held-out
verifier rejects them and *accepts* the rare real improvement, closing an autonomous loop, and (iv) in the
embodied domain, failures are flaky, so verification must be statistical. **The verifier — not the
self-improvement heuristic — is the reliable, transferable piece.**

## 1. The problem
"Self-improving agent" work (Reflexion, ExpeL, Meta-Reflexion, Trace2Skill, ERL, …) distils an agent's
trajectories into rules/skills and reports gains — usually measured on the same tasks the rules were
distilled from. That measurement is exactly what the reward-hacking result breaks: a change that raises the
*seen* score need not be a real, transferable improvement. We separate the two roles: the *inducer* proposes
candidate improvements; the *verifier* decides which are real.

## 2. Method
- **Verifier = held-out transfer.** A candidate improvement is applied and evaluated on data with **disjoint
  keys** from what it was derived from (fact-group splits; held-out parameter values; held-out benchmark
  sources). Accept iff held-out performance rises beyond a margin. In stochastic domains, "performance" is a
  **failure rate over N runs**, not a single run.
- **The loop.** fail → the inducer proposes K candidate rules/policies from the failures → the verifier
  scores each on held-out → adopt only those that transfer. Verification is the *selection* step, not a
  post-hoc check.

## 3. Results
- **Reliability substrate (R1–7).** A zero-shot *reading* judge beats a tuned groundedness rule on a sealed
  benchmark (0.949 vs 0.789; an embedding watcher only ties — reading the trace is load-bearing). A
  calibrated-abstention signal (logit margin) lets it abstain the least-confident 30% → 0.84→0.91, and this
  transfers to 4/6 unseen domains. On public HaluBench, 7B zero-shot 0.688; fine-tuned 0.815 **in-distribution**
  but **not** cross-source — the negative that motivates the thesis.
- **REJECT is the norm (R6, loop_v2, faith_loop).** Prompt-rule induction and cross-source fine-tuning do
  not transfer; the verifier robustly rejects them and is not fooled by seen-gains.
- **ACCEPT + discrimination (R8).** On 4 hidden-convention tasks, the verifier accepts a general rule that
  transfers (held-out +1.00) and rejects an overfit rule that the naive seen-only baseline accepts. Correct 4/4.
- **Autonomous loop closed (R9–10).** A *naive* inducer fails by **prior-anchoring** (it rationalizes its
  failures with its prior, e.g. keeps "overtime after 8h" / "dozen = 12"). A *pattern-fitting* inducer
  (ignore priors; fit output = f(input)) proposes candidates that include the true rule, and the verifier
  adopts the transferring one (2/2). The model's candidates are unreliable; the verifier makes the loop work.
- **The self-verifying agent — {prompt/rule/memory/tool} (R11–R16).** The verifier turned on the agent's OWN
  self-modifications, adopting one only if it clears a HIDDEN TEST (transfers to held-out) AND a REGRESSION
  check (breaks no existing capability). Fine-tuning level (R11): naive deploys the memorizing fine-tune
  (in-dist 0.993 / held-out 0.507), the verifier the generalizer (0.578). Regression (R12): naively fixing
  one source silently regresses another by −0.16; the 2D gate catches it. Flagship (R13): the agent fixes B
  (0→0.83) while keeping A, REJECTING even the best B-fix because it regresses A (0.50→0.92 vs naive 0.67).
  Tool/code (R15): 2 of 3 of the agent's own tool rewrites are lookup-hacks (seen 1.0, executed held-out 0.0)
  — seen can't tell them apart, executing on held-out can. **Honest scope (R14, R16):** the CUMULATIVE
  multi-round advantage is small and noisy at scale (final mean Δ≈+0.03 over 5 rounds); the robust result is
  the SINGLE adoption DECISION, where the verifier reliably catches the fake/regression/hack that fools naive.
- **Embodied statistical verifier (R19–R20, CARLA/Bench2Drive).** Re-running real routes N times (fresh
  CARLA, LEAD tfv6 agent, zero changes) quantifies the flakiness: **11755 = 50% (3/6), 18252 = 20% (1/5)**
  fail — genuinely flaky collision scenarios — while **3436, 2509, 2513 pass 100% (0%)** even though they
  were labeled FAILURES in the single-run taxonomy: their "failure" was one-off noise that does not
  reproduce. **Single-run failure labels — the driving-leaderboard standard — are unreliable** (they
  measure stochastic noise as much as capability; the embodied illusion-of-progress). An embodied verifier
  must use a failure RATE over N runs; a candidate fix is real only if it lowers the rate beyond this noise.

## 4. Findings
1. **Fake improvement is the default**, in text and driving. Held-out / statistical verification is required.
2. **The naive reading (seen-only) is fooled**; the verifier is not. This is the field's exact gap.
3. **Inducers are unreliable** (prior-anchoring); the verifier as a *selection gate over several candidates*
   is what yields reliable improvement.
4. **A self-verifying agent can gate its own {prompt/rule/memory/tool} edits** — accepting real transfer,
   rejecting overfit/lookup-hacks and regressions — reliably per DECISION; cumulative multi-round compounding
   is not (yet) large on toy tasks (honest).
5. **Embodied self-improvement is dominated by flakiness**, raising the bar (statistical, multi-run).

## 5. Related work
Self-improving / experiential agents: Reflexion, Voyager, ExpeL, AutoGuide, Meta-Reflexion, Trace2Skill, ERL,
EvoAgentBench. Trust/eval crisis: the UC-Berkeley reward-hacking result; "Illusion of Progress" for
hallucination detection; "Beyond Outcome Leaderboards". Hallucination detection / faithfulness: Lynx, HHEM,
GPT-4-as-judge. Uncertainty / selective prediction: semantic entropy, conformal prediction, calibration.
Our contribution is orthogonal: not a better inducer, but the *verifier* that separates real from fake
self-improvement — the component the trust crisis says is missing.

## 6. Limitations (honest)
Small model (7B), synthetic + benchmark data, single node — not production, not SOTA raw detection. The
self-verifying agent is shown on hidden-convention tasks + arithmetic tool-code (controlled substrates where
a real improvement exists and transfers); on a hard real task (cross-source faithfulness) self-mods don't
transfer, so the verifier safely adopts little — correct, but not a triumphant climb. The per-DECISION
advantage is robust; the multi-round CUMULATIVE advantage is small/noisy at scale (R16). The **embodied**
loop is **not solved**: failures are flaky and prior driving work (X-MoD) shows naive correction-retraining
backfires and naive data-scaling is flat; the honest lever (retention-DAgger) is a multi-week build. The
verifier, the self-verifying-agent demonstration, and the honest negatives are the deliverable.

## 7. Conclusion
Across text and driving, self-improvement heuristics are unreliable and "improvements" are usually fake; a
held-out / statistical **verifier** reliably separates real from fake and, as a selection gate over candidate
improvements, makes an autonomous loop work. In a field whose benchmarks are reward-hackable, the verifier —
not the self-improvement trick — is the transferable contribution.
