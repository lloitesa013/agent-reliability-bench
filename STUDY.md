# Verified Self-Improvement — an honest study

**Thesis (one line).** Self-improving agents need a *verifier* that confirms an improvement is REAL
(it transfers to held-out cases) and not FAKE (overfit to what it learned from, or a flaky/stochastic
non-reproduction). The field cannot currently do this — in April 2026 UC Berkeley showed all 8 major
agent benchmarks are reward-hackable to ~100%. **The verifier is the missing, valuable piece; and fake
improvement is the norm.** We prove both, in text and in an embodied driving domain, on one RTX 5090.

**The artifact this builds toward: a self-verifying agent** — an agent that treats its OWN
{prompt / rule / memory / tool} self-modifications as UNTRUSTED and adopts one only if it clears a
HIDDEN TEST (improves on held-out, not memorized) AND a REGRESSION check (does not break an existing
capability). This is the safe self-improvement loop: it lets an agent improve without self-deception.
R11–R15 below demonstrate it end-to-end across rule, memory, and tool self-modifications.

## What is proven (both directions of the verifier)
- **REJECT (fake is caught).** Prompt-rule self-improvement (`loop_v2.py`, the ExpeL/AutoGuide family)
  and cross-source fine-tuning (`cross_source.py`, `faith_loop.py`) do **not** transfer. Candidate
  "improvements" that raise the *seen* score fail on held-out; the verifier rejects them robustly and
  is not fooled by seen-gains. Shown on synthetic and on public HaluBench (a real-gap task).
- **ACCEPT (real is recognized).** `proof_loop.py` / `proof_robust.py`: on 4 hidden-convention tasks,
  the verifier ACCEPTS a general rule that transfers to held-out values (held-out +1.00) and REJECTS an
  overfit rule that the naive "measure on what you learned from" baseline wrongly accepts. **Verifier
  correct 4/4; naive-seen fooled by the overfit where it creates a seen-gain** = the reward-hacking /
  illusion-of-progress gap, caught by held-out verification.
- **AUTONOMOUS loop closed.** `proof_autonomous2.py`: the model induces candidate rules from its OWN
  failures and the verifier adopts only the one that transfers (2/2 tasks). Key nuance (`proof_autonomous.py`):
  a *naive* inducer is **prior-anchored** — it rationalizes its failures with its prior (kept "8h"/"dozen=12")
  and induces wrong rules; a *pattern-fitting* inducer (ignore priors, fit output=f(input)) produces
  candidates that include the true rule, and the **verifier selects it**. So the model's self-improvement
  candidates are unreliable; the verifier is the essential selection mechanism that makes the loop work.

## The self-verifying agent — {prompt / rule / memory / tool} (R11–R15)
The verifier, turned on the agent's OWN self-modifications. Each result is `RESULTS.md`.
- **Fine-tuning level (R11, `candidate_ft.py`).** Among three candidate LoRA fine-tunes, the one that
  memorized its single training source (in-dist 0.993, held-out 0.507) is what NAIVE deploys; the
  verifier (held-out) deploys the broad fine-tune (0.578) = the reward-hacking/memorization signature
  caught at the fine-tuning level. Verified ≠ naive, verified generalizes better.
- **Regression verifier (R12, `verified_integration.py`).** Naively fine-tuning to fix one source
  silently REGRESSES another by **−0.16 (16 pts)**; a 1D "did the target improve?" check misses it. The
  2D verifier (gain AND no-regression) catches it; a retention-replay candidate regresses nothing (even
  +0.12 elsewhere). The X-MoD "naive correction backfires" warning, reproduced and gated.
- **FLAGSHIP single-round (R13, `self_verify_agent.py`).** Agent handles A (1.00), fails B (0.00),
  proposes rule-edits to fix B. It ADOPTS only the edit that transfers to held-out B AND keeps A —
  even REJECTING the edit that fixes B best (hidden 1.00) because it regresses A. Capability 0.50→0.92
  vs naive 0.67. Safe self-modification, demonstrated.
- **Multi-round (R14, `self_improve_loop.py`).** Acquiring B,C,D one per round: self-verifying higher
  every round (0.88 vs 0.79), naive adopts a −0.50-regressing edit round 1. **HONEST caveat (R16):** at
  5 rounds this gap is NOT robust — cumulative capability at scale is dominated by new-task-difficulty
  noise + dilution (final mean Δ only +0.03). The robust claim is the SINGLE-DECISION advantage (R11–13,
  R15); large multi-round CUMULATIVE compounding is not demonstrated on these toy tasks.
- **Tool/code self-mods (R15, `self_verify_tool.py`).** The agent rewrites its own Python tool; **2 of 3
  rewrites are lookup-hacks** (perfect on seen inputs, executed-held-out 0.0). Seen score can't tell them
  from the general tool (naive gambles, expected 0.67); EXECUTING each on held-out is decisive → verifier
  adopts the general tool (1.00). The self-coding-agent failure mode, caught by an executing hidden test.

## Embodied confirmation (CARLA driving)
- Reused the operator's own assets (CARLA+LEAD, Bench2Drive-220, X-MoD, failure taxonomy).
- **WATCHER** (`run_taxonomy.ps1`): 201 route failures observed and clustered.
- **JUDGE** (`get_scenarios.ps1`): 7 hard failures attributed to scenarios (EnterActorFlow→collision,
  pedestrian-crossing→collision, …).
- **VERIFY pipeline** works (re-run routes) — and revealed the key embodied fact: **driving failures are
  FLAKY.** The same route+agent, zero changes, flips FAILED↔PASSED across runs (route 11755 collision
  in the batch → PASSED on re-run; 3436/18252 similarly). So single-run "the fix worked" is meaningless;
  verification must be a **statistical failure-RATE** — the embodied analog of the fake-improvement problem.

## The reliability foundation it stands on (the watcher itself)
- A zero-shot *reading* judge beats a tuned groundedness rule on a sealed benchmark (**0.949 vs 0.789**,
  unsafe_pass 0.000); an embedding watcher ties it — reading the trace is load-bearing (`RESULTS.md` R1–2).
- **Calibrated abstention** (the differentiator): the judge's logit margin is a real uncertainty signal —
  abstain the least-confident 30% → accuracy 0.84→0.91; and it **transfers to 4/6 unseen domains** (R3, R7).
- Public field level: 7B zero-shot HaluBench 0.688 (GPT-3.5-tier); fine-tuned 0.815 in-distribution
  (near Lynx-8B) but does **not** generalize cross-source — honest, and it motivated the whole thesis (R4–6).

## Honest scope & limits
- Small models (7B), synthetic + public benchmark data, single node — not production, not SOTA raw
  detection. The self-verifying agent (R11–R15) is demonstrated on **hidden-convention tasks + arithmetic
  tool code** — controlled substrates where a real improvement exists and transfers. On a HARD real task
  (cross-source faithfulness) prompt-rule/fine-tune self-mods do NOT transfer, so the verifier safely
  adopts little (R6, R11–12) — honest, and the correct behavior (refuse fake improvement), but not a
  triumphant climb. The **EMBODIED/CARLA** version remains the open hard core: failures are flaky (need
  statistical verification), and prior X-MoD work shows naive correction-retraining backfires + naive
  data-scaling is flat — the honest lever is retention-DAgger, a multi-week build.
- **Bottom line:** the self-verifying agent — an agent that gates its own {prompt/rule/memory/tool}
  changes through a hidden-test + regression verifier — reliably wins the **single adoption decision**:
  it accepts real transfer and rejects the overfit fine-tune (R11), the −0.16 regression (R12), the
  best-B-fix that breaks A (R13), and the 2/3 lookup-hacks (R15) that all fool the naive agent.
  **Quantified (R17): over 24 tool-acquisition decisions a reward-hack existed in 100%, naive deploys a
  broken tool 59% of the time, the executing verifier 0% (while still improving 88%).** That per-decision
  advantage is the robust, load-bearing result. The **multi-round CUMULATIVE** advantage is
  small and noisy at scale (R16, honest) — not a large-compounding claim. The honest negatives (hard-task
  non-transfer, multi-round dilution, CARLA flakiness) *are* the evidence for what the verifier does and
  does not buy. Next frontier: a harder/real substrate and the embodied loop.

## Repo map
`bench/` sealed benchmark · `RESULTS.md` authoritative results (R1–8) · `loop*.py` self-improvement loops ·
`proof_loop.py`/`proof_robust.py` the verifier proof · `faith_loop.py`/`cross_source*.py` transfer tests ·
`calibrate*.py`/`self_consistency.py` abstention · `carla/` the embodied watcher/judge/verify + flakiness.
