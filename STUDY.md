# Verified Self-Improvement — an honest study

**Thesis (one line).** Self-improving agents need a *verifier* that confirms an improvement is REAL
(it transfers to held-out cases) and not FAKE (overfit to what it learned from, or a flaky/stochastic
non-reproduction). The field cannot currently do this — in April 2026 UC Berkeley showed all 8 major
agent benchmarks are reward-hackable to ~100%. **The verifier is the missing, valuable piece; and fake
improvement is the norm.** We prove both, in text and in an embodied driving domain, on one RTX 5090.

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
  detection. The autonomous loop is closed in TEXT on simple hidden-rule tasks (R10); it needs a
  prior-overcoming inducer and the model's candidates are unreliable (the verifier does the real work).
  The **EMBODIED/CARLA** version remains the open hard core: failures are flaky (need statistical
  verification), and prior X-MoD work shows naive correction-retraining backfires + naive data-scaling
  is flat — the honest lever is retention-DAgger, a multi-week build. So "verified self-improvement on
  driving" is not solved; the verifier + the honest negatives are the deliverable.
- **Bottom line:** the verifier — the thing that separates real from fake self-improvement — is proven,
  robust, and transfers; that is the deliverable. The full autonomous self-improvement loop is not solved
  (transfer is the frontier), and the honest negatives here *are* the evidence for why the verifier matters.

## Repo map
`bench/` sealed benchmark · `RESULTS.md` authoritative results (R1–8) · `loop*.py` self-improvement loops ·
`proof_loop.py`/`proof_robust.py` the verifier proof · `faith_loop.py`/`cross_source*.py` transfer tests ·
`calibrate*.py`/`self_consistency.py` abstention · `carla/` the embodied watcher/judge/verify + flakiness.
