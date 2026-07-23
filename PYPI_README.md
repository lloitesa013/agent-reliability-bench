# vsi-gate

**The verifier for self-improving agents.** An agent that rewrites its own prompts, rules, memories,
tools, or weights must not trust its own changes: across four modification surfaces we measured,
fake improvement is the *norm* — 100% of tool-rewrite decision sets contained a reward-hack, 29–30%
of real coding problems yielded a genuine overfit, and selecting by the score a change was derived
from shipped a broken change 59–88% of the time. This gate shipped **0%** on every benchmark while
still adopting the real improvements, and on a real embodied driving system it both **accepted** a
genuine repair (p = 6e-8) and **rejected the same candidate at deployment scope** (p = 8.5e-45)
before it could ship catastrophic forgetting.

Pure stdlib. No dependencies. Every number above reproduces from the
[research repository](https://github.com/lloitesa013/agent-reliability-bench) (`REPRO.md`).

```
pip install vsi-gate
python -m vsi        # deterministic no-LLM demo
```

## The gate (deterministic surfaces)

```python
from vsi import gate

report = gate(
    candidates=[edit_a, edit_b, edit_c],          # your agent's proposed self-modifications
    hidden_score=lambda c: eval_on_heldout(c),    # data the candidate was NOT derived from
    baseline_hidden=eval_on_heldout(current),
    regression_profile=lambda c: {"taskA": eval_a(c), "taskB": eval_b(c)},
    baseline_profile={"taskA": eval_a(current), "taskB": eval_b(current)},
)
if report.accepted:                                # else: abstain — the safe default
    deploy(report.accepted.candidate)
```

The regression check is not decoration: the best target-fix can be the worst adoption (a candidate
that fixed its target perfectly caused 12 confirmed regressions elsewhere).

## The rate gate (flaky / stochastic surfaces)

A single run is noise — the same route, agent, and configuration flipped PASS/FAIL at 58% in our
measurements. Verify failure *rates*:

```python
from vsi import gate_rate
verdict = gate_rate(baseline_fail=7, baseline_n=12, cand_fail=0, cand_n=19)
# ACCEPT only if the candidate's rate is significantly lower (one-sided exact binomial)
```

## Curtailment (free cost reduction, verdict-identical)

Stop an evaluation arm the moment its verdict is mathematically decided. Recovered 26% of our
deployment-confirm cost with zero assumptions — and beat a naive SPRT on both savings and safety:

```python
from vsi import decided
fails = n = 0
for _ in range(8):
    fails += run_once(); n += 1
    if decided(fails, n, max_fails=2, n_planned=8):   # "PASS" / "FAIL" -> stop
        break
```

## The registry (audit-proof bookkeeping)

Append-only trial ledger: acceptance lines are stored **before** data, verdicts are **computed**
(never declared), and finished trials are frozen. This is what lets a third party replay every
decision your loop ever made:

```python
from vsi import Registry
reg = Registry("trials.jsonl")
reg.register("fix-1", recipe={"lr": 3e-4}, lines={"fix_max_fails": 2, "fix_n_planned": 8})
reg.add_arm("fix-1", "fix", {"fails": 1, "n": 8})
reg.verdict("fix-1")     # computed from raw counts; the trial is now frozen
```

## One rule learned the hard way

A verification-infrastructure crash once produced score-0 records that our failure counter read as
three genuine failures — nearly rejecting a good candidate. *A trustworthy verdict starts at the
definition of a valid run.* Guard your run validity before you count anything.

MIT. Research: "The Verifier Is the Loop" (VSI-0), single RTX 5090.
