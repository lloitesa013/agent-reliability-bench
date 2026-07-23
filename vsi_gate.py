"""
vsi-gate -- the VSI-0 verifier as a reusable library (pure stdlib, no dependencies).

An agent (or a human pipeline) proposes candidate self-modifications -- a new prompt, a rule, a memory
entry, a rewritten tool, a fine-tuned checkpoint. THIS MODULE IS THE GATE: it treats every candidate as
untrusted and ACCEPTs one only if it

  (1) HIDDEN TEST  -- improves the target capability on held-out data the candidate was not derived from,
  (2) REGRESSION   -- does not degrade any existing capability beyond a noise margin.

If no candidate clears both gates, the gate ABSTAINS (adopt nothing) -- which is the safe outcome.
Backed by VSI-0 results R11–R25 (github.com/lloitesa013/agent-reliability-bench): selecting by the score
a candidate was tuned on ships fakes (59% on tool rewrites, 19/29 overfits on MBPP, 12% on solvable
problems); this gate shipped 0% across all of them.

Quick start (scores are floats where higher is better; you supply the evaluators):

    from vsi_gate import gate

    report = gate(
        candidates=[cand_a, cand_b, cand_c],
        hidden_score=lambda c: eval_on_heldout(c),      # held-out target score
        baseline_hidden=eval_on_heldout(current),
        regression_profile=lambda c: {"taskA": eval_a(c), "taskB": eval_b(c)},
        baseline_profile={"taskA": eval_a(current), "taskB": eval_b(current)},
        gain_margin=0.05, reg_margin=0.03,
    )
    if report.accepted is not None:
        deploy(report.accepted.candidate)               # else keep the current version

For FLAKY / stochastic domains (embodied agents, CI, driving) use the statistical form -- a single run is
noise; verify by failure RATE:

    from vsi_gate import gate_rate
    verdict = gate_rate(baseline_fail=7, baseline_n=12, cand_fail=2, cand_n=20, alpha=0.05)
    # ACCEPT only if the candidate's failure rate is significantly LOWER (one-sided) than baseline.

Demo (no LLM, deterministic):  python vsi_gate.py --demo
"""
from dataclasses import dataclass, field
from math import comb
from typing import Any, Callable, Dict, List, Optional

__all__ = ["gate", "gate_rate", "GateReport", "CandidateVerdict"]


# ----------------------------------------------------------------------------- deterministic gate

@dataclass
class CandidateVerdict:
    candidate: Any
    index: int
    hidden: float                       # held-out target score with this candidate applied
    hidden_gain: float                  # hidden - baseline_hidden
    worst_regression: float             # min over capabilities of (cand - baseline); 0.0 if no profile
    regressions: Dict[str, float] = field(default_factory=dict)
    accepted: bool = False
    reason: str = ""


@dataclass
class GateReport:
    verdicts: List[CandidateVerdict]
    accepted: Optional[CandidateVerdict]    # the selected candidate, or None = ABSTAIN (keep current)

    def summary(self) -> str:
        lines = []
        for v in self.verdicts:
            lines.append("cand#%d  hidden=%.3f (gain %+.3f)  worst_reg=%+.3f  -> %s%s"
                         % (v.index, v.hidden, v.hidden_gain, v.worst_regression,
                            "ACCEPT" if v.accepted else "reject",
                            (" [%s]" % v.reason) if v.reason else ""))
        lines.append("DECISION: " + ("adopt cand#%d" % self.accepted.index if self.accepted
                                     else "ABSTAIN -- no candidate cleared both gates (keep current)"))
        return "\n".join(lines)


def gate(candidates: List[Any],
         hidden_score: Callable[[Any], float],
         baseline_hidden: float,
         regression_profile: Optional[Callable[[Any], Dict[str, float]]] = None,
         baseline_profile: Optional[Dict[str, float]] = None,
         gain_margin: float = 0.05,
         reg_margin: float = 0.03) -> GateReport:
    """Score every candidate on the hidden test and the regression profile; ACCEPT the best-hidden
    candidate among those clearing BOTH gates, else abstain.

    hidden_score MUST evaluate on data the candidates were not derived/tuned on -- that is the whole point.
    Reusing a small selection set across many gate() calls re-creates the overfitting this prevents
    (VSI-0 R18: +0.045 optimism from a 20-item reused set); use fresh or large held-out data.
    """
    if (regression_profile is None) != (baseline_profile is None):
        raise ValueError("provide both regression_profile and baseline_profile, or neither")
    verdicts: List[CandidateVerdict] = []
    for i, c in enumerate(candidates, 1):
        hid = float(hidden_score(c))
        gain = hid - baseline_hidden
        regs: Dict[str, float] = {}
        worst = 0.0
        if regression_profile is not None:
            prof = regression_profile(c)
            regs = {k: float(prof[k]) - float(baseline_profile[k]) for k in baseline_profile}
            worst = min(regs.values()) if regs else 0.0
        v = CandidateVerdict(candidate=c, index=i, hidden=hid, hidden_gain=gain,
                             worst_regression=worst, regressions=regs)
        if gain < gain_margin:
            v.reason = "no held-out gain (%.3f < %.3f)" % (gain, gain_margin)
        elif worst < -reg_margin:
            k = min(regs, key=regs.get)
            v.reason = "regresses %s by %.3f" % (k, -regs[k])
        else:
            v.accepted = True
        verdicts.append(v)
    passing = [v for v in verdicts if v.accepted]
    best = max(passing, key=lambda v: v.hidden) if passing else None
    for v in verdicts:                       # only the selected one is the adoption
        v.accepted = (best is not None and v.index == best.index)
    return GateReport(verdicts=verdicts, accepted=best)


# ----------------------------------------------------------------------------- statistical gate (flaky)

def _binom_tail_le(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p). Exact, stdlib-only."""
    return sum(comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(0, k + 1))


def gate_rate(baseline_fail: int, baseline_n: int,
              cand_fail: int, cand_n: int,
              alpha: float = 0.05) -> dict:
    """ACCEPT a candidate fix in a stochastic domain only if its observed failure rate is significantly
    LOWER than the baseline rate (one-sided exact binomial test at the baseline point estimate).

    VSI-0 R19–R25 background: driving failures flip PASS/FAIL run-to-run (route 11755 = 58% flaky), so a
    single run is meaningless, and a plausible fix can make things WORSE (R25: 50%->83%) -- always verify
    by rate. Power guidance (R23): detecting 58%->20% needs ~19 runs/arm; 58%->40% needs ~91.
    """
    if baseline_n <= 0 or cand_n <= 0:
        raise ValueError("need runs in both arms")
    p0 = baseline_fail / baseline_n
    p1 = cand_fail / cand_n
    pval = _binom_tail_le(cand_fail, cand_n, p0)     # P(this few failures | true rate = baseline)
    accept = (p1 < p0) and (pval < alpha)
    return {
        "baseline_rate": p0, "candidate_rate": p1, "p_value": pval, "alpha": alpha,
        "accept": accept,
        "verdict": "ACCEPT -- failure rate significantly lower" if accept else
                   ("REJECT -- rate not lower" if p1 >= p0 else
                    "REJECT -- direction ok but not significant (p=%.3f); add runs (see R23 power table)" % pval),
    }


# ----------------------------------------------------------------------------- demo (no LLM needed)

def _demo() -> int:
    """Deterministic tool-rewrite demo: three candidate implementations of items_in_dozen(n) where the
    true convention is n*10. #2 is a hardcoded lookup of the seen inputs (the reward-hack every naive
    selector ships). Visible test = what the agent saw; hidden = held-out inputs."""
    seen = {3: 30, 7: 70}
    hidden = {5: 50, 11: 110, 20: 200}
    candidates = {
        1: ("general  n*10", lambda n: n * 10),
        2: ("lookup   {3:30,7:70}", lambda n: {3: 30, 7: 70}.get(n)),
        3: ("wrong    n*12", lambda n: n * 12),
    }

    def acc(fn, cases):
        return sum(1.0 for k, v in cases.items() if fn(k) == v) / len(cases)

    print("candidate tool rewrites (visible = seen inputs, hidden = held-out):")
    for i, (name, fn) in candidates.items():
        print("  #%d %-22s visible=%.2f  hidden=%.2f" % (i, name, acc(fn, seen), acc(fn, hidden)))
    best_vis = max(acc(fn, seen) for _, fn in candidates.values())
    tied = [i for i, (_, fn) in candidates.items() if acc(fn, seen) == best_vis]
    exp_hidden = sum(acc(candidates[i][1], hidden) for i in tied) / len(tied)
    print("\nNAIVE (best visible): #%s tie at visible=%.2f -- the seen score CANNOT tell the general tool"
          % (" and #".join(map(str, tied)), best_vis))
    print("from the lookup-hack; naive gambles (expected hidden = %.2f)." % exp_hidden)

    report = gate(
        candidates=[fn for _, fn in candidates.values()],
        hidden_score=lambda fn: acc(fn, hidden),
        baseline_hidden=0.0,
        gain_margin=0.5,
    )
    print("\nvsi-gate:")
    print(report.summary())

    print("\nstatistical gate (flaky domain, from VSI-0 R25 real numbers):")
    r = gate_rate(baseline_fail=3, baseline_n=6, cand_fail=5, cand_n=6)
    print("  conservative-throttle 'fix' 50%%->83%%: %s" % r["verdict"])
    r2 = gate_rate(baseline_fail=7, baseline_n=12, cand_fail=1, cand_n=19)
    print("  hypothetical real fix 58%%->5%% over 19 runs: %s (p=%.4f)" % (r2["verdict"], r2["p_value"]))
    return 0


if __name__ == "__main__":
    import sys as _sys
    if "--demo" in _sys.argv:
        raise SystemExit(_demo())
    print(__doc__)
