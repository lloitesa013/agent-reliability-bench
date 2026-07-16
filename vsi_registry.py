"""
vsi-registry -- the S2 Trial Registry (pure stdlib, append-only).

S1's outer loop was run by a human: propose recipe -> run panel -> gate verdict -> next recipe.
S2 turns the RECORD-KEEPING and the VERDICT into code. This module is the registry: an append-only
JSONL ledger of trials, and a pure verdict computation that applies the pre-registered lines
(S2_PREREG.md) with vsi_gate as the only statistical authority.

Invariants (enforced):
  - append-only: a trial with a computed verdict is FROZEN -- further arms/verdicts raise.
  - no post-hoc lines: the pre-registered thresholds are stored at registration time, before data.
  - the verdict is COMPUTED from raw counts, never stored by fiat.

Trial anatomy (matches the S1 panel design, RESULTS.md R31-R41):
  lines = {
    "fix_max_fails": 2, "fix_n_planned": 8,          # fix axis: <=2/8 failures on the target route
    "ret_max_rate": 0.30, "ret_n_planned": 24,       # retention axis: pooled <=30% on the reg-12 set
  }                                                   # (either axis optional)
  or  lines = {"stat_alpha": 0.05}                    # statistical accept trial (gate_rate)
  or  lines = {"deploy_alpha": 0.05, "deploy_max_confirmed": 0}   # deployment gate

Early stopping is honest: an axis is DECIDED as soon as its outcome is mathematically fixed
(R34: K2a's retention arm was stopped at 12/12 because even 12/24 = 50% > 30%).
"""
import json
import os
from typing import Dict, List, Optional

from vsi_gate import gate_rate, _binom_tail_le

__all__ = ["Registry", "compute_verdict"]


# ------------------------------------------------------------------ axis decisions (pure)

def _axis_fix(fails: int, n: int, max_fails: int, n_planned: int) -> Optional[str]:
    """PASS/FAIL/None(undecided) for a count-threshold axis."""
    if fails > max_fails:
        return "FAIL"                      # already over the line, no matter what remains
    if n >= n_planned:
        return "PASS"
    return None

def _axis_rate(fails: int, n: int, max_rate: float, n_planned: int) -> Optional[str]:
    """PASS/FAIL/None for a pooled-rate axis; FAIL as soon as even a perfect remainder can't recover."""
    if n_planned <= 0:
        return None
    if fails / n_planned > max_rate:       # best case: all remaining runs pass
        return "FAIL"
    if n >= n_planned:
        return "PASS" if (fails / n) <= max_rate else "FAIL"
    return None


def compute_verdict(trial: dict) -> dict:
    """Apply the trial's pre-registered lines to its raw arms. Returns {"verdict": ..., "axes": {...}}.

    Verdict vocabulary (matches the S1 record):
      ACCEPT        -- statistical trial passed hidden test AND regression arm clean
      PANEL_PASS    -- both panel axes passed (candidate may proceed to the deployment gate)
      REJECT        -- any decided axis failed / no improvement evidence
      GLOBAL_REJECT -- deployment gate: confirmed regression(s) or pooled significantly worse
      UNDECIDED     -- arms incomplete and nothing mathematically decided yet
    """
    lines, arms = trial["lines"], trial.get("arms", {})
    axes: Dict[str, str] = {}

    # --- statistical accept trial (hidden test = failure rate significantly lower) ---
    if "stat_alpha" in lines:
        f, r = arms.get("fix"), None
        if f and f.get("baseline_n"):
            r = gate_rate(f["baseline_fails"], f["baseline_n"], f["fails"], f["n"],
                          alpha=lines["stat_alpha"])
            axes["hidden"] = "PASS" if r["accept"] else "FAIL"
            axes["hidden_p"] = "%.3g" % r["p_value"]
        ret = arms.get("retention")
        if ret is not None:
            base = ret.get("baseline_fails", 0)
            axes["regression"] = "PASS" if ret["fails"] <= base else "FAIL"
        if not axes:
            return {"verdict": "UNDECIDED", "axes": axes}
        bad = [k for k, v in axes.items() if v == "FAIL"]
        return {"verdict": "REJECT" if bad else "ACCEPT", "axes": axes}

    # --- deployment gate (screen + interleaved confirm, R33/R37 rules) ---
    if "deploy_alpha" in lines:
        d = arms.get("deploy")
        if d is None:
            return {"verdict": "UNDECIDED", "axes": axes}
        confirmed = d.get("confirmed_regressions", 0)
        axes["confirmed_regressions"] = str(confirmed)
        # pooled: is the CANDIDATE's failure rate significantly ABOVE baseline?
        p0 = d["baseline_fails"] / d["baseline_n"]
        p_worse = max(0.0, 1.0 - _binom_tail_le(d["fails"] - 1, d["n"], p0)) if d["fails"] > 0 else 1.0
        axes["pooled_worse_p"] = "%.3g" % p_worse
        worse = (d["fails"] / d["n"] > p0) and (p_worse < lines["deploy_alpha"])
        if confirmed > lines.get("deploy_max_confirmed", 0) or worse:
            return {"verdict": "GLOBAL_REJECT", "axes": axes}
        return {"verdict": "PANEL_PASS", "axes": axes}   # survived deployment scope

    # --- local panel (fix count line + retention pooled-rate line) ---
    if "fix_max_fails" in lines:
        f = arms.get("fix")
        axes["fix"] = (_axis_fix(f["fails"], f["n"], lines["fix_max_fails"], lines["fix_n_planned"])
                       if f else None) or ("UNDECIDED" if f else "MISSING")
    if "ret_max_rate" in lines:
        ret = arms.get("retention")
        axes["retention"] = (_axis_rate(ret["fails"], ret["n"], lines["ret_max_rate"],
                                        lines["ret_n_planned"])
                             if ret else None) or ("UNDECIDED" if ret else "MISSING")
    decided_fail = any(v == "FAIL" for v in axes.values())
    all_pass = axes and all(v == "PASS" for v in axes.values())
    if decided_fail:
        return {"verdict": "REJECT", "axes": axes}
    if all_pass:
        return {"verdict": "PANEL_PASS", "axes": axes}
    return {"verdict": "UNDECIDED", "axes": axes}


# ------------------------------------------------------------------ the append-only ledger

class Registry:
    """Append-only trial ledger over a JSONL file. Every state change is one appended event."""

    def __init__(self, path: str):
        self.path = path
        self.trials: Dict[str, dict] = {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        self._apply(json.loads(line))

    # -- event application (also used on load) --
    def _apply(self, ev: dict) -> None:
        kind, tid = ev["event"], ev["trial"]
        if kind == "register":
            if tid in self.trials:
                raise ValueError("trial %s already registered" % tid)
            self.trials[tid] = {"id": tid, "recipe": ev["recipe"], "lines": ev["lines"],
                                "arms": {}, "verdict": None, "note": ev.get("note", "")}
        elif kind == "arm":
            t = self._open_trial(tid)
            t["arms"][ev["arm"]] = ev["data"]
        elif kind == "verdict":
            t = self._open_trial(tid)
            t["verdict"] = ev["result"]
        else:
            raise ValueError("unknown event kind %r" % kind)

    def _open_trial(self, tid: str) -> dict:
        t = self.trials.get(tid)
        if t is None:
            raise KeyError("trial %s not registered" % tid)
        if t["verdict"] is not None:
            raise ValueError("trial %s is FROZEN (verdict recorded) -- registry is append-only" % tid)
        return t

    def _append(self, ev: dict) -> None:
        self._apply(ev)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(ev, ensure_ascii=False) + "\n")

    # -- public API --
    def register(self, tid: str, recipe: dict, lines: dict, note: str = "") -> str:
        """Register a trial: recipe + PRE-REGISTERED lines, before any data."""
        self._append({"event": "register", "trial": tid, "recipe": recipe,
                      "lines": lines, "note": note})
        return tid

    def add_arm(self, tid: str, arm: str, data: dict) -> None:
        """Record raw counts for an arm (fix / retention / deploy). Raw counts only -- no verdicts."""
        self._append({"event": "arm", "trial": tid, "arm": arm, "data": data})

    def verdict(self, tid: str) -> dict:
        """Compute the verdict from raw arms + pre-registered lines, record it, freeze the trial."""
        t = self._open_trial(tid)
        result = compute_verdict(t)
        if result["verdict"] == "UNDECIDED":
            return result                     # not frozen -- more runs may be added
        self._append({"event": "verdict", "trial": tid, "result": result})
        return result

    def history(self) -> List[dict]:
        """Trials in registration order (insertion order is file order)."""
        return list(self.trials.values())
