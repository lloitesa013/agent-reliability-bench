"""
Curtailment for count-threshold verification arms: stop the moment the verdict is mathematically
decided regardless of remaining runs. Zero statistical assumptions; the verdict is IDENTICAL to the
fixed-N protocol by construction (VSI-0 R45: 26% of deployment-confirm runs recovered for free;
R46: this beats a naive per-arm SPRT on both savings and verdict safety at small arm sizes).

Usage inside any evaluation loop:

    from vsi.curtail import decided

    fails = passes = 0
    for k in range(N_PLANNED):
        outcome = run_once()                    # True = failed
        fails += outcome; passes += (not outcome)
        verdict = decided(fails, fails + passes, max_fails=2, n_planned=8)
        if verdict is not None:                 # "PASS" or "FAIL" — stop, verdict is final
            break
"""
from typing import Optional, Tuple

__all__ = ["decided", "curtail_counts"]


def decided(fails: int, n_valid: int, max_fails: int, n_planned: int) -> Optional[str]:
    """Return "FAIL" or "PASS" the moment the count-threshold verdict (pass iff final
    fails <= max_fails over n_planned valid runs) is decided; None while undecided."""
    if fails > max_fails:
        return "FAIL"                              # over the line, no matter what remains
    if fails + (n_planned - n_valid) <= max_fails:
        return "PASS"                              # even if every remaining run fails
    return None


def curtail_counts(outcomes, max_fails: int, n_planned: int) -> Tuple[str, int]:
    """Replay a recorded outcome sequence (iterable of truthy=failed) under curtailment.
    Returns (verdict, runs_used). Raises if the sequence ends before the verdict is decided."""
    fails = n = 0
    for o in outcomes:
        n += 1
        fails += bool(o)
        v = decided(fails, n, max_fails, n_planned)
        if v is not None:
            return v, n
    raise ValueError("sequence exhausted while verdict undecided (fails=%d, n=%d)" % (fails, n))
