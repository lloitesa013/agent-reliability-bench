"""
Example: gate a change against a FLAKY test/benchmark with a rate gate + curtailment.

Scenario: your agent (or you) changed something; the affected test fails ~58% of the time on the
old code. One green run proves nothing (42% luck). Verify by failure RATE, and stop early the
moment the verdict is mathematically decided.

Run:  python examples/gate_flaky_ci.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo-root fallback

import random

from vsi import gate_rate, decided

random.seed(7)

BASELINE_FAIL_RATE = 7 / 12          # measured on the old code (7 fails in 12 runs)
TRUE_NEW_RATE = 0.05                 # unknown to you; what the sim below draws from


def run_ci_once() -> bool:
    """Your test command here. Returns True if the run FAILED."""
    return random.random() < TRUE_NEW_RATE


# --- curtailed evaluation arm: plan 19 runs, stop the moment a <=4/19 line is decided ---
MAX_FAILS, N_PLANNED = 4, 19
fails = n = 0
while n < N_PLANNED:
    fails += run_ci_once()
    n += 1
    v = decided(fails, n, MAX_FAILS, N_PLANNED)
    if v is not None:
        print("curtailed: line <=%d/%d decided %s after %d runs" % (MAX_FAILS, N_PLANNED, v, n))
        break

# --- statistical verdict against the measured baseline rate ---
verdict = gate_rate(baseline_fail=7, baseline_n=12, cand_fail=fails, cand_n=n)
print("rate gate: baseline %.0f%% vs candidate %d/%d -> %s"
      % (100 * BASELINE_FAIL_RATE, fails, n, verdict["verdict"]))
print()
print("why this matters: with a 58%-flaky baseline, a single lucky green run would 'confirm' any")
print("change 42% of the time. The rate gate needs ~19 runs for a 58%->20% claim at 80% power --")
print("and curtailment gives many of those runs back whenever the verdict comes early.")
