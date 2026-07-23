"""
Example: gate an LLM agent's PROMPT-RULE self-modification (no LLM needed to run this file --
evaluators are stubbed so you can see the mechanics in 30 seconds, then swap in your own).

Scenario: your agent failed some task and proposed three rule edits to its own system prompt.
One is a genuine generalization, one is an overfit to the exact failures it saw, one regresses an
existing capability. Naive selection by training-set score cannot tell them apart.

Run:  python examples/gate_prompt_change.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo-root fallback

from vsi import gate

# --- your agent's proposed self-modifications (normally: generated from its own failures) ---
CANDIDATES = ["rule: always convert units to SI before comparing",       # genuine general fix
              "rule: if the question mentions 'km', answer 42",          # overfit to seen cases
              "rule: answer numeric questions with numbers only"]        # fixes target, breaks B

# --- your evaluators (swap these stubs for real ones; held-out = data the edit was NOT derived from) ---
SEEN = {0: 1.00, 1: 1.00, 2: 1.00}          # all three ace the failures they were derived from!
HELDOUT = {0: 0.92, 1: 0.15, 2: 0.90}       # ...but only #0 and #2 transfer
PROFILE = {                                  # existing capabilities (regression check)
    0: {"taskA": 0.95, "taskB": 0.90},
    1: {"taskA": 0.95, "taskB": 0.90},
    2: {"taskA": 0.95, "taskB": 0.40},       # #2 regresses taskB by -0.50
}
BASELINE_HELDOUT = 0.55
BASELINE_PROFILE = {"taskA": 0.95, "taskB": 0.90}

report = gate(
    candidates=list(range(3)),
    hidden_score=lambda i: HELDOUT[i],
    baseline_hidden=BASELINE_HELDOUT,
    regression_profile=lambda i: PROFILE[i],
    baseline_profile=BASELINE_PROFILE,
)
print(report.summary())
print()
print("naive (seen-score) selection: all three tie at 1.00 -- it must gamble.")
print("the gate: adopts #%s -- the overfit is rejected on held-out, and the edit that fixes the"
      % (report.accepted.index if report.accepted else "NONE"))
print("target BEST (#3, hidden 0.90) is rejected anyway because it breaks taskB. The best")
print("target-fix can be the worst adoption; the regression check is not decoration.")
