"""
s2-replay -- W1 of the S2 pre-registration (S2_PREREG.md, sealed 2026-07-17).

Replays the S1 search (RESULTS.md R25-R41) through the automated registry + gate + prescriber:

  W1a  every recorded verdict is recomputed from RAW COUNTS by compute_verdict()/gate_rate()
       and must match the human-recorded verdict EXACTLY (12 verdicts across 10 recipes,
       including K1's local-ACCEPT/deployment-REJECT scope split and A2's panel-pass ->
       deployment-REJECT). Required: 12/12.
  W1b  at each recipe transition, the prescriber run on the history prefix must emit the axis
       the human search actually took next. Required: >=7 of 9. (Consistency check of the
       distillation, NOT generality evidence -- see S2_PREREG.md.)

Run:  python s2_replay.py        (exit 0 iff W1a exact and W1b >= 7/9)
"""
import os
import sys
import tempfile

from vsi_registry import Registry
from vsi_prescriber import prescribe

# ---------------------------------------------------------------------------- the S1 record
# (id, recipe, lines, arms, human-recorded verdict, human's NEXT move axis or None)
# Raw counts exactly as recorded in RESULTS.md; recipes carry the move that produced them.

S1 = [
    ("R25", {"name": "R25", "type": "control", "move": "control-probe"},
     {"stat_alpha": 0.05},
     {"fix": {"baseline_fails": 3, "baseline_n": 6, "fails": 5, "n": 6}},
     "REJECT", "learning-repair-narrow"),

    ("K1-local", {"name": "K1", "type": "finetune", "move": "learning-repair-narrow",
                  "data": "narrow", "epochs": 10},
     {"stat_alpha": 0.05},
     {"fix": {"baseline_fails": 7, "baseline_n": 12, "fails": 0, "n": 19},
      "retention": {"baseline_fails": 0, "fails": 0, "n": 12}},
     "ACCEPT", None),                       # stage move (escalate) checked separately

    ("K1-deploy", {"name": "K1", "type": "finetune", "move": "escalate-deployment"},
     {"deploy_alpha": 0.05, "deploy_max_confirmed": 0},
     {"deploy": {"baseline_fails": 5, "baseline_n": 76, "fails": 53, "n": 76,
                 "confirmed_regressions": 12,
                 "diagnosis": {"behavior_warp": True}}},
     "GLOBAL_REJECT", "reduce-epochs"),

    ("K2a", {"name": "K2a", "type": "finetune", "move": "reduce-epochs",
             "data": "narrow", "epochs": 3},
     {"fix_max_fails": 2, "fix_n_planned": 8, "ret_max_rate": 0.30, "ret_n_planned": 24},
     {"fix": {"fails": 0, "n": 8},
      "retention": {"fails": 12, "n": 12}},          # stopped early: 12/24 best case > 30%
     "REJECT", "broaden-retention"),

    ("K2b", {"name": "K2b", "type": "finetune", "move": "broaden-retention",
             "data": "broad", "epochs": 3},
     {"fix_max_fails": 2, "fix_n_planned": 8, "ret_max_rate": 0.30, "ret_n_planned": 24},
     {"fix": {"fails": 4, "n": 8},
      "retention": {"fails": 8, "n": 24}},
     "REJECT", "raise-fix-ratio"),

    ("A1", {"name": "A1", "type": "finetune", "move": "raise-fix-ratio",
            "data": "broad", "epochs": 3, "fix_ratio": 0.4},
     {"fix_max_fails": 2, "fix_n_planned": 8, "ret_max_rate": 0.30, "ret_n_planned": 24},
     {"fix": {"fails": 4, "n": 8},
      "retention": {"fails": 6, "n": 24}},
     "REJECT", "raise-fix-ratio"),

    ("A2-panel", {"name": "A2", "type": "finetune", "move": "raise-fix-ratio",
                  "data": "broad", "epochs": 5, "fix_ratio": 0.6},
     {"fix_max_fails": 2, "fix_n_planned": 8, "ret_max_rate": 0.30, "ret_n_planned": 24},
     {"fix": {"fails": 0, "n": 8},
      "retention": {"fails": 7, "n": 24}},
     "PANEL_PASS", None),                   # stage move (escalate) checked separately

    ("A2-deploy", {"name": "A2", "type": "finetune", "move": "escalate-deployment"},
     {"deploy_alpha": 0.05, "deploy_max_confirmed": 0},
     {"deploy": {"baseline_fails": 15, "baseline_n": 81, "fails": 53, "n": 81,
                 "confirmed_regressions": 6,
                 "diagnosis": {"within_family_overfit": True, "coverage_holes": True}}},
     "GLOBAL_REJECT", "family-fix-breadth+targeted-ret"),

    ("A3", {"name": "A3", "type": "finetune", "move": "family-fix-breadth+targeted-ret",
            "data": "broad", "epochs": 5, "fix_ratio": 0.6, "fix_bucket": "family",
            "targeted_ret": True, "dataset": "modified"},
     {"fix_max_fails": 2, "fix_n_planned": 8, "ret_max_rate": 0.30, "ret_n_planned": 24},
     {"fix": {"fails": 6, "n": 7}},                  # wedge cut the panel; fix axis already decided
     "REJECT", "restore-fix-bucket-route-only"),

    ("A4", {"name": "A4", "type": "finetune", "move": "restore-fix-bucket-route-only",
            "data": "broad", "epochs": 5, "fix_ratio": 0.6, "fix_bucket": "route",
            "targeted_ret": True, "dataset": "modified"},
     {"fix_max_fails": 2, "fix_n_planned": 8, "ret_max_rate": 0.30, "ret_n_planned": 24},
     {"fix": {"fails": 4, "n": 8},
      "retention": {"fails": 8, "n": 24}},
     "REJECT", "freeze-backbone"),

    ("A5", {"name": "A5", "type": "finetune", "move": "freeze-backbone",
            "data": "broad", "epochs": 5, "fix_ratio": 0.6, "fix_bucket": "route",
            "targeted_ret": True, "dataset": "modified", "freeze": True},
     {"fix_max_fails": 2, "fix_n_planned": 8, "ret_max_rate": 0.30, "ret_n_planned": 24},
     {"fix": {"fails": 4, "n": 8},
      "retention": {"fails": 4, "n": 13}},           # partial arm; verdict decided on fix
     "REJECT", "restore-known-good-dataset+freeze"),

    ("A6", {"name": "A6", "type": "finetune", "move": "restore-known-good-dataset+freeze",
            "data": "broad", "epochs": 5, "fix_ratio": 0.6, "fix_bucket": "route",
            "dataset": "known-good", "freeze": True},
     {"fix_max_fails": 2, "fix_n_planned": 8, "ret_max_rate": 0.30, "ret_n_planned": 24},
     {"fix": {"fails": 3, "n": 8},
      "retention": {"fails": 4, "n": 24}},
     "REJECT", "close-bounded-negative"),
]

STAGE_MOVES = {"K1-local": "escalate-deployment", "A2-panel": "escalate-deployment"}


def main() -> int:
    path = os.path.join(tempfile.gettempdir(), "s2_replay_trials.jsonl")
    if os.path.exists(path):
        os.remove(path)
    reg = Registry(path)

    print("== W1a: verdict replay (raw counts -> compute_verdict/gate_rate) ==")
    exact = 0
    for tid, recipe, lines, arms, recorded, _next in S1:
        reg.register(tid, recipe, lines)
        for arm, data in arms.items():
            reg.add_arm(tid, arm, data)
        got = reg.verdict(tid)
        ok = got["verdict"] == recorded
        exact += ok
        print("  %-10s recorded=%-13s computed=%-13s %s  %s"
              % (tid, recorded, got["verdict"], "OK " if ok else "MISMATCH", got["axes"]))
    w1a = exact == len(S1)
    print("W1a: %d/%d exact -> %s" % (exact, len(S1), "PASS" if w1a else "FAIL"))

    print("\n== W1b: prescription replay (prefix -> next axis; consistency check only) ==")
    hist = reg.history()
    hits = misses = 0
    stage_hits = 0
    close_ok = None
    for i, (tid, _r, _l, _a, _v, next_axis) in enumerate(S1):
        p = prescribe(hist[: i + 1])
        if tid in STAGE_MOVES:                        # scope escalations (bonus check, not in the 9)
            ok = p["axis"] == STAGE_MOVES[tid]
            stage_hits += ok
            print("  after %-9s [stage] want=%-28s got=%-28s %s"
                  % (tid, STAGE_MOVES[tid], p["axis"], "OK" if ok else "MISS"))
        elif next_axis == "close-bounded-negative":   # the closure decision (bonus check)
            close_ok = p["axis"] == next_axis
            print("  after %-9s [close] want=%-28s got=%-28s %s"
                  % (tid, next_axis, p["axis"], "OK" if close_ok else "MISS"))
        elif next_axis is not None:                   # one of the 9 recipe transitions
            ok = p["axis"] == next_axis
            hits += ok
            misses += (not ok)
            print("  after %-9s want=%-32s got=%-32s %s"
                  % (tid, next_axis, p["axis"], "OK" if ok else "MISS"))
    w1b = hits >= 7
    print("W1b: %d/%d recipe transitions matched (need >=7); stage escalations %d/2; closure %s"
          % (hits, hits + misses, stage_hits, "OK" if close_ok else "MISS"))
    print("W1b -> %s" % ("PASS" if w1b else "FAIL"))

    print("\n== registry integrity ==")
    try:
        reg.add_arm("A6", "fix", {"fails": 0, "n": 8})
        print("  FROZEN-trial mutation was ALLOWED -- integrity FAIL")
        frozen_ok = False
    except ValueError as e:
        print("  frozen-trial mutation refused (%s) -- OK" % e)
        frozen_ok = True

    ok = w1a and w1b and frozen_ok
    print("\nRESULT: W1a %s | W1b %s | append-only %s  =>  %s"
          % ("PASS" if w1a else "FAIL", "PASS" if w1b else "FAIL",
             "PASS" if frozen_ok else "FAIL", "W1 GREEN" if ok else "W1 RED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
