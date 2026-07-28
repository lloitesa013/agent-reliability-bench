"""
E1 L0 aggregation (stdlib only). Reads e1/data/e1_l0_runs.csv (36 valid runs) and reports, per
route: failure rate + exact Clopper-Pearson 95% CI + CI width (the quantity H1 is measured
against), the OUTCOME MULTISET (distinct outcome values and their counts), and outcome entropy.

Failure definition is the project's standing one, unchanged from R19-R21 and used by every gate
in this repo: failed iff score_composed < 100 OR any collision.

Run:  py -3.10 e1/aggregate_l0.py
"""
import csv
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from vsi_gate import _binom_tail_le  # exact binomial tail, stdlib

CSV = os.path.join(HERE, "data", "e1_l0_runs.csv")
ROUTES = ["11755", "18252", "3436"]
LABEL = {"11755": "EnterActorFlow (high-variance)",
         "18252": "pedestrian crossing (mid-variance)",
         "3436": "HazardSideLane (intended zero-variance control)"}


def cp_interval(k, n, alpha=0.05):
    """Exact Clopper-Pearson 95% CI via bisection on the exact binomial tail."""
    def lo():
        if k == 0:
            return 0.0
        a, b = 0.0, 1.0
        for _ in range(200):                     # P(X >= k | p) = alpha/2
            m = (a + b) / 2
            if 1.0 - _binom_tail_le(k - 1, n, m) < alpha / 2:
                a = m
            else:
                b = m
        return (a + b) / 2

    def hi():
        if k == n:
            return 1.0
        a, b = 0.0, 1.0
        for _ in range(200):                     # P(X <= k | p) = alpha/2
            m = (a + b) / 2
            if _binom_tail_le(k, n, m) > alpha / 2:
                a = m
            else:
                b = m
        return (a + b) / 2

    return lo(), hi()


def main():
    rows = []
    with open(CSV, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            col = (float(r["col_vehicle"]) + float(r["col_pedestrian"]) + float(r["col_layout"]))
            rows.append({
                "route": r["route"], "run": int(r["run"]), "status": r["status"],
                "score": round(float(r["score_composed"]), 3), "col": round(col, 3),
                "blocked": float(r["agent_blocked"] or 0), "dev": float(r["route_dev"] or 0),
                "dur_game": float(r["duration_game"]) if r["duration_game"] else None,
            })
    print("E1 L0 -- %d valid runs\n" % len(rows))
    print("%-8s %-38s %5s %6s %8s %-22s %6s %8s" %
          ("route", "label", "n", "fails", "rate", "95% CI (Clopper-Pearson)", "width", "entropy"))
    summary = []
    for rt in ROUTES:
        rs = [r for r in rows if r["route"] == rt]
        n = len(rs)
        fails = sum(1 for r in rs if r["score"] < 100 or r["col"] > 0)
        lo, hi = cp_interval(fails, n)
        # outcome multiset: an outcome is the exact (score, collisions) pair
        counts = {}
        for r in rs:
            counts[(r["score"], r["col"])] = counts.get((r["score"], r["col"]), 0) + 1
        ent = -sum((c / n) * math.log2(c / n) for c in counts.values())
        print("%-8s %-38s %5d %6d %7.1f%% [%.3f, %.3f]        %6.3f %8.3f" %
              (rt, LABEL[rt], n, fails, 100 * fails / n, lo, hi, hi - lo, ent))
        summary.append((rt, n, fails, lo, hi, hi - lo, ent, counts))

    print("\n--- outcome multisets (an outcome = exact (score, collisions) pair) ---")
    for rt, n, fails, lo, hi, w, ent, counts in summary:
        print("%s: %d distinct outcome(s) over %d runs" % (rt, len(counts), n))
        for (sc, col), c in sorted(counts.items(), key=lambda kv: -kv[1]):
            verdict = "PASS" if (sc >= 100 and col == 0) else "FAIL"
            print("    score=%-8s collisions=%-8s x%-3d  %s" % (sc, col, c, verdict))

    print("\n--- run-order sequences (pass=., fail=X) ---")
    for rt in ROUTES:
        rs = sorted([r for r in rows if r["route"] == rt], key=lambda r: r["run"])
        seq = "".join("." if (r["score"] >= 100 and r["col"] == 0) else "X" for r in rs)
        print("  %-6s %s" % (rt, seq))

    print("\n--- durations (game seconds) ---")
    for rt in ROUTES:
        ds = [r["dur_game"] for r in rows if r["route"] == rt and r["dur_game"] is not None]
        if ds:
            uniq = sorted(set(round(d, 2) for d in ds))
            print("  %-6s n=%d distinct=%d  min=%.2f max=%.2f" % (rt, len(ds), len(uniq), min(ds), max(ds)))

    # H1 baseline record: the CI widths L1 must shrink by >=50% to support H1
    print("\n--- H1 baseline (CI width at L0; H1 SUPPORTED iff L1 width <= 50% of this) ---")
    for rt, n, fails, lo, hi, w, ent, counts in summary:
        print("  %-6s L0 CI width = %.3f  ->  H1 line for L1: width <= %.3f" % (rt, w, w / 2))


if __name__ == "__main__":
    main()
