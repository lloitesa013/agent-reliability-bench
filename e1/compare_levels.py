"""
E1 level comparison (stdlib only).  py -3.10 e1/compare_levels.py l0 l1 [...]

For each level x route: n, fails, rate, exact Clopper-Pearson 95% CI + width, outcome multiset.
Across levels: Fisher exact two-sided on the 2x2 (level x pass/fail) per route, branch-value set
comparison, and the sealed H1 criterion (CI width at L1 <= 50% of L0's width).

Also tests R52's recorded confound: does machine state (GPU memory in use at run start, logged
from L1 on) predict the outcome? Point-biserial-style comparison of mean GPU-MB for pass vs fail.
"""
import csv
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from vsi_gate import _binom_tail_le

ROUTES = ["11755", "18252", "3436"]


def load(level):
    p = os.path.join(HERE, "data", "e1_%s_runs.csv" % level)
    rows = []
    if not os.path.exists(p):
        return rows
    with open(p, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            col = float(r["col_vehicle"]) + float(r["col_pedestrian"]) + float(r["col_layout"])
            rows.append({"level": level, "route": r["route"], "run": int(r["run"]),
                         "score": round(float(r["score_composed"]), 3), "col": round(col, 3),
                         "dur": float(r["duration_game"]) if r["duration_game"] else None})
    return rows


def load_machine(level):
    p = os.path.join(HERE, "data", "e1_%s_machine.csv" % level)
    m = {}
    if not os.path.exists(p):
        return m
    with open(p, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            try:
                m[(r["route"], int(r["run"]))] = float(r["gpu_mem_used_mb"])
            except (ValueError, KeyError):
                pass
    return m


def cp_interval(k, n, alpha=0.05):
    def solve(target_lo):
        a, b = 0.0, 1.0
        for _ in range(200):
            mid = (a + b) / 2
            if target_lo(mid):
                a = mid
            else:
                b = mid
        return (a + b) / 2
    lo = 0.0 if k == 0 else solve(lambda p: 1.0 - _binom_tail_le(k - 1, n, p) < alpha / 2)
    hi = 1.0 if k == n else solve(lambda p: _binom_tail_le(k, n, p) > alpha / 2)
    return lo, hi


def fisher_two_sided(a, b, c, d):
    """2x2 [[a,b],[c,d]] exact test."""
    n = a + b + c + d
    row1, col1 = a + b, a + c

    def p_tab(x):
        y, z, w = row1 - x, col1 - x, n - row1 - col1 + x
        if min(y, z, w) < 0:
            return 0.0
        return (math.comb(row1, x) * math.comb(n - row1, col1 - x)) / math.comb(n, col1)

    obs = p_tab(a)
    lo_x, hi_x = max(0, col1 - (n - row1)), min(row1, col1)
    return min(1.0, sum(p_tab(x) for x in range(lo_x, hi_x + 1) if p_tab(x) <= obs + 1e-12))


def is_fail(r):
    return r["score"] < 100 or r["col"] > 0


def main():
    levels = sys.argv[1:] or ["l0", "l1"]
    data = {lv: load(lv) for lv in levels}
    mach = {lv: load_machine(lv) for lv in levels}

    print("=== per level x route ===")
    print("%-6s %-7s %4s %5s %7s  %-18s %6s  %s" % ("level", "route", "n", "fail", "rate", "95% CI", "width", "outcome values (count)"))
    stats = {}
    for lv in levels:
        for rt in ROUTES:
            rs = [r for r in data[lv] if r["route"] == rt]
            if not rs:
                continue
            n = len(rs)
            f = sum(1 for r in rs if is_fail(r))
            lo, hi = cp_interval(f, n)
            counts = {}
            for r in rs:
                counts[(r["score"], r["col"])] = counts.get((r["score"], r["col"]), 0) + 1
            vals = "  ".join("%s/%s x%d" % (k[0], k[1], v) for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))
            print("%-6s %-7s %4d %5d %6.1f%%  [%.3f,%.3f] %6.3f  %s" % (lv, rt, n, f, 100 * f / n, lo, hi, hi - lo, vals))
            stats[(lv, rt)] = {"n": n, "f": f, "lo": lo, "hi": hi, "w": hi - lo, "vals": set(counts)}

    if len(levels) >= 2:
        base = levels[0]
        for lv in levels[1:]:
            print("\n=== %s vs %s ===" % (base, lv))
            for rt in ROUTES:
                s0, s1 = stats.get((base, rt)), stats.get((lv, rt))
                if not s0 or not s1:
                    continue
                p = fisher_two_sided(s0["f"], s0["n"] - s0["f"], s1["f"], s1["n"] - s1["f"])
                same = s0["vals"] == s1["vals"]
                new = s1["vals"] - s0["vals"]
                print("  %-7s %s %d/%d -> %s %d/%d | Fisher p=%.3f | branch values %s%s"
                      % (rt, base, s0["f"], s0["n"], lv, s1["f"], s1["n"], p,
                         "IDENTICAL" if same else "DIFFER",
                         "" if not new else (" (new: %s)" % ", ".join("%s/%s" % v for v in sorted(new)))))
                if lv == "l1":
                    verdict = "SUPPORTED" if s1["w"] <= 0.5 * s0["w"] else "NOT SUPPORTED"
                    print("        H1 (sealed): L1 width %.3f vs line %.3f (= 50%% of L0 %.3f) -> %s"
                          % (s1["w"], 0.5 * s0["w"], s0["w"], verdict))

    for lv in levels:
        m = mach.get(lv) or {}
        if not m:
            continue
        print("\n=== %s: machine-state confound (R52 caveat) ===" % lv)
        for rt in ROUTES + ["ALL"]:
            rs = [r for r in data[lv] if (rt == "ALL" or r["route"] == rt) and (r["route"], r["run"]) in m]
            if len(rs) < 4:
                continue
            fmb = [m[(r["route"], r["run"])] for r in rs if is_fail(r)]
            pmb = [m[(r["route"], r["run"])] for r in rs if not is_fail(r)]
            if fmb and pmb:
                print("  %-6s GPU MB at start: fail mean=%.0f (n=%d)  pass mean=%.0f (n=%d)  diff=%+.0f"
                      % (rt, sum(fmb) / len(fmb), len(fmb), sum(pmb) / len(pmb), len(pmb),
                         sum(fmb) / len(fmb) - sum(pmb) / len(pmb)))


if __name__ == "__main__":
    main()
