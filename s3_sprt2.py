"""
s3-sprt2 -- Wald-SPRT replay of the deployment confirms (rule fixed BEFORE running; GPU-free).

Rung 2 after curtailment (R45: 26% saved, verdict-identical). SPRT accepts a controlled error to
stop EARLIER than curtailment. Per route, each arm runs its own Wald SPRT:

  H0: p_fail = 0.10 (route is fine / flaky-noise)     H1: p_fail = 0.70 (regression-grade)
  alpha = beta = 0.10 per arm
  LLR(fail) = ln(.7/.1) = +1.946 ; LLR(pass) = ln(.3/.9) = -1.099
  accept H1 when LLR >= ln((1-b)/a) = ln 9 ; accept H0 when LLR <= ln(b/(1-a)) = -ln 9
  => H1 after 2 straight fails; H0 after 2 straight passes.

Route verdict: CONFIRMED iff cand-arm accepts H1 AND base-arm accepts H0. If an arm's SPRT is
undecided when its recorded 4 runs are exhausted, that arm falls back to the fixed rule
(cand>=3/4 fails ; base<=1/4) at full cost.

Honesty: unlike curtailment, SPRT verdicts are NOT guaranteed identical to the fixed protocol --
concordance is measured and every disagreement is printed. Savings only count if concordance holds.
"""
import csv
import math
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "s3_data")

P0, P1 = 0.10, 0.70
LLR_F, LLR_P = math.log(P1 / P0), math.log((1 - P1) / (1 - P0))
A, B = math.log(9.0), -math.log(9.0)          # alpha = beta = 0.10


def load(f):
    with open(os.path.join(D, f), encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def arms_for(rows, run_key):
    out = defaultdict(lambda: {"cand": [], "base": []})
    for r in sorted(rows, key=lambda r: (r["route"], int(r[run_key]) if r[run_key] not in ("", "NA") else 99)):
        if r["failed"] not in ("NA", ""):
            out[r["route"]][r["arm"]].append(int(r["failed"]))
    return out


def sprt_arm(fails, n_max):
    """Return (decision 'H1'|'H0'|None, runs_used, llr_path). Uses recorded outcomes in order."""
    llr, used = 0.0, 0
    for f in fails[:n_max]:
        used += 1
        llr += LLR_F if f else LLR_P
        if llr >= A:
            return "H1", used
        if llr <= B:
            return "H0", used
    return None, used


def fixed_verdict(c, b, nc, nb):
    need_c = 3 if nc >= 4 else nc
    max_b = 1 if nb >= 4 else 0
    return sum(c[:nc]) >= need_c and sum(b[:nb]) <= max_b


def replay(name, arms):
    tot_fixed = tot_sprt = agree = disagree = 0
    disagreements = []
    for rt, a in sorted(arms.items()):
        c, b = a["cand"], a["base"]
        nc, nb = min(len(c), 4), min(len(b), 4)
        if nc == 0 or nb == 0:
            continue
        vf = fixed_verdict(c, b, nc, nb)
        dc, uc = sprt_arm(c, nc)
        db, ub = sprt_arm(b, nb)
        used = uc + ub
        if dc is None:                          # fallback: fixed rule on that arm, full cost
            dc = "H1" if sum(c[:nc]) >= (3 if nc >= 4 else nc) else "H0"
            used += (nc - uc)
        if db is None:
            db = "H1" if sum(b[:nb]) > (1 if nb >= 4 else 0) else "H0"
            used += (nb - ub)
        vs = (dc == "H1" and db == "H0")
        tot_fixed += nc + nb
        tot_sprt += used
        if vs == vf:
            agree += 1
        else:
            disagree += 1
            disagreements.append((rt, "fixed=%s sprt=%s cand=%s base=%s" % (vf, vs, c[:nc], b[:nb])))
    print("== %s ==" % name)
    print("routes %d | agree %d, disagree %d | runs fixed=%d sprt=%d -> SAVED %.0f%%"
          % (agree + disagree, agree, disagree, tot_fixed, tot_sprt,
             100.0 * (1 - tot_sprt / tot_fixed)))
    for rt, msg in disagreements:
        print("  DISAGREE %s: %s" % (rt, msg))
    return tot_fixed, tot_sprt, agree, disagree


def main():
    k1 = arms_for(load("confirm.csv"), "run")
    rows2 = []
    for f in ["conf2_p2000.csv", "conf2_p2004.csv", "conf2_p2008.csv"]:
        rows2 += load(f)
    a2 = arms_for(rows2, "k")
    f1, s1, g1, d1 = replay("R33 / K1 confirms", k1)
    f2, s2, g2, d2 = replay("R37 / A2 confirms", a2)
    print("\n== POOLED ==")
    print("runs fixed=%d  sprt=%d  -> SAVED %.0f%% | concordance %d/%d routes"
          % (f1 + f2, s1 + s2, 100.0 * (1 - (s1 + s2) / (f1 + f2)),
             g1 + g2, g1 + g2 + d1 + d2))
    print("(curtailment floor was 26% with identity; SPRT trades a controlled 10%/arm error for the extra)")


if __name__ == "__main__":
    main()
