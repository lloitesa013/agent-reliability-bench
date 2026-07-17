"""
s3-w4-analysis -- recompute R47 (gate-reuse decay measurement) from the committed W4 CSVs.

Surfaces: FRESH = the 51 baseline-non-clean routes never used in any recipe iteration;
ITERATED = the reg-12 confirmed-regression routes reused across the S1 panels. For each, the
candidate (A6) vs contemporaneous baseline pooled failure rate; decay would show as A6 doing
relatively BETTER on the iterated surface than on fresh (flattered by reuse). R47 found the
opposite: A6's damage is concentrated on the known surface; out-of-sample it never dropped below
baseline. A6's reg-12 candidate rate (4/24) comes from the committed a6p_p*.csv panel files.
Run:  py -3.10 s3_w4_analysis.py
"""
import csv
import os
import collections

from vsi_gate import _binom_tail_le

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "s3_data")
REG12 = {'2143', '24258', '25845', '28243', '2913', '3086', '3410', '3697', '3717', '3737', '4183', '4669'}


def load(f):
    with open(os.path.join(D, f), encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def main():
    rows = []
    for p in (2000, 2004, 2008):
        rows += load("w4_p%d.csv" % p)
    valid = [r for r in rows if r["failed"] not in ("NA", "")]
    print("W4 runs: %d valid / %d invalid" % (len(valid), len(rows) - len(valid)))

    def pool(rs):
        return sum(int(r["failed"]) for r in rs), len(rs)

    fc, nc = pool([r for r in valid if r["route"] not in REG12 and r["arm"] == "cand"])
    fb, nb = pool([r for r in valid if r["route"] not in REG12 and r["arm"] == "base"])
    rb, nrb = pool([r for r in valid if r["route"] in REG12 and r["arm"] == "base"])

    # A6 on reg12 from the committed fleet panel files (candidate arm of the A6 panel)
    a6 = [r for p in (2000, 2004, 2008) for r in load("a6p_p%d.csv" % p)]
    a6v = [r for r in a6 if r.get("failed") not in ("NA", "", None) and r["route"] in REG12]
    ra, nra = pool(a6v)

    print("FRESH surface:    A6 %d/%d = %.0f%%   base %d/%d = %.0f%%   uplift %+.1fpp"
          % (fc, nc, 100 * fc / nc, fb, nb, 100 * fb / nb, 100 * (fc / nc - fb / nb)))
    print("ITERATED (reg12): A6 %d/%d = %.0f%%   base %d/%d = %.0f%%   uplift %+.1fpp"
          % (ra, nra, 100 * ra / nra, rb, nrb, 100 * rb / nrb, 100 * (ra / nra - rb / nrb)))

    p0 = fb / nb
    p_hi = 1.0 - _binom_tail_le(fc - 1, nc, p0) if fc > 0 else 1.0
    print("decay test (A6 significantly WORSE on fresh?): one-sided p = %.3f -> %s"
          % (p_hi, "DECAY SIGNAL" if p_hi < 0.05 else "no decay detected"))

    per = collections.defaultdict(lambda: {"cand": [], "base": []})
    for r in valid:
        if r["route"] not in REG12:
            per[r["route"]][r["arm"]].append(int(r["failed"]))
    worse = sum(1 for a in per.values() if a["cand"] and a["base"]
                and sum(a["cand"]) / len(a["cand"]) > sum(a["base"]) / len(a["base"]))
    better = sum(1 for a in per.values() if a["cand"] and a["base"]
                 and sum(a["cand"]) / len(a["cand"]) < sum(a["base"]) / len(a["base"]))
    print("paired fresh routes: A6 worse %d / better %d / same %d"
          % (worse, better, sum(1 for a in per.values() if a["cand"] and a["base"]) - worse - better))


if __name__ == "__main__":
    main()
