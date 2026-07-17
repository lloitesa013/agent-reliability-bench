"""
s3-sprt-replay -- paper-2 exploratory analysis (GPU-free): how many confirm runs did the fixed-N
deployment gate waste, relative to pure CURTAILMENT?

Curtailment = stop a route's interleaved confirm (cand/base alternating, recorded order) the moment
the pre-registered verdict (confirmed regression iff cand_fails>=3/4 AND base_fails<=1/4; for the
2+2 signature routes: cand 2/2 AND base 0/2) is mathematically decided regardless of the remaining
runs. By construction the verdict is IDENTICAL to the full protocol -- zero statistical assumptions,
zero added error. The saving is therefore a lower bound on what sequential verification buys;
a true SPRT/anytime-valid rule (accepting controlled error) would stop earlier still.

Replays: R33 (K1 confirm batch, confirm.csv) and R37 (A2 fleet confirm, conf2_p*.csv).
Rule discipline: the stopping rule above was fixed before running this script; the script asserts
verdict-identity on every route.
"""
import csv
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "s3_data")


def load(f):
    with open(os.path.join(D, f), encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def arms_for(rows, run_key):
    """route -> {'cand': [fail...], 'base': [fail...]} in recorded order, NA rows dropped."""
    out = defaultdict(lambda: {"cand": [], "base": []})
    for r in sorted(rows, key=lambda r: (r["route"], int(r[run_key]) if r[run_key] not in ("", "NA") else 99)):
        if r["failed"] in ("NA", ""):
            continue
        out[r["route"]][r["arm"]].append(int(r["failed"]))
    return out


def verdict_full(c, b, nc, nb):
    """Final verdict under the pre-registered rule with arm sizes (nc, nb)."""
    need_c = 3 if nc >= 4 else nc            # 4+4 rule: >=3/4 ; 2+2 signature: 2/2
    max_b = 1 if nb >= 4 else 0              # 4+4: <=1/4 ; 2+2: 0/2
    return sum(c[:nc]) >= need_c and sum(b[:nb]) <= max_b


def curtailed_runs(c, b, nc, nb):
    """Interleave cand/base in recorded order; return #runs consumed until decided."""
    need_c = 3 if nc >= 4 else nc
    max_b = 1 if nb >= 4 else 0
    seq = []
    for i in range(max(nc, nb)):
        if i < nc:
            seq.append(("c", c[i]))
        if i < nb:
            seq.append(("b", b[i]))
    cf = bf = cs = bs = 0                     # fails/seen per arm
    for k, (arm, fail) in enumerate(seq, 1):
        if arm == "c":
            cs += 1; cf += fail
        else:
            bs += 1; bf += fail
        # 'confirmed' final state (cf_final>=need_c AND bf_final<=max_b) still reachable?
        conf_possible = (cf + (nc - cs) >= need_c) and (bf <= max_b)
        # 'not confirmed' final state still reachable? cf can only grow, so cf_final<need_c is
        # reachable iff cf<need_c now (remaining cand runs may all pass); bf_final>max_b is
        # reachable iff even the remaining base runs failing pushes past the cap.
        notconf_possible = (cf < need_c) or (bf + (nb - bs) > max_b)
        if conf_possible != notconf_possible:  # exactly one outcome remains -> decided
            return k, conf_possible
    return len(seq), (cf >= need_c and bf <= max_b)


def replay(name, arms):
    total_full = total_curt = n_conf = 0
    per_route = []
    for rt, a in sorted(arms.items()):
        c, b = a["cand"], a["base"]
        nc, nb = min(len(c), 4), min(len(b), 4)
        if nc == 0 or nb == 0:
            continue
        full = nc + nb
        used, v_curt = curtailed_runs(c, b, nc, nb)
        v_full = verdict_full(c, b, nc, nb)
        assert v_curt == v_full, (rt, v_curt, v_full)   # verdict identity, checked per route
        n_conf += v_full
        total_full += full
        total_curt += used
        per_route.append((rt, full, used, v_full))
    sav = 100.0 * (1 - total_curt / total_full)
    print("== %s ==" % name)
    print("routes: %d | confirmed: %d | runs full=%d curtailed=%d -> SAVED %.0f%%"
          % (len(per_route), n_conf, total_full, total_curt, sav))
    worst = sorted(per_route, key=lambda x: x[1] - x[2])[:3]
    best = sorted(per_route, key=lambda x: x[2] - x[1])[:3]
    print("  biggest savers:", [(rt, "%d->%d" % (f, u)) for rt, f, u, _ in sorted(per_route, key=lambda x: x[2]-x[1])[:5]])
    return total_full, total_curt, n_conf


def main():
    k1 = arms_for(load("confirm.csv"), "run")
    rows2 = []
    for f in ["conf2_p2000.csv", "conf2_p2004.csv", "conf2_p2008.csv"]:
        rows2 += load(f)
    a2 = arms_for(rows2, "k")

    f1, c1, n1 = replay("R33 / K1 confirm batch", k1)
    f2, c2, n2 = replay("R37 / A2 fleet confirm", a2)
    tf, tc = f1 + f2, c1 + c2
    print("\n== POOLED ==")
    print("full protocol: %d runs | curtailed: %d runs | SAVED %.0f%% -- with IDENTICAL verdicts"
          % (tf, tc, 100.0 * (1 - tc / tf)))
    print("(curtailment = zero-assumption lower bound; SPRT/anytime-valid stops earlier still)")


if __name__ == "__main__":
    main()
