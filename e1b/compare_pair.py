"""
E1-B paired divergence localization (stdlib only).

  py -3.10 e1b/compare_pair.py <ticksA.csv> <ticksB.csv>

Aligns two tick logs by step and reports, per stream, the FIRST differing tick:
  t_obs   observation entering the network (pre-forward tensor bytes)
  t_net   raw network output (pre-post-processing)
  t_ctrl  applied control (post post-processors / shield / initial-frame override)
plus the sealed X=0.50 m pose-divergence tick, and the MAGNITUDE of the first difference in each
channel (to separate rounding-level from behaviour-level divergence).

Only the first divergence is interpretable; everything after is feedback-contaminated
(E1B_PREREG.md). HB verdict is printed but the human reads the evidence, not the label.
"""
import csv
import os
import sys

X_METERS = 0.50          # sealed in E1B_PREREG.md / E1_PARAMS.md


def load(path):
    rows = {}
    with open(path, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            try:
                rows[int(r["step"])] = r
            except (ValueError, KeyError):
                continue
    return rows


def fnum(r, k):
    try:
        return float(r[k])
    except (ValueError, KeyError, TypeError):
        return None


def first_diff(a, b, steps, key):
    for s in steps:
        if a[s].get(key, "") != b[s].get(key, ""):
            return s
    return None


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    pa, pb = sys.argv[1], sys.argv[2]
    A, B = load(pa), load(pb)
    steps = sorted(set(A) & set(B))
    print("A = %s  (%d ticks)" % (os.path.basename(pa), len(A)))
    print("B = %s  (%d ticks)" % (os.path.basename(pb), len(B)))
    print("common steps: %d  (range %d..%d)" % (len(steps), steps[0], steps[-1]) if steps else "no common steps")
    if not steps:
        return 1

    t_obs = first_diff(A, B, steps, "obs_hash")
    t_net = first_diff(A, B, steps, "net_out_hash")
    t_ctrl = first_diff(A, B, steps, "ctrl_hash")

    t_pose = None
    for s in steps:
        ax, ay = fnum(A[s], "x"), fnum(A[s], "y")
        bx, by = fnum(B[s], "x"), fnum(B[s], "y")
        if None in (ax, ay, bx, by):
            continue
        if ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 > X_METERS:
            t_pose = s
            break

    print("\n--- first divergence (tick index) ---")
    for name, t in (("t_obs ", t_obs), ("t_net ", t_net), ("t_ctrl", t_ctrl),
                    ("t_pose(>%.2fm)" % X_METERS, t_pose)):
        print("  %-14s %s" % (name, "identical over common range" if t is None else t))

    firsts = [t for t in (t_obs, t_net, t_ctrl) if t is not None]
    if firsts:
        s = min(firsts)
        print("\n--- magnitudes at the first divergent tick (step %d) ---" % s)
        for k in ("x", "y", "yaw", "speed", "steer", "throttle", "brake"):
            va, vb = fnum(A[s], k), fnum(B[s], k)
            if va is None or vb is None:
                continue
            print("    %-9s A=%-22r B=%-22r  |diff|=%.3e" % (k, va, vb, abs(va - vb)))
        # how far back does the LAST identical tick sit?
        prev = [x for x in steps if x < s]
        print("    last identical tick: %s" % (prev[-1] if prev else "none"))

    print("\n--- sealed verdict (E1B_PREREG HB1-HB4) ---")
    if not firsts:
        print("  HB1: all three streams identical over the common range -> "
              "source is L3 (kernel selection); determinism is purchasable.")
    elif t_obs is not None and (t_net is None or t_obs < t_net):
        print("  HB2: observation diverges first (t_obs=%s < t_net=%s) -> server-side "
              "(sim/physics/traffic) source." % (t_obs, t_net))
    elif t_net is not None and (t_obs is None or t_net <= t_obs):
        print("  HB3: network output diverges with identical observation "
              "(t_net=%s <= t_obs=%s) -> residual inference nondeterminism." % (t_net, t_obs))
    elif t_ctrl is not None:
        print("  HB4: obs and net_out identical, control differs (t_ctrl=%s) -> control-stack "
              "source (post-processors / shield)." % t_ctrl)
    else:
        print("  AMBIGUOUS -- report as-is, do not resolve by choice.")

    tail_a = [s for s in sorted(A) if s > steps[-1]]
    tail_b = [s for s in sorted(B) if s > steps[-1]]
    if tail_a or tail_b:
        print("\n  note: episode lengths differ (A has %d ticks past the common range, B has %d);"
              % (len(tail_a), len(tail_b)))
        print("        divergence in length is downstream of the first divergence above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
