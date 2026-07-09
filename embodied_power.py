"""
C (embodied, statistical spec) - how many CARLA runs does the failure-rate verifier need to CONFIRM a fix?
The embodied loop can only verify a fix that provably lowers the failure RATE below the flaky baseline
(route 11755 = 7/12 = 58%). Given that stochasticity, this computes the number of runs required to detect a
fix of a given effect size (two-proportion power). This is the honest cost/spec of embodied verified
self-improvement - and why it is the hard core (a real fix still needs retention-DAgger to CREATE it).
Pure computation, no GPU/CARLA.
"""
import math

Z = {0.05: 1.645, 0.025: 1.960}      # one-sided alpha
ZB = {0.80: 0.842, 0.90: 1.282}      # power

p0 = 7.0 / 12.0                       # measured baseline failure rate of route 11755 (R19+R21)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def n_per_arm(p0, p1, alpha=0.05, power=0.80):
    pbar = (p0 + p1) / 2
    za, zb = Z[alpha], ZB[power]
    num = (za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p0 * (1 - p0) + p1 * (1 - p1))) ** 2
    return math.ceil(num / (p0 - p1) ** 2)


def main():
    lo, hi = wilson(7, 12)
    print("baseline (route 11755): failure rate = 7/12 = %.2f, 95%% CI [%.2f, %.2f]" % (p0, lo, hi))
    print("  -> even the BASELINE is only known to +/-%.0f%% at n=12; single runs are worthless.\n" % (100 * (hi - lo) / 2))
    print("runs per arm to CONFIRM a fix lowers 58%% failure to p1 (one-sided a=0.05):")
    print("  target p1 |  fix effect  |  N runs @80%% power |  N runs @90%% power")
    for p1 in [0.40, 0.30, 0.20, 0.10]:
        n80 = n_per_arm(p0, p1, 0.05, 0.80)
        n90 = n_per_arm(p0, p1, 0.05, 0.90)
        print("     %.2f    |   -%.0f pts    |        %3d         |        %3d" % (p1, 100 * (p0 - p1), n80, n90))
    print("\n--- interpretation ---")
    print(">>> Confirming a LARGE fix (58pct -> 20pct, -38pts) needs ~%d runs @80pct power; a modest one" % n_per_arm(p0, 0.20))
    print(">>> (58pct -> 40pct) needs ~%d. At ~2-4 min/run that is 30-360 min of CARLA PER candidate fix," % n_per_arm(p0, 0.40))
    print(">>> and a self-improvement loop tries many candidates. This is the honest COST of embodied verified")
    print(">>> self-improvement, and why the full loop is the hard core: the statistical verifier is cheap")
    print(">>> to SPEC but a real fix must first be CREATED (retention-DAgger, multi-week). VSI-0 delivers")
    print(">>> the verifier + this spec; the fix-generation is future work.")


if __name__ == "__main__":
    main()
