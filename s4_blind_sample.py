"""
s4-blind-sample -- S4-(3) blind arm sampling (S4_PREREG.md, sealed 9b7bfe2).

Enumerates the sealed 6-axis recipe space, applies the sealed exclusions, removes the 9
already-run S1 finetune recipes, and samples N=6 uniformly WITHOUT replacement with the seed
recorded in the registry note at registration time. Each sampled recipe is registered in
s4_trials.jsonl with the SAME panel lines as S1 (fix <=2/8 on 11755; reg12 pooled <=30%),
BEFORE any training or evaluation.

Space (sealed): epochs {3,5,10} x data {narrow,broad} x fix_ratio {none,0.4,0.6}
              x fix_bucket {route,family} x targeted_ret {no,yes} x freeze {no,yes}
Sealed exclusions: (fix_bucket=family AND data=narrow); (fix_ratio=none AND data=narrow).
Run:  py -3.10 s4_blind_sample.py            (idempotent: refuses to re-register)
"""
import itertools
import random

from vsi_registry import Registry

SEED = 20260718          # recorded in every trial note; fixed here at registration time

S1_POINTS = {            # the 9 finetune recipes the S1 search actually ran (R31-R41)
    (10, "narrow", None, "route", False, False),   # K1
    (3, "narrow", None, "route", False, False),    # K2a
    (3, "broad", None, "route", False, False),     # K2b
    (3, "broad", 0.4, "route", False, False),      # A1
    (5, "broad", 0.6, "route", False, False),      # A2
    (5, "broad", 0.6, "family", True, False),      # A3
    (5, "broad", 0.6, "route", True, False),       # A4
    (5, "broad", 0.6, "route", True, True),        # A5
    (5, "broad", 0.6, "route", False, True),       # A6
}


def space():
    out = []
    for ep, data, ratio, bucket, tr, fz in itertools.product(
            (3, 5, 10), ("narrow", "broad"), (None, 0.4, 0.6),
            ("route", "family"), (False, True), (False, True)):
        if data == "narrow" and bucket == "family":
            continue
        if data == "narrow" and ratio is None:
            continue
        out.append((ep, data, ratio, bucket, tr, fz))
    return out


def main():
    sp = space()
    pool = [p for p in sp if p not in S1_POINTS]
    print("space=%d  minus S1=%d  pool=%d" % (len(sp), len(sp) - len(pool), len(pool)))
    rng = random.Random(SEED)
    picks = rng.sample(pool, 6)

    reg = Registry("s4_trials.jsonl")
    for i, (ep, data, ratio, bucket, tr, fz) in enumerate(picks, 1):
        tid = "S4-B%d" % i
        print("%s: epochs=%d data=%s ratio=%s bucket=%s targeted=%s freeze=%s"
              % (tid, ep, data, ratio, bucket, tr, fz))
        if tid in reg.trials:
            print("  (already registered, skipping)")
            continue
        reg.register(tid,
                     recipe={"name": tid, "type": "finetune", "move": "blind-sample",
                             "target": "11755", "epochs": ep, "data": data,
                             "fix_ratio": ratio, "fix_bucket": bucket,
                             "targeted_ret": tr, "freeze": fz},
                     lines={"fix_max_fails": 2, "fix_n_planned": 8,
                            "ret_max_rate": 0.30, "ret_n_planned": 24},
                     note="S4-(3) blind arm, seed=%d, space=96 minus 9 S1 points, uniform without "
                          "replacement (S4_PREREG 9b7bfe2). Method comparison only; any panel-pass "
                          "still faces the deployment gate before adoption claims." % SEED)
    print("registered; lines sealed before any training")


if __name__ == "__main__":
    main()
