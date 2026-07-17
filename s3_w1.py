"""
s3-w1 -- W1 of the S3 pre-registration (S3_PREREG.md, sealed f5e1923): retrospective risk scorer.

Question: can a scorer that NEVER sees candidate outcomes rank the deployment screen's confirmed
regressions into its top-30? Sealed line: leave-one-candidate-out pooled recall@30 >= 12/18
(random expectation ~3.2/18). This is the go/no-go for all S3 GPU spend.

Feature contract (sealed): candidate-independent only --
  route_meta.csv        : town, length, #scenarios, scenario type      (static)
  batch_progress.csv    : BASELINE 220-sweep score/penalty/infractions/crash (baseline model only)
  + per-scenario-TYPE aggregates computed from the baseline sweep alone.
EXCLUDED by construction: anything measured on any candidate; baseline arms of the confirm
batches (only available for flagged routes -> would leak the flag).

LOCO protocol: train on K1's 12 confirmed (positives) vs the rest of the universe -> rank A2's
routes, count A2's 6 in top-30; and vice versa. Pooled hits >= 12 passes. Verdict is computed by
the S2 registry machinery (misses encoded against a count line: fails=misses<=6 of n=18).

Model: L2 logistic regression, pure stdlib (plain gradient descent; ~170 rows x ~20 features).
Run:  py -3.10 s3_w1.py
"""
import csv
import math
import os

from vsi_registry import Registry

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "s3_data")


def load(f):
    with open(os.path.join(D, f), encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def confirmed(rows_by_route):
    return sorted(rt for rt, a in rows_by_route.items()
                  if len(a["cand"]) >= 4 and sum(a["cand"][:4]) >= 3 and sum(a["base"][:4]) <= 1)


def confirm_sets():
    import collections
    k1 = collections.defaultdict(lambda: {"cand": [], "base": []})
    for r in load("confirm.csv"):
        if r["failed"] not in ("NA", ""):
            k1[r["route"]][r["arm"]].append(int(r["failed"]))
    a2 = collections.defaultdict(lambda: {"cand": [], "base": []})
    for f in ["conf2_p2000.csv", "conf2_p2004.csv", "conf2_p2008.csv"]:
        for r in load(f):
            if r["failed"] not in ("NA", ""):
                a2[r["route"]][r["arm"]].append(int(r["failed"]))
    return set(confirmed(k1)), set(confirmed(a2))


def build_features():
    meta = {r["route"]: r for r in load("route_meta.csv")}
    base = {r["route_id"]: r for r in load("batch_progress.csv")}

    # per-scenario-type aggregates from the BASELINE sweep only
    import collections
    ty_n, ty_fail, ty_score = collections.Counter(), collections.Counter(), collections.defaultdict(list)
    for rt, m in meta.items():
        b = base.get(rt)
        if not b:
            continue
        failed = (b["result"] == "FAILED") or (b["status"] != "Completed")
        try:
            sc = float(b["score_composed"])
        except ValueError:
            sc = 0.0
        for ty in (m["scenario_types"].split(";") if m["scenario_types"] else ["none"]):
            ty_n[ty] += 1
            ty_fail[ty] += failed
            ty_score[ty].append(sc)

    towns = sorted({m["town"] for m in meta.values()})

    def fvec(rt):
        m, b = meta[rt], base.get(rt)
        tys = m["scenario_types"].split(";") if m["scenario_types"] else ["none"]
        # baseline behavior (route level); route 2509 has no baseline row -> neutral fill
        if b and b["score_composed"] not in ("", "NA"):
            sc = float(b["score_composed"]) / 100.0
            ninf = min(float(b["num_infractions"] or 0) / 10.0, 2.0)
            crashed = 0.0 if b["status"] == "Completed" else 1.0
        else:
            sc, ninf, crashed = 0.8, 0.2, 1.0
        # scenario-type aggregates (baseline-only), worst over the route's types
        t_fail = max((ty_fail[t] / ty_n[t]) if ty_n[t] else 0.0 for t in tys)
        t_score = min((sum(ty_score[t]) / len(ty_score[t]) / 100.0) if ty_score[t] else 1.0 for t in tys)
        v = [1.0,                                   # bias
             float(m["length_m"]) / 1000.0,
             float(m["n_scenarios"]) / 5.0,
             sc, ninf, crashed, t_fail, t_score]
        v += [1.0 if m["town"] == t else 0.0 for t in towns]
        return v

    return meta, fvec


def train_logistic(X, y, l2=1.0, iters=3000, lr=0.5):
    n, k = len(X), len(X[0])
    w = [0.0] * k
    for _ in range(iters):
        g = [0.0] * k
        for xi, yi in zip(X, y):
            z = sum(wj * xj for wj, xj in zip(w, xi))
            p = 1.0 / (1.0 + math.exp(-max(min(z, 30), -30)))
            e = p - yi
            for j in range(k):
                g[j] += e * xi[j]
        for j in range(k):
            w[j] = w[j] - lr * (g[j] / n + l2 * w[j] / n * (0 if j == 0 else 1))
    return w


def score(w, x):
    return sum(wj * xj for wj, xj in zip(w, x))


def main():
    k1c, a2c = confirm_sets()
    assert len(k1c) == 12 and len(a2c) == 6, (len(k1c), len(a2c))
    meta, fvec = build_features()

    # universe = the deployment-screen route set (A2 fleet screen ledger) + all confirm routes
    g2 = set()
    for f in ["fleet_g2_p2000.csv", "fleet_g2_p2004.csv", "fleet_g2_p2008.csv"]:
        g2 |= {r["route"] for r in load(f)}
    conf_routes = {r["route"] for r in load("confirm.csv")}
    for f in ["conf2_p2000.csv", "conf2_p2004.csv", "conf2_p2008.csv"]:
        conf_routes |= {r["route"] for r in load(f)}
    universe = sorted((g2 | conf_routes) & set(meta))
    print("universe: %d routes | positives: K1 %d, A2 %d (overlap %d)"
          % (len(universe), len(k1c), len(a2c), len(k1c & a2c)))

    folds = [("train-K1 -> test-A2", k1c, a2c), ("train-A2 -> test-K1", a2c, k1c)]
    pooled_hits, pooled_total, report = 0, 0, []
    for name, train_pos, test_pos in folds:
        X = [fvec(rt) for rt in universe]
        y = [1.0 if rt in train_pos else 0.0 for rt in universe]
        w = train_logistic(X, y)
        ranked = sorted(universe, key=lambda rt: -score(w, fvec(rt)))
        top30 = set(ranked[:30])
        hits = sorted(test_pos & top30)
        misses = sorted(test_pos - top30)
        ranks = {rt: ranked.index(rt) + 1 for rt in sorted(test_pos)}
        pooled_hits += len(hits)
        pooled_total += len(test_pos)
        report.append((name, hits, misses, ranks))
        print("%s: %d/%d in top-30 | ranks: %s" % (name, len(hits), len(test_pos), ranks))

    misses_total = pooled_total - pooled_hits
    print("\nPOOLED LOCO recall@30: %d/%d (sealed line >=12/18; random ~3.2)"
          % (pooled_hits, pooled_total))

    # authoritative ledger s3_trials.jsonl is committed and FROZEN; replays verify against a
    # fresh temp ledger so this script stays runnable end-to-end (repro pack requirement)
    import tempfile
    ledger = os.path.join(HERE, "s3_trials.jsonl")
    if os.path.exists(ledger) and "S3-W1" in Registry(ledger).trials:
        ledger = os.path.join(tempfile.gettempdir(), "s3_w1_replay.jsonl")
        if os.path.exists(ledger):
            os.remove(ledger)
    reg = Registry(ledger)
    if "S3-W1" not in reg.trials:
        reg.register("S3-W1",
                     recipe={"name": "S3-W1", "type": "risk-screen", "move": "retrospective-loco"},
                     lines={"fix_max_fails": 6, "fix_n_planned": 18},
                     note="LOCO recall@30; fails=misses (18-hits); line misses<=6 == recall>=12/18. "
                          "Universe=deployment-screen set + confirm routes; features per sealed "
                          "contract (static + baseline-sweep only).")
    reg.add_arm("S3-W1", "fix", {"fails": misses_total, "n": pooled_total})
    v = reg.verdict("S3-W1")
    print("REGISTRY VERDICT: %s  axes=%s" % (v["verdict"], v["axes"]))
    print("(PANEL_PASS => W1 GREEN, GPU spend approved -> W2; REJECT => S3 risk-screen direction "
          "closes as bounded negative, no GPU spent)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
