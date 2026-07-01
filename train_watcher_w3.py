"""
Week-3 watcher trainer + honest A-E scorer for the sealed reliability benchmark.

- FACT-GROUP split (split by `group`, NOT per-record) -> anti-leak (DATA_REPORT §8-4).
- Watcher D = logistic on [ bge(query+evidence+answer) , groundedness ]  -> P(should_escalate).
- Thresholds chosen on TRAIN (maximize effective_reliability), applied to EVAL  (sealed protocol).
- Metrics come from the SEALED bench/run_bench.py (imported as a library; TAU/TAU_W set per
  protocol, the file itself is UNCHANGED). E still uses run_bench's placeholder value_mismatch.
- Reports A-E mean±std over 5 seeds, overall + per case_type + the headline D-vs-C.
"""
import json, os, sys
import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "bench"))
import run_bench as RB  # sealed metric definitions


def trace_text(r):
    return "Question: %s\nEvidence: %s\nAnswer: %s" % (
        r["query"], " ".join(r["retrieved"]), r["rag_answer"])


def main():
    recs = [json.loads(l) for l in open(os.path.join(HERE, "traces_bench.jsonl"), encoding="utf-8")]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("records=%d  device=%s" % (len(recs), device))

    emb = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)
    E = np.asarray(emb.encode([trace_text(r) for r in recs], normalize_embeddings=True,
                              batch_size=64, show_progress_bar=False), dtype=np.float32)
    g = np.array([[float(r.get("groundedness", 1.0))] for r in recs], dtype=np.float32)
    X = np.concatenate([E, g], axis=1)                      # (N, 385): trace embedding + groundedness
    y = np.array([1.0 if r["should_escalate"] else 0.0 for r in recs], dtype=np.float32)
    groups = [r["group"] for r in recs]
    uniq = sorted(set(groups))
    print("distinct fact-groups=%d" % len(uniq))

    grid = [round(v, 2) for v in np.arange(0.05, 1.0, 0.05)]
    SEEDS = [0, 1, 2, 3, 4]
    agg = {s: {k: [] for k in ("effective_reliability", "unsafe_pass", "overblock", "decision_acc")} for s in "ABCDE"}
    agg_ct = {s: {ct: [] for ct in ("direct", "reasoning", "distractor")} for s in "ABCDE"}

    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        gg = uniq[:]; rng.shuffle(gg)
        n_ev = max(1, int(round(len(gg) * 0.3)))
        ev_groups = set(gg[:n_ev])
        tr_idx = [i for i in range(len(recs)) if groups[i] not in ev_groups]
        ev_idx = [i for i in range(len(recs)) if groups[i] in ev_groups]

        # --- train logistic watcher on TRAIN groups only ---
        Xtr = torch.tensor(X[tr_idx]); ytr = torch.tensor(y[tr_idx]).unsqueeze(1)
        torch.manual_seed(seed)
        model = nn.Linear(X.shape[1], 1)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-3)
        lossf = nn.BCEWithLogitsLoss()
        model.train()
        for _ in range(500):
            opt.zero_grad(); loss = lossf(model(Xtr), ytr); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(torch.tensor(X))).squeeze(1).numpy()
        for i, r in enumerate(recs):
            r["watcher_prob"] = float(probs[i])

        tr_recs = [recs[i] for i in tr_idx]; ev_recs = [recs[i] for i in ev_idx]

        # --- choose thresholds on TRAIN (fair to C too: tune its groundedness cut) ---
        def tune(attr, system):
            best_v, best = grid[0], -1.0
            for v in grid:
                setattr(RB, attr, v)
                er = RB.metrics(tr_recs, system)["effective_reliability"]
                if er > best:
                    best, best_v = er, v
            setattr(RB, attr, best_v)
            return best_v
        tune("TAU", "C"); tune("TAU_W", "D")

        for s in "ABCDE":
            m = RB.metrics(ev_recs, s)
            for k in agg[s]:
                agg[s][k].append(m[k])
            for ct in ("direct", "reasoning", "distractor"):
                sub = [r for r in ev_recs if r["case_type"] == ct]
                if sub:
                    agg_ct[s][ct].append(RB.metrics(sub, s)["effective_reliability"])
        print("seed %d: train=%d eval=%d  TAU=%.2f TAU_W=%.2f" % (seed, len(tr_recs), len(ev_recs), RB.TAU, RB.TAU_W))

    def ms(a):
        return (float(np.mean(a)), float(np.std(a)))

    print("\n=== A-E over %d seeds (mean±std), FACT-GROUP split ===" % len(SEEDS))
    print("sys | eff_rel         unsafe_pass     overblock       dec_acc")
    for s in "ABCDE":
        er, up, ob, da = (ms(agg[s][k]) for k in ("effective_reliability", "unsafe_pass", "overblock", "decision_acc"))
        print("%s   | %.3f±%.3f   %.3f±%.3f   %.3f±%.3f   %.3f±%.3f" % (s, er[0], er[1], up[0], up[1], ob[0], ob[1], da[0], da[1]))

    print("\nper case_type eff_rel (mean over seeds):")
    print("      direct   reasoning  distractor")
    for s in "ABCDE":
        row = [np.mean(agg_ct[s][ct]) if agg_ct[s][ct] else float("nan") for ct in ("direct", "reasoning", "distractor")]
        print("  %s   %.3f    %.3f     %.3f" % (s, row[0], row[1], row[2]))

    Cer, Der, Eer = (ms(agg[s]["effective_reliability"])[0] for s in "CDE")
    Cr, Dr = (np.mean(agg_ct[s]["reasoning"]) for s in "CD")
    print("\n=== HEADLINE ===")
    print("D (trained watcher) %.3f  vs  C (groundedness rule) %.3f   -> D beats C: %s" % (Der, Cer, Der > Cer))
    print("D reasoning %.3f vs C reasoning %.3f            -> D>C on reasoning (overblock cut): %s" % (Dr, Cr, Dr > Cr))
    print("E (watcher+placeholder value_mismatch) %.3f     -> E beats D: %s  (real detector = future)" % (Eer, Eer > Der))


if __name__ == "__main__":
    main()
