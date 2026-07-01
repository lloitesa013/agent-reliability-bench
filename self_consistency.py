"""
Self-consistency calibrated abstention -- THE differentiator: a watcher that abstains when its own
repeated judgments disagree.

Sample the (winning-config) judge K times per trace -> P(escalate) = fraction that say ESCALATE.
Agreement = max(P,1-P).  Disagreement (P near 0.5) = the judge is unsure of itself -> ABSTAIN.

Reports: ECE (vs the single-shot 0.30), risk-coverage, abstain-on-disagreement, and the sealed
A-E eval (does the self-consistency watcher still beat the rule C).
"""
import json, os, sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "bench"))
import run_bench as RB
MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
K = 5

JUDGE_SYS = ("You are a strict reliability auditor for AI answers used in high-stakes regulated "
             "domains (finance, healthcare, public services). You decide whether an answer must "
             "be ESCALATED to a human reviewer, or can PASS.")
JUDGE_TMPL = (
    "Decide if the ANSWER below should be ESCALATED or can PASS.\n"
    "ESCALATE if ANY of: the answer is likely wrong; it is not directly supported by the "
    "evidence; it says it does not know; it needed a calculation that may be wrong (recompute "
    "it and check); or the evidence contains conflicting values for the asked item.\n"
    "PASS only if the answer is clearly correct and supported.\n\n"
    "Question: {q}\nEvidence:\n{ev}\nAnswer: {a}\n\nReply with exactly one word: PASS or ESCALATE.")


def ece(p, y, nbins=10):
    p, y = np.asarray(p), np.asarray(y)
    e = np.linspace(0, 1, nbins + 1); tot = 0.0
    for i in range(nbins):
        m = (p >= e[i]) & (p < e[i + 1]) if i < nbins - 1 else (p >= e[i]) & (p <= e[i + 1])
        if m.sum():
            tot += m.mean() * abs(p[m].mean() - y[m].mean())
    return tot


def main():
    recs = [json.loads(l) for l in open(os.path.join(HERE, "traces_bench.jsonl"), encoding="utf-8")]
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda:0")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    torch.manual_seed(0)

    P = []
    B = 24
    for i in range(0, len(recs), B):
        chunk = recs[i:i + B]
        texts = [tok.apply_chat_template(
            [{"role": "system", "content": JUDGE_SYS},
             {"role": "user", "content": JUDGE_TMPL.format(q=r["query"], ev="\n".join("- " + d for d in r["retrieved"]), a=r["rag_answer"])}],
            tokenize=False, add_generation_prompt=True) for r in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to("cuda:0")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=8, do_sample=True, temperature=0.7, top_p=0.9,
                                 num_return_sequences=K, pad_token_id=tok.pad_token_id)
        plen = enc.input_ids.shape[1]
        for j in range(len(chunk)):
            votes = 0
            for k in range(K):
                txt = tok.decode(gen[j * K + k][plen:], skip_special_tokens=True).upper()
                votes += 1 if "ESCALATE" in txt else 0
            P.append(votes / K)
    P = np.array(P)
    y = np.array([1.0 if r["should_escalate"] else 0.0 for r in recs])
    dec = (P >= 0.5).astype(float)
    conf = np.maximum(P, 1 - P)

    print("=== self-consistency (K=%d) calibration ===" % K)
    print("ECE = %.3f   (single-shot logit ECE was 0.301)" % ece(P, y))
    print("decision accuracy (majority) = %.3f   escalate-rate = %.3f" % ((dec == y).mean(), dec.mean()))

    print("\n=== risk-coverage (abstain on most-disagreeing) ===")
    order = np.argsort(-conf)
    print("coverage | selective_error")
    for cov in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
        k = max(1, int(round(cov * len(P)))); idx = order[:k]
        print("  %.2f    |     %.3f" % (cov, (dec[idx] != y[idx]).mean()))

    print("\n=== abstain when NOT unanimous (any of K disagrees) ===")
    unan = (P == 0) | (P == 1)
    print("unanimous cases: %.2f  | accuracy on them: %.3f  | abstained (sent to human): %.2f"
          % (unan.mean(), (dec[unan] == y[unan]).mean(), (~unan).mean()))
    for lo, hi in [(0.2, 0.8), (0.4, 0.6)]:
        keep = (P <= lo) | (P >= hi)
        print("abstain band P in (%.1f,%.1f): abstain %.2f | acc on rest %.3f"
              % (lo, hi, (~keep).mean(), (dec[keep] == y[keep]).mean() if keep.sum() else float("nan")))

    # sealed A-E eval, watcher_prob = self-consistency P, fact-group 5-split
    for i, r in enumerate(recs):
        r["watcher_prob"] = float(P[i])
    groups = [r["group"] for r in recs]; uniq = sorted(set(groups))
    grid = [round(v, 2) for v in np.arange(0.05, 1.0, 0.05)]
    agg = {s: [] for s in "CDE"}
    for seed in range(5):
        rng = np.random.RandomState(seed); gg = uniq[:]; rng.shuffle(gg)
        ev_groups = set(gg[:max(1, int(round(len(gg) * 0.3)))])
        tr = [r for r in recs if r["group"] not in ev_groups]; ev = [r for r in recs if r["group"] in ev_groups]
        for attr, sysid in (("TAU", "C"), ("TAU_W", "D")):
            best_v, best = grid[0], -1.0
            for v in grid:
                setattr(RB, attr, v)
                er = RB.metrics(tr, sysid)["effective_reliability"]
                if er > best:
                    best, best_v = er, v
            setattr(RB, attr, best_v)
        for s in "CDE":
            agg[s].append(RB.metrics(ev, s)["effective_reliability"])
    print("\n=== sealed A-E eval (self-consistency watcher), 5-seed fact-group ===")
    for s in "CDE":
        print("  %s eff_rel = %.3f±%.3f" % (s, np.mean(agg[s]), np.std(agg[s])))
    print("  D beats C: %s" % (np.mean(agg["D"]) > np.mean(agg["C"])))


if __name__ == "__main__":
    main()
