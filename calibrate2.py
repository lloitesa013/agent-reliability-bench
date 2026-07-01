"""
Calibration attempt #2 (winning judge config).
Decision = the winning generated verdict (~0.84 acc).  Confidence = first-token logit margin
z = logit(ESCALATE) - logit(PASS)  ->  P_raw = sigmoid(z).
Then TEMPERATURE-SCALE z on a fact-group train split (fit T by NLL) and report ECE before/after,
risk-coverage (abstain on smallest |z|), and an abstain band.  Temp scaling fixes magnitude (ECE)
but NOT ranking -> risk-coverage tells us if the logit margin carries a real uncertainty signal.
"""
import json, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
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


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def ece(p, y, nb=10):
    p, y = np.asarray(p), np.asarray(y); e = np.linspace(0, 1, nb + 1); t = 0.0
    for i in range(nb):
        m = (p >= e[i]) & (p < e[i + 1]) if i < nb - 1 else (p >= e[i]) & (p <= e[i + 1])
        if m.sum():
            t += m.mean() * abs(p[m].mean() - y[m].mean())
    return t


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

    def fids(w):
        s = set()
        for x in (w, " " + w):
            t = tok.encode(x, add_special_tokens=False)
            if t:
                s.add(t[0])
        return list(s)
    esc_ids, pass_ids = fids("ESCALATE"), fids("PASS")

    Z, V = [], []
    B = 24
    for i in range(0, len(recs), B):
        chunk = recs[i:i + B]
        texts = [tok.apply_chat_template(
            [{"role": "system", "content": JUDGE_SYS},
             {"role": "user", "content": JUDGE_TMPL.format(q=r["query"], ev="\n".join("- " + d for d in r["retrieved"]), a=r["rag_answer"])}],
            tokenize=False, add_generation_prompt=True) for r in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to("cuda:0")
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=8, do_sample=False, output_scores=True,
                               return_dict_in_generate=True, pad_token_id=tok.pad_token_id)
        plen = enc.input_ids.shape[1]; s0 = g.scores[0].float()
        for j in range(len(chunk)):
            V.append(1 if "ESCALATE" in tok.decode(g.sequences[j][plen:], skip_special_tokens=True).upper() else 0)
            Z.append((s0[j, esc_ids].max() - s0[j, pass_ids].max()).item())
    Z = np.array(Z); V = np.array(V)
    y = np.array([1.0 if r["should_escalate"] else 0.0 for r in recs])
    Praw = sig(Z)

    print("verdict-decision acc = %.3f | z-sign-decision acc = %.3f | verdict==sign(z): %.3f"
          % ((V == y).mean(), ((Z > 0) == y).mean(), (V == (Z > 0)).mean()))
    print("raw logit ECE = %.3f" % ece(Praw, y))

    conf = np.abs(Z)
    order = np.argsort(-conf)
    print("\n=== risk-coverage (decision=verdict, rank by |z|, abstain smallest) ===")
    print("coverage | selective_error")
    for cov in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
        k = max(1, int(round(cov * len(Z)))); idx = order[:k]
        print("  %.2f    |     %.3f" % (cov, (V[idx] != y[idx]).mean()))

    # temperature scaling, 5-fold fact-group
    groups = [r["group"] for r in recs]; uniq = sorted(set(groups))
    Ts, e_before, e_after = [], [], []
    Tgrid = np.arange(0.3, 8.0, 0.1)
    for seed in range(5):
        rng = np.random.RandomState(seed); gg = uniq[:]; rng.shuffle(gg)
        ev_g = set(gg[:max(1, int(round(len(gg) * 0.3)))])
        tri = [i for i in range(len(recs)) if groups[i] not in ev_g]
        evi = [i for i in range(len(recs)) if groups[i] in ev_g]
        zt, yt = Z[tri], y[tri]
        best_T, best_nll = 1.0, 1e9
        for T in Tgrid:
            p = np.clip(sig(zt / T), 1e-6, 1 - 1e-6)
            nll = -(yt * np.log(p) + (1 - yt) * np.log(1 - p)).mean()
            if nll < best_nll:
                best_nll, best_T = nll, T
        Ts.append(best_T)
        e_before.append(ece(sig(Z[evi]), y[evi]))
        e_after.append(ece(sig(Z[evi] / best_T), y[evi]))
    print("\n=== temperature scaling (5-fold fact-group) ===")
    print("mean T=%.2f | ECE before=%.3f  after=%.3f" % (np.mean(Ts), np.mean(e_before), np.mean(e_after)))

    print("\n=== abstain band on |z| (decision=verdict) ===")
    for q in (0.0, 0.1, 0.2, 0.3):
        thr = np.quantile(conf, q)
        keep = conf >= thr
        print("  abstain smallest %.0f%%: abstain %.2f | acc on rest %.3f"
              % (q * 100, (~keep).mean(), (V[keep] == y[keep]).mean()))


if __name__ == "__main__":
    main()
