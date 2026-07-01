"""
Calibration / abstention analysis of the reading judge -- our differentiator: a watcher that
knows WHEN IT IS UNSURE and can abstain.

Confidence = logit-based P(ESCALATE) from the first generated token (softmax over the PASS vs
ESCALATE first-token logits) -- more principled than verbal confidence.

Reports:
  - ECE (does stated P(escalate) match the actual escalate rate?)
  - risk-coverage (abstain on the least-confident -> error on the rest should drop)
  - an abstain band (|P-0.5| small -> send to human), and accuracy on the confident remainder
"""
import json, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")

JUDGE_SYS = ("You are a strict reliability auditor for AI answers used in high-stakes regulated "
             "domains. Decide whether an answer must be ESCALATED to a human, or can PASS.")
JUDGE_TMPL = (
    "ESCALATE if the answer is likely wrong, unsupported, uncertain, needs a calculation that may "
    "be wrong, or the evidence has conflicting values. PASS only if clearly correct and supported.\n\n"
    "Question: {q}\nEvidence:\n{ev}\nAnswer: {a}\n\nReply with exactly one word: PASS or ESCALATE.")


def ece(p, y, nbins=10):
    p, y = np.asarray(p), np.asarray(y)
    edges = np.linspace(0, 1, nbins + 1)
    tot = 0.0
    for i in range(nbins):
        m = (p >= edges[i]) & (p < edges[i + 1]) if i < nbins - 1 else (p >= edges[i]) & (p <= edges[i + 1])
        if m.sum() == 0:
            continue
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

    def first_ids(word):
        s = set()
        for w in (word, " " + word):
            t = tok.encode(w, add_special_tokens=False)
            if t:
                s.add(t[0])
        return list(s)
    esc_ids, pass_ids = first_ids("ESCALATE"), first_ids("PASS")

    P = []
    B = 32
    for i in range(0, len(recs), B):
        chunk = recs[i:i + B]
        texts = [tok.apply_chat_template(
            [{"role": "system", "content": JUDGE_SYS},
             {"role": "user", "content": JUDGE_TMPL.format(q=r["query"], ev="\n".join("- " + d for d in r["retrieved"]), a=r["rag_answer"])}],
            tokenize=False, add_generation_prompt=True) for r in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to("cuda:0")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=1, do_sample=False, output_scores=True,
                                 return_dict_in_generate=True, pad_token_id=tok.pad_token_id)
        logits = out.scores[0].float()  # (b, vocab)
        el = logits[:, esc_ids].max(dim=1).values
        pl = logits[:, pass_ids].max(dim=1).values
        p_esc = torch.softmax(torch.stack([pl, el], dim=1), dim=1)[:, 1]
        P.extend(p_esc.cpu().numpy().tolist())

    P = np.array(P)
    y = np.array([1.0 if r["should_escalate"] else 0.0 for r in recs])
    dec = (P >= 0.5).astype(float)
    conf = np.maximum(P, 1 - P)  # decision confidence

    print("=== calibration of judge P(escalate) ===")
    print("ECE = %.3f   (0 = perfectly calibrated)" % ece(P, y))
    print("decision accuracy @0.5 = %.3f   escalate-rate = %.3f" % ((dec == y).mean(), dec.mean()))

    print("\n=== risk-coverage (abstain on least-confident) ===")
    order = np.argsort(-conf)                      # most confident first
    print("coverage | selective_error | abstained")
    for cov in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
        k = max(1, int(round(cov * len(P))))
        idx = order[:k]
        err = (dec[idx] != y[idx]).mean()
        print("  %.2f    |     %.3f       |   %d" % (cov, err, len(P) - k))

    print("\n=== abstain band (send to human if confidence < t) ===")
    for t in (0.6, 0.7, 0.8, 0.9):
        keep = conf >= t
        ab = (~keep).mean()
        acc = (dec[keep] == y[keep]).mean() if keep.sum() else float("nan")
        print("  t=%.1f : abstain %.2f  | accuracy on the rest %.3f" % (t, ab, acc))


if __name__ == "__main__":
    main()
