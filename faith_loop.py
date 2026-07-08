"""
Self-improvement loop on a REAL-gap task (faithfulness judging) — de-risks the CARLA loop mechanism.
The judge fails on some sources -> memory INDUCES general judging rules from the failures -> each
rule is VERIFIED on a HELD-OUT SOURCE the rules were not derived from (cross-source transfer).
Verified self-improvement = held-out-SOURCE gain, not induction-source memorization. Genuine gap
(7B faithfulness ~0.69; FinanceBench is the hard held-out). Runs on one 5090.
"""
import os, re
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
IND_SOURCES = ["DROP", "covidQA", "halueval", "RAGTruth"]   # induce rules from failures here
HELDOUT = ["FinanceBench", "pubmedQA"]                        # verify transfer here (unseen sources)
IND_PER, HO_PER = 120, 200

JUDGE_SYS = ("You are a strict faithfulness auditor. Given a passage, question, and answer, decide "
             "if the answer is faithful to the passage (PASS) or a hallucination (ESCALATE).")
JUDGE_TMPL = ("Passage:\n{p}\nQuestion: {q}\nAnswer: {a}\n\nReply with exactly one word: PASS or ESCALATE.")
INDUCE_SYS = ("You improve a faithfulness auditor. Given cases it JUDGED WRONG (with the correct "
              "label), propose THREE DIFFERENT general rules (numbered 1., 2., 3.) that would make it "
              "judge better on UNSEEN cases. General heuristics only, <=2 sentences each.")


def astr(a):
    return ", ".join(map(str, a)) if isinstance(a, list) else str(a)


def judge(model, tok, rows, rule=None):
    sysp = JUDGE_SYS if not rule else (JUDGE_SYS + "\n\nApply this learned rule:\n" + rule)
    tok.padding_side = "left"
    pred, y = [], []
    B = 16
    for i in range(0, len(rows), B):
        ch = rows[i:i + B]
        texts = [tok.apply_chat_template([{"role": "system", "content": sysp},
                 {"role": "user", "content": JUDGE_TMPL.format(p=r["passage"][:2500], q=r["question"], a=astr(r["answer"]))}],
                 tokenize=False, add_generation_prompt=True) for r in ch]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=3072).to("cuda:0")
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=6, do_sample=False, pad_token_id=tok.pad_token_id)
        plen = enc.input_ids.shape[1]
        for j, r in enumerate(ch):
            o = tok.decode(g[j][plen:], skip_special_tokens=True).upper()
            pred.append(1 if "ESCALATE" in o else 0)          # 1 = judged hallucinated
            y.append(1 if r["label"] == "FAIL" else 0)
    tok.padding_side = "right"
    pred, y = np.array(pred), np.array(y)
    return (pred == y).mean(), pred, y


def main():
    d = load_dataset("PatronusAI/HaluBench", split="test")
    by = {}
    for i, s in enumerate(d["source_ds"]):
        by.setdefault(s, []).append(i)
    rng = np.random.RandomState(0)

    def sample(src, k):
        ii = np.array(by[src]); rng.shuffle(ii)
        return [d[int(x)] for x in ii[:k]]

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda:0")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()

    ho = {s: sample(s, HO_PER) for s in HELDOUT}
    base = {s: judge(model, tok, ho[s])[0] for s in HELDOUT}
    print("baseline held-out:", {s: round(base[s], 3) for s in HELDOUT}, flush=True)

    # collect failures on induction sources
    fails = []
    for s in IND_SOURCES:
        rows = sample(s, IND_PER)
        _, pred, y = judge(model, tok, rows)
        for r, p, yy in zip(rows, pred, y):
            if p != yy:
                fails.append((r, "hallucinated" if yy == 1 else "faithful", "hallucinated" if p == 1 else "faithful"))
    print("watcher: %d judging failures across %d induction sources\n" % (len(fails), len(IND_SOURCES)), flush=True)

    block = "\n".join("- Answer: %s | it said: %s | correct: %s" % (astr(r["answer"])[:80], pj, cj) for r, cj, pj in fails[:16])
    ind = tok.apply_chat_template([{"role": "system", "content": INDUCE_SYS}, {"role": "user", "content": "Cases judged WRONG:\n" + block + "\n\nPropose THREE rules."}], tokenize=False, add_generation_prompt=True)
    enc = tok(ind, return_tensors="pt", truncation=True, max_length=3072).to("cuda:0")
    with torch.no_grad():
        g = model.generate(**enc, max_new_tokens=220, do_sample=False, pad_token_id=tok.pad_token_id)
    rules_text = tok.decode(g[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
    cands = [m.group(1).strip() for m in re.finditer(r"(?m)^\s*\d[\.\)]\s*(.+)", rules_text)][:3]

    print("=== candidate judging rules -> VERIFY on held-out sources ===", flush=True)
    best = None
    for k, rule in enumerate(cands, 1):
        accs = {s: judge(model, tok, ho[s], rule=rule)[0] for s in HELDOUT}
        avg_delta = np.mean([accs[s] - base[s] for s in HELDOUT])
        verdict = "ACCEPT (transfers)" if avg_delta > 0.02 else "REJECT"
        print("rule#%d avg held-out delta %+.3f  %s | %s" % (k, avg_delta, verdict, {s: round(accs[s], 3) for s in HELDOUT}), flush=True)
        print("   %s" % rule[:130], flush=True)
        if avg_delta > 0.02 and (best is None or avg_delta > best[0]):
            best = (avg_delta, rule)

    print("\n=== SELECTION ===", flush=True)
    if best:
        print("ADOPTED (+%.3f held-out transfer):\n  %s" % (best[0], best[1]))
        print("-> VERIFIED self-improvement: rule learned from OTHER sources' failures transfers to UNSEEN sources.")
    else:
        print("ADOPTED: none — no rule transferred to held-out sources. Honest null (watcher rejects all).")


if __name__ == "__main__":
    main()
