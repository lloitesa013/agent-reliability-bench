"""
Verified self-improvement at the FINE-TUNING level (verifier as a model-selection gate).
Generate several candidate fine-tuned judges with DIFFERENT training-data compositions, then let the
VERIFIER pick by HELD-OUT-source accuracy (transfer) vs the NAIVE pick by in-distribution accuracy.
Question: does verified selection choose a better-generalizing fine-tune than naive selection?
Held-out sources (never trained on): FinanceBench, pubmedQA. Pool sources: DROP, covidQA, halueval, RAGTruth.
"""
import os, re, gc
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
HELDOUT = ["FinanceBench", "pubmedQA"]
POOL = ["DROP", "covidQA", "halueval", "RAGTruth"]
JUDGE_SYS = ("You are a strict faithfulness auditor. Given a passage, question, and answer, decide if "
             "the answer is faithful to the passage (PASS) or a hallucination (ESCALATE).")
JUDGE_TMPL = "Passage:\n{p}\nQuestion: {q}\nAnswer: {a}\n\nReply with exactly one word: PASS or ESCALATE."

CANDIDATES = {
    "broad4  (250 x4)": {"DROP": 250, "covidQA": 250, "halueval": 250, "RAGTruth": 250},
    "single  (halu1000)": {"halueval": 1000},
    "pair    (DROP+RAG)": {"DROP": 500, "RAGTruth": 500},
}


def astr(a):
    return ", ".join(map(str, a)) if isinstance(a, list) else str(a)


def prompt(tok, r):
    return tok.apply_chat_template([{"role": "system", "content": JUDGE_SYS},
        {"role": "user", "content": JUDGE_TMPL.format(p=r["passage"][:2500], q=r["question"], a=astr(r["answer"]))}],
        tokenize=False, add_generation_prompt=True)


def evaluate(model, tok, rows):
    tok.padding_side = "left"; pred = []; y = []
    for i in range(0, len(rows), 16):
        ch = rows[i:i + 16]
        enc = tok([prompt(tok, r) for r in ch], return_tensors="pt", padding=True, truncation=True, max_length=3072).to("cuda:0")
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=6, do_sample=False, pad_token_id=tok.pad_token_id)
        pl = enc.input_ids.shape[1]
        for j, r in enumerate(ch):
            o = tok.decode(g[j][pl:], skip_special_tokens=True).upper()
            pred.append(1 if "ESCALATE" in o else 0); y.append(1 if r["label"] == "FAIL" else 0)
    tok.padding_side = "right"
    return float(np.mean(np.array(pred) == np.array(y)))


def main():
    d = load_dataset("PatronusAI/HaluBench", split="test")
    by = {}
    for i, s in enumerate(d["source_ds"]):
        by.setdefault(s, []).append(i)
    rng = np.random.RandomState(0)
    idx = {s: rng.permutation(by[s]) for s in by}
    sizes = {s: len(idx[s]) for s in by}

    def take(s, a, b):
        return [d[int(x)] for x in idx[s][a:b]]

    heldout_rows = sum([take(s, 0, 200) for s in HELDOUT], [])          # verified transfer set (sources never trained)

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    results = {}
    for name, comp in CANDIDATES.items():
        print("\n### training candidate: %s (%s)" % (name, comp), flush=True)
        try:
            model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda:0")
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0")
        model.gradient_checkpointing_enable(); model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
        enc = []
        for s, n in comp.items():
            for r in take(s, 0, n):
                pids = tok(prompt(tok, r), add_special_tokens=False).input_ids
                tids = tok("ESCALATE" if r["label"] == "FAIL" else "PASS", add_special_tokens=False).input_ids + [tok.eos_token_id]
                enc.append(((pids + tids)[-1024:], ([-100] * len(pids) + tids)[-1024:]))
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
        model.train()
        for ep in range(2):
            order = np.random.RandomState(ep).permutation(len(enc)); opt.zero_grad(); last = 0.0
            for step, b in enumerate(range(0, len(order), 4)):
                batch = [enc[order[k]] for k in range(b, min(b + 4, len(order)))]
                ml = max(len(x[0]) for x in batch)
                ii = torch.full((len(batch), ml), tok.pad_token_id, dtype=torch.long); lb = torch.full((len(batch), ml), -100, dtype=torch.long); am = torch.zeros((len(batch), ml), dtype=torch.long)
                for j, (a, c) in enumerate(batch):
                    ii[j, :len(a)] = torch.tensor(a); lb[j, :len(c)] = torch.tensor(c); am[j, :len(a)] = 1
                out = model(input_ids=ii.cuda(), attention_mask=am.cuda(), labels=lb.cuda()); (out.loss / 4).backward(); last = float(out.loss)
                if (step + 1) % 4 == 0:
                    opt.step(); opt.zero_grad()
            print("   epoch %d loss=%.3f" % (ep, last), flush=True)
        model.eval()
        # in-dist = held-out TAIL of THIS candidate's own training sources (disjoint from train [0:n], n<=1000)
        ind_rows = sum([take(s, sizes[s] - 150, sizes[s]) for s in comp.keys()], [])
        ver = evaluate(model, tok, heldout_rows)      # held-out SOURCES (finance/pubmed) = verified transfer
        ind = evaluate(model, tok, ind_rows)          # own-domain held-out = naive selection signal
        results[name] = (ver, ind)
        print("   -> verified(held-out src)=%.3f  in-dist(pool)=%.3f" % (ver, ind), flush=True)
        del model; gc.collect(); torch.cuda.empty_cache()

    print("\n=== CANDIDATE SELECTION ===", flush=True)
    print("candidate            | verified(held-out) | in-dist(naive)")
    for n, (v, i) in results.items():
        print("%-20s |      %.3f        |     %.3f" % (n, v, i), flush=True)
    ver_pick = max(results, key=lambda k: results[k][0])
    naive_pick = max(results, key=lambda k: results[k][1])
    print("\nVERIFIER picks (best held-out): %s  (held-out %.3f)" % (ver_pick, results[ver_pick][0]))
    print("NAIVE picks (best in-dist)    : %s  (held-out %.3f)" % (naive_pick, results[naive_pick][0]))
    if ver_pick != naive_pick:
        print(">>> They DIFFER: verified selection would deploy a candidate that generalizes %.3f on held-out;" % results[ver_pick][0])
        print(">>> naive would deploy one that only scores %.3f on held-out = verified selection is better." % results[naive_pick][0])
    else:
        print(">>> Same pick this run (candidates didn't diverge enough); verified selection still the safe gate.")


if __name__ == "__main__":
    main()
