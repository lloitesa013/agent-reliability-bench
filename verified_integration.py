"""
BRICK 4 - Verified Integration (retention). Adopt a verified improvement WITHOUT regressing old capability.
The verifier is extended from 1D (does the new target improve?) to 2D (new improves AND no old source
regresses). X-MoD warns naive correction-retraining backfires; here we MEASURE it on the judge:
  base            : zero-shot judge, per-source capability profile
  candidate NAIVE : fine-tune on the NEW weak target only        -> expect new UP, old DOWN (regression)
  candidate RETAIN: fine-tune on NEW + replay from OLD sources    -> expect new UP, old kept
The 2D verifier accepts iff (new held-out UP by margin) AND (no old held-out DOWN beyond noise).
Question: does naive adoption regress old capability, and does the 2D verifier catch it + accept retention?
"""
import os, gc
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
OLD = ["DROP", "covidQA", "halueval", "RAGTruth"]     # capabilities the base judge already has
NEW = "FinanceBench"                                    # weak target we want to improve
GAIN_MARGIN = 0.05                                      # new must rise at least this much
REG_MARGIN = 0.03                                       # no old source may fall more than this
JUDGE_SYS = ("You are a strict faithfulness auditor. Given a passage, question, and answer, decide if "
             "the answer is faithful to the passage (PASS) or a hallucination (ESCALATE).")
JUDGE_TMPL = "Passage:\n{p}\nQuestion: {q}\nAnswer: {a}\n\nReply with exactly one word: PASS or ESCALATE."


def astr(a):
    return ", ".join(map(str, a)) if isinstance(a, list) else str(a)


def prompt(tok, r):
    return tok.apply_chat_template([{"role": "system", "content": JUDGE_SYS},
        {"role": "user", "content": JUDGE_TMPL.format(p=r["passage"][:2500], q=r["question"], a=astr(r["answer"]))}],
        tokenize=False, add_generation_prompt=True)


def eval_src(model, tok, rows):
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


def load_base(tok):
    try:
        m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda:0")
    except TypeError:
        m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0")
    return m


def finetune(tok, comp, take):
    m = load_base(tok)
    m.gradient_checkpointing_enable(); m.enable_input_require_grads()
    m = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    enc = []
    for s, n in comp.items():
        for r in take(s, 0, n):
            pids = tok(prompt(tok, r), add_special_tokens=False).input_ids
            tids = tok("ESCALATE" if r["label"] == "FAIL" else "PASS", add_special_tokens=False).input_ids + [tok.eos_token_id]
            enc.append(((pids + tids)[-1024:], ([-100] * len(pids) + tids)[-1024:]))
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=2e-4)
    m.train()
    for ep in range(2):
        order = np.random.RandomState(ep).permutation(len(enc)); opt.zero_grad(); last = 0.0
        for step, b in enumerate(range(0, len(order), 4)):
            batch = [enc[order[k]] for k in range(b, min(b + 4, len(order)))]
            ml = max(len(x[0]) for x in batch)
            ii = torch.full((len(batch), ml), tok.pad_token_id, dtype=torch.long); lb = torch.full((len(batch), ml), -100, dtype=torch.long); am = torch.zeros((len(batch), ml), dtype=torch.long)
            for j, (a, c) in enumerate(batch):
                ii[j, :len(a)] = torch.tensor(a); lb[j, :len(c)] = torch.tensor(c); am[j, :len(a)] = 1
            out = m(input_ids=ii.cuda(), attention_mask=am.cuda(), labels=lb.cuda()); (out.loss / 4).backward(); last = float(out.loss)
            if (step + 1) % 4 == 0:
                opt.step(); opt.zero_grad()
        print("     epoch %d loss=%.3f" % (ep, last), flush=True)
    m.eval()
    return m


def profile(model, tok, evalsets):
    return {s: eval_src(model, tok, rows) for s, rows in evalsets.items()}


def verdict(base_p, cand_p):
    dnew = cand_p[NEW] - base_p[NEW]
    dold = {s: cand_p[s] - base_p[s] for s in OLD}
    worst_old = min(dold.values())
    ok = (dnew >= GAIN_MARGIN) and (worst_old >= -REG_MARGIN)
    return dnew, dold, worst_old, ok


def main():
    d = load_dataset("PatronusAI/HaluBench", split="test")
    by = {}
    for i, s in enumerate(d["source_ds"]):
        by.setdefault(s, []).append(i)
    rng = np.random.RandomState(0)
    idx = {s: rng.permutation(by[s]) for s in by}

    def take(s, a, b):
        return [d[int(x)] for x in idx[s][a:b]]

    # held-out eval sets (disjoint from any training slice which uses [0:n], n<=400; eval uses [800:1000])
    evalsets = {s: take(s, 800, 1000) for s in OLD + [NEW]}
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    print("== BASE profile (zero-shot judge) ==", flush=True)
    base = load_base(tok); base_p = profile(base, tok, evalsets)
    for s in OLD + [NEW]:
        print("   %-12s %.3f" % (s, base_p[s]), flush=True)
    del base; gc.collect(); torch.cuda.empty_cache()

    candidates = {
        "NAIVE  (finance400)": {NEW: 400},
        "RETAIN (fin400+old100x4)": {NEW: 400, "DROP": 100, "covidQA": 100, "halueval": 100, "RAGTruth": 100},
    }
    profiles = {}
    for name, comp in candidates.items():
        print("\n== candidate %s (%s) ==" % (name, comp), flush=True)
        m = finetune(tok, comp, take); profiles[name] = profile(m, tok, evalsets)
        for s in OLD + [NEW]:
            print("   %-12s %.3f  (Δ%+.3f)" % (s, profiles[name][s], profiles[name][s] - base_p[s]), flush=True)
        del m; gc.collect(); torch.cuda.empty_cache()

    print("\n=== 2D VERIFIED INTEGRATION GATE (gain>=%.2f AND no old drop>%.2f) ===" % (GAIN_MARGIN, REG_MARGIN), flush=True)
    for name in candidates:
        dnew, dold, worst_old, ok = verdict(base_p, profiles[name])
        print("%-26s Δnew=%+.3f  worst_old=%+.3f  -> %s" % (name, dnew, worst_old, "ACCEPT" if ok else "REJECT"), flush=True)
    n_ok = verdict(base_p, profiles["NAIVE  (finance400)"])[3]
    r_ok = verdict(base_p, profiles["RETAIN (fin400+old100x4)"])[3]
    print("\n--- interpretation ---", flush=True)
    if (not n_ok) and r_ok:
        print(">>> BRICK 4 DEMONSTRATED: naive adoption regresses old capability (2D verifier REJECTS);", flush=True)
        print(">>> retention keeps old + gains new (2D verifier ACCEPTS). The verifier's retention check is", flush=True)
        print(">>> what makes verified INTEGRATION possible - exactly the X-MoD naive-correction backfire, caught.", flush=True)
    elif n_ok and r_ok:
        print(">>> honest: naive did NOT regress here (no forgetting on this task) - retention not needed;", flush=True)
        print(">>> the 2D gate still correctly accepts both. Retention problem is task/scale dependent.", flush=True)
    else:
        print(">>> honest: retention candidate did not clear the gate either - the improvement itself doesn't", flush=True)
        print(">>> transfer to held-out finance (consistent with cross-source non-transfer); integration blocked upstream.", flush=True)


if __name__ == "__main__":
    main()
