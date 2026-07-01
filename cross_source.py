"""
Cross-SOURCE generalization (leave-one-source-out): for each source, fine-tune a fresh LoRA judge
on the OTHER 5 sources and test on the held-out source it never saw. This is the honest
"does it generalize to an unseen domain?" test. Compare held-out acc to (zero-shot) and (in-dist).

Reference numbers already on record (same 2400-balanced eval):
  zero-shot per-source: DROP .565 FinanceBench .532 RAGTruth .745 covidQA .688 halueval .787 pubmedQA .807
  in-distribution     : DROP .833 FinanceBench .620 RAGTruth .875 covidQA .728 halueval .960 pubmedQA .877
"""
import os, gc
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
TRAIN_PER, TEST_PER, EPOCHS, BS, ACC, MAXLEN = 400, 400, 2, 4, 4, 1280
JUDGE_SYS = ("You are a strict reliability auditor for AI answers used in high-stakes regulated "
             "domains. You decide whether an answer must be ESCALATED to a human, or can PASS.")
JUDGE_TMPL = (
    "Decide if the ANSWER is faithful to the PASSAGE or is a hallucination.\n"
    "ESCALATE if the answer is not supported by the passage, contradicts it, adds facts beyond it, "
    "or is otherwise likely wrong. PASS only if the answer is fully supported by the passage.\n\n"
    "Passage:\n{p}\nQuestion: {q}\nAnswer: {a}\n\nReply with exactly one word: PASS or ESCALATE.")


def astr(a):
    return ", ".join(map(str, a)) if isinstance(a, list) else str(a)


def prompt_of(tok, ex):
    user = JUDGE_TMPL.format(p=ex["passage"][:3000], q=ex["question"], a=astr(ex["answer"]))
    return tok.apply_chat_template([{"role": "system", "content": JUDGE_SYS}, {"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True)


def run_fold(tok, d, by, held):
    rng = np.random.RandomState(0)
    train_idx = []
    for s, ii in by.items():
        if s == held:
            continue
        ii = np.array(ii); rng.shuffle(ii); train_idx += ii[:TRAIN_PER].tolist()
    ii = np.array(by[held]); rng.shuffle(ii); test_idx = ii[:TEST_PER].tolist()

    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda:0")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0")
    model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))

    enc = []
    for i in train_idx:
        ex = d[int(i)]
        pids = tok(prompt_of(tok, ex), add_special_tokens=False).input_ids
        tids = tok("ESCALATE" if ex["label"] == "FAIL" else "PASS", add_special_tokens=False).input_ids + [tok.eos_token_id]
        enc.append(((pids + tids)[-MAXLEN:], ([-100] * len(pids) + tids)[-MAXLEN:]))

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    model.train()
    for ep in range(EPOCHS):
        order = np.random.RandomState(ep).permutation(len(enc)); opt.zero_grad()
        for step, b in enumerate(range(0, len(order), BS)):
            batch = [enc[order[k]] for k in range(b, min(b + BS, len(order)))]
            ml = max(len(x[0]) for x in batch)
            ii = torch.full((len(batch), ml), tok.pad_token_id, dtype=torch.long)
            lb = torch.full((len(batch), ml), -100, dtype=torch.long)
            am = torch.zeros((len(batch), ml), dtype=torch.long)
            for j, (a, c) in enumerate(batch):
                ii[j, :len(a)] = torch.tensor(a); lb[j, :len(c)] = torch.tensor(c); am[j, :len(a)] = 1
            out = model(input_ids=ii.cuda(), attention_mask=am.cuda(), labels=lb.cuda())
            (out.loss / ACC).backward()
            if (step + 1) % ACC == 0:
                opt.step(); opt.zero_grad()

    model.eval(); tok.padding_side = "left"
    esc, y = [], []
    B = 24
    for i in range(0, len(test_idx), B):
        exs = [d[int(k)] for k in test_idx[i:i + B]]
        texts = [prompt_of(tok, e) for e in exs]
        e2 = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=3072).to("cuda:0")
        with torch.no_grad():
            g = model.generate(**e2, max_new_tokens=8, do_sample=False, pad_token_id=tok.pad_token_id)
        plen = e2.input_ids.shape[1]
        for j, e in enumerate(exs):
            esc.append(1 if "ESCALATE" in tok.decode(g[j][plen:], skip_special_tokens=True).upper() else 0)
            y.append(1 if e["label"] == "FAIL" else 0)
    esc, y = np.array(esc), np.array(y)
    tp = ((esc == 1) & (y == 1)).sum(); fp = ((esc == 1) & (y == 0)).sum(); fn = ((esc == 0) & (y == 1)).sum()
    pr = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
    acc = (esc == y).mean(); f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    tok.padding_side = "right"
    del model; gc.collect(); torch.cuda.empty_cache()
    return acc, f1


def main():
    d = load_dataset("PatronusAI/HaluBench", split="test")
    by = {}
    for i, s in enumerate(d["source_ds"]):
        by.setdefault(s, []).append(i)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    ZS = {"DROP": .565, "FinanceBench": .532, "RAGTruth": .745, "covidQA": .688, "halueval": .787, "pubmedQA": .807}
    ID = {"DROP": .833, "FinanceBench": .620, "RAGTruth": .875, "covidQA": .728, "halueval": .960, "pubmedQA": .877}
    print("held-out source | cross-source acc (trained on OTHER 5) | zero-shot | in-dist", flush=True)
    for held in sorted(by):
        acc, f1 = run_fold(tok, d, by, held)
        print("  %-14s cross=%.3f (F1 %.3f)   zs=%.3f  in-dist=%.3f" % (held, acc, f1, ZS.get(held, 0), ID.get(held, 0)), flush=True)


if __name__ == "__main__":
    main()
