"""
Fine-tune a LoRA judge on HaluBench (train) and evaluate on the SAME 2400-example balanced test set
used for the zero-shot judge -> direct "is fine-tuning worth it?" comparison (zero-shot was 0.688).
7B bf16 + LoRA fits one RTX 5090 (no 4-bit). Reports accuracy/F1 overall + per source + risk-coverage.
"""
import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
TEST_PER, TRAIN_PER = 400, 500
EPOCHS, BS, ACC, MAXLEN = 2, 4, 4, 1280

JUDGE_SYS = ("You are a strict reliability auditor for AI answers used in high-stakes regulated "
             "domains. You decide whether an answer must be ESCALATED to a human, or can PASS.")
JUDGE_TMPL = (
    "Decide if the ANSWER is faithful to the PASSAGE or is a hallucination.\n"
    "ESCALATE if the answer is not supported by the passage, contradicts it, adds facts beyond it, "
    "or is otherwise likely wrong. PASS only if the answer is fully supported by the passage.\n\n"
    "Passage:\n{p}\nQuestion: {q}\nAnswer: {a}\n\nReply with exactly one word: PASS or ESCALATE.")


def astr(a):
    return ", ".join(map(str, a)) if isinstance(a, list) else str(a)


def main():
    d = load_dataset("PatronusAI/HaluBench", split="test")
    rng = np.random.RandomState(0)
    by = {}
    for i, s in enumerate(d["source_ds"]):
        by.setdefault(s, []).append(i)
    test_idx, train_idx = [], []
    for s, ii in by.items():
        ii = np.array(ii); rng.shuffle(ii)
        test_idx += ii[:TEST_PER].tolist()
        train_idx += ii[TEST_PER:TEST_PER + TRAIN_PER].tolist()
    print("train n=%d  test n=%d (disjoint)" % (len(train_idx), len(test_idx)), flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda:0")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.print_trainable_parameters()

    def build(i):
        ex = d[int(i)]
        user = JUDGE_TMPL.format(p=ex["passage"][:3000], q=ex["question"], a=astr(ex["answer"]))
        prompt = tok.apply_chat_template([{"role": "system", "content": JUDGE_SYS}, {"role": "user", "content": user}],
                                         tokenize=False, add_generation_prompt=True)
        target = "ESCALATE" if ex["label"] == "FAIL" else "PASS"
        pids = tok(prompt, add_special_tokens=False).input_ids
        tids = tok(target, add_special_tokens=False).input_ids + [tok.eos_token_id]
        ids = (pids + tids)[-MAXLEN:]
        lab = ([-100] * len(pids) + tids)[-MAXLEN:]
        return ids, lab
    train_enc = [build(i) for i in train_idx]

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    model.train()
    for ep in range(EPOCHS):
        order = np.random.RandomState(ep).permutation(len(train_enc))
        opt.zero_grad(); last = 0.0
        for step, b in enumerate(range(0, len(order), BS)):
            batch = [train_enc[order[k]] for k in range(b, min(b + BS, len(order)))]
            ml = max(len(x[0]) for x in batch)
            ii = torch.full((len(batch), ml), tok.pad_token_id, dtype=torch.long)
            lb = torch.full((len(batch), ml), -100, dtype=torch.long)
            am = torch.zeros((len(batch), ml), dtype=torch.long)
            for j, (ids, lab) in enumerate(batch):
                ii[j, :len(ids)] = torch.tensor(ids); lb[j, :len(lab)] = torch.tensor(lab); am[j, :len(ids)] = 1
            out = model(input_ids=ii.cuda(), attention_mask=am.cuda(), labels=lb.cuda())
            (out.loss / ACC).backward(); last = float(out.loss)
            if (step + 1) % ACC == 0:
                opt.step(); opt.zero_grad()
        print("epoch %d  loss=%.4f" % (ep, last), flush=True)

    # ---- eval on the identical 2400 test ----
    model.eval(); tok.padding_side = "left"

    def fids(w):
        s = set()
        for x in (w, " " + w):
            t = tok.encode(x, add_special_tokens=False)
            if t:
                s.add(t[0])
        return list(s)
    esc_ids, pass_ids = fids("ESCALATE"), fids("PASS")

    esc, Z, y, src = [], [], [], []
    B = 24
    for i in range(0, len(test_idx), B):
        sub = test_idx[i:i + B]
        exs = [d[int(k)] for k in sub]
        texts = [tok.apply_chat_template(
            [{"role": "system", "content": JUDGE_SYS},
             {"role": "user", "content": JUDGE_TMPL.format(p=e["passage"][:3000], q=e["question"], a=astr(e["answer"]))}],
            tokenize=False, add_generation_prompt=True) for e in exs]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=3072).to("cuda:0")
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=8, do_sample=False, output_scores=True,
                               return_dict_in_generate=True, pad_token_id=tok.pad_token_id)
        plen = enc.input_ids.shape[1]; s0 = g.scores[0].float()
        for j, e in enumerate(exs):
            esc.append(1 if "ESCALATE" in tok.decode(g.sequences[j][plen:], skip_special_tokens=True).upper() else 0)
            Z.append((s0[j, esc_ids].max() - s0[j, pass_ids].max()).item())
            y.append(1 if e["label"] == "FAIL" else 0); src.append(e["source_ds"])
    esc, Z, y, src = np.array(esc), np.array(Z), np.array(y), np.array(src)

    def prf(e, yy):
        tp = ((e == 1) & (yy == 1)).sum(); fp = ((e == 1) & (yy == 0)).sum(); fn = ((e == 0) & (yy == 1)).sum()
        pr = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
        return (e == yy).mean(), (2 * pr * rc / (pr + rc) if pr + rc else 0.0)

    a, f = prf(esc, y)
    print("\n=== FINE-TUNED LoRA judge on HaluBench (same 2400 test) ===")
    print("overall: accuracy=%.3f  F1=%.3f   (zero-shot was 0.688 / 0.685)" % (a, f))
    print("per source:")
    for s in sorted(set(src.tolist())):
        m = src == s; aa, ff = prf(esc[m], y[m])
        print("  %-14s acc=%.3f  F1=%.3f" % (s, aa, ff))
    conf = np.abs(Z); order = np.argsort(-conf); dec = (Z > 0).astype(int)
    print("risk-coverage: ", end="")
    for cov in (1.0, 0.8, 0.6, 0.5):
        k = max(1, int(round(cov * len(Z)))); ii = order[:k]
        print("cov%.1f=%.3f " % (cov, (dec[ii] == y[ii]).mean()), end="")
    print()


if __name__ == "__main__":
    main()
