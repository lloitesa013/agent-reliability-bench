"""
Phase-0 v3 (clean) — self-improvement by POLICY FINE-TUNING.
Harder 3-hop tasks calibrated so the working-allowed baseline leaves room (~0.5). Agent fails ->
LoRA fine-tune the policy on corrected step-by-step solutions of the FAILURES -> re-measure on a
HELD-OUT set with DISJOINT params. Verified improvement = held-out transfer (not seen memorization).
"""
import os, re
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
EVAL_SYS = "Solve the problem. End with a line exactly: ANSWER: <value>"


def tasks(P):
    T = []
    for n, m in P["fees"]:
        g = 39 * n + 30 * m - 50
        T.append((f"A customer had {n} late payments at $39 each and {m} returned-item fees at $30 each, then received a one-time $50 courtesy credit. Net total in fees?", g,
                  f"{n} x $39 = ${39*n}. {m} x $30 = ${30*m}. Subtotal = ${39*n+30*m}. Minus $50 credit = ${g}.\nANSWER: {g}"))
    for n, d, p in P["dose"]:
        g = 3000 - n * d - p
        T.append((f"Daily acetaminophen limit is 3,000 mg. A patient took {n} doses of {d} mg pills plus {p} mg of liquid. How many mg remain within the daily limit?", g,
                  f"Pills: {n} x {d} = {n*d} mg. Plus liquid {p} mg = {n*d+p} mg used. Remaining: 3000 - {n*d+p} = {g} mg.\nANSWER: {g}"))
    for a, b, c in P["ded"]:
        g = 1500 - a - b - c
        T.append((f"Annual deductible is $1,500. A patient paid ${a}, then ${b}, then ${c}. How much of the deductible remains?", g,
                  f"Total paid: {a} + {b} + {c} = {a+b+c}. Remaining: 1500 - {a+b+c} = {g}.\nANSWER: {g}"))
    return T


TRAIN = tasks({"fees": [(2, 3), (4, 2), (3, 5), (5, 1), (6, 4), (3, 2), (4, 5), (7, 3), (5, 6), (8, 1), (2, 7), (6, 3), (4, 4), (3, 6)],
               "dose": [(2, 500, 300), (3, 400, 200), (4, 300, 150), (2, 700, 100), (3, 500, 250), (4, 250, 300), (5, 200, 400), (2, 600, 350), (3, 350, 450), (2, 450, 600), (4, 350, 200), (3, 300, 500), (2, 550, 250), (5, 300, 150)],
               "ded": [(200, 300, 150), (400, 250, 300), (300, 200, 450), (150, 500, 250), (350, 300, 200), (500, 150, 400), (250, 450, 300), (300, 350, 250), (200, 600, 150), (450, 200, 350), (150, 300, 500), (400, 400, 200), (250, 250, 600), (350, 450, 150)]})
TEST = tasks({"fees": [(6, 2), (3, 4), (7, 1), (5, 3), (8, 2), (9, 1), (2, 6), (6, 5)],
              "dose": [(3, 450, 200), (2, 650, 300), (4, 400, 100), (3, 550, 150), (2, 750, 250), (5, 250, 300), (3, 400, 600), (4, 300, 350)],
              "ded": [(300, 450, 200), (500, 200, 350), (150, 400, 550), (400, 300, 300), (250, 550, 200), (350, 250, 450), (200, 650, 300), (450, 350, 250)]})


def num(s):
    m = re.findall(r"-?\d[\d,]*\.?\d*", s.replace("$", ""))
    return float(m[-1].replace(",", "")) if m else None


def ptext(tok, q):
    return tok.apply_chat_template([{"role": "system", "content": EVAL_SYS}, {"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)


def gen(model, tok, qs):
    tok.padding_side = "left"; outs = []
    for i in range(0, len(qs), 16):
        enc = tok([ptext(tok, q) for q in qs[i:i + 16]], return_tensors="pt", padding=True, truncation=True, max_length=1024).to("cuda:0")
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=220, do_sample=False, pad_token_id=tok.pad_token_id)
        plen = enc.input_ids.shape[1]
        outs += [tok.decode(g[j][plen:], skip_special_tokens=True) for j in range(enc.input_ids.shape[0])]
    tok.padding_side = "right"
    return outs


def acc_and_fails(model, tok, tset):
    outs = gen(model, tok, [q for q, _, _ in tset])
    ok, fails = [], []
    for (q, gold, sol), o in zip(tset, outs):
        mm = re.search(r"ANSWER:\s*(.+)", o, re.IGNORECASE); a = num(mm.group(1) if mm else o)
        good = a is not None and abs(a - float(gold)) < 0.5
        ok.append(good)
        if not good:
            fails.append((q, gold, sol))
    return float(np.mean(ok)), fails


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda:0")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0")

    test0, _ = acc_and_fails(model, tok, TEST)
    train0, fails = acc_and_fails(model, tok, TRAIN)
    print("baseline: TEST(held-out)=%.3f  TRAIN=%.3f | %d/%d train failures\n" % (test0, train0, len(fails), len(TRAIN)), flush=True)
    if len(fails) < 6:
        print("too few failures — baseline near ceiling; skip."); return

    model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    enc = []
    for q, gold, sol in fails:
        pids = tok(ptext(tok, q), add_special_tokens=False).input_ids
        tids = tok(sol, add_special_tokens=False).input_ids + [tok.eos_token_id]
        enc.append((pids + tids, [-100] * len(pids) + tids))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    model.train()
    for ep in range(6):
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
        print("  ft epoch %d loss=%.4f" % (ep, last), flush=True)

    model.eval()
    test1, _ = acc_and_fails(model, tok, TEST); train1, _ = acc_and_fails(model, tok, TRAIN)
    print("\n=== after POLICY FINE-TUNING on corrected solutions ===", flush=True)
    print("TEST (held-out): %.3f -> %.3f   (VERIFIED transfer delta = %+.3f)" % (test0, test1, test1 - test0))
    print("TRAIN (seen)   : %.3f -> %.3f   (delta = %+.3f)" % (train0, train1, train1 - train0))
    print("\nVERDICT:", flush=True)
    if test1 - test0 > 0.05:
        print("  VERIFIED self-improvement: fine-tuning TRANSFERS to unseen params (+%.3f held-out). Watcher ACCEPTS." % (test1 - test0))
    elif train1 - train0 > 0.05:
        print("  overfit: helped seen only -> watcher REJECTS.")
    else:
        print("  honest null: fine-tuning did not transfer.")


if __name__ == "__main__":
    main()
