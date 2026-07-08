"""
Phase-0 closed learning loop (VERIFIED self-improvement).
Loop: agent attempts multi-step tasks -> watcher collects failures -> memory INDUCES a general
rule -> rule is enforced (prepended) -> agent retries -> improvement measured on a HELD-OUT set
with DISJOINT parameters. Verified improvement = held-out gain (transfer), not seen-set gain
(memorization). Reuses the 7B agent; runs on one 5090.
"""
import os, re, random
import numpy as np
os.environ.setdefault("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
from real_rag import LLM

AGENT_SYS = ("You are a precise assistant for numeric policy questions. Solve the problem, showing "
             "brief working, then end with a line exactly: ANSWER: <value>")
INDUCE_SYS = ("You improve an assistant. Given problems it got WRONG (with the correct answers), "
              "write exactly ONE general rule (<=2 sentences) that would prevent these mistakes on "
              "UNSEEN problems. The rule must be general method advice, NOT about specific numbers.")


def tasks(params):
    T = []
    for n in params["late"]:
        T.append((f"A customer had {n} late credit-card payments at $39 each. What is the total in "
                  f"late fees?", 39 * n))
    for u in params["atm"]:
        T.append((f"The daily ATM limit is $1,000. A customer already withdrew ${u} today. How much "
                  f"more can they withdraw today?", 1000 - u))
    for n, d in params["dose"]:
        T.append((f"The maximum acetaminophen dose is 3,000 mg per day. A patient took {n} doses of "
                  f"{d} mg today. How many mg remain within the daily limit?", 3000 - n * d))
    for p in params["ded"]:
        T.append((f"The annual deductible is $1,500. A patient has paid ${p} so far. How much of the "
                  f"deductible remains?", 1500 - p))
    return T


TRAIN = tasks({"late": [2, 3, 4, 5], "atm": [150, 300, 450, 600],
               "dose": [(3, 650), (4, 500), (2, 900), (3, 550)], "ded": [300, 600, 900]})
TEST = tasks({"late": [6, 7, 8, 9], "atm": [250, 500, 700, 850],   # DISJOINT params from TRAIN
              "dose": [(3, 750), (5, 400), (4, 650), (2, 800)], "ded": [450, 1100, 750]})


def norm(x):
    return str(x).lower().replace(",", "").replace("$", "").strip()


def extract(out):
    m = re.search(r"ANSWER:\s*(.+)", out, re.IGNORECASE)
    return (m.group(1) if m else out).strip()


def run(llm, rule, tset):
    sys = AGENT_SYS if not rule else (AGENT_SYS + "\n\nAlways follow this learned rule:\n" + rule)
    outs = llm.chat_batch(sys, [q for q, _ in tset], max_new_tokens=220, batch_size=16)
    res = []
    for (q, gold), o in zip(tset, outs):
        a = extract(o)
        res.append((q, gold, a, norm(gold) in norm(a)))
    acc = np.mean([r[3] for r in res])
    return acc, res


def main():
    llm = LLM(device="cuda:0")
    print("TRAIN n=%d  TEST n=%d (disjoint params)\n" % (len(TRAIN), len(TEST)), flush=True)

    # baseline (no rule)
    test_acc0, _ = run(llm, None, TEST)
    train_acc0, train_res0 = run(llm, None, TRAIN)
    print("baseline: TEST(held-out)=%.3f  TRAIN=%.3f" % (test_acc0, train_acc0), flush=True)

    # watcher: collect TRAIN failures
    fails = [(q, gold, a) for (q, gold, a, ok) in train_res0 if not ok]
    print("watcher: %d/%d train failures observed" % (len(fails), len(TRAIN)), flush=True)
    if not fails:
        print("no failures to learn from — task too easy for this base model."); return

    # memory: induce ONE general rule from failures
    block = "\n".join("- Q: %s\n  wrong answer: %s\n  correct: %s" % (q, a, gold) for q, gold, a in fails[:12])
    rule = llm.chat_batch(INDUCE_SYS, ["Problems it got WRONG:\n" + block + "\n\nWrite ONE general rule."],
                          max_new_tokens=80, batch_size=1)[0].strip()
    print("\nINDUCED RULE:\n  %s\n" % rule, flush=True)

    # enforce rule, re-measure
    test_acc1, _ = run(llm, rule, TEST)
    train_acc1, _ = run(llm, rule, TRAIN)

    print("=== after enforcing the learned rule ===", flush=True)
    print("TEST (held-out): %.3f -> %.3f   (VERIFIED transfer delta = %+.3f)" % (test_acc0, test_acc1, test_acc1 - test_acc0))
    print("TRAIN (seen)   : %.3f -> %.3f   (delta = %+.3f)" % (train_acc0, train_acc1, train_acc1 - train_acc0))
    print("\nVERDICT:", flush=True)
    if test_acc1 - test_acc0 > 0.02:
        print("  VERIFIED self-improvement: the rule TRANSFERS to unseen params (+%.3f on held-out)." % (test_acc1 - test_acc0))
    elif train_acc1 - train_acc0 > 0.05:
        print("  FAKE/overfit: helped seen but NOT held-out — watcher would REJECT this 'improvement'.")
    else:
        print("  no real change (rule didn't help) — honest null.")


if __name__ == "__main__":
    main()
