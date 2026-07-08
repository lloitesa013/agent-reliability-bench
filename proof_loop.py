"""
THE verified-self-improvement proof (both directions of the verifier).
Hidden-convention task: overtime starts after 6h (the model assumes the usual 8h -> fails).
Three candidate self-improvements:
  correct  = the general rule (overtime after 6h) -> fixes SEEN and HELD-OUT (transfers) = REAL
  overfit  = a lookup of the TRAIN h-values only -> fixes SEEN, not HELD-OUT = FAKE (seen-only)
  spurious = a wrong rule -> fixes neither
Verifier accepts iff HELD-OUT (disjoint h) accuracy rises. Proof = verifier ACCEPTS correct,
REJECTS overfit+spurious, while the naive "measure on SEEN" baseline is FOOLED by overfit.
"""
import os, re
import numpy as np
os.environ.setdefault("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
from real_rag import LLM

AGENT_SYS = ("You are a precise HR assistant. Answer with only the number of overtime hours. "
             "End with a line exactly: ANSWER: <number>")
TRAIN_H = [7, 8, 9, 10, 11]
TEST_H = [12, 13, 14, 15, 16]   # disjoint from train


def tasks(hs):
    return [(f"An employee worked {h} hours in a single day. How many of those were overtime hours?", max(0, h - 6)) for h in hs]


TRAIN, TEST = tasks(TRAIN_H), tasks(TEST_H)

CANDIDATES = [
    ("correct  (general rule)", "At this company, overtime is any time worked beyond 6 hours in a day (not 8)."),
    ("overfit  (train lookup)", "Overtime reference table: 7h->1, 8h->2, 9h->3, 10h->4, 11h->5."),
    ("spurious (wrong rule)",   "Overtime equals the total hours worked minus twelve."),
]


def num(s):
    m = re.findall(r"-?\d+", s)
    return int(m[-1]) if m else None


def acc(llm, rule, tset):
    sysp = AGENT_SYS if not rule else (AGENT_SYS + "\n\nApply this company rule:\n" + rule)
    outs = llm.chat_batch(sysp, [q for q, _ in tset], max_new_tokens=64, batch_size=16)
    ok = []
    for (q, gold), o in zip(tset, outs):
        m = re.search(r"ANSWER:\s*(-?\d+)", o, re.IGNORECASE)
        a = int(m.group(1)) if m else num(o)
        ok.append(a is not None and a == gold)
    return float(np.mean(ok))


def main():
    llm = LLM(device="cuda:0")
    base_test = acc(llm, None, TEST)
    base_train = acc(llm, None, TRAIN)
    print("baseline: TEST(held-out)=%.2f  TRAIN(seen)=%.2f  (model assumes the usual 8h -> fails)\n" % (base_test, base_train), flush=True)

    print("candidate            | seen Δ | held-out Δ | VERIFIER (held-out) | naive-seen")
    rows = []
    for name, rule in CANDIDATES:
        te = acc(llm, rule, TEST) - base_test
        tr = acc(llm, rule, TRAIN) - base_train
        verifier = "ACCEPT" if te > 0.2 else "reject"
        naive = "accept" if tr > 0.2 else "reject"
        rows.append((name, tr, te, verifier, naive))
        print("%-20s |  %+.2f |   %+.2f    |   %-7s          | %s" % (name, tr, te, verifier, naive), flush=True)

    print("\n=== PROOF CHECK ===", flush=True)
    correct = [r for r in rows if r[0].startswith("correct")][0]
    overfit = [r for r in rows if r[0].startswith("overfit")][0]
    verifier_correct = (correct[3] == "ACCEPT") and all(r[3] == "reject" for r in rows if not r[0].startswith("correct"))
    naive_fooled = (overfit[4] == "accept")
    print("verifier ACCEPTS the real (transfers), REJECTS the fakes:", verifier_correct)
    print("naive-seen is FOOLED by the overfit fake:", naive_fooled)
    if verifier_correct and naive_fooled:
        print("\n>>> PROVEN: verified self-improvement — the verifier accepts a REAL transferring improvement and")
        print(">>> rejects an overfit one that the naive (seen-only) baseline wrongly accepts. Both directions shown.")
    else:
        print("\n>>> not clean this run — inspect (model may not apply the rule reliably).")


if __name__ == "__main__":
    main()
