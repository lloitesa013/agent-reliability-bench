"""
Phase-0 closed learning loop (VERIFIED self-improvement).
Two-hop numeric tasks + a one-shot baseline (answer immediately, no working) -> reliable failures ->
watcher collects them -> memory INDUCES one general rule -> rule enforced -> retry -> improvement
measured on a HELD-OUT set with DISJOINT params. Verified = held-out transfer, not seen memorization.
"""
import os, re
import numpy as np
os.environ.setdefault("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
from real_rag import LLM

AGENT_SYS = ("You are a numeric policy assistant. Answer IMMEDIATELY with the final number only — do "
             "NOT show any working, steps, or reasoning. End with a line exactly: ANSWER: <value>")
INDUCE_SYS = ("You improve an assistant. Given problems it got WRONG (with correct answers), write "
              "exactly ONE general rule (<=2 sentences) to prevent these mistakes on UNSEEN problems. "
              "General method advice only, NOT about specific numbers.")


def tasks(P):
    T = []
    for n, m in P["fees"]:
        T.append((f"A customer had {n} late payments at $39 each and {m} returned-item fees at $30 "
                  f"each. What is the total in fees?", 39 * n + 30 * m))
    for n, d, e in P["dose"]:
        T.append((f"The daily acetaminophen limit is 3,000 mg. A patient already took {n} doses of "
                  f"{d} mg. How many more {e} mg tablets can they take today without exceeding the "
                  f"limit?", (3000 - n * d) // e))
    for bal, r in P["apr"]:
        T.append((f"A ${bal} balance has an APR of {r}%. What is one month of interest (the APR "
                  f"divided by 12), rounded to the nearest dollar?", round(bal * (r / 100) / 12)))
    for a, b in P["ded"]:
        T.append((f"The annual deductible is $1,500. A patient paid ${a}, then later ${b}. How much "
                  f"of the deductible remains?", 1500 - a - b))
    return T


TRAIN = tasks({"fees": [(2, 3), (4, 2), (3, 5), (5, 1)], "dose": [(3, 650, 500), (2, 800, 400), (4, 500, 250)],
               "apr": [(4200, 18), (2600, 24)], "ded": [(300, 450), (600, 250)]})
TEST = tasks({"fees": [(6, 2), (3, 4), (7, 1), (5, 3)], "dose": [(3, 550, 500), (5, 300, 750), (2, 950, 350)],
              "apr": [(3600, 21), (5100, 15)], "ded": [(450, 350), (800, 200)]})


def num(s):
    m = re.findall(r"-?\d[\d,]*\.?\d*", s.replace("$", ""))
    return float(m[-1].replace(",", "")) if m else None


def extract(o):
    m = re.search(r"ANSWER:\s*(.+)", o, re.IGNORECASE)
    return m.group(1) if m else o


def run(llm, rule, tset):
    sys = AGENT_SYS if not rule else (AGENT_SYS + "\n\nAlways follow this learned rule:\n" + rule)
    outs = llm.chat_batch(sys, [q for q, _ in tset], max_new_tokens=256, batch_size=16)
    res = []
    for (q, gold), o in zip(tset, outs):
        a = num(extract(o))
        res.append((q, gold, o, a is not None and abs(a - float(gold)) < 0.5))
    return np.mean([r[3] for r in res]), res


def main():
    llm = LLM(device="cuda:0")
    print("TRAIN n=%d  TEST n=%d (disjoint params)\n" % (len(TRAIN), len(TEST)), flush=True)

    test0, _ = run(llm, None, TEST)
    train0, tr = run(llm, None, TRAIN)
    print("baseline: TEST(held-out)=%.3f  TRAIN=%.3f" % (test0, train0), flush=True)

    fails = [(q, gold, o) for (q, gold, o, ok) in tr if not ok]
    print("watcher: %d/%d train failures observed" % (len(fails), len(TRAIN)), flush=True)
    if not fails:
        print("no failures — task still too easy."); return

    block = "\n".join("- Q: %s\n  correct answer: %s" % (q, gold) for q, gold, _ in fails[:12])
    rule = llm.chat_batch(INDUCE_SYS, ["Problems it got WRONG:\n" + block + "\n\nWrite ONE general rule."],
                          max_new_tokens=80, batch_size=1)[0].strip()
    print("\nINDUCED RULE:\n  %s\n" % rule, flush=True)

    test1, _ = run(llm, rule, TEST)
    train1, _ = run(llm, rule, TRAIN)
    print("=== after enforcing the learned rule ===", flush=True)
    print("TEST (held-out): %.3f -> %.3f   (VERIFIED transfer delta = %+.3f)" % (test0, test1, test1 - test0))
    print("TRAIN (seen)   : %.3f -> %.3f   (delta = %+.3f)" % (train0, train1, train1 - train0))
    print("\nVERDICT:", flush=True)
    if test1 - test0 > 0.02:
        print("  VERIFIED self-improvement: rule TRANSFERS to unseen params (+%.3f held-out)." % (test1 - test0))
    elif train1 - train0 > 0.05:
        print("  FAKE/overfit: helped seen only, not held-out -> watcher would REJECT it.")
    else:
        print("  honest null: rule did not help.")


if __name__ == "__main__":
    main()
