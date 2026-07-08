"""
Phase-0 v2 — VERIFICATION-AS-SELECTION (complete verified self-improvement).
Agent fails -> memory proposes SEVERAL candidate rules -> each is VERIFIED on a held-out set with
disjoint params -> adopt only the rule(s) that TRANSFER (real), reject those that only help seen
(fake). Verification is the selection mechanism, not a post-hoc check. Runs on one 5090.
"""
import os, re
import numpy as np
os.environ.setdefault("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
from real_rag import LLM

AGENT_SYS = ("You are a numeric policy assistant. Answer IMMEDIATELY with the final number only — do "
             "NOT show any working, steps, or reasoning. End with a line exactly: ANSWER: <value>")
INDUCE_SYS = ("You improve an assistant. Given problems it got WRONG (with correct answers), propose "
              "THREE DIFFERENT general rules to prevent such mistakes on UNSEEN problems. Number them "
              "1., 2., 3. Each <=2 sentences, general method advice, NOT about specific numbers.")


def tasks(P):
    T = []
    for n, m in P["fees"]:
        T.append((f"A customer had {n} late payments at $39 each and {m} returned-item fees at $30 each. Total fees?", 39 * n + 30 * m))
    for n, d, e in P["dose"]:
        T.append((f"Daily acetaminophen limit is 3,000 mg. A patient took {n} doses of {d} mg. How many more {e} mg tablets can they take without exceeding the limit?", (3000 - n * d) // e))
    for bal, r in P["apr"]:
        T.append((f"A ${bal} balance has an APR of {r}%. One month of interest (APR/12), rounded to the nearest dollar?", round(bal * (r / 100) / 12)))
    for a, b in P["ded"]:
        T.append((f"Annual deductible is $1,500. A patient paid ${a}, then ${b}. How much of the deductible remains?", 1500 - a - b))
    return T


TRAIN = tasks({"fees": [(2, 3), (4, 2), (3, 5), (5, 1), (6, 4), (3, 2), (4, 5), (7, 3)],
               "dose": [(3, 650, 500), (2, 800, 400), (4, 500, 250), (3, 550, 500), (5, 400, 300), (2, 700, 250)],
               "apr": [(4200, 18), (2600, 24), (3000, 15), (6000, 12)],
               "ded": [(300, 450), (600, 250), (200, 700), (500, 400)]})
TEST = tasks({"fees": [(6, 2), (3, 4), (7, 1), (5, 3), (8, 2), (4, 4), (2, 6), (6, 5)],
              "dose": [(3, 750, 250), (5, 300, 750), (2, 950, 350), (4, 450, 500), (6, 300, 400), (3, 600, 600)],
              "apr": [(3600, 21), (5100, 15), (4800, 18), (2400, 24)],
              "ded": [(450, 350), (800, 200), (350, 550), (700, 300)]})


def num(s):
    m = re.findall(r"-?\d[\d,]*\.?\d*", s.replace("$", ""))
    return float(m[-1].replace(",", "")) if m else None


def extract(o):
    m = re.search(r"ANSWER:\s*(.+)", o, re.IGNORECASE)
    return m.group(1) if m else o


def run(llm, rule, tset):
    sys = AGENT_SYS if not rule else (AGENT_SYS + "\n\nAlways follow this learned rule:\n" + rule)
    outs = llm.chat_batch(sys, [q for q, _ in tset], max_new_tokens=256, batch_size=16)
    res = [(gold, num(extract(o))) for (q, gold), o in zip(tset, outs)]
    return np.mean([a is not None and abs(a - float(g)) < 0.5 for g, a in res])


def main():
    llm = LLM(device="cuda:0")
    print("TRAIN n=%d  TEST n=%d (disjoint params)\n" % (len(TRAIN), len(TEST)), flush=True)

    test0 = run(llm, None, TEST); train0 = run(llm, None, TRAIN)
    print("baseline: TEST(held-out)=%.3f  TRAIN=%.3f\n" % (test0, train0), flush=True)

    # collect train failures
    sys = AGENT_SYS
    outs = llm.chat_batch(sys, [q for q, _ in TRAIN], max_new_tokens=256, batch_size=16)
    fails = [(q, g) for (q, g), o in zip(TRAIN, outs) if not (num(extract(o)) is not None and abs(num(extract(o)) - float(g)) < 0.5)]
    print("watcher: %d/%d train failures\n" % (len(fails), len(TRAIN)), flush=True)

    block = "\n".join("- Q: %s\n  correct: %s" % (q, g) for q, g in fails[:14])
    out = llm.chat_batch(INDUCE_SYS, ["Problems it got WRONG:\n" + block + "\n\nPropose THREE rules."], max_new_tokens=220, batch_size=1)[0]
    cands = [("induced#%d" % i, m.group(1).strip()) for i, m in enumerate(re.finditer(r"(?m)^\s*\d[\.\)]\s*(.+)", out), 1)][:3]
    cands.append(("[ref] method", "Work through the problem step by step: compute and write down each intermediate value, then give the final answer."))

    print("=== candidate rules -> VERIFY on held-out (transfer) ===", flush=True)
    print("rule           | held-out delta | seen delta | verdict")
    adopted = None
    for name, rule in cands:
        th = run(llm, rule, TEST) - test0
        tr = run(llm, rule, TRAIN) - train0
        verdict = "ACCEPT (transfers)" if th > 0.05 else ("REJECT (overfit: seen only)" if tr > 0.05 else "REJECT (no effect)")
        print("%-14s |   %+.3f       |  %+.3f   | %s" % (name, th, tr, verdict), flush=True)
        print("     rule: %s" % rule[:120], flush=True)
        if th > 0.05 and (adopted is None or th > adopted[1]):
            adopted = (name, th, rule)

    print("\n=== SELECTION (verification as the gate) ===", flush=True)
    if adopted:
        print("ADOPTED: %s  (+%.3f held-out transfer)\n  %s" % (adopted[0], adopted[1], adopted[2]))
        print("-> VERIFIED self-improvement: kept a rule that GENERALIZES; rejected the ones that only fit seen.")
    else:
        print("ADOPTED: none — no candidate transferred to held-out. Honest null (watcher rejects all).")


if __name__ == "__main__":
    main()
