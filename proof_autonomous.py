"""
Autonomous verified self-improvement (stronger than the control-based proof).
The agent fails on a hidden-convention task; the model itself INDUCES candidate rules from its own
failures (not given); each induced rule is verified on HELD-OUT disjoint values. Success = at least one
INDUCED rule transfers (held-out gain) and is ACCEPTED, while non-transferring inductions are rejected.
This closes the loop end to end: fail -> induce -> VERIFY(held-out) -> adopt only the real one.
"""
import os, re
import numpy as np
os.environ.setdefault("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
from real_rag import LLM

SYS = "You are a precise assistant. Answer with only the final number. End with: ANSWER: <number>"
INDUCE = ("An assistant got the problems below WRONG (correct answers shown). Infer the hidden rule and "
          "propose THREE candidate general rules (numbered 1., 2., 3.), each <=1 sentence, that would make "
          "it answer UNSEEN problems of this kind correctly.")

FAMILIES = [
    dict(name="overtime(6h)", q=lambda x: f"An employee worked {x} hours in one day. How many were overtime hours?",
         gold=lambda x: max(0, x - 6), train=[7, 8, 9, 10, 11, 13], test=[12, 14, 15, 16, 18, 20]),
    dict(name="dozen(10)", q=lambda x: f"Here, how many individual items are in {x} dozen?",
         gold=lambda x: x * 10, train=[3, 4, 6, 7, 9, 12], test=[11, 13, 15, 18, 22, 25]),
]


def gnum(o):
    m = re.search(r"ANSWER:\s*(-?\d+)", o, re.IGNORECASE)
    if m:
        return int(m.group(1))
    d = re.findall(r"-?\d+", o)
    return int(d[-1]) if d else None


def acc(llm, rule, items):
    sysp = SYS if not rule else (SYS + "\n\nApply this rule:\n" + rule)
    outs = llm.chat_batch(sysp, [q for q, _ in items], max_new_tokens=48, batch_size=16)
    return float(np.mean([gnum(o) == g for (q, g), o in zip(items, outs)]))


def main():
    llm = LLM(device="cuda:0")
    total, adopted_real = 0, 0
    for F in FAMILIES:
        tr = [(F["q"](x), F["gold"](x)) for x in F["train"]]
        te = [(F["q"](x), F["gold"](x)) for x in F["test"]]
        bt, btr = acc(llm, None, te), acc(llm, None, tr)
        # collect failures on train
        outs = llm.chat_batch(SYS, [q for q, _ in tr], max_new_tokens=48, batch_size=16)
        fails = [(q, g) for (q, g), o in zip(tr, outs) if gnum(o) != g]
        print("\n=== %s === baseline held-out=%.2f seen=%.2f | %d/%d train failures" % (F["name"], bt, btr, len(fails), len(tr)), flush=True)
        block = "\n".join("Q: %s  correct: %s" % (q, g) for q, g in fails)
        ind = llm.chat_batch(INDUCE, ["Wrong problems:\n" + block + "\n\nPropose THREE rules."], max_new_tokens=180, batch_size=1)[0]
        rules = [m.group(1).strip() for m in re.finditer(r"(?m)^\s*\d[\.\)]\s*(.+)", ind)][:3]
        best = None
        for i, r in enumerate(rules, 1):
            dte = acc(llm, r, te) - bt
            v = "ACCEPT" if dte > 0.3 else "reject"
            print("  induced#%d heldout Δ=%+.2f -> %s : %s" % (i, dte, v, r[:90]), flush=True)
            if dte > 0.3 and (best is None or dte > best[1]):
                best = (i, dte, r)
        total += 1
        if best:
            adopted_real += 1
            print("  -> ADOPTED induced#%d (held-out +%.2f) = AUTONOMOUS verified self-improvement" % (best[0], best[1]), flush=True)
        else:
            print("  -> no induced rule transferred (verifier adopts none)", flush=True)

    print("\n=== SUMMARY ===", flush=True)
    print("autonomous verified self-improvement achieved on %d/%d tasks" % (adopted_real, total))
    if adopted_real >= 1:
        print(">>> PROVEN (autonomous): the model induced a rule from its OWN failures that transfers to")
        print(">>> held-out, and the verifier adopted it — the full loop, verified end to end.")


if __name__ == "__main__":
    main()
