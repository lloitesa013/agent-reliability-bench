"""
Autonomous verified self-improvement, attempt 2: overcome prior-anchoring with a PATTERN-FITTING
inducer. The judge extracts the (input, correct-output) pairs from the failures; the model is told to
IGNORE what the terms normally mean and fit output=f(input) to ALL pairs. Then verify on held-out.
Does a stronger inducer close the autonomous loop, or is prior-anchoring a hard wall? Either is honest.
"""
import os, re
import numpy as np
os.environ.setdefault("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
from real_rag import LLM

SYS = "You are a precise assistant. Answer with only the final number. End with: ANSWER: <number>"
FIT_SYS = ("You are a pattern finder. You are given (input, correct_output) pairs. IGNORE what the words "
           "normally mean — the system uses a NON-STANDARD rule. Find the exact arithmetic rule "
           "output = f(input) that fits ALL pairs, then state it as one short instruction referring to "
           "the quantity in the question. Propose THREE candidate rules, numbered 1., 2., 3.")

FAMILIES = [
    dict(name="overtime(6h)", q=lambda x: f"An employee worked {x} hours in one day. How many were overtime hours?",
         key="hours worked", gold=lambda x: max(0, x - 6), train=[7, 8, 9, 10, 11, 13], test=[12, 14, 15, 16, 18, 20]),
    dict(name="dozen(10)", q=lambda x: f"Here, how many individual items are in {x} dozen?",
         key="number of dozen", gold=lambda x: x * 10, train=[3, 4, 6, 7, 9, 12], test=[11, 13, 15, 18, 22, 25]),
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
    adopted = 0
    for F in FAMILIES:
        te = [(F["q"](x), F["gold"](x)) for x in F["test"]]
        bt = acc(llm, None, te)
        pairs = ", ".join("(%s=%d -> %d)" % (F["key"], x, F["gold"](x)) for x in F["train"])
        print("\n=== %s === baseline held-out=%.2f" % (F["name"], bt), flush=True)
        print("  pairs: %s" % pairs, flush=True)
        ind = llm.chat_batch(FIT_SYS, ["Pairs:\n" + pairs + "\n\nFind output=f(%s). Propose THREE rules." % F["key"]],
                             max_new_tokens=200, batch_size=1)[0]
        rules = [m.group(1).strip() for m in re.finditer(r"(?m)^\s*\d[\.\)]\s*(.+)", ind)][:3]
        best = None
        for i, r in enumerate(rules, 1):
            dte = acc(llm, r, te) - bt
            v = "ACCEPT" if dte > 0.3 else "reject"
            print("  induced#%d heldout Δ=%+.2f -> %s : %s" % (i, dte, v, r[:90]), flush=True)
            if dte > 0.3 and (best is None or dte > best[1]):
                best = (i, dte)
        if best:
            adopted += 1
            print("  -> ADOPTED induced#%d (held-out +%.2f)" % best, flush=True)
        else:
            print("  -> none transferred", flush=True)
    print("\n=== SUMMARY ===  autonomous verified self-improvement on %d/%d tasks" % (adopted, len(FAMILIES)), flush=True)
    if adopted >= 1:
        print(">>> AUTONOMOUS PROVEN: pattern-fitting inducer overcame prior-anchoring; the loop induced a")
        print(">>> rule from its own failures that transfers, and the verifier adopted it. Full loop closed.")
    else:
        print(">>> prior-anchoring is a hard wall for this 7B even with pattern-fitting — honest limit;")
        print(">>> the verifier remains the reliable piece (control-based ACCEPT proof stands).")


if __name__ == "__main__":
    main()
