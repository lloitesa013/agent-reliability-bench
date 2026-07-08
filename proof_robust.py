"""
Robustness of the verified-self-improvement proof: repeat across 4 different hidden-convention tasks
(overtime, week length, dozen, century). For each, the verifier should ACCEPT the real (general) rule
that transfers to held-out values and REJECT the overfit (train-lookup) rule that the naive seen-only
baseline is fooled by. Proof is robust iff this holds on ALL tasks (not cherry-picked).
"""
import os, re
import numpy as np
os.environ.setdefault("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
from real_rag import LLM

SYS = "You are a precise assistant. Answer with only the final number. End with: ANSWER: <number>"

FAMILIES = [
    dict(name="overtime(6h)", q=lambda x: f"An employee worked {x} hours in one day. How many were overtime hours?",
         gold=lambda x: max(0, x - 6), train=[7, 8, 9, 10, 11], test=[12, 13, 14, 15, 16],
         correct="At this company, overtime is any time worked beyond 6 hours in a day (not 8).",
         spur="Overtime equals total hours minus twelve."),
    dict(name="week(5d)", q=lambda x: f"In this calendar, how many full weeks are in {x} days?",
         gold=lambda x: x // 5, train=[12, 17, 23, 31, 38], test=[44, 51, 63, 72, 88],
         correct="In this calendar, a week is 5 days long (not 7).",
         spur="In this calendar a week is 9 days long."),
    dict(name="dozen(10)", q=lambda x: f"Here, how many individual items are in {x} dozen?",
         gold=lambda x: x * 10, train=[3, 4, 6, 7, 9], test=[11, 13, 15, 18, 22],
         correct="Here, one dozen means 10 items (not 12).",
         spur="Here, one dozen means 15 items."),
    dict(name="century(50y)", q=lambda x: f"In this system, how many years are in {x} centuries?",
         gold=lambda x: x * 50, train=[2, 3, 4, 6, 7], test=[8, 9, 11, 13, 15],
         correct="In this system, a century is 50 years (not 100).",
         spur="In this system, a century is 70 years."),
]


def acc(llm, rule, items):
    sysp = SYS if not rule else (SYS + "\n\nApply this rule:\n" + rule)
    outs = llm.chat_batch(sysp, [q for q, _ in items], max_new_tokens=48, batch_size=16)
    ok = []
    for (q, g), o in zip(items, outs):
        m = re.search(r"ANSWER:\s*(-?\d+)", o, re.IGNORECASE)
        n = int(m.group(1)) if m else (int(re.findall(r"-?\d+", o)[-1]) if re.findall(r"-?\d+", o) else None)
        ok.append(n is not None and n == g)
    return float(np.mean(ok))


def main():
    llm = LLM(device="cuda:0")
    n_ok = 0; n_fooled = 0
    for F in FAMILIES:
        tr = [(F["q"](x), F["gold"](x)) for x in F["train"]]
        te = [(F["q"](x), F["gold"](x)) for x in F["test"]]
        overfit = "Reference table: " + ", ".join("%d->%d" % (x, F["gold"](x)) for x in F["train"]) + "."
        bt, btr = acc(llm, None, te), acc(llm, None, tr)
        cands = [("correct", F["correct"]), ("overfit", overfit), ("spurious", F["spur"])]
        print("\n=== %s ===  baseline held-out=%.2f seen=%.2f" % (F["name"], bt, btr), flush=True)
        verd = {}
        for nm, rule in cands:
            dte = acc(llm, rule, te) - bt
            dtr = acc(llm, rule, tr) - btr
            v = "ACCEPT" if dte > 0.2 else "reject"
            nv = "accept" if dtr > 0.2 else "reject"
            verd[nm] = (v, nv)
            print("  %-9s heldout Δ=%+.2f seen Δ=%+.2f -> verifier=%-7s naive=%s" % (nm, dte, dtr, v, nv), flush=True)
        ok = verd["correct"][0] == "ACCEPT" and verd["overfit"][0] == "reject" and verd["spurious"][0] == "reject"
        fooled = verd["overfit"][1] == "accept"
        n_ok += ok; n_fooled += fooled
        print("  -> verifier correct: %s | naive fooled by overfit: %s" % (ok, fooled), flush=True)

    print("\n=== ROBUSTNESS SUMMARY ===", flush=True)
    print("verifier correct (accept real, reject fakes) on %d/%d tasks" % (n_ok, len(FAMILIES)))
    print("naive-seen fooled by the overfit fake on %d/%d tasks" % (n_fooled, len(FAMILIES)))
    if n_ok == len(FAMILIES) and n_fooled >= len(FAMILIES) - 1:
        print(">>> ROBUST: verified self-improvement holds across all hidden-rule tasks (not cherry-picked).")
    else:
        print(">>> mixed — inspect per-task (model may not apply some rules reliably).")


if __name__ == "__main__":
    main()
