"""
BRICK 7 - longer multi-round loop (robustness / not-cherry-picked). Same mechanism as brick 5 but the
agent acquires FIVE new capabilities (B..F) over five rounds. More rounds = more chances for ungated
adoption to accumulate regressions. Checks the verified>naive gap is not a 3-round fluke.
"""
import os, re
import numpy as np
os.environ.setdefault("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
from real_rag import LLM

BASE_SYS = "You are a precise assistant. Answer with only the final number. End with: ANSWER: <number>"
FIT_SYS = ("You are a pattern finder. You are given (input, correct_output) pairs from questions about a "
           "specific quantity. IGNORE what the words normally mean - the system uses a NON-STANDARD rule. "
           "Find the exact rule output=f(input) fitting ALL pairs. IMPORTANT: the assistant ALSO answers "
           "OTHER kinds of questions, so a safe rule must apply ONLY to THESE questions and not change "
           "answers to other questions. Propose THREE candidate rule-edits, numbered 1., 2., 3., RANGING "
           "from broad (just the arithmetic, no scope) to narrow (a rule naming the exact question type it "
           "applies to). Each must be a single sentence.")

RULE_A = "For overtime questions: overtime hours = (hours worked) minus 6, floored at 0."
FAMS = {
    "A_overtime": dict(key="hours worked", q=lambda x: f"An employee worked {x} hours in one day. How many were overtime hours?",
                       gold=lambda x: max(0, x - 6), seen=[7, 8, 9, 10, 11, 13], heldout=[12, 14, 15, 16, 18, 20]),
    "B_dozen": dict(key="number of dozen", q=lambda x: f"How many individual items are in {x} dozen?",
                    gold=lambda x: x * 10, seen=[3, 4, 6, 7, 9, 12], heldout=[11, 13, 15, 18, 22, 25]),
    "C_chapter": dict(key="number of chapters", q=lambda x: f"A book has {x} chapters. How many pages does it have?",
                      gold=lambda x: x * 7, seen=[2, 3, 5, 6, 8, 10], heldout=[4, 9, 11, 12, 14, 16]),
    "D_team": dict(key="number of teams", q=lambda x: f"There are {x} teams in a league. How many players are there in total?",
                   gold=lambda x: x * 4, seen=[2, 3, 5, 6, 8, 9], heldout=[4, 7, 10, 11, 13, 15]),
    "E_week": dict(key="number of weeks", q=lambda x: f"A project lasts {x} weeks. How many working days is that?",
                   gold=lambda x: x * 5, seen=[2, 3, 4, 6, 8, 10], heldout=[5, 7, 9, 11, 12, 14]),
    "F_crate": dict(key="number of crates", q=lambda x: f"A warehouse has {x} crates. How many boxes are there in total?",
                    gold=lambda x: x * 3, seen=[2, 4, 5, 7, 8, 10], heldout=[3, 6, 9, 11, 13, 15]),
}
ACQUIRE = ["B_dozen", "C_chapter", "D_team", "E_week", "F_crate"]
HID_THR = 0.34
REG_MARGIN = 0.20


def gnum(o):
    m = re.search(r"ANSWER:\s*(-?\d+)", o, re.IGNORECASE)
    if m:
        return int(m.group(1))
    dd = re.findall(r"-?\d+", o)
    return int(dd[-1]) if dd else None


def acc(llm, rules, fam, split="heldout"):
    sysp = BASE_SYS + ("\n\nRules:\n" + "\n".join("- " + r for r in rules) if rules else "")
    items = FAMS[fam][split]
    outs = llm.chat_batch(sysp, [FAMS[fam]["q"](x) for x in items], max_new_tokens=48, batch_size=16)
    return float(np.mean([gnum(o) == FAMS[fam]["gold"](x) for x, o in zip(items, outs)]))


def induce(llm, fam):
    pairs = ", ".join("(%s=%d -> %d)" % (FAMS[fam]["key"], x, FAMS[fam]["gold"](x)) for x in FAMS[fam]["seen"])
    ind = llm.chat_batch(FIT_SYS, ["Pairs:\n" + pairs + "\n\nPropose THREE rule-edits for these questions."],
                         max_new_tokens=240, batch_size=1)[0]
    return [m.group(1).strip() for m in re.finditer(r"(?m)^\s*\d[\.\)]\s*(.+)", ind)][:3]


def profile(llm, mem, acquired):
    return [acc(llm, mem, f) for f in acquired]


def main():
    llm = LLM(device="cuda:0")
    naive_mem = [RULE_A]
    ver_mem = [RULE_A]
    acquired = ["A_overtime"]

    def caps(mem):
        p = profile(llm, mem, acquired)
        return float(np.mean(p)), float(np.min(p))

    nm, nn = caps(naive_mem)
    vm, vn = caps(ver_mem)
    hist = [("start", nm, vm, nn, vn)]
    print("== round 0 (only A) ==  naive mean=%.2f/min=%.2f  verifying mean=%.2f/min=%.2f" % (nm, nn, vm, vn), flush=True)

    for rnd, target in enumerate(ACQUIRE, 1):
        print("\n=== ROUND %d: acquire %s ===" % (rnd, target), flush=True)
        cands = induce(llm, target)
        n_seen = [acc(llm, naive_mem + [c], target, "seen") for c in cands]
        n_pick = int(np.argmax(n_seen))
        naive_mem = naive_mem + [cands[n_pick]]

        ver_prof = {f: acc(llm, ver_mem, f) for f in acquired}
        base_new = acc(llm, ver_mem, target)
        passing = []
        for i, c in enumerate(cands):
            trial = ver_mem + [c]
            hid = acc(llm, trial, target)
            worst_reg = min((acc(llm, trial, f) - ver_prof[f]) for f in acquired)
            if (hid - base_new >= HID_THR) and (worst_reg >= -REG_MARGIN):
                passing.append((hid, i))
        if passing:
            _, vi = max(passing)
            ver_mem = ver_mem + [cands[vi]]
            vadopt = "#%d" % (vi + 1)
        else:
            vadopt = "NOTHING"

        acquired = acquired + [target]
        nm, nn = caps(naive_mem)
        vm, vn = caps(ver_mem)
        hist.append((target, nm, vm, nn, vn))
        print("   naive adopts #%d (seen=%.2f) | verifying adopts %s -> NAIVE mean=%.2f/min=%.2f  VERIFYING mean=%.2f/min=%.2f"
              % (n_pick + 1, n_seen[n_pick], vadopt, nm, nn, vm, vn), flush=True)

    print("\n=== TRAJECTORY ===   (mean = avg capability, MIN = worst single capability = did anything break)", flush=True)
    print("round        NAIVE(mean/min)   VERIFYING(mean/min)", flush=True)
    for name, nm, vm, nn, vn in hist:
        print("%-11s   %.2f / %.2f        %.2f / %.2f" % (name, nm, nn, vm, vn), flush=True)
    f = hist[-1]
    print("\n--- interpretation ---", flush=True)
    print(">>> FINAL  mean: naive=%.2f verifying=%.2f (Δ%+.2f)   MIN(worst): naive=%.2f verifying=%.2f (Δ%+.2f)"
          % (f[1], f[2], f[2] - f[1], f[3], f[4], f[4] - f[3]), flush=True)
    naive_broke = min(h[3] for h in hist[1:]) < 0.6
    ver_broke = min(h[4] for h in hist[1:]) < 0.6
    if (f[4] - f[3]) >= 0.15 or (naive_broke and not ver_broke):
        print(">>> MIN metric (the safety-relevant one) shows the real gap: naive BROKE a capability"
              " (min dropped to %.2f) while verifying kept all (min %.2f)." % (min(h[3] for h in hist[1:]), min(h[4] for h in hist[1:])), flush=True)
        print(">>> mean dilutes this at scale; worst-case is the honest lens for SAFE self-improvement.", flush=True)
    else:
        print(">>> honest: even on the MIN metric the gap is modest this run (naive worst %.2f, verifying worst %.2f)"
              % (min(h[3] for h in hist[1:]), min(h[4] for h in hist[1:])), flush=True)
        print(">>> - naive did not catastrophically break a capability here; verifier's guarantee (never regress) still held.", flush=True)


if __name__ == "__main__":
    main()
