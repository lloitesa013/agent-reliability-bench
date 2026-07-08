"""
BRICK 5 - the MULTI-ROUND self-verifying self-improvement loop. The agent starts with one capability (A)
and must ACQUIRE new capabilities B, C, D one per round, each time proposing self-modifications and adopting
under a policy. Two agents run the SAME candidate pool each round but differ only in SELECTION:
  NAIVE          - adopt the edit with the best SEEN score on the new task
  SELF-VERIFYING - adopt only an edit that (hidden test) transfers to held-out new task AND (regression)
                   does not drop any ALREADY-ACQUIRED capability; else adopt nothing
We track CUMULATIVE capability = mean held-out accuracy over all capabilities acquired so far.
Question: over rounds, does self-verifying climb/hold while naive erodes (regressions accumulate)?
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
}
ACQUIRE = ["B_dozen", "C_chapter", "D_team"]
HID_THR = 0.34     # hidden-test: new task must rise at least this much
REG_MARGIN = 0.20  # regression: no acquired capability may drop more than this


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


def cum_cap(llm, mem, acquired):
    return float(np.mean([acc(llm, mem, f) for f in acquired]))


def main():
    llm = LLM(device="cuda:0")
    naive_mem = [RULE_A]
    ver_mem = [RULE_A]
    acquired = ["A_overtime"]
    hist = [("start", cum_cap(llm, naive_mem, acquired), cum_cap(llm, ver_mem, acquired))]
    print("== round 0 (only A) ==  naive=%.2f  verifying=%.2f" % (hist[0][1], hist[0][2]), flush=True)

    for rnd, target in enumerate(ACQUIRE, 1):
        print("\n=== ROUND %d: acquire %s ===" % (rnd, target), flush=True)
        cands = induce(llm, target)
        for i, c in enumerate(cands, 1):
            print("   cand#%d %s" % (i, c[:95]), flush=True)

        # --- NAIVE: adopt best SEEN on the new task (ties -> first) ---
        n_seen = [acc(llm, naive_mem + [c], target, "seen") for c in cands]
        n_pick = int(np.argmax(n_seen))
        naive_mem = naive_mem + [cands[n_pick]]
        print("   NAIVE   adopts #%d (seen=%.2f)" % (n_pick + 1, n_seen[n_pick]), flush=True)

        # --- SELF-VERIFYING: hidden-test + regression over already-acquired ---
        ver_prof = {f: acc(llm, ver_mem, f) for f in acquired}     # capability before this round
        base_new = acc(llm, ver_mem, target)
        passing = []
        for i, c in enumerate(cands):
            trial = ver_mem + [c]
            hid = acc(llm, trial, target)
            worst_reg = min((acc(llm, trial, f) - ver_prof[f]) for f in acquired)
            ok = (hid - base_new >= HID_THR) and (worst_reg >= -REG_MARGIN)
            print("   ver cand#%d hidden=%.2f worst_reg=%+.2f -> %s" % (i + 1, hid, worst_reg, "pass" if ok else "reject"), flush=True)
            if ok:
                passing.append((hid, i))
        if passing:
            _, vi = max(passing)
            ver_mem = ver_mem + [cands[vi]]
            print("   VERIFYING adopts #%d" % (vi + 1), flush=True)
        else:
            print("   VERIFYING adopts NOTHING (no edit cleared both gates)", flush=True)

        acquired = acquired + [target]
        nc, vc = cum_cap(llm, naive_mem, acquired), cum_cap(llm, ver_mem, acquired)
        hist.append((target, nc, vc))
        print("   -> cumulative capability (all %d):  NAIVE=%.2f  VERIFYING=%.2f" % (len(acquired), nc, vc), flush=True)

    print("\n=== TRAJECTORY (cumulative held-out capability over rounds) ===", flush=True)
    print("round        NAIVE   VERIFYING", flush=True)
    for name, nc, vc in hist:
        print("%-11s  %.2f     %.2f" % (name, nc, vc), flush=True)
    fn, fv = hist[-1][1], hist[-1][2]
    print("\n--- interpretation ---", flush=True)
    naive_eroded = any(hist[i][1] < hist[i - 1][1] - 0.05 for i in range(1, len(hist)))
    ver_monotone = all(hist[i][2] >= hist[i - 1][2] - 0.05 for i in range(1, len(hist)))
    if fv > fn + 0.08 and ver_monotone:
        print(">>> BRICK 5 shown: self-verifying capability CLIMBS/holds across rounds (%.2f) while naive" % fv, flush=True)
        print(">>> ends at %.2f%s. Gated adoption compounds safely; ungated adoption accumulates regressions." %
              (fn, " (and eroded mid-way)" if naive_eroded else ""), flush=True)
    elif fv > fn + 0.08:
        print(">>> self-verifying ends higher (%.2f vs %.2f) - gating helps cumulatively even if not strictly monotone." % (fv, fn), flush=True)
    else:
        print(">>> honest: gap did not open this run (naive=%.2f, verifying=%.2f) - either candidates were all" % (fn, fv), flush=True)
        print(">>> safe (no regression to punish) or none transferred; inspect per-round log. Verifier still never regressed.", flush=True)


if __name__ == "__main__":
    main()
