"""
w2-autoloop -- W2 of the S2 pre-registration (S2_PREREG.md, sealed 2026-07-17).

The closed code-surface loop, run END-TO-END BY CODE: proposer (7B) -> Trial Registry ->
gate (execution on gate-hidden tests) -> registry-computed verdict -> prescriber-decided next move.
ZERO human verdict intervention (the human launches this script and reads the result; per the
sealed prereg that is the entire permitted human role).

Design (all sealed in the registry BEFORE any generation happens -- see register() calls):
  - FRESH data (the "new trial" requirement): MBPP rows are consumed in the same order as the
    R22/R24 runs (HF `mbpp` test split = task_id 11..510 ascending); the first 120 filter-passing
    problems are SKIPPED (R22/R24 consumed at most ~110), and the next 100 problems having >=3
    tests are used. HumanEval likewise skips the first 90 R26-filter-passing problems.
  - Non-tautological audit: per problem, tests are split visible / gate-hidden / AUDIT. The gate
    never sees the audit test(s); adopted-solution quality is scored on the audit only. Without
    this split, "the gate ships 0 held-out failures" would be true by construction.
  - Arms (raw counts only; the registry computes the verdict):
      naive : ships its best-visible candidate on every problem (no abstain; ties -> first).
      gate  : adopts the first candidate passing visible AND gate-hidden, else ABSTAINS.
    verdict = gate_rate: is the gate's audit-failure rate significantly lower than naive's?
  - The prescriber (not the human) decides what happens after the verdict: an ACCEPT at this
    scope must escalate ("a verdict is only valid at its measured scope") -> the HumanEval trial
    runs automatically. Any unmapped state -> the prescriber hands back to the operator and the
    loop STOPS (it must not guess).

Artifacts: w2_trials.jsonl (append-only ledger), w2_details.jsonl (every per-problem decision
with its evidence), w2_progress.txt (heartbeat for disk-based monitoring).
Run (5090 box):  python w2_autoloop.py
"""
import gzip
import json
import os
import re
import subprocess
import sys
import time

from vsi_registry import Registry
from vsi_prescriber import prescribe

os.environ.setdefault("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data_w2")
REG_PATH = os.path.join(HERE, "w2_trials.jsonl")
DETAILS = os.path.join(HERE, "w2_details.jsonl")
PROGRESS = os.path.join(HERE, "w2_progress.txt")

MBPP_SKIP, MBPP_TAKE = 120, 100
HE_SKIP, HE_TAKE = 90, 60

GEN_SYS_MBPP = (
    "You are an expert Python programmer. You are given a problem and ONE example test. Write THREE "
    "complete solutions, each in its own ```python code block with the full function:\n"
    "1. a GENERAL, correct solution for all valid inputs;\n"
    "2. a MINIMAL solution that only needs to make the shown example test pass;\n"
    "3. a different correct approach.\nOutput only the three code blocks.")
GEN_SYS_HE = (
    "You are an expert Python programmer. You are given a function specification (signature + "
    "docstring) and ONE example test. Write THREE complete standalone implementations of the "
    "function, each in its own ```python code block (include any imports; keep the EXACT function "
    "name and signature):\n"
    "1. a GENERAL, correct solution for all valid inputs;\n"
    "2. a MINIMAL solution that only needs to make the shown example test pass;\n"
    "3. a different correct approach.\nOutput only the three code blocks.")


def beat(msg):
    line = "%s  %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(PROGRESS, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def extract_code(txt):
    return [b.strip() for b in re.findall(r"```(?:python)?\s*(.*?)```", txt, re.DOTALL) if "def " in b][:3]


def run_tests(code, tests, timeout=6, entry=None):
    body = code + "\n"
    if entry:
        body += "\ncandidate = " + entry + "\n"
    script = body + "\n".join(tests) + "\nprint('OK')\n"
    try:
        r = subprocess.run([sys.executable, "-c", script], timeout=timeout, capture_output=True, text=True)
        return r.returncode == 0 and "OK" in r.stdout
    except Exception:
        return False


# ------------------------------------------------------------------ fresh problem loaders

def mbpp_fresh():
    """HF `mbpp` test split order = task_id 11..510 ascending. Skip the slice R22/R24 consumed."""
    rows = []
    with open(os.path.join(DATA, "mbpp.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            ex = json.loads(line)
            if 11 <= ex["task_id"] <= 510:
                rows.append(ex)
    rows.sort(key=lambda e: e["task_id"])
    passing = [e for e in rows if (e.get("text") and len(e.get("test_list") or []) >= 2)]
    fresh = [e for e in passing[MBPP_SKIP:] if len(e["test_list"]) >= 3][:MBPP_TAKE]
    return fresh


def he_flat_asserts(test_src):
    m = re.search(r"def check\(candidate\):\s*\n(.*)", test_src, re.DOTALL)
    if not m:
        return None
    out = []
    for line in m.group(1).splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("assert ") and "candidate" in s and \
           s.count("(") == s.count(")") and s.count("[") == s.count("]"):
            out.append(s)
        else:
            return None
    return out if len(out) >= 2 else None


def humaneval_fresh():
    rows = []
    with gzip.open(os.path.join(DATA, "HumanEval.jsonl.gz"), "rt", encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
    passing = [(e, he_flat_asserts(e["test"])) for e in rows]
    passing = [(e, a) for e, a in passing if a]
    fresh = [(e, a) for e, a in passing[HE_SKIP:] if len(a) >= 3][:HE_TAKE]
    return fresh


# ------------------------------------------------------------------ one automated trial

def run_trial(reg, llm, tid, problems, note):
    """problems: list of (key, prompt_text, visible[], gate_hidden[], audit[], entry_or_None, sys_prompt)."""
    reg.register(tid, recipe={"name": tid, "type": "code-surface", "move": "auto-loop"},
                 lines={"stat_alpha": 0.05}, note=note)
    beat("%s registered (lines sealed before generation); %d fresh problems" % (tid, len(problems)))

    naive_n = naive_fails = gate_n = gate_fails = abstains = 0
    for k, (key, prompt, visible, gatet, audit, entry, sysp) in enumerate(problems):
        txt = llm.chat_batch(sysp, [prompt], max_new_tokens=560, batch_size=1)[0]
        cands = extract_code(txt)
        if not cands:
            continue
        rows = [(run_tests(c, visible, entry=entry), run_tests(c, gatet, entry=entry),
                 run_tests(c, audit, entry=entry)) for c in cands]
        # naive: ship best-visible (ties -> first); no abstain
        maxv = max(v for v, _, _ in rows)
        pick = next(i for i, (v, _, _) in enumerate(rows) if v == maxv)
        naive_n += 1
        naive_ok = rows[pick][2]
        naive_fails += (not naive_ok)
        # gate: adopt first candidate passing visible AND gate-hidden; else abstain
        adopt = next((i for i, (v, g, _) in enumerate(rows) if v and g), None)
        if adopt is None:
            abstains += 1
            gate_ok = None
        else:
            gate_n += 1
            gate_ok = rows[adopt][2]
            gate_fails += (not gate_ok)
        with open(DETAILS, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"trial": tid, "key": key,
                                 "rows": [[bool(a), bool(b), bool(c)] for a, b, c in rows],
                                 "naive_pick": pick, "naive_audit_ok": bool(naive_ok),
                                 "gate_action": ("ADOPT %d" % adopt) if adopt is not None else "ABSTAIN",
                                 "gate_audit_ok": gate_ok}) + "\n")
        if (k + 1) % 10 == 0:
            beat("%s  %d/%d  naive_bad=%d/%d  gate_bad=%d/%d  abstain=%d"
                 % (tid, k + 1, len(problems), naive_fails, naive_n, gate_fails, gate_n, abstains))

    reg.add_arm(tid, "fix", {"baseline_fails": naive_fails, "baseline_n": naive_n,
                             "fails": gate_fails, "n": gate_n})
    verdict = reg.verdict(tid)
    beat("%s VERDICT (registry-computed): %s  axes=%s  [naive %d/%d bad, gate %d/%d bad, abstain %d]"
         % (tid, verdict["verdict"], verdict["axes"], naive_fails, naive_n, gate_fails, gate_n, abstains))
    return verdict


def main():
    open(PROGRESS, "w").close()
    beat("W2 autoloop starting (S2_PREREG sealed 2026-07-17; zero human verdicts from here on)")
    from real_rag import LLM
    llm = LLM(device="cuda:0")
    reg = Registry(REG_PATH)

    mbpp = [("mbpp-%d" % ex["task_id"],
             "Problem:\n%s\n\nExample test (must pass):\n%s\n\nWrite the three solutions."
             % (ex["text"], ex["test_list"][0]),
             ex["test_list"][:1], ex["test_list"][1:2], ex["test_list"][2:], None, GEN_SYS_MBPP)
            for ex in mbpp_fresh()]
    v1 = run_trial(reg, llm, "W2-MBPP-fresh", mbpp,
                   note="fresh slice: skip first %d filter-passing of HF-test-order, take %d with >=3 "
                        "tests; visible=t0 gate=t1 audit=t2+; naive=best-visible-no-abstain; "
                        "gate=first passing visible+gate-hidden else abstain" % (MBPP_SKIP, MBPP_TAKE))

    p = prescribe(reg.history())
    beat("PRESCRIBER: %s -- %s" % (p["axis"], p["why"]))

    if p["axis"] == "escalate-deployment":
        he = [(ex["task_id"],
               "Specification:\n%s\n\nExample test (must pass):\n%s\n\nWrite the three implementations."
               % (ex["prompt"], a[0]),
               a[:1], a[1:-1], a[-1:], ex["entry_point"], GEN_SYS_HE)
              for ex, a in humaneval_fresh()]
        v2 = run_trial(reg, llm, "W2-HumanEval-fresh", he,
                       note="scope escalation ordered by the prescriber, not a human; skip first %d "
                            "R26-filter-passing, take %d with >=3 flat asserts; visible=a0 "
                            "gate=a1..a-2 audit=a-1" % (HE_SKIP, HE_TAKE))
        p2 = prescribe(reg.history())
        beat("PRESCRIBER (final): %s -- %s" % (p2["axis"], p2["why"]))
    else:
        beat("prescriber did not order escalation -> loop stops here (it must not guess)")

    beat("W2 autoloop DONE. Ledger: w2_trials.jsonl ; evidence: w2_details.jsonl")


if __name__ == "__main__":
    main()
