"""
D-lite - replicate R24 on a SECOND real benchmark (HumanEval) to kill the "MBPP-specific" objection.
Same protocol as mbpp_self_verify2.py: the agent sees the function signature+docstring + ONE visible test
assert and proposes THREE solutions; the verifier EXECUTES each on the HELD-OUT asserts. Clean metrics:
solvable-only naive-vs-verifier bad-ship rate + genuine-overfit counts.
HumanEval's `test` field is a `def check(candidate):` body; we only keep problems whose check body is a
flat list of single-line asserts (>=2), split first=visible / rest=hidden.
"""
import os, re, sys, subprocess
import numpy as np
os.environ.setdefault("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
from datasets import load_dataset
from real_rag import LLM

N_PROB = 80
GEN_SYS = ("You are an expert Python programmer. You are given a function specification (signature + "
           "docstring) and ONE example test. Write THREE complete standalone implementations of the "
           "function, each in its own ```python code block (include any imports; keep the EXACT function "
           "name and signature):\n"
           "1. a GENERAL, correct solution for all valid inputs;\n"
           "2. a MINIMAL solution that only needs to make the shown example test pass;\n"
           "3. a different correct approach.\nOutput only the three code blocks.")


def extract_asserts(test_src):
    """Return list of single-line assert statements iff the check body is flat asserts only."""
    m = re.search(r"def check\(candidate\):\s*\n(.*)", test_src, re.DOTALL)
    if not m:
        return None
    asserts = []
    for line in m.group(1).splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("assert ") and "candidate" in s:
            # reject multi-line asserts (unbalanced parens)
            if s.count("(") == s.count(")") and s.count("[") == s.count("]"):
                asserts.append(s)
            else:
                return None
        else:
            return None   # any non-assert statement (helpers, loops) -> skip problem
    return asserts if len(asserts) >= 2 else None


def extract_code(txt):
    return [b.strip() for b in re.findall(r"```(?:python)?\s*(.*?)```", txt, re.DOTALL) if "def " in b][:3]


def run_tests(code, entry, asserts, timeout=6):
    script = code + "\n\ncandidate = " + entry + "\n" + "\n".join(asserts) + "\nprint('OK')\n"
    try:
        r = subprocess.run([sys.executable, "-c", script], timeout=timeout, capture_output=True, text=True)
        return r.returncode == 0 and "OK" in r.stdout
    except Exception:
        return False


def main():
    d = load_dataset("openai/openai_humaneval", split="test")
    llm = LLM(device="cuda:0")

    solvable_naive_bad, solvable_ver_bad = [], []
    overfit_present, overfit_naive_ships = 0, 0
    n_solvable, n_unsolvable, used = 0, 0, 0
    for ex in d:
        if used >= N_PROB:
            break
        asserts = extract_asserts(ex["test"])
        if not asserts:
            continue
        visible, hidden = asserts[:1], asserts[1:]
        txt = llm.chat_batch(GEN_SYS, ["Specification:\n%s\n\nExample test (must pass):\n%s\n\nWrite the three implementations."
                                       % (ex["prompt"], visible[0])], max_new_tokens=560, batch_size=1)[0]
        cands = extract_code(txt)
        if not cands:
            continue
        rows = [(run_tests(c, ex["entry_point"], visible), run_tests(c, ex["entry_point"], hidden)) for c in cands]
        used += 1
        any_hidden = any(hp for _, hp in rows)
        has_overfit = any(vp and not hp for vp, hp in rows)
        if has_overfit:
            overfit_present += 1
        maxv = max(vp for vp, _ in rows)
        tied = [(vp, hp) for vp, hp in rows if vp == maxv]
        naive_bad = float(np.mean([0.0 if hp else 1.0 for _, hp in tied])) if maxv else 1.0
        if any_hidden:
            n_solvable += 1
            solvable_naive_bad.append(naive_bad)
            solvable_ver_bad.append(0.0)     # verifier ships only held-out-passing, else abstains
            if has_overfit and naive_bad > 0:
                overfit_naive_ships += 1
        else:
            n_unsolvable += 1
        if used % 10 == 0:
            print("   ...%d problems (solvable=%d unsolvable=%d overfit-present=%d)"
                  % (used, n_solvable, n_unsolvable, overfit_present), flush=True)

    print("\n=== HUMANEVAL self-verifying coding agent (%d problems) ===" % used, flush=True)
    print("solvable=%d  unsolvable=%d  overfit-present=%d (%.0f%%)"
          % (n_solvable, n_unsolvable, overfit_present, 100 * overfit_present / max(used, 1)), flush=True)
    print("[SOLVABLE only] NAIVE ships held-out-FAILING: %.0f%%   VERIFIER: %.0f%%"
          % (100 * np.mean(solvable_naive_bad), 100 * np.mean(solvable_ver_bad)), flush=True)
    print("of overfit-present problems, NAIVE ships the overfit: %d/%d ; VERIFIER: 0"
          % (overfit_naive_ships, overfit_present), flush=True)
    print("\n--- interpretation ---", flush=True)
    if np.mean(solvable_naive_bad) >= 0.05 and overfit_present >= 5:
        print(">>> R24 REPLICATES on a second real benchmark: the naive-vs-verifier single-decision gap is", flush=True)
        print(">>> not MBPP-specific.", flush=True)
    else:
        print(">>> honest: weaker signal on HumanEval this run (naive %.0f%%, overfit-present %d) - inspect"
              % (100 * np.mean(solvable_naive_bad), overfit_present), flush=True)
        print(">>> per-problem; possibly the model rarely hardcodes HumanEval-style specs.", flush=True)


if __name__ == "__main__":
    main()
