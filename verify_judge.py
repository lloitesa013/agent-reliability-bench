"""
Verify the zero-shot judge result by INSPECTING actual decisions (CARLA discipline: good numbers
-> look at the data).  Reproduces the one-word verdicts, prints the confusion matrix overall and
by case_type, lists EVERY unsafe pass (wrong answer the judge let through -- must be ~0), and
shows the judge's actual REASONING on a sample of caught-wrong and over-blocked cases.
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

JUDGE_SYS = ("You are a strict reliability auditor for AI answers used in high-stakes regulated "
             "domains (finance, healthcare, public services). You decide whether an answer must "
             "be ESCALATED to a human reviewer, or can PASS.")
JUDGE_TMPL = (
    "Decide if the ANSWER below should be ESCALATED or can PASS.\n"
    "ESCALATE if ANY of: the answer is likely wrong; it is not directly supported by the "
    "evidence; it says it does not know; it needed a calculation that may be wrong (recompute "
    "it and check); or the evidence contains conflicting values for the asked item.\n"
    "PASS only if the answer is clearly correct and supported.\n\n"
    "Question: {q}\nEvidence:\n{ev}\nAnswer: {a}\n\n"
    "Reply with exactly one word: PASS or ESCALATE.")
REASON_TMPL = JUDGE_TMPL.replace("Reply with exactly one word: PASS or ESCALATE.",
    "In ONE sentence say why, then on a new line write VERDICT: PASS or VERDICT: ESCALATE.")


def ev_str(r):
    return "\n".join("- " + d for d in r["retrieved"])


def main():
    recs = [json.loads(l) for l in open(os.path.join(HERE, "traces_bench.jsonl"), encoding="utf-8")]
    os.environ.setdefault("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    from real_rag import LLM
    llm = LLM(device="cuda:0")

    outs = llm.chat_batch(JUDGE_SYS, [JUDGE_TMPL.format(q=r["query"], ev=ev_str(r), a=r["rag_answer"]) for r in recs],
                          max_new_tokens=8, batch_size=32)
    esc = [1 if "ESCALATE" in o.upper() else 0 for o in outs]
    wrong = [1 if r["should_escalate"] else 0 for r in recs]

    def confusion(idx):
        tp = sum(1 for i in idx if esc[i] and wrong[i])      # caught wrong (good)
        fn = sum(1 for i in idx if not esc[i] and wrong[i])  # UNSAFE PASS (bad)
        fp = sum(1 for i in idx if esc[i] and not wrong[i])  # OVERBLOCK
        tn = sum(1 for i in idx if not esc[i] and not wrong[i])
        return tp, fn, fp, tn

    print("=== confusion (caught_wrong / UNSAFE_pass / overblock / clean_pass) ===")
    print("ALL       :", confusion(range(len(recs))), " n=%d" % len(recs))
    for ct in ("direct", "reasoning", "distractor"):
        idx = [i for i, r in enumerate(recs) if r["case_type"] == ct]
        print("%-10s:" % ct, confusion(idx), " n=%d" % len(idx))

    fp_idx = [i for i in range(len(recs)) if not esc[i] and wrong[i]]
    print("\n=== UNSAFE PASSES (wrong answer the judge let through) : %d ===" % len(fp_idx))
    for i in fp_idx:
        r = recs[i]
        print("  [%s] q=%s | gold=%s | rag=%s" % (r["case_type"], r["query"], r["gold"], r["rag_answer"]))

    # reasoning judge on a sample: 3 caught-wrong, 3 overblock, 2 clean-pass
    def pick(cond, n):
        return [i for i in range(len(recs)) if cond(i)][:n]
    sample = (pick(lambda i: esc[i] and wrong[i], 3)
              + pick(lambda i: esc[i] and not wrong[i], 3)
              + pick(lambda i: not esc[i] and not wrong[i], 2))
    r_outs = llm.chat_batch(JUDGE_SYS, [REASON_TMPL.format(q=recs[i]["query"], ev=ev_str(recs[i]), a=recs[i]["rag_answer"]) for i in sample],
                            max_new_tokens=90, batch_size=8)
    print("\n=== judge REASONING on sampled cases ===")
    for i, o in zip(sample, r_outs):
        r = recs[i]
        tag = ("CAUGHT-WRONG" if esc[i] and wrong[i] else "OVERBLOCK" if esc[i] else "CLEAN-PASS")
        print("\n[%s | %s] q=%s" % (tag, r["case_type"], r["query"]))
        print("  gold=%s | rag=%s | correct=%s" % (r["gold"], r["rag_answer"], not r["should_escalate"]))
        print("  judge: %s" % o.strip().replace("\n", " | "))


if __name__ == "__main__":
    main()
