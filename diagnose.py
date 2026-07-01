"""
Diagnose WHY the RAG agent failed, with a CALIBRATED confidence and the option to
ABSTAIN ("uncertain") when the evidence does not clearly point to one cause.

Causes: no_failure | retrieval_miss | missing_knowledge | reasoning_error |
        correct_abstention | uncertain (abstain)

The point (the "trustworthy" angle): the tool reports not just a cause but how
much to trust that diagnosis -- and stays silent when it genuinely can't tell.
"""

import json
from pathlib import Path
from agent import KNOWLEDGE


def _tokens(s):
    return {w.strip(".,?!()").lower() for w in s.split() if w.strip(".,?!()")}


def _match(texts, needle):
    """Best fraction of the needle's tokens found in any single text (0..1)."""
    nt = _tokens(needle)
    if not nt:
        return 0.0
    return max((len(nt & _tokens(t)) / len(nt)) for t in texts) if texts else 0.0


# decision thresholds (these are the knobs you'd calibrate against labeled data)
HI, LO, ABSTAIN_BAND = 0.6, 0.34, 0.12


def _conf(margin):
    return round(min(0.99, 0.5 + margin), 2)


def _out(cause, confidence, in_kb, in_retr, in_answer):
    return {
        "cause": cause,
        "confidence": confidence,
        "evidence": {"answer_in_kb": round(in_kb, 2),
                     "answer_in_retrieved": round(in_retr, 2),
                     "answer_match": round(in_answer, 2)},
    }


def diagnose(trace):
    expected = trace.get("expected", "")
    answer = trace.get("answer", "")
    retrieved = next((s["retrieved"] for s in trace["steps"] if s["step"] == "retrieve"), [])

    in_answer = _match([answer], expected)   # did the agent actually answer it?
    in_kb = _match(KNOWLEDGE, expected)      # is the answer available at all?
    in_retr = _match(retrieved, expected)    # was it retrieved?
    said_idk = "don't know" in answer.lower() or "do not know" in answer.lower()

    if in_answer >= HI:
        return _out("no_failure", _conf(in_answer - HI), in_kb, in_retr, in_answer)

    # ABSTAIN when the deciding signal sits in the ambiguous band near a threshold
    if abs(in_kb - LO) < ABSTAIN_BAND or abs(in_retr - LO) < ABSTAIN_BAND:
        return _out("uncertain", 0.3, in_kb, in_retr, in_answer)

    if in_kb < LO:
        cause = "correct_abstention" if said_idk else "missing_knowledge"
        return _out(cause, _conf(LO - in_kb), in_kb, in_retr, in_answer)
    if in_retr < LO:
        return _out("retrieval_miss", _conf(LO - in_retr), in_kb, in_retr, in_answer)
    return _out("reasoning_error", _conf(in_retr - LO), in_kb, in_retr, in_answer)


if __name__ == "__main__":
    for p in sorted(Path("traces").glob("trace_*.json")):
        t = json.loads(p.read_text())
        d = diagnose(t)
        print("Q: {}\n   answer: {}\n   diagnosis: {}  (conf {})  {}\n".format(
            t["question"], t["answer"], d["cause"], d["confidence"], d["evidence"]))
