"""
Runtime reliability guardrail for a high-stakes (e.g. financial) RAG agent.

At RUNTIME there is NO ground-truth answer. So the guardrail scores signals
available at inference time and decides:
    PASS      -> ship the answer
    SAFE      -> the agent itself declined (no risk)
    ESCALATE  -> route to a human + reason   (the money case for finance/health/gov)
Every decision is written to an audit log.

Two independent signals, BOTH required to PASS:
    groundedness -- is the answer supported by the retrieved evidence?
    relevance    -- does the answer actually address the QUERY?

The relevance gate exists because groundedness alone happily ships a
grounded-but-irrelevant answer: a verbatim KB sentence that answers the WRONG
question scores groundedness 1.0. An earlier version of this demo committed
exactly that failure in its own audit log (a business-credit-line query
answered with the ATM-limit sentence at full trust). It is kept as the
motivating lesson: lexical grounding alone is not a safety signal -- the same
blind spot the sealed benchmark's distractor class measures (bench/BENCH_SPEC.md).
"""

import json
import datetime
from pathlib import Path
import agent

# internal policy "documents" -- stand-in for a bank's knowledge base
FINANCE_KB = [
    "Personal loan APR for prime customers is 6.9% with no origination fee.",
    "Early repayment of a personal loan incurs no penalty after the first 12 months.",
    "The daily ATM withdrawal limit for a standard checking account is 1,000 USD.",
    "Wire transfers over 10,000 USD require additional identity verification.",
    "A mortgage pre-approval is valid for 90 days from the date of issue.",
]

GROUNDED_THRESHOLD = 0.5   # min answer-in-evidence support to auto-ship
RELEVANCE_THRESHOLD = 0.34 # min query coverage: the answer must address the question
_STOP = {"the", "a", "is", "of", "for", "to", "and", "in", "no", "with", "on", "an"}


def _tokens(s):
    return {w.strip(".,?!()%$").lower() for w in s.split() if w.strip(".,?!()%$")}


def groundedness(answer, retrieved):
    """Fraction of the answer's content tokens that appear in the retrieved evidence.
    An answer with NO content tokens is unverifiable -> 0.0 (it must not auto-ship)."""
    a = _tokens(answer) - _STOP
    if not a:
        return 0.0
    ev = set().union(*[_tokens(r) for r in retrieved]) if retrieved else set()
    return len(a & ev) / len(a)


def relevance(query, answer):
    """Fraction of the query's content tokens that the answer addresses.
    Catches grounded-but-irrelevant answers (right sentence, wrong question)."""
    q = _tokens(query) - _STOP
    if not q:
        return 0.0
    return len(q & (_tokens(answer) - _STOP)) / len(q)


def assess(query, answer, retrieved):
    idk = "don't know" in answer.lower() or "do not know" in answer.lower()
    if idk:
        return {"query": query, "answer": answer, "groundedness": None, "relevance": None,
                "trust": None, "decision": "SAFE", "reason": "agent declined (no confident answer)"}

    g = round(groundedness(answer, retrieved), 2)
    rel = round(relevance(query, answer), 2)
    trust = min(g, rel)

    if not (_tokens(answer) - _STOP):
        decision, reason = "ESCALATE", "answer has no verifiable content"
    elif rel < RELEVANCE_THRESHOLD:
        decision, reason = "ESCALATE", "answer does not address the question (grounded-but-irrelevant risk)"
    elif g < GROUNDED_THRESHOLD:
        decision, reason = "ESCALATE", "answer NOT supported by retrieved evidence (hallucination risk)"
    else:
        decision, reason = "PASS", "answer addresses the question and is supported by retrieved policy"
    return {"query": query, "answer": answer, "groundedness": g, "relevance": rel,
            "trust": trust, "decision": decision, "reason": reason}


def guard(query):
    agent.KNOWLEDGE = FINANCE_KB
    trace = agent.run_agent(query)
    retrieved = next(s["retrieved"] for s in trace["steps"] if s["step"] == "retrieve")
    return assess(query, trace["answer"], retrieved)


if __name__ == "__main__":
    agent.KNOWLEDGE = FINANCE_KB
    audit = []
    print("{:<48} {:<9} {}".format("QUERY", "DECISION", "REASON"))
    print("-" * 110)

    for q in [
        "What is the personal loan APR for prime customers?",
        "Is there a penalty for early loan repayment?",
        "What is the daily ATM withdrawal limit?",
        "What is the interest rate on a business credit line?",  # off-KB: keyword retrieval returns an IRRELEVANT doc; the relevance gate must ESCALATE (this was the historical unsafe pass)
        "How do I report a lost card?",                          # nothing similar in KB -> the agent itself declines -> SAFE
    ]:
        r = guard(q)
        audit.append(dict(r, ts=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        print("{:<48} {:<9} {}".format(q[:47], r["decision"], r["reason"]))

    # Injected: what a *less reliable* agent would do -- a confident hallucination.
    # The guardrail must catch it (this is the failure that costs money in finance).
    halluc_q = "What is the APR on a business credit line?"
    r = assess(halluc_q, "The APR on a business credit line is 4.5%.", agent.retrieve(halluc_q))
    audit.append(dict(r, ts=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print("{:<48} {:<9} {}".format(halluc_q[:47] + " (halluc)", r["decision"], r["reason"]))

    # Regression probe: an all-stopword answer ("No.") is unverifiable and must NOT auto-ship.
    probe_q = "Is there an overdraft fee?"
    r = assess(probe_q, "No.", agent.retrieve(probe_q))
    audit.append(dict(r, ts=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print("{:<48} {:<9} {}".format(probe_q[:40] + ' (probe "No.")', r["decision"], r["reason"]))

    Path("audit_log.json").write_text(json.dumps(audit, indent=2))
    esc = sum(1 for a in audit if a["decision"] == "ESCALATE")
    print("\n%d of %d routed to a human BEFORE reaching the customer.  Audit trail -> audit_log.json" % (esc, len(audit)))
