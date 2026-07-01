"""
Runtime reliability guardrail for a high-stakes (e.g. financial) RAG agent.

At RUNTIME there is NO ground-truth answer. So the guardrail scores TRUST from
signals available at inference time -- mainly GROUNDEDNESS (is the answer actually
supported by the retrieved evidence?) -- and decides:
    PASS      -> ship the answer
    SAFE      -> the agent itself declined (no risk)
    ESCALATE  -> route to a human + reason   (the money case for finance/health/gov)
Every decision is written to an audit log for compliance.

This is the sellable product shape: catch ungrounded/risky answers BEFORE they
reach the customer, with a human-in-the-loop and an audit trail.
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

GROUNDED_THRESHOLD = 0.5  # min answer-in-evidence support to auto-ship
_STOP = {"the", "a", "is", "of", "for", "to", "and", "in", "no", "with", "on", "an"}


def _tokens(s):
    return {w.strip(".,?!()%$").lower() for w in s.split() if w.strip(".,?!()%$")}


def groundedness(answer, retrieved):
    """Fraction of the answer's content tokens that appear in the retrieved evidence."""
    a = _tokens(answer) - _STOP
    if not a:
        return 1.0
    ev = set().union(*[_tokens(r) for r in retrieved]) if retrieved else set()
    return len(a & ev) / len(a)


def assess(query, answer, retrieved):
    idk = "don't know" in answer.lower() or "do not know" in answer.lower()
    g = groundedness(answer, retrieved)
    if idk:
        decision, reason, trust = "SAFE", "agent declined (no confident answer)", 1.0
    elif g >= GROUNDED_THRESHOLD:
        decision, reason, trust = "PASS", "answer supported by retrieved policy", round(g, 2)
    else:
        decision, reason, trust = "ESCALATE", "answer NOT supported by retrieved evidence (hallucination risk)", round(g, 2)
    return {"query": query, "answer": answer, "trust": trust, "decision": decision, "reason": reason}


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
        "What is the interest rate on a business credit line?",  # not in KB -> agent declines (SAFE)
    ]:
        r = guard(q)
        audit.append(dict(r, ts=datetime.datetime.utcnow().isoformat()))
        print("{:<48} {:<9} {}".format(q[:47], r["decision"], r["reason"]))

    # Injected: what a *less reliable* agent would do -- a confident hallucination.
    # The guardrail must catch it (this is the failure that costs money in finance).
    halluc_q = "What is the APR on a business credit line?"
    r = assess(halluc_q, "The APR on a business credit line is 4.5%.", agent.retrieve(halluc_q))
    audit.append(dict(r, ts=datetime.datetime.utcnow().isoformat()))
    print("{:<48} {:<9} {}".format(halluc_q[:47] + " (halluc)", r["decision"], r["reason"]))

    Path("audit_log.json").write_text(json.dumps(audit, indent=2))
    esc = sum(1 for a in audit if a["decision"] == "ESCALATE")
    print("\n%d of %d routed to a human BEFORE reaching the customer.  Audit trail -> audit_log.json" % (esc, len(audit)))
