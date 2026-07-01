"""
Shared primitives for the Week-2 data engine (SCALED build of the sealed A-E bench).

Reuses the *proven* prototype patterns so we don't re-pay bug-fixes:
  - groundedness()  : copied VERBATIM from guardrail.py (token-overlap of answer
                      content found in retrieved evidence).  This is the exact float
                      that bench/run_bench.py's system C thresholds on.
  - DECLINE list    : expanded per B300_AGENT_BRIEF.md ("expanded DECLINE/abstain list").
  - correctness     : STRICT token-overlap (NOT semantic similarity), per START_HERE.md
                      and the brief.  Numeric golds -> exact value-token match; categorical
                      golds -> >=0.6 token coverage with a negation guard.

Nothing here imports torch; the LLM/embedding wrappers live in real_rag.py so this
module stays importable for offline label auditing.
"""
import re

# ---------------------------------------------------------------------------
# groundedness  (VERBATIM from guardrail.py — must match the sealed run_bench.py input)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# DECLINE / abstain detection  (expanded list — brief lesson)
# ---------------------------------------------------------------------------
DECLINE_PHRASES = (
    "i don't know", "i do not know", "don't know", "do not know",
    "not sure", "cannot determine", "can't determine", "unable to determine",
    "not specified", "not mentioned", "not stated", "not provided",
    "no information", "insufficient information", "not enough information",
    "cannot answer", "can't answer", "unable to answer", "not available",
    "context does not", "context doesn't", "not in the context", "unclear",
)


def is_decline(answer):
    a = (answer or "").lower()
    return any(p in a for p in DECLINE_PHRASES)


# ---------------------------------------------------------------------------
# correctness  (STRICT token-overlap, number-normalized — NOT semantic similarity)
# ---------------------------------------------------------------------------
_CORR_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "of", "for", "to", "and", "in",
    "with", "on", "at", "by", "it", "this", "that", "be", "as", "or", "from",
    "yes", "you", "your", "will", "would", "there", "does", "do", "per", "usd",
    "dollars", "dollar", "answer", "value",
}
_NEG = {"not", "no", "never", "without", "exempt", "cannot", "can't", "isn't",
        "doesn't", "don't", "neither", "none", "ineligible", "denied"}


def _corr_tokens(s):
    """Lowercase; strip thousands-commas inside numbers; %->percent; keep words+numbers."""
    s = s.lower()
    s = re.sub(r"(?<=\d),(?=\d)", "", s)          # 1,000 -> 1000
    s = s.replace("%", " percent ").replace("$", " ")
    toks = re.findall(r"[a-z']+|\d+\.?\d*", s)
    return [t for t in toks if t]


def correct(answer, gold, threshold=0.6):
    """
    STRICT lexical correctness of `answer` against controlled `gold`.
      - numeric gold  -> every numeric value-token of gold must appear in the answer
                         (exact value match; this is the decisive signal for value facts).
      - categorical   -> >= `threshold` of gold's content tokens present in the answer,
                         AND polarity (negation) must agree.
    Declines never count as correct.
    """
    if is_decline(answer):
        return False
    gtoks = _corr_tokens(gold)
    if not gtoks:
        return False
    gnums = [t for t in gtoks if t[0].isdigit()]
    aset = set(_corr_tokens(answer))
    if gnums:
        return all(n in aset for n in gnums)
    # categorical
    gcontent = [t for t in gtoks if t not in _CORR_STOP] or gtoks
    cover = sum(1 for t in gcontent if t in aset) / len(gcontent)
    if cover < threshold:
        return False
    g_neg = any(t in _NEG for t in gtoks)
    a_neg = any(t in _NEG for t in _corr_tokens(answer))
    if g_neg != a_neg:
        return False
    return True


if __name__ == "__main__":
    # sanity checks against the sealed sample_bench.jsonl semantics
    assert correct("6.9%", "6.9%")
    assert correct("The APR is 6.9%.", "6.9%")
    assert not correct("14.9%", "6.9%")               # distractor wrong-but-grounded
    assert correct("1,000 USD", "1,000")
    assert correct("1000", "1,000")
    assert correct("Yes, verification is required.", "required")
    assert not correct("No, it is not required.", "required")   # negation guard
    assert correct("It is not required.", "not required")
    assert not correct("Yes, it is required.", "not required")
    assert correct("2", "2") and not correct("3", "2")
    assert not correct("I don't know.", "6.9%")       # decline
    # groundedness direction
    assert groundedness("14.9%", ["Subprime personal loan APR is 14.9%."]) == 1.0
    assert groundedness("2 visits", ["The plan covers up to 6 visits per year."]) < 0.6
    print("bench_lib self-tests PASSED")
