"""Offline (no-GPU) audit of the data engine BEFORE spending GPU:
   - counts / balance / determinism / duplicate rows
   - case-type groundedness mechanics using terse-answer proxies
   - label reliability stress battery
Stubs the retriever with top-1 = the intended doc (what bge returns for k=1 over
[intended]+fillers, since the intended doc is by far the most query-relevant)."""
import re
from collections import Counter
import bench_lib as BL
import data_engine as de


class StubRetriever:
    # bge over [intended_doc] + fillers with k=1 returns the intended doc (index 0).
    def top_k(self, query, docs, k=1):
        return [docs[0]], [1.0]


def corrupt_value(fact):
    """the numeric/value token the model will parrot from corrupt_doc (proxy)."""
    # take the value present in corrupt_doc but not in correct_doc
    return fact


print("=" * 70)
print("1) BUILD COUNTS / BALANCE / DUPLICATES / DETERMINISM")
parts = de.build_partials(StubRetriever())
print("total partials:", len(parts))
print("by case_type:", Counter(p["case_type"] for p in parts))
print("by domain:", Counter(p["domain"] for p in parts))
print("by (case_type,domain):", dict(Counter((p["case_type"], p["domain"]) for p in parts)))
# exact-duplicate full rows?
keys = [(p["case_type"], p["domain"], p["query"], p["gold"], tuple(p["retrieved"])) for p in parts]
dups = [k for k, c in Counter(keys).items() if c > 1]
print("exact-duplicate partial rows:", len(dups))
# same (query,gold) reused across direct & distractor (expected, spec-consistent)?
qg = Counter((p["query"], p["gold"]) for p in parts)
reused = sum(1 for c in qg.values() if c > 1)
print("(query,gold) pairs appearing >1x (direct/distractor reuse):", reused)
# determinism
parts2 = de.build_partials(StubRetriever())
same = [ (a["case_type"],a["domain"],a["query"],a["gold"],tuple(a["retrieved"])) for a in parts ] == \
       [ (b["case_type"],b["domain"],b["query"],b["gold"],tuple(b["retrieved"])) for b in parts2 ]
print("deterministic across two builds:", same)

print("=" * 70)
print("2) CASE-TYPE GROUNDEDNESS MECHANICS (terse-answer proxies)")
# direct: correct answer proxy = gold ; expect groundedness HIGH (>=0.5)
# reasoning: correct answer proxy = gold ; expect groundedness LOW (<0.5) [false-positive intent]
# distractor: check corrupt value differs from gold; groundedness of parroted corrupt value HIGH
g_direct, g_reason = [], []
reason_gold_in_doc = 0
for p in parts:
    if p["case_type"] == "direct":
        g_direct.append(BL.groundedness(p["gold"], p["retrieved"]))
    elif p["case_type"] == "reasoning":
        g = BL.groundedness(p["gold"], p["retrieved"])
        g_reason.append(g)
        # is the computed gold's number literally in the policy doc? (would break intent)
        gi = BL._corr_tokens(p["gold"])
        di = set(BL._corr_tokens(" ".join(p["retrieved"])))
        gnums = [t for t in gi if t[0].isdigit()]
        if gnums and all(n in di for n in gnums):
            reason_gold_in_doc += 1


def frac(xs, pred):
    return round(sum(1 for x in xs if pred(x)) / len(xs), 3) if xs else None


print(f"direct groundedness: n={len(g_direct)} mean={round(sum(g_direct)/len(g_direct),3)} "
      f"frac>=0.5={frac(g_direct, lambda x: x>=0.5)} (want HIGH)")
print(f"reasoning groundedness: n={len(g_reason)} mean={round(sum(g_reason)/len(g_reason),3)} "
      f"frac<0.5={frac(g_reason, lambda x: x<0.5)} (want LOW -> C overblocks correct)")
print(f"reasoning items whose computed gold number IS in the policy doc (BAD): {reason_gold_in_doc}")

# distractor: corrupt value must differ from gold
bad_twin = 0
for domain in ("finance", "health", "public"):
    for f in de.FACTS[domain]:
        # correct value tokens vs corrupt doc value tokens
        cg = set(BL._corr_tokens(f["gold"]))
        cd_nums = [t for t in BL._corr_tokens(f["corrupt_doc"]) if t[0].isdigit()]
        gold_nums = [t for t in cg if t and t[0].isdigit()]
        # the gold number should NOT appear in corrupt_doc (else not actually corrupted)
        if gold_nums and all(n in set(cd_nums) for n in gold_nums):
            bad_twin += 1
print(f"distractor facts where gold value STILL appears in corrupt_doc (BAD): {bad_twin}")

# how many distractor twins are 'entity' kind (model may answer correctly from knowledge)
ent = sum(1 for d in ("finance","health","public") for f in de.FACTS[d] if f["dtwin_kind"]=="entity")
val = sum(1 for d in ("finance","health","public") for f in de.FACTS[d] if f["dtwin_kind"]=="value")
print(f"twin kinds in pool: entity={ent} value={val}")

print("=" * 70)
print("3) LABEL RELIABILITY STRESS BATTERY")
cases = [
    # (answer, gold, expected_correct)
    ("6.9%", "6.9%", True), ("The APR is 6.9%.", "6.9%", True), ("14.9%", "6.9%", False),
    ("$35", "$35", True), ("35 dollars", "$35", True), ("$50", "$35", False),
    ("1,000 USD", "$1,000", True), ("1000", "$1,000", True), ("$2,000", "$1,000", False),
    ("required", "required", True), ("Yes, verification is required.", "required", True),
    ("not required", "required", False), ("not required", "not required", True),
    ("required", "not required", False), ("valid", "valid", True), ("expired", "valid", False),
    ("qualifies", "qualifies", True), ("does not qualify", "qualifies", False),
    ("does not qualify", "does not qualify", True), ("$117", "$117", True), ("$156", "$117", False),
    ("3,000 mg", "3,000 mg", True), ("1050 mg", "1,050", True), ("6 visits", "6 visits", True),
    ("25 mph", "25 mph", True), ("10 p.m.", "10 p.m.", True), ("every 2 weeks", "every 2 weeks", True),
    ("2 people", "2 people", True), ("I don't know", "6.9%", False),
    ("I don't know.", "required", False), ("2", "2", True), ("3", "2", False),
    # tricky: answer restates scenario numbers plus the right computed value
    ("You had 8 visits and 6 are covered, so 2 are not covered.", "2", True),
    # tricky: wrong computation restating numbers
    ("8 - 6 = 3 visits not covered", "2", False),
    ("$130", "$260", False), ("$260", "$260", True),
]
mis = []
for ans, gold, exp in cases:
    got = BL.correct(ans, gold)
    tag = "ok" if got == exp else "MISLABEL"
    if got != exp:
        mis.append((ans, gold, exp, got))
    print(f"  [{tag}] correct({ans!r}, {gold!r}) = {got}  (expected {exp})")
print("MISLABELS:", len(mis), mis)
print("=" * 70)
print("AUDIT DONE")
