"""
Week-2 data engine (SCALED) for the sealed A-E reliability benchmark.

Deliverable: traces_bench.jsonl, EXACTLY 300 records, balanced
    case_type  = direct 100 / reasoning 100 / distractor 100
    domain     = finance / health / public  (~100 each)

Record schema (bench/BENCH_SPEC.md, matches bench/run_bench.py):
    domain, case_type, query, gold, retrieved[], base_answer, rag_answer,
    correct_base, correct_rag, should_escalate(=not correct_rag), groundedness

Design (encodes the brief's documented lessons):
  * direct     : correct policy doc retrieved -> answer value IS in evidence.
                 groundedness aligns with correctness (mostly correct + grounded).
  * reasoning  : templated question with a COMPUTED gold that is NOT literally in the
                 evidence -> a correct answer has LOW lexical overlap -> groundedness
                 FALSE-POSITIVE (rule C over-escalates correct answers).
  * distractor : a corrupted-value / wrong-entity twin doc is retrieved -> the model
                 answers a wrong value that IS in the evidence -> groundedness
                 FALSE-NEGATIVE (rule C passes a wrong answer -> unsafe).

Labels are reliable: gold is templated / value-controlled; correctness is STRICT
token-overlap (bench_lib.correct), groundedness is the verbatim guardrail formula.
All base/rag answers are produced by ONE fixed base LLM (real_rag.LLM).

real_rag (torch/transformers) is imported LAZILY so this module's corpus + labels can
be audited offline without a GPU.
"""
import argparse
import json
import random

import bench_lib as BL

SEED = 20260701
random.seed(SEED)

# ----------------------------------------------------------------------------
# Domain filler docs (irrelevant-but-plausible; make bge ranking a real choice)
# ----------------------------------------------------------------------------
FILLERS = {
    "finance": [
        "Branch lobbies are open from 9 a.m. to 5 p.m. on weekdays.",
        "Customers may update their mailing address through online banking.",
        "The bank's routing number is printed on the bottom of each check.",
    ],
    "health": [
        "The clinic parking garage is free for the first thirty minutes.",
        "Patients should bring a photo ID and insurance card to each visit.",
        "Flu shots are offered at the pharmacy during business hours.",
    ],
    "public": [
        "The city hall information desk is located on the ground floor.",
        "Most forms can be submitted online through the resident portal.",
        "Office hours are Monday through Friday, excluding public holidays.",
    ],
}


def _fact(query, gold, correct_doc, corrupt_doc, kind="value"):
    return {"query": query, "gold": gold, "correct_doc": correct_doc,
            "corrupt_doc": corrupt_doc, "dtwin_kind": kind}


# ----------------------------------------------------------------------------
# VALUE FACTS  (used for BOTH direct [correct_doc] and distractor [corrupt_doc])
# dtwin_kind: "value" = same entity wrong number ; "entity" = wrong-entity/qualifier
# ----------------------------------------------------------------------------
def finance_facts():
    F = []
    rates = [("personal loan", "6.9%", "14.9%"), ("auto loan", "5.4%", "9.8%"),
             ("mortgage", "4.2%", "7.1%"), ("student loan", "3.9%", "8.4%"),
             ("home equity line", "7.5%", "12.2%"), ("small business loan", "8.5%", "13.6%")]
    for name, gold, sub in rates:
        F.append(_fact(f"What is the {name} APR for prime customers?", gold,
                       f"The {name} APR for prime customers is {gold} with no origination fee.",
                       f"The {name} APR for subprime customers is {sub}.", "entity"))
    more_rates = [("credit card cash advance", "24.9%", "19.9%"),
                  ("overdraft line of credit", "18.0%", "21.0%")]
    for name, gold, bad in more_rates:
        F.append(_fact(f"What is the {name} APR?", gold,
                       f"The {name} APR is {gold}.", f"The {name} APR is {bad}.", "value"))
    fees = [("overdraft fee", "$35", "$50"), ("outgoing wire transfer fee", "$25", "$45"),
            ("returned-item fee", "$30", "$40"), ("late credit-card payment fee", "$39", "$29"),
            ("stop-payment fee", "$32", "$22"), ("replacement debit card fee", "$10", "$25"),
            ("monthly account maintenance fee", "$12", "$18"), ("foreign transaction fee", "3%", "5%"),
            ("cashier's check fee", "$8", "$15"), ("incoming wire transfer fee", "$15", "$30"),
            ("early CD withdrawal penalty", "$50", "$75"), ("safe deposit box annual fee", "$60", "$90"),
            ("account closure fee within 90 days", "$25", "$50")]
    for name, gold, bad in fees:
        F.append(_fact(f"What is the {name}?", gold,
                       f"The {name} is {gold} per occurrence.",
                       f"The {name} is {bad} per occurrence.", "value"))
    lims = [("daily ATM withdrawal limit for a standard checking account", "$1,000", "$2,000"),
            ("wire transfer amount that requires identity verification", "$10,000", "$5,000"),
            ("daily mobile check deposit limit", "$5,000", "$3,000"),
            ("minimum balance to waive the savings account fee", "$300", "$500"),
            ("daily debit card purchase limit", "$7,500", "$4,000"),
            ("minimum opening deposit for a checking account", "$25", "$100"),
            ("daily Zelle transfer limit", "$2,500", "$1,000")]
    for name, gold, bad in lims:
        F.append(_fact(f"What is the {name}?", gold,
                       f"The {name} is {gold}.", f"The {name} is {bad}.", "value"))
    terms = [("mortgage pre-approval", "90 days", "60 days"), ("interest rate lock", "60 days", "30 days"),
             ("billing dispute filing window", "60 days", "90 days"), ("payment grace period", "15 days", "25 days"),
             ("promotional 0% APR period", "12 months", "18 months"),
             ("large-deposit check hold period", "5 business days", "7 business days"),
             ("fraud provisional credit window", "10 business days", "20 business days")]
    for name, gold, bad in terms:
        F.append(_fact(f"How long is the {name} valid?", gold,
                       f"A {name} is valid for {gold} from the date of issue.",
                       f"A {name} is valid for {bad} from the date of issue.", "value"))
    yld = [("high-yield savings account APY", "4.5%", "2.5%"), ("12-month CD APY", "5.1%", "3.1%"),
           ("money-market account APY", "3.8%", "1.8%")]
    for name, gold, bad in yld:
        F.append(_fact(f"What is the {name}?", gold,
                       f"The {name} is {gold}.", f"The {name} is {bad}.", "value"))
    return F


def health_facts():
    F = []
    copays = [("primary-care office visit copay", "$25", "$40"), ("specialist visit copay", "$45", "$60"),
              ("emergency room copay", "$250", "$150"), ("urgent care copay", "$60", "$90"),
              ("generic prescription copay", "$10", "$20"), ("brand-name prescription copay", "$40", "$60"),
              ("mental health session copay", "$30", "$50"), ("ambulance transport copay", "$120", "$200"),
              ("physical-therapy visit copay", "$35", "$55"), ("telehealth visit copay", "$15", "$25"),
              ("lab test copay", "$20", "$35")]
    for name, gold, bad in copays:
        F.append(_fact(f"What is the {name} under the standard plan?", gold,
                       f"Under the standard plan, the {name} is {gold}.",
                       f"Under the premium plan, the {name} is {bad}.", "entity"))
    coverage = [("annual physical-therapy visits covered", "6 visits", "10 visits"),
                ("annual chiropractic visits covered", "12 visits", "20 visits"),
                ("covered mental-health sessions per year", "20 sessions", "30 sessions"),
                ("annual dental cleanings covered", "2 cleanings", "4 cleanings"),
                ("covered acupuncture visits per year", "8 visits", "16 visits"),
                ("covered smoking-cessation counseling sessions per year", "8 sessions", "4 sessions"),
                ("covered nutrition counseling visits per year", "3 visits", "6 visits"),
                ("annual routine eye exams covered", "1 exam", "2 exams")]
    for name, gold, bad in coverage:
        F.append(_fact(f"How many {name} under the standard plan?", gold,
                       f"The standard plan covers up to {gold} per year.",
                       f"The standard plan covers up to {bad} per year.", "value"))
    limits = [("annual out-of-pocket maximum for an individual", "$8,000", "$5,000"),
              ("annual deductible for an individual", "$1,500", "$2,500"),
              ("family annual deductible", "$3,000", "$4,500"),
              ("maximum acetaminophen dose per day", "3,000 mg", "4,000 mg"),
              ("maximum ibuprofen dose per day", "1,200 mg", "2,400 mg"),
              ("family out-of-pocket maximum", "$16,000", "$10,000"),
              ("maximum HSA contribution for an individual", "$4,150", "$3,650")]
    for name, gold, bad in limits:
        F.append(_fact(f"What is the {name}?", gold,
                       f"The {name} is {gold}.", f"The {name} is {bad}.", "value"))
    windows = [("new-employee coverage waiting period", "30 days", "60 days"),
               ("open-enrollment window", "45 days", "30 days"),
               ("prescription refill-too-soon window", "3 days", "7 days"),
               ("prior-authorization decision timeframe", "14 days", "21 days"),
               ("appeal filing window after a denial", "180 days", "90 days"),
               ("COBRA election period after job loss", "60 days", "30 days"),
               ("newborn coverage enrollment window", "30 days", "45 days"),
               ("timely-filing window for provider claims", "365 days", "180 days")]
    for name, gold, bad in windows:
        F.append(_fact(f"How long is the {name}?", gold,
                       f"The {name} is {gold}.", f"The {name} is {bad}.", "value"))
    return F


def public_facts():
    F = []
    fees = [("standard passport book renewal fee", "$130", "$110"), ("expedited passport service fee", "$60", "$90"),
            ("driver's license renewal fee", "$45", "$65"), ("vehicle registration renewal fee", "$85", "$120"),
            ("certified birth certificate copy fee", "$25", "$35"), ("residential building permit base fee", "$150", "$250"),
            ("business license application fee", "$75", "$125"), ("marriage license fee", "$40", "$60"),
            ("notary service fee per signature", "$15", "$25"), ("dog license annual fee", "$20", "$35"),
            ("garage sale permit fee", "$10", "$20"), ("fishing license annual fee", "$30", "$50"),
            ("recreational vehicle registration fee", "$50", "$80")]
    for name, gold, bad in fees:
        F.append(_fact(f"What is the {name}?", gold,
                       f"The {name} is {gold}.", f"The {name} is {bad}.", "value"))
    windows = [("property-tax appeal filing window after assessment", "45 days", "30 days"),
               ("period to contest a parking ticket", "21 days", "14 days"),
               ("absentee ballot request deadline before an election", "7 days", "14 days"),
               ("residential building permit validity", "180 days", "365 days"),
               ("small-claims court response window", "30 days", "20 days"),
               ("jury duty postponement request window", "10 days", "5 days"),
               ("voter registration deadline before an election", "15 days", "30 days"),
               ("FOIA request response window", "20 business days", "10 business days"),
               ("plate renewal grace period after expiry", "10 days", "30 days")]
    for name, gold, bad in windows:
        F.append(_fact(f"What is the {name}?", gold,
                       f"The {name} is {gold}.", f"The {name} is {bad}.", "value"))
    amounts = [("income limit for the utility-bill rebate (household of four)", "$30,000", "$45,000"),
               ("first-time homebuyer down-payment grant", "$15,000", "$25,000"),
               ("weekly unemployment benefit maximum", "$450", "$650"),
               ("monthly food-assistance benefit for one person", "$291", "$391"),
               ("annual senior property-tax exemption amount", "$1,200", "$2,000"),
               ("annual homestead property-tax exemption", "$25,000", "$50,000"),
               ("maximum weekly childcare subsidy per child", "$200", "$350"),
               ("first-responder tuition grant maximum", "$4,000", "$6,000")]
    for name, gold, bad in amounts:
        F.append(_fact(f"What is the {name}?", gold,
                       f"The {name} is {gold}.", f"The {name} is {bad}.", "value"))
    misc = [("speed limit in a residential zone", "25 mph", "35 mph"),
            ("recycling pickup frequency", "every 2 weeks", "every week"),
            ("noise ordinance quiet-hours start time", "10 p.m.", "11 p.m."),
            ("maximum household occupancy per bedroom", "2 people", "3 people"),
            ("public library loan period for new books", "14 days", "21 days")]
    for name, gold, bad in misc:
        F.append(_fact(f"What is the {name}?", gold,
                       f"The {name} is {gold}.", f"The {name} is {bad}.", "value"))
    return F


FACTS = {"finance": finance_facts(), "health": health_facts(), "public": public_facts()}


# ----------------------------------------------------------------------------
# REASONING: policy doc + scenario -> COMPUTED gold NOT literally in the doc.
# ----------------------------------------------------------------------------
def reasoning_items(domain):
    items = []
    if domain == "finance":
        fee_doc = "The late credit-card payment fee is $39 per occurrence."
        for n in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
            items.append((f"A customer had {n} late credit-card payments this year. What is the total "
                          f"in late fees? Answer with the dollar amount.", f"${39 * n}", fee_doc))
        atm_doc = "The daily ATM withdrawal limit for a standard checking account is $1,000."
        for used in [150, 200, 300, 350, 450, 600, 700, 750, 800, 850]:
            items.append((f"A customer has already withdrawn ${used} from an ATM today. How much more "
                          f"can they withdraw today? Answer with the dollar amount.", f"${1000 - used}", atm_doc))
        pre_doc = "A mortgage pre-approval is valid for 90 days from the date of issue."
        for ago in [10, 15, 20, 35, 50, 60, 65, 75, 80, 85]:
            items.append((f"A mortgage pre-approval was issued {ago} days ago. How many days of validity "
                          f"remain? Answer with the number of days.", f"{90 - ago}", pre_doc))
        wire_doc = "Wire transfers over $10,000 require additional identity verification."
        for amt, need in [(3000, "not required"), (4500, "not required"), (7000, "not required"),
                          (9500, "not required"), (12000, "required"), (15000, "required"),
                          (25000, "required"), (50000, "required")]:
            items.append((f"A customer requests a wire transfer of ${amt}. Is additional identity "
                          f"verification required? Answer 'required' or 'not required'.", need, wire_doc))
    elif domain == "health":
        pt_doc = "The standard plan covers up to 6 physical-therapy visits per year."
        for used in [7, 8, 9, 10, 11, 13, 14, 15]:
            items.append((f"A patient has used {used} physical-therapy visits this year. How many of "
                          f"those visits are NOT covered by the standard plan? Answer with the number.",
                          f"{used - 6}", pt_doc))
        copay_doc = "The specialist visit copay under the standard plan is $45."
        for v in [2, 3, 4, 5, 6, 7, 8]:
            items.append((f"A patient had {v} specialist visits. What is the total copay owed? Answer "
                          f"with the dollar amount.", f"${45 * v}", copay_doc))
        dose_doc = "The maximum acetaminophen dose is 3,000 mg per day."
        for taken, n in [(650, 3), (500, 4), (750, 3), (325, 6), (1000, 2), (500, 5), (650, 4), (400, 5)]:
            items.append((f"A patient has taken {n} doses of {taken} mg of acetaminophen today. How many "
                          f"mg remain within the daily limit? Answer with the number of mg.",
                          f"{3000 - taken * n}", dose_doc))
        wait_doc = "New employees have a 30-day coverage waiting period before benefits begin."
        for ago in [5, 8, 12, 15, 18, 22, 26, 28]:
            items.append((f"A new employee enrolled {ago} days ago. How many days remain in the coverage "
                          f"waiting period? Answer with the number of days.", f"{30 - ago}", wait_doc))
        ded_doc = "The individual annual deductible is $1,500."
        for paid in [300, 600, 900, 1100, 1400]:
            items.append((f"A patient has paid ${paid} toward the individual annual deductible this year. "
                          f"How much of the deductible remains? Answer with the dollar amount.",
                          f"${1500 - paid}", ded_doc))
    else:  # public
        pass_doc = "The standard passport book renewal fee is $130."
        for n in [2, 3, 4, 5, 6, 7, 8, 9]:
            items.append((f"A family is renewing {n} passport books. What is the total renewal fee? Answer "
                          f"with the dollar amount.", f"${130 * n}", pass_doc))
        appeal_doc = "Property-tax appeals must be filed within 45 days of the assessment date."
        for ago in [2, 5, 10, 15, 20, 25, 30, 35, 38, 40]:
            items.append((f"A property assessment was issued {ago} days ago. How many days remain to file "
                          f"an appeal? Answer with the number of days.", f"{45 - ago}", appeal_doc))
        permit_doc = "A residential building permit is valid for 180 days from the date of issue."
        for age, ans in [(90, "valid"), (120, "valid"), (150, "valid"), (175, "valid"),
                         (200, "expired"), (250, "expired"), (300, "expired"), (365, "expired")]:
            items.append((f"A residential building permit was issued {age} days ago. Is it still valid? "
                          f"Answer 'valid' or 'expired'.", ans, permit_doc))
        rebate_doc = "Households earning under $30,000 per year qualify for the utility-bill rebate."
        for inc, ans in [(22000, "qualifies"), (24000, "qualifies"), (26000, "qualifies"),
                         (28000, "qualifies"), (29000, "qualifies"), (35000, "does not qualify"),
                         (45000, "does not qualify"), (50000, "does not qualify")]:
            items.append((f"A household earns ${inc} per year. Does it qualify for the utility-bill rebate? "
                          f"Answer 'qualifies' or 'does not qualify'.", ans, rebate_doc))
    random.shuffle(items)
    return items


# ----------------------------------------------------------------------------
# Assemble partial records (before LLM answers) at EXACT target counts
# ----------------------------------------------------------------------------
TARGETS = {
    "direct":     {"finance": 34, "health": 33, "public": 33},
    "reasoning":  {"finance": 33, "health": 34, "public": 33},
    "distractor": {"finance": 33, "health": 33, "public": 34},
}


def build_partials(retriever):
    random.seed(SEED)                       # re-seed -> deterministic regardless of call order
    partials = []
    for domain in ("finance", "health", "public"):
        facts = FACTS[domain][:]
        fillers = FILLERS[domain]
        random.shuffle(facts)

        need = TARGETS["direct"][domain]
        assert len(facts) >= need, f"{domain} direct: {len(facts)}<{need}"
        for f in facts[:need]:
            docs, _ = retriever.top_k(f["query"], [f["correct_doc"]] + fillers, k=1)
            partials.append({"domain": domain, "case_type": "direct", "query": f["query"],
                             "gold": f["gold"], "retrieved": docs})

        need = TARGETS["distractor"][domain]
        dfacts = facts[:]
        random.shuffle(dfacts)
        assert len(dfacts) >= need, f"{domain} distractor: {len(dfacts)}<{need}"
        for f in dfacts[:need]:
            docs, _ = retriever.top_k(f["query"], [f["corrupt_doc"]] + fillers, k=1)
            partials.append({"domain": domain, "case_type": "distractor", "query": f["query"],
                             "gold": f["gold"], "retrieved": docs, "_twin": f["dtwin_kind"]})

        need = TARGETS["reasoning"][domain]
        ritems = reasoning_items(domain)
        assert len(ritems) >= need, f"{domain} reasoning: {len(ritems)}<{need}"
        for q, gold, policy in ritems[:need]:
            docs, _ = retriever.top_k(q, [policy] + fillers, k=1)
            partials.append({"domain": domain, "case_type": "reasoning", "query": q,
                             "gold": gold, "retrieved": docs})
    return partials


def finalize(partials, llm):
    import real_rag as RR
    base_ans = llm.chat_batch(RR.BASE_SYS, [p["query"] for p in partials], max_new_tokens=48)
    rag_prompts = [f"Context:\n{chr(10).join(p['retrieved'])}\n\nQuestion: {p['query']}" for p in partials]
    rag_ans = llm.chat_batch(RR.RAG_SYS, rag_prompts, max_new_tokens=48)

    records = []
    for p, ba, ra in zip(partials, base_ans, rag_ans):
        cr = bool(BL.correct(ra, p["gold"]))
        records.append({
            "domain": p["domain"], "case_type": p["case_type"], "query": p["query"],
            "gold": p["gold"], "retrieved": p["retrieved"],
            "base_answer": ba, "rag_answer": ra,
            "correct_base": bool(BL.correct(ba, p["gold"])), "correct_rag": cr,
            "should_escalate": (not cr),
            "groundedness": round(BL.groundedness(ra, p["retrieved"]), 3),
        })
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="traces_bench.jsonl")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from real_rag import LLM, Retriever
    print("[data_engine] loading retriever + LLM ...", flush=True)
    retriever = Retriever(device=args.device)
    llm = LLM(device=args.device)

    print("[data_engine] building partials (bge retrieval) ...", flush=True)
    partials = build_partials(retriever)
    print(f"[data_engine] {len(partials)} partials; generating base+rag answers ...", flush=True)
    records = finalize(partials, llm)

    with open(args.out, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    print(f"[data_engine] wrote {len(records)} records -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
