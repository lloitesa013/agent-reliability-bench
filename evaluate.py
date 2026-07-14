"""
Evaluate the diagnoser against ground-truth causes (labeled in agent.CASES and
hard_cases.CRAFTED).

Reported numbers (n is tiny -- this is a pipeline smoke test, not a benchmark;
quantitative claims live in bench/ + RESULTS.md):
  - strict accuracy: predicted cause == ground-truth label
  - lenient accuracy: additionally counts cause=correct_abstention as correct
    when gt=missing_knowledge (the agent DID correctly say "I don't know" to an
    out-of-KB question; the two labels describe the same event at different
    granularity -- counted separately so the equivalence is visible, not hidden)
  - abstention: rate, plus precision/recall against gt=uncertain
  - high-confidence check: when the heuristic score was >=0.7, was it right?
    (NOT calibration -- see diagnose.py's honesty note)
"""

import json
from pathlib import Path
from diagnose import diagnose


def main():
    traces = sorted(Path("traces").glob("trace_*.json"))
    if not traces:
        print("No traces. Run `python agent.py` and `python hard_cases.py` first.")
        return

    total = strict = lenient = 0
    abstained = abstain_hits = gt_uncertain = 0
    hi_conf = hi_conf_correct = 0
    for p in traces:
        t = json.loads(p.read_text())
        gt = t.get("ground_truth_cause", "?")
        d = diagnose(t)
        total += 1
        gt_uncertain += int(gt == "uncertain")
        if d["cause"] == "uncertain":
            abstained += 1
            abstain_hits += int(gt == "uncertain")
            verdict = "ABSTAIN" + (" (correct)" if gt == "uncertain" else " (gt={})".format(gt))
        else:
            ok_strict = d["cause"] == gt
            # explicit label equivalence, see module docstring
            ok_lenient = ok_strict or (d["cause"] == "correct_abstention" and gt == "missing_knowledge")
            strict += int(ok_strict)
            lenient += int(ok_lenient)
            verdict = "correct" if ok_strict else ("correct (lenient)" if ok_lenient else "WRONG (gt={})".format(gt))
            if d["confidence"] >= 0.7:
                hi_conf += 1
                hi_conf_correct += int(ok_lenient)
        print("{:<46} -> {:<18} conf {:.2f}  [{}]".format(t["question"][:45], d["cause"], d["confidence"], verdict))

    answered = total - abstained
    print("\n--- summary (n={} traces: smoke test, not a benchmark) ---".format(total))
    if answered:
        print("accuracy strict / lenient:          {}/{} / {}/{}".format(strict, answered, lenient, answered))
    else:
        print("accuracy: n/a (all abstained)")
    print("abstention rate:                    {}/{}".format(abstained, total))
    if abstained:
        print("abstain precision (gt=uncertain):   {}/{}".format(abstain_hits, abstained))
    if gt_uncertain:
        print("abstain recall (gt=uncertain):      {}/{}".format(abstain_hits, gt_uncertain))
    if hi_conf:
        print("high-conf check (conf>=0.7 right):  {}/{}  (heuristic score, not calibration)".format(hi_conf_correct, hi_conf))


if __name__ == "__main__":
    main()
