"""
Evaluate the diagnoser against ground-truth causes (labeled in agent.CASES).

Reports the three numbers that make this a *trustworthy* tool, not just a guesser:
  - accuracy on the cases it actually answered
  - abstention rate (how often it correctly stayed silent)
  - calibration: when it was confident (>=0.7), was it actually right?
"""

import json
from pathlib import Path
from diagnose import diagnose


def main():
    traces = sorted(Path("traces").glob("trace_*.json"))
    if not traces:
        print("No traces. Run `python agent.py` first.")
        return

    total = correct = abstained = hi_conf = hi_conf_correct = 0
    for p in traces:
        t = json.loads(p.read_text())
        gt = t.get("ground_truth_cause", "?")
        d = diagnose(t)
        total += 1
        if d["cause"] == "uncertain":
            abstained += 1
            verdict = "ABSTAIN"
        else:
            ok = (d["cause"] == gt) or (d["cause"] == "correct_abstention" and gt == "missing_knowledge")
            correct += int(ok)
            verdict = "correct" if ok else "WRONG (gt={})".format(gt)
            if d["confidence"] >= 0.7:
                hi_conf += 1
                hi_conf_correct += int(ok)
        print("{:<46} -> {:<18} conf {:.2f}  [{}]".format(t["question"][:45], d["cause"], d["confidence"], verdict))

    answered = total - abstained
    print("\n--- summary ---")
    print("accuracy (answered):              {}/{}".format(correct, answered) if answered else "accuracy: n/a (all abstained)")
    print("abstention rate:                  {}/{}".format(abstained, total))
    if hi_conf:
        print("calibration (conf>=0.7 -> right):  {}/{}".format(hi_conf_correct, hi_conf))


if __name__ == "__main__":
    main()
