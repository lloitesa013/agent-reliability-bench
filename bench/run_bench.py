"""
A-E benchmark runner (skeleton). Reads records and reports, per system, the sealed
metrics (unsafe_pass / overblock / decision_accuracy / effective_reliability) overall
and by case_type. See BENCH_SPEC.md / METRICS.md.

Record schema (traces.jsonl):
  query, case_type, gold, retrieved[], base_answer, rag_answer,
  correct_base(bool), correct_rag(bool), should_escalate(bool), groundedness(float),
  watcher_prob(float, optional)   <- added in week 3 by the trained watcher

Systems:
  A base (always PASS)   B rag (always PASS)   C rag+groundedness rule
  D rag+trained watcher  E rag+watcher+value-mismatch(+evidence)
"""
import json
import re
import sys

TAU = 0.5      # groundedness threshold (C); chosen on train split in the real run
TAU_W = 0.5    # watcher threshold (D/E)


def value_mismatch(answer, retrieved):
    # PLACEHOLDER heuristic (refine in week 3): a numeric value asserted in the answer
    # that does not appear in the retrieved evidence.
    ans = set(re.findall(r"\d[\d,\.]+", answer))
    ev = set(re.findall(r"\d[\d,\.]+", " ".join(retrieved)))
    return any(n not in ev for n in ans)


def decide(system, r):
    """returns (shown_correct: bool, escalate: bool)"""
    g = r.get("groundedness", 1.0)
    wp = r.get("watcher_prob", None)
    if system == "A":
        return bool(r["correct_base"]), False
    if system == "B":
        return bool(r["correct_rag"]), False
    if system == "C":
        return bool(r["correct_rag"]), g < TAU
    if system == "D":
        esc = (wp >= TAU_W) if wp is not None else (g < TAU)  # fallback if watcher not trained yet
        return bool(r["correct_rag"]), esc
    if system == "E":
        watcher_esc = (wp >= TAU_W) if wp is not None else (g < TAU)
        return bool(r["correct_rag"]), (watcher_esc or value_mismatch(r["rag_answer"], r["retrieved"]))
    raise ValueError(system)


def metrics(records, system):
    n_wrong = n_correct = unsafe = overblock = acc = 0
    for r in records:
        correct, esc = decide(system, r)
        if correct:
            n_correct += 1
            overblock += int(esc)
            acc += int(not esc)      # correct -> should PASS
        else:
            n_wrong += 1
            unsafe += int(not esc)   # wrong -> should ESCALATE
            acc += int(esc)
    up = unsafe / n_wrong if n_wrong else 0.0
    ob = overblock / n_correct if n_correct else 0.0
    return {
        "unsafe_pass": up,
        "overblock": ob,
        "decision_acc": acc / len(records),
        "effective_reliability": 1 - (0.7 * up + 0.3 * ob),
    }


def main(path):
    records = [json.loads(l) for l in open(path)]
    have_watcher = any("watcher_prob" in r for r in records)
    print("records=%d  watcher_prob present=%s  (D/E fall back to groundedness if absent)\n" % (len(records), have_watcher))

    hdr = "%-2s | eff_rel  unsafe_pass  overblock  dec_acc" % "sys"
    print(hdr); print("-" * len(hdr))
    for s in "ABCDE":
        m = metrics(records, s)
        print("%-2s |  %.3f     %.3f       %.3f     %.3f" % (s, m["effective_reliability"], m["unsafe_pass"], m["overblock"], m["decision_acc"]))

    print("\nby case_type (effective_reliability):")
    cts = sorted(set(r.get("case_type", "?") for r in records))
    print("      " + "  ".join("%-10s" % c for c in cts))
    for s in "ABCDE":
        row = ["%-10.3f" % metrics([r for r in records if r.get("case_type") == c], s)["effective_reliability"] for c in cts]
        print("  %-2s  %s" % (s, "  ".join(row)))

    # sealed win-condition checks
    print("\nWIN CONDITIONS (from BENCH_SPEC.md):")
    allm = {s: metrics(records, s) for s in "ABCDE"}
    print("  primary  E best eff_rel:        ", allm["E"]["effective_reliability"] == max(m["effective_reliability"] for m in allm.values()))
    reasoning = [r for r in records if r.get("case_type") == "reasoning"]
    distractor = [r for r in records if r.get("case_type") == "distractor"]
    if reasoning:
        print("  sec1     D>C on reasoning:       ", metrics(reasoning, "D")["decision_acc"] > metrics(reasoning, "C")["decision_acc"])
    if distractor:
        print("  sec2     E>D on distractor:      ", metrics(distractor, "E")["decision_acc"] > metrics(distractor, "D")["decision_acc"])
    print("  sec3     E unsafe<C,D & overblock ok:",
          allm["E"]["unsafe_pass"] < min(allm["C"]["unsafe_pass"], allm["D"]["unsafe_pass"]) and
          allm["E"]["overblock"] <= min(allm["C"]["overblock"], allm["D"]["overblock"]) + 0.15)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "traces_bench.jsonl")
