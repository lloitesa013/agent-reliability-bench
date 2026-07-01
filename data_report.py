"""Generate DATA_REPORT.md — Week-2 data quality report for traces_bench.jsonl.
Pure-python (no GPU). Runs the SEALED bench/run_bench.py as a subprocess and embeds
its authoritative output. Label-audit and disclosure sections are appended separately."""
import json
import os
import subprocess
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
TRACES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "traces_bench.jsonl")
OUT = os.path.join(HERE, "DATA_REPORT.md")
CTS = ("direct", "reasoning", "distractor")
DOMS = ("finance", "health", "public")

rs = [json.loads(l) for l in open(TRACES)]
n = len(rs)


def sub(pred):
    return [r for r in rs if pred(r)]


def rate(xs, key):
    return (sum(r[key] for r in xs), len(xs))


def is_decline(a):
    a = a.lower()
    return any(p in a for p in ("don't know", "do not know", "not sure", "cannot",
                                "unable", "not specified", "no information"))


L = []
w = L.append
w("# DATA_REPORT — Week-2 traces_bench.jsonl (SCALED build)\n")
w(f"- Records: **{n}**  (target 300)")
w(f"- Base LLM (fixed for all systems): `Qwen/Qwen2.5-14B-Instruct`  ·  Retriever: `BAAI/bge-small-en-v1.5`")
w(f"- Correctness = STRICT token/value overlap (bench_lib.correct, NOT semantic)  ·  groundedness = verbatim guardrail formula")
w(f"- Generated in Docker `arb-bench:26.03` on GPU 1 (GPU 0 excluded).\n")

# 1. balance
w("## 1. Case-type & domain balance")
w("```")
w("case_type   : " + "  ".join(f"{c}={sum(r['case_type']==c for r in rs)}" for c in CTS))
w("domain      : " + "  ".join(f"{d}={sum(r['domain']==d for r in rs)}" for d in DOMS))
w("")
w("cell counts (case_type x domain):")
w(f"{'':12s}" + "".join(f"{d:>10s}" for d in DOMS))
for c in CTS:
    w(f"{c:12s}" + "".join(f"{sum(1 for r in rs if r['case_type']==c and r['domain']==d):>10d}" for d in DOMS))
w("```")
w(f"Balance target 100/100/100 by case_type and ~100 by domain: "
  f"{'MET' if all(sum(r['case_type']==c for r in rs)==100 for c in CTS) and all(sum(r['domain']==d for r in rs)==100 for d in DOMS) else 'NOT MET'}.\n")

# 2. gold / correctness validation
w("## 2. Gold validation & correctness rates")
w("Labels are value-controlled (templated gold); correct_base/correct_rag are assigned by strict match.")
w("```")
w(f"{'subset':14s}{'n':>5s}{'correct_base':>14s}{'correct_rag':>13s}{'should_esc':>12s}")
cb, _ = rate(rs, "correct_base"); cr, _ = rate(rs, "correct_rag"); se = sum(r["should_escalate"] for r in rs)
w(f"{'ALL':14s}{n:>5d}{cb:>14d}{cr:>13d}{se:>12d}")
for c in CTS:
    xs = sub(lambda r, c=c: r["case_type"] == c)
    w(f"{c:14s}{len(xs):>5d}{sum(r['correct_base'] for r in xs):>14d}{sum(r['correct_rag'] for r in xs):>13d}{sum(r['should_escalate'] for r in xs):>12d}")
for d in DOMS:
    xs = sub(lambda r, d=d: r["domain"] == d)
    w(f"{'  '+d:14s}{len(xs):>5d}{sum(r['correct_base'] for r in xs):>14d}{sum(r['correct_rag'] for r in xs):>13d}{sum(r['should_escalate'] for r in xs):>12d}")
w("```")
w(f"Invariant `should_escalate == not correct_rag`: "
  f"{'HOLDS for all records' if all(r['should_escalate']==(not r['correct_rag']) for r in rs) else 'VIOLATED'}.\n")

# 3. groundedness distribution
w("## 3. Groundedness distribution (the signal rule C thresholds at 0.5)")
w("```")
w(f"{'subset':14s}{'mean':>8s}{'g<0.5':>8s}{'g>=0.5':>8s}{'min':>7s}{'max':>7s}")
for c in ("ALL",) + CTS:
    xs = rs if c == "ALL" else sub(lambda r, c=c: r["case_type"] == c)
    g = [r["groundedness"] for r in xs]
    w(f"{c:14s}{sum(g)/len(g):>8.3f}{sum(1 for x in g if x<0.5):>8d}{sum(1 for x in g if x>=0.5):>8d}{min(g):>7.2f}{max(g):>7.2f}")
w("```")
w("Intended: direct groundedness HIGH (value in evidence); reasoning LOW (computed gold absent from evidence "
  "-> false positives -> rule-C overblock); distractor HIGH (corrupted value present in evidence -> false "
  "negatives -> rule-C unsafe pass).\n")

# 4. failure-mode structure
d = sub(lambda r: r["case_type"] == "distractor")
rz = sub(lambda r: r["case_type"] == "reasoning")
unsafe = sum(1 for r in d if (not r["correct_rag"]) and r["groundedness"] >= 0.5)
catch = sum(1 for r in d if (not r["correct_rag"]) and r["groundedness"] < 0.5)
overb = sum(1 for r in rz if r["correct_rag"] and r["groundedness"] < 0.5)
ent = sum(1 for r in d if ("subprime" in r["retrieved"][0].lower() or "premium" in r["retrieved"][0].lower()))
w("## 4. Failure-mode structure (why groundedness-alone is insufficient)")
w("```")
w(f"distractor wrong-but-grounded (g>=0.5 & wrong)  = {unsafe}/100   <- rule-C UNSAFE PASS (value-mismatch/E target)")
w(f"distractor declined/ungrounded (g<0.5 & wrong)  = {catch}/100   <- rule C catches (entity/qualifier twins)")
w(f"  of distractor, entity/qualifier twins (subprime/premium) ~ {ent}/100")
w(f"reasoning correct-but-ungrounded (g<0.5 & right) = {overb}/100   <- rule-C OVERBLOCK (trained-watcher/D target)")
w("```\n")

# 5. declines
w("## 5. Abstention / decline counts (expanded DECLINE list)")
w("```")
w(f"{'subset':14s}{'base declines':>15s}{'rag declines':>14s}")
for c in ("ALL",) + CTS:
    xs = rs if c == "ALL" else sub(lambda r, c=c: r["case_type"] == c)
    w(f"{c:14s}{sum(is_decline(r['base_answer']) for r in xs):>15d}{sum(is_decline(r['rag_answer']) for r in xs):>14d}")
w("```")
w("base_answer declines heavily (base LLM does not know institution-specific policy values) — expected; it makes "
  "system A (base, always-PASS) the low baseline.\n")

# 6. sealed runner preview
w("## 6. Sealed `run_bench.py` preview (SANITY ONLY — not the official score)")
w("No trained watcher yet (Week 2), so `watcher_prob` is absent and D/E fall back to the groundedness rule "
  "(hence D == C). The official A–E score is produced in Week 4 after the watcher is trained.")
try:
    out = subprocess.run([sys.executable, os.path.join(HERE, "bench", "run_bench.py"), TRACES],
                         capture_output=True, text=True, timeout=120).stdout
except Exception as e:
    out = f"(run_bench failed: {e})"
w("```")
w(out.strip())
w("```")
w("Read: with groundedness alone, rule C has BOTH high unsafe_pass and high overblock — i.e. the data leaves "
  "clear room for a trained watcher (D) to cut reasoning overblock and a value-mismatch layer (E) to cut "
  "distractor unsafe pass. That is exactly what Week 2 must deliver.\n")

open(OUT, "w").write("\n".join(L) + "\n")
print(f"wrote {OUT} ({len(L)} lines)")
