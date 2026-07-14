"""
60-second offline tour of the Layer-1 watcher. No API key, no dependencies.

    python demo.py

Runs the whole story in order:
  1. a tiny RAG agent fails in designed ways (traces logged)
  2. the diagnoser attributes WHY each failure happened -- and abstains when unsure
  3. the evaluator scores the diagnoser itself (accuracy / abstention / high-conf check)
  4. the runtime guardrail gates a finance agent: PASS / SAFE / ESCALATE + audit log
  5. the sealed A-E benchmark runner on the shipped sample

The real numbers (0.946 vs 0.789, abstention 0.84->0.91) come from the GPU pipeline
documented in RESULTS.md R1-R7; this demo shows the working SHAPE of the system.
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    ("1/5  Agent run -- two designed failures (distractor hijack, missing knowledge)",
     [sys.executable, "agent.py"]),
    ("2/5  Inject the two hard cases a mock can't produce (reasoning error, ambiguous)",
     [sys.executable, "hard_cases.py"]),
    ("3/5  Diagnose WHY each trace failed -- with a score and an ABSTAIN option",
     [sys.executable, "diagnose.py"]),
    ("4/5  Score the diagnoser itself (who watches the watcher)",
     [sys.executable, "evaluate.py"]),
    ("5/5  Runtime guardrail -- PASS / SAFE / ESCALATE + audit log "
     "(incl. the grounded-but-irrelevant catch and the 'No.' probe)",
     [sys.executable, "guardrail.py"]),
]


def main():
    for banner, cmd in STEPS:
        print("\n" + "=" * 100)
        print(banner, flush=True)
        print("=" * 100, flush=True)
        r = subprocess.run(cmd, cwd=HERE)
        if r.returncode != 0:
            print("step failed (exit %d): %s" % (r.returncode, " ".join(cmd)))
            sys.exit(r.returncode)
    print("\n" + "=" * 100)
    print("Bonus: sealed A-E benchmark runner on the shipped 300-trace data (skeleton run)")
    print("       python bench/run_bench.py            # full output")
    print("=" * 100)
    subprocess.run([sys.executable, os.path.join("bench", "run_bench.py")], cwd=HERE)
    print("\nDone. Next: RESULTS.md (R1-R7 = watcher evidence, R8-R31 = VSI-0), STUDY.md, bench/BENCH_SPEC.md")


if __name__ == "__main__":
    main()
