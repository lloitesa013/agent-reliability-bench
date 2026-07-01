# Watcher Reliability Benchmark — SPEC (SEALED 2026-06-27)

**Sealed before any results are seen. Win conditions are fixed here and NOT changed after running.**

## Question
> Does adding a calibrated watcher / gate / evidence layer on top of the SAME base LLM
> produce a more reliable system (catches wrong answers, keeps correct ones) for
> high-stakes regulated domains?

## Dataset  (`traces.jsonl`)
- **300 traces**, balanced by case_type: **direct 100 / reasoning 100 / distractor 100**
- Domains: finance, healthcare, public-sector (≈ even split)
- Each record (schema):
  ```json
  {"domain","case_type","query","gold","retrieved":["..."],
   "base_answer":"(LLM, NO retrieval)","rag_answer":"(LLM, WITH retrieval)",
   "correct_base":true,"correct_rag":true,"should_escalate":false,"groundedness":0.0}
  ```
- `should_escalate = NOT correct_rag`  (the RAG answer is wrong/risky -> should be flagged).
- Labels are reliable: gold is known (templated for reasoning, value-controlled for distractor).

## Systems compared
| id | system | answer shown | gate decision |
|----|--------|--------------|---------------|
| A | Base LLM | base_answer | always PASS |
| B | RAG LLM | rag_answer | always PASS |
| C | RAG + groundedness rule | rag_answer | ESCALATE if groundedness < tau |
| D | RAG + trained watcher | rag_answer | ESCALATE if watcher_prob >= tau_w |
| E | RAG + watcher + value-mismatch + evidence | rag_answer | ESCALATE if watcher OR value-mismatch; emit evidence packet |

## Protocol
- Same base LLM for every system (only the layer differs).
- 5 random train/eval splits; report mean +/- std.
- D/E trained ONLY on the train split; metrics on the eval split (no leakage).
- Thresholds (tau, tau_w) chosen on the train split only (no test peeking).

## WIN CONDITIONS (fixed, sealed)
- **Primary:** E has the best **effective_reliability** (METRICS.md) overall.
- **Secondary:**
  1. D beats C on the **reasoning** subset (decision accuracy).
  2. E beats D on the **distractor / value-mismatch** subset.
  3. E lowers **unsafe_pass_rate** vs C and D **without** overblock_rate rising more than +0.15 absolute.
- Every result is reported HONESTLY whether or not these hold. A miss is a finding, not a failure.

See `METRICS.md` (definitions) and `CLAIMS.md` (claims / non-claims).
