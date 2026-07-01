# DATA_REPORT — Week-2 traces_bench.jsonl (SCALED build)

- Records: **300**  (target 300)
- Base LLM (fixed for all systems): `Qwen/Qwen2.5-14B-Instruct`  ·  Retriever: `BAAI/bge-small-en-v1.5`
- Correctness = STRICT token/value overlap (bench_lib.correct, NOT semantic)  ·  groundedness = verbatim guardrail formula
- Generated in Docker `arb-bench:26.03` on GPU 1 (GPU 0 excluded).

## 1. Case-type & domain balance
```
case_type   : direct=100  reasoning=100  distractor=100
domain      : finance=100  health=100  public=100

cell counts (case_type x domain):
               finance    health    public
direct              34        33        33
reasoning           33        34        33
distractor          33        33        34
```
Balance target 100/100/100 by case_type and ~100 by domain: MET.

## 2. Gold validation & correctness rates
Labels are value-controlled (templated gold); correct_base/correct_rag are assigned by strict match.
```
subset            n  correct_base  correct_rag  should_esc
ALL             300             9          197         103
direct          100             1          100           0
reasoning       100             7           97           3
distractor      100             1            0         100
  finance       100             6           67          33
  health        100             3           64          36
  public        100             0           66          34
```
Invariant `should_escalate == not correct_rag`: HOLDS for all records.

## 3. Groundedness distribution (the signal rule C thresholds at 0.5)
```
subset            mean   g<0.5  g>=0.5    min    max
ALL              0.630     112     188   0.00   1.00
direct           1.000       0     100   1.00   1.00
reasoning        0.050      96       4   0.00   1.00
distractor       0.840      16      84   0.00   1.00
```
Intended: direct groundedness HIGH (value in evidence); reasoning LOW (computed gold absent from evidence -> false positives -> rule-C overblock); distractor HIGH (corrupted value present in evidence -> false negatives -> rule-C unsafe pass).

## 4. Failure-mode structure (why groundedness-alone is insufficient)
```
distractor wrong-but-grounded (g>=0.5 & wrong)  = 84/100   <- rule-C UNSAFE PASS (value-mismatch/E target)
distractor declined/ungrounded (g<0.5 & wrong)  = 16/100   <- rule C catches (entity/qualifier twins)
  of distractor, entity/qualifier twins (subprime/premium) ~ 16/100
reasoning correct-but-ungrounded (g<0.5 & right) = 93/100   <- rule-C OVERBLOCK (trained-watcher/D target)
```

## 5. Abstention / decline counts (expanded DECLINE list)
```
subset          base declines  rag declines
ALL                       266            16
direct                     96             0
reasoning                  74             0
distractor                 96            16
```
base_answer declines heavily (base LLM does not know institution-specific policy values) — expected; it makes system A (base, always-PASS) the low baseline.

## 6. Sealed `run_bench.py` preview (SANITY ONLY — not the official score)
No trained watcher yet (Week 2), so `watcher_prob` is absent and D/E fall back to the groundedness rule (hence D == C). The official A–E score is produced in Week 4 after the watcher is trained.
```
records=300  watcher_prob present=False  (D/E fall back to groundedness if absent)

sys | eff_rel  unsafe_pass  overblock  dec_acc
----------------------------------------------
A  |  0.300     1.000       0.000     0.030
B  |  0.300     1.000       0.000     0.657
C  |  0.288     0.816       0.472     0.410
D  |  0.288     0.816       0.472     0.410
E  |  0.441     0.505       0.685     0.377

by case_type (effective_reliability):
      direct      distractor  reasoning 
  A   0.300       0.300       0.300     
  B   1.000       0.300       0.300     
  C   1.000       0.412       0.712     
  D   1.000       0.412       0.712     
  E   0.874       0.636       0.712     

WIN CONDITIONS (from BENCH_SPEC.md):
  primary  E best eff_rel:         True
  sec1     D>C on reasoning:        False
  sec2     E>D on distractor:       True
  sec3     E unsafe<C,D & overblock ok: False
```
Read: with groundedness alone, rule C has BOTH high unsafe_pass and high overblock — i.e. the data leaves clear room for a trained watcher (D) to cut reasoning overblock and a value-mismatch layer (E) to cut distractor unsafe pass. That is exactly what Week 2 must deliver.

## 7. Label audit (independent + offline)
- **Blind judge panel:** a 35-record stratified sample (labels stripped) was independently re-labeled by 3 blind LLM judges using the strict value-match rubric, then reconciled against stored labels: **correct_rag agreement 35/35, correct_base agreement 35/35, disputes 0.**
- **Offline stress battery:** 36/36 hand-constructed correctness cases pass (numeric/comma/%/unit, categorical, negation, decline, scenario-number restatement) — **0 mislabels** (`audit_offline.py`).
- **Invariant:** `should_escalate == not correct_rag` holds on all 300 records.
- Labels are therefore reliable **within the strict string-match definition** (see disclosure 7 below).

## 8. Honest disclosures & limitations (READ before using any number above)
Per `CLAIMS.md` — a miss is a finding; nothing here is overstated.

1. **The Section-6 A–E preview is NOT the official score and confirms no win condition.** Week 2 has no trained watcher, so `watcher_prob` is absent and D falls back to the groundedness rule (**D == C exactly**); E uses the throwaway PLACEHOLDER `value_mismatch` (run_bench.py comment: "refine in week 3"). The `primary E best` / `sec2 E>D` booleans are artifacts of that placeholder + the degenerate structure below — **not** evidence for the thesis. Official score = Week 4, trained watcher, 5 splits.
2. **Per-case correctness is one-sided by construction:** direct = 100/100 correct (0 wrong → `unsafe_pass` on direct is undefined, pinned to 0 for every system); distractor = 0/100 correct (0 correct → `overblock` on distractor is undefined, pinned to 0). Any win condition leaning on the absent axis is near-vacuous — with zero correct distractor traces, extra escalation lowers unsafe_pass at **zero** overblock cost, so `sec2 (E>D on distractor)` is almost automatic.
3. **The placeholder `value_mismatch` is formatting noise, not detection (verified):** it fires on **42/100 correct direct** and **62/97 correct reasoning** answers (e.g. "1000" vs "1,000"), while catching only **32/84** distractor wrong-but-grounded cases. E's preview overblock (0.685) and its distractor "win" come mostly from this misfiring. The real value-mismatch detector is Week-3 work.
4. **Split-leakage risk — ACTION REQUIRED in Week 3/4:** direct and distractor share **94/100** `(query,gold)` pairs (same fact: clean evidence→correct vs corrupted evidence→wrong); only **206** distinct `(query,gold)` across all 300. Record-level 5-split shuffling can put the same fact in train (direct/correct) and eval (distractor/wrong), letting a watcher memorize `query→escalate` and silently break the sealed no-leakage protocol. **Recommendation: split at the fact/query group level, not per-record.** (sample_bench.jsonl reuses queries the same way; the splitter must handle it.)
5. **Failure modes are engineered, not emergent.** Reasoning low groundedness (mean 0.05) and distractor wrong-but-grounded (84/100) are planted by design. Rule C's high unsafe/overblock is an intended property of the fixtures — do **not** read it as a real-world groundedness failure rate (`CLAIMS.md`: not production performance).
6. **Reasoning is overblock-dominated:** arithmetic is easy for the 14B (correct_rag 97/100, only **3** wrong), so reasoning `unsafe_pass` is a k/3 quantity (coarse over 5 splits). `sec1 (D>C on reasoning)` is essentially an overblock-reduction test on correct-but-ungrounded answers, not a hard-error-catching test.
7. **All correctness/groundedness numbers are STRICT string/value-overlap proxies** (`bench_lib.correct` + verbatim guardrail formula), NOT semantic. A semantically-correct but differently-phrased answer can be labeled wrong; every rate is relative to these definitions.
8. **eff_rel for always-PASS systems is a structural constant:** any always-PASS system with ≥1 wrong answer has unsafe_pass=1.0 → eff_rel = 1−0.7 = **0.300**. A=B=0.300 is arithmetic (not "RAG gives no benefit"; decision_acc A=0.030 vs B=0.657). The metric scores the gate, not the base model.
9. **`should_escalate ≡ NOT correct_rag`** by definition — no independent risk signal beyond the correctness label; the gate's task is to reproduce correctness from the trace.
10. **Scope (`CLAIMS.md`):** one fixed base model (Qwen2.5-14B-Instruct), one node, synthetic templated/value-controlled traces, ~300 records over ~206 distinct facts. Not production, not base-model capability, not SOTA — treat any A–E gap below the 5-split std as noise.

