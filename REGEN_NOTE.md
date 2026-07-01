# REGEN REQUIRED — this branch has the FIXED engine, NOT fixed data

`data_engine.py` on this branch (`conflict-fix`) changes two things vs `week2-data`, to resolve
the honest problems the Week-2 `DATA_REPORT.md` flagged:

1. **CONFLICT retrieval for distractor.** Distractor now retrieves BOTH the correct doc AND the
   corrupted twin (retrieved-len = 2), so the evidence literally contradicts itself. The answer
   (right or wrong) stays grounded → rule C still passes (unsafe), but a watcher that READS the
   trace can detect the conflict. A lone corrupt doc was information-theoretically uncatchable
   without the gold (DATA_REPORT §8-3); conflict is the honest, generalizable signal. This also
   breaks the degenerate distractor = 0/100-correct (DATA_REPORT §8-2) — the model will now
   sometimes pick the correct doc.

2. **`group` field on every record** (= the fact key). Week-3 must split train/eval at the
   fact-group level, NOT per-record, or a watcher memorizes `query→escalate` and silently breaks
   the sealed no-leakage protocol (DATA_REPORT §8-4). Direct & distractor of the same fact share a
   `group`, so a group-level splitter keeps them on the same side.

Offline structural check (`validate_conflict.py`, no GPU): 300 records, case_type 100/100/100,
distractor retrieved-len = 2, direct∩distractor shared groups = 94, 119 distinct groups.

**`traces_bench.jsonl` and `DATA_REPORT.md` in this branch are STALE** (produced by the old
single-corrupt-doc engine on `week2-data`). Regenerate on a GPU box before Week 3:

    python data_engine.py            # ~42s on one GPU (Qwen2.5-14B-Instruct + bge-small)

then re-run the offline audit + regenerate the report, confirm distractor now has a healthy
correct/wrong mix, and only THEN train the watcher (Week 3) with a fact-group split.

Sealed files (bench/*, run_bench.py) are untouched — the `group` field is auxiliary metadata that
run_bench.py ignores; it does not change any metric or win condition.
