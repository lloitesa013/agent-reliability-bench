# REGEN NOTE — RESOLVED (kept as changelog)

**Status: resolved.** The concerns below were addressed: `traces_bench.jsonl` in this branch WAS
regenerated with the fixed conflict-distractor engine (week-3 data commit — records carry the
`group` field, distractor retrieved-len = 2, and a healthy correct/wrong distractor mix ~33/100).
`DATA_REPORT.md` still describes the OLD week-2 build and is kept for provenance with a STALE
banner; `RESULTS.md` is the authoritative report.

---

Original note (historical): `data_engine.py` changed two things vs `week2-data`, to resolve the
honest problems the Week-2 `DATA_REPORT.md` flagged:

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

Sealed files (bench/*, run_bench.py) are untouched by the data change — the `group` field is
auxiliary metadata that run_bench.py ignores; it does not change any metric or win condition.
