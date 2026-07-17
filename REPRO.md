# REPRO — reproduction matrix (S4-①, sealed spec in S4_PREREG.md)

Every verdict-bearing number in this repository recomputes from committed artifacts. Two tiers:
**Tier A (GPU-free, minutes):** verdict recomputation from committed raw counts/records — anyone,
any machine, Python ≥3.8 stdlib only. **Tier B (GPU):** regenerating the raw records themselves
(7B experiments: one CUDA GPU; embodied: CARLA 0.9.15 + LEAD stack, single RTX 5090-class).

Verified fresh on 2026-07-18 (`repro_outputs.txt` holds the captured outputs).

## Tier A — recompute the verdicts (no GPU)

| result | claim | command | expected output |
|---|---|---|---|
| gate demo | the gate as a library | `python vsi_gate.py --demo` | lookup-hack rejected; R25 rates REJECT |
| R42 (S2/W1) | S1's 12 verdicts replay exactly; prescriber 9/9 | `python s2_replay.py` | `W1a: 12/12 exact`, `W1b: 9/9`, `W1 GREEN`, exit 0 |
| R43 (S2/W2) | automated-loop ledger + evidence | inspect `results_w2/w2_trials.jsonl` (6 events), `w2_details.jsonl` (126 rows); verdicts recompute via `vsi_registry.compute_verdict` | MBPP ACCEPT p=2.81e-8; HumanEval REJECT p=1 |
| R33/R37 confirmed sets | 12 + 6 confirmed regressions from raw runs | `python -c "import s3_w1; print(s3_w1.confirm_sets())"` | 12 K1 routes; 6 A2 routes (11715, 17598, 25845, 26944, 27532, 3737) |
| R44 (S3/W1) | risk ranking ≈ random | `python s3_w1.py` | `POOLED LOCO recall@30: 3/18`, `REGISTRY VERDICT: REJECT` |
| R45 | curtailment saves 26%, verdicts identical | `python s3_sprt_replay.py` | `338 runs / 249 runs / SAVED 26%` (identity asserted per route) |
| R46 | naive SPRT loses | `python s3_sprt2.py` | `SAVED 22%`, `concordance 44/50` |
| R47 (S3/W4) | no gate-reuse decay | `python s3_w4_analysis.py` | fresh −7.8pp / iterated +12.5pp, `no decay detected` |
| R31 verdict | accepted-fix statistics | `python -c "from vsi_gate import gate_rate; print(gate_rate(7,12,0,19))"` | p ≈ 5.97e-8, ACCEPT |

Raw data locations: `s3_data/` (sweeps, screens, confirms, panels, W4, route metadata),
`results_w2/` (automated-loop ledger + per-problem evidence), `s3_trials.jsonl` (frozen S3-W1),
`RESULTS.md` (the authoritative ledger, R1–R47, each entry naming its script).

## Tier B — regenerate the raw records (GPU)

| result | machinery | environment |
|---|---|---|
| R13/R17 (text/tool) | `self_verify_agent.py`, `stat_robustness.py` | 7B (Qwen2.5-7B-Instruct), 1×CUDA GPU |
| R22/R24/R26 (real code) | `mbpp_self_verify2.py`, `humaneval_self_verify.py` | same |
| R43 (automated loop, live) | `w2_autoloop.py` + `data_w2/` (raw MBPP/HumanEval JSONL) | same; ~11 min on an RTX 5090 |
| R19–R41, R47 (embodied) | LEAD tfv6 + CARLA 0.9.15 (native Windows), fleet runners | single RTX 5090; see `RESULTS.md` entries for exact runners |

Environment pins: replays = stdlib only (Python ≥3.8; `math.comb` required). 7B experiments =
`transformers`/`torch` CUDA env (WSL). Embodied = CARLA 0.9.15 + LEAD, Windows-native (GPU console
session required for rendering). All results in this repository were produced on ONE RTX 5090.
