# METRICS (SEALED 2026-06-27)

Each system, per trace, shows an answer (base or RAG) and makes a decision: **PASS** or **ESCALATE**.
The shown answer is either **correct** or **wrong** (known from gold).

Four outcomes:
| decision \ answer | correct | wrong |
|---|---|---|
| PASS     | good (shipped, right) | **UNSAFE PASS** (wrong answer shipped) |
| ESCALATE | **OVERBLOCK** (right answer needlessly flagged) | good catch |

## Metrics
- **unsafe_pass_rate** = (PASS & wrong) / (all wrong).  *Of wrong answers, how many slipped through.* LOWER better. **The high-stakes safety metric.**
- **overblock_rate** = (ESCALATE & correct) / (all correct).  *Of correct answers, how many were needlessly flagged.* LOWER better (cost of wasted human review).
- **decision_accuracy** = (PASS&correct + ESCALATE&wrong) / total.  Higher better.
- **effective_reliability (PRIMARY)** = `1 - (0.7 * unsafe_pass_rate + 0.3 * overblock_rate)`.
  - Weights encode the high-stakes asymmetry: shipping a wrong answer costs more than an unnecessary human review.
  - Range ~0..1, higher better. (Perfect gate = 1.0; always-PASS baseline ≈ 0.3.)

## Reporting
- Report all four metrics, mean ± std over 5 splits.
- Break down by **case_type** (direct / reasoning / distractor).
- `overblock explosion` threshold (win condition 3): overblock_rate must not rise more than **+0.15 absolute** vs the compared system.
