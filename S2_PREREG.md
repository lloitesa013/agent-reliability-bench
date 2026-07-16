# S2 Pre-registration — Automating the Outer Loop (SEALED 2026-07-17, before any S2 code was written)

**Decision (operator, 2026-07-17):** "S2 GO. 단 8/15 AoE 컷오프 사전등록 + 5090 폴백(텍스트·코드 표면
한정) 명시 + S2 승리조건 봉인 후 착수."

## What S2 is

S1's outer loop — propose recipe → run trials → gate verdict → next recipe — was executed by a human
(the research session). S2 turns that outer loop into code:

- **Trial Registry** (`vsi_registry.py`): append-only, machine-readable record of every trial
  (recipe, raw run outcomes, pre-registered lines, verdict). No verdict may be registered before its
  trial; no trial may be edited after its verdict.
- **Rule-based Prescriber** (`vsi_prescriber.py`): literature-priors-as-rules (continual-learning
  playbook: epochs↓, retention breadth, mixture bracketing, targeted replay, parameter isolation)
  that reads the registry and emits the next prescription. The prescriber may only PROPOSE.
- **The gate stays the sole adoption authority** (`vsi_gate.py`, unchanged): every verdict is
  recomputed from raw run records by `gate()`/`gate_rate()` + the pre-registered threshold lines.
  No human verdict, no prescriber verdict.

## Win conditions (sealed now, judged later — no post-hoc edits)

- **W1a (verdict replay, exact):** replaying the S1 history (R25, K1, K2a, K2b, A1, A2, A3, A4, A5,
  A6) from raw counts through the automated verdict computation reproduces **all 10 recorded verdicts
  exactly**, including K1's scope split (local ACCEPT + deployment REJECT) and A2's panel-pass →
  deployment-REJECT. Deterministic; must be 10/10.
- **W1b (prescription replay, consistency check):** run at each history prefix, the prescriber's
  emitted axis (e.g. "reduce epochs", "broaden retention", "raise fix ratio", "targeted retention",
  "freeze backbone", "restore known-good data", "close: budget exhausted") matches the move the human
  search actually made next in **≥ 7 of 9 transitions**. HONESTY NOTE, stated up front: the rules are
  distilled *from* this search, so W1b shows only that the distillation is faithful — it is a
  consistency check, **not** evidence of generality. Generality evidence can come only from W2.
- **W2 (new trials, gate integrity):** the automated loop runs **≥ 1 new trial end-to-end with zero
  human verdict intervention** (human role limited to: launching the run, ops babysitting of
  CARLA/GPU, and reading the result). Requirements: (i) trial registered before its verdict; (ii) no
  candidate adopted that fails any pre-registered line; (iii) every abstain/reject recorded with its
  computed evidence. The automation does NOT need to find a passing repair and does NOT need to set
  any record — *running without being fooled is the result.*

## Cutoff (Rail 1 — the paper is always shippable)

- **2026-08-15 AoE.** If W1+W2 are green by then → S2 enters the paper as a section (strict addition;
  the S0+S1 narrative and §4 arc are not restructured). If not green by then → the paper ships with
  S0+S1 exactly as committed at `paper_sea/main.tex`, and S2 becomes next-paper material. After the
  cutoff, no S2 result may touch the submission.

## Compute fallback (Rail 2 — sealed branch condition)

- The 5090's availability for embodied trials is outside our control. **If the 5090 is unavailable
  (or operator-paused) for embodied W2 trials before the cutoff:** W2 is satisfied on the
  **text/code surfaces only** (MBPP/HumanEval closed auto-loop: proposer → registry → prescriber →
  gate, same authority rules). In that branch the paper reports embodied automation as
  **"design + replay validation (W1) only"**, labeled exactly that honestly. Embodied W2 runs only if
  5090 access allows within the cutoff.

## Non-goals (sealed)

No new SOTA/performance claims; no reopening of the S1 repair search (its closure at R41 stands); no
gate modification to make automation look better (any gate change = a new pre-registration).
