# PROTOCOL — held-back validation for reused gates (S4-②, sealed spec in S4_PREREG.md)

**The rule.** Any evaluation surface that is REUSED across candidate iterations (a panel, a screen,
a benchmark slice) must be paired, before any final adoption, with **held-back fresh probes never
touched during iteration**, and both surfaces' candidate-vs-baseline uplifts must be reported side
by side.

**Format (the W4 layout):**

| surface | candidate rate | contemporaneous baseline rate | uplift |
|---|---|---|---|
| iterated (reused across selection) | measured | measured | Δ_iter |
| fresh (held back, never used) | measured | measured | Δ_fresh |

Decay signal = Δ_fresh significantly worse than Δ_iter (the candidate was flattered by the reused
surface). Adoption additionally requires the fresh surface alone to clear the same pre-registered
lines as the iterated one.

**Why (the evidence chain).**
- R18 (text): reusing a 20-item hidden set across 8 selections re-inflated scores by +0.045 —
  gate reuse is adaptive overfitting at system level.
- R37 (sealed caution): recipe iteration against a fixed 169-route screen was flagged as a decay
  risk BEFORE any decay was looked for; the 51 non-clean routes were reserved then as fresh probes.
- R47 (measurement): for candidate A6 after a 10-recipe search, NO decay was detected
  (fresh-surface uplift −7.8pp vs iterated +12.5pp; damage concentrated on the already-known
  surface; one-sided decay test p = 0.97).

**Honest boundary.** R47 is one candidate, one budget, 2 runs/arm/route (aggregate power only). The
protocol is justified as INSURANCE, not by an observed exploit — and the insurance is cheap: R45
shows curtailment recovers 26% of confirm cost for free, which more than pays for a fresh-probe
batch. Rule of thumb from these numbers: reserve ≥20% of evaluation surface as held-back from day
one; spend the curtailment savings on it.

**Registry integration.** A trial's `lines` must name which surface they were computed on; a
deployment-scope adoption claim requires BOTH a passed iterated-surface trial and a passed
fresh-surface trial registered before their data (see `vsi_registry.py` invariants).
