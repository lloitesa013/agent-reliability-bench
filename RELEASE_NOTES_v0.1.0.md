# vsi-gate v0.1.0 — the verifier for self-improving agents

First public release of the gate machinery from the VSI-0 study ("The Verifier Is the Loop").

- `gate()` — hidden-test + regression adoption gate for {prompt, rule, memory, tool} self-mods.
  Shipped 0% broken changes across every benchmark where naive selection shipped 59–88%.
- `gate_rate()` — one-sided exact-binomial rate gate for flaky/stochastic surfaces. The same
  machinery that accepted a real embodied repair (p=6e-8) and rejected the same candidate at
  deployment scope (p=8.5e-45).
- `Registry` — append-only trial ledger: lines sealed before data, verdicts computed not declared,
  frozen after verdict. Replays the entire 10-recipe embodied search verdict-exactly.
- `vsi.curtail` — stop any count-threshold arm the moment its verdict is decided. Recovered 26%
  of deployment-confirm cost with verdict identity (and beat a naive SPRT on both axes).
- `python -m vsi` — deterministic no-LLM demo; `examples/` — prompt-change gating + flaky-CI gating.

Pure stdlib, Python ≥3.8, MIT. Every number reproduces from this repository (see REPRO.md).
