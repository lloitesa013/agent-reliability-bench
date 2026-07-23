"""
vsi-gate — the verifier for self-improving agents, as a dependency-free library.

An agent (or pipeline) that modifies its own prompts, rules, memories, tools, or weights must not
trust those modifications. This package is the adoption gate, extracted from the VSI-0 study
(github.com/lloitesa013/agent-reliability-bench — every number below reproduces from that repo):

  - across four self-modification surfaces, fake improvements are the NORM (100% of tool-rewrite
    decision sets contain a reward-hack; 29-30% of real coding problems yield a genuine overfit);
  - selecting by the score a change was derived from ships broken changes 59-88% of the time;
    this gate shipped 0% across every benchmark while still adopting real improvements;
  - in stochastic domains a single run is noise — verify failure RATES (a real driving fix was
    accepted at p=6e-8, and the same candidate REJECTED at deployment scope at p=8.5e-45);
  - curtailment (stop an arm the moment its verdict is mathematically decided) recovers ~26% of
    verification cost with verdicts identical by construction.

Quick start:

    from vsi import gate, gate_rate, Registry

    # deterministic surfaces: hidden-test + regression gate over candidate modifications
    report = gate(candidates, hidden_score=eval_heldout, baseline_hidden=0.60,
                  regression_profile=profile, baseline_profile=base_profile)

    # stochastic surfaces (flaky tests, embodied agents): verify by failure rate
    verdict = gate_rate(baseline_fail=7, baseline_n=12, cand_fail=0, cand_n=19)

    # audit-proof bookkeeping: lines sealed before data, verdicts computed, trials frozen
    reg = Registry("trials.jsonl")
    reg.register("my-fix-1", recipe={...}, lines={"fix_max_fails": 2, "fix_n_planned": 8})
    reg.add_arm("my-fix-1", "fix", {"fails": 1, "n": 8})
    print(reg.verdict("my-fix-1"))

Run the no-LLM demo:  python -m vsi
"""
from vsi_gate import gate, gate_rate, GateReport, CandidateVerdict
from vsi_registry import Registry, compute_verdict
from vsi.curtail import curtail_counts, decided

__version__ = "0.1.0"
__all__ = ["gate", "gate_rate", "GateReport", "CandidateVerdict",
           "Registry", "compute_verdict", "curtail_counts", "decided"]
