"""
vsi-prescriber -- the S2 rule-based prescriber (pure stdlib).

Reads the Trial Registry and emits the NEXT prescription. The rules are the continual-learning
playbook (literature priors: reduce interference -> replay breadth -> replay weighting -> targeted
replay -> parameter isolation) plus the S1 search's recorded diagnostics. The prescriber may only
PROPOSE -- vsi_gate + the registry's pre-registered lines remain the sole adoption authority.

HONESTY (from S2_PREREG.md, sealed 2026-07-17): these rules were distilled FROM the S1 search
(R25-R41). Replaying them against that search (s2_replay.py, W1b) verifies the distillation is
faithful -- it is NOT evidence of generality. Generality evidence = W2 (new trials) only.

Prescription vocabulary (axes):
  control-probe                     cheapest first: no-training control-layer intervention
  learning-repair-narrow            expert demos on the failure + narrow retention, standard epochs
  escalate-deployment               local/panel pass -> deployment-scope gate (scope rule)
  reduce-epochs                     catastrophic forgetting: try the cheapest variable first
  broaden-retention                 forgetting is data-breadth-driven: widen the retention set
  raise-fix-ratio                   fix diluted by breadth: oversample the fix bucket (bracket)
  family-fix-breadth+targeted-ret   deployment diagnosis: family overfit + coverage holes
  restore-fix-bucket-route-only     family demos conflicted: restore the known-good fix bucket
  freeze-backbone                   data axis exhausted: parameter isolation (head-only)
  restore-known-good-dataset+freeze combine the two known-goods (best data x gentlest update)
  close-bounded-negative            budget exhausted: close the search, report the frontier
"""
from typing import List, Optional

__all__ = ["prescribe", "RECIPE_BUDGET"]

RECIPE_BUDGET = 10        # S1's sealed timebox (R40): ten recipes, then the search closes


def _axis(t: dict, name: str) -> Optional[str]:
    v = t.get("verdict") or {}
    return (v.get("axes") or {}).get(name)


def _is_deploy(t: dict) -> bool:
    return "deploy_alpha" in t["lines"]


def _recipes_spent(history: List[dict]) -> int:
    """Distinct recipes tried in the CURRENT campaign (same target as the latest trial). A
    deployment stage re-uses its panel's recipe -- not a new recipe. The budget is a per-campaign
    timebox, not a lifetime cap: other targets' searches in the global history don't count."""
    if not history:
        return 0
    target = history[-1]["recipe"].get("target")
    return len({t["recipe"].get("name", t["id"]) for t in history
                if t["recipe"].get("target") == target})


def prescribe(history: List[dict]) -> dict:
    """history = Registry.history() (frozen trials, registration order). Returns
    {"axis": ..., "why": ...} for the next move. First matching rule wins; rules are ordered
    from cheapest intervention to most invasive (the literature-prior ladder)."""

    if not history:
        return {"axis": "control-probe",
                "why": "no trials yet: try the cheapest plausible fix first (no training)"}

    if _recipes_spent(history) >= RECIPE_BUDGET:
        return {"axis": "close-bounded-negative",
                "why": "recipe budget (%d) exhausted: close, report the measured frontier" % RECIPE_BUDGET}

    last = history[-1]
    verdict = (last.get("verdict") or {}).get("verdict")
    recipe = last["recipe"]

    # -- a pass at the current scope always escalates to the next scope (the scope rule) --
    if verdict in ("ACCEPT", "PANEL_PASS") and not _is_deploy(last):
        return {"axis": "escalate-deployment",
                "why": "passed at %s scope: a verdict is only valid at its measured scope -- "
                       "re-verify at deployment scope" % ("local" if verdict == "ACCEPT" else "panel")}

    # -- control-layer probe rejected: move to learning-based repair --
    if verdict == "REJECT" and recipe.get("type") == "control":
        # accumulated-knowledge rule: if the global registry already holds a recipe that PASSED a
        # local panel on a DIFFERENT target, start this target's learning attempt from it instead
        # of re-climbing the ladder from scratch (the registry is the loop's memory).
        known_good = [t for t in history
                      if (t.get("verdict") or {}).get("verdict") == "PANEL_PASS"
                      and t["recipe"].get("type") == "finetune"
                      and t["recipe"].get("target") != recipe.get("target")]
        if known_good:
            src = known_good[-1]["recipe"]
            return {"axis": "transfer-known-good",
                    "why": "control probe rejected; global history holds a panel-passing recipe "
                           "(%s) from another target -- transfer its mixture to this target"
                           % src.get("name", "?"),
                    "template": src}
        return {"axis": "learning-repair-narrow",
                "why": "control-layer intervention made the rate worse: repair needs learning "
                       "(expert demos on the failure), start with the standard recipe"}

    # -- deployment rejection: cheapest untried variable first, then the recorded diagnosis --
    if verdict == "GLOBAL_REJECT":
        moved = {t["recipe"].get("move") for t in history}
        if "reduce-epochs" not in moved:
            return {"axis": "reduce-epochs",
                    "why": "catastrophic forgetting at deployment scope: try the cheapest "
                           "variable first (fewer epochs; registry shows it untried)"}
        diag = (last.get("arms", {}).get("deploy") or {}).get("diagnosis", {})
        if diag.get("within_family_overfit") or diag.get("coverage_holes"):
            return {"axis": "family-fix-breadth+targeted-ret",
                    "why": "cheap variables exhausted; deployment diagnosis: within-family overfit "
                           "+ retention coverage holes -> fix-family demos + retention on the "
                           "breakage map"}
        return {"axis": "broaden-retention",
                "why": "forgetting persists with fewer epochs: widen the retention data"}

    # -- panel rejections: read which axis failed --
    if verdict == "REJECT":
        fix, ret = _axis(last, "fix"), _axis(last, "retention")

        if fix == "PASS" and ret == "FAIL":
            if recipe.get("data") == "narrow":
                return {"axis": "broaden-retention",
                        "why": "fix retained but forgetting is total on narrow data: forgetting is "
                               "a data-breadth problem (K2a lesson) -> broaden retention"}
            return {"axis": "raise-fix-ratio",
                    "why": "fix retained, retention just over the line: rebalance the mixture"}

        if fix == "FAIL":
            # fix collapsed after a data change away from a known-good bucket -> restore it
            if recipe.get("fix_bucket") == "family":
                return {"axis": "restore-fix-bucket-route-only",
                        "why": "family demos diluted/conflicted with the route repair (fractal "
                               "dilution, A3 lesson): restore the known-good fix bucket"}
            if recipe.get("freeze") and recipe.get("dataset") != "known-good":
                return {"axis": "restore-known-good-dataset+freeze",
                        "why": "head-only update did not rescue this dataset: combine the two "
                               "known-goods (panel-passing data x gentlest update) -- final shot"}
            if recipe.get("targeted_ret") or recipe.get("dataset") == "modified":
                return {"axis": "freeze-backbone",
                        "why": "fix data identical to a passing recipe yet adding retention broke "
                               "it: data-mixture axis exhausted (A4 lesson) -> parameter isolation"}
            if recipe.get("fix_ratio") is None:
                return {"axis": "raise-fix-ratio",
                        "why": "fix diluted by uniform breadth: oversample the fix bucket "
                               "(the selective-replay lever)"}
            return {"axis": "raise-fix-ratio",
                    "why": "fix still diluted at ratio %.2f: continue the bracket upward"
                           % recipe["fix_ratio"]}

    return {"axis": "close-bounded-negative",
            "why": "no rule fired (unmapped state): stop and hand back to the operator -- "
                   "the prescriber must not guess"}
