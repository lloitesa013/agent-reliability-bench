"""python -m vsi — run the deterministic no-LLM demo (gate + rate gate + curtailment)."""
import sys

import vsi_gate
from vsi.curtail import curtail_counts


def main() -> int:
    vsi_gate._demo()
    print("\ncurtailment demo (VSI-0 R45/R50 machinery):")
    outcomes = [1, 0, 1, 1]        # fail, pass, fail, fail -> REJECT decided at run 4 of 8
    v, used = curtail_counts(outcomes, max_fails=2, n_planned=8)
    print("  recorded arm %s under line <=2/8 -> %s after %d runs (fixed-N would use 8)"
          % (outcomes, v, used))
    return 0


if __name__ == "__main__":
    sys.exit(main())
