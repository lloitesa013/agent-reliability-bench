"""
Audit of the rate gate itself (stdlib unittest; run from the repo root):

    python -m unittest tests.test_gate_rate -v

The gate decides adoptions, so its false-ACCEPT rate is checked exactly -- by summing over every
possible outcome of both arms -- rather than trusted.
"""
import unittest
from math import comb

from vsi_gate import (gate_rate, gate_rate_type1, _binom_tail_le,
                      _fisher_cand_le, _fisher_cand_ge)

RATES = (0.1, 0.2, 0.3, 0.4, 0.5, 7 / 12, 0.7, 0.8)


class FisherValues(unittest.TestCase):
    def test_r31_two_sample(self):
        # 7/12 vs 0/19: the observed table is the most extreme one -> C(12,7) / C(31,7)
        r = gate_rate(7, 12, 0, 19)
        self.assertEqual(r["method"], "fisher")
        self.assertAlmostEqual(r["p_value"], comb(12, 7) / comb(31, 7), places=15)
        self.assertAlmostEqual(r["p_value"], 3.0119e-4, places=7)
        self.assertTrue(r["accept"])

    def test_r31_recorded_value_reproduces_with_fixed_baseline(self):
        r = gate_rate(7, 12, 0, 19, method="fixed_baseline")
        self.assertAlmostEqual(r["p_value"], 5.970166e-8, places=13)
        self.assertTrue(r["accept"])

    def test_r43_mbpp(self):
        self.assertAlmostEqual(gate_rate(41, 100, 5, 59)["p_value"], 4.787e-6, places=8)
        self.assertAlmostEqual(gate_rate(41, 100, 5, 59, method="fixed_baseline")["p_value"],
                               2.808e-8, places=10)

    def test_tails_are_complementary(self):
        # same margins (total failures k = bf + cf): P(X <= cf) + P(X >= cf + 1) = 1
        for bf, bn, cf, cn in ((7, 12, 3, 19), (5, 76, 53, 76), (0, 10, 0, 10), (10, 10, 4, 6)):
            total = _fisher_cand_le(bf, bn, cf, cn)
            if bf >= 1 and cf + 1 <= cn:
                total += _fisher_cand_ge(bf - 1, bn, cf + 1, cn)
            self.assertAlmostEqual(total, 1.0, places=12)

    def test_r33_deployment_worse(self):
        self.assertAlmostEqual(_fisher_cand_ge(5, 76, 53, 76), 6.329e-17, places=19)


class Direction(unittest.TestCase):
    def test_worse_candidate_rejected(self):
        for method in ("fisher", "fixed_baseline"):
            r = gate_rate(3, 6, 5, 6, method=method)          # R25: 50% -> 83%
            self.assertFalse(r["accept"])
            self.assertTrue(r["verdict"].startswith("REJECT -- rate not lower"))

    def test_equal_rates_rejected(self):
        self.assertFalse(gate_rate(4, 10, 4, 10)["accept"])

    def test_bad_input(self):
        with self.assertRaises(ValueError):
            gate_rate(1, 0, 1, 5)
        with self.assertRaises(ValueError):
            gate_rate(1, 5, 1, 5, method="binomial")


class FalseAcceptRate(unittest.TestCase):
    """A no-op change must not be adopted more than alpha of the time, whatever the true rate."""

    def test_type1_sums_to_a_probability(self):
        # sanity of the exact enumeration: the joint pmf of both arms sums to 1
        p = 0.37
        total = sum(comb(12, i) * p ** i * (1 - p) ** (12 - i) for i in range(13))
        self.assertAlmostEqual(total, 1.0, places=12)
        self.assertAlmostEqual(_binom_tail_le(12, 12, p), 1.0, places=12)

    def test_fisher_controls_alpha(self):
        for n_base, n_cand in ((12, 19), (19, 19), (8, 8), (24, 24)):
            for p in RATES:
                t = gate_rate_type1(n_base, n_cand, p, alpha=0.05)
                self.assertLessEqual(t, 0.05, "n=%d/%d p=%.3f -> %.4f" % (n_base, n_cand, p, t))

    def test_fixed_baseline_exceeds_alpha(self):
        # documents why the default changed: 12 baseline vs 19 candidate runs (the R31 design)
        worst = max(gate_rate_type1(12, 19, p, method="fixed_baseline") for p in RATES)
        best = min(gate_rate_type1(12, 19, p, method="fixed_baseline") for p in RATES)
        self.assertGreater(best, 0.05)
        self.assertAlmostEqual(worst, 0.1559, places=3)


if __name__ == "__main__":
    unittest.main()
