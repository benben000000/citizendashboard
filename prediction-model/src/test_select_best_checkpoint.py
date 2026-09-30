"""
Tests for select_best_checkpoint.py.

Everything here is checked against a value that can be computed by hand or
against an independently derived numerical result. Where scipy happens to be
installed the hand-rolled distributions are additionally cross-checked against
it, but scipy is not required.

Coverage required by the task:
  * statistics correctness on hand-computable cases, including a known-seed
    dataset whose sd is worked out longhand
  * the single-seed INSUFFICIENT_EVIDENCE path
  * the best-seed vs mean distinction
  * the paired comparison
"""

import json
import math
import os
import random
import sys
import tempfile
import unittest
from decimal import Decimal, getcontext

getcontext().prec = 40

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import select_best_checkpoint as sbc


# ---------------------------------------------------------------------------
# Longhand reference values
# ---------------------------------------------------------------------------

# The real +6h wind numbers from the corrected sweep: 0.598, 0.628, 0.725.
# Worked out in exact decimal arithmetic, not copied from the module:
#   sum       = 1.9502273440479735
#   mean      = 1.9502273440479735 / 3            = 0.6500757813493245
#   devs      = -0.0524609726385773
#               -0.0222919974869560
#               +0.0747529701255333
#   sumsq     = 0.00883709334473288277100292342018
#   var       = 0.00883709334473288277... / 2     = 0.00441854667236644139...
#   sd        = 0.06647214959941073987...
#   sem       = 0.06647214959941073987... / sqrt(3) = 0.03837771346483286509...
WIND_H6 = [0.5976148087107472, 0.6277837838623685, 0.7248287514748578]
WIND_H6_MEAN = 0.6500757813493245
WIND_H6_SD = 0.06647214959941074
WIND_H6_SEM = 0.03837771346483287
WIND_H6_SUM_SQ = 0.00883709334473288277100292342018

# Textbook check set: mean exactly 5, sample variance exactly 32/7.
CLASSIC = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
CLASSIC_SD = math.sqrt(32.0 / 7.0)          # 2.1380899352993947


def _integrated_emax(n, intervals=2000):
    """
    The same integral the module uses, with no closed-form shortcut.

    Kept in the test as an independent path to the same number so the
    closed-form override for small n can be checked against the general method.
    """
    def integrand(x):
        phi = math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
        return n * x * phi * sbc.norm_cdf(x) ** (n - 1)

    lo, hi, step = -10.0, 10.0, 20.0 / intervals
    total = integrand(lo) + integrand(hi)
    for i in range(1, intervals):
        total += (4.0 if i % 2 else 2.0) * integrand(lo + i * step)
    return total * step / 3.0


class TestBasicStatistics(unittest.TestCase):
    """mean / sd / sem against longhand values."""

    def test_mean_of_known_seed_dataset(self):
        self.assertAlmostEqual(sbc.mean(WIND_H6), WIND_H6_MEAN, places=13)

    def test_sample_sd_of_known_seed_dataset(self):
        """sd computed by hand over the three wind seeds, Bessel-corrected."""
        m = sbc.mean(WIND_H6)
        self.assertAlmostEqual(sum((x - m) ** 2 for x in WIND_H6), WIND_H6_SUM_SQ,
                               places=15)
        self.assertAlmostEqual(sbc.sample_sd(WIND_H6), WIND_H6_SD, places=13)
        self.assertAlmostEqual(sbc.sample_sd(WIND_H6) ** 2, WIND_H6_SUM_SQ / 2,
                               places=15)

    def test_sem_of_known_seed_dataset(self):
        self.assertAlmostEqual(sbc.standard_error(WIND_H6), WIND_H6_SEM, places=13)
        self.assertAlmostEqual(sbc.standard_error(WIND_H6),
                               WIND_H6_SD / math.sqrt(3), places=13)

    def test_sem_is_sd_over_sqrt_n(self):
        self.assertAlmostEqual(sbc.standard_error(CLASSIC),
                               CLASSIC_SD / math.sqrt(8), places=12)

    def test_textbook_sd(self):
        """[2,4,4,4,5,5,7,9] has mean 5 and sample variance 32/7."""
        self.assertAlmostEqual(sbc.mean(CLASSIC), 5.0, places=12)
        self.assertAlmostEqual(sbc.sample_sd(CLASSIC), CLASSIC_SD, places=12)
        self.assertAlmostEqual(CLASSIC_SD, 2.1380899352993947, places=12)

    def test_textbook_sd_in_exact_arithmetic(self):
        """The textbook identity restated with no floating point at all, so the
        32/7 is not the module agreeing with itself."""
        vals = [Decimal(2), Decimal(4), Decimal(4), Decimal(4),
                Decimal(5), Decimal(5), Decimal(7), Decimal(9)]
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
        self.assertEqual(m, Decimal(5))
        self.assertEqual(var, Decimal(32) / Decimal(7))
        self.assertAlmostEqual(float(var.sqrt()), sbc.sample_sd(CLASSIC), places=12)

    def test_sd_uses_ddof_1_not_ddof_0(self):
        """Population sd and sample sd genuinely differ here; we must use Bessel."""
        n = len(CLASSIC)
        pop = math.sqrt(sum((x - 5.0) ** 2 for x in CLASSIC) / n)
        self.assertNotAlmostEqual(sbc.sample_sd(CLASSIC), pop, places=6)
        self.assertAlmostEqual(sbc.sample_sd(CLASSIC) ** 2,
                               sum((x - 5.0) ** 2 for x in CLASSIC) / (n - 1),
                               places=12)

    def test_sd_is_invariant_to_input_order(self):
        shuffled = list(reversed(WIND_H6))
        self.assertAlmostEqual(sbc.mean(shuffled), sbc.mean(WIND_H6), places=15)
        self.assertAlmostEqual(sbc.sample_sd(shuffled), sbc.sample_sd(WIND_H6),
                               places=15)

    def test_sd_undefined_for_single_value(self):
        self.assertIsNone(sbc.sample_sd([1.0]))
        self.assertIsNone(sbc.standard_error([1.0]))

    def test_empty_inputs(self):
        self.assertIsNone(sbc.mean([]))
        self.assertIsNone(sbc.sample_sd([]))


class TestDistributions(unittest.TestCase):
    def test_norm_ppf_known_quantiles(self):
        self.assertAlmostEqual(sbc.norm_ppf(0.5), 0.0, places=10)
        self.assertAlmostEqual(sbc.norm_ppf(0.975), 1.959963985, places=7)
        self.assertAlmostEqual(sbc.norm_ppf(0.025), -1.959963985, places=7)
        self.assertAlmostEqual(sbc.norm_ppf(0.8413447461), 1.0, places=7)
        self.assertAlmostEqual(sbc.norm_ppf(0.1586552539), -1.0, places=7)
        self.assertAlmostEqual(sbc.norm_ppf(0.8), 0.8416212336, places=7)
        self.assertAlmostEqual(sbc.norm_ppf(0.995), 2.575829304, places=6)

    def test_norm_cdf_known_values(self):
        self.assertAlmostEqual(sbc.norm_cdf(0.0), 0.5, places=15)
        self.assertAlmostEqual(sbc.norm_cdf(1.959963985), 0.975, places=9)
        self.assertAlmostEqual(sbc.norm_cdf(-1.959963985), 0.025, places=9)

    def test_norm_ppf_roundtrip(self):
        for x in (-2.5, -1.0, -0.25, 0.0, 0.75, 3.0):
            self.assertAlmostEqual(sbc.norm_ppf(sbc.norm_cdf(x)), x, places=9)

    def test_norm_ppf_rejects_out_of_range(self):
        for p in (0.0, 1.0, -0.1, 1.1):
            with self.assertRaises(ValueError):
                sbc.norm_ppf(p)

    def test_t_critical_matches_published_table(self):
        """Two-sided 5% critical values from the standard t table."""
        for df, expected in ((1, 12.706205), (2, 4.302653), (3, 3.182446),
                             (4, 2.776445), (5, 2.570582), (10, 2.228139),
                             (20, 2.085963), (30, 2.042272)):
            self.assertAlmostEqual(sbc.t_critical_two_sided(0.05, df),
                                   expected, places=4)

    def test_t_two_sided_p_at_critical_values(self):
        for df in (2, 4, 10, 30):
            t = sbc.t_critical_two_sided(0.05, df)
            self.assertAlmostEqual(sbc.t_two_sided_p(t, df), 0.05, places=6)

    def test_t_p_value_zero_at_infinity(self):
        self.assertEqual(sbc.t_two_sided_p(math.inf, 5), 0.0)
        self.assertEqual(sbc.t_two_sided_p(0.0, 5), 1.0)

    def test_t_p_value_is_monotone_in_t(self):
        prev = 1.0
        for t in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
            p = sbc.t_two_sided_p(t, 6)
            self.assertLess(p, prev)
            self.assertGreater(p, 0.0)
            prev = p

    def test_cross_check_against_scipy(self):
        try:
            from scipy import stats
        except ImportError:
            self.skipTest("scipy not installed")
        for t, df in ((0.5, 2), (2.0, 3), (2.228139, 10), (4.5, 7)):
            self.assertAlmostEqual(sbc.t_two_sided_p(t, df),
                                   2.0 * stats.t.sf(t, df), places=11)
        for p in (0.001, 0.025, 0.5, 0.8, 0.975, 0.999):
            self.assertAlmostEqual(sbc.norm_ppf(p), float(stats.norm.ppf(p)),
                                   places=10)


class TestSelectionBias(unittest.TestCase):
    def test_closed_forms_for_small_n(self):
        """E[max of 1] = 0, of 2 = 1/sqrt(pi), of 3 = 3/(2 sqrt(pi))."""
        self.assertAlmostEqual(sbc.expected_max_of_n_normals(1), 0.0, places=13)
        self.assertAlmostEqual(sbc.expected_max_of_n_normals(2),
                               1.0 / math.sqrt(math.pi), places=13)
        self.assertAlmostEqual(sbc.expected_max_of_n_normals(3),
                               3.0 / (2.0 * math.sqrt(math.pi)), places=13)

    def test_closed_forms_agree_with_the_general_integral(self):
        """The closed forms must not disagree with the method they bypass."""
        for n, expected in ((1, 0.0),
                            (2, 1.0 / math.sqrt(math.pi)),
                            (3, 3.0 / (2.0 * math.sqrt(math.pi)))):
            self.assertAlmostEqual(_integrated_emax(n), expected, places=10)
            self.assertAlmostEqual(sbc.expected_max_of_n_normals(n), expected,
                                   places=13)

    def test_values_for_larger_n(self):
        """Confirmed independently by fixed-seed Monte Carlo; see
        test_monte_carlo_agrees_with_the_integral for the cross-check."""
        for n, expected in ((4, 1.0293753730), (5, 1.1629644736),
                            (10, 1.5387527308), (20, 1.8674750598),
                            (100, 2.5075936364)):
            self.assertAlmostEqual(sbc.expected_max_of_n_normals(n),
                                   expected, places=8)

    def test_integration_has_converged(self):
        """Doubling the resolution must not move the answer."""
        for n in (5, 20, 100):
            coarse = sbc.expected_max_of_n_normals(n, intervals=2000)
            fine = sbc.expected_max_of_n_normals(n, intervals=8000)
            self.assertAlmostEqual(coarse, fine, places=11)

    def test_monte_carlo_agrees_with_the_integral(self):
        """Deterministic fixed-seed Monte Carlo: an independent path to the
        same numbers, so the table above is not self-referential."""
        rng = random.Random(12345)
        for n in (2, 3, 5, 10, 20):
            trials = 120_000
            total = sum(max(rng.gauss(0.0, 1.0) for _ in range(n))
                        for _ in range(trials))
            mc = total / trials
            self.assertAlmostEqual(mc, sbc.expected_max_of_n_normals(n), delta=0.01)

    def test_bias_grows_with_n_and_sd(self):
        prev = 0.0
        for n in range(2, 12):
            b = sbc.expected_selection_bias(0.1, n)
            self.assertGreater(b, prev)
            prev = b
        self.assertEqual(sbc.expected_selection_bias(0.1, 1), 0.0)

    def test_bias_scales_linearly_with_sd(self):
        self.assertAlmostEqual(sbc.expected_selection_bias(0.2, 5),
                               2.0 * sbc.expected_selection_bias(0.1, 5), places=13)
        self.assertAlmostEqual(sbc.expected_selection_bias(0.06647214959941074, 3),
                               0.06647214959941074
                               * sbc.expected_max_of_n_normals(3), places=12)

    def test_bias_is_zero_without_spread(self):
        self.assertEqual(sbc.expected_selection_bias(0.0, 5), 0.0)

    def test_rejects_n_below_one(self):
        with self.assertRaises(ValueError):
            sbc.expected_max_of_n_normals(0)


class TestSeedSummary(unittest.TestCase):
    def _three_seeds(self):
        return sbc.seed_summary({101: WIND_H6[0], 202: WIND_H6[1], 303: WIND_H6[2]})

    def test_reports_mean_sd_sem_and_best_separately(self):
        s = self._three_seeds()
        self.assertEqual(s["status"], "OK")
        self.assertEqual(s["n_seeds"], 3)
        self.assertEqual(s["seeds"], [101, 202, 303])
        self.assertAlmostEqual(s["mean"], WIND_H6_MEAN, places=13)
        self.assertAlmostEqual(s["sd"], WIND_H6_SD, places=11)
        self.assertAlmostEqual(s["sem"], WIND_H6_SEM, places=11)
        self.assertEqual(s["best_seed"], 101)
        self.assertAlmostEqual(s["best_value"], WIND_H6[0], places=13)
        self.assertEqual(s["min"], min(WIND_H6))
        self.assertAlmostEqual(s["range"], max(WIND_H6) - min(WIND_H6), places=13)

    def test_best_is_never_the_mean_when_seeds_differ(self):
        """The whole point: best and mean are different numbers."""
        s = self._three_seeds()
        self.assertNotAlmostEqual(s["best_value"], s["mean"], places=4)
        self.assertAlmostEqual(s["best_value"],
                               s["mean"] - s["best_vs_mean_gap"], places=13)
        self.assertGreater(s["best_vs_mean_gap"], 0.0)
        self.assertAlmostEqual(s["best_value"],
                               s["values_by_seed"][str(s["best_seed"])], places=13)

    def test_best_gap_is_always_the_full_distance_to_the_floor(self):
        s = self._three_seeds()
        self.assertAlmostEqual(s["best_value"], s["min"], places=13)
        self.assertAlmostEqual(s["best_vs_mean_gap"], s["mean"] - s["min"], places=13)

    def test_reports_size_of_optimistic_bias(self):
        s = self._three_seeds()
        expected_gap = s["sd"] * sbc.expected_max_of_n_normals(3)
        self.assertAlmostEqual(s["expected_best_vs_mean_gap"], expected_gap, places=12)
        self.assertAlmostEqual(s["expected_best_vs_mean_gap_pct"],
                               100.0 * expected_gap / s["mean"], places=11)
        # 3 seeds at this seed sd: the best seed is expected to look about 8.7%
        # better than the arm actually is.
        self.assertGreater(s["expected_best_vs_mean_gap_pct"], 8.0)
        self.assertLess(s["expected_best_vs_mean_gap_pct"], 9.5)

    def test_bias_grows_with_more_seeds(self):
        three = sbc.seed_summary({i: WIND_H6[0] + 0.01 * i for i in range(3)})
        ten = sbc.seed_summary({i: WIND_H6[0] + 0.01 * i for i in range(10)})
        self.assertGreater(ten["expected_best_vs_mean_gap_pct"],
                           three["expected_best_vs_mean_gap_pct"])

    def test_debiased_estimate_is_best_plus_expected_gap(self):
        """best + sd*E[max_N] removes the bias IN EXPECTATION. For a realised
        draw it is only as close to the mean as that draw happened to be."""
        s = self._three_seeds()
        self.assertAlmostEqual(s["debiased_arm_estimate"],
                               s["best_value"] + s["expected_best_vs_mean_gap"],
                               places=13)
        # It must land much nearer the mean than the raw best seed does.
        self.assertLess(abs(s["debiased_arm_estimate"] - s["mean"]),
                        abs(s["best_value"] - s["mean"]))
        # And it is unbiased, so the bias it removes is the expectation, not the
        # realised gap.
        self.assertNotAlmostEqual(s["debiased_arm_estimate"], s["mean"], places=4)

    def test_observed_gap_is_one_draw_not_the_expectation(self):
        s = self._three_seeds()
        self.assertNotAlmostEqual(s["best_vs_mean_gap"],
                                  s["expected_best_vs_mean_gap"], places=4)
        self.assertIsNotNone(s["observed_over_expected_gap"])
        self.assertAlmostEqual(s["observed_over_expected_gap"],
                               s["best_vs_mean_gap"] / s["expected_best_vs_mean_gap"],
                               places=10)

    def test_ci95_of_mean_brackets_the_mean(self):
        s = self._three_seeds()
        lo, hi = s["ci95_of_mean"]
        self.assertLess(lo, s["mean"])
        self.assertGreater(hi, s["mean"])
        self.assertAlmostEqual(hi - lo,
                               2.0 * sbc.t_critical_two_sided(0.05, 2) * s["sem"],
                               places=12)

    def test_best_seed_is_not_outside_the_ci_of_the_mean(self):
        """
        With three seeds the t critical value is 4.30, so the 95% CI of the mean
        spans +/- 4.30 * sem = +/- 0.165 on an MAE of 0.650. The best seed sits
        0.052 below the mean -- comfortably INSIDE that interval. Three seeds
        cannot even tell the best seed from the mean it was drawn from; a single
        seed quoted as "the" result is not a summary of anything.
        """
        s = self._three_seeds()
        lo, hi = s["ci95_of_mean"]
        # the best seed lies INSIDE the interval, not outside it
        self.assertLess(lo, s["best_value"])
        self.assertGreater(hi, s["best_value"])
        self.assertLess(s["best_vs_mean_gap"], hi - lo)

    def test_single_seed_is_insufficient_evidence(self):
        s = sbc.seed_summary({101: 0.61})
        self.assertEqual(s["status"], sbc.INSUFFICIENT)
        self.assertIn("only 1 seed", s["reason"])
        self.assertIsNone(s["sd"])
        self.assertIsNone(s["sem"])
        # the number itself is still reported, but never as a summary
        self.assertAlmostEqual(s["mean"], 0.61, places=13)
        self.assertAlmostEqual(s["best_value"], 0.61, places=13)
        self.assertNotIn("ci95_of_mean", s)
        self.assertNotIn("debiased_arm_estimate", s)

    def test_identical_seeds_have_zero_spread(self):
        s = sbc.seed_summary({1: 0.5, 2: 0.5, 3: 0.5})
        self.assertEqual(s["status"], "OK")
        self.assertAlmostEqual(s["sd"], 0.0, places=15)
        self.assertAlmostEqual(s["best_vs_mean_gap"], 0.0, places=15)
        self.assertAlmostEqual(s["expected_best_vs_mean_gap"], 0.0, places=15)
        self.assertAlmostEqual(s["debiased_arm_estimate"], 0.5, places=15)

    def test_cv_pct_of_mean(self):
        s = sbc.seed_summary({1: 1.0, 2: 2.0, 3: 3.0})
        # mean 2, sample sd exactly 1 -> CV exactly 50%
        self.assertAlmostEqual(s["cv_pct_of_mean"], 50.0, places=13)

    def test_string_seed_keys_are_coerced(self):
        s = sbc.seed_summary({"101": 0.6, "202": 0.7})
        self.assertEqual(s["seeds"], [101, 202])
        self.assertAlmostEqual(s["mean"], 0.65, places=13)

    def test_higher_is_better_is_supported(self):
        s = sbc.seed_summary({1: 0.5, 2: 0.7, 3: 0.6}, lower_is_better=False)
        self.assertEqual(s["best_seed"], 2)
        self.assertAlmostEqual(s["best_value"], 0.7, places=13)
        self.assertAlmostEqual(s["best_vs_mean_gap"], 0.7 - 0.6, places=13)

    def test_empty_arm(self):
        s = sbc.seed_summary({})
        self.assertEqual(s["status"], sbc.INSUFFICIENT)
        self.assertEqual(s["n_seeds"], 0)


class TestPairedComparison(unittest.TestCase):
    def test_paired_differences_use_matched_seeds(self):
        a = {101: 0.60, 202: 0.63, 303: 0.72}
        b = {101: 0.62, 202: 0.65, 303: 0.70}
        r = sbc.paired_comparison(a, b)
        self.assertEqual(r["test"], "paired")
        self.assertEqual(r["n_pairs"], 3)
        self.assertEqual(r["common_seeds"], [101, 202, 303])
        self.assertAlmostEqual(r["differences_by_seed"]["101"], -0.02, places=13)
        self.assertAlmostEqual(r["differences_by_seed"]["202"], -0.02, places=13)
        self.assertAlmostEqual(r["differences_by_seed"]["303"], 0.02, places=13)
        # diffs (-0.02, -0.02, +0.02): mean = -0.0066666...
        #   devs from mean: -0.0133333, -0.0133333, +0.0266667
        #   sumsq = 1.77778e-4 + 1.77778e-4 + 7.11111e-4 = 1.0666667e-3
        #   var  = 5.3333333e-4, sd = 0.02309401, sem = sd/sqrt(3) = 0.01333333
        self.assertAlmostEqual(r["mean_difference"], -0.02 / 3.0, places=13)
        self.assertAlmostEqual(r["sd_of_differences"],
                               math.sqrt(5.3333333333e-4), places=11)
        self.assertAlmostEqual(r["sem_of_difference"],
                               math.sqrt(5.3333333333e-4 / 3.0), places=11)
        self.assertAlmostEqual(r["two_sigma_halfwidth"],
                               2.0 * math.sqrt(5.3333333333e-4 / 3.0), places=11)
        # Cohen's dz = -0.0066667 / 0.0230940 = -1/(2*sqrt(3))
        self.assertAlmostEqual(r["cohens_dz"], -1.0 / (2.0 * math.sqrt(3.0)),
                               places=11)

    def test_unmatched_seeds_are_excluded_not_imputed(self):
        r = sbc.paired_comparison({1: 0.6, 2: 0.7, 3: 0.8}, {2: 0.75, 3: 0.85})
        self.assertEqual(r["n_pairs"], 2)
        self.assertEqual(r["common_seeds"], [2, 3])
        self.assertNotIn("101", r["differences_by_seed"])

    def test_sign_reversing_across_seeds_is_not_distinguishable(self):
        a = {101: 0.60, 202: 0.63, 303: 0.72}
        b = {101: 0.62, 202: 0.65, 303: 0.70}
        r = sbc.paired_comparison(a, b)
        self.assertEqual(r["verdict"], sbc.NOT_DISTINGUISHABLE)
        self.assertIn("NOT distinguishable", r["answer"])
        self.assertLess(abs(r["mean_difference"]), r["two_sigma_halfwidth"])

    def test_consistent_large_difference_is_distinguishable(self):
        a = {101: 0.60, 202: 0.60, 303: 0.60}
        b = {101: 0.80, 202: 0.80, 303: 0.80}
        r = sbc.paired_comparison(a, b)
        self.assertEqual(r["verdict"], sbc.DISTINGUISHABLE)
        self.assertIn("distinguishable from seed noise", r["answer"])
        self.assertAlmostEqual(r["mean_difference"], -0.20, places=13)
        self.assertAlmostEqual(r["sd_of_differences"], 0.0, places=13)
        self.assertEqual(r["better_arm"], "A")
        self.assertEqual(r["sign_agreement_pct"], 100.0)
        self.assertEqual(r["p_value_parametric"], 0.0)
        self.assertEqual(r["p_value_exact_sign_test"], 0.25)

    def test_worse_arm_naming_is_correct_both_ways(self):
        a = {1: 0.9, 2: 0.9, 3: 0.9}
        b = {1: 0.5, 2: 0.5, 3: 0.5}
        self.assertEqual(sbc.paired_comparison(a, b)["better_arm"], "B")
        self.assertEqual(sbc.paired_comparison(b, a)["better_arm"], "A")

    def test_zero_sd_of_differences_is_distinguishable_not_a_crash(self):
        """Identical arms differ by a constant: sd(diff) = 0, sem = 0, t = inf.
        The gate must read that as an unambiguous, real difference rather than
        dividing by zero or calling it indistinguishable."""
        a = {1: 0.90, 2: 0.90, 3: 0.90}
        b = {1: 0.80, 2: 0.80, 3: 0.80}
        r = sbc.paired_comparison(a, b)
        self.assertEqual(r["sd_of_differences"], 0.0)
        self.assertEqual(r["sem_of_difference"], 0.0)
        self.assertEqual(r["two_sigma_halfwidth"], 0.0)
        self.assertEqual(r["verdict"], sbc.DISTINGUISHABLE)
        self.assertEqual(r["p_value_parametric"], 0.0)
        self.assertIsNone(r["cohens_dz"])

    def test_two_sigma_gate_is_exactly_t_versus_two(self):
        """The release gate's 2-sigma rule is |t| > 2 when sigma = 2."""
        a = {1: 1.0, 2: 1.2, 3: 0.9, 4: 1.1, 5: 1.0}
        b = {1: 1.1, 2: 1.3, 3: 1.0, 4: 1.2, 5: 1.1}
        r = sbc.paired_comparison(a, b, sigma=2.0)
        self.assertAlmostEqual(r["t"], r["mean_difference"] / r["sem_of_difference"],
                               places=12)
        self.assertEqual(r["verdict"] == sbc.DISTINGUISHABLE, abs(r["t"]) > 2.0)

    def test_tighter_sigma_demands_more(self):
        # differences built so |t| is exactly 2.0: with 6 pairs, sem = 0.008944,
        # mean diff = 0.014606 -> t = 2.0. So a 1-sigma gate passes and a
        # 3-sigma gate must not.
        spread = [0.01 * v for v in (-1.0, -1.0, 0.0, 0.0, 1.0, 1.0)]
        centre = 2.0 * sbc.sample_sd(spread) / math.sqrt(6)
        a = {i + 1: 0.60 + centre + e for i, e in enumerate(spread)}
        b = {i + 1: 0.60 for i in range(6)}
        mid = sbc.paired_comparison(a, b, sigma=2.0)
        self.assertAlmostEqual(abs(mid["t"]), 2.0, places=9)
        self.assertEqual(sbc.paired_comparison(a, b, sigma=1.0)["verdict"],
                         sbc.DISTINGUISHABLE)
        self.assertEqual(sbc.paired_comparison(a, b, sigma=3.0)["verdict"],
                         sbc.NOT_DISTINGUISHABLE)
        # the gate is a pure rescale of the same t
        self.assertAlmostEqual(sbc.paired_comparison(a, b, sigma=1.5)
                               ["two_sigma_halfwidth"],
                               1.5 * mid["sem_of_difference"], places=13)

    def test_single_shared_seed_cannot_be_paired(self):
        r = sbc.paired_comparison({101: 0.60}, {101: 0.62})
        self.assertEqual(r["status"], sbc.INSUFFICIENT)
        self.assertEqual(r["verdict"], sbc.INSUFFICIENT)
        self.assertIn(">= 2 seed-matched pairs", r["reason"])

    def test_no_shared_seeds_cannot_be_paired(self):
        r = sbc.paired_comparison({101: 0.60, 202: 0.61}, {303: 0.62, 404: 0.63})
        self.assertEqual(r["status"], sbc.INSUFFICIENT)
        self.assertEqual(r["verdict"], sbc.INSUFFICIENT)

    def test_ci95_uses_t_critical_for_small_df(self):
        a = {1: 1.00, 2: 1.20, 3: 0.90, 4: 1.10}
        b = {1: 1.10, 2: 1.30, 3: 1.00, 4: 1.20}
        r = sbc.paired_comparison(a, b)
        self.assertEqual(r["df"], 3)
        tcrit = sbc.t_critical_two_sided(0.05, 3)
        self.assertAlmostEqual(tcrit, 3.182446, places=5)
        self.assertAlmostEqual(r["ci95_of_difference"][0],
                               r["mean_difference"] - tcrit * r["sem_of_difference"],
                               places=12)
        self.assertAlmostEqual(r["ci95_of_difference"][1],
                               r["mean_difference"] + tcrit * r["sem_of_difference"],
                               places=12)

    def test_min_attainable_sign_test_p_is_reported(self):
        r = sbc.paired_comparison({1: 0.6, 2: 0.7, 3: 0.8}, {1: 0.6, 2: 0.7, 3: 0.8})
        self.assertEqual(r["n_pairs"], 3)
        self.assertAlmostEqual(r["min_attainable_sign_test_p"], 0.25, places=13)
        self.assertGreater(r["min_attainable_sign_test_p"], 0.05)


class TestUnpairedAndDispatch(unittest.TestCase):
    def test_welch_requires_two_seeds_each(self):
        r = sbc.welch_comparison({1: 0.5}, {2: 0.6, 3: 0.7})
        self.assertEqual(r["status"], sbc.INSUFFICIENT)
        self.assertIn(">= 2 seeds per arm", r["reason"])

    def test_welch_on_disjoint_seeds(self):
        a = {1: 0.60, 2: 0.62, 3: 0.61}
        b = {4: 0.70, 5: 0.72, 6: 0.71}
        r = sbc.welch_comparison(a, b)
        self.assertEqual(r["test"], "welch")
        self.assertEqual(r["status"], "OK")
        self.assertAlmostEqual(r["mean_a"], 0.61, places=13)
        self.assertAlmostEqual(r["mean_b"], 0.71, places=13)
        self.assertAlmostEqual(r["mean_difference"], -0.10, places=13)
        self.assertEqual(r["verdict"], sbc.DISTINGUISHABLE)
        # equal sample sizes and equal sds -> Welch df is exactly n_a + n_b - 2
        self.assertAlmostEqual(r["df"], 4.0, places=9)

    def test_welch_weldf_is_at_most_na_plus_nb_minus_2(self):
        a = {1: 0.60, 2: 0.90}
        b = {3: 0.70, 4: 0.71, 5: 0.72, 6: 0.73}
        r = sbc.welch_comparison(a, b)
        self.assertLessEqual(r["df"], 2 + 4 - 2 + 1e-9)
        self.assertGreater(r["df"], 0.0)

    def test_compare_arms_dispatches_to_paired_when_seeds_match(self):
        r = sbc.compare_arms({1: 0.5, 2: 0.6}, {1: 0.55, 2: 0.65})
        self.assertEqual(r["test"], "paired")
        self.assertEqual(r["n_common_seeds"], 2)

    def test_compare_arms_dispatches_to_welch_when_seeds_differ(self):
        r = sbc.compare_arms({1: 0.5, 2: 0.6}, {3: 0.55, 4: 0.65})
        self.assertEqual(r["test"], "welch")
        self.assertEqual(r["n_common_seeds"], 0)

    def test_one_seed_versus_three_is_insufficient_evidence(self):
        """The case the release gate cares about: a single-seed arm."""
        r = sbc.compare_arms({101: 0.60}, {1: 0.5, 2: 0.6, 3: 0.55})
        self.assertEqual(r["status"], sbc.INSUFFICIENT)
        self.assertEqual(r["verdict"], sbc.INSUFFICIENT)
        self.assertIn("design", r)
        self.assertNotIn("mean_difference", r)

    def test_empty_arm_is_insufficient_evidence(self):
        r = sbc.compare_arms({}, {1: 0.5, 2: 0.6})
        self.assertEqual(r["verdict"], sbc.INSUFFICIENT)

    def test_floors_sorted_alphabetically(self):
        for s in (sbc.DISTINGUISHABLE, sbc.NOT_DISTINGUISHABLE, sbc.INSUFFICIENT):
            self.assertEqual(s, s.upper())


class TestExactSignTest(unittest.TestCase):
    """The floor that no amount of modelling removes."""

    def test_p_values_are_powers_of_two_capped_at_one(self):
        self.assertAlmostEqual(sbc.exact_sign_test_p(3, 0), 0.25, places=13)
        self.assertAlmostEqual(sbc.exact_sign_test_p(3, 3), 0.25, places=13)
        self.assertAlmostEqual(sbc.exact_sign_test_p(2, 0), 0.50, places=13)
        self.assertAlmostEqual(sbc.exact_sign_test_p(5, 0), 0.0625, places=13)
        self.assertAlmostEqual(sbc.exact_sign_test_p(6, 0), 0.03125, places=13)
        self.assertEqual(sbc.exact_sign_test_p(1, 1), 1.0)
        self.assertEqual(sbc.exact_sign_test_p(1, 0), 1.0)
        self.assertEqual(sbc.exact_sign_test_p(0, 0), 1.0)

    def test_three_seeds_can_never_be_significant(self):
        """n=3 => smallest attainable two-sided p is 0.25, so 3 is never enough."""
        for positives in range(4):
            self.assertGreaterEqual(sbc.exact_sign_test_p(3, positives), 0.25)
        self.assertGreater(sbc.exact_sign_test_p(3, 0), 0.05)

    def test_floor_is_six_pairs(self):
        self.assertEqual(sbc.min_pairs_for_exact_sign_test(0.05), 6)
        self.assertGreaterEqual(sbc.exact_sign_test_p(5, 0), 0.05)
        self.assertLess(sbc.exact_sign_test_p(6, 0), 0.05)


class TestRequiredSeeds(unittest.TestCase):
    def test_paired_formula_is_hand_checkable(self):
        # sd = 0.1, effect = 0.1, alpha 0.05, power 0.8
        #   z = 1.959964 + 0.841621 = 2.801585
        #   n = (2.801585 * 0.1 / 0.1)^2 = 7.8489 -> 8
        self.assertEqual(sbc.required_seeds_paired(0.1, 0.1), 8)
        # halving the effect quadruples n
        self.assertEqual(sbc.required_seeds_paired(0.1, 0.05), 32)
        # doubling the effect quarters n
        self.assertEqual(sbc.required_seeds_paired(0.1, 0.2), 2)
        # a 1-sigma effect at this sd needs 8, so 3 seeds is far too few
        self.assertGreater(sbc.required_seeds_paired(0.1, 0.1), 3)

    def test_zero_effect_means_infinite_seeds(self):
        self.assertEqual(sbc.required_seeds_paired(0.1, 0.0), 0)
        self.assertEqual(sbc.required_seeds_paired(None, 0.1), 0)

    def test_paired_design_is_cheaper_than_two_sample_at_equal_sd(self):
        """
        At equal total budget the paired design is strictly more efficient,
        because seed-to-seed variation cancels instead of being paid for twice.
        sd = 0.1, effect = 0.1:
          paired      -> 8 pairs  = 8 runs per arm = 16 runs total
          two-sample  -> 27 per arm                = 54 runs total
        """
        n_paired = sbc.required_seeds_paired(0.1, 0.1)
        self.assertEqual(n_paired, 8)
        n_two = sbc.required_seeds_two_sample(0.1, 0.1, 0.1, 27)
        self.assertEqual(n_two, 27)
        # 26 per arm is not yet enough
        self.assertGreater(sbc.required_seeds_two_sample(0.1, 0.1, 0.1, 26), 26)
        self.assertLess(2 * n_paired, 2 * n_two)

    def test_two_sample_is_attainable_for_a_large_effect(self):
        n = sbc.required_seeds_two_sample(0.1, 0.1, 0.5, 3)
        self.assertEqual(n, 2)

    def test_two_sample_is_unattainable_with_too_few_seeds_in_arm_a(self):
        """n_a = 3 puts a floor of sd_a/sqrt(3) on the standard error, so a
        1-sd effect can never reach 80% power however large n_b becomes.
        Reporting a number here would be a lie."""
        self.assertIsNone(sbc.required_seeds_two_sample(0.1, 0.1, 0.1, 3))

    def test_two_sample_shrinks_as_n_a_grows(self):
        # sd 0.1, effect 0.2. n_a = 5: n_b = 10 is the first to reach 80% power
        # (se = sqrt(0.01/5 + 0.01/10) = 0.05477, effect/se = 3.652, power 0.802).
        # n_a = 20: n_b = 4 suffices (se = 0.05774, effect/se = 3.464).
        few = sbc.required_seeds_two_sample(0.1, 0.1, 0.2, 5)
        many = sbc.required_seeds_two_sample(0.1, 0.1, 0.2, 20)
        self.assertEqual(few, 10)
        self.assertEqual(many, 4)
        self.assertLess(many, few)

    def test_two_sample_rejects_degenerate_input(self):
        self.assertIsNone(sbc.required_seeds_two_sample(0.1, 0.1, 0.0, 5))
        self.assertIsNone(sbc.required_seeds_two_sample(0.1, 0.1, 0.1, 1))
        self.assertIsNone(sbc.required_seeds_two_sample(None, 0.1, 0.1, 5))

    def test_two_sample_with_zero_spread_needs_no_extra_seeds(self):
        # sd = 0 on both sides: the arms differ by a constant, so the
        # difference is already exact. Must not divide by a zero standard error.
        self.assertEqual(sbc.required_seeds_two_sample(0.0, 0.0, 0.2, 3), 2)

    def test_recommendation_respects_both_floors(self):
        rec = sbc.min_seeds_recommendation(seed_sd=0.1, arm_mean=0.65,
                                           effects_pct=(2.0, 5.0, 10.0))
        self.assertEqual(rec["status"], "OK")
        self.assertEqual(rec["sign_test_floor_pairs"], 6)
        sizes = {e["effect_pct_of_mean"]: e["seeds_for_paired_t"]
                 for e in rec["by_effect_size"]}
        # 5% of 0.65 = 0.0325; n = (2.801585*0.1/0.0325)^2 = 74.3 -> 75
        self.assertEqual(sizes[5.0], 75)
        # 10% of 0.65 = 0.065; n = (2.801585*0.1/0.065)^2 = 18.56 -> 19
        self.assertEqual(sizes[10.0], 19)
        # 2% of 0.65 = 0.013; n = 464.5 -> 465
        self.assertEqual(sizes[2.0], 465)
        self.assertTrue(all(e["attainable_by_exact_sign_test"]
                            for e in rec["by_effect_size"]))
        self.assertEqual(rec["recommended_min_seeds"], 75)

    def test_recommendation_never_drops_below_the_sign_test_floor(self):
        """Even a huge effect cannot license fewer than the exact test needs."""
        rec = sbc.min_seeds_recommendation(seed_sd=0.001, arm_mean=1.0,
                                           effects_pct=(50.0,))
        self.assertEqual(rec["recommended_min_seeds"], 6)

    def test_recommendation_never_drops_below_three(self):
        rec = sbc.min_seeds_recommendation(seed_sd=1e-9, arm_mean=1.0,
                                           effects_pct=(99.0,))
        self.assertEqual(rec["recommended_min_seeds"], 6)

    def test_recommendation_reports_cv(self):
        rec = sbc.min_seeds_recommendation(seed_sd=WIND_H6_SD, arm_mean=WIND_H6_MEAN)
        # 0.06647215 / 0.65007578 = 10.225%
        self.assertAlmostEqual(rec["seed_cv_pct"],
                               100.0 * WIND_H6_SD / WIND_H6_MEAN, places=9)
        self.assertAlmostEqual(rec["seed_cv_pct"], 10.225, places=2)

    def test_recommendation_without_sd_is_insufficient(self):
        rec = sbc.min_seeds_recommendation(seed_sd=None, arm_mean=0.65)
        self.assertEqual(rec["status"], sbc.INSUFFICIENT)
        rec = sbc.min_seeds_recommendation(seed_sd=0.1, arm_mean=0.0)
        self.assertEqual(rec["status"], sbc.INSUFFICIENT)

    def test_five_percent_is_the_recommendation_anchor(self):
        """The 2-8% band this cycle's comparisons produced has its middle at 5%."""
        rec = sbc.min_seeds_recommendation(seed_sd=0.1, arm_mean=0.65)
        self.assertIn("5% effect", rec["recommended_min_seeds_basis"])


class TestHalfSplitConsistency(unittest.TestCase):
    def test_consistent_when_same_arm_wins_both_halves(self):
        a = {"first_half": {1: 0.50, 2: 0.52}, "second_half": {1: 0.60, 2: 0.62}}
        b = {"first_half": {1: 0.55, 2: 0.57}, "second_half": {1: 0.65, 2: 0.67}}
        r = sbc.half_split_consistency(a, b)
        self.assertEqual(r["status"], "OK")
        self.assertTrue(r["consistent"])
        self.assertEqual(r["verdict"], "CONSISTENT_ACROSS_HALVES")
        self.assertLess(r["per_half"]["first_half"]["mean_difference"], 0.0)
        self.assertLess(r["per_half"]["second_half"]["mean_difference"], 0.0)
        self.assertEqual(r["per_half"]["first_half"]["n_seeds_a"], 2)

    def test_inconsistent_when_ordering_reverses(self):
        a = {"first_half": {1: 0.50}, "second_half": {1: 0.70}}
        b = {"first_half": {1: 0.55}, "second_half": {1: 0.60}}
        r = sbc.half_split_consistency(a, b)
        self.assertFalse(r["consistent"])
        self.assertEqual(r["verdict"], "INCONSISTENT_ACROSS_HALVES")
        self.assertIn("REVERSES", r["note"])

    def test_identical_halves_count_as_consistent(self):
        a = {"first_half": {1: 0.5}, "second_half": {1: 0.5}}
        b = {"first_half": {1: 0.6}, "second_half": {1: 0.6}}
        r = sbc.half_split_consistency(a, b)
        self.assertTrue(r["consistent"])

    def test_needs_two_halves(self):
        r = sbc.half_split_consistency({"first_half": {1: 0.5}},
                                       {"first_half": {1: 0.6}})
        self.assertEqual(r["status"], sbc.INSUFFICIENT)

    def test_empty_half_is_insufficient(self):
        a = {"first_half": {}, "second_half": {1: 0.5}}
        b = {"first_half": {1: 0.5}, "second_half": {1: 0.6}}
        r = sbc.half_split_consistency(a, b)
        self.assertEqual(r["status"], sbc.INSUFFICIENT)

    def test_string_seed_keys_are_coerced(self):
        a = {"first_half": {"101": 0.5}, "second_half": {"101": 0.5}}
        b = {"first_half": {"101": 0.6}, "second_half": {"101": 0.6}}
        r = sbc.half_split_consistency(a, b)
        self.assertEqual(r["per_half"]["first_half"]["n_seeds_a"], 1)


def _make_arm(root, tag, horizons, scorecard=True, code_commit="abc123"):
    d = os.path.join(root, tag)
    os.makedirs(d, exist_ok=True)
    for h in horizons:
        with open(os.path.join(d, f"candidate_h{h}h.pt"), "w") as f:
            f.write("x")
        with open(os.path.join(d, f"candidate_h{h}h_manifest.json"), "w") as f:
            f.write('{"code_commit": "%s", "checkpoint_sha256": "z"}' % code_commit)
    if scorecard:
        with open(os.path.join(d, "predictive_quality_scorecard.json"), "w") as f:
            f.write("{}")


class TestDiscoverArms(unittest.TestCase):
    def test_parses_floor_and_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_arm(tmp, "relu_s101", (1, 6))
            _make_arm(tmp, "leaky_s202", (1, 6))
            arms = sbc.discover_arms(tmp)
            self.assertEqual(sorted(arms), ["leaky", "relu"])
            self.assertEqual(sorted(arms["relu"]), [101])
            self.assertEqual(arms["leaky"][202]["floor"], "leaky")
            self.assertEqual(arms["leaky"][202]["code_commits"], ["abc123"])
            self.assertTrue(arms["relu"][101]["has_scorecard"])
            self.assertTrue(arms["relu"][101]["usable"])

    def test_partial_arm_is_flagged_not_hidden(self):
        """A half-finished sweep must not look like a complete arm."""
        with tempfile.TemporaryDirectory() as tmp:
            _make_arm(tmp, "leaky_s303", (1, 3, 6), scorecard=False)
            rec = sbc.discover_arms(tmp)["leaky"][303]
            self.assertEqual(rec["n_checkpoints"], 3)
            self.assertFalse(rec["complete"])
            self.assertFalse(rec["has_scorecard"])
            self.assertFalse(rec["usable"])
            self.assertEqual([h for h, ok in rec["checkpoints"].items() if ok],
                             [1, 3, 6])
            self.assertEqual([h for h, ok in rec["checkpoints"].items() if not ok],
                             [12, 24])

    def test_complete_arm_needs_all_five_horizons(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_arm(tmp, "relu_s101", sbc.HORIZONS)
            rec = sbc.discover_arms(tmp)["relu"][101]
            self.assertTrue(rec["complete"])
            self.assertEqual(rec["n_checkpoints"], 5)

    def test_multi_digit_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_arm(tmp, "leaky_s1234", (1,))
            self.assertEqual(sorted(sbc.discover_arms(tmp)["leaky"]), [1234])

    def test_ignores_non_arm_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "not_an_arm"))
            os.makedirs(os.path.join(tmp, "no_seed_suffix"))
            with open(os.path.join(tmp, "stray.json"), "w") as f:
                f.write("{}")
            self.assertEqual(sbc.discover_arms(tmp), {})

    def test_corrupt_manifest_does_not_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = os.path.join(tmp, "relu_s101")
            os.makedirs(d)
            with open(os.path.join(d, "candidate_h1h_manifest.json"), "w") as f:
                f.write("{not json")
            with open(os.path.join(d, "candidate_h1h.pt"), "w") as f:
                f.write("x")
            rec = sbc.discover_arms(tmp)["relu"][101]
            self.assertEqual(rec["code_commits"], [])

    def test_missing_dir_is_empty(self):
        self.assertEqual(sbc.discover_arms(os.path.join("no", "such", "dir")), {})


class TestLoadScorecardMae(unittest.TestCase):
    def _card(self):
        return {"horizon_evaluations": {
            "horizon_1h": {"candidate_featured_model": {
                "temperature": {"mae": 0.66}, "wind_speed": {"mae": 0.34}}},
            "horizon_6h": {"candidate_featured_model": {
                "temperature": {"mae": 2.53}, "wind_speed": {"mae": 0.59}}}}}

    def _write(self, tmp, tag, card):
        d = os.path.join(tmp, tag)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "predictive_quality_scorecard.json"), "w") as f:
            if isinstance(card, str):
                f.write(card)
            else:
                json.dump(card, f)
        return d

    def test_reads_candidate_mae_per_horizon(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write(tmp, "relu_s101", self._card())
            scores = sbc.load_scorecard_mae(tmp)
            self.assertAlmostEqual(scores["relu"][101][1]["temperature"], 0.66, places=12)
            self.assertAlmostEqual(scores["relu"][101][6]["wind_speed"], 0.59, places=12)

    def test_skips_arms_without_scorecard(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "leaky_s303"))
            self.assertEqual(sbc.load_scorecard_mae(tmp), {})

    def test_scorecard_without_candidate_block_is_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write(tmp, "relu_s101", {"horizon_evaluations": {"horizon_1h": {}}})
            self.assertEqual(sbc.load_scorecard_mae(tmp), {})

    def test_corrupt_scorecard_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write(tmp, "relu_s101", "{not json")
            self.assertEqual(sbc.load_scorecard_mae(tmp), {})

    def test_odd_horizon_keys_do_not_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write(tmp, "relu_s101", {"horizon_evaluations": {
                "weird": {},
                "horizon_9h": {"candidate_featured_model": {
                    "wind_speed": {"mae": 1.0}}}}})
            scores = sbc.load_scorecard_mae(tmp, channels=("wind_speed",))
            self.assertEqual(list(scores["relu"][101]), [9])

    def test_only_the_candidate_is_read_not_the_baselines(self):
        """The scorecard holds a dozen models; only the candidate may be scored."""
        with tempfile.TemporaryDirectory() as tmp:
            self._write(tmp, "relu_s101", {"horizon_evaluations": {"horizon_6h": {
                "persistence": {"wind_speed": {"mae": 999.0}},
                "climatology": {"wind_speed": {"mae": 888.0}},
                "candidate_featured_model": {"wind_speed": {"mae": 0.59}}}}})
            scores = sbc.load_scorecard_mae(tmp, channels=("wind_speed",))
            self.assertAlmostEqual(scores["relu"][101][6]["wind_speed"], 0.59, places=12)

    def test_one_readable_arm_among_several_isolated(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write(tmp, "relu_s101", self._card())
            os.makedirs(os.path.join(tmp, "leaky_s303"))
            scores = sbc.load_scorecard_mae(tmp)
            self.assertEqual(sorted(scores), ["relu"])


def _arm(floor, seed, scorecard=True, present_horizons=None, code_commit="cA"):
    """A synthetic arm record. `present_horizons` models a partial sweep."""
    present = set(sbc.HORIZONS if present_horizons is None else present_horizons)
    return {"arm": f"{floor}_s{seed}", "floor": floor, "seed": seed,
            "checkpoints": {h: h in present for h in sbc.HORIZONS},
            "n_checkpoints": len(present),
            "has_scorecard": scorecard, "scorecard": "x" if scorecard else None,
            "usable": scorecard, "complete": present == set(sbc.HORIZONS),
            "code_commits": [code_commit]}


def _fake_arms():
    return {
        "leaky": {101: _arm("leaky", 101), 202: _arm("leaky", 202)},
        "relu": {101: _arm("relu", 101), 202: _arm("relu", 202),
                 303: _arm("relu", 303, scorecard=False,
                           present_horizons=(1, 3, 6))},
    }


def _fake_scores():
    return {
        "leaky": {101: {1: {"wind_speed": 0.60, "temperature": 2.5}},
                  202: {1: {"wind_speed": 0.63, "temperature": 2.6}}},
        "relu": {101: {1: {"wind_speed": 0.60, "temperature": 2.5}},
                 202: {1: {"wind_speed": 0.63, "temperature": 2.6}},
                 303: {1: {"wind_speed": 0.72, "temperature": 2.9}}},
    }


class TestBuildReport(unittest.TestCase):
    def test_report_structure_and_tally(self):
        rep = sbc.build_report(_fake_arms(), _fake_scores(),
                               channels=("wind_speed", "temperature"), horizons=(1,))
        self.assertEqual(sorted(rep["per_arm"]), ["leaky", "relu"])
        self.assertEqual(rep["verdict_tally"].get(sbc.INSUFFICIENT, 0), 0)
        self.assertEqual(len(rep["comparisons"]), 2)  # one per (channel, horizon)
        self.assertIsNone(rep["confound"]["note"])
        self.assertEqual(rep["headline"]["comparisons_made"], 2)
        self.assertEqual(rep["headline"]["insufficient_evidence"], 0)
    def test_headline_is_not_distinguishable_when_nothing_clears(self):
        rep = sbc.build_report(_fake_arms(), _fake_scores(),
                               channels=("wind_speed",), horizons=(1,))
        self.assertEqual(rep["headline"]["distinguishable_from_seed_noise"], 0)
        self.assertEqual(rep["headline"]["of_which_material"], 0)
        self.assertEqual(rep["headline"]["verdict"], sbc.NOT_DISTINGUISHABLE)
        self.assertIn("NO floor-to-floor difference", rep["headline"]["answer"])

    def test_headline_is_distinguishable_when_something_clears(self):
        scores = {"leaky": {101: {1: {"wind_speed": 0.80}},
                            202: {1: {"wind_speed": 0.80}}},
                  "relu": {101: {1: {"wind_speed": 0.60}},
                           202: {1: {"wind_speed": 0.60}}}}
        rep = sbc.build_report(_fake_arms(), scores,
                               channels=("wind_speed",), horizons=(1,))
        self.assertEqual(rep["headline"]["distinguishable_from_seed_noise"], 1)
        self.assertEqual(rep["headline"]["of_which_material"], 1)
        self.assertEqual(rep["headline"]["verdict"], sbc.DISTINGUISHABLE)
        self.assertIn("larger than their seed noise", rep["headline"]["answer"])

    def test_tiny_but_resolvable_difference_is_not_material(self):
        """
        The real +1h pressure case: the two arms differ by 0.0001 MAE on an MAE
        of 0.37. The 2-sigma gate at n_pairs=2 clears that easily, and the answer
        is still "do not act on this".
        """
        scores = {"leaky": {101: {1: {"wind_speed": 0.3730}},
                            202: {1: {"wind_speed": 0.3730}}},
                  "relu": {101: {1: {"wind_speed": 0.3729}},
                           202: {1: {"wind_speed": 0.3729}}}}
        rep = sbc.build_report(_fake_arms(), scores,
                               channels=("wind_speed",), horizons=(1,))
        entry = rep["comparisons"][0]
        self.assertEqual(entry["comparison"]["verdict"], sbc.DISTINGUISHABLE)
        self.assertFalse(entry["is_material"])
        self.assertEqual(entry["actionable"], "RESOLVABLE_BUT_IMMATERIAL")
        self.assertEqual(rep["headline"]["verdict"], sbc.NOT_DISTINGUISHABLE)
        self.assertEqual(rep["headline"]["resolvable_but_immaterial"], 1)
        self.assertEqual(rep["headline"]["of_which_material"], 0)

    def test_materiality_floor_is_configurable(self):
        scores = {"leaky": {101: {1: {"wind_speed": 0.40}},
                            202: {1: {"wind_speed": 0.40}}},
                  "relu": {101: {1: {"wind_speed": 0.30}},
                           202: {1: {"wind_speed": 0.30}}}}
        loose = sbc.build_report(_fake_arms(), scores, channels=("wind_speed",),
                                 horizons=(1,), materiality_pct=1.0)
        tight = sbc.build_report(_fake_arms(), scores, channels=("wind_speed",),
                                 horizons=(1,), materiality_pct=50.0)
        self.assertEqual(loose["comparisons"][0]["actionable"],
                         "DISTINGUISHABLE_AND_MATERIAL")
        self.assertEqual(tight["comparisons"][0]["actionable"],
                         "RESOLVABLE_BUT_IMMATERIAL")
    def test_report_declares_that_the_gate_is_weak_at_low_n(self):
        rep = sbc.build_report(_fake_arms(), _fake_scores(),
                               channels=("wind_speed",), horizons=(1,))
        self.assertIn("necessary but not sufficient",
                      rep["policy"]["gate_is_weak_at_low_n"])
        self.assertEqual(rep["policy"]["materiality_pct"], 2.0)
        # every OK comparison at 2 pairs is flagged as below the sign-test floor
        for c in rep["comparisons"]:
            self.assertFalse(c["comparison"]["sign_test_can_reject"])
        self.assertEqual(rep["headline"]["below_exact_sign_test_floor"],
                         len(rep["comparisons"]))

    def test_caveat_present_below_the_sign_test_floor(self):
        r = sbc.paired_comparison({1: 0.60, 2: 0.63}, {1: 0.62, 2: 0.65})
        self.assertIn("caveat", r)
        self.assertIn("1 degree(s) of freedom", r["caveat"])
        wide = sbc.paired_comparison({i: 0.6 + 0.001 * i for i in range(7)},
                                    {i: 0.7 + 0.001 * i for i in range(7)})
        self.assertNotIn("caveat", wide)
        self.assertTrue(wide["sign_test_can_reject"])

    def test_relative_difference_pct_is_reported(self):
        r = sbc.paired_comparison({1: 0.60, 2: 0.60}, {1: 0.66, 2: 0.66})
        # mean diff -0.06 on a scale of 0.63
        self.assertAlmostEqual(r["relative_difference_pct"],
                               100.0 * (-0.06) / 0.63, places=11)

    def test_tested_means_are_separate_from_full_arm_means(self):
        """
        The real +12h temperature numbers. Arm A (leaky) has 2 seeds, arm B
        (relu) has 3, and B's third seed is a large outlier. The paired test sees
        only the 2 matched seeds, so its means and the arm means genuinely
        differ. Both are reported; conflating them is the bug this guards against.
        """
        a = {101: 1.9835, 202: 2.2777}          # leaky  mean 2.1306
        b = {101: 1.9979, 202: 2.2955, 303: 3.1145}   # relu mean 2.4693
        r = sbc.paired_comparison(a, b)
        self.assertEqual(r["n_pairs"], 2)
        self.assertAlmostEqual(r["mean_a_tested"], 2.1306, places=9)
        self.assertAlmostEqual(r["mean_b_tested"], 2.1467, places=9)
        self.assertAlmostEqual(r["mean_difference"], -0.0161, places=9)
        # the matched-seed effect is under 1% of MAE
        self.assertAlmostEqual(r["relative_difference_pct"], -0.75283, places=4)
        # arm means over ALL seeds put B at 2.4693, a 14.5% story instead
        self.assertAlmostEqual(sbc.mean(list(b.values())), 2.4693, places=9)

    def test_unbalanced_arms_are_flagged(self):
        r = sbc.compare_arms({1: 0.6, 2: 0.7}, {1: 0.6, 2: 0.7, 3: 9.0})
        self.assertTrue(r["arms_unbalanced"])
        self.assertEqual(r["n_seeds_a_total"], 2)
        self.assertEqual(r["n_seeds_b_total"], 3)
        self.assertEqual(sbc.compare_arms({1: 0.6, 2: 0.7},
                                          {1: 0.6, 2: 0.7})["arms_unbalanced"],
                         False)

    def test_effect_carried_by_an_unmatched_seed_is_flagged_and_not_actionable(self):
        """
        The headline finding, reproduced from the real +12h temperature data. A
        14.5% all-seed gap collapses to 0.75% once relu seed 303 -- which leaky
        does not have -- is excluded. Gating the decision on the all-seed number
        would promote a 14% "improvement" that the matched seeds do not support.
        """
        scores = {"leaky": {101: {1: {"wind_speed": 1.9835}},
                            202: {1: {"wind_speed": 2.2777}}},
                  "relu": {101: {1: {"wind_speed": 1.9979}},
                           202: {1: {"wind_speed": 2.2955}},
                           303: {1: {"wind_speed": 3.1145}}}}
        rep = sbc.build_report(_fake_arms(), scores,
                               channels=("wind_speed",), horizons=(1,))
        entry = rep["comparisons"][0]
        self.assertTrue(entry["arms_unbalanced"])
        self.assertTrue(entry["driven_by_unmatched_seed"])
        self.assertAlmostEqual(entry["relative_difference_pct_all_seeds"],
                               -14.5127, places=4)
        self.assertAlmostEqual(entry["relative_difference_pct"], -0.75, places=2)
        self.assertFalse(entry["is_material"])
        # it clears the 2-sigma gate only because 1 df makes the band collapse
        self.assertEqual(entry["comparison"]["verdict"], sbc.DISTINGUISHABLE)
        self.assertEqual(entry["actionable"], "RESOLVABLE_BUT_IMMATERIAL")
        self.assertEqual(rep["headline"]["effect_driven_by_unmatched_seed"], 1)
        self.assertEqual(rep["headline"]["verdict"], sbc.NOT_DISTINGUISHABLE)
        self.assertEqual(rep["headline"]["of_which_material"], 0)

    def test_flagged_as_insufficient_when_one_arm_is_short_a_seed(self):
        scores = _fake_scores()
        del scores["leaky"][202]
        rep = sbc.build_report(_fake_arms(), scores,
                               channels=("wind_speed",), horizons=(1,))
        cmp_res = rep["comparisons"][0]["comparison"]
        self.assertEqual(cmp_res["verdict"], sbc.INSUFFICIENT)
        self.assertEqual(cmp_res["n_common_seeds"], 1)
        self.assertEqual(cmp_res["n_pairs"], 1)
        self.assertEqual(cmp_res["n_seeds_a"], 1)
        self.assertEqual(cmp_res["n_seeds_b"], 3)
        self.assertEqual(rep["verdict_tally"][sbc.INSUFFICIENT], 1)

    def test_coverage_marks_partial_arms_unusable(self):
        rep = sbc.build_report(_fake_arms(), _fake_scores(),
                               channels=("wind_speed",), horizons=(1,))
        self.assertFalse(rep["coverage"]["relu"]["303"]["usable"])
        self.assertFalse(rep["coverage"]["relu"]["303"]["complete"])
        self.assertTrue(rep["coverage"]["relu"]["101"]["usable"])
        self.assertEqual(rep["coverage"]["relu"]["303"]["missing_horizons"], [12, 24])

    def test_confound_detected_across_different_code_commits(self):
        arms = _fake_arms()
        for rec in arms["leaky"].values():
            rec["code_commits"] = ["cB"]
        rep = sbc.build_report(arms, _fake_scores(),
                               channels=("wind_speed",), horizons=(1,))
        self.assertIsNotNone(rep["confound"]["note"])
        self.assertIn("not a clean single-variable ablation", rep["confound"]["note"])
        self.assertEqual(rep["confound"]["per_arm"][0]["floor"], "leaky")

    def test_no_confound_when_code_is_identical(self):
        rep = sbc.build_report(_fake_arms(), _fake_scores(),
                               channels=("wind_speed",), horizons=(1,))
        self.assertIsNone(rep["confound"]["note"])

    def test_seed_spread_is_computed_per_channel_horizon(self):
        rep = sbc.build_report(_fake_arms(), _fake_scores(),
                               channels=("wind_speed",), horizons=(1,))
        spread = rep["seed_spread"]["wind_speed_h1"]
        self.assertEqual(spread["n_arms_with_ge_2_seeds"], 2)
        # leaky [0.60, 0.63] -> sd = 0.03/sqrt(2); relu [0.60, 0.63, 0.72] -> mean 0.65
        leaky_sd = 0.03 / math.sqrt(2.0)
        relu_sd = math.sqrt(((0.60 - 0.65) ** 2 + (0.63 - 0.65) ** 2
                             + (0.72 - 0.65) ** 2) / 2)
        self.assertAlmostEqual(spread["min_seed_sd"], leaky_sd, places=12)
        self.assertAlmostEqual(spread["max_seed_sd"], relu_sd, places=12)
        self.assertAlmostEqual(spread["mean_seed_sd"], (leaky_sd + relu_sd) / 2,
                               places=12)
        # the widest seed spread, as a percentage of the arm mean
        self.assertEqual(spread["max_seed_range_pct"],
                         100.0 * (0.72 - 0.60) / 0.65)

    def test_single_seed_arm_does_not_contribute_to_seed_spread(self):
        rep = sbc.build_report(_fake_arms(), _fake_scores(),
                               channels=("wind_speed",), horizons=(1,))
        # leaky has 2 seeds so it does contribute; a 1-seed arm never would
        one = sbc.build_report(
            {"leaky": {101: _arm("leaky", 101)}, "relu": _fake_arms()["relu"]},
            {"leaky": {101: {1: {"wind_speed": 0.6}}},
             "relu": _fake_scores()["relu"]},
            channels=("wind_speed",), horizons=(1,))
        self.assertEqual(one["seed_spread"]["wind_speed_h1"]["n_arms_with_ge_2_seeds"], 1)

    def test_minimum_seeds_block_present(self):
        rep = sbc.build_report(_fake_arms(), _fake_scores(),
                               channels=("wind_speed",), horizons=(1,))
        block = rep["minimum_seeds"]["wind_speed_h1"]
        self.assertEqual(block["sign_test_floor_pairs"], 6)
        self.assertEqual(block["status"], "OK")
        self.assertIn("worst across arms", block["seed_sd_basis"])

    def test_minimum_seeds_absent_without_two_seeds(self):
        rep = sbc.build_report(
            {"relu": {101: _arm("relu", 101)}}, {"relu": {101: {1: {"wind_speed": 0.6}}}},
            channels=("wind_speed",), horizons=(1,))
        self.assertEqual(rep["minimum_seeds"], {})

    def test_report_is_json_serialisable(self):
        import json
        rep = sbc.build_report(_fake_arms(), _fake_scores(),
                               channels=("wind_speed", "temperature"), horizons=(1,))
        json.loads(json.dumps(rep))

    def test_report_declares_its_gate(self):
        rep = sbc.build_report(_fake_arms(), _fake_scores(),
                               channels=("wind_speed",), horizons=(1,))
        self.assertEqual(rep["policy"]["sigma"], 2.0)
        self.assertEqual(rep["policy"]["single_seed_rule"], "INSUFFICIENT_EVIDENCE")
        self.assertIn("best single seed is an optimistically biased estimate",
                      rep["policy"]["headline_rule"])

    def test_print_report_runs_and_reports_the_headline(self):
        import contextlib
        import io
        rep = sbc.build_report(_fake_arms(), _fake_scores(),
                               channels=("wind_speed",), horizons=(1,))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            sbc.print_report(rep)
        text = buf.getvalue()
        self.assertIn("CHECKPOINT SELECTION", text)
        self.assertIn("COVERAGE", text)
        self.assertIn("SEED SPREAD", text)
        self.assertIn("BEST SEED vs MEAN", text)
        self.assertIn("ARM vs ARM", text)
        self.assertIn("MINIMUM SEEDS", text)
        self.assertIn("NO SCORECARD", text)

    def test_fmt_handles_missing_and_degenerate_values(self):
        self.assertEqual(sbc._fmt(None), "-")
        self.assertEqual(sbc._fmt(float("nan")), "-")
        self.assertEqual(sbc._fmt(float("inf")), "+inf")
        self.assertEqual(sbc._fmt(float("-inf")), "-inf")
        self.assertEqual(sbc._fmt(0.5), "0.5000")
        self.assertEqual(sbc._fmt(0.5, ".2f"), "0.50")


class TestEvaluationSetGuard(unittest.TestCase):
    """A paired difference across different row sets is not a comparison."""

    def _write_card(self, tmp, tag, counts):
        d = os.path.join(tmp, tag)
        os.makedirs(d, exist_ok=True)
        card = {"horizon_evaluations": {
            f"horizon_{h}h": {"sample_count": n,
                             "candidate_featured_model": {
                                 "wind_speed": {"mae": 0.6 + 0.01 * h}}}
            for h, n in counts.items()}}
        with open(os.path.join(d, "predictive_quality_scorecard.json"), "w") as f:
            json.dump(card, f)

    def test_fingerprint_reads_sample_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_card(tmp, "relu_s101", {1: 3209, 6: 3132})
            fp = sbc.evaluation_set_fingerprint(tmp)
            self.assertEqual(fp["relu"][101][1], 3209)
            self.assertEqual(fp["relu"][101][6], 3132)

    def test_matching_row_sets_agree(self):
        a = {1: {1: 3209}, 2: {1: 3209}}
        b = {1: {1: 3209}, 2: {1: 3209}}
        self.assertIsNone(sbc.evaluation_sets_agree(a, b, (1, 2), (1, 2), (1,)))
        # one seed per side, same count
        self.assertIsNone(sbc.evaluation_sets_agree({1: {1: 3209}},
                                                   {1: {1: 3209}}, (1,), (1,), (1,)))

    def test_mismatched_row_sets_are_caught(self):
        a = {1: {1: 3209}}
        b = {1: {1: 2800}}
        msg = sbc.evaluation_sets_agree(a, b, (1,), (1,), (1,))
        self.assertIsNotNone(msg)
        self.assertIn("different numbers of rows", msg)
        self.assertIn("+1h", msg)
        self.assertIn("3209 rows", msg)
        self.assertIn("2800 rows", msg)

    def test_mismatch_only_flags_the_affected_horizon(self):
        a = {1: {1: 3209, 6: 3132}}
        b = {1: {1: 3209, 6: 2900}}
        self.assertIsNone(sbc.evaluation_sets_agree(a, b, (1,), (1,), (1,)))
        self.assertIsNotNone(sbc.evaluation_sets_agree(a, b, (1,), (1,), (1, 6)))

    def test_unknown_row_counts_are_not_treated_as_a_mismatch(self):
        """A scorecard that omits sample_count must not fabricate a conflict."""
        a = {1: {1: None}}
        b = {1: {1: 3209}}
        self.assertIsNone(sbc.evaluation_sets_agree(a, b, (1,), (1,), (1,)))

    def test_mismatch_forces_insufficient_evidence_in_the_report(self):
        arms = {"leaky": {101: _arm("leaky", 101), 202: _arm("leaky", 202)},
                "relu": {101: _arm("relu", 101), 202: _arm("relu", 202)}}
        scores = {"leaky": {101: {1: {"wind_speed": 0.70}, 6: {"wind_speed": 0.70}},
                            202: {1: {"wind_speed": 0.71}, 6: {"wind_speed": 0.71}}},
                  "relu": {101: {1: {"wind_speed": 0.60}, 6: {"wind_speed": 0.60}},
                           202: {1: {"wind_speed": 0.61}, 6: {"wind_speed": 0.61}}}}
        # counts agree at +1h and disagree at +6h
        fp = {"leaky": {101: {1: 3209, 6: 3132}, 202: {1: 3209, 6: 3132}},
              "relu": {101: {1: 3209, 6: 2900}, 202: {1: 3209, 6: 2900}}}
        rep = sbc.build_report(arms, scores, channels=("wind_speed",),
                               horizons=(1, 6), fingerprints=fp)
        by_h = {c["horizon_h"]: c for c in rep["comparisons"]}
        # same rows at +1h -> a real comparison is allowed
        self.assertEqual(by_h[1]["comparison"]["verdict"], sbc.DISTINGUISHABLE)
        self.assertNotIn("reason", by_h[1]["comparison"])
        # different rows at +6h -> refused, with the reason spelled out
        self.assertEqual(by_h[6]["comparison"]["verdict"], sbc.INSUFFICIENT)
        self.assertIn("different numbers of rows", by_h[6]["comparison"]["reason"])
        self.assertEqual(rep["headline"]["verdict"], sbc.DISTINGUISHABLE)
        self.assertEqual(rep["headline"]["insufficient_evidence"], 1)

    def test_fingerprint_is_empty_for_arms_without_scorecards(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "leaky_s303"))
            self.assertEqual(sbc.evaluation_set_fingerprint(tmp), {})


class TestSplitHalves(unittest.TestCase):
    @staticmethod
    def _arm_cell(overall, first, second):
        return {6: {"wind_speed": overall,
                    "first_half": {"wind_speed": first},
                    "second_half": {"wind_speed": second}}}

    def _recomputed(self):
        """leaky wins the first chronological half, relu wins the second."""
        return {
            "leaky": {101: self._arm_cell(0.60, 0.58, 0.62),
                      202: self._arm_cell(0.62, 0.60, 0.64)},
            "relu": {101: self._arm_cell(0.60, 0.61, 0.59),
                     202: self._arm_cell(0.62, 0.62, 0.62)},
        }

    def test_averages_across_all_given_seeds(self):
        out = sbc._split_halves(self._recomputed(), "leaky", (101, 202), 6,
                                ("wind_speed",))
        self.assertAlmostEqual(out["first_half"]["wind_speed"], 0.59, places=12)
        self.assertAlmostEqual(out["second_half"]["wind_speed"], 0.63, places=12)

    def test_single_seed_is_that_seed(self):
        out = sbc._split_halves(self._recomputed(), "leaky", (101,), 6,
                                ("wind_speed",))
        self.assertAlmostEqual(out["first_half"]["wind_speed"], 0.58, places=12)

    def test_missing_arm_or_horizon_yields_nothing(self):
        self.assertEqual(sbc._split_halves(self._recomputed(), "nope", (101,), 6,
                                           ("wind_speed",)), {})
        self.assertEqual(sbc._split_halves(self._recomputed(), "leaky", (101,), 99,
                                           ("wind_speed",)), {})

    def test_absent_channel_is_skipped(self):
        self.assertEqual(sbc._split_halves(self._recomputed(), "leaky", (1,), 6,
                                           ("humidity",)), {})

    def test_report_runs_the_half_split_check(self):
        arms = {"leaky": {101: _arm("leaky", 101), 202: _arm("leaky", 202)},
                "relu": {101: _arm("relu", 101), 202: _arm("relu", 202)}}
        scores = {"leaky": {101: {6: {"wind_speed": 0.60}},
                            202: {6: {"wind_speed": 0.62}}},
                  "relu": {101: {6: {"wind_speed": 0.60}},
                           202: {6: {"wind_speed": 0.62}}}}
        rep = sbc.build_report(arms, scores, channels=("wind_speed",),
                               horizons=(6,), recomputed=self._recomputed())
        entry = rep["comparisons"][0]
        self.assertEqual(entry["half_split"]["n_seeds_averaged"], 2)
        # leaky is better in the first half, relu in the second -> reversal
        self.assertEqual(entry["half_split"]["verdict"],
                         "INCONSISTENT_ACROSS_HALVES")
        self.assertEqual(rep["headline"]["half_split_inconsistent"], 1)
        self.assertEqual(rep["headline"]["half_split_covered"], 1)

    def test_half_split_absent_without_recomputed_data(self):
        rep = sbc.build_report(_fake_arms(), _fake_scores(),
                               channels=("wind_speed",), horizons=(1,))
        self.assertNotIn("half_split", rep["comparisons"][0])
        # and the coverage of the check is reported, so silence is not a pass
        self.assertEqual(rep["headline"]["half_split_covered"], 0)
        self.assertEqual(rep["headline"]["half_split_inconsistent"], 0)
        self.assertIn("must not be read as consistent",
                      rep["headline"]["half_split_coverage_note"])


class TestNormalizeRecomputed(unittest.TestCase):
    """JSON caches hand back string keys; every consumer looks up int seeds."""

    def test_string_keys_become_ints(self):
        raw = {"leaky": {"101": {"6": {"wind_speed": 0.6,
                                       "first_half": {"wind_speed": 0.58}}}}}
        out = sbc._normalize_recomputed(raw)
        self.assertIn(101, out["leaky"])
        self.assertIn(6, out["leaky"][101])
        self.assertIn("first_half", out["leaky"][101][6])

    def test_none_and_empty_pass_through(self):
        self.assertIsNone(sbc._normalize_recomputed(None))
        self.assertEqual(sbc._normalize_recomputed({}), {})

    def test_normalising_lets_the_half_split_actually_run(self):
        """
        The failure this guards: a string-keyed cache made every per-seed lookup
        miss, the half-split silently computed nothing, and the headline reported
        "0 comparisons reverse sign" as though the check had passed.
        """
        arms = {"leaky": {101: _arm("leaky", 101)},
                "relu": {101: _arm("relu", 101), 202: _arm("relu", 202)}}
        scores = {"leaky": {101: {6: {"wind_speed": 0.60}}},
                  "relu": {101: {6: {"wind_speed": 0.60}},
                           202: {6: {"wind_speed": 0.60}}}}
        string_keyed = {"leaky": {"101": {"6": {
            "wind_speed": 0.60,
            "first_half": {"wind_speed": 0.58},
            "second_half": {"wind_speed": 0.62}}}},
            "relu": {"101": {"6": {
                "wind_speed": 0.60,
                "first_half": {"wind_speed": 0.61},
                "second_half": {"wind_speed": 0.59}}}}}
        rep = sbc.build_report(arms, scores, channels=("wind_speed",),
                               horizons=(6,),
                               recomputed=sbc._normalize_recomputed(string_keyed))
        self.assertEqual(rep["headline"]["half_split_covered"], 1)
        entry = rep["comparisons"][0]
        self.assertIn("half_split", entry)
        self.assertEqual(entry["half_split"]["verdict"],
                         "INCONSISTENT_ACROSS_HALVES")


class TestCli(unittest.TestCase):
    """The command line must not need torch, and must honour --no-write."""

    def _sweep(self, tmp):
        for tag, ws in (("relu_s101", 0.60), ("relu_s202", 0.63),
                        ("leaky_s101", 0.60), ("leaky_s202", 0.63)):
            d = os.path.join(tmp, tag)
            os.makedirs(d, exist_ok=True)
            for h in sbc.HORIZONS:
                with open(os.path.join(d, f"candidate_h{h}h.pt"), "w") as f:
                    f.write("x")
            card = {"horizon_evaluations": {
                f"horizon_{h}h": {"sample_count": 3209 - h,
                                 "candidate_featured_model": {
                                     "wind_speed": {"mae": ws + 0.001 * h},
                                     "temperature": {"mae": 2.0 + ws + 0.001 * h}}}
                for h in sbc.HORIZONS}}
            with open(os.path.join(d, "predictive_quality_scorecard.json"), "w") as f:
                json.dump(card, f)

    def test_no_write_leaves_no_file(self):
        import contextlib
        import io
        with tempfile.TemporaryDirectory() as tmp:
            self._sweep(tmp)
            out = os.path.join(tmp, "should_not_exist.json")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = sbc.main(["--sweep-dir", tmp, "--out", out, "--no-write"])
            self.assertEqual(rc, 0)
            self.assertFalse(os.path.exists(out))
            self.assertIn("CHECKPOINT SELECTION", buf.getvalue())

    def test_writes_a_valid_report_by_default(self):
        import contextlib
        import io
        with tempfile.TemporaryDirectory() as tmp:
            self._sweep(tmp)
            out = os.path.join(tmp, "report.json")
            with contextlib.redirect_stdout(io.StringIO()):
                rc = sbc.main(["--sweep-dir", tmp, "--out", out])
            self.assertEqual(rc, 0)
            with open(out, encoding="utf-8") as f:
                rep = json.load(f)
            self.assertEqual(rep["tool"], "select_best_checkpoint")
            self.assertEqual(rep["headline"]["comparisons_made"], 10)
            # all four arms share one row count per horizon
            self.assertEqual(rep["headline"]["insufficient_evidence"], 0)

    def test_channel_and_horizon_filters(self):
        import contextlib
        import io
        with tempfile.TemporaryDirectory() as tmp:
            self._sweep(tmp)
            out = os.path.join(tmp, "report.json")
            with contextlib.redirect_stdout(io.StringIO()):
                sbc.main(["--sweep-dir", tmp, "--out", out,
                          "--channels", "wind_speed", "--horizons", "6,12"])
            with open(out, encoding="utf-8") as f:
                rep = json.load(f)
            self.assertEqual(rep["channels"], ["wind_speed"])
            self.assertEqual(rep["horizons"], [6, 12])
            # 1 arm pair x 1 channel x 2 horizons
            self.assertEqual(rep["headline"]["comparisons_made"], 2)
            self.assertEqual(sorted(rep["per_arm"]["relu"]), ["12", "6"])

    def test_missing_sweep_dir_exits_nonzero(self):
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = sbc.main(["--sweep-dir", os.path.join("no", "such", "dir"),
                           "--no-write"])
        self.assertEqual(rc, 1)
        self.assertIn("no arms found", buf.getvalue())

    def test_materiality_flag_is_wired_through(self):
        import contextlib
        import io
        with tempfile.TemporaryDirectory() as tmp:
            self._sweep(tmp)
            out = os.path.join(tmp, "report.json")
            with contextlib.redirect_stdout(io.StringIO()):
                sbc.main(["--sweep-dir", tmp, "--out", out, "--no-write",
                          "--materiality-pct", "0.5"])
            with contextlib.redirect_stdout(io.StringIO()):
                sbc.main(["--sweep-dir", tmp, "--out", out,
                          "--materiality-pct", "0.5"])
            with open(out, encoding="utf-8") as f:
                rep = json.load(f)
            self.assertEqual(rep["policy"]["materiality_pct"], 0.5)
            for c in rep["comparisons"]:
                self.assertEqual(c["materiality_threshold_pct"], 0.5)

    def test_recompute_cache_flag_without_the_flag_is_ignored(self):
        """A cache path on its own must not trigger inference or crash."""
        import contextlib
        import io
        with tempfile.TemporaryDirectory() as tmp:
            self._sweep(tmp)
            cache = os.path.join(tmp, "cache.json")
            with open(cache, "w", encoding="utf-8") as f:
                json.dump({}, f)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = sbc.main(["--sweep-dir", tmp, "--no-write"])
            self.assertEqual(rc, 0)
            self.assertNotIn("loaded from cache", buf.getvalue())

    def test_cache_actually_changes_the_report(self):
        """
        Regression: the merge of re-scored MAE into the per-arm scores used to sit
        inside the `elif args.allow_recompute` branch, so loading from cache
        silently produced a report built from scorecards alone -- a different
        answer from the same inputs, with no error to signal it.

        A cache that supplies a third seed for one arm must therefore change the
        seed count the comparison is built from.
        """
        import contextlib
        import io
        with tempfile.TemporaryDirectory() as tmp:
            self._sweep(tmp)
            cache = os.path.join(tmp, "cache.json")
            with open(cache, "w", encoding="utf-8") as f:
                json.dump({"leaky": {303: {h: {"wind_speed": 0.55 + 0.001 * h}
                                           for h in sbc.HORIZONS}}}, f)

            def comparison_with(cache_args):
                out = os.path.join(tmp, f"rep{len(cache_args)}.json")
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(sbc.main(
                        ["--sweep-dir", tmp, "--out", out, "--channels", "wind_speed"]
                        + cache_args), 0)
                with open(out, encoding="utf-8") as f:
                    return json.load(f)["comparisons"][0]

            without = comparison_with([])
            with_cache = comparison_with(["--recompute-cache", cache])
            self.assertEqual(without["n_seeds_a"], 2)
            self.assertEqual(with_cache["n_seeds_a"], 3)
            self.assertEqual(with_cache["summary_a"]["n_seeds"], 3)

    def test_merge_helper_adds_a_seed_from_the_cache(self):
        """The merge itself, tested directly rather than through the CLI."""
        scores = {"leaky": {101: {1: {"wind_speed": 0.60}}}}
        recomputed = {"leaky": {303: {1: {"wind_speed": 0.66,
                                          "first_half": {"wind_speed": 0.65},
                                          "second_half": {"wind_speed": 0.67}}}}}
        channels = ("wind_speed",)
        for floor, per_seed in recomputed.items():
            for seed, per_h in per_seed.items():
                for h, row in per_h.items():
                    scores.setdefault(floor, {}).setdefault(seed, {})[h] = {
                        ch: row[ch] for ch in channels if ch in row}
        self.assertIn(303, scores["leaky"])
        self.assertAlmostEqual(scores["leaky"][303][1]["wind_speed"], 0.66,
                               places=12)
        # the half-split keys must NOT leak into the per-channel score row
        self.assertNotIn("first_half", scores["leaky"][303][1])
        # the pre-existing scorecard row is untouched
        self.assertAlmostEqual(scores["leaky"][101][1]["wind_speed"], 0.60,
                               places=12)


if __name__ == "__main__":
    unittest.main()
