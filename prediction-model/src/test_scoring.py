"""
Contract tests for scoring.py, the single canonical scoring function.

WHAT THIS FILE IS FOR
---------------------
scoring.py is the one implementation of MAE/RMSE/bias for this project. Before
it existed there were thirteen functions returning a dict with an "mae" key
and sixty-four raw mean(|a-b|) call sites, which disagreed about two things
that matter: the SIGN of bias, and what to return when the data is thin.

Those two disagreements are what this file pins. It is deliberately written as
hand-computable arithmetic rather than frozen-from-a-run literals, because the
values it asserts are the ones a reader can check with a pencil, and because a
literal copied out of a previous run only proves the number did not change --
it does not prove the number was ever right.

Note the deliberate overlap with test_scoring_golden_fixtures.py. That file
freezes the OUTPUT of the scorer on an 8-window fixture. This file pins the
SEMANTICS with cases chosen so each rule has its own minimal witness:

    perfect forecast            -> mae/rmse/bias all exactly 0
    constant under-forecast     -> every metric from two rows
    one-sided error             -> bias sign, stated as a physical claim
    1-of-2 at coverage 0.5      -> the gate boundary, inclusive
    1-of-3 at coverage 0.333    -> the first case the gate refuses
    NaN / inf on either side    -> symmetric masking
    empty input                 -> None, not a raise, not a number
    single row                  -> every metric defined, CI honestly absent

CONVENTIONS STATED HERE SO THEY CANNOT DRIFT BACK
-------------------------------------------------
bias is PREDICTED MINUS OBSERVED (mean(pred - truth)), the WMO definition,
which is what the operational dashboard (monitoring.py) already used. So:

    POSITIVE bias -> the model runs WARM, it predicts above observation
    NEGATIVE bias -> the model runs COLD, it predicts below observation

This is the convention that was previously inverted in the benchmark, which
made one forecast read as running warm in one place and cold in another.
"""

import os
import sys

import numpy as np
import pytest

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from scoring import (  # noqa: E402
    BOOTSTRAP_CI,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    MIN_COVERAGE,
    SCORE_KEYS,
    bootstrap_ci,
    metrics,
)

NAN = float("nan")
INF = float("inf")


# ---------------------------------------------------------------------------
# The perfect forecast.
# ---------------------------------------------------------------------------
class TestPerfectForecast:
    def test_perfect_forecast_scores_zero_on_every_metric(self):
        """
        pred == truth, so every signed error is exactly 0. Then:
            mae  = mean(|0|)   = 0
            rmse = sqrt(mean(0^2)) = 0
            bias = mean(0)    = 0
        Asserted with `==` and not a tolerance: a perfect forecast has an
        exactly-zero error, and any implementation that produces 1e-17 here is
        doing something (a subtraction of unequal values, a sqrt of a denormal)
        that is worth knowing about.
        """
        m = metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert m["mae"] == 0.0
        assert m["rmse"] == 0.0
        assert m["bias"] == 0.0

    def test_perfect_forecast_still_reports_full_coverage_and_no_loss(self):
        # The coverage fields describe the data, not the quality of the score.
        # A perfect score over 3 of 3 rows is still a full-coverage score, and
        # the fact that it is perfect must not be confused with the fact that
        # it saw everything.
        m = metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert m["n"] == 3
        assert m["coverage"] == 1.0
        assert m["rows_dropped_nonfinite"] == 0

    def test_perfect_forecast_degenerate_ci_is_an_interval_of_zero_not_a_crack(self):
        # All errors are 0, so every bootstrap resample has mean 0 and the
        # percentile interval is [0, 0]. That is a real interval -- it is
        # reported, not suppressed -- and it is distinct from the [None, None]
        # returned when there are too few rows to bootstrap at all.
        lo, hi = metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])["ci95_mae"]
        assert lo == 0.0 and hi == 0.0


# ---------------------------------------------------------------------------
# Constant under-forecast: every metric from two rows of arithmetic.
# ---------------------------------------------------------------------------
class TestConstantUnderForecast:
    """
    pred = [0, 0] against truth = [2, 4].

        signed errors pred - truth = [-2, -4]
        mae  = (|-2| + |-4|) / 2 = (2 + 4) / 2 = 3.0
        mse  = (4 + 16) / 2      = 20 / 2     = 10.0
        rmse = sqrt(10)                          = 3.1622776601683795
        bias  = (-2 + -4) / 2                   = -3.0

    Every one of these is exactly hand-checkable, which is the point: the RMSE
    is sqrt(10) and NOT 2.0, and the bias is -3.0 and NOT +3.0. Both of those
    wrong-but-plausible values have been produced by this codebase before.
    """

    PRED = [0.0, 0.0]
    TRUTH = [2.0, 4.0]

    def test_mae_is_mean_absolute_error(self):
        assert metrics(self.PRED, self.TRUTH)["mae"] == 3.0

    def test_rmse_is_sqrt_of_the_mean_SQUARED_error(self):
        """
        sqrt(10) = 3.1622776601683795, not 2.0.

        sqrt(4) = 2.0 is what you get by treating the errors as [2, 2] instead
        of [2, 4], i.e. by averaging the absolute error and then squaring, or
        by squaring the mean. mean(e^2) = (4 + 16)/2 = 10, not 4.
        """
        m = metrics(self.PRED, self.TRUTH)
        assert m["rmse"] == 3.1622776601683795
        assert m["rmse"] != 2.0

    def test_under_forecast_has_negative_bias(self):
        """
        The sign convention, as a physical claim: a model that predicts 0
        against observations of 2 and 4 is running COLD, so its bias is
        NEGATIVE under mean(pred - truth).

        The pre-correction convention mean(truth - pred) published this same
        forecast as +3.0, i.e. as running warm. Asserted as a signed literal
        so a return to the inverted convention cannot pass.
        """
        m = metrics(self.PRED, self.TRUTH)
        assert m["bias"] == -3.0
        assert m["bias"] < 0.0

    def test_over_forecast_is_the_mirror_image(self):
        # Symmetry check on the same two rows with the sides swapped: the
        # magnitudes are identical and only the bias sign moves. A scorer that
        # returned the same bias for both would be returning |error|.
        m = metrics(self.TRUTH, self.PRED)
        assert m["mae"] == 3.0
        assert m["rmse"] == 3.1622776601683795
        assert m["bias"] == 3.0

    def test_counting_fields_for_the_two_row_case(self):
        m = metrics(self.PRED, self.TRUTH)
        assert m["n"] == 2
        assert m["coverage"] == 1.0
        assert m["rows_dropped_nonfinite"] == 0


# ---------------------------------------------------------------------------
# The bias sign convention, stated explicitly and separately.
# ---------------------------------------------------------------------------
class TestBiasSignConvention:
    """
    bias = mean(pred - truth), the WMO definition.

    Stated once more because it is the number triage reads most often and the
    one that was previously inverted: POSITIVE bias means the model runs WARM,
    NEGATIVE means it runs COLD. The dashboard (monitoring.py) has always used
    this convention; the benchmark did not, so the same forecast read as
    over-predicting in one and under-predicting in the other.
    """

    TRUTH = [0.0, 0.0, 0.0, 0.0]

    def test_a_model_predicting_one_above_observation_reports_positive_bias(self):
        # "Predicting 1 degree above observation on every window is running
        # warm." If the sign convention is ever argued about again, this is
        # the sentence to point at.
        assert metrics([1.0, 1.0, 1.0, 1.0], self.TRUTH)["bias"] == 1.0

    def test_a_model_predicting_one_below_observation_reports_negative_bias(self):
        # The mirror of the case above, and the one that was previously
        # published with the wrong sign: under mean(truth - pred) this same
        # forecast reads +1.0, i.e. as running warm when it is running cold.
        assert metrics([-1.0, -1.0, -1.0, -1.0], self.TRUTH)["bias"] == -1.0

    def test_bias_is_the_mean_of_signed_errors_not_of_absolute_errors(self):
        """
        pred - truth = [+2, -2] on two rows.

            mae  = (2 + 2) / 2 = 2.0
            bias = (2 + -2) / 2 = 0.0

        A scorer that computed bias from the absolute errors would report
        +2.0 here. This is the minimal witness for sign cancellation: a bias
        of zero with a non-zero MAE is only possible if the signs were kept
        long enough to cancel.
        """
        m = metrics([2.0, 0.0], [0.0, 2.0])
        assert m["mae"] == 2.0
        assert m["bias"] == 0.0

    def test_mae_and_bias_are_independent_quantities(self):
        # Guards the specific refactor that replaces the signed mean with the
        # absolute mean. With errors [+2, -2] the two differ; if a change ever
        # makes them identical, the sign information has been lost even though
        # this particular case would still "look right".
        m = metrics([2.0, 0.0], [0.0, 2.0])
        assert m["mae"] != m["bias"]
        assert m["mae"] == 2.0 and m["bias"] == 0.0

    def test_bias_is_computed_over_surviving_rows_only(self):
        """
        Three rows, the middle one non-finite on both sides.

            surviving errors pred - truth = [+1, -1]
            mae  = (1 + 1) / 2 = 1.0
            bias = (1 + -1) / 2 = 0.0

        Both sides of the input are masked by the same rule, so the row where
        BOTH sides were non-finite is counted once, not twice. A scorer that
        counted dropped rows per side would report 2 here against an input of
        3 and its coverage would not be a fraction of the input at all.
        """
        m = metrics([1.0, NAN, -1.0], [0.0, NAN, 0.0])
        assert m["n"] == 2
        assert m["rows_dropped_nonfinite"] == 1
        assert m["coverage"] == pytest.approx(2.0 / 3.0)
        assert m["mae"] == 1.0
        assert m["bias"] == 0.0


# ---------------------------------------------------------------------------
# The coverage gate. The boundary is INCLUSIVE.
# ---------------------------------------------------------------------------
class TestCoverageGate:
    def test_min_coverage_default_is_half(self):
        """
        The threshold is part of the contract, not an implementation detail.
        Half the rows is the line: a score over more of the input is a score of
        the model, a score over less is a score of whichever rows happened to
        survive, and the run that motivated this gate scored over ZERO rows
        while still printing a rank for all seven models.
        """
        assert MIN_COVERAGE == 0.5

    def test_gate_boundary_is_inclusive_at_exactly_min_coverage(self):
        """
        1 of 2 rows finite -> coverage exactly 0.5 -> SCORED, not refused.

        The comparison is `coverage < min_coverage`, so exactly at the
        threshold survives. This is the kind of boundary that gets flipped to
        `<=` by accident during a refactor, and the flip is invisible except
        as a run that stopped producing a table for one series.

        The surviving row is a PERFECT match, so the score that is published
        is 0.0 -- the most flattering number available. Publishing it is
        correct here only because the coverage was exactly at the documented
        threshold; the same score computed over 1 of 3 rows is refused below.
        """
        m = metrics([1.0, NAN], [1.0, 2.0])
        assert m is not None, (
            "coverage exactly AT min_coverage must still score; the gate is "
            "inclusive, not exclusive")
        assert m["coverage"] == 0.5
        assert m["n"] == 1
        assert m["rows_dropped_nonfinite"] == 1
        assert m["mae"] == 0.0

    def test_one_below_the_threshold_is_refused(self):
        """
        1 of 3 rows finite -> coverage 0.3333... -> refused, returns None.

        The surviving row is again a perfect match, so the number the gate
        declines to publish is 0.0. Refusing to report it is the entire point:
        a model that scored 0.0 because two of three hours were missing is not
        a model that scored 0.0.
        """
        assert metrics([1.0, NAN, NAN], [1.0, 2.0, 3.0]) is None

    def test_refusal_returns_none_not_nan_not_inf_not_a_raise(self):
        # The return type distinguishes the refusal from every other answer
        # this function could give. A caller that does `if m is None` gets
        # exactly the refusal; one that does `if not m["mae"]` cannot.
        m = metrics([NAN, NAN, NAN], [1.0, 2.0, 3.0])
        assert m is None

    def test_the_same_good_rows_stop_being_scored_when_padded_with_junk(self):
        """
        The interaction the gate creates, pinned so it is a decision and not an
        accident.

        Three rows, all finite and all exact, score. The SAME three rows
        padded out to 7 with 4 non-finite are 3/7 = 0.43 coverage and are
        refused. The quality of the surviving forecast is unchanged; only its
        relationship to the input changed, and that relationship is what the
        gate is about.
        """
        good = [1.0, 2.0, 3.0]
        assert metrics(good, good)["coverage"] == 1.0

        padded = good + [NAN] * 4
        assert metrics(padded, padded) is None

    def test_min_coverage_is_overridable_per_call(self):
        """
        The threshold is a parameter, not a constant baked into the body, so a
        caller scoring a legitimately patchy series can demand a lower bar
        without editing the shared function -- and, more importantly, can
        still SEE what bar it got.

            pred  [1.0, nan, 4.0, 5.0]   kept 3 of 4 -> coverage 0.75
            truth [1.0, 2.0, 3.0, 3.0]

        0.75 is below 0.8 (refused) and above 0.5 (scored). The reported
        coverage does not move with the threshold: it describes the data, not
        the policy.
        """
        p = [1.0, NAN, 4.0, 5.0]
        t = [1.0, 2.0, 3.0, 3.0]

        assert metrics(p, t, min_coverage=0.8) is None

        relaxed = metrics(p, t, min_coverage=0.5)
        assert relaxed is not None
        assert relaxed["coverage"] == 0.75
        assert relaxed["rows_dropped_nonfinite"] == 1

        # A threshold of zero disables the gate but not the masking: the score
        # is still computed over the survivors, and coverage still reports the
        # truth about the input.
        no_gate = metrics(p, t, min_coverage=0.0)
        assert no_gate["coverage"] == 0.75
        assert no_gate["n"] == 3

    def test_min_coverage_is_a_keyword_or_positional_third_argument(self):
        # Callers should not have to remember whether the threshold is
        # positional or keyword-only; both work, so a positional call in
        # existing code keeps working.
        assert metrics([1.0, NAN], [1.0, 2.0], 0.5) is not None
        assert metrics([1.0, NAN], [1.0, 2.0], 0.51) is None


# ---------------------------------------------------------------------------
# The coverage identity: n, rows_dropped_nonfinite and coverage describe the
# same rows. This is what makes the coverage fields falsifiable.
# ---------------------------------------------------------------------------
class TestCoverageAccountingIdentity:
    """
    coverage == n / (n + rows_dropped_nonfinite), and
    n + rows_dropped_nonfinite == the number of input rows.

    Every finite-only fixture has coverage 1.0 and dropped 0, so a scorer that
    HARDCODED either of those would pass every other test in this file. These
    cases are what make those two fields falsifiable, and the identity below
    is the strongest form of the check because it survives any ONE of the
    three fields lying: a hardcoded coverage, a hardcoded dropped count, or an
    n computed from the unmasked length all break it.
    """

    def test_full_coverage_breaks_no_part_of_the_identity(self):
        m = metrics([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])
        n, dropped, cov = m["n"], m["rows_dropped_nonfinite"], m["coverage"]
        assert n == 4 and dropped == 0
        assert n + dropped == 4
        assert cov == pytest.approx(n / (n + dropped)) == 1.0

    @pytest.mark.parametrize("drop_pred,drop_truth", [
        ([True, False, False, False], [False, True, False, False]),
        ([True, True, False, False], [False, False, False, False]),
        ([False, False, False, False], [True, True, False, False]),
        ([True, False, True, False], [True, False, False, True]),
    ])
    def test_identity_holds_for_every_masking_pattern(self, drop_pred, drop_truth):
        # Every row dropped is dropped if EITHER side is non-finite, so the
        # union of the two patterns is what must be counted -- not either one
        # alone. A scorer that only masked on the prediction side, or only on
        # the truth side, breaks the identity on three of these four rows.
        base = [1.0, 2.0, 3.0, 4.0]
        pred = [NAN if d else v for d, v in zip(drop_pred, base)]
        truth = [NAN if d else v for d, v in zip(drop_truth, base)]

        expected_dropped = sum(1 for dp, dt in zip(drop_pred, drop_truth) if dp or dt)
        m = metrics(pred, truth)
        if m is None:
            # Legal only when the SURVIVING fraction falls below the gate;
            # then the accounting cannot be read off the return value, so
            # assert that the refusal was the gate and not something else.
            assert (4 - expected_dropped) / 4 < MIN_COVERAGE
            return

        n, dropped, cov = m["n"], m["rows_dropped_nonfinite"], m["coverage"]
        assert n == 4 - expected_dropped
        assert dropped == expected_dropped
        assert n + dropped == 4
        assert cov == pytest.approx(n / (n + dropped))
        assert cov == pytest.approx((4 - expected_dropped) / 4)

    def test_dropped_rows_are_excluded_from_the_denominator(self):
        """
        The mutation a NaN test normally misses.

            pred  (6 rows) 28.5   nan  30.0   nan  29.0  28.0
            truth (6 rows) 28.0  29.0  29.0  30.0  29.5  28.5
            surviving rows 0, 2, 4, 5 -> errors pred - truth = [+0.5, +1.0, -0.5, -0.5]

            MAE  = (0.5 + 1.0 + 0.5 + 0.5) / 4 = 2.5/4 = 0.625
            MSE  = (0.25 + 1.0 + 0.25 + 0.25) / 4 = 1.75/4 = 0.4375
            RMSE = sqrt(0.4375) = 0.6614378277661477
            bias = mean(pred - truth) = 0.5/4 = +0.125

        Dividing by 6 instead of 4 would give MAE 0.4166... and bias 0.0833...
        -- both flattering, both wrong, and both a REWARD for missing data: a
        model that lost half its rows would score better than one that kept
        them. That is the failure this asserts against.
        """
        pred = [28.5, NAN, 30.0, NAN, 29.0, 28.0]
        truth = [28.0, 29.0, 29.0, 30.0, 29.5, 28.5]
        m = metrics(pred, truth)

        assert m["n"] == 4
        assert m["mae"] == 0.625
        assert m["rmse"] == pytest.approx(0.6614378277661477, rel=0.0, abs=1e-12)
        assert m["bias"] == 0.125
        assert m["coverage"] == pytest.approx(4.0 / 6.0)
        assert m["rows_dropped_nonfinite"] == 2

        # And explicitly NOT the unmasked-length versions.
        assert m["mae"] != pytest.approx(2.5 / 6.0)
        assert m["bias"] != pytest.approx(0.5 / 6.0)


# ---------------------------------------------------------------------------
# NaN and infinity.
# ---------------------------------------------------------------------------
class TestNonFiniteHandling:
    def test_nan_in_prediction_is_dropped(self):
        # Three rows, one poisoned in pred. kept = 2 of 3, coverage 2/3 which
        # clears the gate, and the two survivors are exact.
        m = metrics([1.0, NAN, 3.0], [1.0, 2.0, 3.0])
        assert m is not None
        assert m["n"] == 2
        assert m["coverage"] == pytest.approx(2.0 / 3.0)
        assert m["rows_dropped_nonfinite"] == 1

    def test_nan_in_truth_is_dropped_too(self):
        # The mask is symmetric. A gap in the OBSERVATION must not be scored
        # as a model error, and a gap in the model must not be scored as an
        # observation error. Both are dropped, both are counted.
        m = metrics([1.0, 2.0, 3.0], [1.0, NAN, 3.0])
        assert m["n"] == 2
        assert m["rows_dropped_nonfinite"] == 1
        assert m["coverage"] == pytest.approx(2.0 / 3.0)

    @pytest.mark.parametrize("bad", [INF, -INF, NAN])
    def test_infinity_and_negative_infinity_drop_like_nan(self, bad):
        # np.isfinite rejects +-inf as well as nan. A saturated sensor reading
        # is dropped rather than dominating the mean, which is right, and it
        # counts against coverage rather than passing unnoticed.
        #
        # 1 of 2 surviving is coverage exactly 0.5 = MIN_COVERAGE, and the
        # gate is inclusive, so this is scored rather than refused.
        m = metrics([1.0, bad], [1.0, 2.0])
        assert m is not None
        assert m["n"] == 1
        assert m["coverage"] == 0.5
        assert m["rows_dropped_nonfinite"] == 1
        assert m["mae"] == 0.0

    def test_a_single_infinite_survivor_poisons_nothing(self):
        # The whole reason for masking. Without it, one +inf in the error
        # array makes mae = inf and rmse = inf and bias = nan, i.e. a single
        # bad sensor reading destroys the entire score for the run rather than
        # removing one row from it.
        #
        # 3 of 4 survive, coverage 0.75, errors on the survivors are
        # [0, 1, 2] so mae = 1.0 and bias = +1.0.
        m = metrics([1.0, NAN, 4.0, 5.0], [1.0, 2.0, 3.0, 3.0])
        assert m["n"] == 3
        assert m["mae"] == 1.0
        assert m["bias"] == 1.0
        assert np.isfinite(m["rmse"])
        assert m["rmse"] == pytest.approx(1.2909944487358056, rel=0.0, abs=1e-12)

    def test_all_rows_non_finite_returns_none(self):
        # Total loss of coverage is the documented NWP-cache failure: every
        # lookup missed. It yields None, not nan, so it cannot be mistaken for
        # a model that scored badly.
        assert metrics([NAN, NAN], [1.0, 2.0]) is None
        assert metrics([1.0, 2.0], [NAN, NAN]) is None
        assert metrics([INF, -INF], [1.0, 2.0]) is None

    def test_a_row_is_dropped_if_EITHER_side_is_non_finite(self):
        # 2 of 4 survive. Row 0 is poisoned in truth only, row 1 in pred only;
        # both count, and neither is scored.
        m = metrics([1.0, NAN, 3.0, 4.0], [NAN, 2.0, 3.0, 4.0])
        assert m["n"] == 2
        assert m["rows_dropped_nonfinite"] == 2
        assert m["coverage"] == 0.5


# ---------------------------------------------------------------------------
# Empty and single-row inputs.
# ---------------------------------------------------------------------------
class TestDegenerateInputs:
    def test_empty_input_returns_none(self):
        # Not a raise, not nan, not a division by zero: None. The documented
        # contract is that a caller can always write `if m is None` and be
        # right, and an empty horizon is a legitimate thing to ask about.
        assert metrics([], []) is None

    def test_mismatched_input_lengths_raise_rather_than_truncating(self):
        """
        KNOWN LIMITATION, pinned as-is rather than fixed.

        metrics() builds a single pairwise mask, so pred and truth must be the
        same length. Differing lengths raise numpy's broadcast ValueError
        instead of returning None, because a length mismatch between a
        prediction series and its truth series is a programming error at the
        call site, not a data-quality condition. Silently truncating to the
        shorter one -- or padding to the longer -- would let a misaligned
        window set produce a confident score computed over rows that were never
        actually paired, which is the exact class of confidently-wrong number
        this module exists to prevent.

        It is preserved from the reference implementation unchanged. The
        module docstring's "None" guarantee is therefore about missing DATA,
        not about malformed INPUT.
        """
        with pytest.raises(ValueError):
            metrics([], [1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            metrics([1.0, 2.0, 3.0], [])

    def test_both_empty_is_the_one_length_mismatch_that_is_not_an_error(self):
        # Two empty arrays are the same length, so this is the empty-input case
        # and it returns None rather than raising.
        assert metrics([], []) is None

    def test_single_row_input_is_fully_defined(self):
        """
        Every metric is defined for n == 1; only the CI is absent.

            pred [2.0], truth [1.0] -> error +1.0
            mae  = 1.0   rmse = 1.0   bias = +1.0   n = 1
            coverage 1.0, dropped 0
        """
        m = metrics([2.0], [1.0])
        assert m["mae"] == 1.0
        assert m["rmse"] == 1.0
        assert m["bias"] == 1.0
        assert m["n"] == 1
        assert m["coverage"] == 1.0
        assert m["rows_dropped_nonfinite"] == 0

    def test_single_row_reports_no_confidence_interval(self):
        """
        ci95_mae is [None, None] for n == 1, not [x, x].

        A percentile bootstrap over one observation has no spread. Reporting
        the point estimate as both endpoints would present a degenerate
        interval as a real one, and a downstream reader would see a confident
        bound on a single data point.
        """
        m = metrics([2.0], [1.0])
        assert m["ci95_mae"] == [None, None]

    def test_single_row_at_the_gate_boundary(self):
        # 1 of 2 rows surviving is exactly MIN_COVERAGE and is scored; the CI
        # is still absent, so the boundary and the degenerate-CI rules compose
        # rather than conflict.
        m = metrics([1.0, NAN], [1.0, 2.0])
        assert m is not None
        assert m["n"] == 1
        assert m["coverage"] == 0.5
        assert m["ci95_mae"] == [None, None]

    def test_single_non_finite_row_is_nothing_at_all(self):
        # One row, and it is dropped: kept == 0, which returns None before the
        # coverage gate is even reached.
        assert metrics([NAN], [1.0]) is None

    def test_inputs_are_not_modified(self):
        # A scorer that sorted or filled its input in place would corrupt the
        # caller's arrays between horizons. Cheap to check, and the kind of bug
        # that only shows up as "the second horizon is wrong".
        pred = [3.0, 1.0, NAN, 2.0]
        truth = [1.0, 1.0, 9.0, 2.0]
        p_copy, t_copy = list(pred), list(truth)
        metrics(pred, truth)
        assert pred == p_copy
        assert truth == t_copy


# ---------------------------------------------------------------------------
# The shape of the return value.
# ---------------------------------------------------------------------------
class TestReturnContract:
    def test_key_set_is_exact(self):
        """
        Exactly these seven keys, no more and no fewer.

        Consumers index these by name, so a missing key is a crash at the call
        site and an extra key is a field nobody knows the provenance of. The
        set is published as SCORE_KEYS so a caller can check it rather than
        guess.
        """
        m = metrics([1.0, 2.0], [1.0, 3.0])
        assert set(m) == set(SCORE_KEYS)
        assert set(m) == {"mae", "rmse", "bias", "n", "coverage",
                          "rows_dropped_nonfinite", "ci95_mae"}

    def test_values_are_plain_python_floats_and_ints(self):
        # JSON-serialisable without a default=float handler, and comparable
        # with == rather than only approximately. numpy scalars leak into
        # results files otherwise and break round-tripping.
        m = metrics([1.0, 2.0], [1.0, 3.0])
        assert type(m["mae"]) is float
        assert type(m["rmse"]) is float
        assert type(m["bias"]) is float
        assert type(m["n"]) is int
        assert type(m["coverage"]) is float
        assert type(m["rows_dropped_nonfinite"]) is int
        assert all(type(v) is float for v in m["ci95_mae"])

    def test_accepts_lists_tuples_and_arrays_alike(self):
        # Callers pass whatever they have; a scorer that only accepts ndarrays
        # forces a conversion at every call site and is one refactor away from
        # only accepting ndarrays.
        expected = metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])["mae"]
        assert metrics((1.0, 2.0, 3.0), (1.0, 2.0, 3.0))["mae"] == expected
        assert metrics(np.array([1.0, 2.0, 3.0]),
                       np.array([1.0, 2.0, 3.0]))["mae"] == expected
        assert metrics([np.float64(1.0), np.float64(2.0)],
                       [1, 2])["mae"] == 0.0

    def test_integers_are_accepted_without_caller_side_conversion(self):
        # A scorer that rejected ints, or that truncated them, would be a
        # nuisance on the first call and a wrong number on the second.
        assert metrics([1, 2, 3], [1, 2, 3])["mae"] == 0.0
        assert metrics([1, 2, 3], [2, 3, 4])["mae"] == 1.0

    def test_rmse_is_never_below_mae(self):
        # sqrt(mean(e^2)) >= mean(|e|) for any e, with equality only when all
        # |e| are equal. A change that loses the square, or that computes RMSE
        # on a different mask than MAE, breaks this immediately.
        rng = np.random.default_rng(0)
        for _ in range(25):
            t = rng.normal(30.0, 5.0, 200)
            p = t + rng.normal(0.0, 1.5, 200)
            m = metrics(p, t)
            assert m["rmse"] >= m["mae"] - 1e-12

    def test_rmse_equals_mae_when_every_error_is_identical(self):
        # The equality case of the inequality above, which is what pins the
        # direction of it: a constant error has no spread between the two
        # metrics, so rmse == mae and any "improvement" is arithmetic drift.
        m = metrics([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        assert m["mae"] == 1.0
        assert m["rmse"] == 1.0

    def test_a_constant_offset_shifts_the_bias_by_exactly_that_constant(self):
        """
        pred = 0 against truth [1, 2, 3] gives errors [-1, -2, -3], bias -2.0.
        pred = 5 gives errors [+4, +3, +2], bias +3.0.

        A constant offset of 5 moves the bias by exactly 5, which is the
        signed quantity behaving like a signed quantity. (mae and rmse do NOT
        shift by 5 -- they are magnitudes and are not translation invariant --
        which is precisely why bias cannot be derived from them.)
        """
        base = metrics([0.0, 0.0, 0.0], [1.0, 2.0, 3.0])
        shifted = metrics([5.0, 5.0, 5.0], [1.0, 2.0, 3.0])
        assert base["bias"] == -2.0
        assert shifted["bias"] == 3.0
        assert shifted["bias"] - base["bias"] == pytest.approx(5.0)

    def test_metrics_are_invariant_to_the_order_of_the_rows(self):
        # Permuting the rows permutes the errors and changes no mean, so all
        # four of the aggregate metrics must be identical. A scorer that
        # sorted one input but not the other would pair the wrong rows and get
        # a clean-looking answer from mismatched data -- the silent failure
        # this checks for.
        p = [1.0, 5.0, 2.0, 9.0, 3.0, 7.0]
        t = [2.0, 4.0, 1.0, 9.0, 5.0, 6.0]
        order = [3, 0, 5, 1, 4, 2]
        a = metrics(p, t)
        b = metrics([p[i] for i in order], [t[i] for i in order])
        for k in ("mae", "rmse", "bias", "n", "coverage", "rows_dropped_nonfinite"):
            assert a[k] == b[k]


# ---------------------------------------------------------------------------
# The bootstrap.
# ---------------------------------------------------------------------------
class TestBootstrapCi:
    def test_bootstrap_settings_are_frozen(self):
        # 400 resamples at 95% with seed 42. These pin every published
        # confidence interval, so a silent change to any of them moves numbers
        # that are already written down.
        assert BOOTSTRAP_RESAMPLES == 400
        assert BOOTSTRAP_CI == 0.95
        assert BOOTSTRAP_SEED == 42

    def test_bootstrap_is_deterministic_for_a_fixed_seed(self):
        # Same input, same interval, every time. Without a fixed seed the
        # published CIs would move on every rerun and a diff between two runs
        # of the same benchmark would be unreadable.
        a = metrics([1.0, 2.0, 3.0, 4.0, 5.0], [1.5, 2.5, 2.5, 4.5, 5.5])
        b = metrics([1.0, 2.0, 3.0, 4.0, 5.0], [1.5, 2.5, 2.5, 4.5, 5.5])
        assert a["ci95_mae"] == b["ci95_mae"]

    def test_ci_brackets_the_mae(self):
        # A 95% percentile interval of the resampled MEAN must contain the
        # observed mean in essentially every case; when it does not, the CI is
        # not an interval of the quantity being reported.
        rng = np.random.default_rng(7)
        for _ in range(20):
            t = rng.normal(25.0, 3.0, 60)
            p = t + rng.normal(0.0, 1.0, 60)
            m = metrics(p, t)
            lo, hi = m["ci95_mae"]
            assert lo <= m["mae"] <= hi

    def test_ci_is_ordered_and_has_the_right_width(self):
        # Lower endpoint below upper endpoint, and the interval is finite and
        # non-degenerate for a sample of this size -- not [None, None], not
        # [x, x] with a nonzero x, not reversed.
        m = metrics([1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    [1.5, 2.0, 3.5, 4.0, 5.5, 6.0])
        lo, hi = m["ci95_mae"]
        assert lo < hi
        assert lo >= 0.0
        assert all(np.isfinite(v) for v in (lo, hi))

    def test_ci_tightens_as_the_sample_grows(self):
        # More data, less uncertainty. A CI that does not narrow with n means
        # the resampling is not estimating the spread of the mean.
        rng = np.random.default_rng(3)
        widths = []
        for n in (10, 100, 2000):
            t = rng.normal(0.0, 1.0, n)
            p = t + rng.normal(0.0, 1.0, n)
            lo, hi = metrics(p, t)["ci95_mae"]
            widths.append(hi - lo)
        assert widths[0] > widths[1] > widths[2]

    def test_bootstrap_ci_helper_agrees_with_the_scorer(self):
        # The helper is public, so a caller can compute a CI for something the
        # scorer does not return. It must be the SAME function, or two
        # "bootstrap 95% CIs" with different settings end up in one report.
        err = np.abs(np.array([0.0, 2.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0]))
        assert bootstrap_ci(err) == metrics([1.0, 3.0, 2.0, 1.0, 2.0, 3.0, 1.0, 2.0],
                                            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])["ci95_mae"]

    def test_bootstrap_ci_returns_none_pair_for_zero_or_one_row(self):
        assert bootstrap_ci(np.array([])) == [None, None]
        assert bootstrap_ci(np.array([1.0])) == [None, None]
