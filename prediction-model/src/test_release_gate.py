"""
Tests for the promotion gate: a gate that CANNOT silently pass.

WHAT THESE TESTS ARE FOR
------------------------
A promotion gate has one dangerous failure mode: returning a verdict it did not
earn. Not crashing -- crashing is loud. Returning "PROMOTE" because a number
was absent, or because a list was iterated over whatever happened to be in it,
or because the decision quietly ran on the test split.

The incident that motivates this file had exactly that shape. In
``train_predictive_quality.py`` the Workstream G promotion audit built
``_target_routes`` by looping over seven 3-field tuples while unpacking four
names, so every run raised ``ValueError: not enough values to unpack`` at the
LAST stage -- after all five checkpoints had been written. Each run exited rc=1
and left a complete, valid, plausible artifact set on disk. Nobody could tell
from the filesystem that the promotion verdict had never been computed.
``test_benchmark_harness.py`` guards that specific arity. This file guards the
class: every route by which absence, omission, or a mislabelled split could turn
into a pass.

The tests are grouped by the property they defend, and the two headline ones are
in ``TestAbsentEvidenceIsNeverAPass`` and ``TestChannelIndependence`` -- a
challenger that is better at wind and worse at rain must SPLIT, because
rejecting it wholesale throws the wind gain away and keeps the rain regression.
"""

import json
import math
import os
import random
import statistics
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from release_gate import (  # noqa: E402
    BEATS_PERSISTENCE,
    BEATS_SERVED_MODEL,
    CHANNEL_SPECS,
    DEFAULT_BASELINE_PATH,
    DEFAULT_CHANNELS,
    DEFAULT_POLICY_PATH,
    DRIFT_DRIFT,
    DRIFT_MATCH,
    DRIFT_NOT_CHECKED,
    DRIFT_REL_TOLERANCE,
    DRIFT_SKIPPED,
    EXIT_BLOCKED,
    EXIT_OK,
    EXIT_RETAIN,
    EXIT_UNUSABLE,
    GateInputError,
    GateInputs,
    HORIZONS,
    MIN_HALF_PAIRED_ROWS,
    MIN_PAIRED_ROWS,
    REASON_HALF_REGRESSION,
    REASON_NOT_SIGNIFICANT,
    REASON_PROMOTED,
    REASON_WORSE,
    R_BLOCKED,
    R_PROMOTE_ALL,
    R_RETAIN_ALL,
    R_SPLIT,
    SIGMA_MULTIPLIER,
    SPLIT_DECISION,
    SPLIT_REPORTING,
    V_INSUFFICIENT,
    V_PROMOTE,
    V_REJECT,
    VOLATILE_BASELINE_FIELDS,
    BaselineCell,
    BaselineDocument,
    CellEvidence,
    CellMetric,
    PairedBlock,
    PolicyView,
    ServedSource,
    compare_baseline_documents,
    compare_baseline_to_fresh,
    evaluate_promotion,
    load_policy,
    load_release_baseline,
    main,
    write_verdict,
)

# The corpus SHA-256 recorded in data/release_baseline.md. Synthetic fixtures
# reuse it so the corpus check passes by default and can be made to fail on
# purpose; the real-file tests below assert it against the real documents.
BASELINE_CORPUS = "86ce906453aa071e12726e91da6c96c516285cf0bc7c3e3fb51d8d111f3bea4a"

# Incumbent scores taken from the +6h row of data/release_baseline.md. Real
# numbers, so a test that accidentally starts judging skill has something
# plausible to look at.
DEFAULT_INCUMBENT = {
    "temperature": 1.477,
    "humidity": 4.884,
    "pressure": 1.315,
    "wind_speed": 1.217,
    "rain_occurrence": 0.2209,
}

# A comfortable, unambiguous win: 0.02 better against a stderr of 0.002 is ten
# sigma, so nothing in these tests is decided by rounding.
WIN = (0.02, 0.002)


# --------------------------------------------------------------------------- #
# fixtures and builders
# --------------------------------------------------------------------------- #


def _baseline_cell_value(channel, _horizon=None):
    return DEFAULT_INCUMBENT[channel]


def _synthetic_baseline(corpus=BASELINE_CORPUS, values=None):
    """An in-memory incumbent record, so the rule tests never depend on a file.

    The real data/release_baseline.md is exercised separately in
    TestRealReleaseBaselineDocument, where a skip is an acceptable answer. Here
    a missing file would be a confusing failure in twenty unrelated tests.
    """
    cells = {}
    for channel in CHANNEL_SPECS:
        for horizon in HORIZONS:
            base = (values or {}).get((channel, horizon), _baseline_cell_value(channel, horizon))
            cells[(channel, horizon)] = BaselineCell(
                channel=channel,
                horizon=horizon,
                model=float(base),
                persistence=float(base) * 1.1,
                skill_pct=9.0909,
            )
    return BaselineDocument(
        path="<synthetic>",
        corpus_sha256=corpus,
        captured="2026-09-30T08:01:40Z",
        commit="ab204e2c2b3b",
        cells=cells,
    )


def _synthetic_policy(corpus=BASELINE_CORPUS, served=None, read_error=None,
                      fill_missing=True):
    """An in-memory served-source map.

    ``served`` overrides individual cells. With ``fill_missing=False`` the map
    is used exactly as given, so a test can express "the policy does not resolve
    this cell" instead of quietly getting a default.
    """
    sources = {}
    for channel in CHANNEL_SPECS:
        for horizon in HORIZONS:
            key = (channel, horizon)
            if key in (served or {}):
                sources[key] = served[key]
            elif fill_missing:
                sources[key] = ServedSource("persistence_fallback", "synthetic", "fixture")
    return PolicyView(
        policy_version="fixture", weather_telemetry_sha256=corpus, sources=sources,
        read_error=read_error,
    )


def _matching_drift(baseline=None):
    """The healthy regression guard: a fresh scoring identical to the record."""
    baseline = baseline or _synthetic_baseline()
    return compare_baseline_to_fresh(
        baseline, {k: c.model for k, c in dict(baseline.cells).items()}
    )


def _blocks(mean_diff, stderr, n_full=120, half_means=None, ids=None, split=SPLIT_DECISION):
    """Three PairedBlocks (full / early / late) with consistent row counts."""
    n_early = n_full // 2
    n_late = n_full - n_early
    e_mean, l_mean = half_means if half_means else (None, None)
    return {
        "full": PairedBlock(
            n_pairs=n_full, mean_difference=mean_diff, stderr=stderr, split=split,
            source="fixture", sample_ids=None if ids is None else tuple(ids),
        ),
        "early": PairedBlock(
            n_pairs=n_early,
            mean_difference=e_mean if e_mean is not None else mean_diff,
            stderr=stderr, split=split, source="fixture",
            sample_ids=None if ids is None else tuple(ids[0]),
        ),
        "late": PairedBlock(
            n_pairs=n_late,
            mean_difference=l_mean if l_mean is not None else mean_diff,
            stderr=stderr, split=split, source="fixture",
            sample_ids=None if ids is None else tuple(ids[1]),
        ),
    }


def _win_vectors(n, incumbent_value, improvement, sd, seed):
    """Per-row absolute-error vectors for a challenger beating the incumbent."""
    rng = random.Random(seed)
    incumbent = [abs(incumbent_value + rng.gauss(0.0, sd)) for _ in range(n)]
    challenger = [abs(incumbent_value - improvement + rng.gauss(0.0, sd)) for _ in range(n)]
    return incumbent, challenger


def _blocks_from_rows(incumbent, challenger, ids):
    """PairedBlocks computed from real per-row errors, in three spans."""
    def _one(inc, cand, span_ids):
        diffs = [i - c for i, c in zip(inc, cand)]
        return PairedBlock(
            n_pairs=len(diffs),
            mean_difference=statistics.fmean(diffs),
            stderr=statistics.stdev(diffs) / math.sqrt(len(diffs)),
            split=SPLIT_DECISION,
            source="fixture:per_row",
            sample_ids=tuple(span_ids),
            incumbent_errors=tuple(inc),
            candidate_errors=tuple(cand),
        )

    half = len(incumbent) // 2
    return {
        "full": _one(incumbent, challenger, ids),
        "early": _one(incumbent[:half], challenger[:half], ids[:half]),
        "late": _one(incumbent[half:], challenger[half:], ids[half:]),
    }


def _cell(value, paired, test_value=None, n_rows=120):
    return CellEvidence(
        validation=CellMetric(
            value=value, n_rows=n_rows, split=SPLIT_DECISION, source="fixture"
        ),
        test=(
            None
            if test_value is None
            else CellMetric(
                value=test_value, n_rows=n_rows, split=SPLIT_REPORTING, source="fixture"
            )
        ),
        paired=paired,
    )


def _grid(plan, channels=DEFAULT_CHANNELS, horizons=HORIZONS, incumbent_values=None,
          with_test=True):
    """Build (challenger, incumbent) evidence maps from a per-cell plan.

    ``plan`` maps (channel, horizon) to one of:
      (mean_diff, stderr)      -- a uniform win/loss at that spread
      ("missing",)             -- a None validation metric
      ("no_pairs",)            -- a good point estimate with no paired evidence
    """
    cand: dict = {}
    inc: dict = {}
    for key in [(c, h) for c in channels for h in horizons]:
        channel, _horizon = key
        spec = plan.get(key, WIN)
        base = (incumbent_values or {}).get(channel, _baseline_cell_value(channel))
        if spec == ("missing",):
            cand[key] = _cell(None, _blocks(*WIN))
            inc[key] = _cell(base, _blocks(*WIN))
            continue
        if spec == ("no_pairs",):
            cand[key] = _cell(base - WIN[0], {})
            inc[key] = _cell(base, _blocks(*WIN))
            continue
        mean_diff, stderr = spec
        halves = (mean_diff / 2.0, mean_diff / 2.0)
        blocks = _blocks(mean_diff, stderr, half_means=halves)
        cand[key] = _cell(
            base - mean_diff, blocks,
            test_value=(base - mean_diff) if with_test else None,
        )
        inc[key] = _cell(base, _blocks(*WIN), test_value=base if with_test else None)
    return cand, inc


def _gate_inputs(candidate=None, incumbent=None, *, corpus_sha256=BASELINE_CORPUS,
                 baseline=None, policy=None, drift=None, channels=DEFAULT_CHANNELS,
                 horizons=HORIZONS, candidate_name="challenger",
                 incumbent_name="incumbent"):
    """A GateInputs with a healthy baseline/policy/drift unless told otherwise."""
    baseline = _synthetic_baseline() if baseline is None else baseline
    if policy is None:
        policy = _synthetic_policy(corpus=baseline.corpus_sha256 or BASELINE_CORPUS)
    if drift is None and policy.read_error is None and baseline is not None:
        drift = _matching_drift(baseline)
    if candidate is None or incumbent is None:
        default_cand, default_inc = _grid({}, channels=channels, horizons=horizons)
        candidate = default_cand if candidate is None else candidate
        incumbent = default_inc if incumbent is None else incumbent
    return GateInputs(
        corpus_sha256=corpus_sha256,
        candidate=candidate,
        incumbent=incumbent,
        channels=tuple(channels),
        horizons=tuple(horizons),
        baseline=baseline,
        policy=policy,
        drift=drift,
        candidate_name=candidate_name,
        incumbent_name=incumbent_name,
    )


def _run(inputs):
    return evaluate_promotion(inputs)


def _verdicts(result):
    return {"%s|h%d" % (c.channel, c.horizon): c.verdict for c in result.cells}


def result_undecidable_is_published(view):
    """The dropped channels must appear in the published policy summary.

    A channel silently absent from the served-source map is a channel silently
    absent from every verdict, which is the shape of failure this gate exists to
    prevent -- so the omission has to be visible in the JSON.
    """
    return sorted(view.undecidable_channels) == list(view.undecidable_channels) and (
        "undecidable_channels" in view.to_dict())


# --------------------------------------------------------------------------- #
# THE RULE
# --------------------------------------------------------------------------- #

class TestRuleMechanics:
    """Step 1 (2-sigma), Step 2 (both halves), and the reject path."""

    def test_a_uniformly_better_challenger_promotes_every_cell(self):
        result, code = _run(_gate_inputs())
        assert code == EXIT_OK
        assert result.recommendation == R_PROMOTE_ALL
        assert len(result.cells) == len(DEFAULT_CHANNELS) * len(HORIZONS)
        assert all(c.verdict == V_PROMOTE for c in result.cells)
        assert all(c.reason == REASON_PROMOTED for c in result.cells)

    def test_exactly_at_two_sigma_is_rejected(self):
        """The boundary, stated as a hard case.

        At exactly the multiple the one-sided p is 0.0228. The rule says
        ``>``, so this is a REJECT. A gate that promotes on ``>=`` has quietly
        moved its own significance level, and the only way anyone would find out
        is by reading this line.
        """
        stderr = 0.005
        required = SIGMA_MULTIPLIER * stderr
        cand, inc = _grid({}, with_test=False)
        plan = {}
        for key in cand:
            plan[key] = (required, stderr)
        cand, inc = _grid(plan, with_test=False)
        # halves must be positive but are not themselves significance-tested
        for key, cell in cand.items():
            cell.paired["early"] = PairedBlock(
                n_pairs=60, mean_difference=required / 2.0, stderr=stderr,
                split=SPLIT_DECISION, source="fixture",
            )
            cell.paired["late"] = PairedBlock(
                n_pairs=60, mean_difference=required / 2.0, stderr=stderr,
                split=SPLIT_DECISION, source="fixture",
            )
        result, code = _run(_gate_inputs(cand, inc))
        assert all(
            c.evidence["mean_difference"] == c.evidence["required_difference"]
            for c in result.cells
        ), "fixture is broken: it is supposed to sit exactly on the threshold"
        assert all(c.verdict == V_REJECT for c in result.cells)
        assert all(c.reason == REASON_NOT_SIGNIFICANT for c in result.cells)
        assert code == EXIT_RETAIN

    @pytest.mark.parametrize("stderr", [0.0005, 0.005, 0.05, 1.0])
    def test_exactly_at_two_sigma_is_rejected_at_every_spread(self, stderr):
        required = SIGMA_MULTIPLIER * stderr
        plan = {(c, h): (required, stderr)
                for c in DEFAULT_CHANNELS for h in HORIZONS}
        cand, inc = _grid(plan, with_test=False)
        for cell in cand.values():
            cell.paired["early"] = PairedBlock(
                n_pairs=60, mean_difference=required / 2.0, stderr=stderr,
                split=SPLIT_DECISION, source="fixture")
            cell.paired["late"] = PairedBlock(
                n_pairs=60, mean_difference=required / 2.0, stderr=stderr,
                split=SPLIT_DECISION, source="fixture")
        result, _code = _run(_gate_inputs(cand, inc))
        assert all(c.verdict == V_REJECT for c in result.cells), (
            "a cell sitting exactly on %g sigma promoted at stderr=%r"
            % (SIGMA_MULTIPLIER, stderr))

    def test_the_next_float_past_two_sigma_promotes(self):
        """The complement of the boundary test, so the comparison is strict and
        not accidentally over-tight."""
        stderr = 0.002
        required = SIGMA_MULTIPLIER * stderr
        mean = math.nextafter(required, math.inf)
        assert mean > required
        plan = {(c, h): (mean, stderr)
                for c in DEFAULT_CHANNELS for h in HORIZONS}
        cand, inc = _grid(plan, with_test=False)
        for cell in cand.values():
            cell.paired["early"] = PairedBlock(
                n_pairs=60, mean_difference=mean / 2.0, stderr=stderr,
                split=SPLIT_DECISION, source="fixture")
            cell.paired["late"] = PairedBlock(
                n_pairs=60, mean_difference=mean / 2.0, stderr=stderr,
                split=SPLIT_DECISION, source="fixture")
        result, code = _run(_gate_inputs(cand, inc))
        assert all(c.verdict == V_PROMOTE for c in result.cells)
        assert code == EXIT_OK

    def test_a_worse_challenger_is_rejected_explicitly_not_omitted(self):
        plan = {(c, h): (-0.02, 0.002) for c in DEFAULT_CHANNELS for h in HORIZONS}
        cand, inc = _grid(plan, with_test=False)
        result, code = _run(_gate_inputs(cand, inc))
        assert all(c.verdict == V_REJECT for c in result.cells)
        assert all(c.reason == REASON_WORSE for c in result.cells)
        # The point of the test: the losing cells are PRESENT and NAMED.
        assert result.verdict["summary"]["rejected"], "no rejected cell was reported"
        assert len(result.verdict["summary"]["rejected"]) == len(result.cells)
        assert result.verdict["summary"]["promoted"] == []
        assert code == EXIT_RETAIN

    def test_a_zero_difference_is_rejected(self):
        plan = {(c, h): (0.0, 0.002) for c in DEFAULT_CHANNELS for h in HORIZONS}
        cand, inc = _grid(plan, with_test=False)
        for cell in cand.values():
            cell.paired["early"] = PairedBlock(
                n_pairs=60, mean_difference=0.0001, stderr=0.002,
                split=SPLIT_DECISION, source="fixture")
            cell.paired["late"] = PairedBlock(
                n_pairs=60, mean_difference=0.0001, stderr=0.002,
                split=SPLIT_DECISION, source="fixture")
        result, _code = _run(_gate_inputs(cand, inc))
        assert all(c.verdict == V_REJECT for c in result.cells)
        assert all(c.reason == REASON_NOT_SIGNIFICANT for c in result.cells)

    def test_a_real_gain_drowned_in_spread_is_rejected(self):
        """Same point improvement, absurd spread. The paired spread, not the
        headline number, is what the rule turns on."""
        plan = {(c, h): (0.02, 0.5) for c in DEFAULT_CHANNELS for h in HORIZONS}
        cand, inc = _grid(plan, with_test=False)
        for cell in cand.values():
            cell.paired["early"] = PairedBlock(
                n_pairs=60, mean_difference=0.01, stderr=0.5,
                split=SPLIT_DECISION, source="fixture")
            cell.paired["late"] = PairedBlock(
                n_pairs=60, mean_difference=0.01, stderr=0.5,
                split=SPLIT_DECISION, source="fixture")
        result, _code = _run(_gate_inputs(cand, inc))
        assert all(c.verdict == V_REJECT for c in result.cells)
        assert all(c.evidence["mean_difference"] > 0.0 for c in result.cells)

    def test_a_loss_confined_to_the_early_half_is_a_half_regression(self):
        cand, inc = _grid({})
        for cell in cand.values():
            cell.paired["early"] = PairedBlock(
                n_pairs=60, mean_difference=-0.01, stderr=0.002,
                split=SPLIT_DECISION, source="fixture")
        result, _code = _run(_gate_inputs(cand, inc))
        assert all(c.verdict == V_REJECT for c in result.cells)
        assert all(c.reason == REASON_HALF_REGRESSION for c in result.cells)
        assert all(c.evidence["failing_half"] == "early" for c in result.cells)

    def test_a_loss_confined_to_the_late_half_is_a_half_regression(self):
        cand, inc = _grid({})
        for cell in cand.values():
            cell.paired["late"] = PairedBlock(
                n_pairs=60, mean_difference=-0.01, stderr=0.002,
                split=SPLIT_DECISION, source="fixture")
        result, _code = _run(_gate_inputs(cand, inc))
        assert all(c.verdict == V_REJECT for c in result.cells)
        assert all(c.evidence["failing_half"] == "late" for c in result.cells)

    def test_a_tie_in_one_half_is_not_a_win_in_that_half(self):
        """Exactly zero in a half fails the half test, the same way a loss
        does. 'Ahead' is strict."""
        cand, inc = _grid({})
        for cell in cand.values():
            cell.paired["late"] = PairedBlock(
                n_pairs=60, mean_difference=0.0, stderr=0.002,
                split=SPLIT_DECISION, source="fixture")
        result, _code = _run(_gate_inputs(cand, inc))
        assert all(c.verdict == V_REJECT for c in result.cells)
        assert all(c.evidence["failing_half"] == "late" for c in result.cells)

    def test_evidence_computed_from_real_per_row_errors_is_honoured(self):
        """The trustworthy path: the gate derives the paired statistic itself."""
        inc_err, cand_err = _win_vectors(200, 1.477, 0.06, 0.10, seed=7)
        ids = ["row%04d" % i for i in range(200)]
        blocks = _blocks_from_rows(inc_err, cand_err, ids)
        cand, inc = _grid({})
        for key in cand:
            cand[key] = _cell(1.477 - 0.06, blocks)
        result, code = _run(_gate_inputs(cand, inc))
        assert code == EXIT_OK
        assert all(c.verdict == V_PROMOTE for c in result.cells)
        assert all(c.evidence["stderr_source"] == "COMPUTED_FROM_ROWS"
                   for c in result.cells)
        assert all(c.evidence["halves"]["early"]["ahead"] for c in result.cells)
        assert all(c.evidence["halves"]["late"]["ahead"] for c in result.cells)

    def test_rows_paired_in_the_wrong_order_are_caught(self):
        """The mis-pairing guard. Shuffling the challenger column produces a
        block whose supplied mean no longer matches its own rows; the gate
        refuses rather than reporting a confident wrong number."""
        inc_err, cand_err = _win_vectors(200, 1.477, 0.06, 0.10, seed=7)
        ids = ["row%04d" % i for i in range(200)]
        blocks = _blocks_from_rows(inc_err, cand_err, ids)
        full = blocks["full"]
        shuffled = list(full.candidate_errors)
        random.Random(3).shuffle(shuffled)
        tampered = PairedBlock(
            n_pairs=full.n_pairs, mean_difference=full.mean_difference,
            stderr=full.stderr, split=SPLIT_DECISION, source="fixture:per_row",
            sample_ids=full.sample_ids, incumbent_errors=full.incumbent_errors,
            candidate_errors=tuple(shuffled),
        )
        cand, inc = _grid({})
        for key in cand:
            bad = dict(blocks)
            bad["full"] = tampered
            cand[key] = _cell(1.477 - 0.06, bad)
        result, code = _run(_gate_inputs(cand, inc))
        assert code == EXIT_BLOCKED
        assert all(c.verdict == V_INSUFFICIENT for c in result.cells)
        # Either half of the paired statistic can be the one that moves; both
        # are the same mistake and both must stop the run.
        assert all(c.reason in ("PAIRED_STATISTIC_DISAGREEMENT",
                                "PAIRED_STDERR_DISAGREEMENT") for c in result.cells), (
            "a mis-paired challenger column was not caught: %s"
            % sorted({c.reason for c in result.cells}))
        assert "does not match" in result.cells[0].detail

    def test_supplied_paired_statistics_are_used_when_no_rows_are_given(self):
        result, _code = _run(_gate_inputs())
        assert all(c.evidence["stderr_source"] == "SUPPLIED" for c in result.cells)

    def test_a_zero_standard_error_is_insufficient_not_equal(self):
        plan = {(c, h): (0.02, 0.0) for c in DEFAULT_CHANNELS for h in HORIZONS}
        cand, inc = _grid(plan, with_test=False)
        for cell in cand.values():
            cell.paired["early"] = PairedBlock(
                n_pairs=60, mean_difference=0.01, stderr=0.0,
                split=SPLIT_DECISION, source="fixture")
            cell.paired["late"] = PairedBlock(
                n_pairs=60, mean_difference=0.01, stderr=0.0,
                split=SPLIT_DECISION, source="fixture")
        result, code = _run(_gate_inputs(cand, inc))
        assert code == EXIT_BLOCKED
        assert all(c.verdict == V_INSUFFICIENT for c in result.cells)
        assert all(c.reason == "PAIRED_STDERR_NOT_POSITIVE" for c in result.cells)


# --------------------------------------------------------------------------- #
# CHANNEL INDEPENDENCE
# --------------------------------------------------------------------------- #

class TestChannelIndependence:
    """The reason per-channel verdicts exist at all.

    Rejecting a candidate wholesale because it regressed on one channel throws
    away the gains on the others while keeping the regression. That is the
    failure this project is trying to avoid, so it gets its own test class.
    """

    def test_better_at_wind_and_worse_at_rain_splits(self):
        plan = {}
        for channel in DEFAULT_CHANNELS:
            for horizon in HORIZONS:
                plan[(channel, horizon)] = (-0.02, 0.002) if channel == "rain_occurrence" \
                    else WIN
        cand, inc = _grid(plan)
        result, code = _run(_gate_inputs(cand, inc))

        assert result.recommendation == R_SPLIT
        assert code == EXIT_OK, "a split is a fully-earned decision, not a failure"
        rain = ["%s|h%d" % ("rain_occurrence", h) for h in HORIZONS]
        others = ["%s|h%d" % (c, h) for c in DEFAULT_CHANNELS
                  if c != "rain_occurrence" for h in HORIZONS]
        assert all(_verdicts(result)[k] == V_PROMOTE for k in others)
        assert all(_verdicts(result)[k] == V_REJECT for k in rain)
        assert all(
            result.cells[i].reason == REASON_WORSE
            for i, c in enumerate(result.cells) if c.channel == "rain_occurrence"
        )

    def test_the_losing_channel_is_named_in_the_summary(self):
        plan = {(c, h): (WIN if c == "wind_speed" else (-0.02, 0.002))
                for c in DEFAULT_CHANNELS for h in HORIZONS}
        cand, inc = _grid(plan)
        result, _code = _run(_gate_inputs(cand, inc))
        summary = result.verdict["summary"]
        assert summary["promoted"] and summary["rejected"]
        assert all(k.startswith("wind_speed|") for k in summary["promoted"])
        assert not any(k.startswith("wind_speed|") for k in summary["rejected"])
        assert len(summary["promoted"]) + len(summary["rejected"]) == len(result.cells)

    def test_an_unevaluable_channel_does_not_contaminate_an_evaluable_one(self):
        plan = {(c, h): WIN for c in DEFAULT_CHANNELS for h in HORIZONS}
        plan[("rain_occurrence", 6)] = ("missing",)
        cand, inc = _grid(plan)
        result, code = _run(_gate_inputs(cand, inc))
        by_key = _verdicts(result)
        assert by_key["rain_occurrence|h6"] == V_INSUFFICIENT
        assert by_key["temperature|h6"] == V_PROMOTE
        # The run is blocked overall, but the evaluable cell keeps its verdict:
        # recording the wins is the entire point of deciding per cell.
        assert result.recommendation == R_BLOCKED
        assert code == EXIT_BLOCKED
        assert result.verdict["promotion_decision_computed"] is False

    def test_every_requested_cell_appears_in_the_verdict_document(self):
        channels = ("temperature", "rain_occurrence")
        horizons = (6, 12)
        cand, inc = _grid({}, channels=channels, horizons=horizons)
        result, _code = _run(_gate_inputs(cand, inc, channels=channels,
                                          horizons=horizons))
        cells = result.verdict["cells"]
        assert set(cells) == {"%s|h%d" % (c, h) for c in channels for h in horizons}
        assert len(result.cells) == 4

    def test_the_reported_improvement_numbers_are_derived_not_asserted(self):
        """A published percentage is quotable, so it has to be arithmetic on the
        two numbers beside it. A reader who sees '4.9% better' and cannot
        reproduce it has been handed exactly the over-claim this project
        shipped once."""
        cand, inc = _grid({})
        result, _code = _run(_gate_inputs(cand, inc))
        checked = 0
        for cell in result.cells:
            evidence = cell.evidence
            incumbent = evidence["incumbent"]
            challenger = evidence["challenger"]
            assert evidence["absolute_improvement"] == pytest.approx(
                incumbent - challenger, abs=1e-12)
            assert evidence["relative_improvement"] == pytest.approx(
                (incumbent - challenger) / abs(incumbent), rel=1e-9)
            assert evidence["sigma"] == pytest.approx(
                evidence["mean_difference"] / evidence["stderr"], rel=1e-9)
            assert evidence["required_difference"] == pytest.approx(
                SIGMA_MULTIPLIER * evidence["stderr"], abs=1e-12)
            checked += 1
        assert checked == len(DEFAULT_CHANNELS) * len(HORIZONS)

    def test_an_improvement_is_never_reported_for_a_cell_that_did_not_promote(self):
        """`relative_improvement` is a description of the measurement, not a
        verdict. A rejected cell may legitimately show a positive number (it
        beat persistence without clearing 2 sigma), and the verdict is what
        governs. Pin both, so neither can be read off the other."""
        plan = {(c, h): (0.001, 0.002) for c in DEFAULT_CHANNELS for h in HORIZONS}
        cand, inc = _grid(plan, with_test=False)
        for cell in cand.values():
            cell.paired["early"] = PairedBlock(
                n_pairs=60, mean_difference=0.0005, stderr=0.002,
                split=SPLIT_DECISION, source="fixture")
            cell.paired["late"] = PairedBlock(
                n_pairs=60, mean_difference=0.0005, stderr=0.002,
                split=SPLIT_DECISION, source="fixture")
        result, code = _run(_gate_inputs(cand, inc))
        assert all(c.verdict == V_REJECT for c in result.cells)
        assert all(c.reason == REASON_NOT_SIGNIFICANT for c in result.cells)
        assert all(c.evidence["mean_difference"] > 0.0 for c in result.cells)
        assert all(c.evidence["relative_improvement"] > 0.0 for c in result.cells)
        assert result.verdict["summary"]["promoted"] == []
        assert code == EXIT_RETAIN


# --------------------------------------------------------------------------- #
# ABSENT EVIDENCE
# --------------------------------------------------------------------------- #

class TestAbsentEvidenceIsNeverAPass:
    """The headline property. Absence in any shape yields a verdict that is
    never PROMOTE, and a run that cannot decide exits non-zero."""

    def test_a_candidate_with_no_metrics_promotes_nothing(self):
        plan = {(c, h): ("missing",) for c in DEFAULT_CHANNELS for h in HORIZONS}
        cand, inc = _grid(plan)
        result, code = _run(_gate_inputs(cand, inc))
        assert all(c.verdict == V_INSUFFICIENT for c in result.cells)
        assert all(c.reason == "CANDIDATE_METRIC_MISSING" for c in result.cells)
        assert result.verdict["summary"]["promoted"] == []
        assert result.recommendation == R_BLOCKED
        assert code == EXIT_BLOCKED

    def test_a_missing_incumbent_metric_is_insufficient_not_a_free_pass(self):
        plan = {(c, h): WIN for c in DEFAULT_CHANNELS for h in HORIZONS}
        cand, inc = _grid(plan)
        # wipe the incumbent value for exactly one cell, leaving the challenger
        # intact so the incumbent branch is the one under test
        inc = dict(inc)
        original = inc[("humidity", 12)]
        inc[("humidity", 12)] = CellEvidence(
            validation=CellMetric(value=None, split=SPLIT_DECISION, source="fixture"),
            paired=original.paired,
        )
        result, code = _run(_gate_inputs(cand, inc))
        cell = result.verdict_for("humidity", 12)
        assert cell.verdict == V_INSUFFICIENT
        assert cell.reason == "INCUMBENT_METRIC_MISSING"
        assert result.recommendation == R_BLOCKED
        assert code == EXIT_BLOCKED
        # and the neighbouring cell is untouched
        assert result.verdict_for("humidity", 6).verdict == V_PROMOTE

    @pytest.mark.parametrize("absence", [
        "value_none", "key_absent", "block_absent", "stderr_absent",
        "half_absent", "rows_below_minimum", "half_rows_below_minimum",
        "channel_unknown", "policy_source_unknown",
    ])
    def test_no_shape_of_absence_can_produce_a_promotion(self, absence):
        cand, inc = _grid({})
        if absence == "value_none":
            cell = cand[("temperature", 6)]
            cand[("temperature", 6)] = CellEvidence(paired=cell.paired)
        elif absence == "key_absent":
            cand = {k: v for k, v in cand.items() if k != ("temperature", 6)}
        elif absence == "block_absent":
            cell = cand[("temperature", 6)]
            cand[("temperature", 6)] = CellEvidence(
                validation=cell.validation, test=cell.test, paired={})
        elif absence == "stderr_absent":
            cell = cand[("temperature", 6)]
            blocks = dict(cell.paired)
            blocks["full"] = PairedBlock(split=SPLIT_DECISION, source="fixture")
            cand[("temperature", 6)] = _cell(cell.validation.value, blocks)
        elif absence == "half_absent":
            cell = cand[("temperature", 6)]
            blocks = {k: v for k, v in cell.paired.items() if k != "early"}
            cand[("temperature", 6)] = _cell(cell.validation.value, blocks)
        elif absence == "rows_below_minimum":
            cell = cand[("temperature", 6)]
            blocks = dict(cell.paired)
            blocks["full"] = PairedBlock(
                n_pairs=MIN_PAIRED_ROWS - 1, mean_difference=0.02, stderr=0.002,
                split=SPLIT_DECISION, source="fixture")
            blocks["early"] = PairedBlock(
                n_pairs=5, mean_difference=0.01, stderr=0.002,
                split=SPLIT_DECISION, source="fixture")
            blocks["late"] = PairedBlock(
                n_pairs=5, mean_difference=0.01, stderr=0.002,
                split=SPLIT_DECISION, source="fixture")
            cand[("temperature", 6)] = _cell(cell.validation.value, blocks)
        elif absence == "half_rows_below_minimum":
            cell = cand[("temperature", 6)]
            blocks = dict(cell.paired)
            blocks["full"] = PairedBlock(
                n_pairs=2 * MIN_HALF_PAIRED_ROWS, mean_difference=0.02, stderr=0.002,
                split=SPLIT_DECISION, source="fixture")
            blocks["early"] = PairedBlock(
                n_pairs=MIN_HALF_PAIRED_ROWS - 1, mean_difference=0.01, stderr=0.002,
                split=SPLIT_DECISION, source="fixture")
            blocks["late"] = PairedBlock(
                n_pairs=MIN_HALF_PAIRED_ROWS + 1, mean_difference=0.01, stderr=0.002,
                split=SPLIT_DECISION, source="fixture")
            cand[("temperature", 6)] = _cell(cell.validation.value, blocks)
        elif absence == "channel_unknown":
            cand[("solar_flux", 6)] = cand[("temperature", 6)]
            inputs = _gate_inputs(cand, inc, channels=("solar_flux",))
            result, code = _run(inputs)
            assert result.cells[0].verdict == V_INSUFFICIENT
            assert result.cells[0].reason == "UNKNOWN_CHANNEL"
            assert result.verdict["summary"]["promoted"] == []
            assert code == EXIT_BLOCKED
            return
        elif absence == "policy_source_unknown":
            # a policy that resolves every cell EXCEPT this one, so the cell's
            # own reason is the one under test rather than a gate-wide blocker
            served = {(c, h): ServedSource("persistence_fallback", "synthetic", "fixture")
                      for c in DEFAULT_CHANNELS for h in HORIZONS}
            del served[("temperature", 6)]
            policy = _synthetic_policy(served=served, fill_missing=False)
            result, code = _run(_gate_inputs(cand, inc, policy=policy))
            assert result.verdict_for("temperature", 6).verdict == V_INSUFFICIENT
            assert result.verdict_for("temperature", 6).reason == "POLICY_SOURCE_UNKNOWN"
            assert "temperature|h6" in result.verdict["summary"]["insufficient_evidence"]
            assert "temperature|h6" not in result.verdict["summary"]["promoted"]
            assert result.recommendation == R_BLOCKED
            assert code == EXIT_BLOCKED
            return

        result, code = _run(_gate_inputs(cand, inc))
        cell = result.verdict_for("temperature", 6)
        assert cell.verdict == V_INSUFFICIENT, (
            "temperature|h6 promoted on absent evidence: %s" % cell.reason)
        assert cell.reason
        # the rest of the grid keeps its honest verdicts
        assert result.verdict_for("pressure", 6).verdict == V_PROMOTE
        assert result.recommendation == R_BLOCKED
        assert code == EXIT_BLOCKED
        assert "temperature|h6" in result.verdict["summary"]["insufficient_evidence"]

    def test_better_point_estimates_with_no_paired_evidence_promote_nothing(self):
        """The exact silent-pass vector: a challenger whose every headline
        number is better, and which therefore 'looks promoted' to anything
        reading the metric table, with no paired evidence behind it at all.

        The incumbent's paired blocks are also removed below, so a
        point-estimate-only reader would see a uniform sweep and no verdict
        blocking it.
        """
        plan = {(c, h): ("no_pairs",) for c in DEFAULT_CHANNELS for h in HORIZONS}
        cand, inc = _grid(plan)
        result, code = _run(_gate_inputs(cand, inc))
        assert result.verdict["summary"]["promoted"] == []
        assert all(c.verdict == V_INSUFFICIENT for c in result.cells)
        assert all(c.reason == "PAIRED_BLOCK_MISSING" for c in result.cells)
        assert result.recommendation == R_BLOCKED
        assert code == EXIT_BLOCKED

    def test_an_empty_evidence_map_blocks_rather_than_passes(self):
        result, code = _run(_gate_inputs({}, {}))
        assert all(c.verdict == V_INSUFFICIENT for c in result.cells)
        assert all(c.reason == "CELL_ABSENT_FROM_EVIDENCE" for c in result.cells)
        assert result.verdict["summary"]["promoted"] == []
        assert result.recommendation == R_BLOCKED
        assert code == EXIT_BLOCKED

    def test_an_empty_grid_is_refused_rather_than_returned_as_a_verdict(self):
        """A gate over zero cells must not produce a decision. `_recommendation`
        on an empty cell list would tally nothing and fall through to
        RETAIN_ALL, which is a verdict nobody earned."""
        cand, inc = _grid({})
        with pytest.raises(GateInputError):
            _gate_inputs(cand, inc, channels=())
        with pytest.raises(GateInputError):
            _gate_inputs(cand, inc, horizons=())
        assert len(DEFAULT_CHANNELS) > 0 and len(HORIZONS) > 0

    def test_non_finite_metrics_are_refused_at_the_boundary(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with pytest.raises(GateInputError):
                CellMetric(value=bad, split=SPLIT_DECISION, source="fixture")
        with pytest.raises(GateInputError):
            PairedBlock(n_pairs=10, mean_difference=float("nan"), stderr=0.1,
                        split=SPLIT_DECISION, source="fixture")
        with pytest.raises(GateInputError):
            CellMetric(value=True, split=SPLIT_DECISION, source="fixture")
        with pytest.raises(GateInputError):
            CellMetric(value="0.5", split=SPLIT_DECISION, source="fixture")

    def test_every_non_promotion_carries_a_reason(self):
        plan = {}
        for channel in DEFAULT_CHANNELS:
            for horizon in HORIZONS:
                if channel == "rain_occurrence":
                    plan[(channel, horizon)] = ("missing",)
                elif channel == "wind_speed":
                    plan[(channel, horizon)] = (-0.02, 0.002)
                else:
                    plan[(channel, horizon)] = WIN
        cand, inc = _grid(plan)
        result, _code = _run(_gate_inputs(cand, inc))
        for cell in result.cells:
            if cell.verdict != V_PROMOTE:
                assert cell.reason and cell.reason.strip(), (
                    "%s|h%d is %s with no reason" % (cell.channel, cell.horizon,
                                                     cell.verdict))


# --------------------------------------------------------------------------- #
# NO TEST-SPLIT LEAKAGE
# --------------------------------------------------------------------------- #

class TestNoTestSplitLeakage:
    """The rule says validation only. The test split is for reporting."""

    def test_a_paired_block_labelled_test_is_refused(self):
        with pytest.raises(GateInputError) as exc:
            PairedBlock(n_pairs=100, mean_difference=0.02, stderr=0.002,
                        split=SPLIT_REPORTING, source="fixture")
        assert "validation" in str(exc.value)

    def test_a_validation_metric_labelled_test_is_refused(self):
        with pytest.raises(GateInputError):
            CellEvidence(
                validation=CellMetric(value=1.0, split=SPLIT_REPORTING, source="fixture"),
                paired={},
            )

    def test_a_test_metric_in_the_validation_slot_is_refused(self):
        with pytest.raises(GateInputError):
            CellEvidence(
                validation=CellMetric(value=1.0, split=SPLIT_REPORTING, source="fixture"),
                test=CellMetric(value=1.0, split=SPLIT_DECISION, source="fixture"),
            )

    def test_test_metrics_cannot_change_a_single_verdict(self):
        """Structural proof: the verdict document is byte-identical whether or
        not test metrics were supplied. If they ever reached the rule, this
        fails."""
        with_test, incumbent_with = _grid({}, with_test=True)
        without_test, incumbent_without = _grid({}, with_test=False)
        # non-vacuity: the two inputs really do differ
        assert with_test[("temperature", 6)].test is not None
        assert without_test[("temperature", 6)].test is None
        a, _ = _run(_gate_inputs(with_test, incumbent_with))
        b, _ = _run(_gate_inputs(without_test, incumbent_without))
        assert json.dumps(a.verdict["cells"], sort_keys=True) == \
            json.dumps(b.verdict["cells"], sort_keys=True)
        assert a.verdict["counts"] == b.verdict["counts"]
        assert a.verdict["summary"] == b.verdict["summary"]

    def test_test_metrics_are_reported_separately_from_the_decision_inputs(self):
        """They are carried, and under a name that says what they are for."""
        cand, inc = _grid({}, with_test=True)
        result, _code = _run(_gate_inputs(cand, inc))
        decision_inputs = result.verdict["decision_inputs_used"]
        assert decision_inputs["split_used_for_decision"] == SPLIT_DECISION
        assert decision_inputs["split_used_for_reporting"] == SPLIT_REPORTING
        cell = result.verdict["cells"]["temperature|h6"]
        assert "test" not in json.dumps(cell.get("evidence", {}))
        assert cand[("temperature", 6)].test.split == SPLIT_REPORTING
        assert cand[("temperature", 6)].test.value is not None


# --------------------------------------------------------------------------- #
# HALF-SPLIT INTEGRITY
# --------------------------------------------------------------------------- #

class TestHalfSplitIntegrity:
    """'Both halves' is a claim about rows. These guard it from being theatre."""

    def test_overlapping_halves_are_refused(self):
        ids = ["row%04d" % i for i in range(120)]
        overlapping = (ids[:60], ids[30:90])   # 30 rows in both halves
        cand, inc = _grid({})
        for cell in cand.values():
            cell.paired["full"] = PairedBlock(
                n_pairs=120, mean_difference=0.02, stderr=0.002,
                split=SPLIT_DECISION, source="fixture", sample_ids=tuple(ids))
            cell.paired["early"] = PairedBlock(
                n_pairs=60, mean_difference=0.01, stderr=0.002,
                split=SPLIT_DECISION, source="fixture", sample_ids=tuple(overlapping[0]))
            cell.paired["late"] = PairedBlock(
                n_pairs=60, mean_difference=0.01, stderr=0.002,
                split=SPLIT_DECISION, source="fixture", sample_ids=tuple(overlapping[1]))
        result, code = _run(_gate_inputs(cand, inc))
        assert all(c.verdict == V_INSUFFICIENT for c in result.cells)
        assert all(c.reason == "HALVES_NOT_DISJOINT" for c in result.cells)
        assert code == EXIT_BLOCKED

    def test_halves_that_do_not_partition_the_period_are_refused(self):
        ids = ["row%04d" % i for i in range(120)]
        cand, inc = _grid({})
        for cell in cand.values():
            cell.paired["full"] = PairedBlock(
                n_pairs=120, mean_difference=0.02, stderr=0.002,
                split=SPLIT_DECISION, source="fixture", sample_ids=tuple(ids))
            cell.paired["early"] = PairedBlock(
                n_pairs=60, mean_difference=0.01, stderr=0.002,
                split=SPLIT_DECISION, source="fixture", sample_ids=tuple(ids[:60]))
            cell.paired["late"] = PairedBlock(
                n_pairs=60, mean_difference=0.01, stderr=0.002,
                split=SPLIT_DECISION, source="fixture", sample_ids=tuple(ids[60:110]))
        result, _code = _run(_gate_inputs(cand, inc))
        assert all(c.reason == "HALVES_DO_NOT_PARTITION_THE_PERIOD"
                   for c in result.cells)

    def test_halves_labelled_backwards_are_refused(self):
        """The realistic mistake: the caller swapped which half is early and
        which is late. The halves are disjoint and cover the period, so nothing
        else catches it -- the split is simply not a split."""
        ids = ["row%04d" % i for i in range(120)]
        cand, inc = _grid({})
        for cell in cand.values():
            cell.paired["full"] = PairedBlock(
                n_pairs=120, mean_difference=0.02, stderr=0.002,
                split=SPLIT_DECISION, source="fixture", sample_ids=tuple(ids))
            cell.paired["early"] = PairedBlock(
                n_pairs=60, mean_difference=0.01, stderr=0.002,
                split=SPLIT_DECISION, source="fixture", sample_ids=tuple(ids[60:120]))
            cell.paired["late"] = PairedBlock(
                n_pairs=60, mean_difference=0.01, stderr=0.002,
                split=SPLIT_DECISION, source="fixture", sample_ids=tuple(ids[0:60]))
        result, _code = _run(_gate_inputs(cand, inc))
        assert all(c.reason == "HALVES_OVERLAP_IN_TIME" for c in result.cells)
        assert all(c.verdict == V_INSUFFICIENT for c in result.cells)

    def test_halves_with_unsorted_row_ids_are_refused(self):
        ids = ["row%04d" % i for i in range(120)]
        cand, inc = _grid({})
        for cell in cand.values():
            cell.paired["full"] = PairedBlock(
                n_pairs=120, mean_difference=0.02, stderr=0.002,
                split=SPLIT_DECISION, source="fixture", sample_ids=tuple(ids))
            cell.paired["early"] = PairedBlock(
                n_pairs=60, mean_difference=0.01, stderr=0.002,
                split=SPLIT_DECISION, source="fixture",
                sample_ids=tuple(reversed(ids[0:60])))
            cell.paired["late"] = PairedBlock(
                n_pairs=60, mean_difference=0.01, stderr=0.002,
                split=SPLIT_DECISION, source="fixture",
                sample_ids=tuple(reversed(ids[60:120])))
        result, _code = _run(_gate_inputs(cand, inc))
        assert all(c.reason == "HALVES_NOT_CHRONOLOGICALLY_ORDERED"
                   for c in result.cells)

    def test_half_row_counts_must_sum_to_the_full_period(self):
        cand, inc = _grid({})
        for key in list(cand):
            cell = cand[key]
            blocks = dict(cell.paired)
            blocks["early"] = PairedBlock(
                n_pairs=60, mean_difference=0.01, stderr=0.002,
                split=SPLIT_DECISION, source="fixture")
            blocks["late"] = PairedBlock(
                n_pairs=40, mean_difference=0.01, stderr=0.002,
                split=SPLIT_DECISION, source="fixture")
            cand[key] = _cell(cell.validation.value, blocks)
        result, _code = _run(_gate_inputs(cand, inc))
        assert all(c.reason == "HALF_ROW_COUNTS_DO_NOT_SUM" for c in result.cells), (
            "a full period of 120 rows was covered by halves of 60 and 40 and "
            "the gate did not notice")
        assert all(c.verdict == V_INSUFFICIENT for c in result.cells)


# --------------------------------------------------------------------------- #
# CORPUS
# --------------------------------------------------------------------------- #

class TestCorpusIntegrity:
    """A difference between two models scored on different data is not a
    difference between the models."""

    def test_a_corpus_mismatch_blocks_every_cell(self):
        cand, inc = _grid({})
        inputs = _gate_inputs(cand, inc, corpus_sha256="0" * 64)
        result, code = _run(inputs)
        assert code == EXIT_BLOCKED
        assert result.recommendation == R_BLOCKED
        assert all(c.verdict == V_INSUFFICIENT for c in result.cells)
        assert all(c.reason == "GATE_INTEGRITY_BLOCKED" for c in result.cells)
        assert all(c.evidence["blocker"] == "CORPUS_MISMATCH" for c in result.cells)
        assert result.verdict["summary"]["promoted"] == []

    def test_a_missing_corpus_sha_blocks_and_is_named(self):
        """An empty SHA is allowed to reach the gate so the verdict document can
        say WHICH input is missing. A non-string is malformed and is refused
        before the gate runs at all."""
        cand, inc = _grid({})
        result, code = _run(_gate_inputs(cand, inc, corpus_sha256="   "))
        assert code == EXIT_BLOCKED
        assert result.verdict["integrity"]["corpus"]["status"] == "FAIL"
        assert result.verdict["integrity"]["corpus"]["reason"] == "CORPUS_SHA_MISSING"
        assert all(c.verdict == V_INSUFFICIENT for c in result.cells)
        assert result.verdict["summary"]["promoted"] == []
        with pytest.raises(GateInputError):
            _gate_inputs(cand, inc, corpus_sha256=None)
        with pytest.raises(GateInputError):
            _gate_inputs(cand, inc, corpus_sha256=12345)

    def test_a_policy_refitted_onto_another_corpus_blocks(self):
        baseline = _synthetic_baseline()
        policy = _synthetic_policy(corpus="a" * 64)
        cand, inc = _grid({})
        result, code = _run(_gate_inputs(cand, inc, baseline=baseline, policy=policy))
        assert code == EXIT_BLOCKED
        assert result.verdict["integrity"]["policy_corpus"]["reason"] == \
            "POLICY_CORPUS_DIVERGED"
        assert all(c.verdict == V_INSUFFICIENT for c in result.cells)


# --------------------------------------------------------------------------- #
# POLICY / SERVED SOURCE
# --------------------------------------------------------------------------- #

class TestPolicySourceResolution:
    """13 of 20 cells in the reference baseline are persistence, and beating
    persistence is not the same question as beating a model."""

    def test_an_unknown_served_source_is_insufficient(self):
        """A gap in the policy blocks the run without erasing the per-cell wins.

        The other 24 cells have a complete decision and keep it; the unresolved
        one is named; nothing is promoted overall because the run did not
        complete.
        """
        served = {(c, h): ServedSource("persistence_fallback", "synthetic", "fixture")
                  for c in DEFAULT_CHANNELS for h in HORIZONS}
        del served[("pressure", 24)]
        policy = _synthetic_policy(served=served, fill_missing=False)
        cand, inc = _grid({})
        result, code = _run(_gate_inputs(cand, inc, policy=policy))
        assert code == EXIT_BLOCKED
        assert result.recommendation == R_BLOCKED
        gap = result.verdict_for("pressure", 24)
        assert gap.verdict == V_INSUFFICIENT
        assert gap.reason == "POLICY_SOURCE_UNKNOWN"
        assert all(c.verdict == V_PROMOTE for c in result.cells
                   if c.key != ("pressure", 24))
        summary = result.verdict["summary"]
        assert "pressure|h24" in summary["insufficient_evidence"]
        assert "pressure|h24" not in summary["promoted"]
        assert len(summary["promoted"]) == len(result.cells) - 1

    def test_a_policy_that_resolves_nothing_blocks_on_its_own(self):
        policy = PolicyView(
            policy_version="test", weather_telemetry_sha256=BASELINE_CORPUS, sources={})
        cand, inc = _grid({})
        result, code = _run(_gate_inputs(cand, inc, policy=policy))
        assert code == EXIT_BLOCKED
        assert result.verdict["integrity"]["policy"]["reason"] == "POLICY_EMPTY"
        assert all(c.verdict == V_INSUFFICIENT for c in result.cells)
        assert result.verdict["summary"]["promoted"] == []

    def test_an_unreadable_policy_blocks_and_is_named(self):
        policy = load_policy(os.path.join(HERE, "no_such_policy_file.json"))
        assert policy.read_error and "not found" in policy.read_error
        cand, inc = _grid({})
        result, code = _run(_gate_inputs(cand, inc, policy=policy, drift=None))
        assert code == EXIT_BLOCKED
        assert result.verdict["integrity"]["policy"]["reason"] == "POLICY_UNREADABLE"
        assert all(c.verdict == V_INSUFFICIENT for c in result.cells)

    def test_persistence_cells_are_tagged_and_counted_separately(self):
        served = {}
        for channel in DEFAULT_CHANNELS:
            for horizon in HORIZONS:
                served[(channel, horizon)] = ServedSource(
                    source=("learned_model" if channel == "wind_speed"
                            else "persistence_fallback"),
                    detail="synthetic", origin="fixture")
        policy = _synthetic_policy(served=served)
        cand, inc = _grid({})
        result, code = _run(_gate_inputs(cand, inc, policy=policy))
        assert code == EXIT_OK
        assert result.verdict["summary"]["promotions_beating_served_model"]
        assert result.verdict["summary"]["promotions_beating_persistence"]
        assert (set(result.verdict["summary"]["promotions_beating_served_model"])
                | set(result.verdict["summary"]["promotions_beating_persistence"])
                == set(result.verdict["summary"]["promoted"]))
        for cell in result.cells:
            expected = (BEATS_SERVED_MODEL if cell.channel == "wind_speed"
                        else BEATS_PERSISTENCE)
            assert cell.evidence["promotion_class"] == expected

    def test_rain_is_resolved_from_the_blend_weights_not_left_unknown(self):
        """inference_policy.json has no selected_sources entry for rain; it has
        a blend. Treating that as 'unknown' would make every rain cell
        undecidable for a reason that is really 'the gate only read one field'."""
        if not os.path.exists(DEFAULT_POLICY_PATH):
            pytest.skip("inference_policy.json is not present")
        view = load_policy()
        if view.read_error:
            pytest.skip("inference_policy.json is unreadable: %s" % view.read_error)
        for horizon in HORIZONS:
            served = view.resolve("rain_occurrence", horizon)
            assert served is not None, "rain is unresolved at +%dh" % horizon
            assert served.origin == "rain_blend_weights"
            assert "rain_model_weight" in served.detail

    def test_derived_and_undecidable_sources_are_not_indexed_as_served(self):
        """heat_index is `derived_*` and wind_direction is a scored channel the
        gate holds no rule for. Indexing either would offer a verdict nobody
        defined, so they are named in `undecidable_channels` instead of
        vanishing from the report."""
        if not os.path.exists(DEFAULT_POLICY_PATH):
            pytest.skip("inference_policy.json is not present")
        view = load_policy()
        if view.read_error:
            pytest.skip("inference_policy.json is unreadable: %s" % view.read_error)
        for (channel, _horizon), served in dict(view.sources).items():
            assert channel in CHANNEL_SPECS, (
                "%s is not a decided channel; the gate has no rule for it" % channel)
            assert not served.source.startswith("derived_")
            assert served.source not in ("blocked", "daylight_beta")
        assert "heat_index" in view.undecidable_channels
        assert "wind_direction" in view.undecidable_channels, (
            "wind_direction is served but the gate does not decide it; it must "
            "be listed as undecidable rather than indexed as a decided channel")
        assert result_undecidable_is_published(view)

    def test_the_committed_policy_resolves_twenty_served_cells(self):
        if not os.path.exists(DEFAULT_POLICY_PATH):
            pytest.skip("inference_policy.json is not present")
        view = load_policy()
        if view.read_error:
            pytest.skip("inference_policy.json is unreadable: %s" % view.read_error)
        mae_channels = [c for c in CHANNEL_SPECS
                        if CHANNEL_SPECS[c].metric == "mae"]
        decided = [k for k in dict(view.sources) if k[0] in mae_channels]
        assert len(decided) == 20, (
            "expected the 4 MAE channels x 5 horizons = 20 served cells, got %d"
            % len(decided))
        persistence = [k for k in decided
                       if dict(view.sources)[k].source == "persistence_fallback"]
        learned = [k for k in decided
                   if dict(view.sources)[k].source == "learned_model"]
        assert len(persistence) + len(learned) == len(decided), (
            "a served source is neither learned_model nor persistence_fallback; "
            "the gate would have to guess what it means. Seen: %s"
            % sorted({dict(view.sources)[k].source for k in decided}))
        assert learned, "the served policy selects the model nowhere at all"
        # rain is resolved from the blend weights, so the full index is 25
        assert len(dict(view.sources)) == len(decided) + len(HORIZONS)


# --------------------------------------------------------------------------- #
# BASELINE REGRESSION GUARD
# --------------------------------------------------------------------------- #

class TestBaselineRegressionGuard:
    """A promotion is only meaningful against a fixed point, so the fixed point
    is checked too."""

    def test_a_matching_rescoring_lets_the_run_proceed(self):
        cand, inc = _grid({})
        result, code = _run(_gate_inputs(cand, inc))
        assert result.verdict["integrity"]["baseline_drift"]["status"] == "PASS"
        assert code == EXIT_OK

    def test_a_baseline_that_did_not_parse_blocks(self):
        """A section present but yielding no rows is a damaged file, not an
        empty section. `capture_release_baseline.py` skips such rows with a bare
        `continue`, which reads downstream as 'the incumbent has no humidity
        data' -- which looks exactly like a policy change."""
        healthy = _synthetic_baseline()
        damaged = BaselineDocument(
            path=healthy.path, corpus_sha256=healthy.corpus_sha256,
            captured=healthy.captured, commit=healthy.commit,
            cells=dict(healthy.cells),
            parse_errors=("baseline is missing the '### humidity (MAE %)' section",),
        )
        cand, inc = _grid({})
        result, code = _run(_gate_inputs(
            cand, inc, baseline=damaged, drift=_matching_drift(damaged)))
        assert code == EXIT_BLOCKED
        report = result.verdict["integrity"]["baseline_document"]
        assert report["status"] == "FAIL"
        assert report["reason"] == "BASELINE_PARSE_ERRORS"
        assert "humidity" in report["parse_errors"][0]
        assert all(c.verdict == V_INSUFFICIENT for c in result.cells)
        assert result.verdict["summary"]["promoted"] == []

    def test_drift_blocks_the_whole_run(self):
        baseline = _synthetic_baseline()
        fresh = {k: c.model for k, c in dict(baseline.cells).items()}
        fresh[("wind_speed", 6)] = 1.30   # the incumbent moved under the record
        drift = compare_baseline_to_fresh(baseline, fresh)
        assert drift.status == DRIFT_DRIFT
        assert ("wind_speed", 6) in drift.drifted
        cand, inc = _grid({})
        result, code = _run(_gate_inputs(cand, inc, baseline=baseline, drift=drift))
        assert code == EXIT_BLOCKED
        assert result.verdict["integrity"]["baseline_drift"]["reason"] == "BASELINE_DRIFT"
        assert all(c.verdict == V_INSUFFICIENT for c in result.cells)
        assert result.verdict["summary"]["promoted"] == []

    def test_drift_detection_is_actually_sensitive(self):
        """Non-vacuity: the same comparison at the tolerance must PASS, so the
        drift test above is discriminating and not a permanent alarm."""
        baseline = _synthetic_baseline()
        fresh = {k: c.model for k, c in dict(baseline.cells).items()}
        assert compare_baseline_to_fresh(baseline, fresh).status == DRIFT_MATCH
        nudged = dict(fresh)
        nudged[("temperature", 6)] = fresh[("temperature", 6)] * 1.005
        assert compare_baseline_to_fresh(baseline, nudged).status == DRIFT_MATCH
        pushed = dict(fresh)
        pushed[("temperature", 6)] = fresh[("temperature", 6)] * 1.05
        assert compare_baseline_to_fresh(baseline, pushed).status == DRIFT_DRIFT

    def test_no_rescoring_is_not_a_pass(self):
        drift = compare_baseline_to_fresh(_synthetic_baseline(), None)
        assert drift.status == DRIFT_SKIPPED
        cand, inc = _grid({})
        result, code = _run(_gate_inputs(cand, inc, drift=drift))
        assert code == EXIT_BLOCKED
        assert result.verdict["integrity"]["baseline_drift"]["reason"] == \
            "BASELINE_RECHECK_SKIPPED"
        assert result.verdict["summary"]["promoted"] == []

    def test_a_skipped_rescoring_keeps_its_justification_in_the_record(self):
        from release_gate import DriftReport
        drift = DriftReport(status=DRIFT_SKIPPED,
                            justification="nightly run, scoring harness offline")
        cand, inc = _grid({})
        result, code = _run(_gate_inputs(cand, inc, drift=drift))
        assert code == EXIT_BLOCKED
        published = result.verdict["integrity"]["baseline_drift"]
        assert published["justification"] == "nightly run, scoring harness offline"

    def test_a_recorded_cell_the_rescoring_omits_blocks(self):
        baseline = _synthetic_baseline()
        fresh = {k: c.model for k, c in dict(baseline.cells).items()}
        del fresh[("pressure", 12)]
        drift = compare_baseline_to_fresh(baseline, fresh)
        assert drift.status == DRIFT_DRIFT
        assert ("pressure", 12) in drift.missing
        cand, inc = _grid({})
        result, code = _run(_gate_inputs(cand, inc, baseline=baseline, drift=drift))
        assert code == EXIT_BLOCKED
        assert "pressure|h12" in result.verdict["integrity"]["baseline_drift"]["not_rechecked"]

    def test_a_newly_covered_cell_does_not_block(self):
        """The recorded document was never expected to describe a cell it does
        not contain, so its appearance is not evidence of movement."""
        baseline = _synthetic_baseline()
        fresh = {k: c.model for k, c in dict(baseline.cells).items()}
        fresh[("precipitation", 6)] = 0.42
        drift = compare_baseline_to_fresh(baseline, fresh)
        assert drift.status == DRIFT_MATCH
        assert ("precipitation", 6) in drift.newly_covered
        cand, inc = _grid({})
        result, code = _run(_gate_inputs(cand, inc, baseline=baseline, drift=drift))
        assert code == EXIT_OK
        assert result.verdict["integrity"]["baseline_drift"]["newly_covered"] == \
            ["precipitation|h6"]

    def test_a_capture_timestamp_change_is_not_drift(self):
        """The historical trap. `capture_release_baseline.py --check` excludes
        exactly one line, the capture timestamp, because an earlier version
        reported drift on every run and an alarm that always fires is not an
        alarm. The same exclusion, at the record level."""
        a = _synthetic_baseline()
        b = BaselineDocument(path="<synthetic>", corpus_sha256=a.corpus_sha256,
                             captured="2099-01-01T00:00:00Z", commit=a.commit,
                             cells=dict(a.cells))
        report = compare_baseline_documents(a, b)
        assert report.status == DRIFT_MATCH
        assert "captured" in VOLATILE_BASELINE_FIELDS

    def test_a_changed_corpus_sha_is_drift(self):
        a = _synthetic_baseline()
        b = _synthetic_baseline(corpus="b" * 64)
        report = compare_baseline_documents(a, b)
        assert report.status == DRIFT_DRIFT

    def test_a_changed_scoreboard_row_is_drift(self):
        a = _synthetic_baseline()
        cells = dict(a.cells)
        cells[("wind_speed", 6)] = BaselineCell("wind_speed", 6, 1.400, 1.5, 6.6)
        b = BaselineDocument(path="<synthetic>", corpus_sha256=a.corpus_sha256,
                             captured=a.captured, commit=a.commit, cells=cells)
        assert compare_baseline_documents(a, b).status == DRIFT_DRIFT

    def test_absent_drift_report_is_its_own_failure(self):
        cand, inc = _grid({})
        inputs = _gate_inputs(cand, inc, drift=None)
        inputs = GateInputs(
            corpus_sha256=inputs.corpus_sha256, candidate=inputs.candidate,
            incumbent=inputs.incumbent, channels=inputs.channels,
            horizons=inputs.horizons, baseline=inputs.baseline,
            policy=inputs.policy, drift=None,
        )
        result, code = _run(inputs)
        assert code == EXIT_BLOCKED
        assert result.verdict["integrity"]["baseline_drift"]["reason"] == \
            "DRIFT_CHECK_NOT_SUPPLIED"
        assert result.verdict["summary"]["promoted"] == []


# --------------------------------------------------------------------------- #
# THE REAL REFERENCE DOCUMENTS
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def doc():
    """The committed incumbent record, or a skip."""
    if not os.path.exists(DEFAULT_BASELINE_PATH):
        pytest.skip("data/release_baseline.md is not present")
    parsed = load_release_baseline(DEFAULT_BASELINE_PATH)
    if parsed.parse_errors:
        pytest.skip("release_baseline.md did not parse: %s" % parsed.parse_errors)
    return parsed


class TestRealReleaseBaselineDocument:
    """release_baseline.md and inference_policy.json, as committed.

    These may skip. When they are present they must agree with each other, and
    the facts release_baseline.md is cited for must be reproducible from it --
    because the whole point of that file is to be a fixed point.
    """

    def test_the_baseline_and_the_policy_agree_on_the_corpus(self, doc):
        """A baseline captured on one corpus and a policy refitted on another
        describe different systems, and comparing against it is meaningless."""
        if not os.path.exists(DEFAULT_POLICY_PATH):
            pytest.skip("inference_policy.json is not present")
        view = load_policy()
        if view.read_error:
            pytest.skip("inference_policy.json is unreadable: %s" % view.read_error)
        assert view.weather_telemetry_sha256 == doc.corpus_sha256, (
            "inference_policy.json was refitted against %s but the release "
            "baseline was captured on %s"
            % (view.weather_telemetry_sha256, doc.corpus_sha256)
        )

    def test_every_mae_cell_parses_and_its_skill_recomputes(self, doc):
        for channel in ("temperature", "humidity", "pressure", "wind_speed"):
            for horizon in HORIZONS:
                cell = doc.value and dict(doc.cells).get((channel, horizon))
                assert cell is not None, "%s|h%d is missing from the baseline" % (
                    channel, horizon)
                recomputed = (cell.persistence - cell.model) / cell.persistence * 100.0
                assert cell.skill_pct == pytest.approx(recomputed, abs=0.15), (
                    "%s|h%d records %+.1f%% but model/persistence give %+.1f%%; the "
                    "published table is not derivable from itself"
                    % (channel, horizon, cell.skill_pct, recomputed))

    def test_exactly_four_of_twenty_cells_carry_real_skill(self, doc):
        """The claim the baseline is cited for. If this fails, the baseline was
        legitimately re-captured and the number in the task description and in
        every report that quotes it needs updating -- do not silently relax it.
        """
        mae_channels = ("temperature", "humidity", "pressure", "wind_speed")
        cells = [(c, h) for c in mae_channels for h in HORIZONS]
        real = ["%s|h%d" % (c, h) for c, h in cells
                if dict(doc.cells)[(c, h)].skill_pct >= 1.0]
        assert sorted(real) == ["temperature|h12", "temperature|h6",
                                "wind_speed|h12", "wind_speed|h6"], (
            "the served model no longer beats persistence in exactly 4 of 20 "
            "cells; it now does so in %s" % sorted(real))
        assert len(cells) == 20
        assert len(cells) - len(real) == 16, (
            "the 16 cells the baseline describes as returning persistence are "
            "not all still below the 1%% line")

    def test_the_rain_brier_section_parses(self, doc):
        for horizon in HORIZONS:
            cell = dict(doc.cells).get(("rain_occurrence", horizon))
            assert cell is not None, "rain_occurrence|h%d is missing" % horizon
            assert 0.0 < cell.model < 1.0, "a Brier score outside (0, 1)"
            assert cell.model < cell.persistence, (
                "release_baseline.md states the rain model beats persistence at "
                "every horizon; +%dh has model %.4f >= persistence %.4f"
                % (horizon, cell.model, cell.persistence))

    def test_the_capture_timestamp_is_parsed_but_never_compared(self, doc):
        assert doc.captured, "the capture timestamp was not parsed"
        assert "captured" in VOLATILE_BASELINE_FIELDS
        again = load_release_baseline(DEFAULT_BASELINE_PATH)
        assert compare_baseline_documents(doc, again).status == DRIFT_MATCH


# --------------------------------------------------------------------------- #
# EXIT CODES AND THE VERDICT DOCUMENT
# --------------------------------------------------------------------------- #

class TestExitCodesAndVerdictDocument:
    """The gate's contract with whatever runs it."""

    def _code(self, inputs):
        result, code = _run(inputs)
        assert code == result.exit_code
        assert code == result.verdict["exit_code"]
        return result, code

    def test_every_recommendation_maps_to_the_documented_exit_code(self):
        # PROMOTE_ALL -> 0
        cand, inc = _grid({})
        result, code = self._code(_gate_inputs(cand, inc))
        assert result.recommendation == R_PROMOTE_ALL and code == EXIT_OK
        # SPLIT -> 0
        plan = {(c, h): (WIN if c == "wind_speed" else (-0.02, 0.002))
                for c in DEFAULT_CHANNELS for h in HORIZONS}
        cand, inc = _grid(plan)
        result, code = self._code(_gate_inputs(cand, inc))
        assert result.recommendation == R_SPLIT and code == EXIT_OK
        # RETAIN_ALL -> 2
        plan = {(c, h): (-0.02, 0.02) for c in DEFAULT_CHANNELS for h in HORIZONS}
        cand, inc = _grid(plan)
        result, code = self._code(_gate_inputs(cand, inc))
        assert result.recommendation == R_RETAIN_ALL and code == EXIT_RETAIN
        # BLOCKED -> 1
        plan = {(c, h): ("missing",) for c in DEFAULT_CHANNELS for h in HORIZONS}
        cand, inc = _grid(plan)
        result, code = self._code(_gate_inputs(cand, inc))
        assert result.recommendation == R_BLOCKED and code == EXIT_BLOCKED

    def test_only_zero_means_the_gate_permitted_the_promotion(self):
        """`if rc == 0: deploy` has to be safe. Every other outcome, including
        a legitimate 'no', is non-zero."""
        for plan, expected_zero in (
                ({}, True),
                ({(c, h): (WIN if c == "wind_speed" else (-0.02, 0.002))
                  for c in DEFAULT_CHANNELS for h in HORIZONS}, True),
                ({(c, h): (-0.02, 0.02) for c in DEFAULT_CHANNELS for h in HORIZONS},
                 False),
                ({(c, h): ("missing",) for c in DEFAULT_CHANNELS for h in HORIZONS},
                 False)):
            cand, inc = _grid(plan)
            _result, code = _run(_gate_inputs(cand, inc))
            assert (code == EXIT_OK) is expected_zero

    def test_the_verdict_document_is_json_serialisable_and_round_trips(self, tmp_path):
        cand, inc = _grid({})
        result, _code = _run(_gate_inputs(cand, inc))
        path = write_verdict(result, os.path.join(str(tmp_path), "nested", "v.json"))
        with open(path, encoding="utf-8") as handle:
            reloaded = json.load(handle)
        assert reloaded["recommendation"] == result.recommendation
        assert reloaded["cells"].keys() == result.verdict["cells"].keys()
        assert reloaded["rule"] == result.verdict["rule"]

    def test_the_published_rule_matches_the_constants_that_ran(self):
        cand, inc = _grid({})
        result, _code = _run(_gate_inputs(cand, inc))
        rule = result.verdict["rule"]
        assert rule["sigma_multiplier"] == SIGMA_MULTIPLIER
        assert rule["min_paired_rows"] == MIN_PAIRED_ROWS
        assert rule["min_paired_rows_per_half"] == MIN_HALF_PAIRED_ROWS
        assert rule["decision_split"] == SPLIT_DECISION
        assert "strictly greater than" in rule["significance_comparison"]

    def test_the_rule_constants_are_the_documented_ones(self):
        """Pinned by value, not by self-reference.

        A silently relaxed threshold is the quietest way to turn this gate into
        an always-passes formality: every other test would still be consistent,
        because they all read the same constant. These literals are the
        specification.
        """
        assert SIGMA_MULTIPLIER == 2.0
        assert MIN_PAIRED_ROWS == 30
        assert MIN_HALF_PAIRED_ROWS == 10
        assert DRIFT_REL_TOLERANCE == 0.01
        assert HORIZONS == (1, 3, 6, 12, 24)
        assert EXIT_OK == 0 and EXIT_BLOCKED == 1
        assert EXIT_RETAIN == 2 and EXIT_UNUSABLE == 3
        assert SPLIT_DECISION == "validation" and SPLIT_REPORTING == "test"
        # every declared channel is LOWER-is-better; a channel that is higher-is
        # better would need its sign flipped in the rule and must not appear here
        for channel, spec in CHANNEL_SPECS.items():
            assert spec.better == "lower_is_better", channel
        assert CHANNEL_SPECS["rain_occurrence"].row_error == "squared"
        for channel in ("temperature", "humidity", "pressure", "wind_speed"):
            assert CHANNEL_SPECS[channel].row_error == "absolute"
            assert CHANNEL_SPECS[channel].metric == "mae"

    def test_the_module_docstring_describes_the_rule_that_ran(self):
        """The docstring is what the next person reads. If it and the code
        disagree, the code is what runs and the docstring is what is believed."""
        import release_gate
        text = release_gate.__doc__ or ""
        for fragment in (
                "mean_difference = mean( incumbent_row_error - candidate_row_error )",
                "mean_difference > SIGMA_MULTIPLIER * stderr",
                "STRICTLY greater",
                "BOTH",
                "REJECT",
                "INSUFFICIENT_EVIDENCE",
                "0  gate ran, decision complete",
                "1  BLOCKED",
                "2  RETAIN_ALL",
                "3  UNUSABLE_INPUT",
                "(p - y)^2",
                "Step 1 -- DIRECTION",
                "Step 2 -- HALF-SPLIT STABILITY",
                "Step 3 -- SIGNIFICANCE",
                "Only Steps 2 and 3 together produce PROMOTE",
        ):
            assert fragment in text, (
                "the module docstring no longer states %r; the rule it describes "
                "and the rule the code runs have drifted apart" % fragment)
        # the step order in the docstring must be the order decide_cell applies
        import release_gate as rg
        source = open(rg.__file__, encoding="utf-8").read()
        body = source.split("def decide_cell(", 1)[1].split("\ndef ", 1)[0]
        order = [body.index("if full.mean_difference < 0.0:"),
                 body.index("reason, extra = _half_reason(early, late)"),
                 body.index("if full.mean_difference <= required:")]
        assert order == sorted(order), (
            "decide_cell applies the steps in a different order from the one the "
            "docstring states: direction@%d, halves@%d, significance@%d"
            % tuple(order))

    def test_the_summary_names_a_blocked_run_as_not_a_pass(self, tmp_path, capsys):
        from release_gate import summarize
        plan = {(c, h): ("missing",) for c in DEFAULT_CHANNELS for h in HORIZONS}
        cand, inc = _grid(plan)
        result, code = _run(_gate_inputs(cand, inc))
        text = summarize(result)
        assert "BLOCKED" in text
        assert "0 promote" in text, "a blocked run reported a promotion: %s" % text
        assert "promotion was NOT evaluated" in text
        assert "not a pass" in text.lower()
        assert "INSUFFICIENT_EVIDENCE" in text

    def test_the_verdict_states_whether_a_decision_was_computed(self):
        cand, inc = _grid({})
        result, _code = _run(_gate_inputs(cand, inc))
        assert result.verdict["promotion_decision_computed"] is True
        plan = {(c, h): ("missing",) for c in DEFAULT_CHANNELS for h in HORIZONS}
        cand, inc = _grid(plan)
        result, _code = _run(_gate_inputs(cand, inc))
        assert result.verdict["promotion_decision_computed"] is False

    def test_main_writes_the_verdict_and_returns_the_exit_code(self, tmp_path):
        if not os.path.exists(DEFAULT_BASELINE_PATH):
            pytest.skip("data/release_baseline.md is not present")
        parsed = load_release_baseline(DEFAULT_BASELINE_PATH)
        if parsed.parse_errors:
            pytest.skip("release_baseline.md did not parse: %s" % parsed.parse_errors)
        cand_map, inc_map = _grid({})

        def _as_json(evidence):
            return {
                "%s|h%d" % (k[0], k[1]): {
                    "validation": {"value": v.validation.value, "n_rows": 120},
                    "test": {"value": v.validation.value, "n_rows": 120,
                             "split": "test"},
                    "paired": {
                        span: {"n_pairs": b.n_pairs,
                               "mean_difference": b.mean_difference,
                               "stderr": b.stderr, "split": "validation",
                               "source": "fixture"}
                        for span, b in v.paired.items()
                    },
                } for k, v in evidence.items()
            }

        inputs_doc = {
            "corpus_sha256": parsed.corpus_sha256,
            "challenger_name": "nightly-2026-09-30",
            "incumbent_name": "served-bundles",
            "baseline_path": DEFAULT_BASELINE_PATH,
            "policy_path": DEFAULT_POLICY_PATH,
            "fresh_incumbent_metrics": {
                "%s|%d" % (k[0], k[1]): c.model for k, c in dict(parsed.cells).items()
            },
            "challenger": _as_json(cand_map),
            "incumbent": {
                key: {k: v for k, v in cell.items() if k != "test"}
                for key, cell in _as_json(inc_map).items()
            },
        }
        inputs_path = os.path.join(str(tmp_path), "inputs.json")
        with open(inputs_path, "w", encoding="utf-8") as handle:
            json.dump(inputs_doc, handle)

        out = os.path.join(str(tmp_path), "verdict.json")
        code = main(["--inputs", inputs_path, "--out", out, "--quiet"])
        assert code == EXIT_OK
        with open(out, encoding="utf-8") as handle:
            verdict = json.load(handle)
        assert verdict["recommendation"] == R_PROMOTE_ALL
        assert verdict["gate"].endswith("release_gate.py")
        assert verdict["decision_inputs_used"]["challenger"] == "nightly-2026-09-30"
        assert verdict["integrity"]["baseline_drift"]["status"] == "PASS"
        assert verdict["integrity"]["corpus"]["status"] == "PASS"
        # the real policy really does resolve every cell, so nothing blocked
        assert verdict["integrity"]["policy"]["status"] == "PASS"

    def test_main_refuses_a_test_labelled_decision_input_and_writes_nothing(self,
                                                                           tmp_path):
        inputs_path = os.path.join(str(tmp_path), "bad.json")
        with open(inputs_path, "w", encoding="utf-8") as handle:
            json.dump({
                "corpus_sha256": BASELINE_CORPUS,
                "challenger": {
                    "temperature|h6": {
                        "validation": {"value": 1.4, "n_rows": 120},
                        "paired": {
                            "full": {"n_pairs": 120, "mean_difference": 0.02,
                                     "stderr": 0.002, "split": "test"},
                            "early": {"n_pairs": 60, "mean_difference": 0.01,
                                      "stderr": 0.002, "split": "test"},
                            "late": {"n_pairs": 60, "mean_difference": 0.01,
                                     "stderr": 0.002, "split": "test"},
                        },
                    },
                },
            }, handle)
        out = os.path.join(str(tmp_path), "verdict.json")
        code = main(["--inputs", inputs_path, "--out", out, "--quiet"])
        assert code == EXIT_UNUSABLE
        assert not os.path.exists(out), (
            "a run that could not decide still wrote a verdict file; a stale "
            "verdict on disk is indistinguishable from a fresh one")
