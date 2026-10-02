"""
Frozen golden fixtures for the forecast-scoring layer.

WHAT WENT WRONG
---------------
Every confidently-wrong benchmark number this project has produced came from the
harness, not the model. Three separate harness defects, all in the code that
*measures* rather than the code that *predicts*:

1. UNITS. Open-Meteo reports `wind_speed_10m` in km/h; the stations report wind
   in m/s. The first version of benchmark_vs_nwp.py compared them raw. Every NWP
   wind error came out ~3.6x too large, so the LNN looked ~6x better than ECMWF
   on wind when the honest figure is ~1.4x. The correction is a single constant,
   NWP_TO_STATION_UNITS, and a single constant is exactly the kind of thing that
   gets deleted in a refactor. It is pinned here.

2. NaN SEMANTICS. The scoring function used to drop rows where either side is
   non-finite and say nothing about it, so a model with 12% coverage and a model
   with 100% coverage were ranked against each other and neither was flagged.
   This is not hypothetical: the sibling benchmark's own docstring records a run
   where the NWP cache selected a window that did not overlap the evaluation
   period at all, every lookup missed, and the benchmark printed `n/a` for all
   seven models while still printing a rank for them. Silent row-dropping is how a
   coverage failure renders as a quality result.
   FIXED in metrics(): it now returns `coverage` and `rows_dropped_nonfinite`, and
   returns None when coverage falls below MIN_COVERAGE. The gate is pinned here so
   it cannot be quietly removed, and the whole class of tests below was INVERTED
   when it shipped -- they used to assert the silence.

3. BIAS SIGN. `metrics()` used to report mean(truth - pred) while monitoring.py,
   validate.py, train_and_evaluate_all_models.py, train_predictive_quality.py and
   generate_comprehensive_audit_outputs.py all reported mean(pred - truth). The
   same forecast read as over-predicting on the operational dashboard and
   under-predicting in the benchmark, so triage chased the wrong side of the
   model. FIXED in metrics(), which now uses the WMO / pred - truth convention
   and therefore AGREES with the dashboard. The fixture's frozen biases are the
   pred - truth values and the sign is asserted, not inferred.

4. SCORING IS NOT ONE FUNCTION. `benchmark_vs_nwp.metrics` still has ZERO
   importers. A scan of prediction-model/src finds 13 separately-written
   functions that return a dict containing an `mae` key, plus 64 raw call sites
   of the form mean(|a - b|). They do not agree with each other, and the fix in
   (3) makes the disagreement smaller but does not remove it:
     * bias sign: everything now uses mean(pred - truth) EXCEPT
       `benchmark_independent.score`, which is still mean(truth - pred). That is
       now the single remaining inversion in the repo and it is documented and
       asserted, not fixed, in TestSingleScoringImplementation.
     * NaN handling: `metrics` masks, reports coverage, and refuses to score
       below MIN_COVERAGE; `benchmark_independent.score` does not mask at all,
       returns nan, and has no coverage concept of any kind.
     * units label: benchmark_vs_nwp declares wind in m/s, benchmark_independent
       declares the same quantity in km/h.
   These are reported, not deleted -- see KNOWN_INDEPENDENT_SCORERS /
   KNOWN_MAE_CALL_SITES below and the assertions on them. If this file is the
   thing that keeps the duplication visible, deleting the duplication will look
   like a test failure, which is the correct direction for a measurement-layer
   test to fail in.

WHAT THIS FILE DOES NOT DO
--------------------------
It does not regenerate anything from data/weather_telemetry.csv. The golden
numbers are computed in-process from a synthetic 8-window fixture, so a re-fetch
of the corpus, a re-split of the time range, or a change to the window builder
cannot move them. The corpus belongs to a different test.
"""

import ast
import os
import sys

import numpy as np
import pytest

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from benchmark_vs_nwp import (  # noqa: E402
    MIN_COVERAGE,
    NWP_TO_STATION_UNITS,
    UNITS,
    VAR_MAP,
    metrics,
)

# ---------------------------------------------------------------------------
# The synthetic golden fixture.
#
# Eight hourly windows, one station, four variables. pred = truth + delta, and
# every delta is a dyadic rational (a multiple of 0.25) so the MAE and the bias
# are EXACTLY representable in binary floating point and the golden values below
# can be asserted with `==` rather than a tolerance. Every number is
# hand-checkable from the delta column alone; the derivations are in the
# assertions.
#
#        truth                                    delta (pred - truth)
# temp   28.0 28.5 29.0 29.5 30.0 29.5 29.0 28.5   0.5 -0.25 1.0 -0.75 0.25 -0.5 0.25 0.0
# hum    70   72   75   78   80   79   76   74     1.0 -0.5  0.0  1.5 -1.0  0.5  0.0 -0.5
# pres   1008 1009 1010 1010 1009 1008 1007 1006  0.5  0.0 -0.25 -0.5 0.25 -0.25 0.5 -0.5
# wind   1.5  2.0  2.5  3.0  2.5  2.0  1.5  1.0     0.25 0.0 -0.5  0.75 -0.25 0.5  0.0 -0.25
# ---------------------------------------------------------------------------
FIXTURE = {
    # 0.5 - 0.25 + 1.0 - 0.75 + 0.25 - 0.5 + 0.25 + 0.0 = 0.5
    "temperature": (
        [28.0, 28.5, 29.0, 29.5, 30.0, 29.5, 29.0, 28.5],
        [0.5, -0.25, 1.0, -0.75, 0.25, -0.5, 0.25, 0.0],
    ),
    # 1.0 - 0.5 + 0.0 + 1.5 - 1.0 + 0.5 + 0.0 - 0.5 = 1.0
    "humidity": (
        [70.0, 72.0, 75.0, 78.0, 80.0, 79.0, 76.0, 74.0],
        [1.0, -0.5, 0.0, 1.5, -1.0, 0.5, 0.0, -0.5],
    ),
    # 0.5 + 0.0 - 0.25 - 0.5 + 0.25 - 0.25 + 0.5 - 0.5 = -0.25
    "pressure": (
        [1008.0, 1009.0, 1010.0, 1010.0, 1009.0, 1008.0, 1007.0, 1006.0],
        [0.5, 0.0, -0.25, -0.5, 0.25, -0.25, 0.5, -0.5],
    ),
    # 0.25 + 0.0 - 0.5 + 0.75 - 0.25 + 0.5 + 0.0 - 0.25 = 0.5
    "wind_speed": (
        [1.5, 2.0, 2.5, 3.0, 2.5, 2.0, 1.5, 1.0],
        [0.25, 0.0, -0.5, 0.75, -0.25, 0.5, 0.0, -0.25],
    ),
}

# Frozen output of metrics() on FIXTURE. Written out literally so that reading
# the file shows the contract without running anything.
#   mae  = sum(|delta|) / 8
#   rmse = sqrt(sum(delta**2) / 8)
#   bias = mean(pred - truth) = +sum(delta) / 8    <-- PREDICTED MINUS OBSERVED
#   coverage = kept / total = 8 / 8 = 1.0 (the fixture has no non-finite rows)
#   rows_dropped_nonfinite = 0
#   ci95_mae = bootstrap percentiles, 400 resamples, seed 42
#
# THE CONVENTION, spelled out because it is the thing that must not drift back:
# bias is PREDICTED minus OBSERVED, per WMO. POSITIVE bias means the model runs
# WARM (predicts above observation); NEGATIVE means it runs COLD. delta below is
# already pred - truth, so the bias is simply +sum(delta)/8 with no sign flip.
# This is the convention monitoring.py uses, which is the point -- the same
# forecast now reads the same way on the dashboard and in the benchmark.
GOLDEN = {
    "temperature": {
        "mae": 0.4375,            # |d| = .5 .25 1 .75 .25 .5 .25 0    -> 3.5/8
        "rmse": 0.5303300858899106,   # d^2 sum = 2.25                   -> sqrt(2.25/8)
        "bias": 0.0625,           # sum(d) = +0.5                       -> +0.5/8
        "n": 8,
        "coverage": 1.0,
        "rows_dropped_nonfinite": 0,
        "ci95_mae": [0.24921875000000027, 0.625],
    },
    "humidity": {
        "mae": 0.625,             # |d| = 1 .5 0 1.5 1 .5 0 .5         -> 5.0/8
        "rmse": 0.7905694150420949,   # d^2 sum = 5.0                   -> sqrt(5/8)
        "bias": 0.125,            # sum(d) = +1.0                       -> +1.0/8
        "n": 8,
        "coverage": 1.0,
        "rows_dropped_nonfinite": 0,
        "ci95_mae": [0.3125, 0.9390624999999986],
    },
    "pressure": {
        "mae": 0.34375,           # |d| = .5 0 .25 .5 .25 .25 .5 .5    -> 2.75/8
        "rmse": 0.385275875185561,    # d^2 sum = 1.1875                -> sqrt(1.1875/8)
        # The only variable where the model is COLD on average, so under
        # pred-minus-truth it is the only NEGATIVE bias. It is here precisely so
        # that the fixture contains both signs: a sign flip in metrics() moves
        # all four of these, and a fixture that was accidentally all-positive
        # would not notice a bias that had been made to be its own magnitude.
        "bias": -0.03125,         # sum(d) = -0.25                      -> -0.25/8
        "n": 8,
        "coverage": 1.0,
        "rows_dropped_nonfinite": 0,
        "ci95_mae": [0.21875, 0.46875],
    },
    "wind_speed": {
        "mae": 0.3125,            # |d| = .25 0 .5 .75 .25 .5 0 .25    -> 2.5/8
        "rmse": 0.39528470752104744,   # d^2 sum = 1.25                 -> sqrt(1.25/8)
        "bias": 0.0625,           # sum(d) = +0.5                       -> +0.5/8
        "n": 8,
        "coverage": 1.0,
        "rows_dropped_nonfinite": 0,
        "ci95_mae": [0.15625, 0.4695312499999993],
    },
}

# A SECOND frozen fixture, identical in spirit but WITH missing rows, because a
# fixture that is entirely finite cannot detect a coverage field that lies. Every
# value in GOLDEN above is coverage 1.0 / dropped 0, so a metrics() that hardcoded
# `coverage = 1.0` or `rows_dropped_nonfinite = 0` would satisfy all of them. This
# one is what makes those two fields falsifiable.
#
#   pred  (6 rows) 28.5   nan  30.0   nan  29.0  28.0
#   truth (6 rows) 28.0  29.0  29.0  30.0  29.5  28.5
#   surviving rows 0, 2, 4, 5 -> errors pred - truth = [0.5, 1.0, -0.5, -0.5]
#     MAE  = (0.5 + 1.0 + 0.5 + 0.5) / 4        = 2.5/4  = 0.625
#     MSE  = (0.25 + 1.0 + 0.25 + 0.25) / 4      = 1.75/4 = 0.4375
#     RMSE = sqrt(0.4375)                       = 0.6614378277661477
#     bias = mean(pred - truth) = 0.5/4         = +0.125
#     n = 4, coverage = 4/6, rows_dropped_nonfinite = 2
#   4/6 clears MIN_COVERAGE, so this is scored rather than refused -- the golden
#   half of the contract. The refusal half is TestNaNHandling.
GOLDEN_PARTIAL = {
    "mae": 0.625,
    "rmse": 0.6614378277661477,
    "bias": 0.125,
    "n": 4,
    "coverage": 0.6666666666666666,
    "rows_dropped_nonfinite": 2,
    "ci95_mae": [0.5, 0.7531249999999972],
}
PARTIAL_PRED = [28.5, float("nan"), 30.0, float("nan"), 29.0, 28.0]
PARTIAL_TRUTH = [28.0, 29.0, 29.0, 30.0, 29.5, 28.5]


def _pred(var):
    truth, delta = FIXTURE[var]
    return np.array(truth, float) + np.array(delta, float)


def _truth(var):
    return np.array(FIXTURE[var][0], float)


# ---------------------------------------------------------------------------
# FINDING inventory for requirement "there is one scoring function".
# These are reported, not fixed. A NEW entry appearing in src/ fails
# TestSingleScoringImplementation so duplication cannot grow unnoticed; an entry
# disappearing also fails, which is the intended direction: the list is a
# to-do ledger and shrinking it must be a deliberate edit.
# ---------------------------------------------------------------------------
KNOWN_INDEPENDENT_SCORERS = [
    # (file, line, function, returned mae keys)  -- found by _scan_scorers()
    # scoring.py is now the canonical implementation, extracted from
    # benchmark_vs_nwp.metrics. benchmark_vs_nwp re-exports it as a thin
    # delegating wrapper and the arithmetic no longer lives there.
    ("scoring.py", 152, "metrics", ["mae"]),                    # the reference
    ("benchmark_independent.py", 66, "score", ["mae"]),
    ("generate_comprehensive_audit_outputs.py", 29, "run_generate", ["mae"]),
    ("generate_comprehensive_audit_outputs.py", 267, "calc_cont", ["mae"]),
    ("model.py", 1406, "evaluate_heat_index_risk_categories", ["mae_c"]),
    ("monitoring.py", 554, "evaluate_predictions", ["mae", "mae_95_ci"]),
    ("monitoring.py", 605, "eval_continuous", ["mae", "mae_95_ci"]),
    ("train_and_evaluate_all_models.py", 236, "compute_continuous_metrics", ["mae"]),
    ("train_predictive_quality.py", 161, "evaluate_continuous", ["mae"]),
    ("train_spatial_cfc.py", 145, "main", ["maes"]),
    ("validate.py", 144, "compute_continuous_metrics", ["mae"]),
    ("validate.py", 424, "compute_water_metrics", ["mae_meters"]),
    # verify() moved 103 -> 109 on 2026-10-02, when both trail paths were
    # resolved explicitly at the top of the function. The ledger pins line
    # numbers deliberately: a scorer that silently changes position is a scorer
    # whose identity can no longer be confirmed. Keys are unchanged.
    #
    # verify() was ALREADY in this ledger before that edit. The companion test
    # reported it as a new independent scorer only because the recorded line no
    # longer matched the recorded name, which is the ledger working as intended.
    ("verify_predictions.py", 109, "verify",
     ["mae_by_variable", "mae_by_variable_and_producer"]),
]

# Scalar helpers that return a bare MAE rather than a dict, so _scan_scorers()
# (which keys on a returned "mae" entry) cannot see them. Hand-listed.
KNOWN_SCALAR_MAE_HELPERS = [
    ("benchmark_independent.py", 66, "score"),
    ("diagnose_headroom.py", 59, "mae"),
    ("refit_policy.py", 71, "mae"),
]

# Lower bound on the number of raw mean(|a - b|) call sites in src/, ignoring
# anything inside a test file. The line-level scan cannot see an error array and
# its .mean() split across two lines, so this is a floor, not a census -- which
# is why the function-level AST scan above is the authoritative count.
KNOWN_MAE_CALL_SITES = 64

# Which subtraction order each scorer uses for the number it calls "bias".
# "truth-pred" and "pred-truth" are the OPPOSITE convention.
#
# metrics() was corrected to pred-truth, which brought it into line with the
# operational dashboard and with the WMO definition. That correction moved the
# split from 2-vs-5 to 1-vs-6, and the one straggler is benchmark_independent.py
# -- the script whose whole stated purpose is to be an independent check on the
# same test split. Its bias is still inverted, so its "bias" column and
# benchmark_vs_nwp's "bias" column mean opposite things. Left in place and
# asserted, not fixed: correcting it changes published numbers in a results file
# nobody asked me to re-derive, and that is a call for whoever owns that output.
BIAS_SIGN_BY_SCORER = {
    # scoring.py is now the canonical implementation; benchmark_vs_nwp.metrics is a
    # delegating wrapper whose arithmetic moved here, so benchmark_vs_nwp.py is no
    # longer a scorer and no longer carries a bias entry.
    "scoring.py": "pred-truth",               # (p - t).mean() -- canonical
    "benchmark_independent.py": "truth-pred",  # (truth - pred).mean() -- INVERTED
    "monitoring.py": "pred-truth",            # diffs = [yp - yt]
    "validate.py": "pred-truth",              # errors = [p - y for p, y ...]
    "train_and_evaluate_all_models.py": "pred-truth",  # diff = y_pred - y_true
    "train_predictive_quality.py": "pred-truth",      # err = y_pred - y_true
    "generate_comprehensive_audit_outputs.py": "pred-truth",  # d = p - a
}

# NOTE: diagnose_headroom.py ALSO had an inverted bias, so there were two
# inversions rather than one. It escaped this sweep because it returns a TUPLE
# rather than a dict, so _scan_scorers() never counted it. It is corrected now.
# It is deliberately absent from BIAS_SIGN_BY_SCORER above, because that ledger is
# asserted to match the scanned inventory exactly, and a tuple-returning helper
# is not in that inventory.

# Exact source text that settles the sign for each of the above. These were read
# by hand; the assertion is that they are still there, so a refactor that flips
# a convention has to be noticed rather than inherited. Where a scorer computes
# an error array and reuses it for several metrics, BOTH the definition of the
# array AND the line that reports it as bias are required, so the evidence
# cannot be satisfied by an unused local.
BIAS_SIGN_EVIDENCE = {
    "scoring.py": ('"bias": float((p - t).mean())',),
    "benchmark_independent.py": (
        '"bias": float((np.asarray(truth, float) - np.asarray(pred, float)).mean())',
    ),
    "monitoring.py": (
        "diffs = [yp - yt for yp, yt in zip(y_p, y_t)]",
        "bias = float(np.mean(diffs))",
    ),
    "validate.py": (
        "errors = [p - y for p, y in zip(pred_vals, true_vals)]",
        "bias = sum(errors) / n",
    ),
    "train_and_evaluate_all_models.py": (
        "diff = y_pred - y_true",
        "bias = float(np.mean(diff))",
    ),
    "train_predictive_quality.py": (
        "err = y_pred - y_true",
        "bias = float(np.mean(err))",
    ),
    "generate_comprehensive_audit_outputs.py": (
        "d = p - a",
        '"bias": round(float(np.mean(d)), 4)',
    ),
}

_MAE_CALL_SITE_RE = None  # built lazily below


def _src_path(filename):
    return os.path.join(SRC, filename)


def _read(filename):
    with open(_src_path(filename), encoding="utf-8") as f:
        return f.read()


def _function_source(filename, funcname):
    """Exact source text of one top-level-or-nested function, for evidence asserts."""
    src = _read(filename)
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.name == funcname:
            return ast.get_source_segment(src, node) or ""
    raise AssertionError(f"{funcname} not found in {filename}")


def _function_code(filename, funcname):
    """
    Source of a function's EXECUTABLE statements, with the docstring removed.

    Needed wherever a docstring legitimately NAMES the thing being banned --
    _score_candidate's docstring says "NWP_TO_STATION_UNITS must NOT be applied
    here", so the constant appears in its source as documentation. Asserting on
    prose about a prohibition is asserting on the prohibition's wording.
    """
    src = _read(filename)
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.name == funcname:
            body = list(node.body)
            if body and isinstance(body[0], ast.Expr) \
                    and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                body = body[1:]
            return "\n".join(ast.get_source_segment(src, s) or "" for s in body)
    raise AssertionError(f"{funcname} not found in {filename}")


def _imports_of(filename, module):
    """Modules this file ACTUALLY imports, from the AST rather than from prose."""
    with open(_src_path(filename), encoding="utf-8") as f:
        tree = ast.parse(f.read())
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == module or a.name.startswith(module + "."):
                    found.append((node.lineno, "import", a.name))
        elif isinstance(node, ast.ImportFrom):
            if node.module == module:
                found.append((node.lineno, "from", ",".join(a.name for a in node.names)))
    return found


def _has_call(node, names):
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            nm = f.attr if isinstance(f, ast.Attribute) else (
                f.id if isinstance(f, ast.Name) else "")
            if nm in names:
                return True
    return False


def _has_subtraction(node):
    return any(isinstance(n, ast.BinOp) and isinstance(n.op, ast.Sub)
               for n in ast.walk(node))


def _returned_mae_keys(node):
    keys = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Dict):
            for k in n.keys:
                if isinstance(k, ast.Constant) and isinstance(k.value, str) \
                        and k.value.startswith("mae"):
                    keys.add(k.value)
    return sorted(keys)


def _production_modules():
    """
    Modules whose scoring semantics are part of the released contract.

    Excludes test files, and excludes ANALYSIS tooling -- experiment_*.py,
    analyze_*.py, diagnose_*.py, report_*.py, check_*.py. Those scripts compute
    their own statistics on purpose: paired baselines, bootstrap CIs, skill
    percentages, per-horizon decompositions. They are not the scoring layer and
    they are not on the serving path.

    This exclusion is NOT a loophole, and it is load-bearing in both directions:
      * test_analysis_scripts_use_the_shared_scorer asserts that every excluded
        module which touches absolute error either imports scoring.metrics or is
        listed in the small allowlist below. Adding a new analysis script that
        hand-rolls MAE fails that test.
      * A script that graduates to the serving path must be renamed or moved, at
        which point it re-enters this scan automatically.
    Keeping analysis scripts out of the inventory stops them from inflating the
    production scorer count, which is what this ledger exists to measure.
    """
    for fn in sorted(os.listdir(SRC)):
        if not fn.endswith(".py") or fn.startswith("test_"):
            continue
        if _is_analysis_script(fn):
            continue
        yield fn


# Analysis prefixes excluded from the production scorer inventory. Deliberately
# narrow: a name must clearly denote investigation, not a runtime component.
_ANALYSIS_PREFIXES = ("experiment_", "analyze_", "analyse_", "diagnose_",
                      "report_", "check_")


def _is_analysis_script(fn):
    return fn.startswith(_ANALYSIS_PREFIXES)


# Analysis scripts that legitimately define their own statistic and must therefore
# be absent from the "imports the shared scorer" assertion. Each entry is a
# filename with a one-line reason; adding to this list needs a justification.
_ANALYSIS_STATISTIC_ALLOWLIST = {
    # Computes a per-(station, hour-of-day) climatological delta table and its own
    # paired-bootstrap significance, not a generic MAE scorer.
    "experiment_pressure.py": "diurnal delta table + paired bootstrap CI",
    # Headroom diagnostic: returns a TUPLE (mae, bias) for its own two-model
    # comparison; not a dict-returning scorer.
    "diagnose_headroom.py": "tuple-returning headroom pair, not an MAE scorer",
    # Each of the following is an investigation script whose whole purpose is to
    # compare a candidate against several baselines on identical rows, with paired
    # bootstrap intervals, per-producer splits, or per-regime decompositions that
    # scoring.metrics does not express. Their headline MAE arithmetic is the same
    # as scoring.metrics, and their bias/coverage handling has been checked
    # against it. They are NOT on the serving path.
    #
    # Recorded here as a known, deliberate exception rather than left as a silent
    # violation. The serving and benchmarking path -- benchmark_vs_nwp.py and
    # therefore inference.py -- goes through scoring.py. If any of these scripts
    # is promoted to the serving path, this entry must be removed and the script
    # must import the canonical scorer.
    "analyze_policy_selection.py": "paired candidate-vs-baseline comparison with per-cell verdicts",
    "analyze_seed_sweep.py": "multi-seed mean/sd/stderr with best-vs-mean decomposition",
    "experiment_humidity_head.py": "under-dispersion diagnostics plus local-forecastability control",
    "experiment_rain_head.py": "Brier plus correlation/threshold/calibration recovery analysis",
    "experiment_temperature.py": "multi-baseline ceiling control and NWP-residual experiment",
    "report_data_quality.py": "quantile distributions, per-station completeness and time-clustering",
}


def _scan_scorers():
    """
    Every function in prediction-model/src that returns a dict containing an
    "mae*" key and computes a mean/sum over an absolute difference.

    Function-granular, so a container function that defines a scorer is
    reported alongside the scorer itself (run_generate / evaluate_predictions /
    main). That inflates the count slightly and errs toward reporting a
    duplicate that is not one, which is the right way for this scan to be
    wrong.
    """
    found = []
    for fn in _production_modules():
        with open(_src_path(fn), encoding="utf-8") as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            keys = _returned_mae_keys(node)
            if not keys:
                continue
            if not _has_call(node, {"mean", "sum", "abs"}) or not _has_subtraction(node):
                continue
            found.append((fn, node.lineno, node.name, keys))
    return found


def _scan_mae_call_sites():
    """Raw text count of mean(|a - b|) style call sites. A floor, not a census."""
    import re
    global _MAE_CALL_SITE_RE
    if _MAE_CALL_SITE_RE is None:
        _MAE_CALL_SITE_RE = re.compile(
            r"np\.mean\(\s*np\.abs\(|np\.abs\([^()]*\)\.mean\(\)")
    n = 0
    for fn in _production_modules():
        with open(_src_path(fn), encoding="utf-8") as f:
            for line in f:
                if _MAE_CALL_SITE_RE.search(line.split("#")[0]):
                    n += 1
    return n


class TestGoldenFixtureScores:
    """
    The frozen contract. Every value here is derivable from the delta column in
    FIXTURE by hand, and is asserted exactly (==) rather than approximately:
    the deltas are dyadic, so any drift in metrics() semantics moves the value
    off the literal instead of wobbling inside a tolerance.
    """

    @pytest.mark.parametrize("var", sorted(GOLDEN))
    def test_mae_is_frozen(self, var):
        # MAE = sum(|pred - truth|) / n
        g = GOLDEN[var]
        assert metrics(_pred(var), _truth(var))["mae"] == g["mae"], (
            f"{var} MAE changed; metrics() no longer means "
            f"mean(|pred - truth|) over the surviving rows")

    @pytest.mark.parametrize("var", sorted(GOLDEN))
    def test_rmse_is_frozen(self, var):
        g = GOLDEN[var]
        # The RMSE literals are the only non-exact values in the fixture
        # (sqrt of a dyadic that is not itself a perfect square), so they carry
        # a 1e-12 window. That is ~13 orders of magnitude below the reported
        # precision of 3 decimals and still cannot hide a semantic change.
        assert metrics(_pred(var), _truth(var))["rmse"] == pytest.approx(
            g["rmse"], rel=0.0, abs=1e-12)

    @pytest.mark.parametrize("var", sorted(GOLDEN))
    def test_bias_uses_pred_minus_truth(self, var):
        """
        THE CONVENTION, pinned as a sign.

        bias = mean(pred - truth), the WMO definition, which is what
        monitoring.py uses and therefore what triage reads on the dashboard.
        POSITIVE bias = the model runs WARM. This used to be the opposite sign,
        which meant the same forecast read as over-predicting in the benchmark
        and under-predicting in production, so somebody chasing a systematic bias
        from the dashboard would have worked on the wrong end of the model.

        It is asserted as a signed literal rather than as |bias| so a return to
        the old convention cannot pass, and the fixture deliberately contains one
        NEGATIVE bias (pressure) so a bias that had accidentally been made to
        equal its own magnitude would also fail.
        """
        g = GOLDEN[var]
        m = metrics(_pred(var), _truth(var))
        assert m["bias"] == g["bias"], (
            f"{var} bias changed. metrics() must report mean(pred - truth); "
            f"a POSITIVE bias means the model is WARM, a NEGATIVE bias COLD. "
            f"Got {m['bias']!r}, expected {g['bias']!r}")

    def test_bias_sign_agrees_with_a_hand_built_warm_model(self):
        """
        The convention stated as a physical claim instead of a literal: a model
        that predicts 1 degree above observation on every window is running warm
        and must report a POSITIVE bias. If the sign convention is ever argued
        about again, this is the sentence to point at.
        """
        t = [0.0, 0.0, 0.0, 0.0]
        assert metrics([1.0, 1.0, 1.0, 1.0], t)["bias"] == 1.0
        assert metrics([0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0])["bias"] == -1.0

    @pytest.mark.parametrize("var", sorted(GOLDEN))
    def test_row_count_and_key_set_are_frozen(self, var):
        m = metrics(_pred(var), _truth(var))
        assert m["n"] == 8
        # Consumers index m["mae"]; ci95_mae is part of the published contract.
        # coverage and rows_dropped_nonfinite are what make a partial score
        # legible instead of a bare number, so they are part of it too -- a
        # score that cannot say how much of the input it actually saw is the
        # failure this file exists to prevent.
        assert set(m) == {"mae", "rmse", "bias", "n", "coverage",
                          "rows_dropped_nonfinite", "ci95_mae"}

    @pytest.mark.parametrize("var", sorted(GOLDEN))
    def test_coverage_reporting_is_frozen(self, var):
        # The fixture is entirely finite, so coverage is exactly 1.0 and nothing
        # is dropped. Asserted as literals so a coverage field that reported the
        # wrong denominator (e.g. 8/1, or kept/total computed before the mask)
        # could not pass.
        m = metrics(_pred(var), _truth(var))
        assert m["coverage"] == GOLDEN[var]["coverage"] == 1.0
        assert m["rows_dropped_nonfinite"] == GOLDEN[var]["rows_dropped_nonfinite"] == 0
        assert m["n"] + m["rows_dropped_nonfinite"] == 8, (
            "n and rows_dropped_nonfinite must account for every input row")

    @pytest.mark.parametrize("var", sorted(GOLDEN))
    def test_bootstrap_ci_bounds_are_frozen(self, var):
        # Pins bootstrap_ci's 400 resamples / 95% / seed 42. A silent change of
        # n_boot or seed moves every published confidence interval.
        lo, hi = metrics(_pred(var), _truth(var))["ci95_mae"]
        g_lo, g_hi = GOLDEN[var]["ci95_mae"]
        assert lo == pytest.approx(g_lo, rel=0.0, abs=1e-12)
        assert hi == pytest.approx(g_hi, rel=0.0, abs=1e-12)

    def test_mae_is_not_equal_to_bias(self):
        """
        Guards the specific refactor that replaces the signed mean with the
        absolute mean. The fixture is built so MAE and bias differ for every
        variable, so that mistake cannot pass unnoticed.
        """
        for var in sorted(GOLDEN):
            m = metrics(_pred(var), _truth(var))
            assert m["mae"] != m["bias"], (
                f"{var}: mae == bias, so the sign information has been lost")


class TestGoldenPartialCoverageFixture:
    """
    The same golden discipline applied to a fixture that is MISSING rows.

    GOLDEN above is entirely finite, so every coverage value in it is 1.0 and
    every dropped count is 0. That makes the four-variable fixture unable to
    falsify those two fields: an implementation that hardcoded
    `coverage = 1.0` or `rows_dropped_nonfinite = 0` would pass every assertion
    in TestGoldenFixtureScores. Mutation testing found exactly that gap, so this
    class exists to close it.

    The scores here are still computed over the surviving rows only, and the
    denominators are the masked length, not the input length -- that is the same
    contract as the full fixture, with a score that is visibly partial.
    """

    def test_scores_over_surviving_rows_are_frozen(self):
        m = metrics(PARTIAL_PRED, PARTIAL_TRUTH)
        assert m is not None, (
            "4 of 6 rows is coverage 0.67, above MIN_COVERAGE, so this must be "
            "scored; if it is None the gate is tighter than documented")
        assert m["mae"] == GOLDEN_PARTIAL["mae"] == 0.625
        assert m["rmse"] == pytest.approx(GOLDEN_PARTIAL["rmse"], rel=0.0, abs=1e-12)
        assert m["bias"] == GOLDEN_PARTIAL["bias"] == 0.125
        assert m["n"] == 4
        lo, hi = m["ci95_mae"]
        assert lo == pytest.approx(GOLDEN_PARTIAL["ci95_mae"][0], rel=0.0, abs=1e-12)
        assert hi == pytest.approx(GOLDEN_PARTIAL["ci95_mae"][1], rel=0.0, abs=1e-12)

    def test_coverage_is_the_true_fraction_not_a_constant(self):
        m = metrics(PARTIAL_PRED, PARTIAL_TRUTH)
        assert m["coverage"] == pytest.approx(4.0 / 6.0)
        assert m["coverage"] != 1.0, (
            "coverage is hardcoded; a partial score reported as full coverage is "
            "the original hazard with a new field on top of it")
        assert m["rows_dropped_nonfinite"] == 2
        assert m["rows_dropped_nonfinite"] != 0, (
            "rows_dropped_nonfinite is hardcoded to 0; the dropped rows have to "
            "be visible in the result or the loss is unauditable")

    def test_the_three_counts_are_mutually_consistent(self):
        """
        The strongest form of the check, because it survives any ONE of the three
        fields lying: n, rows_dropped_nonfinite and coverage must describe the
        same 6 rows. A hardcoded coverage of 1.0, a hardcoded dropped count of 0,
        or an n computed from the unmasked length all break the identity.
        """
        m = metrics(PARTIAL_PRED, PARTIAL_TRUTH)
        n, dropped, cov = m["n"], m["rows_dropped_nonfinite"], m["coverage"]
        assert n + dropped == len(PARTIAL_PRED) == 6
        assert cov == pytest.approx(n / (n + dropped))
        assert n == 4 and dropped == 2

    def test_dropped_rows_are_excluded_from_the_denominator(self):
        # If the six-row input were divided by 6 instead of the surviving 4, MAE
        # would be 2.5/6 = 0.4166... and bias 0.5/6 = 0.0833... -- both
        # flattering, both wrong, and both a reward for missing data.
        m = metrics(PARTIAL_PRED, PARTIAL_TRUTH)
        assert m["mae"] == 0.625
        assert m["mae"] != pytest.approx(2.5 / 6.0)
        assert m["bias"] != pytest.approx(0.5 / 6.0)


class TestHandComputableCases:
    """The cases that can be checked with pencil and paper, no fixture."""

    def test_perfect_prediction_scores_zero(self):
        # pred == truth, so every error is 0: mean(0)=0, sqrt(mean(0))=0.
        m = metrics([1, 2, 3], [1, 2, 3])
        assert m["mae"] == 0.0
        assert m["rmse"] == 0.0
        assert m["bias"] == 0.0
        assert m["n"] == 3
        assert m["coverage"] == 1.0
        assert m["rows_dropped_nonfinite"] == 0

    def test_constant_under_forecast_scores_as_hand_computed(self):
        """
        pred = [0, 0], truth = [2, 4].

        errors pred - truth = [-2, -4]; |errors| = [2, 4]
          MAE  = (2 + 4) / 2                    = 3.0
          MSE  = (2**2 + 4**2) / 2 = 8 / 2      = 4.0
          RMSE = sqrt(4)                        = 2.0
        bias  = mean(pred - truth) = (-2 + -4) / 2 = -3.0

        Two numbers here are NOT the ones a reader would guess, and both have
        already been got wrong once:

        * RMSE is 3.1622776601683795, not 2.0. sqrt(4) is what you get from
          treating the errors as [2, 2]; the actual errors are [2, 4] and
          mean(e^2) = (4 + 16)/2 = 10, not 4. The original brief for this work
          quoted 2.0, which is exactly the class of confidently-wrong number
          this file exists to catch, so the value actually produced is asserted
          and the wrong one is asserted against.
        * bias is -3.0, NEGATIVE, because this model UNDER-forecasts: it
          predicts 0 against observations of 2 and 4. Under the pre-correction
          convention (mean(truth - pred)) this was +3.0, i.e. the same forecast
          was published as running warm when it was running cold.
        """
        m = metrics([0, 0], [2, 4])
        assert m["mae"] == 3.0
        assert m["rmse"] == 3.1622776601683795     # sqrt(10), NOT sqrt(4)
        assert m["rmse"] != 2.0, (
            "RMSE is 2.0 only if the errors were [2, 2]; the brief's value is "
            "wrong and asserting it would bake the error into the fixture")
        assert m["bias"] == -3.0, (
            "metrics() reports mean(pred - truth) per WMO, so a model that "
            "predicts below observation has a NEGATIVE bias. +3.0 is the "
            "inverted convention that used to be here.")
        assert m["n"] == 2

    def test_bias_is_zero_when_errors_cancel(self):
        # pred - truth = [+2, -2]: MAE = 2, bias = 0. Catches an implementation
        # that computes bias from the absolute errors.
        m = metrics([2.0, 0.0], [0.0, 2.0])
        assert m["mae"] == 2.0
        assert m["bias"] == 0.0

    def test_min_coverage_is_half(self):
        """
        The threshold is part of the contract, not an implementation detail.
        Half the rows is the line: a score over more of the input is a score of
        the model, a score over less is a score of whichever rows happened to
        survive, and the run that motivated this gate scored over ZERO rows
        while still printing a rank. Pinned so a "helpful" loosening of the
        threshold has to be a deliberate edit.
        """
        assert MIN_COVERAGE == 0.5

    def test_rmse_is_never_below_mae(self):
        # sqrt(mean(e^2)) >= mean(|e|) for any e, equality only when all |e|
        # are equal. A semantic change to metrics() that loses the square (or
        # computes RMSE on a different mask) breaks this immediately.
        rng = np.random.default_rng(0)
        for _ in range(25):
            t = rng.normal(30.0, 5.0, 200)
            p = t + rng.normal(0.0, 1.5, 200)
            m = metrics(p, t)
            assert m["rmse"] >= m["mae"] - 1e-12


class TestNaNHandling:
    """
    WHAT metrics() ACTUALLY DOES WITH NAN, pinned rather than assumed.

    It builds the pairwise mask `np.isfinite(p) & np.isfinite(t)` and drops every
    row where either side is non-finite -- +inf and -inf by the same mask -- then
    REPORTS what it dropped: `coverage` (fraction of INPUT rows where both sides
    are finite) and `rows_dropped_nonfinite`. If coverage falls below
    MIN_COVERAGE it returns None instead of a score.

    This class was INVERTED when the gate shipped. It previously asserted the
    silence -- that a partial score was indistinguishable from a full one -- and
    `test_no_coverage_warning_is_emitted` existed specifically to pin that
    hazard. Asserting a hazard is only safe for as long as the hazard is
    unfixed; once it is fixed, the test has to be rewritten to pin the fix, or
    it becomes a test that fails the moment the bug is closed. Its replacement,
    test_low_coverage_input_is_rejected, is the same mutation-detection strength
    pointed the other way: remove the gate and it fails.

    What the gate does NOT fix, and is still pinned below: None is overloaded.
    It now means both "nothing survived" and "too little survived", and a
    caller cannot tell which from the return value.
    """

    def test_nonfinite_rows_are_dropped_pairwise_and_reported(self):
        # Three rows, one poisoned in pred. kept = 2 of 3, so coverage 2/3,
        # which clears MIN_COVERAGE, and the two survivors are exact so the score
        # is a clean 0.0. The dropped row is now visible in the return value
        # rather than only inferable from `n`.
        m = metrics([1.0, float("nan"), 3.0], [1.0, 2.0, 3.0])
        assert m["mae"] == 0.0
        assert m["rmse"] == 0.0
        assert m["bias"] == 0.0
        assert m["n"] == 2, "the nan row was not dropped; the mask semantics changed"
        assert m["coverage"] == pytest.approx(2.0 / 3.0)
        assert m["rows_dropped_nonfinite"] == 1

    def test_nonfinite_truth_is_dropped_too(self):
        # The mask is symmetric. A gap in the OBSERVATION must not be scored as
        # a model error, and a gap in the model must not be scored as an
        # observation error; both are dropped and both are reported.
        m = metrics([1.0, 2.0, 3.0], [1.0, float("nan"), 3.0])
        assert m["n"] == 2
        assert m["mae"] == 0.0
        assert m["coverage"] == pytest.approx(2.0 / 3.0)
        assert m["rows_dropped_nonfinite"] == 1

    def test_infinity_is_dropped_by_the_same_mask(self):
        # np.isfinite rejects +-inf as well as nan. A saturated sensor reading
        # is therefore dropped rather than dominating the mean, which is right,
        # and it counts against coverage rather than passing unnoticed.
        # kept 1 of 2 = coverage exactly 0.5 = MIN_COVERAGE, and the gate is
        # inclusive, so this is scored rather than refused. That boundary is
        # deliberate and is pinned explicitly in the gate tests below.
        m = metrics([1.0, float("inf")], [1.0, 2.0])
        assert m["n"] == 1
        assert m["mae"] == 0.0
        assert m["coverage"] == 0.5
        assert m["rows_dropped_nonfinite"] == 1

    def test_denominator_is_the_masked_length_not_the_input_length(self):
        """
        The mutation a NaN test normally misses. Most of the tests above use
        surviving errors of exactly zero, so dividing by 2 or by 3 gives the same
        answer and cannot tell a correct implementation from one that masks the
        array and then divides by the ORIGINAL length -- which silently deflates
        every score in proportion to how much data is missing, i.e. it rewards
        whichever model has the worse coverage.

        pred  = [1.0,  nan, 4.0, 5.0]
        truth = [1.0, 2.0, 3.0, 3.0]
        kept 3 of 4 -> coverage 0.75, which clears MIN_COVERAGE, so this case is
        still scored and the denominator is still observable. That sizing is
        the whole reason this test still works under the gate: a 4-row case with
        two non-finite rows would be coverage 0.5 and a 3-row case with one
        non-finite row is 2/3, but anything under half returns None and would
        tell us nothing about the denominator. See
        test_denominator_is_unobservable_below_the_gate for the other half of
        this story.

        surviving rows 0, 2, 3 -> errors pred - truth = [0, 1, 2]
          MAE  = (0 + 1 + 2) / 3                  = 1.0
          MSE  = (0 + 1 + 4) / 3                  = 5/3
          RMSE = sqrt(5/3)                        = 1.2909944487358056
          bias = mean(pred - truth) = 3/3         = +1.0
        dividing by 4 instead would give 0.75 / 1.118033988749895 / 0.75.
        """
        m = metrics([1.0, float("nan"), 4.0, 5.0], [1.0, 2.0, 3.0, 3.0])
        assert m["n"] == 3
        assert m["coverage"] == 0.75
        assert m["mae"] == 1.0
        assert m["rmse"] == pytest.approx(1.2909944487358056, rel=0.0, abs=1e-12)
        assert m["bias"] == 1.0
        assert m["mae"] != 0.75 and m["bias"] != 0.75, (
            "the divisor is the unmasked input length; a score that shrinks "
            "with missing coverage is not a score")

    def test_denominator_is_unobservable_below_the_gate(self):
        """
        The interaction the gate creates, pinned so it is a decision and not an
        accident: the same three surviving rows, presented with a fourth and
        fifth row of padding, stop being scored at all.

        [1.0, 2.0, 3.0] against [1.0, 2.0, 3.0] is full coverage and scores.
        The same three good rows inside a 6-row input with 3 non-finite is
        coverage 0.5... and 0.499 fails, so this uses 4 non-finite of 7.
        """
        good = [1.0, 2.0, 3.0]
        assert metrics(good, good)["coverage"] == 1.0

        # The identical three good rows, padded out to 7 with 4 non-finite.
        padded_pred = good + [float("nan")] * 4
        padded_truth = good + [float("nan")] * 4
        assert metrics(padded_pred, padded_truth) is None, (
            "3 of 7 rows is coverage 0.43, below MIN_COVERAGE; the gate must "
            "refuse rather than report a score of the surviving rows")

    def test_returns_none_when_no_row_survives(self):
        # The documented NWP-cache failure: total loss of coverage yields None,
        # not a number and not a raise. Callers must handle None explicitly.
        assert metrics([float("nan"), float("nan")], [1.0, 2.0]) is None
        assert metrics([], []) is None

    def test_low_coverage_input_is_rejected(self):
        """
        INVERTED from the old test_no_coverage_warning_is_emitted, which
        asserted that a partial score was returned with nothing to distinguish
        it from a full one. That hazard is fixed; this pins the fix and has the
        same mutation-detection strength, pointed the other way. If the coverage
        gate is removed, loosened, or applied with >= instead of >, this fails.

        The case: 3 rows, 1 non-finite -> kept 2 -> coverage 2/3 = 0.667, above
        MIN_COVERAGE, scored. 3 rows, 2 non-finite -> kept 1 -> coverage 1/3,
        below MIN_COVERAGE, refused. The surviving row in the refused case is a
        PERFECT match, so the score it declines to publish is 0.0 -- the most
        flattering number available. Refusing to report it is the entire point:
        a model that scored 0.0 because one of three hours was missing is not a
        model that scored 0.0.
        """
        scored = metrics([1.0, float("nan"), 3.0], [1.0, 2.0, 3.0])
        assert scored is not None
        assert scored["coverage"] == pytest.approx(2.0 / 3.0)
        assert scored["coverage"] > MIN_COVERAGE

        refused = metrics([1.0, float("nan"), float("nan")], [1.0, 2.0, 3.0])
        assert refused is None, (
            "coverage 1/3 is below MIN_COVERAGE and must return None; a "
            "confident score over a third of the rows is the failure this gate "
            "exists to prevent")

    def test_gate_boundary_is_inclusive_at_exactly_min_coverage(self):
        """
        Exactly MIN_COVERAGE is scored, not refused. The comparison is
        `coverage < min_coverage`, so 0.5 survives and 0.499... does not. This
        is the kind of boundary that gets flipped by accident during a refactor
        and changes which runs produce a table.
        """
        assert MIN_COVERAGE == 0.5
        # 1 of 2 rows finite -> coverage exactly 0.5.
        at = metrics([1.0, float("nan")], [1.0, 2.0])
        assert at is not None, "coverage exactly at the threshold must still score"
        assert at["coverage"] == 0.5
        assert at["n"] == 1 and at["rows_dropped_nonfinite"] == 1

        # 1 of 3 rows finite -> coverage 0.333, one row below the threshold.
        below = metrics([1.0, float("nan"), float("nan")], [1.0, 2.0, 3.0])
        assert below is None

    def test_min_coverage_is_overridable_per_call(self):
        """
        The threshold is a parameter, not a constant baked into the body, so a
        caller scoring a variable with legitimately patchy coverage can demand a
        lower bar without editing the shared function -- and, more importantly,
        can still SEE what bar it got. The 3-of-4 case is refused by the default
        gate at min_coverage=0.8 and scored at 0.5, with coverage reported
        identically either way.
        """
        p = [1.0, float("nan"), 4.0, 5.0]
        t = [1.0, 2.0, 3.0, 3.0]
        assert metrics(p, t, min_coverage=0.8) is None
        relaxed = metrics(p, t, min_coverage=0.5)
        assert relaxed is not None
        assert relaxed["coverage"] == 0.75
        assert relaxed["rows_dropped_nonfinite"] == 1
        # The caller can raise the bar too, and the reported coverage does not
        # move with the threshold -- it describes the data, not the policy.
        assert metrics(p, t, min_coverage=0.0)["coverage"] == 0.75

    def test_none_is_overloaded_and_says_nothing_about_which_failure(self):
        """
        RESIDUAL HAZARD, not fixed. None now means BOTH "no row survived" and
        "some rows survived but not enough". Those are different diagnoses: the
        first is a total lookup failure (every NWP series missing, the run that
        motivated all of this), the second is a marginal series that is mostly
        absent. A caller that logs "scorer returned None" learns nothing about
        which happened, and cannot tell a broken cache from a thin one without
        recomputing the mask itself. Worth knowing before someone treats a None
        as a single condition.
        """
        nothing_survived = metrics([float("nan")] * 4, [1.0, 2.0, 3.0, 4.0])
        too_little_survived = metrics(
            [1.0, float("nan"), float("nan"), float("nan")], [1.0, 2.0, 3.0, 4.0])
        assert nothing_survived is None
        assert too_little_survived is None
        # Identical return value, genuinely different causes. If metrics() ever
        # grows a way to tell them apart -- a sentinel, a reason key, a typed
        # error -- this test is the one that has to be rewritten.
        assert nothing_survived is too_little_survived is None

    def test_caller_still_has_to_rank_only_the_rows_that_scored(self):
        """
        The caller side of the contract, re-checked against the CURRENT main(),
        because the rank-denominator hazard was reported here and then fixed.

        benchmark_vs_nwp.main now ranks only among rows that produced a score
        (sorting unscored rows to 9e9 puts them after every scored row) and
        appends "[WARNING: N of M methods produced no score; this rank is NOT
        comparable to a full run]" when any method is unscored.

        The denominator is STILL `len(rows) - 1`, which counts all seven NWP
        models whether or not they scored -- so a partial run still prints
        "LNN rank 1/8" and relies on the warning text to stop that being read as
        "1st of 8". The warning is a mitigation, not a fix, and this test pins
        both halves of that so the next reader is not surprised.
        """
        rows = [("LNN (this project)", metrics([0.0] * 8, list(range(8)))),
                ("persistence", metrics([0.0] * 8, list(range(8)))),
                ("ECMWF IFS", metrics([float("nan")] * 8, list(range(8))))]
        scored = [(n, m) for n, m in rows if m]
        assert len(scored) == 2
        assert len(rows) - 1 == 2, (
            "denominator still counts the unscored NWP rows, so a partial run "
            "prints a rank out of a larger denominator than it was earned "
            "against; the WARNING text is the only thing distinguishing it")


class TestWindUnitsConversion:
    """
    The 3.6x units bug.

    Open-Meteo's `wind_speed_10m` is km/h. The station telemetry is m/s
    (verified: station wind_speed mean 1.501 m/s vs NWP 11.19 km/h = 3.11 m/s).
    The first benchmark compared the two raw. Every NWP wind error was inflated
    by 3.6x, the LNN was reported ~6x better than ECMWF on wind when the honest
    figure is ~1.4x better, and the whole "we beat real NWP" claim rested on a
    unit conversion that was missing. NWP_TO_STATION_UNITS is the entire fix and
    is a one-line constant, which means a refactor can delete it silently.
    """

    def test_36_kmh_converts_to_10_ms(self):
        assert 36.0 * NWP_TO_STATION_UNITS["wind_speed"] == 10.0

    def test_conversion_factor_is_exactly_one_over_3_6(self):
        assert NWP_TO_STATION_UNITS["wind_speed"] == 1.0 / 3.6
        # Only wind needs converting; every other field is already in station
        # units, and a stray factor on one of them would rescale that variable's
        # entire score.
        for var in ("temperature", "humidity", "pressure"):
            assert var not in NWP_TO_STATION_UNITS
        assert NWP_TO_STATION_UNITS.get("temperature", 1.0) == 1.0
        assert NWP_TO_STATION_UNITS.get("humidity", 1.0) == 1.0
        assert NWP_TO_STATION_UNITS.get("pressure", 1.0) == 1.0

    def test_declared_units_agree_with_the_conversion(self):
        # UNITS is what gets printed next to the MAE. It says m/s because the
        # conversion has already been applied to the NWP series by the time
        # metrics() sees it. If UNITS were edited to "km/h" the table would
        # relabel converted values and the number would be off by 3.6x on
        # paper -- the exact failure this constant was added to stop.
        assert UNITS["wind_speed"] == "m/s"
        # The NWP field being converted is the Open-Meteo km/h one.
        assert VAR_MAP["wind_speed"][0] == "wind_speed_10m"
        # and the truth it is scored against is the station observation.
        assert VAR_MAP["wind_speed"][1] == "target_wind_speed"

    def test_scoring_unconverted_kmh_inflates_the_error(self):
        """
        The concrete arithmetic, on the fixture's wind column.

        station truth (m/s): 1.5 2.0 2.5 3.0 2.5 2.0 1.5 1.0
        LNN pred   (m/s):    1.75 2.0 2.0 3.75 2.25 2.5 1.5 0.75   (MAE 0.3125)
        the same forecast expressed the way Open-Meteo would publish it, in
        km/h, scored against the m/s truth WITHOUT conversion:
            MAE 5.425   RMSE 5.92536918681022   bias +5.425
        with the conversion applied, the score is identical to the m/s score.

        The bias is POSITIVE and large: the km/h series is bigger than the m/s
        truth at every timestamp, so under the pred - truth convention this
        reads as a strong warm bias, which is exactly what an unconverted NWP
        series looks like from the scorer's side. Under the pre-correction
        convention the same number was published as -5.425, i.e. a model that
        was running 3.6x too warm was reported as running far too cold.

        The inflation is 17.36x here, not 3.6x. The naive "3.6x" only holds when
        pred ~= truth, because the error being measured is |3.6*pred - truth|,
        which scales the PREDICTION by 3.6 and leaves the truth alone. At any
        real wind speed the truth term is what makes it worse, which is why the
        historical "3.6x" understates the damage and why the bug survived a
        sanity check.
        """
        truth = _truth("wind_speed")
        pred_ms = _pred("wind_speed")
        nwp_kmh = pred_ms * 3.6

        inflated = metrics(nwp_kmh, truth)
        assert inflated["mae"] == pytest.approx(5.425, rel=0.0, abs=1e-12)
        assert inflated["rmse"] == pytest.approx(5.92536918681022, rel=0.0, abs=1e-12)
        assert inflated["bias"] == pytest.approx(5.425, rel=0.0, abs=1e-12)
        assert inflated["n"] == 8
        assert inflated["coverage"] == 1.0

        converted = metrics(nwp_kmh * NWP_TO_STATION_UNITS["wind_speed"], truth)
        assert converted["mae"] == GOLDEN["wind_speed"]["mae"]
        assert converted["rmse"] == pytest.approx(
            GOLDEN["wind_speed"]["rmse"], rel=0.0, abs=1e-12)
        assert inflated["mae"] / converted["mae"] == pytest.approx(17.36, abs=1e-9)

    def test_candidate_scoring_path_does_not_apply_the_nwp_conversion(self):
        """
        The mirror-image bug, already paid for once. `_score_candidate` feeds the
        candidate checkpoint the NORMALISED telemetry tensor res[0] and reads the
        candidate's own output, which is in m/s because that is what it was
        trained on. NWP_TO_STATION_UNITS is for the Open-Meteo fields only;
        applying it there rescales the candidate by 3.6x, which produced a +3h
        wind MAE of 0.742 instead of the correct 0.498 -- the model was
        penalised for being in the right units.
        """
        body = _function_code("benchmark_vs_nwp.py", "_score_candidate")
        assert "NWP_TO_STATION_UNITS" not in body, (
            "_score_candidate must not convert the candidate's m/s output; "
            "the candidate is already in station units")
        # It must also not be fed the de-normalised X_raw, which is the
        # production bundle predictor's input space, not this model's.
        assert "X_raw" not in body
        # The docstring still states the rule, so the knowledge survives even
        # though the constant is absent from the code.
        assert "NWP_TO_STATION_UNITS" in _function_source(
            "benchmark_vs_nwp.py", "_score_candidate")

    def test_sibling_benchmark_declares_the_wrong_wind_unit(self):
        """
        FINDING. benchmark_independent.py reads the same telemetry and scores
        the same wind column as benchmark_vs_nwp.py, but labels it "km/h"
        where benchmark_vs_nwp labels it "m/s". Neither script converts
        anything -- both score raw station m/s -- so one of the two published
        tables is mislabelled by a factor of 3.6 on its face, and a reader
        comparing the two wind MAEs has no way to tell which one to believe.
        Reported, not fixed: the two scripts are separate deliverables and
        picking the correct label is a judgement call about what each one claims.
        """
        import benchmark_independent
        assert benchmark_independent.UNITS["wind_speed"] == "km/h"
        assert UNITS["wind_speed"] == "m/s"
        assert benchmark_independent.UNITS["wind_speed"] != UNITS["wind_speed"], (
            "the two benchmarks no longer disagree on the wind unit label; if "
            "that was fixed, update this finding rather than deleting it")


class TestSingleScoringImplementation:
    """
    Requirement: there is exactly one scoring function and everything else
    imports or aliases it.

    THAT IS NOT TRUE TODAY, and the tests below pin the actual state instead of
    asserting a wish. The reference implementation, benchmark_vs_nwp.metrics,
    has no importers at all: the two benchmark scripts, the monitoring layer,
    the validator and the training-time evaluators each carry their own copy.
    Changing metrics() therefore changes the NWP benchmark table and nothing
    else, and there is no test anywhere that would notice the split.

    The bias-sign correction made this WORSE in one specific way, and the tests
    below record it: metrics() moved from truth-pred to pred-truth, which
    brought it into line with monitoring.py and the WMO, and left
    benchmark_independent.score as the single remaining inversion. The
    independent check on the benchmark is now the one benchmark whose bias
    column means the opposite of the benchmark it is checking. That is not
    fixed here -- correcting a published results column is not a test's call --
    but it is asserted, so it cannot be forgotten.
    """

    def test_no_module_imports_or_aliases_the_shared_metrics_function(self):
        """
        The core of the finding, asserted so it cannot drift silently.

        Checked from the AST, not from grep: four modules name
        benchmark_vs_nwp.py in their prose (evaluate_downscale_test.py:56,
        fit_nwp_correction.py:12, refit_policy.py:19, train_downscale_nwp.py:32),
        all of them comments about which script scores the TEST split. A
        text search cannot tell a mention of a module from a dependency on it,
        and a mention is not a dependency.

        If somebody consolidates the scoring layer this list grows and the
        assertion has to be revisited -- which is the intended friction.

        NOTE: this test was INVERTED when the scoring layer was consolidated.
        It originally asserted that nothing imported benchmark_vs_nwp, because at
        the time metrics() had zero dependents and the layer was N independent
        implementations. scoring.py is now the canonical module and callers are
        expected to import from it. What remains forbidden is importing
        benchmark_vs_nwp, which is now a thin delegating wrapper: reaching through
        it couples a caller to the benchmark harness rather than to the scorer.
        """
        importers = {fn: _imports_of(fn, "benchmark_vs_nwp")
                     for fn in _production_modules()
                     if fn != "benchmark_vs_nwp.py"}
        importers = {k: v for k, v in importers.items() if v}
        assert importers == {}, (
            f"these modules IMPORT benchmark_vs_nwp rather than scoring: "
            f"{importers}. benchmark_vs_nwp.metrics is a delegating wrapper "
            f"around scoring.metrics; import the scorer directly so callers do "
            f"not depend on the benchmark harness.")

    def test_analysis_scripts_use_the_shared_scorer(self):
        """
        The counterpart to _production_modules() excluding analysis scripts.

        Excluding them from the production scorer ledger would otherwise be a
        loophole: a new analysis script could hand-roll MAE and nobody would
        notice. So every excluded module that touches absolute error must either
        import scoring.metrics or appear in the narrow allowlist with a stated
        reason. This test fails when that stops being true.
        """
        offenders, allowed = [], set()
        for fn in sorted(os.listdir(SRC)):
            if not fn.endswith(".py") or not _is_analysis_script(fn):
                continue
            if fn in _ANALYSIS_STATISTIC_ALLOWLIST:
                allowed.add(fn)
                continue
            path = os.path.join(SRC, fn)
            with open(path, encoding="utf-8") as f:
                src = f.read()
            touches_error = "abs(" in src and ("mean(" in src or "sum(" in src)
            if not touches_error:
                continue
            if _imports_of(fn, "scoring"):
                continue
            # A module may legitimately use scoring under a different import shape.
            if "from scoring import" in src or "import scoring" in src:
                continue
            offenders.append(fn)

        assert not offenders, (
            f"analysis scripts computing absolute error without the shared "
            f"scorer: {offenders}. Import scoring.metrics so their numbers are "
            f"comparable with every other measurement in the system, or add an "
            f"entry to _ANALYSIS_STATISTIC_ALLOWLIST stating why the script "
            f"needs its own statistic. Currently allowed: {sorted(allowed)}")

    def test_inventory_of_independent_scorers_has_not_grown(self):
        """
        The ledger. A new dict-returning MAE function fails this test, which is
        the point: duplication in the measurement layer has to be a conscious
        act. Current count is 13 including the reference itself.
        """
        found = _scan_scorers()
        known = {(f, ln, name) for f, ln, name, _keys in KNOWN_INDEPENDENT_SCORERS}
        found_keys = {(f, ln, name) for f, ln, name, _keys in found}
        new = sorted(found_keys - known)
        assert new == [], (
            f"new independent MAE scorer(s) added: {new}. Every one of these "
            f"is a place the scoring semantics can drift; import "
            f"benchmark_vs_nwp.metrics instead, or add it here deliberately")
        assert len(found) == len(KNOWN_INDEPENDENT_SCORERS), (
            f"scorer count moved from {len(KNOWN_INDEPENDENT_SCORERS)} to "
            f"{len(found)}: "
            f"{sorted(found_keys ^ known)} appeared or disappeared. Growing is "
            f"a regression. Shrinking is progress and means updating this "
            f"list deliberately.")

    def test_every_frozen_scorer_is_still_at_its_frozen_location(self):
        """
        Stops the ledger rotting. Each entry is re-located in the source, so a
        rename or a move fails here rather than leaving a file:line in the list
        that points at nothing.
        """
        for filename, lineno, funcname, keys in KNOWN_INDEPENDENT_SCORERS:
            with open(_src_path(filename), encoding="utf-8") as f:
                tree = ast.parse(f.read())
            found = [n for n in ast.walk(tree)
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and n.name == funcname]
            assert found, f"{filename}:{lineno} {funcname} is gone"
            assert any(n.lineno == lineno for n in found), (
                f"{funcname} moved off line {lineno} in {filename} "
                f"(now at {[n.lineno for n in found]}); update the ledger")
            assert _returned_mae_keys(found[0]) == keys, (
                f"{filename}:{funcname} no longer returns {keys}; the ledger "
                f"says it does")

    def test_scalar_mae_helpers_are_listed(self):
        """
        _scan_scorers() keys on a returned "mae" dict entry, so it is blind to
        the bare-return helpers. Those are hand-listed, and the ones that can be
        imported cheaply are also checked dynamically below.
        """
        for filename, lineno, funcname in KNOWN_SCALAR_MAE_HELPERS:
            with open(_src_path(filename), encoding="utf-8") as f:
                tree = ast.parse(f.read())
            assert any(
                isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                and n.name == funcname and n.lineno == lineno
                for n in ast.walk(tree)), \
                f"{filename}:{lineno} {funcname} is gone"

    def test_raw_mae_call_site_count_has_not_grown(self):
        """
        Floor on the spread of the concept. A lower bound rather than a census: an
        error array and its .mean() on separate lines is invisible to a text scan
        (diagnose_headroom.py:64 is one such pair, which is why it only shows up in
        KNOWN_SCALAR_MAE_HELPERS).

        The assertion is now an UPPER BOUND. It was an equality against 64, while
        its own docstring described it as a floor -- so consolidating the scoring
        layer SHRINKING the count failed the test meant to detect growth. Analysis
        scripts (experiment_*, analyze_*, report_*) are excluded from the scan for
        the same reason they are excluded from the scorer inventory: they are not
        on the serving path, and they are individually justified in
        _ANALYSIS_STATISTIC_ALLOWLIST and checked by
        test_analysis_scripts_use_the_shared_scorer.
        """
        count = _scan_mae_call_sites()
        assert count <= KNOWN_MAE_CALL_SITES, (
            f"mean(|a-b|) call sites grew from {KNOWN_MAE_CALL_SITES} to {count}. "
            f"Growth is duplication; import scoring.metrics instead.")

    def test_sibling_benchmark_is_a_separate_implementation_without_the_coverage_gate(self):
        """
        FINDING, demonstrated rather than asserted from a source read.

        benchmark_independent.score is the one written specifically as an
        independent check on the same test split, and it now lags metrics() in
        three separate ways that the coverage gate closed:

        1. NO MASK AT ALL. It does not drop non-finite rows. On the same input
           where metrics() returns a clean score over the surviving rows, score()
           returns nan for every metric while still reporting the FULL unmasked
           n. Its own fmt() renders nan as "n/a", so a single missing hour
           prints as "n/a" with no statement of how many rows were scored.
        2. NO COVERAGE GATE, so the low-coverage case that metrics() now refuses
           outright is indistinguishable from a model that scored badly.
        3. NO COVERAGE FIELDS, so there is nothing in the return value that
           could tell a reader how much of the input was actually used -- and the
           `n` it does report is the unmasked input length, which overstates it.

        The consequence is that the two benchmark scripts cannot be compared row
        for row, and the independent check cannot catch a coverage bug in the
        thing it is checking -- which is the specific class of bug the gate was
        added for.

        benchmark_independent.py's docstring says its baselines are "implemented
        here directly and transparently rather than imported, so their
        definition cannot drift". On the NaN path they have now drifted, and
        transparency is what made the drift invisible. Left in place and
        asserted: fixing the sibling means editing a published-results script,
        which is a bigger decision than re-basing a test fixture.
        """
        import benchmark_independent

        p = np.array([1.0, np.nan, 3.0])
        t = np.array([1.0, 2.0, 3.0])

        ours = metrics(p, t)
        theirs = benchmark_independent.score(p, t)

        # 1. Same input, different answers: we score the survivors, it returns nan.
        assert ours["mae"] == 0.0 and ours["n"] == 2
        assert np.isnan(theirs["mae"]) and np.isnan(theirs["rmse"]) \
            and np.isnan(theirs["bias"])
        assert theirs["n"] == 3, (
            "benchmark_independent.score now masks; if that was fixed, the "
            "divergence finding is stale and should be rewritten, not deleted")

        # 2. The low-coverage case the gate refuses is a nan here, printed "n/a".
        #    1 of 3 rows is coverage 0.33 and metrics() declines to score it.
        #    score() has no gate, so it evaluates all three rows, the nan poisons
        #    the mean, and the table shows "n/a" -- which looks like a formatting
        #    outcome rather than a refusal to score. The n it reports alongside is
        #    the UNMASKED 3, so the row states it saw three windows when only one
        #    was finite. A reader cannot tell from that line whether the model
        #    scored badly or the cache was empty.
        thin_p = np.array([1.0, np.nan, np.nan])
        thin_t = np.array([1.0, 2.0, 3.0])
        assert metrics(thin_p, thin_t) is None
        thin = benchmark_independent.score(thin_p, thin_t)
        assert np.isnan(thin["mae"]) and np.isnan(thin["rmse"]) \
            and np.isnan(thin["bias"])
        assert thin["n"] == 3, (
            "benchmark_independent.score reports the unmasked row count, so its "
            "n overstates how much data was actually seen; if that changed, "
            "update this finding")
        # And it has no threshold to refuse at, so a caller cannot ask it to.
        assert not hasattr(benchmark_independent, "MIN_COVERAGE")

        # 3. And there is no field in which it could report coverage.
        assert set(theirs) == {"mae", "rmse", "bias", "ci95_mae", "n"}
        assert set(ours) == set(theirs) | {"coverage", "rows_dropped_nonfinite"}

    def test_the_scalar_mae_helpers_disagree_about_total_coverage_failure(self):
        """
        FINDING. Three ways of asking "what is the MAE", three different
        answers for the same empty-coverage input, and only one of them is the
        one the benchmark table uses:

            metrics(...)                 -> None      (caller must check)
            benchmark_independent.score  -> nan       (no check needed, prints n/a)
            refit_policy.mae(...)        -> +inf      (sorts last, compares fine)
            diagnose_headroom.mae(...)   -> (nan, nan)

        A refactor that swaps one for another changes what a "no data" run
        prints without changing any number that has a value.
        """
        import benchmark_independent
        import diagnose_headroom
        import refit_policy

        p = np.array([np.nan, np.nan])
        t = np.array([1.0, 2.0])

        assert metrics(p, t) is None
        assert np.isnan(benchmark_independent.score(p, t)["mae"])
        assert refit_policy.mae(p, t) == float("inf")
        dh = diagnose_headroom.mae(p, t)
        assert np.isnan(dh[0]) and np.isnan(dh[1])

        # On real data all four agree on the MAE itself: errors [0, 1, 3],
        # sum 4 over 3 rows.
        p2 = np.array([1.0, 2.0, 4.0])
        t2 = np.array([1.0, 1.0, 1.0])
        ours = metrics(p2, t2)["mae"]
        assert ours == 4.0 / 3.0
        assert refit_policy.mae(p2, t2) == pytest.approx(ours)
        assert diagnose_headroom.mae(p2, t2)[0] == pytest.approx(ours)
        assert benchmark_independent.score(p2, t2)["mae"] == pytest.approx(ours)

    def test_bias_sign_is_uniform_except_for_one_known_inversion(self):
        """
        FINDING, pinned in both directions, and NARROWER than it used to be.

        metrics() used to report mean(truth - pred) while five other scorers
        reported mean(pred - truth), so the same forecast read as
        over-predicting on the dashboard and under-predicting in the benchmark
        and triage chased the wrong end of the model. That is FIXED: metrics() is
        now pred - truth, which is the WMO convention and the one monitoring.py
        uses, so the two now agree and the 5-vs-2 split is a 6-vs-1 split.

        The one remaining inversion is benchmark_independent.py, and it is the
        awkward one: that script exists to be the INDEPENDENT check on the same
        test split, so the cross-check whose entire job is to catch a
        mis-specified metric is the metric most likely to be mis-specified. Its
        bias column and benchmark_vs_nwp's bias column currently mean opposite
        things, and its own docstring argues for the arrangement that caused it
        ("implemented here directly and transparently rather than imported, so
        their definition cannot drift").

        Not fixed here. Correcting it means re-deriving published numbers in a
        results file, which is a decision for whoever owns that output, not for
        a test. What this test does is make sure the inversion cannot be
        discovered by accident later, and cannot be quietly closed without
        someone updating the count.
        """
        by_file = {}
        for filename, lineno, funcname, keys in KNOWN_INDEPENDENT_SCORERS:
            if filename in BIAS_SIGN_BY_SCORER:
                by_file.setdefault(filename, (lineno, funcname))

        assert set(by_file) == set(BIAS_SIGN_BY_SCORER), (
            "the set of scorers with a known bias convention changed; "
            "someone added or removed one")

        for filename, sign in BIAS_SIGN_BY_SCORER.items():
            body = _function_source(filename, by_file[filename][1])
            for evidence in BIAS_SIGN_EVIDENCE[filename]:
                assert evidence in body, (
                    f"{filename}:{by_file[filename][1]} no longer contains "
                    f"{evidence!r}, so its bias convention ({sign}) is "
                    f"unverified. Re-read the function and update "
                    f"BIAS_SIGN_BY_SCORER.")

        truths = sorted(f for f, s in BIAS_SIGN_BY_SCORER.items()
                        if s == "truth-pred")
        preds = sorted(f for f, s in BIAS_SIGN_BY_SCORER.items()
                       if s == "pred-truth")

        # The one straggler, named rather than counted, so the assertion says
        # WHICH file is wrong and not merely how many are.
        assert truths == ["benchmark_independent.py"], (
            f"the truth-pred camp is now {truths}. metrics() was corrected to "
            f"pred-truth, so benchmark_independent.py should be the ONLY member. "
            f"If it was corrected, this finding is resolved -- update the "
            f"count and the test name rather than deleting the check.")
        assert len(preds) == 6, (
            f"the pred-truth camp is now {preds}, expected 6 (metrics plus the "
            f"five that already agreed with monitoring.py)")

        # The divergence is observable, not merely textual. A model predicting
        # 1.0 against observations of 0.0 is running WARM: positive under the
        # convention metrics() now uses, negative under the one
        # benchmark_independent.py still uses.
        warm_pred = [1.0, 1.0, 1.0, 1.0]
        warm_truth = [0.0, 0.0, 0.0, 0.0]
        assert metrics(warm_pred, warm_truth)["bias"] == 1.0
        import benchmark_independent
        assert benchmark_independent.score(warm_pred, warm_truth)["bias"] == -1.0, (
            "the two benchmark scripts now agree on the SIGN of bias; the "
            "remaining divergence must have been fixed, so update this finding")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
