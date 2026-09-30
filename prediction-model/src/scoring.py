"""
The canonical forecast-scoring function for this project.

WHY THIS EXISTS
---------------
There was no single scoring function. An audit of prediction-model/src found
thirteen separately-written functions returning a dict with an "mae" key, and
sixty-four raw `mean(|a - b|)` call sites. They did not agree with each other,
and the disagreements were not cosmetic -- they were four different answers to
"what happens when the data is thin", and two opposite sign conventions for the
single number triage reads most often:

    scorer                        thin data        bias convention
    ----------------------------- ---------------- -----------------
    metrics (the reference)       None             mean(pred - truth)
    benchmark_independent.score   NaN, n unmasked  mean(truth - pred)
    refit_policy.mae              +inf             n/a (scalar)
    diagnose_headroom.mae         (nan, nan)       mean(truth - pred)

A refactor that swaps one for another changes what a "no data" run prints
without changing any number that has a value. That is the failure mode this
module closes: one implementation, one set of semantics, imported by callers
rather than reimplemented by them.

THE CONTRACT
------------
`metrics(pred, truth, min_coverage=MIN_COVERAGE)` returns either None or a dict
with exactly these keys:

    mae                     mean(|pred - truth|) over the surviving rows
    rmse                    sqrt(mean((pred - truth)**2)), same rows
    bias                    mean(pred - truth)   <-- PREDICTED MINUS OBSERVED
    n                       number of surviving rows (the denominator)
    coverage                surviving / input rows, fraction of the input used
    rows_dropped_nonfinite  input rows dropped for missing data
    ci95_mae                [lo, hi] bootstrap 95% CI on the MAE

Four properties are load-bearing and are pinned by test_scoring.py:

1. BIAS SIGN. bias = mean(pred - truth), the WMO definition. POSITIVE means the
   model runs WARM (predicts above observation); NEGATIVE means it runs COLD.
   This was the opposite sign in the original benchmark, so the same forecast
   read as over-predicting on the operational dashboard (monitoring.py, which
   has always used pred - truth) and under-predicting in the benchmark, and
   somebody triaging a systematic bias from the dashboard would have worked on
   the wrong end of the model.

2. THE COVERAGE GATE. A score computed over 12% of the rows is not a score of
   the model, it is a score of whichever rows happened to be present, and it is
   indistinguishable from a full-coverage score by the number alone. That is
   not hypothetical: when NWP window selection picked a cached series that did
   not overlap the test split, every NWP lookup missed, coverage fell to zero,
   and the run printed `n/a` for all seven models while still emitting a rank.
   Below `min_coverage` this returns None rather than a confident number. The
   comparison is INCLUSIVE: coverage exactly equal to min_coverage is scored.

3. THE LOSS IS AUDITABLE. Non-finite rows are dropped, and what was dropped is
   reported rather than vanishing. `coverage` and `rows_dropped_nonfinite` are
   in the return value so a partial score is visibly partial, and
   n + rows_dropped_nonfinite always equals the input length.

4. THE DENOMINATOR IS THE SURVIVING COUNT, never the input length. Dividing by
   the unmasked length silently deflates every score in proportion to how much
   data is missing, i.e. it rewards whichever model has the worse coverage.

WHAT None MEANS
---------------
None is overloaded: it means both "nothing survived" (a total lookup failure,
the run that motivated the coverage gate) and "some survived but not enough"
(a marginal series that is mostly absent). These are different diagnoses and
the return value does not distinguish them; a caller that logs "scorer returned
None" cannot tell a broken cache from a thin one. That residual hazard is
pinned by a test rather than fixed, so it stays a known, deliberate gap.

The None guarantee is about missing DATA, not about malformed INPUT. `pred` and
`truth` must be the same length: a mismatch raises numpy's broadcast ValueError
rather than returning None, because a prediction series paired with a
differently-sized truth series is a bug at the call site, and truncating to the
shorter one would produce a confident score over rows that were never actually
paired. Callers build both sides from the same window list, so the lengths agree
by construction; this note exists so the raise is not a surprise.

This module is the single implementation. Import it; do not copy it. A caller
that needs different thin-data behaviour -- a scalar MAE that returns +inf so
it sorts last, say -- should wrap `metrics` explicitly at the call site so the
divergence is visible where it is introduced.
"""

import numpy as np

__all__ = [
    "MIN_COVERAGE",
    "BOOTSTRAP_CI",
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "SCORE_KEYS",
    "bootstrap_ci",
    "metrics",
]

# ---------------------------------------------------------------------------
# Contract constants. These are part of the published output, not tunables:
# changing any of them moves numbers that are already written down.
# ---------------------------------------------------------------------------

#: Minimum fraction of input rows that must survive masking for a score to be
#: published at all. Half the rows is the line: a score over more of the input
#: is a score of the model, a score over less is a score of whichever rows
#: happened to survive. The gate is INCLUSIVE at exactly this value.
MIN_COVERAGE = 0.5

#: Bootstrap settings for ci95_mae. These pin every published confidence
#: interval, so they are frozen rather than derived.
BOOTSTRAP_CI = 0.95
BOOTSTRAP_RESAMPLES = 400
BOOTSTRAP_SEED = 42

#: The exact key set a non-None result carries. Consumers index these, so a
#: partially-successful score that is missing one of them is a type error at
#: the call site rather than a silently-absent metric.
SCORE_KEYS = frozenset({
    "mae",
    "rmse",
    "bias",
    "n",
    "coverage",
    "rows_dropped_nonfinite",
    "ci95_mae",
})


def bootstrap_ci(err, n_boot=BOOTSTRAP_RESAMPLES, ci=BOOTSTRAP_CI,
                 seed=BOOTSTRAP_SEED):
    """
    Percentile bootstrap CI for the MEAN of `err`.

    Resamples with replacement, n_boot times, and returns the 2.5th/97.5th
    percentiles of the resampled means. Returns [None, None] for fewer than
    two observations, because a bootstrap over one point has no spread and
    reporting [x, x] would present a degenerate interval as a real one.
    """
    rng = np.random.default_rng(seed)
    n = len(err)
    if n < 2:
        return [None, None]
    idx = rng.integers(0, n, (n_boot, n))
    m = err[idx].mean(axis=1)
    return [float(np.percentile(m, 100 * (1 - ci) / 2)),
            float(np.percentile(m, 100 * (1 + ci) / 2))]


def metrics(pred, truth, min_coverage=MIN_COVERAGE):
    """
    MAE / RMSE / bias over the rows where BOTH prediction and truth are finite.

    Returns None -- not a number, not a raise -- when the surviving rows fall
    below `min_coverage`, so a caller cannot accidentally publish a score of a
    mostly-missing series. Callers must handle None explicitly.

    Non-finite means nan and either infinity, on either side: a gap in the
    observation is not a model error and a gap in the model is not an
    observation error, so both are dropped symmetrically and both are counted
    in `rows_dropped_nonfinite`.

    `bias` is PREDICTED MINUS OBSERVED, per WMO. Positive means the model runs
    warm, negative means it runs cold. See the module docstring.

    See the module docstring for the full contract; every property there is
    pinned by test_scoring.py and by test_scoring_golden_fixtures.py.
    """
    p_all = np.asarray(pred, float)
    t_all = np.asarray(truth, float)
    finite = np.isfinite(p_all) & np.isfinite(t_all)
    total = len(p_all)
    kept = int(finite.sum())
    coverage = kept / total if total else 0.0
    if kept == 0:
        return None
    p, t = p_all[finite], t_all[finite]
    # INCLUSIVE gate: coverage exactly equal to min_coverage is scored.
    if coverage < min_coverage:
        return None
    e = np.abs(p - t)
    return {"mae": float(e.mean()), "rmse": float(np.sqrt((e ** 2).mean())),
            "bias": float((p - t).mean()), "n": int(len(p)),
            "coverage": coverage,
            "rows_dropped_nonfinite": total - kept,
            "ci95_mae": bootstrap_ci(e)}
