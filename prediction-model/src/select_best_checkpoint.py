"""
select_best_checkpoint.py -- choose a checkpoint using seed-variance statistics,
not "the one with the lowest MAE".

THE PROBLEM THIS EXISTS TO FIX
------------------------------
In this system seed variance is larger than every architecture delta measured so
far. Wind MAE at +6h spanned 0.598 -> 0.725 across three seeds on IDENTICAL data
-- a 21% swing from initialisation alone -- while every architecture comparison
this cycle produced 2-8% deltas. A single-seed "winner" is therefore, with high
probability, the luckiest draw rather than the better model. This module exists
to make that statement measurable instead of rhetorical.

WHAT IT REPORTS
---------------
For every (arm, seed, horizon, channel):

  * MEAN across seeds, sd (ddof=1), standard error of the mean, 95% CI of the mean
  * BEST single seed, reported SEPARATELY from the mean, plus the observed gap
  * the EXPECTED gap between best-of-N and the mean, and therefore the size of
    the optimistic bias incurred by shipping the best seed
  * a bias-corrected ("debiased") arm estimate: best + sd * E[max of N normals]

Comparing two arms it runs a seed-matched PAIRED test (or Welch when the arms
have no seeds in common) against the same 2-sigma noise band the release gate
uses, and answers one question explicitly:

    is this difference distinguishable from seed noise?

One seed on either side returns INSUFFICIENT_EVIDENCE -- never a comparison.

FOUR GUARDS ON TOP OF THE GATE
------------------------------
Each of these exists because the gate alone would have produced a wrong answer
on the data in this repository:

1. MATERIALITY. The 2-sigma band is a WEAK gate, not a conservative one: its
   halfwidth is sigma * sd_of_differences / sqrt(n_pairs) and sd_of_differences
   has only n_pairs - 1 degrees of freedom. At n_pairs = 2 the band collapses and
   a difference of 0.0001 MAE clears it. A difference is reported as actionable
   only if it is BOTH outside the noise band AND above a materiality floor.
2. UNMATCHED SEEDS. When the arms carry different numbers of seeds, the all-seed
   gap can be carried entirely by a seed only one arm has. That is flagged
   explicitly, because a 14.5% all-seed gap that collapses to 0.75% on matched
   seeds is the most dangerous shape of result this tool can encounter.
3. THE EXACT SIGN TEST. An exact two-sided sign test over N seed pairs has a
   smallest attainable p of 2 * 0.5**N, so 3 pairs can never reach p < 0.25 and
   you need 6 pairs before an assumption-free test can reject at all. Below that
   floor the parametric verdict is annotated as resting entirely on normality.
4. THE EVALUATION ROW SET. A paired difference is a statement about the same rows
   measured two ways. If the two arms were scored on different numbers of rows
   the difference is not noisy, it is meaningless, so the comparison is refused.

It also checks the chronological half-split the release gate requires: a mean
difference that reverses sign between the two halves of the test split is a
property of the window, not of the model.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
It never trains, never calls a promotion gate, and never reuses
`train_predictive_quality.py`'s incumbent comparison. It answers "does the seed
explain this difference", which is a different and prior question. Promotion
against production is a separate decision on a separate comparison.

DEPENDENCIES
------------
The statistics core is pure standard library so it is cheap, deterministic and
testable in isolation. `torch`/`numpy` are imported lazily and ONLY by the
opt-in `--allow-recompute` path, which runs inference over checkpoints that
already exist. It never trains. Re-scoring costs real CPU, so its output is
cacheable with `--recompute-cache` and the statistics can be re-derived from the
cache without paying for inference again.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(ROOT, "prediction-model", "data")

DEFAULT_SWEEP_DIR = os.path.join(DATA, "seed_sweep")
DEFAULT_OUT = os.path.join(DATA, "checkpoint_selection.json")
DEFAULT_WEATHER_CSV = os.path.join(DATA, "weather_telemetry_current.csv")

HORIZONS: Tuple[int, ...] = (1, 3, 6, 12, 24)
CHANNELS: Tuple[str, ...] = ("temperature", "humidity", "pressure", "wind_speed")
ORIGIN_COLS = ("origin_temperature", "origin_humidity", "origin_pressure",
               "origin_wind_speed", "origin_wind_u", "origin_wind_v")

# The release gate's own wording: a difference must clear 2 sigma to count.
DEFAULT_SIGMA = 2.0
DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.8
# A difference can clear the 2-sigma gate and still be far too small to matter.
# 2% is the floor of the 2-8% band every architecture comparison this cycle
# produced, so anything under it was never worth chasing.
DEFAULT_MATERIALITY_PCT = 2.0

DISTINGUISHABLE = "DISTINGUISHABLE"
NOT_DISTINGUISHABLE = "NOT_DISTINGUISHABLE"
INSUFFICIENT = "INSUFFICIENT_EVIDENCE"

# Closed forms for the expected maximum of N standard normals, used so the
# bias figure is exact for the small-N cases that are actually being run.
_EMAX_EXACT = {
    1: 0.0,
    2: 1.0 / math.sqrt(math.pi),                 # 0.5641895835...
    3: 3.0 / (2.0 * math.sqrt(math.pi)),         # 0.8462843753...
}


# --------------------------------------------------------------------------
# Distributions (standard library only, so every number is checkable by hand)
# --------------------------------------------------------------------------

def norm_cdf(x: float) -> float:
    """P(Z <= x) for a standard normal."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _horner(coeffs: Sequence[float], r: float) -> float:
    """Evaluate a0*r^n + a1*r^(n-1) + ... + an, least-significant term first."""
    acc = 0.0
    for c in reversed(coeffs):
        acc = acc * r + c
    return acc


# Wichura AS241 rational approximation coefficients, in Horner order
# (index 0 is the constant term, index 7 multiplies r**7).
_AS241_CENTRAL_NUM = (3.3871328727963666080e0, 1.3314166789178437745e+2,
                      1.9715909503065514427e+3, 1.3731693765094611250e+4,
                      4.5921953931549981457e+4, 6.7265770927008700853e+4,
                      3.3430575583588128105e+4, 2.5090809287301226727e+3)
_AS241_CENTRAL_DEN = (1.0, 4.2313330701600911252e+1, 6.8718700749205790830e+2,
                      5.3941960214247511077e+3, 2.1213794301585958670e+4,
                      3.9307895800092710610e+4, 2.8729085735721942674e+4,
                      5.2264952788528545610e+3)
_AS241_TAIL_NUM = (1.42343711074968357734e0, 4.6303378461565452959e0,
                   5.7694972214606914055e0, 3.64784832476320460504e0,
                   1.27045825245236838258e0, 2.41780725177450611770e-1,
                   2.27238449892691845833e-2, 7.74545014278341407640e-4)
_AS241_TAIL_DEN = (1.0, 2.05319162663775882187e0, 1.67638483018380384940e0,
                   6.89767334985100004550e-1, 1.48103976427480074590e-1,
                   1.51986665636164571966e-2, 5.47593808499534494600e-4,
                   1.05075007164441684324e-9)


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Wichura AS241, ~1e-15 absolute error)."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"norm_ppf requires 0 < p < 1, got {p}")
    q = p - 0.5
    if abs(q) <= 0.425:
        r = 0.180625 - q * q
        return q * _horner(_AS241_CENTRAL_NUM, r) / _horner(_AS241_CENTRAL_DEN, r)

    r = p if q < 0.0 else 1.0 - p
    r = math.sqrt(-math.log(r))
    r = r - 1.6 if r <= 5.0 else r - 5.0
    val = _horner(_AS241_TAIL_NUM, r) / _horner(_AS241_TAIL_DEN, r)
    return -val if q < 0.0 else val


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (modified Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 301):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-16:
            break
    return h


def betai(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log1p(-x))
    front = math.exp(lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t: float, df: float) -> float:
    """Two-sided p-value for Student's t with `df` degrees of freedom."""
    if df <= 0:
        return float("nan")
    t = abs(float(t))
    if math.isinf(t):
        return 0.0
    return betai(0.5 * df, 0.5, df / (df + t * t))


def t_critical_two_sided(alpha: float, df: float) -> float:
    """Smallest t with two-sided p == alpha, by bisection on `t_two_sided_p`."""
    if df <= 0:
        return float("nan")
    lo, hi = 0.0, 1.0
    while t_two_sided_p(hi, df) > alpha:
        hi *= 2.0
        if hi > 1e6:
            return float("inf")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_two_sided_p(mid, df) > alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def exact_sign_test_p(n_pairs: int, positives: int) -> float:
    """
    Exact two-sided sign test.

    This is the assumption-free floor on evidence. With N seed pairs the
    smallest attainable two-sided p is 2 * 0.5**N, because every pair agreeing
    in one direction is the only way to be significant.
    """
    if n_pairs <= 0:
        return 1.0
    k = min(positives, n_pairs - positives)
    tail = sum(math.comb(n_pairs, i) for i in range(k + 1)) / (2.0 ** n_pairs)
    return min(1.0, 2.0 * tail)


def min_pairs_for_exact_sign_test(alpha: float = DEFAULT_ALPHA) -> int:
    """Smallest N of seed pairs for which the exact sign test can reject."""
    n = 1
    while n < 10_000:
        if exact_sign_test_p(n, 0) < alpha:
            return n
        n += 1
    return n  # pragma: no cover


# --------------------------------------------------------------------------
# Selection bias: how far below the mean does the best of N seeds sit?
# --------------------------------------------------------------------------

def expected_max_of_n_normals(n: int, intervals: int = 2000) -> float:
    """
    E[max of N iid standard normals].

    Closed forms are used for N in {1, 2, 3} because those are the N actually
    being run and they are exact. For N >= 4 the expectation is integrated as

        E[max] = N * integral_{-inf}^{inf} x * phi(x) * Phi(x)^(N-1) dx

    by composite Simpson over [-10, 10] (the integrand is 0 for all x <= 0
    after that truncation and below 1e-12 elsewhere at N = 2000).
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if n in _EMAX_EXACT:
        return _EMAX_EXACT[n]

    def integrand(x: float) -> float:
        phi = math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
        return n * x * phi * norm_cdf(x) ** (n - 1)

    lo, hi = -10.0, 10.0
    if intervals % 2:
        intervals += 1
    step = (hi - lo) / intervals
    total = integrand(lo) + integrand(hi)
    for i in range(1, intervals):
        total += (4.0 if i % 2 else 2.0) * integrand(lo + i * step)
    return total * step / 3.0


def expected_selection_bias(sd: float, n: int) -> float:
    """
    Expected amount by which the best of N seeds understates the arm's true mean,
    for a lower-is-better metric such as MAE.

    Under normality this is sd * E[max of N standard normals]. It is a property
    of the sweep, not of the model: the best seed is guaranteed to look better
    than the architecture that produced it.
    """
    if sd is None or n < 1:
        return 0.0
    return sd * expected_max_of_n_normals(n)


# --------------------------------------------------------------------------
# Basic descriptive statistics
# --------------------------------------------------------------------------

def mean(xs: Sequence[float]) -> Optional[float]:
    return float(sum(xs) / len(xs)) if xs else None


def sample_sd(xs: Sequence[float]) -> Optional[float]:
    """Sample standard deviation with Bessel's correction (ddof=1)."""
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def standard_error(xs: Sequence[float]) -> Optional[float]:
    """Standard error of the mean. None when n < 2 (the spread is unknown)."""
    sd = sample_sd(xs)
    return None if sd is None else sd / math.sqrt(len(xs))


def _coerce(values_by_seed: Dict[object, float]) -> Dict[int, float]:
    return {int(k): float(v) for k, v in values_by_seed.items()}


def seed_summary(values_by_seed: Dict[object, float],
                 lower_is_better: bool = True,
                 alpha: float = DEFAULT_ALPHA) -> Dict:
    """
    Everything a selection decision needs about ONE arm at ONE (horizon, channel).

    The best seed and the mean are returned as separate fields on purpose. They
    are different quantities; reporting the best seed as if it were the arm's
    performance is how a lucky draw gets promoted.
    """
    by_seed = _coerce(values_by_seed)
    seeds = sorted(by_seed)
    vals = [by_seed[s] for s in seeds]
    n = len(vals)

    out: Dict = {
        "n_seeds": n,
        "seeds": seeds,
        "values_by_seed": {str(s): by_seed[s] for s in seeds},
        "lower_is_better": lower_is_better,
    }

    if n == 0:
        out.update({"status": INSUFFICIENT, "reason": "no seeds present"})
        return out

    m = mean(vals)
    sd = sample_sd(vals)
    sem = standard_error(vals)
    best_seed = min(seeds, key=lambda s: by_seed[s]) if lower_is_better \
        else max(seeds, key=lambda s: by_seed[s])
    best = by_seed[best_seed]

    out.update({
        "mean": m,
        "sd": sd,
        "sem": sem,
        "min": min(vals),
        "max": max(vals),
        "range": max(vals) - min(vals),
        "best_seed": best_seed,
        "best_value": best,
        "cv_pct_of_mean": None if (sd is None or not m) else 100.0 * sd / abs(m),
    })

    if sd is None:
        # One seed: the number is real, the spread is not measurable. Report the
        # number, refuse the summary.
        out.update({
            "status": INSUFFICIENT,
            "reason": (f"only 1 seed; mean and best are the same observation, "
                       f"sd and sem are undefined, and a single seed carries an "
                       f"unquantified optimistic bias"),
        })
        return out

    tcrit = t_critical_two_sided(alpha, n - 1)
    half = tcrit * sem
    emax = expected_max_of_n_normals(n)
    expected_gap = expected_selection_bias(sd, n)
    observed_gap = (m - best) if lower_is_better else (best - m)

    out.update({
        "status": "OK",
        "ci95_of_mean": [m - half, m + half],
        "t_critical": tcrit,
        "best_vs_mean_gap": observed_gap,
        "best_vs_mean_gap_pct": 100.0 * observed_gap / abs(m) if m else None,
        "expected_best_vs_mean_gap": expected_gap,
        "expected_best_vs_mean_gap_pct": 100.0 * expected_gap / abs(m) if m else None,
        "expected_max_of_n_normals": emax,
        "observed_over_expected_gap": (observed_gap / expected_gap
                                       if expected_gap > 0 else None),
        "debiased_arm_estimate": (best + expected_gap) if lower_is_better
                                 else (best - expected_gap),
    })
    return out


# --------------------------------------------------------------------------
# Comparing two arms
# --------------------------------------------------------------------------

def _verdict(mean_diff: float, halfwidth: float) -> str:
    if halfwidth is None or math.isnan(halfwidth):
        return INSUFFICIENT
    return DISTINGUISHABLE if abs(mean_diff) > halfwidth else NOT_DISTINGUISHABLE


def paired_comparison(a_by_seed: Dict[object, float],
                      b_by_seed: Dict[object, float],
                      sigma: float = DEFAULT_SIGMA,
                      alpha: float = DEFAULT_ALPHA,
                      lower_is_better: bool = True) -> Dict:
    """
    Seed-matched paired comparison.

    Pairing is the right test here because both arms see the same corpus, the
    same split and the same seeds; the seed-to-seed nuisance variation cancels
    and only the architecture change remains in the differences.
    """
    a, b = _coerce(a_by_seed), _coerce(b_by_seed)
    common = sorted(set(a) & set(b))
    out: Dict = {"test": "paired", "n_pairs": len(common), "common_seeds": common,
                 "sigma": sigma, "alpha": alpha}

    if len(common) < 2:
        out.update({
            "status": INSUFFICIENT,
            "reason": (f"paired test needs >= 2 seed-matched pairs; got "
                       f"{len(common)} (arm A seeds {sorted(a)}, "
                       f"arm B seeds {sorted(b)})"),
            "verdict": INSUFFICIENT,
        })
        return out

    diffs = [a[s] - b[s] for s in common]
    n = len(diffs)
    m = mean(diffs)
    sd = sample_sd(diffs)
    sem = sd / math.sqrt(n)
    halfwidth = sigma * sem
    t = m / sem if sem > 0 else (math.inf if m > 0 else -math.inf)
    tcrit = t_critical_two_sided(alpha, n - 1)
    positives = sum(1 for d in diffs if d > 0)
    negatives = sum(1 for d in diffs if d < 0)
    sign_p = exact_sign_test_p(n, positives)

    better = "A" if (m < 0) == lower_is_better else "B"
    mean_a = mean([a[s] for s in common])
    mean_b = mean([b[s] for s in common])
    scale = abs(mean([a[s] for s in common] + [b[s] for s in common]))

    out.update({
        "status": "OK",
        "differences_by_seed": {str(s): a[s] - b[s] for s in common},
        "mean_difference": m,                 # negative => A lower => A better
        # Means over the seeds the TEST actually used. When the arms are
        # unbalanced these differ from the full arm means, and conflating the
        # two is how a 0.34 gap gets reported as 0.016.
        "mean_a_tested": mean_a,
        "mean_b_tested": mean_b,
        "relative_difference_pct": (100.0 * m / scale) if scale else None,
        "sd_of_differences": sd,
        "sem_of_difference": sem,
        "two_sigma_halfwidth": halfwidth,
        "ci95_of_difference": [m - tcrit * sem, m + tcrit * sem],
        "t": t,
        "df": n - 1,
        "p_value_parametric": t_two_sided_p(t, n - 1),
        "p_value_exact_sign_test": sign_p,
        "min_attainable_sign_test_p": exact_sign_test_p(n, 0),
        "sign_test_can_reject": exact_sign_test_p(n, 0) < alpha,
        "cohens_dz": (m / sd) if sd > 0 else None,
        "sign_agreement_pct": 100.0 * max(positives, negatives) / n,
        "better_arm": better,
        "verdict": _verdict(m, halfwidth),
        "answer": ("the difference is distinguishable from seed noise"
                   if abs(m) > halfwidth else
                   "the difference is NOT distinguishable from seed noise"),
    })
    if n < min_pairs_for_exact_sign_test(alpha):
        out["caveat"] = (
            f"n_pairs={n} leaves {n - 1} degree(s) of freedom, and an exact sign "
            f"test over {n} pairs can never reach p<{alpha} (its floor is "
            f"{exact_sign_test_p(n, 0):.4f}). This verdict therefore rests "
            f"entirely on the normality assumption of a t test on {n - 1} df, "
            f"and the two tests disagree by construction. Treat it as a "
            f"provisional number, not a decision.")
    return out


def welch_comparison(a_by_seed: Dict[object, float],
                     b_by_seed: Dict[object, float],
                     sigma: float = DEFAULT_SIGMA,
                     alpha: float = DEFAULT_ALPHA,
                     lower_is_better: bool = True) -> Dict:
    """Two-sample Welch comparison, for arms that share no seed."""
    a, b = _coerce(a_by_seed), _coerce(b_by_seed)
    na, nb = len(a), len(b)
    out: Dict = {"test": "welch", "n_a": na, "n_b": nb,
                 "seeds_a": sorted(a), "seeds_b": sorted(b),
                 "sigma": sigma, "alpha": alpha}

    if na < 2 or nb < 2:
        out.update({
            "status": INSUFFICIENT,
            "reason": (f"two-sample test needs >= 2 seeds per arm; got "
                       f"{na} and {nb}"),
            "verdict": INSUFFICIENT,
        })
        return out

    va, vb = [a[s] for s in sorted(a)], [b[s] for s in sorted(b)]
    ma, mb = mean(va), mean(vb)
    sda, sdb = sample_sd(va), sample_sd(vb)
    se = math.sqrt(sda ** 2 / na + sdb ** 2 / nb)
    m = ma - mb
    halfwidth = sigma * se
    t = m / se if se > 0 else (math.inf if m > 0 else -math.inf)
    num = se ** 4
    den = ((sda ** 2 / na) ** 2 / (na - 1)) + ((sdb ** 2 / nb) ** 2 / (nb - 1))
    df = num / den if den > 0 else float(na + nb - 2)

    out.update({
        "status": "OK",
        "mean_a": ma, "mean_b": mb,
        "mean_difference": m,
        "sd_a": sda, "sd_b": sdb,
        "standard_error": se,
        "two_sigma_halfwidth": halfwidth,
        "ci95_of_difference": [m - t_critical_two_sided(alpha, df) * se,
                               m + t_critical_two_sided(alpha, df) * se],
        "t": t, "df": df,
        "p_value_parametric": t_two_sided_p(t, df),
        "verdict": _verdict(m, halfwidth),
        "answer": ("the difference is distinguishable from seed noise"
                   if abs(m) > halfwidth else
                   "the difference is NOT distinguishable from seed noise"),
    })
    return out


def compare_arms(a_by_seed: Dict[object, float],
                 b_by_seed: Dict[object, float],
                 sigma: float = DEFAULT_SIGMA,
                 alpha: float = DEFAULT_ALPHA,
                 lower_is_better: bool = True) -> Dict:
    """Paired where seeds match, Welch where they do not, INSUFFICIENT otherwise."""
    a, b = _coerce(a_by_seed), _coerce(b_by_seed)
    n_common = len(set(a) & set(b))
    if n_common >= 2:
        res = paired_comparison(a, b, sigma, alpha, lower_is_better)
    else:
        res = welch_comparison(a, b, sigma, alpha, lower_is_better)
        if res.get("status") == INSUFFICIENT:
            res["design"] = (f"arms share only {n_common} seed(s); a paired test "
                             f"is impossible and a two-sample test needs >= 2 each")
    # Normalise the shape so a caller never has to know which test ran.
    res["n_common_seeds"] = n_common
    res.setdefault("n_pairs", n_common)
    res.setdefault("n_seeds_a", len(a))
    res.setdefault("n_seeds_b", len(b))
    # When the arms carry different numbers of seeds the paired test silently
    # discards the unmatched ones. Say so, rather than letting the reader assume
    # the tested means describe the whole arms.
    res["arms_unbalanced"] = len(a) != len(b)
    res["n_seeds_a_total"] = len(a)
    res["n_seeds_b_total"] = len(b)
    return res


# --------------------------------------------------------------------------
# How many seeds are needed before a comparison should be believed at all
# --------------------------------------------------------------------------

def required_seeds_paired(sd_of_difference: float, effect: float,
                          alpha: float = DEFAULT_ALPHA,
                          power: float = DEFAULT_POWER) -> int:
    """N seed pairs for a two-sided paired t-test at the given power."""
    if effect <= 0 or sd_of_difference is None:
        return 0
    z = norm_ppf(1.0 - alpha / 2.0) + norm_ppf(power)
    return max(2, int(math.ceil((z * sd_of_difference / effect) ** 2)))


def required_seeds_two_sample(sd_a: float, sd_b: float, effect: float,
                              n_a: int, alpha: float = DEFAULT_ALPHA,
                              power: float = DEFAULT_POWER,
                              cap: int = 100_000) -> Optional[int]:
    """
    n_b for a two-sided two-sample z-test at the given power, given a fixed n_a.

    This is a normal approximation, used only for planning how many seeds to run
    and never to decide a result. It returns None when the requested power is
    UNREACHABLE no matter how large n_b gets: with n_a fixed, the standard error
    has a floor of sd_a / sqrt(n_a), so a small n_a cannot resolve a small effect
    at high power no matter how many seeds the other arm has. Reporting a number
    there would be a lie; reporting the impossibility is the finding.
    """
    if effect <= 0 or sd_a is None or sd_b is None or n_a < 2:
        return None
    if sd_a == 0.0 and sd_b == 0.0:
        # No seed variance at all: the two arms differ by a constant, so 2 seeds
        # already establish the difference exactly and more seeds add nothing.
        return 2
    z = norm_ppf(1.0 - alpha / 2.0) + norm_ppf(power)
    for n_b in range(2, cap + 1):
        se = math.sqrt(sd_a ** 2 / n_a + sd_b ** 2 / n_b)
        if se <= 0.0:
            return 2
        if (norm_cdf(effect / se - z) + norm_cdf(-effect / se - z)) >= power:
            return n_b
    return None


def min_seeds_recommendation(seed_sd: float,
                             arm_mean: float,
                             effects_pct: Sequence[float] = (2.0, 5.0, 10.0),
                             alpha: float = DEFAULT_ALPHA,
                             power: float = DEFAULT_POWER) -> Dict:
    """
    The minimum number of seeds before an architecture comparison should be
    believed, stated as a function of how large an effect would have to be to
    matter, plus the hard floor imposed by the exact sign test.
    """
    sign_floor = min_pairs_for_exact_sign_test(alpha)
    out: Dict = {
        "sign_test_floor_pairs": sign_floor,
        "sign_test_explanation": (
            f"an exact two-sided sign test over N seed pairs has a smallest "
            f"attainable p of 2*0.5**N, so N={sign_floor} is the first N that can "
            f"reach p<{alpha} at all. N=2 -> p>=0.50, N=3 -> p>=0.25, "
            f"N=5 -> p>=0.0625. No parametric model rescues a smaller N."),
        "by_effect_size": [],
    }

    if seed_sd is None or not arm_mean:
        out["status"] = INSUFFICIENT
        out["reason"] = "sd or mean unavailable; cannot size the sweep"
        return out

    sd = seed_sd
    out["seed_sd"] = sd
    out["arm_mean"] = arm_mean
    out["seed_cv_pct"] = 100.0 * sd / abs(arm_mean)

    for pct in effects_pct:
        effect = abs(arm_mean) * pct / 100.0
        # Paired design: the difference sd is the seed sd when the two arms are
        # seed-matched, which is the design this repo's sweep produces.
        n_paired = required_seeds_paired(sd, effect, alpha, power)
        n_two = required_seeds_two_sample(sd, sd, effect, 3, alpha, power)
        out["by_effect_size"].append({
            "effect_pct_of_mean": pct,
            "effect_absolute": effect,
            "seeds_for_paired_t": n_paired,
            "seeds_per_arm_for_two_sample": n_two,
            "two_sample_attainable": n_two is not None,
            "attainable_by_exact_sign_test": n_paired >= sign_floor,
        })

    # The recommended floor: enough seeds to detect a 5% delta (the middle of
    # the 2-8% band every architecture comparison this cycle produced) AND at
    # least enough for the assumption-free test to be capable of rejecting.
    mid = next((e for e in out["by_effect_size"] if e["effect_pct_of_mean"] == 5.0),
               out["by_effect_size"][min(1, len(out["by_effect_size"]) - 1)])
    out["recommended_min_seeds"] = max(3, mid["seeds_for_paired_t"], sign_floor)
    out["recommended_min_seeds_basis"] = (
        f"paired test at 80% power for a 5% effect at the observed seed sd "
        f"({sd:.4g}) needs {mid['seeds_for_paired_t']} pairs; the exact sign test "
        f"needs {sign_floor}; 3 is the smallest N with a meaningful sd at all, "
        f"so the floor is max of the two")
    out["status"] = "OK"
    return out


# --------------------------------------------------------------------------
# Chronological half-split consistency (part of the release gate)
# --------------------------------------------------------------------------

def half_split_consistency(a_halves: Dict[str, Dict[object, float]],
                           b_halves: Dict[str, Dict[object, float]],
                           lower_is_better: bool = True) -> Dict:
    """
    Do the two arms order the same way in both chronological halves of the test
    split? A mean difference that reverses sign between halves is a coincidence
    of the window, not an effect of the architecture.
    """
    halves = sorted(set(a_halves) & set(b_halves))
    out: Dict = {"halves": halves, "per_half": {}}

    if len(halves) < 2:
        out.update({"status": INSUFFICIENT,
                    "reason": "fewer than two chronological halves available"})
        return out

    signs = []
    for h in halves:
        a, b = _coerce(a_halves[h]), _coerce(b_halves[h])
        if not a or not b:
            out["per_half"][h] = {"status": INSUFFICIENT, "reason": "empty half"}
            continue
        ma, mb = mean(list(a.values())), mean(list(b.values()))
        d = ma - mb
        signs.append(1 if d > 0 else (-1 if d < 0 else 0))
        out["per_half"][h] = {
            "status": "OK",
            "n_seeds_a": len(a), "n_seeds_b": len(b),
            "mean_a": ma, "mean_b": mb, "mean_difference": d,
        }

    live = [s for s in signs if s != 0]
    if len(live) < 2:
        out.update({"status": INSUFFICIENT,
                    "reason": "at least one half had no data"})
        return out

    consistent = all(s == live[0] for s in live)
    out.update({
        "status": "OK",
        "signs": signs,
        "consistent": consistent,
        "verdict": "CONSISTENT_ACROSS_HALVES" if consistent
                   else "INCONSISTENT_ACROSS_HALVES",
        "note": ("the same arm wins both chronological halves"
                 if consistent else
                 "the arm ordering REVERSES between chronological halves; the "
                 "mean difference is a property of the window, not the model"),
    })
    return out


# --------------------------------------------------------------------------
# Data loading -- scorecards first, checkpoints only on explicit request
# --------------------------------------------------------------------------

def discover_arms(sweep_dir: str) -> Dict[str, Dict]:
    """
    Map floor -> seed -> arm record, and record what each arm actually has.

    An arm is a directory named "<floor>_s<seed>". A directory existing is NOT
    evidence that the arm is usable: a sweep that is still running leaves
    partial arms behind, and a partial arm is a trap for any tool that counts
    directories instead of completed artefacts.
    """
    arms: Dict[str, Dict] = {}
    if not os.path.isdir(sweep_dir):
        return arms
    for tag in sorted(os.listdir(sweep_dir)):
        full = os.path.join(sweep_dir, tag)
        if not os.path.isdir(full):
            continue
        floor, sep, seed_txt = tag.rpartition("_s")
        if not sep or not floor or not seed_txt.isdigit():
            continue
        checkpoints = {h: os.path.exists(os.path.join(full, f"candidate_h{h}h.pt"))
                       for h in HORIZONS}
        scorecard = os.path.join(full, "predictive_quality_scorecard.json")
        has_scorecard = os.path.exists(scorecard)
        code_commits = set()
        for h in HORIZONS:
            man = os.path.join(full, f"candidate_h{h}h_manifest.json")
            if os.path.exists(man):
                try:
                    with open(man, encoding="utf-8") as f:
                        code_commits.add(json.load(f).get("code_commit", ""))
                except (ValueError, OSError):
                    pass
        n_ck = sum(checkpoints.values())
        arms.setdefault(floor, {})[int(seed_txt)] = {
            "arm": tag,
            "floor": floor,
            "seed": int(seed_txt),
            "dir": full,
            "checkpoints": checkpoints,
            "n_checkpoints": n_ck,
            "has_scorecard": has_scorecard,
            "scorecard": scorecard if has_scorecard else None,
            # Usable for cross-seed statistics without re-running inference.
            "usable": has_scorecard,
            # Usable for anything that needs all five horizons, e.g. the arms
            # exist as directories but a partial arm cannot be compared at +24h.
            "complete": n_ck == len(HORIZONS),
            "code_commits": sorted(c for c in code_commits if c),
        }
    return arms


def load_scorecard_mae(sweep_dir: str,
                       channels: Sequence[str] = CHANNELS) -> Dict[str, Dict[int, Dict[str, float]]]:
    """
    Per-(floor, seed) -> horizon -> channel -> MAE, read from each arm's
    predictive quality scorecard. No torch, no inference, no training.
    """
    out: Dict[str, Dict[int, Dict[str, float]]] = {}
    for floor, seeds in discover_arms(sweep_dir).items():
        for seed, rec in sorted(seeds.items()):
            if not rec["has_scorecard"]:
                continue
            try:
                with open(rec["scorecard"], encoding="utf-8") as f:
                    card = json.load(f)
            except (ValueError, OSError):
                continue
            per_h: Dict[int, Dict[str, float]] = {}
            for key, block in (card.get("horizon_evaluations") or {}).items():
                try:
                    h = int(key.rsplit("_", 1)[1].rstrip("h"))
                except (IndexError, ValueError):
                    continue
                cand = block.get("candidate_featured_model") or {}
                # Build the row first: a horizon whose candidate block is absent
                # or carries none of the requested channels must not leave an
                # empty husk behind for a downstream consumer to trip over.
                row = {}
                for ch in channels:
                    val = (cand.get(ch) or {}).get("mae")
                    if isinstance(val, (int, float)) and not math.isnan(val):
                        row[ch] = float(val)
                if row:
                    per_h[h] = row
            if not per_h:
                continue
            # Only materialise the floor once a seed actually contributes rows,
            # so a floor with no readable scorecard cannot be mistaken for a
            # floor that merely has no data.
            out.setdefault(floor, {})[seed] = per_h
    return out


def recompute_from_checkpoints(sweep_dir: str,
                               weather_csv: str = DEFAULT_WEATHER_CSV,
                               channels: Sequence[str] = CHANNELS,
                               horizons: Sequence[int] = HORIZONS,
                               verbose: bool = False) -> Dict[str, Dict[int, Dict]]:
    """
    Re-score EXISTING checkpoints on the test split, overall and split into the
    two chronological halves of that split.

    This is inference over artefacts that already exist. It never trains. It is
    opt-in because it needs torch and CPU, and because it is only needed to fill
    in arms whose scorecard was never written, or to obtain the half-split
    numbers the release gate asks for -- neither of which the scorecards carry.
    """
    import numpy as np
    import torch
    from model import GarciaWeatherLNNFeatured
    from dataset import (TelemetryDataPipeline, build_forecast_windows,
                         build_feature_augmented_forecast_windows, DEFAULT_SEQ_LEN)

    pipe = TelemetryDataPipeline(weather_csv=weather_csv)
    cache = {}
    for h in horizons:
        res = build_forecast_windows(pipeline=pipe, split="test", horizon=h,
                                     seq_len=DEFAULT_SEQ_LEN, return_metadata=True)
        aug = build_feature_augmented_forecast_windows(
            pipe, split="test", horizon=h, seq_len=DEFAULT_SEQ_LEN)
        stamps = [m["target_timestamp"] for m in res[6]]
        order = sorted(range(len(stamps)), key=lambda i: stamps[i])
        mid = len(order) // 2
        halves = {"first_half": set(order[:mid]), "second_half": set(order[mid:])}
        cache[h] = (res[0], aug[1], res[1], res[6], halves)
        if verbose:
            print(f"  +{h:>2}h windows={len(order)} "
                  f"first={len(halves['first_half'])} second={len(halves['second_half'])}")

    out: Dict[str, Dict[int, Dict]] = {}
    for floor, seeds in discover_arms(sweep_dir).items():
        out.setdefault(floor, {})
        for seed, rec in sorted(seeds.items()):
            for h in horizons:
                ckpt = os.path.join(rec["dir"], f"candidate_h{h}h.pt")
                if not os.path.exists(ckpt):
                    continue
                ck = torch.load(ckpt, map_location="cpu", weights_only=False)
                man = ck.get("manifest", {}) or {}
                mdl = GarciaWeatherLNNFeatured(
                    input_dim=int(man.get("input_dim", 8)),
                    context_dim=int(man.get("context_dim", 75)),
                    hidden_dim=int(man.get("hidden_dim", 32)),
                    use_two_stage_precipitation=True)
                mdl.load_state_dict(ck["model_state_dict"])
                mdl.eval()
                tele, ctx, dt, meta, halves = cache[h]
                origin = torch.tensor(
                    np.column_stack([[m[c] for m in meta] for c in ORIGIN_COLS]),
                    dtype=torch.float32)
                with torch.no_grad():
                    outp = mdl(tele, ctx, dt, origin_weather=origin)

                row: Dict = {"halves": {}}
                for ch in channels:
                    pred = outp[ch].squeeze(-1).numpy()
                    tgt = np.array([m[f"target_{ch}"] for m in meta], dtype=float)
                    ae = np.abs(pred - tgt)
                    row[ch] = float(ae.mean())
                    for hname, idx in halves.items():
                        sel = np.fromiter(idx, dtype=int, count=len(idx))
                        row.setdefault(hname, {})[ch] = float(ae[sel].mean())
                out[floor].setdefault(seed, {})[h] = row
                if verbose:
                    print(f"  {rec['arm']:<12} +{h:>2}h " +
                          " ".join(f"{c}={row[c]:.4f}" for c in channels))
    return out


def _split_halves(recomputed: Dict, floor: str, seeds: Iterable[int], h: int,
                  channels: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """Per-chronological-half MAE for every seed of one arm at one horizon."""
    out: Dict[str, Dict[str, float]] = {}
    for hname in ("first_half", "second_half"):
        row: Dict[str, float] = {}
        for ch in channels:
            vals = []
            for s in seeds:
                block = ((recomputed.get(floor) or {}).get(s) or {}).get(h) or {}
                v = (block.get(hname) or {}).get(ch)
                if isinstance(v, (int, float)):
                    vals.append(float(v))
            if vals:
                row[ch] = mean(vals)
        if row:
            out[hname] = row
    return out


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def build_report(arms: Dict[str, Dict],
                 scores: Dict[str, Dict[int, Dict[str, float]]],
                 channels: Sequence[str] = CHANNELS,
                 horizons: Sequence[int] = HORIZONS,
                 sigma: float = DEFAULT_SIGMA,
                 alpha: float = DEFAULT_ALPHA,
                 power: float = DEFAULT_POWER,
                 materiality_pct: float = DEFAULT_MATERIALITY_PCT,
                 fingerprints: Optional[Dict] = None,
                 recomputed: Optional[Dict] = None) -> Dict:
    """Assemble the full selection report from discovered arms and their scores."""
    per_arm: Dict[str, Dict[str, Dict[int, Dict]]] = {}
    comparisons: List[Dict] = []
    floors = sorted(set(arms) | set(scores))

    # Iterate the union of DISCOVERED arms and SCORED arms. A seed can be scored
    # without a scorecard on disk -- re-scored from its checkpoint, for instance
    # -- and silently dropping it because no scorecard was found would quietly
    # shrink the sweep and change the answer.
    def seeds_for(floor):
        # Seed keys can arrive as strings from a JSON cache, so normalise before
        # ordering: mixing "303" and 303 in one sort raises.
        raw = set(arms.get(floor, {})) | set(scores.get(floor, {}))
        return sorted(int(s) for s in raw)

    for floor in floors:
        seeds = seeds_for(floor)
        per_arm[floor] = {}
        for h in horizons:
            per_channel: Dict[str, Dict] = {}
            for ch in channels:
                vals = {s: scores[floor][s][h][ch]
                        for s in seeds
                        if s in scores.get(floor, {}) and ch in scores[floor][s].get(h, {})}
                if not vals:
                    continue
                summ = seed_summary(vals, lower_is_better=True, alpha=alpha)
                summ["expected_seeds_for_5pct_effect"] = (
                    required_seeds_paired(summ["sd"],
                                          abs(summ["mean"]) * 0.05, alpha, power)
                    if summ.get("sd") else None)
                per_channel[ch] = summ
            if per_channel:
                per_arm[floor][str(h)] = per_channel

    for i, fa in enumerate(floors):
        for fb in floors[i + 1:]:
            for h in horizons:
                for ch in channels:
                    a = {s: scores[fa][s][h][ch]
                         for s in seeds_for(fa)
                         if s in scores.get(fa, {}) and ch in scores[fa][s].get(h, {})}
                    b = {s: scores[fb][s][h][ch]
                         for s in seeds_for(fb)
                         if s in scores.get(fb, {}) and ch in scores[fb][s].get(h, {})}
                    if not a and not b:
                        continue
                    # Precondition for ANY comparison: both arms must have been
                    # scored on the same rows. Checked, never assumed.
                    mismatch = None
                    if fingerprints:
                        mismatch = evaluation_sets_agree(
                            fingerprints.get(fa, {}), fingerprints.get(fb, {}),
                            sorted(a), sorted(b), (h,))
                    cmp_res = compare_arms(a, b, sigma, alpha, lower_is_better=True)
                    if mismatch:
                        cmp_res = {"status": INSUFFICIENT, "verdict": INSUFFICIENT,
                                   "reason": mismatch,
                                   "n_pairs": cmp_res.get("n_pairs", 0),
                                   "n_common_seeds": cmp_res.get("n_common_seeds", 0)}
                    sum_a, sum_b = seed_summary(a, True, alpha), seed_summary(b, True, alpha)
                    # The relative gap over ALL available seeds: descriptive only.
                    # When the arms are unbalanced this is computed over different
                    # seed sets on each side, so it is NOT an effect a test can
                    # speak to. Denominator is the mean of every value on both
                    # sides, matching the convention used inside the paired test.
                    pooled_all = [v for s in sorted(a) for v in (a[s],)] + \
                                [v for s in sorted(b) for v in (b[s],)]
                    scale_all = abs(mean(pooled_all)) if pooled_all else 0.0
                    if scale_all and sum_a.get("mean") is not None \
                            and sum_b.get("mean") is not None:
                        rel_all = 100.0 * (sum_a["mean"] - sum_b["mean"]) / scale_all
                    else:
                        rel_all = None
                    # The relative gap over the MATCHED seeds: this is the only
                    # difference a test can legitimately be about, so it is the
                    # one the decision is gated on.
                    rel_tested = cmp_res.get("relative_difference_pct")
                    if rel_tested is None:
                        rel_tested = rel_all

                    material = rel_tested is not None and abs(rel_tested) >= materiality_pct
                    # A large all-seed gap next to a small matched-seed gap means
                    # the apparent effect is being carried by a seed that only one
                    # arm has. That is the single most dangerous shape of result
                    # this tool can encounter.
                    driven_by_unmatched = bool(
                        rel_all is not None and rel_tested is not None
                        and abs(rel_all) >= materiality_pct
                        and abs(rel_tested) < materiality_pct)

                    if cmp_res.get("verdict") == INSUFFICIENT:
                        actionable = INSUFFICIENT
                    elif cmp_res.get("verdict") == DISTINGUISHABLE and material:
                        actionable = "DISTINGUISHABLE_AND_MATERIAL"
                    elif cmp_res.get("verdict") == DISTINGUISHABLE:
                        actionable = "RESOLVABLE_BUT_IMMATERIAL"
                    else:
                        actionable = "NOT_DISTINGUISHABLE"
                    entry = {
                        "arm_a": fa, "arm_b": fb, "horizon_h": h, "channel": ch,
                        "n_seeds_a": len(a), "n_seeds_b": len(b),
                        "summary_a": sum_a,
                        "summary_b": sum_b,
                        "comparison": cmp_res,
                        "relative_difference_pct": rel_tested,
                        "relative_difference_pct_all_seeds": rel_all,
                        "relative_difference_pct_tested": cmp_res.get(
                            "relative_difference_pct"),
                        "arms_unbalanced": cmp_res.get("arms_unbalanced", False),
                        "materiality_threshold_pct": materiality_pct,
                        "is_material": material,
                        "driven_by_unmatched_seed": driven_by_unmatched,
                        "actionable": actionable,
                    }
                    if recomputed:
                        # Only over the seeds BOTH arms have, so the half-split
                        # check is as paired as the main test. _split_halves
                        # returns {half: {channel: mean over seeds}}, and
                        # half_split_consistency wants {half: {seed: value}}, so
                        # the already-averaged arm mean is passed as a single
                        # pseudo-seed.
                        shared = sorted(set(a) & set(b))
                        ha = _split_halves(recomputed, fa, shared, h, [ch])
                        hb = _split_halves(recomputed, fb, shared, h, [ch])
                        pa = {k: {0: v[ch]} for k, v in ha.items() if ch in v}
                        pb = {k: {0: v[ch]} for k, v in hb.items() if ch in v}
                        if pa and pb:
                            entry["half_split"] = half_split_consistency(pa, pb)
                            entry["half_split"]["n_seeds_averaged"] = len(shared)
                    comparisons.append(entry)

    # Seed spread, pooled per (channel, horizon) across every arm with >= 2 seeds.
    spread: Dict[str, Dict] = {}
    for h in horizons:
        for ch in channels:
            cvs, sds, rels = [], [], []
            for floor in floors:
                vals = [scores[floor][s][h][ch] for s in seeds_for(floor)
                        if s in scores.get(floor, {})
                        and ch in scores[floor][s].get(h, {})]
                if len(vals) >= 2:
                    sds.append(sample_sd(vals))
                    m = mean(vals)
                    cvs.append(100.0 * sample_sd(vals) / abs(m))
                    rels.append(100.0 * (max(vals) - min(vals)) / abs(m))
            if sds:
                spread[f"{ch}_h{h}"] = {
                    "n_arms_with_ge_2_seeds": len(sds),
                    "min_seed_sd": min(sds),
                    "median_seed_sd": sorted(sds)[len(sds) // 2],
                    "mean_seed_sd": mean(sds),
                    "max_seed_sd": max(sds),
                    "median_seed_cv_pct": sorted(cvs)[len(cvs) // 2],
                    "max_seed_cv_pct": max(cvs),
                    "median_seed_range_pct": sorted(rels)[len(rels) // 2],
                    "max_seed_range_pct": max(rels),
                }

    min_seeds: Dict[str, Dict] = {}
    for h in horizons:
        for ch in channels:
            # Plan against the WORST arm-to-arm seed spread observed at this
            # (channel, horizon), not the average one: a sweep sized for the
            # median spread is under-powered exactly where the spread is worst.
            sds, means = [], []
            for floor in floors:
                summ = (per_arm.get(floor, {})
                        .get(str(h), {})
                        .get(ch))
                if summ and summ.get("status") == "OK":
                    sds.append(summ["sd"])
                    means.append(summ["mean"])
            if not sds or not means:
                continue
            ref_mean = mean(means)
            if not ref_mean:
                continue
            rec = min_seeds_recommendation(max(sds), ref_mean,
                                           (2.0, 5.0, 10.0), alpha, power)
            rec["seed_sd_used"] = max(sds)
            rec["seed_sd_basis"] = "worst across arms at this (channel, horizon)"
            min_seeds[f"{ch}_h{h}"] = rec

    # Confound check: were the arms trained by the same code?
    confounds = []
    for floor in floors:
        commits = set()
        for rec in arms.get(floor, {}).values():
            commits.update(rec["code_commits"])
        if commits:
            confounds.append({"floor": floor, "code_commits": sorted(commits)})
    if len(confounds) > 1 and len({tuple(c["code_commits"]) for c in confounds}) > 1:
        confound_note = (
            "arms were trained by DIFFERENT code commits, so any floor-to-floor "
            "difference is confounded with everything else that changed between "
            "those commits; this is not a clean single-variable ablation")
    else:
        confound_note = None

    coverage = {
        floor: {
            str(seed): {
                "checkpoints_present": sorted(
                    h for h, ok in (rec.get("checkpoints") or {}).items() if ok),
                "missing_horizons": sorted(
                    h for h, ok in (rec.get("checkpoints") or {}).items() if not ok),
                "has_scorecard": rec.get("has_scorecard", False),
                "usable": rec.get("usable", rec.get("has_scorecard", False)),
                "complete": rec.get("complete", False),
                "on_disk": True,
            } for seed, rec in sorted(arms.get(floor, {}).items())
        } for floor in floors
    }
    # A seed can be scored without appearing in the sweep directory at all. List
    # it, so a report never silently rests on fewer seeds than it appears to.
    for floor in floors:
        for seed in seeds_for(floor):
            if str(seed) not in coverage[floor]:                coverage[floor][str(seed)] = {
                    "checkpoints_present": [], "missing_horizons": [],
                    "has_scorecard": False, "usable": True, "complete": False,
                    "on_disk": False,
                }

    verdicts = [c["comparison"].get("verdict") for c in comparisons]
    tally: Dict[str, int] = {}
    for v in verdicts:
        tally[v] = tally.get(v, 0) + 1

    # The headline: is any floor-to-floor difference larger than the seed noise
    # it is being measured against? "not one" is a valid and important result.
    distinguishable = [c for c in comparisons
                      if c["comparison"].get("verdict") == DISTINGUISHABLE]
    material = [c for c in comparisons if c.get("actionable")
                == "DISTINGUISHABLE_AND_MATERIAL"]
    immaterial = [c for c in comparisons if c.get("actionable")
                  == "RESOLVABLE_BUT_IMMATERIAL"]
    inconsistent = [c for c in comparisons
                    if (c.get("half_split") or {}).get("verdict")
                    == "INCONSISTENT_ACROSS_HALVES"]
    half_split_covered = [c for c in comparisons if c.get("half_split")]
    below_floor = [c for c in comparisons
                   if c["comparison"].get("sign_test_can_reject") is False
                   and c["comparison"].get("status") == "OK"]
    unmatched = [c for c in comparisons if c.get("driven_by_unmatched_seed")]
    headline = {
        "comparisons_made": len(comparisons),
        "distinguishable_from_seed_noise": len(distinguishable),
        "of_which_material": len(material),
        "resolvable_but_immaterial": len(immaterial),
        "not_distinguishable": tally.get(NOT_DISTINGUISHABLE, 0),
        "insufficient_evidence": tally.get(INSUFFICIENT, 0),
        "half_split_inconsistent": len(inconsistent),
        "half_split_covered": len(half_split_covered),
        "half_split_coverage_note": (
            f"{len(half_split_covered)}/{len(comparisons)} comparisons had the "
            f"chronological half-split check run; the remainder have no verdict "
            f"and must not be read as consistent"
            if len(half_split_covered) < len(comparisons)
            else f"the half-split check ran on all {len(comparisons)} comparisons"),
        "below_exact_sign_test_floor": len(below_floor),
        "effect_driven_by_unmatched_seed": len(unmatched),
        "largest_relative_difference_pct": max(
            (abs(c["relative_difference_pct"]) for c in comparisons
             if c.get("relative_difference_pct") is not None), default=None),
        "largest_relative_difference_all_seeds_pct": max(
            (abs(c["relative_difference_pct_all_seeds"]) for c in comparisons
             if c.get("relative_difference_pct_all_seeds") is not None), default=None),
        "widest_seed_range_pct": max(
            (v["max_seed_range_pct"] for v in spread.values()), default=None),
        "verdict": (DISTINGUISHABLE if material else NOT_DISTINGUISHABLE),
        "answer": (
            (f"{len(material)} of {len(comparisons)} floor-to-floor differences are "
             f"both larger than their seed noise and larger than the "
             f"{materiality_pct:g}% materiality floor")
            if material else
            (f"NO floor-to-floor difference on this corpus is both larger than its "
             f"seed noise and larger than the {materiality_pct:g}% materiality "
             f"floor. The largest difference between architectures on matched seeds "
             f"is {max((abs(c['relative_difference_pct']) for c in comparisons if c.get('relative_difference_pct') is not None), default=0.0):.2f}% "
             f"of MAE, and the largest difference over all seeds is "
             f"{max((abs(c['relative_difference_pct_all_seeds']) for c in comparisons if c.get('relative_difference_pct_all_seeds') is not None), default=0.0):.2f}% "
             f"-- while seed-to-seed spread alone reaches "
             f"{max((v['max_seed_range_pct'] for v in spread.values()), default=0.0):.1f}% "
             f"at the same (channel, horizon). No architecture ranking from this "
             f"sweep can be trusted.")),
    }

    return {
        "tool": "select_best_checkpoint",
        "version": "1.0.0",
        "policy": {
            "gate": "paired 2-sigma on seed-matched differences",
            "sigma": sigma,
            "alpha": alpha,
            "power": power,
            "materiality_pct": materiality_pct,
            "single_seed_rule": "INSUFFICIENT_EVIDENCE",
            "gate_is_weak_at_low_n": (
                "the 2-sigma halfwidth is sigma * sd_of_differences / sqrt(n_pairs) "
                "and sd_of_differences has only n_pairs - 1 degrees of freedom, so "
                "at n_pairs=2 the band collapses and trivially small differences "
                "clear it. A DISTINGUISHABLE verdict is therefore necessary but "
                "not sufficient; the effect size and the exact sign test are "
                "reported alongside it."),
            "headline_rule": (
                "report the mean across seeds; the best single seed is an "
                "optimistically biased estimate of the arm, not the arm"),
        },
        "coverage": coverage,
        "confound": {"per_arm": confounds, "note": confound_note},
        "evaluation_row_counts": fingerprints or {},
        "seed_spread": spread,
        "per_arm": per_arm,
        "comparisons": comparisons,
        "verdict_tally": tally,
        "headline": headline,
        "minimum_seeds": min_seeds,
        "horizons": list(horizons),
        "channels": list(channels),
    }


def evaluation_set_fingerprint(sweep_dir: str) -> Dict[str, Dict[int, Optional[int]]]:
    """
    Per-(floor, seed) -> horizon -> number of evaluation rows.

    A paired test is only meaningful if both arms were scored on the SAME rows.
    Different sample counts mean different evaluation sets, and every difference
    computed across them is meaningless -- not noisy, meaningless. This is
    checked rather than assumed, because the two arms here were trained by
    different code commits and the trainer carried window-selection logic that
    differed between them.
    """
    out: Dict[str, Dict[int, Optional[int]]] = {}
    for floor, seeds in discover_arms(sweep_dir).items():
        for seed, rec in sorted(seeds.items()):
            if not rec["has_scorecard"]:
                continue
            try:
                with open(rec["scorecard"], encoding="utf-8") as f:
                    card = json.load(f)
            except (ValueError, OSError):
                continue
            row: Dict[int, Optional[int]] = {}
            for key, block in (card.get("horizon_evaluations") or {}).items():
                try:
                    h = int(key.rsplit("_", 1)[1].rstrip("h"))
                except (IndexError, ValueError):
                    continue
                n = block.get("sample_count")
                row[h] = n if isinstance(n, int) else None
            if row:
                out.setdefault(floor, {})[seed] = row
    return out


def evaluation_sets_agree(fp_a: Dict[int, Dict[int, Optional[int]]],
                          fp_b: Dict[int, Dict[int, Optional[int]]],
                          seeds_a: Iterable[int],
                          seeds_b: Iterable[int],
                          horizons: Sequence[int]) -> Optional[str]:
    """
    None when both arms were scored on the same number of rows at every horizon,
    otherwise a description of where they diverge.

    A paired difference is a statement about the same rows measured two ways. If
    the row counts differ the difference is not noisy, it is meaningless, and it
    must not be reported with a confidence interval attached.
    """
    for h in horizons:
        counts: Dict[Optional[int], int] = {}
        for table, seeds in ((fp_a, seeds_a), (fp_b, seeds_b)):
            for s in seeds:
                n = (table.get(s) or {}).get(h)
                if n is not None:
                    counts[n] = counts.get(n, 0) + 1
        if len(counts) > 1:
            detail = ", ".join(f"{n} rows x{k}" for n, k in sorted(counts.items()))
            return (f"the two arms were evaluated on different numbers of rows at "
                    f"+{h}h ({detail}); a paired difference across different row "
                    f"sets is not a comparison")
    return None


def _normalize_recomputed(recomputed: Optional[Dict]) -> Optional[Dict]:
    """
    Force seed and horizon keys of a re-scored block to ints.

    A block read back from a JSON cache has string keys, and every consumer
    downstream looks seeds up by int. Normalising once, here, means the cache
    path and the fresh-compute path cannot drift apart -- which is exactly the
    failure mode where a half-split check silently computes nothing and a report
    reads as though it had passed.
    """
    if not recomputed:
        return recomputed
    out: Dict[str, Dict[int, Dict]] = {}
    for floor, per_seed in recomputed.items():
        out[floor] = {}
        for seed, per_h in per_seed.items():
            out[floor][int(seed)] = {int(h): row for h, row in per_h.items()}
    return out


def _fmt(v, spec: str = ".4f") -> str:
    """Format a number for a fixed-width table cell, or '-' when it is absent."""
    if v is None:
        return "-"
    if isinstance(v, float) and math.isnan(v):
        return "-"
    if isinstance(v, float) and math.isinf(v):
        return "-inf" if v < 0.0 else "+inf"
    return f"{v:{spec}}"


def print_report(report: Dict) -> None:
    print("=" * 104)
    print("CHECKPOINT SELECTION -- seed-variance aware   (MAE, lower is better)")
    print("=" * 104)

    print("\nCOVERAGE (what actually exists)")
    for floor, seeds in report["coverage"].items():
        for seed, rec in seeds.items():
            have = ",".join(f"{h}h" for h in rec["checkpoints_present"]) or "none"
            missing = ",".join(f"{h}h" for h in rec.get("missing_horizons", []))
            if not rec.get("on_disk", True):
                flag = "scored, no artefacts on disk"
            elif rec["has_scorecard"]:
                flag = "ok"
            else:
                flag = "NO SCORECARD (needs re-scoring to be usable)"
            tail = f"  missing:{missing}" if missing else ""
            print(f"  {floor:<6} seed {seed:<4} checkpoints[{have:<24}] {flag}{tail}")
    if report["confound"]["note"]:
        print(f"\n  !! CONFOUND: {report['confound']['note']}")
        for c in report["confound"]["per_arm"]:
            print(f"     {c['floor']:<6} code_commit(s) {', '.join(c['code_commits'])}")

    print("\n" + "-" * 104)
    print("SEED SPREAD  (sd of MAE across seeds, within one arm)")
    print("-" * 104)
    print(f"{'channel':<12}{'hz':>4}{'arms':>6}{'median sd':>12}{'max sd':>10}"
          f"{'med CV%':>10}{'max CV%':>10}{'med rng%':>11}")
    for key, row in report["seed_spread"].items():
        ch, _, h = key.rpartition("_h")
        print(f"{ch:<12}{h:>4}{row['n_arms_with_ge_2_seeds']:>6}"
              f"{_fmt(row['median_seed_sd']):>12}{_fmt(row['max_seed_sd']):>10}"
              f"{_fmt(row['median_seed_cv_pct'], '.2f'):>10}"
              f"{_fmt(row['max_seed_cv_pct'], '.2f'):>10}"
              f"{_fmt(row['median_seed_range_pct'], '.2f'):>11}")

    print("\n" + "-" * 104)
    print("BEST SEED vs MEAN  (the optimistic bias of shipping the best seed)")
    print("-" * 104)
    print(f"{'arm':<7}{'channel':<12}{'hz':>4}{'n':>3}{'mean':>9}{'sd':>9}{'sem':>9}"
          f"{'best':>9}{'gap':>9}{'E[gap]':>9}{'bias%':>8}")
    for floor, per_h in report["per_arm"].items():
        for h, per_ch in per_h.items():
            for ch, s in per_ch.items():
                print(f"{floor:<7}{ch:<12}{h:>4}{s['n_seeds']:>3}"
                      f"{_fmt(s.get('mean')):>9}{_fmt(s.get('sd')):>9}"
                      f"{_fmt(s.get('sem')):>9}{_fmt(s.get('best_value')):>9}"
                      f"{_fmt(s.get('best_vs_mean_gap')):>9}"
                      f"{_fmt(s.get('expected_best_vs_mean_gap')):>9}"
                      f"{_fmt(s.get('expected_best_vs_mean_gap_pct'), '.2f'):>8}")

    print("\n" + "-" * 104)
    print(f"ARM vs ARM  (paired 2-sigma on seed-matched differences; "
          f"material = >= {report['policy']['materiality_pct']:g}% of MAE)")
    print("-" * 104)
    print(f"{'A':<7}{'B':<7}{'channel':<12}{'hz':>4}{'nA':>3}{'nB':>3}"
          f"{'meanA*':>9}{'meanB*':>9}{'diff':>9}{'2sig':>9}{'rel%*':>8}"
          f"{'relAll%':>9}{'p':>8}{'signP':>8}  verdict")
    for c in report["comparisons"]:
        cm = c["comparison"]
        # meanA*/meanB*/diff/rel%* are over the seeds the test used; relAll% is
        # over every seed available for each arm.
        ma = cm.get("mean_a_tested")
        mb = cm.get("mean_b_tested")
        if ma is None:
            ma = c["summary_a"].get("mean")
        if mb is None:
            mb = c["summary_b"].get("mean")
        print(f"{c['arm_a']:<7}{c['arm_b']:<7}{c['channel']:<12}{c['horizon_h']:>4}"
              f"{c['n_seeds_a']:>3}{c['n_seeds_b']:>3}"
              f"{_fmt(ma):>9}{_fmt(mb):>9}"
              f"{_fmt(cm.get('mean_difference')):>9}"
              f"{_fmt(cm.get('two_sigma_halfwidth')):>9}"
              f"{_fmt(cm.get('relative_difference_pct'), '.2f'):>8}"
              f"{_fmt(c.get('relative_difference_pct_all_seeds'), '.2f'):>9}"
              f"{_fmt(cm.get('p_value_parametric'), '.3f'):>8}"
              f"{_fmt(cm.get('p_value_exact_sign_test'), '.3f'):>8}"
              f"  {c.get('actionable')}")
        if cm.get("status") == INSUFFICIENT:
            print(f"{'':<27} reason: {cm.get('reason')}")
        if c.get("driven_by_unmatched_seed"):
            print(f"{'':<27} ** effect is carried by a seed only one arm has: "
                  f"{c['relative_difference_pct_all_seeds']:+.2f}% over all seeds "
                  f"vs {c['relative_difference_pct']:+.2f}% on matched seeds **")
    unbalanced = [(c["arm_a"], c["n_seeds_a"], c["arm_b"], c["n_seeds_b"],
                   c["comparison"].get("n_pairs"))
                  for c in report["comparisons"] if c.get("arms_unbalanced")]
    for fa, na, fb, nb, npairs in sorted(set(unbalanced)):
        print(f"\n  * {fa} has {na} usable seed(s) and {fb} has {nb}. The paired "
              f"test above used only the {npairs} shared seed(s), so meanA*/meanB*/"
              f"diff/rel%* describe that matched subset. relAll% and the BEST SEED "
              f"vs MEAN table describe every usable seed, which is why the two "
              f"relative figures differ.")

    print("\n" + "-" * 104)
    print("MINIMUM SEEDS BEFORE A COMPARISON SHOULD BE BELIEVED  (80% power, alpha 0.05)")
    print("-" * 104)
    for key, m in report["minimum_seeds"].items():
        ch, _, h = key.rpartition("_h")
        parts = "  ".join(f"{e['effect_pct_of_mean']:g}%->{e['seeds_for_paired_t']}"
                          for e in m["by_effect_size"])
        print(f"  {ch:<12}+{h:<3}h  seedCV={m['seed_cv_pct']:>6.2f}%   "
              f"seeds for: {parts}   RECOMMENDED MIN = {m['recommended_min_seeds']}")
    if report["minimum_seeds"]:
        any_m = next(iter(report["minimum_seeds"].values()))
        print(f"\n  sign-test floor: {any_m['sign_test_floor_pairs']} pairs")
        print(f"  {any_m['sign_test_explanation']}")

    print("\n" + "=" * 104)
    print(f"VERDICT TALLY  {report['verdict_tally']}")
    print("=" * 104)
    head = report.get("headline")
    if head:
        print(f"\nHEADLINE: {head['answer']}")
        print(f"          2-sigma gate: {head['distinguishable_from_seed_noise']}"
              f"/{head['comparisons_made']} distinguishable, of which "
              f"{head['of_which_material']} clear the materiality floor and "
              f"{head['resolvable_but_immaterial']} do not | "
              f"{head['insufficient_evidence']} insufficient | "
              f"{head['below_exact_sign_test_floor']} rest on fewer pairs than the "
              f"exact sign test needs | "
              f"{head['effect_driven_by_unmatched_seed']} effect(s) carried by an "
              f"unmatched seed | "
              f"{head['half_split_inconsistent']}/{head['half_split_covered']} "
              f"comparisons reverse sign across chronological halves")
        if head["half_split_covered"] < head["comparisons_made"]:
            print(f"          {head['half_split_coverage_note']}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Select a checkpoint across seeds using statistics that "
                    "account for seed variance. Never trains.")
    ap.add_argument("--sweep-dir", default=DEFAULT_SWEEP_DIR)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--channels", default=",".join(CHANNELS))
    ap.add_argument("--horizons", default=",".join(str(h) for h in HORIZONS))
    ap.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--power", type=float, default=DEFAULT_POWER)
    ap.add_argument("--materiality-pct", type=float, default=DEFAULT_MATERIALITY_PCT,
                    help="minimum relative effect size, in percent of MAE, for a "
                         "difference to count as worth acting on rather than "
                         "merely resolvable")
    ap.add_argument("--allow-recompute", action="store_true",
                    help="re-score existing checkpoints (inference only, never "
                         "training) to fill arms with no scorecard and to obtain "
                         "chronological half-split numbers. Needs torch + CPU.")
    ap.add_argument("--recompute-cache", default=None,
                    help="path to a JSON cache of the re-scored MAE. Loaded if "
                         "present, written after a fresh --allow-recompute run. "
                         "Re-scoring costs real CPU, so the cache exists so the "
                         "statistics can be re-derived without paying for it "
                         "again.")
    ap.add_argument("--weather-csv", default=DEFAULT_WEATHER_CSV)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)

    channels = tuple(c.strip() for c in args.channels.split(",") if c.strip())
    horizons = tuple(int(h) for h in args.horizons.split(",") if h.strip())

    arms = discover_arms(args.sweep_dir)
    if not arms:
        print(f"no arms found under {args.sweep_dir}")
        return 1

    scores = load_scorecard_mae(args.sweep_dir, channels)
    fingerprints = evaluation_set_fingerprint(args.sweep_dir)
    recomputed = None
    if args.recompute_cache and os.path.exists(args.recompute_cache):
        with open(args.recompute_cache, encoding="utf-8") as f:
            recomputed = json.load(f)
        print(f"  re-scored MAE loaded from cache: {args.recompute_cache}")
    elif args.allow_recompute:
        recomputed = recompute_from_checkpoints(
            args.sweep_dir, args.weather_csv, channels, horizons, verbose=True)
        if args.recompute_cache:
            with open(args.recompute_cache, "w", encoding="utf-8", newline="\n") as f:
                json.dump(recomputed, f, indent=2)
            print(f"  re-scored MAE cached: {args.recompute_cache}")
    if recomputed:
        # Normalise once so the cache path and the fresh-compute path are
        # indistinguishable to every consumer below.
        recomputed = _normalize_recomputed(recomputed)
        # Merge on BOTH paths. Re-scored MAE is the better number (it is computed
        # from the checkpoint on the shared test split rather than read out of a
        # per-arm scorecard) and it is the only source for arms that never got a
        # scorecard written, so it fills those gaps too.
        for floor, per_seed in recomputed.items():
            for seed, per_h in per_seed.items():
                for h, row in per_h.items():
                    scores.setdefault(floor, {}).setdefault(seed, {})[h] = {
                        ch: row[ch] for ch in channels if ch in row}

    report = build_report(arms, scores, channels, horizons,
                          args.sigma, args.alpha, args.power,
                          args.materiality_pct, fingerprints, recomputed)
    print_report(report)

    if not args.no_write:
        with open(args.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(report, f, indent=2)
        print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
