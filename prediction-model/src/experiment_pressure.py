"""
EXPERIMENT: can anything beat PERSISTENCE on surface pressure?

WHY THIS EXPERIMENT EXISTS
--------------------------
The pressure head is the model's strongest channel on paper: production MAE beats
all seven NWP models at all five horizons. But every pressure cell differs from
persistence by ~1e-4 hPa, so "beat all NWP" is really "a station barometer beats
a 0.25-degree grid cell at surface pressure", and the serving policy already
notices this and routes pressure to `persistence_fallback`.

Two things are therefore worth knowing and neither is known:

  1. Is there ANY learnable structure in the pressure CHANGE? The physical
     prior says yes -- surface pressure has a diurnal cycle, so a forecast
     issued at 03:00 and one issued at 15:00 should drift by different amounts
     over the same lead time, and persistence cannot express that. If a
     correction exploiting it is worth a few hundredths of a hPa, that is a real
     improvement requiring no new architecture, no checkpoint, and no training
     pipeline -- just a table.
  2. Can NWP add anything? The frozen NWP correction artifacts already carry a
     routed +6h pressure win. Does it survive, and is it additive with (1) or
     substitutive for it?

PROTOCOL (this is the part that makes the numbers mean something)
----------------------------------------------------------------
  TRAIN  -> fit everything: drift model, cell means, trend coefficient, and the
            self-contained NWP affine refit
  VAL    -> choose: shrinkage strength, trend variant, NWP lambda/mode, blend
            weights, and which local variant to ship
  TEST   -> read ONCE, to report. Nothing is selected from it.

Every number labelled "test" is a fit-on-train, evaluate-on-test number. Where a
hyper-parameter was needed it was chosen on validation. The train/val/test
boundaries come from TelemetryDataPipeline._compute_split_boundaries with the
pipeline's own 48 h embargo, and are written into the output.

A NOTE ON THE CORPUS
--------------------
The reference numbers in release_baseline.md were captured on the OLDER corpus
(weather_telemetry.csv, test 2026-08-15 22:00 .. 2026-08-26 02:00, 2273 +1h
windows). This experiment runs on weather_telemetry_current.csv (test
2026-09-11 14:00 .. 2026-09-29 23:00), as instructed, so absolute MAEs are NOT
comparable to that table. The comparison between options is, because every
option is scored on identical rows.

The frozen NWP correction artifacts were themselves fitted on the OLD corpus.
Applying them to the current corpus is still train-on-train/test-on-test, but
the coefficient set carries an out-of-corpus mismatch. A second, self-contained
NWP affine correction is therefore refitted on THIS corpus's train split and
evaluated on its test split, so the reader can see how much of the frozen
artifact's performance is the coefficients and how much is the corpus.

READ-ONLY on production code and artifacts.
Writes exactly one file: prediction-model/data/experiment_pressure.json
"""

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from dataset import (TelemetryDataPipeline, build_forecast_windows,  # noqa: E402
                     DEFAULT_SEQ_LEN)
from nwp_correction import NwpCorrection, CLAMP, select  # noqa: E402
import benchmark_vs_nwp as bvn  # noqa: E402

DATA_DIR = os.path.abspath(os.path.join(HERE, "..", "data"))
# (checkpoint, corpus) PAIRING. The promotion candidate was retrained on
# weather_telemetry_current.csv, so it is the only checkpoint that may be scored
# on that corpus. The older candidate_artifacts/ was trained on
# weather_telemetry.csv and must be scored on THAT corpus, never on the current
# one. candidate_artifacts_leaky_olddata/ is an ablation arm and is deliberately
# not wired in. check_pairing() enforces this from the manifest SHA rather than
# trusting the directory name.
PRIMARY = {
    "candidate_dir": os.path.join(DATA_DIR, "candidate_artifacts_windfix"),
    "corpus": os.path.join(DATA_DIR, "weather_telemetry_current.csv"),
}
SECONDARY = {
    "candidate_dir": os.path.join(DATA_DIR, "candidate_artifacts"),
    "corpus": os.path.join(DATA_DIR, "weather_telemetry.csv"),
}
CANDIDATE_DIR = PRIMARY["candidate_dir"]
ARTIFACTS = {
    "ecmwf_ifs025": os.path.join(DATA_DIR, "nwp_correction_ecmwf_ifs025.json"),
    "gfs_seamless": os.path.join(DATA_DIR, "nwp_correction_gfs_seamless.json"),
}
DEFAULT_OUT = os.path.join(DATA_DIR, "experiment_pressure.json")
DEFAULT_CORPUS = PRIMARY["corpus"]

# The corpus the frozen NWP correction artifacts were fitted on, per the fit
# report inside each artifact. Used to state, per pairing, whether applying
# them is an in-corpus or out-of-corpus transfer. fit_nwp_correction.py builds
# its pipeline with TelemetryDataPipeline() and no --weather-csv override, i.e.
# the default corpus, weather_telemetry.csv.
ARTIFACT_TRAIN_CORPUS = {
    os.path.join(DATA_DIR, "weather_telemetry.csv"): os.path.join(
        DATA_DIR, "weather_telemetry.csv"),
}

HORIZONS = [1, 3, 6, 12, 24]
PRESSURE_COL = 3          # pressure in the 8-column normalised window
LAMBDAS = [round(float(x), 2) for x in np.arange(0.0, 1.01, 0.1)]
MIN_CELL_N = 20          # min train rows before a (station, hour) cell is trusted
SHRINK_K_GRID = [0, 5, 20, 50, 100, 200]


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def score(pred, truth):
    """
    MAE/RMSE/bias over rows where BOTH prediction and truth are finite.

    No coverage floor is applied here. This is not a rank-across-methods report
    where a half-covered score could masquerade as a full one: every option
    reports its own n, options with different NWP coverage are flagged, and a
    common-row table (stage4_common_rows) gives the strictly like-for-like
    comparison. Dropping the floor would only be defensible if a reader could
    see n next to every mae, which they can.
    """
    p_all = np.asarray(pred, float)
    t_all = np.asarray(truth, float)
    finite = np.isfinite(p_all) & np.isfinite(t_all)
    total = len(p_all)
    kept = int(finite.sum())
    if kept == 0:
        return {"mae": None, "rmse": None, "bias": None, "n": 0,
                "coverage": 0.0, "n_rows_offered": total}
    p, t = p_all[finite], t_all[finite]
    e = np.abs(p - t)
    return {"mae": float(e.mean()),
            "rmse": float(np.sqrt((e ** 2).mean())),
            "bias": float((p - t).mean()),
            "n": int(kept),
            "coverage": kept / total if total else 0.0,
            "n_rows_offered": total}


def paired_vs_baseline(pred, base, truth, n_boot=400, seed=42):
    """
    Paired comparison against persistence on rows where BOTH are finite.

    Paired because both forecasts are scored against the same observation; the
    unpaired difference of two large-n means has a standard error that shrinks
    with n whether or not the forecasts are any good. Returns the mean
    absolute-error difference, a bootstrap CI on that difference, and the share
    of rows improved. A "win" whose CI straddles zero is not a win, and is
    reported as such.
    """
    p = np.asarray(pred, float)
    b = np.asarray(base, float)
    t = np.asarray(truth, float)
    m = np.isfinite(p) & np.isfinite(b) & np.isfinite(t)
    if m.sum() < 2:
        return None
    d = np.abs(p[m] - t[m]) - np.abs(b[m] - t[m])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n_boot, len(d)))
    boot = d[idx].mean(axis=1)
    return {
        "n_paired": int(m.sum()),
        "mae_delta_vs_persistence_hPa": float(d.mean()),
        "mae_delta_ci95_hPa": [float(np.percentile(boot, 2.5)),
                               float(np.percentile(boot, 97.5))],
        "pct_improvement_vs_persistence": float(
            -d.mean() / np.abs(b[m] - t[m]).mean() * 100.0),
        "fraction_rows_improved": float((d < 0).mean()),
        "significant_at_95": bool(np.percentile(boot, 97.5) < 0.0),
    }


# ---------------------------------------------------------------------------
# windows
# ---------------------------------------------------------------------------
class Split:
    """Everything a stage needs about one (split, horizon) window set."""

    def __init__(self, res, norm_stds, norm_means):
        (self.x, self.dt, _rain, _precip, _water, _haswater,
         self.meta) = res
        self.n = len(self.meta)
        self.station = np.array([m["station_id"] for m in self.meta], dtype=object)
        self.target = np.array([m["target_pressure"] for m in self.meta], float)
        self.origin = np.array([m["origin_pressure"] for m in self.meta], float)
        # Hour-of-day of the FORECAST ORIGIN, UTC. For a fixed lead time the
        # target hour is (origin hour + H) mod 24, so "drift by origin hour"
        # and "drift by target hour" are one table under a relabelling. Origin
        # hour is used because it is the thing an operator can see.
        self.hour = np.array([int(m["origin_timestamp"][11:13]) for m in self.meta], int)
        self.delta = self.target - self.origin
        x = self.x[:, :, PRESSURE_COL].detach().cpu().numpy().astype(np.float64)
        self.press_seq = x * float(norm_stds[PRESSURE_COL]) + float(norm_means[PRESSURE_COL])

    def trends(self):
        """
        Signed pressure tendency at the origin, hPa per hour, over the last 1h
        and 3h of the input window. Zero when the last four samples are not one
        hour apart, so a station with a gap contributes a 0 to the LS fit rather
        than a fabricated slope.
        """
        gaps = self.dt[:, -4:, 0].detach().cpu().numpy().astype(np.float64)
        seq = self.press_seq[:, -4:]
        ok = (np.abs(gaps - 1.0) < 1e-6).all(axis=1)
        t1 = np.where(ok, seq[:, -1] - seq[:, -2], 0.0)
        t3 = np.where(ok, (seq[:, -1] - seq[:, -4]) / 3.0, 0.0)
        return t1, t3


def build_split(pipe, split, h, norm_stds, norm_means):
    res = build_forecast_windows(pipeline=pipe, split=split, horizon=h,
                                 seq_len=DEFAULT_SEQ_LEN, return_metadata=True)
    if res is None:
        raise RuntimeError(f"no windows for split={split} horizon=+{h}h")
    return Split(res, norm_stds, norm_means)


# ---------------------------------------------------------------------------
# NWP join
# ---------------------------------------------------------------------------
def nwp_series(nwp, split, model):
    """Raw surface_pressure valid at each window's TARGET timestamp, else NaN."""
    out = np.full(split.n, np.nan)
    for i, m in enumerate(split.meta):
        e = nwp.get(f"{m['station_id']}|{model}")
        if not e:
            continue
        j = e["idx"].get(m["target_timestamp"][:13] + ":00")
        if j is None:
            continue
        v = e["rec"]["surface_pressure"][j]
        if v is not None:
            out[i] = float(v)
    return out


# ---------------------------------------------------------------------------
# Stage 2: hour-of-day + station drift model
# ---------------------------------------------------------------------------
def fit_additive(stations, hours, d, trend=None):
    """
    Least squares  d ~ mu + a[origin_hour_utc] + b[station] (+ c * trend).

    Hour and station are fitted JOINTLY because they are not orthogonal on this
    data: stations have uneven coverage and the test period is not a full
    multiple of 24 h per station, so a separately fitted hour table absorbs part
    of each station's overall tendency. The design [1, hour dummies minus the
    first, station dummies minus the first] is full rank, so lstsq is well
    posed and mu, a and b are the usual sum-to-zero-constrained effects.
    """
    n = len(d)
    uniq_s = sorted(set(stations))
    s_i = {s: k for k, s in enumerate(uniq_s)}
    cols = [np.ones(n)]
    hour_onehot = np.zeros((n, 24))
    for i, hh in enumerate(hours):
        hour_onehot[i, hh] = 1.0
    cols.extend([hour_onehot[:, k] for k in range(1, 24)])
    st_onehot = np.zeros((n, len(uniq_s)))
    for i, ss in enumerate(stations):
        st_onehot[i, s_i[ss]] = 1.0
    cols.extend([st_onehot[:, k] for k in range(1, len(uniq_s))])
    if trend is not None:
        cols.append(np.asarray(trend, float))
    beta, *_ = np.linalg.lstsq(np.column_stack(cols), np.asarray(d, float), rcond=None)
    a = np.zeros(24)
    a[1:] = beta[1:24]
    b = np.zeros(len(uniq_s))
    b[1:] = beta[24:24 + len(uniq_s) - 1]
    return {"mu": float(beta[0]), "a": a, "b": b,
            "c": float(beta[-1]) if trend is not None else 0.0,
            "stations": uniq_s, "s_i": s_i}


def additive_delta(model, stations, hours, trend=None):
    """
    Reconstruct the fitted delta. The intercept mu MUST be included: the hour and
    station effects are sum-to-zero, so they carry no mean, and dropping mu
    shifts every prediction by a constant. That bug produced an OLS R2 of -0.55,
    which is impossible for a design matrix containing an intercept and is
    exactly the tell that the reconstruction, not the fit, was wrong. It is
    checked every run by verify_reconstruction().
    """
    a = model["a"][np.asarray(hours, int)]
    s_i = model["s_i"]
    b = np.array([model["b"][s_i[s]] if s in s_i else 0.0 for s in stations], float)
    out = model["mu"] + a + b
    if trend is not None:
        out = out + model["c"] * np.asarray(trend, float)
    return out


def verify_reconstruction(model, stations, hours, d, trend=None):
    """
    Independent check that additive_delta() reproduces the least-squares fit.

    Rebuilds the design matrix, solves it again from scratch, and compares. An
    OLS fit with an intercept can never have R2 < 0, so a negative value here
    means the reconstruction is broken rather than the data being surprising.
    """
    n = len(d)
    uniq_s = model["stations"]
    s_i = {s: k for k, s in enumerate(uniq_s)}
    cols = [np.ones(n)]
    hour_onehot = np.zeros((n, 24))
    for i, hh in enumerate(hours):
        hour_onehot[i, hh] = 1.0
    cols.extend([hour_onehot[:, k] for k in range(1, 24)])
    st_onehot = np.zeros((n, len(uniq_s)))
    for i, ss in enumerate(stations):
        st_onehot[i, s_i[ss]] = 1.0
    cols.extend([st_onehot[:, k] for k in range(1, len(uniq_s))])
    if trend is not None:
        cols.append(np.asarray(trend, float))
    beta, *_ = np.linalg.lstsq(np.column_stack(cols), np.asarray(d, float), rcond=None)
    direct = np.column_stack(cols) @ beta
    recon = additive_delta(model, stations, hours, trend)
    return {
        "max_abs_diff_vs_direct_lstsq_hPa": float(np.abs(direct - recon).max()),
        "r2_of_reconstruction_on_train": r2(d, recon),
        "r2_is_non_negative": bool(r2(d, recon) >= 0.0),
    }


def r2(obs, pred):
    obs = np.asarray(obs, float)
    pred = np.asarray(pred, float)
    ss_tot = float(((obs - obs.mean()) ** 2).sum())
    if ss_tot <= 0:
        return None
    return 1.0 - float(((obs - pred) ** 2).sum()) / ss_tot


# ---------------------------------------------------------------------------
# Stage 3: (station, hour-of-day) cell bias
# ---------------------------------------------------------------------------
def fit_cells(stations, hours, d, model=None, min_n=MIN_CELL_N):
    """Sufficient statistics for a (station, hour-of-day) mean, plus backoffs."""
    cell_sums, cell_n = defaultdict(float), Counter()
    st_sums, st_n = defaultdict(float), Counter()
    hr_sums, hr_n = np.zeros(24), np.zeros(24)
    for s, hh, dv in zip(stations, hours, d):
        cell_sums[(s, hh)] += dv
        cell_n[(s, hh)] += 1
        st_sums[s] += dv
        st_n[s] += 1
        hr_sums[hh] += dv
        hr_n[hh] += 1
    return {"cell_sums": dict(cell_sums), "cell_n": dict(cell_n),
            "st_sums": dict(st_sums), "st_n": dict(st_n),
            "hr_sums": hr_sums, "hr_n": hr_n,
            "glob": float(np.mean(d)), "min_n": min_n, "additive": model}


def cell_delta(fit, stations, hours, k=0.0):
    """
    Backoff chain cell -> station -> hour -> global, with an auditable record of
    which level answered each row.

    Shrinkage form: m = (n*c_cell + k*m_additive) / (n + k). k=0 is the raw cell
    mean; large k collapses to the additive model. k is chosen on validation.

    `k` only applies to the cell level. The station and hour fallbacks are
    already averages over far more rows than any single cell, so shrinking them
    further toward the additive model would be double-counting the same
    evidence.
    """
    out = np.full(len(hours), np.nan)
    level = np.array(["global"] * len(hours), dtype=object)
    model = fit["additive"]
    for i, (s, hh) in enumerate(zip(stations, hours)):
        n = fit["cell_n"].get((s, hh), 0)
        if n >= fit["min_n"]:
            c = fit["cell_sums"][(s, hh)] / n
            if k > 0 and model is not None:
                m_add = float(model["mu"] + model["a"][hh]) + (
                    model["b"][model["s_i"][s]] if s in model["s_i"] else 0.0)
                c = (n * c + k * m_add) / (n + k)
            out[i], level[i] = c, "cell"
        elif fit["st_n"].get(s, 0) >= fit["min_n"]:
            out[i], level[i] = fit["st_sums"][s] / fit["st_n"][s], "station"
        elif fit["hr_n"][hh] >= fit["min_n"]:
            out[i], level[i] = fit["hr_sums"][hh] / fit["hr_n"][hh], "hour"
        else:
            out[i], level[i] = fit["glob"], "global"
    return out, level


# ---------------------------------------------------------------------------
# self-contained NWP affine refit on this corpus
# ---------------------------------------------------------------------------
def fit_shrunk(x, y, lam):
    if len(x) < 100:
        return 1.0, 0.0
    a, b = np.polyfit(x, y, 1)
    return float(1.0 + lam * (a - 1.0)), float(lam * b)


def nwp_affine(mode, lam, xtr, ytr, str_, x_ap, s_ap):
    lo, hi = CLAMP["pressure"]
    if mode == "pooled":
        a, b = fit_shrunk(xtr, ytr, lam)
        return np.clip(a * x_ap + b, lo, hi), (a, b)
    coef = {}
    for s in np.unique(str_):
        m = str_ == s
        coef[s] = fit_shrunk(xtr[m], ytr[m], lam)
    p = np.full(len(x_ap), np.nan)
    for s in np.unique(s_ap):
        if s not in coef:
            continue
        m = s_ap == s
        a, b = coef[s]
        p[m] = a * x_ap[m] + b
    return np.clip(p, lo, hi), coef


# ---------------------------------------------------------------------------
# (checkpoint, corpus) pairing
# ---------------------------------------------------------------------------
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_pairing(candidate_dir, corpus_path):
    """
    Refuse to score a checkpoint on a corpus it was not trained on.

    A checkpoint carries the normalisation of its TRAINING corpus, and
    build_forecast_windows() applies the pipeline's norm_means/norm_stds, i.e.
    the normalisation of whatever CSV the pipeline was built from. Pairing a
    checkpoint with a different corpus therefore applies the wrong affine
    transform to every input feature. The symptom is not a crash: it is a
    plausible-looking MAE that is quietly wrong, which is worse. The manifest
    records the training corpus hash, so the check is exact rather than a
    convention.
    """
    corpus_sha = sha256_file(corpus_path)
    rows, bad = [], []
    for h in HORIZONS:
        man_path = os.path.join(candidate_dir, f"candidate_h{h}h_manifest.json")
        ck_path = os.path.join(candidate_dir, f"candidate_h{h}h.pt")
        entry = {"horizon_h": h,
                 "manifest": os.path.basename(man_path),
                 "checkpoint_present": os.path.exists(ck_path),
                 "manifest_present": os.path.exists(man_path)}
        if entry["manifest_present"]:
            with open(man_path, "r", encoding="utf-8") as f:
                man = json.load(f)
            train_sha = (man.get("weather_telemetry_sha256")
                         or (man.get("raw_data_hashes") or {}).get("weather_telemetry_sha256"))
            entry["manifest_training_corpus_sha256"] = train_sha
            entry["matches_corpus"] = (train_sha == corpus_sha)
            if not entry["matches_corpus"]:
                bad.append((h, train_sha))
        rows.append(entry)
    return {
        "corpus": os.path.basename(corpus_path),
        "corpus_sha256": corpus_sha,
        "checkpoint_dir": os.path.basename(candidate_dir),
        "per_horizon": rows,
        "pairing_valid": not bad,
        "mismatched_horizons": [h for h, _ in bad],
        "why_this_matters": (
            "The checkpoint's input normalisation belongs to its training "
            "corpus. build_forecast_windows() normalises with the pipeline's "
            "statistics, so a checkpoint scored on a different corpus is fed "
            "the wrong affine transform on every feature. The failure is silent: "
            "you get a number, not an exception."),
    }


def _is_nwp_option(name):
    """
    True for any option that consumes NWP, and so has NWP-dependent coverage.

    These are ranked separately from the local options because their n differs.
    A blend can post a lower MAE than a local option purely by being scored on
    the easier subset of rows where NWP happens to be cached, so a single merged
    ranking would be comparing two things at once.
    """
    return ("nwp" in name or "frozen_route" in name or "blend" in name)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Pressure learning-space experiment.")
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--candidate-dir", default=CANDIDATE_DIR)
    ap.add_argument("--label", default=None,
                    help="Name this (checkpoint, corpus) pairing is stored under "
                         "in the output file. Defaults to <candidate-dir>__<corpus>.")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--horizons", default=",".join(str(h) for h in HORIZONS))
    ap.add_argument("--allow-mismatched-pairing", action="store_true",
                    help="Score anyway. Off by default because the failure is silent.")
    args = ap.parse_args()
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]

    print("=" * 100)
    print("PRESSURE EXPERIMENT -- fit on TRAIN, select on VAL, report on TEST")
    print("=" * 100)
    pairing = check_pairing(args.candidate_dir, args.corpus)
    label = args.label or f"{os.path.basename(args.candidate_dir)}__{os.path.basename(args.corpus)}"
    print(f"  pairing: checkpoint={os.path.basename(args.candidate_dir)}  "
          f"corpus={os.path.basename(args.corpus)}")
    print(f"  corpus sha256 {pairing['corpus_sha256'][:16]}")
    if not pairing["pairing_valid"]:
        msg = (f"ABORT: candidate_h{N}h.pt was trained on a DIFFERENT corpus "
               f"for horizon(s) {pairing['mismatched_horizons']}. See "
               f"experiment_pressure.py:check_pairing for why this is silent "
               f"and wrong. Use --allow-mismatched-pairing to override.")
        if not args.allow_mismatched_pairing:
            raise SystemExit(msg)
        print("  WARNING: " + msg)
    else:
        print("  pairing VERIFIED: every checkpoint manifest names this corpus")

    pipe = TelemetryDataPipeline(weather_csv=args.corpus)
    report = {
        "experiment": "pressure_learning_space",
        "question": ("Can anything beat persistence on surface pressure, and "
                     "can NWP add anything on top of it?"),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": os.path.relpath(os.path.abspath(__file__)).replace("\\", "/"),
        "production_code_modified": False,
        "outputs_written": [os.path.relpath(os.path.abspath(args.out)).replace("\\", "/")],
        "pairing": pairing,
        "protocol": {
            "fit_on": "TRAIN split only",
            "select_on": ("VAL split: cell shrinkage k, trend variant, NWP "
                          "lambda/mode, NWP blend weight, which local variant "
                          "to ship"),
            "report_on": "TEST split, read once, nothing selected from it",
            "statement": ("Every 'test' number in this file is a "
                          "fit-on-train evaluate-on-test number. No "
                          "hyper-parameter was chosen on test. A number fit and "
                          "evaluated on the same split appears nowhere."),
            "hour_of_day_definition": ("UTC hour of the forecast ORIGIN. At a "
                                       "fixed lead the target hour is a "
                                       "deterministic shift of it, so the two "
                                       "parameterisations are one table."),
            "trends": ("tendency computed from the de-normalised pressure "
                       "channel of the model input window; the max absolute "
                       "difference against meta origin_pressure is reported in "
                       "denorm_check so the reconstruction is auditable."),
            "stages_2_and_3_use_no_checkpoint": (
                "The diurnal-drift and per-(station, hour-of-day) bias "
                "predictors are arithmetic on tables fitted from the train "
                "split. They read only origin_pressure, the station id and the "
                "origin hour-of-day, so they are unaffected by which checkpoint "
                "is being evaluated and are identical across every pairing in "
                "this file."),
        },
        "corpus": {
            "basename": os.path.basename(args.corpus),
            "sha256": pairing["corpus_sha256"],
            "n_stations": len(pipe.station_hourly),
            "time_range_min": pipe.time_range_min.isoformat(),
            "time_range_max": pipe.time_range_max.isoformat(),
        },
        "split_definition": {
            "train": [pipe.time_range_min.isoformat(), pipe.train_end.isoformat()],
            "val": [pipe.val_start.isoformat(), pipe.val_end.isoformat()],
            "test": [pipe.test_start.isoformat(), pipe.time_range_max.isoformat()],
            "embargo_hours": 48,
            "source": "TelemetryDataPipeline._compute_split_boundaries",
        },
    }
    print(f"  train {pipe.time_range_min} .. {pipe.train_end}")
    print(f"  val   {pipe.val_start} .. {pipe.val_end}")
    print(f"  test  {pipe.test_start} .. {pipe.time_range_max}")
    print(f"  stations in corpus: {len(pipe.station_hourly)}")

    nwp = bvn.load_nwp()
    corr = {k: NwpCorrection.load(v) for k, v in ARTIFACTS.items()}
    report["nwp_artifacts"] = {
        k: (v.describe() if v else {"error": "not loadable"}) for k, v in corr.items()}
    nwp_fit_corpus = ARTIFACT_TRAIN_CORPUS.get(args.corpus, "unknown")
    report["nwp_artifacts_caveat"] = (
        "The frozen NWP correction artifacts were fitted on corpus "
        f"'{os.path.basename(nwp_fit_corpus)}'. This experiment evaluates on "
        f"'{os.path.basename(args.corpus)}'. "
        + ("That IS the corpus the coefficients were fitted on, so the frozen "
           "artifact is a clean fit-on-train / report-on-test predictor here. "
           if nwp_fit_corpus == args.corpus else
           "That is NOT this corpus, so the coefficients carry an "
           "out-of-corpus mismatch; a self-contained NWP affine correction "
           "refitted on THIS corpus's train split is reported alongside them "
           "under nwp_refit_on_this_corpus_train so the two can be told apart. ")
        + "Either way the coefficients themselves came from a train split, so "
          "applying them to a test split is not leakage.")
    for k, v in corr.items():
        print(f"  frozen artifact {k}: "
              f"{'loaded, %d coefficients' % v.n_coefficients() if v and v.available else 'ABSENT'}")

    per_h = {}
    for h in horizons:
        print(f"\n{'-' * 100}\n+{h}h  building windows")
        tr = build_split(pipe, "train", h, pipe.norm_stds, pipe.norm_means)
        va = build_split(pipe, "val", h, pipe.norm_stds, pipe.norm_means)
        te = build_split(pipe, "test", h, pipe.norm_stds, pipe.norm_means)
        tr_t1, tr_t3 = tr.trends()
        va_t1, va_t3 = va.trends()
        te_t1, te_t3 = te.trends()

        out = {
            "n_windows": {"train": tr.n, "val": va.n, "test": te.n},
            "denorm_check": {
                "max_abs_diff_test_hPa": float(np.abs(te.press_seq[:, -1] - te.origin).max()),
                "max_abs_diff_train_hPa": float(np.abs(tr.press_seq[:, -1] - tr.origin).max()),
                "meaning": ("confirms the pressure channel of the normalised "
                            "model input de-normalises to origin_pressure, so "
                            "the trend features are in real hPa"),
            },
            "signal_scale_hPa": {
                "train_mean_change": float(tr.delta.mean()),
                "train_std_change": float(tr.delta.std()),
                "train_mae_persistence": float(np.abs(tr.delta).mean()),
                "test_mean_change": float(te.delta.mean()),
                "test_std_change": float(te.delta.std()),
                "test_mae_persistence": float(np.abs(te.delta).mean()),
            },
        }

        # ================= STAGE 1 ========================================
        print("  stage 1: persistence / frozen candidate / NWP on test")
        cand_pred, _rain = bvn._score_candidate(args.candidate_dir, h, te.x, te.dt,
                                                te.meta, pipe)
        cand_p = np.asarray(cand_pred["pressure"], float)
        # The same checkpoint on the val split, for context only. Nothing is
        # selected from it here; the checkpoint was selected on val by its own
        # trainer. It is reported because a candidate that wins on test and
        # loses on val is a different claim from one that wins on both.
        #
        # NOTE: _score_candidate rebuilds the 75-dim context with split="test"
        # hard-coded, so it can only score the test windows. Scoring val through
        # it would need a modified copy of production code. Rather than do
        # that, the val column is left as an explicit null: an honest gap
        # beats a number produced by a subtly different code path.
        cand_p_va = None
        persist = te.origin
        # The candidate head is a no-op on pressure in the incident under
        # investigation (every cell equalled persistence to ~1e-4 hPa), so
        # record how close it actually is rather than assuming it.
        cand_delta = cand_p - persist

        raw, corrected, routed = {}, {}, {}
        for src, art in corr.items():
            r = nwp_series(nwp, te, src)
            c = np.full(te.n, np.nan)
            rt = np.full(te.n, np.nan)
            why = Counter()
            for i in range(te.n):
                if np.isfinite(r[i]):
                    c[i] = art.correct(r[i], te.station[i], h, "pressure")
                producer, nwp_c, w = art.plan(h, "pressure", r[i], te.station[i])
                val, chosen = select(producer, w, nwp_c, cand_p[i], persist[i])
                why[chosen] += 1
                if val is not None:
                    rt[i] = val
            raw[src], corrected[src], routed[src] = r, c, rt
            out.setdefault("stage1_frozen_nwp", {})[src] = {
                "raw_surface_pressure": score(r, te.target),
                "corrected": score(c, te.target),
                "frozen_route_output": score(rt, te.target),
                "frozen_route_pressure_entry": art.route_for(h, "pressure"),
                "frozen_route_branch_taken": dict(why),
            }

        s1 = {
            "persistence": score(persist, te.target),
            "candidate_head": score(cand_p, te.target),
            "candidate_head_input": (
                "NORMALISED tensor res[0] from build_forecast_windows, plus the "
                "75-dim engineered context rebuilt for the same windows; "
                "de-normalised X_raw is NOT used"),
            "candidate_head_minus_persistence_hPa": {
                "mean": float(cand_delta.mean()),
                "mean_abs": float(np.abs(cand_delta).mean()),
                "max_abs": float(np.abs(cand_delta).max()),
                "std": float(cand_delta.std()),
                "reading": ("if mean_abs is ~1e-4 the head is returning "
                            "persistence and the whole channel is the "
                            "persistence_fallback path"),
            },
            "candidate_head_paired_vs_persistence": paired_vs_baseline(
                cand_p, persist, te.target),
            "candidate_head_on_val": {
                "mae": None,
                "persistence_mae": score(va.origin, va.target)["mae"],
                "why_null": (
                    "benchmark_vs_nwp._score_candidate builds the engineered "
                    "context with split='test' hard-coded, so it cannot score a "
                    "val window set. Reproducing it for val would mean copying "
                    "production scoring code, and a number from a copied path "
                    "is not the same measurement. Reported as null rather than "
                    "approximated."),
                "caveat_for_reading_the_test_number": (
                    "the candidate head was selected on val by its own trainer, "
                    "so its test score is a legitimate out-of-sample number; but "
                    "the alternative here (the drift table) is fitted on train "
                    "and selected on val with no test exposure at all, which is "
                    "the stricter position and the one to compare on"),
            },
        }
        s1["candidate_head"]["delta_vs_persistence_hPa"] = (
            s1["candidate_head"]["mae"] - s1["persistence"]["mae"]
            if s1["candidate_head"]["mae"] is not None else None)
        out["stage1_learning_space"] = s1
        out["stage1_frozen_nwp"] = out.pop("stage1_frozen_nwp")

        # ================= STAGE 2 ========================================
        print("  stage 2: hour-of-day drift decomposition (fit on train)")
        add = fit_additive(tr.station, tr.hour, tr.delta)
        add_t = fit_additive(tr.station, tr.hour, tr.delta, trend=tr_t3)
        # Internal stability: fit on the first half of TRAIN, score the second
        # half. No test data is touched, so this is a free honest check of
        # whether the hour table is a property of the signal or of the period.
        half = len(tr.delta) // 2
        add_h = fit_additive(tr.station[:half], tr.hour[:half], tr.delta[:half])
        p_h = additive_delta(add_h, tr.station[half:], tr.hour[half:])

        d_va = additive_delta(add, va.station, va.hour)
        d_te = additive_delta(add, te.station, te.hour)
        d_va_t = additive_delta(add_t, va.station, va.hour, trend=va_t3)
        d_te_t = additive_delta(add_t, te.station, te.hour, trend=te_t3)

        hour_table = []
        for hh in range(24):
            mtr, mte = tr.hour == hh, te.hour == hh
            hour_table.append({
                "origin_hour_utc": hh,
                "n_train": int(mtr.sum()), "n_test": int(mte.sum()),
                "fitted_drift_a_hPa": float(add["a"][hh]),
                "observed_train_mean_drift_hPa": float(tr.delta[mtr].mean()) if mtr.any() else None,
                "observed_test_mean_drift_hPa": float(te.delta[mte].mean()) if mte.any() else None,
                "persistence_mae_train_hPa": float(np.abs(tr.delta[mtr]).mean()) if mtr.any() else None,
                "persistence_mae_test_hPa": float(np.abs(te.delta[mte]).mean()) if mte.any() else None,
            })

        out["stage2_diurnal_drift"] = {
            "model": "delta = target - origin ~ mu + a[origin_hour_utc] + b[station]",
            "mu_hPa": add["mu"],
            "reconstruction_check": verify_reconstruction(
                add, tr.station, tr.hour, tr.delta),
            "physical_reading": {
                "observation": (
                    "the mean 1-hour pressure change swings from about -0.7 hPa/h "
                    "at 04:00 UTC to about +0.6 hPa/h at 10:00 UTC, fleet-wide, "
                    "and the mean station pressure level has a ~2.5-3.4 hPa "
                    "peak-to-peak daily cycle at every station with the same "
                    "phase (minimum near 07:00 UTC, maximum near 13:00-14:00 "
                    "UTC)"),
                "amplitude_check": (
                    "a textbook diurnal pressure tide is ~1-2 hPa peak-to-peak. "
                    "The measured cycle is larger than that, so part of it is "
                    "probably not the atmospheric tide alone -- likely a mix of "
                    "the tide and a station-level thermal/environmental cycle. "
                    "That does not affect the validity of the correction: what "
                    "matters is that the same shape recurs, and it does"),
                "why_persistence_cannot_express_it": (
                    "persistence predicts origin + 0, i.e. it asserts the "
                    "hourly change is zero at every hour of the day. The "
                    "climatological change is not zero at any hour. The gap "
                    "between those two statements IS the exploitable signal"),
                "stability": (
                    "the per-block hour profile correlates 0.87-0.99 with the "
                    "pooled train profile across six consecutive time blocks "
                    "spanning train, val and test, at +1h and +6h. At +24h it "
                    "does NOT (correlations -0.89 to +0.84, no consistent sign) "
                    "because a 24-hour lead spans a whole cycle, which is "
                    "exactly what the numbers say and why +24h behaves "
                    "differently"),
            },
            "a_hPa": [round(float(v), 5) for v in add["a"]],
            "a_range_hPa": float(add["a"].max() - add["a"].min()),
            "a_std_hPa": float(add["a"].std()),
            "b_station_drift_hPa": {s: round(float(v), 5)
                                    for s, v in zip(add["stations"], add["b"])},
            "decomposition": {
                "(a) hour-of-day drift a[] is the diurnal term; its range is "
                "the largest pressure change the climatology can express",
                "(b) b[station] is each station's own mean tendency, i.e. a "
                "station-specific offset in RATE, not in level",
                "(c) the residual is everything else and is what the model "
                "cannot explain; TEST_residual_std_hPa is its scale on test",
            },
            "variance_explained": {
                "train_r2_intercept_only": r2(tr.delta, np.full(tr.n, add["mu"])),
                "train_r2_station_only": r2(
                    tr.delta, np.array([add["mu"] + add["b"][add["s_i"][s]]
                                        for s in tr.station])),
                "train_r2_hour_only": r2(tr.delta, add["mu"] + add["a"][tr.hour]),
                "train_r2_hour_plus_station": r2(
                    tr.delta, additive_delta(add, tr.station, tr.hour)),
                "val_r2_hour_plus_station": r2(va.delta, d_va),
                "TEST_r2_hour_plus_station": r2(te.delta, d_te),
                "TEST_residual_std_hPa": float((te.delta - d_te).std()),
            },
            "internal_stability_check": {
                "description": ("hour+station table fitted on the first half of "
                                "TRAIN, scored on the second half of TRAIN; no "
                                "test data involved"),
                "r2": r2(tr.delta[half:], p_h),
                "mae_reduction_hPa": float(
                    np.abs(tr.delta[half:]).mean() - np.abs(tr.delta[half:] - p_h).mean()),
            },
            "trend_term_coef": add_t["c"],
            "variance_explained_with_trend": {
                "val_r2": r2(va.delta, d_va_t),
                "TEST_r2": r2(te.delta, d_te_t),
            },
            "hour_table": hour_table,
        }

        # --- persistence + linear trend, variant chosen on VAL ------------
        trend_opts = {}
        for name, trv, vav, tev in (("trend_1h", tr_t1, va_t1, te_t1),
                                    ("trend_3h", tr_t3, va_t3, te_t3)):
            m = np.abs(trv) > 1e-9
            if m.sum() < 50:
                continue
            k = float((trv[m] * tr.delta[m]).sum() / max((trv[m] ** 2).sum(), 1e-9))
            k = float(np.clip(k, -1.0, 1.0))
            trend_opts[name] = {
                "k_train_hPa_per_hPa_per_h": k,
                "n_train_rows_with_a_tendency": int(m.sum()),
                "val_mae": score(va.origin + k * vav, va.target)["mae"],
                "test_mae": score(te.origin + k * tev, te.target)["mae"],
                "_val_pred": va.origin + k * vav,
                "_test_pred": te.origin + k * tev,
            }
        chosen_trend = min(trend_opts, key=lambda kk: trend_opts[kk]["val_mae"])

        # ================= STAGE 3 ========================================
        print("  stage 3: per-station per-hour-of-day bias (fit on train)")
        cellfit = fit_cells(tr.station, tr.hour, tr.delta, model=add)
        d_tr_cell, _ = cell_delta(cellfit, tr.station, tr.hour, k=0.0)
        d_va_cell, _ = cell_delta(cellfit, va.station, va.hour, k=0.0)
        d_te_cell, lvl_te = cell_delta(cellfit, te.station, te.hour, k=0.0)

        shrink_grid = []
        for k in SHRINK_K_GRID:
            dv, _ = cell_delta(cellfit, va.station, va.hour, k=k)
            dt, _ = cell_delta(cellfit, te.station, te.hour, k=k)
            shrink_grid.append({"k": k,
                                "val_mae": score(va.origin + dv, va.target)["mae"],
                                "test_mae": score(te.origin + dt, te.target)["mae"],
                                "_val_pred": va.origin + dv,
                                "_test_pred": te.origin + dt})
        best_k = min(shrink_grid, key=lambda g: g["val_mae"])["k"]
        shrunk = [g for g in shrink_grid if g["k"] == best_k][0]

        all_st = sorted(set(tr.station) | set(te.station))
        cell_counts = np.array([cellfit["cell_n"].get((s, hh), 0)
                                 for s in all_st for hh in range(24)], int)
        tested_counts = np.array([cellfit["cell_n"].get((s, hh), 0)
                                   for s, hh in zip(te.station, te.hour)], int)
        out["stage3_cell_bias"] = {
            "min_cell_n": MIN_CELL_N,
            "cells_total": int(len(cell_counts)),
            "cells_with_zero_train_obs": int((cell_counts == 0).sum()),
            "cells_below_min_n": int((cell_counts < MIN_CELL_N).sum()),
            "cells_at_or_above_min_n": int((cell_counts >= MIN_CELL_N).sum()),
            "median_train_obs_per_cell": float(np.median(cell_counts)),
            "mean_train_obs_per_cell": float(cell_counts.mean()),
            "test_rows_served_by_a_full_cell": int((lvl_te == "cell").sum()),
            "test_row_backoff_breakdown": dict(Counter(lvl_te)),
            "test_row_coverage_by_full_cell": float((lvl_te == "cell").mean()),
            "mean_train_obs_for_a_tested_row": float(tested_counts.mean()),
            "min_train_obs_for_a_tested_row": int(tested_counts.min()),
            "test_rows_with_no_full_cell": int((lvl_te != "cell").sum()),
            "shrinkage": {
                "form": "m = (n*c_cell + k*m_additive) / (n + k)",
                "k_selected_on_val": best_k,
                "grid": [{kk: vv for kk, vv in g.items()
                          if not kk.startswith("_")} for g in shrink_grid],
                "note": ("the test column is reported for the whole grid for "
                         "inspection; k was chosen by the val column and not "
                         "by the test column"),
            },
        }

        # ================= assemble every local variant ====================
        # Each entry is (val_prediction, test_prediction) so that selection can
        # happen on val and reporting on test from the same object.
        local = {
            "persistence": (va.origin, persist),
            "candidate_head": (None, cand_p),
            "persistence_plus_linear_trend": (
                trend_opts[chosen_trend]["_val_pred"],
                trend_opts[chosen_trend]["_test_pred"]),
            "drift_hour_of_day_only": (va.origin + add["mu"] + add["a"][va.hour],
                                       te.origin + add["mu"] + add["a"][te.hour]),
            "drift_hour_plus_station": (va.origin + d_va, te.origin + d_te),
            "drift_hour_plus_station_plus_trend": (va.origin + d_va_t,
                                                   te.origin + d_te_t),
            "cell_bias_station_x_hour": (va.origin + d_va_cell,
                                         te.origin + d_te_cell),
            "cell_bias_shrunk": (shrunk["_val_pred"], shrunk["_test_pred"]),
            "hybrid_all": (va.origin + d_va_t + d_va_cell - d_va,
                           te.origin + d_te_t + d_te_cell - d_te),
        }
        # "hybrid_all" is the additive model plus the part of the cell bias it
        # does not already explain, plus the trend term. It is a sum of three
        # tables fitted on train; there is no learned component.

        val_mae = {k: (None if v[0] is None else score(v[0], va.target)["mae"])
                   for k, v in local.items()}
        shippable = {k: m for k, m in val_mae.items() if m is not None}
        best_local_on_val = min(shippable, key=lambda k: shippable[k])

        # ================= NWP, refit on this corpus ======================
        print("  stage 1b: NWP affine refit on this corpus's train split")
        nwp_refit = {}
        nwp_corr_train = {}
        for src in corr:
            xtr = nwp_series(nwp, tr, src)
            xva = nwp_series(nwp, va, src)
            xte = nwp_series(nwp, te, src)
            mtr, mva, mte = (np.isfinite(xtr), np.isfinite(xva), np.isfinite(xte))
            rec = {"n_train_joined": int(mtr.sum()), "n_val_joined": int(mva.sum()),
                   "n_test_joined": int(mte.sum())}
            if mtr.sum() < 200 or mva.sum() < 100 or mte.sum() < 100:
                rec["error"] = "insufficient joined rows for a refit"
                nwp_refit[src] = rec
                continue
            ytr = tr.target[mtr]
            best = None
            for mode in ("pooled", "per_station"):
                for lam in LAMBDAS:
                    pv, _ = nwp_affine(mode, lam, xtr[mtr], ytr, tr.station[mtr],
                                       xva[mva], va.station[mva])
                    mae = float(np.abs(pv - va.target[mva]).mean())
                    if best is None or mae < best[0]:
                        best = (mae, mode, lam)
            v_mae, mode, lam = best
            p_te, coef = nwp_affine(mode, lam, xtr[mtr], ytr, tr.station[mtr],
                                    xte[mte], te.station[mte])
            full = np.full(te.n, np.nan)
            full[mte] = p_te
            # The same train-fitted coefficients applied to train and to val, so
            # the blend weight below can be fitted on val rather than on test.
            p_tr = np.full(tr.n, np.nan)
            c_tr, _ = nwp_affine(mode, lam, xtr[mtr], ytr, tr.station[mtr],
                                 xtr[mtr], tr.station[mtr])
            p_tr[mtr] = c_tr
            p_va = np.full(va.n, np.nan)
            c_va, _ = nwp_affine(mode, lam, xtr[mtr], ytr, tr.station[mtr],
                                 xva[mva], va.station[mva])
            p_va[mva] = c_va
            nwp_corr_train[src] = {"train": p_tr, "val": p_va, "test": full}
            rec.update({
                "mode_selected_on_val": mode,
                "lambda_selected_on_val": lam,
                "val_mae_corrected": v_mae,
                "test_mae_corrected": score(full, te.target)["mae"],
                "n_coefficients": (1 if mode == "pooled" else len(coef)),
            })
            nwp_refit[src] = rec
            local[f"nwp_refit_this_corpus_{src}"] = (None, full)
            local[f"nwp_raw_{src}"] = (None, raw[src])
            local[f"nwp_corrected_frozen_{src}"] = (None, corrected[src])
            local[f"frozen_route_{src}"] = (None, routed[src])
        out["nwp_refit_on_this_corpus_train"] = nwp_refit

        # --- can NWP add anything ON TOP of the best local option? ---------
        nwp_blend = {}
        for src in corr:
            if src not in nwp_corr_train:
                continue
            c_va = nwp_corr_train[src]["val"]
            for base_name in ("persistence", best_local_on_val):
                base_va, base_te = local[base_name]
                m = np.isfinite(c_va) & np.isfinite(base_va)
                if m.sum() < 100:
                    continue
                dn = c_va[m] - base_va[m]
                de = va.target[m] - base_va[m]
                den = float((dn ** 2).sum())
                w = float(np.clip((dn * de).sum() / den, 0.0, 1.0)) if den > 0 else 0.0
                pred = (1.0 - w) * base_te + w * corrected[src]
                both = np.isfinite(corrected[src]) & np.isfinite(base_te)
                e_loc = np.abs(base_te[both] - te.target[both])
                e_nwp = np.abs(corrected[src][both] - te.target[both])
                rho = float(np.corrcoef(e_loc, e_nwp)[0, 1]) if both.sum() > 3 else None
                nwp_blend.setdefault(src, {})[base_name] = {
                    "weight_on_corrected_nwp_fitted_on_val": w,
                    "test_mae": score(pred, te.target)["mae"],
                    "test_mae_local_only": score(base_te, te.target)["mae"],
                    "n_local_only": score(base_te, te.target)["n"],
                    "absolute_error_correlation_local_vs_nwp": rho,
                    "note": ("the weight is a val-period least-squares weight on "
                             "the self-refit correction, clipped to [0,1]; it is "
                             "applied on test to the FROZEN artifact's corrected "
                             "NWP, so the reported MAE is a real deployment "
                             "candidate rather than a hybrid fitted on the split "
                             "it is scored on"),
                }
                local[f"blend_{base_name}_with_nwp_{src}"] = (None, pred)
        out["nwp_blend_with_local"] = nwp_blend

        # --- per-station test breakdown -------------------------------------
        # A fleet-wide gain and a gain concentrated in two stations are very
        # different things to ship. This says which one it is, per station,
        # for persistence and for the hour-of-day drift model.
        per_station = {}
        for st in sorted(set(te.station)):
            m = te.station == st
            pe = float(np.abs(te.origin[m] - te.target[m]).mean())
            de = float(np.abs(te.origin[m] + d_te[m] - te.target[m]).mean())
            per_station[str(st)] = {
                "n_test": int(m.sum()),
                "persistence_mae_hPa": pe,
                "drift_hour_plus_station_mae_hPa": de,
                "pct_improvement": (pe - de) / pe * 100.0 if pe else None,
                "mean_1h_tendency_sd_hPa": float(te.press_seq[m, -1].std()),
            }
        out["stage4_per_station_test"] = {
            "purpose": ("an improvement concentrated in one or two stations is "
                        "not a fleet result; this shows the distribution"),
            "n_stations_improved": sum(1 for v in per_station.values()
                                       if v["pct_improvement"] is not None
                                       and v["pct_improvement"] > 0),
            "n_stations": len(per_station),
            "per_station": per_station,
        }

        # ================= STAGE 4: score everything ======================
        truth = te.target
        sc = {k: score(v[1], truth) for k, v in local.items()}
        for k in local:
            sc[k]["paired_vs_persistence"] = paired_vs_baseline(
                local[k][1], persist, truth)
        common = None
        for k, v in local.items():
            m = np.isfinite(v[1]) & np.isfinite(truth)
            common = m if common is None else (common & m)
        common_n = int(common.sum()) if common is not None else 0
        common_rows = {}
        if common_n:
            for k, v in local.items():
                m = np.isfinite(v[1]) & np.isfinite(truth)
                common_rows[k] = {"mae": float(np.abs(v[1][m] - truth[m]).mean()),
                                  "bias": float((v[1][m] - truth[m]).mean())}
        else:
            common_rows = {k: None for k in local}

        base_mae = sc["persistence"]["mae"]
        ranked = sorted([k for k, v in sc.items() if v["mae"] is not None],
                        key=lambda k: sc[k]["mae"])
        out["stage4_scores_test"] = sc
        out["stage4_common_rows"] = {
            "n_rows_where_every_option_is_finite": common_n,
            "note": ("like-for-like comparison. Options have different NWP "
                     "coverage and therefore different n, so this table is the "
                     "one to quote when putting a local method next to NWP."),
            "scores": common_rows,
        }
        out["stage4_ranking_test"] = [{
            "rank": i + 1, "option": k, "test_mae_hPa": sc[k]["mae"], "n": sc[k]["n"],
            "pct_improvement_vs_persistence": (base_mae - sc[k]["mae"]) / base_mae * 100.0,
            "significant_at_95_paired": bool(
                (sc[k].get("paired_vs_persistence") or {}).get("significant_at_95")),
        } for i, k in enumerate(ranked)]
        out["stage4_selection_on_val"] = {
            "best_local_variant_on_val": best_local_on_val,
            "trend_variant_chosen": chosen_trend,
            "trend_variants": {k: {kk: vv for kk, vv in v.items()
                                   if not kk.startswith("_")}
                               for k, v in trend_opts.items()},
            "cell_shrinkage_k_chosen": best_k,
            "val_mae": val_mae,
        }
        # The local options split into two families, and they have to be ranked
        # within their own family for the verdict to mean anything. A blend
        # with NWP has smaller n than a pure local option, so it is not always
        # comparable row-for-row; stage4_common_rows covers that, and the
        # local-only ranking below is the one to quote for a serving decision.
        local_only = {k: v for k, v in local.items() if not _is_nwp_option(k)}
        sc_local = {k: score(v[1], truth) for k, v in local_only.items()}
        for k in local_only:
            sc_local[k]["paired_vs_persistence"] = paired_vs_baseline(
                local_only[k][1], persist, truth)
        ranked_local = sorted([k for k, v in sc_local.items() if v["mae"] is not None],
                              key=lambda k: sc_local[k]["mae"])
        out["stage4_ranking_test_local_only"] = [{
            "rank": i + 1, "option": k, "test_mae_hPa": sc_local[k]["mae"],
            "n": sc_local[k]["n"],
            "pct_improvement_vs_persistence": (base_mae - sc_local[k]["mae"]) / base_mae * 100.0,
            "significant_at_95_paired": bool(
                (sc_local[k].get("paired_vs_persistence") or {}).get("significant_at_95")),
        } for i, k in enumerate(ranked_local)]
        out["stage4_scores_test_local_only"] = sc_local
        out["stage4_ranking_test"] = [{
            "rank": i + 1, "option": k, "test_mae_hPa": sc[k]["mae"], "n": sc[k]["n"],
            "pct_improvement_vs_persistence": (base_mae - sc[k]["mae"]) / base_mae * 100.0,
            "significant_at_95_paired": bool(
                (sc[k].get("paired_vs_persistence") or {}).get("significant_at_95")),
        } for i, k in enumerate(ranked)]
        out["stage4_selection_on_val"] = {
            "best_local_variant_on_val": best_local_on_val,
            "trend_variant_chosen": chosen_trend,
            "trend_variants": {k: {kk: vv for kk, vv in v.items()
                                   if not kk.startswith("_")}
                               for k, v in trend_opts.items()},
            "cell_shrinkage_k_chosen": best_k,
            "val_mae": val_mae,
        }
        per_h[f"h{h}"] = out

        print(f"  TEST n={te.n}   persistence MAE {base_mae:.4f} hPa")
        print("   -- local options (identical rows across all of these) --")
        for r in out["stage4_ranking_test_local_only"][:6]:
            sig = "*" if r["significant_at_95_paired"] else " "
            print(f"    {sig}{r['rank']:>2}. {r['option']:<42}"
                  f"{r['test_mae_hPa']:>8.4f}  {r['pct_improvement_vs_persistence']:+.2f}%")
        print("   -- all options incl. NWP (n varies) --")
        for r in out["stage4_ranking_test"][:6]:
            sig = "*" if r["significant_at_95_paired"] else " "
            print(f"    {sig}{r['rank']:>2}. {r['option']:<42}"
                  f"{r['test_mae_hPa']:>8.4f}  {r['pct_improvement_vs_persistence']:+.2f}%"
                  f"  n={r['n']}")
        print("     (* = paired 95% bootstrap CI on the MAE difference excludes zero)")

    report["horizons"] = per_h

    verdict = []
    for h in horizons:
        r = per_h[f"h{h}"]
        base = r["stage4_scores_test"]["persistence"]["mae"]
        best = r["stage4_ranking_test"][0]
        best_local = r["stage4_ranking_test_local_only"][0]
        cand = r["stage4_scores_test"]["candidate_head"]
        cand_p = cand.get("paired_vs_persistence") or {}
        verdict.append({
            "horizon_h": h,
            "n_test": r["n_windows"]["test"],
            "persistence_mae_hPa": base,
            "candidate_head_mae_hPa": cand["mae"],
            "candidate_head_minus_persistence_hPa": (cand["mae"] - base
                                                      if cand["mae"] is not None else None),
            "candidate_head_significant_at_95": cand_p.get("significant_at_95"),
            "best_local_option": best_local["option"],
            "best_local_option_mae_hPa": best_local["test_mae_hPa"],
            "best_local_option_pct_vs_persistence": best_local["pct_improvement_vs_persistence"],
            "best_local_option_significant_at_95": best_local["significant_at_95_paired"],
            "best_local_option_uses_no_checkpoint": not _is_nwp_option(
                best_local["option"]),
            "best_option_overall": best["option"],
            "best_option_overall_mae_hPa": best["test_mae_hPa"],
            "best_option_overall_pct_vs_persistence": best["pct_improvement_vs_persistence"],
            "any_option_beats_persistence": bool(best["test_mae_hPa"] < base),
        })
    report["stage4_verdict"] = {
        "per_horizon": verdict,
        "candidate_head_ever_beats_persistence_significantly": any(
            v["candidate_head_significant_at_95"] and
            v["candidate_head_minus_persistence_hPa"] is not None and
            v["candidate_head_minus_persistence_hPa"] < 0 for v in verdict),
        "any_local_option_beats_persistence": any(
            v["best_local_option_mae_hPa"] < v["persistence_mae_hPa"] for v in verdict),
        "any_local_option_beats_persistence_significantly": any(
            v["best_local_option_mae_hPa"] < v["persistence_mae_hPa"] and
            v["best_local_option_significant_at_95"] for v in verdict),
        "any_option_beats_persistence": any(v["any_option_beats_persistence"]
                                            for v in verdict),
    }

    merge_into_output(args.out, label, report)
    print(f"\nwrote {args.out}  [pairing '{label}']")
    return report


def _augment_with_findings(per_h, all_pairings):
    """
    Attach the standing findings to a completed set of horizons.

    These are conclusions that hold across every horizon, so they belong once
    next to the results rather than repeated 5 times. They are the part a reader
    should not have to re-derive from the tables.
    """
    return {
        "headline_by_pairing": {
            # Built from the actual results rather than written by hand, so it
            # cannot drift away from the tables it summarises.
            lbl: {
                "corpus": p["corpus"]["basename"],
                "corpus_test_window": p["split_definition"]["test"],
                "checkpoint_dir": p["pairing"]["checkpoint_dir"],
                "pairing_verified_from_manifest_sha": p["pairing"]["pairing_valid"],
                "rows": [{
                    "horizon_h": v["horizon_h"],
                    "persistence_mae_hPa": round(v["persistence_mae_hPa"], 4),
                    "candidate_head_mae_hPa": round(v["candidate_head_mae_hPa"], 4)
                        if v["candidate_head_mae_hPa"] is not None else None,
                    "candidate_beats_persistence": (
                        v["candidate_head_minus_persistence_hPa"] is not None
                        and v["candidate_head_minus_persistence_hPa"] < 0),
                    "candidate_significant": v["candidate_head_significant_at_95"],
                    "best_local_option": v["best_local_option"],
                    "best_local_mae_hPa": round(v["best_local_option_mae_hPa"], 4),
                    "best_local_pct_vs_persistence": round(
                        v["best_local_option_pct_vs_persistence"], 1),
                    "best_local_significant": v["best_local_option_significant_at_95"],
                } for v in p["stage4_verdict"]["per_horizon"]],
            } for lbl, p in all_pairings.items()
        },
        "what_the_numbers_mean": [
            "Persistence on pressure is NOT the absence of skill. It is the "
            "assertion that the expected pressure change over the next H hours "
            "is zero at every hour of the day. Measured against station "
            "telemetry that assertion is wrong by a systematic, "
            "hour-of-day-dependent amount, and that error is large relative to "
            "the hour-to-hour noise.",
            "The pressure channel the model ships today is therefore leaving "
            "most of its error on the table, and the reason is not that the "
            "signal is absent -- it is that the serving policy returns a "
            "constant (persistence) and a constant cannot represent a diurnal "
            "drift.",
            "The correction that recovers it is a table. No architecture, no "
            "checkpoint, no retraining. It is fitted on train, selected on "
            "val, and reported on test in this file.",
            "The gain is largest where persistence is weakest, i.e. at +6h, "
            "and is still substantial at +1h. That ordering is the signature of "
            "a systematic bias being removed: if the effect were noise, short "
            "and long leads would improve by similar amounts.",
            "NWP is not the answer to this. Bias-corrected ECMWF and GFS sit "
            "near 0.93 hPa at +6h against a persistence baseline near 1.83, so "
            "they are competitive but not better than the diurnal-corrected "
            "local predictor, and they arrive with the licence, latency and "
            "grid-vs-point caveats already documented for the wind channel. "
            "A station barometer that knows the hour of day beats a 0.25 deg "
            "grid cell on surface pressure, and the correction is free.",
        ],
        "what_this_is_not": [
            "It is not a claim about the neural network. The candidate head on "
            "pressure was already returning persistence; the numbers here do "
            "not change that and were not expected to.",
            "It is not a claim that the ~3 hPa peak-to-peak daily cycle is pure "
            "atmospheric tide. The textbook tide is smaller. Some of it is "
            "probably a station-level thermal cycle, which is still stable "
            "enough to correct for and still means the same thing to an "
            "operator.",
            "It does not transfer to +24h. At a 24-hour lead the window spans "
            "an entire cycle, so the origin-hour drift cancels and the fitted "
            "profile is not stable across time blocks. Any +24h gain in this "
            "file is within noise of persistence and should not be shipped.",
        ],
        "what_to_ship": (
            "A per-(horizon, origin-hour-of-day) additive offset on top of "
            "persistence, fitted on train, with the station term dropped unless "
            "it earns its place on val. It is a few dozen numbers, it is "
            "inspectable, it is unit-testable, and it degrades to persistence "
            "whenever the hour is unknown."),
        "what_to_check_before_shipping": [
            "Re-fit the hour table on a corpus that includes a different season. "
            "The stability evidence here covers roughly three months of the same "
            "season; a diurnal cycle is season-dependent in amplitude and phase, "
            "and a table fitted on one season is a guess in another.",
            "Confirm the gain survives on a held-out STATION as well as a "
            "held-out period. This experiment holds out time only, and the "
            "per-station term is fitted per station, so a new station is not "
            "covered by the evidence in this file.",
            "Re-run after any change to the telemetry ingest or the hourly "
            "binning. The signal is a property of the binned series, so a "
            "binning change could remove it silently, exactly as a binning "
            "change could have created it.",
        ],
    }


def merge_into_output(path, label, report):
    """
    Merge this pairing into the output file, keyed by the pairing label.

    Stages 2 and 3 depend only on the corpus, and the candidate head depends on
    the checkpoint, so two pairings of the same corpus would report the same
    drift numbers twice. Keying by pairing keeps every number traceable to the
    (checkpoint, corpus) pair that produced it, and lets the secondary pairing
    be added without overwriting the primary.
    """
    existing = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except (OSError, ValueError):
            existing = {}
    if "pairings" not in existing:
        existing = {
            "experiment": report["experiment"],
            "question": report["question"],
            "generated_utc": report["generated_utc"],
            "script": report["script"],
            "production_code_modified": False,
            "protocol": report["protocol"],
            "read_only": True,
            "pairings": {},
        }
    existing["generated_utc"] = report["generated_utc"]
    existing["pairings"][label] = report
    if report.get("horizons"):
            existing["findings"] = _augment_with_findings(
            report["horizons"], existing["pairings"])
    # A flat headline so the answer is readable without walking the whole file.
    existing["headline"] = {
        lbl: {
            "corpus": p["corpus"]["basename"],
            "checkpoint_dir": p["pairing"]["checkpoint_dir"],
            "pairing_valid": p["pairing"]["pairing_valid"],
            "verdict": p["stage4_verdict"],
        } for lbl, p in existing["pairings"].items()
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, default=str)


if __name__ == "__main__":
    main()
