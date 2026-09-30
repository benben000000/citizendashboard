"""
experiment_temperature.py -- DIAGNOSTIC EXPERIMENT (read-only analysis).

PURPOSE
    Establish what temperature skill is ACTUALLY ACHIEVABLE per horizon, decide
    whether the gap to ECMWF is closable by the neural head, and produce an
    honest per-horizon verdict.

CORPUS / CHECKPOINT PAIRING  (this is the whole ballgame -- see below)
    The checkpoint's input normalisation belongs to its TRAINING corpus, and
    build_forecast_windows() applies the pipeline's norm_means/norm_stds to the
    window it builds. Pairing a checkpoint with the wrong corpus applies the
    wrong affine transform to every input and scores the model out of its own
    distribution. Every number below is therefore tagged with its pair.

        PAIR A  (PRIMARY, correctly paired)
            checkpoint : data/candidate_artifacts_windfix/candidate_h{N}h.pt
            corpus     : data/weather_telemetry_current.csv  (fc8f7b4edc11)
        PAIR B  (SECONDARY, correctly paired)
            checkpoint : data/candidate_artifacts/candidate_h{N}h.pt
            corpus     : data/weather_telemetry.csv           (86ce906453aa)
    data/candidate_artifacts_leaky_olddata/ is an ablation arm and is EXCLUDED.

STAGE 1 -- MEASUREMENT (per horizon, on IDENTICAL windows, TEST split)
    persistence | candidate head | sklearn GradientBoostingRegressor |
    sklearn Ridge | ECMWF IFS 0.25 (raw / bias-corrected) | NOAA GFS |
    diurnal-delta persistence
    The GBM and Ridge are the CRITICAL CONTROL. If the GBM cannot beat
    persistence, the horizon is not forecastable from station-local telemetry
    and further architecture work is wasted. Their numbers set the ceiling.

STAGE 2 -- ARCHITECTURE DIAGNOSIS
    * cross-horizon correlation of the head's temperature error
    * sd(d_temp) vs sd(target - origin): under-dispersion / over-regularisation
    * neural + GBM ensemble
    * NWP temperature as an INPUT feature, and learning ECMWF's systematic
      error as a residual (the most promising route to beating ECMWF)

STAGE 3 -- HONEST VERDICT per horizon.

SPLIT CONTRACT
    Chronological 60/20/20 with a 48h embargo, from
    dataset.TelemetryDataPipeline. EVERY model fit here (GBM, Ridge, diurnal
    delta, the ECMWF/GFS bias correction, the dispersion scale, the ensemble
    weights) is fit on the TRAIN windows of ITS OWN corpus and applied
    unchanged to that corpus's TEST windows. Nothing is selected on TEST.

    The two frozen artifacts that predate this experiment
    (data/inference_policy.json and data/nwp_correction_*.json) carry
    dataset_hashes / test_report values that pin them to weather_telemetry.csv,
    so they are CORRECTLY PAIRED with PAIR B and are reported there. Under
    PAIR A they are a corpus mismatch and are reported only with an explicit
    flag, alongside a bias correction refitted on PAIR A's own TRAIN split.

OUTPUT: data/experiment_temperature.json
Does not modify model.py, dataset.py, train.py, or any artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(HERE, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from dataset import (  # noqa: E402
    TelemetryDataPipeline,
    build_feature_augmented_forecast_windows,
    DEFAULT_SEQ_LEN,
)

PAIRS = [
    {
        "id": "A_windfix_on_current",
        "role": "PRIMARY",
        "csv": "weather_telemetry_current.csv",
        "ckpt_dir": "candidate_artifacts_windfix",
    },
    {
        "id": "B_candidate_on_original",
        "role": "SECONDARY",
        "csv": "weather_telemetry.csv",
        "ckpt_dir": "candidate_artifacts",
    },
]

HORIZONS = [1, 3, 6, 12, 24]
SEED = 42

# Station flagged as unreliable in BOTH corpora: its lag-1 |dT| is 1.8-3.8 degC
# against 0.6-0.8 for every other station, and its lag-24 |dT| (5.3-11.7) is
# larger than its lag-12 (8.6-11.8), i.e. it has no usable diurnal structure.
# Sensitivity numbers are reported with it removed.
BROKEN_STATION = "VEpdDpBK"

OUT_PATH = os.path.join(HERE, "data", "experiment_temperature.json")

_log_t0 = time.time()


def log(msg):
    print(f"[{time.time() - _log_t0:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def mae(p, t):
    return float(np.mean(np.abs(np.asarray(p, float) - np.asarray(t, float))))


def rmse(p, t):
    e = np.asarray(p, float) - np.asarray(t, float)
    return float(np.sqrt(np.mean(e ** 2)))


def _block_index(blocks):
    uniq = np.array(sorted(set(blocks)))
    return uniq, {b: np.where(np.asarray(blocks) == b)[0] for b in uniq}


def block_bootstrap_mae(err, blocks, n_boot=1500, seed=SEED):
    """Percentile CI of the MAE, resampling contiguous (station, day) blocks.

    Consecutive hourly windows share 23 of 24 input hours, so an i.i.d.
    window bootstrap would badly understate the CI. Blocks are (station,
    calendar day of origin), which keeps the serial correlation inside a
    resampled unit."""
    err = np.asarray(err, float)
    if len(err) < 2:
        return {"mae": float(err.mean()) if len(err) else None,
                "ci95": [None, None], "n_blocks": 0}
    uniq, idx_by_block = _block_index(blocks)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for k in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by_block[b] for b in pick])
        stats[k] = err[rows].mean()
    return {
        "mae": float(err.mean()),
        "ci95": [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))],
        "n_blocks": int(len(uniq)),
    }


def paired_block_test(err_a, err_b, blocks, n_boot=1500, seed=SEED + 1):
    """Paired (station, day)-block bootstrap on the MAE DIFFERENCE a - b.
    Negative delta => a is better. p is read off the bootstrap distribution."""
    err_a = np.asarray(err_a, float)
    err_b = np.asarray(err_b, float)
    d = err_a - err_b
    uniq, idx_by_block = _block_index(blocks)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for k in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by_block[b] for b in pick])
        stats[k] = d[rows].mean()
    lo, hi = np.percentile(stats, [2.5, 97.5])
    sd = float(stats.std(ddof=1))
    z = abs(float(d.mean())) / (sd + 1e-12)
    return {
        "delta_mae": float(d.mean()),
        "ci95_delta": [float(lo), float(hi)],
        "p_two_sided": float(math.erfc(z / math.sqrt(2.0))),
        "significant_at_5pct": bool(hi < 0 or lo > 0),
    }


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# NWP
# ---------------------------------------------------------------------------

def load_nwp():
    """Same merge rule as benchmark_vs_nwp.load_nwp: shortest cached window is
    applied first and only fills timestamps that are still missing."""
    cache = os.path.join(HERE, "data", "nwp_benchmark_cache.json")
    with open(cache, "r", encoding="utf-8") as f:
        raw = json.load(f)
    FIELDS = ["temperature_2m", "relative_humidity_2m", "surface_pressure",
              "wind_speed_10m", "precipitation"]
    grouped = {}
    for k, v in raw.items():
        if k.startswith("_") or not isinstance(v, dict):
            continue
        times = v.get("time") or []
        if not times:
            continue
        base = f"{v.get('station_id')}|{v.get('model')}"
        grouped.setdefault(base, []).append((len(times), times[-1], v))
    out = {}
    for base, entries in grouped.items():
        entries.sort(key=lambda t: (t[0], [-ord(c) for c in t[1]]))
        merged = {}
        for _span, _end, v in entries:
            for i, t in enumerate(v.get("time", [])):
                row = merged.setdefault(t, {})
                for fld in FIELDS:
                    arr = v.get(fld)
                    if arr is not None and i < len(arr) and arr[i] is not None:
                        row.setdefault(fld, arr[i])
        out[base] = {t: {fld: merged[t].get(fld) for fld in FIELDS}
                     for t in sorted(merged)}
    return out


_NWP_CACHE = None


def nwp_for(nwp, station_id, model, iso_ts):
    e = nwp.get(f"{station_id}|{model}")
    if e is None:
        return None
    return e.get(iso_ts[:13] + ":00")


def arrayise(nwp, meta, model, ts_key="target_timestamp", field="temperature_2m"):
    out = np.full(len(meta), np.nan)
    for i, m in enumerate(meta):
        rec = nwp_for(nwp, m["station_id"], model, m[ts_key])
        if rec is not None and rec.get(field) is not None:
            out[i] = float(rec[field])
    return out


def shrunk_affine_fit(x, y, lam):
    """Replicates nwp_correction's estimator:
         corrected = a*nwp + b,  a = 1 + lam*(a_raw - 1),  b = lam*b_raw
    Shrinkage toward the IDENTITY map. Returns (a, b) or None."""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 200:
        return None
    a_raw, b_raw = np.polyfit(x[m], y[m], 1)
    return float(1.0 + lam * (a_raw - 1.0)), float(lam * b_raw)


def shrunk_affine_apply(x, a, b, lo=-20.0, hi=60.0):
    out = np.full(len(x), np.nan)
    m = np.isfinite(x)
    out[m] = np.clip(a * x[m] + b, lo, hi)
    return out


# ---------------------------------------------------------------------------
# candidate head
# ---------------------------------------------------------------------------

def score_candidate_head(ckpt_dir, h, aug, meta):
    """Run candidate_h{N}h.pt on the NORMALISED telemetry tensor aug[0] plus the
    75-dim context.

    Feeding de-normalised X_raw here is the historical bug that cost ~35% MAE at
    +12h, so this function takes the normalised tensor and has no code path that
    can accept a de-normalised one. The context MUST be built for exactly the
    same windows (aug comes from the same call that produced meta)."""
    import torch
    from model import GarciaWeatherLNNFeatured

    ck_path = os.path.join(HERE, "data", ckpt_dir, f"candidate_h{h}h.pt")
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    man = ck.get("manifest", {}) or {}
    model = GarciaWeatherLNNFeatured(
        input_dim=int(man.get("input_dim", 8)),
        context_dim=int(man.get("context_dim", 75)),
        hidden_dim=int(man.get("hidden_dim", 32)),
        use_two_stage_precipitation=True,
    )
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    telemetry = aug[0].float()
    context = aug[1].float()
    dt = aug[2].float()
    origin = torch.tensor(
        np.column_stack([[m[c] for m in meta] for c in
                         ("origin_temperature", "origin_humidity", "origin_pressure",
                          "origin_wind_speed", "origin_wind_u", "origin_wind_v")]),
        dtype=torch.float32)

    with torch.no_grad():
        out = model(telemetry, context, dt, origin_weather=origin)
    pred = out["temperature"].squeeze(-1).cpu().numpy().astype(np.float64)
    return pred, {
        "checkpoint": f"data/{ckpt_dir}/candidate_h{h}h.pt",
        "manifest_hidden_dim": int(man.get("hidden_dim", 32)),
        "manifest_weather_telemetry_sha256": man.get("weather_telemetry_sha256"),
        "manifest_training_date": man.get("training_date"),
        "manifest_seed": man.get("seed"),
    }


# ---------------------------------------------------------------------------
# one (pair, horizon) cell
# ---------------------------------------------------------------------------

def run_cell(pair, h, pipe, nwp, ecmwf_frozen, gfs_frozen, verbose=True):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import Ridge

    def new_gbm():
        return GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.06, max_depth=3,
            max_features=0.6, subsample=0.9, random_state=SEED)

    tr = build_feature_augmented_forecast_windows(
        pipeline=pipe, split="train", horizon=h, seq_len=DEFAULT_SEQ_LEN,
        return_metadata=True)
    te = build_feature_augmented_forecast_windows(
        pipeline=pipe, split="test", horizon=h, seq_len=DEFAULT_SEQ_LEN,
        return_metadata=True)
    n_tr, n_te = tr[0].shape[0], te[0].shape[0]
    Ctr, Cte = tr[1].numpy().astype(np.float64), te[1].numpy().astype(np.float64)
    mtr, mte = tr[7], te[7]

    o_tr = np.array([m["origin_temperature"] for m in mtr], float)
    y_tr = np.array([m["target_temperature"] for m in mtr], float)
    o_te = np.array([m["origin_temperature"] for m in mte], float)
    y_te = np.array([m["target_temperature"] for m in mte], float)
    resid_tr, resid_te = y_tr - o_tr, y_te - o_te
    blocks = np.array([f"{m['station_id']}|{m['origin_timestamp'][:10]}" for m in mte])
    healthy = np.array([m["station_id"] != BROKEN_STATION for m in mte])

    # ---- neural head ----------------------------------------------------
    p_nn, nn_meta = score_candidate_head(pair["ckpt_dir"], h, te, mte)
    dtemp_te = p_nn - o_te
    p_nn_tr, _ = score_candidate_head(pair["ckpt_dir"], h, tr, mtr)
    dtemp_tr = p_nn_tr - o_tr

    # ---- NWP on the same rows -------------------------------------------
    ecm_t = arrayise(nwp, mte, "ecmwf_ifs025")
    ecm_o = arrayise(nwp, mte, "ecmwf_ifs025", ts_key="origin_timestamp")
    ecm_ttr = arrayise(nwp, mtr, "ecmwf_ifs025")
    ecm_otr = arrayise(nwp, mtr, "ecmwf_ifs025", ts_key="origin_timestamp")
    gfs_t = arrayise(nwp, mte, "gfs_seamless")
    gfs_ttr = arrayise(nwp, mtr, "gfs_seamless")

    # ECMWF bias correction REFITTED ON THIS CORPUS'S TRAIN SPLIT.
    # lambda = 0.5 mirrors the frozen artifact's temperature shrinkage.
    ab = shrunk_affine_fit(ecm_ttr, y_tr, 0.5)
    ab_full = shrunk_affine_fit(ecm_ttr, y_tr, 1.0)
    ecm_corr = shrunk_affine_apply(ecm_t, *ab) if ab else np.full(n_te, np.nan)
    ecm_corr_full = shrunk_affine_apply(ecm_t, *ab_full) if ab_full else np.full(n_te, np.nan)
    gfs_ab = shrunk_affine_fit(gfs_ttr, y_tr, 0.5)
    gfs_corr = shrunk_affine_apply(gfs_t, *gfs_ab) if gfs_ab else np.full(n_te, np.nan)

    # frozen artifact, applied unchanged -- flagged as a cross-corpus
    # application when the corpus does not match the one it was fitted on.
    def frozen_apply(obj, raw, meta):
        out = np.full(len(meta), np.nan)
        if obj is None:
            return out
        for i, m in enumerate(meta):
            if np.isfinite(raw[i]):
                v = obj.correct(raw[i], m["station_id"], h, "temperature")
                if v is not None:
                    out[i] = v
        return out

    ecm_frozen_te = frozen_apply(ecmwf_frozen, ecm_t, mte)
    gfs_frozen_te = frozen_apply(gfs_frozen, gfs_t, mte)

    # ---- classical baselines --------------------------------------------
    p_pers = o_te.copy()

    if verbose:
        log("  GBM(residual | 75 ctx) ...")
    gbm = new_gbm()
    gbm.fit(Ctr, resid_tr)
    p_gbm = o_te + gbm.predict(Cte)

    ridge_r = Ridge(alpha=1.0).fit(Ctr, resid_tr)
    p_ridge = o_te + ridge_r.predict(Cte)
    ridge_a = Ridge(alpha=1.0).fit(Ctr, y_tr)
    p_ridge_abs = ridge_a.predict(Cte)

    # diurnal-delta persistence: train mean (target - origin) by local hour of
    # the TARGET time. The classical strong temperature baseline.
    acc = defaultdict(list)
    for i, m in enumerate(mtr):
        acc[(int(m["target_timestamp"][11:13]) + 8) % 24].append(resid_tr[i])
    table = {k: float(np.mean(v)) for k, v in acc.items()}
    glob = float(np.mean(resid_tr))
    p_diurnal = np.array([
        table.get((int(m["target_timestamp"][11:13]) + 8) % 24, glob) + o_te[i]
        for i, m in enumerate(mte)])

    # ---- NWP as an INPUT feature ----------------------------------------
    if verbose:
        log("  GBM(residual | ctx + NWP) ...")
    def nwp_block(n_tgt, n_org, n_corr, origin):
        with np.errstate(invalid="ignore"):
            return np.column_stack([n_tgt - origin,
                                    n_corr - origin,
                                    n_tgt - n_org])
    # for TRAIN rows the correction must be computed from TRAIN-fitted a,b
    ecm_corr_tr = shrunk_affine_apply(ecm_ttr, *ab) if ab else np.full(len(mtr), np.nan)
    Ftr = np.hstack([Ctr, nwp_block(ecm_ttr, ecm_otr, ecm_corr_tr, o_tr)])
    Fte = np.hstack([Cte, np.nan_to_num(nwp_block(ecm_t, ecm_o, ecm_corr, o_te), nan=0.0)])
    ok = np.isfinite(Ftr).all(axis=1)
    gbm_nwp = new_gbm()
    gbm_nwp.fit(Ftr[ok], resid_tr[ok])
    p_gbm_nwp = o_te + gbm_nwp.predict(Fte)

    # ---- learn ECMWF's systematic error (downscale) ---------------------
    if verbose:
        log("  GBM(target - ECMWFcorr | ctx) ...")
    okd = np.isfinite(ecm_corr_tr)
    gbm_dn = new_gbm()
    gbm_dn.fit(Ctr[okd], (y_tr - ecm_corr_tr)[okd])
    p_dn = ecm_corr + gbm_dn.predict(Cte)

    # ---- dispersion rescale of d_temp, alpha fitted on TRAIN -------------
    var_d = float(np.var(dtemp_tr))
    alpha = float(np.cov(dtemp_tr, resid_tr, ddof=1)[0, 1] / var_d) if var_d > 1e-12 else 1.0
    p_nn_scaled = o_te + alpha * dtemp_te

    preds = {
        "persistence": p_pers,
        "diurnal_delta_persistence": p_diurnal,
        "ridge_residual_on_75ctx": p_ridge,
        "ridge_absolute_on_75ctx": p_ridge_abs,
        "gbm_residual_on_75ctx_CRITICAL_CONTROL": p_gbm,
        "candidate_head_nn": p_nn,
        "candidate_head_nn_dispersion_scaled": p_nn_scaled,
        "ecmwf_raw": ecm_t,
        "ecmwf_bias_corrected_refit_train_lambda0.5": ecm_corr,
        "ecmwf_bias_corrected_refit_train_lambda1.0": ecm_corr_full,
        "ecmwf_FROZEN_artifact_uncorrected_pairing_flagged": ecm_frozen_te,
        "gfs_raw": gfs_t,
        "gfs_bias_corrected_refit_train_lambda0.5": gfs_corr,
        "gfs_FROZEN_artifact_uncorrected_pairing_flagged": gfs_frozen_te,
        "gbm_residual_ctx_plus_nwp": p_gbm_nwp,
        "gbm_downscaling_of_ecmwf_corr": p_dn,
    }

    scored = {}
    for k, v in preds.items():
        m = np.isfinite(v)
        if m.sum() < 30:
            scored[k] = {"mae": None, "n": int(m.sum()),
                         "coverage_of_test_rows": float(m.mean())}
            continue
        e = np.abs(v[m] - y_te[m])
        bs = block_bootstrap_mae(e, blocks[m])
        scored[k] = {
            "mae": bs["mae"],
            "rmse": rmse(v[m], y_te[m]),
            "bias_pred_minus_obs": float(np.mean(v[m] - y_te[m])),
            "n": int(m.sum()),
            "coverage_of_test_rows": float(m.mean()),
            "ci95_mae": bs["ci95"],
            "vs_persistence": paired_block_test(e, np.abs(p_pers[m] - y_te[m]), blocks[m]),
            "mae_healthy_stations_only": mae(v[m & healthy], y_te[m & healthy]),
            "vs_persistence_healthy_only": paired_block_test(
                np.abs(v[m & healthy] - y_te[m & healthy]),
                np.abs(p_pers[m & healthy] - y_te[m & healthy]),
                blocks[m & healthy]),
        }

    # ---- ensembles -------------------------------------------------------
    def ens(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        return {"mae": mae((a[m] + b[m]) / 2.0, y_te[m]), "n_rows": int(m.sum())}

    ensembles = {
        "nn_plus_gbm": ens(p_nn, p_gbm),
        "nn_plus_ecmwf_corr_refit": ens(p_nn, ecm_corr),
        "gbm_plus_ecmwf_corr_refit": ens(p_gbm, ecm_corr),
        "nn_plus_gbm_nwp": ens(p_nn, p_gbm_nwp),
        "ecmwf_corr_refit_plus_persistence": ens(ecm_corr, p_pers),
        "nn_plus_persistence": ens(p_nn, p_pers),
    }

    # blend weight fitted on TRAIN (caveat: the head saw TRAIN in training)
    def fit_blend(a_tr, b_tr, tgt):
        m = np.isfinite(a_tr) & np.isfinite(b_tr)
        if m.sum() < 50:
            return None
        d = a_tr[m] - b_tr[m]
        denom = float(np.mean(d ** 2))
        if denom < 1e-9:
            return None
        return float(np.clip(-np.mean((b_tr[m] - tgt[m]) * d) / denom, 0.0, 1.0))

    fitted_blends = {}
    w = fit_blend(p_nn_tr, ecm_corr_tr, y_tr)
    if w is not None:
        m = np.isfinite(p_nn) & np.isfinite(ecm_corr)
        fitted_blends["nn_w_+_ecmwf_corr_refit"] = {
            "weight_on_nn": w,
            "mae": float(mae(w * p_nn[m] + (1 - w) * ecm_corr[m], y_te[m])),
            "n_rows": int(m.sum())}
    w2 = fit_blend(p_nn_tr, o_tr + gbm.predict(Ctr), y_tr)
    if w2 is not None:
        fitted_blends["nn_w_+_gbm"] = {
            "weight_on_nn": w2,
            "mae": float(mae(w2 * p_nn + (1 - w2) * p_gbm, y_te)),
            "n_rows": int(n_te)}

    # ---- residual diagnostics -------------------------------------------
    sd_dt = float(np.std(dtemp_te))
    sd_r = float(np.std(resid_te))
    diagnostics = {
        "sd_target_minus_origin_TEST": sd_r,
        "sd_d_temp_TEST": sd_dt,
        "sd_d_temp_TRAIN": float(np.std(dtemp_tr)),
        "dispersion_ratio_sd_dtemp_over_sd_target_change": (sd_dt / sd_r) if sd_r > 1e-12 else None,
        "corr_d_temp_with_target_minus_origin_TEST":
            float(np.corrcoef(dtemp_te, resid_te)[0, 1]) if sd_dt > 1e-9 and sd_r > 1e-9 else None,
        "mean_d_temp_TEST": float(np.mean(dtemp_te)),
        "mean_target_minus_origin_TEST": float(np.mean(resid_te)),
        "var_share_of_change_explained_by_head_r2":
            float(1.0 - np.var(resid_te - dtemp_te) / np.var(resid_te)) if sd_r > 1e-12 else None,
        "train_fitted_dispersion_alpha": alpha,
        "reading": "alpha>1 means the head IS under-dispersed (shrunk toward "
                   "persistence); rescaling d_temp by alpha is the cheapest possible "
                   "fix and its TEST effect is reported as "
                   "candidate_head_nn_dispersion_scaled.",
    }

    per_station = {}
    for st in sorted({m["station_id"] for m in mte}):
        sm = np.array([m["station_id"] == st for m in mte])
        row = {"n": int(sm.sum()),
               "persistence": mae(p_pers[sm], y_te[sm]),
               "gbm_control": mae(p_gbm[sm], y_te[sm]),
               "candidate_head": mae(p_nn[sm], y_te[sm])}
        smf = sm & np.isfinite(ecm_corr)
        row["ecmwf_bias_corrected_refit"] = mae(ecm_corr[smf], y_te[smf]) if smf.any() else None
        per_station[st] = row

    # ---- stage-2 bookkeeping --------------------------------------------
    keys = [f"{m['station_id']}|{m['origin_timestamp']}" for m in mte]
    errs = {
        "nn": {keys[i]: float(p_nn[i] - y_te[i]) for i in range(n_te)},
        "persistence": {keys[i]: float(o_te[i] - y_te[i]) for i in range(n_te)},
        "ecmwf_corr": {keys[i]: float(ecm_corr[i] - y_te[i])
                       for i in range(n_te) if np.isfinite(ecm_corr[i])},
        "gbm": {keys[i]: float(p_gbm[i] - y_te[i]) for i in range(n_te)},
    }

    if verbose:
        g = lambda k: scored[k]["mae"]
        log(f"  persistence {g('persistence'):.4f} | NN {g('candidate_head_nn'):.4f} "
            f"| GBM {g('gbm_residual_on_75ctx_CRITICAL_CONTROL'):.4f} "
            f"| ECMWFcorr {g('ecmwf_bias_corrected_refit_train_lambda0.5'):.4f} "
            f"| GBM+NWP {g('gbm_residual_ctx_plus_nwp'):.4f} "
            f"| downscale {g('gbm_downscaling_of_ecmwf_corr'):.4f}")

    return {
        "n_train_windows": int(n_tr),
        "n_test_windows": int(n_te),
        "n_test_windows_healthy_stations": int(healthy.sum()),
        "broken_station_excluded_from_sensitivity": BROKEN_STATION,
        "candidate_checkpoint": nn_meta,
        "nwp_coverage_test_rows": {
            "ecmwf_raw": float(np.isfinite(ecm_t).mean()),
            "gfs_raw": float(np.isfinite(gfs_t).mean()),
        },
        "ecmwf_refit_coefficients_train": (
            {"a": ab[0], "b": ab[1], "lambda": 0.5} if ab else None),
        "mae_degC": scored,
        "ensembles_simple_average": ensembles,
        "ensembles_blend_weight_fit_on_train": fitted_blends,
        "residual_diagnostics": diagnostics,
        "per_station_mae_degC": per_station,
    }, errs


# ---------------------------------------------------------------------------
# cross-horizon correlation
# ---------------------------------------------------------------------------

def cross_horizon(errs_by_h):
    """Pearson correlation of per-window error (pred - obs) between horizons,
    on the windows present at every horizon. High off-diagonal correlation means
    the horizons fail on the SAME windows, i.e. a shared systematic term."""
    out = {}
    for name in ("nn", "persistence", "gbm", "ecmwf_corr"):
        store = {h: errs_by_h[h][name] for h in HORIZONS}
        common = set(store[HORIZONS[0]].keys())
        for h in HORIZONS[1:]:
            common &= set(store[h].keys())
        ck = sorted(common)
        if len(ck) < 50:
            out[name] = {"n_common_windows": len(ck), "note": "too few common windows"}
            continue
        M = np.array([[store[h][k] for k in ck] for h in HORIZONS])
        C = np.corrcoef(M)
        od = [float(C[i, j]) for i in range(len(HORIZONS))
              for j in range(len(HORIZONS)) if i != j]
        out[name] = {
            "n_common_windows": len(ck),
            "matrix": [[round(float(v), 4) for v in r] for r in C],
            "mean_offdiagonal": round(float(np.mean(od)), 4),
            "min_offdiagonal": round(float(np.min(od)), 4),
        }
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    from nwp_correction import NwpCorrection

    log("loading NWP cache ...")
    nwp = load_nwp()
    ecmwf_frozen = NwpCorrection.load(os.path.join(HERE, "data", "nwp_correction_ecmwf_ifs025.json"))
    gfs_frozen = NwpCorrection.load(os.path.join(HERE, "data", "nwp_correction_gfs_seamless.json"))

    with open(os.path.join(HERE, "data", "inference_policy.json"), encoding="utf-8") as f:
        policy = json.load(f)

    # ---- read the routing decision off the policy file directly ----------
    sel_vars = ["temperature", "humidity", "pressure", "wind_speed", "wind_direction"]
    policy_cells = {}
    for hz, cfg in policy["horizons"].items():
        for v in sel_vars:
            src = cfg["selected_sources"].get(v)
            policy_cells[f"h{hz}:{v}"] = src
    n_pers = sum(1 for s in policy_cells.values() if s == "persistence_fallback")
    policy_summary = {
        "file": "data/inference_policy.json",
        "policy_version": policy.get("policy_version"),
        "dataset_hashes": policy.get("dataset_hashes"),
        "which_corpus_the_policy_was_fitted_on": (
            "weather_telemetry.csv (86ce906453aa...)"
            if policy.get("dataset_hashes", {}).get("weather_telemetry_sha256")
            == sha256(os.path.join(HERE, "data", "weather_telemetry.csv"))
            else "unmatched"),
        "temperature_source_by_horizon": {
            hz: cfg["selected_sources"].get("temperature")
            for hz, cfg in policy["horizons"].items()},
        "persistence_fallback_count_excluding_wind_direction":
            sum(1 for v in sel_vars if v != "wind_direction"
                for cfg in policy["horizons"].values()
                if cfg["selected_sources"].get(v) == "persistence_fallback"),
        "cells_excluding_wind_direction": 4 * len(policy["horizons"]),
        "persistence_fallback_count_including_wind_direction": n_pers,
        "cells_including_wind_direction": len(sel_vars) * len(policy["horizons"]),
        "reading": "the +24h temperature cell is persistence_fallback, so the "
                   "identical production/persistence MAE at +24h is a ROUTING "
                   "decision, not a measurement that the head lacks skill. "
                   "The head is scored directly in this report.",
    }

    report = {
        "experiment": "temperature_channel_achievability_and_gap_diagnosis",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "pairing_contract": {
            "why": "the checkpoint's input normalisation belongs to its training "
                   "corpus and build_forecast_windows() applies the pipeline's "
                   "norm_means/norm_stds, so a checkpoint paired with the wrong "
                   "corpus is scored out of its own input distribution",
            "excluded": "data/candidate_artifacts_leaky_olddata/ (ablation arm)",
            "pairs": [],
        },
        "inference_policy_readout": policy_summary,
        "csv_sha256": {
            "weather_telemetry_current.csv": sha256(os.path.join(HERE, "data", "weather_telemetry_current.csv")),
            "weather_telemetry.csv": sha256(os.path.join(HERE, "data", "weather_telemetry.csv")),
        },
        "frozen_artifact_provenance": {
            "nwp_correction_ecmwf_ifs025.json": {
                "created_utc": ecmwf_frozen.artifact.get("created_utc") if ecmwf_frozen else None,
                "carries_corpus_sha256": False,
                "inferred_corpus": "weather_telemetry.csv -- its test_report "
                                   "reproduces that corpus's persistence MAEs "
                                   "(h1 0.5936, h3 1.1367, h6 1.6987) exactly, "
                                   "and weather_telemetry.csv's test split does "
                                   "not overlap the current corpus's",
                "consequence": "CORRECTLY PAIRED with PAIR B. Under PAIR A it is a "
                               "corpus mismatch, so a bias correction refitted on "
                               "PAIR A's own TRAIN split is used for the headline "
                               "ECMWF number and the frozen artifact is reported "
                               "separately, flagged.",
            },
            "inference_policy.json": {
                "dataset_hashes": policy.get("dataset_hashes"),
                "consequence": "fitted on weather_telemetry.csv; its temperature "
                               "routing decisions are PAIR B decisions.",
            },
        },
        "cross_corpus_comparability_warning":
            "The two corpora are different products, not different slices of one: "
            "weather_telemetry.csv is 756156 minute-resolution rows -> 12036 "
            "station-hours; weather_telemetry_current.csv is 30470 rows -> 27798 "
            "station-hours, i.e. pre-aggregated to one value per hour. Their test "
            "periods also differ in difficulty (mean |dT| over 6h is ~2.8 degC on "
            "the current corpus vs ~2.1 degC on the original). Absolute MAE values "
            "are therefore NOT comparable between PAIR A and PAIR B; only the "
            "within-pair ordering of methods is.",
        "pairs": {},
    }

    for pair in PAIRS:
        pid = pair["id"]
        csv_path = os.path.join(HERE, "data", pair["csv"])
        csv_sha = sha256(csv_path)
        log("=" * 100)
        log(f"PAIR {pid}  corpus={pair['csv']}  ckpt={pair['ckpt_dir']}")
        pipe = TelemetryDataPipeline(weather_csv=csv_path)

        entry = {
            "role": pair["role"],
            "corpus": f"data/{pair['csv']}",
            "corpus_sha256": csv_sha,
            "checkpoint_dir": f"data/{pair['ckpt_dir']}",
            "split": {
                "strategy": "chronological 60/20/20 with 48h embargo",
                "train_start": pipe.time_range_min.isoformat(),
                "train_end": pipe.train_end.isoformat(),
                "val_start": pipe.val_start.isoformat(),
                "val_end": pipe.val_end.isoformat(),
                "test_start": pipe.test_start.isoformat(),
                "test_end": pipe.time_range_max.isoformat(),
                "n_stations": len(pipe.station_hourly),
            },
            "fit_rule": "every fitted object (GBM, Ridge, diurnal delta, ECMWF/GFS "
                        "bias correction, dispersion alpha, blend weights) is fit on "
                        "this pair's TRAIN windows and applied to its TEST windows",
            "horizons": {},
        }

        errs_by_h = {}
        for h in HORIZONS:
            log(f"  --- PAIR {pid}  +{h}h ---")
            cell, errs = run_cell(pair, h, pipe, nwp, ecmwf_frozen, gfs_frozen)
            entry["horizons"][f"h{h}"] = cell
            errs_by_h[h] = errs

        entry["stage2_cross_horizon_error_correlation"] = cross_horizon(errs_by_h)
        entry["stage2_cross_horizon_error_correlation"]["horizons"] = HORIZONS
        entry["stage2_cross_horizon_error_correlation"]["reading"] = (
            "for each predictor family, the Pearson correlation of per-window "
            "temperature error (pred - obs) between every pair of horizons, on the "
            "windows that exist at all five horizons. High off-diagonal values mean "
            "the horizons fail on the SAME windows, which points at a shared "
            "systematic term (the shared CfC trunk state) rather than at "
            "horizon-specific noise. Note the +1h target lies INSIDE the +6h/12h/24h "
            "input window, so the +1h row is partly tautological.")
        report["pairs"][pid] = entry
        del pipe

    with open(OUT_PATH, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=2, default=float)
    log(f"written: {OUT_PATH}")
    return report


if __name__ == "__main__":
    main()
