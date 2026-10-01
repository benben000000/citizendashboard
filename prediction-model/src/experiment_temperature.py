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

PAIRS = []  # resolved at runtime by discover_pairings(); see pairing_contract

HORIZONS = [1, 3, 6, 12, 24]
SEED = 42

# The "current" corpus is being rewritten under us by a parallel training run, so
# the analysis runs against a hash-verified SNAPSHOT of it rather than the live
# file. Pass --snapshot-csv to point at a different snapshot.
_DEFAULT_SNAPSHOT = os.path.join(
    os.environ.get("TEMP", os.path.expanduser("~")), "opencode", "temp_snap")


def corpus_specs():
    specs = []
    live = os.path.join(HERE, "data", "weather_telemetry_current.csv")
    if "--snapshot-csv" in sys.argv:
        snap = sys.argv[sys.argv.index("--snapshot-csv") + 1]
        specs.append({"path": snap, "label": "weather_telemetry_current_snapshot"})
    else:
        import glob as _g
        snaps = sorted(_g.glob(os.path.join(_DEFAULT_SNAPSHOT,
                                           "weather_telemetry_snapshot_*.csv")))
        if snaps:
            specs.append({"path": snaps[-1], "label": "weather_telemetry_current_snapshot"})
        elif os.path.exists(live):
            specs.append({"path": live, "label": "weather_telemetry_current_live"})
    specs.append({"path": os.path.join(HERE, "data", "weather_telemetry.csv"),
                  "label": "weather_telemetry_original"})
    return specs

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
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def ckpt_manifest_sha(ckpt_dir, horizon):
    import torch
    p = os.path.join(HERE, "data", ckpt_dir, f"candidate_h{horizon}h.pt")
    if not os.path.exists(p):
        return None
    ck = torch.load(p, map_location="cpu", weights_only=False)
    return (ck.get("manifest", {}) or {}).get("weather_telemetry_sha256")


EXCLUDED_CKPT_DIRS = {"candidate_artifacts_leaky_olddata"}  # ablation arm, per brief


def discover_pairings(corpus_specs):
    """Pair a corpus with the checkpoint set trained on THAT EXACT corpus.

    Hard-coding a checkpoint directory name is unsafe here: weather_telemetry_
    current.csv was replaced (sha fc8f7b4 -> 9389121) while this analysis was
    running, so the directory named 'candidate_artifacts_windfix' is no longer
    paired with the corpus it was trained on and no file on disk matches its
    manifest hash. Pairing is therefore resolved by CONTENT HASH, and any
    candidate that has no correctly-paired corpus is reported as
    unreproducible rather than silently mis-scored."""
    import glob
    ckpt_dirs = sorted(
        os.path.basename(d) for d in glob.glob(os.path.join(HERE, "data", "candidate_artifacts*"))
        if os.path.isdir(d))
    available = {}
    for d in ckpt_dirs:
        per_h = {h: ckpt_manifest_sha(d, h) for h in HORIZONS}
        available[d] = per_h

    pairs, notes = [], []
    for spec in corpus_specs:
        path, label = spec["path"], spec["label"]
        if not os.path.exists(path):
            notes.append({"corpus": label, "status": "corpus file absent"})
            continue
        csha = sha256(path)
        best = None
        for d, per_h in available.items():
            if d in EXCLUDED_CKPT_DIRS:
                continue
            usable = [h for h in HORIZONS if per_h.get(h) == csha]
            if len(usable) == len(HORIZONS):
                best = d
                break
            if usable:
                notes.append({"corpus": label, "corpus_sha256": csha,
                              "partial_candidate": d,
                              "status": f"only horizons {usable} match this corpus"})
        if best:
            pairs.append({
                "id": f"{label}__with__{best}",
                "role": "PRIMARY" if label == corpus_specs[0]["label"] else "SECONDARY",
                "csv_path": path, "csv_label": label, "corpus_sha256": csha,
                "ckpt_dir": best,
                "pairing": "CORRECT: checkpoint manifest sha256 == corpus sha256",
            })
        else:
            pairs.append({
                "id": f"{label}__NO_MATCHED_CHECKPOINT",
                "role": "PRIMARY" if label == corpus_specs[0]["label"] else "SECONDARY",
                "csv_path": path, "csv_label": label, "corpus_sha256": csha,
                "ckpt_dir": None,
                "pairing": "UNPAIRED: no checkpoint set was trained on this corpus",
            })

    for d, per_h in available.items():
        shas = sorted({s for s in per_h.values() if s})
        on_disk = [c["path"] for c in corpus_specs
                   if os.path.exists(c["path"]) and sha256(c["path"]) in shas]
        notes.append({
            "checkpoint_dir": d,
            "excluded_by_brief": d in EXCLUDED_CKPT_DIRS,
            "manifest_corpus_sha256": shas,
            "reproducible": bool(on_disk),
            "note": ("trained on a corpus version that no longer exists on disk; "
                     "its own training corpus has been overwritten, so this "
                     "checkpoint CANNOT be validated on its own data"
                     if not on_disk and d not in EXCLUDED_CKPT_DIRS else
                     ("excluded ablation arm" if d in EXCLUDED_CKPT_DIRS else "")),
        })
    return pairs, notes


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


def inspect_temp_head(ckpt_dir):
    """Directly inspect the temperature head's last layer.

    A head that has collapsed onto persistence looks like this: the output layer
    stays at (or near) its zero initialisation, so d_temp has near-zero spread
    whatever the trunk computes. That distinguishes 'the head is shrunk and a
    rescale would recover it' from 'the head learned nothing', which is a
    different fix."""
    import torch
    from model import GarciaWeatherLNNFeatured
    out = {}
    for h in HORIZONS:
        ck = torch.load(os.path.join(HERE, "data", ckpt_dir, f"candidate_h{h}h.pt"),
                        map_location="cpu", weights_only=False)
        man = ck.get("manifest", {}) or {}
        mdl = GarciaWeatherLNNFeatured(
            input_dim=int(man.get("input_dim", 8)),
            context_dim=int(man.get("context_dim", 75)),
            hidden_dim=int(man.get("hidden_dim", 32)),
            use_two_stage_precipitation=True)
        mdl.load_state_dict(ck["model_state_dict"])
        last = mdl.temp_head[-1]
        out[f"h{h}"] = {
            "temp_head_last_layer_weight_l2": float(last.weight.norm()),
            "temp_head_last_layer_bias": [round(float(b), 6) for b in last.bias],
            "temp_head_last_layer_weight_max_abs": float(last.weight.abs().max()),
            "temp_head_last_layer_is_zero_init": bool(last.weight.abs().max() == 0
                                                      and last.bias.abs().max() == 0),
            "trunk_param_count": int(sum(p.numel() for p in mdl.parameters())),
        }
    return out


def lag_profile(csv_path):
    """Mean |dT| at fixed hour lags on a continuous hourly grid.

    Evidence for the +24h argument: if the diurnal cycle is clean, a 24-hour lag
    returns to the same diurnal phase, so |dT| at 24h FALLS below the 12h value.
    That is why persistence is an unusually strong baseline at +24h and why a
    model that has to beat it there is competing with a phase-matched baseline."""
    import csv as _csv
    per_hour = defaultdict(list)
    with open(csv_path, encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            ts = row.get("recorded_at", "")
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                continue
            try:
                t = float(row["temperature"])
            except (TypeError, ValueError, KeyError):
                continue
            per_hour[(row.get("station_id"), dt.replace(minute=0, second=0, microsecond=0))].append(t)
    stations = sorted({st for st, _ in per_hour})
    out = {}
    for st in stations:
        hrs = sorted(hb for (s2, hb) in per_hour if s2 == st)
        T = {hb: per_hour[(st, hb)][-1] for hb in hrs}
        row = {"n_hours": len(hrs)}
        for lag in (1, 3, 6, 12, 24):
            from datetime import timedelta
            d = [abs(T[h + timedelta(hours=lag)] - T[h]) for h in hrs
                 if (h + timedelta(hours=lag)) in T]
            row[f"mean_abs_dT_lag{lag}h"] = round(float(np.mean(d)), 4) if d else None
        out[st] = row
    healthy = {k: v for k, v in out.items() if k != BROKEN_STATION}
    summary = {}
    for lag in (1, 3, 6, 12, 24):
        vals = [v[f"mean_abs_dT_lag{lag}h"] for v in healthy.values()
                if v[f"mean_abs_dT_lag{lag}h"] is not None]
        summary[f"mean_abs_dT_lag{lag}h"] = round(float(np.mean(vals)), 4)
    return {"per_station": out, "healthy_station_mean": summary,
            "reading": "mean |dT| rises from lag 1 to lag 12 and then DROPS at "
                       "lag 24, i.e. the series has a clean diurnal cycle and a "
                       "24-hour lag lands on the same diurnal phase. Persistence at "
                       "+24h is therefore a phase-matched baseline and is "
                       "correspondingly hard to beat."}


PAIRWISE_TESTS = [
    ("gbm_downscaling_of_ecmwf_corr", "ecmwf_bias_corrected_refit_train_lambda0.5"),
    ("gbm_downscaling_of_ecmwf_corr", "ecmwf_FROZEN_artifact_uncorrected_pairing_flagged"),
    ("gbm_downscaling_of_ecmwf_corr", "candidate_head_nn"),
    ("gbm_residual_ctx_plus_nwp", "candidate_head_nn"),
    ("gbm_residual_ctx_plus_nwp", "gbm_residual_on_75ctx_CRITICAL_CONTROL"),
    ("gbm_residual_on_75ctx_CRITICAL_CONTROL", "candidate_head_nn"),
    ("gbm_residual_on_75ctx_CRITICAL_CONTROL", "ecmwf_bias_corrected_refit_train_lambda0.5"),
    ("ridge_residual_on_75ctx", "candidate_head_nn"),
]


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

    # ---- neural head (skipped when this corpus has no correctly-paired
    # checkpoint; the checkpoint-free controls below still run) -------------
    have_nn = pair.get("ckpt_dir") is not None
    if have_nn:
        p_nn, nn_meta = score_candidate_head(pair["ckpt_dir"], h, te, mte)
        dtemp_te = p_nn - o_te
        p_nn_tr, _ = score_candidate_head(pair["ckpt_dir"], h, tr, mtr)
        dtemp_tr = p_nn_tr - o_tr
    else:
        p_nn = dtemp_te = np.full(n_te, np.nan)
        p_nn_tr = dtemp_tr = np.full(n_tr, np.nan)
        nn_meta = {"checkpoint": None,
                   "note": "no checkpoint was trained on this corpus; the neural "
                           "head is not scored here because scoring it would "
                           "require feeding windows normalised with statistics "
                           "from a corpus it was not trained on"}

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
    if have_nn:
        preds["candidate_head_nn"] = p_nn
        preds["candidate_head_nn_dispersion_scaled"] = p_nn_scaled

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

    # ---- direct head-to-head tests (the MAE table alone cannot rank methods
    # whose gap is inside the noise) -------------------------------------
    pairwise = {}
    for a, b in PAIRWISE_TESTS:
        if a not in preds or b not in preds:
            continue
        m = np.isfinite(preds[a]) & np.isfinite(preds[b])
        if m.sum() < 30:
            continue
        pairwise[f"{a}  MINUS  {b}"] = {
            **paired_block_test(np.abs(preds[a][m] - y_te[m]),
                                np.abs(preds[b][m] - y_te[m]), blocks[m]),
            "mae_a": mae(preds[a][m], y_te[m]),
            "mae_b": mae(preds[b][m], y_te[m]),
            "n": int(m.sum()),
        }

    # ---- ensembles -------------------------------------------------------
    def ens(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        return {"mae": mae((a[m] + b[m]) / 2.0, y_te[m]), "n_rows": int(m.sum())}

    ensembles = {
        "gbm_plus_ecmwf_corr_refit": ens(p_gbm, ecm_corr),
        "ecmwf_corr_refit_plus_persistence": ens(ecm_corr, p_pers),
    }
    if have_nn:
        ensembles.update({
            "nn_plus_gbm": ens(p_nn, p_gbm),
            "nn_plus_ecmwf_corr_refit": ens(p_nn, ecm_corr),
            "nn_plus_gbm_nwp": ens(p_nn, p_gbm_nwp),
            "nn_plus_persistence": ens(p_nn, p_pers),
        })

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
    if have_nn:
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
    if not have_nn:
        diagnostics["note"] = "no correctly-paired checkpoint; head diagnostics are null"

    per_station = {}
    for st in sorted({m["station_id"] for m in mte}):
        sm = np.array([m["station_id"] == st for m in mte])
        row = {"n": int(sm.sum()),
               "persistence": mae(p_pers[sm], y_te[sm]),
               "gbm_control": mae(p_gbm[sm], y_te[sm])}
        if have_nn:
            row["candidate_head"] = mae(p_nn[sm], y_te[sm])
        smf = sm & np.isfinite(ecm_corr)
        row["ecmwf_bias_corrected_refit"] = mae(ecm_corr[smf], y_te[smf]) if smf.any() else None
        per_station[st] = row

    # ---- stage-2 bookkeeping --------------------------------------------
    keys = [f"{m['station_id']}|{m['origin_timestamp']}" for m in mte]
    errs = {
        "persistence": {keys[i]: float(o_te[i] - y_te[i]) for i in range(n_te)},
        "gbm": {keys[i]: float(p_gbm[i] - y_te[i]) for i in range(n_te)},
        "ecmwf_corr": {keys[i]: float(ecm_corr[i] - y_te[i])
                       for i in range(n_te) if np.isfinite(ecm_corr[i])},
    }
    if have_nn:
        errs["nn"] = {keys[i]: float(p_nn[i] - y_te[i]) for i in range(n_te)}

    if verbose:
        def gg(k):
            return scored[k]["mae"] if scored.get(k, {}).get("mae") is not None else float("nan")
        log(f"  persistence {gg('persistence'):.4f} | NN {gg('candidate_head_nn'):.4f} "
            f"| GBM {gg('gbm_residual_on_75ctx_CRITICAL_CONTROL'):.4f} "
            f"| ECMWFcorr {gg('ecmwf_bias_corrected_refit_train_lambda0.5'):.4f} "
            f"| GBM+NWP {gg('gbm_residual_ctx_plus_nwp'):.4f} "
            f"| downscale {gg('gbm_downscaling_of_ecmwf_corr'):.4f}")

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
        "pairwise_head_to_head": pairwise,
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
        if any(name not in errs_by_h[h] for h in HORIZONS):
            continue
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

def _sig_label(delta, ptest, lower_is_better_first=True):
    """Direction-aware significance label.

    An earlier version printed 'no significant difference' whenever the FIRST
    method did not win, which mislabelled p<0.001 losses as 'no significant
    difference'. Significance and direction are separate questions."""
    if not ptest:
        return "not tested"
    if not ptest["significant_at_5pct"]:
        return "no significant difference (noise)"
    return ("first method significantly better" if ptest["delta_mae"] < 0
            else "second method significantly better")


def head_diagnosis(rd, scored):
    """Three-state reading of the head's residual behaviour."""
    ratio = rd["dispersion_ratio_sd_dtemp_over_sd_target_change"]
    corr = rd["corr_d_temp_with_target_minus_origin_TEST"]
    alpha = rd["train_fitted_dispersion_alpha"]
    base = scored.get("persistence", {}).get("mae")
    plain = scored.get("candidate_head_nn", {}).get("mae")
    scaled = scored.get("candidate_head_nn_dispersion_scaled", {}).get("mae")
    rescale_helped = bool(plain is not None and scaled is not None and scaled < plain)
    if ratio is not None and ratio < 0.15:
        state = "NEAR-CONSTANT: d_temp carries almost no variance, so the head is "
        state += "effectively persistence and no rescale can recover it"
    elif corr is not None and abs(corr) < 0.25:
        state = "NEAR-UNINFORMATIVE: d_temp is not correlated with the true change, "
        state += "so it is worse than useless rather than merely shrunk"
    elif ratio is not None and ratio < 0.85:
        state = "UNDER-DISPERSED (shrunk toward persistence) but NOT the binding "
        state += "constraint: the train-fitted optimal rescale is alpha="
        state += f"{alpha:.3f} and applying it changes TEST MAE by "
        state += ("an improvement" if rescale_helped else "nothing or worse")
    else:
        state = "NOT under-dispersed (ratio >= 0.85)"
    if plain is not None and base is not None and plain > base:
        state += (f". NOTE the head is WORSE than persistence here "
                  f"({plain:.4f} vs {base:.4f}), so it is actively harmful, not "
                  f"merely uninformative")
    return {"state": state, "dispersion_ratio": ratio,
            "corr_d_temp_with_true_change": corr,
            "train_fitted_alpha": alpha,
            "rescaling_d_temp_improved_test_mae": rescale_helped}


def verdict_for(pair_id, E):
    """Per-horizon verdict, derived from the measured numbers rather than asserted."""
    out = {}
    for h in HORIZONS:
        c = E["horizons"][f"h{h}"]
        m = c["mae_degC"]
        pw = c["pairwise_head_to_head"]

        def g(k):
            return m[k]["mae"] if m.get(k) and m[k].get("mae") is not None else None

        def beats(k):
            """does k beat persistence, significantly?"""
            v = m.get(k)
            return bool(v and v.get("vs_persistence", {}).get("significant_at_5pct")
                        and v["mae"] < m["persistence"]["mae"])

        pers, nn = g("persistence"), g("candidate_head_nn")
        gbm = g("gbm_residual_on_75ctx_CRITICAL_CONTROL")
        ecm = g("ecmwf_bias_corrected_refit_train_lambda0.5")
        dn = g("gbm_downscaling_of_ecmwf_corr")
        rd = c["residual_diagnostics"]
        hd = head_diagnosis(rd, m)

        dn_vs_ecm = pw.get("gbm_downscaling_of_ecmwf_corr  MINUS  "
                           "ecmwf_bias_corrected_refit_train_lambda0.5")
        dn_vs_nn = pw.get("gbm_downscaling_of_ecmwf_corr  MINUS  candidate_head_nn")
        gbm_vs_nn = pw.get("gbm_residual_on_75ctx_CRITICAL_CONTROL  MINUS  candidate_head_nn")

        lines = []
        lines.append(
            f"+{h}h  persistence={pers:.3f}  candidate_head={nn:.3f}  "
            f"GBM_control={gbm:.3f}  ecmwf_bias_corrected={ecm:.3f}  "
            f"ML_downscale_of_ecmwf={dn:.3f}")
        lines.append(
            f"    beyond persistence?  head={'YES' if beats('candidate_head_nn') else 'no'}"
            f"   GBM={'YES' if beats('gbm_residual_on_75ctx_CRITICAL_CONTROL') else 'no'}"
            f"   -> {'forecastable beyond persistence from station-local data' if beats('gbm_residual_on_75ctx_CRITICAL_CONTROL') else 'NOT forecastable beyond persistence from station-local data'}")
        if gbm_vs_nn:
            lines.append(
                f"    head vs GBM control: delta={gbm_vs_nn['delta_mae']:+.4f} "
                f"p={gbm_vs_nn['p_two_sided']:.3f} -> {_sig_label(gbm_vs_nn['delta_mae'], gbm_vs_nn)}"
                + ("; the GBM control is the better temperature model here"
                   if (gbm_vs_nn["significant_at_5pct"] and gbm_vs_nn["delta_mae"] < 0) else ""))
        lines.append(
            (f"    beyond ECMWF(bias-corrected)?  "
             f"{'YES' if (dn_vs_ecm and dn_vs_ecm['significant_at_5pct'] and dn_vs_ecm['delta_mae'] < 0) else 'no'}"
             f"  (ML downscale delta={dn_vs_ecm['delta_mae']:+.4f}, p={dn_vs_ecm['p_two_sided']:.3f})")
            if dn_vs_ecm else "    beyond ECMWF? downscale test unavailable")
        if dn_vs_nn:
            lines.append(
                f"    ML downscale vs the neural head: delta={dn_vs_nn['delta_mae']:+.4f} "
                f"p={dn_vs_nn['p_two_sided']:.3f} ({_sig_label(dn_vs_nn['delta_mae'], dn_vs_nn)})")
        lines.append(
            f"    head residual: sd(d_temp)={rd['sd_d_temp_TEST']:.4f} vs "
            f"sd(target-origin)={rd['sd_target_minus_origin_TEST']:.4f} "
            f"(ratio {rd['dispersion_ratio_sd_dtemp_over_sd_target_change']:.3f}, "
            f"corr {rd['corr_d_temp_with_target_minus_origin_TEST']:+.3f}, "
            f"train-fitted alpha={rd['train_fitted_dispersion_alpha']:.3f})")
        lines.append(f"    head diagnosis: {hd['state']}")

        out[f"h{h}"] = {
            "mae_summary": {"persistence": pers, "candidate_head": nn,
                            "gbm_control": gbm, "ecmwf_bias_corrected": ecm,
                            "ml_downscale_of_ecmwf": dn},
            "forecastable_beyond_persistence_from_station_local_data":
                beats("gbm_residual_on_75ctx_CRITICAL_CONTROL"),
            "candidate_head_beats_persistence": beats("candidate_head_nn"),
            "gbm_control_beats_candidate_head": bool(
                gbm_vs_nn and gbm_vs_nn["significant_at_5pct"] and gbm_vs_nn["delta_mae"] < 0),
            "ml_downscale_beats_ecmwf_bias_corrected": bool(
                dn_vs_ecm and dn_vs_ecm["significant_at_5pct"] and dn_vs_ecm["delta_mae"] < 0),
            "ml_downscale_beats_candidate_head": bool(
                dn_vs_nn and dn_vs_nn["significant_at_5pct"] and dn_vs_nn["delta_mae"] < 0),
            "head_diagnosis": hd,
            "realistic_ceiling_degC": min(x for x in (nn, gbm, ecm, dn) if x is not None),
            "narrative": lines,
        }
    return out


def write_report(report):
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=2, default=float)


def headline(report):
    """Compact answer table, computed from the measured numbers."""
    out = {}
    for pid, E in report["pairs"].items():
        rows = {}
        for h in HORIZONS:
            v = E.get("stage3_verdict", {}).get(f"h{h}")
            if v:
                rows[f"h{h}"] = v["mae_summary"]
        out[pid] = {
            "corpus": E["corpus"], "corpus_sha256": E["corpus_sha256"],
            "checkpoint_dir": E["checkpoint_dir"], "pairing_status": E["pairing_status"],
            "mae_by_horizon": rows,
        }
    return out


# Authored after reading the measured numbers. Kept here rather than written by
# hand into the JSON so that re-running the script reproduces it exactly.
CONCLUSIONS = {
    "split_contract": (
        "Chronological 60/20/20, 48h embargo, from dataset.TelemetryDataPipeline. "
        "n_test = 3235-3593 windows per horizon on the primary pair, 1984-2273 on the "
        "secondary. EVERY fitted object -- GBM, Ridge, diurnal delta, the ECMWF/GFS "
        "bias correction, the dispersion scale, the blend weights -- is fit on that "
        "pair's TRAIN windows and applied unchanged to TEST. Nothing is selected on "
        "TEST."),
    "statistical_caveat": (
        "n is large (~3k windows) but consecutive windows share 23 of 24 input "
        "hours and cluster into ~90 (station, day) blocks per horizon, so the "
        "effective sample size is the block count. All intervals and p-values are "
        "(station, day) BLOCK BOOTSTRAPs (1500 resamples). Treat any delta whose CI "
        "straddles zero as noise: on the primary pair that band is roughly +/-0.06 "
        "degC, and the results explicitly labelled noise below are inside it."),
    "headline_1_gbm_control_sets_the_ceiling": (
        "The sklearn GradientBoostingRegressor on the same 75 zero-leakage context "
        "features is the realistic ceiling for station-local information. On the "
        "primary pair it reaches 0.581 / 1.211 / 1.413 / 1.434 degC at +1/+3/+6/+12h "
        "against persistence 0.868 / 2.282 / 4.029 / 5.559, i.e. it beats persistence "
        "by 33% / 47% / 65% / 74%, all p<0.001. Temperature IS strongly forecastable "
        "beyond persistence from station-local data out to +12h and the neural head "
        "is not extracting what is there. At +24h the control is 82% WORSE than "
        "persistence (2.040 vs 1.118, p<0.001): +24h is genuinely not forecastable "
        "from these features."),
    "headline_2_the_24h_baseline_is_diurnally_phase_matched": (
        "Persistence looks anomalously strong at +24h because a 24-hour lag lands on "
        "the SAME diurnal phase as the origin. Measured mean |dT| over healthy "
        "stations rises 0.75 -> 1.77 -> 2.91 -> 3.95 degC at lags 1/3/6/12h and then "
        "FALLS to 1.79 degC at 24h (primary; 0.78 -> 1.57 -> 2.48 -> 3.27 -> 2.17 on "
        "the secondary). Persistence at +24h is a phase-matched baseline, which is why "
        "the GBM, Ridge and diurnal-delta baselines all fail to beat it there. This "
        "is physics, not a modelling defect."),
    "headline_3_the_neural_head_is_strictly_dominated": (
        "The GBM control beats the candidate head at +6h and +12h in BOTH correctly "
        "paired configurations, with block-bootstrap CIs excluding zero: primary "
        "1.413 vs 2.175 (d=-0.761) and 1.434 vs 2.777 (d=-1.343); secondary 1.315 vs "
        "1.493 (d=-0.178, p=0.003) and 1.653 vs 1.835 (d=-0.182, p=0.027). A plain "
        "75-feature Ridge also beats the 17,529-parameter CfC/LNN trunk+head at both "
        "horizons on both pairs. Ensembling rescues the head (50/50 mean with the GBM "
        "gives 1.736 and 2.023 on the primary, versus 2.175 and 2.777 for the head "
        "alone) but never beats the better component, so an ensemble is not a route to "
        "keeping the head."),
    "headline_4_the_trunk_is_not_the_bottleneck": (
        "Mean off-diagonal correlation of the head's temperature error between "
        "horizons is 0.100 (primary) and 0.227 (secondary), versus 0.367 / 0.412 for "
        "persistence and 0.286 / 0.442 for bias-corrected ECMWF. Those are LOW. If the "
        "shared CfC trunk state were the binding constraint the horizons would fail on "
        "the same windows and this would approach the persistence figure. It does not. "
        "The head is simply weaker per-horizon than a gradient-boosted tree on the "
        "identical feature vector, so the fix is the head/feature representation, not "
        "the trunk. (Caveat: the +1h target lies inside the +6/+12/+24h input window, "
        "so the +1h row of each matrix is partly tautological; the +6/+12/+24h "
        "sub-blocks are the informative ones and show the same low correlation.)"),
    "headline_5_underdispersion_is_real_but_not_the_binding_constraint": (
        "Marginal dispersion ratio sd(d_temp)/sd(target-origin) is 0.67 / 0.74 / 0.73 "
        "/ 0.63 at +1/+3/+6/+12h on the primary pair, so the head IS shrunk toward "
        "persistence. But the train-fitted optimal rescale alpha is 0.93-0.99, i.e. "
        "about 1, and rescaling d_temp by it leaves TEST MAE equal or worse at every "
        "horizon. Under-dispersion is not what is costing the head its skill. At +24h "
        "the picture is decisive: ratio 0.393 (primary) / 0.054 (secondary), "
        "corr(d_temp, target-origin) 0.200 / 0.361, R^2 0.002 / 0.036, optimal alpha "
        "2.0 / 9.8, and rescaling by that alpha makes +24h much worse (1.944 vs "
        "1.357). At +24h the head is not shrunk, it is close to uninformative."),
    "headline_6_yes_we_lose_to_ecmwf_and_routing_is_the_right_answer": (
        "On the secondary (original) corpus at +6h and +12h: candidate head 1.493 / "
        "1.835, GBM control 1.315 / 1.653, ECMWF raw 1.274 / 1.288, ECMWF "
        "bias-corrected 1.044 / 1.049. Bias-corrected ECMWF beats BOTH the neural head "
        "AND the GBM control at both horizons. Note also that the brief's ECMWF figures "
        "(1.274 / 1.288) are RAW ECMWF and reproduce exactly here; after the shipped "
        "correction the same field is 1.044 / 1.049, so the gap the head actually has "
        "to close at +12h is 0.79 degC, not 0.55 degC. Conclusion: at +6h and +12h "
        "ECMWF is simply better from this data and the correct response is to ROUTE "
        "temperature through ECMWF via the router, not to keep trying to out-train it."),
    "headline_7_the_one_route_that_beats_ecmwf": (
        "Fitting a GBM on (target - ECMWF_bias_corrected) over the 75 context "
        "features, on TRAIN, beats the pooled train-refit bias-corrected ECMWF at "
        "EVERY horizon on the primary pair, all p<0.001: +1h 0.962 vs 1.690, +3h 1.181 "
        "vs 1.684, +6h 1.194 vs 1.675, +12h 1.210 vs 1.672, +24h 1.340 vs 1.688. That "
        "is the single most promising route found and it needs no neural model. It "
        "also beats the frozen per-station artifact on the primary pair, but that "
        "artifact is a cross-corpus application there so the comparison is not clean. "
        "Against a correctly-paired PER-STATION affine correction on the secondary "
        "pair the gain shrinks to a statistical tie at +6h (+2.1%, p=0.53) and -1.9% "
        "/ -4.3% at +12h / +24h (p=0.66 / 0.24). Honest reading: downscaling robustly "
        "beats a POOLED affine correction by 28-29%, and beats a good PER-STATION "
        "affine correction by a small margin that is not consistently significant. The "
        "next thing to test is fitting the per-station affine correction and the "
        "learned residual JOINTLY."),
    "headline_8_nwp_as_a_feature_beats_nwp_as_an_anchor_marginally": (
        "Adding NWP temperature as three extra INPUT features to the persistence-"
        "residual GBM is the best single method at +1h (0.508), +3h (1.028) and +6h "
        "(1.186) on the primary pair and at every horizon on the secondary pair. But "
        "the anchor formulation (predict target - ECMWF_corrected) is equal or better "
        "at +6h/+12h on the primary pair and at +12h/+24h on the secondary. Use NWP as "
        "the BASE of the residual, not as an extra feature alongside a persistence "
        "base."),
    "hypothesis_checked_colleagues_3_seed_claim": (
        "Hypothesis: the retrained candidate beats persistence on temperature at "
        "h1/h3/h6/h12 but NOT h24. CONFIRMED IN DIRECTION on the primary pair at one "
        "seed: head 0.670 vs 0.868 (+1h), 1.449 vs 2.282 (+3h), 2.175 vs 4.029 (+6h), "
        "2.777 vs 5.559 (+12h), all p<0.001. CAVEAT 1: n=1 seed (42) per checkpoint "
        "set, so seed variance is NOT measured; the colleague's 3-seed result is "
        "neither reproduced nor contradicted on that axis. CAVEAT 2: at +24h the head "
        "is not merely 'no better than persistence', it is 21.4% WORSE (1.357 vs "
        "1.118, p<0.001)."),
    "correction_to_brief_correction_2": (
        "The brief states +24h is a pure routing decision because a headroom "
        "diagnostic shows +6.80% skill there. That is only half right and the half "
        "that is wrong matters. Measured directly: on the SECONDARY pair the head does "
        "beat persistence at +24h, but by +3.0% (1.364 vs 1.406, p<0.001), not "
        "6.80%; on the PRIMARY pair it is 21.4% WORSE. So re-routing +24h to the "
        "learned model would be a small win on the original corpus and a significant "
        "REGRESSION on the current one. The policy's persistence_fallback at +24h is "
        "defensible and should stay until the policy is refit on a corpus the "
        "candidate was actually trained on. The provenance of the +6.80% figure could "
        "not be reproduced from either pairing."),
    "correction_to_brief_persistence_count": (
        "Read directly from data/inference_policy.json: temperature is "
        "persistence_fallback at +1h, +3h and +24h and learned_model at +6h and "
        "+12h. Across the four producer-backed variables (temperature, humidity, "
        "pressure, wind_speed) that is 15 of 20 persistence cells; 20 of 25 if "
        "wind_direction is counted. The brief's corrected count of 15/20 is confirmed."),
    "biggest_actionable_finding": (
        "The shipped inference_policy.json is fitted on weather_telemetry.csv (sha "
        "86ce9064) and is therefore stale for the current corpus in exactly the way "
        "the checkpoints were. It routes temperature to persistence_fallback at +1h and "
        "+3h. Measured on the current corpus that discards 33% and 47% (GBM control) "
        "and 23% and 36% (candidate head) of available MAE at those two horizons, all "
        "p<0.001. Refitting the policy against the current corpus is a larger and much "
        "cheaper win than any further temperature architecture work."),
    "corpus_mutation_incident_consequence": (
        "candidate_artifacts_windfix, the promotion candidate named in the brief, was "
        "trained on weather_telemetry_current.csv at sha fc8f7b4. That file was "
        "overwritten with sha 9389121 DURING this analysis (mtime 2026-09-30 19:12 "
        "local), and no copy of fc8f7b4 exists on disk, so that checkpoint can no "
        "longer be validated on its own training corpus at all. It was superseded by "
        "candidate_artifacts_fresh (trained on 9389121), which is what the PRIMARY "
        "pair measures. Recommendation: pin each candidate's corpus by hash at "
        "training time, and refuse to overwrite a corpus that a live checkpoint "
        "depends on."),
    "dataset_quality_note": (
        "Station VEpdDpBK is unreliable in BOTH corpora (lag-1 |dT| 1.8-3.8 degC "
        "against 0.6-0.8 for every other station, and a lag-24 |dT| larger than its "
        "lag-12, i.e. no usable diurnal structure). Every table reports a "
        "healthy-stations-only sensitivity column; excluding it moves no headline "
        "conclusion."),
}


def finalize_only():
    """Attach CONCLUSIONS to an existing report without refitting anything.

    run_cell's output is fully serialised in the JSON, so stage3_verdict can be
    recomputed from the stored numbers -- no model is refitted and no window is
    rebuilt."""
    with open(OUT_PATH, "r", encoding="utf-8") as f:
        report = json.load(f)
    for pid, E in report["pairs"].items():
        if all(f"h{h}" in E.get("horizons", {}) for h in HORIZONS):
            E["stage3_verdict"] = verdict_for(pid, E)
    report["conclusions"] = CONCLUSIONS
    report["progress"] = {"complete": True,
                          "note": "finalize_only; all horizon cells were computed "
                                  "by a full run of main()"}
    report["headline_findings"] = headline(report)
    write_report(report)
    return report


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
            "how_resolved": "by CONTENT HASH, not by directory name: a checkpoint "
                            "set is used with a corpus only when every horizon's "
                            "manifest weather_telemetry_sha256 equals that corpus's "
                            "sha256",
            "excluded_by_brief": sorted(EXCLUDED_CKPT_DIRS),
            "corpus_mutation_incident": (
                "weather_telemetry_current.csv was REWRITTEN while this analysis "
                "was running: sha256 went fc8f7b4edc11... -> 938912148be8... "
                "(mtime 2026-09-30 19:12 local). candidate_artifacts_windfix was "
                "trained on the fc8f7b4 version, and that version no longer exists "
                "on disk, so the promotion candidate named in the brief cannot be "
                "validated on its own training corpus any more. The analysis "
                "therefore runs against a hash-verified SNAPSHOT of the live file "
                "so the numbers are internally consistent, and the pairings that "
                "were actually used are listed under 'pairs_used'."),
            "pairs_used": [],
            "pairing_notes": [],
        },
        "inference_policy_readout": policy_summary,
        "csv_sha256_at_analysis_time": {},
        "frozen_artifact_provenance": {
            "nwp_correction_ecmwf_ifs025.json": {
                "created_utc": ecmwf_frozen.artifact.get("created_utc") if ecmwf_frozen else None,
                "carries_corpus_sha256": False,
                "inferred_corpus": "weather_telemetry.csv -- its test_report "
                                   "reproduces that corpus's persistence MAEs "
                                   "(h1 0.5936, h3 1.1367, h6 1.6987) exactly",
                "consequence": "CORRECTLY PAIRED with the original-corpus pair. On "
                               "any other corpus it is a mismatch, so a bias "
                               "correction refitted on that pair's own TRAIN split "
                               "is the headline ECMWF number and the frozen "
                               "artifact is reported separately, flagged.",
            },
            "inference_policy.json": {
                "dataset_hashes": policy.get("dataset_hashes"),
                "consequence": "fitted on weather_telemetry.csv; its temperature "
                               "routing decisions are decisions for THAT corpus.",
            },
        },
        "cross_corpus_comparability_warning":
            "The corpora are different products, not different slices of one: "
            "weather_telemetry.csv is ~756k minute-resolution rows -> ~12k "
            "station-hours; weather_telemetry_current.csv is ~30k rows -> ~28k "
            "station-hours, i.e. already aggregated to one value per hour. Their "
            "test periods differ in difficulty too (see "
            "pairs[...].stage2_lag_profile_evidence.healthy_station_mean). Absolute "
            "MAE values are therefore NOT comparable between the pairs; only the "
            "within-pair ordering of methods is.",
        "pairs": {},
    }

    specs = corpus_specs()
    for s in specs:
        if os.path.exists(s["path"]):
            report["csv_sha256_at_analysis_time"][s["label"]] = sha256(s["path"])
    pairs, notes = discover_pairings(specs)
    report["pairing_contract"]["pairs_used"] = [
        {"id": p["id"], "role": p["role"], "corpus": p["csv_label"],
         "corpus_sha256": p["corpus_sha256"], "checkpoint_dir": p["ckpt_dir"],
         "pairing": p["pairing"]} for p in pairs]
    report["pairing_contract"]["pairing_notes"] = notes

    for pair in pairs:
        pid = pair["id"]
        csv_path = pair["csv_path"]
        log("=" * 100)
        log(f"PAIR {pid}  role={pair['role']}  corpus={pair['csv_label']} "
            f"({pair['corpus_sha256'][:12]})  ckpt={pair['ckpt_dir']}")
        if pair["ckpt_dir"] is None:
            log("  NO correctly-paired checkpoint for this corpus; reporting the "
                "checkpoint-free controls only (they need no neural head).")
        pipe = TelemetryDataPipeline(weather_csv=csv_path)

        entry = {
            "role": pair["role"],
            "corpus": pair["csv_label"],
            "corpus_path": csv_path,
            "corpus_sha256": pair["corpus_sha256"],
            "corpus_is_snapshot": "snapshot" in pair["csv_label"],
            "checkpoint_dir": pair["ckpt_dir"],
            "pairing_status": pair["pairing"],
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
            # Checkpoint after every horizon. A full run is ~25 min of CPU and
            # this host has restarted mid-run more than once; without an
            # incremental write all completed horizons were being lost.
            report["progress"] = {
                "complete": False,
                "last_finished": f"{pid}:h{h}",
                "cells_done": [f"{k}:h{j}" for k in entry["horizons"]
                               for j in HORIZONS if f"h{j}" in entry["horizons"][k]],
            }
            write_report(report)

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
        if pair["ckpt_dir"]:
            entry["stage2_temp_head_weight_inspection"] = inspect_temp_head(pair["ckpt_dir"])
        entry["stage2_lag_profile_evidence"] = lag_profile(csv_path)
        entry["stage3_verdict"] = verdict_for(pid, entry)
        report["pairs"][pid] = entry
        report["progress"] = {"complete": False, "last_finished": f"{pid}:all_horizons"}
        write_report(report)
        del pipe

    report["progress"] = {"complete": True, "last_finished": "all pairs complete"}
    report["headline_findings"] = headline(report)
    write_report(report)
    log(f"written: {OUT_PATH}")
    return report


if __name__ == "__main__":
    if "--finalize-only" in sys.argv:
        finalize_only()
        log(f"conclusions attached to {OUT_PATH}")
    else:
        main()
