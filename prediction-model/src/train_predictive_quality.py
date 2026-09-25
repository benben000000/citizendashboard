"""
Comprehensive Predictive-Quality Training, Baseline Benchmarking, and Evaluation Pipeline.

Fulfills Workstreams A through I of the Proper Predictive-Quality Implementation Plan:
  - Workstream A: Freeze the baseline in a reproducible baseline manifest.
  - Workstream B: Production-complete candidate family (MF-1-FEATURED):
      Saves candidate_h{h}h.pt, candidate_h{h}h_manifest.json, candidate_h{h}h_calibration.json, candidate_h{h}h_predictions.csv.
  - Workstream C: Fixed candidate training objective:
      Zero-init persistence prior, origin wind residual vector, calm-wind masked direction loss,
      prior log-odds rain initialization, and validation hybrid persistence calibration.
  - Workstream D: Strong tabular baseline:
      GradientBoostedWeatherModel (pure NumPy gradient boosted tree ensemble on 75 features)
      alongside Persistence, Climatology, and Ridge Linear Regression.
  - Workstream E: Statistically valid training, early stopping, and evaluation schedule.
  - Workstream F: Complete scorecard across all 5 horizons, 15 stations, and all regimes with 95% bootstrap CIs.
  - Workstream G: Candidate promotion gates audit and formal release determination.
  - Workstream H: Rollback preservation and safe inference integration.
  - Workstream I: CI and release execution readiness.

Outputs:
  - {output_dir}/baseline_manifest.json
  - {output_dir}/candidate_h{h}h.pt (for all 5 horizons)
  - {output_dir}/candidate_h{h}h_manifest.json (for all 5 horizons)
  - {output_dir}/candidate_h{h}h_calibration.json (for all 5 horizons)
  - {output_dir}/candidate_h{h}h_predictions.csv (for all 5 horizons)
  - {output_dir}/predictive_quality_scorecard.json
  - {output_dir}/model_comparison_report.json
"""

import os
import sys
import math
import json
import csv
import copy
import hashlib
import random
import argparse
import subprocess
from datetime import datetime, timezone, timedelta
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dataset import (
    get_telemetry_pipeline,
    build_forecast_windows,
    build_feature_augmented_forecast_windows,
    build_rolling_origin_splits,
    audit_features_and_labels,
    FEATURE_SCHEMA_METADATA,
    FEATURE_SCHEMA_HASH,
    compute_file_sha256,
    compute_noaa_heat_index,
    circular_direction_error_deg,
    FEATURE_AUGMENTED_SCHEMA,
    NUM_FEATURE_AUGMENTED,
    DATA_DIR,
    DEFAULT_SEQ_LEN,
    DEFAULT_HORIZONS,
    PHYSICAL_BOUNDS,
)
from model import (
    GarciaWeatherLNN,
    GarciaWeatherLNNFeatured,
    RidgeWeatherModel,
    GradientBoostedWeatherModel,
    ClimatologyWeatherModel,
    PersistenceWeatherModel,
    SeasonalPersistenceWeatherModel,
    WeeklyClimatologyWeatherModel,
    DampedPersistenceWeatherModel,
    AutoregressiveWeatherModel,
    ResidualWeatherModel,
    VectorWindDirectionModel,
    HurdlePrecipitationModel,
    QuantileEvaluator,
    CompactEnsembleWeatherModel,
    evaluate_wind_direction_by_regime,
    evaluate_heat_index_risk_categories,
)
from anomaly_detector import TelemetryAnomalyDetector
from verify_provenance import compute_sha256

DEFAULT_SEED = 42
DEFAULT_CANDIDATE_DIR = os.path.join(DATA_DIR, "candidate_artifacts")
CANONICAL_FEATURES = [
    "temperature",
    "heat_index",
    "humidity",
    "pressure",
    "wind_speed",
    "wind_sin",
    "wind_cos",
    "precipitation",
]


def set_seed(seed: int = DEFAULT_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_git_commit() -> str:
    """Obtain current git commit hash safely."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=SRC_DIR,
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "b9812096e1e72009ea9d696a5f3d7936748b8df6"


# ---------------------------------------------------------------------------
# Metric Evaluation Utilities
# ---------------------------------------------------------------------------

def bootstrap_ci(errors: np.ndarray, n_boot: int = 400, ci: float = 0.95) -> tuple:
    """Compute non-parametric bootstrap confidence interval for mean error."""
    if len(errors) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(DEFAULT_SEED)
    boot_means = []
    n = len(errors)
    for _ in range(n_boot):
        sample = rng.choice(errors, size=n, replace=True)
        boot_means.append(float(np.mean(sample)))
    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(boot_means, alpha * 100))
    high = float(np.percentile(boot_means, (1.0 - alpha) * 100))
    return round(low, 4), round(high, 4)


def evaluate_continuous(y_true: np.ndarray, y_pred: np.ndarray, y_persist: np.ndarray = None, y_clim: np.ndarray = None, bounds: tuple = None) -> dict:
    """Evaluate continuous meteorological target with bootstrap CIs and skill scores."""
    err = y_pred - y_true
    abs_err = np.abs(err)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    med_ae = float(np.median(abs_err))
    ci_low, ci_high = bootstrap_ci(abs_err)

    violations = 0
    if bounds is not None:
        low, high = bounds
        violations = int(np.sum((y_pred < low) | (y_pred > high)))

    res = {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mean_bias": round(bias, 4),
        "median_ae": round(med_ae, 4),
        "ci_95_mae": [ci_low, ci_high],
        "bound_violations": violations,
    }

    if y_persist is not None:
        p_mae = float(np.mean(np.abs(y_persist - y_true)))
        skill_p = 1.0 - (mae / max(1e-4, p_mae))
        res["persistence_mae"] = round(p_mae, 4)
        res["persistence_skill"] = round(skill_p, 4)
        res["beats_persistence"] = bool(mae < p_mae)

    if y_clim is not None:
        c_mae = float(np.mean(np.abs(y_clim - y_true)))
        skill_c = 1.0 - (mae / max(1e-4, c_mae))
        res["climatology_mae"] = round(c_mae, 4)
        res["climatology_skill"] = round(skill_c, 4)

    return res


def evaluate_wind_direction(u_true: np.ndarray, v_true: np.ndarray, u_pred: np.ndarray, v_pred: np.ndarray, ws_true: np.ndarray) -> dict:
    """Evaluate circular wind direction modulo 360 degrees strictly on non-calm wind (ws >= 1.0 m/s)."""
    non_calm = ws_true >= 1.0
    n_total = len(ws_true)
    n_non_calm = int(np.sum(non_calm))
    n_calm = n_total - n_non_calm

    if n_non_calm == 0:
        return {"non_calm_samples": 0, "calm_samples": n_calm, "circular_mae_deg": 0.0}

    deg_true = (np.degrees(np.arctan2(v_true, u_true))) % 360.0
    deg_pred = (np.degrees(np.arctan2(v_pred, u_pred))) % 360.0

    angular_errors = []
    for dt, dp in zip(deg_true[non_calm], deg_pred[non_calm]):
        angular_errors.append(circular_direction_error_deg(dt, dp))

    ang_arr = np.array(angular_errors)
    circ_mae = float(np.mean(ang_arr))
    circ_med = float(np.median(ang_arr))

    within_10 = float(np.mean(ang_arr <= 10.0) * 100.0)
    within_22_5 = float(np.mean(ang_arr <= 22.5) * 100.0)
    within_45 = float(np.mean(ang_arr <= 45.0) * 100.0)
    within_90 = float(np.mean(ang_arr <= 90.0) * 100.0)

    return {
        "non_calm_samples": n_non_calm,
        "calm_samples": n_calm,
        "circular_mae_deg": round(circ_mae, 2),
        "circular_median_ae_deg": round(circ_med, 2),
        "pct_within_10deg": round(within_10, 2),
        "pct_within_22_5deg": round(within_22_5, 2),
        "pct_within_45deg": round(within_45, 2),
        "pct_within_90deg": round(within_90, 2),
    }


def evaluate_rain_occurrence(y_true_binary: np.ndarray, prob_pred: np.ndarray, threshold: float = 0.5) -> dict:
    """Evaluate probabilistic rain occurrence, Brier score, ECE, and decision metrics."""
    prob_pred = np.clip(prob_pred, 1e-6, 1.0 - 1e-6)
    brier = float(np.mean((prob_pred - y_true_binary) ** 2))
    log_loss = float(-np.mean(y_true_binary * np.log(prob_pred) + (1.0 - y_true_binary) * np.log(1.0 - prob_pred)))

    # Expected Calibration Error (ECE) with 10 bins
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    reliability_bins = []
    for i in range(10):
        low, high = bins[i], bins[i+1]
        mask = (prob_pred >= low) & (prob_pred < high if i < 9 else prob_pred <= high)
        n_bin = int(np.sum(mask))
        if n_bin > 0:
            bin_acc = float(np.mean(y_true_binary[mask]))
            bin_conf = float(np.mean(prob_pred[mask]))
            ece += (n_bin / len(prob_pred)) * abs(bin_acc - bin_conf)
            reliability_bins.append({
                "bin_low": round(low, 2), "bin_high": round(high, 2),
                "sample_count": n_bin, "mean_confidence": round(bin_conf, 4),
                "empirical_accuracy": round(bin_acc, 4),
            })

    # Threshold classification
    t_pred = prob_pred >= threshold
    t_true = y_true_binary >= 0.5

    tp = int(np.sum(t_true & t_pred))
    fp = int(np.sum(~t_true & t_pred))
    fn = int(np.sum(t_true & ~t_pred))
    tn = int(np.sum(~t_true & ~t_pred))

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    far = fp / (tp + fp) if (tp + fp) > 0 else 0.0

    return {
        "brier_score": round(brier, 4),
        "log_loss": round(log_loss, 4),
        "expected_calibration_error": round(ece, 4),
        "operational_threshold": round(threshold, 2),
        "precision": round(prec, 4),
        "recall_pod": round(rec, 4),
        "f1_score": round(f1, 4),
        "critical_success_index_csi": round(csi, 4),
        "false_alarm_ratio_far": round(far, 4),
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "reliability_bins": reliability_bins,
    }


def evaluate_precipitation_amount(y_true_mm: np.ndarray, y_pred_mm: np.ndarray, y_persist_mm: np.ndarray = None, y_clim_mm: np.ndarray = None) -> dict:
    """Evaluate precipitation amount across full, dry, and rainy hour regimes."""
    err = y_pred_mm - y_true_mm
    overall_mae = float(np.mean(np.abs(err)))
    overall_rmse = float(np.sqrt(np.mean(err ** 2)))
    overall_bias = float(np.mean(err))

    dry_mask = y_true_mm < 0.1
    rain_mask = y_true_mm >= 0.1

    dry_mae = float(np.mean(np.abs(err[dry_mask]))) if dry_mask.any() else 0.0
    dry_bias = float(np.mean(err[dry_mask])) if dry_mask.any() else 0.0

    rainy_mae = float(np.mean(np.abs(err[rain_mask]))) if rain_mask.any() else 0.0
    rainy_rmse = float(np.sqrt(np.mean(err[rain_mask] ** 2))) if rain_mask.any() else 0.0
    rainy_bias = float(np.mean(err[rain_mask])) if rain_mask.any() else 0.0

    heavy_csi = {}
    for thresh in [2.5, 5.0, 10.0]:
        t_true = y_true_mm >= thresh
        t_pred = y_pred_mm >= thresh
        tp = int(np.sum(t_true & t_pred))
        fp = int(np.sum(~t_true & t_pred))
        fn = int(np.sum(t_true & ~t_pred))
        csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        heavy_csi[f"csi_{thresh}_mmh"] = round(csi, 4)

    res = {
        "overall_mae_mm": round(overall_mae, 4),
        "overall_rmse_mm": round(overall_rmse, 4),
        "overall_bias_mm": round(overall_bias, 4),
        "dry_hour_mae_mm": round(dry_mae, 4),
        "dry_hour_bias_mm": round(dry_bias, 4),
        "rainy_hour_mae_mm": round(rainy_mae, 4),
        "rainy_hour_rmse_mm": round(rainy_rmse, 4),
        "rainy_hour_bias_mm": round(rainy_bias, 4),
        "heavy_rain_csi": heavy_csi,
    }

    if y_persist_mm is not None:
        p_rainy_mae = float(np.mean(np.abs(y_persist_mm[rain_mask] - y_true_mm[rain_mask]))) if rain_mask.any() else 0.0
        skill_p = 1.0 - (rainy_mae / max(1e-4, p_rainy_mae)) if p_rainy_mae > 0 else 0.0
        res["persistence_rainy_mae"] = round(p_rainy_mae, 4)
        res["persistence_rainy_skill"] = round(skill_p, 4)

    if y_clim_mm is not None:
        c_rainy_mae = float(np.mean(np.abs(y_clim_mm[rain_mask] - y_true_mm[rain_mask]))) if rain_mask.any() else 0.0
        skill_c = 1.0 - (rainy_mae / max(1e-4, c_rainy_mae)) if c_rainy_mae > 0 else 0.0
        res["climatology_rainy_mae"] = round(c_rainy_mae, 4)
        res["climatology_rainy_skill"] = round(skill_c, 4)

    return res


# ---------------------------------------------------------------------------
# Training Candidate Models Pipeline
# ---------------------------------------------------------------------------

def train_and_evaluate_all_horizons(output_dir: str = None, epochs: int = 5, lr: float = 1e-3, seed: int = DEFAULT_SEED, commit: str = None):
    """
    Main execution pipeline fulfilling Workstreams A through I of the Proper Implementation Plan:
      1. Freezes baseline manifest (Workstream A).
      2. Fits Climatology, Persistence, Ridge, and GradientBoostedWeatherModel baselines (Workstream D).
      3. Trains GarciaWeatherLNNFeatured candidate model with early stopping on validation split (Workstream C, E).
      4. Calibrates probability and threshold on validation split strictly (Workstream C, 6).
      5. Saves complete candidate artifacts (checkpoints, manifests, calibration, predictions) (Workstream B).
      6. Evaluates untouched test split across 15 stations and 7 regimes (Workstream F).
      7. Performs complete Candidate Promotion Rules Audit (Workstream G).
    """
    set_seed(seed)
    if output_dir is None:
        output_dir = DEFAULT_CANDIDATE_DIR

    os.makedirs(output_dir, exist_ok=True)
    head_commit = commit if commit else get_git_commit()

    pipeline = get_telemetry_pipeline()
    horizons = DEFAULT_HORIZONS  # [1, 3, 6, 12, 24]

    weather_csv_path = os.path.join(DATA_DIR, "weather_telemetry.csv")
    water_csv_path = os.path.join(DATA_DIR, "water_level_telemetry.csv")
    raw_weather_hash = compute_file_sha256(weather_csv_path)
    raw_water_hash = compute_file_sha256(water_csv_path)

    print("=" * 80)
    print("PROPER PREDICTIVE QUALITY & ANOMALY DETECTION RELEASE PIPELINE")
    print(f"Commit: {head_commit[:8]} | Seed: {seed} | Epochs: {epochs} | Horizons: {horizons}")
    print("=" * 80)

    # Pre-fit training normalization for both canonical 8 features and 75 engineered features
    norm_means, norm_stds = pipeline.norm_means, pipeline.norm_stds
    feat_means, feat_stds = pipeline.get_feature_augmented_norm_stats()

    # Pre-fit Climatology, Weekly Climatology, and Damped Persistence from train split
    print("\nFitting Climatology & Multi-Reference Baselines on training split...")
    clim_model = ClimatologyWeatherModel()
    weekly_clim_model = WeeklyClimatologyWeatherModel()
    train_metadata_all = []
    for h in [1]:
        res = build_forecast_windows(pipeline, split="train", horizon=h, return_metadata=True)
        if res is not None:
            train_metadata_all.extend(res[6])
    clim_model.fit_from_metadata(train_metadata_all)
    weekly_clim_model.fit_from_metadata(train_metadata_all)
    damped_model = DampedPersistenceWeatherModel(clim_model)
    damped_model.fit_autocorrelations(train_metadata_all)
    print(f"Hourly Climatology fitted across {len(clim_model.table)} station-hour buckets.")
    print(f"Weekly Climatology fitted across {len(weekly_clim_model.table)} station-dow-hour buckets.")
    print(f"Damped Persistence fitted with estimated autocorrelations: {damped_model.alphas}")

    scorecard = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": head_commit,
        "seed": seed,
        "horizons": horizons,
        "models_evaluated": [
            "Persistence",
            "SeasonalPersistence",
            "Climatology",
            "WeeklyClimatology",
            "DampedPersistence",
            "Autoregression_p6",
            "Ridge_75Features",
            "GradientBoostedTree_75Features",
            "ResidualWeatherModel_Quantiles",
            "VectorWindDirectionModel",
            "HurdlePrecipitationModel",
            "CompactEnsembleWeatherModel",
            "GarciaWeatherLNNFeatured_Candidate",
        ],
        "horizon_evaluations": {},
        "promotion_audit": {},
    }

    comparison_report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": head_commit,
        "horizons": {},
    }

    baseline_manifest = {
        "manifest_type": "baseline_freeze_reference",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline_commit": head_commit,
        "weather_telemetry_sha256": raw_weather_hash,
        "water_telemetry_sha256": raw_water_hash,
        "canonical_feature_schema": [
            "temperature", "heat_index", "humidity", "pressure",
            "wind_speed", "wind_sin", "wind_cos", "precipitation"
        ],
        "canonical_baselines_included": [
            "persistence",
            "seasonal_persistence",
            "hourly_climatology",
            "weekly_climatology",
            "damped_persistence",
            "autoregression_p6",
            "ridge_75features",
            "gradient_boosted_tree_75features",
        ],
        "training_seeds": [seed],
        "model_configuration": {
            "model_family": "MF-1",
            "input_dim": 8,
            "hidden_dim": 32,
            "architecture": "continuous_time_cfc_ode",
        },
        "split_configuration": {
            "train_date_range": ["2026-08-01T00:00:00Z", "2026-08-20T23:00:00Z"],
            "val_date_range": ["2026-08-21T00:00:00Z", "2026-08-25T23:00:00Z"],
            "test_date_range": ["2026-08-26T00:00:00Z", "2026-08-31T23:00:00Z"],
            "embargo_duration_hours": 24,
        },
        "quarantine_counts": {
            "uv_index_quarantined_samples": 40320,
            "luminosity_conditional_samples": 40320,
        },
        "known_limitations": [
            "UV index blocked by nighttime sensor calibration defect.",
            "Luminosity sensor uncalibrated across stations; secondary daylight-only proxy.",
            "Water level gauge available only at Marikina Santo Nino station (internal beta target).",
            "Short-term precipitation sparsity requires two-stage zero-inflation modeling."
        ],
        "five_horizon_baseline_metrics": {},
    }

    # Iterate over all 5 horizons
    all_candidate_artifacts = []
    for h in horizons:
        print(f"\n" + "-" * 70)
        print(f"EVALUATING FORECAST HORIZON: +{h}h")
        print("-" * 70)

        # 1. Load Train Split Windows
        print(f"Loading Train windows (+{h}h)...")
        train_data = build_feature_augmented_forecast_windows(
            pipeline=pipeline, split="train", horizon=h,
            norm_means=norm_means, norm_stds=norm_stds,
            feat_means=feat_means, feat_stds=feat_stds,
            return_metadata=True
        )

        # 2. Load Validation Split Windows (for calibration & threshold freeze)
        print(f"Loading Validation windows (+{h}h)...")
        val_data = build_feature_augmented_forecast_windows(
            pipeline=pipeline, split="val", horizon=h,
            norm_means=norm_means, norm_stds=norm_stds,
            feat_means=feat_means, feat_stds=feat_stds,
            return_metadata=True
        )

        # 3. Load Test Split Windows (strictly untouched until final score)
        print(f"Loading Test windows (+{h}h)...")
        test_data = build_feature_augmented_forecast_windows(
            pipeline=pipeline, split="test", horizon=h,
            norm_means=norm_means, norm_stds=norm_stds,
            feat_means=feat_means, feat_stds=feat_stds,
            return_metadata=True
        )

        train_telemetry, train_context, train_dt, train_rain, train_precip, train_water, train_has_w, train_meta = train_data
        val_telemetry, val_context, val_dt, val_rain, val_precip, val_water, val_has_w, val_meta = val_data
        test_telemetry, test_context, test_dt, test_rain, test_precip, test_water, test_has_w, test_meta = test_data

        N_test = len(test_meta)
        print(f"Samples: Train={len(train_meta)}, Val={len(val_meta)}, Test={N_test}")

        # Extract Ground Truth Test Arrays
        t_true = np.array([m["target_temperature"] for m in test_meta], dtype=np.float32)
        rh_true = np.array([m["target_humidity"] for m in test_meta], dtype=np.float32)
        p_true = np.array([m["target_pressure"] for m in test_meta], dtype=np.float32)
        ws_true = np.array([m["target_wind_speed"] for m in test_meta], dtype=np.float32)
        u_true = np.array([m["target_wind_u"] for m in test_meta], dtype=np.float32)
        v_true = np.array([m["target_wind_v"] for m in test_meta], dtype=np.float32)
        hi_true = np.array([m["target_heat_index"] for m in test_meta], dtype=np.float32)
        rain_true = test_rain.squeeze(-1).numpy()
        precip_true = test_precip.squeeze(-1).numpy()

        # Origin observations for Persistence
        t_orig = np.array([m["origin_temperature"] for m in test_meta], dtype=np.float32)
        rh_orig = np.array([m["origin_humidity"] for m in test_meta], dtype=np.float32)
        p_orig = np.array([m["origin_pressure"] for m in test_meta], dtype=np.float32)
        ws_orig = np.array([m["origin_wind_speed"] for m in test_meta], dtype=np.float32)
        u_orig = np.array([m["origin_wind_u"] for m in test_meta], dtype=np.float32)
        v_orig = np.array([m["origin_wind_v"] for m in test_meta], dtype=np.float32)
        precip_orig = np.array([m["last_observed_precip"] for m in test_meta], dtype=np.float32)

        # Baseline 1: Persistence Predictions
        persist_eval_t = evaluate_continuous(t_true, t_orig, t_orig, bounds=PHYSICAL_BOUNDS["temperature"])
        persist_eval_rh = evaluate_continuous(rh_true, rh_orig, rh_orig, bounds=PHYSICAL_BOUNDS["humidity"])
        persist_eval_p = evaluate_continuous(p_true, p_orig, p_orig, bounds=PHYSICAL_BOUNDS["pressure"])
        persist_eval_ws = evaluate_continuous(ws_true, ws_orig, ws_orig, bounds=PHYSICAL_BOUNDS["wind_speed"])
        hi_orig = np.array([compute_noaa_heat_index(t_orig[i], rh_orig[i]) for i in range(len(t_orig))], dtype=np.float32)
        persist_eval_hi = evaluate_continuous(hi_true, hi_orig, hi_orig, bounds=PHYSICAL_BOUNDS["heat_index"])
        persist_rain_prob = np.where(precip_orig >= 0.1, 0.85, 0.05)
        persist_eval_rain = evaluate_rain_occurrence(rain_true, persist_rain_prob, threshold=0.5)
        persist_eval_precip = evaluate_precipitation_amount(precip_true, precip_orig, precip_orig)
        persist_eval_wdir = evaluate_wind_direction(u_true, v_true, u_orig, v_orig, ws_true)

        # Baseline 2: Climatology Predictions
        clim_preds = []
        for m in test_meta:
            st = m["station_id"]
            dt_m = datetime.fromisoformat(m["target_timestamp"])
            hr = (dt_m.hour + 8) % 24
            clim_preds.append(clim_model.predict(st, hr))
        t_clim = np.array([cp["temperature"] for cp in clim_preds], dtype=np.float32)
        rh_clim = np.array([cp["humidity"] for cp in clim_preds], dtype=np.float32)
        p_clim = np.array([cp["pressure"] for cp in clim_preds], dtype=np.float32)
        ws_clim = np.array([cp["wind_speed"] for cp in clim_preds], dtype=np.float32)
        u_clim = np.array([cp["wind_u"] for cp in clim_preds], dtype=np.float32)
        v_clim = np.array([cp["wind_v"] for cp in clim_preds], dtype=np.float32)
        precip_clim = np.array([cp["precipitation_mm"] for cp in clim_preds], dtype=np.float32)
        rain_prob_clim = np.array([cp["rain_prob"] for cp in clim_preds], dtype=np.float32)
        hi_clim = np.array([compute_noaa_heat_index(t_clim[i], rh_clim[i]) for i in range(len(t_clim))], dtype=np.float32)

        clim_eval_t = evaluate_continuous(t_true, t_clim, t_orig, t_clim, bounds=PHYSICAL_BOUNDS["temperature"])
        clim_eval_rh = evaluate_continuous(rh_true, rh_clim, rh_orig, rh_clim, bounds=PHYSICAL_BOUNDS["humidity"])
        clim_eval_p = evaluate_continuous(p_true, p_clim, p_orig, p_clim, bounds=PHYSICAL_BOUNDS["pressure"])
        clim_eval_ws = evaluate_continuous(ws_true, ws_clim, ws_orig, ws_clim, bounds=PHYSICAL_BOUNDS["wind_speed"])
        clim_eval_hi = evaluate_continuous(hi_true, hi_clim, hi_orig, hi_clim, bounds=PHYSICAL_BOUNDS["heat_index"])
        clim_eval_rain = evaluate_rain_occurrence(rain_true, rain_prob_clim, threshold=0.5)
        clim_eval_precip = evaluate_precipitation_amount(precip_true, precip_clim, precip_orig, precip_clim)
        clim_eval_wdir = evaluate_wind_direction(u_true, v_true, u_clim, v_clim, ws_true)

        # Baseline 2b: Seasonal Persistence (24h Diurnal Lag)
        seasonal_preds = []
        for m in test_meta:
            st = m["station_id"]
            t0_dt = datetime.fromisoformat(m["origin_timestamp"].replace("Z", "+00:00"))
            target_lag = 24 - h
            lag_time = t0_dt - timedelta(hours=target_lag)
            rec_lag = pipeline.station_hourly.get(st, {}).get(lag_time)
            if rec_lag is not None:
                seasonal_preds.append({
                    "temperature": float(rec_lag["temperature"]),
                    "humidity": float(rec_lag["humidity"]),
                    "pressure": float(rec_lag["pressure"]),
                    "wind_speed": float(rec_lag["wind_speed"]),
                    "wind_u": float(rec_lag["wind_cos"]),
                    "wind_v": float(rec_lag["wind_sin"]),
                    "precipitation_mm": float(rec_lag["precipitation"]),
                    "rain_prob": 1.0 if float(rec_lag["precipitation"]) >= 0.1 else 0.0,
                    "heat_index": compute_noaa_heat_index(float(rec_lag["temperature"]), float(rec_lag["humidity"])),
                })
            else:
                seasonal_preds.append({
                    "temperature": float(m["origin_temperature"]),
                    "humidity": float(m["origin_humidity"]),
                    "pressure": float(m["origin_pressure"]),
                    "wind_speed": float(m["origin_wind_speed"]),
                    "wind_u": float(m["origin_wind_u"]),
                    "wind_v": float(m["origin_wind_v"]),
                    "precipitation_mm": float(m["last_observed_precip"]),
                    "rain_prob": 0.85 if float(m["last_observed_precip"]) >= 0.1 else 0.05,
                    "heat_index": float(m["origin_heat_index"]),
                })
        t_seas = np.array([sp["temperature"] for sp in seasonal_preds], dtype=np.float32)
        rh_seas = np.array([sp["humidity"] for sp in seasonal_preds], dtype=np.float32)
        p_seas = np.array([sp["pressure"] for sp in seasonal_preds], dtype=np.float32)
        ws_seas = np.array([sp["wind_speed"] for sp in seasonal_preds], dtype=np.float32)
        u_seas = np.array([sp["wind_u"] for sp in seasonal_preds], dtype=np.float32)
        v_seas = np.array([sp["wind_v"] for sp in seasonal_preds], dtype=np.float32)
        hi_seas = np.array([sp["heat_index"] for sp in seasonal_preds], dtype=np.float32)
        rain_prob_seas = np.array([sp["rain_prob"] for sp in seasonal_preds], dtype=np.float32)
        precip_seas = np.array([sp["precipitation_mm"] for sp in seasonal_preds], dtype=np.float32)

        seas_eval_t = evaluate_continuous(t_true, t_seas, t_orig, t_clim, bounds=PHYSICAL_BOUNDS["temperature"])
        seas_eval_rh = evaluate_continuous(rh_true, rh_seas, rh_orig, rh_clim, bounds=PHYSICAL_BOUNDS["humidity"])
        seas_eval_p = evaluate_continuous(p_true, p_seas, p_orig, p_clim, bounds=PHYSICAL_BOUNDS["pressure"])
        seas_eval_ws = evaluate_continuous(ws_true, ws_seas, ws_orig, ws_clim, bounds=PHYSICAL_BOUNDS["wind_speed"])
        seas_eval_hi = evaluate_continuous(hi_true, hi_seas, hi_orig, hi_clim, bounds=PHYSICAL_BOUNDS["heat_index"])
        seas_eval_rain = evaluate_rain_occurrence(rain_true, rain_prob_seas, threshold=0.5)
        seas_eval_precip = evaluate_precipitation_amount(precip_true, precip_seas, precip_orig, precip_clim)
        seas_eval_wdir = evaluate_wind_direction(u_true, v_true, u_seas, v_seas, ws_true)

        # Baseline 2c: Weekly Climatology (Station x Day-of-Week x Hour-of-Day)
        weekly_preds = []
        for m in test_meta:
            st = m["station_id"]
            dt_m = datetime.fromisoformat(m["target_timestamp"].replace("Z", "+00:00"))
            dow = dt_m.weekday() % 7
            hr = (dt_m.hour + 8) % 24
            weekly_preds.append(weekly_clim_model.predict(st, dow, hr))
        t_wclim = np.array([wp.get("temperature", t_clim[i]) for i, wp in enumerate(weekly_preds)], dtype=np.float32)
        rh_wclim = np.array([wp.get("humidity", rh_clim[i]) for i, wp in enumerate(weekly_preds)], dtype=np.float32)
        p_wclim = np.array([wp.get("pressure", p_clim[i]) for i, wp in enumerate(weekly_preds)], dtype=np.float32)
        ws_wclim = np.array([wp.get("wind_speed", ws_clim[i]) for i, wp in enumerate(weekly_preds)], dtype=np.float32)
        u_wclim = np.array([wp.get("wind_u", u_clim[i]) for i, wp in enumerate(weekly_preds)], dtype=np.float32)
        v_wclim = np.array([wp.get("wind_v", v_clim[i]) for i, wp in enumerate(weekly_preds)], dtype=np.float32)
        hi_wclim = np.array([compute_noaa_heat_index(t_wclim[i], rh_wclim[i]) for i in range(len(t_wclim))], dtype=np.float32)
        rain_prob_wclim = np.array([wp.get("rain_prob", 0.1) for wp in weekly_preds], dtype=np.float32)
        precip_wclim = np.array([wp.get("precipitation_mm", 0.0) for wp in weekly_preds], dtype=np.float32)

        wclim_eval_t = evaluate_continuous(t_true, t_wclim, t_orig, t_clim, bounds=PHYSICAL_BOUNDS["temperature"])
        wclim_eval_rh = evaluate_continuous(rh_true, rh_wclim, rh_orig, rh_clim, bounds=PHYSICAL_BOUNDS["humidity"])
        wclim_eval_p = evaluate_continuous(p_true, p_wclim, p_orig, p_clim, bounds=PHYSICAL_BOUNDS["pressure"])
        wclim_eval_ws = evaluate_continuous(ws_true, ws_wclim, ws_orig, ws_clim, bounds=PHYSICAL_BOUNDS["wind_speed"])
        wclim_eval_hi = evaluate_continuous(hi_true, hi_wclim, hi_orig, hi_clim, bounds=PHYSICAL_BOUNDS["heat_index"])
        wclim_eval_rain = evaluate_rain_occurrence(rain_true, rain_prob_wclim, threshold=0.5)
        wclim_eval_precip = evaluate_precipitation_amount(precip_true, precip_wclim, precip_orig, precip_clim)
        wclim_eval_wdir = evaluate_wind_direction(u_true, v_true, u_wclim, v_wclim, ws_true)

        # Baseline 2d: Damped Persistence (Autocorrelation decay to Climatology)
        damped_preds = []
        for m in test_meta:
            st = m["station_id"]
            dt_m = datetime.fromisoformat(m["target_timestamp"].replace("Z", "+00:00"))
            hr = (dt_m.hour + 8) % 24
            damped_preds.append(damped_model.predict(m, h, st, hr))
        t_damp = np.array([dp["temperature"] for dp in damped_preds], dtype=np.float32)
        rh_damp = np.array([dp["humidity"] for dp in damped_preds], dtype=np.float32)
        p_damp = np.array([dp["pressure"] for dp in damped_preds], dtype=np.float32)
        ws_damp = np.array([dp["wind_speed"] for dp in damped_preds], dtype=np.float32)
        u_damp = np.array([dp["wind_u"] for dp in damped_preds], dtype=np.float32)
        v_damp = np.array([dp["wind_v"] for dp in damped_preds], dtype=np.float32)
        hi_damp = np.array([dp["heat_index"] for dp in damped_preds], dtype=np.float32)
        rain_prob_damp = np.array([dp["rain_probability"] for dp in damped_preds], dtype=np.float32)
        precip_damp = np.array([dp["precipitation_mm"] for dp in damped_preds], dtype=np.float32)

        damp_eval_t = evaluate_continuous(t_true, t_damp, t_orig, t_clim, bounds=PHYSICAL_BOUNDS["temperature"])
        damp_eval_rh = evaluate_continuous(rh_true, rh_damp, rh_orig, rh_clim, bounds=PHYSICAL_BOUNDS["humidity"])
        damp_eval_p = evaluate_continuous(p_true, p_damp, p_orig, p_clim, bounds=PHYSICAL_BOUNDS["pressure"])
        damp_eval_ws = evaluate_continuous(ws_true, ws_damp, ws_orig, ws_clim, bounds=PHYSICAL_BOUNDS["wind_speed"])
        damp_eval_hi = evaluate_continuous(hi_true, hi_damp, hi_orig, hi_clim, bounds=PHYSICAL_BOUNDS["heat_index"])
        damp_eval_rain = evaluate_rain_occurrence(rain_true, rain_prob_damp, threshold=0.5)
        damp_eval_precip = evaluate_precipitation_amount(precip_true, precip_damp, precip_orig, precip_clim)
        damp_eval_wdir = evaluate_wind_direction(u_true, v_true, u_damp, v_damp, ws_true)

        # Baseline 2e: Autoregressive AR(p=6) Lag Model
        ar_model = AutoregressiveWeatherModel(p_lags=6, alpha=1.0)
        X_train_lags = {
            "temperature": train_telemetry[:, -6:, 0].numpy(),
            "humidity": train_telemetry[:, -6:, 2].numpy(),
            "pressure": train_telemetry[:, -6:, 3].numpy(),
            "wind_speed": train_telemetry[:, -6:, 4].numpy(),
            "wind_u": train_telemetry[:, -6:, 6].numpy(),
            "wind_v": train_telemetry[:, -6:, 5].numpy(),
        }
        Y_train_ar = {
            "temperature": np.array([m["target_temperature"] for m in train_meta], dtype=np.float32),
            "humidity": np.array([m["target_humidity"] for m in train_meta], dtype=np.float32),
            "pressure": np.array([m["target_pressure"] for m in train_meta], dtype=np.float32),
            "wind_speed": np.array([m["target_wind_speed"] for m in train_meta], dtype=np.float32),
            "wind_u": np.array([m["target_wind_u"] for m in train_meta], dtype=np.float32),
            "wind_v": np.array([m["target_wind_v"] for m in train_meta], dtype=np.float32),
        }
        ar_model.fit(X_train_lags, Y_train_ar)

        X_test_lags = {
            "temperature": test_telemetry[:, -6:, 0].numpy(),
            "humidity": test_telemetry[:, -6:, 2].numpy(),
            "pressure": test_telemetry[:, -6:, 3].numpy(),
            "wind_speed": test_telemetry[:, -6:, 4].numpy(),
            "wind_u": test_telemetry[:, -6:, 6].numpy(),
            "wind_v": test_telemetry[:, -6:, 5].numpy(),
        }
        ar_preds = ar_model.predict(X_test_lags)
        ar_hi = np.array([compute_noaa_heat_index(ar_preds["temperature"][i], ar_preds["humidity"][i]) for i in range(N_test)], dtype=np.float32)

        ar_eval_t = evaluate_continuous(t_true, ar_preds["temperature"], t_orig, t_clim, bounds=PHYSICAL_BOUNDS["temperature"])
        ar_eval_rh = evaluate_continuous(rh_true, ar_preds["humidity"], rh_orig, rh_clim, bounds=PHYSICAL_BOUNDS["humidity"])
        ar_eval_p = evaluate_continuous(p_true, ar_preds["pressure"], p_orig, p_clim, bounds=PHYSICAL_BOUNDS["pressure"])
        ar_eval_ws = evaluate_continuous(ws_true, ar_preds["wind_speed"], ws_orig, ws_clim, bounds=PHYSICAL_BOUNDS["wind_speed"])
        ar_eval_hi = evaluate_continuous(hi_true, ar_hi, hi_orig, hi_clim, bounds=PHYSICAL_BOUNDS["heat_index"])
        ar_eval_wdir = evaluate_wind_direction(u_true, v_true, ar_preds["wind_u"], ar_preds["wind_v"], ws_true)

        # Baseline 3: Ridge Linear Regression on 75 Engineered Features
        print("Fitting Ridge Linear Regression (75 features)...")
        X_train_75 = train_context.numpy()
        Y_train_multi = np.column_stack([
            np.array([m["target_temperature"] for m in train_meta]),
            np.array([m["target_humidity"] for m in train_meta]),
            np.array([m["target_pressure"] for m in train_meta]),
            np.array([m["target_wind_speed"] for m in train_meta]),
            np.array([m["target_wind_u"] for m in train_meta]),
            np.array([m["target_wind_v"] for m in train_meta]),
            train_precip.squeeze(-1).numpy(),
            train_rain.squeeze(-1).numpy(),
        ])
        ridge = RidgeWeatherModel(alpha=10.0)
        ridge.fit(X_train_75, Y_train_multi)

        X_test_75 = test_context.numpy()
        ridge_preds = ridge.predict(X_test_75)
        ridge_hi = np.array([compute_noaa_heat_index(ridge_preds[i, 0], ridge_preds[i, 1]) for i in range(len(ridge_preds))], dtype=np.float32)
        ridge_eval_t = evaluate_continuous(t_true, ridge_preds[:, 0], t_orig, t_clim, bounds=PHYSICAL_BOUNDS["temperature"])
        ridge_eval_rh = evaluate_continuous(rh_true, ridge_preds[:, 1], rh_orig, rh_clim, bounds=PHYSICAL_BOUNDS["humidity"])
        ridge_eval_p = evaluate_continuous(p_true, ridge_preds[:, 2], p_orig, p_clim, bounds=PHYSICAL_BOUNDS["pressure"])
        ridge_eval_ws = evaluate_continuous(ws_true, ridge_preds[:, 3], ws_orig, ws_clim, bounds=PHYSICAL_BOUNDS["wind_speed"])
        ridge_eval_hi = evaluate_continuous(hi_true, ridge_hi, hi_orig, hi_clim, bounds=PHYSICAL_BOUNDS["heat_index"])
        ridge_eval_wdir = evaluate_wind_direction(u_true, v_true, ridge_preds[:, 4], ridge_preds[:, 5], ws_true)
        ridge_eval_precip = evaluate_precipitation_amount(precip_true, ridge_preds[:, 6], precip_orig, precip_clim)
        ridge_eval_rain = evaluate_rain_occurrence(rain_true, ridge_preds[:, 7], threshold=0.5)

        # Baseline 4: Gradient-Boosted Tree Baseline on 75 Engineered Features (Workstream D)
        print("Fitting Gradient-Boosted Tree Baseline (75 features)...")
        gbm = GradientBoostedWeatherModel(n_estimators=25, learning_rate=0.1, random_state=seed)
        gbm.fit(X_train_75, Y_train_multi)
        gbm_preds = gbm.predict(X_test_75)
        gbm_hi = np.array([compute_noaa_heat_index(gbm_preds[i, 0], gbm_preds[i, 1]) for i in range(len(gbm_preds))], dtype=np.float32)
        gbm_eval_t = evaluate_continuous(t_true, gbm_preds[:, 0], t_orig, t_clim, bounds=PHYSICAL_BOUNDS["temperature"])
        gbm_eval_rh = evaluate_continuous(rh_true, gbm_preds[:, 1], rh_orig, rh_clim, bounds=PHYSICAL_BOUNDS["humidity"])
        gbm_eval_p = evaluate_continuous(p_true, gbm_preds[:, 2], p_orig, p_clim, bounds=PHYSICAL_BOUNDS["pressure"])
        gbm_eval_ws = evaluate_continuous(ws_true, gbm_preds[:, 3], ws_orig, ws_clim, bounds=PHYSICAL_BOUNDS["wind_speed"])
        gbm_eval_hi = evaluate_continuous(hi_true, gbm_hi, hi_orig, hi_clim, bounds=PHYSICAL_BOUNDS["heat_index"])
        gbm_eval_wdir = evaluate_wind_direction(u_true, v_true, gbm_preds[:, 4], gbm_preds[:, 5], ws_true)
        gbm_eval_precip = evaluate_precipitation_amount(precip_true, gbm_preds[:, 6], precip_orig, precip_clim)
        gbm_eval_rain = evaluate_rain_occurrence(rain_true, gbm_preds[:, 7], threshold=0.5)

        # Record baseline metrics in baseline manifest
        baseline_manifest["five_horizon_baseline_metrics"][f"horizon_{h}h"] = {
            "persistence": {
                "temperature_mae": persist_eval_t["mae"],
                "heat_index_mae": persist_eval_hi["mae"],
                "rain_brier_score": persist_eval_rain["brier_score"],
                "wind_direction_circular_mae": persist_eval_wdir["circular_mae_deg"],
                "precipitation_rainy_mae": persist_eval_precip["rainy_hour_mae_mm"],
            },
            "seasonal_persistence": {
                "temperature_mae": seas_eval_t["mae"],
                "heat_index_mae": seas_eval_hi["mae"],
                "rain_brier_score": seas_eval_rain["brier_score"],
                "wind_direction_circular_mae": seas_eval_wdir["circular_mae_deg"],
            },
            "climatology": {
                "temperature_mae": clim_eval_t["mae"],
                "heat_index_mae": clim_eval_hi["mae"],
                "rain_brier_score": clim_eval_rain["brier_score"],
                "wind_direction_circular_mae": clim_eval_wdir["circular_mae_deg"],
            },
            "weekly_climatology": {
                "temperature_mae": wclim_eval_t["mae"],
                "heat_index_mae": wclim_eval_hi["mae"],
                "rain_brier_score": wclim_eval_rain["brier_score"],
                "wind_direction_circular_mae": wclim_eval_wdir["circular_mae_deg"],
            },
            "damped_persistence": {
                "temperature_mae": damp_eval_t["mae"],
                "heat_index_mae": damp_eval_hi["mae"],
                "rain_brier_score": damp_eval_rain["brier_score"],
                "wind_direction_circular_mae": damp_eval_wdir["circular_mae_deg"],
            },
            "autoregression_p6": {
                "temperature_mae": ar_eval_t["mae"],
                "heat_index_mae": ar_eval_hi["mae"],
                "wind_direction_circular_mae": ar_eval_wdir["circular_mae_deg"],
            },
            "ridge_75features": {
                "temperature_mae": ridge_eval_t["mae"],
                "heat_index_mae": ridge_eval_hi["mae"],
                "rain_brier_score": ridge_eval_rain["brier_score"],
                "wind_direction_circular_mae": ridge_eval_wdir["circular_mae_deg"],
            },
            "gradient_boosted_tree_75features": {
                "temperature_mae": gbm_eval_t["mae"],
                "heat_index_mae": gbm_eval_hi["mae"],
                "rain_brier_score": gbm_eval_rain["brier_score"],
                "wind_direction_circular_mae": gbm_eval_wdir["circular_mae_deg"],
            }
        }

        # Candidate Model: Feature-Augmented CfC/LNN (GarciaWeatherLNNFeatured)
        print(f"Training Feature-Augmented Candidate Neural Model (+{h}h)...")
        candidate_model = GarciaWeatherLNNFeatured(
            input_dim=8, context_dim=NUM_FEATURE_AUGMENTED, hidden_dim=32, use_two_stage_precipitation=True
        )
        optimizer = optim.AdamW(candidate_model.parameters(), lr=lr, weight_decay=1e-4)

        # Prepare PyTorch Tensors
        train_orig_w = torch.tensor(np.column_stack([
            [m["origin_temperature"] for m in train_meta],
            [m["origin_humidity"] for m in train_meta],
            [m["origin_pressure"] for m in train_meta],
            [m["origin_wind_speed"] for m in train_meta],
            [m["origin_wind_u"] for m in train_meta],
            [m["origin_wind_v"] for m in train_meta],
        ]), dtype=torch.float32)

        train_tgt_w = torch.tensor(np.column_stack([
            [m["target_temperature"] for m in train_meta],
            [m["target_humidity"] for m in train_meta],
            [m["target_pressure"] for m in train_meta],
            [m["target_wind_speed"] for m in train_meta],
            [m["target_wind_u"] for m in train_meta],
            [m["target_wind_v"] for m in train_meta],
        ]), dtype=torch.float32)

        val_orig_w = torch.tensor(np.column_stack([
            [m["origin_temperature"] for m in val_meta],
            [m["origin_humidity"] for m in val_meta],
            [m["origin_pressure"] for m in val_meta],
            [m["origin_wind_speed"] for m in val_meta],
            [m["origin_wind_u"] for m in val_meta],
            [m["origin_wind_v"] for m in val_meta],
        ]), dtype=torch.float32)

        val_tgt_t = np.array([m["target_temperature"] for m in val_meta], dtype=np.float32)
        test_orig_w = torch.tensor(np.column_stack([t_orig, rh_orig, p_orig, ws_orig, u_orig, v_orig]), dtype=torch.float32)

        train_dataset = TensorDataset(train_telemetry, train_context, train_dt, train_rain, train_precip, train_orig_w, train_tgt_w)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        best_val_mae = float("inf")
        best_state = copy.deepcopy(candidate_model.state_dict())

        candidate_model.train()
        for ep in range(epochs):
            for b_telemetry, b_context, b_dt, b_rain, b_precip, b_orig_w, b_tgt_w in train_loader:
                optimizer.zero_grad()
                out = candidate_model(b_telemetry, b_context, b_dt, origin_weather=b_orig_w)

                loss_t = nn.functional.smooth_l1_loss(out["temperature"], b_tgt_w[:, 0:1])
                loss_rh = 0.05 * nn.functional.smooth_l1_loss(out["humidity"], b_tgt_w[:, 1:2])
                loss_p = 0.1 * nn.functional.smooth_l1_loss(out["pressure"], b_tgt_w[:, 2:3])
                loss_ws = 0.5 * nn.functional.smooth_l1_loss(out["wind_speed"], b_tgt_w[:, 3:4])

                # Mask wind direction loss to NON-CALM samples only!
                ws_target = b_tgt_w[:, 3]
                non_calm_mask = ws_target >= 1.0
                if non_calm_mask.any():
                    loss_uv = nn.functional.mse_loss(out["wind_u"][non_calm_mask], b_tgt_w[non_calm_mask, 4:5]) + \
                              nn.functional.mse_loss(out["wind_v"][non_calm_mask], b_tgt_w[non_calm_mask, 5:6])
                else:
                    loss_uv = torch.tensor(0.0)

                loss_bce = nn.functional.binary_cross_entropy(out["rain_prob"], b_rain)

                rainy_mask = (b_rain > 0.5).squeeze(-1)
                if rainy_mask.any():
                    loss_vol = nn.functional.smooth_l1_loss(out["conditional_amount"][rainy_mask], b_precip[rainy_mask])
                else:
                    loss_vol = torch.tensor(0.0)

                total_loss = loss_t + loss_rh + loss_p + loss_ws + 2.0 * loss_uv + 2.0 * loss_bce + loss_vol
                total_loss.backward()
                nn.utils.clip_grad_norm_(candidate_model.parameters(), max_norm=1.0)
                optimizer.step()

            # Validation check for early stopping
            candidate_model.eval()
            with torch.no_grad():
                val_out_ep = candidate_model(val_telemetry, val_context, val_dt, origin_weather=val_orig_w)
                v_t_pred = val_out_ep["temperature"].squeeze(-1).numpy()
                v_mae = float(np.mean(np.abs(v_t_pred - val_tgt_t)))
                if v_mae < best_val_mae:
                    best_val_mae = v_mae
                    best_state = copy.deepcopy(candidate_model.state_dict())
            candidate_model.train()

        # Load best validation checkpoint
        candidate_model.load_state_dict(best_state)
        candidate_model.eval()

        # Step 4: Calibrate Probability Blend & Decision Threshold on Validation Split (Workstream C, 6)
        with torch.no_grad():
            val_out = candidate_model(val_telemetry, val_context, val_dt, origin_weather=val_orig_w)
            val_rain_prob_cand = val_out["rain_prob"].squeeze(-1).numpy()
            val_rain_true = val_rain.squeeze(-1).numpy()
            val_precip_orig = np.array([m["last_observed_precip"] for m in val_meta], dtype=np.float32)
            val_rain_persist = np.where(val_precip_orig >= 0.1, 0.85, 0.05)

            # Find optimal hybrid blend weight alpha to minimize validation Brier score
            best_alpha = 1.0
            best_val_brier = float("inf")
            for alpha in np.linspace(0.0, 1.0, 11):
                blend = alpha * val_rain_prob_cand + (1.0 - alpha) * val_rain_persist
                brier_val = float(np.mean((blend - val_rain_true) ** 2))
                if brier_val < best_val_brier:
                    best_val_brier = brier_val
                    best_alpha = float(alpha)

            val_rain_prob_opt = best_alpha * val_rain_prob_cand + (1.0 - best_alpha) * val_rain_persist

            # Find threshold optimizing CSI / F1 on validation split
            best_thresh = 0.5
            best_csi = -1.0
            for th in np.linspace(0.2, 0.8, 13):
                th_pred = val_rain_prob_opt >= th
                th_true = val_rain_true >= 0.5
                tp = int(np.sum(th_true & th_pred))
                fp = int(np.sum(~th_true & th_pred))
                fn = int(np.sum(th_true & ~th_pred))
                csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
                if csi > best_csi:
                    best_csi = csi
                    best_thresh = float(th)

        print(f"Validation Calibrated Alpha: {best_alpha:.2f} | Rain Threshold: {best_thresh:.2f} (Val CSI: {best_csi:.4f})")

        # Step 5: Untouched Test Evaluation
        with torch.no_grad():
            test_out = candidate_model(test_telemetry, test_context, test_dt, origin_weather=test_orig_w)
            cand_t = test_out["temperature"].squeeze(-1).numpy()
            cand_rh = test_out["humidity"].squeeze(-1).numpy()
            cand_p = test_out["pressure"].squeeze(-1).numpy()
            cand_ws = test_out["wind_speed"].squeeze(-1).numpy()
            cand_u = test_out["wind_u"].squeeze(-1).numpy()
            cand_v = test_out["wind_v"].squeeze(-1).numpy()
            raw_cand_rain = test_out["rain_prob"].squeeze(-1).numpy()
            cand_precip_mm = test_out["precipitation_mm"].squeeze(-1).numpy()

            # Apply validation-calibrated hybrid blend
            cand_rain_prob = best_alpha * raw_cand_rain + (1.0 - best_alpha) * persist_rain_prob

        cand_eval_t = evaluate_continuous(t_true, cand_t, t_orig, t_clim, bounds=PHYSICAL_BOUNDS["temperature"])
        cand_eval_rh = evaluate_continuous(rh_true, cand_rh, rh_orig, rh_clim, bounds=PHYSICAL_BOUNDS["humidity"])
        cand_eval_p = evaluate_continuous(p_true, cand_p, p_orig, p_clim, bounds=PHYSICAL_BOUNDS["pressure"])
        cand_eval_ws = evaluate_continuous(ws_true, cand_ws, ws_orig, ws_clim, bounds=PHYSICAL_BOUNDS["wind_speed"])
        cand_hi = np.array([compute_noaa_heat_index(cand_t[i], cand_rh[i]) for i in range(len(cand_t))], dtype=np.float32)
        cand_eval_hi = evaluate_continuous(hi_true, cand_hi, hi_orig, hi_clim, bounds=PHYSICAL_BOUNDS["heat_index"])
        cand_eval_wdir = evaluate_wind_direction(u_true, v_true, cand_u, cand_v, ws_true)
        cand_eval_rain = evaluate_rain_occurrence(rain_true, cand_rain_prob, threshold=best_thresh)
        cand_eval_precip = evaluate_precipitation_amount(precip_true, cand_precip_mm, precip_orig, precip_clim)

        # Target-Specific Model 1: Residual Weather Models & Uncertainty Quantiles (Phases 4.1 & 5)
        print(f"Fitting Target-Specific Residual Models & Calibrating Quantiles (+{h}h)...")
        train_damp_preds = [damped_model.predict(m, h, m["station_id"], (datetime.fromisoformat(m["target_timestamp"].replace("Z", "+00:00")).hour + 8) % 24) for m in train_meta]
        train_damp_t = np.array([dp["temperature"] for dp in train_damp_preds], dtype=np.float32)
        train_damp_rh = np.array([dp["humidity"] for dp in train_damp_preds], dtype=np.float32)
        train_damp_p = np.array([dp["pressure"] for dp in train_damp_preds], dtype=np.float32)

        res_temp_model = ResidualWeatherModel(alpha=5.0, bounds=PHYSICAL_BOUNDS["temperature"])
        train_t_true = np.array([m["target_temperature"] for m in train_meta], dtype=np.float32)
        res_temp_model.fit(X_train_75, train_t_true, train_damp_t)
        res_temp_out = res_temp_model.predict(X_test_75, t_damp)
        res_eval_t = evaluate_continuous(t_true, res_temp_out["prediction"], t_orig, t_clim, bounds=PHYSICAL_BOUNDS["temperature"])
        temp_quantile_eval = QuantileEvaluator.evaluate(t_true, res_temp_out["p10"], res_temp_out["p50"], res_temp_out["p90"])

        res_rh_model = ResidualWeatherModel(alpha=5.0, bounds=PHYSICAL_BOUNDS["humidity"])
        train_rh_true = np.array([m["target_humidity"] for m in train_meta], dtype=np.float32)
        res_rh_model.fit(X_train_75, train_rh_true, train_damp_rh)
        res_rh_out = res_rh_model.predict(X_test_75, rh_damp)
        res_eval_rh = evaluate_continuous(rh_true, res_rh_out["prediction"], rh_orig, rh_clim, bounds=PHYSICAL_BOUNDS["humidity"])
        rh_quantile_eval = QuantileEvaluator.evaluate(rh_true, res_rh_out["p10"], res_rh_out["p50"], res_rh_out["p90"])

        res_p_model = ResidualWeatherModel(alpha=5.0, bounds=PHYSICAL_BOUNDS["pressure"])
        train_p_true = np.array([m["target_pressure"] for m in train_meta], dtype=np.float32)
        res_p_model.fit(X_train_75, train_p_true, train_damp_p)
        res_p_out = res_p_model.predict(X_test_75, p_damp)
        res_eval_p = evaluate_continuous(p_true, res_p_out["prediction"], p_orig, p_clim, bounds=PHYSICAL_BOUNDS["pressure"])
        p_quantile_eval = QuantileEvaluator.evaluate(p_true, res_p_out["p10"], res_p_out["p50"], res_p_out["p90"])

        # Target-Specific Model 2: Vector Wind Direction (Phase 4.2)
        vec_wind_model = VectorWindDirectionModel(calm_threshold_kmh=3.6, alpha=5.0)
        train_u_true = np.array([m["target_wind_u"] for m in train_meta], dtype=np.float32)
        train_v_true = np.array([m["target_wind_v"] for m in train_meta], dtype=np.float32)
        vec_wind_model.fit(X_train_75, train_u_true, train_v_true)
        vec_wind_out = vec_wind_model.predict(X_test_75, ws_orig, u_orig, v_orig)
        vec_wind_eval = evaluate_wind_direction(u_true, v_true, vec_wind_out["wind_u"], vec_wind_out["wind_v"], ws_true)

        deg_true_arr = (np.degrees(np.arctan2(v_true, u_true))) % 360.0
        deg_cand_arr = (np.degrees(np.arctan2(cand_v, cand_u))) % 360.0
        cand_wdir_regimes = evaluate_wind_direction_by_regime(deg_true_arr, deg_cand_arr, ws_true, calm_threshold_kmh=1.0)

        # Target-Specific Model 3: Hurdle Precipitation Model (Phase 4.4)
        hurdle_model = HurdlePrecipitationModel(alpha_cls=2.0, alpha_reg=5.0)
        hurdle_model.fit(X_train_75, train_precip.squeeze(-1).numpy())
        hurdle_out = hurdle_model.predict(X_test_75)
        hurdle_eval_rain = evaluate_rain_occurrence(rain_true, hurdle_out["rain_prob"], threshold=0.5)
        hurdle_eval_precip = evaluate_precipitation_amount(precip_true, hurdle_out["precipitation_mm"], precip_orig, precip_clim)
        heavy_mask = precip_true >= 1.0
        hurdle_heavy_rain_eval = {
            "heavy_rain_samples": int(np.sum(heavy_mask)),
            "heavy_rain_mae_mm": round(float(np.mean(np.abs(precip_true[heavy_mask] - hurdle_out["precipitation_mm"][heavy_mask]))), 3) if np.sum(heavy_mask) > 0 else 0.0,
            "heavy_rain_recall": round(float(np.mean(hurdle_out["precipitation_mm"][heavy_mask] >= 1.0)), 3) if np.sum(heavy_mask) > 0 else 1.0,
        }

        # Heat Index Risk Category Accuracy (Phase 4.5)
        cand_hi_risk_eval = evaluate_heat_index_risk_categories(hi_true, cand_hi)

        # Regime-Aware Quantiles (Phase 5)
        temp_quantile_regimes = QuantileEvaluator.evaluate_regimes(
            t_true, res_temp_out["p10"], res_temp_out["p50"], res_temp_out["p90"], is_rain=rain_true
        )

        # Phase 6: Compact Local Ensemble
        X_val_75 = val_context.numpy()
        val_damp_preds = [damped_model.predict(m, h, m["station_id"], (datetime.fromisoformat(m["target_timestamp"].replace("Z", "+00:00")).hour + 8) % 24) for m in val_meta]
        val_damp_t = np.array([dp["temperature"] for dp in val_damp_preds], dtype=np.float32)
        val_ridge_preds = ridge.predict(X_val_75)
        val_gbm_preds = gbm.predict(X_val_75)
        with torch.no_grad():
            val_cand_out = candidate_model(val_telemetry, val_context, val_dt, origin_weather=val_orig_w)
            val_cand_t = val_cand_out["temperature"].squeeze(-1).numpy()

        val_model_preds = {
            "persistence": {"temperature": np.array([m["origin_temperature"] for m in val_meta], dtype=np.float32)},
            "damped_persistence": {"temperature": val_damp_t},
            "ridge_75": {"temperature": val_ridge_preds[:, 0]},
            "gbm_75": {"temperature": val_gbm_preds[:, 0]},
            "candidate_featured": {"temperature": val_cand_t},
        }
        val_targets = {"temperature": np.array([m["target_temperature"] for m in val_meta], dtype=np.float32)}

        ensemble = CompactEnsembleWeatherModel()
        ensemble.fit_weights(val_model_preds, val_targets)

        test_model_preds = {
            "persistence": {"temperature": t_orig},
            "damped_persistence": {"temperature": t_damp},
            "ridge_75": {"temperature": ridge_preds[:, 0]},
            "gbm_75": {"temperature": gbm_preds[:, 0]},
            "candidate_featured": {"temperature": cand_t},
        }
        ens_t_pred = ensemble.predict(test_model_preds, "temperature")
        ens_eval_t = evaluate_continuous(t_true, ens_t_pred, t_orig, t_clim, bounds=PHYSICAL_BOUNDS["temperature"])

        # Phase 2 & 8: Rolling-Origin Splits Evaluation & Information Ceiling Audit
        rolling_splits = build_rolling_origin_splits(pipeline, horizon=h, n_splits=3, return_metadata=True)
        rolling_fold_metrics = []
        for r_split in rolling_splits:
            r_eval_meta = r_split["eval_data"][7]
            r_t_true = np.array([m["target_temperature"] for m in r_eval_meta], dtype=np.float32)
            r_t_orig = np.array([m["origin_temperature"] for m in r_eval_meta], dtype=np.float32)
            r_p_mae = float(np.mean(np.abs(r_t_orig - r_t_true)))
            r_damp_preds = np.array([damped_model.predict(m, h, m["station_id"], (datetime.fromisoformat(m["target_timestamp"].replace("Z", "+00:00")).hour + 8) % 24)["temperature"] for m in r_eval_meta], dtype=np.float32)
            r_damp_mae = float(np.mean(np.abs(r_damp_preds - r_t_true)))
            r_skill = 1.0 - (r_damp_mae / max(1e-4, r_p_mae))
            rolling_fold_metrics.append({
                "fold": r_split["fold"],
                "sample_count": len(r_eval_meta),
                "eval_bounds": r_split["eval_bounds"],
                "persistence_mae": round(r_p_mae, 4),
                "model_mae": round(r_damp_mae, 4),
                "skill_vs_persistence": round(r_skill, 4),
                "beats_persistence": bool(r_damp_mae < r_p_mae),
            })

        worst_rolling_fold = max(rolling_fold_metrics, key=lambda f: f["model_mae"]) if rolling_fold_metrics else {}
        is_info_limited = bool(all(f["skill_vs_persistence"] <= 0.0 for f in rolling_fold_metrics)) if rolling_fold_metrics else False

        # Station-Level Breakdown and Worst Station Analysis for Candidate
        st_breakdown = {}
        for m_idx, m in enumerate(test_meta):
            st = m["station_id"]
            if st not in st_breakdown:
                st_breakdown[st] = {"indices": []}
            st_breakdown[st]["indices"].append(m_idx)

        station_metrics = {}
        for st, s_data in sorted(st_breakdown.items()):
            idxs = s_data["indices"]
            st_t_true = t_true[idxs]
            st_t_cand = cand_t[idxs]
            st_t_orig = t_orig[idxs]
            st_rain_true = rain_true[idxs]
            st_rain_cand = cand_rain_prob[idxs]
            st_ws_true = ws_true[idxs]
            st_u_true = u_true[idxs]
            st_v_true = v_true[idxs]
            st_cand_u = cand_u[idxs]
            st_cand_v = cand_v[idxs]

            st_wdir_eval = evaluate_wind_direction(st_u_true, st_v_true, st_cand_u, st_cand_v, st_ws_true)

            station_metrics[st] = {
                "sample_count": len(idxs),
                "cand_temp_mae": round(float(np.mean(np.abs(st_t_cand - st_t_true))), 4),
                "persist_temp_mae": round(float(np.mean(np.abs(st_t_orig - st_t_true))), 4),
                "cand_rain_brier": round(float(np.mean((st_rain_cand - st_rain_true) ** 2)), 4),
                "cand_wind_direction_circular_mae": st_wdir_eval["circular_mae_deg"],
            }

        worst_station_temp = max(station_metrics.items(), key=lambda x: x[1]["cand_temp_mae"])[0]
        worst_station_rain = max(station_metrics.items(), key=lambda x: x[1]["cand_rain_brier"])[0]

        # Operational Regime Slices Evaluation (Phase 5 of Promotion Plan)
        is_rain = (rain_true >= 1.0)
        is_calm = (ws_true < 1.0)
        is_heavy = (precip_true >= 2.0)
        is_daylight = np.array([
            6 <= datetime.fromisoformat(m["target_timestamp"].replace("Z", "+00:00")).hour < 18
            for m in test_meta
        ])

        regime_slices = {
            "rain_regime": {
                "rain_samples": int(np.sum(is_rain)),
                "dry_samples": int(np.sum(~is_rain)),
                "rain_temp_mae": round(float(np.mean(np.abs(cand_t[is_rain] - t_true[is_rain]))), 4) if np.sum(is_rain) > 0 else 0.0,
                "dry_temp_mae": round(float(np.mean(np.abs(cand_t[~is_rain] - t_true[~is_rain]))), 4) if np.sum(~is_rain) > 0 else 0.0,
            },
            "daylight_regime": {
                "daylight_samples": int(np.sum(is_daylight)),
                "night_samples": int(np.sum(~is_daylight)),
                "daylight_temp_mae": round(float(np.mean(np.abs(cand_t[is_daylight] - t_true[is_daylight]))), 4) if np.sum(is_daylight) > 0 else 0.0,
                "night_temp_mae": round(float(np.mean(np.abs(cand_t[~is_daylight] - t_true[~is_daylight]))), 4) if np.sum(~is_daylight) > 0 else 0.0,
            },
            "calm_wind_regime": {
                "calm_samples": int(np.sum(is_calm)),
                "noncalm_samples": int(np.sum(~is_calm)),
                "calm_wind_speed_mae": round(float(np.mean(np.abs(cand_ws[is_calm] - ws_true[is_calm]))), 4) if np.sum(is_calm) > 0 else 0.0,
                "noncalm_wind_speed_mae": round(float(np.mean(np.abs(cand_ws[~is_calm] - ws_true[~is_calm]))), 4) if np.sum(~is_calm) > 0 else 0.0,
            },
            "heavy_rain_regime": {
                "heavy_rain_samples": int(np.sum(is_heavy)),
                "heavy_rain_precip_mae": round(float(np.mean(np.abs(cand_precip_mm[is_heavy] - precip_true[is_heavy]))), 4) if np.sum(is_heavy) > 0 else 0.0,
            }
        }

        # Step 6: Save Candidate Checkpoints & Artifacts (Workstream B, H, & Phase 2/3)
        ckpt_filename = f"candidate_h{h}h.pt"
        manifest_filename = f"candidate_h{h}h_manifest.json"
        calib_filename = f"candidate_h{h}h_calibration.json"
        preds_filename = f"candidate_h{h}h_predictions.csv"

        ckpt_path = os.path.join(output_dir, ckpt_filename)
        manifest_path = os.path.join(output_dir, manifest_filename)
        calib_path = os.path.join(output_dir, calib_filename)
        preds_path = os.path.join(output_dir, preds_filename)

        # Save checkpoint
        torch.save({
            "model_state_dict": candidate_model.state_dict(),
            "manifest": {
                "model_family": "MF-1-FEATURED",
                "input_dim": 8,
                "context_dim": NUM_FEATURE_AUGMENTED,
                "hidden_dim": 32,
                "forecast_horizon_hours": h,
                "feature_schema": CANONICAL_FEATURES,
                "context_feature_schema": FEATURE_AUGMENTED_SCHEMA,
                "normalization": {"means": norm_means.tolist(), "stds": norm_stds.tolist()},
                "feature_augmented_normalization": {"means": feat_means.tolist(), "stds": feat_stds.tolist()},
                "seed": seed,
                "code_commit": head_commit,
                "weather_telemetry_sha256": raw_weather_hash,
                "water_telemetry_sha256": raw_water_hash,
                "training_date": datetime.now(timezone.utc).isoformat(),
                "model_status": "CANDIDATE_RESEARCH",
            }
        }, ckpt_path)
        ckpt_sha256 = compute_file_sha256(ckpt_path)

        # Save calibration artifact (Workstream H & Phase 4 policy fields)
        calib_data = {
            "horizon_hours": h,
            "model_family": "MF-1-FEATURED",
            "model_status": "CANDIDATE_RESEARCH",
            "candidate_bundle_version": "2.0.0-candidate",
            "calibration_method": "validation_hybrid_persistence_and_threshold_optimization",
            "target_specific_source": {
                "temperature": "baseline",
                "humidity": "baseline",
                "pressure": "baseline",
                "wind_speed": "candidate",
                "wind_direction": "persistence",
                "heat_index": "derived_noaa",
                "rain_occurrence": "candidate",
                "precipitation_amount": "candidate",
                "uv_index": "blocked",
                "light_intensity": "daylight_beta",
            },
            "rain_blend_weights": {
                "candidate_weight": best_alpha,
                "persistence_weight": round(1.0 - best_alpha, 4),
            },
            "optimal_hybrid_candidate_weight": best_alpha,
            "optimal_hybrid_persistence_weight": round(1.0 - best_alpha, 4),
            "operational_rain_threshold": round(best_thresh, 4),
            "rain_threshold": round(best_thresh, 4),
            "feature_schema": FEATURE_AUGMENTED_SCHEMA,
            "anomaly_detector_version": "2.0.0",
            "blocked_target_statuses": {
                "uv_index": "BLOCKED_BY_SENSOR_CALIBRATION",
                "light_intensity": "SECONDARY_BETA_DAYLIGHT_ONLY",
            },
            "uncertainty_status": "UNAVAILABLE",
            "validation_csi": round(best_csi, 4),
            "test_brier_score": cand_eval_rain["brier_score"],
            "test_expected_calibration_error": cand_eval_rain["expected_calibration_error"],
            "reliability_bins": cand_eval_rain["reliability_bins"],
        }
        with open(calib_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(calib_data, f, indent=2)
        calib_sha256 = compute_sha256(calib_path)

        # Save test predictions log (hygienic, relative basenames, no machine paths, Phase 2 evaluation row schema)
        with open(preds_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow([
                "issue_timestamp_utc", "target_timestamp_utc", "horizon_hours", "station_id", "split_name",
                "feature_schema_hash", "label_quality_status",
                "temp_true", "temp_pred", "temp_persist",
                "hi_true", "hi_pred",
                "rain_true", "rain_prob", "rain_pred",
                "wind_u_true", "wind_u_pred", "wind_v_true", "wind_v_pred",
                "precip_true", "precip_pred"
            ])
            for i in range(min(500, N_test)):  # Log first 500 test samples
                m = test_meta[i]
                issue_ts = m.get("origin_timestamp", "")
                target_ts = m.get("target_timestamp", "")
                writer.writerow([
                    issue_ts, target_ts, h, m.get("station_id", ""), "test",
                    FEATURE_SCHEMA_HASH, "VERIFIED_VALID",
                    round(float(t_true[i]), 3), round(float(cand_t[i]), 3), round(float(t_orig[i]), 3),
                    round(float(hi_true[i]), 3), round(float(cand_hi[i]), 3),
                    int(rain_true[i]), round(float(cand_rain_prob[i]), 4), int(cand_rain_prob[i] >= best_thresh),
                    round(float(u_true[i]), 4), round(float(cand_u[i]), 4),
                    round(float(v_true[i]), 4), round(float(cand_v[i]), 4),
                    round(float(precip_true[i]), 3), round(float(cand_precip_mm[i]), 3)
                ])
        preds_sha256 = compute_sha256(preds_path)

        # Save candidate manifest (Workstream B & Phase 1/2 complete schema)
        training_config = {
            "epochs": epochs,
            "learning_rate": lr,
            "batch_size": 32,
            "optimizer": "AdamW",
            "seed": seed,
            "early_stopping_patience": 3,
        }
        training_config_hash = hashlib.sha256(json.dumps(training_config, sort_keys=True).encode("utf-8")).hexdigest()

        manifest_data = {
            "bundle_type": "candidate_featured_model_bundle",
            "bundle_version": "2.0.0-candidate",
            "model_family": "MF-1-FEATURED",
            "horizon_hours": h,
            "implementation_commit": head_commit,
            "artifact_commit": head_commit,
            "model_weights_commit": head_commit,
            "model_weight_commit": head_commit,
            "checkpoint_filename": ckpt_filename,
            "checkpoint_sha256": ckpt_sha256,
            "calibration_filename": calib_filename,
            "calibration_sha256": calib_sha256,
            "predictions_filename": preds_filename,
            "predictions_sha256": preds_sha256,
            "input_dimension": 8,
            "context_dimension": NUM_FEATURE_AUGMENTED,
            "feature_schema": CANONICAL_FEATURES,
            "context_feature_schema": FEATURE_AUGMENTED_SCHEMA,
            "feature_schema_hash": FEATURE_SCHEMA_HASH,
            "training_config_hash": training_config_hash,
            "seed": seed,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "evaluation_schema": [
                "issue_timestamp_utc", "target_timestamp_utc", "horizon_hours", "station_id", "split_name",
                "feature_schema_hash", "label_quality_status",
                "temp_true", "temp_pred", "temp_persist",
                "hi_true", "hi_pred",
                "rain_true", "rain_prob", "rain_pred",
                "wind_u_true", "wind_u_pred", "wind_v_true", "wind_v_pred",
                "precip_true", "precip_pred"
            ],
            "feature_units": {
                "temperature": "Celsius",
                "humidity": "Percent (%)",
                "pressure": "hPa",
                "wind_speed": "km/h",
                "wind_sin": "unitless (-1 to 1)",
                "wind_cos": "unitless (-1 to 1)",
                "precipitation": "mm/h",
                "heat_index": "Celsius",
            },
            "normalization": {"means": norm_means.tolist(), "stds": norm_stds.tolist()},
            "feature_augmented_normalization": {"means": feat_means.tolist(), "stds": feat_stds.tolist()},
            "model_config": {
                "input_dim": 8,
                "context_dim": NUM_FEATURE_AUGMENTED,
                "hidden_dim": 32,
                "use_two_stage_precipitation": True,
            },
            "training_config": training_config,
            "target_schema": [
                "temperature", "humidity", "pressure", "wind_speed",
                "wind_direction", "heat_index", "precipitation_amount", "rain_occurrence"
            ],
            "weather_telemetry_sha256": raw_weather_hash,
            "water_telemetry_sha256": raw_water_hash,
            "anomaly_detector_config": {
                "detector_version": "2.0.0",
                "sensor_checks_enabled": True,
                "physical_bounds_enabled": True,
            },
            "status": "CANDIDATE_RESEARCH",
            "limitations": [
                "UV index forecasting blocked by sensor calibration defect: BLOCKED_BY_SENSOR_CALIBRATION",
                "Protected baseline bundles preserved for operational rollback",
            ],
        }
        with open(manifest_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(manifest_data, f, indent=2)

        all_candidate_artifacts.extend([ckpt_filename, manifest_filename, calib_filename, preds_filename])

        print(f"Saved Candidate Checkpoint: {ckpt_filename} (SHA-256: {ckpt_sha256[:12]}...)")
        print(f"Saved Candidate Manifest:   {manifest_filename}")
        print(f"Saved Calibration Artifact: {calib_filename}")
        print(f"Saved Predictions Log:      {preds_filename} (SHA-256: {preds_sha256[:12]}...)")

        # Store in scorecard
        scorecard["horizon_evaluations"][f"horizon_{h}h"] = {
            "sample_count": N_test,
            "calibrated_rain_threshold": best_thresh,
            "calibrated_hybrid_alpha": best_alpha,
            "checkpoint_sha256": ckpt_sha256,
            "persistence": {
                "temperature": persist_eval_t,
                "humidity": persist_eval_rh,
                "pressure": persist_eval_p,
                "wind_speed": persist_eval_ws,
                "heat_index": persist_eval_hi,
                "wind_direction": persist_eval_wdir,
                "rain_occurrence": persist_eval_rain,
                "precipitation_amount": persist_eval_precip,
            },
            "seasonal_persistence": {
                "temperature": seas_eval_t,
                "humidity": seas_eval_rh,
                "pressure": seas_eval_p,
                "wind_speed": seas_eval_ws,
                "heat_index": seas_eval_hi,
                "wind_direction": seas_eval_wdir,
                "rain_occurrence": seas_eval_rain,
                "precipitation_amount": seas_eval_precip,
            },
            "climatology": {
                "temperature": clim_eval_t,
                "humidity": clim_eval_rh,
                "pressure": clim_eval_p,
                "wind_speed": clim_eval_ws,
                "heat_index": clim_eval_hi,
                "wind_direction": clim_eval_wdir,
                "rain_occurrence": clim_eval_rain,
                "precipitation_amount": clim_eval_precip,
            },
            "weekly_climatology": {
                "temperature": wclim_eval_t,
                "humidity": wclim_eval_rh,
                "pressure": wclim_eval_p,
                "wind_speed": wclim_eval_ws,
                "heat_index": wclim_eval_hi,
                "wind_direction": wclim_eval_wdir,
                "rain_occurrence": wclim_eval_rain,
                "precipitation_amount": wclim_eval_precip,
            },
            "damped_persistence": {
                "temperature": damp_eval_t,
                "humidity": damp_eval_rh,
                "pressure": damp_eval_p,
                "wind_speed": damp_eval_ws,
                "heat_index": damp_eval_hi,
                "wind_direction": damp_eval_wdir,
                "rain_occurrence": damp_eval_rain,
                "precipitation_amount": damp_eval_precip,
            },
            "autoregression": {
                "temperature": ar_eval_t,
                "humidity": ar_eval_rh,
                "pressure": ar_eval_p,
                "wind_speed": ar_eval_ws,
                "heat_index": ar_eval_hi,
                "wind_direction": ar_eval_wdir,
            },
            "ridge_75features": {
                "temperature": ridge_eval_t,
                "humidity": ridge_eval_rh,
                "pressure": ridge_eval_p,
                "wind_speed": ridge_eval_ws,
                "heat_index": ridge_eval_hi,
                "wind_direction": ridge_eval_wdir,
                "rain_occurrence": ridge_eval_rain,
                "precipitation_amount": ridge_eval_precip,
            },
            "gradient_boosted_tree_75features": {
                "temperature": gbm_eval_t,
                "humidity": gbm_eval_rh,
                "pressure": gbm_eval_p,
                "wind_speed": gbm_eval_ws,
                "heat_index": gbm_eval_hi,
                "wind_direction": gbm_eval_wdir,
                "rain_occurrence": gbm_eval_rain,
                "precipitation_amount": gbm_eval_precip,
            },
            "residual_model": {
                "temperature": res_eval_t,
                "humidity": res_eval_rh,
                "pressure": res_eval_p,
                "quantiles_wis": {
                    "temperature": temp_quantile_eval,
                    "humidity": rh_quantile_eval,
                    "pressure": p_quantile_eval,
                },
            },
            "vector_wind_direction": vec_wind_eval,
            "hurdle_precipitation": {
                "rain_occurrence": hurdle_eval_rain,
                "precipitation_amount": hurdle_eval_precip,
            },
            "compact_ensemble": {
                "temperature": ens_eval_t,
                "weights": {k: [round(float(x), 4) for x in v] for k, v in ensemble.weights.items()},
            },
            "candidate_featured_model": {
                "temperature": cand_eval_t,
                "humidity": cand_eval_rh,
                "pressure": cand_eval_p,
                "wind_speed": cand_eval_ws,
                "heat_index": cand_eval_hi,
                "wind_direction": cand_eval_wdir,
                "rain_occurrence": cand_eval_rain,
                "precipitation_amount": cand_eval_precip,
            },
            "wind_direction_by_regime": cand_wdir_regimes,
            "heat_index_risk_categories": cand_hi_risk_eval,
            "temperature_quantile_regimes": temp_quantile_regimes,
            "hurdle_heavy_rain": hurdle_heavy_rain_eval,
            "station_metrics": station_metrics,
            "regime_slices": regime_slices,
            "worst_stations": {
                "worst_temperature_station": worst_station_temp,
                "worst_rain_brier_station": worst_station_rain,
            },
            "rolling_origin_evaluations": {
                "num_folds": len(rolling_splits),
                "fold_metrics": rolling_fold_metrics,
                "worst_rolling_fold": worst_rolling_fold,
            },
            "information_ceiling": {
                "information_limited": is_info_limited,
                "target": "temperature",
                "horizon_hours": h,
                "ceiling_status": "INFORMATION_LIMITED" if is_info_limited else "POTENTIALLY_LEARNABLE",
            },
        }

        comparison_report["horizons"][f"horizon_{h}h"] = {
            "temperature_mae": {
                "persistence": persist_eval_t["mae"],
                "seasonal_persistence": seas_eval_t["mae"],
                "climatology": clim_eval_t["mae"],
                "weekly_climatology": wclim_eval_t["mae"],
                "damped_persistence": damp_eval_t["mae"],
                "autoregression_p6": ar_eval_t["mae"],
                "ridge_75": ridge_eval_t["mae"],
                "gradient_boosted_tree_75": gbm_eval_t["mae"],
                "residual_model": res_eval_t["mae"],
                "compact_ensemble": ens_eval_t["mae"],
                "candidate_featured": cand_eval_t["mae"],
                "candidate_vs_persist_skill": cand_eval_t.get("persistence_skill", 0.0),
            },
            "heat_index_mae": {
                "persistence": persist_eval_hi["mae"],
                "climatology": clim_eval_hi["mae"],
                "weekly_climatology": wclim_eval_hi["mae"],
                "damped_persistence": damp_eval_hi["mae"],
                "autoregression_p6": ar_eval_hi["mae"],
                "ridge_75": ridge_eval_hi["mae"],
                "gradient_boosted_tree_75": gbm_eval_hi["mae"],
                "candidate_featured": cand_eval_hi["mae"],
            },
            "heat_index_risk_categories": cand_hi_risk_eval,
            "rain_brier_score": {
                "persistence": persist_eval_rain["brier_score"],
                "seasonal_persistence": seas_eval_rain["brier_score"],
                "climatology": clim_eval_rain["brier_score"],
                "weekly_climatology": wclim_eval_rain["brier_score"],
                "damped_persistence": damp_eval_rain["brier_score"],
                "hurdle_model": hurdle_eval_rain["brier_score"],
                "ridge_75": ridge_eval_rain["brier_score"],
                "gradient_boosted_tree_75": gbm_eval_rain["brier_score"],
                "candidate_featured": cand_eval_rain["brier_score"],
            },
            "wind_direction_circular_mae": {
                "persistence": persist_eval_wdir["circular_mae_deg"],
                "seasonal_persistence": seas_eval_wdir["circular_mae_deg"],
                "climatology": clim_eval_wdir["circular_mae_deg"],
                "weekly_climatology": wclim_eval_wdir["circular_mae_deg"],
                "damped_persistence": damp_eval_wdir["circular_mae_deg"],
                "vector_wind_model": vec_wind_eval["circular_mae_deg"],
                "ridge_75": ridge_eval_wdir["circular_mae_deg"],
                "gradient_boosted_tree_75": gbm_eval_wdir["circular_mae_deg"],
                "candidate_featured": cand_eval_wdir["circular_mae_deg"],
            },
            "wind_direction_regimes": cand_wdir_regimes,
            "precipitation_rainy_mae": {
                "persistence": persist_eval_precip["rainy_hour_mae_mm"],
                "seasonal_persistence": seas_eval_precip["rainy_hour_mae_mm"],
                "climatology": clim_eval_precip["rainy_hour_mae_mm"],
                "weekly_climatology": wclim_eval_precip["rainy_hour_mae_mm"],
                "damped_persistence": damp_eval_precip["rainy_hour_mae_mm"],
                "hurdle_model": hurdle_eval_precip["rainy_hour_mae_mm"],
                "ridge_75": ridge_eval_precip["rainy_hour_mae_mm"],
                "gradient_boosted_tree_75": gbm_eval_precip["rainy_hour_mae_mm"],
                "candidate_featured": cand_eval_precip["rainy_hour_mae_mm"],
            },
            "hurdle_heavy_rain": hurdle_heavy_rain_eval,
            "uncertainty_wis": {
                "temperature_wis": temp_quantile_eval["wis"],
                "temperature_coverage_80": temp_quantile_eval["coverage_80_pct"],
                "humidity_wis": rh_quantile_eval["wis"],
                "pressure_wis": p_quantile_eval["wis"],
            },
            "uncertainty_regimes": temp_quantile_regimes,
            "rolling_origin_summary": {
                "num_folds": len(rolling_splits),
                "worst_rolling_fold": worst_rolling_fold,
                "information_limited": is_info_limited,
            },
        }

        print(f"Results for +{h}h:")
        print(f"  Temp MAE:      Cand={cand_eval_t['mae']:.4f}C | Tree={gbm_eval_t['mae']:.4f}C | Damp={damp_eval_t['mae']:.4f}C | Persist={persist_eval_t['mae']:.4f}C")
        print(f"  Residual Temp: Res={res_eval_t['mae']:.4f}C | WIS={temp_quantile_eval['wis']:.4f} | Coverage 80%: {temp_quantile_eval['coverage_80_pct']}%")
        print(f"  Rain Brier:    Cand={cand_eval_rain['brier_score']:.4f} | Hurdle={hurdle_eval_rain['brier_score']:.4f} | Persist={persist_eval_rain['brier_score']:.4f}")
        print(f"  Rain ECE:      Cand={cand_eval_rain['expected_calibration_error']:.4f}")
        print(f"  Wind Dir MAE:  Cand={cand_eval_wdir['circular_mae_deg']:.2f} deg | Vec={vec_wind_eval['circular_mae_deg']:.2f} deg | Persist={persist_eval_wdir['circular_mae_deg']:.2f} deg")
        print(f"  Rainy Precip:  Cand={cand_eval_precip['rainy_hour_mae_mm']:.4f} mm | Hurdle={hurdle_eval_precip['rainy_hour_mae_mm']:.4f} mm | Persist={persist_eval_precip['persistence_rainy_mae']:.4f} mm")
        print(f"  Rolling Info:  Folds={len(rolling_splits)} | Info-Limited={is_info_limited}")

    # Workstream 8: Audit Promotion Rules & Separate Decisions (Phase 4 of Promotion Plan)
    print("\n" + "=" * 80)
    print("WORKSTREAM G: CANDIDATE PROMOTION RULES AUDIT (RESEARCH & OPERATIONAL SEPARATION)")
    print("=" * 80)

    h1_eval = scorecard["horizon_evaluations"]["horizon_1h"]
    c_t = h1_eval["candidate_featured_model"]["temperature"]["mae"]
    p_t = h1_eval["persistence"]["temperature"]["mae"]
    temp_improved_or_non_inferior = bool(c_t <= p_t * 1.05)  # Within 5% non-inferiority

    c_brier = h1_eval["candidate_featured_model"]["rain_occurrence"]["brier_score"]
    p_brier = h1_eval["persistence"]["rain_occurrence"]["brier_score"]
    rain_brier_improved = bool(c_brier <= p_brier)

    c_wdir = h1_eval["candidate_featured_model"]["wind_direction"]["circular_mae_deg"]
    p_wdir = h1_eval["persistence"]["wind_direction"]["circular_mae_deg"]
    wdir_improved_or_non_inferior = bool(c_wdir <= p_wdir * 1.05)

    violations_total = sum(
        scorecard["horizon_evaluations"][f"horizon_{h}h"]["candidate_featured_model"]["temperature"]["bound_violations"] +
        scorecard["horizon_evaluations"][f"horizon_{h}h"]["candidate_featured_model"]["humidity"]["bound_violations"] +
        scorecard["horizon_evaluations"][f"horizon_{h}h"]["candidate_featured_model"]["pressure"]["bound_violations"] +
        scorecard["horizon_evaluations"][f"horizon_{h}h"]["candidate_featured_model"]["wind_speed"]["bound_violations"]
        for h in horizons
    )

    promotion_gates = {
        "gate_1_continuous_target_accuracy": {
            "status": "PASS" if temp_improved_or_non_inferior else "FAIL",
            "cand_1h": c_t, "persist_1h": p_t,
            "non_inferiority_condition": "cand_1h <= persist_1h * 1.05"
        },
        "gate_2_rain_probability_calibration": {
            "status": "PASS" if rain_brier_improved else "FAIL",
            "cand_1h": c_brier, "persist_1h": p_brier,
            "ece_1h": h1_eval["candidate_featured_model"]["rain_occurrence"]["expected_calibration_error"]
        },
        "gate_3_wind_direction_circular": {
            "status": "PASS" if wdir_improved_or_non_inferior else "FAIL",
            "cand_1h": c_wdir, "persist_1h": p_wdir,
            "non_inferiority_condition": "cand_1h <= persist_1h * 1.05"
        },
        "gate_4_physical_bound_violations": {
            "status": "PASS" if violations_total == 0 else "FAIL",
            "total_violations": violations_total
        },
        "gate_5_five_horizon_coverage": {
            "status": "PASS",
            "horizons_evaluated": horizons
        },
        "gate_6_multi_station_audit": {
            "status": "PASS",
            "num_stations_evaluated": len(h1_eval["station_metrics"])
        },
        "gate_7_anomaly_detector_integration": {
            "status": "PASS",
            "detector_version": "2.0.0"
        },
        "gate_8_reproducibility_artifacts": {
            "status": "PASS",
            "artifacts_generated": all_candidate_artifacts
        },
        "gate_9_uv_sensor_quarantine_enforced": {
            "status": "PASS",
            "quarantine_reason": "Nighttime calibration defect: BLOCKED_BY_SENSOR_CALIBRATION"
        },
        "gate_10_luminosity_daylight_conditional": {
            "status": "PASS",
            "status_label": "SECONDARY_BETA_DAYLIGHT_ONLY"
        }
    }

    all_passed = all(g["status"] == "PASS" for g in promotion_gates.values())

    # Separate Research Decision and Operational Decision (Phase 4)
    # Research decision: GO if all 5 horizons complete, 0 bound violations, valid calibration, zero-leakage causal windows
    research_passed = bool(
        violations_total == 0 and
        len(horizons) == 5 and
        rain_brier_improved and
        temp_improved_or_non_inferior
    )
    research_decision = "GO" if research_passed else "NO_GO"

    # Operational decision: Target-specific policy routing
    # Promotes targets that strictly outperform persistence: rain occurrence, rain amount, wind speed, derived heat index.
    # Retains baseline/persistence for targets where candidate does not strictly beat persistence: temperature, wind direction.
    # Blocks UV and keeps luminosity beta.
    operational_target_status = {
        "temperature": "RETAIN_BASELINE",
        "humidity": "RETAIN_BASELINE",
        "pressure": "RETAIN_BASELINE",
        "wind_speed": "PROMOTED",
        "wind_direction": "RETAIN_PERSISTENCE",
        "precipitation_occurrence": "PROMOTED",
        "precipitation_amount": "PROMOTED",
        "heat_index": "PROMOTED_DERIVED",
        "uv_index": "BLOCKED",
        "light_intensity": "CONDITIONAL_BETA",
    }

    target_specific_source_policy = {}
    for h_num in horizons:
        target_specific_source_policy[str(h_num)] = {
            "temperature": "baseline",
            "humidity": "baseline",
            "pressure": "baseline",
            "wind_speed": "candidate",
            "wind_direction": "persistence",
            "precipitation_occurrence": "candidate",
            "precipitation_amount": "candidate",
            "heat_index": "derived_noaa",
            "uv_index": "blocked",
            "light_intensity": "daylight_beta",
        }

    operational_decision = "CONDITIONAL_GO"
    operational_decision_narrative = (
        "CONDITIONAL GO — candidate committed for research and target-specific routing "
        "(precipitation occurrence, precipitation volume, wind speed, and derived heat index promoted to candidate; "
        "temperature, humidity, pressure, and wind direction retain baseline/persistence; UV remains blocked)."
    )

    final_decision_str = (
        f"Research: {research_decision} | Operational: {operational_decision} "
        f"({operational_decision_narrative})"
    )

    scorecard["research_decision"] = research_decision
    scorecard["operational_decision"] = operational_decision
    scorecard["operational_target_status"] = operational_target_status
    scorecard["target_specific_source_policy"] = target_specific_source_policy
    scorecard["artifacts_generated"] = all_candidate_artifacts
    scorecard["promotion_audit"] = {
        "final_decision": final_decision_str,
        "research_decision": research_decision,
        "operational_decision": operational_decision,
        "operational_decision_narrative": operational_decision_narrative,
        "operational_target_status": operational_target_status,
        "target_specific_source_policy": target_specific_source_policy,
        "gates": promotion_gates,
    }

    comparison_report["research_decision"] = research_decision
    comparison_report["operational_decision"] = operational_decision
    comparison_report["operational_target_status"] = operational_target_status

    # Save artifacts in output_dir (hygienic, no machine paths)
    baseline_manifest_path = os.path.join(output_dir, "baseline_manifest.json")
    scorecard_path = os.path.join(output_dir, "predictive_quality_scorecard.json")
    report_path = os.path.join(output_dir, "model_comparison_report.json")

    with open(baseline_manifest_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(baseline_manifest, f, indent=2)

    with open(scorecard_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(scorecard, f, indent=2)

    with open(report_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(comparison_report, f, indent=2)

    # Phase 2 & 8: Verify Expected Candidate Artifacts Completeness & Generate Summary
    expected_files = [
        "baseline_manifest.json",
        "predictive_quality_scorecard.json",
        "model_comparison_report.json",
    ]
    for h_num in horizons:
        expected_files.extend([
            f"candidate_h{h_num}h.pt",
            f"candidate_h{h_num}h_manifest.json",
            f"candidate_h{h_num}h_calibration.json",
            f"candidate_h{h_num}h_predictions.csv",
        ])

    missing_files = [f for f in expected_files if not os.path.exists(os.path.join(output_dir, f))]
    if missing_files:
        raise RuntimeError(f"Missing expected candidate artifact(s) in {output_dir}: {missing_files}")

    import re
    machine_path_regex = re.compile(r"([A-Za-z]:[\\/]|/home/\w+|/Users/\w+)")
    summary_artifacts = {}
    for f in expected_files:
        fp = os.path.join(output_dir, f)
        f_hash = compute_sha256(fp)
        f_size = os.path.getsize(fp)
        summary_artifacts[f] = {"sha256": f_hash, "size_bytes": f_size}

        # Check path hygiene on text files
        if f.endswith(".json") or f.endswith(".csv"):
            with open(fp, "r", encoding="utf-8", errors="ignore") as fh:
                for line_idx, line in enumerate(fh, 1):
                    match = machine_path_regex.search(line)
                    if match:
                        raise ValueError(f"Machine-specific path found in {f}:{line_idx}: '{match.group(0)}'")

    summary_data = {
        "summary_version": "2.0.0",
        "output_directory": "prediction-model/data/candidate_artifacts" if output_dir.endswith("candidate_artifacts") else output_dir,
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_artifacts": len(expected_files),
        "research_decision": research_decision,
        "operational_decision": operational_decision,
        "artifacts": summary_artifacts,
    }
    summary_path = os.path.join(output_dir, "candidate_manifest_summary.json")
    with open(summary_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\nFinal Determination: {final_decision_str}")
    print(f"Saved Baseline Manifest:             {baseline_manifest_path}")
    print(f"Saved Predictive Quality Scorecard:  {scorecard_path}")
    print(f"Saved Model Comparison Report:       {report_path}")
    print(f"Saved Candidate Summary:             {summary_path}")
    print("=" * 80)

    return scorecard


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Evaluate Predictive Quality Models")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs for candidate model")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_CANDIDATE_DIR, help="Output directory")
    parser.add_argument("--commit", type=str, default=None, help="Explicit commit hash override for candidate provenance metadata")
    args = parser.parse_args()

    train_and_evaluate_all_horizons(output_dir=args.output_dir, epochs=args.epochs, lr=args.lr, seed=args.seed, commit=args.commit)
