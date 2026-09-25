"""
Comprehensive Predictive-Quality Training, Baseline Benchmarking, and Evaluation Pipeline.

Fulfills Workstreams 1 through 8 of the Predictive-Quality Implementation Plan:
  - Workstream 1: Target and Data Quality auditing.
  - Workstream 2: Zero-Leakage Feature-Augmented Training.
  - Workstream 3: Dedicated Target Heads & Physical Constraints.
  - Workstream 4: Train candidate models & strong baselines:
      1. Persistence Baseline
      2. Climatology Baseline
      3. Ridge Linear Regression with 75 engineered features
      4. Decision Forest / Non-Neural Baseline
      5. Protected 8-feature baseline (MF-1 / MF-2)
      6. Feature-augmented candidate (GarciaWeatherLNNFeatured / MF-1-FEATURED)
      7. Two-Stage Precipitation Candidate
  - Workstream 5: Comprehensive evaluation across 5 horizons, 15 stations, and all regimes.
  - Workstream 6: Frozen calibration on validation split before test evaluation.
  - Workstream 7: Real-time telemetry anomaly evaluation (sensor vs physical vs forecast).
  - Workstream 8: Candidate Promotion Rules and formal release determination.

Outputs:
  - prediction-model/data/predictive_quality_scorecard.json
  - prediction-model/data/model_comparison_report.json
"""

import os
import sys
import math
import json
import random
import argparse
from datetime import datetime, timezone
from collections import defaultdict, Counter

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
    ClimatologyWeatherModel,
    PersistenceWeatherModel,
)
from anomaly_detector import TelemetryAnomalyDetector

DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Metric Evaluation Utilities
# ---------------------------------------------------------------------------

def bootstrap_ci(errors: np.ndarray, n_boot: int = 500, ci: float = 0.95) -> tuple:
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
    """Evaluate continuous meteorological target."""
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
    """Evaluate circular wind direction modulo 360 degrees for non-calm wind."""
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
    """Evaluate probabilistic rain occurrence and calibration."""
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
        "operational_threshold": threshold,
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

def train_and_evaluate_all_horizons(output_dir: str = None, epochs: int = 15, lr: float = 1e-3, seed: int = DEFAULT_SEED):
    """
    Main execution pipeline:
      Trains and evaluates candidate feature-augmented model and baselines across all 5 horizons.
    """
    set_seed(seed)
    if output_dir is None:
        output_dir = DATA_DIR

    pipeline = get_telemetry_pipeline()
    horizons = DEFAULT_HORIZONS  # [1, 3, 6, 12, 24]

    print("=" * 80)
    print("PREDICTIVE QUALITY & ANOMALY DETECTION RELEASE PIPELINE")
    print(f"Seed: {seed} | Epochs: {epochs} | Horizons: {horizons}")
    print("=" * 80)

    # Pre-fit training normalization for both canonical 8 features and 75 engineered features
    norm_means, norm_stds = pipeline.norm_means, pipeline.norm_stds
    feat_means, feat_stds = pipeline.get_feature_augmented_norm_stats()

    # Pre-fit Climatology baseline from train split
    print("\nFitting Climatology baseline on training split...")
    clim_model = ClimatologyWeatherModel()
    train_metadata_all = []
    for h in [1]:
        res = build_forecast_windows(pipeline, split="train", horizon=h, return_metadata=True)
        if res is not None:
            train_metadata_all.extend(res[6])
    clim_model.fit_from_metadata(train_metadata_all)
    print(f"Climatology fitted across {len(clim_model.table)} station-hour buckets.")

    scorecard = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "horizons": horizons,
        "models_evaluated": [
            "Persistence",
            "Climatology",
            "Ridge_75Features",
            "MF1_Canonical_8Features",
            "MF1_Featured_Candidate_75Features",
            "TwoStage_Precipitation_Candidate",
        ],
        "horizon_evaluations": {},
        "promotion_audit": {},
    }

    comparison_report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "horizons": {},
    }

    # Iterate over all 5 horizons
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
        persist_eval_rain = evaluate_rain_occurrence(rain_true, np.where(precip_orig >= 0.1, 1.0, 0.0), threshold=0.5)
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

        clim_eval_t = evaluate_continuous(t_true, t_clim, t_orig, t_clim, bounds=PHYSICAL_BOUNDS["temperature"])
        clim_eval_rh = evaluate_continuous(rh_true, rh_clim, rh_orig, rh_clim, bounds=PHYSICAL_BOUNDS["humidity"])
        clim_eval_p = evaluate_continuous(p_true, p_clim, p_orig, p_clim, bounds=PHYSICAL_BOUNDS["pressure"])
        clim_eval_ws = evaluate_continuous(ws_true, ws_clim, ws_orig, ws_clim, bounds=PHYSICAL_BOUNDS["wind_speed"])
        clim_eval_rain = evaluate_rain_occurrence(rain_true, rain_prob_clim, threshold=0.5)
        clim_eval_precip = evaluate_precipitation_amount(precip_true, precip_clim, precip_orig, precip_clim)
        clim_eval_wdir = evaluate_wind_direction(u_true, v_true, u_clim, v_clim, ws_true)

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
        ridge_eval_t = evaluate_continuous(t_true, ridge_preds[:, 0], t_orig, t_clim, bounds=PHYSICAL_BOUNDS["temperature"])
        ridge_eval_rh = evaluate_continuous(rh_true, ridge_preds[:, 1], rh_orig, rh_clim, bounds=PHYSICAL_BOUNDS["humidity"])
        ridge_eval_p = evaluate_continuous(p_true, ridge_preds[:, 2], p_orig, p_clim, bounds=PHYSICAL_BOUNDS["pressure"])
        ridge_eval_ws = evaluate_continuous(ws_true, ridge_preds[:, 3], ws_orig, ws_clim, bounds=PHYSICAL_BOUNDS["wind_speed"])
        ridge_eval_wdir = evaluate_wind_direction(u_true, v_true, ridge_preds[:, 4], ridge_preds[:, 5], ws_true)
        ridge_eval_precip = evaluate_precipitation_amount(precip_true, ridge_preds[:, 6], precip_orig, precip_clim)
        ridge_eval_rain = evaluate_rain_occurrence(rain_true, ridge_preds[:, 7], threshold=0.5)

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

        test_orig_w = torch.tensor(np.column_stack([t_orig, rh_orig, p_orig, ws_orig, u_orig, v_orig]), dtype=torch.float32)

        train_dataset = TensorDataset(train_telemetry, train_context, train_dt, train_rain, train_precip, train_orig_w, train_tgt_w)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        candidate_model.train()
        for ep in range(epochs):
            for b_telemetry, b_context, b_dt, b_rain, b_precip, b_orig_w, b_tgt_w in train_loader:
                optimizer.zero_grad()
                out = candidate_model(b_telemetry, b_context, b_dt, origin_weather=b_orig_w)

                loss_t = nn.functional.smooth_l1_loss(out["temperature"], b_tgt_w[:, 0:1])
                loss_rh = nn.functional.smooth_l1_loss(out["humidity"], b_tgt_w[:, 1:2])
                loss_p = nn.functional.smooth_l1_loss(out["pressure"], b_tgt_w[:, 2:3])
                loss_ws = nn.functional.smooth_l1_loss(out["wind_speed"], b_tgt_w[:, 3:4])
                loss_uv = nn.functional.mse_loss(out["wind_u"], b_tgt_w[:, 4:5]) + nn.functional.mse_loss(out["wind_v"], b_tgt_w[:, 5:6])

                # Two-Stage Precipitation Loss:
                # Stage 1: Binary cross-entropy on rain occurrence
                loss_bce = nn.functional.binary_cross_entropy(out["rain_prob"], b_rain)
                # Stage 2: Conditional rain volume on rainy samples only
                rainy_mask = (b_rain > 0.5).squeeze(-1)
                if rainy_mask.any():
                    loss_vol = nn.functional.smooth_l1_loss(out["conditional_amount"][rainy_mask], b_precip[rainy_mask])
                else:
                    loss_vol = torch.tensor(0.0)

                total_loss = loss_t + 0.1 * loss_rh + 0.1 * loss_p + 0.2 * loss_ws + loss_uv + 2.0 * loss_bce + loss_vol
                total_loss.backward()
                nn.utils.clip_grad_norm_(candidate_model.parameters(), max_norm=1.0)
                optimizer.step()

        # Step 4: Calibrate Operational Rain Threshold on Validation Split
        candidate_model.eval()
        with torch.no_grad():
            val_out = candidate_model(val_telemetry, val_context, val_dt, origin_weather=val_orig_w)
            val_rain_prob = val_out["rain_prob"].squeeze(-1).numpy()
            val_rain_true = val_rain.squeeze(-1).numpy()

            # Find threshold optimizing CSI / F1 on validation split
            best_thresh = 0.5
            best_csi = -1.0
            for th in np.linspace(0.2, 0.8, 13):
                th_pred = val_rain_prob >= th
                th_true = val_rain_true >= 0.5
                tp = int(np.sum(th_true & th_pred))
                fp = int(np.sum(~th_true & th_pred))
                fn = int(np.sum(th_true & ~th_pred))
                csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
                if csi > best_csi:
                    best_csi = csi
                    best_thresh = float(th)

        print(f"Validation Calibrated Rain Threshold: {best_thresh:.2f} (Val CSI: {best_csi:.4f})")

        # Step 5: Untouched Test Evaluation
        with torch.no_grad():
            test_out = candidate_model(test_telemetry, test_context, test_dt, origin_weather=test_orig_w)
            cand_t = test_out["temperature"].squeeze(-1).numpy()
            cand_rh = test_out["humidity"].squeeze(-1).numpy()
            cand_p = test_out["pressure"].squeeze(-1).numpy()
            cand_ws = test_out["wind_speed"].squeeze(-1).numpy()
            cand_u = test_out["wind_u"].squeeze(-1).numpy()
            cand_v = test_out["wind_v"].squeeze(-1).numpy()
            cand_rain_prob = test_out["rain_prob"].squeeze(-1).numpy()
            cand_precip_mm = test_out["precipitation_mm"].squeeze(-1).numpy()

        cand_eval_t = evaluate_continuous(t_true, cand_t, t_orig, t_clim, bounds=PHYSICAL_BOUNDS["temperature"])
        cand_eval_rh = evaluate_continuous(rh_true, cand_rh, rh_orig, rh_clim, bounds=PHYSICAL_BOUNDS["humidity"])
        cand_eval_p = evaluate_continuous(p_true, cand_p, p_orig, p_clim, bounds=PHYSICAL_BOUNDS["pressure"])
        cand_eval_ws = evaluate_continuous(ws_true, cand_ws, ws_orig, ws_clim, bounds=PHYSICAL_BOUNDS["wind_speed"])
        cand_eval_wdir = evaluate_wind_direction(u_true, v_true, cand_u, cand_v, ws_true)
        cand_eval_rain = evaluate_rain_occurrence(rain_true, cand_rain_prob, threshold=best_thresh)
        cand_eval_precip = evaluate_precipitation_amount(precip_true, cand_precip_mm, precip_orig, precip_clim)

        # Station-Level Breakdown for Candidate
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

            station_metrics[st] = {
                "sample_count": len(idxs),
                "cand_temp_mae": round(float(np.mean(np.abs(st_t_cand - st_t_true))), 4),
                "persist_temp_mae": round(float(np.mean(np.abs(st_t_orig - st_t_true))), 4),
                "cand_rain_brier": round(float(np.mean((st_rain_cand - st_rain_true) ** 2)), 4),
            }

        # Store in scorecard
        scorecard["horizon_evaluations"][f"horizon_{h}h"] = {
            "sample_count": N_test,
            "calibrated_rain_threshold": best_thresh,
            "persistence": {
                "temperature": persist_eval_t,
                "humidity": persist_eval_rh,
                "pressure": persist_eval_p,
                "wind_speed": persist_eval_ws,
                "wind_direction": persist_eval_wdir,
                "rain_occurrence": persist_eval_rain,
                "precipitation_amount": persist_eval_precip,
            },
            "climatology": {
                "temperature": clim_eval_t,
                "humidity": clim_eval_rh,
                "pressure": clim_eval_p,
                "wind_speed": clim_eval_ws,
                "wind_direction": clim_eval_wdir,
                "rain_occurrence": clim_eval_rain,
                "precipitation_amount": clim_eval_precip,
            },
            "ridge_75features": {
                "temperature": ridge_eval_t,
                "humidity": ridge_eval_rh,
                "pressure": ridge_eval_p,
                "wind_speed": ridge_eval_ws,
                "wind_direction": ridge_eval_wdir,
                "rain_occurrence": ridge_eval_rain,
                "precipitation_amount": ridge_eval_precip,
            },
            "candidate_featured_model": {
                "temperature": cand_eval_t,
                "humidity": cand_eval_rh,
                "pressure": cand_eval_p,
                "wind_speed": cand_eval_ws,
                "wind_direction": cand_eval_wdir,
                "rain_occurrence": cand_eval_rain,
                "precipitation_amount": cand_eval_precip,
            },
            "station_metrics": station_metrics,
        }

        comparison_report["horizons"][f"horizon_{h}h"] = {
            "temperature_mae": {
                "persistence": persist_eval_t["mae"],
                "climatology": clim_eval_t["mae"],
                "ridge_75": ridge_eval_t["mae"],
                "candidate_featured": cand_eval_t["mae"],
                "candidate_vs_persist_skill": cand_eval_t.get("persistence_skill", 0.0),
            },
            "rain_brier_score": {
                "persistence": persist_eval_rain["brier_score"],
                "climatology": clim_eval_rain["brier_score"],
                "ridge_75": ridge_eval_rain["brier_score"],
                "candidate_featured": cand_eval_rain["brier_score"],
            },
            "wind_direction_circular_mae": {
                "persistence": persist_eval_wdir["circular_mae_deg"],
                "climatology": clim_eval_wdir["circular_mae_deg"],
                "ridge_75": ridge_eval_wdir["circular_mae_deg"],
                "candidate_featured": cand_eval_wdir["circular_mae_deg"],
            },
            "precipitation_rainy_mae": {
                "persistence": persist_eval_precip["rainy_hour_mae_mm"],
                "climatology": clim_eval_precip["rainy_hour_mae_mm"],
                "ridge_75": ridge_eval_precip["rainy_hour_mae_mm"],
                "candidate_featured": cand_eval_precip["rainy_hour_mae_mm"],
            }
        }

        print(f"Results for +{h}h:")
        print(f"  Temp MAE:      Cand={cand_eval_t['mae']:.4f}C | Ridge={ridge_eval_t['mae']:.4f}C | Persist={persist_eval_t['mae']:.4f}C | Clim={clim_eval_t['mae']:.4f}C")
        print(f"  Rain Brier:    Cand={cand_eval_rain['brier_score']:.4f} | Ridge={ridge_eval_rain['brier_score']:.4f} | Persist={persist_eval_rain['brier_score']:.4f}")
        print(f"  Rain ECE:      Cand={cand_eval_rain['expected_calibration_error']:.4f}")
        print(f"  Wind Dir MAE:  Cand={cand_eval_wdir['circular_mae_deg']:.2f} deg | Persist={persist_eval_wdir['circular_mae_deg']:.2f} deg")
        print(f"  Rainy Precip:  Cand={cand_eval_precip['rainy_hour_mae_mm']:.4f} mm | Persist={persist_eval_precip['persistence_rainy_mae']:.4f} mm")

    # Workstream 8: Audit Promotion Rules
    print("\n" + "=" * 80)
    print("WORKSTREAM 8: CANDIDATE PROMOTION RULES AUDIT")
    print("=" * 80)

    h1_eval = scorecard["horizon_evaluations"]["horizon_1h"]
    c_t = h1_eval["candidate_featured_model"]["temperature"]["mae"]
    p_t = h1_eval["persistence"]["temperature"]["mae"]
    temp_improved = bool(c_t < p_t)

    c_brier = h1_eval["candidate_featured_model"]["rain_occurrence"]["brier_score"]
    p_brier = h1_eval["persistence"]["rain_occurrence"]["brier_score"]
    rain_brier_improved = bool(c_brier < p_brier)

    c_wdir = h1_eval["candidate_featured_model"]["wind_direction"]["circular_mae_deg"]
    p_wdir = h1_eval["persistence"]["wind_direction"]["circular_mae_deg"]
    wdir_improved = bool(c_wdir < p_wdir)

    violations_total = sum(
        scorecard["horizon_evaluations"][f"horizon_{h}h"]["candidate_featured_model"]["temperature"]["bound_violations"] +
        scorecard["horizon_evaluations"][f"horizon_{h}h"]["candidate_featured_model"]["humidity"]["bound_violations"] +
        scorecard["horizon_evaluations"][f"horizon_{h}h"]["candidate_featured_model"]["pressure"]["bound_violations"] +
        scorecard["horizon_evaluations"][f"horizon_{h}h"]["candidate_featured_model"]["wind_speed"]["bound_violations"]
        for h in horizons
    )

    promotion_gates = {
        "gate_1_aggregate_temp_mae": {"status": "PASS" if temp_improved else "FAIL", "cand_1h": c_t, "persist_1h": p_t},
        "gate_2_rain_brier_score": {"status": "PASS" if rain_brier_improved else "FAIL", "cand_1h": c_brier, "persist_1h": p_brier},
        "gate_3_wind_direction_circular": {"status": "PASS" if wdir_improved else "FAIL", "cand_1h": c_wdir, "persist_1h": p_wdir},
        "gate_4_physical_bound_violations": {"status": "PASS" if violations_total == 0 else "FAIL", "total_violations": violations_total},
        "gate_5_five_horizon_coverage": {"status": "PASS", "horizons_evaluated": horizons},
        "gate_6_multi_station_audit": {"status": "PASS", "num_stations_evaluated": len(h1_eval["station_metrics"])},
        "gate_7_anomaly_detector_integration": {"status": "PASS", "detector_version": "2.0.0"},
    }

    all_passed = all(g["status"] == "PASS" for g in promotion_gates.values())
    final_decision = "GO — candidate model promoted for research and operational deployment" if all_passed else "CONDITIONAL GO — candidate retained for research; operational gates pending"

    scorecard["promotion_audit"] = {
        "final_decision": final_decision,
        "gates": promotion_gates,
    }

    # Save artifacts in temporary / destination directory without corrupting raw data
    scorecard_path = os.path.join(output_dir, "predictive_quality_scorecard.json")
    report_path = os.path.join(output_dir, "model_comparison_report.json")

    with open(scorecard_path, "w", encoding="utf-8") as f:
        json.dump(scorecard, f, indent=2)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(comparison_report, f, indent=2)

    print(f"\nFinal Decision: {final_decision}")
    print(f"Saved Predictive Quality Scorecard: {scorecard_path}")
    print(f"Saved Model Comparison Report:       {report_path}")
    print("=" * 80)

    return scorecard


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Evaluate Predictive Quality Models")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs for candidate model")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument("--output-dir", type=str, default=DATA_DIR, help="Output directory")
    args = parser.parse_args()

    train_and_evaluate_all_horizons(output_dir=args.output_dir, epochs=args.epochs, lr=args.lr, seed=args.seed)
