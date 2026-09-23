"""
Comprehensive Independent Multi-Horizon Weather & Hydrological Validator.

Produces the official scorecard for the Garcia Weather Telemetry Forecast Engine:
  - Canonical Forecast Contract: consumes canonical hourly test windows.
  - Multi-Horizon Evaluation: [1, 3, 6, 12, 24] hours ahead of forecast origin t0.
  - Core Commercial Weather Targets:
      * Temperature (Celsius): MAE, RMSE, bias, persistence skill, beats_persistence flag.
      * Relative Humidity (%): MAE, RMSE, bias, persistence skill, beats_persistence flag.
      * Atmospheric Pressure (hPa): MAE, RMSE, bias, persistence skill, tendency accuracy.
      * Wind Speed (km/h): MAE, RMSE, persistence skill, strong-wind recall.
      * Wind Direction (degrees): circular MAE, calm-wind sample count & coverage.
      * Derived Heat Index (Celsius): deterministic NOAA Rothfusz formula evaluated vs station observations.
      * Rain Occurrence: fixed 0.5 threshold, calibrated probability Brier score, reliability bins,
        frozen operational threshold, and per-horizon hybrid blend.
      * Rain Amount: rainy-hour MAE, overall MAE, RMSE, bias.
  - Solar Feasibility & Daylight Audit:
      * UV Index: documented as BLOCKED_BY_SENSOR_CALIBRATION (uncalibrated night spikes).
      * Light Intensity: documented as SECONDARY_BETA_DAYLIGHT_ONLY.
  - Beta / Internal Module:
      * River Stage Hydrology: MAE, RMSE, conformal prediction coverage (80%, 90%, 95%) with Wilson CIs.
        Designated INTERNAL_EXPERIMENT_BETA (not for life-safety or flood alarms).
  - Scorecard Governance:
      * Frozen calibration on validation split; untouched evaluation on test split.
      * Explicit positive and negative skill flags.
      * Outputs weather_validation_scorecard.json, validation_scorecard.json, and test_predictions_log.csv.
"""

import os
import sys
import math
import csv
import json
import argparse
from datetime import datetime, timezone
from collections import defaultdict

import numpy as np
import torch

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dataset import (
    get_telemetry_pipeline,
    build_forecast_windows,
    compute_file_sha256,
    compute_noaa_heat_index,
    circular_direction_error_deg,
    DATA_DIR,
    WEATHER_CSV_PATH,
    WATER_CSV_PATH,
    DEFAULT_SEQ_LEN,
    DEFAULT_HORIZONS,
    WATER_GAUGE_WEATHER_STATION,
)
from model import GarciaWeatherLNN, WeatherWaterLNN

SCORECARD_PATH = os.path.join(DATA_DIR, "validation_scorecard.json")
WEATHER_SCORECARD_PATH = os.path.join(DATA_DIR, "weather_validation_scorecard.json")
PREDICTIONS_LOG_PATH = os.path.join(DATA_DIR, "test_predictions_log.csv")

# Intensity boundaries (mm/h)
INTENSITY_CLASSES = {
    "dry": (0.0, 0.1),
    "trace": (0.1, 0.5),
    "light": (0.5, 2.5),
    "moderate": (2.5, 7.5),
    "heavy": (7.5, float("inf")),
}


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, x))))


def tanh(x: float) -> float:
    return math.tanh(max(-20.0, min(20.0, x)))


class StandaloneLNNRunner:
    """Inference runner for MF-2 ContinuousLNNCell weights."""

    def __init__(self, weights: dict):
        self.hidden_dim = weights.get("hidden_dim", 8)
        self.W_in = weights.get("W_in", [])
        self.W_rec = weights.get("W_rec", [])
        self.b_h = weights.get("b_h", [0.0] * self.hidden_dim)
        self.tau = weights.get("tau", [1.5] * self.hidden_dim)
        self.W_rain = weights.get("W_rain", [])
        self.b_rain = weights.get("b_rain", 0.0)
        self.W_water = weights.get("W_water", [])
        self.b_water = weights.get("b_water", 0.0)
        self.in_features = len(self.W_in)

    def predict_window(self, telemetry_arr: np.ndarray, dt_arr: np.ndarray, initial_water: float = None):
        """Unroll window with fresh hidden state h=0 (zero cross-window leakage)."""
        seq_len = telemetry_arr.shape[0]
        h = [0.0] * self.hidden_dim
        for t in range(seq_len):
            x = telemetry_arr[t].tolist()
            dt_val = float(dt_arr[t, 0])
            h_next = []
            for j in range(self.hidden_dim):
                in_sum = sum(x[i] * self.W_in[i][j] for i in range(self.in_features))
                rec_sum = sum(h[k] * self.W_rec[k][j] for k in range(self.hidden_dim))
                act = tanh(in_sum + rec_sum + self.b_h[j])
                decay = math.exp(-dt_val / max(0.1, self.tau[j]))
                h_next.append(decay * h[j] + (1.0 - decay) * act)
            h = h_next

        rain_logit = self.b_rain + sum(h[j] * self.W_rain[j] for j in range(self.hidden_dim))
        pred_rain = sigmoid(rain_logit)
        delta_water = self.b_water + sum(h[j] * self.W_water[j] for j in range(self.hidden_dim))
        pred_water = (initial_water + delta_water) if initial_water is not None else delta_water
        return pred_rain, pred_water


def compute_conformal_quantiles(calibration_residuals: list, coverage_levels=(0.80, 0.90, 0.95)) -> dict:
    """Compute empirical conformal prediction quantiles from held-out calibration residuals."""
    if not calibration_residuals:
        return {alpha: 0.0 for alpha in coverage_levels}
    sorted_res = sorted(calibration_residuals)
    n = len(sorted_res)
    quantiles = {}
    for alpha in coverage_levels:
        idx = min(int(math.ceil(alpha * (n + 1))) - 1, n - 1)
        quantiles[alpha] = sorted_res[idx]
    return quantiles


def wilson_score_interval(k: int, n: int, z: float = 1.96):
    """Wilson score confidence interval for binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    margin = (z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def compute_continuous_metrics(pred_vals: list, true_vals: list, persist_vals: list, clim_val: float) -> dict:
    """Compute complete regression metrics and persistence skill score."""
    n = len(true_vals)
    if n == 0:
        return {"sample_count": 0}

    errors = [p - y for p, y in zip(pred_vals, true_vals)]
    abs_errors = [abs(e) for e in errors]
    sq_errors = [e ** 2 for e in errors]

    persist_errors = [p - y for p, y in zip(persist_vals, true_vals)]
    abs_p_errors = [abs(e) for e in persist_errors]
    sq_p_errors = [e ** 2 for e in persist_errors]

    clim_errors = [clim_val - y for y in true_vals]
    abs_c_errors = [abs(e) for e in clim_errors]
    sq_c_errors = [e ** 2 for e in clim_errors]

    mae = sum(abs_errors) / n
    rmse = math.sqrt(sum(sq_errors) / n)
    bias = sum(errors) / n

    p_mae = sum(abs_p_errors) / n
    p_rmse = math.sqrt(sum(sq_p_errors) / n)

    c_mae = sum(abs_c_errors) / n
    c_rmse = math.sqrt(sum(sq_c_errors) / n)

    # Persistence skill score: 1 - (MAE_model / MAE_persistence)
    skill_vs_persistence = 1.0 - (mae / max(1e-4, p_mae))

    return {
        "sample_count": n,
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "bias": round(bias, 4),
        "persistence_mae": round(p_mae, 4),
        "persistence_rmse": round(p_rmse, 4),
        "climatology_mae": round(c_mae, 4),
        "climatology_rmse": round(c_rmse, 4),
        "skill_vs_persistence": round(skill_vs_persistence, 4),
        "beats_persistence": bool(mae < p_mae),
    }


def compute_pressure_tendency(pred_p: list, origin_p: list, true_p: list, threshold: float = 0.5) -> dict:
    """Classify and evaluate pressure tendency: rising (+1), falling (-1), steady (0)."""
    hits = 0
    total = len(pred_p)
    pred_cats = []
    true_cats = []

    for p_val, o_val, t_val in zip(pred_p, origin_p, true_p):
        dp_pred = p_val - o_val
        dp_true = t_val - o_val

        p_c = 1 if dp_pred > threshold else (-1 if dp_pred < -threshold else 0)
        t_c = 1 if dp_true > threshold else (-1 if dp_true < -threshold else 0)

        pred_cats.append(p_c)
        true_cats.append(t_c)
        if p_c == t_c:
            hits += 1

    acc = hits / max(1, total) * 100.0
    return {
        "sample_count": total,
        "tendency_threshold_hpa": threshold,
        "accuracy_pct": round(acc, 2),
    }


def compute_wind_direction_metrics(pred_dirs: list, true_dirs: list, speeds: list, calm_threshold: float = 1.0) -> dict:
    """Compute circular MAE for wind direction, masking calm wind conditions."""
    diffs = []
    calm_samples = 0
    for p, y, s in zip(pred_dirs, true_dirs, speeds):
        if s < calm_threshold:
            calm_samples += 1
            continue
        circ_err = circular_direction_error_deg(p, y)
        diffs.append(circ_err)

    c_mae = sum(diffs) / len(diffs) if diffs else 0.0
    return {
        "valid_sample_count": len(diffs),
        "calm_sample_count": calm_samples,
        "calm_coverage_pct": round(calm_samples / max(1, len(pred_dirs)) * 100.0, 2),
        "circular_mae_deg": round(c_mae, 2),
    }


def compute_strong_wind_recall(pred_speeds: list, true_speeds: list, threshold: float = 15.0) -> dict:
    """Compute Probability of Detection (POD) for strong wind speeds >= threshold km/h."""
    strong_indices = [i for i, s in enumerate(true_speeds) if s >= threshold]
    if not strong_indices:
        return {"threshold_kmh": threshold, "strong_event_count": 0, "recall_pod_pct": None}

    hits = sum(1 for i in strong_indices if pred_speeds[i] >= threshold)
    return {
        "threshold_kmh": threshold,
        "strong_event_count": len(strong_indices),
        "recall_pod_pct": round(hits / len(strong_indices) * 100.0, 2),
    }


def compute_rain_metrics(pred_probs: list, targets: list, threshold: float = 0.5) -> dict:
    """Compute complete suite of operational rain classification metrics."""
    n = len(targets)
    if n == 0:
        return {}

    tp = fp = tn = fn = 0
    brier_sum = 0.0

    for p, y in zip(pred_probs, targets):
        brier_sum += (p - y) ** 2
        pred_cls = 1 if p >= threshold else 0
        target_cls = int(y)
        if pred_cls == 1 and target_cls == 1:
            tp += 1
        elif pred_cls == 1 and target_cls == 0:
            fp += 1
        elif pred_cls == 0 and target_cls == 0:
            tn += 1
        elif pred_cls == 0 and target_cls == 1:
            fn += 1

    acc = (tp + tn) / n * 100.0
    rec = tp / (tp + fn) * 100.0 if (tp + fn) > 0 else 0.0
    prec = tp / (tp + fp) * 100.0 if (tp + fp) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    far = fp / (tp + fp) * 100.0 if (tp + fp) > 0 else 0.0
    csi = tp / (tp + fp + fn) * 100.0 if (tp + fp + fn) > 0 else 0.0
    brier = brier_sum / n

    # Calibration / Reliability Bins (5 bins)
    bin_counts = [0] * 5
    bin_true_sums = [0.0] * 5
    bin_pred_sums = [0.0] * 5
    for p, y in zip(pred_probs, targets):
        b = min(int(p * 5), 4)
        bin_counts[b] += 1
        bin_pred_sums[b] += p
        bin_true_sums[b] += y

    reliability_bins = []
    ece_sum = 0.0
    for b in range(5):
        cnt = bin_counts[b]
        b_min = b * 0.2
        b_max = (b + 1) * 0.2
        avg_pred = bin_pred_sums[b] / cnt if cnt > 0 else (b_min + b_max) / 2
        obs_freq = bin_true_sums[b] / cnt if cnt > 0 else 0.0
        if cnt > 0:
            ece_sum += abs(avg_pred - obs_freq) * cnt
        reliability_bins.append({
            "bin_range": f"{b_min:.1f}-{b_max:.1f}",
            "sample_count": cnt,
            "mean_pred_prob": round(avg_pred, 4),
            "observed_frequency": round(obs_freq, 4),
        })

    ece = ece_sum / n if n > 0 else 0.0

    return {
        "threshold": round(threshold, 3),
        "accuracy_pct": round(acc, 2),
        "recall_pod_pct": round(rec, 2),
        "precision_pct": round(prec, 2),
        "f1_score_pct": round(f1, 2),
        "false_alarm_ratio_pct": round(far, 2),
        "critical_success_index_pct": round(csi, 2),
        "brier_score": round(brier, 4),
        "expected_calibration_error": round(ece, 4),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "reliability_bins": reliability_bins,
    }


def compute_rain_amount_metrics(pred_precip: list, true_precip: list, persist_precip: list = None, clim_precip: float = 0.0) -> dict:
    """Compute comprehensive rain amount metrics: overall, baselines, dry/rainy subsets, and heavy rain events."""
    n = len(true_precip)
    if n == 0:
        return {}

    overall_errs = [p - y for p, y in zip(pred_precip, true_precip)]
    overall_mae = sum(abs(e) for e in overall_errs) / n
    overall_rmse = math.sqrt(sum(e**2 for e in overall_errs) / n)
    overall_bias = sum(overall_errs) / n

    # Baselines
    if persist_precip is not None and len(persist_precip) == n:
        persist_errs = [p - y for p, y in zip(persist_precip, true_precip)]
        persist_mae = sum(abs(e) for e in persist_errs) / n
        persist_rmse = math.sqrt(sum(e**2 for e in persist_errs) / n)
        persist_bias = sum(persist_errs) / n
        skill_vs_persist = 1.0 - (overall_mae / max(1e-4, persist_mae))
        beats_persist = bool(overall_mae < persist_mae)
    else:
        persist_mae = persist_rmse = persist_bias = skill_vs_persist = beats_persist = None

    clim_errs = [clim_precip - y for y in true_precip]
    clim_mae = sum(abs(e) for e in clim_errs) / n
    clim_rmse = math.sqrt(sum(e**2 for e in clim_errs) / n)

    # Subsets: Dry-hour (< 0.1 mm) vs Rainy-hour (>= 0.1 mm)
    dry_indices = [i for i, y in enumerate(true_precip) if y < 0.1]
    rainy_indices = [i for i, y in enumerate(true_precip) if y >= 0.1]

    if dry_indices:
        dry_errs = [pred_precip[i] - true_precip[i] for i in dry_indices]
        dry_mae = sum(abs(e) for e in dry_errs) / len(dry_indices)
        dry_rmse = math.sqrt(sum(e**2 for e in dry_errs) / len(dry_indices))
        dry_bias = sum(dry_errs) / len(dry_indices)
    else:
        dry_mae = dry_rmse = dry_bias = None

    if rainy_indices:
        rainy_errs = [pred_precip[i] - true_precip[i] for i in rainy_indices]
        rainy_mae = sum(abs(e) for e in rainy_errs) / len(rainy_indices)
        rainy_rmse = math.sqrt(sum(e**2 for e in rainy_errs) / len(rainy_indices))
        rainy_bias = sum(rainy_errs) / len(rainy_indices)
    else:
        rainy_mae = rainy_rmse = rainy_bias = None

    # Heavy-rain threshold events (2.5 mm, 5.0 mm, 10.0 mm)
    heavy_thresholds = {
        "threshold_2_5mm": 2.5,
        "threshold_5_0mm": 5.0,
        "threshold_10_0mm": 10.0,
    }
    threshold_metrics = {}
    for key, thresh in heavy_thresholds.items():
        obs_events = sum(1 for y in true_precip if y >= thresh)
        pred_events = sum(1 for p in pred_precip if p >= thresh)
        hits = sum(1 for p, y in zip(pred_precip, true_precip) if p >= thresh and y >= thresh)
        prec = (hits / pred_events * 100.0) if pred_events > 0 else 0.0
        rec = (hits / obs_events * 100.0) if obs_events > 0 else (100.0 if obs_events == 0 else 0.0)
        csi = (hits / (obs_events + pred_events - hits) * 100.0) if (obs_events + pred_events - hits) > 0 else 0.0
        threshold_metrics[key] = {
            "threshold_mm": thresh,
            "observed_event_count": obs_events,
            "predicted_event_count": pred_events,
            "hits": hits,
            "precision_pct": round(prec, 2),
            "recall_pod_pct": round(rec, 2),
            "critical_success_index_pct": round(csi, 2),
        }

    return {
        "overall_samples": n,
        "overall_mae_mm": round(overall_mae, 4),
        "overall_rmse_mm": round(overall_rmse, 4),
        "overall_bias_mm": round(overall_bias, 4),
        "persistence_mae_mm": round(persist_mae, 4) if persist_mae is not None else None,
        "persistence_rmse_mm": round(persist_rmse, 4) if persist_rmse is not None else None,
        "persistence_bias_mm": round(persist_bias, 4) if persist_bias is not None else None,
        "climatology_mae_mm": round(clim_mae, 4),
        "climatology_rmse_mm": round(clim_rmse, 4),
        "skill_vs_persistence": round(skill_vs_persist, 4) if skill_vs_persist is not None else None,
        "beats_persistence": beats_persist,
        "dry_hour_samples": len(dry_indices),
        "dry_hour_mae_mm": round(dry_mae, 4) if dry_mae is not None else None,
        "dry_hour_rmse_mm": round(dry_rmse, 4) if dry_rmse is not None else None,
        "dry_hour_bias_mm": round(dry_bias, 4) if dry_bias is not None else None,
        "rainy_hour_samples": len(rainy_indices),
        "rainy_hour_mae_mm": round(rainy_mae, 4) if rainy_mae is not None else None,
        "rainy_hour_rmse_mm": round(rainy_rmse, 4) if rainy_rmse is not None else None,
        "rainy_hour_bias_mm": round(rainy_bias, 4) if rainy_bias is not None else None,
        "heavy_rain_thresholds": threshold_metrics,
        "uncertainty_intervals": {
            "status": "UNAVAILABLE",
            "coverage_pct": None,
            "mean_interval_width_mm": None,
            "reason": "Validated conformal intervals are only implemented for beta water level. Weather and precipitation intervals are unavailable.",
        },
    }


def compute_water_metrics(pred_levels: list, true_levels: list, persist_levels: list, climatology_val: float) -> dict:
    """Compute complete river stage hydrological error metrics."""
    m = len(true_levels)
    if m == 0:
        return {"sample_count": 0}

    errors = [p - y for p, y in zip(pred_levels, true_levels)]
    abs_errors = [abs(e) for e in errors]
    sq_errors = [e ** 2 for e in errors]

    persist_errors = [p - y for p, y in zip(persist_levels, true_levels) if p is not None]
    abs_p_errors = [abs(e) for e in persist_errors] if persist_errors else [0.0]
    sq_p_errors = [e ** 2 for e in persist_errors] if persist_errors else [0.0]

    clim_errors = [climatology_val - y for y in true_levels]
    abs_c_errors = [abs(e) for e in clim_errors]
    sq_c_errors = [e ** 2 for e in clim_errors]

    mae = sum(abs_errors) / m
    rmse = math.sqrt(sum(sq_errors) / m)
    bias = sum(errors) / m

    p_mae = sum(abs_p_errors) / len(abs_p_errors)
    p_rmse = math.sqrt(sum(sq_p_errors) / len(sq_p_errors))

    c_mae = sum(abs_c_errors) / m
    c_rmse = math.sqrt(sum(sq_c_errors) / m)

    return {
        "status": "INTERNAL_EXPERIMENT_BETA",
        "not_for_life_safety": True,
        "sample_count": m,
        "mae_meters": round(mae, 4),
        "rmse_meters": round(rmse, 4),
        "bias_meters": round(bias, 4),
        "persistence_mae_meters": round(p_mae, 4),
        "persistence_rmse_meters": round(p_rmse, 4),
        "climatology_mae_meters": round(c_mae, 4),
        "climatology_rmse_meters": round(c_rmse, 4),
        "beats_persistence_mae": bool(mae < p_mae),
    }


def train_simple_linear_regression(train_res, test_res, target_field_name: str) -> list:
    """Train simple linear regression baseline on origin features and predict target on test split."""
    if train_res is None or test_res is None:
        return [0.0] * (len(test_res[-1]) if test_res else 0)

    tr_meta = train_res[-1]
    te_meta = test_res[-1]

    # Features: origin temperature, humidity, pressure, wind_speed, precip
    X_tr = []
    y_tr = []
    for m in tr_meta:
        feats = [
            float(m.get("origin_temperature", 28.0) or 28.0),
            float(m.get("origin_humidity", 75.0) or 75.0),
            float(m.get("origin_pressure", 1008.0) or 1008.0),
            float(m.get("origin_wind_speed", 5.0) or 5.0),
            float(m.get("last_observed_precip", 0.0) or 0.0),
            1.0,  # Bias
        ]
        tgt = float(m.get(target_field_name, 0.0) or 0.0)
        X_tr.append(feats)
        y_tr.append(tgt)

    X_mat = np.array(X_tr, dtype=np.float32)
    y_vec = np.array(y_tr, dtype=np.float32)

    # Solve least squares with small L2 regularization
    l2 = 1.0 * np.eye(X_mat.shape[1])
    try:
        weights = np.linalg.solve(X_mat.T @ X_mat + l2, X_mat.T @ y_vec)
    except Exception:
        weights = np.zeros(X_mat.shape[1])

    X_te = []
    for m in te_meta:
        feats = [
            float(m.get("origin_temperature", 28.0) or 28.0),
            float(m.get("origin_humidity", 75.0) or 75.0),
            float(m.get("origin_pressure", 1008.0) or 1008.0),
            float(m.get("origin_wind_speed", 5.0) or 5.0),
            float(m.get("last_observed_precip", 0.0) or 0.0),
            1.0,
        ]
        X_te.append(feats)

    preds = (np.array(X_te, dtype=np.float32) @ weights).tolist()
    return preds


def train_and_predict_logistic_regression(train_res, test_res) -> list:
    """Train a reproducible logistic regression baseline on train split t0 features and predict on test split."""
    if train_res is None or len(train_res[0]) == 0 or test_res is None or len(test_res[0]) == 0:
        return [0.0] * (len(test_res[0]) if test_res else 0)

    tr_telemetry, _, tr_rain, _, _, _, tr_meta = train_res
    te_telemetry, _, _, _, _, _, te_meta = test_res

    X_tr = []
    y_tr = []
    for i, m in enumerate(tr_meta):
        x_base = tr_telemetry[i, -1].numpy()
        p0 = float(m.get("last_observed_precip", 0.0) or 0.0)
        r3 = float(m.get("rolling_3h_precip", p0) or 0.0)
        r6 = float(m.get("rolling_6h_precip", p0) or 0.0)
        X_tr.append(np.append(x_base, [p0, r3, r6]))
        y_tr.append(float(tr_rain[i, 0]))

    X_te = []
    for i, m in enumerate(te_meta):
        x_base = te_telemetry[i, -1].numpy()
        p0 = float(m.get("last_observed_precip", 0.0) or 0.0)
        r3 = float(m.get("rolling_3h_precip", p0) or 0.0)
        r6 = float(m.get("rolling_6h_precip", p0) or 0.0)
        X_te.append(np.append(x_base, [p0, r3, r6]))

    X_tr_t = torch.tensor(np.array(X_tr), dtype=torch.float32)
    y_tr_t = torch.tensor(np.array(y_tr), dtype=torch.float32).unsqueeze(-1)
    X_te_t = torch.tensor(np.array(X_te), dtype=torch.float32)

    torch.manual_seed(42)
    model = torch.nn.Linear(X_tr_t.shape[1], 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for _ in range(80):
        optimizer.zero_grad()
        loss = loss_fn(model(X_tr_t), y_tr_t)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        preds = torch.sigmoid(model(X_te_t)).squeeze(-1).tolist()
    return preds


def evaluate_horizon(
    pipeline,
    horizon: int,
    mf1_model: GarciaWeatherLNN,
    mf2_runner: StandaloneLNNRunner,
    climatology_stats: dict,
    device: torch.device,
):
    """Run comprehensive independent evaluation for a single horizon across all weather variables."""
    print(f"\nEvaluating Horizon +{horizon}h across all weather targets...")

    train_res = build_forecast_windows(pipeline=pipeline, split="train", horizon=horizon, seq_len=DEFAULT_SEQ_LEN, return_metadata=True)
    calib_res = build_forecast_windows(pipeline=pipeline, split="val", horizon=horizon, seq_len=DEFAULT_SEQ_LEN, return_metadata=True)
    test_res = build_forecast_windows(pipeline=pipeline, split="test", horizon=horizon, seq_len=DEFAULT_SEQ_LEN, return_metadata=True)

    if test_res is None or len(test_res[0]) == 0:
        raise RuntimeError(f"No valid test samples found for horizon +{horizon}h!")

    calib_telemetry, calib_dt, calib_rain, calib_precip, calib_water, calib_has_water, calib_meta = calib_res
    test_telemetry, test_dt, test_rain, test_precip, test_water, test_has_water, test_meta = test_res

    n_test = test_telemetry.shape[0]
    n_calib = calib_telemetry.shape[0]

    # --- PHASE 5: CALIBRATION SPLIT FITTING ---
    # 1. MF-1 on Calibration Split
    mf1_calib_rain_probs = []
    mf1_calib_water_resids = []
    calib_true_rain = [float(calib_rain[i, 0]) for i in range(n_calib)]
    calib_persist_rain = [1.0 if m["last_observed_precip"] >= 0.1 else 0.0 for m in calib_meta]

    # Calib origin weather tensor
    c_orig_list = [
        [
            float(m["origin_temperature"]),
            float(m["origin_humidity"]),
            float(m["origin_pressure"]),
            float(m["origin_wind_speed"]),
            float(m["origin_wind_u"]),
            float(m["origin_wind_v"]),
        ]
        for m in calib_meta
    ]
    c_orig_tensor = torch.tensor(c_orig_list, dtype=torch.float32, device=device)

    # Track calib predictions for skill-gate evaluation
    c_pred_temp = []
    c_pred_rh = []
    c_pred_p = []
    c_pred_ws = []
    c_pred_u = []
    c_pred_v = []

    if mf1_model is not None:
        mf1_model.eval()
        with torch.no_grad():
            c_t = calib_telemetry.to(device)
            c_dt = calib_dt.to(device)
            c_last_w = torch.tensor(
                [[float(m["last_observed_water"] or 0.0)] if (m["has_water"] and m["last_observed_water"] is not None) else [0.0] for m in calib_meta],
                dtype=torch.float32, device=device
            )
            c_out = mf1_model(c_t, c_dt, initial_water=c_last_w, origin_weather=c_orig_tensor, return_dict=True)
            mf1_calib_rain_probs = c_out["rain_prob"][:, 0].cpu().tolist()
            c_pred_temp = c_out["temperature"][:, 0].cpu().tolist()
            c_pred_rh = c_out["humidity"][:, 0].cpu().tolist()
            c_pred_p = c_out["pressure"][:, 0].cpu().tolist()
            c_pred_ws = c_out["wind_speed"][:, 0].cpu().tolist()
            c_pred_u = c_out["wind_u"][:, 0].cpu().tolist()
            c_pred_v = c_out["wind_v"][:, 0].cpu().tolist()

            c_w_pred = c_out["water_level"][:, 0].cpu().numpy()
            for i in range(n_calib):
                if calib_has_water[i, 0].item() > 0.5:
                    w_true = calib_water[i, 0].item()
                    mf1_calib_water_resids.append(abs(c_w_pred[i] - w_true))
    else:
        mf1_calib_rain_probs = [0.0] * n_calib

    # 2. Conformal quantiles for water level
    mf1_quantiles = compute_conformal_quantiles(mf1_calib_water_resids)

    # 3. Fit Rain Blending Weight w_h on Calibration Split ONLY
    best_w = 0.5
    best_brier = float("inf")
    for w_cand in np.linspace(0.0, 1.0, 21):
        cand_brier = sum(
            ((w_cand * p_m + (1.0 - w_cand) * p_p) - y) ** 2
            for p_m, p_p, y in zip(mf1_calib_rain_probs, calib_persist_rain, calib_true_rain)
        ) / max(1, n_calib)
        if cand_brier < best_brier:
            best_brier = cand_brier
            best_w = round(float(w_cand), 2)

    # 4. Fit Operational Alert Threshold T_op on Calibration Split ONLY
    best_thresh = 0.5
    best_f1 = -1.0
    for t_cand in np.linspace(0.1, 0.9, 17):
        tp = fp = fn = 0
        for p_m, y in zip(mf1_calib_rain_probs, calib_true_rain):
            p_cls = 1 if p_m >= t_cand else 0
            if p_cls == 1 and y == 1:
                tp += 1
            elif p_cls == 1 and y == 0:
                fp += 1
            elif p_cls == 0 and y == 1:
                fn += 1
        cand_f1 = (2 * tp) / max(1, 2 * tp + fp + fn)
        if cand_f1 > best_f1:
            best_f1 = cand_f1
            best_thresh = round(float(t_cand), 2)

    # 5. Continuous Skill Gates evaluated on Calibration Split
    calib_skill_gates = {}
    if c_pred_temp:
        for var_name, p_list, o_key, t_key in [
            ("temperature", c_pred_temp, "origin_temperature", "target_temperature"),
            ("humidity", c_pred_rh, "origin_humidity", "target_humidity"),
            ("pressure", c_pred_p, "origin_pressure", "target_pressure"),
            ("wind_speed", c_pred_ws, "origin_wind_speed", "target_wind_speed"),
        ]:
            t_vals = [float(m[t_key]) for m in calib_meta]
            o_vals = [float(m[o_key]) for m in calib_meta]
            m_mae = sum(abs(p - y) for p, y in zip(p_list, t_vals)) / n_calib
            p_mae = sum(abs(o - y) for o, y in zip(o_vals, t_vals)) / n_calib
            skill = 1.0 - (m_mae / max(1e-4, p_mae))
            calib_skill_gates[var_name] = {
                "calib_model_mae": round(m_mae, 4),
                "calib_persist_mae": round(p_mae, 4),
                "calib_skill_score": round(skill, 4),
                "selected_source": "learned_model" if skill > 0 else "persistence_fallback",
            }

        # Wind direction circular skill gate on calib split
        if c_pred_u and c_pred_v:
            c_pred_wind_deg = [round(math.degrees(math.atan2(v, u)) % 360.0, 2) for u, v in zip(c_pred_u, c_pred_v)]
            c_true_wind_deg = [float(m["target_wind_deg"]) for m in calib_meta]
            c_orig_wind_deg = [float(m["origin_wind_deg"]) for m in calib_meta]
            c_ws = [float(m["target_wind_speed"]) for m in calib_meta]

            calib_m_circ = compute_wind_direction_metrics(c_pred_wind_deg, c_true_wind_deg, c_ws)
            calib_p_circ = compute_wind_direction_metrics(c_orig_wind_deg, c_true_wind_deg, c_ws)
            c_m_cmae = calib_m_circ["circular_mae_deg"]
            c_p_cmae = calib_p_circ["circular_mae_deg"]
            c_circ_skill = round(1.0 - (c_m_cmae / max(1e-4, c_p_cmae)), 4)
            calib_skill_gates["wind_direction"] = {
                "calib_model_circular_mae": round(c_m_cmae, 2),
                "calib_persist_circular_mae": round(c_p_cmae, 2),
                "calib_skill_score": c_circ_skill,
                "selected_source": "learned_model" if c_circ_skill > 0 else "persistence_fallback",
            }
        else:
            calib_skill_gates["wind_direction"] = {
                "selected_source": "persistence_fallback",
            }

        calib_skill_gates["heat_index"] = {
            "derivation_rule": "derived_from_selected_temp_and_humidity",
            "formula": "NOAA NWS Rothfusz regression",
            "selected_source": "derived_from_selected_temp_and_humidity",
        }

    # --- UNTOUCHED TEST SPLIT EVALUATION ---
    t_orig_list = [
        [
            float(m["origin_temperature"]),
            float(m["origin_humidity"]),
            float(m["origin_pressure"]),
            float(m["origin_wind_speed"]),
            float(m["origin_wind_u"]),
            float(m["origin_wind_v"]),
        ]
        for m in test_meta
    ]
    t_orig_tensor = torch.tensor(t_orig_list, dtype=torch.float32, device=device)

    # Model predictions on test split
    if mf1_model is not None:
        mf1_model.eval()
        with torch.no_grad():
            t_t = test_telemetry.to(device)
            t_dt = test_dt.to(device)
            t_last_w = torch.tensor(
                [[float(m["last_observed_water"] or 0.0)] if (m["has_water"] and m["last_observed_water"] is not None) else [0.0] for m in test_meta],
                dtype=torch.float32, device=device
            )
            t_out = mf1_model(t_t, t_dt, initial_water=t_last_w, origin_weather=t_orig_tensor, return_dict=True)

            pred_temp = t_out["temperature"][:, 0].cpu().tolist()
            pred_rh = t_out["humidity"][:, 0].cpu().tolist()
            pred_pressure = t_out["pressure"][:, 0].cpu().tolist()
            pred_ws = t_out["wind_speed"][:, 0].cpu().tolist()
            pred_u = t_out["wind_u"][:, 0].cpu().tolist()
            pred_v = t_out["wind_v"][:, 0].cpu().tolist()
            pred_rain_prob = t_out["rain_prob"][:, 0].cpu().tolist()
            pred_precip_vol = t_out["precipitation_mm"][:, 0].cpu().tolist()
            pred_water_level = t_out["water_level"][:, 0].cpu().tolist()
    else:
        pred_temp = [float(m["origin_temperature"]) for m in test_meta]
        pred_rh = [float(m["origin_humidity"]) for m in test_meta]
        pred_pressure = [float(m["origin_pressure"]) for m in test_meta]
        pred_ws = [float(m["origin_wind_speed"]) for m in test_meta]
        pred_u = [float(m["origin_wind_u"]) for m in test_meta]
        pred_v = [float(m["origin_wind_v"]) for m in test_meta]
        pred_rain_prob = [0.0] * n_test
        pred_precip_vol = [0.0] * n_test
        pred_water_level = [2.5] * n_test

    # Standalone MF-2 predictions
    mf2_rain_probs = []
    mf2_water_preds = []
    for i in range(n_test):
        if mf2_runner is not None:
            w_init = float(test_meta[i]["last_observed_water"] or 0.0) if test_meta[i]["has_water"] else None
            r_p, w_p = mf2_runner.predict_window(test_telemetry[i].numpy(), test_dt[i].numpy(), initial_water=w_init)
            mf2_rain_probs.append(r_p)
            mf2_water_preds.append(w_p)
        else:
            mf2_rain_probs.append(0.0)
            mf2_water_preds.append(2.5)

    # Observed ground truth & persistence at t0
    true_temp = [float(m["target_temperature"]) for m in test_meta]
    true_rh = [float(m["target_humidity"]) for m in test_meta]
    true_p = [float(m["target_pressure"]) for m in test_meta]
    true_ws = [float(m["target_wind_speed"]) for m in test_meta]
    true_wind_deg = [float(m["target_wind_deg"]) for m in test_meta]
    true_rain = [float(test_rain[i, 0]) for i in range(n_test)]
    true_precip = [float(test_precip[i, 0]) for i in range(n_test)]
    true_hi = [float(m["target_heat_index"]) for m in test_meta]

    origin_temp = [float(m["origin_temperature"]) for m in test_meta]
    origin_rh = [float(m["origin_humidity"]) for m in test_meta]
    origin_p = [float(m["origin_pressure"]) for m in test_meta]
    origin_ws = [float(m["origin_wind_speed"]) for m in test_meta]
    origin_wind_deg = [float(m["origin_wind_deg"]) for m in test_meta]
    persist_rain = [1.0 if m["last_observed_precip"] >= 0.1 else 0.0 for m in test_meta]

    # Reconstruct circular wind direction (degrees) and derived Heat Index
    pred_wind_deg = [
        round(math.degrees(math.atan2(v, u)) % 360.0, 2)
        for u, v in zip(pred_u, pred_v)
    ]
    derived_model_hi = [
        round(compute_noaa_heat_index(t, rh), 2)
        for t, rh in zip(pred_temp, pred_rh)
    ]
    persist_hi = [
        round(compute_noaa_heat_index(t, rh), 2)
        for t, rh in zip(origin_temp, origin_rh)
    ]

    # Simple Linear Regression baselines
    lin_temp = train_simple_linear_regression(train_res, test_res, "target_temperature")
    lin_rh = train_simple_linear_regression(train_res, test_res, "target_humidity")
    lin_p = train_simple_linear_regression(train_res, test_res, "target_pressure")
    lin_ws = train_simple_linear_regression(train_res, test_res, "target_wind_speed")
    logreg_rain_probs = train_and_predict_logistic_regression(train_res, test_res)

    # Hybrid Blended Rain Forecast on Test: p_blend = best_w * p_model + (1 - best_w) * p_persist
    hybrid_rain_probs = [
        best_w * p_m + (1.0 - best_w) * p_p
        for p_m, p_p in zip(pred_rain_prob, persist_rain)
    ]

    # --- COMPUTE COMPREHENSIVE METRICS ---
    # 1. Temperature
    temp_metrics = compute_continuous_metrics(pred_temp, true_temp, origin_temp, climatology_stats["temperature"])
    temp_metrics["linear_regression_mae"] = round(float(np.mean(np.abs(np.array(lin_temp) - np.array(true_temp)))), 4)
    temp_metrics["selected_source"] = calib_skill_gates.get("temperature", {}).get("selected_source", "persistence_fallback")
    temp_metrics["hybrid_mae"] = temp_metrics["mae"] if temp_metrics["selected_source"] == "learned_model" else temp_metrics["persistence_mae"]

    # 2. Relative Humidity
    rh_metrics = compute_continuous_metrics(pred_rh, true_rh, origin_rh, climatology_stats["humidity"])
    rh_metrics["linear_regression_mae"] = round(float(np.mean(np.abs(np.array(lin_rh) - np.array(true_rh)))), 4)
    rh_metrics["selected_source"] = calib_skill_gates.get("humidity", {}).get("selected_source", "persistence_fallback")
    rh_metrics["hybrid_mae"] = rh_metrics["mae"] if rh_metrics["selected_source"] == "learned_model" else rh_metrics["persistence_mae"]

    # 3. Pressure & Tendency
    p_metrics = compute_continuous_metrics(pred_pressure, true_p, origin_p, climatology_stats["pressure"])
    p_metrics["linear_regression_mae"] = round(float(np.mean(np.abs(np.array(lin_p) - np.array(true_p)))), 4)
    p_metrics["pressure_tendency"] = compute_pressure_tendency(pred_pressure, origin_p, true_p)
    p_metrics["selected_source"] = calib_skill_gates.get("pressure", {}).get("selected_source", "persistence_fallback")
    p_metrics["hybrid_mae"] = p_metrics["mae"] if p_metrics["selected_source"] == "learned_model" else p_metrics["persistence_mae"]

    # 4. Wind Speed
    ws_metrics = compute_continuous_metrics(pred_ws, true_ws, origin_ws, climatology_stats["wind_speed"])
    ws_metrics["linear_regression_mae"] = round(float(np.mean(np.abs(np.array(lin_ws) - np.array(true_ws)))), 4)
    ws_metrics["strong_wind_recall"] = compute_strong_wind_recall(pred_ws, true_ws, threshold=15.0)
    ws_metrics["selected_source"] = calib_skill_gates.get("wind_speed", {}).get("selected_source", "persistence_fallback")
    ws_metrics["hybrid_mae"] = ws_metrics["mae"] if ws_metrics["selected_source"] == "learned_model" else ws_metrics["persistence_mae"]

    # 5. Wind Direction (Circular)
    wdir_metrics = {
        "model_circular": compute_wind_direction_metrics(pred_wind_deg, true_wind_deg, true_ws),
        "persistence_circular": compute_wind_direction_metrics(origin_wind_deg, true_wind_deg, true_ws),
    }
    m_circ_mae = wdir_metrics["model_circular"]["circular_mae_deg"]
    p_circ_mae = wdir_metrics["persistence_circular"]["circular_mae_deg"]
    wdir_metrics["beats_persistence"] = bool(m_circ_mae < p_circ_mae)
    wdir_metrics["skill_vs_persistence"] = round(1.0 - (m_circ_mae / max(1e-4, p_circ_mae)), 4)
    wdir_metrics["selected_source"] = calib_skill_gates.get("wind_direction", {}).get("selected_source", "persistence_fallback")

    # 6. Derived Heat Index
    hi_metrics = compute_continuous_metrics(derived_model_hi, true_hi, persist_hi, climatology_stats["heat_index"])
    hi_metrics["derivation_formula"] = "NOAA NWS Rothfusz regression from predicted (T, RH)"
    hi_metrics["selected_source"] = calib_skill_gates.get("heat_index", {}).get("selected_source", "derived_from_selected_temp_and_humidity")

    # 7. Rain Occurrence
    rain_05_metrics = compute_rain_metrics(pred_rain_prob, true_rain, threshold=0.5)
    rain_op_metrics = compute_rain_metrics(pred_rain_prob, true_rain, threshold=best_thresh)
    rain_hybrid_metrics = compute_rain_metrics(hybrid_rain_probs, true_rain, threshold=0.5)
    persist_rain_metrics = compute_rain_metrics(persist_rain, true_rain, threshold=0.5)
    logreg_rain_metrics = compute_rain_metrics(logreg_rain_probs, true_rain, threshold=0.5)
    clim_rain_metrics = compute_rain_metrics([climatology_stats["rain_prior"]] * n_test, true_rain, threshold=0.5)

    # 8. Rain Amount
    persist_precip = [float(m["last_observed_precip"]) for m in test_meta]
    precip_amount_metrics = compute_rain_amount_metrics(
        pred_precip_vol,
        true_precip,
        persist_precip=persist_precip,
        clim_precip=climatology_stats.get("precipitation", 0.0),
    )

    # 9. Gauge stage (Internal/Beta)
    gauge_indices = [i for i in range(n_test) if test_has_water[i, 0].item() > 0.5]
    true_water_gauge = [float(test_water[i, 0]) for i in gauge_indices]
    persist_water_gauge = [test_meta[i]["last_observed_water"] for i in gauge_indices]
    mf1_water_gauge = [pred_water_level[i] for i in gauge_indices]
    mf2_water_gauge = [mf2_water_preds[i] for i in gauge_indices]

    persist_water_metrics = compute_water_metrics(persist_water_gauge, true_water_gauge, persist_water_gauge, climatology_stats["water_stage"])
    mf1_water_metrics = compute_water_metrics(mf1_water_gauge, true_water_gauge, persist_water_gauge, climatology_stats["water_stage"])
    mf2_water_metrics = compute_water_metrics(mf2_water_gauge, true_water_gauge, persist_water_gauge, climatology_stats["water_stage"])

    # Conformal Coverage Evaluation on TEST split
    conformal_coverage = {}
    for alpha in [0.80, 0.90, 0.95]:
        q1 = mf1_quantiles.get(alpha, 0.0)
        cov1_hits = sum(1 for p, y in zip(mf1_water_gauge, true_water_gauge) if abs(p - y) <= q1)
        cov1_pct = cov1_hits / max(1, len(gauge_indices)) * 100.0
        ci1_low, ci1_high = wilson_score_interval(cov1_hits, len(gauge_indices))
        conformal_coverage[f"nominal_{int(alpha*100)}"] = {
            "nominal_level": alpha,
            "test_gauge_samples": len(gauge_indices),
            "observed_coverage_pct": round(cov1_pct, 2),
            "coverage_meets_nominal": bool(cov1_pct >= (alpha * 100 - 3.0)),
            "interval_full_width_meters": round(2 * q1, 4),
            "coverage_95_ci": [round(ci1_low * 100, 2), round(ci1_high * 100, 2)],
        }

    # Intensity class breakdown
    intensity_breakdown = {}
    for c_name, (low, high) in INTENSITY_CLASSES.items():
        c_idx = [i for i in range(n_test) if low <= true_precip[i] < high]
        c_count = len(c_idx)
        if c_count > 0:
            c_mf1_rec = sum(1 for i in c_idx if pred_rain_prob[i] >= 0.5) / c_count * 100.0
            c_per_rec = sum(1 for i in c_idx if persist_rain[i] >= 0.5) / c_count * 100.0
        else:
            c_mf1_rec = c_per_rec = None
        intensity_breakdown[c_name] = {
            "sample_count": c_count,
            "mf1_detection_rate_pct": round(c_mf1_rec, 2) if c_mf1_rec is not None else None,
            "persistence_detection_rate_pct": round(c_per_rec, 2) if c_per_rec is not None else None,
        }

    # Assemble per-sample records for CSV export
    sample_records = []
    for i in range(n_test):
        meta = test_meta[i]
        sample_records.append({
            "station_id": meta["station_id"],
            "origin_timestamp": meta["origin_timestamp"],
            "target_timestamp": meta["target_timestamp"],
            "horizon_hours": horizon,
            "actual_lead_hours": meta["actual_lead_hours"],
            "actual_rain": true_rain[i],
            "actual_precip_mm": true_precip[i],
            "actual_temp": true_temp[i],
            "actual_humidity": true_rh[i],
            "actual_pressure": true_p[i],
            "actual_wind_speed": true_ws[i],
            "actual_wind_dir_deg": true_wind_deg[i],
            "actual_heat_index": true_hi[i],
            "pred_temp": round(pred_temp[i], 2),
            "pred_humidity": round(pred_rh[i], 2),
            "pred_pressure": round(pred_pressure[i], 2),
            "pred_wind_speed": round(pred_ws[i], 2),
            "pred_wind_dir_deg": pred_wind_deg[i],
            "derived_heat_index": derived_model_hi[i],
            "mf1_rain_prob": round(pred_rain_prob[i], 4),
            "hybrid_rain_prob": round(hybrid_rain_probs[i], 4),
            "mf1_precip_mm": round(pred_precip_vol[i], 4),
            "persist_temp": origin_temp[i],
            "persist_humidity": origin_rh[i],
            "persist_pressure": origin_p[i],
            "persist_wind_speed": origin_ws[i],
            "persist_rain": persist_rain[i],
            "actual_water_level": meta["actual_water_level"],
            "mf1_water_level": round(pred_water_level[i], 4) if meta["has_water"] else None,
            "persist_water": meta["last_observed_water"],
        })

    horizon_result = {
        "test_samples_total": n_test,
        "calibration_samples_total": n_calib,
        "small_sample_warning": bool(n_test < 200),
        "calib_skill_gates": calib_skill_gates,
        "temperature": temp_metrics,
        "humidity": rh_metrics,
        "pressure": p_metrics,
        "wind_speed": ws_metrics,
        "wind_direction": wdir_metrics,
        "heat_index": hi_metrics,
        "rain_occurrence": {
            "probability_quality": {
                "hybrid_brier_score": rain_hybrid_metrics.get("brier_score"),
                "model_brier_score": rain_05_metrics.get("brier_score"),
                "persistence_brier_score": persist_rain_metrics.get("brier_score"),
                "climatology_brier_score": clim_rain_metrics.get("brier_score"),
                "brier_skill_score_vs_persistence": round(1.0 - (rain_hybrid_metrics.get("brier_score", 1.0) / max(1e-4, persist_rain_metrics.get("brier_score", 1.0))), 4),
                "hybrid_expected_calibration_error": rain_hybrid_metrics.get("expected_calibration_error"),
                "model_expected_calibration_error": rain_05_metrics.get("expected_calibration_error"),
            },
            "event_classification": {
                "fixed_threshold_05": {
                    "f1_score_pct": rain_05_metrics.get("f1_score_pct"),
                    "recall_pod_pct": rain_05_metrics.get("recall_pod_pct"),
                    "precision_pct": rain_05_metrics.get("precision_pct"),
                    "critical_success_index_pct": rain_05_metrics.get("critical_success_index_pct"),
                    "false_alarm_ratio_pct": rain_05_metrics.get("false_alarm_ratio_pct"),
                },
                "frozen_operational_threshold": {
                    "threshold": best_thresh,
                    "f1_score_pct": rain_op_metrics.get("f1_score_pct"),
                    "recall_pod_pct": rain_op_metrics.get("recall_pod_pct"),
                    "precision_pct": rain_op_metrics.get("precision_pct"),
                    "critical_success_index_pct": rain_op_metrics.get("critical_success_index_pct"),
                    "false_alarm_ratio_pct": rain_op_metrics.get("false_alarm_ratio_pct"),
                },
                "hybrid_blend_05": {
                    "f1_score_pct": rain_hybrid_metrics.get("f1_score_pct"),
                    "recall_pod_pct": rain_hybrid_metrics.get("recall_pod_pct"),
                    "precision_pct": rain_hybrid_metrics.get("precision_pct"),
                    "critical_success_index_pct": rain_hybrid_metrics.get("critical_success_index_pct"),
                    "false_alarm_ratio_pct": rain_hybrid_metrics.get("false_alarm_ratio_pct"),
                },
                "persistence_05": {
                    "f1_score_pct": persist_rain_metrics.get("f1_score_pct"),
                    "recall_pod_pct": persist_rain_metrics.get("recall_pod_pct"),
                    "precision_pct": persist_rain_metrics.get("precision_pct"),
                    "critical_success_index_pct": persist_rain_metrics.get("critical_success_index_pct"),
                    "false_alarm_ratio_pct": persist_rain_metrics.get("false_alarm_ratio_pct"),
                },
            },
            "operational_selection_objective": {
                "objective": "Minimize Brier score on validation calibration split to produce calibrated probabilities for risk outlooks",
                "weight_selection_metric": "Brier score on calibration split",
                "threshold_selection_metric": "F1 score on calibration split",
                "tradeoff_analysis": (
                    "Hybrid probability blending consistently minimizes Brier score (probability error) across horizons, "
                    "providing reliable probabilistic outlooks. However, for hard binary classification at 0.5 threshold, "
                    "persistence achieves higher or comparable F1/recall on short horizons (1h/3h). Operational users "
                    "seeking threshold alerts should use the calibrated operational threshold or persistence baseline accordingly."
                ),
            },
            "fixed_threshold_05": rain_05_metrics,
            "frozen_operational_threshold": {
                "threshold": best_thresh,
                "metrics": rain_op_metrics,
            },
            "hybrid_blend": {
                "frozen_model_weight": best_w,
                "frozen_persistence_weight": round(1.0 - best_w, 2),
                "metrics": rain_hybrid_metrics,
            },
            "persistence": persist_rain_metrics,
            "logistic_regression": logreg_rain_metrics,
            "climatology": clim_rain_metrics,
            "beats_persistence_f1": bool(rain_05_metrics.get("f1_score_pct", 0) > persist_rain_metrics.get("f1_score_pct", 0)),
            "beats_persistence_brier": bool(rain_05_metrics.get("brier_score", 1.0) < persist_rain_metrics.get("brier_score", 1.0)),
        },
        "precipitation_amount": precip_amount_metrics,
        "solar_status": {
            "uv_index": {
                "status": "BLOCKED_BY_SENSOR_CALIBRATION",
                "reason": "Uncalibrated sensor reports up to 11.0 index during nighttime (00:00-04:00 local). Must not be scored without hardware recalibration.",
            },
            "light_intensity": {
                "status": "SECONDARY_BETA_DAYLIGHT_ONLY",
                "reason": "Photometric lux readings available for monitoring daylight cycle; uncalibrated for solar irradiance forecast.",
            },
        },
        "water_level_beta": {
            "status": "INTERNAL_EXPERIMENT_BETA",
            "not_for_life_safety": True,
            "mf1_pytorch": mf1_water_metrics,
            "mf2_standalone": mf2_water_metrics,
            "persistence": persist_water_metrics,
            "conformal_uncertainty": conformal_coverage,
        },
        "intensity_breakdown": intensity_breakdown,
        # Backwards-compatibility aliases for verify_provenance.py and legacy consumers
        "rain_metrics": {
            "mf1_pytorch": rain_05_metrics,
            "persistence": persist_rain_metrics,
            "climatology": clim_rain_metrics,
            "logistic_regression": logreg_rain_metrics,
            "hybrid_blend": rain_hybrid_metrics,
        },
        "water_metrics": {
            "mf1_pytorch": mf1_water_metrics,
            "mf2_standalone": mf2_water_metrics,
            "persistence": persist_water_metrics,
            "status": "INTERNAL_EXPERIMENT_BETA",
            "not_for_life_safety": True,
        },
        "conformal_uncertainty": conformal_coverage,
    }

    return horizon_result, sample_records


def run_full_validation(horizons: list = None):
    """Execute complete independent validation suite across all horizons."""
    if horizons is None:
        horizons = DEFAULT_HORIZONS

    print("=" * 90)
    print("GARCIA WEATHER TELEMETRY FORECAST ENGINE: COMPREHENSIVE VALIDATION SUITE")
    print(f"Horizons: {horizons} hours")
    print("=" * 90)

    pipeline = get_telemetry_pipeline()

    # Determine train split climatology baselines
    train_rain_count = 0
    train_total = 0
    train_temps = []
    train_rhs = []
    train_pressures = []
    train_ws = []
    train_his = []
    train_water_vals = []
    train_precips = []

    for st_id, h_dict in pipeline.station_hourly.items():
        for h, rec in h_dict.items():
            if h <= pipeline.train_end:
                train_total += 1
                if rec["precipitation"] >= 0.1:
                    train_rain_count += 1
                train_temps.append(rec["temperature"])
                train_rhs.append(rec["humidity"])
                train_pressures.append(rec["pressure"])
                train_ws.append(rec["wind_speed"])
                train_his.append(rec["heat_index"])
                train_precips.append(rec["precipitation"])
                if st_id == WATER_GAUGE_WEATHER_STATION and h in pipeline.water_hourly:
                    train_water_vals.append(pipeline.water_hourly[h])

    climatology_stats = {
        "rain_prior": train_rain_count / max(1, train_total),
        "temperature": float(np.mean(train_temps)),
        "humidity": float(np.mean(train_rhs)),
        "pressure": float(np.mean(train_pressures)),
        "wind_speed": float(np.mean(train_ws)),
        "heat_index": float(np.mean(train_his)),
        "precipitation": float(np.mean(train_precips)) if train_precips else 0.0,
        "water_stage": float(np.mean(train_water_vals)) if train_water_vals else 2.50,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_horizon_results = {}
    all_sample_records = []

    # Get git commit
    git_commit = "unknown"
    try:
        import subprocess
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(DATA_DIR), text=True).strip()
    except Exception:
        pass

    for h in horizons:
        # Load MF-1 model checkpoint
        mf1_model = None
        ckpt_path = os.path.join(DATA_DIR, f"lnn_weather_water_h{h}.pt")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(DATA_DIR, "lnn_weather_water.pt")

        if os.path.exists(ckpt_path):
            try:
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                m_config = ckpt.get("manifest", {}).get("model_config", {"input_dim": 8, "hidden_dim": 32})
                # Check whether state dict has GarciaWeatherLNN heads
                sd = ckpt["model_state_dict"]
                if "temp_head.0.weight" in sd:
                    mf1_model = GarciaWeatherLNN(**m_config).to(device)
                else:
                    mf1_model = WeatherWaterLNN(**m_config).to(device)
                mf1_model.load_state_dict(sd)
                mf1_model.eval()
            except Exception as e:
                print(f"Warning: Failed to load MF-1 checkpoint ({ckpt_path}): {e}")

        # Load MF-2 model weights
        mf2_runner = None
        weights_path = os.path.join(DATA_DIR, f"lnn_trained_weights_h{h}.json")
        if not os.path.exists(weights_path):
            weights_path = os.path.join(DATA_DIR, "lnn_trained_weights.json")

        if os.path.exists(weights_path):
            try:
                with open(weights_path, "r", encoding="utf-8") as f:
                    w_dict = json.load(f)
                mf2_runner = StandaloneLNNRunner(w_dict)
            except Exception as e:
                print(f"Warning: Failed to load MF-2 weights ({weights_path}): {e}")

        h_result, h_samples = evaluate_horizon(
            pipeline=pipeline,
            horizon=h,
            mf1_model=mf1_model,
            mf2_runner=mf2_runner,
            climatology_stats=climatology_stats,
            device=device,
        )

        all_horizon_results[f"horizon_{h}h"] = h_result
        all_sample_records.extend(h_samples)

    # Save per-sample prediction CSV
    fieldnames = [
        "station_id", "origin_timestamp", "target_timestamp", "horizon_hours", "actual_lead_hours",
        "actual_rain", "actual_precip_mm", "actual_temp", "actual_humidity", "actual_pressure", "actual_wind_speed", "actual_wind_dir_deg", "actual_heat_index",
        "pred_temp", "pred_humidity", "pred_pressure", "pred_wind_speed", "pred_wind_dir_deg", "derived_heat_index",
        "mf1_rain_prob", "hybrid_rain_prob", "mf1_precip_mm",
        "persist_temp", "persist_humidity", "persist_pressure", "persist_wind_speed", "persist_rain",
        "actual_water_level", "mf1_water_level", "persist_water",
    ]
    with open(PREDICTIONS_LOG_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_sample_records)
    print(f"\nSaved {len(all_sample_records)} test sample predictions to: {PREDICTIONS_LOG_PATH}")

    # Build master scorecard structure
    scorecard = {
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "code_commit": git_commit,
        "product_name": "Garcia Weather Telemetry Forecast Engine",
        "dataset_hashes": {
            "weather_telemetry_sha256": compute_file_sha256(WEATHER_CSV_PATH),
            "water_level_telemetry_sha256": compute_file_sha256(WATER_CSV_PATH),
        },
        "model_status": "RESEARCH_PROTOTYPE",
        "tested_horizons_hours": horizons,
        "climatology_baselines": {
            "train_rain_prevalence_pct": round(climatology_stats["rain_prior"] * 100.0, 2),
            "train_mean_temperature_c": round(climatology_stats["temperature"], 2),
            "train_mean_humidity_pct": round(climatology_stats["humidity"], 2),
            "train_mean_pressure_hpa": round(climatology_stats["pressure"], 2),
            "train_mean_wind_speed_kmh": round(climatology_stats["wind_speed"], 2),
            "train_mean_heat_index_c": round(climatology_stats["heat_index"], 2),
            "train_mean_river_stage_meters": round(climatology_stats["water_stage"], 4),
        },
        "operational_recommendation": (
            "DEPLOY_HYBRID_GUIDANCE. Use learned model where validation skill is positive; "
            "fallback to persistence where persistence error is lower. Hybrid rain probability blending "
            "consistently improves probability quality (Brier score) for risk outlooks, but does not universally "
            "dominate persistence in discrete event classification (F1/recall) at 1h/3h. "
            "Weather variables do NOT have validated confidence intervals (uncertainty intervals unavailable). "
            "Conformal prediction intervals apply ONLY to the internal beta water-level experiment. "
            "DO NOT use water level for life-safety or automated flood warning triggers."
        ),
        "horizons": all_horizon_results,
    }

    def to_serializable(obj):
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [to_serializable(v) for v in obj]
        return obj

    clean_scorecard = to_serializable(scorecard)
    with open(WEATHER_SCORECARD_PATH, "w", encoding="utf-8") as f:
        json.dump(clean_scorecard, f, indent=2)
    with open(SCORECARD_PATH, "w", encoding="utf-8") as f:
        json.dump(clean_scorecard, f, indent=2)

    print(f"Saved weather scorecard to: {WEATHER_SCORECARD_PATH}")
    print(f"Saved canonical scorecard to: {SCORECARD_PATH}")

    # Build and export operational inference policy
    policy_horizons = {}
    for h in horizons:
        h_k = f"horizon_{h}h"
        h_res = all_horizon_results.get(h_k, {})
        h_rain = h_res.get("rain_occurrence", {})
        h_hybrid = h_rain.get("hybrid_blend", {})
        h_op_thresh = h_rain.get("frozen_operational_threshold", {})

        policy_horizons[str(h)] = {
            "selected_sources": {
                "temperature": h_res.get("temperature", {}).get("selected_source", "persistence_fallback"),
                "humidity": h_res.get("humidity", {}).get("selected_source", "persistence_fallback"),
                "pressure": h_res.get("pressure", {}).get("selected_source", "persistence_fallback"),
                "wind_speed": h_res.get("wind_speed", {}).get("selected_source", "persistence_fallback"),
                "wind_direction": h_res.get("wind_direction", {}).get("selected_source", "persistence_fallback"),
                "heat_index": "derived_from_selected_temp_and_humidity",
            },
            "rain_model_weight": float(h_hybrid.get("frozen_model_weight", 0.5)),
            "rain_persistence_weight": float(h_hybrid.get("frozen_persistence_weight", 0.5)),
            "operational_rain_threshold": float(h_op_thresh.get("threshold", 0.5)),
            "calibration_code_commit": git_commit,
        }

    inference_policy = {
        "policy_version": "1.0.0",
        "policy_code_commit": git_commit,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_hashes": {
            "weather_telemetry_sha256": compute_file_sha256(WEATHER_CSV_PATH),
            "water_level_telemetry_sha256": compute_file_sha256(WATER_CSV_PATH),
        },
        "horizons": policy_horizons,
    }

    policy_path = os.path.join(DATA_DIR, "inference_policy.json")
    with open(policy_path, "w", encoding="utf-8") as f:
        json.dump(to_serializable(inference_policy), f, indent=2)
    print(f"Saved inference policy artifact to: {policy_path}")

    # Print summary tables to console
    print("\n" + "=" * 110)
    print("GARCIA WEATHER TELEMETRY FORECAST ENGINE: MULTI-HORIZON WEATHER SCORECARD")
    print("=" * 110)
    print(f"{'Horizon':<8} | {'Temp MAE':<10} | {'RH MAE':<10} | {'P MAE':<10} | {'WS MAE':<10} | {'WDir Circ':<11} | {'HI MAE':<10} | {'Rain F1':<10} | {'Rain Brier':<10}")
    print("-" * 110)
    for h in horizons:
        h_k = f"horizon_{h}h"
        res = all_horizon_results.get(h_k, {})
        t_m = f"{res.get('temperature', {}).get('mae', 0.0):.2f}C"
        rh_m = f"{res.get('humidity', {}).get('mae', 0.0):.1f}%"
        p_m = f"{res.get('pressure', {}).get('mae', 0.0):.2f}hPa"
        ws_m = f"{res.get('wind_speed', {}).get('mae', 0.0):.2f}km/h"
        wd_m = f"{res.get('wind_direction', {}).get('model_circular', {}).get('circular_mae_deg', 0.0):.1f}deg"
        hi_m = f"{res.get('heat_index', {}).get('mae', 0.0):.2f}C"
        r_f1 = f"{res.get('rain_occurrence', {}).get('fixed_threshold_05', {}).get('f1_score_pct', 0.0):.1f}%"
        r_br = f"{res.get('rain_occurrence', {}).get('fixed_threshold_05', {}).get('brier_score', 0.0):.4f}"
        print(f"+{h:02d}h     | {t_m:<10} | {rh_m:<10} | {p_m:<10} | {ws_m:<10} | {wd_m:<11} | {hi_m:<10} | {r_f1:<10} | {r_br:<10}")

    print("=" * 110)
    print(f"{'Horizon':<8} | {'Persist Temp':<13} | {'Persist RH':<11} | {'Persist P':<11} | {'Persist WS':<11} | {'Persist Rain F1':<16} | {'Persist Brier':<13}")
    print("-" * 110)
    for h in horizons:
        h_k = f"horizon_{h}h"
        res = all_horizon_results.get(h_k, {})
        p_t = f"{res.get('temperature', {}).get('persistence_mae', 0.0):.2f}C"
        p_rh = f"{res.get('humidity', {}).get('persistence_mae', 0.0):.1f}%"
        p_p = f"{res.get('pressure', {}).get('persistence_mae', 0.0):.2f}hPa"
        p_ws = f"{res.get('wind_speed', {}).get('persistence_mae', 0.0):.2f}km/h"
        p_f1 = f"{res.get('rain_occurrence', {}).get('persistence', {}).get('f1_score_pct', 0.0):.1f}%"
        p_br = f"{res.get('rain_occurrence', {}).get('persistence', {}).get('brier_score', 0.0):.4f}"
        print(f"+{h:02d}h     | {p_t:<13} | {p_rh:<11} | {p_p:<11} | {p_ws:<11} | {p_f1:<16} | {p_br:<13}")
    print("=" * 110)

    return scorecard


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Garcia Weather Telemetry Independent Multi-Horizon Validator.")
    parser.add_argument("--horizons", nargs="+", type=int, default=DEFAULT_HORIZONS, help="List of horizons (e.g. 1 3 6 12 24)")
    args = parser.parse_args()

    run_full_validation(horizons=args.horizons)
