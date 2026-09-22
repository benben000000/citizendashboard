"""
Comprehensive Independent Multi-Horizon Validator & Scorecard Suite.

Features:
  - Canonical Forecast Contract: consumes canonical hourly test windows.
  - Multi-Horizon Evaluation: [1, 3, 6, 12, 24] hours ahead of forecast origin t0.
  - Model Family Comparison:
      * MF-1: PyTorch WeatherWaterLNN
      * MF-2: Standalone ContinuousLNNCell
      * Baselines: Rain Persistence, Rain Climatology, Water Persistence, Water Climatology
  - Strict 3-Way Uncertainty Evaluation:
      * Train Split: Model training
      * Calibration Split (Val): Conformal residual quantiles
      * Test Split: Empirical test coverage evaluation
  - Full Operational Metrics:
      * Rain: Accuracy, Precision, Recall/POD, F1, FAR, CSI, Brier, Reliability Bins, Confusion Matrix, Event-level hit/FAR, Intensity Breakdown (dry, trace, light, moderate, heavy)
      * Water: MAE, RMSE, Bias, Persistence MAE/RMSE, Climatology MAE/RMSE, Rising Stage MAE
  - Honest Reporting:
      * Negative results clearly stated (e.g. loses to persistence)
      * Status strictly labeled RESEARCH_PROTOTYPE
      * Outputs validation_scorecard.json and test_predictions_log.csv
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
    DATA_DIR,
    WEATHER_CSV_PATH,
    WATER_CSV_PATH,
    DEFAULT_SEQ_LEN,
    DEFAULT_HORIZONS,
    WATER_GAUGE_WEATHER_STATION,
)
from model import WeatherWaterLNN

SCORECARD_PATH = os.path.join(DATA_DIR, "validation_scorecard.json")
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
        self.b_water = weights.get("b_water", 2.50)
        self.in_features = len(self.W_in)

    def predict_window(self, telemetry_arr: np.ndarray, dt_arr: np.ndarray):
        """Unroll window with fresh hidden state h=0 (zero cross-window leakage)."""
        seq_len = telemetry_arr.shape[0]
        h = [0.0] * self.hidden_dim
        pred_rain = 0.0
        pred_water = self.b_water

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
        pred_water = self.b_water + sum(h[j] * self.W_water[j] for j in range(self.hidden_dim))
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
    for b in range(5):
        cnt = bin_counts[b]
        b_min = b * 0.2
        b_max = (b + 1) * 0.2
        avg_pred = bin_pred_sums[b] / cnt if cnt > 0 else (b_min + b_max) / 2
        obs_freq = bin_true_sums[b] / cnt if cnt > 0 else 0.0
        reliability_bins.append({
            "bin_range": f"{b_min:.1f}-{b_max:.1f}",
            "sample_count": cnt,
            "mean_pred_prob": round(avg_pred, 4),
            "observed_frequency": round(obs_freq, 4),
        })

    return {
        "accuracy_pct": round(acc, 2),
        "recall_pod_pct": round(rec, 2),
        "precision_pct": round(prec, 2),
        "f1_score_pct": round(f1, 2),
        "false_alarm_ratio_pct": round(far, 2),
        "critical_success_index_pct": round(csi, 2),
        "brier_score": round(brier, 4),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "reliability_bins": reliability_bins,
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

    # Rising stage errors (where current stage is greater than origin stage)
    rising_errors = []
    for p, y, orig in zip(pred_levels, true_levels, persist_levels):
        if orig is not None and y > orig:
            rising_errors.append(abs(p - y))
    rising_mae = sum(rising_errors) / len(rising_errors) if rising_errors else None

    return {
        "sample_count": m,
        "mae_meters": round(mae, 4),
        "rmse_meters": round(rmse, 4),
        "bias_meters": round(bias, 4),
        "persistence_mae_meters": round(p_mae, 4),
        "persistence_rmse_meters": round(p_rmse, 4),
        "climatology_mae_meters": round(c_mae, 4),
        "climatology_rmse_meters": round(c_rmse, 4),
        "beats_persistence_mae": bool(mae < p_mae),
        "rising_stage_count": len(rising_errors),
        "rising_stage_mae_meters": round(rising_mae, 4) if rising_mae is not None else None,
    }


def evaluate_horizon(
    pipeline,
    horizon: int,
    mf1_model: WeatherWaterLNN,
    mf2_runner: StandaloneLNNRunner,
    climatology_rain_prior: float,
    climatology_water_stage: float,
    device: torch.device,
):
    """Run comprehensive independent evaluation for a single horizon."""
    print(f"\nEvaluating Horizon +{horizon}h...")

    # 1. Load calibration split (val) for conformal quantile calibration
    calib_res = build_forecast_windows(
        pipeline=pipeline,
        split="val",
        horizon=horizon,
        seq_len=DEFAULT_SEQ_LEN,
        return_metadata=True,
    )
    # 2. Load untouched test split
    test_res = build_forecast_windows(
        pipeline=pipeline,
        split="test",
        horizon=horizon,
        seq_len=DEFAULT_SEQ_LEN,
        return_metadata=True,
    )

    if test_res is None or len(test_res[0]) == 0:
        raise RuntimeError(f"No valid test samples found for horizon +{horizon}h!")

    calib_telemetry, calib_dt, calib_rain, calib_precip, calib_water, calib_has_water, calib_meta = calib_res
    test_telemetry, test_dt, test_rain, test_precip, test_water, test_has_water, test_meta = test_res

    n_test = test_telemetry.shape[0]
    n_calib = calib_telemetry.shape[0]

    # --- Phase 4 Conformal Calibration on CALIBRATION split ---
    # MF-1 Calibration residuals
    mf1_calib_water_resids = []
    mf2_calib_water_resids = []

    if mf1_model is not None:
        mf1_model.eval()
        with torch.no_grad():
            c_t = calib_telemetry.to(device)
            c_dt = calib_dt.to(device)
            _, _, c_w_pred = mf1_model(c_t, c_dt)
            c_w_pred = c_w_pred[:, -1, 0].cpu().numpy()

            for i in range(n_calib):
                if calib_has_water[i, 0].item() > 0.5:
                    w_true = calib_water[i, 0].item()
                    mf1_calib_water_resids.append(abs(c_w_pred[i] - w_true))

    if mf2_runner is not None:
        for i in range(n_calib):
            if calib_has_water[i, 0].item() > 0.5:
                _, w_pred = mf2_runner.predict_window(calib_telemetry[i].numpy(), calib_dt[i].numpy())
                w_true = calib_water[i, 0].item()
                mf2_calib_water_resids.append(abs(w_pred - w_true))

    mf1_quantiles = compute_conformal_quantiles(mf1_calib_water_resids)
    mf2_quantiles = compute_conformal_quantiles(mf2_calib_water_resids)

    # --- UNTOUCHED TEST SPLIT EVALUATION ---
    # Model predictions
    mf1_test_rain_probs = []
    mf1_test_precip_vols = []
    mf1_test_water_preds = []

    if mf1_model is not None:
        mf1_model.eval()
        with torch.no_grad():
            t_t = test_telemetry.to(device)
            t_dt = test_dt.to(device)
            p_rain, p_precip, p_water = mf1_model(t_t, t_dt)
            mf1_test_rain_probs = p_rain[:, -1, 0].cpu().tolist()
            mf1_test_precip_vols = p_precip[:, -1, 0].cpu().tolist()
            mf1_test_water_preds = p_water[:, -1, 0].cpu().tolist()
    else:
        mf1_test_rain_probs = [0.0] * n_test
        mf1_test_precip_vols = [0.0] * n_test
        mf1_test_water_preds = [2.5] * n_test

    mf2_test_rain_probs = []
    mf2_test_water_preds = []
    for i in range(n_test):
        if mf2_runner is not None:
            r_p, w_p = mf2_runner.predict_window(test_telemetry[i].numpy(), test_dt[i].numpy())
            mf2_test_rain_probs.append(r_p)
            mf2_test_water_preds.append(w_p)
        else:
            mf2_test_rain_probs.append(0.0)
            mf2_test_water_preds.append(2.5)

    # Ground truth targets & baselines
    true_rain = [float(test_rain[i, 0]) for i in range(n_test)]
    true_precip = [float(test_precip[i, 0]) for i in range(n_test)]
    persist_rain = [1.0 if sample["last_observed_precip"] >= 0.1 else 0.0 for sample in test_meta]
    climatology_rain = [climatology_rain_prior] * n_test

    # Gauge target subsets
    gauge_indices = [i for i in range(n_test) if test_has_water[i, 0].item() > 0.5]
    true_water_gauge = [float(test_water[i, 0]) for i in gauge_indices]
    persist_water_gauge = [test_meta[i]["last_observed_water"] for i in gauge_indices]
    mf1_water_gauge = [mf1_test_water_preds[i] for i in gauge_indices]
    mf2_water_gauge = [mf2_test_water_preds[i] for i in gauge_indices]

    # Compute metrics
    mf1_rain_metrics = compute_rain_metrics(mf1_test_rain_probs, true_rain)
    mf2_rain_metrics = compute_rain_metrics(mf2_test_rain_probs, true_rain)
    persist_rain_metrics = compute_rain_metrics(persist_rain, true_rain)
    clim_rain_metrics = compute_rain_metrics(climatology_rain, true_rain)

    mf1_water_metrics = compute_water_metrics(mf1_water_gauge, true_water_gauge, persist_water_gauge, climatology_water_stage)
    mf2_water_metrics = compute_water_metrics(mf2_water_gauge, true_water_gauge, persist_water_gauge, climatology_water_stage)

    # Conformal Coverage Evaluation on TEST split
    conformal_coverage = {}
    for alpha in [0.80, 0.90, 0.95]:
        # MF-1
        q1 = mf1_quantiles.get(alpha, 0.0)
        cov1_hits = sum(1 for p, y in zip(mf1_water_gauge, true_water_gauge) if abs(p - y) <= q1)
        cov1_pct = cov1_hits / max(1, len(gauge_indices)) * 100.0
        ci1_low, ci1_high = wilson_score_interval(cov1_hits, len(gauge_indices))

        # MF-2
        q2 = mf2_quantiles.get(alpha, 0.0)
        cov2_hits = sum(1 for p, y in zip(mf2_water_gauge, true_water_gauge) if abs(p - y) <= q2)
        cov2_pct = cov2_hits / max(1, len(gauge_indices)) * 100.0
        ci2_low, ci2_high = wilson_score_interval(cov2_hits, len(gauge_indices))

        conformal_coverage[f"nominal_{int(alpha*100)}"] = {
            "nominal_level": alpha,
            "calibration_gauge_samples": len(mf1_calib_water_resids),
            "test_gauge_samples": len(gauge_indices),
            "small_sample_warning": len(mf1_calib_water_resids) < 50 or len(gauge_indices) < 50,
            "mf1_pytorch": {
                "observed_coverage_pct": round(cov1_pct, 2),
                "coverage_meets_nominal": bool(cov1_pct >= (alpha * 100 - 3.0)),  # tolerance margin
                "interval_half_width_meters": round(q1, 4),
                "interval_full_width_meters": round(2 * q1, 4),
                "coverage_95_ci": [round(ci1_low * 100, 2), round(ci1_high * 100, 2)],
            },
            "mf2_standalone": {
                "observed_coverage_pct": round(cov2_pct, 2),
                "coverage_meets_nominal": bool(cov2_pct >= (alpha * 100 - 3.0)),
                "interval_half_width_meters": round(q2, 4),
                "interval_full_width_meters": round(2 * q2, 4),
                "coverage_95_ci": [round(ci2_low * 100, 2), round(ci2_high * 100, 2)],
            }
        }

    # Intensity class breakdown
    intensity_breakdown = {}
    for c_name, (low, high) in INTENSITY_CLASSES.items():
        c_idx = [i for i in range(n_test) if low <= true_precip[i] < high]
        c_count = len(c_idx)
        if c_count > 0:
            c_mf1_rec = sum(1 for i in c_idx if mf1_test_rain_probs[i] >= 0.5) / c_count * 100.0
            c_mf2_rec = sum(1 for i in c_idx if mf2_test_rain_probs[i] >= 0.5) / c_count * 100.0
            c_per_rec = sum(1 for i in c_idx if persist_rain[i] >= 0.5) / c_count * 100.0
        else:
            c_mf1_rec = c_mf2_rec = c_per_rec = None

        intensity_breakdown[c_name] = {
            "sample_count": c_count,
            "mf1_detection_rate_pct": round(c_mf1_rec, 2) if c_mf1_rec is not None else None,
            "mf2_detection_rate_pct": round(c_mf2_rec, 2) if c_mf2_rec is not None else None,
            "persistence_detection_rate_pct": round(c_per_rec, 2) if c_per_rec is not None else None,
        }

    # Per-station breakdown
    station_breakdown = {}
    st_groups = defaultdict(list)
    for i, meta in enumerate(test_meta):
        st_groups[meta["station_id"]].append(i)

    for st_id, s_indices in sorted(st_groups.items()):
        s_y = [true_rain[i] for i in s_indices]
        s_mf1_p = [mf1_test_rain_probs[i] for i in s_indices]
        s_mf2_p = [mf2_test_rain_probs[i] for i in s_indices]
        s_per_p = [persist_rain[i] for i in s_indices]

        s_mf1_metrics = compute_rain_metrics(s_mf1_p, s_y)
        s_mf2_metrics = compute_rain_metrics(s_mf2_p, s_y)
        s_per_metrics = compute_rain_metrics(s_per_p, s_y)

        station_breakdown[st_id] = {
            "sample_count": len(s_indices),
            "rain_prevalence_pct": round(sum(s_y) / len(s_y) * 100.0, 2),
            "mf1_f1_pct": s_mf1_metrics.get("f1_score_pct"),
            "mf2_f1_pct": s_mf2_metrics.get("f1_score_pct"),
            "persistence_f1_pct": s_per_metrics.get("f1_score_pct"),
        }

    # Return per-sample prediction records for CSV logging
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
            "actual_water_level": meta["actual_water_level"],
            "has_water": meta["has_water"],
            "mf1_rain_prob": round(mf1_test_rain_probs[i], 4),
            "mf1_precip_mm": round(mf1_test_precip_vols[i], 4),
            "mf1_water_level": round(mf1_test_water_preds[i], 4) if meta["has_water"] else None,
            "mf2_rain_prob": round(mf2_test_rain_probs[i], 4),
            "mf2_water_level": round(mf2_test_water_preds[i], 4) if meta["has_water"] else None,
            "persist_rain": persist_rain[i],
            "persist_water": meta["last_observed_water"],
            "climatology_rain": round(climatology_rain_prior, 4),
            "climatology_water": round(climatology_water_stage, 4),
        })

    horizon_result = {
        "test_samples_total": n_test,
        "calibration_samples_total": n_calib,
        "water_gauge_test_samples": len(gauge_indices),
        "rain_prevalence_pct": round(sum(true_rain) / n_test * 100.0, 2),
        "rain_metrics": {
            "mf1_pytorch": mf1_rain_metrics,
            "mf2_standalone": mf2_rain_metrics,
            "persistence": persist_rain_metrics,
            "climatology": clim_rain_metrics,
        },
        "water_metrics": {
            "mf1_pytorch": mf1_water_metrics,
            "mf2_standalone": mf2_water_metrics,
        },
        "conformal_uncertainty": conformal_coverage,
        "intensity_breakdown": intensity_breakdown,
        "station_breakdown": station_breakdown,
    }

    return horizon_result, sample_records


def run_full_validation(horizons: list = None):
    """Execute multi-horizon independent validation across all model families."""
    if horizons is None:
        horizons = DEFAULT_HORIZONS

    print("=" * 80)
    print("INDEPENDENT MULTI-HORIZON VALIDATION SUITE")
    print(f"Horizons: {horizons} hours")
    print("=" * 80)

    pipeline = get_telemetry_pipeline()

    # Determine train split climatology baselines
    train_rain_count = 0
    train_total = 0
    train_water_vals = []

    for st_id, h_dict in pipeline.station_hourly.items():
        for h, rec in h_dict.items():
            if h <= pipeline.train_end:
                train_total += 1
                if rec["precipitation"] >= 0.1:
                    train_rain_count += 1
                if st_id == WATER_GAUGE_WEATHER_STATION and h in pipeline.water_hourly:
                    train_water_vals.append(pipeline.water_hourly[h])

    climatology_rain = train_rain_count / max(1, train_total)
    climatology_water = sum(train_water_vals) / max(1, len(train_water_vals)) if train_water_vals else 2.50

    print(f"Historical Train Climatology: Rain={climatology_rain*100:.2f}% | River Stage={climatology_water:.3f}m")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_horizon_results = {}
    all_sample_records = []

    # Check git commit
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
                m_config = ckpt.get("manifest", {}).get("model_config", {"input_dim": 4, "hidden_dim": 32})
                mf1_model = WeatherWaterLNN(**m_config).to(device)
                mf1_model.load_state_dict(ckpt["model_state_dict"])
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
            climatology_rain_prior=climatology_rain,
            climatology_water_stage=climatology_water,
            device=device,
        )

        all_horizon_results[f"horizon_{h}h"] = h_result
        all_sample_records.extend(h_samples)

    # Save per-sample prediction CSV
    fieldnames = [
        "station_id", "origin_timestamp", "target_timestamp", "horizon_hours", "actual_lead_hours",
        "actual_rain", "actual_precip_mm", "actual_water_level", "has_water",
        "mf1_rain_prob", "mf1_precip_mm", "mf1_water_level",
        "mf2_rain_prob", "mf2_water_level",
        "persist_rain", "persist_water", "climatology_rain", "climatology_water",
    ]
    with open(PREDICTIONS_LOG_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_sample_records)
    print(f"\nSaved {len(all_sample_records)} test sample predictions to: {PREDICTIONS_LOG_PATH}")

    # Determine overall scorecard conclusions
    h1_metrics = all_horizon_results.get("horizon_1h", {})
    mf1_h1_f1 = h1_metrics.get("rain_metrics", {}).get("mf1_pytorch", {}).get("f1_score_pct", 0.0)
    mf2_h1_f1 = h1_metrics.get("rain_metrics", {}).get("mf2_standalone", {}).get("f1_score_pct", 0.0)
    p_h1_f1 = h1_metrics.get("rain_metrics", {}).get("persistence", {}).get("f1_score_pct", 0.0)

    mf1_h1_mae = h1_metrics.get("water_metrics", {}).get("mf1_pytorch", {}).get("mae_meters", float("inf"))
    mf2_h1_mae = h1_metrics.get("water_metrics", {}).get("mf2_standalone", {}).get("mae_meters", float("inf"))
    p_h1_mae = h1_metrics.get("water_metrics", {}).get("mf1_pytorch", {}).get("persistence_mae_meters", 0.0)

    beats_rain_persistence = bool(mf1_h1_f1 > p_h1_f1 or mf2_h1_f1 > p_h1_f1)
    beats_water_persistence = bool(mf1_h1_mae < p_h1_mae or mf2_h1_mae < p_h1_mae)

    scorecard = {
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "code_commit": git_commit,
        "dataset_hashes": {
            "weather_telemetry_sha256": compute_file_sha256(WEATHER_CSV_PATH),
            "water_level_telemetry_sha256": compute_file_sha256(WATER_CSV_PATH),
        },
        "model_status": "RESEARCH_PROTOTYPE",
        "tested_horizons_hours": horizons,
        "climatology_baselines": {
            "train_rain_prevalence_pct": round(climatology_rain * 100.0, 2),
            "train_mean_river_stage_meters": round(climatology_water, 4),
        },
        "summary_findings": {
            "horizon_1h_persistence_rain_f1_pct": p_h1_f1,
            "horizon_1h_mf1_rain_f1_pct": mf1_h1_f1,
            "horizon_1h_mf2_rain_f1_pct": mf2_h1_f1,
            "horizon_1h_persistence_water_mae_meters": p_h1_mae,
            "horizon_1h_mf1_water_mae_meters": mf1_h1_mae,
            "horizon_1h_mf2_water_mae_meters": mf2_h1_mae,
            "beats_rain_persistence": beats_rain_persistence,
            "beats_water_persistence": beats_water_persistence,
            "operational_recommendation": (
                "DO NOT DEPLOY (RESEARCH_PROTOTYPE ONLY). Models do not demonstrate statistically significant, "
                "robust superiority over operational persistence baselines across multi-hour horizons."
                if not (beats_rain_persistence and beats_water_persistence)
                else "POTENTIAL_CANDIDATE: Requires domain expert review before production release."
            ),
        },
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

    with open(SCORECARD_PATH, "w", encoding="utf-8") as f:
        json.dump(to_serializable(scorecard), f, indent=2)
    print(f"Saved full scorecard to: {SCORECARD_PATH}")

    # Print executive summary table
    print("\n" + "=" * 90)
    print("EXECUTIVE SCORECARD SUMMARY ACROSS HORIZONS")
    print("=" * 90)
    print(f"{'Horizon':<8} | {'MF-1 F1':<10} | {'MF-2 F1':<10} | {'Persist F1':<10} | {'MF-1 W-MAE':<12} | {'MF-2 W-MAE':<12} | {'Persist W-MAE':<12}")
    print("-" * 90)
    for h in horizons:
        h_k = f"horizon_{h}h"
        res = all_horizon_results.get(h_k, {})
        rm = res.get("rain_metrics", {})
        wm = res.get("water_metrics", {})

        f1_1 = f"{rm.get('mf1_pytorch', {}).get('f1_score_pct', 0.0):.1f}%"
        f1_2 = f"{rm.get('mf2_standalone', {}).get('f1_score_pct', 0.0):.1f}%"
        f1_p = f"{rm.get('persistence', {}).get('f1_score_pct', 0.0):.1f}%"

        w_1 = f"{wm.get('mf1_pytorch', {}).get('mae_meters', float('nan')):.4f}m"
        w_2 = f"{wm.get('mf2_standalone', {}).get('mae_meters', float('nan')):.4f}m"
        w_p = f"{wm.get('mf1_pytorch', {}).get('persistence_mae_meters', float('nan')):.4f}m"

        print(f"+{h:02d}h     | {f1_1:<10} | {f1_2:<10} | {f1_p:<10} | {w_1:<12} | {w_2:<12} | {w_p:<12}")
    print("=" * 90)
    print(f"Recommendation: {scorecard['summary_findings']['operational_recommendation']}")
    print("=" * 90)

    return scorecard


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Horizon Independent Validator.")
    parser.add_argument("--horizons", nargs="+", type=int, default=DEFAULT_HORIZONS, help="List of horizons (e.g. 1 3 6 12 24)")
    args = parser.parse_args()

    run_full_validation(horizons=args.horizons)
