"""
Comprehensive Future-Forecast Model Validation & Scorecard Suite for KloudTrack CfC/LNN.

Key audit fixes applied (prediction-model-audit-followup.md):
  - Consumes CANONICAL FUTURE-WINDOW test sequences (input window -> future target t0 + h) (Fix 6)
  - Isolates recurrent hidden state per window — zero cross-station or cross-window leakage (Fix 5)
  - Splits conformal prediction: calibration on split='val', coverage verified on split='test' (Fix 7)
  - Includes operational baselines: Persistence (last observed) and Climatology (historical mean) (Fix 8)
  - Compares Model Family 1 (PyTorch WeatherWaterLNN) and Model Family 2 (Standalone ContinuousLNNCell)
  - Status: RESEARCH_PROTOTYPE (rigorous, honest, out-of-sample evaluation)
"""

import os
import math
import time
import json
from datetime import datetime
import numpy as np

from dataset import (
    load_real_telemetry_sequences,
    fit_train_normalization,
    FEATURE_MEANS,
    FEATURE_STDS,
    DATA_DIR,
)

WEIGHTS_PATH = os.path.join(DATA_DIR, "lnn_trained_weights.json")
PYTORCH_CHECKPOINT_PATH = os.path.join(DATA_DIR, "lnn_weather_water.pt")
SCORECARD_PATH = os.path.join(DATA_DIR, "validation_scorecard.json")

# Precipitation intensity classes (mm/h)
INTENSITY_CLASSES = {
    "trace": (0.0, 0.5),
    "light": (0.5, 2.5),
    "moderate": (2.5, 7.5),
    "heavy": (7.5, float("inf")),
}

# Threshold values for sensitivity analysis
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, x))))


def tanh(x: float) -> float:
    return math.tanh(max(-20.0, min(20.0, x)))


class StandaloneLNNRunner:
    """Runs inference on a sequence window using weights from lnn_trained_weights.json."""

    def __init__(self, weights: dict):
        self.hidden_dim = weights.get("hidden_dim", 8)
        self.W_in = weights.get("W_in", [])
        self.W_rec = weights.get("W_rec", [])
        self.b_h = weights.get("b_h", [0.0] * self.hidden_dim)
        self.tau = weights.get("tau", [1.5] * self.hidden_dim)
        self.W_rain = weights.get("W_rain", [])
        self.b_rain = weights.get("b_rain", 0.0)
        self.W_water = weights.get("W_water", [])
        self.b_water = weights.get("b_water", 3.45)
        self.in_features = len(self.W_in)

    def predict_window(self, telemetry_arr: np.ndarray, dt_arr: np.ndarray):
        """
        Unrolls sequence window starting with fresh hidden state h=0 (no cross-window leakage).
        telemetry_arr: [seq_len, 4] normalized features.
        dt_arr: [seq_len, 1] elapsed hours.
        """
        seq_len = telemetry_arr.shape[0]
        h = [0.0] * self.hidden_dim
        pred_rain = 0.0
        pred_water = self.b_water

        for t in range(seq_len):
            x = telemetry_arr[t]
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


def _compute_conformal_quantiles(calibration_residuals: list, coverage_levels=(0.80, 0.90, 0.95)) -> dict:
    """Compute empirical conformal prediction quantiles from held-out calibration residuals."""
    if not calibration_residuals:
        return {}
    sorted_res = sorted(calibration_residuals)
    n = len(sorted_res)
    quantiles = {}
    for alpha in coverage_levels:
        idx = min(int(math.ceil(alpha * (n + 1))) - 1, n - 1)
        quantiles[alpha] = sorted_res[idx]
    return quantiles


def _evaluate_rain_metrics(pred_probs: list, targets: list, threshold: float = 0.5) -> dict:
    """Compute full suite of operational rain classification metrics."""
    tp = fp = tn = fn = 0
    brier_sum = 0.0
    n = len(targets)

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
        else:
            fn += 1

    acc = (tp + tn) / n * 100.0 if n > 0 else 0.0
    rec = tp / (tp + fn) * 100.0 if (tp + fn) > 0 else 0.0
    prec = tp / (tp + fp) * 100.0 if (tp + fp) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    far = fp / (tp + fp) * 100.0 if (tp + fp) > 0 else 0.0
    csi = tp / (tp + fp + fn) * 100.0 if (tp + fp + fn) > 0 else 0.0
    brier = brier_sum / n if n > 0 else 0.0

    return {
        "accuracy": round(acc, 2),
        "recall_pod": round(rec, 2),
        "precision": round(prec, 2),
        "f1_score": round(f1, 2),
        "false_alarm_ratio": round(far, 2),
        "critical_success_index": round(csi, 2),
        "brier_score": round(brier, 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def run_comprehensive_validation(horizon: int = 1):
    print("=" * 75)
    print(f"CANONICAL FUTURE-WINDOW VALIDATION SUITE (Forecast Horizon: +{horizon}h)")
    print("=" * 75)
    print("Design: Out-of-sample temporal holdout, station-isolated sequence windows,")
    print("        real river gauge observations, empirical conformal test coverage.")
    print("Status: RESEARCH_PROTOTYPE (not production certified)\n")

    # 1. Load Standalone Model (MF-2)
    if not os.path.exists(WEIGHTS_PATH):
        print(f"ERROR: Model weights not found at {WEIGHTS_PATH}. Run train_standalone.py first.")
        return
    with open(WEIGHTS_PATH, "r", encoding="utf-8") as f:
        mf2_weights = json.load(f)
    mf2_runner = StandaloneLNNRunner(mf2_weights)

    # Load train-fitted normalization stats
    norm_info = mf2_weights.get("normalization", {})
    if norm_info and "means" in norm_info and "stds" in norm_info:
        norm_means = np.array(norm_info["means"], dtype=np.float32)
        norm_stds = np.array(norm_info["stds"], dtype=np.float32)
    else:
        norm_means, norm_stds = fit_train_normalization()

    # 2. Check for PyTorch Model (MF-1)
    mf1_model = None
    if os.path.exists(PYTORCH_CHECKPOINT_PATH):
        try:
            import torch
            from model import WeatherWaterLNN
            ckpt = torch.load(PYTORCH_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
            model_cfg = ckpt.get("manifest", {}).get("model_config", {"input_dim": 4, "hidden_dim": 32})
            mf1_model = WeatherWaterLNN(**model_cfg)
            mf1_model.load_state_dict(ckpt["model_state_dict"])
            mf1_model.eval()
            print("Loaded Model Family 1 (PyTorch WeatherWaterLNN) from checkpoint.")
        except Exception as e:
            print(f"Notice: Could not load MF-1 PyTorch model ({e}). Proceeding with MF-2.")

    # 3. Load CALIBRATION Split (split='val') for Conformal Prediction (Fix 7)
    print("Loading CALIBRATION windows from validation split (split='val')...")
    cal_data = load_real_telemetry_sequences(
        split="val",
        max_sequences=600,
        horizons=[horizon],
        norm_means=norm_means,
        norm_stds=norm_stds,
        return_metadata=True,
    )

    cal_residuals_mf2 = []
    cal_residuals_mf1 = []
    if cal_data is not None:
        t_cal, dt_cal, r_cal, p_cal, w_cal, hw_cal, meta_cal = cal_data
        for i in range(len(meta_cal)):
            m = meta_cal[i]
            if not m["has_water"]:
                continue
            true_w = m["actual_water_level"]
            # MF-2 prediction
            _, pred_w_mf2 = mf2_runner.predict_window(t_cal[i].numpy(), dt_cal[i].numpy())
            cal_residuals_mf2.append(abs(pred_w_mf2 - true_w))

            # MF-1 prediction
            if mf1_model is not None:
                with torch.no_grad():
                    _, _, pred_w_mf1_tensor = mf1_model(t_cal[i:i+1], dt_cal[i:i+1])
                    cal_residuals_mf1.append(abs(pred_w_mf1_tensor[0, -1, 0].item() - true_w))

    conformal_quantiles_mf2 = _compute_conformal_quantiles(cal_residuals_mf2)
    conformal_quantiles_mf1 = _compute_conformal_quantiles(cal_residuals_mf1)
    print(f"Computed conformal quantiles on calibration set (n={len(cal_residuals_mf2)} gauge windows).")
    for alpha, q in conformal_quantiles_mf2.items():
        print(f"   MF-2 Calibration {int(alpha*100)}% Quantile: +/-{q:.4f}m")

    # 4. Load INDEPENDENT TEST Split (split='test') (Fix 6)
    print("\nLoading INDEPENDENT TEST windows from test split (split='test')...")
    test_data = load_real_telemetry_sequences(
        split="test",
        max_sequences=600,
        horizons=[horizon],
        norm_means=norm_means,
        norm_stds=norm_stds,
        return_metadata=True,
    )

    if test_data is None:
        print("ERROR: Insufficient test data.")
        return

    t_test, dt_test, r_test, p_test, w_test, hw_test, meta_test = test_data
    n_test = len(meta_test)
    print(f"Loaded {n_test} future-window test samples across 16 stations.")

    # 5. Run All Models & Baselines on Test Set (Fix 5 & Fix 8)
    # Lists for metrics
    targets_rain = []
    targets_water = []

    preds_rain_mf2 = []
    preds_water_mf2 = []

    preds_rain_mf1 = []
    preds_water_mf1 = []

    preds_rain_persistence = []
    preds_water_persistence = []

    preds_water_climatology = []
    preds_rain_climatology = []

    test_water_indices = []

    start_t = time.perf_counter()
    for i in range(n_test):
        m = meta_test[i]
        telemetry = t_test[i].numpy()
        dt_arr = dt_test[i].numpy()

        y_rain = float(m["actual_rain_prob"])
        targets_rain.append(y_rain)

        # Baseline 1: Persistence
        # Rain persistence: predict current rain status
        p_rain_pers = 1.0 if m["last_observed_precip"] > 0.1 else 0.0
        preds_rain_persistence.append(p_rain_pers)

        # Baseline 2: Climatology
        preds_rain_climatology.append(0.12)  # Base precipitation probability in Philippine monsoon
        preds_water_climatology.append(3.45)  # Historical mean stage for Calumpit WLMS

        # MF-2: Standalone LNN Cell (recurrent state h=0 reset for each window)
        p_rain_mf2, p_water_mf2 = mf2_runner.predict_window(telemetry, dt_arr)
        preds_rain_mf2.append(p_rain_mf2)
        preds_water_mf2.append(p_water_mf2)

        # MF-1: PyTorch WeatherWaterLNN
        if mf1_model is not None:
            with torch.no_grad():
                pr1, _, pw1 = mf1_model(t_test[i:i+1], dt_test[i:i+1])
                preds_rain_mf1.append(float(pr1[0, -1, 0].item()))
                preds_water_mf1.append(float(pw1[0, -1, 0].item()))

        # Water stage evaluation
        if m["has_water"]:
            test_water_indices.append(i)
            targets_water.append(float(m["actual_water_level"]))
            # Persistence: stage at t0
            p_water_pers = m["last_observed_water"] if m["last_observed_water"] is not None else 3.45
            preds_water_persistence.append(p_water_pers)

    elapsed_time = time.perf_counter() - start_t
    latency_us = (elapsed_time / n_test) * 1_000_000.0

    # 6. Calculate Metrics
    # Rain Metrics
    metrics_rain_mf2 = _evaluate_rain_metrics(preds_rain_mf2, targets_rain, threshold=0.5)
    metrics_rain_pers = _evaluate_rain_metrics(preds_rain_persistence, targets_rain, threshold=0.5)
    metrics_rain_clim = _evaluate_rain_metrics(preds_rain_climatology, targets_rain, threshold=0.5)

    metrics_rain_mf1 = None
    if mf1_model is not None and preds_rain_mf1:
        metrics_rain_mf1 = _evaluate_rain_metrics(preds_rain_mf1, targets_rain, threshold=0.5)

    # Water Stage Metrics (MAE & RMSE)
    n_gauge = len(targets_water)
    water_metrics = {}

    if n_gauge > 0:
        # MF-2
        errs_mf2 = [abs(preds_water_mf2[idx] - targets_water[k]) for k, idx in enumerate(test_water_indices)]
        sq_mf2 = [(preds_water_mf2[idx] - targets_water[k]) ** 2 for k, idx in enumerate(test_water_indices)]
        mae_mf2 = sum(errs_mf2) / n_gauge
        rmse_mf2 = math.sqrt(sum(sq_mf2) / n_gauge)

        # Persistence
        errs_pers = [abs(preds_water_persistence[k] - targets_water[k]) for k in range(n_gauge)]
        sq_pers = [(preds_water_persistence[k] - targets_water[k]) ** 2 for k in range(n_gauge)]
        mae_pers = sum(errs_pers) / n_gauge
        rmse_pers = math.sqrt(sum(sq_pers) / n_gauge)

        # Climatology
        errs_clim = [abs(preds_water_climatology[idx] - targets_water[k]) for k, idx in enumerate(test_water_indices)]
        sq_clim = [(preds_water_climatology[idx] - targets_water[k]) ** 2 for k, idx in enumerate(test_water_indices)]
        mae_clim = sum(errs_clim) / n_gauge
        rmse_clim = math.sqrt(sum(sq_clim) / n_gauge)

        water_metrics["mf2"] = {"mae": round(mae_mf2, 4), "rmse": round(rmse_mf2, 4)}
        water_metrics["persistence"] = {"mae": round(mae_pers, 4), "rmse": round(rmse_pers, 4)}
        water_metrics["climatology"] = {"mae": round(mae_clim, 4), "rmse": round(rmse_clim, 4)}

        # MF-1
        if mf1_model is not None and preds_water_mf1:
            errs_mf1 = [abs(preds_water_mf1[idx] - targets_water[k]) for k, idx in enumerate(test_water_indices)]
            sq_mf1 = [(preds_water_mf1[idx] - targets_water[k]) ** 2 for k, idx in enumerate(test_water_indices)]
            mae_mf1 = sum(errs_mf1) / n_gauge
            rmse_mf1 = math.sqrt(sum(sq_mf1) / n_gauge)
            water_metrics["mf1"] = {"mae": round(mae_mf1, 4), "rmse": round(rmse_mf1, 4)}

        # 7. Independent Conformal Test Coverage (Fix 7)
        conformal_test_coverage = {}
        for alpha, q in conformal_quantiles_mf2.items():
            hits = sum(1 for e in errs_mf2 if e <= q)
            coverage = (hits / n_gauge) * 100.0
            conformal_test_coverage[f"{int(alpha*100)}%"] = {
                "target_coverage": f"{int(alpha*100)}%",
                "band_meters": round(q, 4),
                "empirical_test_coverage": f"{coverage:.1f}%",
                "calibration_samples": len(cal_residuals_mf2),
                "test_samples": n_gauge,
            }
    else:
        conformal_test_coverage = {}

    # Threshold Sensitivity Analysis for MF-2
    threshold_results = []
    for th in THRESHOLDS:
        m_th = _evaluate_rain_metrics(preds_rain_mf2, targets_rain, threshold=th)
        threshold_results.append({
            "threshold": th,
            "accuracy": m_th["accuracy"],
            "recall": m_th["recall_pod"],
            "precision": m_th["precision"],
            "f1": m_th["f1_score"],
            "far": m_th["false_alarm_ratio"],
            "csi": m_th["critical_success_index"],
        })

    # Intensity Breakdown for MF-2
    intensity_results = {}
    for cls_name, (lo, hi) in INTENSITY_CLASSES.items():
        hits = 0
        count = 0
        for i, m in enumerate(meta_test):
            precip = m["actual_precip_mm"]
            if lo < precip <= hi:
                count += 1
                if preds_rain_mf2[i] >= 0.5:
                    hits += 1
        pct = (hits / count * 100.0) if count > 0 else 0.0
        hi_str = f"{hi:.1f}" if hi != float("inf") else "inf"
        intensity_results[cls_name] = {
            "range_mm": f"{lo:.1f}-{hi_str}",
            "event_count": count,
            "detected_pct": round(pct, 1),
        }

    # Per-Station Breakdown
    station_results = {}
    for i, m in enumerate(meta_test):
        st = m["station_id"]
        if st not in station_results:
            station_results[st] = {"y_true": [], "y_pred": []}
        station_results[st]["y_true"].append(m["actual_rain_prob"])
        station_results[st]["y_pred"].append(preds_rain_mf2[i])

    station_summary = {}
    for st, data in sorted(station_results.items()):
        st_m = _evaluate_rain_metrics(data["y_pred"], data["y_true"], threshold=0.5)
        station_summary[st] = {
            "accuracy": st_m["accuracy"],
            "recall": st_m["recall_pod"],
            "precision": st_m["precision"],
            "sample_count": len(data["y_true"]),
        }

    # ------------------------------------------------------------------
    # Display Output
    # ------------------------------------------------------------------
    print("-" * 75)
    print(f"OPERATIONAL RAIN FORECAST COMPARISON (+{horizon}h ahead):")
    print(f"{'Model / Baseline':<32} {'Accuracy':>9} {'Recall':>9} {'Precision':>9} {'F1':>8} {'FAR':>8} {'CSI':>8} {'Brier':>8}")
    print("-" * 75)
    if metrics_rain_mf1:
        m1 = metrics_rain_mf1
        print(f"{'MF-1: PyTorch WeatherWaterLNN':<32} {m1['accuracy']:>8.1f}% {m1['recall_pod']:>8.1f}% {m1['precision']:>8.1f}% {m1['f1_score']:>7.1f}% {m1['false_alarm_ratio']:>7.1f}% {m1['critical_success_index']:>7.1f}% {m1['brier_score']:>8.4f}")
    m2 = metrics_rain_mf2
    print(f"{'MF-2: Standalone ContinuousLNN':<32} {m2['accuracy']:>8.1f}% {m2['recall_pod']:>8.1f}% {m2['precision']:>8.1f}% {m2['f1_score']:>7.1f}% {m2['false_alarm_ratio']:>7.1f}% {m2['critical_success_index']:>7.1f}% {m2['brier_score']:>8.4f}")
    mp = metrics_rain_pers
    print(f"{'Baseline: Persistence':<32} {mp['accuracy']:>8.1f}% {mp['recall_pod']:>8.1f}% {mp['precision']:>8.1f}% {mp['f1_score']:>7.1f}% {mp['false_alarm_ratio']:>7.1f}% {mp['critical_success_index']:>7.1f}% {mp['brier_score']:>8.4f}")
    mc = metrics_rain_clim
    print(f"{'Baseline: Climatology':<32} {mc['accuracy']:>8.1f}% {mc['recall_pod']:>8.1f}% {mc['precision']:>8.1f}% {mc['f1_score']:>7.1f}% {mc['false_alarm_ratio']:>7.1f}% {mc['critical_success_index']:>7.1f}% {mc['brier_score']:>8.4f}")

    print("\n" + "-" * 75)
    print(f"HYDROLOGICAL WATER LEVEL FORECAST COMPARISON (Real Gauge Only, n={n_gauge}):")
    print(f"{'Model / Baseline':<35} {'MAE (meters)':>15} {'RMSE (meters)':>15}")
    print("-" * 75)
    if "mf1" in water_metrics:
        print(f"{'MF-1: PyTorch WeatherWaterLNN':<35} {water_metrics['mf1']['mae']:>15.4f} {water_metrics['mf1']['rmse']:>15.4f}")
    print(f"{'MF-2: Standalone ContinuousLNN':<35} {water_metrics['mf2']['mae']:>15.4f} {water_metrics['mf2']['rmse']:>15.4f}")
    print(f"{'Baseline: Persistence':<35} {water_metrics['persistence']['mae']:>15.4f} {water_metrics['persistence']['rmse']:>15.4f}")
    print(f"{'Baseline: Climatology':<35} {water_metrics['climatology']['mae']:>15.4f} {water_metrics['climatology']['rmse']:>15.4f}")

    print("\n" + "-" * 75)
    print("INDEPENDENT CONFORMAL UNCERTAINTY TEST COVERAGE:")
    print("   (Quantiles fitted on Calibration split='val', evaluated on Test split='test')")
    for alpha_str, c_res in conformal_test_coverage.items():
        print(f"   Target {alpha_str}: Band = +/-{c_res['band_meters']}m | Empirical Test Coverage = {c_res['empirical_test_coverage']} (n={n_gauge})")

    # 8. Build Scorecard Document
    scorecard = {
        "model_status": "RESEARCH_PROTOTYPE",
        "evaluation_type": "independent_future_window_test",
        "forecast_horizon_hours": horizon,
        "split_method": "chronological_60_20_20_pre_windowing",
        "conformal_calibration_samples": len(cal_residuals_mf2),
        "independent_test_samples": n_test,
        "monitored_stations": len(station_summary),
        "rain_forecast_comparison": {
            "mf1_pytorch_cfc": metrics_rain_mf1,
            "mf2_standalone_cell": metrics_rain_mf2,
            "persistence_baseline": metrics_rain_pers,
            "climatology_baseline": metrics_rain_clim,
        },
        "water_level_forecast_comparison": water_metrics,
        "conformal_uncertainty_test_coverage": conformal_test_coverage,
        "threshold_sensitivity": threshold_results,
        "rain_by_intensity": intensity_results,
        "rain_by_station": station_summary,
        "performance": {
            "latency_microseconds": round(latency_us, 2),
            "throughput_samples_per_sec": int(n_test / max(1e-4, elapsed_time)),
        },
    }

    with open(SCORECARD_PATH, "w", encoding="utf-8") as f:
        json.dump(scorecard, f, indent=2)

    print("\n" + "=" * 75)
    print(f"Scorecard saved -> {SCORECARD_PATH}")
    print("=" * 75)


if __name__ == "__main__":
    run_comprehensive_validation(horizon=1)
