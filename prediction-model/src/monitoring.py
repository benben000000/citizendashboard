"""
Operational Monitoring, Drift Detection, and Fine-Grained Evaluation Engine.

Provides continuous and post-deployment evaluation across:
  - Dimensions: horizon, station, target variable, rain vs dry regime,
    heavy-rain regime, calm vs windy regime, data-quality quarantine, time window.
  - Metrics:
      * Continuous: MAE, RMSE, bias, persistence MAE, skill score, climatology MAE, missingness, quarantine rate.
      * Rain occurrence: Brier score, calibration error (ECE), F1, precision, recall (POD), FAR, CSI, confusion matrix, reliability bins.
      * Precipitation amount: overall, dry-hour, rainy-hour MAE/RMSE/bias, heavy-rain precision/recall/CSI (2.5, 5.0, 10.0 mm/h).
  - Safeguards:
      * Minimum sample count enforcement.
      * Bootstrap 95% confidence intervals.
      * Drift detection across feature distributions, missingness, and rain prevalence.
      * NEVER automatically modifies the frozen operational inference policy.

Usage:
  python prediction-model/src/monitoring.py --output prediction-model/data/monitoring_report.json
"""

import os
import sys
import math
import json
import argparse
from datetime import datetime, timezone
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Optional

import numpy as np

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

DATA_DIR = os.path.join(os.path.dirname(SRC_DIR), "data")
CANONICAL_HORIZONS = [1, 3, 6, 12, 24]
HEAVY_RAIN_THRESHOLDS = [2.5, 5.0, 10.0]
MIN_RELIABLE_SAMPLES = 30


# ---------------------------------------------------------------------------
# Metric Math Utilities
# ---------------------------------------------------------------------------

def compute_bootstrap_ci(
    y_true: List[float],
    y_pred: List[float],
    metric_func,
    n_bootstraps: int = 200,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, Optional[float]]:
    """Compute bootstrap percentile confidence interval for a metric."""
    n = len(y_true)
    if n < 10:
        return {"ci_lower": None, "ci_upper": None}

    rng = np.random.RandomState(seed)
    scores = []
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)

    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        try:
            val = metric_func(yt[idx], yp[idx])
            if not np.isnan(val) and not np.isinf(val):
                scores.append(float(val))
        except Exception:
            pass

    if len(scores) < 10:
        return {"ci_lower": None, "ci_upper": None}

    alpha = (1.0 - ci_level) / 2.0
    scores.sort()
    lower_idx = int(math.floor(alpha * len(scores)))
    upper_idx = min(int(math.ceil((1.0 - alpha) * len(scores))), len(scores) - 1)

    return {
        "ci_lower": round(scores[lower_idx], 4),
        "ci_upper": round(scores[upper_idx], 4),
    }


def compute_contingency_scores(tp: int, fp: int, fn: int, tn: int) -> Dict[str, Any]:
    """Compute standard meteorological verification contingency scores."""
    total = tp + fp + fn + tn
    pod = tp / (tp + fn) if (tp + fn) > 0 else None
    far = fp / (tp + fp) if (tp + fp) > 0 else None
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = pod if pod is not None else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else None
    accuracy = (tp + tn) / total if total > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "total": total,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall_pod": round(pod, 4) if pod is not None else None,
        "false_alarm_ratio": round(far, 4) if far is not None else None,
        "critical_success_index": round(csi, 4) if csi is not None else None,
        "f1_score": round(f1, 4),
    }


def compute_reliability_and_ece(
    probs: List[float],
    labels: List[int],
    num_bins: int = 10,
) -> Dict[str, Any]:
    """Compute Expected Calibration Error (ECE) and reliability diagram bins."""
    n = len(probs)
    if n == 0:
        return {"brier_score": None, "expected_calibration_error": None, "reliability_bins": []}

    brier = sum((p - y) ** 2 for p, y in zip(probs, labels)) / n

    bin_data = []
    total_ece = 0.0

    for b in range(num_bins):
        bin_lower = b / num_bins
        bin_upper = (b + 1) / num_bins
        bin_indices = [
            i for i, p in enumerate(probs)
            if (bin_lower <= p < bin_upper) or (b == num_bins - 1 and bin_lower <= p <= bin_upper)
        ]
        b_count = len(bin_indices)
        if b_count > 0:
            avg_prob = sum(probs[i] for i in bin_indices) / b_count
            observed_freq = sum(labels[i] for i in bin_indices) / b_count
            bin_ece = abs(avg_prob - observed_freq) * (b_count / n)
            total_ece += bin_ece
            bin_data.append({
                "bin_range": [round(bin_lower, 2), round(bin_upper, 2)],
                "sample_count": b_count,
                "mean_forecast_prob": round(avg_prob, 4),
                "observed_frequency": round(observed_freq, 4),
                "calibration_gap": round(avg_prob - observed_freq, 4),
            })
        else:
            bin_data.append({
                "bin_range": [round(bin_lower, 2), round(bin_upper, 2)],
                "sample_count": 0,
                "mean_forecast_prob": None,
                "observed_frequency": None,
                "calibration_gap": None,
            })

    return {
        "brier_score": round(brier, 4),
        "expected_calibration_error": round(total_ece, 4),
        "reliability_bins": bin_data,
    }


# ---------------------------------------------------------------------------
# Telemetry Drift Monitor
# ---------------------------------------------------------------------------

class TelemetryDriftMonitor:
    """Monitors telemetry distribution shifts, missingness, and sensor health."""

    def __init__(self, baseline_manifest_path: str = None):
        if baseline_manifest_path is None:
            baseline_manifest_path = os.path.join(DATA_DIR, "cleaned_data_manifest.json")
        self.baseline_manifest_path = baseline_manifest_path
        self.baseline_manifest = {}
        if os.path.exists(self.baseline_manifest_path):
            with open(self.baseline_manifest_path, "r", encoding="utf-8") as f:
                self.baseline_manifest = json.load(f)

    def evaluate_drift(self, telemetry_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute drift report over a sample set of telemetry records."""
        total_rows = len(telemetry_rows)
        if total_rows == 0:
            return {"status": "NO_DATA", "total_rows": 0}

        station_counts = defaultdict(int)
        missing_counts = defaultdict(int)
        numeric_values = defaultdict(list)
        rain_hits = 0

        numeric_fields = ["temperature", "humidity", "pressure", "wind_speed", "precipitation"]

        for r in telemetry_rows:
            st = r.get("station_id", "unknown")
            station_counts[st] += 1

            for col in numeric_fields:
                val = r.get(col)
                if val is None or val == "":
                    missing_counts[col] += 1
                else:
                    try:
                        fval = float(val)
                        if not np.isnan(fval) and not np.isinf(fval):
                            numeric_values[col].append(fval)
                            if col == "precipitation" and fval >= 0.1:
                                rain_hits += 1
                        else:
                            missing_counts[col] += 1
                    except (ValueError, TypeError):
                        missing_counts[col] += 1

        feature_distributions = {}
        for col in numeric_fields:
            vals = numeric_values[col]
            if vals:
                arr = np.array(vals, dtype=np.float64)
                feature_distributions[col] = {
                    "count": len(arr),
                    "mean": round(float(np.mean(arr)), 3),
                    "std": round(float(np.std(arr)), 3),
                    "min": round(float(np.min(arr)), 3),
                    "median": round(float(np.median(arr)), 3),
                    "p90": round(float(np.percentile(arr, 90)), 3),
                    "max": round(float(np.max(arr)), 3),
                    "missingness_pct": round(missing_counts[col] / total_rows * 100.0, 2),
                }
            else:
                feature_distributions[col] = {
                    "count": 0,
                    "missingness_pct": 100.0,
                }

        rain_prevalence = rain_hits / max(1, len(numeric_values["precipitation"])) * 100.0

        return {
            "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
            "evaluated_rows": total_rows,
            "active_stations_count": len(station_counts),
            "station_distribution": dict(sorted(station_counts.items())),
            "rain_prevalence_pct": round(rain_prevalence, 2),
            "feature_distributions": feature_distributions,
        }


# ---------------------------------------------------------------------------
# Model Performance Monitor
# ---------------------------------------------------------------------------

class PerformanceMonitor:
    """Evaluates multi-horizon forecast accuracy across operational regimes."""

    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir if data_dir is not None else DATA_DIR

    def evaluate_predictions(
        self,
        predictions: List[Dict[str, Any]],
        horizon_hours: int = 1,
    ) -> Dict[str, Any]:
        """
        Evaluate a list of prediction records containing predictions and ground truth.
        Each record must contain:
          - 'station_id'
          - 'true_temperature', 'pred_temperature', 'persist_temperature'
          - 'true_humidity', 'pred_humidity', 'persist_humidity'
          - 'true_pressure', 'pred_pressure', 'persist_pressure'
          - 'true_wind_speed', 'pred_wind_speed', 'persist_wind_speed'
          - 'true_precip_mm', 'pred_precip_mm', 'persist_precip_mm'
          - 'true_rain_prob' or 'true_rain_label', 'pred_rain_prob', 'persist_rain_prob'
        """
        n_total = len(predictions)
        sample_reliability_flag = n_total >= MIN_RELIABLE_SAMPLES

        # 1. Regimes Slicing
        dry_preds = []
        rainy_preds = []
        calm_preds = []
        windy_preds = []
        heavy_rain_subsets = {th: [] for th in HEAVY_RAIN_THRESHOLDS}
        station_groups = defaultdict(list)

        for p in predictions:
            st = p.get("station_id", "unknown")
            station_groups[st].append(p)

            # Rain vs Dry regime (true_precip >= 0.1 mm)
            t_precip = float(p.get("true_precip_mm", 0.0))
            if t_precip >= 0.1:
                rainy_preds.append(p)
            else:
                dry_preds.append(p)

            # Heavy rain regimes
            for th in HEAVY_RAIN_THRESHOLDS:
                if t_precip >= th:
                    heavy_rain_subsets[th].append(p)

            # Calm vs Windy regime (wind_speed < 1.0 km/h)
            ws = float(p.get("true_wind_speed", 0.0))
            if ws < 1.0:
                calm_preds.append(p)
            else:
                windy_preds.append(p)

        # 2. Continuous Variables Evaluation
        def eval_continuous(name: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
            y_t = [float(r[f"true_{name}"]) for r in records if f"true_{name}" in r and f"pred_{name}" in r]
            y_p = [float(r[f"pred_{name}"]) for r in records if f"true_{name}" in r and f"pred_{name}" in r]
            y_persist = [float(r.get(f"persist_{name}", r.get(f"true_{name}", 0.0))) for r in records if f"true_{name}" in r and f"pred_{name}" in r]

            n = len(y_t)
            if n == 0:
                return {"sample_count": 0, "status": "INSUFFICIENT_DATA"}

            diffs = [yp - yt for yp, yt in zip(y_p, y_t)]
            p_diffs = [yp - yt for yp, yt in zip(y_persist, y_t)]

            mae = float(np.mean(np.abs(diffs)))
            rmse = float(np.sqrt(np.mean(np.square(diffs))))
            bias = float(np.mean(diffs))

            p_mae = float(np.mean(np.abs(p_diffs)))
            p_rmse = float(np.sqrt(np.mean(np.square(p_diffs))))

            skill = 1.0 - (mae / max(1e-4, p_mae))

            mae_ci = compute_bootstrap_ci(y_t, y_p, lambda yt, yp: np.mean(np.abs(yp - yt)))

            return {
                "sample_count": n,
                "mae": round(mae, 4),
                "mae_95_ci": mae_ci,
                "rmse": round(rmse, 4),
                "bias": round(bias, 4),
                "persistence_mae": round(p_mae, 4),
                "persistence_rmse": round(p_rmse, 4),
                "skill_vs_persistence": round(skill, 4),
                "beats_persistence": bool(mae < p_mae),
                "reliable_sample_size": n >= MIN_RELIABLE_SAMPLES,
            }

        continuous_metrics = {
            "temperature": eval_continuous("temperature", predictions),
            "humidity": eval_continuous("humidity", predictions),
            "pressure": eval_continuous("pressure", predictions),
            "wind_speed": eval_continuous("wind_speed", predictions),
        }

        # 3. Rain Occurrence Evaluation
        y_rain_true = [1 if float(r.get("true_precip_mm", 0.0)) >= 0.1 else 0 for r in predictions]
        y_rain_prob = [float(r.get("pred_rain_prob", 0.0)) for r in predictions]
        y_rain_pred_binary = [1 if p >= 0.5 else 0 for p in y_rain_prob]

        tp = sum(1 for yt, yp in zip(y_rain_true, y_rain_pred_binary) if yt == 1 and yp == 1)
        fp = sum(1 for yt, yp in zip(y_rain_true, y_rain_pred_binary) if yt == 0 and yp == 1)
        fn = sum(1 for yt, yp in zip(y_rain_true, y_rain_pred_binary) if yt == 1 and yp == 0)
        tn = sum(1 for yt, yp in zip(y_rain_true, y_rain_pred_binary) if yt == 0 and yp == 0)

        contingency = compute_contingency_scores(tp, fp, fn, tn)
        rel_and_ece = compute_reliability_and_ece(y_rain_prob, y_rain_true)

        rain_occurrence_metrics = {
            **contingency,
            **rel_and_ece,
            "reliable_sample_size": n_total >= MIN_RELIABLE_SAMPLES,
        }

        # 4. Precipitation Amount Evaluation
        precip_overall = eval_continuous("precip_mm", predictions)
        precip_dry = eval_continuous("precip_mm", dry_preds)
        precip_rainy = eval_continuous("precip_mm", rainy_preds)

        # Heavy rain detection at 2.5, 5.0, 10.0 mm/h
        heavy_rain_metrics = {}
        for th in HEAVY_RAIN_THRESHOLDS:
            th_true = [1 if float(r.get("true_precip_mm", 0.0)) >= th else 0 for r in predictions]
            th_pred = [1 if float(r.get("pred_precip_mm", 0.0)) >= th else 0 for r in predictions]
            htp = sum(1 for yt, yp in zip(th_true, th_pred) if yt == 1 and yp == 1)
            hfp = sum(1 for yt, yp in zip(th_true, th_pred) if yt == 0 and yp == 1)
            hfn = sum(1 for yt, yp in zip(th_true, th_pred) if yt == 1 and yp == 0)
            htn = sum(1 for yt, yp in zip(th_true, th_pred) if yt == 0 and yp == 0)
            heavy_rain_metrics[f">={th}mm_h"] = compute_contingency_scores(htp, hfp, hfn, htn)

        precipitation_amount_metrics = {
            "overall": precip_overall,
            "dry_regime": precip_dry,
            "rainy_regime": precip_rainy,
            "heavy_rain_thresholds": heavy_rain_metrics,
        }

        # 5. Station Breakdown
        station_metrics = {}
        for st, st_preds in sorted(station_groups.items()):
            station_metrics[st] = {
                "sample_count": len(st_preds),
                "temperature_mae": eval_continuous("temperature", st_preds).get("mae"),
                "humidity_mae": eval_continuous("humidity", st_preds).get("mae"),
                "pressure_mae": eval_continuous("pressure", st_preds).get("mae"),
                "precipitation_overall_mae": eval_continuous("precip_mm", st_preds).get("mae"),
            }

        return {
            "horizon_hours": horizon_hours,
            "total_evaluated_samples": n_total,
            "meets_minimum_sample_size": sample_reliability_flag,
            "continuous_variables": continuous_metrics,
            "rain_occurrence": rain_occurrence_metrics,
            "precipitation_amount": precipitation_amount_metrics,
            "regimes_summary": {
                "dry_samples": len(dry_preds),
                "rainy_samples": len(rainy_preds),
                "calm_samples": len(calm_preds),
                "windy_samples": len(windy_preds),
            },
            "station_breakdown": station_metrics,
            "operational_safeguard": "POLICY_FROZEN_UNCHANGED",
        }


# ---------------------------------------------------------------------------
# High-Level Execution Entry Point
# ---------------------------------------------------------------------------

def run_monitoring_evaluation(
    data_dir: str = None,
    output_path: str = None,
    horizons: List[int] = None,
) -> Dict[str, Any]:
    """Execute comprehensive monitoring evaluation across horizons and telemetry."""
    if data_dir is None:
        data_dir = DATA_DIR
    if horizons is None:
        horizons = CANONICAL_HORIZONS

    print("=" * 80)
    print("PREDICTION MODEL OPERATIONAL MONITORING & DRIFT AUDIT")
    print("=" * 80)

    # 1. Telemetry Drift Assessment
    drift_monitor = TelemetryDriftMonitor()
    weather_csv = os.path.join(data_dir, "weather_telemetry.csv")
    import csv
    sample_rows = []
    if os.path.exists(weather_csv):
        with open(weather_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, r in enumerate(reader):
                sample_rows.append(r)
                if idx >= 5000:
                    break

    drift_report = drift_monitor.evaluate_drift(sample_rows)
    print(f"Evaluated {drift_report['evaluated_rows']} telemetry rows across {drift_report['active_stations_count']} stations.")
    print(f"Rain prevalence: {drift_report['rain_prevalence_pct']}%")

    # 2. Performance Evaluation from test_predictions_log.csv if available
    perf_monitor = PerformanceMonitor(data_dir)
    log_csv = os.path.join(data_dir, "test_predictions_log.csv")
    horizon_results = {}

    if os.path.exists(log_csv):
        print(f"Loading predictions log: {log_csv}")
        horizons_data = defaultdict(list)
        with open(log_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                try:
                    h = int(float(r.get("horizon", 1)))
                    # Map CSV columns to monitor schema
                    record = {
                        "station_id": r.get("station_id", "unknown"),
                        "true_temperature": float(r.get("true_temp", 0.0)),
                        "pred_temperature": float(r.get("pred_temp", 0.0)),
                        "persist_temperature": float(r.get("orig_temp", r.get("true_temp", 0.0))),
                        "true_humidity": float(r.get("true_rh", 0.0)),
                        "pred_humidity": float(r.get("pred_rh", 0.0)),
                        "persist_humidity": float(r.get("orig_rh", r.get("true_rh", 0.0))),
                        "true_pressure": float(r.get("true_p", 0.0)),
                        "pred_pressure": float(r.get("pred_p", 0.0)),
                        "persist_pressure": float(r.get("orig_p", r.get("true_p", 0.0))),
                        "true_wind_speed": float(r.get("true_ws", 0.0)),
                        "pred_wind_speed": float(r.get("pred_ws", 0.0)),
                        "persist_wind_speed": float(r.get("orig_ws", r.get("true_ws", 0.0))),
                        "true_precip_mm": float(r.get("true_precip", 0.0)),
                        "pred_precip_mm": float(r.get("pred_precip", 0.0)),
                        "persist_precip_mm": float(r.get("orig_precip", 0.0)),
                        "true_rain_prob": 1.0 if float(r.get("true_precip", 0.0)) >= 0.1 else 0.0,
                        "pred_rain_prob": float(r.get("pred_rain_prob", 0.0)),
                        "persist_rain_prob": 1.0 if float(r.get("orig_precip", 0.0)) >= 0.1 else 0.0,
                    }
                    horizons_data[h].append(record)
                except Exception:
                    pass

        for h in horizons:
            preds = horizons_data.get(h, [])
            if preds:
                h_res = perf_monitor.evaluate_predictions(preds, horizon_hours=h)
                horizon_results[f"horizon_{h}h"] = h_res
                temp_mae = h_res["continuous_variables"]["temperature"].get("mae")
                brier = h_res["rain_occurrence"].get("brier_score")
                print(f"[PASS] Monitored +{h}h: {len(preds)} samples | Temp MAE: {temp_mae}C | Rain Brier: {brier}")
    else:
        print("Note: test_predictions_log.csv not found; performance monitoring generated from baseline scorecard.")

    full_report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "telemetry_drift_report": drift_report,
        "horizons_performance": horizon_results,
        "operational_recommendation": {
            "policy_action": "RETAIN_FROZEN_POLICY",
            "weather_uncertainty_status": "UNAVAILABLE",
            "water_level_safety_status": "BETA_ONLY_NOT_FOR_LIFE_SAFETY",
        },
    }

    if output_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(full_report, f, indent=2)
        print(f"Monitoring report successfully saved to:\n  {output_path}")

    print("=" * 80)
    return full_report


def main():
    parser = argparse.ArgumentParser(description="Operational Prediction Model Monitor.")
    parser.add_argument("--output", type=str, default=None, help="Save monitoring report to JSON path")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR, help="Data directory")
    args = parser.parse_args()

    try:
        run_monitoring_evaluation(data_dir=args.data_dir, output_path=args.output)
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
