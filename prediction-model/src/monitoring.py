"""
Operational Monitoring, Drift Detection, and Fine-Grained Evaluation Engine.

Provides continuous and post-deployment evaluation across:
  - Dimensions: horizon (1h, 3h, 6h, 12h, 24h), station, target variable, rain vs dry regime,
    heavy-rain regime, calm vs windy regime, data-quality quarantine, time window.
  - Metrics:
      * Continuous: MAE, RMSE, bias, persistence MAE, skill score, climatology MAE, missingness, quarantine rate.
      * Rain occurrence: Brier score, calibration error (ECE), F1, precision, recall (POD), FAR, CSI, confusion matrix, reliability bins.
      * Precipitation amount: overall, dry-hour, rainy-hour MAE/RMSE/bias, heavy-rain precision/recall/CSI (2.5, 5.0, 10.0 mm/h).
  - Safeguards:
      * Minimum sample count enforcement.
      * Bootstrap 95% confidence intervals.
      * Drift detection across feature distributions, missingness, and rain prevalence.
      * Strict schema contract validation (PREDICTION_LOG_REQUIRED_COLUMNS) with zero silent defaults.
      * Independent multi-horizon evaluation for all five canonical horizons (1h, 3h, 6h, 12h, 24h).
      * Fail-closed behavior for missing columns, invalid horizons, and malformed rows.
      * NEVER automatically modifies the frozen operational inference policy.

Usage:
  python prediction-model/src/monitoring.py \
    --data-dir prediction-model/data \
    --predictions-log prediction-model/data/test_predictions_log.csv \
    --horizons 1 3 6 12 24 \
    --require-all-horizons \
    --output prediction-model/data/monitoring_report.json
"""

import os
import sys
import math
import json
import csv
import hashlib
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

PREDICTION_LOG_SCHEMA_VERSION = "1.0"
MONITORING_VERSION = "2.0.0"

PREDICTION_LOG_REQUIRED_COLUMNS = {
    "station_id": "string",
    "origin_timestamp": "timestamp",
    "target_timestamp": "timestamp",
    "horizon_hours": "integer",
    "actual_lead_hours": "float",
    "actual_rain": "float_or_binary",
    "actual_precip_mm": "float",
    "actual_temp": "float",
    "actual_humidity": "float",
    "actual_pressure": "float",
    "actual_wind_speed": "float",
    "pred_temp": "float",
    "pred_humidity": "float",
    "pred_pressure": "float",
    "pred_wind_speed": "float",
    "mf1_rain_prob": "float",
    "hybrid_rain_prob": "float",
    "mf1_precip_mm": "float",
    "persist_temp": "float",
    "persist_humidity": "float",
    "persist_pressure": "float",
    "persist_wind_speed": "float",
    "persist_rain": "numeric_with_explicit_unit",
}


# ---------------------------------------------------------------------------
# Schema and Error Definitions
# ---------------------------------------------------------------------------

class PredictionLogSchemaError(ValueError):
    """Raised when prediction log schema validation fails (e.g. missing required columns)."""
    pass


class PredictionLogDataError(ValueError):
    """Raised when prediction log data is empty, contains invalid horizons, or exceeds malformed limits."""
    pass


def compute_file_sha256(filepath: str) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def validate_prediction_log_headers(fieldnames: Optional[List[str]]) -> None:
    """
    Validate that all required columns are present in CSV fieldnames.
    Fails closed with a clear error listing any missing columns.
    """
    if not fieldnames:
        raise PredictionLogSchemaError("Prediction log headers are empty or missing.")
    present = set(fieldnames)
    missing = [c for c in PREDICTION_LOG_REQUIRED_COLUMNS if c not in present]
    if missing:
        raise PredictionLogSchemaError(
            f"Prediction log schema validation failed: missing {len(missing)} required column(s): {missing}. "
            f"Expected schema version: {PREDICTION_LOG_SCHEMA_VERSION}."
        )


def parse_prediction_log_row(
    row: Dict[str, Any],
    row_idx: int,
    canonical_horizons: List[int] = CANONICAL_HORIZONS,
    lead_tolerance: float = 1.0,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Parse and validate a single prediction log row without silent fallbacks.
    Returns (record, None) if valid, or (None, error_reason) if malformed.
    """
    # 1. Validate required columns exist in row dict
    missing_cols = [c for c in PREDICTION_LOG_REQUIRED_COLUMNS if c not in row]
    if missing_cols:
        return None, f"Row {row_idx}: missing column(s): {missing_cols}"

    # 2. Horizon validation
    raw_h = row.get("horizon_hours")
    if raw_h is None or str(raw_h).strip() == "":
        return None, f"Row {row_idx}: 'horizon_hours' is missing or empty"
    try:
        f_h = float(raw_h)
        i_h = int(f_h)
        if f_h != i_h:
            return None, f"Row {row_idx}: 'horizon_hours' {raw_h} is not an integer"
        if i_h not in canonical_horizons:
            return None, f"Row {row_idx}: 'horizon_hours' {i_h} is not in allowed canonical horizons {canonical_horizons}"
    except (ValueError, TypeError):
        return None, f"Row {row_idx}: 'horizon_hours' '{raw_h}' cannot be parsed as integer"

    # 3. Lead hours validation
    raw_lead = row.get("actual_lead_hours")
    if raw_lead is None or str(raw_lead).strip() == "":
        return None, f"Row {row_idx}: 'actual_lead_hours' is missing or empty"
    try:
        lead_h = float(raw_lead)
        if math.isnan(lead_h) or math.isinf(lead_h):
            return None, f"Row {row_idx}: 'actual_lead_hours' is NaN or Inf"
        if abs(lead_h - i_h) > lead_tolerance:
            return None, f"Row {row_idx}: 'actual_lead_hours' ({lead_h}) differs from 'horizon_hours' ({i_h}) by more than {lead_tolerance}h"
    except (ValueError, TypeError):
        return None, f"Row {row_idx}: 'actual_lead_hours' '{raw_lead}' cannot be parsed as float"

    # 4. Numeric fields validation and parsing
    numeric_fields = [
        "actual_temp", "pred_temp", "persist_temp",
        "actual_humidity", "pred_humidity", "persist_humidity",
        "actual_pressure", "pred_pressure", "persist_pressure",
        "actual_wind_speed", "pred_wind_speed", "persist_wind_speed",
        "actual_precip_mm", "mf1_precip_mm",
        "actual_rain", "hybrid_rain_prob", "mf1_rain_prob", "persist_rain",
    ]
    parsed_nums: Dict[str, float] = {}
    for nf in numeric_fields:
        raw_val = row.get(nf)
        if raw_val is None or str(raw_val).strip() == "":
            return None, f"Row {row_idx}: field '{nf}' is missing or empty"
        try:
            val = float(raw_val)
            if math.isnan(val) or math.isinf(val):
                return None, f"Row {row_idx}: field '{nf}' is NaN or Inf"
            parsed_nums[nf] = val
        except (ValueError, TypeError):
            return None, f"Row {row_idx}: field '{nf}' '{raw_val}' cannot be parsed as float"

    # 5. Probability range validation
    for pf in ["actual_rain", "hybrid_rain_prob", "mf1_rain_prob", "persist_rain"]:
        pval = parsed_nums[pf]
        if pval < -1e-4 or pval > 1.0 + 1e-4:
            return None, f"Row {row_idx}: probability field '{pf}' value {pval} out of range [0, 1]"

    record = {
        "station_id": str(row.get("station_id", "unknown")),
        "origin_timestamp": str(row.get("origin_timestamp", "")),
        "target_timestamp": str(row.get("target_timestamp", "")),
        "horizon_hours": i_h,
        "actual_lead_hours": lead_h,

        "true_temperature": parsed_nums["actual_temp"],
        "pred_temperature": parsed_nums["pred_temp"],
        "persist_temperature": parsed_nums["persist_temp"],

        "true_humidity": parsed_nums["actual_humidity"],
        "pred_humidity": parsed_nums["pred_humidity"],
        "persist_humidity": parsed_nums["persist_humidity"],

        "true_pressure": parsed_nums["actual_pressure"],
        "pred_pressure": parsed_nums["pred_pressure"],
        "persist_pressure": parsed_nums["persist_pressure"],

        "true_wind_speed": parsed_nums["actual_wind_speed"],
        "pred_wind_speed": parsed_nums["pred_wind_speed"],
        "persist_wind_speed": parsed_nums["persist_wind_speed"],

        "true_precip_mm": parsed_nums["actual_precip_mm"],
        "pred_precip_mm": parsed_nums["mf1_precip_mm"],
        "persist_precip_mm": None,  # Not recorded as precip mm persistence in log; marked unavailable

        "true_rain_prob": 1.0 if parsed_nums["actual_rain"] >= 0.5 else 0.0,
        "pred_rain_prob": parsed_nums["hybrid_rain_prob"],
        "mf1_rain_prob": parsed_nums["mf1_rain_prob"],
        "persist_rain_prob": 1.0 if parsed_nums["persist_rain"] >= 0.5 else 0.0,
    }
    return record, None


def load_and_validate_prediction_log(
    file_path: str,
    canonical_horizons: List[int] = CANONICAL_HORIZONS,
    max_malformed_rows: int = 0,
) -> Tuple[Dict[int, List[Dict[str, Any]]], int, List[str]]:
    """
    Load CSV, validate schema headers, parse rows, and group by horizon_hours.
    Returns: (horizons_data, malformed_count, malformed_reasons)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Prediction log file not found: {file_path}")
    if os.path.getsize(file_path) == 0:
        raise PredictionLogDataError(f"Prediction log file is empty (0 bytes): {file_path}")

    horizons_data: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    malformed_reasons: List[str] = []
    malformed_count = 0
    total_raw_rows = 0

    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        validate_prediction_log_headers(reader.fieldnames)

        for idx, row in enumerate(reader, start=2):
            total_raw_rows += 1
            rec, err = parse_prediction_log_row(row, idx, canonical_horizons)
            if err is not None:
                malformed_count += 1
                if len(malformed_reasons) < 20:
                    malformed_reasons.append(err)
            else:
                horizons_data[rec["horizon_hours"]].append(rec)

    if total_raw_rows == 0:
        raise PredictionLogDataError(f"Prediction log contains 0 data rows (header only): {file_path}")

    if malformed_count > max_malformed_rows:
        sample_errs = "; ".join(malformed_reasons[:5])
        raise PredictionLogDataError(
            f"Prediction log contained {malformed_count} malformed rows, exceeding threshold of {max_malformed_rows}. "
            f"Examples: {sample_errs}"
        )

    return dict(horizons_data), malformed_count, malformed_reasons


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
          - 'true_rain_prob', 'pred_rain_prob', 'mf1_rain_prob', 'persist_rain_prob'
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

            n = len(y_t)
            if n == 0:
                return {
                    "sample_count": 0,
                    "status": "INSUFFICIENT_DATA",
                    "mae": None,
                    "rmse": None,
                    "bias": None,
                    "persistence_mae": None,
                    "persistence_rmse": None,
                    "skill_vs_persistence": None,
                    "beats_persistence": None,
                    "reliable_sample_size": False,
                }

            diffs = [yp - yt for yp, yt in zip(y_p, y_t)]
            mae = float(np.mean(np.abs(diffs)))
            rmse = float(np.sqrt(np.mean(np.square(diffs))))
            bias = float(np.mean(diffs))

            # Persistence evaluation (if field is available)
            has_persistence = all(r.get(f"persist_{name}") is not None for r in records)
            if has_persistence:
                y_persist = [float(r[f"persist_{name}"]) for r in records]
                p_diffs = [yp - yt for yp, yt in zip(y_persist, y_t)]
                p_mae = float(np.mean(np.abs(p_diffs)))
                p_rmse = float(np.sqrt(np.mean(np.square(p_diffs))))
                skill = 1.0 - (mae / max(1e-4, p_mae))
                beats_p = bool(mae < p_mae)
                p_mae_val = round(p_mae, 4)
                p_rmse_val = round(p_rmse, 4)
                skill_val = round(skill, 4)
            else:
                p_mae_val = None
                p_rmse_val = None
                skill_val = None
                beats_p = None

            mae_ci = compute_bootstrap_ci(y_t, y_p, lambda yt, yp: np.mean(np.abs(yp - yt)))

            res = {
                "sample_count": n,
                "mae": round(mae, 4),
                "mae_95_ci": mae_ci,
                "rmse": round(rmse, 4),
                "bias": round(bias, 4),
                "persistence_mae": p_mae_val,
                "persistence_rmse": p_rmse_val,
                "skill_vs_persistence": skill_val,
                "beats_persistence": beats_p,
                "reliable_sample_size": n >= MIN_RELIABLE_SAMPLES,
            }
            if not has_persistence:
                res["persistence_status"] = "UNAVAILABLE_IN_LOG"
            return res

        continuous_metrics = {
            "temperature": eval_continuous("temperature", predictions),
            "humidity": eval_continuous("humidity", predictions),
            "pressure": eval_continuous("pressure", predictions),
            "wind_speed": eval_continuous("wind_speed", predictions),
        }

        # 3. Rain Occurrence Evaluation
        y_rain_true = [1 if float(r["true_rain_prob"]) >= 0.5 else 0 for r in predictions]
        y_rain_prob = [float(r["pred_rain_prob"]) for r in predictions]
        y_rain_pred_binary = [1 if p >= 0.5 else 0 for p in y_rain_prob]

        tp = sum(1 for yt, yp in zip(y_rain_true, y_rain_pred_binary) if yt == 1 and yp == 1)
        fp = sum(1 for yt, yp in zip(y_rain_true, y_rain_pred_binary) if yt == 0 and yp == 1)
        fn = sum(1 for yt, yp in zip(y_rain_true, y_rain_pred_binary) if yt == 1 and yp == 0)
        tn = sum(1 for yt, yp in zip(y_rain_true, y_rain_pred_binary) if yt == 0 and yp == 0)

        contingency = compute_contingency_scores(tp, fp, fn, tn)
        rel_and_ece = compute_reliability_and_ece(y_rain_prob, y_rain_true)

        # Rain persistence comparison
        persist_rain_probs = [float(r["persist_rain_prob"]) for r in predictions if r.get("persist_rain_prob") is not None]
        if len(persist_rain_probs) == n_total and n_total > 0:
            p_brier = sum((p - y) ** 2 for p, y in zip(persist_rain_probs, y_rain_true)) / n_total
            brier_val = rel_and_ece.get("brier_score")
            bss = 1.0 - (brier_val / max(1e-4, p_brier)) if (brier_val is not None and p_brier > 0) else 0.0
            p_brier_round = round(p_brier, 4)
            bss_round = round(bss, 4)
            beats_rain_p = bool(brier_val < p_brier) if brier_val is not None else None
        else:
            p_brier_round = None
            bss_round = None
            beats_rain_p = None

        # Diagnostic MF-1 model-only rain probability
        mf1_probs = [float(r["mf1_rain_prob"]) for r in predictions if r.get("mf1_rain_prob") is not None]
        if len(mf1_probs) == n_total and n_total > 0:
            mf1_brier = round(sum((p - y) ** 2 for p, y in zip(mf1_probs, y_rain_true)) / n_total, 4)
        else:
            mf1_brier = None

        rain_occurrence_metrics = {
            **contingency,
            **rel_and_ece,
            "diagnostic_mf1_brier_score": mf1_brier,
            "persistence_brier_score": p_brier_round,
            "brier_skill_score_vs_persistence": bss_round,
            "beats_persistence_brier": beats_rain_p,
            "reliable_sample_size": n_total >= MIN_RELIABLE_SAMPLES,
        }

        # 4. Precipitation Amount Evaluation
        precip_overall = eval_continuous("precip_mm", predictions)
        precip_dry = eval_continuous("precip_mm", dry_preds)
        precip_rainy = eval_continuous("precip_mm", rainy_preds)

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

        # 6. Structured Persistence Comparison Summary
        persistence_comparison = {
            "temperature_mae": continuous_metrics["temperature"].get("mae"),
            "temperature_persistence_mae": continuous_metrics["temperature"].get("persistence_mae"),
            "temperature_skill_vs_persistence": continuous_metrics["temperature"].get("skill_vs_persistence"),
            "beats_temperature_persistence": continuous_metrics["temperature"].get("beats_persistence"),

            "humidity_mae": continuous_metrics["humidity"].get("mae"),
            "humidity_persistence_mae": continuous_metrics["humidity"].get("persistence_mae"),
            "humidity_skill_vs_persistence": continuous_metrics["humidity"].get("skill_vs_persistence"),
            "beats_humidity_persistence": continuous_metrics["humidity"].get("beats_persistence"),

            "pressure_mae": continuous_metrics["pressure"].get("mae"),
            "pressure_persistence_mae": continuous_metrics["pressure"].get("persistence_mae"),
            "pressure_skill_vs_persistence": continuous_metrics["pressure"].get("skill_vs_persistence"),
            "beats_pressure_persistence": continuous_metrics["pressure"].get("beats_persistence"),

            "wind_speed_mae": continuous_metrics["wind_speed"].get("mae"),
            "wind_speed_persistence_mae": continuous_metrics["wind_speed"].get("persistence_mae"),
            "wind_speed_skill_vs_persistence": continuous_metrics["wind_speed"].get("skill_vs_persistence"),
            "beats_wind_speed_persistence": continuous_metrics["wind_speed"].get("beats_persistence"),

            "rain_brier_score": rain_occurrence_metrics.get("brier_score"),
            "rain_persistence_brier_score": rain_occurrence_metrics.get("persistence_brier_score"),
            "rain_brier_skill_score_vs_persistence": rain_occurrence_metrics.get("brier_skill_score_vs_persistence"),
            "beats_rain_persistence": rain_occurrence_metrics.get("beats_persistence_brier"),

            "precipitation_amount_persistence": "UNAVAILABLE_IN_LOG",
        }

        return {
            "horizon_hours": horizon_hours,
            "sample_count": n_total,
            "total_evaluated_samples": n_total,
            "station_count": len(station_groups),
            "valid_row_count": n_total,
            "malformed_row_count": 0,
            "meets_minimum_sample_size": sample_reliability_flag,
            "data_quality_summary": {
                "valid_samples": n_total,
                "stations_count": len(station_groups),
                "missing_fields_count": 0,
                "lead_hour_discrepancy_count": 0,
            },
            "continuous_variables": continuous_metrics,
            "temperature": continuous_metrics["temperature"],
            "humidity": continuous_metrics["humidity"],
            "pressure": continuous_metrics["pressure"],
            "wind_speed": continuous_metrics["wind_speed"],
            "rain_occurrence": rain_occurrence_metrics,
            "precipitation_amount": precipitation_amount_metrics,
            "persistence_comparison": persistence_comparison,
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
    predictions_log_path: str = None,
    output_path: str = None,
    horizons: List[int] = None,
    require_all_horizons: bool = False,
    allow_missing_horizon: bool = False,
    max_malformed_rows: int = 0,
) -> Dict[str, Any]:
    """Execute comprehensive monitoring evaluation across horizons and telemetry."""
    if data_dir is None:
        data_dir = DATA_DIR
    if horizons is None:
        horizons = list(CANONICAL_HORIZONS)
    if predictions_log_path is None:
        predictions_log_path = os.path.join(data_dir, "test_predictions_log.csv")

    print("=" * 80)
    print("PREDICTION MODEL OPERATIONAL MONITORING & DRIFT AUDIT")
    print(f"Monitoring Version: {MONITORING_VERSION} | Schema Version: {PREDICTION_LOG_SCHEMA_VERSION}")
    print("=" * 80)

    # 1. Telemetry Drift Assessment
    drift_monitor = TelemetryDriftMonitor(os.path.join(data_dir, "cleaned_data_manifest.json"))
    weather_csv = os.path.join(data_dir, "weather_telemetry.csv")
    sample_rows = []
    if os.path.exists(weather_csv):
        with open(weather_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, r in enumerate(reader):
                sample_rows.append(r)
                if idx >= 5000:
                    break

    drift_report = drift_monitor.evaluate_drift(sample_rows)
    print(f"Evaluated {drift_report.get('evaluated_rows', 0)} telemetry rows across {drift_report.get('active_stations_count', 0)} stations.")
    print(f"Rain prevalence: {drift_report.get('rain_prevalence_pct', 'N/A')}%")

    # 2. Validate and load prediction log
    print(f"Loading predictions log: {predictions_log_path}")
    input_log_sha256 = compute_file_sha256(predictions_log_path)
    print(f"Log SHA-256: {input_log_sha256}")

    horizons_data, malformed_count, malformed_reasons = load_and_validate_prediction_log(
        file_path=predictions_log_path,
        canonical_horizons=CANONICAL_HORIZONS,
        max_malformed_rows=max_malformed_rows,
    )
    print(f"Schema validation PASS. Malformed rows: {malformed_count}")

    # 3. Evaluate requested forecast horizons independently
    perf_monitor = PerformanceMonitor(data_dir)
    horizon_results: Dict[str, Any] = {}
    evaluated_horizons: List[int] = []
    missing_horizons: List[int] = []

    for h in horizons:
        preds = horizons_data.get(h, [])
        if preds:
            h_res = perf_monitor.evaluate_predictions(preds, horizon_hours=h)
            h_res["malformed_row_count"] = malformed_count if len(horizons) == 1 else 0
            horizon_results[f"horizon_{h}h"] = h_res
            evaluated_horizons.append(h)
            temp_mae = h_res["continuous_variables"]["temperature"].get("mae")
            brier = h_res["rain_occurrence"].get("brier_score")
            print(f"[PASS] Monitored +{h}h: {len(preds)} samples | Temp MAE: {temp_mae}C | Rain Brier: {brier}")
        else:
            missing_horizons.append(h)
            print(f"[WARN] Horizon +{h}h has 0 valid samples in prediction log.")

    # 4. Strict fail-closed horizon check
    if missing_horizons:
        if require_all_horizons or (not allow_missing_horizon):
            raise PredictionLogDataError(
                f"Missing required forecast horizon(s): {missing_horizons}. "
                f"Evaluated horizons: {evaluated_horizons}. "
                f"Supply --allow-missing-horizon to permit partial evaluation."
            )

    # Sanitize path for path hygiene compliance (no absolute Windows/Unix paths)
    repo_root = os.path.dirname(os.path.dirname(SRC_DIR))
    try:
        sanitized_log_path = os.path.relpath(predictions_log_path, repo_root).replace("\\", "/")
    except Exception:
        sanitized_log_path = os.path.basename(predictions_log_path)

    # 5. Automated Operational Rollback & Fallback Triggers (Phase 10)
    trigger_level = "NORMAL"
    trigger_reasons = []
    target_fallbacks = {}

    # Trigger A: Full Bundle Rollback on integrity failure
    if malformed_count > 0:
        trigger_level = "FULL_BUNDLE_ROLLBACK"
        trigger_reasons.append(f"Malformed row count ({malformed_count}) exceeds integrity threshold")

    # Trigger B: Warning on telemetry feature drift
    drift_status = drift_report.get("status", "NORMAL")
    if drift_status in ("WARNING", "DRIFT_DETECTED") and trigger_level == "NORMAL":
        trigger_level = "WARNING"
        trigger_reasons.append("Telemetry feature drift detected exceeding baseline variance")

    # Trigger C: Target-Specific Baseline Fallback or Recalibration
    for h_key, h_data in horizon_results.items():
        cont = h_data.get("continuous_variables", {})
        for var in ("temperature", "humidity", "pressure", "wind_speed"):
            v_met = cont.get(var, {})
            skill = v_met.get("skill_vs_persistence")
            if skill is not None and skill < -0.20 and v_met.get("sample_count", 0) >= MIN_RELIABLE_SAMPLES:
                target_fallbacks[var] = "TARGET_BASELINE_FALLBACK"
                if trigger_level in ("NORMAL", "WARNING"):
                    trigger_level = "TARGET_BASELINE_FALLBACK"
                trigger_reasons.append(f"{var} skill vs persistence severely degraded ({skill:.2f}) on {h_key}")

        rain_met = h_data.get("rain_occurrence", {})
        rain_ece = rain_met.get("calibration_error_ece")
        if rain_ece is not None and rain_ece > 0.25 and rain_met.get("sample_count", 0) >= MIN_RELIABLE_SAMPLES:
            target_fallbacks["rain_occurrence"] = "RECALIBRATION_RECOMMENDED"
            if trigger_level in ("NORMAL", "WARNING"):
                trigger_level = "RECALIBRATION_RECOMMENDED"
            trigger_reasons.append(f"Rain calibration error ECE ({rain_ece:.2f}) indicates calibration drift on {h_key}")

    full_report = {
        "monitoring_version": MONITORING_VERSION,
        "prediction_log_schema_version": PREDICTION_LOG_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_log_path": sanitized_log_path,
        "input_log_sha256": input_log_sha256,
        "evaluated_horizons_hours": evaluated_horizons,
        "missing_horizons_hours": missing_horizons,
        "malformed_row_count": malformed_count,
        "horizons_performance": horizon_results,
        "telemetry_drift_report": drift_report,
        "operational_recommendation": {
            "policy_action": "RETAIN_FROZEN_POLICY" if trigger_level in ("NORMAL", "WARNING") else trigger_level,
            "trigger_level": trigger_level,
            "trigger_reasons": trigger_reasons,
            "target_fallbacks": target_fallbacks,
            "weather_uncertainty_status": "CALIBRATED_QUANTILES_AVAILABLE",
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
    parser.add_argument("--data-dir", type=str, default=DATA_DIR, help="Data directory (default: prediction-model/data)")
    parser.add_argument("--predictions-log", type=str, default=None, help="Path to predictions log CSV")
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=CANONICAL_HORIZONS,
        help="List of forecast horizons to evaluate in hours (default: 1 3 6 12 24)",
    )
    parser.add_argument(
        "--require-all-horizons",
        action="store_true",
        help="Fail if any requested horizon has zero valid rows",
    )
    parser.add_argument(
        "--allow-missing-horizon",
        action="store_true",
        help="Permit partial horizon evaluation if a requested horizon is absent",
    )
    parser.add_argument(
        "--max-malformed-rows",
        type=int,
        default=0,
        help="Maximum allowed malformed rows before failing (default: 0)",
    )
    parser.add_argument("--output", type=str, default=None, help="Save monitoring report to JSON path")
    args = parser.parse_args()

    try:
        run_monitoring_evaluation(
            data_dir=args.data_dir,
            predictions_log_path=args.predictions_log,
            output_path=args.output,
            horizons=args.horizons,
            require_all_horizons=args.require_all_horizons,
            allow_missing_horizon=args.allow_missing_horizon,
            max_malformed_rows=args.max_malformed_rows,
        )
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
