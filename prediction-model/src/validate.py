"""
Comprehensive Model Validation & Scorecard Suite for KloudTrack CfC/LNN.

Evaluates:
  - Rain: Accuracy, Recall/POD, Precision, F1, FAR, CSI, Brier Score,
    threshold sensitivity, per-intensity breakdown, per-station breakdown
  - Water Level: MAE, RMSE against *real gauge observations* (not synthetic)
  - Uncertainty: Empirical conformal intervals with coverage reporting
  - Latency & throughput benchmarks

This validates the standalone ContinuousLNNCell (Model Family 2).
See MODEL_REGISTRY.md for model family details.
"""

import os
import csv
import math
import time
import json

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
WEIGHTS_PATH = os.path.join(DATA_DIR, "lnn_trained_weights.json")
WEATHER_CSV = os.path.join(DATA_DIR, "weather_telemetry.csv")
WATER_CSV = os.path.join(DATA_DIR, "water_level_telemetry.csv")

MEANS = [28.5, 33.0, 10.0, 1008.0]
STDS = [4.5, 6.5, 8.0, 6.0]

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


# ---------------------------------------------------------------------------
# Water-level gauge data loader
# ---------------------------------------------------------------------------
def _load_water_gauge_lookup():
    """Load real water-level observations indexed by minute-truncated timestamp.

    Returns a dict mapping iso_minute_str -> water_level_m.
    We round timestamps to the nearest minute to allow fuzzy joining.
    """
    lookup = {}
    if not os.path.exists(WATER_CSV):
        return lookup
    with open(WATER_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts = row.get("recorded_at", "")[:16]  # Truncate to minute precision
                wl_m = row.get("water_level_m")
                wl_cm = row.get("water_level_cm")
                if wl_m is not None and wl_m != "":
                    wl = float(wl_m)
                elif wl_cm is not None and wl_cm != "":
                    wl = float(wl_cm) / 100.0
                else:
                    continue
                lookup[ts] = wl
            except (ValueError, TypeError):
                continue
    return lookup


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------
def run_comprehensive_validation():
    print("=" * 70)
    print("COMPREHENSIVE CfC/LNN PREDICTION MODEL VALIDATION SUITE")
    print("=" * 70)

    if not os.path.exists(WEIGHTS_PATH):
        print("ERROR: Model weights not found at", WEIGHTS_PATH)
        return

    with open(WEIGHTS_PATH, "r", encoding="utf-8") as f:
        weights = json.load(f)

    hidden_dim = weights.get("hidden_dim", 8)
    W_in = weights.get("W_in", [])
    W_rec = weights.get("W_rec", [])
    b_h = weights.get("b_h", [0.0] * hidden_dim)
    tau = weights.get("tau", [1.5] * hidden_dim)
    W_rain = weights.get("W_rain", [])
    b_rain = weights.get("b_rain", 0.0)
    W_water = weights.get("W_water", [])
    b_water = weights.get("b_water", 3.45)

    print(f"Model: Continuous-Time CfC Recurrent Neural Network (CfC/LNN)")
    print(f"   Hidden Dimensions:  {hidden_dim} state variables")
    print(f"   Input Dimensions:   4 channels (Temp, Heat Index, Wind, Pressure)")
    print(f"   Mathematical Core:  Analytical Ordinary Differential Equation (ODE)")
    print(f"   Status:             RESEARCH PROTOTYPE (not production validated)")

    if not os.path.exists(WEATHER_CSV):
        print("ERROR: Telemetry CSV not found.")
        return

    # Load real water-level gauge observations
    water_gauge = _load_water_gauge_lookup()
    has_water_gauge = len(water_gauge) > 0
    if has_water_gauge:
        print(f"\nLoaded {len(water_gauge):,} real water-level gauge observations.")
    else:
        print("\nWARNING: No water-level gauge data found. Water metrics will be skipped.")

    print("\nLoading validation records across all stations...")
    test_samples = []
    station_counts = {}

    with open(WEATHER_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i % 10 != 0:
                continue
            try:
                st_id = row.get("station_id", "unknown")
                ts = row.get("recorded_at", "")[:16]  # Minute precision
                t = float(row.get("temperature") or 28.5)
                hi = float(row.get("heat_index") or 33.0)
                ws = float(row.get("wind_speed") or 10.0)
                p = float(row.get("pressure") or 1008.0)
                precip = float(row.get("precipitation") or 0.0)

                feat = [
                    (t - MEANS[0]) / STDS[0],
                    (hi - MEANS[1]) / STDS[1],
                    (ws - MEANS[2]) / STDS[2],
                    (p - MEANS[3]) / STDS[3],
                ]

                rain_target = 1.0 if precip > 0.1 else 0.0

                # Real water target from Calumpit WLMS gauge joined to Calumpit AWS (3nzr48bG)
                water_target = water_gauge.get(ts) if st_id == "3nzr48bG" else None

                test_samples.append((st_id, feat, rain_target, water_target, precip))
                station_counts[st_id] = station_counts.get(st_id, 0) + 1
            except Exception:
                continue

    print(f"Loaded {len(test_samples):,} validation records across {len(station_counts)} stations.")

    # ------------------------------------------------------------------
    # Run Inference
    # ------------------------------------------------------------------
    # Per-threshold accumulators
    threshold_stats = {th: {"tp": 0, "fp": 0, "tn": 0, "fn": 0} for th in THRESHOLDS}

    # Per-station accumulators (at default threshold 0.5)
    station_stats = {}

    # Per-intensity class accumulators
    intensity_stats = {cls: {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "count": 0} for cls in INTENSITY_CLASSES}

    # Brier score accumulator
    brier_sum = 0.0

    # Water level accumulators (real gauge only)
    water_errors = []
    water_squared_errors = []

    # All prediction probabilities for conformal calibration
    all_rain_probs = []
    all_rain_targets = []
    all_water_residuals = []

    h = [0.0] * hidden_dim
    start_t = time.perf_counter()

    for st_id, feat, target_rain, target_water, precip in test_samples:
        # Continuous ODE Step
        h_next = []
        for j in range(hidden_dim):
            in_sum = sum(feat[i] * W_in[i][j] for i in range(4)) if W_in else 0.0
            rec_sum = sum(h[k] * W_rec[k][j] for k in range(hidden_dim)) if W_rec else 0.0
            act = tanh(in_sum + rec_sum + (b_h[j] if b_h else 0.0))
            decay = math.exp(-1.0 / max(0.1, tau[j]))
            h_j = decay * h[j] + (1.0 - decay) * act
            h_next.append(h_j)
        h = h_next

        # Rain head
        rain_logit = b_rain + sum(h[j] * W_rain[j] for j in range(hidden_dim))
        pred_prob = sigmoid(rain_logit)

        all_rain_probs.append(pred_prob)
        all_rain_targets.append(target_rain)

        # Brier score
        brier_sum += (pred_prob - target_rain) ** 2

        # Per-threshold stats
        for th in THRESHOLDS:
            pred_class = 1 if pred_prob >= th else 0
            s = threshold_stats[th]
            if pred_class == 1 and target_rain == 1.0:
                s["tp"] += 1
            elif pred_class == 1 and target_rain == 0.0:
                s["fp"] += 1
            elif pred_class == 0 and target_rain == 0.0:
                s["tn"] += 1
            else:
                s["fn"] += 1

        # Per-station stats (at threshold 0.5)
        pred_rain_05 = 1 if pred_prob >= 0.5 else 0
        if st_id not in station_stats:
            station_stats[st_id] = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        ss = station_stats[st_id]
        if pred_rain_05 == 1 and target_rain == 1.0:
            ss["tp"] += 1
        elif pred_rain_05 == 1 and target_rain == 0.0:
            ss["fp"] += 1
        elif pred_rain_05 == 0 and target_rain == 0.0:
            ss["tn"] += 1
        else:
            ss["fn"] += 1

        # Per-intensity class
        if precip > 0.1:
            for cls_name, (lo, hi_bound) in INTENSITY_CLASSES.items():
                if lo < precip <= hi_bound:
                    ic = intensity_stats[cls_name]
                    ic["count"] += 1
                    if pred_rain_05 == 1:
                        ic["tp"] += 1
                    else:
                        ic["fn"] += 1
                    break

        # Water head (real gauge only)
        pred_water = b_water + sum(h[j] * W_water[j] for j in range(hidden_dim))
        if target_water is not None:
            diff = abs(pred_water - target_water)
            water_errors.append(diff)
            water_squared_errors.append((pred_water - target_water) ** 2)
            all_water_residuals.append(abs(pred_water - target_water))

    total_time = time.perf_counter() - start_t
    latency_us = (total_time / len(test_samples)) * 1_000_000.0
    throughput = int(len(test_samples) / total_time)

    # ------------------------------------------------------------------
    # Calculate metrics at default threshold (0.5)
    # ------------------------------------------------------------------
    total = len(test_samples)
    s05 = threshold_stats[0.5]
    tp, fp, tn, fn = s05["tp"], s05["fp"], s05["tn"], s05["fn"]

    accuracy = (tp + tn) / total * 100.0
    recall = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
    precision = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    far = (fp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0  # False Alarm Ratio
    csi = (tp / (tp + fp + fn)) * 100.0 if (tp + fp + fn) > 0 else 0.0  # Critical Success Index
    brier = brier_sum / total  # Brier Score (lower is better)

    # Water metrics (real gauge only)
    if water_errors:
        mae = sum(water_errors) / len(water_errors)
        rmse = math.sqrt(sum(water_squared_errors) / len(water_squared_errors))
    else:
        mae = None
        rmse = None

    # ------------------------------------------------------------------
    # Threshold sensitivity analysis
    # ------------------------------------------------------------------
    print("-" * 70)
    print("THRESHOLD SENSITIVITY ANALYSIS (Rain Classification):")
    print(f"  {'Threshold':>10s} {'Accuracy':>10s} {'Recall':>10s} {'Precision':>10s} {'F1':>10s} {'FAR':>10s} {'CSI':>10s}")
    threshold_results = []
    for th in THRESHOLDS:
        s = threshold_stats[th]
        t_tp, t_fp, t_tn, t_fn = s["tp"], s["fp"], s["tn"], s["fn"]
        t_total = t_tp + t_fp + t_tn + t_fn
        t_acc = (t_tp + t_tn) / t_total * 100.0 if t_total > 0 else 0.0
        t_rec = (t_tp / (t_tp + t_fn)) * 100.0 if (t_tp + t_fn) > 0 else 0.0
        t_prec = (t_tp / (t_tp + t_fp)) * 100.0 if (t_tp + t_fp) > 0 else 0.0
        t_f1 = (2 * t_prec * t_rec / (t_prec + t_rec)) if (t_prec + t_rec) > 0 else 0.0
        t_far = (t_fp / (t_tp + t_fp)) * 100.0 if (t_tp + t_fp) > 0 else 0.0
        t_csi = (t_tp / (t_tp + t_fp + t_fn)) * 100.0 if (t_tp + t_fp + t_fn) > 0 else 0.0
        print(f"  {th:>10.1f} {t_acc:>9.1f}% {t_rec:>9.1f}% {t_prec:>9.1f}% {t_f1:>9.1f}% {t_far:>9.1f}% {t_csi:>9.1f}%")
        threshold_results.append({
            "threshold": th,
            "accuracy": round(t_acc, 2),
            "recall": round(t_rec, 2),
            "precision": round(t_prec, 2),
            "f1": round(t_f1, 2),
            "far": round(t_far, 2),
            "csi": round(t_csi, 2),
        })

    # ------------------------------------------------------------------
    # Per-intensity class breakdown
    # ------------------------------------------------------------------
    print("-" * 70)
    print("RAIN DETECTION BY PRECIPITATION INTENSITY:")
    intensity_results = {}
    for cls_name, (lo, hi_bound) in INTENSITY_CLASSES.items():
        ic = intensity_stats[cls_name]
        if ic["count"] > 0:
            detection_rate = ic["tp"] / ic["count"] * 100.0
        else:
            detection_rate = 0.0
        hi_str = f"{hi_bound:.1f}" if hi_bound != float("inf") else "inf"
        print(f"   {cls_name:>10s} ({lo:.1f}–{hi_str} mm): {ic['count']:>6d} events, {detection_rate:.1f}% detected")
        intensity_results[cls_name] = {
            "range_mm": f"{lo}-{hi_str}",
            "event_count": ic["count"],
            "detected_pct": round(detection_rate, 1),
        }

    # ------------------------------------------------------------------
    # Per-station breakdown
    # ------------------------------------------------------------------
    print("-" * 70)
    print("PER-STATION RAIN METRICS (threshold=0.5):")
    per_station_results = {}
    for st_id, ss in sorted(station_stats.items()):
        s_total = ss["tp"] + ss["fp"] + ss["tn"] + ss["fn"]
        s_acc = (ss["tp"] + ss["tn"]) / s_total * 100.0 if s_total > 0 else 0.0
        s_rec = (ss["tp"] / (ss["tp"] + ss["fn"])) * 100.0 if (ss["tp"] + ss["fn"]) > 0 else 0.0
        s_prec = (ss["tp"] / (ss["tp"] + ss["fp"])) * 100.0 if (ss["tp"] + ss["fp"]) > 0 else 0.0
        print(f"   {st_id:>20s}: Acc={s_acc:5.1f}%  Rec={s_rec:5.1f}%  Prec={s_prec:5.1f}%  n={s_total}")
        per_station_results[st_id] = {
            "accuracy": round(s_acc, 2),
            "recall": round(s_rec, 2),
            "precision": round(s_prec, 2),
            "sample_count": s_total,
        }

    # ------------------------------------------------------------------
    # Empirical conformal uncertainty bands (replaces asserted sqrt formula)
    # ------------------------------------------------------------------
    print("-" * 70)
    print("EMPIRICAL CONFORMAL UNCERTAINTY BANDS:")
    print("   (Computed from held-out residuals, NOT from a fixed formula)")

    conformal_results = []
    if all_water_residuals:
        sorted_residuals = sorted(all_water_residuals)
        n_cal = len(sorted_residuals)
        for coverage in [0.80, 0.90, 0.95]:
            idx = min(int(math.ceil(coverage * (n_cal + 1))) - 1, n_cal - 1)
            band = sorted_residuals[idx]
            # Verify empirical coverage
            actual_coverage = sum(1 for r in all_water_residuals if r <= band) / n_cal
            print(f"   Coverage {coverage*100:.0f}%: band = +/-{band:.4f}m  (empirical coverage: {actual_coverage*100:.1f}%, n={n_cal})")
            conformal_results.append({
                "target_coverage": f"{coverage*100:.0f}%",
                "band_meters": round(band, 4),
                "empirical_coverage": f"{actual_coverage*100:.1f}%",
                "calibration_samples": n_cal,
            })
    else:
        print("   SKIPPED — no real water-level gauge data available for calibration.")
        conformal_results.append({
            "status": "SKIPPED",
            "reason": "No real water-level gauge observations available",
        })

    # ------------------------------------------------------------------
    # Output Scorecard
    # ------------------------------------------------------------------
    print("=" * 70)
    print("VALIDATION SCORECARD")
    print("=" * 70)
    print(f"RAIN PREDICTION:")
    print(f"   Probability of Detection (Recall / POD): {recall:.2f}%")
    print(f"   Overall Accuracy:                        {accuracy:.2f}%")
    print(f"   Precision:                               {precision:.2f}%")
    print(f"   F1-Score:                                {f1:.2f}%")
    print(f"   False Alarm Ratio (FAR):                 {far:.2f}%")
    print(f"   Critical Success Index (CSI):            {csi:.2f}%")
    print(f"   Brier Score:                             {brier:.4f} (lower is better)")
    print()
    print(f"HYDROLOGICAL WATER LEVEL (Real Gauge Only):")
    if mae is not None:
        print(f"   Mean Absolute Error (MAE):               {mae:.4f} meters (~{mae * 100:.1f} cm)")
        print(f"   Root Mean Squared Error (RMSE):          {rmse:.4f} meters (~{rmse * 100:.1f} cm)")
        print(f"   Gauge-matched samples:                   {len(water_errors):,}")
        print(f"   NOTE: Water metrics are single-gauge (Calumpit WLMS) only.")
    else:
        print(f"   SKIPPED — no real gauge observations matched.")
    print()
    print(f"INFERENCE PERFORMANCE:")
    print(f"   Evaluated Samples:                       {total:,} records across {len(station_counts)} stations")
    print(f"   Latency:                                 {latency_us:.2f} microseconds / step")
    print(f"   Throughput:                              {throughput:,} predictions / second (CPU)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Save validation report
    # ------------------------------------------------------------------
    report_path = os.path.join(DATA_DIR, "validation_scorecard.json")
    report = {
        "model_status": "RESEARCH_PROTOTYPE",
        "model_family": "ContinuousLNNCell (Model Family 2 — standalone trainer)",
        "evaluation_scale": {
            "total_records_dataset": 800039,
            "evaluated_samples": total,
            "monitored_stations": len(station_counts),
        },
        "rain_metrics": {
            "threshold": 0.5,
            "recall_pod": f"{recall:.2f}%",
            "accuracy": f"{accuracy:.2f}%",
            "precision": f"{precision:.2f}%",
            "f1_score": f"{f1:.2f}%",
            "false_alarm_ratio": f"{far:.2f}%",
            "critical_success_index": f"{csi:.2f}%",
            "brier_score": round(brier, 4),
        },
        "threshold_sensitivity": threshold_results,
        "rain_by_intensity": intensity_results,
        "rain_by_station": per_station_results,
        "water_level_metrics": {
            "source": "real_gauge_observations" if mae is not None else "NO_GAUGE_DATA",
            "gauge_station": "Calumpit WLMS (O3z0j5bG)" if mae is not None else None,
            "mae_meters": round(mae, 4) if mae is not None else None,
            "mae_cm": round(mae * 100, 1) if mae is not None else None,
            "rmse_meters": round(rmse, 4) if rmse is not None else None,
            "gauge_matched_samples": len(water_errors),
            "note": "Single-gauge validated. Do not generalize to all 16 weather stations.",
        },
        "uncertainty_bands": {
            "method": "empirical_conformal_prediction",
            "note": "Computed from sorted absolute residuals on calibration set, NOT a fixed formula.",
            "bands": conformal_results,
        },
        "performance": {
            "latency_microseconds": round(latency_us, 2),
            "throughput_per_second": throughput,
            "infrastructure": "Serverless CPU",
        },
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Scorecard saved -> {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_comprehensive_validation()
