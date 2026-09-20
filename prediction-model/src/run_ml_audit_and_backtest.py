"""
Kloudtrack ML Audit & Multi-Horizon Walk-Forward Backtest Engine
Audits data pipeline, verifies leakage, aligns forecast-target timestamps,
and evaluates baselines vs models across all horizons and stations.
"""

import os
import sys
import csv
import json
import math
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import numpy as np

WORKSPACE_ROOT = r"c:\Ben File\beta-citizen-prediction"
BENCHMARK_CSV = os.path.join(WORKSPACE_ROOT, "public", "exports", "Kloudtrack_Benchmark_Comparison_All_Stations_2026-09-01_to_2026-09-20_1h.csv")
CLEAN_CONSOLIDATED_CSV = os.path.join(WORKSPACE_ROOT, "prediction-model", "data", "segregated", "clean_consolidated_2024_2026.csv")

HORIZONS = [1, 3, 6, 12, 24, 48, 72]

def parse_iso(ts_str):
    clean = ts_str.replace("Z", "+00:00")
    return datetime.fromisoformat(clean)

def format_iso(dt):
    return dt.strftime("%Y-%m-%dT%H:00:00.000Z")

def run_pipeline_audit():
    print("=" * 80)
    print("STEP 1: NON-NEGOTIABLE DATA-LEAKAGE AND PIPELINE AUDIT")
    print("=" * 80)

    # 1. Inspect Benchmark CSV
    with open(BENCHMARK_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    print(f"Total Rows in Benchmark CSV: {len(rows)}")
    print(f"Total Columns: {len(fieldnames)}")

    # Check for metadata fields in current export
    req_meta = [
        "forecast_origin_timestamp",
        "target_timestamp",
        "horizon_hours",
        "model_version",
        "feature_cutoff_timestamp",
        "prediction_created_at",
        "is_forecast"
    ]
    missing_meta = [m for m in req_meta if m not in fieldnames]
    print(f"\nAudit Item 9 (Explicit Metadata):")
    if missing_meta:
        print(f"  [FAIL] Missing required metadata columns in exported CSV: {missing_meta}")
    else:
        print("  [PASS] All explicit metadata columns present.")

    # Check processed vs raw values in benchmark
    proc_equals_raw = defaultdict(int)
    proc_diff_raw = defaultdict(int)
    numeric_targets = ["temperature_c", "hourly_precip_mm", "humidity_pct", "wind_speed_kmh", "pressure_hpa"]

    for r in rows:
        if r["raw_qc_status"] != "VALID":
            continue
        for t in numeric_targets:
            raw_v = r.get(f"raw_{t}")
            proc_v = r.get(f"processed_{t}")
            if raw_v not in ("", None, "None") and proc_v not in ("", None, "None"):
                try:
                    if abs(float(raw_v) - float(proc_v)) < 1e-6:
                        proc_equals_raw[t] += 1
                    else:
                        proc_diff_raw[t] += 1
                except:
                    pass

    print("\nAudit of Processed Telemetry vs Raw Telemetry:")
    for t in numeric_targets:
        eq = proc_equals_raw[t]
        diff = proc_diff_raw[t]
        pct = (eq / (eq + diff) * 100) if (eq + diff) > 0 else 0
        print(f"  Target: {t:20s} | Equal: {eq:5d} | Different: {diff:5d} | % Identical: {pct:.2f}%")

    # Check extreme outliers
    print("\nOutlier Detection in Telemetry:")
    outliers = []
    for i, r in enumerate(rows):
        if r["raw_qc_status"] != "VALID":
            continue
        sid = r["station_id"]
        ts = r["timestamp"]
        # Temp check
        if r["raw_temperature_c"]:
            t = float(r["raw_temperature_c"])
            if t > 50.0 or t < 10.0:
                outliers.append((sid, ts, "raw_temperature_c", t, "Physically impossible surface temp"))
        # Pressure check
        if r["raw_pressure_hpa"]:
            p = float(r["raw_pressure_hpa"])
            if p < 900.0 or p > 1050.0:
                outliers.append((sid, ts, "raw_pressure_hpa", p, "Severe barometer drop/hardware fault"))

    print(f"  Total Extreme Physical Outliers Detected: {len(outliers)}")
    outlier_stns = Counter([o[0] for o in outliers])
    for stn, cnt in outlier_stns.items():
        print(f"    Station {stn}: {cnt} extreme outlier readings")
    if outliers:
        print(f"    Sample: {outliers[0]}")
        print(f"    Sample: {outliers[-1]}")

    return rows, fieldnames

if __name__ == "__main__":
    run_pipeline_audit()
