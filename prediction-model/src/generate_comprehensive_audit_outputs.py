"""
Generates the Corrected Benchmark CSV and Comprehensive Metrics Breakdown
for all 12 deliverables in the ML Audit & Improvement prompt.
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
CORRECTED_BENCHMARK_CSV = os.path.join(WORKSPACE_ROOT, "public", "exports", "Kloudtrack_Corrected_Audited_Benchmark_2026-09-01_to_2026-09-20.csv")
DETAILED_METRICS_OUTPUT = os.path.join(WORKSPACE_ROOT, "prediction-model", "data", "detailed_audit_report_data.json")

HORIZONS = [1, 3, 6, 12, 24, 48, 72]

def parse_iso(ts_str):
    return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))

def format_iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")

def run_generate():
    print("Loading data...")
    data = {}
    stn_names = {}
    with open(BENCHMARK_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            sid = r["station_id"]
            ts = r["timestamp"]
            stn_names[sid] = r["station_name"]
            data[(sid, ts)] = r

    # Station hourly climatology from clean historical dataset
    stn_hour_climatology = defaultdict(lambda: defaultdict(list))
    with open(CLEAN_CONSOLIDATED_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            ts_str = r["timestamp"]
            sid = r["station_id"]
            dt = parse_iso(ts_str)
            if dt < datetime(2026, 8, 1, tzinfo=dt.tzinfo):
                pht_hour = (dt.hour + 8) % 24
                try:
                    t = float(r["temperature_c"]) if r["temperature_c"] else None
                    if t is not None and 15.0 <= t <= 45.0:
                        stn_hour_climatology[sid][pht_hour].append(t)
                except:
                    pass

    clim_means = {}
    for sid, h_dict in stn_hour_climatology.items():
        clim_means[sid] = {}
        for h_val, t_list in h_dict.items():
            clim_means[sid][h_val] = float(np.mean(t_list))

    print("Building corrected long-format benchmark dataset with explicit metadata...")
    corrected_rows = []

    # Store metrics per horizon and target
    metrics_summary = {
        "outliers": {
            "temperature_spikes_over_45c": 0,
            "pressure_dropouts_under_940hpa": 0,
            "stations_affected": []
        },
        "horizons": {}
    }

    stn_affected = set()

    for h in HORIZONS:
        h_key = f"{h}h"
        metrics_summary["horizons"][h_key] = {
            "sample_count": 0,
            "temperature": {},
            "water_level": {},
            "rain_state": {},
            "humidity": {},
            "pressure": {},
            "wind_speed": {}
        }

        # Accumulators for this horizon
        act_t, mod_t, per_t, imp_t = [], [], [], []
        act_t_raw, mod_t_raw, per_t_raw = [], [], []

        act_wl, mod_wl, per_wl, imp_wl = [], [], [], []

        act_r, mod_r, per_r, imp_r, imp_prob = [], [], [], [], []

        act_h_list, mod_h_list, per_h_list = [], [], []
        act_p_list, mod_p_list, per_p_list = [], [], []
        act_w_list, mod_w_list, per_w_list = [], [], []

        for (sid, ts_origin_str), r_origin in sorted(data.items(), key=lambda x: (x[0][0], x[0][1])):
            if r_origin["raw_qc_status"] != "VALID":
                continue

            t_origin = parse_iso(ts_origin_str)
            t_target = t_origin + timedelta(hours=h)
            ts_target_str = format_iso(t_target)

            r_target = data.get((sid, ts_target_str))
            if not r_target or r_target["raw_qc_status"] != "VALID":
                continue

            # Check for outliers
            raw_t_orig = float(r_origin["raw_temperature_c"]) if r_origin["raw_temperature_c"] else None
            raw_t_targ = float(r_target["raw_temperature_c"]) if r_target["raw_temperature_c"] else None
            pred_t_mod = float(r_origin[f"pred_{h}h_temperature_c"]) if r_origin[f"pred_{h}h_temperature_c"] else None

            raw_p_orig = float(r_origin["raw_pressure_hpa"]) if r_origin["raw_pressure_hpa"] else None
            raw_p_targ = float(r_target["raw_pressure_hpa"]) if r_target["raw_pressure_hpa"] else None
            pred_p_mod = float(r_origin[f"pred_{h}h_pressure_hpa"]) if r_origin[f"pred_{h}h_pressure_hpa"] else None

            qc_flags = []
            if raw_t_orig is not None and (raw_t_orig > 45.0 or raw_t_orig < 10.0):
                qc_flags.append("ORIGIN_TEMP_OUTLIER")
                metrics_summary["outliers"]["temperature_spikes_over_45c"] += 1
                stn_affected.add(sid)
            if raw_t_targ is not None and (raw_t_targ > 45.0 or raw_t_targ < 10.0):
                qc_flags.append("TARGET_TEMP_OUTLIER")
                metrics_summary["outliers"]["temperature_spikes_over_45c"] += 1
                stn_affected.add(sid)
            if raw_p_orig is not None and raw_p_orig < 940.0:
                qc_flags.append("ORIGIN_PRESSURE_DROPOUT")
                metrics_summary["outliers"]["pressure_dropouts_under_940hpa"] += 1
                stn_affected.add(sid)
            if raw_p_targ is not None and raw_p_targ < 940.0:
                qc_flags.append("TARGET_PRESSURE_DROPOUT")
                metrics_summary["outliers"]["pressure_dropouts_under_940hpa"] += 1
                stn_affected.add(sid)

            qc_status_final = "REJECTED_OUTLIER: " + "|".join(qc_flags) if qc_flags else "PASSED_QC"

            # Compute improved models
            # 1. Improved Temp
            t_improved = None
            if raw_t_orig is not None:
                target_pht_hour = (t_target.hour + 8) % 24
                orig_pht_hour = (t_origin.hour + 8) % 24
                clim_t = clim_means.get(sid, {}).get(target_pht_hour, 28.5)
                diurnal_shift = 3.2 * (math.cos(2 * math.pi * (target_pht_hour - 14.0) / 24.0) - math.cos(2 * math.pi * (orig_pht_hour - 14.0) / 24.0))
                decay_factor = math.exp(-h / 36.0)
                t_improved = round(decay_factor * (raw_t_orig + diurnal_shift) + (1 - decay_factor) * clim_t, 2)

            # 2. Improved Water Level
            wl_improved = None
            raw_wl_orig = float(r_origin["raw_water_level_m"]) if r_origin["raw_water_level_m"] else None
            raw_wl_targ = float(r_target["raw_water_level_m"]) if r_target["raw_water_level_m"] else None
            pred_wl_mod = float(r_origin[f"pred_{h}h_water_level_m"]) if r_origin[f"pred_{h}h_water_level_m"] else None

            if sid == "O3z0j5bG" and raw_wl_orig is not None:
                wl_improved = round(raw_wl_orig * math.exp(-0.0012 * h), 3)

            # 3. Improved Rain
            raw_rain_orig = r_origin["raw_is_raining"] == "TRUE"
            raw_rain_targ = r_target["raw_is_raining"] == "TRUE"
            pred_rain_mod = r_origin[f"pred_{h}h_is_raining"] == "TRUE"

            orig_h_val = float(r_origin["raw_humidity_pct"]) if r_origin["raw_humidity_pct"] else 75.0
            orig_p_val = float(r_origin["raw_pressure_hpa"]) if r_origin["raw_pressure_hpa"] else 1008.0
            p_drop = max(0.0, (1010.0 - orig_p_val) / 10.0) if orig_p_val >= 900.0 else 0.2
            h_excess = max(0.0, (orig_h_val - 75.0) / 25.0) if orig_h_val <= 100.0 else 0.5
            tau_rain = 4.0
            decay = math.exp(-h / tau_rain)
            improved_prob = decay * (0.65 if raw_rain_orig else 0.08) + (1 - decay) * (0.12 + 0.15 * h_excess + 0.08 * p_drop)
            improved_prob = round(min(0.90, max(0.05, improved_prob)), 3)
            rain_improved = improved_prob >= 0.24

            # Write row to corrected benchmark export
            row_out = {
                "forecast_origin_timestamp": ts_origin_str,
                "target_timestamp": ts_target_str,
                "horizon_hours": h,
                "model_version": "PINN-LNN-v3.2-Audited",
                "feature_cutoff_timestamp": ts_origin_str,
                "prediction_created_at": ts_origin_str,
                "is_forecast": "TRUE",
                "station_id": sid,
                "station_name": stn_names.get(sid, ""),
                "qc_status": qc_status_final,
                # Temperature
                "actual_temperature_c": raw_t_targ,
                "pred_model_temperature_c": pred_t_mod,
                "pred_persist_temperature_c": raw_t_orig,
                "pred_improved_temperature_c": t_improved,
                "err_model_temperature_c": round(pred_t_mod - raw_t_targ, 2) if (pred_t_mod is not None and raw_t_targ is not None) else "",
                "err_persist_temperature_c": round(raw_t_orig - raw_t_targ, 2) if (raw_t_orig is not None and raw_t_targ is not None) else "",
                "err_improved_temperature_c": round(t_improved - raw_t_targ, 2) if (t_improved is not None and raw_t_targ is not None) else "",
                # Water Level
                "actual_water_level_m": raw_wl_targ if sid == "O3z0j5bG" else "",
                "pred_model_water_level_m": pred_wl_mod if sid == "O3z0j5bG" else "",
                "pred_persist_water_level_m": raw_wl_orig if sid == "O3z0j5bG" else "",
                "pred_improved_water_level_m": wl_improved if sid == "O3z0j5bG" else "",
                # Rain State
                "actual_is_raining": str(raw_rain_targ).upper(),
                "pred_model_is_raining": str(pred_rain_mod).upper(),
                "pred_persist_is_raining": str(raw_rain_orig).upper(),
                "pred_improved_is_raining": str(rain_improved).upper(),
                "pred_improved_rain_prob": improved_prob,
                # Rain Intensity
                "actual_rain_intensity": r_target.get("raw_rain_intensity", ""),
                "pred_model_rain_intensity": r_origin.get(f"pred_{h}h_rain_intensity", ""),
                # Flood Stage
                "actual_flood_stage": r_target.get("raw_flood_stage", ""),
                "pred_model_flood_stage": r_origin.get(f"pred_{h}h_flood_stage", "")
            }
            corrected_rows.append(row_out)

            # Accumulate for metrics if clean
            if "REJECTED_OUTLIER" not in qc_status_final:
                if raw_t_orig is not None and raw_t_targ is not None and pred_t_mod is not None:
                    act_t.append(raw_t_targ)
                    mod_t.append(pred_t_mod)
                    per_t.append(raw_t_orig)
                    imp_t.append(t_improved)

                if sid == "O3z0j5bG" and raw_wl_orig is not None and raw_wl_targ is not None and pred_wl_mod is not None:
                    act_wl.append(raw_wl_targ)
                    mod_wl.append(pred_wl_mod)
                    per_wl.append(raw_wl_orig)
                    imp_wl.append(wl_improved)

                act_r.append(1 if raw_rain_targ else 0)
                mod_r.append(1 if pred_rain_mod else 0)
                per_r.append(1 if raw_rain_orig else 0)
                imp_r.append(1 if rain_improved else 0)
                imp_prob.append(improved_prob)

                # Humidity
                raw_h_orig = float(r_origin["raw_humidity_pct"]) if r_origin["raw_humidity_pct"] else None
                raw_h_targ = float(r_target["raw_humidity_pct"]) if r_target["raw_humidity_pct"] else None
                pred_h_mod = float(r_origin[f"pred_{h}h_humidity_pct"]) if r_origin[f"pred_{h}h_humidity_pct"] else None
                if raw_h_orig and raw_h_targ and pred_h_mod and 20 <= raw_h_orig <= 100 and 20 <= raw_h_targ <= 100:
                    act_h_list.append(raw_h_targ)
                    mod_h_list.append(pred_h_mod)
                    per_h_list.append(raw_h_orig)

                # Pressure
                if raw_p_orig and raw_p_targ and pred_p_mod and 940 <= raw_p_orig <= 1040 and 940 <= raw_p_targ <= 1040:
                    act_p_list.append(raw_p_targ)
                    mod_p_list.append(pred_p_mod)
                    per_p_list.append(raw_p_orig)

                # Wind
                raw_w_orig = float(r_origin["raw_wind_speed_kmh"]) if r_origin["raw_wind_speed_kmh"] else None
                raw_w_targ = float(r_target["raw_wind_speed_kmh"]) if r_target["raw_wind_speed_kmh"] else None
                pred_w_mod = float(r_origin[f"pred_{h}h_wind_speed_kmh"]) if r_origin[f"pred_{h}h_wind_speed_kmh"] else None
                if raw_w_orig is not None and raw_w_targ is not None and pred_w_mod is not None and 0 <= raw_w_orig <= 120 and 0 <= raw_w_targ <= 120:
                    act_w_list.append(raw_w_targ)
                    mod_w_list.append(pred_w_mod)
                    per_w_list.append(raw_w_orig)

        # Store summary metrics
        metrics_summary["horizons"][h_key]["sample_count"] = len(act_t)
        
        # Helper for continuous metrics
        def calc_cont(act, prd):
            a, p = np.array(act), np.array(prd)
            d = p - a
            return {
                "mae": round(float(np.mean(np.abs(d))), 4),
                "rmse": round(float(math.sqrt(np.mean(d ** 2))), 4),
                "bias": round(float(np.mean(d)), 4),
                "max_err": round(float(np.max(np.abs(d))), 4)
            }

        # Temperature metrics
        metrics_summary["horizons"][h_key]["temperature"]["model"] = calc_cont(act_t, mod_t)
        metrics_summary["horizons"][h_key]["temperature"]["persistence"] = calc_cont(act_t, per_t)
        metrics_summary["horizons"][h_key]["temperature"]["improved"] = calc_cont(act_t, imp_t)

        # Water level metrics
        if act_wl:
            metrics_summary["horizons"][h_key]["water_level"]["model"] = calc_cont(act_wl, mod_wl)
            metrics_summary["horizons"][h_key]["water_level"]["persistence"] = calc_cont(act_wl, per_wl)
            metrics_summary["horizons"][h_key]["water_level"]["improved"] = calc_cont(act_wl, imp_wl)

        # Humidity metrics
        if act_h_list:
            metrics_summary["horizons"][h_key]["humidity"]["model"] = calc_cont(act_h_list, mod_h_list)
            metrics_summary["horizons"][h_key]["humidity"]["persistence"] = calc_cont(act_h_list, per_h_list)

        # Pressure metrics
        if act_p_list:
            metrics_summary["horizons"][h_key]["pressure"]["model"] = calc_cont(act_p_list, mod_p_list)
            metrics_summary["horizons"][h_key]["pressure"]["persistence"] = calc_cont(act_p_list, per_p_list)

        # Wind metrics
        if act_w_list:
            metrics_summary["horizons"][h_key]["wind_speed"]["model"] = calc_cont(act_w_list, mod_w_list)
            metrics_summary["horizons"][h_key]["wind_speed"]["persistence"] = calc_cont(act_w_list, per_w_list)

        # Rain classification metrics helper
        def calc_cls(act, prd, prb=None):
            a, p = np.array(act), np.array(prd)
            tp = int(np.sum((a == 1) & (p == 1)))
            fp = int(np.sum((a == 0) & (p == 1)))
            tn = int(np.sum((a == 0) & (p == 0)))
            fn = int(np.sum((a == 1) & (p == 0)))
            total = len(a)
            acc = (tp + tn) / total if total > 0 else 0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            b_acc = (rec + spec) / 2.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            far = fp / (fp + tn) if (fp + tn) > 0 else 0
            denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            mcc = ((tp * tn) - (fp * fn)) / denom if denom > 0 else 0

            brier = None
            if prb is not None:
                brier = round(float(np.mean((np.array(prb) - a) ** 2)), 4)

            return {
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "specificity": round(spec, 4),
                "balanced_accuracy": round(b_acc, 4),
                "f1": round(f1, 4),
                "false_alarm_rate": round(far, 4),
                "mcc": round(mcc, 4),
                "brier_score": brier
            }

        metrics_summary["horizons"][h_key]["rain_state"]["model"] = calc_cls(act_r, mod_r)
        metrics_summary["horizons"][h_key]["rain_state"]["persistence"] = calc_cls(act_r, per_r)
        metrics_summary["horizons"][h_key]["rain_state"]["improved"] = calc_cls(act_r, imp_r, imp_prob)

    metrics_summary["outliers"]["stations_affected"] = list(stn_affected)

    # Save corrected benchmark export CSV
    fieldnames = list(corrected_rows[0].keys())
    with open(CORRECTED_BENCHMARK_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in corrected_rows:
            writer.writerow(r)

    print(f"Saved {len(corrected_rows)} long-format auditable rows to: {CORRECTED_BENCHMARK_CSV}")

    # Save metrics JSON
    with open(DETAILED_METRICS_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    print(f"Saved detailed metrics breakdown to: {DETAILED_METRICS_OUTPUT}")

if __name__ == "__main__":
    run_generate()
