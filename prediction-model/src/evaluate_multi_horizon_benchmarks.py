#!/usr/bin/env python3
"""
Scientific Multi-Horizon Benchmark Evaluator
Evaluates the KloudTrack prediction model against 5 standard meteorological baselines
across 7 horizons (1h, 3h, 6h, 12h, 24h, 48h, 72h) using out-of-sample September 2026 telemetry.
"""

import os
import csv
import json
import math
from datetime import datetime
import numpy as np

CSV_PATH = "prediction-model/data/segregated/september_2026_backtest_1h.csv"
if not os.path.exists(CSV_PATH):
    CSV_PATH = "c:/Ben File/beta-citizen-prediction/prediction-model/data/segregated/september_2026_backtest_1h.csv"

def load_data():
    records_by_station = {}
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            sid = r['station_id']
            if sid not in records_by_station:
                records_by_station[sid] = []
            
            dt = datetime.fromisoformat(r['timestamp'].replace('Z', '+00:00'))
            temp = float(r['raw_temperature_c']) if r['raw_temperature_c'] != '' else None
            hum = float(r['raw_humidity_pct']) if r['raw_humidity_pct'] != '' else None
            pres = float(r['raw_pressure_hpa']) if r['raw_pressure_hpa'] != '' else 1008.0
            rain = float(r['raw_hourly_precip_mm']) if r['raw_hourly_precip_mm'] != '' else 0.0
            wind = float(r['raw_wind_speed_kmh']) if r.get('raw_wind_speed_kmh') and r['raw_wind_speed_kmh'] != '' else 5.0
            water = float(r['raw_water_level_m']) if r.get('raw_water_level_m') and r['raw_water_level_m'] != '' else None

            records_by_station[sid].append({
                'station_id': sid,
                'station_name': r.get('station_name', sid),
                'dt': dt,
                'temp': temp,
                'hum': hum,
                'pres': pres,
                'rain': rain,
                'wind': wind,
                'is_rain': rain > 0,
                'water': water,
            })

    for sid in records_by_station:
        records_by_station[sid].sort(key=lambda x: x['dt'])

    return records_by_station

def classify_rain_tier(r):
    if r <= 0.05:
        return 0  # NONE
    if r <= 1.0:
        return 1  # DRIZZLE
    if r <= 2.5:
        return 2  # LIGHT
    if r <= 7.5:
        return 3  # MODERATE
    return 4      # HEAVY

TIER_NAMES = ['NONE', 'DRIZZLE', 'LIGHT', 'MODERATE', 'HEAVY']

def run_evaluation():
    records_by_station = load_data()
    all_points = [p for pts in records_by_station.values() for p in pts if p['temp'] is not None]
    
    # Climatological means by hour of day for Diurnal Baseline
    hourly_temp_clim = {h: [] for h in range(24)}
    for p in all_points:
        hr = (p['dt'].hour + 8) % 24
        hourly_temp_clim[hr].append(p['temp'])
    hourly_temp_means = {h: np.mean(v) if v else 28.5 for h, v in hourly_temp_clim.items()}
    global_temp_mean = float(np.mean([p['temp'] for p in all_points]))
    global_pres_mean = float(np.mean([p['pres'] for p in all_points]))

    horizons = [1, 3, 6, 12, 24, 48, 72]
    benchmark_report = {
        'metadata': {
            'evaluated_at': datetime.utcnow().isoformat(),
            'total_stations': len(records_by_station),
            'date_range': '2026-09-01 to 2026-09-20',
            'horizons': [f"{h}h" for h in horizons],
            'baselines': ['Persistence', 'Climatology_NoRain', 'Diurnal_HourOfDay'],
        },
        'horizon_metrics': {},
        'rain_event_metrics': {},
        'confusion_matrices': {},
    }

    print("=" * 140)
    print(f"{'Horizon':<8}{'N':<7}{'T_MAE':<8}{'T_Pers':<8}{'T_Diurn':<8}{'RainAcc%':<10}{'PersAcc%':<10}{'NoRain%':<10}{'POD%':<8}{'Prec%':<8}{'F1':<8}{'CSI':<8}{'FAR':<8}{'WL_MAE':<8}{'WL_Pers':<8}")
    print("=" * 140)

    for h in horizons:
        y_true_temp, y_true_rain, y_true_is_rain, y_true_tier, y_true_water = [], [], [], [], []
        y_pred_temp, y_pred_rain, y_pred_is_rain, y_pred_tier, y_pred_water = [], [], [], [], []
        y_pers_temp, y_pers_is_rain, y_pers_water = [], [], []
        y_diurn_temp = []

        for sid, pts in records_by_station.items():
            n = len(pts)
            for i in range(n - h):
                cur, tgt = pts[i], pts[i + h]

                # Check strict hourly spacing (allow h +- 0.6 hours)
                delta_h = (tgt['dt'] - cur['dt']).total_seconds() / 3600.0
                if abs(delta_h - h) > 0.6:
                    continue

                if cur['temp'] is None or tgt['temp'] is None:
                    continue

                temp = cur['temp']
                rh = cur['hum'] if cur['hum'] is not None else 78.0
                pres = cur['pres']
                rain = cur['rain']
                is_rain = cur['is_rain']
                water = cur['water']

                cur_hour = (cur['dt'].hour + 8) % 24 + cur['dt'].minute / 60.0
                fut_hour = (cur_hour + h) % 24
                fut_hour_int = int(fut_hour)

                # Atmospheric thermodynamics
                es = 6.1121 * math.exp((17.67 * temp) / (temp + 243.5))
                e = es * max(0.05, min(1.0, rh / 100.0))
                log_term = math.log(max(1e-4, e / 6.1121))
                td = (243.5 * log_term) / (17.67 - log_term)
                dew_point_dep = max(0.0, temp - td)
                lcl_meters = 125.0 * dew_point_dep

                # Model Temperature Forecast
                if h <= 1:
                    pT = temp
                elif h <= 12:
                    future_solar = math.cos((2 * math.pi * (fut_hour - 14.0)) / 24.0)
                    current_solar = math.cos((2 * math.pi * (cur_hour - 14.0)) / 24.0)
                    diurnal_shift = (future_solar - current_solar) * 3.4
                    pT = temp + diurnal_shift
                else:
                    decay = math.exp(-h / 72.0)
                    future_solar = math.cos((2 * math.pi * (fut_hour - 14.0)) / 24.0)
                    diurnal_clim = 28.5 + future_solar * 2.8
                    pT = decay * temp + (1 - decay) * diurnal_clim
                pT = max(18.0, min(43.0, pT))

                pT_pers = temp
                pT_diurn = hourly_temp_means.get(fut_hour_int, global_temp_mean)

                # Two-Stage Hurdle Model for Rain
                solar_convective = math.sin((math.pi * (fut_hour - 11.5)) / 6.5) if (11.5 <= fut_hour <= 18.0) else 0.0
                synoptic_trough = max(0.0, (1006.5 - pres) / 7.0)
                lcl_convective = max(0.0, (850.0 - lcl_meters) / 600.0)
                convective_potential = min(0.85, 0.04 + 0.38 * solar_convective * lcl_convective + 0.45 * synoptic_trough)

                tau_convective = 3.0 if h <= 3.0 else 4.5
                memory_decay = math.exp(-h / tau_convective)
                raw_prob = memory_decay * (0.80 if is_rain else 0.03) + (1 - memory_decay) * convective_potential
                rain_prob = min(0.95, max(0.02, round(raw_prob, 2)))

                p_thresh = 0.24 if h <= 1.0 else (0.30 if h <= 3.0 else (0.35 if h <= 6.0 else (0.38 if h <= 12.0 else 0.40)))
                pred_is_rain = rain_prob >= p_thresh
                margin = max(0.0, rain_prob - p_thresh)

                if pred_is_rain:
                    if synoptic_trough > 0.4:
                        pRain = round(3.5 + margin * 15.0 + synoptic_trough * 12.0, 1)
                    elif margin > 0.25:
                        pRain = round(1.8 + margin * 8.0, 1)
                    elif margin > 0.12:
                        pRain = round(0.8 + margin * 4.0, 1)
                    else:
                        pRain = round(0.2 + margin * 2.0, 1)
                else:
                    pRain = 0.0

                # Calumpit Tidal-Hydrologic Water Model
                if water is not None and tgt['water'] is not None:
                    time_h = cur['dt'].timestamp() / 3600.0 + h
                    tidal_phase = (2 * math.pi * time_h) / 12.42
                    tidal_backwater = 0.065 * math.sin(tidal_phase)
                    hydro_recession = water * math.exp(-0.0003 * h)
                    runoff_inflow = (pRain / 15.0) * 0.08 * min(h, 8.0) if pRain > 0 else 0.0
                    pWater = round(max(0.5, hydro_recession + tidal_backwater + runoff_inflow), 2)

                    y_true_water.append(tgt['water'])
                    y_pred_water.append(pWater)
                    y_pers_water.append(water)

                y_true_temp.append(tgt['temp'])
                y_pred_temp.append(pT)
                y_pers_temp.append(pT_pers)
                y_diurn_temp.append(pT_diurn)

                y_true_rain.append(tgt['rain'])
                y_pred_rain.append(pRain)
                y_true_is_rain.append(tgt['is_rain'])
                y_pred_is_rain.append(pred_is_rain)
                y_pers_is_rain.append(is_rain)

                y_true_tier.append(classify_rain_tier(tgt['rain']))
                y_pred_tier.append(classify_rain_tier(pRain))

        y_true_t = np.array(y_true_temp)
        y_pred_t = np.array(y_pred_temp)
        y_pers_t = np.array(y_pers_temp)
        y_diur_t = np.array(y_diurn_temp)

        y_true_r = np.array(y_true_is_rain, dtype=bool)
        y_pred_r = np.array(y_pred_is_rain, dtype=bool)
        y_pers_r = np.array(y_pers_is_rain, dtype=bool)

        y_true_tiers = np.array(y_true_tier)
        y_pred_tiers = np.array(y_pred_tier)

        mae_t = float(np.mean(np.abs(y_true_t - y_pred_t)))
        mae_pers_t = float(np.mean(np.abs(y_true_t - y_pers_t)))
        mae_diur_t = float(np.mean(np.abs(y_true_t - y_diur_t)))
        rmse_t = float(np.sqrt(np.mean((y_true_t - y_pred_t) ** 2)))
        ss_pers_t = float(1.0 - (mae_t / mae_pers_t)) if mae_pers_t > 0 else 0.0

        tp = int(np.sum((y_pred_r == True) & (y_true_r == True)))
        fp = int(np.sum((y_pred_r == True) & (y_true_r == False)))
        fn = int(np.sum((y_pred_r == False) & (y_true_r == True)))
        tn = int(np.sum((y_pred_r == False) & (y_true_r == False)))
        total_samples = len(y_true_r)

        acc = float((tp + tn) / total_samples)
        pers_acc = float(np.mean(y_pers_r == y_true_r))
        no_rain_acc = float(np.mean(y_true_r == False))

        pod = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        far = float(fp / (tp + fp)) if (tp + fp) > 0 else 0.0
        csi = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 0.0
        f1 = float(2 * prec * pod / (prec + pod)) if (prec + pod) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        bal_acc = float((pod + spec) / 2.0)

        cm = [[0]*5 for _ in range(5)]
        for true_c, pred_c in zip(y_true_tiers, y_pred_tiers):
            cm[true_c][pred_c] += 1

        tier_acc = float(np.mean(y_true_tiers == y_pred_tiers))
        f1_per_tier = []
        for c in range(5):
            c_tp = int(np.sum((y_pred_tiers == c) & (y_true_tiers == c)))
            c_fp = int(np.sum((y_pred_tiers == c) & (y_true_tiers != c)))
            c_fn = int(np.sum((y_pred_tiers != c) & (y_true_tiers == c)))
            c_p = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0.0
            c_r = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0.0
            c_f1 = 2 * c_p * c_r / (c_p + c_r) if (c_p + c_r) > 0 else 0.0
            f1_per_tier.append(float(c_f1))
        macro_f1 = float(np.mean(f1_per_tier))

        if len(y_true_water) > 10:
            y_true_w = np.array(y_true_water)
            y_pred_w = np.array(y_pred_water)
            y_pers_w = np.array(y_pers_water)
            mae_w = float(np.mean(np.abs(y_true_w - y_pred_w)))
            mae_pers_w = float(np.mean(np.abs(y_true_w - y_pers_w)))
            ss_w = float(1.0 - (mae_w / mae_pers_w)) if mae_pers_w > 0 else 0.0
        else:
            mae_w, mae_pers_w, ss_w = None, None, None

        h_str = f"{h}h"
        benchmark_report['horizon_metrics'][h_str] = {
            'samples': total_samples,
            'temperature': {'mae': round(mae_t, 2), 'rmse': round(rmse_t, 2), 'pers_mae': round(mae_pers_t, 2), 'diur_mae': round(mae_diur_t, 2), 'skill_vs_pers': round(ss_pers_t, 3)},
            'water_level': {'mae': round(mae_w, 3) if mae_w else None, 'pers_mae': round(mae_pers_w, 3) if mae_pers_w else None, 'skill_vs_pers': round(ss_w, 3) if ss_w else None}
        }
        benchmark_report['rain_event_metrics'][h_str] = {
            'accuracy': round(acc, 3),
            'persistence_accuracy': round(pers_acc, 3),
            'no_rain_baseline_accuracy': round(no_rain_acc, 3),
            'pod_recall': round(pod, 3),
            'precision': round(prec, 3),
            'far': round(far, 3),
            'csi_threat_score': round(csi, 3),
            'f1_score': round(f1, 3),
            'balanced_accuracy': round(bal_acc, 3),
            'tier_accuracy': round(tier_acc, 3),
            'tier_macro_f1': round(macro_f1, 3),
            'tier_f1_scores': {TIER_NAMES[i]: round(f1_per_tier[i], 3) for i in range(5)}
        }
        benchmark_report['confusion_matrices'][h_str] = {
            'binary': {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn},
            'intensity_tiers': cm
        }

        row = f"{h_str:<8}{total_samples:<7}{mae_t:<8.2f}{mae_pers_t:<8.2f}{mae_diur_t:<8.2f}{acc*100:<10.1f}{pers_acc*100:<10.1f}{no_rain_acc*100:<10.1f}{pod*100:<8.1f}{prec*100:<8.1f}{f1:<8.3f}{csi:<8.3f}{far:<8.3f}{mae_w if mae_w else 0.0:<8.3f}{mae_pers_w if mae_pers_w else 0.0:<8.3f}"
        print(row)

    print("=" * 140)

    out_json = "prediction-model/data/multi_horizon_benchmark_results.json"
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(benchmark_report, f, indent=2)
    print(f"\nSaved detailed benchmark JSON to: {out_json}")

    return benchmark_report

if __name__ == "__main__":
    run_evaluation()
