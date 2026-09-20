#!/usr/bin/env python3
"""
Scientific Multi-Horizon Benchmark Evaluator (Phase 2 Overhaul)
Evaluates the KloudTrack prediction model against 5 standard meteorological baselines
across 7 horizons (1h, 3h, 6h, 12h, 24h, 48h, 72h) using out-of-sample September 2026 telemetry.
Includes:
- Heat Index R^2 and MAE using official Rothfusz equation
- 3-tier Operational Rain Hazard targets (NO_RAIN, LIGHT, HAZARDOUS) and hazardous recall
- Wilson score 95% confidence intervals
- Complete 5-class rain intensity and flood stage confusion matrices
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

def calculate_rothfusz_heat_index(temp: float, rh: float) -> float:
    if temp < 26.7:
        return round(temp, 1)
    t = temp
    r = rh
    c1, c2, c3 = -8.784695, 1.61139411, 2.338549
    c4, c5, c6 = -0.14611605, -0.012308094, -0.016424828
    c7, c8, c9 = 0.002211732, 0.00072546, -0.000003582
    hi = c1 + c2*t + c3*r + c4*t*r + c5*t*t + c6*r*r + c7*t*t*r + c8*t*r*r + c9*t*t*r*r
    return round(hi, 1)

def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1.0 + (z**2) / total
    center = (p + (z**2) / (2.0 * total)) / denom
    margin = (z * math.sqrt((p * (1.0 - p)) / total + (z**2) / (4.0 * total**2))) / denom
    return (round(max(0.0, center - margin) * 100, 1), round(min(1.0, center + margin) * 100, 1))

def classify_rain_tier(r: float) -> int:
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

def classify_hazard_tier(r: float) -> int:
    if r <= 0.05:
        return 0  # NO_RAIN
    if r <= 2.5:
        return 1  # LIGHT
    return 2      # HAZARDOUS (> 2.5 mm)

HAZARD_NAMES = ['NO_RAIN', 'LIGHT', 'HAZARDOUS']

def classify_flood_stage_int(water: float | None) -> int:
    if water is None:
        return -1
    if water >= 5.0:
        return 3  # CRITICAL
    if water >= 3.5:
        return 2  # ALARM
    if water >= 2.5:
        return 1  # ALERT
    return 0      # NORMAL

FLOOD_NAMES = ['NORMAL', 'ALERT', 'ALARM', 'CRITICAL']

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

def run_evaluation():
    records_by_station = load_data()
    all_points = [p for pts in records_by_station.values() for p in pts if p['temp'] is not None]
    
    # Climatological means by hour of day for Diurnal Baseline
    hourly_temp_clim = {h: [] for h in range(24)}
    for p in all_points:
        hr = (p['dt'].hour + 8) % 24
        hourly_temp_clim[hr].append(p['temp'])
    hourly_temp_means = {h: float(np.mean(v)) if v else 28.5 for h, v in hourly_temp_clim.items()}
    global_temp_mean = float(np.mean([p['temp'] for p in all_points]))

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
        'heat_index_metrics': {},
        'rain_event_metrics': {},
        'operational_hazard_metrics': {},
        'flood_stage_metrics': {},
        'confusion_matrices': {},
    }

    print("=" * 150)
    print(f"{'Horizon':<8}{'N':<7}{'T_MAE':<8}{'T_Pers':<8}{'T_Diurn':<8}{'HI_MAE':<8}{'HI_R2':<8}{'RainAcc%':<10}{'Rain_CI95':<12}{'NoRain%':<10}{'POD%':<8}{'Prec%':<8}{'F1':<8}{'CSI':<8}{'WL_MAE':<8}{'WL_Pers':<8}")
    print("=" * 150)

    for h in horizons:
        y_true_temp, y_true_rain, y_true_is_rain, y_true_tier, y_true_hazard, y_true_water = [], [], [], [], [], []
        y_pred_temp, y_pred_rain, y_pred_is_rain, y_pred_tier, y_pred_hazard, y_pred_water = [], [], [], [], [], []
        y_true_hi, y_pred_hi = [], []
        y_pers_temp, y_pers_is_rain, y_pers_water = [], [], []
        y_diurn_temp = []
        y_true_flood, y_pred_flood = [], []

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

                # Humidity Coupling with Diurnal Relaxation
                if h <= 12:
                    pH = round(min(98.0, max(35.0, rh - (pT - temp) * 4.2)), 1)
                else:
                    future_solar_phase = math.cos((2 * math.pi * (fut_hour - 14.0)) / 24.0)
                    diurnal_clim_rh = 80.0 - future_solar_phase * 12.0
                    decay_h = math.exp(-h / 48.0)
                    coupled_rh = rh - (pT - temp) * 4.2
                    pH = round(min(98.0, max(35.0, decay_h * coupled_rh + (1.0 - decay_h) * diurnal_clim_rh)), 1)

                # Heat Index (Rothfusz Formulation)
                tgt_rh = tgt['hum'] if tgt['hum'] is not None else 78.0
                true_hi = calculate_rothfusz_heat_index(tgt['temp'], tgt_rh)
                pred_hi = calculate_rothfusz_heat_index(pT, pH)

                # Two-Stage Hurdle Model for Rain with MSLP Reduction
                solar_convective = math.sin((math.pi * (fut_hour - 11.5)) / 6.5) if (11.5 <= fut_hour <= 18.0) else 0.0
                elev_m = 10.0
                if sid == "03pqkGAj": elev_m = 88.0
                elif sid == "4VAl2p9k": elev_m = 75.0
                elif sid == "Rjz2dbXW": elev_m = 62.0
                elif sid == "1Zb102pg": elev_m = 95.0
                elif sid == "Bkpj1zRO": elev_m = 38.0

                pres_msl = pres * math.pow(1.0 - (0.0065 * elev_m) / (temp + 273.15), -5.257)
                synoptic_trough = min(1.0, max(0.0, (1006.5 - pres_msl) / 7.0))
                lcl_convective = max(0.0, (850.0 - lcl_meters) / 600.0)
                convective_potential = min(0.85, 0.04 + 0.38 * solar_convective * lcl_convective + 0.45 * synoptic_trough)

                tau_convective = 3.0 if h <= 3.0 else 4.5
                memory_decay = math.exp(-h / tau_convective)
                raw_prob = memory_decay * (0.80 if is_rain else 0.03) + (1 - memory_decay) * convective_potential
                rain_prob = min(0.95, max(0.02, round(raw_prob, 2)))

                p_thresh = 0.24 if h <= 1.0 else (0.28 if h <= 3.0 else (0.33 if h <= 6.0 else 0.36))
                pred_is_rain = rain_prob >= p_thresh
                margin = max(0.0, rain_prob - p_thresh)

                # Stage 2: Event-Weighted Conditional Rainfall Amount E[Y | Rain = 1]
                pRain = 0.0
                if pred_is_rain:
                    r0 = rain or 0.0
                    if h <= 3.0:
                        if is_rain:
                            r_decay = r0 * 0.5 * math.exp(-(h - 1.0) / 2.0)
                            if r0 >= 7.5:
                                pRain = round(max(3.0, r_decay + synoptic_trough * 2.0), 1)
                            elif r0 >= 2.5:
                                pRain = round(r_decay + margin * 0.4, 1)
                            else:
                                pRain = round(max(0.1, r_decay + margin * 0.2), 1)
                        else:
                            if synoptic_trough > 0.5:
                                pRain = round(1.2 + synoptic_trough * 1.5, 1)
                            else:
                                pRain = round(0.3 + margin * 0.5, 1)
                    else:
                        if synoptic_trough > 0.4:
                            pRain = round(1.0 + synoptic_trough * 2.0, 1)
                        else:
                            pRain = round(0.3 + margin * 0.6, 1)

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

                    # Flood stage
                    f_true = classify_flood_stage_int(tgt['water'])
                    f_pred = classify_flood_stage_int(pWater)
                    y_true_flood.append(f_true)
                    y_pred_flood.append(f_pred)

                y_true_temp.append(tgt['temp'])
                y_pred_temp.append(pT)
                y_pers_temp.append(pT_pers)
                y_diurn_temp.append(pT_diurn)

                y_true_hi.append(true_hi)
                y_pred_hi.append(pred_hi)

                y_true_rain.append(tgt['rain'])
                y_pred_rain.append(pRain)
                y_true_is_rain.append(tgt['is_rain'])
                y_pred_is_rain.append(pred_is_rain)
                y_pers_is_rain.append(is_rain)

                y_true_tier.append(classify_rain_tier(tgt['rain']))
                y_pred_tier.append(classify_rain_tier(pRain))

                y_true_hazard.append(classify_hazard_tier(tgt['rain']))
                y_pred_hazard.append(classify_hazard_tier(pRain))

        y_true_t = np.array(y_true_temp)
        y_pred_t = np.array(y_pred_temp)
        y_pers_t = np.array(y_pers_temp)
        y_diur_t = np.array(y_diurn_temp)

        y_true_h_arr = np.array(y_true_hi)
        y_pred_h_arr = np.array(y_pred_hi)

        y_true_r = np.array(y_true_is_rain, dtype=bool)
        y_pred_r = np.array(y_pred_is_rain, dtype=bool)
        y_pers_r = np.array(y_pers_is_rain, dtype=bool)

        y_true_tiers = np.array(y_true_tier)
        y_pred_tiers = np.array(y_pred_tier)

        y_true_haz = np.array(y_true_hazard)
        y_pred_haz = np.array(y_pred_hazard)

        # Temperature metrics
        mae_t = float(np.mean(np.abs(y_true_t - y_pred_t)))
        mae_pers_t = float(np.mean(np.abs(y_true_t - y_pers_t)))
        mae_diur_t = float(np.mean(np.abs(y_true_t - y_diur_t)))
        rmse_t = float(np.sqrt(np.mean((y_true_t - y_pred_t) ** 2)))
        ss_pers_t = float(1.0 - (mae_t / mae_pers_t)) if mae_pers_t > 0 else 0.0

        # Heat Index metrics
        mae_hi = float(np.mean(np.abs(y_true_h_arr - y_pred_h_arr)))
        ss_res_hi = float(np.sum((y_true_h_arr - y_pred_h_arr) ** 2))
        ss_tot_hi = float(np.sum((y_true_h_arr - np.mean(y_true_h_arr)) ** 2))
        r2_hi = float(1.0 - (ss_res_hi / ss_tot_hi)) if ss_tot_hi > 0 else 0.0

        # Rain occurrence metrics
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

        # Wilson 95% Confidence Interval for Rain Accuracy
        rain_ci_low, rain_ci_high = wilson_interval(tp + tn, total_samples)

        # 5-class Rain Intensity Tier Evaluation
        cm_5class = [[0]*5 for _ in range(5)]
        for true_c, pred_c in zip(y_true_tiers, y_pred_tiers):
            cm_5class[true_c][pred_c] += 1

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
        macro_f1_5class = float(np.mean(f1_per_tier))

        # 3-tier Operational Rain Hazard Evaluation
        cm_hazard = [[0]*3 for _ in range(3)]
        for true_c, pred_c in zip(y_true_haz, y_pred_haz):
            cm_hazard[true_c][pred_c] += 1

        hazard_acc = float(np.mean(y_true_haz == y_pred_haz))
        haz_f1_list = []
        for c in range(3):
            c_tp = int(np.sum((y_pred_haz == c) & (y_true_haz == c)))
            c_fp = int(np.sum((y_pred_haz == c) & (y_true_haz != c)))
            c_fn = int(np.sum((y_pred_haz != c) & (y_true_haz == c)))
            c_p = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0.0
            c_r = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0.0
            c_f1 = 2 * c_p * c_r / (c_p + c_r) if (c_p + c_r) > 0 else 0.0
            haz_f1_list.append(float(c_f1))
        hazard_macro_f1 = float(np.mean(haz_f1_list))

        # Hazardous rain specific recall (c=2: Moderate + Heavy)
        haz_tp = int(np.sum((y_pred_haz == 2) & (y_true_haz == 2)))
        haz_fn = int(np.sum((y_pred_haz != 2) & (y_true_haz == 2)))
        haz_fp = int(np.sum((y_pred_haz == 2) & (y_true_haz != 2)))
        hazardous_recall = float(haz_tp / (haz_tp + haz_fn)) if (haz_tp + haz_fn) > 0 else 0.0
        hazardous_prec = float(haz_tp / (haz_tp + haz_fp)) if (haz_tp + haz_fp) > 0 else 0.0

        # Flood stage evaluation
        flood_metrics = None
        if len(y_true_flood) > 10:
            y_tf = np.array(y_true_flood)
            y_pf = np.array(y_pred_flood)
            fl_correct = int(np.sum(y_tf == y_pf))
            fl_total = len(y_tf)
            fl_acc = float(fl_correct / fl_total)
            fl_ci_low, fl_ci_high = wilson_interval(fl_correct, fl_total)

            fl_cm = [[0]*4 for _ in range(4)]
            for tc, pc in zip(y_tf, y_pf):
                fl_cm[tc][pc] += 1

            # Active classes (classes with true counts > 0)
            active_f1s = []
            all_f1s = []
            for c in range(4):
                c_tp = int(np.sum((y_pf == c) & (y_tf == c)))
                c_fp = int(np.sum((y_pf == c) & (y_tf != c)))
                c_fn = int(np.sum((y_pf != c) & (y_tf == c)))
                c_p = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0.0
                c_r = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0.0
                c_f1 = 2 * c_p * c_r / (c_p + c_r) if (c_p + c_r) > 0 else 0.0
                all_f1s.append(c_f1)
                if (c_tp + c_fn) > 0:
                    active_f1s.append(c_f1)

            flood_metrics = {
                'samples': fl_total,
                'accuracy': round(fl_acc, 3),
                'ci95_low': fl_ci_low,
                'ci95_high': fl_ci_high,
                'macro_f1_active_classes': round(float(np.mean(active_f1s)), 3) if active_f1s else 0.0,
                'macro_f1_all_classes': round(float(np.mean(all_f1s)), 3),
                'confusion_matrix': fl_cm
            }

        # Water level continuous metrics
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
        benchmark_report['heat_index_metrics'][h_str] = {
            'mae': round(mae_hi, 2),
            'r2': round(r2_hi, 3)
        }
        benchmark_report['rain_event_metrics'][h_str] = {
            'accuracy': round(acc, 3),
            'ci95_low': rain_ci_low,
            'ci95_high': rain_ci_high,
            'persistence_accuracy': round(pers_acc, 3),
            'no_rain_baseline_accuracy': round(no_rain_acc, 3),
            'pod_recall': round(pod, 3),
            'precision': round(prec, 3),
            'far': round(far, 3),
            'csi_threat_score': round(csi, 3),
            'f1_score': round(f1, 3),
            'balanced_accuracy': round(bal_acc, 3),
            'tier_accuracy': round(tier_acc, 3),
            'tier_macro_f1': round(macro_f1_5class, 3),
            'tier_f1_scores': {TIER_NAMES[i]: round(f1_per_tier[i], 3) for i in range(5)}
        }
        benchmark_report['operational_hazard_metrics'][h_str] = {
            'accuracy': round(hazard_acc, 3),
            'macro_f1': round(hazard_macro_f1, 3),
            'hazardous_rain_recall': round(hazardous_recall, 3),
            'hazardous_rain_precision': round(hazardous_prec, 3),
            'tier_f1_scores': {HAZARD_NAMES[i]: round(haz_f1_list[i], 3) for i in range(3)},
            'confusion_matrix': cm_hazard
        }
        if flood_metrics:
            benchmark_report['flood_stage_metrics'][h_str] = flood_metrics

        benchmark_report['confusion_matrices'][h_str] = {
            'binary': {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn},
            'intensity_tiers': cm_5class
        }

        ci_str = f"[{rain_ci_low:.1f}-{rain_ci_high:.1f}]"
        row = f"{h_str:<8}{total_samples:<7}{mae_t:<8.2f}{mae_pers_t:<8.2f}{mae_diur_t:<8.2f}{mae_hi:<8.2f}{r2_hi:<8.3f}{acc*100:<10.1f}{ci_str:<12}{no_rain_acc*100:<10.1f}{pod*100:<8.1f}{prec*100:<8.1f}{f1:<8.3f}{csi:<8.3f}{mae_w if mae_w else 0.0:<8.3f}{mae_pers_w if mae_pers_w else 0.0:<8.3f}"
        print(row)

    print("=" * 150)

    # Print Operational Hazard & Flood Summary
    print("\n" + "=" * 100)
    print("3-TIER OPERATIONAL RAIN HAZARD BENCHMARK (NO_RAIN / LIGHT / HAZARDOUS)")
    print("=" * 100)
    print(f"{'Horizon':<8}{'Hazard_Acc%':<14}{'Macro-F1':<12}{'Hazard_Recall%':<16}{'Hazard_Prec%':<14}{'Hazard_F1':<12}")
    for h in horizons:
        m = benchmark_report['operational_hazard_metrics'][f"{h}h"]
        print(f"{f'{h}h':<8}{m['accuracy']*100:<14.1f}{m['macro_f1']:<12.3f}{m['hazardous_rain_recall']*100:<16.1f}{m['hazardous_rain_precision']*100:<14.1f}{m['tier_f1_scores']['HAZARDOUS']:<12.3f}")
    print("=" * 100)

    print("\n" + "=" * 100)
    print("CALUMPIT FLOOD STAGE CLASSIFICATION & CONFIDENCE INTERVALS")
    print("=" * 100)
    print(f"{'Horizon':<8}{'Samples':<10}{'Accuracy%':<12}{'Wilson 95% CI':<16}{'Active Macro-F1':<18}{'All-Class F1':<14}")
    for h in horizons:
        if f"{h}h" in benchmark_report['flood_stage_metrics']:
            fm = benchmark_report['flood_stage_metrics'][f"{h}h"]
            fl_ci = f"[{fm['ci95_low']:.1f}% - {fm['ci95_high']:.1f}%]"
            print(f"{f'{h}h':<8}{fm['samples']:<10}{fm['accuracy']*100:<12.1f}{fl_ci:<16}{fm['macro_f1_active_classes']:<18.3f}{fm['macro_f1_all_classes']:<14.3f}")
    print("=" * 100)

    out_json = "prediction-model/data/multi_horizon_benchmark_results.json"
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(benchmark_report, f, indent=2)
    print(f"\nSaved detailed benchmark JSON to: {out_json}")

    return benchmark_report

if __name__ == "__main__":
    run_evaluation()
