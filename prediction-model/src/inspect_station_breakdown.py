import csv
import json
from collections import defaultdict
import numpy as np

BENCHMARK_CSV = "public/exports/Kloudtrack_Corrected_Audited_Benchmark_2026-09-01_to_2026-09-20.csv"

# Accumulate per station and horizon
stn_metrics = defaultdict(lambda: defaultdict(lambda: {"act_t": [], "mod_t": [], "per_t": [], "act_r": [], "mod_r": [], "per_r": []}))

with open(BENCHMARK_CSV, "r", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r["qc_status"] != "PASSED_QC": continue
        sid = r["station_id"]
        h = r["horizon_hours"]
        if r["actual_temperature_c"] and r["pred_model_temperature_c"]:
            act_t = float(r["actual_temperature_c"])
            mod_t = float(r["pred_model_temperature_c"])
            per_t = float(r["pred_persist_temperature_c"])
            stn_metrics[sid][h]["act_t"].append(act_t)
            stn_metrics[sid][h]["mod_t"].append(mod_t)
            stn_metrics[sid][h]["per_t"].append(per_t)

        act_r = 1 if r["actual_is_raining"] == "TRUE" else 0
        mod_r = 1 if r["pred_model_is_raining"] == "TRUE" else 0
        per_r = 1 if r["pred_persist_is_raining"] == "TRUE" else 0
        stn_metrics[sid][h]["act_r"].append(act_r)
        stn_metrics[sid][h]["mod_r"].append(mod_r)
        stn_metrics[sid][h]["per_r"].append(per_r)

print("Sample Per-Station Breakdown for 1h and 24h horizons:")
print(f"{'Station ID':<12} | {'Horizon':<7} | {'Sample N':<8} | {'Model Temp MAE':<15} | {'Persist Temp MAE':<17} | {'Rain Recall (Model/Persist)':<25}")
print("-" * 90)

for sid in sorted(stn_metrics.keys()):
    for h in ["1", "24"]:
        d = stn_metrics[sid][h]
        if not d["act_t"]: continue
        t_mae_mod = np.mean(np.abs(np.array(d["mod_t"]) - np.array(d["act_t"])))
        t_mae_per = np.mean(np.abs(np.array(d["per_t"]) - np.array(d["act_t"])))
        
        # Rain recall
        act_arr = np.array(d["act_r"])
        mod_arr = np.array(d["mod_r"])
        per_arr = np.array(d["per_r"])
        
        rain_events = sum(act_arr)
        rec_mod = (np.sum((act_arr == 1) & (mod_arr == 1)) / rain_events) if rain_events > 0 else 0
        rec_per = (np.sum((act_arr == 1) & (per_arr == 1)) / rain_events) if rain_events > 0 else 0
        
        print(f"{sid:<12} | {h+'h':<7} | {len(d['act_t']):<8} | {t_mae_mod:<15.3f} | {t_mae_per:<17.3f} | Mod: {rec_mod:.2f} / Per: {rec_per:.2f} ({rain_events} events)")
