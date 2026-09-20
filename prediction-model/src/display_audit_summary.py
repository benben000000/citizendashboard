import json

with open("prediction-model/data/detailed_audit_report_data.json", "r") as f:
    d = json.load(f)

print("OUTLIER REPORT:")
print(f"  Temp spikes > 45C: {d['outliers']['temperature_spikes_over_45c']}")
print(f"  Pressure dropouts < 940 hPa: {d['outliers']['pressure_dropouts_under_940hpa']}")
print(f"  Stations affected: {d['outliers']['stations_affected']}")

print("\n" + "="*80)
print(f"{'Horizon':<8} | {'Target':<12} | {'Model':<12} | {'Metric 1 (MAE/Acc)':<18} | {'Metric 2 (RMSE/F1)':<18} | {'Bias/Recall':<14}")
print("="*80)

for h_key in ["1h", "3h", "6h", "12h", "24h", "48h", "72h"]:
    h_data = d["horizons"][h_key]
    n = h_data["sample_count"]

    # Temp
    tm = h_data["temperature"]["model"]
    tp = h_data["temperature"]["persistence"]
    ti = h_data["temperature"]["improved"]
    print(f"{h_key:<8} | {'Temp (C)':<12} | {'PINN-LNN':<12} | MAE: {tm['mae']:<13.3f} | RMSE: {tm['rmse']:<12.3f} | Bias: {tm['bias']:+.3f}")
    print(f"{'':<8} | {'':<12} | {'Persistence':<12} | MAE: {tp['mae']:<13.3f} | RMSE: {tp['rmse']:<12.3f} | Bias: {tp['bias']:+.3f}")
    print(f"{'':<8} | {'':<12} | {'Improved':<12} | MAE: {ti['mae']:<13.3f} | RMSE: {ti['rmse']:<12.3f} | Bias: {ti['bias']:+.3f}")

    # Rain
    rm = h_data["rain_state"]["model"]
    rp = h_data["rain_state"]["persistence"]
    ri = h_data["rain_state"]["improved"]
    print(f"{'':<8} | {'Rain State':<12} | {'PINN-LNN':<12} | Acc: {rm['accuracy']:<13.3f} | F1: {rm['f1']:<14.3f} | Rec: {rm['recall']:.3f} (FAR: {rm['false_alarm_rate']:.3f})")
    print(f"{'':<8} | {'':<12} | {'Persistence':<12} | Acc: {rp['accuracy']:<13.3f} | F1: {rp['f1']:<14.3f} | Rec: {rp['recall']:.3f} (FAR: {rp['false_alarm_rate']:.3f})")
    print(f"{'':<8} | {'':<12} | {'Improved':<12} | Acc: {ri['accuracy']:<13.3f} | F1: {ri['f1']:<14.3f} | Rec: {ri['recall']:.3f} (FAR: {ri['false_alarm_rate']:.3f})")

    # Water Level
    if "water_level" in h_data and "model" in h_data["water_level"]:
        wm = h_data["water_level"]["model"]
        wp = h_data["water_level"]["persistence"]
        wi = h_data["water_level"]["improved"]
        print(f"{'':<8} | {'Water Lvl (m)':<12} | {'PINN-LNN':<12} | MAE: {wm['mae']:<13.4f} | RMSE: {wm['rmse']:<12.4f} | Bias: {wm['bias']:+.4f}")
        print(f"{'':<8} | {'':<12} | {'Persistence':<12} | MAE: {wp['mae']:<13.4f} | RMSE: {wp['rmse']:<12.4f} | Bias: {wp['bias']:+.4f}")
        print(f"{'':<8} | {'':<12} | {'Improved':<12} | MAE: {wi['mae']:<13.4f} | RMSE: {wi['rmse']:<12.4f} | Bias: {wi['bias']:+.4f}")

    print("-" * 80)

print("\n" + "="*80)
print("RAIN STATE CONFUSION MATRICES (TP, FP, TN, FN)")
print("="*80)
for h_key in ["1h", "3h", "6h", "12h", "24h", "48h", "72h"]:
    m = d["horizons"][h_key]["rain_state"]["model"]
    p = d["horizons"][h_key]["rain_state"]["persistence"]
    i = d["horizons"][h_key]["rain_state"]["improved"]
    print(f"Horizon {h_key}:")
    print(f"  PINN-LNN:    TP={m['tp']:4d} | FP={m['fp']:4d} | TN={m['tn']:4d} | FN={m['fn']:4d} | Recall={m['recall']:.3f} | FAR={m['false_alarm_rate']:.3f}")
    print(f"  Persistence: TP={p['tp']:4d} | FP={p['fp']:4d} | TN={p['tn']:4d} | FN={p['fn']:4d} | Recall={p['recall']:.3f} | FAR={p['false_alarm_rate']:.3f}")
    print(f"  Improved:    TP={i['tp']:4d} | FP={i['fp']:4d} | TN={i['tn']:4d} | FN={i['fn']:4d} | Recall={i['recall']:.3f} | FAR={i['false_alarm_rate']:.3f}")

