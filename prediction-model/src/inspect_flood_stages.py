import csv
from collections import Counter

BENCHMARK_CSV = "public/exports/Kloudtrack_Corrected_Audited_Benchmark_2026-09-01_to_2026-09-20.csv"

flood_counts = Counter()
actual_stages = Counter()
pred_stages = Counter()
stage_matches = Counter()

with open(BENCHMARK_CSV, "r", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r["station_id"] != "O3z0j5bG": continue
        h = r["horizon_hours"]
        act = r["actual_flood_stage"]
        prd = r["pred_model_flood_stage"]
        actual_stages[act] += 1
        pred_stages[prd] += 1
        flood_counts[h] += 1
        if act == prd:
            stage_matches[h] += 1

print("Actual Flood Stage Distribution (Calumpit WLMS across all rows):")
for k, v in actual_stages.items():
    print(f"  {k}: {v}")

print("\nModel Flood Stage Accuracy by Horizon:")
for h in [1, 3, 6, 12, 24, 48, 72]:
    tot = flood_counts[str(h)]
    mat = stage_matches[str(h)]
    pct = (mat / tot * 100) if tot > 0 else 0
    print(f"  Horizon {h:2d}h: {mat}/{tot} matches ({pct:.2f}%)")
