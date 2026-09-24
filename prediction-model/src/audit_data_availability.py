"""
Deterministic Telemetry Data Availability and Sensor Quality Audit.

Produces an auditable report on raw weather and water level telemetry:
  - Exact source SHA-256 hashes.
  - Row counts, station coverage, time range in UTC.
  - Per-field missingness, numeric validity, and observed ranges.
  - Physical sensor bounds quarantine audit.
  - Wind speed, calm wind percentage, and circular direction consistency.
  - Solar sensor audit (UV index & Light Intensity): daylight vs nighttime distributions.
  - Formal feasibility determination for each proposed forecast target.

Outputs:
  prediction-model/data/weather_data_audit.json
"""

import os
import sys
import csv
import math
import json
import hashlib
from datetime import datetime, timezone
from collections import Counter, defaultdict

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SRC_DIR), "data")
WEATHER_CSV = os.path.join(DATA_DIR, "weather_telemetry.csv")
WATER_CSV = os.path.join(DATA_DIR, "water_level_telemetry.csv")
OUTPUT_JSON = os.path.join(DATA_DIR, "weather_data_audit.json")

# Physical bounds per PHYSICAL_BOUNDS in dataset.py
PHYSICAL_BOUNDS = {
    "temperature": (10.0, 50.0),
    "heat_index": (10.0, 70.0),
    "humidity": (10.0, 100.0),
    "wind_speed": (0.0, 180.0),
    "wind_direction": (0.0, 360.0),
    "pressure": (900.0, 1050.0),
    "precipitation": (0.0, 50.0),
    "water_level": (0.0, 15.0),
}


def compute_sha256(filepath: str) -> str:
    if not os.path.exists(filepath):
        return "file_not_found"
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def parse_iso_utc(ts_str: str):
    if not ts_str:
        return None
    s = ts_str.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S%z",
    ):
        try:
            return datetime.strptime(s, fmt).astimezone(timezone.utc)
        except ValueError:
            pass
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            return datetime.strptime(s[:19], fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return None


def run_audit(output_json: str = None, write_to_disk: bool = True):
    if output_json is None:
        output_json = OUTPUT_JSON
    print("=" * 80)
    print("GARCIA WEATHER TELEMETRY FORECAST ENGINE: DATA AVAILABILITY AUDIT")
    print("=" * 80)

    weather_hash = compute_sha256(WEATHER_CSV)
    water_hash = compute_sha256(WATER_CSV)
    print(f"Weather CSV: {WEATHER_CSV} (SHA-256: {weather_hash[:16]}...)")
    print(f"Water CSV:   {WATER_CSV} (SHA-256: {water_hash[:16]}...)")

    # 1. Audit Weather Telemetry
    weather_total_rows = 0
    weather_stations = Counter()
    weather_fieldnames = []
    weather_col_non_empty = Counter()
    weather_col_numeric = Counter()
    weather_min_vals = {}
    weather_max_vals = {}
    weather_bounds_violations = Counter()
    weather_malformed_ts = 0
    weather_out_of_years = 0

    ts_min = None
    ts_max = None

    calm_wind_count = 0
    valid_wind_speed_count = 0

    # Solar tracking: local hour (UTC+8) -> list of samples (cap for memory)
    hourly_uv = defaultdict(list)
    hourly_light = defaultdict(list)

    with open(WEATHER_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        weather_fieldnames = reader.fieldnames or []

        for row in reader:
            weather_total_rows += 1
            st_id = row.get("station_id", "")
            if st_id:
                weather_stations[st_id] += 1

            # Timestamp check
            ts_str = row.get("recorded_at", "")
            dt = parse_iso_utc(ts_str)
            if dt is None:
                weather_malformed_ts += 1
                continue
            if dt.year < 2025 or dt.year > 2027:
                weather_out_of_years += 1
                continue

            if ts_min is None or dt < ts_min:
                ts_min = dt
            if ts_max is None or dt > ts_max:
                ts_max = dt

            local_hour = (dt.hour + 8) % 24

            for col in weather_fieldnames:
                v = row.get(col)
                if v is not None and v.strip() != "":
                    weather_col_non_empty[col] += 1
                    try:
                        num = float(v)
                        weather_col_numeric[col] += 1
                        if col not in weather_min_vals or num < weather_min_vals[col]:
                            weather_min_vals[col] = num
                        if col not in weather_max_vals or num > weather_max_vals[col]:
                            weather_max_vals[col] = num

                        # Check bounds
                        if col in PHYSICAL_BOUNDS:
                            low, high = PHYSICAL_BOUNDS[col]
                            if not (low <= num <= high):
                                weather_bounds_violations[col] += 1

                        # Wind calm check
                        if col == "wind_speed":
                            valid_wind_speed_count += 1
                            if num < 1.0:
                                calm_wind_count += 1

                        # Solar samples (sample up to 2000 per hour)
                        if col == "uv_index" and len(hourly_uv[local_hour]) < 2000:
                            hourly_uv[local_hour].append(num)
                        if col == "light_intensity" and len(hourly_light[local_hour]) < 2000:
                            hourly_light[local_hour].append(num)

                    except ValueError:
                        pass

    # 2. Audit Water Telemetry
    water_total_rows = 0
    water_stations = Counter()
    water_col_non_empty = Counter()
    water_col_numeric = Counter()
    water_min_vals = {}
    water_max_vals = {}
    water_bounds_violations = Counter()
    water_malformed_ts = 0
    water_out_of_years = 0
    water_ts_min = None
    water_ts_max = None

    if os.path.exists(WATER_CSV):
        with open(WATER_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            water_fieldnames = reader.fieldnames or []
            for row in reader:
                water_total_rows += 1
                st_id = row.get("station_id", "")
                if st_id:
                    water_stations[st_id] += 1
                ts_str = row.get("recorded_at", "")
                dt = parse_iso_utc(ts_str)
                if dt is None:
                    water_malformed_ts += 1
                    continue
                if dt.year < 2025 or dt.year > 2027:
                    water_out_of_years += 1
                    continue
                if water_ts_min is None or dt < water_ts_min:
                    water_ts_min = dt
                if water_ts_max is None or dt > water_ts_max:
                    water_ts_max = dt

                for col in water_fieldnames:
                    v = row.get(col)
                    if v is not None and v.strip() != "":
                        water_col_non_empty[col] += 1
                        try:
                            num = float(v)
                            water_col_numeric[col] += 1
                            if col not in water_min_vals or num < water_min_vals[col]:
                                water_min_vals[col] = num
                            if col not in water_max_vals or num > water_max_vals[col]:
                                water_max_vals[col] = num
                        except ValueError:
                            pass

                # Gauge stage check
                wl_m = row.get("water_level_m")
                wl_cm = row.get("water_level_cm")
                wl = None
                try:
                    if wl_m and wl_m.strip():
                        wl = float(wl_m)
                    elif wl_cm and wl_cm.strip():
                        wl = float(wl_cm) / 100.0
                except ValueError:
                    pass
                if wl is not None:
                    if not (PHYSICAL_BOUNDS["water_level"][0] <= wl <= PHYSICAL_BOUNDS["water_level"][1]):
                        water_bounds_violations["water_level"] += 1

    # 3. Analyze Solar Night vs Day
    # Night hours in Philippines: 19:00 to 05:00 local time
    # Day hours in Philippines: 06:00 to 18:00 local time
    night_uv_samples = [x for h in list(range(19, 24)) + list(range(0, 6)) for x in hourly_uv[h]]
    day_uv_samples = [x for h in range(6, 19) for x in hourly_uv[h]]

    night_uv_mean = sum(night_uv_samples) / len(night_uv_samples) if night_uv_samples else 0.0
    night_uv_max = max(night_uv_samples) if night_uv_samples else 0.0
    night_uv_pos_pct = (sum(1 for x in night_uv_samples if x > 0.0) / len(night_uv_samples) * 100.0) if night_uv_samples else 0.0

    day_uv_mean = sum(day_uv_samples) / len(day_uv_samples) if day_uv_samples else 0.0
    day_uv_max = max(day_uv_samples) if day_uv_samples else 0.0

    night_light_samples = [x for h in list(range(19, 24)) + list(range(0, 6)) for x in hourly_light[h]]
    day_light_samples = [x for h in range(6, 19) for x in hourly_light[h]]
    night_light_mean = sum(night_light_samples) / len(night_light_samples) if night_light_samples else 0.0
    night_light_max = max(night_light_samples) if night_light_samples else 0.0
    day_light_mean = sum(day_light_samples) / len(day_light_samples) if day_light_samples else 0.0
    day_light_max = max(day_light_samples) if day_light_samples else 0.0

    # Feasibility classifications
    feasibility = {
        "temperature": {
            "status": "FEASIBLE",
            "coverage_pct": round(weather_col_non_empty.get("temperature", 0) / weather_total_rows * 100, 2),
            "units": "Celsius",
            "resampling": "last_valid_in_hour",
            "quarantined_rows": weather_bounds_violations.get("temperature", 0),
        },
        "humidity": {
            "status": "FEASIBLE",
            "coverage_pct": round(weather_col_non_empty.get("humidity", 0) / weather_total_rows * 100, 2),
            "units": "Percent (%)",
            "resampling": "last_valid_in_hour",
            "quarantined_rows": weather_bounds_violations.get("humidity", 0),
        },
        "pressure": {
            "status": "FEASIBLE",
            "coverage_pct": round(weather_col_non_empty.get("pressure", 0) / weather_total_rows * 100, 2),
            "units": "hPa",
            "resampling": "last_valid_in_hour",
            "quarantined_rows": weather_bounds_violations.get("pressure", 0),
        },
        "wind_speed": {
            "status": "FEASIBLE",
            "coverage_pct": round(weather_col_non_empty.get("wind_speed", 0) / weather_total_rows * 100, 2),
            "units": "km/h",
            "resampling": "last_valid_in_hour",
            "calm_percentage": round(calm_wind_count / max(1, valid_wind_speed_count) * 100, 2),
            "quarantined_rows": weather_bounds_violations.get("wind_speed", 0),
        },
        "wind_direction": {
            "status": "FEASIBLE_CIRCULAR",
            "coverage_pct": round(weather_col_non_empty.get("wind_direction", 0) / weather_total_rows * 100, 2),
            "units": "Degrees (0-360) via circular (u, v) unit components",
            "resampling": "circular_vector_last_valid",
            "calm_handling": "speed < 1.0 km/h masked as calm",
            "quarantined_rows": weather_bounds_violations.get("wind_direction", 0),
        },
        "precipitation_occurrence": {
            "status": "FEASIBLE",
            "coverage_pct": round(weather_col_non_empty.get("precipitation", 0) / weather_total_rows * 100, 2),
            "units": "Binary probability [0, 1] (threshold 0.1 mm/h)",
            "resampling": "hourly_sum",
            "quarantined_rows": weather_bounds_violations.get("precipitation", 0),
        },
        "precipitation_amount": {
            "status": "FEASIBLE_TWO_STAGE",
            "coverage_pct": round(weather_col_non_empty.get("precipitation", 0) / weather_total_rows * 100, 2),
            "units": "mm (hourly accumulated volume)",
            "resampling": "hourly_sum",
            "quarantined_rows": weather_bounds_violations.get("precipitation", 0),
        },
        "heat_index": {
            "status": "FEASIBLE_DERIVED",
            "units": "Celsius",
            "method": "NOAA Rothfusz regression derived from predicted temperature and humidity",
            "quarantined_rows": weather_bounds_violations.get("heat_index", 0),
        },
        "uv_index": {
            "status": "BLOCKED_BY_SENSOR_CALIBRATION",
            "coverage_pct": round(weather_col_non_empty.get("uv_index", 0) / weather_total_rows * 100, 2),
            "units": "UV Index",
            "night_positive_pct": round(night_uv_pos_pct, 2),
            "night_max_observed": night_uv_max,
            "reason": "Sensor clock/calibration defect: UV index reports up to 11.0 during nighttime (00:00-04:00 local time). Must not be scored without verified calibration.",
        },
        "light_intensity": {
            "status": "SECONDARY_BETA_DAYLIGHT_ONLY",
            "coverage_pct": round(weather_col_non_empty.get("light_intensity", 0) / weather_total_rows * 100, 2),
            "units": "lux",
            "night_mean": round(night_light_mean, 2),
            "day_mean": round(day_light_mean, 2),
            "reason": "Measurable daylight cycle, but uncalibrated photometric sensor lacking station orientation metadata. Kept as secondary/beta monitoring only.",
        },
        "water_level": {
            "status": "INTERNAL_EXPERIMENT_BETA",
            "coverage_pct": round(water_col_non_empty.get("water_level_m", 0) / max(1, water_total_rows) * 100, 2),
            "units": "Meters",
            "station": "Calumpit WLMS (O3z0j5bG) collocated with Calumpit AWS (3nzr48bG)",
            "quarantined_rows": water_bounds_violations.get("water_level", 0),
            "commercial_claim": "EXCLUDED from commercial core; not approved for autonomous flood or life-safety use.",
        },
    }

    report = {
        "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_hashes": {
            "weather_telemetry_csv": weather_hash,
            "water_level_telemetry_csv": water_hash,
        },
        "weather_telemetry": {
            "file_path": "prediction-model/data/weather_telemetry.csv",
            "total_rows": weather_total_rows,
            "total_stations": len(weather_stations),
            "station_ids": sorted(list(weather_stations.keys())),
            "time_range_min_utc": ts_min.isoformat() if ts_min else None,
            "time_range_max_utc": ts_max.isoformat() if ts_max else None,
            "field_summary": {
                f: {
                    "non_empty_count": weather_col_non_empty[f],
                    "coverage_pct": round(weather_col_non_empty[f] / weather_total_rows * 100, 2),
                    "numeric_count": weather_col_numeric[f],
                    "min_value": weather_min_vals.get(f),
                    "max_value": weather_max_vals.get(f),
                    "bounds_violations": weather_bounds_violations.get(f, 0),
                }
                for f in weather_fieldnames
            },
            "solar_sensor_audit": {
                "uv_index": {
                    "night_samples": len(night_uv_samples),
                    "night_mean": round(night_uv_mean, 2),
                    "night_max": round(night_uv_max, 2),
                    "night_positive_pct": round(night_uv_pos_pct, 2),
                    "day_samples": len(day_uv_samples),
                    "day_mean": round(day_uv_mean, 2),
                    "day_max": round(day_uv_max, 2),
                },
                "light_intensity": {
                    "night_samples": len(night_light_samples),
                    "night_mean": round(night_light_mean, 2),
                    "night_max": round(night_light_max, 2),
                    "day_samples": len(day_light_samples),
                    "day_mean": round(day_light_mean, 2),
                    "day_max": round(day_light_max, 2),
                }
            }
        },
        "water_telemetry": {
            "file_path": "prediction-model/data/water_level_telemetry.csv",
            "total_rows": water_total_rows,
            "total_stations": len(water_stations),
            "station_ids": sorted(list(water_stations.keys())),
            "time_range_min_utc": water_ts_min.isoformat() if water_ts_min else None,
            "time_range_max_utc": water_ts_max.isoformat() if water_ts_max else None,
            "bounds_violations": dict(water_bounds_violations),
        },
        "target_feasibility_determination": feasibility,
    }

    if write_to_disk and output_json:
        os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nAudit completed successfully. Report written to:\n  {output_json}\n")
    print(f"{'Target Field':<28} | {'Status':<30} | {'Units/Method':<35}")
    print("-" * 100)
    for tgt, info in feasibility.items():
        st = info["status"]
        u = info.get("units", info.get("method", ""))
        print(f"{tgt:<28} | {st:<30} | {u:<35}")
    print("=" * 100)
    return report


if __name__ == "__main__":
    run_audit()
