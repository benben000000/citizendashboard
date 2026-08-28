"""
KloudTrack Historical Dataset Ingestion Script (ALL Stations).
Downloads full historical telemetry for all 16+ weather and river monitoring stations
in single bulk batch requests without exhausting rate limits.
"""

import os
import json
import urllib.request
import urllib.parse
import csv
import time

API_BASE_URL = "https://api.kloudtechsea.com/api/v1"
API_TOKEN = os.environ.get("KLOUDTRACK_API_TOKEN", "kloud_live_d2c3dece36db0668228537f7846be15a3b0e9303aeeb704d")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# Load all stations from constants
STATIONS_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src", "lib", "constants", "stations.json"
)

def get_stations_to_fetch():
    with open(STATIONS_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    
    weather_stations = cfg.get("weather", {}).get("stationIdToFetch", [])
    water_stations = cfg.get("waterLevel", {}).get("stationIdToFetch", [])
    maintenance_stations = cfg.get("maintenance", {}).get("stationIdToFetch", [])

    all_weather = weather_stations + maintenance_stations
    return all_weather, water_stations


def fetch_api(endpoint: str, params: dict = None) -> dict:
    url = f"{API_BASE_URL}{endpoint}"
    if params:
        query_str = urllib.parse.urlencode(params)
        url = f"{url}?{query_str}"

    headers = {
        "Content-Type": "application/json",
        "x-kloudtrack-key": API_TOKEN,
    }

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=45) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP Error {response.status}: {response.reason}")
        return json.loads(response.read().decode("utf-8"))


def download_all_stations():
    os.makedirs(DATA_DIR, exist_ok=True)
    weather_stations, water_stations = get_stations_to_fetch()

    print("=" * 70)
    print(f"Starting Bulk Telemetry Ingestion for ALL {len(weather_stations)} Weather & {len(water_stations)} Water Stations")
    print(f"Output Directory: {DATA_DIR}")
    print("=" * 70)

    all_weather_records = []
    station_stats = {}

    # 1. Fetch telemetry for all weather stations
    for idx, st in enumerate(weather_stations, 1):
        station_id = st["stationId"]
        location = st.get("location", station_id)
        print(f"[{idx:02d}/{len(weather_stations):02d}] 📥 Fetching Weather: {location.upper():<22} ({station_id})...", end=" ", flush=True)

        try:
            res = fetch_api(f"/telemetry/station/{station_id}/history", {"take": 50000})
            records = res.get("data", {}).get("telemetry", [])
            st_info = res.get("data", {}).get("station", {})
            st_name = st_info.get("stationName", location)

            # Save individual raw station JSON
            raw_file = os.path.join(DATA_DIR, f"raw_weather_{station_id}.json")
            with open(raw_file, "w", encoding="utf-8") as f:
                json.dump(res, f)

            station_stats[station_id] = {
                "name": st_name,
                "location": location,
                "type": "WEATHERSTATION",
                "count": len(records),
            }

            for r in records:
                wind = r.get("wind") or {}
                wind_speed = wind.get("speed") if isinstance(wind, dict) else None
                wind_dir = wind.get("direction") if isinstance(wind, dict) else None

                all_weather_records.append({
                    "station_id": station_id,
                    "station_name": st_name,
                    "location": location,
                    "recorded_at": r.get("recordedAt"),
                    "temperature": r.get("temperature"),
                    "heat_index": r.get("heatIndex"),
                    "humidity": r.get("humidity"),
                    "pressure": r.get("pressure"),
                    "wind_speed": wind_speed,
                    "wind_direction": wind_dir,
                    "precipitation": r.get("precipitation"),
                    "uv_index": r.get("uvIndex"),
                    "light_intensity": r.get("lightIntensity"),
                })

            print(f"✅ {len(records):,} records.")
        except Exception as e:
            print(f"⚠️ Warning/Skipped: {e}")

        time.sleep(0.3)  # Gentle spacing between stations

    # 2. Fetch telemetry for all water level stations
    all_water_records = []
    for idx, st in enumerate(water_stations, 1):
        station_id = st["stationId"]
        location = st.get("location", station_id)
        print(f"[{idx:02d}/{len(water_stations):02d}] 📥 Fetching Water Level: {location.upper():<22} ({station_id})...", end=" ", flush=True)

        try:
            res = fetch_api(f"/water-level/station/{station_id}/history/distance")
            records = res.get("data", {}).get("waterLevel", [])
            st_info = res.get("data", {}).get("station", {})
            st_name = st_info.get("stationName", location)

            raw_file = os.path.join(DATA_DIR, f"raw_water_level_{station_id}.json")
            with open(raw_file, "w", encoding="utf-8") as f:
                json.dump(res, f)

            station_stats[station_id] = {
                "name": st_name,
                "location": location,
                "type": "RIVERLEVEL",
                "count": len(records),
            }

            for r in records:
                val_cm = r.get("value")
                val_m = round(val_cm / 100.0, 3) if val_cm is not None else None
                all_water_records.append({
                    "station_id": station_id,
                    "station_name": st_name,
                    "location": location,
                    "recorded_at": r.get("recordedAt"),
                    "water_level_cm": val_cm,
                    "water_level_m": val_m,
                })

            print(f"✅ {len(records):,} records.")
        except Exception as e:
            print(f"⚠️ Warning/Skipped: {e}")

    # 3. Export Comprehensive Clean CSVs
    weather_csv = os.path.join(DATA_DIR, "weather_telemetry.csv")
    if all_weather_records:
        fieldnames = list(all_weather_records[0].keys())
        with open(weather_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_weather_records)

    water_csv = os.path.join(DATA_DIR, "water_level_telemetry.csv")
    if all_water_records:
        fieldnames = list(all_water_records[0].keys())
        with open(water_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_water_records)

    # Save metadata summary
    summary_path = os.path.join(DATA_DIR, "dataset_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_weather_records": len(all_weather_records),
            "total_water_records": len(all_water_records),
            "total_records": len(all_weather_records) + len(all_water_records),
            "stations": station_stats,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        }, f, indent=2)

    print("=" * 70)
    print(f"🎉 Complete Ingestion Finished!")
    print(f"📊 Total Weather Telemetry Rows: {len(all_weather_records):,}")
    print(f"🌊 Total Water Level Rows:      {len(all_water_records):,}")
    print(f"📁 Grand Total Telemetry Rows:   {len(all_weather_records) + len(all_water_records):,}")
    print("=" * 70)


if __name__ == "__main__":
    download_all_stations()
