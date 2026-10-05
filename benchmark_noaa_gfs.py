# benchmark_noaa_gfs.py -- licence-clean NWP benchmark

"""
Replace Open-Meteo benchmark with NOAA GFS from AWS Open Data (public domain).
This provides a licence-clean NWP comparison for commercial use.

Usage:
  python benchmark_noaa_gfs.py --weather-csv prediction-model/data/weather_telemetry.csv --out prediction-model/data/nwp_benchmark_noaa_gfs.json
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "prediction-model" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dataset import TelemetryDataPipeline, build_forecast_windows, parse_utc_timestamp, DEFAULT_SEQ_LEN

# Check if AWS S3 & eccodes stack is available
try:
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    import eccodes
    HAS_GRIB_STACK = True
except ImportError:
    HAS_GRIB_STACK = False

S3_BUCKET = "noaa-gfs-bdp-pds"

# Station coordinates dictionary (Public Telemetry IDs + KT Expansion Nodes)
STATION_COORDS = {
    "95pM7BAV": {"name": "Dona Maria AWS", "lat": 14.6852, "lon": 120.5284},
    "lMAZe9b3": {"name": "Abucay AWS", "lat": 14.7358, "lon": 120.5372},
    "2Dpo5DAK": {"name": "1Bataan Command Center", "lat": 14.6784, "lon": 120.5412},
    "QgbGldAY": {"name": "Pag-asa Bagac AWS", "lat": 14.6012, "lon": 120.4012},
    "nDbyYbR1": {"name": "Sabang Morong AWS", "lat": 14.6812, "lon": 120.2741},
    "rqAkmpKG": {"name": "Barretto AWS", "lat": 14.8542, "lon": 120.2641},
    "Bkpj1zRO": {"name": "Old Cabalan AWS", "lat": 14.8621, "lon": 120.3102},
    "wkAWLzlm": {"name": "Lazatin AWS", "lat": 15.0341, "lon": 120.6812},
    "3nzr8bGo": {"name": "Alasas AWS", "lat": 15.0298, "lon": 120.6894},
    "3nzr48bG": {"name": "Calumpit AWS", "lat": 14.9201, "lon": 120.7657},
    "Rjz2dbXW": {"name": "Popolon AWS", "lat": 15.5368, "lon": 121.0577},
    "4VAl2p9k": {"name": "Sapang Buho AWS", "lat": 15.5521, "lon": 121.0843},
    "nDby4YpR": {"name": "General Natividad AWS", "lat": 15.6023, "lon": 121.0541},
    "03pqkGAj": {"name": "Bongabon Water District AWS", "lat": 15.6312, "lon": 121.1458},
    "1Zb102pg": {"name": "San Jose City AWS", "lat": 15.7912, "lon": 120.9984},
    "VEpdDpBK": {"name": "San Luis AWS - Aurora", "lat": 15.7012, "lon": 121.5201},
    # Expansion / Hardware IDs
    "KT-8CEE47DC5194": {"name": "1Bataan Command Center", "lat": 14.6812, "lon": 120.5414},
    "KT-6CBD47DC5194": {"name": "Old Cabcaben Pier - Bataan", "lat": 14.4532, "lon": 120.5978},
    "KT-CC380371FE68": {"name": "Dinalupihan AWS - Bataan", "lat": 14.8778, "lon": 120.4636},
    "KT-A86039DC5194": {"name": "Pag Asa Orani AWS - Bataan", "lat": 14.8000, "lon": 120.5333},
    "KT-D032325C7BCC": {"name": "Poblacion Mariveles AWS - Bataan", "lat": 14.4333, "lon": 120.4833},
    "KT-94AD8332A7B0": {"name": "Wawa Limay AWS - Bataan", "lat": 14.5667, "lon": 120.5833},
    "KT-A80A1B29E748": {"name": "Avida Asten AWS - Makati", "lat": 14.5583, "lon": 121.0111},
    "O3z0j5bG": {"name": "Calumpit WLMS", "lat": 14.9201, "lon": 120.7657},
}

DEFAULT_STATION_COORDS = {"lat": 15.0298, "lon": 120.6894}  # Central Luzon Regional fallback

def get_station_coords(station_id: str):
    if station_id in STATION_COORDS:
        info = STATION_COORDS[station_id]
        return info["lat"], info["lon"]
    clean_id = station_id.replace("KT-", "").replace("KT", "").upper()
    for k, v in STATION_COORDS.items():
        if clean_id in k.upper():
            return v["lat"], v["lon"]
    return DEFAULT_STATION_COORDS["lat"], DEFAULT_STATION_COORDS["lon"]


def fetch_gfs_step_from_s3(s3_client, date_str: str, init_hour: int, fhour: int):
    """
    Download GFS 0.25deg GRIB2 variables via byte-range and extract values for all stations.
    Returns (key, dict_of_station_forecasts).
    """
    key = f"gfs.{date_str}/{init_hour:02d}/atmos/gfs.t{init_hour:02d}z.pgrb2.0p25.f{fhour:03d}"
    idx_key = key + ".idx"
    try:
        idx_text = s3_client.get_object(Bucket=S3_BUCKET, Key=idx_key)["Body"].read().decode("utf-8")
        lines = idx_text.strip().splitlines()
        offsets = [int(l.split(":")[1]) for l in lines]
        
        tags = {
            "TMP": "TMP:2 m above ground",
            "RH": "RH:2 m above ground",
            "PRES": "PRES:surface",
            "UGRD": "UGRD:10 m above ground",
            "VGRD": "VGRD:10 m above ground",
            "APCP": "APCP:surface"
        }
        var_idx = {}
        for k, tag in tags.items():
            matches = [i for i, l in enumerate(lines) if tag in l]
            if matches:
                var_idx[k] = matches[0]

        if len(var_idx) < 4:
            print(f"Warning: Missing required variables in {idx_key}")
            return (date_str, init_hour, fhour), None

        min_i = min(var_idx.values())
        max_i = max(var_idx.values())
        r_start = offsets[min_i]
        r_end = offsets[max_i + 1] - 1 if max_i + 1 < len(offsets) else ""

        chunk = s3_client.get_object(Bucket=S3_BUCKET, Key=key, Range=f"bytes={r_start}-{r_end}")["Body"].read()

        gids = {}
        for k, i in var_idx.items():
            s = offsets[i] - r_start
            e = offsets[i + 1] - r_start if i + 1 < len(offsets) else len(chunk)
            gids[k] = eccodes.codes_new_from_message(chunk[s:e])

        station_results = {}
        for st_id in STATION_COORDS.keys():
            lat, lon = get_station_coords(st_id)
            v = {}
            for k, gid in gids.items():
                res = eccodes.codes_grib_find_nearest(gid, lat, lon)
                v[k] = res[0]["value"] if res else 0.0

            temp_c = v.get("TMP", 273.15 + 25.0) - 273.15
            rh_pct = v.get("RH", 80.0)
            pres_hpa = v.get("PRES", 101325.0) / 100.0
            wind_ms = math.hypot(v.get("UGRD", 0.0), v.get("VGRD", 0.0))
            apcp_mm = v.get("APCP", 0.0)
            rain_prob = 0.05 if apcp_mm <= 0.01 else (0.95 if apcp_mm >= 2.0 else 0.05 + 0.90 * (apcp_mm / 2.0))

            station_results[st_id] = {
                "temperature_c": float(temp_c),
                "humidity_pct": float(rh_pct),
                "pressure_hpa": float(pres_hpa),
                "wind_speed_ms": float(wind_ms),
                "precipitation_mm": float(apcp_mm),
                "rain_prob": float(rain_prob)
            }

        for gid in gids.values():
            eccodes.codes_release(gid)

        return (date_str, init_hour, fhour), station_results

    except Exception as exc:
        print(f"  [fetch error] {key}: {exc}")
        return (date_str, init_hour, fhour), None


def load_gfs_cache(cache_path: str) -> dict:
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_gfs_cache(cache: dict, cache_path: str):
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def run_benchmark(
    weather_csv: str,
    out_path: str,
    horizons: list = [1, 3, 6, 12, 24],
    hour_step: int = 6,
    max_workers: int = 12,
    cache_path: str = None
):
    """
    Run benchmark against NOAA GFS (AWS Open Data, public domain).
    Downloads GFS 0.25deg GRIB2 byte ranges, extracts physical surface variables,
    and computes MAE/Brier scores on the validation set.
    """
    DATA = ROOT / "prediction-model" / "data"
    csv = weather_csv or str(DATA / "weather_telemetry_current.csv")
    if not Path(csv).exists():
        csv = str(DATA / "weather_telemetry.csv")

    if cache_path is None:
        cache_path = str(DATA / "nwp_gfs_cache.json")

    print(f"=== NOAA GFS Benchmark (AWS Open Data / Public Domain) ===")
    print(f"  Telemetry CSV : {csv}")
    print(f"  Output path   : {out_path}")
    print(f"  Cache path    : {cache_path}")
    print(f"  Horizons      : {horizons}")
    print(f"  Hour step     : {hour_step}h (synoptic cycles)")
    print(f"  Workers       : {max_workers}")
    print(f"  GRIB2 Stack   : {'eccodes present' if HAS_GRIB_STACK else 'MISSING'}")

    cache = load_gfs_cache(cache_path)
    print(f"  Existing cached GFS forecast entries: {len(cache)}")

    # Initialize telemetry pipeline ONCE for efficiency
    t_start = time.time()
    print("\n[1/3] Loading telemetry pipeline...")
    pipe = TelemetryDataPipeline(weather_csv=csv)
    print(f"Pipeline initialized ({time.time() - t_start:.2f}s).")

    # Step 1: Collect all validation windows and needed GFS steps across horizons
    print("\n[2/3] Building validation windows and identifying GFS forecast steps...")
    horizon_windows = {}
    needed_gfs_steps = set()

    for h in horizons:
        try:
            res = build_forecast_windows(
                pipeline=pipe,
                split="val",
                horizon=h,
                seq_len=DEFAULT_SEQ_LEN,
                return_metadata=True
            )
            meta = res[6] if res and len(res) > 6 else []
            n_rows = len(res[0]) if res is not None else 0
        except Exception as exc:
            meta = []
            n_rows = 0
            print(f"  h{h}h error building windows: {exc}")

        # Filter by hour_step if specified (e.g. synoptic 00Z, 06Z, 12Z, 18Z cycles)
        if hour_step > 1:
            sampled_meta = [
                m for m in meta
                if parse_utc_timestamp(m["origin_timestamp"]).hour % hour_step == 0
            ]
        else:
            sampled_meta = meta

        horizon_windows[h] = {
            "n_val_rows": n_rows,
            "meta": sampled_meta
        }

        for m in sampled_meta:
            t0 = parse_utc_timestamp(m["origin_timestamp"])
            tt = parse_utc_timestamp(m["target_timestamp"])
            init_h = (t0.hour // 6) * 6
            t_init = t0.replace(hour=init_h, minute=0, second=0, microsecond=0)
            fhour = int(round((tt - t_init).total_seconds() / 3600))
            d_str = t_init.strftime("%Y%m%d")
            needed_gfs_steps.add((d_str, init_h, fhour))

        print(f"  +{h}h: {n_rows} total val rows, {len(sampled_meta)} evaluated rows")

    print(f"Total unique GFS forecast files required across all horizons: {len(needed_gfs_steps)}")

    # Check which steps are missing from cache
    missing_steps = []
    for d_str, init_h, fhour in needed_gfs_steps:
        cache_key = f"{d_str}_{init_h:02d}z_f{fhour:03d}"
        if cache_key not in cache:
            missing_steps.append((d_str, init_h, fhour))

    print(f"  Cached steps : {len(needed_gfs_steps) - len(missing_steps)}")
    print(f"  To download  : {len(missing_steps)}")

    # Step 2: Download missing steps from S3 via multi-threaded byte-range requests
    if missing_steps and HAS_GRIB_STACK:
        print(f"\n[3/3] Downloading {len(missing_steps)} GFS forecast files from AWS S3 (noaa-gfs-bdp-pds)...")
        s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED, max_pool_connections=max_workers * 2))
        t_dl_start = time.time()
        downloaded = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_step = {
                executor.submit(fetch_gfs_step_from_s3, s3, d, h, f): (d, h, f)
                for d, h, f in missing_steps
            }
            for fut in as_completed(future_to_step):
                (d, h, f), station_data = fut.result()
                if station_data:
                    cache_key = f"{d}_{h:02d}z_f{f:03d}"
                    cache[cache_key] = station_data
                    downloaded += 1
                    if downloaded % 10 == 0 or downloaded == len(missing_steps):
                        print(f"    Progress: {downloaded}/{len(missing_steps)} files downloaded ({time.time() - t_dl_start:.1f}s)...")

        save_gfs_cache(cache, cache_path)
        print(f"Download complete: {downloaded}/{len(missing_steps)} fetched in {time.time() - t_dl_start:.2f}s.")
    elif not HAS_GRIB_STACK:
        print("Warning: eccodes/boto3 not available. Using cached forecasts if present.")

    # Step 3: Compute MAE and Brier scores for each horizon
    horizon_results = {}
    print("\n=== Scoring NOAA GFS Against Ground Truth Telemetry ===")

    for h in horizons:
        info = horizon_windows[h]
        meta = info["meta"]
        n_rows = info["n_val_rows"]

        err_temp = []
        err_rh = []
        err_pres = []
        err_wind = []
        err_rain = []

        for m in meta:
            t0 = parse_utc_timestamp(m["origin_timestamp"])
            tt = parse_utc_timestamp(m["target_timestamp"])
            init_h = (t0.hour // 6) * 6
            t_init = t0.replace(hour=init_h, minute=0, second=0, microsecond=0)
            fhour = int(round((tt - t_init).total_seconds() / 3600))
            d_str = t_init.strftime("%Y%m%d")
            cache_key = f"{d_str}_{init_h:02d}z_f{fhour:03d}"
            st_id = m["station_id"]

            if cache_key in cache and st_id in cache[cache_key]:
                p = cache[cache_key][st_id]
                t_target = float(m["target_temperature"])
                rh_target = float(m["target_humidity"])
                p_target = float(m["target_pressure"])
                w_target = float(m["target_wind_speed"])
                rain_target = 1.0 if (float(m.get("target_precipitation", 0.0) or 0.0) > 0.1) else 0.0

                err_temp.append(abs(p["temperature_c"] - t_target))
                err_rh.append(abs(p["humidity_pct"] - rh_target))
                err_pres.append(abs(p["pressure_hpa"] - p_target))
                err_wind.append(abs(p["wind_speed_ms"] - w_target))
                err_rain.append((p["rain_prob"] - rain_target) ** 2)

        n_scored = len(err_temp)

        if n_scored > 0:
            mae_temp = round(float(np.mean(err_temp)), 4)
            mae_rh = round(float(np.mean(err_rh)), 4)
            mae_pres = round(float(np.mean(err_pres)), 4)
            mae_wind = round(float(np.mean(err_wind)), 4)
            brier_rain = round(float(np.mean(err_rain)), 4)
            note = f"Evaluated against NOAA GFS 0.25deg GRIB2 via AWS S3 Open Data (public domain, {n_scored} scored rows)"
        else:
            mae_temp = None
            mae_rh = None
            mae_pres = None
            mae_wind = None
            brier_rain = None
            note = "No GFS data available for scoring"

        horizon_results[f"+{h}h"] = {
            "n_val_rows": n_rows,
            "n_scored_rows": n_scored,
            "gfs_mae_temperature_c": mae_temp,
            "gfs_mae_humidity_pct": mae_rh,
            "gfs_mae_pressure_hpa": mae_pres,
            "gfs_mae_wind_speed_ms": mae_wind,
            "gfs_brier_rain": brier_rain,
            "note": note,
        }

        print(f"  +{h}h (N={n_scored}/{n_rows} rows):")
        print(f"      Temperature MAE : {mae_temp} degC")
        print(f"      Humidity MAE    : {mae_rh} %")
        print(f"      Pressure MAE    : {mae_pres} hPa")
        print(f"      Wind Speed MAE  : {mae_wind} m/s")
        print(f"      Rain Brier      : {brier_rain}")

    results = {
        "benchmark": "NOAA GFS (AWS Open Data)",
        "licence": "public domain \u2014 https://registry.opendata.aws/noaa-gfs-bdp-pds/",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "horizons": horizon_results,
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWritten benchmark results -> {out_path}")


if __name__ == "__main__":
    DATA = ROOT / "prediction-model" / "data"
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weather-csv", default=str(DATA / "weather_telemetry_current.csv"))
    ap.add_argument("--out", default=str(DATA / "nwp_benchmark_noaa_gfs.json"))
    ap.add_argument("--horizons", default="1,3,6,12,24")
    ap.add_argument("--hour-step", type=int, default=6, help="Step between sampled validation origin hours (default 6h for synoptic cycles)")
    ap.add_argument("--workers", type=int, default=12, help="Number of concurrent S3 download threads")
    ap.add_argument("--cache", default=str(DATA / "nwp_gfs_cache.json"), help="Path to local GFS forecast cache")
    args = ap.parse_args()

    run_benchmark(
        weather_csv=args.weather_csv,
        out_path=args.out,
        horizons=[int(h) for h in args.horizons.split(",")],
        hour_step=args.hour_step,
        max_workers=args.workers,
        cache_path=args.cache
    )
