
# benchmark_noaa_gfs.py -- licence-clean NWP benchmark

"""
Replace Open-Meteo benchmark with NOAA GFS from AWS Open Data (public domain).
This provides a licence-clean NWP comparison for commercial use.

Usage:
  python benchmark_noaa_gfs.py --weather-csv prediction-model/data/weather_telemetry.csv --out prediction-model/data/nwp_benchmark_noaa_gfs.json
"""

import json, os, sys, time, boto3, numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset import get_telemetry_pipeline
from inference import LNNServerlessPredictor
from scoring import mae, brier

# NOAA GFS on AWS Open Data
# s3://noaa-gfs-bdp-pds/gfs.YYYYMMDD/HH/atmos/gfs.tHHz.pgrb2.0p25.fFFF
# Public domain, no authentication required for public bucket

S3_BUCKET = "noaa-gfs-bdp-pds"
S3_PREFIX = "gfs."

GFS_VARS = {
    "temperature": "tmp_2m",
    "humidity": "rh_2m", 
    "pressure": "pressfc",
    "wind_u": "ugrd_10m",
    "wind_v": "vgrd_10m",
    "precipitation": "apcp_sfc",
}

# Station coordinates (need to be in station registry)
STATION_COORDS = {
    "KT-8CEE47DC5194": {"lat": 14.5995, "lon": 120.9842},
    # ... add all stations
}

def fetch_gfs_for_station(s3, station_id, init_time, horizons):
    """Fetch GFS forecast for a station at init_time for given horizons."""
    lat, lon = STATION_COORDS.get(station_id, (None, None))
    if lat is None: return None
    
    # Find nearest grid point
    # GFS 0.25 degree grid
    grid_lat = round(lat * 4) / 4
    grid_lon = round(lon * 4) / 4
    
    results = {}
    for h in horizons:
        fhr = f"{h:03d}"
        date_str = init_time.strftime("%Y%m%d")
        hour_str = f"{init_time.hour:02d}"
        key = f"gfs.{date_str}/{hour_str}/atmos/gfs.t{hour_str}z.pgrb2.0p25.f{fhr}"
        
        try:
            obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
            # Parse GRIB2 - use eccodes or cfgrib
            # For now, return placeholder
            pass
        except Exception as e:
            print(f"Failed to fetch {key}: {e}")
    
    return results


def run_benchmark(weather_csv, out_path, horizons=[1,3,6,12,24]):
    """Run benchmark against NOAA GFS."""
    pipe = get_telemetry_pipeline()
    
    # Get validation windows
    windows = build_forecast_windows(pipe, split="val", horizon=1, seq_len=24)
    # ... for each horizon, get GFS forecast, compute MAE/Brier
    
    # For now, create the structure
    results = {
        "benchmark": "NOAA GFS (AWS Open Data)",
        "licence": "public domain",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "horizons": {},
    }
    
    # Save
    Path(out_path).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Written to {out_path}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--weather-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--horizons", default="1,3,6,12,24")
    args = ap.parse_args()
    run_benchmark(args.weather_csv, args.out, [int(h) for h in args.horizons.split(",")])
