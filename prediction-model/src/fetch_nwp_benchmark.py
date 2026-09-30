"""
Fetch real numerical weather prediction (NWP) data for benchmarking the local
forecast engine.

SOURCES (all free at the point of retrieval)
--------------------------------------------
1. NOAA GFS — US National Centers for Environmental Prediction. Public domain,
   no key, no terms of use. Archived on AWS Open Data:
       s3://noaa-gfs-bdp-pds/enkfgdas.YYYYMMDD/HH/atmos/gfs.tHHz.pgrb2.0p25.fFFF
   This is the licence-clean primary source for commercial use, and GFS is the
   global model that drives most regional operations.

2. Open-Meteo Historical Forecast API — serves ARCHIVED FORECASTS (not analyses)
   from ECMWF IFS, GFS, ICON, Meteo-France, JMA, UK Met Office and ECCC GEM:
       https://historical-forecast-api.open-meteo.com/v1/forecast
   The free tier is NON-COMMERCIAL. Use it for research benchmarking; for
   commercial deployment buy their tier or fall back to raw open data.

       ecmwf_ifs025         ECMWF IFS 0.25   (operational, world-leading)
       gfs_seamless         NOAA GFS
       icon_seamless        DWD ICON
       meteofrance_seamless Meteo-France
       jma_seamless         JMA (Japan Met Agency)
       ukmo_seamless        UK Met Office
       gem_seamless         ECCC GEM (Environment Canada)

3. PAGASA — there is NO free programmatic NWP feed. PAGASA runs a WRF model
   nested in GFS but publishes human-written bulletins, not gridded data.
   The GFS comparison is the closest legitimate proxy for the PAGASA
   forecasting chain, because PAGASA's own runs are GFS-driven.

CACHING
-------
Results are written to prediction-model/data/nwp_benchmark_cache.json and reused
on subsequent runs; re-fetching 16 stations x 7 models is rate-limited.
"""

import json
import os
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(HERE, "..", "data"))
COORDS_PATH = os.path.join(DATA_DIR, "station_coords.json")
CACHE_PATH = os.path.join(DATA_DIR, "nwp_benchmark_cache.json")

NWP_MODELS = {
    "ecmwf_ifs025": ("ECMWF IFS 0.25", "ECMWF"),
    "gfs_seamless": ("NOAA GFS", "NOAA/NCEP"),
    "icon_seamless": ("DWD ICON", "DWD Germany"),
    "meteofrance_seamless": ("Meteo-France", "Meteo-France"),
    "jma_seamless": ("JMA", "Japan Met Agency"),
    "ukmo_seamless": ("UK Met Office", "Met Office"),
    "gem_seamless": ("ECCC GEM", "Environment Canada"),
}

# Variables directly comparable with the station telemetry.
HOURLY_VARS = ["temperature_2m", "relative_humidity_2m", "surface_pressure",
               "wind_speed_10m", "precipitation"]

API = "https://historical-forecast-api.open-meteo.com/v1/forecast"


def load_coords():
    """
    Station coordinates, read straight through.

    This used to swap the two fields, because station_coords.json was written
    with longitude stored under the key "lat" (inherited from default-stations.ts,
    which holds a Mapbox [lon, lat] pair). That file has since been corrected, so
    the compensation is removed -- keeping it would have swapped a second time and
    pointed every fetch at the wrong grid cell.

    The ranges are asserted rather than trusted: a transposed axis is exactly the
    kind of error that produces plausible-looking numbers at the wrong place.
    """
    with open(COORDS_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out = {}
    for sid, v in raw.items():
        lat, lon = float(v["lat"]), float(v["lon"])
        if not (-90.0 <= lat <= 90.0):
            raise ValueError(
                f"{sid}: latitude {lat} is out of range, so lat/lon are "
                f"transposed in {os.path.basename(COORDS_PATH)}")
        out[sid] = {"name": v["name"], "lat": lat, "lon": lon}
    return out


def fetch_one(lat, lon, start, end, model, retries=3, timeout=90):
    params = {
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
        "hourly": ",".join(HOURLY_VARS),
        "start_date": start,
        "end_date": end,
        "models": model,
        "timezone": "UTC",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
    }
    url = API + "?" + urllib.parse.urlencode(params)
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "kloudtrack-nwp-benchmark/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"fetch failed for {model}: {last}")


def main():
    coords = load_coords()
    with open(os.path.join(DATA_DIR, "station_index.json"), "r", encoding="utf-8") as f:
        index = json.load(f)
    in_model = [s["station_id"] for s in index]

    start = os.environ.get("NWP_START", "2026-08-15")
    end = os.environ.get("NWP_END", "2026-08-27")

    cache = {}
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            cache = json.load(f)
    meta = cache.get("_meta", {})
    meta.update({
        "start_date": start, "end_date": end,
        "source": "Open-Meteo Historical Forecast API (archived forecasts, not analyses)",
        "licence_note": (
            "Open-Meteo free tier is NON-COMMERCIAL. NOAA GFS on AWS Open Data is "
            "public domain and is the licence-clean route for commercial use."
        ),
        "variables": HOURLY_VARS,
    })
    cache["_meta"] = meta

    # The cache key MUST include the requested window: the same station/model
    # fetched for a different date range is different data. Keying on
    # station|model alone silently returned the previous (narrower) window.
    tag = f"{start}_{end}"
    todo = [(sid, m) for sid in in_model for m in NWP_MODELS
            if f"{sid}|{m}|{tag}" not in cache]
    print(f"stations: {len(in_model)}   models: {len(NWP_MODELS)}   "
          f"to fetch: {len(todo)}   cached: {len([k for k in cache if not k.startswith('_')])}")
    print(f"window  : {start} .. {end} (UTC)")

    done = 0
    for sid, model in todo:
        c = coords.get(sid)
        if not c:
            print(f"  [SKIP] {sid}: no coordinates")
            continue
        try:
            payload = fetch_one(c["lat"], c["lon"], start, end, model)
            h = payload.get("hourly", {})
            cache[f"{sid}|{model}|{tag}"] = {
                "station_id": sid,
                "lat": c["lat"],
                "lon": c["lon"],
                "model": model,
                "window": tag,
                "time": h.get("time", []),
                "temperature_2m": h.get("temperature_2m", []),
                "relative_humidity_2m": h.get("relative_humidity_2m", []),
                "surface_pressure": h.get("surface_pressure", []),
                "wind_speed_10m": h.get("wind_speed_10m", []),
                "precipitation": h.get("precipitation", []),
            }
            done += 1
            print(f"  [{done:>3}/{len(todo)}] {sid:<12} {model:<22} n={len(h.get('time', []))}")
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL] {sid} {model}: {exc}")
        time.sleep(0.35)

    with open(CACHE_PATH, "w", encoding="utf-8", newline="\n") as f:
        json.dump(cache, f)
    total = len([k for k in cache if not k.startswith("_")])
    print(f"\ncached series: {total} -> {CACHE_PATH}")


if __name__ == "__main__":
    main()
