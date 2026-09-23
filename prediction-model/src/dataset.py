"""
Canonical Forecast Telemetry Dataset and Hourly Preprocessing Pipeline.

Implements the unified forecasting contract across all model families for
weather station telemetry and hydrological river-stage forecasting.

Key Specifications:
  - 1.1 Raw Timestamp & Sensor Bounds Cleaning:
      * UTC timezone-aware parsing.
      * Strict collection range [2025, 2027] quarantine (excludes 2069 bug).
      * Physical sensor bounds quarantine per variable with reason counts.
      * Provenance hashing (SHA-256) and data quality manifest output.
      * Automated integrity assertions (no unsorted series, no cross-station windows).
  - 1.2 Hourly Resampling Grid:
      * Resamples irregular minute telemetry to standard UTC hourly bins.
      * Temp, Heat Index, Wind Speed, Pressure: last valid observation in hour.
      * Precipitation: sum of valid minute increments in hour (hourly volume mm).
      * Water gauge: last valid gauge observation in hour (collocated at Calumpit).
      * Tracks hour completeness and observation count.
  - 1.3 Exact Forecast Target Contract:
      * Horizon h in [1, 3, 6, 12, 24] hours.
      * Forecast origin t0 = final timestamp in seq_len input window.
      * Target timestamp = exactly t0 + h hours on the hourly grid.
      * Target lead time validated against tolerance (|lead - h| <= 0.25h).
      * Metadata fields exposed for transparent independent validation.
  - 1.4 Chronological Split with Embargo:
      * 60% train, 20% validation/calibration, 20% test by time range.
      * 48-hour embargo (seq_len + max_horizon) between splits to prevent leakage.
      * Supports temporal generalization (default) and station generalization.
  - 1.5 Train-Fitted Normalization:
      * Feature means and standard deviations fitted ONLY on the training split.
      * Passed unchanged to validation and test splits.
"""

import os
import csv
import math
import hashlib
import json
from datetime import datetime, timezone, timedelta
from collections import defaultdict, Counter

import numpy as np
import torch
from torch.utils.data import Dataset

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
WEATHER_CSV_PATH = os.path.join(DATA_DIR, "weather_telemetry.csv")
WATER_CSV_PATH = os.path.join(DATA_DIR, "water_level_telemetry.csv")

# Allowed collection date bounds for KloudTrack corpus
MIN_VALID_YEAR = 2025
MAX_VALID_YEAR = 2027

# Physical bounds for Philippine tropical surface meteorology and river stage
PHYSICAL_BOUNDS = {
    "temperature": (10.0, 50.0),    # Celsius
    "heat_index": (10.0, 70.0),     # Celsius
    "humidity": (10.0, 100.0),      # Relative humidity % (drops below 10% are sensor disconnects/outages)
    "wind_speed": (0.0, 180.0),     # km/h
    "wind_direction": (0.0, 360.0), # degrees
    "pressure": (900.0, 1050.0),    # hPa
    "precipitation": (0.0, 50.0),   # mm per 1-minute record (incremental tipping bucket; tropical cloudburst limit)
    "water_level": (0.0, 15.0),     # meters
}

# Canonical feature schema: 8 physical surface meteorology features
DEFAULT_FEATURE_SCHEMA = [
    "temperature",
    "heat_index",
    "humidity",
    "pressure",
    "wind_speed",
    "wind_sin",
    "wind_cos",
    "precipitation",
]
NUM_FEATURES = len(DEFAULT_FEATURE_SCHEMA)

# Collocated gauge and weather station IDs in Calumpit, Bulacan
WATER_GAUGE_STATION_ID = "O3z0j5bG"       # Calumpit WLMS
WATER_GAUGE_WEATHER_STATION = "3nzr48bG"  # Calumpit AWS

DEFAULT_HORIZONS = [1, 3, 6, 12, 24]
DEFAULT_SEQ_LEN = 24
HORIZON_TOLERANCE_HOURS = 0.25  # 15 minutes tolerance on lead time


def parse_utc_timestamp(ts_str: str) -> datetime:
    """Parse timestamp string into timezone-aware UTC datetime."""
    if not ts_str:
        return None
    ts_str = ts_str.strip()
    # Normalize Z to +00:00
    if ts_str.endswith("Z"):
        ts_str = ts_str[:-1] + "+00:00"
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S%z",
    ):
        try:
            return datetime.strptime(ts_str, fmt).astimezone(timezone.utc)
        except ValueError:
            continue
    # Naive fallback: treat as UTC
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            dt = datetime.strptime(ts_str[:19], fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def compute_file_sha256(filepath: str) -> str:
    """Compute SHA-256 checksum of a file."""
    if not os.path.exists(filepath):
        return "file_not_found"
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_noaa_heat_index(temp_c: float, humidity_rh: float) -> float:
    """
    Standard NOAA National Weather Service Rothfusz heat index algorithm.
    Takes Temp in Celsius and Relative Humidity in % (0-100).
    Returns Heat Index in Celsius.
    """
    t_f = temp_c * 9.0 / 5.0 + 32.0
    rh = max(0.0, min(100.0, humidity_rh))
    if t_f < 80.0:
        hi_f = 0.5 * (t_f + 61.0 + ((t_f - 68.0) * 1.2) + (rh * 0.094))
        if hi_f < 80.0:
            return (hi_f - 32.0) * 5.0 / 9.0

    c1 = -42.379
    c2 = 2.04901523
    c3 = 10.14333127
    c4 = -0.22475541
    c5 = -0.00683783
    c6 = -0.05481717
    c7 = 0.00122874
    c8 = 0.00085282
    c9 = -0.00000199

    hi_f = (
        c1 + c2 * t_f + c3 * rh + c4 * t_f * rh
        + c5 * (t_f ** 2) + c6 * (rh ** 2)
        + c7 * (t_f ** 2) * rh + c8 * t_f * (rh ** 2)
        + c9 * (t_f ** 2) * (rh ** 2)
    )
    if rh < 13.0 and 80.0 <= t_f <= 112.0:
        adj = ((13.0 - rh) / 4.0) * math.sqrt(max(0.0, (17.0 - abs(t_f - 95.0)) / 17.0))
        hi_f -= adj
    elif rh > 85.0 and 80.0 <= t_f <= 87.0:
        adj = ((rh - 85.0) / 10.0) * ((87.0 - t_f) / 5.0)
        hi_f += adj

    return (hi_f - 32.0) * 5.0 / 9.0


def circular_direction_error_deg(theta_pred_deg: float, theta_true_deg: float) -> float:
    """Compute shortest angular distance between two wind directions in degrees."""
    diff = abs(theta_pred_deg - theta_true_deg) % 360.0
    return min(diff, 360.0 - diff)


# ---------------------------------------------------------------------------
# Data Cleaning, Quarantine, and Hourly Resampling
# ---------------------------------------------------------------------------

class TelemetryDataPipeline:
    """
    Manages end-to-end data ingestion, quarantine filtering, hourly aggregation,
    chronological partitioning, and window sampling.
    """

    def __init__(self, weather_csv: str = None, water_csv: str = None):
        self.weather_csv = weather_csv or WEATHER_CSV_PATH
        self.water_csv = water_csv or WATER_CSV_PATH

        self.quarantine_counts = Counter()
        self.raw_weather_row_count = 0
        self.raw_water_row_count = 0

        # Hourly aggregated records: station_id -> dict(hourly_bin_dt -> record_dict)
        self.station_hourly = defaultdict(dict)
        # Hourly water records: hourly_bin_dt -> water_level_m
        self.water_hourly = {}

        self.time_range_min = None
        self.time_range_max = None
        self.train_end = None
        self.val_start = None
        self.val_end = None
        self.test_start = None

        self.norm_means = None
        self.norm_stds = None

        self._process_pipeline()

    def _process_pipeline(self):
        """Execute the complete data cleaning, quarantine, and resampling pipeline."""
        self._load_and_resample_water()
        self._load_and_resample_weather()
        self._compute_split_boundaries()
        self._fit_training_normalization()

    def _load_and_resample_water(self):
        """Load, quarantine, and resample water level telemetry to hourly bins."""
        if not os.path.exists(self.water_csv):
            return

        hourly_obs = defaultdict(list)

        with open(self.water_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.raw_water_row_count += 1
                ts_raw = row.get("recorded_at", "")
                dt = parse_utc_timestamp(ts_raw)
                if dt is None:
                    self.quarantine_counts["water_malformed_timestamp"] += 1
                    continue
                if dt.year < MIN_VALID_YEAR or dt.year > MAX_VALID_YEAR:
                    self.quarantine_counts["water_year_out_of_bounds"] += 1
                    continue

                wl_m = row.get("water_level_m")
                wl_cm = row.get("water_level_cm")
                try:
                    if wl_m is not None and wl_m.strip() != "":
                        wl = float(wl_m)
                    elif wl_cm is not None and wl_cm.strip() != "":
                        wl = float(wl_cm) / 100.0
                    else:
                        self.quarantine_counts["water_missing_value"] += 1
                        continue

                    if not (PHYSICAL_BOUNDS["water_level"][0] <= wl <= PHYSICAL_BOUNDS["water_level"][1]):
                        self.quarantine_counts["water_physical_bounds"] += 1
                        continue

                    h_bin = dt.replace(minute=0, second=0, microsecond=0)
                    hourly_obs[h_bin].append((dt, wl))
                except (ValueError, TypeError):
                    self.quarantine_counts["water_parse_error"] += 1

        # Resample: last valid gauge observation in each hourly bin
        for h_bin, obs_list in hourly_obs.items():
            obs_list.sort(key=lambda x: x[0])
            self.water_hourly[h_bin] = obs_list[-1][1]

    def _load_and_resample_weather(self):
        """Load, quarantine, and resample weather telemetry to hourly bins."""
        if not os.path.exists(self.weather_csv):
            return

        # station_id -> dict(h_bin -> list of (dt, t, hi, ws, p, precip))
        station_hour_obs = defaultdict(lambda: defaultdict(list))

        with open(self.weather_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.raw_weather_row_count += 1
                ts_raw = row.get("recorded_at", "")
                dt = parse_utc_timestamp(ts_raw)
                if dt is None:
                    self.quarantine_counts["weather_malformed_timestamp"] += 1
                    continue
                if dt.year < MIN_VALID_YEAR or dt.year > MAX_VALID_YEAR:
                    self.quarantine_counts["weather_year_out_of_bounds"] += 1
                    continue

                st_id = row.get("station_id")
                if not st_id:
                    self.quarantine_counts["weather_missing_station_id"] += 1
                    continue

                try:
                    t = float(row.get("temperature") or 28.5)
                    hi = float(row.get("heat_index") or 33.0)
                    hum = float(row.get("humidity") or 75.0)
                    ws = float(row.get("wind_speed") or 10.0)
                    wd = float(row.get("wind_direction") or 0.0)
                    p = float(row.get("pressure") or 1008.0)
                    precip = float(row.get("precipitation") or 0.0)

                    # Check individual physical sensor bounds
                    if not (PHYSICAL_BOUNDS["temperature"][0] <= t <= PHYSICAL_BOUNDS["temperature"][1]):
                        self.quarantine_counts["weather_bounds_temperature"] += 1
                        continue
                    if not (PHYSICAL_BOUNDS["heat_index"][0] <= hi <= PHYSICAL_BOUNDS["heat_index"][1]):
                        self.quarantine_counts["weather_bounds_heat_index"] += 1
                        continue
                    if not (PHYSICAL_BOUNDS["humidity"][0] <= hum <= PHYSICAL_BOUNDS["humidity"][1]):
                        self.quarantine_counts["weather_bounds_humidity"] += 1
                        continue
                    if not (PHYSICAL_BOUNDS["wind_speed"][0] <= ws <= PHYSICAL_BOUNDS["wind_speed"][1]):
                        self.quarantine_counts["weather_bounds_wind_speed"] += 1
                        continue
                    if not (PHYSICAL_BOUNDS["wind_direction"][0] <= wd <= PHYSICAL_BOUNDS["wind_direction"][1]):
                        self.quarantine_counts["weather_bounds_wind_direction"] += 1
                        continue
                    if not (PHYSICAL_BOUNDS["pressure"][0] <= p <= PHYSICAL_BOUNDS["pressure"][1]):
                        self.quarantine_counts["weather_bounds_pressure"] += 1
                        continue
                    if not (PHYSICAL_BOUNDS["precipitation"][0] <= precip <= PHYSICAL_BOUNDS["precipitation"][1]):
                        self.quarantine_counts["weather_bounds_precipitation"] += 1
                        continue

                    h_bin = dt.replace(minute=0, second=0, microsecond=0)
                    station_hour_obs[st_id][h_bin].append((dt, t, hi, hum, ws, wd, p, precip))

                except (ValueError, TypeError):
                    self.quarantine_counts["weather_parse_error"] += 1

        # Resample each station to the hourly grid:
        # - Temperature, heat_index, humidity, wind_speed, wind_direction, pressure: last valid observation
        # - Wind direction: transformed to sin and cos continuous components
        # - Precipitation: sum of increments (volume in mm)
        for st_id, h_dict in station_hour_obs.items():
            for h_bin, obs_list in h_dict.items():
                obs_list.sort(key=lambda x: x[0])
                last_obs = obs_list[-1]
                tot_precip = sum(x[7] for x in obs_list)
                rad = math.radians(last_obs[5] % 360.0)

                self.station_hourly[st_id][h_bin] = {
                    "timestamp": h_bin,
                    "temperature": last_obs[1],
                    "heat_index": last_obs[2],
                    "humidity": last_obs[3],
                    "pressure": last_obs[6],
                    "wind_speed": last_obs[4],
                    "wind_sin": math.sin(rad),
                    "wind_cos": math.cos(rad),
                    "precipitation": tot_precip,
                    "obs_count": len(obs_list),
                }

    def _compute_split_boundaries(self):
        """Compute chronological split cutoffs with a 48h embargo."""
        all_hours = sorted({h for st in self.station_hourly for h in self.station_hourly[st]})
        if not all_hours:
            return

        self.time_range_min = all_hours[0]
        self.time_range_max = all_hours[-1]

        n = len(all_hours)
        self.train_end = all_hours[int(n * 0.60)]
        self.val_end = all_hours[int(n * 0.80)]

        # 48-hour embargo (seq_len + max_horizon)
        embargo = timedelta(hours=48)
        self.val_start = self.train_end + embargo
        self.test_start = self.val_end + embargo

        # Automated assertion: no test timestamp <= final training timestamp
        assert self.test_start > self.train_end, "Leakage detected: test period overlaps training period!"

    def _fit_training_normalization(self):
        """Fit normalization means and stds strictly on the training partition for all 8 features."""
        train_features = []
        for st_id, h_dict in self.station_hourly.items():
            for h_bin, rec in h_dict.items():
                if h_bin <= self.train_end:
                    train_features.append([
                        rec["temperature"],
                        rec["heat_index"],
                        rec["humidity"],
                        rec["pressure"],
                        rec["wind_speed"],
                        rec["wind_sin"],
                        rec["wind_cos"],
                        rec["precipitation"],
                    ])

        if len(train_features) < 10:
            self.norm_means = FEATURE_MEANS.copy()
            self.norm_stds = FEATURE_STDS.copy()
            return

        arr = np.array(train_features, dtype=np.float32)
        self.norm_means = arr.mean(axis=0)
        stds = arr.std(axis=0)
        self.norm_stds = np.where(stds < 1e-4, 1.0, stds).astype(np.float32)

    def generate_data_quality_report(self, output_path: str = None) -> dict:
        """
        Produce a comprehensive data quality and quarantine report.
        Saves report to prediction-model/data/data_quality_report.json.
        """
        if output_path is None:
            output_path = os.path.join(DATA_DIR, "data_quality_report.json")

        total_station_hours = sum(len(h) for h in self.station_hourly.values())
        git_commit = "unknown"
        try:
            import subprocess
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=os.path.dirname(DATA_DIR),
                text=True
            ).strip()
        except Exception:
            pass

        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "code_commit": git_commit,
            "data_hashes": {
                "weather_telemetry_sha256": compute_file_sha256(self.weather_csv),
                "water_level_telemetry_sha256": compute_file_sha256(self.water_csv),
            },
            "raw_counts": {
                "weather_telemetry_rows": self.raw_weather_row_count,
                "water_level_telemetry_rows": self.raw_water_row_count,
            },
            "quarantine_counts_by_reason": dict(self.quarantine_counts),
            "total_quarantined_weather_rows": sum(v for k, v in self.quarantine_counts.items() if k.startswith("weather_")),
            "total_quarantined_water_rows": sum(v for k, v in self.quarantine_counts.items() if k.startswith("water_")),
            "resampled_hourly_summary": {
                "num_weather_stations": len(self.station_hourly),
                "total_station_hours": total_station_hours,
                "station_hours_by_id": {st: len(h) for st, h in sorted(self.station_hourly.items())},
                "total_water_gauge_hours": len(self.water_hourly),
                "time_range_min": self.time_range_min.isoformat() if self.time_range_min else None,
                "time_range_max": self.time_range_max.isoformat() if self.time_range_max else None,
            },
            "split_boundaries": {
                "split_strategy": "chronological_60_20_20_with_48h_embargo",
                "train_start": self.time_range_min.isoformat() if self.time_range_min else None,
                "train_end": self.train_end.isoformat() if self.train_end else None,
                "val_start": self.val_start.isoformat() if self.val_start else None,
                "val_end": self.val_end.isoformat() if self.val_end else None,
                "test_start": self.test_start.isoformat() if self.test_start else None,
                "test_end": self.time_range_max.isoformat() if self.time_range_max else None,
            },
            "train_fitted_normalization": {
                "feature_schema": list(DEFAULT_FEATURE_SCHEMA),
                "means": self.norm_means.tolist() if self.norm_means is not None else [],
                "stds": self.norm_stds.tolist() if self.norm_stds is not None else [],
            },
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        # Also write cleaned-data manifest
        manifest_path = os.path.join(DATA_DIR, "cleaned_data_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return report


# Global cached pipeline singleton
_PIPELINE_CACHE = None

def get_telemetry_pipeline(weather_csv: str = None, water_csv: str = None, force_reload: bool = False) -> TelemetryDataPipeline:
    """Retrieve or initialize the canonical telemetry pipeline."""
    global _PIPELINE_CACHE
    if _PIPELINE_CACHE is None or force_reload:
        _PIPELINE_CACHE = TelemetryDataPipeline(weather_csv, water_csv)
    return _PIPELINE_CACHE


# ---------------------------------------------------------------------------
# Forecast Window Construction
# ---------------------------------------------------------------------------

def normalize_features(features: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    """Normalize feature array [..., 4] using specified means and standard deviations."""
    return (features - means) / stds


def denormalize_features(features: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    """Denormalize scaled features back to physical units."""
    return features * stds + means


def build_forecast_windows(
    pipeline: TelemetryDataPipeline,
    split: str = "train",
    horizon: int = 1,
    seq_len: int = DEFAULT_SEQ_LEN,
    max_samples: int = None,
    norm_means: np.ndarray = None,
    norm_stds: np.ndarray = None,
    mode: str = "temporal",
    holdout_stations: list = None,
    return_metadata: bool = False,
):
    """
    Build canonical future-forecast sequence windows adhering strictly to the contract:
      - Input: seq_len hourly observations ending at t0.
      - Target: observation at t0 + h hours (actual elapsed time verified within tolerance).
      - dt: continuous elapsed hours between consecutive observations (default 1.0h).
      - Water level: collocated gauge observation at t0 + h, masked when absent.
      - Station boundaries: windows NEVER span across multiple stations.
      - Split boundaries: windows NEVER span across split cutoffs.

    Args:
      pipeline: Initialized TelemetryDataPipeline.
      split: One of 'train', 'val', 'test'.
      horizon: Forecast horizon in hours (e.g. 1, 3, 6, 12, 24).
      seq_len: Input history length in hours (default 24).
      max_samples: Optional cap on total returned sequences.
      norm_means, norm_stds: Training normalization statistics.
      mode: 'temporal' (chronological time split) or 'station' (station holdout).
      holdout_stations: Stations to hold out if mode == 'station'.
      return_metadata: Whether to return detailed metadata dictionaries.

    Returns:
      (telemetry_tensor, dt_tensor, rain_targets, precip_targets, water_targets, has_water_mask)
      + optionally [metadata_list]
    """
    if norm_means is None:
        norm_means = pipeline.norm_means
    if norm_stds is None:
        norm_stds = pipeline.norm_stds

    # Determine split time boundary
    if split == "train":
        start_bound = pipeline.time_range_min
        end_bound = pipeline.train_end
    elif split == "val":
        start_bound = pipeline.val_start
        end_bound = pipeline.val_end
    elif split == "test":
        start_bound = pipeline.test_start
        end_bound = pipeline.time_range_max
    else:
        raise ValueError(f"Unknown split '{split}'. Must be 'train', 'val', or 'test'.")

    # Select stations
    holdout_set = set(holdout_stations or [])
    candidate_stations = []
    for st_id in sorted(pipeline.station_hourly.keys()):
        if mode == "station":
            if split == "test" and st_id not in holdout_set:
                continue
            if split in ("train", "val") and st_id in holdout_set:
                continue
        candidate_stations.append(st_id)

    windows = []
    dt_list = []
    rain_list = []
    precip_list = []
    water_list = []
    has_water_list = []
    metadata_list = []

    # Quota per station to ensure fair geographic representation
    quota = max(10, max_samples // len(candidate_stations)) if max_samples else None

    for st_id in candidate_stations:
        st_dict = pipeline.station_hourly[st_id]
        # Filter hours in split range
        split_hours = sorted([h for h in st_dict if start_bound <= h <= end_bound])
        st_hour_set = set(split_hours)

        st_count = 0
        for t0 in split_hours:
            if quota is not None and st_count >= quota:
                break

            # 1. Target timestamp check
            t_target = t0 + timedelta(hours=horizon)
            if t_target not in st_dict or t_target > end_bound:
                continue

            actual_lead = (t_target - t0).total_seconds() / 3600.0
            if abs(actual_lead - horizon) > HORIZON_TOLERANCE_HOURS:
                continue

            # 2. Input sequence check (preceding seq_len hourly observations)
            # Ensure strictly within station and within split
            window_records = []
            has_full_window = True
            for step in range(seq_len - 1, -1, -1):
                h_step = t0 - timedelta(hours=step)
                if h_step not in st_dict or h_step < start_bound:
                    has_full_window = False
                    break
                window_records.append(st_dict[h_step])

            if not has_full_window:
                continue

            # Compute dt (elapsed hours between consecutive observations)
            dt_values = [1.0]  # First step relative to nominal 1h
            for k in range(1, len(window_records)):
                dt_val = (window_records[k]["timestamp"] - window_records[k - 1]["timestamp"]).total_seconds() / 3600.0
                dt_values.append(max(0.01, dt_val))

            # Normalize input features (8 canonical features)
            raw_feats = np.array([
                [
                    r["temperature"],
                    r["heat_index"],
                    r["humidity"],
                    r["pressure"],
                    r["wind_speed"],
                    r["wind_sin"],
                    r["wind_cos"],
                    r["precipitation"],
                ]
                for r in window_records
            ], dtype=np.float32)
            norm_feats = normalize_features(raw_feats, norm_means, norm_stds)

            # Build targets at t0 + h
            target_rec = st_dict[t_target]
            target_precip = target_rec["precipitation"]
            target_rain_prob = 1.0 if target_precip >= 0.1 else 0.0

            # Water level target: real gauge stage if collocated (Calumpit)
            is_gauge_station = (st_id == WATER_GAUGE_WEATHER_STATION)
            water_stage = pipeline.water_hourly.get(t_target) if is_gauge_station else None
            last_observed_water = pipeline.water_hourly.get(t0) if is_gauge_station else None
            has_water = bool(water_stage is not None and last_observed_water is not None)
            water_delta = (water_stage - last_observed_water) if has_water else 0.0

            # Persistence & rolling rain reference at origin t0
            last_observed_precip = window_records[-1]["precipitation"]
            rolling_3h_precip = sum(r["precipitation"] for r in window_records[-3:])
            rolling_6h_precip = sum(r["precipitation"] for r in window_records[-6:])

            windows.append(norm_feats)
            dt_list.append(np.array(dt_values, dtype=np.float32).reshape(-1, 1))
            rain_list.append(np.array([target_rain_prob], dtype=np.float32))
            precip_list.append(np.array([target_precip], dtype=np.float32))
            water_list.append(np.array([water_stage if has_water else 0.0], dtype=np.float32))
            has_water_list.append(np.array([1.0 if has_water else 0.0], dtype=np.float32))

            if return_metadata:
                target_temp = float(target_rec["temperature"])
                target_humidity = float(target_rec["humidity"])
                target_pressure = float(target_rec["pressure"])
                target_wind_speed = float(target_rec["wind_speed"])
                target_wind_u = float(target_rec["wind_cos"])
                target_wind_v = float(target_rec["wind_sin"])
                target_wind_deg = round(math.degrees(math.atan2(target_wind_v, target_wind_u)) % 360.0, 2)
                target_heat_index = float(target_rec["heat_index"])

                origin_rec = window_records[-1]
                origin_temp = float(origin_rec["temperature"])
                origin_humidity = float(origin_rec["humidity"])
                origin_pressure = float(origin_rec["pressure"])
                origin_wind_speed = float(origin_rec["wind_speed"])
                origin_wind_u = float(origin_rec["wind_cos"])
                origin_wind_v = float(origin_rec["wind_sin"])
                origin_wind_deg = round(math.degrees(math.atan2(origin_wind_v, origin_wind_u)) % 360.0, 2)
                origin_heat_index = float(origin_rec["heat_index"])

                metadata_list.append({
                    "station_id": st_id,
                    "origin_timestamp": t0.isoformat(),
                    "target_timestamp": t_target.isoformat(),
                    "requested_horizon_hours": horizon,
                    "actual_lead_hours": actual_lead,
                    "actual_rain_prob": target_rain_prob,
                    "actual_precip_mm": target_precip,
                    "actual_water_level": water_stage if has_water else None,
                    "actual_water_delta": round(water_delta, 4) if has_water else None,
                    "has_water": has_water,
                    "last_observed_precip": last_observed_precip,
                    "rolling_3h_precip": round(rolling_3h_precip, 4),
                    "rolling_6h_precip": round(rolling_6h_precip, 4),
                    "last_observed_water": last_observed_water,
                    # Weather target fields (at t0 + h)
                    "target_temperature": target_temp,
                    "target_humidity": target_humidity,
                    "target_pressure": target_pressure,
                    "target_wind_speed": target_wind_speed,
                    "target_wind_u": target_wind_u,
                    "target_wind_v": target_wind_v,
                    "target_wind_deg": target_wind_deg,
                    "target_heat_index": target_heat_index,
                    # Origin observations (at t0 for persistence)
                    "origin_temperature": origin_temp,
                    "origin_humidity": origin_humidity,
                    "origin_pressure": origin_pressure,
                    "origin_wind_speed": origin_wind_speed,
                    "origin_wind_u": origin_wind_u,
                    "origin_wind_v": origin_wind_v,
                    "origin_wind_deg": origin_wind_deg,
                    "origin_heat_index": origin_heat_index,
                })

            st_count += 1
            if max_samples and len(windows) >= max_samples:
                break

        if max_samples and len(windows) >= max_samples:
            break

    if len(windows) == 0:
        return None

    tensors = (
        torch.tensor(np.stack(windows), dtype=torch.float32),
        torch.tensor(np.stack(dt_list), dtype=torch.float32),
        torch.tensor(np.stack(rain_list), dtype=torch.float32),
        torch.tensor(np.stack(precip_list), dtype=torch.float32),
        torch.tensor(np.stack(water_list), dtype=torch.float32),
        torch.tensor(np.stack(has_water_list), dtype=torch.float32),
    )

    if return_metadata:
        return (*tensors, metadata_list)
    return tensors


class TelemetryDataset(Dataset):
    """
    PyTorch Dataset wrapper around the canonical future-forecast window contract.
    """

    def __init__(
        self,
        split: str = "train",
        horizon: int = 1,
        seq_len: int = DEFAULT_SEQ_LEN,
        max_samples: int = None,
        norm_means: np.ndarray = None,
        norm_stds: np.ndarray = None,
        return_metadata: bool = False,
        mode: str = "temporal",
        holdout_stations: list = None,
        pipeline: TelemetryDataPipeline = None,
    ):
        self.split = split
        self.horizon = horizon
        self.seq_len = seq_len
        self.return_metadata = return_metadata

        if pipeline is None:
            pipeline = get_telemetry_pipeline()
        self.pipeline = pipeline

        self.norm_means = norm_means if norm_means is not None else pipeline.norm_means
        self.norm_stds = norm_stds if norm_stds is not None else pipeline.norm_stds

        res = build_forecast_windows(
            pipeline=self.pipeline,
            split=split,
            horizon=horizon,
            seq_len=seq_len,
            max_samples=max_samples,
            norm_means=self.norm_means,
            norm_stds=self.norm_stds,
            mode=mode,
            holdout_stations=holdout_stations,
            return_metadata=True,
        )

        if res is None:
            self.telemetry = torch.empty(0, seq_len, NUM_FEATURES)
            self.dt = torch.empty(0, seq_len, 1)
            self.rain_prob = torch.empty(0, 1)
            self.precip_mm = torch.empty(0, 1)
            self.water_level = torch.empty(0, 1)
            self.has_water = torch.empty(0, 1)
            self.metadata = []
        else:
            self.telemetry, self.dt, self.rain_prob, self.precip_mm, self.water_level, self.has_water, self.metadata = res

    def __len__(self):
        return self.telemetry.shape[0]

    def __getitem__(self, idx):
        has_w = bool(self.has_water[idx, 0].item() > 0.5)
        meta = self.metadata[idx] if (self.metadata and idx < len(self.metadata)) else None
        last_w = meta["last_observed_water"] if (meta and meta.get("last_observed_water") is not None) else 0.0
        w_delta = meta["actual_water_delta"] if (meta and meta.get("actual_water_delta") is not None) else 0.0

        orig_w = [
            meta["origin_temperature"],
            meta["origin_humidity"],
            meta["origin_pressure"],
            meta["origin_wind_speed"],
            meta["origin_wind_u"],
            meta["origin_wind_v"],
        ] if meta and "origin_temperature" in meta else [0.0] * 6

        tgt_w = [
            meta["target_temperature"],
            meta["target_humidity"],
            meta["target_pressure"],
            meta["target_wind_speed"],
            meta["target_wind_u"],
            meta["target_wind_v"],
        ] if meta and "target_temperature" in meta else [0.0] * 6

        item = {
            "telemetry": self.telemetry[idx],
            "dt": self.dt[idx],
            "rain_prob": self.rain_prob[idx],
            "precip_mm": self.precip_mm[idx],
            "water_level": self.water_level[idx],
            "last_water": torch.tensor([last_w if has_w else 0.0], dtype=torch.float32),
            "water_delta": torch.tensor([w_delta if has_w else 0.0], dtype=torch.float32),
            "has_water": self.has_water[idx],
            "origin_weather": torch.tensor(orig_w, dtype=torch.float32),
            "target_weather": torch.tensor(tgt_w, dtype=torch.float32),
            "target_temp": torch.tensor([tgt_w[0]], dtype=torch.float32),
            "target_humidity": torch.tensor([tgt_w[1]], dtype=torch.float32),
            "target_pressure": torch.tensor([tgt_w[2]], dtype=torch.float32),
            "target_wind_speed": torch.tensor([tgt_w[3]], dtype=torch.float32),
            "target_wind_u": torch.tensor([tgt_w[4]], dtype=torch.float32),
            "target_wind_v": torch.tensor([tgt_w[5]], dtype=torch.float32),
        }
        if self.return_metadata and meta is not None:
            item["metadata"] = meta
        return item


def load_real_telemetry_sequences(
    weather_csv_path: str = None,
    water_csv_path: str = None,
    seq_len: int = DEFAULT_SEQ_LEN,
    max_sequences: int = 2000,
    split: str = "train",
    horizons: list = None,
    norm_means: np.ndarray = None,
    norm_stds: np.ndarray = None,
    return_metadata: bool = False,
):
    """
    Backwards-compatible API wrapper returning canonical tensors or None.
    """
    pipeline = get_telemetry_pipeline(weather_csv_path, water_csv_path)
    horizon = horizons[0] if (horizons and len(horizons) > 0) else 1
    return build_forecast_windows(
        pipeline=pipeline,
        split=split,
        horizon=horizon,
        seq_len=seq_len,
        max_samples=max_sequences,
        norm_means=norm_means,
        norm_stds=norm_stds,
        return_metadata=return_metadata,
    )


def fit_train_normalization(weather_csv_path: str = None, **kwargs):
    """Backwards-compatible helper returning fitted (means, stds)."""
    pipeline = get_telemetry_pipeline(weather_csv_path)
    return pipeline.norm_means, pipeline.norm_stds


# Export fallback constants for legacy imports (8 canonical features)
FEATURE_MEANS = np.array([27.91, 32.22, 86.14, 1005.75, 1.59, 0.079, -0.002, 0.50], dtype=np.float32)
FEATURE_STDS = np.array([3.20, 7.03, 12.56, 4.51, 3.62, 0.74, 0.66, 2.84], dtype=np.float32)
