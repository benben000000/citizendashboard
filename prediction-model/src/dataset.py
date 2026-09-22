"""
Telemetry Dataset and Normalization Pipeline for Weather Station & Hydrological Forecasting.

Supports loading real historical KloudTrack CSV datasets with:
  - Data quality filtering & outlier quarantine (date validation, physical bounds)
  - Station identity preservation (no cross-station windows)
  - Temporal continuity (sorted by station + timestamp, actual dt)
  - Chronological train/val/test splits (strict temporal holdout)
  - Train-fitted normalization (means and stds computed exclusively on train split)
  - Real water-level gauge targets (joined by station/timestamp, masked unobserved)
  - Future-forecasting targets (input window -> future horizon targets)
  - Fair multi-station window representation (no single-station dominance)

See prediction-model-audit-followup.md for the rationale behind these design decisions.
"""

import os
import csv
import hashlib
import json
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# Fallback normalization constants (used ONLY when fit_train_normalization cannot run).
FEATURE_MEANS = np.array([27.2, 31.0, 2.0, 1004.5], dtype=np.float32)
FEATURE_STDS = np.array([2.8, 6.3, 3.0, 4.6], dtype=np.float32)

# Forecast horizons (hours ahead of the forecast origin t0)
DEFAULT_HORIZONS = [1, 3, 6, 12, 24]

# Maximum gap (hours) allowed within an observation window before rejecting
MAX_GAP_HOURS = 2.0

# Allowed collection date bounds for KloudTrack 2026 telemetry corpus (Fix 1)
MIN_VALID_YEAR = 2025
MAX_VALID_YEAR = 2027

# Physically plausible sensor ranges for tropical Philippine surface telemetry
PHYSICAL_BOUNDS = {
    "temperature": (10.0, 50.0),    # Celsius
    "heat_index": (10.0, 70.0),     # Celsius
    "wind_speed": (0.0, 180.0),     # km/h
    "pressure": (900.0, 1050.0),    # hPa
    "precipitation": (0.0, 300.0),  # mm/h
}

# Water-level gauge station ID (only one gauge available)
WATER_GAUGE_STATION_ID = "O3z0j5bG"  # Calumpit WLMS - Bulacan
# Nearest weather station to the gauge
WATER_GAUGE_WEATHER_STATION = "3nzr48bG"  # Calumpit AWS - Bulacan


def normalize_features(features: np.ndarray, means: np.ndarray = None, stds: np.ndarray = None) -> np.ndarray:
    """Normalize raw telemetry array [..., 4] to zero mean and unit variance."""
    if means is None:
        means = FEATURE_MEANS
    if stds is None:
        stds = FEATURE_STDS
    return (features - means) / stds


def denormalize_features(features: np.ndarray, means: np.ndarray = None, stds: np.ndarray = None) -> np.ndarray:
    """Denormalize scaled features back to physical units."""
    if means is None:
        means = FEATURE_MEANS
    if stds is None:
        stds = FEATURE_STDS
    return features * stds + means


def _parse_timestamp(ts_str: str):
    """Parse an ISO timestamp string robustly."""
    if not ts_str:
        return None
    ts_str = ts_str.strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(ts_str, fmt)
        except ValueError:
            continue
    try:
        return datetime.strptime(ts_str[:19], "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Water-level gauge loader
# ---------------------------------------------------------------------------
_WATER_LOOKUP_CACHE = {}

def _load_water_level_lookup(water_csv_path: str = None):
    """
    Load real water-level gauge observations into a lookup table.
    Returns dict mapping timestamp_str (minute precision) -> water_level_m.
    Only contains data from the single available gauge station (Calumpit WLMS).
    """
    if water_csv_path is None:
        water_csv_path = os.path.join(DATA_DIR, "water_level_telemetry.csv")

    if water_csv_path in _WATER_LOOKUP_CACHE:
        return _WATER_LOOKUP_CACHE[water_csv_path]

    lookup = {}
    if not os.path.exists(water_csv_path):
        return lookup

    with open(water_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts_raw = row.get("recorded_at", "")
                dt = _parse_timestamp(ts_raw)
                if dt is None or dt.year < MIN_VALID_YEAR or dt.year > MAX_VALID_YEAR:
                    continue
                wl_m = row.get("water_level_m")
                wl_cm = row.get("water_level_cm")
                if wl_m is not None and wl_m != "":
                    wl = float(wl_m)
                elif wl_cm is not None and wl_cm != "":
                    wl = float(wl_cm) / 100.0
                else:
                    continue
                # Round to 5-minute buckets for fuzzy join
                bucket = dt.replace(minute=(dt.minute // 5) * 5, second=0, microsecond=0)
                key = bucket.isoformat()
                lookup[key] = wl
            except (ValueError, TypeError):
                continue
    _WATER_LOOKUP_CACHE[water_csv_path] = lookup
    return lookup


# ---------------------------------------------------------------------------
# Per-station data loading with temporal ordering and data quality quarantine
# ---------------------------------------------------------------------------
_STATION_DATA_CACHE = {}

def _load_station_sorted_data(weather_csv_path: str = None):
    """
    Load weather telemetry sorted by (station_id, recorded_at).
    Applies data-quality quarantine:
      1. Rejects date outliers outside [MIN_VALID_YEAR, MAX_VALID_YEAR] (e.g. 2069 bug).
      2. Rejects corrupt hardware sensor spikes outside physical bounds.
    Returns a dict: station_id -> list of (datetime, temp, hi, ws, pressure, precip).
    """
    if weather_csv_path is None:
        weather_csv_path = os.path.join(DATA_DIR, "weather_telemetry.csv")

    if weather_csv_path in _STATION_DATA_CACHE:
        return _STATION_DATA_CACHE[weather_csv_path]

    station_data = defaultdict(list)
    quarantined_dates = 0
    quarantined_spikes = 0

    with open(weather_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                st_id = row.get("station_id", "unknown")
                ts_str = row.get("recorded_at", "")
                dt = _parse_timestamp(ts_str)
                if dt is None:
                    continue

                # Data Quality Rule 1: Date outlier quarantine (Fix 1)
                if dt.year < MIN_VALID_YEAR or dt.year > MAX_VALID_YEAR:
                    quarantined_dates += 1
                    continue

                t = float(row.get("temperature") or 28.5)
                hi = float(row.get("heat_index") or 33.0)
                ws = float(row.get("wind_speed") or 10.0)
                p = float(row.get("pressure") or 1008.0)
                precip = float(row.get("precipitation") or 0.0)

                # Data Quality Rule 2: Physical sensor bounds quarantine
                if not (
                    PHYSICAL_BOUNDS["temperature"][0] <= t <= PHYSICAL_BOUNDS["temperature"][1] and
                    PHYSICAL_BOUNDS["heat_index"][0] <= hi <= PHYSICAL_BOUNDS["heat_index"][1] and
                    PHYSICAL_BOUNDS["wind_speed"][0] <= ws <= PHYSICAL_BOUNDS["wind_speed"][1] and
                    PHYSICAL_BOUNDS["pressure"][0] <= p <= PHYSICAL_BOUNDS["pressure"][1] and
                    PHYSICAL_BOUNDS["precipitation"][0] <= precip <= PHYSICAL_BOUNDS["precipitation"][1]
                ):
                    quarantined_spikes += 1
                    continue

                station_data[st_id].append((dt, t, hi, ws, p, precip))
            except (ValueError, TypeError):
                continue

    # Sort each station by timestamp
    for st_id in station_data:
        station_data[st_id].sort(key=lambda x: x[0])

    _STATION_DATA_CACHE[weather_csv_path] = dict(station_data)
    return _STATION_DATA_CACHE[weather_csv_path]


def _compute_chronological_split(station_data: dict, train_frac=0.6, val_frac=0.2):
    """
    Compute chronological split cutoff timestamps using the global time range.
    Returns:
        (train_end_dt, val_end_dt)
        train: ts <= train_end_dt
        val:   train_end_dt < ts <= val_end_dt
        test:  ts > val_end_dt
    """
    all_times = []
    for rows in station_data.values():
        for row in rows:
            all_times.append(row[0])

    all_times.sort()
    n = len(all_times)
    train_end = all_times[int(n * train_frac)]
    val_end = all_times[int(n * (train_frac + val_frac))]
    return train_end, val_end


# ---------------------------------------------------------------------------
# Train-fitted Normalization (Fix 2)
# ---------------------------------------------------------------------------
def fit_train_normalization(weather_csv_path: str = None, station_data: dict = None, train_end: datetime = None):
    """
    Compute mean and std strictly on the clean TRAINING partition (ts <= train_end).
    Prevents data leakage into validation or test sets.

    Returns:
        (means, stds) as np.float32 arrays of shape (4,).
    """
    if station_data is None:
        station_data = _load_station_sorted_data(weather_csv_path)
    if not station_data:
        return FEATURE_MEANS.copy(), FEATURE_STDS.copy()

    if train_end is None:
        train_end, _ = _compute_chronological_split(station_data)

    train_vals = []
    for rows in station_data.values():
        for r in rows:
            if r[0] <= train_end:
                train_vals.append([r[1], r[2], r[3], r[4]])

    if len(train_vals) < 100:
        return FEATURE_MEANS.copy(), FEATURE_STDS.copy()

    arr = np.array(train_vals, dtype=np.float32)
    means = arr.mean(axis=0)
    stds = arr.std(axis=0)
    stds = np.where(stds < 1e-6, 1.0, stds)
    return means, stds


# ---------------------------------------------------------------------------
# Main dataset builder with all audit fixes
# ---------------------------------------------------------------------------
def load_real_telemetry_sequences(
    weather_csv_path: str = None,
    water_csv_path: str = None,
    seq_len: int = 24,
    max_sequences: int = 2000,
    split: str = "train",
    horizons: list = None,
    norm_means: np.ndarray = None,
    norm_stds: np.ndarray = None,
    return_metadata: bool = False,
):
    """
    Loads and preprocesses real historical telemetry into sequential sliding windows.

    Key design decisions (audit-driven):
    - Date outliers and sensor glitches are quarantined (Fix 1).
    - Normalization parameters are strictly fitted on the train split (Fix 2).
    - Windows are built PER STATION — never cross station boundaries (Fix 5).
    - dt is computed from actual timestamps, not hard-coded (Fix 5).
    - Targets are at future horizons (t0 + h), not same-step reconstruction (Fix 6).
    - Water-level targets come from real gauge observations where available (Fix 6).
    - Split is chronological (train/val/test by time, not random) (Fix 6).
    - Balanced multi-station sampling to prevent single-station dominance.

    Args:
        weather_csv_path: Path to weather CSV.
        water_csv_path: Path to water level CSV.
        seq_len: Length of input observation window.
        max_sequences: Maximum number of sequences to generate.
        split: One of "train", "val", "test".
        horizons: List of forecast horizon offsets (in hours). Defaults to [1].
        norm_means: Normalization means (fitted on training data).
        norm_stds: Normalization stds (fitted on training data).
        return_metadata: Whether to return list of sample metadata dicts.

    Returns:
        If return_metadata is False:
          (telemetry, dt, rain_targets, precip_targets, water_targets, has_water_mask)
        If return_metadata is True:
          (telemetry, dt, rain_targets, precip_targets, water_targets, has_water_mask, metadata_list)
        or None if insufficient data.
    """
    if horizons is None:
        horizons = [1]  # Default: predict 1 hour ahead
    max_horizon = max(horizons)

    # Load station-sorted data (with date quarantine & physical bounds filter)
    station_data = _load_station_sorted_data(weather_csv_path)
    if not station_data:
        return None

    # Compute chronological split boundaries
    train_end, val_end = _compute_chronological_split(station_data)

    # Ensure train-fitted normalization if not explicitly provided
    if norm_means is None or norm_stds is None:
        norm_means, norm_stds = fit_train_normalization(station_data=station_data, train_end=train_end)

    # Load real water-level gauge observations
    water_lookup = _load_water_level_lookup(water_csv_path)

    # Select split filter
    if split == "train":
        time_filter = lambda dt_val: dt_val <= train_end
    elif split == "val":
        time_filter = lambda dt_val: train_end < dt_val <= val_end
    elif split == "test":
        time_filter = lambda dt_val: dt_val > val_end
    else:
        raise ValueError(f"Unknown split: {split}. Use 'train', 'val', or 'test'.")

    # Filter station rows
    valid_station_data = {}
    for st_id, rows in station_data.items():
        s_rows = [r for r in rows if time_filter(r[0])]
        if len(s_rows) >= seq_len + max_horizon:
            valid_station_data[st_id] = s_rows

    if not valid_station_data:
        return None

    # Fair allocation across all available stations
    num_stations = len(valid_station_data)
    quota_per_station = max(10, max_sequences // num_stations)

    telemetry_seqs = []
    dt_seqs = []
    rain_target_seqs = []
    precip_target_seqs = []
    water_target_seqs = []
    has_water_seqs = []
    metadata_list = []

    for st_id, split_rows in valid_station_data.items():
        station_window_count = 0
        total_possible = len(split_rows) - seq_len - max_horizon
        stride = max(1, total_possible // quota_per_station) if quota_per_station > 0 else 1

        for i in range(0, total_possible, stride):
            if station_window_count >= quota_per_station and len(telemetry_seqs) >= max_sequences:
                break

            window = split_rows[i: i + seq_len]
            future_rows = split_rows[i + seq_len: i + seq_len + max_horizon]

            # Check for excessive gaps within the input window
            has_gap = False
            dt_values = []
            for k in range(1, len(window)):
                elapsed = (window[k][0] - window[k - 1][0]).total_seconds() / 3600.0
                if elapsed > MAX_GAP_HOURS:
                    has_gap = True
                    break
                dt_values.append(max(0.01, elapsed))  # Minimum 0.01h to avoid zero
            if has_gap:
                continue

            # First timestep has no predecessor in the window; default to 1.0h
            dt_values.insert(0, 1.0)

            # Build input features
            raw_feat = np.array([[r[1], r[2], r[3], r[4]] for r in window], dtype=np.float32)
            norm_feat = normalize_features(raw_feat, norm_means, norm_stds)

            # Build future targets at the specified horizon
            target_idx = max_horizon - 1
            if target_idx >= len(future_rows):
                continue

            future_row = future_rows[target_idx]
            future_precip = future_row[5]
            future_rain_prob = 1.0 if future_precip > 0.1 else 0.0

            # Real water-level target from gauge
            future_dt_obj = future_row[0]
            bucket = future_dt_obj.replace(
                minute=(future_dt_obj.minute // 5) * 5, second=0, microsecond=0
            )
            water_key = bucket.isoformat()
            # Water observations only matched for gauge-collocated station
            is_gauge_station = (st_id == WATER_GAUGE_WEATHER_STATION)
            water_target = water_lookup.get(water_key) if is_gauge_station else None
            has_water = water_target is not None

            # Persistence reference: water stage at origin t0
            origin_bucket = window[-1][0].replace(
                minute=(window[-1][0].minute // 5) * 5, second=0, microsecond=0
            ).isoformat()
            last_water_obs = water_lookup.get(origin_bucket) if is_gauge_station else None
            last_precip_obs = window[-1][5]

            dt_arr = np.array(dt_values, dtype=np.float32).reshape(-1, 1)

            telemetry_seqs.append(norm_feat)
            dt_seqs.append(dt_arr)
            rain_target_seqs.append(np.array([future_rain_prob], dtype=np.float32))
            precip_target_seqs.append(np.array([future_precip], dtype=np.float32))
            water_target_seqs.append(np.array([water_target if has_water else 0.0], dtype=np.float32))
            has_water_seqs.append(np.array([1.0 if has_water else 0.0], dtype=np.float32))

            if return_metadata:
                metadata_list.append({
                    "station_id": st_id,
                    "origin_timestamp": window[-1][0].isoformat(),
                    "target_timestamp": future_dt_obj.isoformat(),
                    "horizon": max_horizon,
                    "actual_rain_prob": future_rain_prob,
                    "actual_precip_mm": future_precip,
                    "actual_water_level": water_target if has_water else None,
                    "last_observed_water": last_water_obs,
                    "last_observed_precip": last_precip_obs,
                    "has_water": has_water,
                })

            station_window_count += 1
            if len(telemetry_seqs) >= max_sequences:
                break

    if len(telemetry_seqs) < 2:
        return None

    tensors = (
        torch.tensor(np.stack(telemetry_seqs), dtype=torch.float32),
        torch.tensor(np.stack(dt_seqs), dtype=torch.float32),
        torch.tensor(np.stack(rain_target_seqs), dtype=torch.float32),
        torch.tensor(np.stack(precip_target_seqs), dtype=torch.float32),
        torch.tensor(np.stack(water_target_seqs), dtype=torch.float32),
        torch.tensor(np.stack(has_water_seqs), dtype=torch.float32),
    )

    if return_metadata:
        return (*tensors, metadata_list)
    return tensors


class TelemetryDataset(Dataset):
    """
    PyTorch Dataset for weather/water telemetry future-forecasting.

    Supports chronological splitting, station-aware windowing,
    train-fitted normalization, real gauge targets, and future-horizon forecasting.
    """

    def __init__(self, seq_len: int = 24, max_samples: int = 2000,
                 split: str = "train", horizon: int = 1,
                 norm_means: np.ndarray = None, norm_stds: np.ndarray = None,
                 return_metadata: bool = False):
        self.split = split
        self.horizon = horizon
        self.return_metadata = return_metadata

        # Compute or retain train-fitted normalization stats
        if norm_means is None or norm_stds is None:
            norm_means, norm_stds = fit_train_normalization()
        self.norm_means = norm_means
        self.norm_stds = norm_stds

        real_data = load_real_telemetry_sequences(
            seq_len=seq_len,
            max_sequences=max_samples,
            split=split,
            horizons=[horizon],
            norm_means=self.norm_means,
            norm_stds=self.norm_stds,
            return_metadata=return_metadata,
        )
        if real_data is not None:
            if return_metadata:
                (
                    self.telemetry,
                    self.dt,
                    self.rain_prob,
                    self.precip_mm,
                    self.water_level,
                    self.has_water,
                    self.metadata,
                ) = real_data
            else:
                (
                    self.telemetry,
                    self.dt,
                    self.rain_prob,
                    self.precip_mm,
                    self.water_level,
                    self.has_water,
                ) = real_data
                self.metadata = None
            print(f"Loaded {len(self.telemetry)} {split} sequences from real KloudTrack dataset (horizon={horizon}h).")
        else:
            print(f"Warning: Real dataset not found or insufficient for {split} split, generating benchmark synthetic batch.")
            try:
                from dataset_synth import generate_synthetic_telemetry_batch
                (
                    self.telemetry,
                    self.dt,
                    self.rain_prob,
                    self.precip_mm,
                    self.water_level,
                ) = generate_synthetic_telemetry_batch(num_samples=max_samples, seq_len=seq_len)
                self.has_water = torch.zeros(len(self.telemetry), 1)
                self.metadata = None
            except ImportError:
                raise RuntimeError(
                    f"No real data available for {split} split and no synthetic generator found. "
                    f"Ensure weather_telemetry.csv exists in the data/ directory."
                )

    def __len__(self):
        return len(self.telemetry)

    def __getitem__(self, idx):
        item = {
            "telemetry": self.telemetry[idx],
            "dt": self.dt[idx],
            "rain_prob": self.rain_prob[idx],
            "precip_mm": self.precip_mm[idx],
            "water_level": self.water_level[idx],
            "has_water": self.has_water[idx],
        }
        if self.return_metadata and self.metadata is not None:
            item["metadata"] = self.metadata[idx]
        return item
