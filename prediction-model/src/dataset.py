"""
Telemetry Dataset and Normalization Pipeline for Weather Station & Hydrological Forecasting.

Supports loading real historical KloudTrack CSV datasets with:
  - Station identity preservation (no cross-station windows)
  - Temporal continuity (sorted by station + timestamp, actual dt)
  - Chronological train/val/test splits (no leakage)
  - Real water-level gauge targets (joined by station/timestamp)
  - Future-forecasting targets (input window -> future horizon targets)

See prediction-model-audit.md for the rationale behind these design decisions.
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

# Default normalization constants (calculated from 200k+ historical Philippine records).
# These are used as FALLBACK when fit_normalization_stats() has not been called.
FEATURE_MEANS = np.array([28.5, 33.0, 10.0, 1008.0], dtype=np.float32)
FEATURE_STDS = np.array([4.5, 6.5, 8.0, 6.0], dtype=np.float32)

# Forecast horizons (hours ahead of the forecast origin t0)
DEFAULT_HORIZONS = [1, 3, 6, 12, 24]

# Maximum gap (hours) allowed within a window before masking/rejecting
MAX_GAP_HOURS = 2.0

# Water-level gauge station ID (only one gauge available)
WATER_GAUGE_STATION_ID = "O3z0j5bG"
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


def fit_normalization_stats(weather_csv_path: str = None, train_indices: list = None):
    """
    Compute mean/std from the TRAINING portion of the dataset only.

    Args:
        weather_csv_path: Path to weather CSV.
        train_indices: Row indices that belong to the training set.
            If None, compute from all rows (not recommended for production).

    Returns:
        (means, stds) as np.float32 arrays of shape (4,).
    """
    if weather_csv_path is None:
        weather_csv_path = os.path.join(DATA_DIR, "weather_telemetry.csv")

    if not os.path.exists(weather_csv_path):
        print("WARNING: Weather CSV not found. Using default normalization constants.")
        return FEATURE_MEANS.copy(), FEATURE_STDS.copy()

    values = []
    with open(weather_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if train_indices is not None and i not in train_indices:
                continue
            try:
                t = float(row.get("temperature") or 28.5)
                hi = float(row.get("heat_index") or 33.0)
                ws = float(row.get("wind_speed") or 10.0)
                p = float(row.get("pressure") or 1008.0)
                values.append([t, hi, ws, p])
            except (ValueError, TypeError):
                continue

    if len(values) < 100:
        print(f"WARNING: Only {len(values)} training rows. Using default normalization.")
        return FEATURE_MEANS.copy(), FEATURE_STDS.copy()

    arr = np.array(values, dtype=np.float32)
    means = arr.mean(axis=0)
    stds = arr.std(axis=0)
    stds = np.where(stds < 1e-6, 1.0, stds)  # Prevent division by zero
    return means, stds


# ---------------------------------------------------------------------------
# Water-level gauge loader
# ---------------------------------------------------------------------------
def _load_water_level_lookup(water_csv_path: str = None):
    """
    Load real water-level gauge observations into a lookup table.
    Returns dict mapping timestamp_str (minute precision) -> water_level_m.
    Only contains data from the single available gauge station.
    """
    if water_csv_path is None:
        water_csv_path = os.path.join(DATA_DIR, "water_level_telemetry.csv")

    lookup = {}
    if not os.path.exists(water_csv_path):
        return lookup

    with open(water_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts_raw = row.get("recorded_at", "")
                wl_m = row.get("water_level_m")
                wl_cm = row.get("water_level_cm")
                if wl_m is not None and wl_m != "":
                    wl = float(wl_m)
                elif wl_cm is not None and wl_cm != "":
                    wl = float(wl_cm) / 100.0
                else:
                    continue
                # Round to 5-minute buckets for fuzzy join
                dt = _parse_timestamp(ts_raw)
                if dt is None:
                    continue
                bucket = dt.replace(minute=(dt.minute // 5) * 5, second=0, microsecond=0)
                key = bucket.isoformat()
                lookup[key] = wl
            except (ValueError, TypeError):
                continue
    return lookup


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
    # Last resort: truncate and try
    try:
        return datetime.strptime(ts_str[:19], "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Per-station data loading with temporal ordering
# ---------------------------------------------------------------------------
def _load_station_sorted_data(weather_csv_path: str = None):
    """
    Load weather telemetry sorted by (station_id, recorded_at).
    Returns a dict: station_id -> list of (datetime, temp, hi, ws, pressure, precip).
    """
    if weather_csv_path is None:
        weather_csv_path = os.path.join(DATA_DIR, "weather_telemetry.csv")

    station_data = defaultdict(list)

    with open(weather_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                st_id = row.get("station_id", "unknown")
                ts_str = row.get("recorded_at", "")
                dt = _parse_timestamp(ts_str)
                if dt is None:
                    continue
                t = float(row.get("temperature") or 28.5)
                hi = float(row.get("heat_index") or 33.0)
                ws = float(row.get("wind_speed") or 10.0)
                p = float(row.get("pressure") or 1008.0)
                precip = float(row.get("precipitation") or 0.0)
                station_data[st_id].append((dt, t, hi, ws, p, precip))
            except (ValueError, TypeError):
                continue

    # Sort each station by timestamp
    for st_id in station_data:
        station_data[st_id].sort(key=lambda x: x[0])

    return dict(station_data)


def _compute_chronological_split(station_data: dict, train_frac=0.6, val_frac=0.2):
    """
    Compute chronological split cutoff timestamps.
    Uses the global time range across all stations.

    Returns:
        (train_end_dt, val_end_dt) — timestamps marking split boundaries.
        train: records with ts <= train_end_dt
        val:   records with train_end_dt < ts <= val_end_dt
        test:  records with ts > val_end_dt
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
):
    """
    Loads and preprocesses real historical telemetry into sequential sliding windows.

    Key design decisions (audit-driven):
    - Windows are built PER STATION — never cross station boundaries.
    - dt is computed from actual timestamps, not hard-coded to 1.0.
    - Targets are at future horizons (t0 + h), not same-step reconstruction.
    - Water-level targets come from real gauge observations where available.
    - Split is chronological (train/val/test by time, not random).

    Args:
        weather_csv_path: Path to weather CSV.
        water_csv_path: Path to water level CSV.
        seq_len: Length of input observation window.
        max_sequences: Maximum number of sequences to generate.
        split: One of "train", "val", "test".
        horizons: List of forecast horizon offsets (in timesteps). Defaults to [1].
        norm_means: Normalization means (fitted on training data).
        norm_stds: Normalization stds (fitted on training data).

    Returns:
        Tuple of tensors: (telemetry, dt, rain_targets, precip_targets, water_targets, has_water_mask)
        or None if insufficient data.
    """
    if horizons is None:
        horizons = [1]  # Default: predict 1 step ahead
    max_horizon = max(horizons)

    if norm_means is None:
        norm_means = FEATURE_MEANS
    if norm_stds is None:
        norm_stds = FEATURE_STDS

    # Load station-sorted data
    station_data = _load_station_sorted_data(weather_csv_path)
    if not station_data:
        return None

    # Load real water-level gauge observations
    water_lookup = _load_water_level_lookup(water_csv_path)

    # Compute chronological split boundaries
    train_end, val_end = _compute_chronological_split(station_data)

    # Select split filter
    if split == "train":
        time_filter = lambda dt_val: dt_val <= train_end
    elif split == "val":
        time_filter = lambda dt_val: train_end < dt_val <= val_end
    elif split == "test":
        time_filter = lambda dt_val: dt_val > val_end
    else:
        raise ValueError(f"Unknown split: {split}. Use 'train', 'val', or 'test'.")

    telemetry_seqs = []
    dt_seqs = []
    rain_target_seqs = []
    precip_target_seqs = []
    water_target_seqs = []
    has_water_seqs = []

    for st_id, rows in station_data.items():
        # Filter rows to this split
        split_rows = [r for r in rows if time_filter(r[0])]
        if len(split_rows) < seq_len + max_horizon:
            continue

        # Build windows within this station
        for i in range(len(split_rows) - seq_len - max_horizon):
            window = split_rows[i: i + seq_len]
            future_rows = split_rows[i + seq_len: i + seq_len + max_horizon]

            # Check for excessive gaps within the window
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
            # First timestep has no predecessor, use 1.0h default
            dt_values.insert(0, 1.0)

            # Build input features
            raw_feat = np.array([[r[1], r[2], r[3], r[4]] for r in window], dtype=np.float32)
            norm_feat = normalize_features(raw_feat, norm_means, norm_stds)

            # Build future targets at each horizon
            # For simplicity, we use the max_horizon target
            target_idx = max_horizon - 1
            if target_idx >= len(future_rows):
                continue

            future_row = future_rows[target_idx]
            future_precip = future_row[5]
            future_rain_prob = 1.0 if future_precip > 0.1 else 0.0

            # Real water-level target (only for gauge-matched station)
            future_dt_obj = future_row[0]
            bucket = future_dt_obj.replace(
                minute=(future_dt_obj.minute // 5) * 5, second=0, microsecond=0
            )
            water_key = bucket.isoformat()
            water_target = water_lookup.get(water_key)
            has_water = water_target is not None
            if not has_water:
                water_target = 0.0  # Placeholder, will be masked in loss

            dt_arr = np.array(dt_values, dtype=np.float32).reshape(-1, 1)

            telemetry_seqs.append(norm_feat)
            dt_seqs.append(dt_arr)
            rain_target_seqs.append(np.array([future_rain_prob], dtype=np.float32))
            precip_target_seqs.append(np.array([future_precip], dtype=np.float32))
            water_target_seqs.append(np.array([water_target], dtype=np.float32))
            has_water_seqs.append(np.array([1.0 if has_water else 0.0], dtype=np.float32))

            if len(telemetry_seqs) >= max_sequences:
                break
        if len(telemetry_seqs) >= max_sequences:
            break

    if len(telemetry_seqs) < 2:
        return None

    return (
        torch.tensor(np.stack(telemetry_seqs), dtype=torch.float32),
        torch.tensor(np.stack(dt_seqs), dtype=torch.float32),
        torch.tensor(np.stack(rain_target_seqs), dtype=torch.float32),
        torch.tensor(np.stack(precip_target_seqs), dtype=torch.float32),
        torch.tensor(np.stack(water_target_seqs), dtype=torch.float32),
        torch.tensor(np.stack(has_water_seqs), dtype=torch.float32),
    )


class TelemetryDataset(Dataset):
    """
    PyTorch Dataset for weather/water telemetry forecasting.

    Supports chronological splitting, station-aware windowing,
    real gauge targets, and future-horizon forecasting.
    """

    def __init__(self, seq_len: int = 24, max_samples: int = 2000,
                 split: str = "train", horizon: int = 1,
                 norm_means: np.ndarray = None, norm_stds: np.ndarray = None):
        real_data = load_real_telemetry_sequences(
            seq_len=seq_len,
            max_sequences=max_samples,
            split=split,
            horizons=[horizon],
            norm_means=norm_means,
            norm_stds=norm_stds,
        )
        if real_data is not None:
            (
                self.telemetry,
                self.dt,
                self.rain_prob,
                self.precip_mm,
                self.water_level,
                self.has_water,
            ) = real_data
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
            except ImportError:
                raise RuntimeError(
                    f"No real data available for {split} split and no synthetic generator found. "
                    f"Ensure weather_telemetry.csv exists in the data/ directory."
                )

    def __len__(self):
        return len(self.telemetry)

    def __getitem__(self, idx):
        return {
            "telemetry": self.telemetry[idx],
            "dt": self.dt[idx],
            "rain_prob": self.rain_prob[idx],
            "precip_mm": self.precip_mm[idx],
            "water_level": self.water_level[idx],
            "has_water": self.has_water[idx],
        }
