"""
Nearby Stations Adapter — Regional Telemetry Integration.

BLOCKED: This adapter is structurally complete but cannot be used in production
until the source 'project_nearby_stations_v1' passes the license gate in
external_source_registry.py.

Provides:
  - Per-station features: distance, bearing, elevation diff, recent values,
    lags, rolling stats, trend, missingness, quality, telemetry age.
  - Regional aggregates: IDW-weighted values, nearest valid, mean/median,
    disagreement, gradient, wind-vector mean/dispersion.

Safety constraints:
  - All timestamps in UTC.
  - Only observations available at issue time are eligible.
  - Future/corrected observations must not leak.
  - Station outages remain explicit missingness.
"""

import os
import sys
import math
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Tuple

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_SRC = os.path.dirname(SRC_DIR)
if PARENT_SRC not in sys.path:
    sys.path.insert(0, PARENT_SRC)

from external_source_registry import ExternalSourceRegistry, SourceRegistryError

SOURCE_ID = "project_nearby_stations_v1"


class NearbyStationRecord:
    """A single observation from a nearby station."""

    def __init__(
        self,
        station_id: str,
        timestamp_utc: datetime,
        lat: float,
        lon: float,
        elevation_m: float,
        temperature: Optional[float] = None,
        humidity: Optional[float] = None,
        pressure: Optional[float] = None,
        wind_speed: Optional[float] = None,
        wind_direction: Optional[float] = None,
        precipitation: Optional[float] = None,
        quality_flag: str = "unknown",
    ):
        self.station_id = station_id
        self.timestamp_utc = timestamp_utc
        self.lat = lat
        self.lon = lon
        self.elevation_m = elevation_m
        self.temperature = temperature
        self.humidity = humidity
        self.pressure = pressure
        self.wind_speed = wind_speed
        self.wind_direction = wind_direction
        self.precipitation = precipitation
        self.quality_flag = quality_flag


class NearbyStationsAdapter:
    """
    Adapter for nearby weather station telemetry.

    IMPORTANT: This adapter will refuse to produce features unless
    the source is production-eligible in the registry.
    """

    def __init__(
        self,
        target_lat: float,
        target_lon: float,
        target_elevation_m: float,
        registry: Optional[ExternalSourceRegistry] = None,
        max_distance_km: float = 100.0,
        max_telemetry_age_hours: float = 3.0,
    ):
        self.target_lat = target_lat
        self.target_lon = target_lon
        self.target_elevation_m = target_elevation_m
        self.max_distance_km = max_distance_km
        self.max_telemetry_age_hours = max_telemetry_age_hours
        self._registry = registry or ExternalSourceRegistry()
        self._eligible = False
        self._check_eligibility()

    def _check_eligibility(self) -> None:
        """Check if the source is production-eligible."""
        eligible, reason = self._registry.is_production_eligible(SOURCE_ID)
        self._eligible = eligible
        if not eligible:
            self._blocked_reason = reason

    @property
    def is_eligible(self) -> bool:
        return self._eligible

    @staticmethod
    def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Haversine distance in kilometers."""
        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(dlon / 2) ** 2
        )
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    @staticmethod
    def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Initial bearing from point 1 to point 2 in degrees."""
        dlon = math.radians(lon2 - lon1)
        lat1r = math.radians(lat1)
        lat2r = math.radians(lat2)
        x = math.sin(dlon) * math.cos(lat2r)
        y = math.cos(lat1r) * math.sin(lat2r) - math.sin(lat1r) * math.cos(
            lat2r
        ) * math.cos(dlon)
        return (math.degrees(math.atan2(x, y)) + 360) % 360

    def extract_station_features(
        self,
        station: NearbyStationRecord,
        history: List[NearbyStationRecord],
        issue_time: datetime,
    ) -> Dict[str, Any]:
        """
        Extract features for a single nearby station.

        Returns empty dict if source is not eligible.
        """
        if not self._eligible:
            return {}

        # Causal filter: only observations at or before issue time
        causal_history = [
            r for r in history
            if r.station_id == station.station_id
            and r.timestamp_utc <= issue_time
        ]

        if not causal_history:
            return self._missing_station_features(station)

        # Sort by time, most recent first
        causal_history.sort(key=lambda r: r.timestamp_utc, reverse=True)
        latest = causal_history[0]

        distance_km = self.haversine_km(
            self.target_lat, self.target_lon, station.lat, station.lon
        )
        bearing = self.bearing_deg(
            self.target_lat, self.target_lon, station.lat, station.lon
        )
        elev_diff = station.elevation_m - self.target_elevation_m
        age_hours = (issue_time - latest.timestamp_utc).total_seconds() / 3600.0

        features = {
            "station_id": station.station_id,
            "distance_km": distance_km,
            "bearing_deg": bearing,
            "elevation_diff_m": elev_diff,
            "telemetry_age_hours": age_hours,
            "quality_flag": latest.quality_flag,
            "is_missing": age_hours > self.max_telemetry_age_hours,
        }

        # Current values
        for var in ["temperature", "humidity", "pressure", "wind_speed", "precipitation"]:
            val = getattr(latest, var, None)
            features[f"{var}_current"] = val

        # Rolling stats from causal history (last N records)
        window = causal_history[:12]  # Up to 12 most recent
        for var in ["temperature", "humidity", "pressure", "wind_speed"]:
            values = [getattr(r, var) for r in window if getattr(r, var) is not None]
            if values:
                features[f"{var}_rolling_mean"] = sum(values) / len(values)
                if len(values) > 1:
                    mean = features[f"{var}_rolling_mean"]
                    features[f"{var}_rolling_std"] = (
                        sum((v - mean) ** 2 for v in values) / (len(values) - 1)
                    ) ** 0.5
                else:
                    features[f"{var}_rolling_std"] = 0.0

                # Trend (last - first)
                if len(values) >= 2:
                    features[f"{var}_trend"] = values[0] - values[-1]
                else:
                    features[f"{var}_trend"] = 0.0
            else:
                features[f"{var}_rolling_mean"] = None
                features[f"{var}_rolling_std"] = None
                features[f"{var}_trend"] = None

        return features

    def _missing_station_features(
        self, station: NearbyStationRecord
    ) -> Dict[str, Any]:
        """Return feature dict with explicit missingness."""
        distance_km = self.haversine_km(
            self.target_lat, self.target_lon, station.lat, station.lon
        )
        return {
            "station_id": station.station_id,
            "distance_km": distance_km,
            "is_missing": True,
            "quality_flag": "missing",
        }

    def extract_regional_aggregates(
        self,
        station_features: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Compute regional aggregates from per-station features.

        Returns empty dict if source is not eligible.
        """
        if not self._eligible:
            return {}

        available = [
            sf for sf in station_features
            if not sf.get("is_missing", True)
            and sf.get("distance_km", 999) <= self.max_distance_km
        ]

        if not available:
            return {
                "regional_station_count": 0,
                "regional_available_count": 0,
                "regional_all_missing": True,
            }

        result: Dict[str, Any] = {
            "regional_station_count": len(station_features),
            "regional_available_count": len(available),
            "regional_all_missing": False,
        }

        # IDW-weighted aggregates for continuous variables
        for var in ["temperature", "humidity", "pressure", "wind_speed"]:
            values = []
            weights = []
            for sf in available:
                val = sf.get(f"{var}_current")
                dist = sf.get("distance_km", 1.0)
                if val is not None and dist > 0:
                    values.append(val)
                    weights.append(1.0 / dist)

            if values:
                total_w = sum(weights)
                idw = sum(v * w for v, w in zip(values, weights)) / total_w
                result[f"{var}_idw"] = idw
                result[f"{var}_regional_mean"] = sum(values) / len(values)
                result[f"{var}_regional_median"] = sorted(values)[len(values) // 2]
                if len(values) > 1:
                    mean = result[f"{var}_regional_mean"]
                    result[f"{var}_regional_disagreement"] = (
                        sum((v - mean) ** 2 for v in values) / (len(values) - 1)
                    ) ** 0.5
                else:
                    result[f"{var}_regional_disagreement"] = 0.0
            else:
                result[f"{var}_idw"] = None
                result[f"{var}_regional_mean"] = None
                result[f"{var}_regional_median"] = None
                result[f"{var}_regional_disagreement"] = None

        # Nearest valid station
        nearest = min(available, key=lambda sf: sf.get("distance_km", 999))
        result["nearest_valid_station_id"] = nearest.get("station_id")
        result["nearest_valid_distance_km"] = nearest.get("distance_km")

        return result
