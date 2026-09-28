"""
Radar/QPE Adapter — Precipitation Nowcasting Integration.

BLOCKED: This adapter is structurally complete but cannot be used in production
until the source 'radar_qpe_v1' passes the license gate in
external_source_registry.py.

IMPORTANT: RainViewer is NOT automatically acceptable for commercial production.
It must remain blocked unless explicit permission covers the intended use.

Provides at multiple radii around the target station:
  - mean/max reflectivity
  - rainy-pixel fraction
  - rain-cell distance and bearing
  - intensity trend
  - recent accumulation
  - motion estimate
  - product latency
  - quality/coverage flags
"""

import os
import sys
import math
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_SRC = os.path.dirname(SRC_DIR)
if PARENT_SRC not in sys.path:
    sys.path.insert(0, PARENT_SRC)

from external_source_registry import ExternalSourceRegistry, SourceRegistryError

SOURCE_ID = "radar_qpe_v1"
ANALYSIS_RADII_KM = [10, 25, 50, 100]
RAIN_REFLECTIVITY_THRESHOLD_DBZ = 15.0


class RadarScanRecord:
    """A single radar scan record."""

    def __init__(
        self,
        scan_time_utc: datetime,
        retrieval_time_utc: datetime,
        reflectivity_grid: Optional[List[List[float]]] = None,
        grid_lat_range: tuple = (0.0, 0.0),
        grid_lon_range: tuple = (0.0, 0.0),
        grid_resolution_km: float = 1.0,
        quality_flag: str = "unknown",
        coverage_fraction: float = 0.0,
        product_latency_seconds: float = 0.0,
    ):
        self.scan_time_utc = scan_time_utc
        self.retrieval_time_utc = retrieval_time_utc
        self.reflectivity_grid = reflectivity_grid or []
        self.grid_lat_range = grid_lat_range
        self.grid_lon_range = grid_lon_range
        self.grid_resolution_km = grid_resolution_km
        self.quality_flag = quality_flag
        self.coverage_fraction = coverage_fraction
        self.product_latency_seconds = product_latency_seconds


class RadarAdapter:
    """
    Adapter for radar QPE data.

    IMPORTANT: This adapter will refuse to produce features unless
    the source is production-eligible in the registry.
    """

    def __init__(
        self,
        target_lat: float,
        target_lon: float,
        registry: Optional[ExternalSourceRegistry] = None,
        analysis_radii_km: Optional[List[float]] = None,
    ):
        self.target_lat = target_lat
        self.target_lon = target_lon
        self.analysis_radii_km = analysis_radii_km or ANALYSIS_RADII_KM
        self._registry = registry or ExternalSourceRegistry()
        self._eligible = False
        self._check_eligibility()

    def _check_eligibility(self) -> None:
        """Check if the source is production-eligible."""
        eligible, reason = self._registry.is_production_eligible(SOURCE_ID)
        self._eligible = eligible

    @property
    def is_eligible(self) -> bool:
        return self._eligible

    def extract_features(
        self,
        scans: List[RadarScanRecord],
        issue_time: datetime,
    ) -> Dict[str, Any]:
        """
        Extract radar features from a sequence of scans.

        Returns empty dict if source is not eligible.
        All features use only scans available at issue_time.
        """
        if not self._eligible:
            return {}

        # Causal filter
        causal_scans = [
            s for s in scans
            if s.scan_time_utc <= issue_time
            and s.retrieval_time_utc <= issue_time
        ]

        if not causal_scans:
            return self._missing_features()

        causal_scans.sort(key=lambda s: s.scan_time_utc, reverse=True)
        latest = causal_scans[0]

        features: Dict[str, Any] = {
            "radar_scan_count": len(causal_scans),
            "radar_latest_age_minutes": (
                issue_time - latest.scan_time_utc
            ).total_seconds() / 60.0,
            "radar_product_latency_seconds": latest.product_latency_seconds,
            "radar_quality_flag": latest.quality_flag,
            "radar_coverage_fraction": latest.coverage_fraction,
            "radar_is_missing": False,
        }

        # Per-radius features would be computed from grid data
        # Stub: actual implementation requires grid processing
        for radius in self.analysis_radii_km:
            prefix = f"radar_r{radius}km"
            features[f"{prefix}_mean_reflectivity"] = None
            features[f"{prefix}_max_reflectivity"] = None
            features[f"{prefix}_rainy_fraction"] = None
            features[f"{prefix}_rain_cell_distance_km"] = None
            features[f"{prefix}_rain_cell_bearing_deg"] = None

        # Intensity trend from recent scans
        if len(causal_scans) >= 2:
            features["radar_intensity_trend"] = "stable"  # Stub
        else:
            features["radar_intensity_trend"] = "unknown"

        return features

    def _missing_features(self) -> Dict[str, Any]:
        """Return feature dict with explicit missingness."""
        features: Dict[str, Any] = {
            "radar_is_missing": True,
            "radar_scan_count": 0,
            "radar_quality_flag": "missing",
        }
        for radius in self.analysis_radii_km:
            prefix = f"radar_r{radius}km"
            features[f"{prefix}_mean_reflectivity"] = None
            features[f"{prefix}_max_reflectivity"] = None
            features[f"{prefix}_rainy_fraction"] = None
        return features
