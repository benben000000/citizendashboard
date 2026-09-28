"""
Himawari-9 Adapter — Cloud and Convection Feature Integration.

BLOCKED: This adapter is structurally complete but cannot be used in production
until the source 'himawari9_derived_v1' passes the license gate in
external_source_registry.py.

Do not add raw-image deep learning until numeric derived features demonstrate
value and the data-rights record is complete.

Provides numeric derived features:
  - IR brightness-temperature summaries
  - cloud-top cooling rate
  - cloud-cover fraction
  - visible reflectance (daylight-valid only)
  - water-vapor summaries
  - cloud motion
  - spatial gradients
  - solar-geometry flags
  - source quality and latency
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

SOURCE_ID = "himawari9_derived_v1"


class HimawariRecord:
    """A single Himawari-9 derived product record."""

    def __init__(
        self,
        valid_time_utc: datetime,
        retrieval_time_utc: datetime,
        ir_brightness_temp_k: Optional[float] = None,
        cloud_top_temp_k: Optional[float] = None,
        cloud_cover_fraction: Optional[float] = None,
        visible_reflectance: Optional[float] = None,
        water_vapor_brightness_temp_k: Optional[float] = None,
        is_daylight: bool = True,
        quality_flag: str = "unknown",
        product_latency_seconds: float = 0.0,
    ):
        self.valid_time_utc = valid_time_utc
        self.retrieval_time_utc = retrieval_time_utc
        self.ir_brightness_temp_k = ir_brightness_temp_k
        self.cloud_top_temp_k = cloud_top_temp_k
        self.cloud_cover_fraction = cloud_cover_fraction
        self.visible_reflectance = visible_reflectance
        self.water_vapor_brightness_temp_k = water_vapor_brightness_temp_k
        self.is_daylight = is_daylight
        self.quality_flag = quality_flag
        self.product_latency_seconds = product_latency_seconds


class HimawariAdapter:
    """
    Adapter for Himawari-9 derived features.

    IMPORTANT: This adapter will refuse to produce features unless
    the source is production-eligible in the registry.
    """

    def __init__(
        self,
        target_lat: float,
        target_lon: float,
        registry: Optional[ExternalSourceRegistry] = None,
    ):
        self.target_lat = target_lat
        self.target_lon = target_lon
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
        records: List[HimawariRecord],
        issue_time: datetime,
    ) -> Dict[str, Any]:
        """
        Extract Himawari-9 derived features.

        Returns empty dict if source is not eligible.
        All features use only records available at issue_time.
        """
        if not self._eligible:
            return {}

        # Causal filter
        causal = [
            r for r in records
            if r.valid_time_utc <= issue_time
            and r.retrieval_time_utc <= issue_time
        ]

        if not causal:
            return self._missing_features()

        causal.sort(key=lambda r: r.valid_time_utc, reverse=True)
        latest = causal[0]

        features: Dict[str, Any] = {
            "himawari_record_count": len(causal),
            "himawari_latest_age_minutes": (
                issue_time - latest.valid_time_utc
            ).total_seconds() / 60.0,
            "himawari_quality_flag": latest.quality_flag,
            "himawari_is_daylight": latest.is_daylight,
            "himawari_is_missing": False,
        }

        # IR brightness temperature
        features["himawari_ir_brightness_temp_k"] = latest.ir_brightness_temp_k
        features["himawari_cloud_top_temp_k"] = latest.cloud_top_temp_k
        features["himawari_cloud_cover_fraction"] = latest.cloud_cover_fraction
        features["himawari_water_vapor_temp_k"] = latest.water_vapor_brightness_temp_k

        # Visible reflectance (daylight only)
        if latest.is_daylight:
            features["himawari_visible_reflectance"] = latest.visible_reflectance
        else:
            features["himawari_visible_reflectance"] = None
            features["himawari_visible_nighttime_na"] = True

        # Cloud-top cooling rate from recent records
        if len(causal) >= 2:
            prev = causal[1]
            dt_hours = (latest.valid_time_utc - prev.valid_time_utc).total_seconds() / 3600.0
            if (
                dt_hours > 0
                and latest.cloud_top_temp_k is not None
                and prev.cloud_top_temp_k is not None
            ):
                features["himawari_cloud_top_cooling_rate_k_per_h"] = (
                    latest.cloud_top_temp_k - prev.cloud_top_temp_k
                ) / dt_hours
            else:
                features["himawari_cloud_top_cooling_rate_k_per_h"] = None
        else:
            features["himawari_cloud_top_cooling_rate_k_per_h"] = None

        # Solar geometry flags
        features["himawari_solar_valid"] = latest.is_daylight

        return features

    def _missing_features(self) -> Dict[str, Any]:
        """Return feature dict with explicit missingness."""
        return {
            "himawari_is_missing": True,
            "himawari_record_count": 0,
            "himawari_quality_flag": "missing",
        }
