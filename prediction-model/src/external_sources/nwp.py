"""
NWP Adapter — Numerical Weather Prediction Residual Model Integration.

BLOCKED: This adapter is structurally complete but cannot be used in production
until the source 'pagasa_nwp_v1' passes the license gate in
external_source_registry.py.

Model form:
  local forecast = approved NWP forecast + learned local residual

No NWP feature is eligible when only the later analysis or revised value is
available. Training must replay the original operational availability.

Required fields per NWP record:
  - forecast issue time
  - valid time
  - lead time
  - model cycle
  - grid coordinates
  - interpolation method
  - revision status
  - source quality
  - license registry hash
"""

import os
import sys
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_SRC = os.path.dirname(SRC_DIR)
if PARENT_SRC not in sys.path:
    sys.path.insert(0, PARENT_SRC)

from external_source_registry import ExternalSourceRegistry, SourceRegistryError

SOURCE_ID = "pagasa_nwp_v1"

# NWP availability delay from issue time
NWP_AVAILABILITY_DELAY_HOURS = 3.5


class NWPForecastRecord:
    """A single NWP forecast record interpolated to the target station."""

    def __init__(
        self,
        issue_time_utc: datetime,
        valid_time_utc: datetime,
        lead_hours: float,
        model_cycle: str,
        grid_lat: float,
        grid_lon: float,
        interpolation_method: str = "bilinear",
        revision_status: str = "original",
        quality_flag: str = "unknown",
        temperature_k: Optional[float] = None,
        humidity_pct: Optional[float] = None,
        pressure_hpa: Optional[float] = None,
        wind_u_ms: Optional[float] = None,
        wind_v_ms: Optional[float] = None,
        precipitation_prob: Optional[float] = None,
        license_registry_hash: str = "",
    ):
        self.issue_time_utc = issue_time_utc
        self.valid_time_utc = valid_time_utc
        self.lead_hours = lead_hours
        self.model_cycle = model_cycle
        self.grid_lat = grid_lat
        self.grid_lon = grid_lon
        self.interpolation_method = interpolation_method
        self.revision_status = revision_status
        self.quality_flag = quality_flag
        self.temperature_k = temperature_k
        self.humidity_pct = humidity_pct
        self.pressure_hpa = pressure_hpa
        self.wind_u_ms = wind_u_ms
        self.wind_v_ms = wind_v_ms
        self.precipitation_prob = precipitation_prob
        self.license_registry_hash = license_registry_hash


class NWPAdapter:
    """
    Adapter for NWP forecast data as a residual model prior.

    IMPORTANT: This adapter will refuse to produce features unless
    the source is production-eligible in the registry.

    Gate: No NWP feature is eligible when only the later analysis
    or revised value is available.
    """

    def __init__(
        self,
        target_lat: float,
        target_lon: float,
        registry: Optional[ExternalSourceRegistry] = None,
        availability_delay_hours: float = NWP_AVAILABILITY_DELAY_HOURS,
    ):
        self.target_lat = target_lat
        self.target_lon = target_lon
        self.availability_delay_hours = availability_delay_hours
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
        forecasts: List[NWPForecastRecord],
        issue_time: datetime,
        target_horizon_hours: float,
    ) -> Dict[str, Any]:
        """
        Extract NWP features for use as a residual model prior.

        Returns empty dict if source is not eligible.
        Enforces strict causal availability: only NWP forecasts that were
        operationally available at issue_time are eligible.
        """
        if not self._eligible:
            return {}

        # Causal filter:
        # NWP forecast is available only if:
        #   1. issue_time_utc + availability_delay <= current issue_time
        #   2. revision_status is 'original' or 'preliminary' (not 'revised' or 'final')
        #   3. valid_time matches the target forecast period
        availability_cutoff = issue_time - timedelta(
            hours=self.availability_delay_hours
        )

        eligible_forecasts = []
        target_valid_time = issue_time + timedelta(hours=target_horizon_hours)

        for fc in forecasts:
            # Must have been issued before the availability cutoff
            if fc.issue_time_utc > availability_cutoff:
                continue
            # Must not be a revised analysis
            if fc.revision_status in ("revised", "final"):
                continue
            # Valid time should be close to target
            valid_diff_hours = abs(
                (fc.valid_time_utc - target_valid_time).total_seconds() / 3600.0
            )
            if valid_diff_hours <= max(target_horizon_hours * 0.5, 3.0):
                eligible_forecasts.append((fc, valid_diff_hours))

        if not eligible_forecasts:
            return self._missing_features()

        # Select the best matching forecast (closest valid time)
        eligible_forecasts.sort(key=lambda x: x[1])
        best_fc, valid_diff = eligible_forecasts[0]

        features: Dict[str, Any] = {
            "nwp_is_missing": False,
            "nwp_model_cycle": best_fc.model_cycle,
            "nwp_lead_hours": best_fc.lead_hours,
            "nwp_valid_time_diff_hours": valid_diff,
            "nwp_quality_flag": best_fc.quality_flag,
            "nwp_interpolation_method": best_fc.interpolation_method,
        }

        # NWP prior values (for residual model: local = NWP + residual)
        if best_fc.temperature_k is not None:
            features["nwp_temperature_c"] = best_fc.temperature_k - 273.15
        else:
            features["nwp_temperature_c"] = None

        features["nwp_humidity_pct"] = best_fc.humidity_pct
        features["nwp_pressure_hpa"] = best_fc.pressure_hpa

        if best_fc.wind_u_ms is not None and best_fc.wind_v_ms is not None:
            import math
            features["nwp_wind_speed_ms"] = math.sqrt(
                best_fc.wind_u_ms ** 2 + best_fc.wind_v_ms ** 2
            )
            # Meteorological convention is the bearing of the vector FROM which
            # the wind blows: atan2(v, u). Passing (u, v) transposes the angle
            # and misreports every direction by a reflection about the 45 deg
            # axis (e.g. a north wind 0 deg would report as 90 deg).
            features["nwp_wind_direction_deg"] = (
                math.degrees(math.atan2(best_fc.wind_v_ms, best_fc.wind_u_ms)) + 360
            ) % 360
        else:
            features["nwp_wind_speed_ms"] = None
            features["nwp_wind_direction_deg"] = None

        features["nwp_precipitation_prob"] = best_fc.precipitation_prob

        return features

    def _missing_features(self) -> Dict[str, Any]:
        """Return feature dict with explicit missingness."""
        return {
            "nwp_is_missing": True,
            "nwp_quality_flag": "missing",
        }
