"""
Test Suite for NWP Adapter.

Tests:
  1. Adapter refuses features when source is blocked
  2. Adapter produces features when source is approved
  3. Causal filter rejects future-issued forecasts
  4. Causal filter rejects revised analyses
  5. Causal filter accepts operationally available forecasts
  6. Missing features have explicit missingness
  7. NWP temperature converted from Kelvin to Celsius
  8. NWP wind components converted to speed and direction
  9. Best-match forecast selected by valid time proximity
  10. Availability delay enforcement
  11. Precipitation probability propagated
  12. Model cycle recorded
"""

import os
import sys
import json
import math
import tempfile
import unittest
from datetime import datetime, timezone, timedelta

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from external_source_registry import ExternalSourceRegistry
from external_sources.nwp import (
    NWPAdapter,
    NWPForecastRecord,
    SOURCE_ID,
    NWP_AVAILABILITY_DELAY_HOURS,
)


def _make_approved_registry() -> str:
    """Create a temporary registry with the NWP source approved."""
    future_review = (
        datetime.now(timezone.utc) + timedelta(days=365)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    registry = {
        "registry_version": "1.0.0-test",
        "registry_schema_version": "2026-09-27",
        "description": "Test registry with approved NWP",
        "valid_decisions": [
            "APPROVED_FREE_COMMERCIAL",
            "APPROVED_WITH_EXPLICIT_NO_COST_PERMISSION",
            "UNKNOWN_BLOCKED",
            "PENDING_REVIEW",
            "RESEARCH_ONLY",
            "NOT_ALLOWED",
        ],
        "production_eligible_decisions": [
            "APPROVED_FREE_COMMERCIAL",
            "APPROVED_WITH_EXPLICIT_NO_COST_PERMISSION",
        ],
        "sources": {
            SOURCE_ID: {
                "source_id": SOURCE_ID,
                "provider": "Test NWP Provider",
                "product": "Test NWP output",
                "source_url": "https://test.example.com/nwp",
                "license_url": "https://test.example.com/nwp-license",
                "terms_version_or_date": "2026-09-01",
                "commercial_use_allowed": True,
                "training_use_allowed": True,
                "commercial_inference_allowed": True,
                "derived_features_allowed": True,
                "raw_caching_allowed": True,
                "redistribution_allowed": False,
                "attribution_required": True,
                "rate_limits": "20 req/min",
                "permission_reference": "Test NWP agreement",
                "decision": "APPROVED_FREE_COMMERCIAL",
                "reviewed_at_utc": "2026-09-01T00:00:00Z",
                "review_due_utc": future_review,
            }
        },
    }
    tmpf = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    json.dump(registry, tmpf, indent=2)
    tmpf.close()
    return tmpf.name


def _make_forecast(
    issue_time=None,
    valid_time=None,
    lead_hours=6.0,
    model_cycle="00Z",
    revision_status="original",
    temperature_k=303.15,
    humidity_pct=75.0,
    pressure_hpa=1013.0,
    wind_u_ms=3.0,
    wind_v_ms=4.0,
    precip_prob=0.3,
    quality="good",
):
    """Create a NWPForecastRecord with defaults."""
    if issue_time is None:
        issue_time = datetime(2026, 9, 1, 0, 0, 0, tzinfo=timezone.utc)
    if valid_time is None:
        valid_time = issue_time + timedelta(hours=lead_hours)
    return NWPForecastRecord(
        issue_time_utc=issue_time,
        valid_time_utc=valid_time,
        lead_hours=lead_hours,
        model_cycle=model_cycle,
        grid_lat=14.5,
        grid_lon=121.0,
        revision_status=revision_status,
        quality_flag=quality,
        temperature_k=temperature_k,
        humidity_pct=humidity_pct,
        pressure_hpa=pressure_hpa,
        wind_u_ms=wind_u_ms,
        wind_v_ms=wind_v_ms,
        precipitation_prob=precip_prob,
    )


class TestNWPBlocked(unittest.TestCase):
    """Tests that the adapter refuses features when source is blocked."""

    def test_blocked_adapter_returns_empty(self):
        """Adapter with blocked source must return empty features."""
        adapter = NWPAdapter(target_lat=14.5, target_lon=121.0)
        self.assertFalse(adapter.is_eligible)
        features = adapter.extract_features(
            [_make_forecast()],
            issue_time=datetime(2026, 9, 1, 12, 0, 0, tzinfo=timezone.utc),
            target_horizon_hours=6.0,
        )
        self.assertEqual(features, {})


class TestNWPApproved(unittest.TestCase):
    """Tests with an approved registry for NWP feature extraction."""

    def setUp(self):
        self.registry_path = _make_approved_registry()
        self.registry = ExternalSourceRegistry(self.registry_path)
        self.adapter = NWPAdapter(
            target_lat=14.5,
            target_lon=121.0,
            registry=self.registry,
            availability_delay_hours=3.5,
        )

    def tearDown(self):
        os.unlink(self.registry_path)

    def test_adapter_is_eligible(self):
        """Adapter with approved source must report eligible."""
        self.assertTrue(self.adapter.is_eligible)

    def test_causal_filter_rejects_future_issued(self):
        """Forecasts issued after the availability cutoff must be rejected."""
        # Issue time is 06:00, availability delay is 3.5h
        # So only forecasts issued before 02:30 are available
        issue_time = datetime(2026, 9, 1, 6, 0, 0, tzinfo=timezone.utc)
        future_fc = _make_forecast(
            issue_time=datetime(2026, 9, 1, 3, 0, 0, tzinfo=timezone.utc),
            lead_hours=6.0,
        )
        features = self.adapter.extract_features(
            [future_fc], issue_time, target_horizon_hours=6.0
        )
        # Issued at 03:00 > cutoff 02:30 → rejected
        self.assertTrue(features.get("nwp_is_missing", True))

    def test_causal_filter_rejects_revised(self):
        """Revised analyses must be excluded."""
        issue_time = datetime(2026, 9, 1, 12, 0, 0, tzinfo=timezone.utc)
        revised_fc = _make_forecast(
            issue_time=datetime(2026, 9, 1, 0, 0, 0, tzinfo=timezone.utc),
            revision_status="revised",
        )
        features = self.adapter.extract_features(
            [revised_fc], issue_time, target_horizon_hours=6.0
        )
        self.assertTrue(features.get("nwp_is_missing", True))

    def test_causal_filter_accepts_available(self):
        """Operationally available original forecasts must be accepted."""
        # Issue time is 12:00, delay is 3.5h, so cutoff is 08:30
        # Forecast issued at 00:00 (well before 08:30) should pass
        # target_valid_time = issue_time + target_horizon = 18:00 UTC
        issue_time = datetime(2026, 9, 1, 12, 0, 0, tzinfo=timezone.utc)
        fc = _make_forecast(
            issue_time=datetime(2026, 9, 1, 0, 0, 0, tzinfo=timezone.utc),
            lead_hours=18.0,
            valid_time=datetime(2026, 9, 1, 18, 0, 0, tzinfo=timezone.utc),
        )
        features = self.adapter.extract_features(
            [fc], issue_time, target_horizon_hours=6.0
        )
        self.assertFalse(features.get("nwp_is_missing", True))

    def test_missing_features_explicit(self):
        """No eligible forecasts must produce missing features."""
        issue_time = datetime(2026, 9, 1, 1, 0, 0, tzinfo=timezone.utc)
        features = self.adapter.extract_features(
            [_make_forecast()], issue_time, target_horizon_hours=6.0
        )
        self.assertTrue(features["nwp_is_missing"])

    def test_temperature_kelvin_to_celsius(self):
        """NWP temperature must be converted from Kelvin to Celsius."""
        issue_time = datetime(2026, 9, 1, 12, 0, 0, tzinfo=timezone.utc)
        fc = _make_forecast(
            issue_time=datetime(2026, 9, 1, 0, 0, 0, tzinfo=timezone.utc),
            temperature_k=303.15,
            lead_hours=18.0,
            valid_time=datetime(2026, 9, 1, 18, 0, 0, tzinfo=timezone.utc),
        )
        features = self.adapter.extract_features(
            [fc], issue_time, target_horizon_hours=6.0
        )
        self.assertAlmostEqual(features["nwp_temperature_c"], 30.0, delta=0.1)

    def test_wind_components_to_speed_direction(self):
        """Wind u/v must be converted to speed and direction."""
        issue_time = datetime(2026, 9, 1, 12, 0, 0, tzinfo=timezone.utc)
        fc = _make_forecast(
            issue_time=datetime(2026, 9, 1, 0, 0, 0, tzinfo=timezone.utc),
            wind_u_ms=3.0,
            wind_v_ms=4.0,
            lead_hours=18.0,
            valid_time=datetime(2026, 9, 1, 18, 0, 0, tzinfo=timezone.utc),
        )
        features = self.adapter.extract_features(
            [fc], issue_time, target_horizon_hours=6.0
        )
        self.assertAlmostEqual(features["nwp_wind_speed_ms"], 5.0, delta=0.01)
        self.assertIsNotNone(features["nwp_wind_direction_deg"])
        self.assertGreaterEqual(features["nwp_wind_direction_deg"], 0.0)
        self.assertLess(features["nwp_wind_direction_deg"], 360.0)

    def test_best_match_by_valid_time(self):
        """Closest valid-time forecast must be selected."""
        issue_time = datetime(2026, 9, 1, 12, 0, 0, tzinfo=timezone.utc)
        fc_far = _make_forecast(
            issue_time=datetime(2026, 9, 1, 0, 0, 0, tzinfo=timezone.utc),
            lead_hours=24.0,
            valid_time=datetime(2026, 9, 2, 0, 0, 0, tzinfo=timezone.utc),
            temperature_k=290.0,
        )
        fc_close = _make_forecast(
            issue_time=datetime(2026, 9, 1, 0, 0, 0, tzinfo=timezone.utc),
            lead_hours=18.0,
            valid_time=datetime(2026, 9, 1, 18, 0, 0, tzinfo=timezone.utc),
            temperature_k=300.0,
        )
        # Target horizon: 6h from issue_time = 18:00 UTC
        features = self.adapter.extract_features(
            [fc_far, fc_close], issue_time, target_horizon_hours=6.0
        )
        # fc_close has valid_time 18:00 which exactly matches target
        self.assertAlmostEqual(features["nwp_temperature_c"], 300.0 - 273.15, delta=0.1)

    def test_precipitation_prob_propagated(self):
        """Precipitation probability must be in features."""
        issue_time = datetime(2026, 9, 1, 12, 0, 0, tzinfo=timezone.utc)
        fc = _make_forecast(
            issue_time=datetime(2026, 9, 1, 0, 0, 0, tzinfo=timezone.utc),
            precip_prob=0.65,
            lead_hours=18.0,
            valid_time=datetime(2026, 9, 1, 18, 0, 0, tzinfo=timezone.utc),
        )
        features = self.adapter.extract_features(
            [fc], issue_time, target_horizon_hours=6.0
        )
        self.assertAlmostEqual(
            features["nwp_precipitation_prob"], 0.65, delta=0.01
        )

    def test_model_cycle_recorded(self):
        """Model cycle must appear in features."""
        issue_time = datetime(2026, 9, 1, 12, 0, 0, tzinfo=timezone.utc)
        fc = _make_forecast(
            issue_time=datetime(2026, 9, 1, 0, 0, 0, tzinfo=timezone.utc),
            model_cycle="00Z",
            lead_hours=18.0,
            valid_time=datetime(2026, 9, 1, 18, 0, 0, tzinfo=timezone.utc),
        )
        features = self.adapter.extract_features(
            [fc], issue_time, target_horizon_hours=6.0
        )
        self.assertEqual(features["nwp_model_cycle"], "00Z")


if __name__ == "__main__":
    unittest.main()
