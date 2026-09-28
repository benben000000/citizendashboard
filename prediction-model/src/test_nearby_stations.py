"""
Test Suite for Nearby Stations Adapter.

Tests:
  1. Adapter refuses features when source is blocked
  2. Adapter produces features when source is approved (synthetic registry)
  3. Haversine distance calculation correctness
  4. Bearing calculation correctness
  5. Causal filter rejects future observations
  6. Causal filter accepts past observations
  7. Missing station features have explicit missingness
  8. Rolling statistics computed correctly
  9. Trend computed from causal window
  10. Regional aggregates computed when stations available
  11. Regional aggregates report all-missing when no stations
  12. IDW weighting correctness
  13. Telemetry age computed correctly
  14. Quality flag propagated
  15. Elevation difference computed
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

from external_source_registry import ExternalSourceRegistry, SourceRegistryError
from external_sources.nearby_stations import (
    NearbyStationsAdapter,
    NearbyStationRecord,
    SOURCE_ID,
)


def _make_approved_registry() -> str:
    """Create a temporary registry with the nearby stations source approved."""
    future_review = (
        datetime.now(timezone.utc) + timedelta(days=365)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    registry = {
        "registry_version": "1.0.0-test",
        "registry_schema_version": "2026-09-27",
        "description": "Test registry with approved nearby stations",
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
                "provider": "Test Project Stations",
                "product": "Nearby station telemetry",
                "source_url": "https://test.example.com/stations",
                "license_url": "https://test.example.com/license",
                "terms_version_or_date": "2026-09-01",
                "commercial_use_allowed": True,
                "training_use_allowed": True,
                "commercial_inference_allowed": True,
                "derived_features_allowed": True,
                "raw_caching_allowed": True,
                "redistribution_allowed": False,
                "attribution_required": False,
                "rate_limits": "none",
                "permission_reference": "Test partnership agreement",
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


def _make_station(
    station_id="STN001",
    lat=14.6,
    lon=121.1,
    elev=50.0,
    timestamp=None,
    temperature=30.0,
    humidity=75.0,
    pressure=1013.0,
    wind_speed=5.0,
    wind_direction=180.0,
    precipitation=0.0,
    quality="good",
):
    """Create a NearbyStationRecord with defaults."""
    if timestamp is None:
        timestamp = datetime(2026, 9, 1, 6, 0, 0, tzinfo=timezone.utc)
    return NearbyStationRecord(
        station_id=station_id,
        timestamp_utc=timestamp,
        lat=lat,
        lon=lon,
        elevation_m=elev,
        temperature=temperature,
        humidity=humidity,
        pressure=pressure,
        wind_speed=wind_speed,
        wind_direction=wind_direction,
        precipitation=precipitation,
        quality_flag=quality,
    )


class TestNearbyStationsBlocked(unittest.TestCase):
    """Tests that the adapter refuses features when source is blocked."""

    def test_blocked_adapter_returns_empty(self):
        """Adapter with blocked source must return empty features."""
        # Use the production registry where nearby stations are blocked
        adapter = NearbyStationsAdapter(
            target_lat=14.5, target_lon=121.0, target_elevation_m=20.0
        )
        self.assertFalse(adapter.is_eligible)

        station = _make_station()
        features = adapter.extract_station_features(
            station, [station],
            issue_time=datetime(2026, 9, 1, 7, 0, 0, tzinfo=timezone.utc),
        )
        self.assertEqual(features, {})

    def test_blocked_adapter_regional_empty(self):
        """Regional aggregates must return empty when source is blocked."""
        adapter = NearbyStationsAdapter(
            target_lat=14.5, target_lon=121.0, target_elevation_m=20.0
        )
        result = adapter.extract_regional_aggregates([{"some": "data"}])
        self.assertEqual(result, {})


class TestNearbyStationsApproved(unittest.TestCase):
    """Tests with an approved registry for feature extraction."""

    def setUp(self):
        self.registry_path = _make_approved_registry()
        self.registry = ExternalSourceRegistry(self.registry_path)
        self.adapter = NearbyStationsAdapter(
            target_lat=14.5,
            target_lon=121.0,
            target_elevation_m=20.0,
            registry=self.registry,
        )

    def tearDown(self):
        os.unlink(self.registry_path)

    def test_adapter_is_eligible(self):
        """Adapter with approved source must report eligible."""
        self.assertTrue(self.adapter.is_eligible)

    def test_haversine_known_distance(self):
        """Haversine must return approximately correct distance."""
        # Manila to roughly 1 degree north
        d = NearbyStationsAdapter.haversine_km(14.5, 121.0, 15.5, 121.0)
        self.assertAlmostEqual(d, 111.0, delta=2.0)

    def test_haversine_same_point(self):
        """Haversine of same point must be zero."""
        d = NearbyStationsAdapter.haversine_km(14.5, 121.0, 14.5, 121.0)
        self.assertAlmostEqual(d, 0.0, delta=0.001)

    def test_bearing_north(self):
        """Bearing due north must be approximately 0 degrees."""
        b = NearbyStationsAdapter.bearing_deg(14.5, 121.0, 15.5, 121.0)
        self.assertAlmostEqual(b, 0.0, delta=1.0)

    def test_bearing_east(self):
        """Bearing due east must be approximately 90 degrees."""
        b = NearbyStationsAdapter.bearing_deg(14.5, 121.0, 14.5, 122.0)
        self.assertAlmostEqual(b, 90.0, delta=2.0)

    def test_causal_filter_rejects_future(self):
        """Features must not use observations after issue time."""
        issue_time = datetime(2026, 9, 1, 6, 0, 0, tzinfo=timezone.utc)
        future_station = _make_station(
            timestamp=datetime(2026, 9, 1, 7, 0, 0, tzinfo=timezone.utc)
        )
        features = self.adapter.extract_station_features(
            future_station, [future_station], issue_time
        )
        # Should return missing features since only future data
        self.assertTrue(features.get("is_missing", True))

    def test_causal_filter_accepts_past(self):
        """Features must use observations at or before issue time."""
        issue_time = datetime(2026, 9, 1, 7, 0, 0, tzinfo=timezone.utc)
        past_station = _make_station(
            timestamp=datetime(2026, 9, 1, 6, 0, 0, tzinfo=timezone.utc)
        )
        features = self.adapter.extract_station_features(
            past_station, [past_station], issue_time
        )
        self.assertIn("temperature_current", features)
        self.assertEqual(features["temperature_current"], 30.0)

    def test_missing_station_explicit(self):
        """Missing station features must have is_missing=True and quality=missing."""
        issue_time = datetime(2026, 9, 1, 7, 0, 0, tzinfo=timezone.utc)
        station = _make_station()
        # No causal history → missing
        features = self.adapter.extract_station_features(
            station, [], issue_time
        )
        self.assertTrue(features["is_missing"])
        self.assertEqual(features["quality_flag"], "missing")

    def test_rolling_statistics(self):
        """Rolling mean and std must be computed from causal window."""
        issue_time = datetime(2026, 9, 1, 8, 0, 0, tzinfo=timezone.utc)
        history = [
            _make_station(
                timestamp=datetime(2026, 9, 1, 5, 0, 0, tzinfo=timezone.utc),
                temperature=28.0,
            ),
            _make_station(
                timestamp=datetime(2026, 9, 1, 6, 0, 0, tzinfo=timezone.utc),
                temperature=30.0,
            ),
            _make_station(
                timestamp=datetime(2026, 9, 1, 7, 0, 0, tzinfo=timezone.utc),
                temperature=32.0,
            ),
        ]
        station = history[-1]
        features = self.adapter.extract_station_features(
            station, history, issue_time
        )
        self.assertAlmostEqual(features["temperature_rolling_mean"], 30.0, delta=0.01)
        self.assertGreater(features["temperature_rolling_std"], 0.0)

    def test_trend_computed(self):
        """Trend must be last - first in the causal window."""
        issue_time = datetime(2026, 9, 1, 8, 0, 0, tzinfo=timezone.utc)
        history = [
            _make_station(
                timestamp=datetime(2026, 9, 1, 6, 0, 0, tzinfo=timezone.utc),
                temperature=28.0,
            ),
            _make_station(
                timestamp=datetime(2026, 9, 1, 7, 0, 0, tzinfo=timezone.utc),
                temperature=32.0,
            ),
        ]
        features = self.adapter.extract_station_features(
            history[-1], history, issue_time
        )
        # Trend = most recent - oldest = 32 - 28 = 4
        self.assertAlmostEqual(features["temperature_trend"], 4.0, delta=0.01)

    def test_elevation_difference(self):
        """Elevation difference must be station - target."""
        issue_time = datetime(2026, 9, 1, 8, 0, 0, tzinfo=timezone.utc)
        station = _make_station(elev=100.0)
        features = self.adapter.extract_station_features(
            station, [station], issue_time
        )
        # Target elevation is 20.0, station is 100.0 → diff = 80.0
        self.assertAlmostEqual(features["elevation_diff_m"], 80.0, delta=0.01)

    def test_telemetry_age(self):
        """Telemetry age must reflect time difference from issue time."""
        issue_time = datetime(2026, 9, 1, 9, 0, 0, tzinfo=timezone.utc)
        station = _make_station(
            timestamp=datetime(2026, 9, 1, 6, 0, 0, tzinfo=timezone.utc)
        )
        features = self.adapter.extract_station_features(
            station, [station], issue_time
        )
        self.assertAlmostEqual(features["telemetry_age_hours"], 3.0, delta=0.01)

    def test_quality_flag_propagated(self):
        """Quality flag from latest observation must appear in features."""
        issue_time = datetime(2026, 9, 1, 8, 0, 0, tzinfo=timezone.utc)
        station = _make_station(quality="suspect")
        features = self.adapter.extract_station_features(
            station, [station], issue_time
        )
        self.assertEqual(features["quality_flag"], "suspect")


class TestRegionalAggregates(unittest.TestCase):
    """Regional aggregate computation tests."""

    def setUp(self):
        self.registry_path = _make_approved_registry()
        self.registry = ExternalSourceRegistry(self.registry_path)
        self.adapter = NearbyStationsAdapter(
            target_lat=14.5,
            target_lon=121.0,
            target_elevation_m=20.0,
            registry=self.registry,
            max_distance_km=200.0,
        )

    def tearDown(self):
        os.unlink(self.registry_path)

    def test_regional_all_missing(self):
        """No available stations must report all_missing."""
        features = [
            {"station_id": "A", "is_missing": True, "distance_km": 10.0},
        ]
        result = self.adapter.extract_regional_aggregates(features)
        self.assertTrue(result["regional_all_missing"])
        self.assertEqual(result["regional_available_count"], 0)

    def test_regional_aggregates_computed(self):
        """IDW and mean must be computed from available stations."""
        features = [
            {
                "station_id": "A",
                "is_missing": False,
                "distance_km": 10.0,
                "temperature_current": 30.0,
                "humidity_current": 70.0,
                "pressure_current": 1013.0,
                "wind_speed_current": 5.0,
            },
            {
                "station_id": "B",
                "is_missing": False,
                "distance_km": 20.0,
                "temperature_current": 32.0,
                "humidity_current": 80.0,
                "pressure_current": 1012.0,
                "wind_speed_current": 3.0,
            },
        ]
        result = self.adapter.extract_regional_aggregates(features)
        self.assertFalse(result["regional_all_missing"])
        self.assertEqual(result["regional_available_count"], 2)
        self.assertIsNotNone(result["temperature_idw"])
        self.assertIsNotNone(result["temperature_regional_mean"])
        self.assertAlmostEqual(result["temperature_regional_mean"], 31.0, delta=0.01)

    def test_idw_weights_closer_station(self):
        """IDW-weighted value must be closer to the nearer station."""
        features = [
            {
                "station_id": "CLOSE",
                "is_missing": False,
                "distance_km": 1.0,
                "temperature_current": 20.0,
                "humidity_current": 50.0,
                "pressure_current": 1010.0,
                "wind_speed_current": 1.0,
            },
            {
                "station_id": "FAR",
                "is_missing": False,
                "distance_km": 100.0,
                "temperature_current": 40.0,
                "humidity_current": 90.0,
                "pressure_current": 1020.0,
                "wind_speed_current": 10.0,
            },
        ]
        result = self.adapter.extract_regional_aggregates(features)
        # IDW should be much closer to 20 than 40
        self.assertLess(result["temperature_idw"], 25.0)

    def test_nearest_valid_station(self):
        """Nearest valid station ID must be the closest available station."""
        features = [
            {"station_id": "FAR", "is_missing": False, "distance_km": 50.0,
             "temperature_current": 30.0, "humidity_current": 70.0,
             "pressure_current": 1013.0, "wind_speed_current": 5.0},
            {"station_id": "CLOSE", "is_missing": False, "distance_km": 5.0,
             "temperature_current": 28.0, "humidity_current": 65.0,
             "pressure_current": 1014.0, "wind_speed_current": 3.0},
        ]
        result = self.adapter.extract_regional_aggregates(features)
        self.assertEqual(result["nearest_valid_station_id"], "CLOSE")
        self.assertAlmostEqual(result["nearest_valid_distance_km"], 5.0)


if __name__ == "__main__":
    unittest.main()
