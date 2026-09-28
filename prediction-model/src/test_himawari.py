"""
Test Suite for Himawari-9 Adapter.

Tests:
  1. Adapter refuses features when source is blocked
  2. Adapter produces features when source is approved
  3. Causal filter rejects future records
  4. Causal filter accepts past records
  5. Missing features have explicit missingness
  6. Daylight visible reflectance included
  7. Nighttime visible reflectance excluded
  8. Cloud-top cooling rate computed from two records
  9. Single-record cooling rate is None
  10. Solar geometry flag propagated
  11. IR brightness temperature recorded
  12. Water vapor temperature recorded
"""

import os
import sys
import json
import tempfile
import unittest
from datetime import datetime, timezone, timedelta

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from external_source_registry import ExternalSourceRegistry
from external_sources.himawari import (
    HimawariAdapter,
    HimawariRecord,
    SOURCE_ID,
)


def _make_approved_registry() -> str:
    """Create a temporary registry with the himawari source approved."""
    future_review = (
        datetime.now(timezone.utc) + timedelta(days=365)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    registry = {
        "registry_version": "1.0.0-test",
        "registry_schema_version": "2026-09-27",
        "description": "Test registry with approved Himawari-9",
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
                "provider": "Test JMA/BoM",
                "product": "Test Himawari-9 derived",
                "source_url": "https://test.example.com/himawari",
                "license_url": "https://test.example.com/himawari-license",
                "terms_version_or_date": "2026-09-01",
                "commercial_use_allowed": True,
                "training_use_allowed": True,
                "commercial_inference_allowed": True,
                "derived_features_allowed": True,
                "raw_caching_allowed": True,
                "redistribution_allowed": False,
                "attribution_required": True,
                "rate_limits": "30 req/min",
                "permission_reference": "Test distribution agreement",
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


def _make_record(
    valid_time=None,
    retrieval_time=None,
    ir_temp=220.0,
    cloud_top_temp=210.0,
    cloud_cover=0.6,
    visible_ref=0.35,
    wv_temp=240.0,
    is_daylight=True,
    quality="good",
    latency=180.0,
):
    """Create a HimawariRecord with defaults."""
    if valid_time is None:
        valid_time = datetime(2026, 9, 1, 6, 0, 0, tzinfo=timezone.utc)
    if retrieval_time is None:
        retrieval_time = valid_time + timedelta(minutes=10)
    return HimawariRecord(
        valid_time_utc=valid_time,
        retrieval_time_utc=retrieval_time,
        ir_brightness_temp_k=ir_temp,
        cloud_top_temp_k=cloud_top_temp,
        cloud_cover_fraction=cloud_cover,
        visible_reflectance=visible_ref,
        water_vapor_brightness_temp_k=wv_temp,
        is_daylight=is_daylight,
        quality_flag=quality,
        product_latency_seconds=latency,
    )


class TestHimawariBlocked(unittest.TestCase):
    """Tests that the adapter refuses features when source is blocked."""

    def test_blocked_adapter_returns_empty(self):
        """Adapter with blocked source must return empty features."""
        adapter = HimawariAdapter(target_lat=14.5, target_lon=121.0)
        self.assertFalse(adapter.is_eligible)
        features = adapter.extract_features(
            [_make_record()],
            issue_time=datetime(2026, 9, 1, 7, 0, 0, tzinfo=timezone.utc),
        )
        self.assertEqual(features, {})


class TestHimawariApproved(unittest.TestCase):
    """Tests with an approved registry for Himawari feature extraction."""

    def setUp(self):
        self.registry_path = _make_approved_registry()
        self.registry = ExternalSourceRegistry(self.registry_path)
        self.adapter = HimawariAdapter(
            target_lat=14.5,
            target_lon=121.0,
            registry=self.registry,
        )

    def tearDown(self):
        os.unlink(self.registry_path)

    def test_adapter_is_eligible(self):
        """Adapter with approved source must report eligible."""
        self.assertTrue(self.adapter.is_eligible)

    def test_causal_filter_rejects_future(self):
        """Future records must not appear in features."""
        issue_time = datetime(2026, 9, 1, 5, 0, 0, tzinfo=timezone.utc)
        future_rec = _make_record(
            valid_time=datetime(2026, 9, 1, 6, 0, 0, tzinfo=timezone.utc),
        )
        features = self.adapter.extract_features([future_rec], issue_time)
        self.assertTrue(features.get("himawari_is_missing", True))

    def test_causal_filter_accepts_past(self):
        """Past records must be accepted."""
        issue_time = datetime(2026, 9, 1, 7, 0, 0, tzinfo=timezone.utc)
        past_rec = _make_record()
        features = self.adapter.extract_features([past_rec], issue_time)
        self.assertFalse(features.get("himawari_is_missing", True))

    def test_missing_features_explicit(self):
        """No causal records must produce missing features."""
        issue_time = datetime(2026, 9, 1, 5, 0, 0, tzinfo=timezone.utc)
        features = self.adapter.extract_features(
            [_make_record()], issue_time
        )
        self.assertTrue(features["himawari_is_missing"])
        self.assertEqual(features["himawari_record_count"], 0)

    def test_daylight_visible_reflectance(self):
        """Daylight records must include visible reflectance."""
        issue_time = datetime(2026, 9, 1, 7, 0, 0, tzinfo=timezone.utc)
        rec = _make_record(is_daylight=True, visible_ref=0.42)
        features = self.adapter.extract_features([rec], issue_time)
        self.assertAlmostEqual(
            features["himawari_visible_reflectance"], 0.42, delta=0.01
        )

    def test_nighttime_visible_reflectance_excluded(self):
        """Nighttime records must have None visible reflectance."""
        issue_time = datetime(2026, 9, 1, 7, 0, 0, tzinfo=timezone.utc)
        rec = _make_record(is_daylight=False)
        features = self.adapter.extract_features([rec], issue_time)
        self.assertIsNone(features["himawari_visible_reflectance"])
        self.assertTrue(features.get("himawari_visible_nighttime_na", False))

    def test_cloud_top_cooling_rate(self):
        """Cooling rate must be computed from two consecutive records."""
        issue_time = datetime(2026, 9, 1, 8, 0, 0, tzinfo=timezone.utc)
        records = [
            _make_record(
                valid_time=datetime(2026, 9, 1, 5, 30, 0, tzinfo=timezone.utc),
                cloud_top_temp=215.0,
            ),
            _make_record(
                valid_time=datetime(2026, 9, 1, 6, 30, 0, tzinfo=timezone.utc),
                cloud_top_temp=210.0,
            ),
        ]
        features = self.adapter.extract_features(records, issue_time)
        # Cooling rate = (210 - 215) / 1.0 hours = -5.0 K/h
        self.assertIsNotNone(features["himawari_cloud_top_cooling_rate_k_per_h"])
        self.assertAlmostEqual(
            features["himawari_cloud_top_cooling_rate_k_per_h"], -5.0, delta=0.1
        )

    def test_single_record_no_cooling_rate(self):
        """Single record must produce None cooling rate."""
        issue_time = datetime(2026, 9, 1, 7, 0, 0, tzinfo=timezone.utc)
        features = self.adapter.extract_features([_make_record()], issue_time)
        self.assertIsNone(features["himawari_cloud_top_cooling_rate_k_per_h"])

    def test_solar_geometry_flag(self):
        """Solar valid flag must match is_daylight."""
        issue_time = datetime(2026, 9, 1, 7, 0, 0, tzinfo=timezone.utc)
        day_rec = _make_record(is_daylight=True)
        features = self.adapter.extract_features([day_rec], issue_time)
        self.assertTrue(features["himawari_solar_valid"])

        night_rec = _make_record(is_daylight=False)
        features2 = self.adapter.extract_features([night_rec], issue_time)
        self.assertFalse(features2["himawari_solar_valid"])

    def test_ir_brightness_temp(self):
        """IR brightness temperature must be recorded."""
        issue_time = datetime(2026, 9, 1, 7, 0, 0, tzinfo=timezone.utc)
        rec = _make_record(ir_temp=225.5)
        features = self.adapter.extract_features([rec], issue_time)
        self.assertAlmostEqual(
            features["himawari_ir_brightness_temp_k"], 225.5, delta=0.01
        )

    def test_water_vapor_temp(self):
        """Water vapor temperature must be recorded."""
        issue_time = datetime(2026, 9, 1, 7, 0, 0, tzinfo=timezone.utc)
        rec = _make_record(wv_temp=242.0)
        features = self.adapter.extract_features([rec], issue_time)
        self.assertAlmostEqual(
            features["himawari_water_vapor_temp_k"], 242.0, delta=0.01
        )


if __name__ == "__main__":
    unittest.main()
