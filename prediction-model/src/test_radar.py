"""
Test Suite for Radar/QPE Adapter.

Tests:
  1. Adapter refuses features when source is blocked
  2. Adapter produces features when source is approved
  3. Causal filter rejects future scans
  4. Causal filter accepts past scans
  5. Missing features have explicit missingness
  6. Per-radius feature keys present
  7. Intensity trend from multiple scans
  8. Single-scan intensity trend unknown
  9. Product latency recorded
  10. Coverage fraction recorded
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
from external_sources.radar import (
    RadarAdapter,
    RadarScanRecord,
    SOURCE_ID,
    ANALYSIS_RADII_KM,
)


def _make_approved_registry() -> str:
    """Create a temporary registry with the radar source approved."""
    future_review = (
        datetime.now(timezone.utc) + timedelta(days=365)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    registry = {
        "registry_version": "1.0.0-test",
        "registry_schema_version": "2026-09-27",
        "description": "Test registry with approved radar",
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
                "provider": "Test Radar Provider",
                "product": "Test QPE Composite",
                "source_url": "https://test.example.com/radar",
                "license_url": "https://test.example.com/radar-license",
                "terms_version_or_date": "2026-09-01",
                "commercial_use_allowed": True,
                "training_use_allowed": True,
                "commercial_inference_allowed": True,
                "derived_features_allowed": True,
                "raw_caching_allowed": True,
                "redistribution_allowed": False,
                "attribution_required": True,
                "rate_limits": "60 req/min",
                "permission_reference": "Test radar agreement",
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


def _make_scan(
    scan_time=None,
    retrieval_time=None,
    quality="good",
    coverage=0.95,
    latency=120.0,
):
    """Create a RadarScanRecord with defaults."""
    if scan_time is None:
        scan_time = datetime(2026, 9, 1, 6, 0, 0, tzinfo=timezone.utc)
    if retrieval_time is None:
        retrieval_time = scan_time + timedelta(minutes=2)
    return RadarScanRecord(
        scan_time_utc=scan_time,
        retrieval_time_utc=retrieval_time,
        quality_flag=quality,
        coverage_fraction=coverage,
        product_latency_seconds=latency,
    )


class TestRadarBlocked(unittest.TestCase):
    """Tests that the adapter refuses features when source is blocked."""

    def test_blocked_adapter_returns_empty(self):
        """Adapter with blocked source must return empty features."""
        adapter = RadarAdapter(target_lat=14.5, target_lon=121.0)
        self.assertFalse(adapter.is_eligible)
        features = adapter.extract_features(
            [_make_scan()],
            issue_time=datetime(2026, 9, 1, 7, 0, 0, tzinfo=timezone.utc),
        )
        self.assertEqual(features, {})


class TestRadarApproved(unittest.TestCase):
    """Tests with an approved registry for radar feature extraction."""

    def setUp(self):
        self.registry_path = _make_approved_registry()
        self.registry = ExternalSourceRegistry(self.registry_path)
        self.adapter = RadarAdapter(
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
        """Future scans must not appear in features."""
        issue_time = datetime(2026, 9, 1, 6, 0, 0, tzinfo=timezone.utc)
        future_scan = _make_scan(
            scan_time=datetime(2026, 9, 1, 7, 0, 0, tzinfo=timezone.utc),
            retrieval_time=datetime(2026, 9, 1, 7, 2, 0, tzinfo=timezone.utc),
        )
        features = self.adapter.extract_features([future_scan], issue_time)
        self.assertTrue(features.get("radar_is_missing", True))

    def test_causal_filter_accepts_past(self):
        """Past scans must be accepted."""
        issue_time = datetime(2026, 9, 1, 7, 0, 0, tzinfo=timezone.utc)
        past_scan = _make_scan(
            scan_time=datetime(2026, 9, 1, 6, 0, 0, tzinfo=timezone.utc),
        )
        features = self.adapter.extract_features([past_scan], issue_time)
        self.assertFalse(features.get("radar_is_missing", True))

    def test_missing_features_explicit(self):
        """No causal scans must produce missing features."""
        issue_time = datetime(2026, 9, 1, 5, 0, 0, tzinfo=timezone.utc)
        future_scan = _make_scan(
            scan_time=datetime(2026, 9, 1, 6, 0, 0, tzinfo=timezone.utc),
        )
        features = self.adapter.extract_features([future_scan], issue_time)
        self.assertTrue(features["radar_is_missing"])
        self.assertEqual(features["radar_scan_count"], 0)

    def test_per_radius_keys_present(self):
        """Feature keys for each analysis radius must be present."""
        issue_time = datetime(2026, 9, 1, 7, 0, 0, tzinfo=timezone.utc)
        scan = _make_scan()
        features = self.adapter.extract_features([scan], issue_time)
        for radius in ANALYSIS_RADII_KM:
            prefix = f"radar_r{radius}km"
            self.assertIn(f"{prefix}_mean_reflectivity", features)
            self.assertIn(f"{prefix}_max_reflectivity", features)
            self.assertIn(f"{prefix}_rainy_fraction", features)

    def test_intensity_trend_multiple_scans(self):
        """Multiple scans must produce a trend value."""
        issue_time = datetime(2026, 9, 1, 7, 0, 0, tzinfo=timezone.utc)
        scans = [
            _make_scan(scan_time=datetime(2026, 9, 1, 5, 50, 0, tzinfo=timezone.utc)),
            _make_scan(scan_time=datetime(2026, 9, 1, 6, 0, 0, tzinfo=timezone.utc)),
        ]
        features = self.adapter.extract_features(scans, issue_time)
        self.assertIn("radar_intensity_trend", features)
        self.assertNotEqual(features["radar_intensity_trend"], "unknown")

    def test_intensity_trend_single_scan(self):
        """Single scan must report unknown intensity trend."""
        issue_time = datetime(2026, 9, 1, 7, 0, 0, tzinfo=timezone.utc)
        features = self.adapter.extract_features([_make_scan()], issue_time)
        self.assertEqual(features["radar_intensity_trend"], "unknown")

    def test_product_latency_recorded(self):
        """Product latency must be captured in features."""
        issue_time = datetime(2026, 9, 1, 7, 0, 0, tzinfo=timezone.utc)
        scan = _make_scan(latency=300.0)
        features = self.adapter.extract_features([scan], issue_time)
        self.assertAlmostEqual(
            features["radar_product_latency_seconds"], 300.0, delta=0.1
        )

    def test_coverage_fraction_recorded(self):
        """Coverage fraction must be captured in features."""
        issue_time = datetime(2026, 9, 1, 7, 0, 0, tzinfo=timezone.utc)
        scan = _make_scan(coverage=0.85)
        features = self.adapter.extract_features([scan], issue_time)
        self.assertAlmostEqual(
            features["radar_coverage_fraction"], 0.85, delta=0.01
        )


if __name__ == "__main__":
    unittest.main()
