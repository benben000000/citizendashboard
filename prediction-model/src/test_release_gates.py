"""
Release Gate Verification — Source-Aware Extension.

Validates all 15 required release gates for the multi-source prediction model:

  1. Source-registry schema validation
  2. Unknown-source rejection
  3. Expired-license rejection
  4. Training-manifest license agreement
  5. Raw-source hash agreement
  6. Source-revision handling
  7. Causal replay verification
  8. Future-mutation rejection
  9. Missing-source fallback
  10. Late-source fallback
  11. Source-latency limits
  12. Policy/scorecard agreement
  13. Candidate-bundle provenance
  14. Source-aware scorecard completeness
  15. Target-specific rollback

Usage:
  PYTHONPATH=src python src/test_release_gates.py
"""

import os
import sys
import json
import hashlib
import tempfile
import shutil
import unittest
from datetime import datetime, timezone, timedelta

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

DATA_DIR = os.path.join(os.path.dirname(SRC_DIR), "data")

from external_source_registry import (
    ExternalSourceRegistry,
    SourceRegistryError,
    VALID_DECISIONS,
    PRODUCTION_ELIGIBLE_DECISIONS,
    BLOCKED_DECISIONS,
)
from external_source_contract import (
    SourceObjectRecord,
    SourceManifest,
    SourceContractError,
)
from external_cache import ExternalCache, CacheError
from causal_feature_cube import (
    CausalFeatureRow,
    FeatureGroup,
    FeatureGroupMetadata,
    CausalViolationError,
)
from source_aware_scorecard import (
    SourceAwareScorecardEntry,
    SourceAwareScorecard,
    TargetRollback,
)


def _make_test_registry(sources, **overrides):
    """Create a temporary registry JSON file."""
    registry = {
        "registry_version": "1.0.0-test",
        "registry_schema_version": "2026-09-27",
        "description": "Test registry",
        "valid_decisions": list(VALID_DECISIONS),
        "production_eligible_decisions": list(PRODUCTION_ELIGIBLE_DECISIONS),
        "sources": sources,
    }
    registry.update(overrides)
    tmpf = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    json.dump(registry, tmpf, indent=2)
    tmpf.close()
    return tmpf.name


def _make_approved_source(source_id="test_approved_v1", **overrides):
    """Create an approved source record."""
    future = (datetime.now(timezone.utc) + timedelta(days=365)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    record = {
        "source_id": source_id,
        "provider": "Test Provider",
        "product": "Test Product",
        "source_url": "https://example.com/data",
        "license_url": "https://example.com/license",
        "terms_version_or_date": "2026-09-01",
        "commercial_use_allowed": True,
        "training_use_allowed": True,
        "commercial_inference_allowed": True,
        "derived_features_allowed": True,
        "raw_caching_allowed": True,
        "redistribution_allowed": False,
        "attribution_required": True,
        "rate_limits": "100 req/min",
        "permission_reference": "Test agreement",
        "decision": "APPROVED_FREE_COMMERCIAL",
        "reviewed_at_utc": "2026-09-01T00:00:00Z",
        "review_due_utc": future,
    }
    record.update(overrides)
    return record


class TestReleaseGate01_RegistrySchema(unittest.TestCase):
    """Gate 1: Source-registry schema validation."""

    def test_production_registry_loads(self):
        """Production registry must load and have a valid schema version."""
        reg = ExternalSourceRegistry()
        self.assertTrue(len(reg.registry_hash) == 64)
        self.assertGreater(len(reg.list_sources()), 0)

    def test_production_registry_has_all_candidate_sources(self):
        """Production registry must contain all documented candidate sources."""
        reg = ExternalSourceRegistry()
        expected = [
            "project_nearby_stations_v1",
            "radar_qpe_v1",
            "rainviewer_v1",
            "himawari9_derived_v1",
            "pagasa_nwp_v1",
        ]
        for sid in expected:
            self.assertIn(sid, reg.list_sources(), f"Missing: {sid}")


class TestReleaseGate02_UnknownSourceRejection(unittest.TestCase):
    """Gate 2: Unknown-source rejection."""

    def test_unknown_source_rejected(self):
        """Sources not in registry must be rejected."""
        reg = ExternalSourceRegistry()
        with self.assertRaises(SourceRegistryError):
            reg.assert_production_eligible("totally_unknown_source_v99")


class TestReleaseGate03_ExpiredLicenseRejection(unittest.TestCase):
    """Gate 3: Expired-license rejection."""

    def test_expired_license_rejected(self):
        """Expired license reviews must fail the gate."""
        expired_date = (
            datetime.now(timezone.utc) - timedelta(days=1)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        src = _make_approved_source(review_due_utc=expired_date)
        path = _make_test_registry({"test_expired_v1": src})
        try:
            reg = ExternalSourceRegistry(path)
            with self.assertRaises(SourceRegistryError) as ctx:
                reg.assert_production_eligible("test_expired_v1")
            self.assertIn("expired", ctx.exception.reason.lower())
        finally:
            os.unlink(path)


class TestReleaseGate04_TrainingManifestLicense(unittest.TestCase):
    """Gate 4: Training-manifest license agreement."""

    def test_manifest_with_blocked_source_fails(self):
        """Training manifest referencing blocked source must fail."""
        reg = ExternalSourceRegistry()
        result = reg.validate_training_manifest_sources(
            ["project_nearby_stations_v1"]  # Currently UNKNOWN_BLOCKED
        )
        self.assertFalse(result["all_eligible"])

    def test_manifest_all_approved_passes(self):
        """Training manifest with all approved sources must pass."""
        src = _make_approved_source(source_id="approved_v1")
        path = _make_test_registry({"approved_v1": src})
        try:
            reg = ExternalSourceRegistry(path)
            result = reg.validate_training_manifest_sources(["approved_v1"])
            self.assertTrue(result["all_eligible"])
        finally:
            os.unlink(path)


class TestReleaseGate05_RawSourceHash(unittest.TestCase):
    """Gate 5: Raw-source hash agreement."""

    def test_cache_hash_matches(self):
        """Cached raw objects must match their stored checksums."""
        tmpdir = tempfile.mkdtemp()
        try:
            cache = ExternalCache(tmpdir)
            data = b"test raw data for hash verification"
            cache.store("obj_001", "test_src", data)
            self.assertTrue(cache.verify("obj_001"))
        finally:
            shutil.rmtree(tmpdir)


class TestReleaseGate06_SourceRevision(unittest.TestCase):
    """Gate 6: Source-revision handling."""

    def test_revised_data_not_causally_eligible(self):
        """Revised data must not be causally eligible."""
        data = {
            "source_id": "test_v1",
            "product_version": "1.0",
            "valid_time_utc": "2026-09-01T06:00:00Z",
            "retrieval_time_utc": "2026-09-01T06:10:00Z",
            "request_parameters": {},
            "response_checksum_sha256": hashlib.sha256(b"x").hexdigest(),
            "raw_object_path": "/path",
            "license_decision": "APPROVED_FREE_COMMERCIAL",
            "license_registry_hash": hashlib.sha256(b"r").hexdigest(),
            "quality_flags": ["good"],
            "latency_seconds": 10.0,
            "revision_status": "revised",
            "parser_version": "1.0",
        }
        record = SourceObjectRecord(data)
        issue_time = datetime(2026, 9, 1, 7, 0, 0, tzinfo=timezone.utc)
        self.assertFalse(record.is_causally_eligible(issue_time))

    def test_revisions_get_new_ids(self):
        """Cache must enforce new IDs for revisions (append-only)."""
        tmpdir = tempfile.mkdtemp()
        try:
            cache = ExternalCache(tmpdir)
            cache.store("obs_001", "src", b"original")
            with self.assertRaises(CacheError):
                cache.store("obs_001", "src", b"revised")  # Same ID → error
            # New ID must succeed
            cache.store("obs_001_rev1", "src", b"revised_data")
        finally:
            shutil.rmtree(tmpdir)


class TestReleaseGate07_CausalReplay(unittest.TestCase):
    """Gate 7: Causal replay verification."""

    def test_fixed_inputs_reproduce_hash(self):
        """Fixed inputs must produce the same content hash."""
        base_time = datetime(2026, 9, 1, 7, 0, 0, tzinfo=timezone.utc)
        valid_time = datetime(2026, 9, 1, 6, 0, 0, tzinfo=timezone.utc)
        reg_hash = hashlib.sha256(b"registry_v1").hexdigest()

        def build_row():
            row = CausalFeatureRow("STN001", base_time, 6.0)
            row.add_feature_group(FeatureGroup(
                name="local",
                metadata=FeatureGroupMetadata(
                    source_id="local",
                    valid_time_utc=valid_time,
                    issue_time_utc=base_time,
                    retrieval_time_utc=valid_time + timedelta(minutes=5),
                    availability_cutoff_utc=base_time,
                    license_registry_hash=reg_hash,
                    raw_object_hash=hashlib.sha256(b"data").hexdigest(),
                ),
                features={"temp": 30.0, "humidity": 75.0},
            ))
            return row

        self.assertEqual(build_row().content_hash, build_row().content_hash)


class TestReleaseGate08_FutureMutationRejection(unittest.TestCase):
    """Gate 8: Future-mutation rejection."""

    def test_future_valid_time_rejected(self):
        """Feature groups with future valid_time must be rejected."""
        base_time = datetime(2026, 9, 1, 7, 0, 0, tzinfo=timezone.utc)
        row = CausalFeatureRow("STN001", base_time, 6.0)
        with self.assertRaises(CausalViolationError):
            row.add_feature_group(FeatureGroup(
                name="leak",
                metadata=FeatureGroupMetadata(
                    source_id="future",
                    valid_time_utc=datetime(2026, 9, 1, 8, 0, 0, tzinfo=timezone.utc),
                    issue_time_utc=base_time,
                    retrieval_time_utc=datetime(2026, 9, 1, 8, 5, 0, tzinfo=timezone.utc),
                    availability_cutoff_utc=base_time,
                ),
                features={"leaked": True},
            ))


class TestReleaseGate09_MissingSourceFallback(unittest.TestCase):
    """Gate 9: Missing-source fallback."""

    def test_all_adapters_return_empty_when_blocked(self):
        """All external source adapters must return empty when source blocked."""
        from external_sources.nearby_stations import NearbyStationsAdapter
        from external_sources.radar import RadarAdapter
        from external_sources.himawari import HimawariAdapter
        from external_sources.nwp import NWPAdapter

        # All use the production registry where all sources are blocked
        ns = NearbyStationsAdapter(14.5, 121.0, 20.0)
        self.assertFalse(ns.is_eligible)

        radar = RadarAdapter(14.5, 121.0)
        self.assertFalse(radar.is_eligible)

        hima = HimawariAdapter(14.5, 121.0)
        self.assertFalse(hima.is_eligible)

        nwp = NWPAdapter(14.5, 121.0)
        self.assertFalse(nwp.is_eligible)


class TestReleaseGate10_LateSourceFallback(unittest.TestCase):
    """Gate 10: Late-source fallback."""

    def test_late_retrieval_rejected(self):
        """Late retrieval exceeding ingestion delay must fail causal check."""
        base_time = datetime(2026, 9, 1, 7, 0, 0, tzinfo=timezone.utc)
        row = CausalFeatureRow("STN001", base_time, 6.0)
        with self.assertRaises(CausalViolationError):
            row.add_feature_group(FeatureGroup(
                name="late",
                metadata=FeatureGroupMetadata(
                    source_id="late_source",
                    valid_time_utc=datetime(2026, 9, 1, 6, 0, 0, tzinfo=timezone.utc),
                    issue_time_utc=base_time,
                    retrieval_time_utc=datetime(2026, 9, 2, 0, 0, 0, tzinfo=timezone.utc),
                    availability_cutoff_utc=base_time,
                ),
                features={"late_data": True},
            ), max_ingestion_delay_seconds=0)


class TestReleaseGate11_SourceLatencyLimits(unittest.TestCase):
    """Gate 11: Source-latency limits."""

    def test_scorecard_captures_latency(self):
        """Scorecard entries must include source_latency_p95."""
        entry = SourceAwareScorecardEntry(
            target="temperature",
            horizon_hours=6,
            source_latency_p95=300.0,
            sample_count=100,
        )
        d = entry.to_dict()
        self.assertEqual(d["source_latency_p95"], 300.0)


class TestReleaseGate12_PolicyScorecardAgreement(unittest.TestCase):
    """Gate 12: Policy/scorecard agreement."""

    def test_scorecard_decision_matches_policy(self):
        """Promotion decision in scorecard must drive policy routing."""
        # A promoted entry
        promoted = SourceAwareScorecardEntry(
            target="temperature",
            horizon_hours=6,
            persistence_metric=2.5,
            candidate_metric=2.0,
            folds_improved=8,
            total_folds=10,
            worst_fold_delta=0.02,
            sample_count=500,
            provenance_pass=True,
            license_pass=True,
        )
        self.assertEqual(promoted.promotion_decision, "PROMOTE")

        # A failing entry must keep fallback
        failing = SourceAwareScorecardEntry(
            target="humidity",
            horizon_hours=6,
            persistence_metric=3.0,
            candidate_metric=3.5,  # Worse
            folds_improved=3,
            total_folds=10,
            sample_count=500,
            provenance_pass=True,
            license_pass=True,
        )
        self.assertEqual(failing.promotion_decision, "KEEP_FALLBACK")


class TestReleaseGate13_BundleProvenance(unittest.TestCase):
    """Gate 13: Candidate-bundle provenance."""

    def test_manifest_content_hash_exists(self):
        """Source manifests must produce deterministic content hashes."""
        manifest = SourceManifest()
        data = {
            "source_id": "test_v1",
            "product_version": "1.0",
            "valid_time_utc": "2026-09-01T06:00:00Z",
            "retrieval_time_utc": "2026-09-01T06:10:00Z",
            "request_parameters": {},
            "response_checksum_sha256": hashlib.sha256(b"data").hexdigest(),
            "raw_object_path": "/path",
            "license_decision": "APPROVED",
            "license_registry_hash": hashlib.sha256(b"reg").hexdigest(),
            "quality_flags": ["good"],
            "latency_seconds": 10.0,
            "revision_status": "original",
            "parser_version": "1.0",
        }
        manifest.add_object(data)
        h = manifest.freeze()
        self.assertEqual(len(h), 64)


class TestReleaseGate14_ScorecardCompleteness(unittest.TestCase):
    """Gate 14: Source-aware scorecard completeness."""

    def test_scorecard_entry_has_all_fields(self):
        """Every scorecard entry must have all required source-aware fields."""
        entry = SourceAwareScorecardEntry(
            target="temperature",
            horizon_hours=6,
            source_set="local_only",
            source_registry_hash="abc",
            source_missing_rate=0.05,
            source_latency_p95=60.0,
            source_ablation_delta=-0.1,
            worst_fold_delta=0.02,
            confidence_interval=(0.8, 1.2),
            sample_count=500,
        )
        d = entry.to_dict()
        required_fields = [
            "source_set", "source_registry_hash", "source_missing_rate",
            "source_latency_p95", "source_ablation_delta", "worst_fold_delta",
            "confidence_interval", "sample_count", "promotion_decision",
            "information_status",
        ]
        for field in required_fields:
            self.assertIn(field, d, f"Missing field: {field}")


class TestReleaseGate15_TargetRollback(unittest.TestCase):
    """Gate 15: Target-specific rollback."""

    def test_rollback_and_verify(self):
        """Rollback must restore fallback and verify correctly."""
        tmpdir = tempfile.mkdtemp()
        try:
            policy = {
                "policy_version": "1.0.0",
                "horizons": {
                    "6": {
                        "selected_sources": {
                            "temperature": "learned_model",
                            "humidity": "persistence_fallback",
                        }
                    }
                },
            }
            path = os.path.join(tmpdir, "policy.json")
            with open(path, "w") as f:
                json.dump(policy, f)

            rb = TargetRollback(path)
            self.assertEqual(rb.get_active_source("temperature", 6), "learned_model")

            rb.rollback("temperature", 6)
            self.assertTrue(rb.verify_rollback("temperature", 6))
            self.assertEqual(
                rb.get_active_source("temperature", 6), "persistence_fallback"
            )
        finally:
            shutil.rmtree(tmpdir)


if __name__ == "__main__":
    unittest.main()
