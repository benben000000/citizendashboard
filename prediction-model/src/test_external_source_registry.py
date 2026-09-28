"""
Test Suite for External Source Registry — License Gate Verification.

Tests:
  1. Missing source rejection
  2. Unknown source rejection
  3. Research-only source rejection
  4. Personal/not-allowed source rejection
  5. Valid commercial source acceptance
  6. Valid explicit-permission source acceptance
  7. Expired approval rejection
  8. Missing license URL rejection
  9. Contradictory permission fields rejection
  10. Source revocation
  11. Registry hash consistency
  12. Training manifest validation
  13. Registry schema completeness
  14. Blocked source listing
  15. Eligible source listing
"""

import os
import sys
import json
import copy
import tempfile
import unittest
from datetime import datetime, timezone, timedelta

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from external_source_registry import (
    ExternalSourceRegistry,
    SourceRegistryError,
    VALID_DECISIONS,
    PRODUCTION_ELIGIBLE_DECISIONS,
    BLOCKED_DECISIONS,
)


def _make_test_registry(sources: dict, **overrides) -> str:
    """Create a temporary registry JSON and return its path."""
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


def _make_approved_source(
    source_id: str = "test_approved_v1",
    decision: str = "APPROVED_FREE_COMMERCIAL",
    **overrides,
) -> dict:
    """Create a valid approved source record."""
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
        "permission_reference": "Email from provider dated 2026-09-01",
        "decision": decision,
        "reviewed_at_utc": "2026-09-01T00:00:00Z",
        "review_due_utc": (
            datetime.now(timezone.utc) + timedelta(days=365)
        ).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    record.update(overrides)
    return record


def _make_blocked_source(
    source_id: str = "test_blocked_v1",
    decision: str = "UNKNOWN_BLOCKED",
    **overrides,
) -> dict:
    """Create a blocked source record."""
    record = {
        "source_id": source_id,
        "provider": "Blocked Provider",
        "product": "Blocked Product",
        "source_url": "",
        "license_url": "",
        "terms_version_or_date": "",
        "commercial_use_allowed": False,
        "training_use_allowed": False,
        "commercial_inference_allowed": False,
        "derived_features_allowed": False,
        "raw_caching_allowed": False,
        "redistribution_allowed": False,
        "attribution_required": False,
        "rate_limits": "unknown",
        "permission_reference": "",
        "decision": decision,
        "reviewed_at_utc": "",
        "review_due_utc": "",
        "blocked_reason": "Test blocked reason",
    }
    record.update(overrides)
    return record


class TestExternalSourceRegistry(unittest.TestCase):
    """External source registry gate tests."""

    # ── Test 1: Missing source ───────────────────────────────────────
    def test_missing_source_rejected(self):
        """Registry must reject a source that does not exist."""
        path = _make_test_registry({})
        try:
            reg = ExternalSourceRegistry(path)
            with self.assertRaises(SourceRegistryError) as ctx:
                reg.assert_production_eligible("nonexistent_source_v1")
            self.assertIn("not found", ctx.exception.reason.lower())
        finally:
            os.unlink(path)

    # ── Test 2: Unknown source ───────────────────────────────────────
    def test_unknown_source_rejected(self):
        """Registry must reject a source with an unknown decision value."""
        src = _make_approved_source(decision="MAYBE_OK")
        path = _make_test_registry({"test_unknown_v1": src})
        try:
            reg = ExternalSourceRegistry(path)
            with self.assertRaises(SourceRegistryError) as ctx:
                reg.assert_production_eligible("test_unknown_v1")
            self.assertIn("unknown decision", ctx.exception.reason.lower())
        finally:
            os.unlink(path)

    # ── Test 3: Research-only source ─────────────────────────────────
    def test_research_only_source_rejected(self):
        """RESEARCH_ONLY sources must be blocked from production."""
        src = _make_blocked_source(decision="RESEARCH_ONLY")
        path = _make_test_registry({"test_research_v1": src})
        try:
            reg = ExternalSourceRegistry(path)
            with self.assertRaises(SourceRegistryError) as ctx:
                reg.assert_production_eligible("test_research_v1")
            self.assertIn("not production-eligible", ctx.exception.reason.lower())
        finally:
            os.unlink(path)

    # ── Test 4: NOT_ALLOWED source ───────────────────────────────────
    def test_not_allowed_source_rejected(self):
        """NOT_ALLOWED sources must be blocked from production."""
        src = _make_blocked_source(decision="NOT_ALLOWED")
        path = _make_test_registry({"test_denied_v1": src})
        try:
            reg = ExternalSourceRegistry(path)
            with self.assertRaises(SourceRegistryError) as ctx:
                reg.assert_production_eligible("test_denied_v1")
            self.assertIn("not production-eligible", ctx.exception.reason.lower())
        finally:
            os.unlink(path)

    # ── Test 5: Valid commercial source ───────────────────────────────
    def test_approved_free_commercial_accepted(self):
        """APPROVED_FREE_COMMERCIAL with valid fields must pass."""
        src = _make_approved_source(decision="APPROVED_FREE_COMMERCIAL")
        path = _make_test_registry({"test_free_v1": src})
        try:
            reg = ExternalSourceRegistry(path)
            record = reg.assert_production_eligible("test_free_v1")
            self.assertEqual(record["decision"], "APPROVED_FREE_COMMERCIAL")
        finally:
            os.unlink(path)

    # ── Test 6: Valid explicit permission ─────────────────────────────
    def test_approved_explicit_permission_accepted(self):
        """APPROVED_WITH_EXPLICIT_NO_COST_PERMISSION must pass when fields valid."""
        src = _make_approved_source(
            decision="APPROVED_WITH_EXPLICIT_NO_COST_PERMISSION"
        )
        path = _make_test_registry({"test_explicit_v1": src})
        try:
            reg = ExternalSourceRegistry(path)
            record = reg.assert_production_eligible("test_explicit_v1")
            self.assertEqual(
                record["decision"], "APPROVED_WITH_EXPLICIT_NO_COST_PERMISSION"
            )
        finally:
            os.unlink(path)

    # ── Test 7: Expired approval ─────────────────────────────────────
    def test_expired_approval_rejected(self):
        """An approved source with an expired review_due_utc must be rejected."""
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

    # ── Test 8: Missing license URL ──────────────────────────────────
    def test_missing_license_url_rejected(self):
        """An approved source without a license_url must be rejected."""
        src = _make_approved_source(license_url="")
        path = _make_test_registry({"test_no_license_v1": src})
        try:
            reg = ExternalSourceRegistry(path)
            with self.assertRaises(SourceRegistryError) as ctx:
                reg.assert_production_eligible("test_no_license_v1")
            self.assertIn("license_url", ctx.exception.reason.lower())
        finally:
            os.unlink(path)

    # ── Test 9: Contradictory permission fields ──────────────────────
    def test_contradictory_fields_rejected(self):
        """Approved decision with training_use_allowed=false must be rejected."""
        src = _make_approved_source(training_use_allowed=False)
        path = _make_test_registry({"test_contradictory_v1": src})
        try:
            reg = ExternalSourceRegistry(path)
            with self.assertRaises(SourceRegistryError) as ctx:
                reg.assert_production_eligible("test_contradictory_v1")
            self.assertIn("contradictory", ctx.exception.reason.lower())
        finally:
            os.unlink(path)

    def test_contradictory_commercial_inference_rejected(self):
        """Approved decision with commercial_inference_allowed=false must be rejected."""
        src = _make_approved_source(commercial_inference_allowed=False)
        path = _make_test_registry({"test_contradictory_inf_v1": src})
        try:
            reg = ExternalSourceRegistry(path)
            with self.assertRaises(SourceRegistryError) as ctx:
                reg.assert_production_eligible("test_contradictory_inf_v1")
            self.assertIn("contradictory", ctx.exception.reason.lower())
        finally:
            os.unlink(path)

    def test_contradictory_commercial_use_rejected(self):
        """Approved decision with commercial_use_allowed=false must be rejected."""
        src = _make_approved_source(commercial_use_allowed=False)
        path = _make_test_registry({"test_contradictory_com_v1": src})
        try:
            reg = ExternalSourceRegistry(path)
            with self.assertRaises(SourceRegistryError) as ctx:
                reg.assert_production_eligible("test_contradictory_com_v1")
            self.assertIn("contradictory", ctx.exception.reason.lower())
        finally:
            os.unlink(path)

    # ── Test 10: Source revocation ────────────────────────────────────
    def test_source_revocation(self):
        """Revoking a source must make it fail the production gate."""
        src = _make_approved_source()
        path = _make_test_registry({"test_revoke_v1": src})
        try:
            reg = ExternalSourceRegistry(path)
            # Initially eligible
            record = reg.assert_production_eligible("test_revoke_v1")
            self.assertIsNotNone(record)

            # Revoke
            reg.revoke_source("test_revoke_v1", reason="Latency breach")

            # Must now be blocked
            with self.assertRaises(SourceRegistryError) as ctx:
                reg.assert_production_eligible("test_revoke_v1")
            self.assertIn("not production-eligible", ctx.exception.reason.lower())
        finally:
            os.unlink(path)

    # ── Test 11: Registry hash consistency ───────────────────────────
    def test_registry_hash_consistent(self):
        """Two loads of the same file must produce the same hash."""
        src = _make_approved_source()
        path = _make_test_registry({"test_hash_v1": src})
        try:
            reg1 = ExternalSourceRegistry(path)
            reg2 = ExternalSourceRegistry(path)
            self.assertEqual(reg1.registry_hash, reg2.registry_hash)
            self.assertTrue(len(reg1.registry_hash) == 64)
        finally:
            os.unlink(path)

    # ── Test 12: Training manifest validation ────────────────────────
    def test_training_manifest_validation(self):
        """validate_training_manifest_sources must flag ineligible sources."""
        approved = _make_approved_source(source_id="approved_v1")
        blocked = _make_blocked_source(source_id="blocked_v1")
        path = _make_test_registry({
            "approved_v1": approved,
            "blocked_v1": blocked,
        })
        try:
            reg = ExternalSourceRegistry(path)
            result = reg.validate_training_manifest_sources(
                ["approved_v1", "blocked_v1", "missing_v1"]
            )
            self.assertFalse(result["all_eligible"])
            self.assertTrue(result["sources"]["approved_v1"]["eligible"])
            self.assertFalse(result["sources"]["blocked_v1"]["eligible"])
            self.assertFalse(result["sources"]["missing_v1"]["eligible"])
        finally:
            os.unlink(path)

    # ── Test 13: Registry schema completeness ────────────────────────
    def test_production_registry_schema(self):
        """The production registry must contain all candidate sources from the plan."""
        reg = ExternalSourceRegistry()
        expected_sources = [
            "project_nearby_stations_v1",
            "radar_qpe_v1",
            "rainviewer_v1",
            "himawari9_derived_v1",
            "pagasa_nwp_v1",
        ]
        for sid in expected_sources:
            self.assertIn(sid, reg.list_sources(), f"Missing source: {sid}")

    # ── Test 14: Blocked source listing ──────────────────────────────
    def test_production_registry_all_blocked(self):
        """All sources in the production registry must be blocked (no approved sources yet)."""
        reg = ExternalSourceRegistry()
        blocked = reg.list_blocked_sources()
        self.assertEqual(len(blocked), 5, "Expected 5 blocked sources")
        eligible = reg.list_eligible_sources()
        self.assertEqual(len(eligible), 0, "Expected 0 eligible sources")

    # ── Test 15: Eligible source listing ─────────────────────────────
    def test_eligible_source_listing(self):
        """Eligible list must include only properly approved sources."""
        approved = _make_approved_source(source_id="good_v1")
        blocked = _make_blocked_source(source_id="bad_v1")
        path = _make_test_registry({"good_v1": approved, "bad_v1": blocked})
        try:
            reg = ExternalSourceRegistry(path)
            eligible = reg.list_eligible_sources()
            self.assertEqual(eligible, ["good_v1"])
            blocked_list = reg.list_blocked_sources()
            self.assertEqual(blocked_list, ["bad_v1"])
        finally:
            os.unlink(path)

    # ── Test 16: PENDING_REVIEW rejection ────────────────────────────
    def test_pending_review_source_rejected(self):
        """PENDING_REVIEW sources must be blocked."""
        src = _make_blocked_source(decision="PENDING_REVIEW")
        path = _make_test_registry({"test_pending_v1": src})
        try:
            reg = ExternalSourceRegistry(path)
            with self.assertRaises(SourceRegistryError):
                reg.assert_production_eligible("test_pending_v1")
        finally:
            os.unlink(path)

    # ── Test 17: UNKNOWN_BLOCKED rejection ───────────────────────────
    def test_unknown_blocked_source_rejected(self):
        """UNKNOWN_BLOCKED sources must be blocked."""
        src = _make_blocked_source(decision="UNKNOWN_BLOCKED")
        path = _make_test_registry({"test_unk_v1": src})
        try:
            reg = ExternalSourceRegistry(path)
            with self.assertRaises(SourceRegistryError):
                reg.assert_production_eligible("test_unk_v1")
        finally:
            os.unlink(path)

    # ── Test 18: Missing registry file ───────────────────────────────
    def test_missing_registry_file_raises(self):
        """Loading a non-existent registry must raise FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            ExternalSourceRegistry("/nonexistent/path/registry.json")

    # ── Test 19: Provenance record ───────────────────────────────────
    def test_provenance_record(self):
        """Provenance record must contain required fields."""
        reg = ExternalSourceRegistry()
        prov = reg.get_provenance_record()
        self.assertIn("registry_hash", prov)
        self.assertIn("registry_version", prov)
        self.assertIn("source_count", prov)
        self.assertIn("eligible_sources", prov)
        self.assertIn("blocked_sources", prov)
        self.assertEqual(prov["source_count"], 5)

    # ── Test 20: is_production_eligible returns tuple ─────────────────
    def test_is_production_eligible_returns_tuple(self):
        """is_production_eligible must return (bool, reason) tuple."""
        reg = ExternalSourceRegistry()
        eligible, reason = reg.is_production_eligible("project_nearby_stations_v1")
        self.assertFalse(eligible)
        self.assertIsInstance(reason, str)
        self.assertGreater(len(reason), 0)

    # ── Test 21: Revoke unknown source raises ────────────────────────
    def test_revoke_unknown_source_raises(self):
        """Revoking an unknown source must raise SourceRegistryError."""
        reg = ExternalSourceRegistry()
        with self.assertRaises(SourceRegistryError):
            reg.revoke_source("totally_unknown_v99")


if __name__ == "__main__":
    unittest.main()
