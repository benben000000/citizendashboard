"""
Test Suite for External Source Contract and Cache.

Tests:
  1. Valid source object creation
  2. Missing required field rejection
  3. Invalid checksum format rejection
  4. Invalid revision status rejection
  5. Invalid quality flag rejection
  6. Invalid timestamp rejection
  7. Negative latency rejection
  8. Causal eligibility — eligible case
  9. Causal eligibility — future valid time rejected
  10. Causal eligibility — revised data rejected
  11. Manifest add and freeze
  12. Manifest frozen rejects new objects
  13. Manifest content hash determinism
  14. Manifest causal filtering
  15. Manifest save and load roundtrip
  16. Cache store and verify
  17. Cache append-only enforcement
  18. Cache read_verified requires verification
  19. Cache checksum mismatch detection
  20. Manifest verify against saved
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

from external_source_contract import (
    SourceObjectRecord,
    SourceManifest,
    SourceContractError,
    VALID_REVISION_STATUSES,
    VALID_QUALITY_FLAGS,
)
from external_cache import ExternalCache, CacheError


def _make_valid_object(**overrides) -> dict:
    """Create a valid source object record dict."""
    data = {
        "source_id": "test_source_v1",
        "product_version": "1.0",
        "valid_time_utc": "2026-09-01T06:00:00Z",
        "retrieval_time_utc": "2026-09-01T06:15:00Z",
        "request_parameters": {"lat": 14.5, "lon": 121.0, "radius_km": 50},
        "response_checksum_sha256": hashlib.sha256(b"test_data").hexdigest(),
        "raw_object_path": "/cache/test_source_v1/obj_001.raw",
        "license_decision": "APPROVED_FREE_COMMERCIAL",
        "license_registry_hash": hashlib.sha256(b"registry").hexdigest(),
        "quality_flags": ["good"],
        "latency_seconds": 15.0,
        "revision_status": "original",
        "parser_version": "1.0.0",
    }
    data.update(overrides)
    return data


class TestSourceObjectRecord(unittest.TestCase):
    """Source object record validation tests."""

    # ── Test 1: Valid object ─────────────────────────────────────────
    def test_valid_object_creation(self):
        """A complete valid object must be created without errors."""
        data = _make_valid_object()
        record = SourceObjectRecord(data)
        self.assertEqual(record.source_id, "test_source_v1")
        self.assertEqual(record.revision_status, "original")
        self.assertEqual(record.latency_seconds, 15.0)

    # ── Test 2: Missing required field ───────────────────────────────
    def test_missing_required_field(self):
        """Missing a required field must raise SourceContractError."""
        data = _make_valid_object()
        del data["source_id"]
        with self.assertRaises(SourceContractError) as ctx:
            SourceObjectRecord(data)
        self.assertIn("source_id", ctx.exception.field)

    # ── Test 3: Invalid checksum format ──────────────────────────────
    def test_invalid_checksum_format(self):
        """A non-hex or wrong-length checksum must be rejected."""
        data = _make_valid_object(response_checksum_sha256="not_a_valid_hash")
        with self.assertRaises(SourceContractError) as ctx:
            SourceObjectRecord(data)
        self.assertIn("checksum", ctx.exception.field.lower())

    # ── Test 4: Invalid revision status ──────────────────────────────
    def test_invalid_revision_status(self):
        """An invalid revision status must be rejected."""
        data = _make_valid_object(revision_status="reanalysis")
        with self.assertRaises(SourceContractError) as ctx:
            SourceObjectRecord(data)
        self.assertIn("revision_status", ctx.exception.field)

    # ── Test 5: Invalid quality flag ─────────────────────────────────
    def test_invalid_quality_flag(self):
        """An invalid quality flag must be rejected."""
        data = _make_valid_object(quality_flags=["excellent"])
        with self.assertRaises(SourceContractError) as ctx:
            SourceObjectRecord(data)
        self.assertIn("quality_flags", ctx.exception.field)

    # ── Test 6: Invalid timestamp ────────────────────────────────────
    def test_invalid_timestamp(self):
        """An invalid timestamp must be rejected."""
        data = _make_valid_object(valid_time_utc="not-a-timestamp")
        with self.assertRaises(SourceContractError) as ctx:
            SourceObjectRecord(data)
        self.assertIn("valid_time_utc", ctx.exception.field)

    # ── Test 7: Negative latency ─────────────────────────────────────
    def test_negative_latency_rejected(self):
        """Negative latency must be rejected."""
        data = _make_valid_object(latency_seconds=-5.0)
        with self.assertRaises(SourceContractError) as ctx:
            SourceObjectRecord(data)
        self.assertIn("latency", ctx.exception.field.lower())

    # ── Test 8: Causal eligibility — eligible ────────────────────────
    def test_causal_eligibility_eligible(self):
        """Object with valid_time before issue_time must be eligible."""
        data = _make_valid_object(
            valid_time_utc="2026-09-01T06:00:00Z",
            retrieval_time_utc="2026-09-01T06:10:00Z",
        )
        record = SourceObjectRecord(data)
        issue = datetime(2026, 9, 1, 7, 0, 0, tzinfo=timezone.utc)
        self.assertTrue(record.is_causally_eligible(issue, max_ingestion_delay_seconds=900))

    # ── Test 9: Causal eligibility — future valid time ───────────────
    def test_causal_eligibility_future_rejected(self):
        """Object with valid_time after issue_time must be ineligible."""
        data = _make_valid_object(
            valid_time_utc="2026-09-01T08:00:00Z",
            retrieval_time_utc="2026-09-01T08:05:00Z",
        )
        record = SourceObjectRecord(data)
        issue = datetime(2026, 9, 1, 7, 0, 0, tzinfo=timezone.utc)
        self.assertFalse(record.is_causally_eligible(issue))

    # ── Test 10: Causal eligibility — revised data ───────────────────
    def test_causal_eligibility_revised_rejected(self):
        """Revised data must be ineligible for causal use."""
        data = _make_valid_object(
            revision_status="revised",
            valid_time_utc="2026-09-01T06:00:00Z",
            retrieval_time_utc="2026-09-01T06:10:00Z",
        )
        record = SourceObjectRecord(data)
        issue = datetime(2026, 9, 1, 7, 0, 0, tzinfo=timezone.utc)
        self.assertFalse(record.is_causally_eligible(issue))


class TestSourceManifest(unittest.TestCase):
    """Source manifest tests."""

    # ── Test 11: Manifest add and freeze ─────────────────────────────
    def test_manifest_add_and_freeze(self):
        """Objects can be added before freeze; freeze returns content hash."""
        manifest = SourceManifest()
        obj = _make_valid_object()
        manifest.add_object(obj)
        self.assertEqual(manifest.object_count, 1)
        self.assertFalse(manifest.is_frozen)

        content_hash = manifest.freeze()
        self.assertTrue(manifest.is_frozen)
        self.assertEqual(len(content_hash), 64)

    # ── Test 12: Frozen manifest rejects ─────────────────────────────
    def test_frozen_manifest_rejects_add(self):
        """A frozen manifest must reject new objects."""
        manifest = SourceManifest()
        manifest.freeze()
        with self.assertRaises(SourceContractError):
            manifest.add_object(_make_valid_object())

    # ── Test 13: Content hash determinism ────────────────────────────
    def test_content_hash_determinism(self):
        """Same objects must produce the same content hash regardless of add order."""
        obj_a = _make_valid_object(
            source_id="source_a",
            response_checksum_sha256=hashlib.sha256(b"data_a").hexdigest(),
        )
        obj_b = _make_valid_object(
            source_id="source_b",
            response_checksum_sha256=hashlib.sha256(b"data_b").hexdigest(),
        )

        m1 = SourceManifest()
        m1.add_object(obj_a)
        m1.add_object(obj_b)

        m2 = SourceManifest()
        m2.add_object(obj_b)
        m2.add_object(obj_a)

        self.assertEqual(m1.content_hash, m2.content_hash)

    # ── Test 14: Causal filtering ────────────────────────────────────
    def test_manifest_causal_filtering(self):
        """get_causally_eligible_objects must filter correctly."""
        past_obj = _make_valid_object(
            valid_time_utc="2026-09-01T06:00:00Z",
            retrieval_time_utc="2026-09-01T06:10:00Z",
            response_checksum_sha256=hashlib.sha256(b"past").hexdigest(),
        )
        future_obj = _make_valid_object(
            valid_time_utc="2026-09-01T10:00:00Z",
            retrieval_time_utc="2026-09-01T10:05:00Z",
            response_checksum_sha256=hashlib.sha256(b"future").hexdigest(),
        )

        manifest = SourceManifest()
        manifest.add_object(past_obj)
        manifest.add_object(future_obj)

        issue = datetime(2026, 9, 1, 8, 0, 0, tzinfo=timezone.utc)
        eligible = manifest.get_causally_eligible_objects(issue, max_ingestion_delay_seconds=900)
        self.assertEqual(len(eligible), 1)
        self.assertEqual(eligible[0].valid_time.hour, 6)

    # ── Test 15: Save and load roundtrip ─────────────────────────────
    def test_manifest_save_load_roundtrip(self):
        """Save and load must preserve content hash."""
        manifest = SourceManifest()
        manifest.add_object(_make_valid_object())
        original_hash = manifest.freeze()

        tmpdir = tempfile.mkdtemp()
        try:
            path = os.path.join(tmpdir, "manifest.json")
            manifest.save(path)

            loaded = SourceManifest.load(path)
            self.assertEqual(loaded.content_hash, original_hash)
            self.assertTrue(loaded.is_frozen)
            self.assertEqual(loaded.object_count, 1)
        finally:
            shutil.rmtree(tmpdir)


class TestExternalCache(unittest.TestCase):
    """External cache tests."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    # ── Test 16: Store and verify ────────────────────────────────────
    def test_store_and_verify(self):
        """Storing and verifying an object must succeed."""
        cache = ExternalCache(self.tmpdir)
        data = b"test raw data content"
        record = cache.store("obj_001", "test_source", data)

        self.assertEqual(record["checksum_sha256"], hashlib.sha256(data).hexdigest())
        self.assertFalse(record["verified"])

        # Verify
        self.assertTrue(cache.verify("obj_001"))
        verified_record = cache.get_record("obj_001")
        self.assertTrue(verified_record["verified"])

    # ── Test 17: Append-only enforcement ─────────────────────────────
    def test_append_only_enforcement(self):
        """Storing the same object_id twice must raise CacheError."""
        cache = ExternalCache(self.tmpdir)
        cache.store("obj_001", "test_source", b"data_v1")

        with self.assertRaises(CacheError) as ctx:
            cache.store("obj_001", "test_source", b"data_v2_replacement")
        self.assertIn("already exists", str(ctx.exception).lower())

    # ── Test 18: read_verified requires verification ─────────────────
    def test_read_verified_requires_verification(self):
        """read_verified must fail for unverified objects."""
        cache = ExternalCache(self.tmpdir)
        cache.store("obj_001", "test_source", b"data")

        with self.assertRaises(CacheError) as ctx:
            cache.read_verified("obj_001")
        self.assertIn("not been verified", str(ctx.exception).lower())

        # After verification, read should succeed
        cache.verify("obj_001")
        result = cache.read_verified("obj_001")
        self.assertEqual(result, b"data")

    # ── Test 19: Checksum mismatch detection ─────────────────────────
    def test_checksum_mismatch_detection(self):
        """Modifying a cached file must cause verification failure."""
        cache = ExternalCache(self.tmpdir)
        cache.store("obj_001", "test_source", b"original_data")

        # Tamper with the file
        record = cache.get_record("obj_001")
        with open(record["raw_path"], "wb") as f:
            f.write(b"tampered_data")

        with self.assertRaises(CacheError) as ctx:
            cache.verify("obj_001")
        self.assertIn("mismatch", str(ctx.exception).lower())

    # ── Test 20: Manifest verify against saved ───────────────────────
    def test_manifest_verify_against_saved(self):
        """verify_against_saved must confirm hash match."""
        manifest = SourceManifest()
        manifest.add_object(_make_valid_object())
        manifest.freeze()

        path = os.path.join(self.tmpdir, "manifest.json")
        manifest.save(path)

        # Same manifest should verify
        self.assertTrue(manifest.verify_against_saved(path))

        # Different manifest should not verify
        other = SourceManifest()
        other.add_object(_make_valid_object(
            response_checksum_sha256=hashlib.sha256(b"different").hexdigest()
        ))
        other.freeze()
        self.assertFalse(other.verify_against_saved(path))

    # ── Test 21: Cache list operations ───────────────────────────────
    def test_cache_list_operations(self):
        """List operations must filter correctly."""
        cache = ExternalCache(self.tmpdir)
        cache.store("src_a_001", "source_a", b"data_a1")
        cache.store("src_a_002", "source_a", b"data_a2")
        cache.store("src_b_001", "source_b", b"data_b1")

        self.assertEqual(cache.object_count, 3)
        self.assertEqual(len(cache.list_objects()), 3)
        self.assertEqual(len(cache.list_objects("source_a")), 2)
        self.assertEqual(len(cache.list_objects("source_b")), 1)

        # Initially no verified objects
        self.assertEqual(len(cache.list_verified_objects()), 0)

        # Verify one
        cache.verify("src_a_001")
        self.assertEqual(len(cache.list_verified_objects()), 1)
        self.assertEqual(len(cache.list_verified_objects("source_a")), 1)
        self.assertEqual(len(cache.list_verified_objects("source_b")), 0)

    # ── Test 22: Cache index hash stability ──────────────────────────
    def test_cache_index_hash(self):
        """Cache index hash must be a valid SHA-256."""
        cache = ExternalCache(self.tmpdir)
        cache.store("obj_001", "test_source", b"data")
        h = cache.cache_index_hash
        self.assertEqual(len(h), 64)


if __name__ == "__main__":
    unittest.main()
