"""
Test Suite for Source-Aware Scorecard and Target Rollback.

Tests:
  1. Entry that beats persistence → PROMOTE
  2. Entry that fails persistence → KEEP_FALLBACK
  3. Insufficient sample count → INFORMATION_LIMITED
  4. Leakage detected → KEEP_FALLBACK
  5. Provenance failure → KEEP_FALLBACK
  6. License failure → KEEP_FALLBACK
  7. Worst fold degradation → KEEP_FALLBACK
  8. Calibration degradation → KEEP_FALLBACK
  9. High missing rate → KEEP_FALLBACK
  10. Scorecard serialization roundtrip
  11. Promotion summary counts
  12. Rollback single target
  13. Rollback all targets
  14. Verify rollback state
  15. Information status labels
"""

import os
import sys
import json
import tempfile
import shutil
import unittest
from datetime import datetime, timezone

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from source_aware_scorecard import (
    SourceAwareScorecardEntry,
    SourceAwareScorecard,
    TargetRollback,
)


def _make_promoted_entry(**overrides):
    """Create an entry that should pass all promotion criteria."""
    defaults = dict(
        target="temperature",
        horizon_hours=6,
        source_set="local_only",
        source_registry_hash="abc123",
        source_missing_rate=0.05,
        source_latency_p95=60.0,
        source_ablation_delta=-0.1,
        worst_fold_delta=0.02,
        confidence_interval=(0.8, 1.2),
        sample_count=500,
        persistence_metric=2.5,
        candidate_metric=2.0,  # Lower MAE = better
        metric_name="MAE",
        folds_improved=8,
        total_folds=10,
        worst_fold_metric=2.8,
        calibration_delta=0.01,
        leakage_detected=False,
        provenance_pass=True,
        license_pass=True,
    )
    defaults.update(overrides)
    return SourceAwareScorecardEntry(**defaults)


def _make_failing_entry(**overrides):
    """Create an entry that should fail promotion."""
    defaults = dict(
        target="humidity",
        horizon_hours=6,
        source_set="local_only",
        source_registry_hash="abc123",
        source_missing_rate=0.05,
        source_latency_p95=60.0,
        source_ablation_delta=0.1,
        worst_fold_delta=0.02,
        confidence_interval=(2.0, 3.0),
        sample_count=500,
        persistence_metric=3.0,
        candidate_metric=3.5,  # Worse than persistence
        metric_name="MAE",
        folds_improved=3,
        total_folds=10,
        worst_fold_metric=4.0,
        calibration_delta=0.01,
        leakage_detected=False,
        provenance_pass=True,
        license_pass=True,
    )
    defaults.update(overrides)
    return SourceAwareScorecardEntry(**defaults)


class TestPromotionDecision(unittest.TestCase):
    """Promotion decision tests."""

    def test_promoted_when_beats_persistence(self):
        """Entry beating persistence in all criteria must be PROMOTE."""
        entry = _make_promoted_entry()
        decision, reason = entry.evaluate_promotion()
        self.assertEqual(decision, "PROMOTE")
        self.assertIn("criteria met", reason)

    def test_fallback_when_fails_persistence(self):
        """Entry not beating persistence must be KEEP_FALLBACK."""
        entry = _make_failing_entry()
        decision, reason = entry.evaluate_promotion()
        self.assertEqual(decision, "KEEP_FALLBACK")
        self.assertIn("does not beat", reason)

    def test_information_limited_low_samples(self):
        """Entry with insufficient samples must be INFORMATION_LIMITED."""
        entry = _make_failing_entry(sample_count=10)
        decision, reason = entry.evaluate_promotion()
        self.assertEqual(decision, "INFORMATION_LIMITED")
        self.assertIn("Insufficient evidence", reason)

    def test_fallback_on_leakage(self):
        """Leakage detected must prevent promotion."""
        entry = _make_promoted_entry(leakage_detected=True)
        decision, reason = entry.evaluate_promotion()
        self.assertEqual(decision, "KEEP_FALLBACK")
        self.assertIn("Leakage", reason)

    def test_fallback_on_provenance_failure(self):
        """Provenance failure must prevent promotion."""
        entry = _make_promoted_entry(provenance_pass=False)
        decision, reason = entry.evaluate_promotion()
        self.assertEqual(decision, "KEEP_FALLBACK")
        self.assertIn("Provenance", reason)

    def test_fallback_on_license_failure(self):
        """License gate failure must prevent promotion."""
        entry = _make_promoted_entry(license_pass=False)
        decision, reason = entry.evaluate_promotion()
        self.assertEqual(decision, "KEEP_FALLBACK")
        self.assertIn("License", reason)

    def test_fallback_on_worst_fold_degradation(self):
        """Worst fold degradation exceeding threshold must prevent promotion."""
        entry = _make_promoted_entry(worst_fold_delta=0.20)
        decision, reason = entry.evaluate_promotion()
        self.assertEqual(decision, "KEEP_FALLBACK")
        self.assertIn("Worst fold", reason)

    def test_fallback_on_calibration_degradation(self):
        """Calibration degradation must prevent promotion."""
        entry = _make_promoted_entry(calibration_delta=0.10)
        decision, reason = entry.evaluate_promotion()
        self.assertEqual(decision, "KEEP_FALLBACK")
        self.assertIn("Calibration", reason)

    def test_fallback_on_high_missing_rate(self):
        """High source missing rate must prevent promotion."""
        entry = _make_promoted_entry(source_missing_rate=0.50)
        decision, reason = entry.evaluate_promotion()
        self.assertEqual(decision, "KEEP_FALLBACK")
        self.assertIn("missing rate", reason)


class TestSourceAwareScorecard(unittest.TestCase):
    """Scorecard collection tests."""

    def test_serialization_roundtrip(self):
        """Save and load must preserve entries."""
        sc = SourceAwareScorecard()
        sc.add_entry(_make_promoted_entry())
        sc.add_entry(_make_failing_entry())

        tmpdir = tempfile.mkdtemp()
        try:
            path = os.path.join(tmpdir, "scorecard.json")
            sc.save(path)

            loaded = SourceAwareScorecard.load(path)
            self.assertEqual(
                len(loaded.to_dict()["entries"]),
                len(sc.to_dict()["entries"]),
            )
        finally:
            shutil.rmtree(tmpdir)

    def test_promotion_summary(self):
        """Promotion summary must categorize entries correctly."""
        sc = SourceAwareScorecard()
        sc.add_entry(_make_promoted_entry())
        sc.add_entry(_make_failing_entry())
        sc.add_entry(_make_failing_entry(
            target="wind_direction", sample_count=10
        ))

        summary = sc.get_promotion_summary()
        self.assertEqual(len(summary["promoted"]), 1)
        self.assertEqual(len(summary["fallback"]), 1)
        self.assertEqual(len(summary["information_limited"]), 1)

    def test_information_status_labels(self):
        """Information status must use correct labels."""
        promoted = _make_promoted_entry()
        self.assertEqual(promoted.information_status, "Promoted")

        fallback = _make_failing_entry()
        self.assertEqual(fallback.information_status, "Fallback Active")

        limited = _make_failing_entry(sample_count=10)
        self.assertEqual(limited.information_status, "Information Limited")


class TestTargetRollback(unittest.TestCase):
    """Target-specific rollback tests."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # Create a test policy file
        self.policy = {
            "policy_version": "1.0.0",
            "horizons": {
                "1": {
                    "selected_sources": {
                        "temperature": "persistence_fallback",
                        "humidity": "persistence_fallback",
                        "pressure": "persistence_fallback",
                    }
                },
                "6": {
                    "selected_sources": {
                        "temperature": "learned_model",
                        "humidity": "persistence_fallback",
                        "pressure": "persistence_fallback",
                    }
                },
            },
        }
        self.policy_path = os.path.join(self.tmpdir, "policy.json")
        with open(self.policy_path, "w", encoding="utf-8") as f:
            json.dump(self.policy, f)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_rollback_single_target(self):
        """Rolling back a single target must restore fallback."""
        rb = TargetRollback(self.policy_path)
        self.assertEqual(rb.get_active_source("temperature", 6), "learned_model")

        record = rb.rollback("temperature", 6)
        self.assertEqual(record["old_source"], "learned_model")
        self.assertEqual(record["new_source"], "persistence_fallback")
        self.assertEqual(rb.get_active_source("temperature", 6), "persistence_fallback")

    def test_rollback_all(self):
        """Rolling back all targets must restore all to fallback."""
        rb = TargetRollback(self.policy_path)
        records = rb.rollback_all()
        self.assertGreater(len(records), 0)
        # After rollback all, temperature h6 should be persistence
        self.assertEqual(rb.get_active_source("temperature", 6), "persistence_fallback")

    def test_verify_rollback(self):
        """verify_rollback must confirm fallback state."""
        rb = TargetRollback(self.policy_path)
        # Temperature h1 is already persistence → verify_rollback = True
        self.assertTrue(rb.verify_rollback("temperature", 1))
        # Temperature h6 is learned → verify_rollback = False
        self.assertFalse(rb.verify_rollback("temperature", 6))
        # After rollback, verify should be True
        rb.rollback("temperature", 6)
        self.assertTrue(rb.verify_rollback("temperature", 6))

    def test_policy_summary(self):
        """Policy summary must contain all active and fallback entries."""
        rb = TargetRollback(self.policy_path)
        summary = rb.get_policy_summary()
        self.assertIn("active", summary)
        self.assertIn("fallback", summary)
        self.assertIn("temperature_h6", summary["active"])


if __name__ == "__main__":
    unittest.main()
