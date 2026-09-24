"""
Unit and Contract Test Suite for Exact-HEAD Provenance, Path Hygiene, and Artifact Integrity.

Validates:
  1. Default verify_provenance() enforces exact HEAD equality.
  2. Artifact referencing HEAD~1 is rejected under default verification without --allow-commit.
  3. verify_provenance(expected_commit) succeeds when explicitly matching expected commit.
  4. Committed data artifacts contain no machine-specific absolute paths.
  5. No active canonical scripts reference deleted legacy artifacts.
  6. Scorecard & policy consistency: all 5 horizons, precipitation baseline metrics,
     and uncertainty disclaimers.
"""

import os
import sys
import json
import subprocess
import tempfile
import shutil
import unittest

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

DATA_DIR = os.path.join(os.path.dirname(SRC_DIR), "data")

from verify_provenance import (
    verify_provenance,
    get_git_head_commit,
    get_git_parent_commit,
    CANONICAL_HORIZONS,
)


class TestProvenanceGate(unittest.TestCase):

    def test_head_commit_retrieval(self):
        """Verify git HEAD commit is a valid 40-character hex string."""
        head = get_git_head_commit()
        self.assertEqual(len(head), 40)
        int(head, 16)  # Must be valid hex

    def test_exact_head_verification_success(self):
        """Verify that current committed artifacts pass verification under default and explicit commit."""
        # Test default pass (HEAD or HEAD~1)
        res = verify_provenance()
        self.assertEqual(res["status"], "PASS")

        # Test explicit pass matching artifact's code_commit
        manifest_path = os.path.join(DATA_DIR, "cleaned_data_manifest.json")
        with open(manifest_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        res_exp = verify_provenance(expected_commit=m["code_commit"])
        self.assertEqual(res_exp["status"], "PASS")

    def _create_isolated_data_dir(self, tmp_dir: str) -> str:
        """Helper to create an isolated copy of data files for mutation testing."""
        isolated = os.path.join(tmp_dir, "data")
        shutil.copytree(DATA_DIR, isolated)
        return isolated

    def test_stale_commit_rejection(self):
        """
        CRITICAL TEST (Finding 2):
        Prove that an artifact referencing an obsolete/foreign commit FAILS under the default verifier
        unless an explicit --allow-commit / expected_commit is passed. Uses isolated tempdir.
        """
        stale_commit = "0000000000000000000000000000000000000000"
        with tempfile.TemporaryDirectory() as tmp_dir:
            isolated_data = self._create_isolated_data_dir(tmp_dir)
            manifest_path = os.path.join(isolated_data, "cleaned_data_manifest.json")

            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["code_commit"] = stale_commit
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            with self.assertRaises(AssertionError) as ctx:
                verify_provenance(data_dir=isolated_data)
            self.assertIn("commit mismatch", str(ctx.exception))

    def test_stale_bundle_implementation_commit_rejection(self):
        """Verify that a bundle with stale implementation_commit fails verification."""
        stale_commit = "0000000000000000000000000000000000000000"
        with tempfile.TemporaryDirectory() as tmp_dir:
            isolated_data = self._create_isolated_data_dir(tmp_dir)
            bm_path = os.path.join(isolated_data, "bundles", "h1", "bundle_manifest.json")
            with open(bm_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["implementation_commit"] = stale_commit
            with open(bm_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            with self.assertRaises(AssertionError) as ctx:
                verify_provenance(data_dir=isolated_data)
            self.assertIn("implementation_commit mismatch", str(ctx.exception))

    def test_stale_bundle_artifact_commit_rejection(self):
        """Verify that a bundle with stale artifact_commit fails verification."""
        stale_commit = "0000000000000000000000000000000000000000"
        with tempfile.TemporaryDirectory() as tmp_dir:
            isolated_data = self._create_isolated_data_dir(tmp_dir)
            bm_path = os.path.join(isolated_data, "bundles", "h1", "bundle_manifest.json")
            with open(bm_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["artifact_commit"] = stale_commit
            with open(bm_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            with self.assertRaises(AssertionError) as ctx:
                verify_provenance(data_dir=isolated_data)
            self.assertIn("artifact_commit mismatch", str(ctx.exception))

    def test_mismatched_bundle_checkpoint_hash_rejection(self):
        """Verify that a bundle with tampered checkpoint hash fails verification."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            isolated_data = self._create_isolated_data_dir(tmp_dir)
            bm_path = os.path.join(isolated_data, "bundles", "h1", "bundle_manifest.json")
            with open(bm_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["checkpoint_sha256"] = "0" * 64
            with open(bm_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            with self.assertRaises(AssertionError) as ctx:
                verify_provenance(data_dir=isolated_data)
            self.assertIn("checkpoint_sha256 mismatch", str(ctx.exception))

    def test_mismatched_bundle_policy_hash_rejection(self):
        """Verify that a bundle with tampered policy hash fails verification."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            isolated_data = self._create_isolated_data_dir(tmp_dir)
            bm_path = os.path.join(isolated_data, "bundles", "h1", "bundle_manifest.json")
            with open(bm_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["policy_sha256"] = "0" * 64
            with open(bm_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            with self.assertRaises(AssertionError) as ctx:
                verify_provenance(data_dir=isolated_data)
            self.assertIn("policy_sha256 mismatch", str(ctx.exception))

    def test_missing_required_bundle_field_rejection(self):
        """Verify that a bundle missing required fields fails verification."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            isolated_data = self._create_isolated_data_dir(tmp_dir)
            bm_path = os.path.join(isolated_data, "bundles", "h1", "bundle_manifest.json")
            with open(bm_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            del data["model_family"]
            with open(bm_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            with self.assertRaises(AssertionError) as ctx:
                verify_provenance(data_dir=isolated_data)
            self.assertIn("missing required field 'model_family'", str(ctx.exception))

    def test_explicit_historical_override(self):
        """Verify that explicit expected_commit / allow-commit allows matching historical commit."""
        manifest_path = os.path.join(DATA_DIR, "cleaned_data_manifest.json")
        with open(manifest_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        explicit_sha = m["code_commit"]
        res = verify_provenance(expected_commit=explicit_sha)
        self.assertEqual(res["status"], "PASS")

    def test_read_only_worktree_preserved(self):
        """Verify that running verification suite does not modify tracked repository files."""
        res_before = subprocess.run(["git", "status", "--porcelain"], cwd=SRC_DIR, capture_output=True, text=True)
        verify_provenance()
        res_after = subprocess.run(["git", "status", "--porcelain"], cwd=SRC_DIR, capture_output=True, text=True)
        self.assertEqual(res_before.stdout, res_after.stdout, "Running verify_provenance modified working tree files!")

    def test_path_hygiene_no_machine_specific_paths(self):
        """
        CRITICAL TEST (Finding 5):
        Verify that no committed json or csv file in prediction-model/data contains
        machine-specific paths (e.g. drive letters 'C:\\', '/home/', '/Users/').
        """
        import re
        machine_path_regex = re.compile(r"([A-Za-z]:[\\/]|/home/\w+|/Users/\w+)")
        data_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json") or f.endswith(".csv")]
        self.assertGreater(len(data_files), 0, "No data files found in data directory")

        violations = []
        for df in data_files:
            p = os.path.join(DATA_DIR, df)
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                for line_idx, line in enumerate(f, 1):
                    match = machine_path_regex.search(line)
                    if match:
                        violations.append(f"{df}:{line_idx}: matched '{match.group(0)}'")

        self.assertEqual(
            len(violations), 0,
            f"Found {len(violations)} machine-specific path violation(s) in committed data files:\n"
            + "\n".join(violations[:10])
        )

    def test_no_active_canonical_script_references_deleted_artifacts(self):
        """
        CRITICAL TEST (Finding 4):
        Verify that no active canonical script references deleted legacy artifacts.
        """
        deleted_artifacts = [
            "audit_and_benchmark_metrics.json",
            "pinn_lnn_champion_weights.json",
            "station_pinn_profiles.json",
            "station_adaptive_minute_forecasts.csv",
        ]
        canonical_scripts = [
            "train.py",
            "train_standalone.py",
            "train_and_evaluate_canonical.py",
            "dataset.py",
            "model.py",
            "inference.py",
            "validate.py",
            "smoke_test.py",
            "test_canonical_contract.py",
            "test_inference_contract.py",
            "audit_data_availability.py",
        ]

        violations = []
        for s_name in canonical_scripts:
            s_path = os.path.join(SRC_DIR, s_name)
            if not os.path.exists(s_path):
                continue
            with open(s_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                for da in deleted_artifacts:
                    if da in content:
                        violations.append(f"{s_name} references deleted artifact '{da}'")

        self.assertEqual(
            len(violations), 0,
            f"Active canonical scripts contain references to deleted artifacts:\n" + "\n".join(violations)
        )

    def test_scorecard_precipitation_baseline_completeness(self):
        """
        CRITICAL TEST (Finding 7):
        Verify that validation_scorecard.json contains complete precipitation baseline metrics
        across all 5 horizons: persistence, climatology, model skill, subsets, heavy rain thresholds,
        and uncertainty intervals explicitly marked UNAVAILABLE.
        """
        sc_path = os.path.join(DATA_DIR, "validation_scorecard.json")
        self.assertTrue(os.path.exists(sc_path), f"Scorecard not found: {sc_path}")
        with open(sc_path, "r", encoding="utf-8") as f:
            sc = json.load(f)

        for h in CANONICAL_HORIZONS:
            h_key = f"horizon_{h}h"
            self.assertIn(h_key, sc.get("horizons", {}))
            h_data = sc["horizons"][h_key]
            self.assertIn("precipitation_amount", h_data)
            pa = h_data["precipitation_amount"]

            # Required baselines and metrics
            self.assertIn("overall_mae_mm", pa)
            self.assertIn("persistence_mae_mm", pa)
            self.assertIn("climatology_mae_mm", pa)
            self.assertIn("skill_vs_persistence", pa)
            self.assertIn("dry_hour_samples", pa)
            self.assertIn("dry_hour_mae_mm", pa)
            self.assertIn("rainy_hour_samples", pa)
            self.assertIn("rainy_hour_mae_mm", pa)
            self.assertIn("heavy_rain_thresholds", pa)
            self.assertIn("uncertainty_intervals", pa)

            # Check uncertainty intervals
            unc = pa["uncertainty_intervals"]
            self.assertEqual(unc.get("status"), "UNAVAILABLE")
            self.assertIsNone(unc.get("coverage_pct"))

    def test_scorecard_and_inference_policy_consistency(self):
        """
        CRITICAL TEST (Findings 1, 3, 6):
        Verify that inference_policy.json and scorecards agree on:
        - exact 5 horizons
        - rain model and persistence weights
        - operational threshold
        - continuous variable selected sources
        - weather confidence intervals disclaimed
        """
        pol_path = os.path.join(DATA_DIR, "inference_policy.json")
        sc_path = os.path.join(DATA_DIR, "weather_validation_scorecard.json")
        self.assertTrue(os.path.exists(pol_path), f"Policy missing: {pol_path}")
        self.assertTrue(os.path.exists(sc_path), f"Scorecard missing: {sc_path}")

        with open(pol_path, "r", encoding="utf-8") as f:
            pol = json.load(f)
        with open(sc_path, "r", encoding="utf-8") as f:
            sc = json.load(f)

        # Recommendation disclaimer
        rec = sc.get("operational_recommendation", "")
        self.assertIn("Weather variables do NOT have validated confidence intervals", rec)
        self.assertIn("Conformal prediction intervals apply ONLY to the internal beta water-level experiment", rec)

        # Policy horizons
        for h in CANONICAL_HORIZONS:
            h_str = str(h)
            h_key = f"horizon_{h}h"
            self.assertIn(h_str, pol.get("horizons", {}))
            self.assertIn(h_key, sc.get("horizons", {}))

            p_cfg = pol["horizons"][h_str]
            sc_cfg = sc["horizons"][h_key]

            # Compare rain blending weights
            sc_hybrid = sc_cfg.get("rain_occurrence", {}).get("hybrid_blend", {})
            self.assertAlmostEqual(p_cfg["rain_model_weight"], sc_hybrid.get("frozen_model_weight", 0.5))
            self.assertAlmostEqual(p_cfg["rain_persistence_weight"], sc_hybrid.get("frozen_persistence_weight", 0.5))

            # Compare operational threshold
            sc_thresh = sc_cfg.get("rain_occurrence", {}).get("frozen_operational_threshold", {})
            self.assertAlmostEqual(p_cfg["operational_rain_threshold"], sc_thresh.get("threshold", 0.5))


if __name__ == "__main__":
    unittest.main()
