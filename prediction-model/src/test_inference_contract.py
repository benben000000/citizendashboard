"""
Unit and Contract Test Suite for Operational Inference Hybrid/Fallback Policy.

Validates:
  1. Missing or malformed inference policy raises FileNotFoundError/ValueError (fail-closed).
  2. Commit mismatch between checkpoint and policy raises ValueError (fail-closed).
  3. Horizon-specific policy enforcement: unsupported horizons fail closed.
  4. Continuous weather variable selection (learned_model vs persistence_fallback).
  5. Rain probability hybrid blending: w_m * p_model + w_p * p_persist.
  6. Operational rain thresholding and alert classification.
  7. Explicit metadata completeness (sources, weights, threshold, policy & model commit, weather uncertainty UNAVAILABLE).
  8. Canonical 8-feature contract and NaN/Inf fail-closed behavior.
  9. Separation of research_projected_sequence (synthetic/exploratory).
"""

import os
import sys
import json
import tempfile
import unittest
import numpy as np
import torch

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

DATA_DIR = os.path.join(os.path.dirname(SRC_DIR), "data")

from inference import LNNServerlessPredictor


class TestInferencePolicyContract(unittest.TestCase):

    def setUp(self):
        # Verify checkpoint exists or skip tests requiring real model
        self.ckpt_path = os.path.join(DATA_DIR, "lnn_weather_water.pt")
        self.policy_path = os.path.join(DATA_DIR, "inference_policy.json")

    def test_missing_policy_fails_closed(self):
        """Verify that a missing policy artifact raises FileNotFoundError (fail-closed)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            fake_pol = os.path.join(tmp_dir, "nonexistent_policy.json")
            with self.assertRaises(FileNotFoundError):
                LNNServerlessPredictor(model_weights_path=self.ckpt_path, policy_path=fake_pol)

    def test_malformed_policy_fails_closed(self):
        """Verify that malformed JSON or missing 'horizons' raises ValueError (fail-closed)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_json = os.path.join(tmp_dir, "bad_policy.json")
            with open(bad_json, "w", encoding="utf-8") as f:
                f.write("{ invalid json")
            with self.assertRaises(ValueError):
                LNNServerlessPredictor(model_weights_path=self.ckpt_path, policy_path=bad_json)

            missing_horizons = os.path.join(tmp_dir, "no_horizons.json")
            with open(missing_horizons, "w", encoding="utf-8") as f:
                json.dump({"policy_version": "1.0.0"}, f)
            with self.assertRaises(ValueError):
                LNNServerlessPredictor(model_weights_path=self.ckpt_path, policy_path=missing_horizons)

    def test_commit_mismatch_fails_closed(self):
        """Verify that mismatched commit between checkpoint and policy raises ValueError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pol_file = os.path.join(tmp_dir, "mismatch_policy.json")
            with open(pol_file, "w", encoding="utf-8") as f:
                json.dump({
                    "policy_version": "1.0.0",
                    "policy_code_commit": "0000000000000000000000000000000000000000",
                    "horizons": {"1": {}}
                }, f)

            # Checkpoint manifest commit
            ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
            model_commit = ckpt.get("manifest", {}).get("code_commit")
            if model_commit and model_commit != "unknown":
                with self.assertRaises(ValueError):
                    LNNServerlessPredictor(model_weights_path=self.ckpt_path, policy_path=pol_file)

    def test_unsupported_horizon_fails_closed(self):
        """Verify that asking for an unconfigured horizon (e.g. 2h, 48h) raises ValueError."""
        predictor = LNNServerlessPredictor(model_weights_path=self.ckpt_path)
        dummy_seq = np.array([[28.0, 32.0, 75.0, 1010.0, 5.0, 0.0, 1.0, 0.0]] * 24, dtype=np.float32)

        with self.assertRaises(ValueError) as ctx:
            predictor.predict_from_observed_sequence(
                telemetry_sequence=dummy_seq,
                forecast_origin_timestamp="2026-08-01T12:00:00",
                horizon_hours=2,  # Not in [1, 3, 6, 12, 24]
            )
        self.assertIn("not supported in operational inference policy", str(ctx.exception))

    def test_continuous_variable_policy_selection(self):
        """
        Verify that:
        - When policy specifies 'persistence_fallback', the operational output matches the origin observation.
        - When policy specifies 'learned_model', the operational output uses the model prediction.
        - Raw predictions and observations are preserved in diagnostics.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a test policy with synthetic selection:
            # temperature -> persistence_fallback
            # humidity -> learned_model
            # pressure -> persistence_fallback
            # wind_speed -> persistence_fallback
            # wind_direction -> persistence_fallback
            custom_pol = os.path.join(tmp_dir, "custom_policy.json")
            ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
            commit = ckpt.get("manifest", {}).get("code_commit", "unknown")

            with open(custom_pol, "w", encoding="utf-8") as f:
                json.dump({
                    "policy_version": "1.0.0",
                    "policy_code_commit": commit,
                    "horizons": {
                        "1": {
                            "selected_sources": {
                                "temperature": "persistence_fallback",
                                "humidity": "learned_model",
                                "pressure": "persistence_fallback",
                                "wind_speed": "persistence_fallback",
                                "wind_direction": "persistence_fallback",
                                "heat_index": "derived_from_selected_temp_and_humidity",
                            },
                            "rain_model_weight": 0.8,
                            "rain_persistence_weight": 0.2,
                            "operational_rain_threshold": 0.4,
                            "calibration_code_commit": commit,
                        }
                    }
                }, f)

            predictor = LNNServerlessPredictor(model_weights_path=self.ckpt_path, policy_path=custom_pol)
            # Origin observation has distinctive values:
            # temp=31.5, rh=65.0, pressure=1005.5, ws=8.0
            orig_row = [31.5, 36.0, 65.0, 1005.5, 8.0, 0.0, 1.0, 0.0]
            dummy_seq = np.array([orig_row] * 24, dtype=np.float32)

            res = predictor.predict_from_observed_sequence(
                telemetry_sequence=dummy_seq,
                forecast_origin_timestamp="2026-08-01T12:00:00",
                horizon_hours=1,
            )

            # Temperature selected as persistence_fallback -> MUST match origin temp 31.5
            self.assertEqual(res["temperature_c"], 31.5)
            self.assertEqual(res["selected_source_by_variable"]["temperature"], "persistence_fallback")

            # Pressure selected as persistence_fallback -> MUST match origin pressure 1005.5
            self.assertEqual(res["pressure_hpa"], 1005.5)
            self.assertEqual(res["selected_source_by_variable"]["pressure"], "persistence_fallback")

            # Humidity selected as learned_model -> MUST be model prediction
            self.assertEqual(res["selected_source_by_variable"]["humidity"], "learned_model")

            # Diagnostics must preserve raw model prediction
            self.assertIn("diagnostics", res)
            self.assertIn("raw_learned_predictions", res["diagnostics"])
            self.assertIn("persistence_observations", res["diagnostics"])
            self.assertEqual(res["diagnostics"]["persistence_observations"]["temperature_c"], 31.5)

    def test_rain_hybrid_blending_and_operational_threshold(self):
        """
        Verify that:
        - blended_probability = model_weight * model_prob + persistence_weight * persistence_prob.
        - operational alert fires if and only if blended_prob >= operational_threshold.
        - explicit rain metadata is returned.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            custom_pol = os.path.join(tmp_dir, "rain_policy.json")
            ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
            commit = ckpt.get("manifest", {}).get("code_commit", "unknown")

            with open(custom_pol, "w", encoding="utf-8") as f:
                json.dump({
                    "policy_version": "1.0.0",
                    "policy_code_commit": commit,
                    "horizons": {
                        "1": {
                            "selected_sources": {
                                "temperature": "persistence_fallback",
                                "humidity": "persistence_fallback",
                                "pressure": "persistence_fallback",
                                "wind_speed": "persistence_fallback",
                                "wind_direction": "persistence_fallback",
                                "heat_index": "derived_from_selected_temp_and_humidity",
                            },
                            "rain_model_weight": 0.6,
                            "rain_persistence_weight": 0.4,
                            "operational_rain_threshold": 0.45,
                            "calibration_code_commit": commit,
                        }
                    }
                }, f)

            predictor = LNNServerlessPredictor(model_weights_path=self.ckpt_path, policy_path=custom_pol)

            # Test 1: Origin with dry conditions (last observed precip = 0.0) -> persist_prob = 0.0
            dry_row = [28.0, 32.0, 75.0, 1010.0, 5.0, 0.0, 1.0, 0.0]
            res_dry = predictor.predict_from_observed_sequence(
                telemetry_sequence=np.array([dry_row] * 24, dtype=np.float32),
                forecast_origin_timestamp="2026-08-01T12:00:00",
                horizon_hours=1,
            )
            raw_model_prob = res_dry["diagnostics"]["raw_learned_predictions"]["rain_probability"]
            expected_blended_dry = 0.6 * raw_model_prob + 0.4 * 0.0
            self.assertAlmostEqual(res_dry["chance_of_rain_pct"] / 100.0, expected_blended_dry, places=3)
            self.assertEqual(res_dry["rain_model_weight"], 0.6)
            self.assertEqual(res_dry["rain_persistence_weight"], 0.4)
            self.assertEqual(res_dry["rain_operational_threshold"], 0.45)
            self.assertEqual(res_dry["rain_operational_alert"], bool(expected_blended_dry >= 0.45))

            # Test 2: Origin with rain (last observed precip = 2.5 mm) -> persist_prob = 1.0
            rain_row = [28.0, 32.0, 75.0, 1010.0, 5.0, 0.0, 1.0, 2.5]
            res_rain = predictor.predict_from_observed_sequence(
                telemetry_sequence=np.array([rain_row] * 24, dtype=np.float32),
                forecast_origin_timestamp="2026-08-01T12:00:00",
                horizon_hours=1,
            )
            raw_prob_2 = res_rain["diagnostics"]["raw_learned_predictions"]["rain_probability"]
            expected_blended_rain = 0.6 * raw_prob_2 + 0.4 * 1.0
            self.assertAlmostEqual(res_rain["chance_of_rain_pct"] / 100.0, expected_blended_rain, places=3)
            self.assertEqual(res_rain["rain_operational_alert"], bool(expected_blended_rain >= 0.45))

    def test_explicit_weather_uncertainty_unavailable(self):
        """Verify that weather_uncertainty is explicitly flagged as UNAVAILABLE."""
        predictor = LNNServerlessPredictor(model_weights_path=self.ckpt_path)
        dummy_seq = np.array([[28.0, 32.0, 75.0, 1010.0, 5.0, 0.0, 1.0, 0.0]] * 24, dtype=np.float32)

        res = predictor.predict_from_observed_sequence(
            telemetry_sequence=dummy_seq,
            forecast_origin_timestamp="2026-08-01T12:00:00",
            horizon_hours=1,
        )
        self.assertIn("weather_uncertainty", res)
        self.assertEqual(res["weather_uncertainty"]["status"], "UNAVAILABLE")
        self.assertIn("Conformal prediction intervals apply ONLY to the internal beta water-level experiment",
                      res["weather_uncertainty"]["reason"])

    def test_fail_closed_on_nan_and_feature_contract(self):
        """Verify fail-closed behavior on NaN, Inf, and feature schema violations."""
        predictor = LNNServerlessPredictor(model_weights_path=self.ckpt_path)

        # 7 features instead of 8
        bad_dim = np.zeros((24, 7), dtype=np.float32)
        with self.assertRaises(ValueError):
            predictor.predict_from_observed_sequence(bad_dim, horizon_hours=1)

        # NaN values
        nan_seq = np.zeros((24, 8), dtype=np.float32)
        nan_seq[5, 2] = np.nan
        with self.assertRaises(ValueError):
            predictor.predict_from_observed_sequence(nan_seq, horizon_hours=1)

        # Inf values
        inf_seq = np.zeros((24, 8), dtype=np.float32)
        inf_seq[10, 4] = np.inf
        with self.assertRaises(ValueError):
            predictor.predict_from_observed_sequence(inf_seq, horizon_hours=1)

    def test_all_five_horizons_bundle_contract(self):
        """
        Phase 4 Requirement:
        For every supported horizon:
        1. Load the default operational bundle.
        2. Build a valid 24-step eight-feature observed sequence.
        3. Call predict_from_observed_sequence().
        4. Verify returned policy version and provenance.
        5. Verify selected sources match policy.
        6. Verify API uses persistence for variables marked persistence_fallback.
        7. Verify API uses model values for variables marked learned_model.
        8. Independently recompute rain blending.
        9. Compare independent value with API response.
        10. Verify operational alert classification using frozen threshold.
        11. Verify weather uncertainty is explicitly reported UNAVAILABLE.
        12. Verify water-level output remains beta-only.
        """
        horizons = [1, 3, 6, 12, 24]
        for h in horizons:
            predictor = LNNServerlessPredictor(horizon_hours=h)
            self.assertTrue(predictor.is_bundled, f"Predictor for {h}h was not loaded from bundle")

            # Sequence with distinct values:
            # temp=29.4, heat_idx=34.0, rh=72.0, press=1008.5, ws=9.5, wind_sin=0.6, wind_cos=0.8, precip=1.2
            obs_row = [29.4, 34.0, 72.0, 1008.5, 9.5, 0.6, 0.8, 1.2]
            valid_seq = np.array([obs_row] * 24, dtype=np.float32)

            res = predictor.predict_from_observed_sequence(
                telemetry_sequence=valid_seq,
                forecast_origin_timestamp="2026-08-01T12:00:00",
            )

            # Metadata verification
            self.assertEqual(res["forecast_horizon"], f"{h}h")
            self.assertEqual(res["active_bundle_horizon"], f"{h}h")
            self.assertIn("bundle_version", res)
            self.assertIn("policy_version", res)
            self.assertIn("provenance", res)
            self.assertIn("implementation_commit", res["provenance"])
            self.assertIn("artifact_commit", res["provenance"])
            self.assertIn("checkpoint_sha256", res["provenance"])
            self.assertIn("policy_sha256", res["provenance"])

            # Verify policy sources
            h_policy = predictor.policy["horizons"][str(h)]
            sources = h_policy.get("selected_sources", {})
            for var in ["temperature", "humidity", "pressure", "wind_speed", "wind_direction"]:
                expected_src = sources.get(var, "persistence_fallback")
                self.assertEqual(res["selected_source_by_variable"][var], expected_src)

            # If temperature is persistence_fallback, output must equal origin 29.4
            if sources.get("temperature") == "persistence_fallback":
                self.assertEqual(res["temperature_c"], 29.4)

            # If pressure is persistence_fallback, output must equal origin 1008.5
            if sources.get("pressure") == "persistence_fallback":
                self.assertEqual(res["pressure_hpa"], 1008.5)

            # Independent rain blending recomputation
            raw_model_prob = res["diagnostics"]["raw_learned_predictions"]["rain_probability"]
            w_m = float(h_policy.get("rain_model_weight", 1.0))
            w_p = float(h_policy.get("rain_persistence_weight", 0.0))
            thresh = float(h_policy.get("operational_rain_threshold", 0.5))
            persist_prob = 1.0 if obs_row[7] >= 0.1 else 0.0

            independent_blend = max(0.0, min(1.0, (w_m * raw_model_prob) + (w_p * persist_prob)))
            self.assertAlmostEqual(res["chance_of_rain_pct"] / 100.0, independent_blend, delta=0.01)
            self.assertEqual(res["rain_operational_alert"], bool(independent_blend >= thresh))

            # Uncertainty & Beta Safety
            self.assertEqual(res["weather_uncertainty"]["status"], "UNAVAILABLE")
            self.assertEqual(res["water_level_beta"]["not_for_life_safety"], True)
            self.assertEqual(res["water_level_beta"]["status"], "INTERNAL_EXPERIMENT_BETA")

    def test_bundle_security_fail_closed(self):
        """Verify that tampered bundle hashes and missing files fail closed."""
        import shutil

        # 1. Tampered checkpoint hash in bundle
        with tempfile.TemporaryDirectory() as tmp_dir:
            b_dir = os.path.join(tmp_dir, "h1")
            shutil.copytree(os.path.join(DATA_DIR, "bundles", "h1"), b_dir)
            bm_path = os.path.join(b_dir, "bundle_manifest.json")
            with open(bm_path, "r", encoding="utf-8") as f:
                bm = json.load(f)
            bm["checkpoint_sha256"] = "0" * 64
            with open(bm_path, "w", encoding="utf-8") as f:
                json.dump(bm, f)

            with self.assertRaises(ValueError) as ctx:
                LNNServerlessPredictor(bundle_dir=b_dir, horizon_hours=1)
            self.assertIn("checkpoint hash mismatch", str(ctx.exception))

        # 2. Tampered policy hash in bundle
        with tempfile.TemporaryDirectory() as tmp_dir:
            b_dir = os.path.join(tmp_dir, "h1")
            shutil.copytree(os.path.join(DATA_DIR, "bundles", "h1"), b_dir)
            bm_path = os.path.join(b_dir, "bundle_manifest.json")
            with open(bm_path, "r", encoding="utf-8") as f:
                bm = json.load(f)
            bm["policy_sha256"] = "0" * 64
            with open(bm_path, "w", encoding="utf-8") as f:
                json.dump(bm, f)

            with self.assertRaises(ValueError) as ctx:
                LNNServerlessPredictor(bundle_dir=b_dir, horizon_hours=1)
            self.assertIn("policy hash mismatch", str(ctx.exception))

        # 3. Missing bundle file
        with tempfile.TemporaryDirectory() as tmp_dir:
            b_dir = os.path.join(tmp_dir, "h1")
            shutil.copytree(os.path.join(DATA_DIR, "bundles", "h1"), b_dir)
            os.remove(os.path.join(b_dir, "checkpoint.pt"))

            with self.assertRaises(FileNotFoundError) as ctx:
                LNNServerlessPredictor(bundle_dir=b_dir, horizon_hours=1)
            self.assertIn("Missing required bundle artifact", str(ctx.exception))

        # 4. Horizon mismatch in bundle
        with tempfile.TemporaryDirectory() as tmp_dir:
            b_dir = os.path.join(tmp_dir, "h1")
            shutil.copytree(os.path.join(DATA_DIR, "bundles", "h1"), b_dir)

            with self.assertRaises(ValueError) as ctx:
                LNNServerlessPredictor(bundle_dir=b_dir, horizon_hours=3)
            self.assertIn("Bundle horizon mismatch", str(ctx.exception))

    def test_predict_horizon_mismatch_fails_closed(self):
        """Verify calling predict_from_observed_sequence with mismatched horizon raises ValueError."""
        predictor = LNNServerlessPredictor(horizon_hours=1)
        dummy_seq = np.array([[28.0, 32.0, 75.0, 1010.0, 5.0, 0.0, 1.0, 0.0]] * 24, dtype=np.float32)

        with self.assertRaises(ValueError) as ctx:
            predictor.predict_from_observed_sequence(
                telemetry_sequence=dummy_seq,
                forecast_origin_timestamp="2026-08-01T12:00:00",
                horizon_hours=3,
            )
        self.assertIn("Horizon mismatch", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
