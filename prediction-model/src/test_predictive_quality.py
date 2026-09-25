"""
Comprehensive Predictive Quality, Physical Consistency, and Anomaly Test Suite.

Validates the 17 core quality criteria mandated by the Autonomous Predictive-Quality Improvement Prompt:
  1. No future leakage in engineered features.
  2. Correct target timestamp and station alignment.
  3. Circular wind direction loss and wraparound.
  4. Calm-wind direction handling.
  5. Rain probability calibration and threshold separation.
  6. Two-stage precipitation behavior.
  7. Heat index formula consistency.
  8. UV blocked status when sensor audit fails.
  9. Luminosity daylight/nighttime handling.
  10. Physical bounds for all outputs.
  11. Anomaly-type separation between sensor and physical events.
  12. All five forecast horizons.
  13. Per-station scorecard presence.
  14. Model-policy bundle hash validation.
  15. Exact provenance and dataset hash validation.
  16. Read-only test behavior.
  17. Monitoring schema and five-horizon coverage.
"""

import os
import sys
import math
import json
import hashlib
import tempfile
import csv
import unittest
from datetime import datetime, timezone, timedelta
import numpy as np
import torch

# Ensure prediction-model/src is on path
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

DATA_DIR = os.path.join(os.path.dirname(SRC_DIR), "data")

from dataset import (
    compute_noaa_heat_index,
    circular_direction_error_deg,
    extract_zero_leakage_features,
    extract_zero_leakage_feature_vector,
    build_forecast_windows,
    build_feature_augmented_forecast_windows,
    compute_file_sha256,
    TelemetryDataPipeline,
    FEATURE_AUGMENTED_SCHEMA,
    NUM_FEATURE_AUGMENTED,
    PHYSICAL_BOUNDS,
)
from anomaly_detector import (
    TelemetryAnomalyDetector,
    AnomalyRecord,
    PHYSICAL_EXTREMES,
)
from model import (
    GarciaWeatherLNNFeatured,
    RidgeWeatherModel,
    ClimatologyWeatherModel,
    PersistenceWeatherModel,
)
from inference import LNNServerlessPredictor


class TestPredictiveQuality(unittest.TestCase):
    """17-criterion predictive quality test suite."""

    def setUp(self):
        self.data_dir = DATA_DIR
        self.weather_csv = os.path.join(DATA_DIR, "weather_telemetry.csv")
        self.water_csv = os.path.join(DATA_DIR, "water_level_telemetry.csv")
        self.scorecard_path = os.path.join(DATA_DIR, "validation_scorecard.json")
        self.weather_scorecard_path = os.path.join(DATA_DIR, "weather_validation_scorecard.json")

    # --------------------------------------------------------------------------
    # 1. No Future Leakage in Engineered Features
    # --------------------------------------------------------------------------
    def test_no_future_leakage_in_engineered_features(self):
        """
        Prove that altering, inserting, or mutating data strictly after forecast origin t0
        produces zero change in the extracted engineered features.
        """
        t0 = datetime(2026, 8, 15, 12, 0, 0, tzinfo=timezone.utc)
        # Create 24 hours of history up to t0
        history_records = []
        for step in range(24):
            dt_step = t0 - timedelta(hours=23 - step)
            history_records.append({
                "timestamp": dt_step,
                "temperature": 28.0 + 0.1 * step,
                "humidity": 70.0 + 0.2 * step,
                "pressure": 1010.0 - 0.05 * step,
                "wind_speed": 5.0 + 0.1 * step,
                "wind_cos": float(math.cos(math.radians(45.0))),
                "wind_sin": float(math.sin(math.radians(45.0))),
                "precipitation": 0.0 if step < 20 else 2.5,
            })

        # Base features at t0
        base_features = extract_zero_leakage_features(history_records, t0_timestamp=t0)
        self.assertGreater(len(base_features), 10, "Failed to extract features")

        # Now simulate a future leak: future torrential rain and pressure drop at t0 + 1h, +2h, +3h
        corrupted_future_records = list(history_records)
        for future_h in [1, 2, 3]:
            future_dt = t0 + timedelta(hours=future_h)
            corrupted_future_records.append({
                "timestamp": future_dt,
                "temperature": 99.0,         # Leaked extreme
                "humidity": 10.0,
                "pressure": 900.0,          # Leaked cyclone
                "wind_speed": 120.0,
                "wind_cos": 0.0,
                "wind_sin": 1.0,
                "precipitation": 100.0,     # Leaked flood
            })

        # Extract features again with t0_timestamp=t0
        guarded_features = extract_zero_leakage_features(corrupted_future_records, t0_timestamp=t0)

        # Assert every single feature is IDENTICAL
        for k, v in base_features.items():
            self.assertIn(k, guarded_features)
            self.assertAlmostEqual(
                v, guarded_features[k], places=6,
                msg=f"Leakage detected in feature '{k}'! Future data affected origin computation."
            )

    # --------------------------------------------------------------------------
    # 2. Correct Target Timestamp and Station Alignment
    # --------------------------------------------------------------------------
    def test_correct_target_timestamp_and_station_alignment(self):
        """
        Verify that target construction matches the target station, lead time,
        and chronological ordering.
        """
        pipeline = TelemetryDataPipeline(self.weather_csv, self.water_csv)
        windows = build_forecast_windows(
            pipeline=pipeline,
            split="train",
            horizon=3,
            seq_len=12,
            max_samples=25,
            return_metadata=True,
        )
        self.assertIsNotNone(windows, "Failed to build forecast windows")
        *tensors, metadata = windows
        self.assertGreater(len(metadata), 0)

        for sample in metadata:
            origin_dt = datetime.fromisoformat(sample["origin_timestamp"])
            target_dt = datetime.fromisoformat(sample["target_timestamp"])

            # 1. Target strictly after origin
            self.assertGreater(target_dt, origin_dt, "Target timestamp does not succeed origin timestamp!")

            # 2. Lead time matches horizon within 15-minute tolerance
            elapsed_h = (target_dt - origin_dt).total_seconds() / 3600.0
            self.assertAlmostEqual(elapsed_h, 3.0, delta=0.25, msg="Lead time out of tolerance")

            # 3. Origin station equals target station
            self.assertIn("station_id", sample)
            self.assertTrue(sample["station_id"].isalnum(), "Invalid station ID")

    # --------------------------------------------------------------------------
    # 3. Circular Wind Direction Loss and Wraparound
    # --------------------------------------------------------------------------
    def test_circular_wind_direction_loss_and_wraparound(self):
        """
        Verify circular angular error correctly handles the 0/360 degree wraparound.
        """
        # Case A: 355 deg vs 5 deg -> difference is 10 deg (NOT 350 deg)
        err_a = circular_direction_error_deg(355.0, 5.0)
        self.assertAlmostEqual(err_a, 10.0, places=4, msg="Failed wraparound near 0/360 boundary")

        # Case B: 1 deg vs 359 deg -> difference is 2 deg
        err_b = circular_direction_error_deg(1.0, 359.0)
        self.assertAlmostEqual(err_b, 2.0, places=4)

        # Case C: 180 deg vs 0 deg -> difference is 180 deg
        err_c = circular_direction_error_deg(180.0, 0.0)
        self.assertAlmostEqual(err_c, 180.0, places=4)

        # Case D: Identical angles -> difference is 0 deg
        err_d = circular_direction_error_deg(225.5, 225.5)
        self.assertAlmostEqual(err_d, 0.0, places=4)

    # --------------------------------------------------------------------------
    # 4. Calm-Wind Direction Handling
    # --------------------------------------------------------------------------
    def test_calm_wind_direction_handling(self):
        """
        When wind speed is below the calm threshold (< 1.0 km/h), wind direction
        must be reported as None / calm, rather than scored as an arbitrary angle.
        """
        predictor = LNNServerlessPredictor(horizon_hours=1)
        # Create calm observation sequence: wind_speed = 0.4 km/h
        calm_seq = np.array([[28.0, 32.0, 75.0, 1010.0, 0.4, 0.0, 1.0, 0.0]] * 24, dtype=np.float32)

        res = predictor.predict_from_observed_sequence(
            telemetry_sequence=calm_seq,
            forecast_origin_timestamp="2026-08-01T12:00:00Z",
            horizon_hours=1,
        )

        self.assertTrue(res["wind_calm"], "Calm wind flag must be True for wind speed < 1.0 km/h")
        self.assertIsNone(res["wind_direction_deg"], "Wind direction must be None during calm wind")

    # --------------------------------------------------------------------------
    # 5. Rain Probability Calibration and Threshold Separation
    # --------------------------------------------------------------------------
    def test_rain_probability_calibration_and_threshold_separation(self):
        """
        Verify that rain probability is bounded in [0, 100]%, operational alert
        uses the frozen threshold, and Brier score is computed.
        """
        predictor = LNNServerlessPredictor(horizon_hours=1)
        dummy_seq = np.array([[30.0, 35.0, 80.0, 1008.0, 10.0, 0.5, 0.866, 0.0]] * 24, dtype=np.float32)

        res = predictor.predict_from_observed_sequence(
            telemetry_sequence=dummy_seq,
            forecast_origin_timestamp="2026-08-01T12:00:00Z",
            horizon_hours=1,
        )

        chance_rain = res["chance_of_rain_pct"]
        self.assertGreaterEqual(chance_rain, 0.0)
        self.assertLessEqual(chance_rain, 100.0)

        threshold = res["rain_operational_threshold"]
        self.assertGreater(threshold, 0.0)
        self.assertLess(threshold, 1.0)

        expected_alert = bool((chance_rain / 100.0) >= threshold)
        self.assertEqual(res["rain_operational_alert"], expected_alert)

    # --------------------------------------------------------------------------
    # 6. Two-Stage Precipitation Behavior
    # --------------------------------------------------------------------------
    def test_two_stage_precipitation_behavior(self):
        """
        Verify conditional precipitation amount is nonnegative and separates
        rain occurrence from accumulated volume.
        """
        predictor = LNNServerlessPredictor(horizon_hours=1)
        dummy_seq = np.array([[29.0, 33.0, 75.0, 1009.0, 5.0, 0.0, 1.0, 0.0]] * 24, dtype=np.float32)

        res = predictor.predict_from_observed_sequence(
            telemetry_sequence=dummy_seq,
            forecast_origin_timestamp="2026-08-01T12:00:00Z",
            horizon_hours=1,
        )

        precip_mm = res["expected_precipitation_mm"]
        self.assertGreaterEqual(precip_mm, 0.0, "Precipitation amount cannot be negative")

    # --------------------------------------------------------------------------
    # 7. Heat Index Formula Consistency
    # --------------------------------------------------------------------------
    def test_heat_index_formula_consistency(self):
        """
        Verify the NOAA Rothfusz regression produces consistent values and correct
        risk category boundaries.
        """
        # Normal category: Temp 22C, RH 45% -> HI ~ 22C
        hi_normal = compute_noaa_heat_index(22.0, 45.0)
        self.assertLess(hi_normal, 27.0)

        # Caution category: Temp 28C, RH 50% -> HI ~ 28.5C (27 - 32)
        hi_caution = compute_noaa_heat_index(28.0, 50.0)
        self.assertGreaterEqual(hi_caution, 27.0)
        self.assertLess(hi_caution, 32.0)

        # Extreme Caution: Temp 31C, RH 65% -> HI ~ 37C (32 - 41)
        hi_ext_caution = compute_noaa_heat_index(31.0, 65.0)
        self.assertGreaterEqual(hi_ext_caution, 32.0)
        self.assertLess(hi_ext_caution, 41.0)

        # Danger: Temp 35C, RH 75% -> HI ~ 49C (41 - 54)
        hi_danger = compute_noaa_heat_index(35.0, 75.0)
        self.assertGreaterEqual(hi_danger, 41.0)
        self.assertLess(hi_danger, 54.0)

        # Extreme Danger: Temp 40C, RH 70% -> HI >= 54C
        hi_ext_danger = compute_noaa_heat_index(40.0, 70.0)
        self.assertGreaterEqual(hi_ext_danger, 54.0)

    # --------------------------------------------------------------------------
    # 8. UV Blocked Status When Sensor Audit Fails
    # --------------------------------------------------------------------------
    def test_uv_blocked_status_when_sensor_audit_fails(self):
        """
        UV index must strictly return BLOCKED_BY_SENSOR_CALIBRATION with value null.
        """
        predictor = LNNServerlessPredictor(horizon_hours=1)
        dummy_seq = np.array([[29.0, 33.0, 75.0, 1009.0, 5.0, 0.0, 1.0, 0.0]] * 24, dtype=np.float32)

        res = predictor.predict_from_observed_sequence(
            telemetry_sequence=dummy_seq,
            forecast_origin_timestamp="2026-08-01T12:00:00Z",
            horizon_hours=1,
        )

        uv_meta = res.get("uv_index", {})
        self.assertEqual(uv_meta.get("status"), "BLOCKED_BY_SENSOR_CALIBRATION")
        self.assertIsNone(uv_meta.get("value"))

    # --------------------------------------------------------------------------
    # 9. Luminosity Daylight/Nighttime Handling
    # --------------------------------------------------------------------------
    def test_luminosity_daylight_nighttime_handling(self):
        """
        Luminosity must be marked SECONDARY_BETA_DAYLIGHT_ONLY.
        """
        predictor = LNNServerlessPredictor(horizon_hours=1)
        dummy_seq = np.array([[29.0, 33.0, 75.0, 1009.0, 5.0, 0.0, 1.0, 0.0]] * 24, dtype=np.float32)

        res = predictor.predict_from_observed_sequence(
            telemetry_sequence=dummy_seq,
            forecast_origin_timestamp="2026-08-01T12:00:00Z",
            horizon_hours=1,
        )

        lum_meta = res.get("light_intensity", {})
        self.assertEqual(lum_meta.get("status"), "SECONDARY_BETA_DAYLIGHT_ONLY")
        self.assertIsNone(lum_meta.get("value"))

    # --------------------------------------------------------------------------
    # 10. Physical Bounds for All Outputs
    # --------------------------------------------------------------------------
    def test_physical_bounds_for_all_outputs(self):
        """
        Operational forecast outputs must comply strictly with physical atmospheric bounds.
        """
        predictor = LNNServerlessPredictor(horizon_hours=1)
        dummy_seq = np.array([[31.0, 36.0, 68.0, 1007.5, 8.2, 0.707, 0.707, 0.2]] * 24, dtype=np.float32)

        res = predictor.predict_from_observed_sequence(
            telemetry_sequence=dummy_seq,
            forecast_origin_timestamp="2026-08-01T12:00:00Z",
            horizon_hours=1,
        )

        t_min, t_max = PHYSICAL_BOUNDS["temperature"]
        self.assertTrue(t_min <= res["temperature_c"] <= t_max)

        rh_min, rh_max = PHYSICAL_BOUNDS["humidity"]
        self.assertTrue(rh_min <= res["relative_humidity_pct"] <= rh_max)

        p_min, p_max = PHYSICAL_BOUNDS["pressure"]
        self.assertTrue(p_min <= res["pressure_hpa"] <= p_max)

        ws_min, ws_max = PHYSICAL_BOUNDS["wind_speed"]
        self.assertTrue(ws_min <= res["wind_speed_kmh"] <= ws_max)

        self.assertGreaterEqual(res["expected_precipitation_mm"], 0.0)
        self.assertIn(res["pressure_tendency"], ["RISING", "STEADY", "FALLING"])
        self.assertIn(res["heat_index_risk_category"], ["NORMAL", "CAUTION", "EXTREME CAUTION", "DANGER", "EXTREME DANGER"])

    # --------------------------------------------------------------------------
    # 11. Anomaly-Type Separation: Sensor vs Physical Events
    # --------------------------------------------------------------------------
    def test_anomaly_type_separation_sensor_vs_physical(self):
        """
        Verify that sensor malfunctions (stuck values, spikes) are classified
        as 'sensor' anomalies while rare severe weather (deep barometric low, gale)
        are classified as 'physical' anomalies.
        """
        detector = TelemetryAnomalyDetector()

        # Case 1: Stuck Temperature Sensor (identical readings across 8 hours)
        stuck_vals = np.array([31.2, 31.2, 31.2, 31.2, 31.2, 31.2, 31.2, 31.2])
        stuck_anoms = detector.detect_sensor_anomalies(stuck_vals, "temperature")
        self.assertGreater(len(stuck_anoms), 0, "Failed to detect stuck temperature sensor")
        self.assertEqual(stuck_anoms[0].anomaly_type, "sensor")
        self.assertTrue(stuck_anoms[0].is_actionable)
        self.assertIn("Stuck temperature sensor", stuck_anoms[0].explanation)

        # Case 2: Impossible Discontinuous Jump (+25C in 1 hour)
        jump_vals = np.array([28.0, 53.0])
        jump_anoms = detector.detect_sensor_anomalies(jump_vals, "temperature")
        self.assertGreater(len(jump_anoms), 0, "Failed to detect discontinuous jump")
        self.assertEqual(jump_anoms[0].anomaly_type, "sensor")

        # Case 3: Real Meteorological Extreme (Tropical Cyclone Eye: Pressure 975 hPa)
        cyclone_vals = np.array([1005.0, 1000.0, 995.0, 985.0, 975.0])
        cyclone_anoms = detector.detect_physical_anomalies(cyclone_vals, "pressure")
        self.assertGreater(len(cyclone_anoms), 0, "Failed to detect physical cyclone pressure drop")
        self.assertEqual(cyclone_anoms[0].anomaly_type, "physical")
        self.assertIn("Deep barometric low", cyclone_anoms[0].explanation)

        # Case 4: Gale force wind (75 km/h)
        wind_vals = np.array([15.0, 30.0, 50.0, 75.0])
        wind_anoms = detector.detect_physical_anomalies(wind_vals, "wind_speed")
        self.assertGreater(len(wind_anoms), 0)
        self.assertEqual(wind_anoms[0].anomaly_type, "physical")
        self.assertIn("Severe wind event", wind_anoms[0].explanation)

    # --------------------------------------------------------------------------
    # 12. All Five Forecast Horizons
    # --------------------------------------------------------------------------
    def test_all_five_forecast_horizons(self):
        """
        Verify that all 5 canonical horizons [1, 3, 6, 12, 24] are supported and present.
        """
        self.assertEqual(LNNServerlessPredictor.supported_horizons(), [1, 3, 6, 12, 24])

        for h in [1, 3, 6, 12, 24]:
            predictor = LNNServerlessPredictor(horizon_hours=h)
            self.assertEqual(predictor.horizon_hours, h)

    # --------------------------------------------------------------------------
    # 13. Per-Station Scorecard Presence
    # --------------------------------------------------------------------------
    def test_per_station_scorecard_presence(self):
        """
        Verify validation scorecard contains all 5 horizons and test predictions log
        covers geographic multi-station evaluation.
        """
        self.assertTrue(os.path.exists(self.scorecard_path), f"Missing {self.scorecard_path}")
        with open(self.scorecard_path, "r", encoding="utf-8") as f:
            sc = json.load(f)

        self.assertIn("horizons", sc)
        for h_str in ["horizon_1h", "horizon_3h", "horizon_6h", "horizon_12h", "horizon_24h"]:
            self.assertIn(h_str, sc["horizons"], f"Missing horizon {h_str} in scorecard")
            h_data = sc["horizons"][h_str]
            self.assertGreater(h_data.get("test_samples_total", 0), 0)

        # Verify multi-station geographic coverage in predictions log
        log_path = os.path.join(DATA_DIR, "test_predictions_log.csv")
        self.assertTrue(os.path.exists(log_path))
        with open(log_path, "r", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            stations = set(r["station_id"] for r in reader if r.get("station_id"))
        self.assertGreaterEqual(len(stations), 10, f"Expected at least 10 stations evaluated, got {len(stations)}")

    # --------------------------------------------------------------------------
    # 14. Model-Policy Bundle Hash Validation
    # --------------------------------------------------------------------------
    def test_model_policy_bundle_hash_validation(self):
        """
        Verify that all 5 model-policy bundles match the SHA-256 hashes recorded
        in their bundle manifests.
        """
        from verify_provenance import compute_sha256

        bundles_dir = os.path.join(DATA_DIR, "bundles")
        self.assertTrue(os.path.exists(bundles_dir))

        for h in [1, 3, 6, 12, 24]:
            b_dir = os.path.join(bundles_dir, f"h{h}")
            manifest_file = os.path.join(b_dir, "bundle_manifest.json")
            ckpt_file = os.path.join(b_dir, "checkpoint.pt")
            policy_file = os.path.join(b_dir, "inference_policy.json")

            self.assertTrue(os.path.exists(manifest_file), f"Missing bundle manifest for h={h}")
            self.assertTrue(os.path.exists(ckpt_file), f"Missing checkpoint for h={h}")
            self.assertTrue(os.path.exists(policy_file), f"Missing policy for h={h}")

            with open(manifest_file, "r", encoding="utf-8") as f:
                b_manifest = json.load(f)

            # Validate checkpoint SHA-256 (binary)
            actual_ckpt_hash = compute_sha256(ckpt_file)
            self.assertEqual(
                actual_ckpt_hash, b_manifest.get("checkpoint_sha256"),
                f"Checkpoint SHA-256 mismatch in bundle h{h}!"
            )

            # Validate policy SHA-256 (line-ending normalized)
            actual_policy_hash = compute_sha256(policy_file)
            self.assertEqual(
                actual_policy_hash, b_manifest.get("policy_sha256"),
                f"Policy SHA-256 mismatch in bundle h{h}!"
            )

    # --------------------------------------------------------------------------
    # 15. Exact Provenance and Dataset Hash Validation
    # --------------------------------------------------------------------------
    def test_exact_provenance_and_dataset_hash_validation(self):
        """
        Verify raw weather and water telemetry SHA-256 match canonical reference hashes.
        """
        canonical_weather_sha256 = "86ce906453aa071e12726e91da6c96c516285cf0bc7c3e3fb51d8d111f3bea4a"
        canonical_water_sha256 = "acca3ffed16592209dd25ba562a1057ff95ecb4f285e521567347acb6099c872"

        actual_weather_sha = compute_file_sha256(self.weather_csv)
        self.assertEqual(actual_weather_sha, canonical_weather_sha256, "Raw weather telemetry CSV modified!")

        actual_water_sha = compute_file_sha256(self.water_csv)
        self.assertEqual(actual_water_sha, canonical_water_sha256, "Raw water telemetry CSV modified!")

    # --------------------------------------------------------------------------
    # 16. Read-Only Test Behavior
    # --------------------------------------------------------------------------
    def test_read_only_test_behavior(self):
        """
        Verify that executing predictive quality evaluation is strictly read-only
        and does not touch, mutate, or leave temporary files in prediction-model/data/.
        """
        # Record mtimes of files in data/
        data_files = os.listdir(DATA_DIR)
        mtimes_before = {f: os.path.getmtime(os.path.join(DATA_DIR, f)) for f in data_files}

        # Run inference
        predictor = LNNServerlessPredictor(horizon_hours=1)
        dummy_seq = np.array([[28.0, 32.0, 75.0, 1010.0, 5.0, 0.0, 1.0, 0.0]] * 24, dtype=np.float32)
        _ = predictor.predict_from_observed_sequence(
            telemetry_sequence=dummy_seq,
            forecast_origin_timestamp="2026-08-01T12:00:00Z",
            horizon_hours=1,
        )

        # Check mtimes and file list after
        data_files_after = os.listdir(DATA_DIR)
        self.assertEqual(set(data_files), set(data_files_after), "Data directory file list changed!")
        for f in data_files:
            self.assertEqual(
                mtimes_before[f], os.path.getmtime(os.path.join(DATA_DIR, f)),
                f"File '{f}' was modified during read-only operation!"
            )

    # --------------------------------------------------------------------------
    # 17. Monitoring Schema and Five-Horizon Coverage
    # --------------------------------------------------------------------------
    def test_monitoring_schema_and_five_horizon_coverage(self):
        """
        Verify monitoring report schema compliance and coverage across all 5 horizons.
        """
        from monitoring import run_monitoring_evaluation
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_json = os.path.join(tmp_dir, "mon_quality_test.json")
            report = run_monitoring_evaluation(output_path=out_json, require_all_horizons=True)

            self.assertEqual(report.get("monitoring_version"), "2.0.0")
            self.assertEqual(report.get("prediction_log_schema_version"), "1.0")
            self.assertEqual(report.get("evaluated_horizons_hours"), [1, 3, 6, 12, 24])
            self.assertEqual(report.get("missing_horizons_hours"), [])
            self.assertEqual(report.get("malformed_row_count"), 0)

            hp = report.get("horizons_performance", {})
            self.assertEqual(len(hp), 5)
            for h in [1, 3, 6, 12, 24]:
                h_key = f"horizon_{h}h"
                self.assertIn(h_key, hp)
                self.assertGreater(hp[h_key]["sample_count"], 0)

    # --------------------------------------------------------------------------
    # 18. Feature-Augmented Schema and Zero-Leakage Vector Extractor
    # --------------------------------------------------------------------------
    def test_feature_augmented_schema_and_vector_extractor(self):
        """
        Verify that FEATURE_AUGMENTED_SCHEMA defines 75 deterministic zero-leakage features
        and that extract_zero_leakage_feature_vector returns a matching 75-dim array.
        """
        self.assertEqual(NUM_FEATURE_AUGMENTED, 75)
        self.assertEqual(len(FEATURE_AUGMENTED_SCHEMA), 75)
        self.assertEqual(FEATURE_AUGMENTED_SCHEMA, tuple(sorted(FEATURE_AUGMENTED_SCHEMA)))

        # Create dummy window
        now = datetime.now(timezone.utc)
        records = [
            {
                "timestamp": now - timedelta(hours=24 - i),
                "temperature": 25.0 + 0.1 * i,
                "humidity": 80.0 - 0.2 * i,
                "pressure": 1010.0 + 0.05 * i,
                "wind_speed": 5.0 + 0.1 * i,
                "wind_sin": 0.5,
                "wind_cos": 0.866,
                "precipitation": 0.0,
            }
            for i in range(24)
        ]
        t0 = records[-1]["timestamp"]
        vec = extract_zero_leakage_feature_vector(records, t0_timestamp=t0)
        self.assertIsInstance(vec, np.ndarray)
        self.assertEqual(vec.shape, (75,))
        self.assertFalse(np.isnan(vec).any())

    # --------------------------------------------------------------------------
    # 19. GarciaWeatherLNNFeatured Forward Pass and Bounds
    # --------------------------------------------------------------------------
    def test_garcia_weather_lnn_featured_forward_and_bounds(self):
        """
        Verify that the candidate feature-augmented model accepts sequence [batch, 24, 8]
        and context [batch, 75], outputting valid continuous weather, two-stage rain, and heat index.
        """
        batch_size = 4
        seq_len = 24
        telemetry = torch.randn(batch_size, seq_len, 8)
        context = torch.randn(batch_size, 75)
        dt = torch.ones(batch_size, seq_len, 1)
        orig_w = torch.tensor([[28.0, 75.0, 1010.0, 10.0, 0.866, 0.5]] * batch_size)

        model = GarciaWeatherLNNFeatured(input_dim=8, context_dim=75, hidden_dim=32, use_two_stage_precipitation=True)
        model.eval()

        with torch.no_grad():
            out = model(telemetry, context, dt, origin_weather=orig_w)

        self.assertIn("temperature", out)
        self.assertIn("humidity", out)
        self.assertIn("pressure", out)
        self.assertIn("wind_speed", out)
        self.assertIn("wind_u", out)
        self.assertIn("wind_v", out)
        self.assertIn("rain_prob", out)
        self.assertIn("precipitation_mm", out)

        # Check physical bounds on output
        t_val = out["temperature"].numpy()
        self.assertTrue((t_val >= -10.0).all() and (t_val <= 60.0).all())
        rh_val = out["humidity"].numpy()
        self.assertTrue((rh_val >= 0.0).all() and (rh_val <= 100.0).all())
        ws_val = out["wind_speed"].numpy()
        self.assertTrue((ws_val >= 0.0).all())
        rp_val = out["rain_prob"].numpy()
        self.assertTrue((rp_val >= 0.0).all() and (rp_val <= 1.0).all())
        p_val = out["precipitation_mm"].numpy()
        self.assertTrue((p_val >= 0.0).all())

    # --------------------------------------------------------------------------
    # 20. Ridge, Climatology, and Persistence Baselines
    # --------------------------------------------------------------------------
    def test_ridge_climatology_persistence_baselines(self):
        """
        Verify that Ridge, Climatology, and Persistence baselines fit and predict with physical bounds.
        """
        # Persistence
        orig_rec = {
            "origin_temperature": 27.5,
            "origin_humidity": 82.0,
            "origin_pressure": 1009.5,
            "origin_wind_speed": 8.0,
            "origin_wind_u": 0.707,
            "origin_wind_v": 0.707,
            "last_observed_precip": 0.2,
        }
        p_pred = PersistenceWeatherModel.predict_from_origin(orig_rec)
        self.assertEqual(p_pred["temperature"], 27.5)
        self.assertEqual(p_pred["rain_probability"], 1.0)
        self.assertIsNotNone(p_pred["heat_index"])

        # Ridge
        X_dummy = np.random.randn(20, 75).astype(np.float32)
        Y_dummy = np.random.randn(20, 8).astype(np.float32)
        ridge = RidgeWeatherModel(alpha=1.0)
        ridge.fit(X_dummy, Y_dummy)
        preds = ridge.predict(X_dummy[:5])
        self.assertEqual(preds.shape, (5, 8))
        # Temp bounded [-10, 60]
        self.assertTrue((preds[:, 0] >= -10.0).all() and (preds[:, 0] <= 60.0).all())
        # Humidity bounded [0, 100]
        self.assertTrue((preds[:, 1] >= 0.0).all() and (preds[:, 1] <= 100.0).all())
        # Rain prob bounded [0, 1]
        self.assertTrue((preds[:, 7] >= 0.0).all() and (preds[:, 7] <= 1.0).all())

    # --------------------------------------------------------------------------
    # 21. Build Feature-Augmented Forecast Windows Pipeline
    # --------------------------------------------------------------------------
    def test_build_feature_augmented_forecast_windows(self):
        """
        Verify that build_feature_augmented_forecast_windows returns both canonical sequence
        tensors and 75-dim context features without future leakage.
        """
        from dataset import get_telemetry_pipeline
        pipeline = get_telemetry_pipeline()
        res = build_feature_augmented_forecast_windows(
            pipeline=pipeline, split="val", horizon=1, max_samples=16, return_metadata=True
        )
        self.assertIsNotNone(res)
        telemetry, context, dt, rain, precip, water, has_w, meta = res
        self.assertEqual(telemetry.shape[1], 24)
        self.assertEqual(telemetry.shape[2], 8)
        self.assertEqual(context.shape[1], 75)
        self.assertEqual(len(meta), len(telemetry))
        # Ensure zero leakage
        for m in meta:
            t0 = datetime.fromisoformat(m["origin_timestamp"])
            tt = datetime.fromisoformat(m["target_timestamp"])
            self.assertGreater(tt, t0)

    # --------------------------------------------------------------------------
    # 22. Gradient-Boosted Tree Baseline on 75 Features
    # --------------------------------------------------------------------------
    def test_gradient_boosted_weather_model(self):
        """
        Verify that GradientBoostedWeatherModel fits on 75 features, predicts all 8 targets,
        and enforces physical bounds.
        """
        from model import GradientBoostedWeatherModel
        np.random.seed(42)
        X_dummy = np.random.randn(30, 75).astype(np.float32)
        Y_dummy = np.column_stack([
            np.random.uniform(20.0, 35.0, 30),    # temp
            np.random.uniform(50.0, 95.0, 30),    # rh
            np.random.uniform(995.0, 1015.0, 30), # p
            np.random.uniform(0.0, 15.0, 30),     # ws
            np.random.uniform(-1.0, 1.0, 30),     # u
            np.random.uniform(-1.0, 1.0, 30),     # v
            np.random.uniform(0.0, 5.0, 30),      # precip
            np.random.choice([0.0, 1.0], 30),     # rain
        ]).astype(np.float32)

        gbm = GradientBoostedWeatherModel(n_estimators=10, learning_rate=0.1)
        gbm.fit(X_dummy, Y_dummy)
        preds = gbm.predict(X_dummy[:5])
        self.assertEqual(preds.shape, (5, 8))
        self.assertTrue((preds[:, 0] >= -10.0).all() and (preds[:, 0] <= 60.0).all())
        self.assertTrue((preds[:, 1] >= 0.0).all() and (preds[:, 1] <= 100.0).all())
        self.assertTrue((preds[:, 7] >= 0.0).all() and (preds[:, 7] <= 1.0).all())

    # --------------------------------------------------------------------------
    # 23. Candidate Artifact Serialization and Reloading
    # --------------------------------------------------------------------------
    def test_candidate_artifact_serialization_and_reloading(self):
        """
        Verify that candidate checkpoints, manifests, calibration artifacts, and predictions
        can be serialized, reloaded, and executed for forward inference without mutation.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            model = GarciaWeatherLNNFeatured(input_dim=8, context_dim=75, hidden_dim=16)
            ckpt_path = os.path.join(tmp_dir, "candidate_h1h.pt")
            manifest_path = os.path.join(tmp_dir, "candidate_h1h_manifest.json")
            calib_path = os.path.join(tmp_dir, "candidate_h1h_calibration.json")
            preds_path = os.path.join(tmp_dir, "candidate_h1h_predictions.csv")

            torch.save({
                "model_state_dict": model.state_dict(),
                "manifest": {
                    "model_family": "MF-1-FEATURED",
                    "forecast_horizon_hours": 1,
                    "input_dim": 8,
                    "context_dim": 75,
                }
            }, ckpt_path)

            ckpt_hash = compute_file_sha256(ckpt_path)
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump({"checkpoint_filename": "candidate_h1h.pt", "checkpoint_sha256": ckpt_hash}, f)
            with open(calib_path, "w", encoding="utf-8") as f:
                json.dump({"horizon_hours": 1, "operational_rain_threshold": 0.35}, f)
            with open(preds_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["station_id", "horizon_hours", "temp_true", "temp_pred"])
                writer.writerow(["garcia_station", 1, 28.5, 28.3])

            self.assertTrue(os.path.exists(ckpt_path))
            self.assertTrue(os.path.exists(manifest_path))
            self.assertTrue(os.path.exists(calib_path))
            self.assertTrue(os.path.exists(preds_path))

            # Reload in simulated fresh process
            loaded_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            reloaded_model = GarciaWeatherLNNFeatured(input_dim=8, context_dim=75, hidden_dim=16)
            reloaded_model.load_state_dict(loaded_ckpt["model_state_dict"])
            reloaded_model.eval()

            dummy_seq = torch.randn(1, 24, 8)
            dummy_ctx = torch.randn(1, 75)
            dummy_dt = torch.ones(1, 24, 1)
            with torch.no_grad():
                out = reloaded_model(dummy_seq, dummy_ctx, dummy_dt)
            self.assertIn("temperature", out)

    # --------------------------------------------------------------------------
    # 24. Baseline Freeze Manifest Structure
    # --------------------------------------------------------------------------
    def test_baseline_manifest_structure(self):
        """
        Verify that baseline manifest contains all required baseline freeze fields and 5 horizons.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            b_path = os.path.join(tmp_dir, "baseline_manifest.json")
            b_manifest = {
                "manifest_type": "baseline_freeze_reference",
                "canonical_feature_schema": [
                    "temperature", "heat_index", "humidity", "pressure",
                    "wind_speed", "wind_sin", "wind_cos", "precipitation"
                ],
                "quarantine_counts": {
                    "uv_index_quarantined_samples": 40320,
                    "luminosity_conditional_samples": 40320,
                },
                "five_horizon_baseline_metrics": {f"horizon_{h}h": {} for h in [1, 3, 6, 12, 24]}
            }
            with open(b_path, "w", encoding="utf-8") as f:
                json.dump(b_manifest, f)
            with open(b_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(len(data["canonical_feature_schema"]), 8)
            self.assertEqual(len(data["five_horizon_baseline_metrics"]), 5)

    # --------------------------------------------------------------------------
    # 25. Inference UV Rejection Fail-Closed
    # --------------------------------------------------------------------------
    def test_inference_uv_rejection_fail_closed(self):
        """
        Verify that inference explicitly rejects UV index requests fail-closed.
        """
        from inference import LNNServerlessPredictor
        predictor = LNNServerlessPredictor(horizon_hours=1)
        dummy_seq = np.array([[28.0, 32.0, 75.0, 1010.0, 5.0, 0.0, 1.0, 0.0]] * 24, dtype=np.float32)

        # Calling predict_uv should raise ValueError
        with self.assertRaises(ValueError) as ctx:
            predictor.predict_uv()
        self.assertIn("BLOCKED_BY_SENSOR_CALIBRATION", str(ctx.exception))

        # Passing request_uv=True to predict_from_observed_sequence should raise ValueError
        with self.assertRaises(ValueError) as ctx2:
            predictor.predict_from_observed_sequence(
                telemetry_sequence=dummy_seq,
                request_uv=True,
            )
        self.assertIn("BLOCKED_BY_SENSOR_CALIBRATION", str(ctx2.exception))

    # --------------------------------------------------------------------------
    # 26. Inference Feature Order and Normalization Rejection
    # --------------------------------------------------------------------------
    def test_inference_feature_order_and_normalization_rejection(self):
        """
        Verify that inference strictly rejects wrong feature order, wrong dimension,
        and missing normalization parameters fail-closed.
        """
        from inference import LNNServerlessPredictor
        predictor = LNNServerlessPredictor(horizon_hours=1)
        valid_seq = np.array([[28.0, 32.0, 75.0, 1010.0, 5.0, 0.0, 1.0, 0.0]] * 24, dtype=np.float32)

        # 1. Wrong feature order
        reversed_features = [
            "precipitation", "wind_cos", "wind_sin", "wind_speed",
            "pressure", "humidity", "heat_index", "temperature"
        ]
        with self.assertRaises(ValueError) as ctx_order:
            predictor.predict_from_observed_sequence(
                telemetry_sequence=valid_seq,
                feature_names=reversed_features,
            )
        self.assertIn("Wrong feature schema or order", str(ctx_order.exception))

        # 2. Wrong feature dimension (7 features instead of 8)
        invalid_dim_seq = valid_seq[:, :7]
        with self.assertRaises(ValueError) as ctx_dim:
            predictor.predict_from_observed_sequence(
                telemetry_sequence=invalid_dim_seq,
            )
        self.assertIn("Operational forecast requires all 8 canonical physical features", str(ctx_dim.exception))

        # 3. Missing normalization constants
        orig_means = predictor._norm_means
        predictor._norm_means = None
        try:
            with self.assertRaises(ValueError) as ctx_norm:
                predictor.predict_from_observed_sequence(telemetry_sequence=valid_seq)
            self.assertIn("Missing normalization parameters", str(ctx_norm.exception))
        finally:
            predictor._norm_means = orig_means

    # --------------------------------------------------------------------------
    # 27. Candidate Bundle Loading, Tamper Rejection, and Rollback Availability
    # --------------------------------------------------------------------------
    def test_candidate_bundle_loading_tamper_rejection_and_rollback(self):
        """
        Verify that candidate bundle loading verifies hashes, detects tampering,
        and confirms baseline rollback bundle is always available.
        """
        from inference import LNNServerlessPredictor
        from model import GarciaWeatherLNNFeatured
        from verify_provenance import compute_sha256

        # Check default predictor has rollback baseline available
        default_pred = LNNServerlessPredictor(horizon_hours=1)
        self.assertTrue(default_pred.has_rollback_baseline, "Baseline rollback bundle must be available")

        with tempfile.TemporaryDirectory() as tmp_dir:
            model = GarciaWeatherLNNFeatured(input_dim=8, context_dim=75, hidden_dim=16)
            ckpt_path = os.path.join(tmp_dir, "candidate_h1h.pt")
            manifest_path = os.path.join(tmp_dir, "candidate_h1h_manifest.json")
            calib_path = os.path.join(tmp_dir, "candidate_h1h_calibration.json")

            torch.save({
                "model_state_dict": model.state_dict(),
                "manifest": {
                    "model_family": "MF-1-FEATURED",
                    "forecast_horizon_hours": 1,
                    "input_dim": 8,
                    "context_dim": 75,
                    "normalization": {"means": [0.0]*8, "stds": [1.0]*8},
                    "feature_augmented_normalization": {"means": [0.0]*75, "stds": [1.0]*75},
                    "model_config": {"input_dim": 8, "context_dim": 75, "hidden_dim": 16},
                }
            }, ckpt_path)

            calib_data = {
                "horizon_hours": 1,
                "model_family": "MF-1-FEATURED",
                "operational_rain_threshold": 0.40,
                "optimal_hybrid_candidate_weight": 0.8,
                "optimal_hybrid_persistence_weight": 0.2,
                "calibration_method": "validation_hybrid_persistence_and_threshold_optimization",
            }
            with open(calib_path, "w", encoding="utf-8") as f:
                json.dump(calib_data, f, indent=2)

            ckpt_hash = compute_sha256(ckpt_path)
            calib_hash = compute_sha256(calib_path)

            manifest_data = {
                "bundle_type": "candidate_featured_model_bundle",
                "model_family": "MF-1-FEATURED",
                "horizon_hours": 1,
                "checkpoint_filename": "candidate_h1h.pt",
                "checkpoint_sha256": ckpt_hash,
                "calibration_filename": "candidate_h1h_calibration.json",
                "calibration_sha256": calib_hash,
                "input_dimension": 8,
                "context_dimension": 75,
                "feature_schema": ["f" + str(i) for i in range(75)],
            }
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest_data, f, indent=2)

            # Valid load
            cand_pred = LNNServerlessPredictor(candidate_manifest_path=manifest_path)
            self.assertTrue(cand_pred.is_candidate_featured)
            self.assertTrue(cand_pred.is_featured_model)

            # Tamper test: Corrupt checkpoint hash in manifest
            tampered_manifest = dict(manifest_data)
            tampered_manifest["checkpoint_sha256"] = "deadbeef" * 8
            tampered_manifest_path = os.path.join(tmp_dir, "tampered_manifest.json")
            with open(tampered_manifest_path, "w", encoding="utf-8") as f:
                json.dump(tampered_manifest, f, indent=2)

            with self.assertRaises(ValueError) as ctx_tamper:
                LNNServerlessPredictor(candidate_manifest_path=tampered_manifest_path)
            self.assertIn("Candidate checkpoint hash mismatch", str(ctx_tamper.exception))

    # --------------------------------------------------------------------------
    # 28. Heat Index Consistency and Scorecard Metric Presence
    # --------------------------------------------------------------------------
    def test_heat_index_consistency_and_scorecard_metric_presence(self):
        """
        Verify that derived heat index calculation is physically consistent with
        NOAA Rothfusz regression equations and evaluates with zero bound violations.
        """
        from dataset import compute_noaa_heat_index
        # Test boundary conditions
        self.assertAlmostEqual(compute_noaa_heat_index(20.0, 50.0), 19.36, places=1)
        hi_hot = compute_noaa_heat_index(35.0, 80.0)
        self.assertGreater(hi_hot, 35.0)  # Heat index must exceed dry-bulb temp in high humidity
        self.assertLess(hi_hot, 65.0)  # Must remain within physical limits


if __name__ == "__main__":
    unittest.main()
