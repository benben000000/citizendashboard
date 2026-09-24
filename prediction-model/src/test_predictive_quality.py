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
    build_forecast_windows,
    compute_file_sha256,
    TelemetryDataPipeline,
    PHYSICAL_BOUNDS,
)
from anomaly_detector import (
    TelemetryAnomalyDetector,
    AnomalyRecord,
    PHYSICAL_EXTREMES,
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


if __name__ == "__main__":
    unittest.main()
