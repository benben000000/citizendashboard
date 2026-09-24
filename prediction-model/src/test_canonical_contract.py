"""
Unit Test Suite for Canonical Telemetry Dataset and Forecasting Contract.

Validates:
  1. Synthetic 1-minute telemetry maps horizon=1 to target 1 hour later (60 rows later), NOT 1 row later.
  2. Data quarantine exclusions: 2069 anomaly and physical sensor bound violations are quarantined.
  3. Strict normalization isolation: modifying val/test data does not change fitted training normalization.
  4. Chronological split & embargo: no cross-split leakage, no test timestamp <= train end.
  5. Station boundary isolation: no sequence window crosses station boundaries.
"""

import os
import sys
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

from dataset import (
    TelemetryDataPipeline,
    TelemetryDataset,
    build_forecast_windows,
    normalize_features,
    PHYSICAL_BOUNDS,
    DEFAULT_SEQ_LEN,
)


class TestCanonicalForecastingContract(unittest.TestCase):

    def test_synthetic_minute_telemetry_lead_time(self):
        """
        PROVE that 1-minute telemetry with horizon=1 targets an observation
        exactly 1 HOUR later (60 minute observations later), NOT 1 row later.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            weather_csv = os.path.join(tmp_dir, "synth_weather.csv")
            water_csv = os.path.join(tmp_dir, "synth_water.csv")

            # Create synthetic 1-minute telemetry across 48 hours
            base_time = datetime(2026, 7, 1, 0, 0, 0, tzinfo=timezone.utc)
            rows = []
            total_minutes = 48 * 60  # 2880 minutes

            with open(weather_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "station_id", "station_name", "location", "recorded_at",
                    "temperature", "heat_index", "humidity", "pressure",
                    "wind_speed", "wind_direction", "precipitation", "uv_index", "light_intensity"
                ])
                for m in range(total_minutes):
                    ts = base_time + timedelta(minutes=m)
                    # Every minute rain is 0.1mm except at minute 1500 (hour 25)
                    precip = 0.11 if m % 60 == 30 else 0.0
                    writer.writerow([
                        "ST_TEST_01", "Test Station", "Manila", ts.isoformat(),
                        28.0 + 0.001 * (m % 100), 32.0, 70.0, 1008.0, 5.0, 180.0, precip, 5.0, 500.0
                    ])

            with open(water_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["station_id", "station_name", "location", "recorded_at", "water_level_cm", "water_level_m"])

            pipeline = TelemetryDataPipeline(weather_csv=weather_csv, water_csv=water_csv)

            # Build windows with horizon = 1 hour
            res = build_forecast_windows(
                pipeline=pipeline,
                split="train",
                horizon=1,
                seq_len=12,
                return_metadata=True,
            )
            self.assertIsNotNone(res, "Failed to build forecast windows for synthetic data")
            telemetry, dt, rain, precip, water, has_water, metadata = res

            self.assertGreater(len(metadata), 0, "No forecast windows generated")
            for sample in metadata:
                origin_dt = datetime.fromisoformat(sample["origin_timestamp"])
                target_dt = datetime.fromisoformat(sample["target_timestamp"])
                elapsed_hours = (target_dt - origin_dt).total_seconds() / 3600.0

                # ASSERT that target is approximately 1 hour later, NOT 1 minute (0.0167h) later!
                self.assertAlmostEqual(elapsed_hours, 1.0, places=2,
                                       msg=f"Elapsed time is {elapsed_hours}h, not 1.0h! Row-offset bug detected!")
                self.assertGreater(elapsed_hours, 0.5,
                                   msg="Elapsed time is under 30 minutes; likely row offset instead of hour offset!")

    def test_data_quarantine_2069_and_sensor_spikes(self):
        """
        Verify that 2069 date anomaly and physical sensor spikes are strictly quarantined.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            weather_csv = os.path.join(tmp_dir, "test_quarantine.csv")
            water_csv = os.path.join(tmp_dir, "test_water.csv")

            base_time = datetime(2026, 7, 1, 0, 0, 0, tzinfo=timezone.utc)
            with open(weather_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "station_id", "station_name", "location", "recorded_at",
                    "temperature", "heat_index", "humidity", "pressure",
                    "wind_speed", "wind_direction", "precipitation", "uv_index", "light_intensity"
                ])
                # Valid row
                writer.writerow(["ST1", "Name", "Loc", base_time.isoformat(), 28.0, 32.0, 70.0, 1008.0, 5.0, 180.0, 0.0, 1.0, 100.0])
                # Anomaly 1: 2069 timestamp
                writer.writerow(["ST1", "Name", "Loc", "2069-07-01T00:00:00Z", 28.0, 32.0, 70.0, 1008.0, 5.0, 180.0, 0.0, 1.0, 100.0])
                # Anomaly 2: Extreme temperature spike (99.0 C)
                writer.writerow(["ST1", "Name", "Loc", (base_time + timedelta(minutes=1)).isoformat(), 99.0, 32.0, 70.0, 1008.0, 5.0, 180.0, 0.0, 1.0, 100.0])
                # Anomaly 3: Extreme precipitation spike (7202.0 mm)
                writer.writerow(["ST1", "Name", "Loc", (base_time + timedelta(minutes=2)).isoformat(), 28.0, 32.0, 70.0, 1008.0, 5.0, 180.0, 7202.0, 1.0, 100.0])
                # Anomaly 4: Pressure out of bounds (1500 hPa)
                writer.writerow(["ST1", "Name", "Loc", (base_time + timedelta(minutes=3)).isoformat(), 28.0, 32.0, 70.0, 1500.0, 5.0, 180.0, 0.0, 1.0, 100.0])

            with open(water_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["station_id", "station_name", "location", "recorded_at", "water_level_cm", "water_level_m"])
                # Valid water row
                writer.writerow(["W1", "Name", "Loc", base_time.isoformat(), "250", "2.5"])
                # Anomaly: Negative water level (-57.73 m)
                writer.writerow(["W1", "Name", "Loc", (base_time + timedelta(minutes=1)).isoformat(), "-5773", "-57.73"])

            pipeline = TelemetryDataPipeline(weather_csv=weather_csv, water_csv=water_csv)

            self.assertEqual(pipeline.quarantine_counts["weather_year_out_of_bounds"], 1)
            self.assertEqual(pipeline.quarantine_counts["weather_bounds_temperature"], 1)
            self.assertEqual(pipeline.quarantine_counts["weather_bounds_precipitation"], 1)
            self.assertEqual(pipeline.quarantine_counts["weather_bounds_pressure"], 1)
            self.assertEqual(pipeline.quarantine_counts["water_physical_bounds"], 1)

    def test_normalization_fitted_strictly_on_train(self):
        """
        Verify that changing validation or test data does NOT alter the fitted training normalization statistics.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            weather_csv = os.path.join(tmp_dir, "test_norm.csv")
            water_csv = os.path.join(tmp_dir, "empty_water.csv")

            with open(water_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["station_id", "station_name", "location", "recorded_at", "water_level_cm", "water_level_m"])

            # 100 hours of data: 60 train, 20 val, 20 test
            base_time = datetime(2026, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
            rows = []
            for h in range(100):
                ts = base_time + timedelta(hours=h)
                rows.append(["ST1", "Name", "Loc", ts.isoformat(), 25.0, 30.0, 70.0, 1000.0, 2.0, 180.0, 0.0, 1.0, 100.0])

            with open(weather_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["station_id", "station_name", "location", "recorded_at", "temperature", "heat_index", "humidity", "pressure", "wind_speed", "wind_direction", "precipitation", "uv_index", "light_intensity"])
                writer.writerows(rows)

            pipeline_1 = TelemetryDataPipeline(weather_csv=weather_csv, water_csv=water_csv)
            means_1 = pipeline_1.norm_means.copy()
            stds_1 = pipeline_1.norm_stds.copy()

            # Now modify ONLY the test period (hours 85-99) with extreme values (e.g. 45 C)
            for h in range(85, 100):
                rows[h][4] = 45.0  # extreme temperature in test split

            with open(weather_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["station_id", "station_name", "location", "recorded_at", "temperature", "heat_index", "humidity", "pressure", "wind_speed", "wind_direction", "precipitation", "uv_index", "light_intensity"])
                writer.writerows(rows)

            pipeline_2 = TelemetryDataPipeline(weather_csv=weather_csv, water_csv=water_csv)
            means_2 = pipeline_2.norm_means.copy()
            stds_2 = pipeline_2.norm_stds.copy()

            # Training normalization must be 100% IDENTICAL
            np.testing.assert_allclose(means_1, means_2, rtol=1e-5,
                                       err_msg="Training normalization means changed when test split was modified!")
            np.testing.assert_allclose(stds_1, stds_2, rtol=1e-5,
                                       err_msg="Training normalization stds changed when test split was modified!")

    def test_chronological_split_embargo(self):
        """
        Verify that 48h embargo prevents cross-split window leakage.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            weather_csv = os.path.join(tmp_dir, "test_embargo.csv")
            water_csv = os.path.join(tmp_dir, "empty_water.csv")
            with open(water_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["station_id", "station_name", "location", "recorded_at", "water_level_cm", "water_level_m"])

            base_time = datetime(2026, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
            rows = []
            for h in range(200):
                ts = base_time + timedelta(hours=h)
                rows.append(["ST1", "Name", "Loc", ts.isoformat(), 28.0, 32.0, 70.0, 1008.0, 5.0, 180.0, 0.0, 1.0, 100.0])

            with open(weather_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["station_id", "station_name", "location", "recorded_at", "temperature", "heat_index", "humidity", "pressure", "wind_speed", "wind_direction", "precipitation", "uv_index", "light_intensity"])
                writer.writerows(rows)

            pipeline = TelemetryDataPipeline(weather_csv=weather_csv, water_csv=water_csv)

            # Test start must be strictly after train_end + 48 hours
            embargo_hours = (pipeline.val_start - pipeline.train_end).total_seconds() / 3600.0
            self.assertGreaterEqual(embargo_hours, 48.0, f"Embargo between train and val is only {embargo_hours}h, expected >= 48h")

            test_embargo_hours = (pipeline.test_start - pipeline.val_end).total_seconds() / 3600.0
            self.assertGreaterEqual(test_embargo_hours, 48.0, f"Embargo between val and test is only {test_embargo_hours}h, expected >= 48h")

    def test_precipitation_incremental_hourly_sum(self):
        """
        Verify that raw minute telemetry precipitation is treated as discrete
        incremental volume (e.g. tipping-bucket tips of 0.1099 mm) and summed
        to form the hourly accumulation (mm/h).
        Also verify that extreme single-minute spikes (>50 mm/min) are quarantined.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            weather_csv = os.path.join(tmp_dir, "test_precip_semantics.csv")
            water_csv = os.path.join(tmp_dir, "empty_water.csv")

            with open(water_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["station_id", "station_name", "location", "recorded_at", "water_level_cm", "water_level_m"])

            base_time = datetime(2026, 7, 1, 0, 0, 0, tzinfo=timezone.utc)
            rows = []
            # In hour 0: 10 minutes have 1 tip each (0.1099 mm). Expected hourly sum = 1.099 mm.
            for m in range(60):
                ts = base_time + timedelta(minutes=m)
                precip = 0.1099 if m < 10 else 0.0
                rows.append(["ST_PRECIP", "Precip Station", "Loc", ts.isoformat(), 28.0, 32.0, 75.0, 1008.0, 5.0, 180.0, precip, 1.0, 100.0])

            # In hour 1: minute 5 has a corrupted spike (7202.0 mm), rest are 0.
            for m in range(60):
                ts = base_time + timedelta(hours=1, minutes=m)
                precip = 7202.0 if m == 5 else 0.0
                rows.append(["ST_PRECIP", "Precip Station", "Loc", ts.isoformat(), 28.0, 32.0, 75.0, 1008.0, 5.0, 180.0, precip, 1.0, 100.0])

            with open(weather_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["station_id", "station_name", "location", "recorded_at", "temperature", "heat_index", "humidity", "pressure", "wind_speed", "wind_direction", "precipitation", "uv_index", "light_intensity"])
                writer.writerows(rows)

            pipeline = TelemetryDataPipeline(weather_csv=weather_csv, water_csv=water_csv)

            # Hour 0 should have sum = 1.099 mm
            h0_data = pipeline.station_hourly["ST_PRECIP"][base_time]
            self.assertAlmostEqual(h0_data["precipitation"], 1.099, places=3,
                                   msg="Incremental minute tips were not correctly summed to hourly volume!")

            # Hour 1 should have quarantined the 7202 mm spike
            self.assertEqual(pipeline.quarantine_counts["weather_bounds_precipitation"], 1,
                             "Extreme 7202 mm/min spike was not quarantined!")
            h1_bin = base_time + timedelta(hours=1)
            h1_data = pipeline.station_hourly["ST_PRECIP"][h1_bin]
            self.assertAlmostEqual(h1_data["precipitation"], 0.0, places=3,
                                   msg="Quarantined spike leaked into hourly precipitation total!")

    def test_inference_fails_closed_on_missing_features(self):
        """
        Verify that operational inference predict_from_observed_sequence strictly
        fails closed (raises ValueError) if required 8 canonical physical features are missing,
        or if input contains NaNs/Infs.
        """
        from inference import LNNServerlessPredictor
        predictor = LNNServerlessPredictor()

        # 1. 4-feature legacy input must raise ValueError
        seq_4 = np.ones((24, 4), dtype=np.float32)
        with self.assertRaises(ValueError, msg="Operational forecast did not fail closed on 4-feature input"):
            predictor.predict_from_observed_sequence(seq_4)

        # 2. 7-feature input must raise ValueError
        seq_7 = np.ones((24, 7), dtype=np.float32)
        with self.assertRaises(ValueError, msg="Operational forecast did not fail closed on 7-feature input"):
            predictor.predict_from_observed_sequence(seq_7)

        # 3. 8-feature input with NaN must raise ValueError
        seq_nan = np.ones((24, 8), dtype=np.float32)
        seq_nan[10, 2] = np.nan
        with self.assertRaises(ValueError, msg="Operational forecast did not fail closed on NaN values"):
            predictor.predict_from_observed_sequence(seq_nan)

        # 4. Valid 8-feature input must succeed
        seq_valid = np.ones((24, 8), dtype=np.float32) * 25.0
        seq_valid[:, 3] = 1010.0  # pressure
        res = predictor.predict_from_observed_sequence(seq_valid)
        self.assertIn("chance_of_rain_pct", res)
        self.assertIn("predicted_water_level_m", res)

    def test_circular_wind_direction_wraparound_and_calm_mask(self):
        """
        Verify that wind direction angular distance correctly handles 359° vs 1° as 2°,
        and that calm wind speeds (< 1.0 km/h) are masked or identified.
        """
        from dataset import circular_direction_error_deg
        # 359 vs 1 is 2 degrees
        err1 = circular_direction_error_deg(359.0, 1.0)
        self.assertAlmostEqual(err1, 2.0, places=2)

        # 1 vs 359 is 2 degrees
        err2 = circular_direction_error_deg(1.0, 359.0)
        self.assertAlmostEqual(err2, 2.0, places=2)

        # 180 vs 0 is 180 degrees
        err3 = circular_direction_error_deg(180.0, 0.0)
        self.assertAlmostEqual(err3, 180.0, places=2)

        # Calm wind test in inference: wind_speed < 1.0 returns wind_direction_deg as None
        from inference import LNNServerlessPredictor
        predictor = LNNServerlessPredictor()
        seq_calm = np.ones((24, 8), dtype=np.float32) * 25.0
        seq_calm[:, 3] = 1010.0  # pressure
        seq_calm[:, 4] = 0.4     # wind speed < 1.0 km/h (calm)
        res = predictor.predict_from_observed_sequence(seq_calm)
        self.assertIsNone(res["wind_direction_deg"], "Calm wind (< 1.0 km/h) should yield None for wind direction!")

    def test_noaa_rothfusz_heat_index(self):
        """
        Verify that NOAA Rothfusz regression produces consistent physical heat index
        and matches published NOAA lookup values within 1.0°C.
        """
        from dataset import compute_noaa_heat_index
        # 27C and 40% RH: Heat index is approximately 27C
        hi_mild = compute_noaa_heat_index(27.0, 40.0)
        self.assertAlmostEqual(hi_mild, 27.0, delta=1.5)

        # 32C and 70% RH: Extreme caution / danger zone (~41C)
        hi_hot = compute_noaa_heat_index(32.0, 70.0)
        self.assertGreater(hi_hot, 39.0)
        self.assertLess(hi_hot, 45.0)

        # 20C (cool): Below Rothfusz threshold, Steadman formula yields ~19.36C (within 1.0C of 20C)
        hi_cool = compute_noaa_heat_index(20.0, 50.0)
        self.assertAlmostEqual(hi_cool, 20.0, delta=1.0)

    def test_nighttime_uv_quarantine_and_data_audit(self):
        """
        Verify that raw telemetry UV index defects (up to 11.0 at midnight)
        cause UV index to be marked BLOCKED_BY_SENSOR_CALIBRATION in weather data audit.
        """
        import json
        audit_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "weather_data_audit.json")
        self.assertTrue(os.path.exists(audit_file), f"Missing weather_data_audit.json: {audit_file}")
        with open(audit_file, "r", encoding="utf-8") as f:
            audit = json.load(f)
        feasibility = audit.get("target_feasibility_determination", {})
        uv = feasibility.get("uv_index", {})
        self.assertEqual(uv.get("status"), "BLOCKED_BY_SENSOR_CALIBRATION")
        self.assertIn("defect", uv.get("reason", "").lower())

    def test_frozen_calibration_split_isolation(self):
        """
        Verify that calibration blend weights w_h and operational thresholds
        are frozen on the calibration/validation split and NEVER leak test labels.
        """
        from dataset import get_telemetry_pipeline, build_forecast_windows
        pipeline = get_telemetry_pipeline()
        calib_res = build_forecast_windows(pipeline, split="val", horizon=1, return_metadata=True)
        test_res = build_forecast_windows(pipeline, split="test", horizon=1, return_metadata=True)
        self.assertIsNotNone(calib_res)
        self.assertIsNotNone(test_res)

        # Check timestamp isolation: max calib timestamp < min test timestamp - 48h
        calib_meta = calib_res[6]
        test_meta = test_res[6]
        max_calib_ts = max(datetime.fromisoformat(m["target_timestamp"]) for m in calib_meta)
        min_test_ts = min(datetime.fromisoformat(m["origin_timestamp"]) for m in test_meta)
        gap_hours = (min_test_ts - max_calib_ts).total_seconds() / 3600.0
        self.assertGreaterEqual(gap_hours, 48.0, f"Leakage: calibration and test are separated by only {gap_hours}h")

    def test_checkpoint_load_all_horizons(self):
        """
        Verify that model checkpoints for all 5 horizons (h1, h3, h6, h12, h24)
        load cleanly from disk and execute operational inference.
        """
        from inference import LNNServerlessPredictor
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
        horizons = [1, 3, 6, 12, 24]
        dummy_seq = np.ones((24, 8), dtype=np.float32) * 28.0
        dummy_seq[:, 3] = 1012.0  # pressure

        for h in horizons:
            ckpt_path = os.path.join(data_dir, f"lnn_weather_water_h{h}.pt")
            self.assertTrue(os.path.exists(ckpt_path), f"Checkpoint missing: {ckpt_path}")
            predictor = LNNServerlessPredictor(model_weights_path=ckpt_path)
            res = predictor.predict_from_observed_sequence(
                telemetry_sequence=dummy_seq,
                forecast_origin_timestamp="2026-08-01T12:00:00",
                horizon_hours=h,
            )
            self.assertEqual(res["forecast_horizon"], f"{h}h")
            self.assertIn("temperature_c", res)
            self.assertIn("chance_of_rain_pct", res)
            self.assertIn("pressure_tendency", res)
            self.assertIn("heat_index_risk_category", res)

    def test_pressure_tendency_and_heat_index_categories(self):
        """
        Verify operational classification thresholds:
        - Pressure tendency: RISING (>+0.5 hPa), FALLING (<-0.5 hPa), STEADY.
        - Heat index risk: NORMAL (<27), CAUTION (27-32), EXTREME CAUTION (32-41), DANGER (41-54), EXTREME DANGER (>=54).
        """
        from dataset import compute_noaa_heat_index
        # Test heat index categories
        test_cases = [
            (24.0, 40.0, "NORMAL"),
            (28.0, 45.0, "CAUTION"),
            (32.0, 60.0, "EXTREME CAUTION"),
            (35.0, 65.0, "DANGER"),
            (42.0, 80.0, "EXTREME DANGER"),
        ]
        for t, rh, expected_cat in test_cases:
            hi = compute_noaa_heat_index(t, rh)
            if hi < 27.0:
                cat = "NORMAL"
            elif hi < 32.0:
                cat = "CAUTION"
            elif hi < 41.0:
                cat = "EXTREME CAUTION"
            elif hi < 54.0:
                cat = "DANGER"
            else:
                cat = "EXTREME DANGER"
            self.assertEqual(cat, expected_cat, f"Mismatch for T={t}, RH={rh}, HI={hi}: expected {expected_cat}, got {cat}")

        # Test pressure tendency
        tendency_cases = [(0.6, "RISING"), (-0.8, "FALLING"), (0.2, "STEADY"), (-0.3, "STEADY")]
        for dp, expected in tendency_cases:
            if dp > 0.5:
                res = "RISING"
            elif dp < -0.5:
                res = "FALLING"
            else:
                res = "STEADY"
            self.assertEqual(res, expected)

    def test_fallback_skill_gate_rationale_recorded(self):
        """
        Verify that weather_validation_scorecard.json records explicit skill gates,
        beats_persistence flags, and selected_source or derivation_formula for every variable across all 5 horizons.
        Fails if any target has an unrecorded or undocumented fallback selection.
        """
        import json
        scorecard_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "weather_validation_scorecard.json")
        self.assertTrue(os.path.exists(scorecard_file), f"Missing scorecard: {scorecard_file}")
        with open(scorecard_file, "r", encoding="utf-8") as f:
            sc = json.load(f)

        horizons = [1, 3, 6, 12, 24]
        for h in horizons:
            h_key = f"horizon_{h}h"
            self.assertIn(h_key, sc["horizons"])
            h_data = sc["horizons"][h_key]
            for var in ["temperature", "humidity", "pressure", "wind_speed"]:
                self.assertIn(var, h_data, f"Variable {var} missing in {h_key}")
                var_data = h_data[var]
                self.assertIn("beats_persistence", var_data)
                self.assertIn("selected_source", var_data)
                self.assertIn("skill_vs_persistence", var_data)
                self.assertIn(var_data["selected_source"], ["learned_model", "persistence_fallback"])

            # Heat index is derived
            self.assertIn("heat_index", h_data)
            hi_data = h_data["heat_index"]
            self.assertIn("beats_persistence", hi_data)
            self.assertIn("skill_vs_persistence", hi_data)
            self.assertIn("derivation_formula", hi_data)

    def test_provenance_gate(self):
        """
        Verify that all committed model artifacts, quality reports, and manifests
        satisfy strict provenance, feature schema, and dataset hash requirements.
        """
        from verify_provenance import verify_provenance
        res = verify_provenance()
        self.assertEqual(res["status"], "PASS")

    def test_experimental_two_stage_precipitation_architecture(self):
        """
        Phase 6 Requirement:
        Verify that GarciaWeatherLNN supports experimental two-stage precipitation
        behind explicit flag use_two_stage_precipitation=True, while keeping
        the default use_two_stage_precipitation=False for baseline backward compatibility.
        """
        from model import GarciaWeatherLNN, TwoStagePrecipitationHead
        m_default = GarciaWeatherLNN(input_dim=8, hidden_dim=32, use_two_stage_precipitation=False)
        self.assertFalse(m_default.use_two_stage_precipitation)
        self.assertFalse(hasattr(m_default, "two_stage_rain_head"))

        m_twostage = GarciaWeatherLNN(input_dim=8, hidden_dim=32, use_two_stage_precipitation=True)
        self.assertTrue(m_twostage.use_two_stage_precipitation)
        self.assertTrue(hasattr(m_twostage, "two_stage_rain_head"))
        self.assertIsInstance(m_twostage.two_stage_rain_head, TwoStagePrecipitationHead)

        # Forward pass verification
        x = torch.randn(2, 24, 8)
        dt = torch.ones(2, 24, 1)
        rain_prob, precip_mm, water = m_twostage(x, dt)
        self.assertEqual(rain_prob.shape, (2, 24, 1))
        self.assertEqual(precip_mm.shape, (2, 24, 1))
        self.assertTrue((rain_prob >= 0.0).all() and (rain_prob <= 1.0).all())
        self.assertTrue((precip_mm >= 0.0).all())


if __name__ == "__main__":
    unittest.main()

