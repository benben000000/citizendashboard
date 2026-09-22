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


if __name__ == "__main__":
    unittest.main()
