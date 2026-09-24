"""
Regression Test Suite for Operational Monitoring Engine and Five-Horizon Evaluation.

Validates:
  Test 1: Current production schema parses without error.
  Test 2: All five forecast horizons (1h, 3h, 6h, 12h, 24h) are present.
  Test 3: Sample counts match independent CSV verification.
  Test 4: Correct temperature mapping (no default-derived values).
  Test 5: Correct rain probability mapping and Brier score calculation.
  Test 6: Missing required columns fail closed with clear schema errors.
  Test 7: Invalid horizons fail closed and are never defaulted to +1h.
  Test 8: Malformed numeric values (NaN, Inf, strings) fail closed or are counted explicitly.
  Test 9: Committed CSV metrics are finite, plausible, and not silently defaulted to zero.
  Test 10: In-memory evaluation is strictly read-only and leaves the working tree clean.
"""

import os
import sys
import csv
import math
import tempfile
import unittest
import subprocess
from collections import Counter
from typing import Dict, Any, List

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

DATA_DIR = os.path.join(os.path.dirname(SRC_DIR), "data")
COMMITTED_PREDICTIONS_LOG = os.path.join(DATA_DIR, "test_predictions_log.csv")

from monitoring import (
    CANONICAL_HORIZONS,
    PREDICTION_LOG_REQUIRED_COLUMNS,
    PREDICTION_LOG_SCHEMA_VERSION,
    MONITORING_VERSION,
    PredictionLogSchemaError,
    PredictionLogDataError,
    validate_prediction_log_headers,
    parse_prediction_log_row,
    load_and_validate_prediction_log,
    run_monitoring_evaluation,
    compute_file_sha256,
)


def make_valid_row(
    horizon_hours: int = 1,
    actual_lead_hours: float = 1.0,
    actual_temp: float = 25.0,
    pred_temp: float = 27.0,
    persist_temp: float = 24.0,
    actual_rain: float = 1.0,
    hybrid_rain_prob: float = 0.8,
    mf1_rain_prob: float = 0.75,
    actual_precip_mm: float = 1.5,
    mf1_precip_mm: float = 1.2,
    persist_rain: float = 1.0,
) -> Dict[str, str]:
    """Helper to construct a valid dictionary row with all required columns."""
    return {
        "station_id": "03pqkGAj",
        "origin_timestamp": "2026-08-19T00:00:00+00:00",
        "target_timestamp": f"2026-08-19T{horizon_hours:02d}:00:00+00:00",
        "horizon_hours": str(horizon_hours),
        "actual_lead_hours": str(actual_lead_hours),
        "actual_rain": str(actual_rain),
        "actual_precip_mm": str(actual_precip_mm),
        "actual_temp": str(actual_temp),
        "actual_humidity": "85.0",
        "actual_pressure": "1010.5",
        "actual_wind_speed": "5.2",
        "actual_wind_dir_deg": "180.0",
        "actual_heat_index": "28.0",
        "pred_temp": str(pred_temp),
        "pred_humidity": "83.5",
        "pred_pressure": "1010.8",
        "pred_wind_speed": "4.8",
        "pred_wind_dir_deg": "175.0",
        "derived_heat_index": "28.5",
        "mf1_rain_prob": str(mf1_rain_prob),
        "hybrid_rain_prob": str(hybrid_rain_prob),
        "mf1_precip_mm": str(mf1_precip_mm),
        "persist_temp": str(persist_temp),
        "persist_humidity": "86.0",
        "persist_pressure": "1010.2",
        "persist_wind_speed": "5.0",
        "persist_rain": str(persist_rain),
        "actual_water_level": "",
        "mf1_water_level": "",
        "persist_water": "",
    }


def write_fixture_csv(filepath: str, rows: List[Dict[str, Any]], fieldnames: List[str] = None):
    """Write rows to a CSV file."""
    if fieldnames is None:
        fieldnames = list(PREDICTION_LOG_REQUIRED_COLUMNS.keys())
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


class TestMonitoringEngine(unittest.TestCase):

    def setUp(self):
        self.assertTrue(
            os.path.exists(COMMITTED_PREDICTIONS_LOG),
            f"Committed log must exist at: {COMMITTED_PREDICTIONS_LOG}",
        )

    # -----------------------------------------------------------------------
    # Test 1: Current production schema parses
    # -----------------------------------------------------------------------
    def test_01_current_production_schema_parses(self):
        """Confirm that the committed test_predictions_log.csv matches the required schema and parses without errors."""
        horizons_data, malformed_count, malformed_reasons = load_and_validate_prediction_log(
            COMMITTED_PREDICTIONS_LOG,
            canonical_horizons=CANONICAL_HORIZONS,
            max_malformed_rows=0,
        )
        self.assertGreater(len(horizons_data), 0, "Parsed prediction log must contain data.")
        self.assertEqual(malformed_count, 0, f"Expected 0 malformed rows, got {malformed_count}: {malformed_reasons[:3]}")
        total_rows = sum(len(v) for v in horizons_data.values())
        self.assertEqual(total_rows, 13411, f"Expected exactly 13,411 rows in committed log, got {total_rows}")

    # -----------------------------------------------------------------------
    # Test 2: All five horizons are present
    # -----------------------------------------------------------------------
    def test_02_all_five_horizons_present(self):
        """Confirm monitoring evaluates exactly the five canonical horizons."""
        report = run_monitoring_evaluation(
            data_dir=DATA_DIR,
            predictions_log_path=COMMITTED_PREDICTIONS_LOG,
            horizons=[1, 3, 6, 12, 24],
            require_all_horizons=True,
        )
        expected_keys = {"horizon_1h", "horizon_3h", "horizon_6h", "horizon_12h", "horizon_24h"}
        self.assertEqual(set(report["horizons_performance"].keys()), expected_keys)
        self.assertEqual(report["evaluated_horizons_hours"], [1, 3, 6, 12, 24])
        self.assertEqual(report["missing_horizons_hours"], [])
        self.assertEqual(report["prediction_log_schema_version"], PREDICTION_LOG_SCHEMA_VERSION)
        self.assertEqual(report["monitoring_version"], MONITORING_VERSION)

    # -----------------------------------------------------------------------
    # Test 3: Correct sample counts
    # -----------------------------------------------------------------------
    def test_03_correct_sample_counts(self):
        """Compare each horizon's monitoring count with an independent CSV count."""
        independent_counts = Counter()
        with open(COMMITTED_PREDICTIONS_LOG, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                independent_counts[int(float(row["horizon_hours"]))] += 1

        report = run_monitoring_evaluation(
            data_dir=DATA_DIR,
            predictions_log_path=COMMITTED_PREDICTIONS_LOG,
            horizons=[1, 3, 6, 12, 24],
        )

        for h in [1, 3, 6, 12, 24]:
            h_key = f"horizon_{h}h"
            monitored_count = report["horizons_performance"][h_key]["sample_count"]
            expected = independent_counts[h]
            self.assertEqual(
                monitored_count,
                expected,
                f"Sample count mismatch for +{h}h: monitored {monitored_count} vs independent {expected}",
            )
        # Expected canonical counts
        self.assertEqual(independent_counts[1], 2820)
        self.assertEqual(independent_counts[3], 2784)
        self.assertEqual(independent_counts[6], 2731)
        self.assertEqual(independent_counts[12], 2633)
        self.assertEqual(independent_counts[24], 2443)

    # -----------------------------------------------------------------------
    # Test 4: Correct temperature mapping
    # -----------------------------------------------------------------------
    def test_04_correct_temperature_mapping(self):
        """Construct a fixture where actual_temp=25, pred_temp=27, persist_temp=24, and verify MAE=2, not a default."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = os.path.join(tmp_dir, "test_temp_mapping.csv")
            rows = [
                make_valid_row(horizon_hours=1, actual_lead_hours=1.0, actual_temp=25.0, pred_temp=27.0, persist_temp=24.0)
            ]
            write_fixture_csv(csv_path, rows)

            report = run_monitoring_evaluation(
                data_dir=DATA_DIR,
                predictions_log_path=csv_path,
                horizons=[1],
                allow_missing_horizon=True,
            )
            temp_res = report["horizons_performance"]["horizon_1h"]["continuous_variables"]["temperature"]
            self.assertEqual(temp_res["sample_count"], 1)
            self.assertAlmostEqual(temp_res["mae"], 2.0, places=4)
            self.assertAlmostEqual(temp_res["bias"], 2.0, places=4)
            self.assertAlmostEqual(temp_res["persistence_mae"], 1.0, places=4)
            self.assertAlmostEqual(temp_res["persistence_rmse"], 1.0, places=4)

    # -----------------------------------------------------------------------
    # Test 5: Correct rain mapping
    # -----------------------------------------------------------------------
    def test_05_correct_rain_mapping(self):
        """Construct a fixture with known actual_rain and hybrid_rain_prob, independently calculate Brier score."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = os.path.join(tmp_dir, "test_rain_mapping.csv")
            # 4 rows:
            # Row 1: y=1, p=0.8 -> (0.8 - 1)^2 = 0.04
            # Row 2: y=0, p=0.2 -> (0.2 - 0)^2 = 0.04
            # Row 3: y=1, p=0.9 -> (0.9 - 1)^2 = 0.01
            # Row 4: y=0, p=0.1 -> (0.1 - 0)^2 = 0.01
            # Mean Brier = 0.10 / 4 = 0.025
            rows = [
                make_valid_row(horizon_hours=3, actual_lead_hours=3.0, actual_rain=1.0, hybrid_rain_prob=0.8),
                make_valid_row(horizon_hours=3, actual_lead_hours=3.0, actual_rain=0.0, hybrid_rain_prob=0.2),
                make_valid_row(horizon_hours=3, actual_lead_hours=3.0, actual_rain=1.0, hybrid_rain_prob=0.9),
                make_valid_row(horizon_hours=3, actual_lead_hours=3.0, actual_rain=0.0, hybrid_rain_prob=0.1),
            ]
            write_fixture_csv(csv_path, rows)

            report = run_monitoring_evaluation(
                data_dir=DATA_DIR,
                predictions_log_path=csv_path,
                horizons=[3],
                allow_missing_horizon=True,
            )
            rain_res = report["horizons_performance"]["horizon_3h"]["rain_occurrence"]
            self.assertEqual(rain_res["total"], 4)
            self.assertAlmostEqual(rain_res["brier_score"], 0.025, places=4)
            self.assertEqual(rain_res["tp"], 2)
            self.assertEqual(rain_res["tn"], 2)
            self.assertEqual(rain_res["fp"], 0)
            self.assertEqual(rain_res["fn"], 0)
            self.assertAlmostEqual(rain_res["accuracy"], 1.0, places=4)

    # -----------------------------------------------------------------------
    # Test 6: Missing columns fail closed
    # -----------------------------------------------------------------------
    def test_06_missing_columns_fail_closed(self):
        """Remove actual_temp, horizon_hours, and hybrid_rain_prob individually; confirm schema error."""
        critical_columns = ["actual_temp", "horizon_hours", "hybrid_rain_prob"]

        for col in critical_columns:
            with tempfile.TemporaryDirectory() as tmp_dir:
                csv_path = os.path.join(tmp_dir, f"missing_{col}.csv")
                fields = [c for c in PREDICTION_LOG_REQUIRED_COLUMNS.keys() if c != col]
                row = make_valid_row()
                del row[col]
                write_fixture_csv(csv_path, [row], fieldnames=fields)

                with self.assertRaises(PredictionLogSchemaError) as ctx:
                    load_and_validate_prediction_log(csv_path)
                self.assertIn(col, str(ctx.exception), f"Error message should mention missing column '{col}'")

    # -----------------------------------------------------------------------
    # Test 7: Invalid horizon fails closed
    # -----------------------------------------------------------------------
    def test_07_invalid_horizon_fails_closed(self):
        """Use horizon 2 or empty value and confirm it is rejected rather than assigned to 1h."""
        for invalid_h in ["2", "4", "7", ""]:
            with tempfile.TemporaryDirectory() as tmp_dir:
                csv_path = os.path.join(tmp_dir, f"invalid_h_{invalid_h}.csv")
                row = make_valid_row()
                row["horizon_hours"] = invalid_h
                write_fixture_csv(csv_path, [row])

                with self.assertRaises(PredictionLogDataError) as ctx:
                    load_and_validate_prediction_log(csv_path, max_malformed_rows=0)
                self.assertIn("malformed rows", str(ctx.exception))

    # -----------------------------------------------------------------------
    # Test 8: Malformed numeric values fail closed or are counted explicitly
    # -----------------------------------------------------------------------
    def test_08_malformed_numeric_values_fail_closed_or_counted_explicitly(self):
        """Use NaN, Inf, and nonnumeric text; confirm behavior matches malformed-row policy."""
        malformed_values = ["NaN", "Inf", "-Inf", "corrupted_text"]

        for bad_val in malformed_values:
            with tempfile.TemporaryDirectory() as tmp_dir:
                csv_path = os.path.join(tmp_dir, "bad_numeric.csv")
                row = make_valid_row()
                row["actual_temp"] = bad_val
                write_fixture_csv(csv_path, [row])

                # With max_malformed_rows=0, must fail closed
                with self.assertRaises(PredictionLogDataError) as ctx:
                    load_and_validate_prediction_log(csv_path, max_malformed_rows=0)
                self.assertIn("malformed rows", str(ctx.exception))

                # With max_malformed_rows=5, must record malformed count
                data, malformed_count, reasons = load_and_validate_prediction_log(
                    csv_path, max_malformed_rows=5
                )
                self.assertEqual(malformed_count, 1)
                self.assertEqual(len(data.get(1, [])), 0)

    # -----------------------------------------------------------------------
    # Test 9: Current CSV is not silently defaulted
    # -----------------------------------------------------------------------
    def test_09_current_csv_not_silently_defaulted(self):
        """Run monitoring against committed log and assert plausible, non-default metrics."""
        report = run_monitoring_evaluation(
            data_dir=DATA_DIR,
            predictions_log_path=COMMITTED_PREDICTIONS_LOG,
            horizons=[1, 3, 6, 12, 24],
            require_all_horizons=True,
        )

        self.assertEqual(report["malformed_row_count"], 0)
        self.assertEqual(len(report["horizons_performance"]), 5)

        for h in [1, 3, 6, 12, 24]:
            h_res = report["horizons_performance"][f"horizon_{h}h"]

            # Temperature MAE must be realistic (not 0.0 and not 27.36)
            temp_mae = h_res["continuous_variables"]["temperature"]["mae"]
            self.assertTrue(math.isfinite(temp_mae))
            self.assertGreater(temp_mae, 0.2, f"+{h}h Temp MAE too small: {temp_mae}")
            self.assertLess(temp_mae, 3.0, f"+{h}h Temp MAE too large (defaulted?): {temp_mae}")

            # Rain Brier score must be realistic (not 0.0 and not > 0.5)
            rain_brier = h_res["rain_occurrence"]["brier_score"]
            self.assertTrue(math.isfinite(rain_brier))
            self.assertGreater(rain_brier, 0.05, f"+{h}h Rain Brier score suspiciously zero: {rain_brier}")
            self.assertLess(rain_brier, 0.35, f"+{h}h Rain Brier score too high: {rain_brier}")

            # Humidity MAE must be plausible
            rh_mae = h_res["continuous_variables"]["humidity"]["mae"]
            self.assertTrue(math.isfinite(rh_mae))
            self.assertGreater(rh_mae, 1.0)
            self.assertLess(rh_mae, 15.0)

            # Pressure MAE must be plausible
            p_mae = h_res["continuous_variables"]["pressure"]["mae"]
            self.assertTrue(math.isfinite(p_mae))
            self.assertGreater(p_mae, 0.1)
            self.assertLess(p_mae, 5.0)

            # Persistence comparison must be present and well-formed
            p_comp = h_res["persistence_comparison"]
            self.assertIsNotNone(p_comp["temperature_persistence_mae"])
            self.assertIsNotNone(p_comp["rain_persistence_brier_score"])
            self.assertEqual(p_comp["precipitation_amount_persistence"], "UNAVAILABLE_IN_LOG")

    # -----------------------------------------------------------------------
    # Test 10: Read-only behavior
    # -----------------------------------------------------------------------
    def test_10_read_only_behavior(self):
        """Run monitoring without --output and verify no tracked or data files are modified."""
        # Check git status for prediction-model/data
        repo_root = os.path.dirname(os.path.dirname(SRC_DIR))
        status_before = subprocess.run(
            ["git", "status", "--porcelain", "prediction-model/data"],
            cwd=repo_root,
            capture_output=True,
            text=True,
        ).stdout.strip()

        # Execute monitoring in memory without output file
        run_monitoring_evaluation(
            data_dir=DATA_DIR,
            predictions_log_path=COMMITTED_PREDICTIONS_LOG,
            output_path=None,
            horizons=[1, 3, 6, 12, 24],
        )

        status_after = subprocess.run(
            ["git", "status", "--porcelain", "prediction-model/data"],
            cwd=repo_root,
            capture_output=True,
            text=True,
        ).stdout.strip()

        self.assertEqual(
            status_before,
            status_after,
            f"Monitoring execution altered prediction-model/data!\nBefore: {status_before}\nAfter: {status_after}",
        )


if __name__ == "__main__":
    unittest.main()
