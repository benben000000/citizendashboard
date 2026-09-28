"""Tests for independently recomputing held-out paired-row evidence."""
import csv
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from recompute_paired_evidence import COLUMNS, HORIZONS, compare, evaluate


def rows():
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    result = []
    for horizon in HORIZONS:
        for offset in (0, 1):
            issue = base + timedelta(days=offset)
            result.append(dict(station_id='station-a', issue_timestamp_utc=issue.isoformat(), target_timestamp_utc=(issue + timedelta(hours=horizon)).isoformat(), horizon_hours=str(horizon), split_name='test', label_quality_status='valid', forecast_temperature_p10='20', forecast_temperature_p90='30', observed_temperature_c='25' if offset == 0 else '35', rain_probability='0.8' if offset == 0 else '0.2', rain_observed='1' if offset == 0 else '0'))
    return result


class PairedEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'pairs.csv'
        self.data = rows()

    def tearDown(self):
        self.temp.cleanup()

    def save(self):
        with self.path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(self.data)

    def test_recomputes_each_horizon(self):
        self.save()
        metrics = evaluate(self.path, min_count=2)
        self.assertEqual(metrics['horizon_1h']['temperature_coverage_80'], 50.0)
        self.assertAlmostEqual(metrics['horizon_1h']['rain_brier_score'], 0.04)

    def test_rejects_future_time_misalignment(self):
        self.data[0]['target_timestamp_utc'] = self.data[0]['issue_timestamp_utc']
        self.save()
        with self.assertRaisesRegex(ValueError, 'issue/target/horizon mismatch'):
            evaluate(self.path, min_count=2)

    def test_rejects_duplicate_pairs(self):
        self.data.append(self.data[0].copy())
        self.save()
        with self.assertRaisesRegex(ValueError, 'duplicate forecast/observation pair'):
            evaluate(self.path, min_count=2)

    def test_rejects_bad_label(self):
        self.data[0]['label_quality_status'] = 'unverified'
        self.save()
        with self.assertRaisesRegex(ValueError, 'valid label required'):
            evaluate(self.path, min_count=2)

    def test_comparison_catches_inconsistent_report(self):
        self.save()
        metrics = evaluate(self.path, min_count=2)
        card = {'interval_metrics': {}, 'calibration_metrics': {}}
        report = {'horizons': {}}
        self.assertTrue(compare(metrics, card, report))


if __name__ == '__main__':
    unittest.main()
