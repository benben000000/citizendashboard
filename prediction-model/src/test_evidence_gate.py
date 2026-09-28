"""Tests for the read-only release gate; no trained weights or telemetry needed."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from check_evidence_gate import HORIZONS, check


def fixture(root):
    root = Path(root)
    payload = b'fixture: not real forecast evidence'
    (root / 'pairs.csv').write_bytes(payload)
    evidence = {'evidence_filename': 'pairs.csv', 'evidence_sha256': hashlib.sha256(payload).hexdigest()}
    intervals = {}
    events = {}
    scores = {}
    calibration = {}
    for horizon in HORIZONS:
        key = f'horizon_{horizon}h'
        intervals[key] = dict(evidence, temperature_sample_count=100, temperature_covered_count=80, temperature_coverage_80=80.0)
        events[key] = dict(evidence, true_positives=8, false_negatives=2, false_positives=1, observed_station_days=2.0, heavy_rain_recall=0.8, false_alarm_rate_per_day=0.5)
        scores[key] = {'temperature_coverage_80': 80.0}
        calibration[key] = {'brier_score': 0.1}
    documents = {
        'uncertainty_report.json': {'status': 'PASS', 'target_nominal_coverage_pct': 80, 'horizons': intervals},
        'anomaly_report.json': {'status': 'PASS', 'false_alarm_budget_per_day': 2, 'horizons': events},
        'predictive_quality_scorecard.json': {'interval_metrics': scores, 'calibration_metrics': calibration},
    }
    for name, document in documents.items():
        (root / name).write_text(json.dumps(document), encoding='utf-8')
    return documents


class EvidenceGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.docs = fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def save(self, name):
        (self.root / name).write_text(json.dumps(self.docs[name]), encoding='utf-8')

    def test_consistent_fixture_passes_structural_gate_only(self):
        self.assertEqual(check(self.root), [])

    def test_unsupported_coverage_fails(self):
        self.docs['uncertainty_report.json']['horizons']['horizon_3h']['temperature_covered_count'] = 64
        self.save('uncertainty_report.json')
        self.assertTrue(any('horizon_3h' in error for error in check(self.root)))

    def test_perfect_recall_without_counts_fails(self):
        event = self.docs['anomaly_report.json']['horizons']['horizon_1h']
        del event['true_positives']
        event['heavy_rain_recall'] = 1.0
        self.save('anomaly_report.json')
        self.assertTrue(any('missing reviewed event counts' in error for error in check(self.root)))

    def test_tampered_support_file_fails(self):
        (self.root / 'pairs.csv').write_bytes(b'tampered')
        self.assertTrue(any('evidence hash mismatch' in error for error in check(self.root)))

    def test_nonzero_brier_with_zero_ci_fails(self):
        self.docs['predictive_quality_scorecard.json']['confidence_intervals'] = {'horizon_1h': {'rain_brier_ci_95': [0.0, 0.0]}}
        self.save('predictive_quality_scorecard.json')
        self.assertTrue(any('zero-width zero Brier CI' in error for error in check(self.root)))


if __name__ == '__main__':
    unittest.main()
