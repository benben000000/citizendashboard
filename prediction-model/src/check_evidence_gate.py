"""Read-only release gate for the checked-in candidate evidence.

This checks report consistency and the presence/hash of supporting files; it does
not independently prove forecast skill. A failed or missing check blocks promotion.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

HORIZONS = (1, 3, 6, 12, 24)
DATA_DIR = Path(__file__).resolve().parents[1] / 'data' / 'candidate_artifacts'


def _json(directory, name):
    path = directory / name
    with path.open(encoding='utf-8') as handle:
        return json.load(handle)


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _support(directory, record, errors, label):
    name = record.get('evidence_filename')
    expected = record.get('evidence_sha256')
    if not isinstance(name, str) or Path(name).name != name or not name or not isinstance(expected, str) or len(expected) != 64:
        errors.append(f'{label}: missing or unsafe evidence file/hash')
        return
    try:
        int(expected, 16)
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
    except (OSError, ValueError):
        errors.append(f'{label}: evidence file cannot be verified')
        return
    if actual != expected.lower():
        errors.append(f'{label}: evidence hash mismatch')


def check(directory):
    """Return blockers; never infer a PASS from missing or invalid evidence."""
    directory = Path(directory)
    errors = []
    try:
        uncertainty = _json(directory, 'uncertainty_report.json')
        anomaly = _json(directory, 'anomaly_report.json')
        scorecard = _json(directory, 'predictive_quality_scorecard.json')
    except (OSError, ValueError) as exc:
        return [f'required report missing or invalid: {exc}']
    nominal = uncertainty.get('target_nominal_coverage_pct')
    if not _number(nominal) or not 0 < nominal < 100:
        errors.append('uncertainty: invalid nominal coverage')
    for horizon in HORIZONS:
        key = f'horizon_{horizon}h'
        interval = uncertainty.get('horizons', {}).get(key, {})
        label = f'uncertainty {key}'
        count = interval.get('temperature_sample_count')
        covered = interval.get('temperature_covered_count')
        reported = interval.get('temperature_coverage_80')
        if not isinstance(count, int) or isinstance(count, bool) or count < 100 or not isinstance(covered, int) or isinstance(covered, bool) or not 0 <= covered <= count or not _number(reported):
            errors.append(f'{label}: missing/invalid counts or coverage (minimum 100 pairs)')
        else:
            measured = 100 * covered / count
            if not math.isclose(measured, reported, abs_tol=0.02):
                errors.append(f'{label}: reported coverage disagrees with counts')
            if _number(nominal) and measured < nominal - 5:
                errors.append(f'{label}: {measured:.2f}% is below {nominal - 5:.2f}% minimum')
        _support(directory, interval, errors, label)
        card = scorecard.get('interval_metrics', {}).get(key, {}).get('temperature_coverage_80')
        if not _number(card) or not _number(reported) or not math.isclose(card, reported, abs_tol=0.02):
            errors.append(f'{label}: report/scorecard coverage mismatch')
        event = anomaly.get('horizons', {}).get(key, {})
        label = f'anomaly {key}'
        tp, fn, fp, days = (event.get(field) for field in ('true_positives', 'false_negatives', 'false_positives', 'observed_station_days'))
        recall, rate = event.get('heavy_rain_recall'), event.get('false_alarm_rate_per_day')
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in (tp, fn, fp)) or not _number(days) or days <= 0 or not _number(recall) or not _number(rate) or tp + fn == 0:
            errors.append(f'{label}: missing reviewed event counts, exposure, or measured rates')
        else:
            if not math.isclose(recall, tp / (tp + fn), abs_tol=1e-6):
                errors.append(f'{label}: recall disagrees with event counts')
            if not math.isclose(rate, fp / days, abs_tol=1e-6):
                errors.append(f'{label}: false-alarm rate disagrees with exposure')
            budget = anomaly.get('false_alarm_budget_per_day')
            if not _number(budget) or fp / days > budget:
                errors.append(f'{label}: false-alarm budget missing or exceeded')
        _support(directory, event, errors, label)
    for report, name in ((uncertainty, 'uncertainty'), (anomaly, 'anomaly')):
        if report.get('status') != 'PASS':
            errors.append(f'{name}: not approved for release')
    for key, values in scorecard.get('confidence_intervals', {}).items():
        bounds = values.get('rain_brier_ci_95')
        brier = scorecard.get('calibration_metrics', {}).get(key, {}).get('brier_score')
        if bounds == [0.0, 0.0] and _number(brier) and brier > 0:
            errors.append(f'{key}: zero-width zero Brier CI contradicts nonzero score')
    return errors


def main():
    parser = argparse.ArgumentParser(description='Block unsupported candidate evidence claims')
    parser.add_argument('--directory', type=Path, default=DATA_DIR)
    args = parser.parse_args()
    errors = check(args.directory)
    for error in errors:
        print('BLOCK:', error)
    if errors:
        print(f'NO-GO: {len(errors)} evidence blocker(s); no artifacts changed')
        return 1
    print('Structural evidence gate passed; independent scientific validation still required')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
