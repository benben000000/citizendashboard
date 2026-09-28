"""Independently recompute temperature coverage and rain Brier from paired test rows.

This is an audit-only entry point: no training, inference, or report writes.
"""
import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

HORIZONS = (1, 3, 6, 12, 24)
COLUMNS = (
    'station_id', 'issue_timestamp_utc', 'target_timestamp_utc',
    'horizon_hours', 'split_name', 'label_quality_status',
    'forecast_temperature_p10', 'forecast_temperature_p90',
    'observed_temperature_c', 'rain_probability', 'rain_observed',
)


def _time(value):
    timestamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if timestamp.tzinfo is None or timestamp.utcoffset() != timedelta(0):
        raise ValueError('timestamps must have explicit UTC timezone')
    return timestamp.astimezone(timezone.utc)


def _float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('non-finite numeric value')
    return result


def evaluate(path, min_count=100):
    """Reject invalid or duplicate test pairs; do not silently discard rows."""
    if min_count < 1:
        raise ValueError('min_count must be positive')
    grouped = defaultdict(lambda: {'count': 0, 'covered': 0, 'brier_sum': 0.0})
    seen = set()
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)) or set(COLUMNS) - set(reader.fieldnames):
            raise ValueError('missing or duplicate required CSV columns')
        for row_number, row in enumerate(reader, 2):
            try:
                if not row['station_id'] or row['split_name'] != 'test' or row['label_quality_status'] != 'valid':
                    raise ValueError('station, test split, and valid label required')
                horizon = int(row['horizon_hours'])
                if horizon not in HORIZONS or str(horizon) != row['horizon_hours']:
                    raise ValueError('unsupported horizon')
                issued, target = _time(row['issue_timestamp_utc']), _time(row['target_timestamp_utc'])
                if issued + timedelta(hours=horizon) != target:
                    raise ValueError('issue/target/horizon mismatch')
                identity = (row['station_id'], issued, target, horizon)
                if identity in seen:
                    raise ValueError('duplicate forecast/observation pair')
                seen.add(identity)
                lo, hi, observed = (_float(row[name]) for name in ('forecast_temperature_p10', 'forecast_temperature_p90', 'observed_temperature_c'))
                probability = _float(row['rain_probability'])
                if lo > hi or not 0 <= probability <= 1 or row['rain_observed'] not in ('0', '1'):
                    raise ValueError('invalid interval, rain probability, or binary label')
                label = int(row['rain_observed'])
                totals = grouped[horizon]
                totals['count'] += 1
                totals['covered'] += lo <= observed <= hi
                totals['brier_sum'] += (probability - label) ** 2
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f'row {row_number}: {exc}') from exc
    output = {}
    for horizon in HORIZONS:
        totals = grouped[horizon]
        n = totals['count']
        if n < min_count:
            raise ValueError(f'horizon {horizon}h has {n} valid pairs; requires {min_count}')
        output[f'horizon_{horizon}h'] = {
            'temperature_sample_count': n,
            'temperature_covered_count': totals['covered'],
            'temperature_coverage_80': 100.0 * totals['covered'] / n,
            'rain_brier_score': totals['brier_sum'] / n,
        }
    return output


def compare(metrics, scorecard, uncertainty):
    """Return contradictions against existing reports; do not change their content."""
    blockers = []
    for key, measured in metrics.items():
        sources = (
            ('scorecard interval coverage', scorecard.get('interval_metrics', {}).get(key, {}).get('temperature_coverage_80'), measured['temperature_coverage_80'], 0.02),
            ('uncertainty report coverage', uncertainty.get('horizons', {}).get(key, {}).get('temperature_coverage_80'), measured['temperature_coverage_80'], 0.02),
            ('scorecard rain Brier', scorecard.get('calibration_metrics', {}).get(key, {}).get('brier_score'), measured['rain_brier_score'], 0.0001),
        )
        for name, reported, actual, tolerance in sources:
            if isinstance(reported, bool) or not isinstance(reported, (float, int)) or not math.isfinite(reported) or abs(reported - actual) > tolerance:
                blockers.append(f'{key}: {name} absent or disagrees with paired test rows')
    return blockers


def main():
    parser = argparse.ArgumentParser(description='Read-only recomputation from held-out paired observations')
    parser.add_argument('--paired-csv', type=Path, required=True)
    parser.add_argument('--candidate-dir', type=Path, required=True)
    args = parser.parse_args()
    try:
        metrics = evaluate(args.paired_csv)
        with (args.candidate_dir / 'predictive_quality_scorecard.json').open(encoding='utf-8') as handle:
            scorecard = json.load(handle)
        with (args.candidate_dir / 'uncertainty_report.json').open(encoding='utf-8') as handle:
            uncertainty = json.load(handle)
        blockers = compare(metrics, scorecard, uncertainty)
    except (OSError, ValueError, TypeError) as exc:
        print(f'NO-GO: {exc}')
        return 1
    print(json.dumps({'recomputed': metrics, 'blockers': blockers}, indent=2))
    return 1 if blockers else 0


if __name__ == '__main__':
    raise SystemExit(main())
