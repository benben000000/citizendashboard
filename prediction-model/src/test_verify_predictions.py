"""
Tests for the prediction verification layer.

The behaviour that matters is not the arithmetic. It is that scoring never
rewrites history, never invents a truth, and never scores a forecast against
something that is not actually the target time. Those are the three ways an
audit layer can lie while looking correct.
"""

import json
import os
from datetime import datetime, timedelta, timezone

import pytest

from prediction_audit import PredictionAudit, ObservationAudit, variable_entry
from verify_predictions import verify

NOW = datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc)
FORECAST = {"temperature_c": 28.0, "humidity_pct": 70.0,
            "pressure_hpa": 1005.0, "wind_speed_kmh": 2.0}
TRUTH = {"temperature_c": 27.0, "humidity_pct": 75.0,
         "pressure_hpa": 1005.5, "wind_speed_kmh": 2.4}


def _iso(dt):
    return dt.isoformat().replace("+00:00", "Z")


@pytest.fixture()
def trail(tmp_path):
    """A prediction trail and an observation trail, joined by station."""
    pa = PredictionAudit(path=str(tmp_path / "pred.jsonl"))
    oa = ObservationAudit(path=str(tmp_path / "obs.jsonl"))
    return pa, oa, str(pa.path), str(oa.path)


def _predict(pa, station, origin, horizon, forecast=FORECAST, producer="lln"):
    variables = {}
    for var, key in (("temperature", "temperature_c"), ("humidity", "humidity_pct"),
                     ("pressure", "pressure_hpa"), ("wind_speed", "wind_speed_kmh")):
        variables[var] = variable_entry(forecast[key], producer)
    return pa.record(station_id=station, horizon_hours=horizon, forecast=forecast,
                     origin_timestamp=_iso(origin), device_id=station,
                     variables=variables)


def _observe(oa, station, when, telemetry=TRUTH):
    return oa.record(station_id=station, observed_at_utc=_iso(when),
                     device_id=station, telemetry=telemetry)


class TestScoring:
    def test_scores_a_matured_prediction(self, trail):
        pa, oa, pp, op = trail
        origin = NOW - timedelta(hours=6)
        _predict(pa, "S1", origin, 6)
        _observe(oa, "S1", origin + timedelta(hours=6))
        s = verify(pp, op, str(pp) + ".ver", now=NOW)
        assert s["scored"] == 1
        assert s["mae_by_variable"]["temperature"] == pytest.approx(1.0)

    def test_computes_signed_and_absolute_error(self, trail):
        pa, oa, pp, op = trail
        origin = NOW - timedelta(hours=3)
        _predict(pa, "S1", origin, 3, {"temperature_c": 30.0, "humidity_pct": 70.0,
                                       "pressure_hpa": 1005.0, "wind_speed_kmh": 2.0})
        _observe(oa, "S1", origin + timedelta(hours=3),
                 {"temperature_c": 28.0, "humidity_pct": 70.0,
                  "pressure_hpa": 1005.0, "wind_speed_kmh": 2.0})
        s = verify(pp, op, str(pp) + ".ver", now=NOW)
        with open(s["writer"]["path"], encoding="utf-8") as f:
            rec = json.loads(f.readline())
        t = rec["variables"]["temperature"]
        assert t["predicted"] == 30.0 and t["actual"] == 28.0
        assert t["error"] == pytest.approx(2.0)
        assert t["abs_error"] == pytest.approx(2.0)

    def test_breaks_down_by_producer(self, trail):
        pa, oa, pp, op = trail
        origin = NOW - timedelta(hours=6)
        _predict(pa, "S1", origin, 6, producer="lln")
        _observe(oa, "S1", origin + timedelta(hours=6))
        _predict(pa, "S2", origin, 6, producer="nwp",
                 forecast={"temperature_c": 99.0, "humidity_pct": 70.0,
                           "pressure_hpa": 1005.0, "wind_speed_kmh": 2.0})
        _observe(oa, "S2", origin + timedelta(hours=6))
        s = verify(pp, op, str(pp) + ".ver", now=NOW)
        # Compared per variable, never pooled across units: an MAE that mixes
        # degC with hPa is a number without a meaning.
        per = s["mae_by_variable_and_producer"]["temperature"]
        assert per["lln"] == pytest.approx(1.0)
        assert per["nwp"] == pytest.approx(72.0)
        n = s["n_by_variable_and_producer"]["temperature"]
        assert n == {"lln": 1, "nwp": 1}
        assert "mae_by_producer" not in s


class TestDoesNotInventTruth:
    def test_unmatched_prediction_is_not_scored(self, trail):
        pa, oa, pp, op = trail
        _predict(pa, "S1", NOW - timedelta(hours=6), 6)  # no observation at all
        s = verify(pp, op, str(pp) + ".ver", now=NOW)
        assert s["scored"] == 0 and s["unmatched"] == 1
        assert s["mae_by_variable"] == {}

    def test_station_mismatch_is_not_scored(self, trail):
        """An observation from a different station is not this station's truth."""
        pa, oa, pp, op = trail
        origin = NOW - timedelta(hours=6)
        _predict(pa, "S1", origin, 6)
        _observe(oa, "OTHER", origin + timedelta(hours=6))
        s = verify(pp, op, str(pp) + ".ver", now=NOW)
        assert s["scored"] == 0 and s["unmatched"] == 1

    def test_observation_outside_tolerance_is_not_scored(self, trail):
        pa, oa, pp, op = trail
        origin = NOW - timedelta(hours=6)
        _predict(pa, "S1", origin, 6)
        _observe(oa, "S1", origin + timedelta(hours=6, minutes=90))
        s = verify(pp, op, str(pp) + ".ver", tolerance_minutes=30, now=NOW)
        assert s["scored"] == 0 and s["unmatched"] == 1

    def test_observation_inside_tolerance_is_scored(self, trail):
        pa, oa, pp, op = trail
        origin = NOW - timedelta(hours=6)
        _predict(pa, "S1", origin, 6)
        _observe(oa, "S1", origin + timedelta(hours=6, minutes=5))
        s = verify(pp, op, str(pp) + ".ver", tolerance_minutes=30, now=NOW)
        assert s["scored"] == 1

    def test_nearest_observation_wins(self, trail):
        pa, oa, pp, op = trail
        origin = NOW - timedelta(hours=6)
        _predict(pa, "S1", origin, 6)
        _observe(oa, "S1", origin + timedelta(hours=5, minutes=-25),
                 {"temperature_c": 50.0, "humidity_pct": 70.0,
                  "pressure_hpa": 1005.0, "wind_speed_kmh": 2.0})
        _observe(oa, "S1", origin + timedelta(hours=6, minutes=1),
                 {"temperature_c": 27.0, "humidity_pct": 70.0,
                  "pressure_hpa": 1005.0, "wind_speed_kmh": 2.0})
        s = verify(pp, op, str(pp) + ".ver", tolerance_minutes=30, now=NOW)
        assert s["mae_by_variable"]["temperature"] == pytest.approx(1.0)

    def test_unmatured_prediction_is_pending_not_scored(self, trail):
        pa, oa, pp, op = trail
        _predict(pa, "S1", NOW, 6)                     # target is in the future
        _observe(oa, "S1", NOW + timedelta(hours=6))   # even if the truth exists
        s = verify(pp, op, str(pp) + ".ver", now=NOW)
        assert s["scored"] == 0 and s["pending"] == 1

    def test_missing_truth_field_is_skipped_not_zeroed(self, trail):
        """A null truth is not an error of zero; it is an absence."""
        pa, oa, pp, op = trail
        origin = NOW - timedelta(hours=6)
        _predict(pa, "S1", origin, 6)
        _observe(oa, "S1", origin + timedelta(hours=6),
                 {"humidity_pct": 75.0, "pressure_hpa": 1005.5,
                  "wind_speed_kmh": 2.4})
        s = verify(pp, op, str(pp) + ".ver", now=NOW)
        assert "temperature" not in s["mae_by_variable"]
        assert "humidity" in s["mae_by_variable"]


class TestHistoryIsImmutable:
    def test_prediction_file_is_not_modified(self, trail):
        pa, oa, pp, op = trail
        origin = NOW - timedelta(hours=6)
        _predict(pa, "S1", origin, 6)
        _observe(oa, "S1", origin + timedelta(hours=6))
        before = open(pp, encoding="utf-8").read()
        verify(pp, op, str(pp) + ".ver", now=NOW)
        assert open(pp, encoding="utf-8").read() == before, \
            "verification rewrote the prediction trail"

    def test_verification_references_the_prediction(self, trail):
        pa, oa, pp, op = trail
        origin = NOW - timedelta(hours=6)
        _predict(pa, "S1", origin, 6)
        _observe(oa, "S1", origin + timedelta(hours=6))
        s = verify(pp, op, str(pp) + ".ver", now=NOW)
        with open(s["writer"]["path"], encoding="utf-8") as f:
            rec = json.loads(f.readline())
        with open(pp, encoding="utf-8") as f:
            pred = json.loads(f.readline())
        assert rec["prediction_record_id"] == pred["record_id"]
        assert rec["observation_record_id"]

    def test_rerunning_is_idempotent(self, trail):
        pa, oa, pp, op = trail
        origin = NOW - timedelta(hours=6)
        _predict(pa, "S1", origin, 6)
        _observe(oa, "S1", origin + timedelta(hours=6))
        vp = str(pp) + ".ver"
        first = verify(pp, op, vp, now=NOW)
        second = verify(pp, op, vp, now=NOW)
        assert first["scored"] == 1
        assert second["scored"] == 0 and second["already_verified"] == 1
        assert len(open(vp, encoding="utf-8").read().strip().splitlines()) == 1

    def test_provenance_is_carried_onto_the_verification(self, trail):
        pa, oa, pp, op = trail
        origin = NOW - timedelta(hours=6)
        _predict(pa, "S1", origin, 6)
        _observe(oa, "S1", origin + timedelta(hours=6))
        s = verify(pp, op, str(pp) + ".ver", now=NOW)
        with open(s["writer"]["path"], encoding="utf-8") as f:
            rec = json.loads(f.readline())
        assert rec["policy_version"] == "2.0.0"
        assert rec["model_bundle"] == "h6"
        assert rec["target_timestamp_utc"] == _iso(origin + timedelta(hours=6))


class TestDegradesQuietly:
    def test_missing_files_score_nothing(self, tmp_path):
        s = verify(str(tmp_path / "nope.jsonl"), str(tmp_path / "nope2.jsonl"),
                   str(tmp_path / "v.jsonl"), now=NOW)
        assert s["scored"] == 0 and s["predictions_seen"] == 0

    def test_malformed_prediction_line_is_skipped(self, trail):
        pa, oa, pp, op = trail
        origin = NOW - timedelta(hours=6)
        _predict(pa, "S1", origin, 6)
        _observe(oa, "S1", origin + timedelta(hours=6))
        with open(pp, "a", encoding="utf-8") as f:
            f.write("{not json\n")
            f.write(json.dumps({"station_id": "X", "horizon_hours": None,
                                "origin_timestamp_utc": None}) + "\n")
        s = verify(pp, op, str(pp) + ".ver", now=NOW)
        assert s["scored"] == 1
        assert s["unmatched"] == 1

    def test_unwritable_output_scores_nothing_but_does_not_raise(self, trail, tmp_path):
        pa, oa, pp, op = trail
        origin = NOW - timedelta(hours=6)
        _predict(pa, "S1", origin, 6)
        _observe(oa, "S1", origin + timedelta(hours=6))
        blocked = str(tmp_path / "sub")
        os.makedirs(blocked, exist_ok=True)
        verify(pp, op, blocked, now=NOW)   # must not raise
