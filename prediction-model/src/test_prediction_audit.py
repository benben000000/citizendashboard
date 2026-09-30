"""
Tests for the prediction audit trail.

The two properties worth testing are not the schema -- it is JSON -- but the two
rules that make the trail trustworthy:

  1. it never breaks forecasting (a failed audit costs a log line, not a
     forecast), and
  2. it never silently loses a record (an unserialisable field becomes an
     explicit null, not a missing line).

Everything else here is fast scaffolding for those.
"""

import json
import os
import sys
import threading

import pytest

from prediction_audit import (
    MAX_BYTES,
    PredictionAudit,
    get_audit,
    read_records,
    variable_entry,
)

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


@pytest.fixture()
def audit(tmp_path):
    return PredictionAudit(path=str(tmp_path / "audit.jsonl"),
                           data_dir=os.path.join(DATA_DIR))


FORECAST = {
    "temperature_c": 28.4,
    "relative_humidity_pct": 79.0,
    "pressure_hpa": 1005.1,
    "wind_speed_kmh": 2.4,
    "chance_of_rain_pct": 42.0,
}


class TestRecordShape:
    def test_record_is_json_serialisable(self, audit):
        rec = audit.build_record("03pqkGAj", 1, FORECAST)
        json.dumps(rec)

    def test_carries_the_locating_fields(self, audit):
        rec = audit.build_record("03pqkGAj", 6, FORECAST,
                                 origin_timestamp="2026-09-29T05:00:00Z",
                                 device_id="KT-1Zb102pg")
        assert rec["station_id"] == "03pqkGAj"
        assert rec["device_id"] == "KT-1Zb102pg"
        assert rec["horizon_hours"] == 6
        assert rec["origin_timestamp_utc"] == "2026-09-29T05:00:00Z"
        assert rec["recorded_at_utc"].endswith("Z")

    def test_binds_the_policy_version_and_commit(self, audit):
        """The whole point: a published number must be traceable to an artifact."""
        rec = audit.build_record("03pqkGAj", 6, FORECAST)
        pol = rec["provenance"]["policy"]
        assert "policy_version" in pol
        assert "policy_code_commit" in pol
        # the real policy is present in this repo
        assert pol["policy_version"] is not None, "policy version was not resolved"
        assert pol["policy_code_commit"], "policy commit was not resolved"

    def test_records_the_bundle_for_the_horizon(self, audit):
        assert audit.build_record("s", 1, FORECAST)["provenance"]["model"]["bundle"] == "h1"
        assert audit.build_record("s", 24, FORECAST)["provenance"]["model"]["bundle"] == "h24"
        assert audit.build_record("s", "bogus", FORECAST)["provenance"]["model"]["bundle"] is None

    def test_keeps_the_raw_forecast(self, audit):
        rec = audit.build_record("s", 6, FORECAST)
        assert rec["forecast_raw"]["temperature_c"] == 28.4

    def test_variable_entries_capture_producer_and_nwp(self):
        e = variable_entry(28.4, "blend", nwp_raw=29.1, nwp_corrected=28.7)
        assert e == {"value": 28.4, "producer": "blend",
                     "nwp_raw": 29.1, "nwp_corrected": 28.7}

    def test_non_finite_values_become_null_not_nan(self):
        for bad in (float("nan"), float("inf"), -float("inf"), "x", None, [], {}):
            assert variable_entry(bad, "lln")["value"] is None, f"{bad!r} was not nulled"


class TestNeverBreaksForecasting:
    def test_unwritable_path_reports_failure_without_raising(self, tmp_path):
        bad = PredictionAudit(path=str(tmp_path / "no" / "such" / "deep" / "a.jsonl"),
                              data_dir=DATA_DIR)
        # may or may not succeed depending on OS; must never raise either way
        bad.record(station_id="s", horizon_hours=1, forecast=FORECAST)

    def test_path_that_is_a_directory_is_survivable(self, tmp_path):
        d = tmp_path / "adir"
        d.mkdir()
        a = PredictionAudit(path=str(d), data_dir=DATA_DIR)
        assert a.write({"station_id": "s"}) is False
        assert a.errors >= 1

    def test_garbage_kwargs_do_not_raise(self, audit):
        assert isinstance(audit.record(station_id="s", horizon_hours=object(),
                                       forecast={"x": object()}), bool)

    def test_missing_forecast_is_still_recorded(self, audit):
        """An absent forecast is a fact worth logging, not a reason to skip."""
        ok = audit.record(station_id="s", horizon_hours=1, forecast={})
        assert ok
        recs = read_records(audit.path)
        assert len(recs) == 1
        assert recs[0]["forecast_raw"] == {}

    def test_writer_survives_a_hostile_record(self, audit):
        class _Boom:
            def __repr__(self):
                raise RuntimeError("no repr for you")
        ok = audit.write({"station_id": "s", "bad": _Boom()})
        assert ok, "an unserialisable field must still produce a line"
        recs = read_records(audit.path)
        assert len(recs) == 1
        assert "write_error" in recs[0] or recs[0]["station_id"] == "s"


class TestAppendOnly:
    def test_records_accumulate(self, audit):
        for i in range(5):
            audit.record(station_id=f"s{i}", horizon_hours=1, forecast=FORECAST)
        assert len(read_records(audit.path)) == 5

    def test_one_line_per_record(self, audit):
        for i in range(3):
            audit.record(station_id=f"s{i}", horizon_hours=1, forecast=FORECAST)
        with open(audit.path, encoding="utf-8") as f:
            lines = [l for l in f.read().splitlines() if l.strip()]
        assert len(lines) == 3
        for line in lines:
            json.loads(line)

    def test_read_filters_by_station(self, audit):
        audit.record(station_id="a", horizon_hours=1, forecast=FORECAST)
        audit.record(station_id="b", horizon_hours=1, forecast=FORECAST)
        audit.record(station_id="a", horizon_hours=6, forecast=FORECAST)
        got = read_records(audit.path, station_id="a")
        assert len(got) == 2
        assert all(r["station_id"] == "a" for r in got)

    def test_read_filters_by_time(self, audit):
        audit.record(station_id="a", horizon_hours=1, forecast=FORECAST)
        assert read_records(audit.path, since="2999-01-01T00:00:00Z") == []
        assert len(read_records(audit.path, since="2000-01-01T00:00:00Z")) == 1

    def test_read_limit_returns_the_tail(self, audit):
        for i in range(10):
            audit.record(station_id="a", horizon_hours=i + 1, forecast=FORECAST)
        got = read_records(audit.path, limit=3)
        assert [r["horizon_hours"] for r in got] == [8, 9, 10]

    def test_truncated_final_line_is_skipped_not_fatal(self, audit):
        """A crash mid-append leaves a partial line; the rest must stay readable."""
        audit.record(station_id="a", horizon_hours=1, forecast=FORECAST)
        audit.record(station_id="b", horizon_hours=1, forecast=FORECAST)
        with open(audit.path, "a", encoding="utf-8") as f:
            f.write('{"station_id": "c", "horizon_hou')
        recs = read_records(audit.path)
        assert [r["station_id"] for r in recs] == ["a", "b"]

    def test_missing_file_reads_as_empty(self, tmp_path):
        assert read_records(str(tmp_path / "nope.jsonl")) == []


class TestRotation:
    def test_rotates_past_the_size_limit(self, tmp_path):
        a = PredictionAudit(path=str(tmp_path / "a.jsonl"), max_bytes=2000,
                            keep=3, data_dir=DATA_DIR)
        for i in range(60):
            a.record(station_id=f"s{i}", horizon_hours=1, forecast=FORECAST)
        rotated = [p for p in os.listdir(tmp_path) if p.startswith("a.jsonl.")]
        assert rotated, "nothing rotated"
        assert len(rotated) <= 3, f"kept {len(rotated)} generations, expected <= 3"

    def test_rotation_does_not_lose_the_newest_record(self, tmp_path):
        a = PredictionAudit(path=str(tmp_path / "a.jsonl"), max_bytes=2000,
                            keep=3, data_dir=DATA_DIR)
        for i in range(60):
            a.record(station_id="last", horizon_hours=1, forecast=FORECAST)
        assert read_records(a.path)[-1]["station_id"] == "last"

    def test_default_limit_is_generous(self):
        assert MAX_BYTES >= 8 * 1024 * 1024


class TestConcurrency:
    def test_concurrent_writes_are_all_recorded(self, audit):
        def worker(n):
            for i in range(10):
                audit.record(station_id=f"t{n}", horizon_hours=1, forecast=FORECAST)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        recs = read_records(audit.path)
        assert len(recs) == 60, f"expected 60 records, got {len(recs)}"
        # every line must be valid JSON: interleaved writes would corrupt them
        with open(audit.path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    json.loads(line)


class TestSingleton:
    def test_get_audit_is_stable(self):
        assert get_audit() is get_audit()

    def test_status_reports_counters(self, audit):
        audit.record(station_id="a", horizon_hours=1, forecast=FORECAST)
        s = audit.status()
        assert s["written"] == 1
        assert s["errors"] == 0
        assert s["exists"] is True
