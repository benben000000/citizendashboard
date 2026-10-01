"""
Tests for the live ingestor's supervision and sequence-admissibility rules.

Three properties, and each one is a way the evidence pipeline has actually
failed in production rather than a hypothetical:

  1. a gap longer than the tolerance discards the window, and the discard is
     RECORDED. A silent discard is indistinguishable from a station that
     stopped predicting, which is how a 93-minute hole went unnoticed.

  2. a gap inside the tolerance does not discard the window. The old 180s limit
     against a 60s publish cadence meant one missed cycle cost 24 minutes of
     refilling, and 24 minutes of irreplaceable evidence.

  3. the watchdog reports a live-but-stale ingestor as a FAILURE, not as
     healthy. A process publishing nothing looks exactly like a process working
     perfectly, because the only evidence of liveness is what it writes.

Import note: the ingestor reads its MQTT configuration at import time, so the
environment has to be populated before it is imported. Credentials are pointed
at temporary placeholder files because nothing here opens a socket.
"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

_GAP_LIMIT = 300
_ORIGIN = datetime(2026, 9, 30, 12, 0, 0, tzinfo=timezone.utc)


def _iso(moment: datetime) -> str:
    return moment.isoformat().replace("+00:00", "Z")


def _telemetry() -> dict:
    """A complete, in-range telemetry frame. Every field must be present."""
    return {
        "temperature_c": 24.5, "humidity_pct": 61.0, "pressure_hpa": 1013.2,
        "wind_speed_kmh": 11.0, "wind_direction_deg": 180.0,
        "water_level_m": 0.42, "rain_mm": 0.0,
    }


@pytest.fixture()
def ingestor(tmp_path, monkeypatch):
    """The ingestor module, wired to throwaway paths and placeholder certs."""
    fake = tmp_path / "creds"
    fake.mkdir()
    for name in ("ca.pem", "cert.pem", "key.pem"):
        (fake / name).write_text("placeholder", encoding="utf-8")

    monkeypatch.setenv("MQTT_ENDPOINT", "example.iot.region.amazonaws.com")
    monkeypatch.setenv("MQTT_CA_PATH", str(fake / "ca.pem"))
    monkeypatch.setenv("MQTT_CERT_PATH", str(fake / "cert.pem"))
    monkeypatch.setenv("MQTT_PRIVATE_KEY_PATH", str(fake / "key.pem"))
    monkeypatch.setenv("MQTT_OBSERVATION_HISTORY_PATH", str(tmp_path / "history.json"))
    monkeypatch.setenv("MQTT_EVENT_LOG_PATH", str(tmp_path / "events.jsonl"))
    monkeypatch.setenv("MQTT_MAX_SEQUENCE_GAP_SECONDS", str(_GAP_LIMIT))
    monkeypatch.setenv("MQTT_PREDICTION_SEQUENCE_LENGTH", "4")

    for name in ("mqtt_live_ingestor", "ingestor_watchdog"):
        sys.modules.pop(name, None)
    import mqtt_live_ingestor as module
    return module


def _events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class TestGapTolerance:
    def test_gap_inside_tolerance_keeps_the_window(self, ingestor, tmp_path):
        """A missed cycle must not cost the window -- this is the regression."""
        for step in range(3):
            ingestor.record_observation("KT-A", _iso(_ORIGIN + timedelta(seconds=60 * step)),
                                        _telemetry())
        # 200s gap: past a 180s limit, inside the 300s limit. Under the old
        # setting this discarded the window and began a 24-minute refill.
        history = ingestor.record_observation("KT-A", _iso(_ORIGIN + timedelta(seconds=260)),
                                              _telemetry())
        assert len(history) == 4, "a 200s gap must not discard the window"
        assert _events(tmp_path / "events.jsonl") == [], \
            "an accepted gap is not an event; only discards are"

    def test_gap_beyond_tolerance_discards_and_records(self, ingestor, tmp_path, monkeypatch):
        # Keep the age window wide so the GAP rule is what fires, not age
        # pruning. See test_long_outage_is_recorded_by_the_age_filter for the
        # other path -- they are different causes and were previously
        # conflated, which is how a real 68-minute outage went unrecorded.
        monkeypatch.setenv("MQTT_HISTORY_MAX_AGE_SECONDS", "100000")
        import mqtt_live_ingestor
        mqtt_live_ingestor.HISTORY_MAX_AGE_SECONDS = 100000
        for step in range(3):
            mqtt_live_ingestor.record_observation(
                "KT-B", _iso(_ORIGIN + timedelta(seconds=60 * step)), _telemetry())
        history = mqtt_live_ingestor.record_observation(
            "KT-B", _iso(_ORIGIN + timedelta(seconds=4000)), _telemetry())
        assert len(history) == 1, "a gap past the limit must discard the window"

        events = _events(tmp_path / "events.jsonl")
        resets = [e for e in events if e["kind"] == "sequence_reset"]
        assert len(resets) == 1, "the discard must be recorded, not silent"
        reset = resets[0]
        assert reset["station_id"] == "KT-B"
        assert reset["discarded_samples"] == 3
        assert reset["limit_seconds"] == _GAP_LIMIT
        # Last stored sample is at +120s and the new one lands at +4000s.
        assert reset["gap_seconds"] == pytest.approx(3880, abs=1)

    def test_long_outage_is_recorded_by_the_age_filter(self, ingestor, tmp_path):
        """
        A silence longer than HISTORY_MAX_AGE_SECONDS empties the window in the
        age filter, so the gap rule never sees a discontinuity. That path must
        still be recorded -- it is the one that actually ran during the
        68-minute upstream silence on 2026-10-01.
        """
        for step in range(3):
            ingestor.record_observation("KT-F", _iso(_ORIGIN + timedelta(seconds=60 * step)),
                                        _telemetry())
        history = ingestor.record_observation("KT-F", _iso(_ORIGIN + timedelta(seconds=7200)),
                                              _telemetry())
        assert len(history) == 1, "stale samples must not survive the age window"

        prunes = [e for e in _events(tmp_path / "events.jsonl")
                  if e["kind"] == "history_pruned"]
        assert len(prunes) == 1, "age-driven discards must be recorded too"
        assert prunes[0]["station_id"] == "KT-F"
        assert prunes[0]["dropped_samples"] == 3

    def test_event_records_the_station_and_the_limit(self, ingestor, tmp_path, monkeypatch):
        """An event must name the station and the limit, or it cannot be acted on."""
        monkeypatch.setenv("MQTT_HISTORY_MAX_AGE_SECONDS", "100000")
        import mqtt_live_ingestor
        mqtt_live_ingestor.HISTORY_MAX_AGE_SECONDS = 100000
        mqtt_live_ingestor.record_observation("KT-C", _iso(_ORIGIN), _telemetry())
        mqtt_live_ingestor.record_observation(
            "KT-C", _iso(_ORIGIN + timedelta(seconds=9000)), _telemetry())
        reset = [e for e in _events(tmp_path / "events.jsonl")
                 if e["kind"] == "sequence_reset"][0]
        assert reset["station_id"] == "KT-C"
        assert reset["limit_seconds"] == _GAP_LIMIT
        assert "recorded_at_utc" in reset

    def test_incomplete_telemetry_does_not_reset_or_append(self, ingestor, tmp_path):
        """A frame missing a required field must not count as a sample."""
        ingestor.record_observation("KT-D", _iso(_ORIGIN), _telemetry())
        partial = _telemetry()
        partial["humidity_pct"] = None
        history = ingestor.record_observation("KT-D", _iso(_ORIGIN + timedelta(seconds=90)),
                                              partial)
        assert len(history) == 1, "an incomplete frame must not enter the window"

    def test_station_publishing_nothing_usable_is_reported(self, ingestor, tmp_path):
        """
        Three stations on 2026-10-01 sent 2,661 messages and produced zero
        samples, because their frames lack required fields. With no diagnostic,
        they were indistinguishable from stations that had gone offline.
        """
        for step in range(4):
            partial = _telemetry()
            partial["wind_direction_deg"] = None
            history = ingestor.record_observation(
                "KT-G", _iso(_ORIGIN + timedelta(seconds=60 * step)), partial)
        assert history == [], "no usable frame means no sample, at all"

        events = [e for e in _events(tmp_path / "events.jsonl")
                  if e["kind"] == "incomplete_frame"]
        assert len(events) == 1, "repeated identical failures must report once, not per message"
        assert events[0]["station_id"] == "KT-G"
        assert events[0]["missing_fields"] == ["wind_direction_deg"]

    def test_a_recovered_station_reports_again_if_it_breaks_again(self, ingestor, tmp_path):
        """Suppression must be per distinct fault, not permanent per station."""
        partial = _telemetry()
        partial["wind_direction_deg"] = None
        ingestor.record_observation("KT-H", _iso(_ORIGIN), partial)
        ingestor.record_observation("KT-H", _iso(_ORIGIN + timedelta(seconds=60)), _telemetry())
        broken = _telemetry()
        broken["pressure_hpa"] = None
        ingestor.record_observation("KT-H", _iso(_ORIGIN + timedelta(seconds=120)), broken)
        reported = [(e["missing_fields"]) for e in _events(tmp_path / "events.jsonl")
                    if e["kind"] == "incomplete_frame"]
        assert reported == [["wind_direction_deg"], ["pressure_hpa"]]


class TestProducerLabelling:
    """
    The audit trail must say WHO produced each value.

    Regression: the policy serves most cells by persistence, so labelling
    everything "lln" claims model authorship for numbers the network never
    produced. The fix looked correct and was not -- selected_source_by_variable
    sits inside the predictor's "forecast" dict, and the lookup read one level
    too high, so every producer silently degraded to "unknown". That is worse
    than the original bug in one specific way: it looks like the honest answer.
    """

    @staticmethod
    def _prediction(selection, values=None):
        forecast = {
            "temperature_c": 24.0, "relative_humidity_pct": 60.0,
            "pressure_hpa": 1013.0, "wind_speed_kmh": 10.0,
        }
        if selection is not None:
            forecast["selected_source_by_variable"] = selection
        return {"generated_at": "2026-09-30T12:00:00Z", "horizon": "1h",
                "forecast": forecast}

    def _entries(self, tmp_path, prediction):
        """Write one audit record and return it, or fail with the reason."""
        import mqtt_live_ingestor as ingestor
        # get_audit() memoises the target path, so the redirect has to be given
        # explicitly rather than by environment variable alone.
        ingestor.audit_prediction("KT-Z", "2026-09-30T12:00:00Z", prediction, None,
                                  audit_path=str(tmp_path / "audit.jsonl"))
        target = tmp_path / "audit.jsonl"
        assert target.exists(), (
            "audit_prediction wrote nothing -- it swallows every failure by "
            "design, so an empty file here means the call raised internally")
        return json.loads(target.read_text(encoding="utf-8").splitlines()[0])

    def test_persistence_is_not_credited_to_the_model(self, tmp_path):
        record = self._entries(tmp_path, self._prediction({
            "temperature": "persistence_fallback", "humidity": "learned_model",
            "pressure": "persistence_fallback", "wind_speed": "learned_model"}))
        assert record is not None
        producers = {k: v["producer"] for k, v in record["variables"].items()}
        assert producers["temperature"] == "persistence"
        assert producers["humidity"] == "lln"
        assert producers["pressure"] == "persistence"
        assert producers["wind_speed"] == "lln"

    def test_unrecognised_source_is_unknown_not_lln(self, tmp_path):
        """An unrecognised source is a fact worth surfacing, not a default."""
        record = self._entries(tmp_path, self._prediction({
            "temperature": "some_future_router", "humidity": "learned_model",
            "pressure": "learned_model", "wind_speed": "learned_model"}))
        assert record["variables"]["temperature"]["producer"] == "unknown"
        assert record["variables"]["humidity"]["producer"] == "lln"

    def test_absent_selection_yields_unknown_never_a_false_lln(self, tmp_path):
        record = self._entries(tmp_path, self._prediction(None))
        assert {v["producer"] for v in record["variables"].values()} == {"unknown"}


class TestEventTrail:
    def test_events_are_one_json_object_per_line(self, ingestor, tmp_path):
        ingestor.record_event("unit_test", station_id="KT-E", value=1)
        ingestor.record_event("unit_test", station_id="KT-F", value=2)
        raw = (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(raw) == 2
        for line in raw:
            assert json.loads(line)["kind"] == "unit_test"

    def test_unserialisable_value_does_not_raise(self, ingestor, tmp_path):
        """An event write must never be able to break the ingestor."""
        ingestor.record_event("unit_test", odd=object())
        assert _events(tmp_path / "events.jsonl")[0]["kind"] == "unit_test"


class TestWatchdogAssessment:
    """The watchdog's judgement, tested without touching a real process."""

    @pytest.fixture()
    def watchdog(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MQTT_EVENT_LOG_PATH", str(tmp_path / "events.jsonl"))
        for name in ("mqtt_live_ingestor", "ingestor_watchdog"):
            sys.modules.pop(name, None)
        import ingestor_watchdog
        return ingestor_watchdog

    def test_no_process_is_not_running(self, watchdog, tmp_path, monkeypatch):
        monkeypatch.setattr(watchdog, "running_pids", lambda: [])
        monkeypatch.setattr(watchdog, "cache_age_seconds", lambda: 1.0)
        assert watchdog.assess()["state"] == "NOT_RUNNING"

    def test_live_process_with_stale_cache_is_stale_not_healthy(self, watchdog,
                                                                tmp_path, monkeypatch):
        """The failure mode that matters: alive, connected, publishing nothing."""
        monkeypatch.setattr(watchdog, "running_pids", lambda: [4242])
        monkeypatch.setattr(watchdog, "cache_age_seconds", lambda: 7200.0)
        report = watchdog.assess()
        assert report["state"] == "STALE"
        assert "stale" in report["reason"] or "old" in report["reason"]

    def test_process_with_no_cache_yet_is_reported(self, watchdog, monkeypatch):
        monkeypatch.setattr(watchdog, "running_pids", lambda: [4242])
        monkeypatch.setattr(watchdog, "cache_age_seconds", lambda: None)
        assert watchdog.assess()["state"] == "NO_CACHE"

    def test_fresh_cache_with_live_process_is_healthy(self, watchdog, monkeypatch):
        monkeypatch.setattr(watchdog, "running_pids", lambda: [4242])
        monkeypatch.setattr(watchdog, "cache_age_seconds", lambda: 2.0)
        monkeypatch.setattr(watchdog, "read_events", lambda: [])
        assert watchdog.assess()["state"] == "HEALTHY"

    def test_torn_final_line_does_not_break_event_reading(self, watchdog, tmp_path):
        """A partial write must not stop the watchdog reading its own trail."""
        (tmp_path / "events.jsonl").write_text(
            '{"kind": "a", "recorded_at_utc": "2026-09-30T12:00:00Z"}\n'
            '{"kind": "b", "recorded_at', encoding="utf-8")
        assert [e["kind"] for e in watchdog.read_events()] == ["a"]

    def test_absent_event_file_is_empty_not_an_error(self, watchdog, tmp_path):
        assert watchdog.read_events() == []
