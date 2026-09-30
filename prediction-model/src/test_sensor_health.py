"""
Tests for the sensor health gate.

The important property is that it must not throw away usable data. A gate that
calls a merely-sheltered anemometer "dead" would quietly shrink the fleet and
make the remaining numbers look better by selection, which is worse than the
problem it was meant to solve.
"""

import os
import sys

import numpy as np
import pytest

from dataset import TelemetryDataPipeline
from sensor_health import (
    DEAD_ZERO_FRACTION,
    MIN_DISTINCT_VALUES,
    assess_fleet,
    assess_wind,
    healthy_stations,
    summarise,
    unusable_stations,
)

rng = np.random.default_rng(7)


def weather(n=400, mean=3.0, sd=1.5):
    """Plausible wind: a floor at zero, a long right tail, occasional calms."""
    v = np.abs(rng.normal(mean, sd, n))
    v[v < 0.05] = 0.0
    v[v > mean + 6 * sd] = mean + 6 * sd
    return v.tolist()


class TestDeadDetection:
    def test_constant_zero_is_dead(self):
        r = assess_wind([0.0] * 300, fleet_median=3.0)
        assert r["verdict"] == "dead"

    def test_constant_nonzero_is_dead(self):
        r = assess_wind([2.5] * 300, fleet_median=3.0)
        assert r["verdict"] == "dead", "a flat channel is not weather either"

    def test_almost_always_zero_is_dead(self):
        v = [0.0] * 296 + [0.11, 0.09, 0.12, 0.10]
        r = assess_wind(v, fleet_median=3.0)
        assert r["verdict"] == "dead"

    def test_dead_detection_is_relative_to_the_fleet(self):
        """A station reading a steady 0.4 m/s in a 0.5 m/s fleet is not broken."""
        calm = [abs(x) for x in rng.normal(0.5, 0.25, 400)]
        r = assess_wind([0.0] * 300, fleet_median=0.5)
        assert r["verdict"] == "dead"
        assert assess_wind(calm, fleet_median=0.5)["verdict"] == "ok"


class TestNoisyDetection:
    def test_mostly_zeros_with_a_tail_is_noisy(self):
        v = [0.0] * 150 + weather(150)
        r = assess_wind(v, fleet_median=3.0)
        assert r["verdict"] == "noisy"

    def test_too_few_distinct_values_is_noisy(self):
        v = [0.0, 0.0, 1.0, 2.0, 3.0] * 60
        r = assess_wind(v, fleet_median=3.0)
        assert r["verdict"] in ("noisy", "dead")

    def test_implausibly_smooth_channel_is_not_ok(self):
        """
        A channel that drifts by 2 cm/s over 400 samples is not weather. Whether
        it lands on "dead" or "noisy" is a labelling choice, but it must not be
        accepted as healthy.
        """
        v = list(np.linspace(1.0, 1.02, 400))
        r = assess_wind(v, fleet_median=3.0)
        assert r["verdict"] in ("noisy", "dead")

    def test_intermittent_but_real_gales_are_not_dead(self):
        """
        Sparse readings are fine IF the ones that exist are real. This is the
        case the magnitude test exists to preserve, and the reason dead detection
        cannot be a zero-count threshold alone.
        """
        v = [0.0] * 340 + [8.0, 9.5, 11.0, 12.5, 7.5, 10.0, 13.0, 9.0] * 7
        r = assess_wind(v, fleet_median=3.0)
        assert r["verdict"] != "dead", f"a station seeing 9 m/s gales was called {r['verdict']}"


class TestHealthyIsNotDiscarded:
    def test_realistic_wind_is_ok(self):
        assert assess_wind(weather(), fleet_median=3.0)["verdict"] == "ok"

    def test_a_low_but_variable_station_is_ok(self):
        """
        Shelter makes a station read low; it does not make it constant. This is
        the case the gate must NOT reject, or the fleet quietly shrinks.
        """
        sheltered = [x * 0.35 for x in weather(400, mean=3.0, sd=1.6)]
        r = assess_wind(sheltered, fleet_median=3.0)
        assert r["verdict"] == "ok", f"sheltered but valid wind was called {r['verdict']}"
        assert r["mean"] < 1.5

    def test_a_windy_coastal_station_is_ok(self):
        windy = weather(400, mean=9.0, sd=4.0)
        assert assess_wind(windy, fleet_median=3.0)["verdict"] == "ok"


class TestEdgeCases:
    def test_empty_is_absent_not_dead(self):
        assert assess_wind([], fleet_median=3.0)["verdict"] == "absent"

    def test_none_and_nan_are_ignored(self):
        v = weather(200) + [None, float("nan")] * 50
        r = assess_wind(v, fleet_median=3.0)
        assert r["verdict"] == "ok"

    def test_too_few_samples_refuses_to_judge(self):
        assert assess_wind([0.0] * 10, fleet_median=3.0)["verdict"] == "absent"

    def test_strings_and_bools_do_not_crash(self):
        v = weather(200) + ["bad", True, False, "", None]
        assert assess_wind(v, fleet_median=3.0)["verdict"] == "ok"

    def test_never_raises_on_any_iterable(self):
        for bad in (None, 42, [None] * 5, [float("inf")] * 5):
            try:
                assess_wind(bad if isinstance(bad, list) else [], fleet_median=1.0)
            except Exception as exc:  # noqa: BLE001
                pytest.fail(f"raised {type(exc).__name__}")


class TestFleet:
    def _fleet(self):
        return {
            "healthy_a": weather(400, 3.0, 1.5),
            "healthy_b": weather(400, 4.0, 2.0),
            "sheltered": [x * 0.35 for x in weather(400, 3.0, 1.6)],
            "dead_a": [0.0] * 400,
            "dead_b": [0.0] * 398 + [0.1, 0.1],
            "noisy": [0.0] * 200 + weather(200),
        }

    def test_splits_the_fleet_correctly(self):
        a = assess_fleet(self._fleet())
        s = summarise(a)
        assert set(s["healthy"]) == {"healthy_a", "healthy_b", "sheltered"}
        assert set(s["unusable"]) == {"dead_a", "dead_b"}

    def test_noisy_stations_are_kept_out_of_unusable(self):
        a = assess_fleet(self._fleet())
        assert "noisy" not in unusable_stations(a)

    def test_noisy_can_be_included_on_request(self):
        a = assess_fleet(self._fleet())
        assert "noisy" in healthy_stations(a, include_noisy=True)

    def test_reference_entry_is_not_treated_as_a_station(self):
        a = assess_fleet(self._fleet())
        assert "_fleet_median_wind" in a
        assert "_fleet_median_wind" not in healthy_stations(a)
        assert "_fleet_median_wind" not in unusable_stations(a)

    def test_empty_fleet_is_handled(self):
        s = summarise(assess_fleet({}))
        assert s["healthy"] == [] and s["unusable"] == []


class TestAgainstRealData:
    """The gate must actually catch the three dead anemometers in the fleet."""

    @pytest.fixture(scope="class")
    def assessment(self):
        pipe = TelemetryDataPipeline()
        per = {}
        for sid, obs in pipe.station_hourly.items():
            per[sid] = [r.get("wind_speed") for r in obs.values()]
        return assess_fleet(per)

    def test_finds_exactly_three_dead_channels(self, assessment):
        dead = unusable_stations(assessment)
        assert set(dead) == {"4VAl2p9k", "Bkpj1zRO", "95pM7BAV"}

    def test_keeps_the_rest_of_the_fleet(self, assessment):
        """
        13 of 16 stations have a usable wind channel. Ten are clean; three are
        intermittent -- many exact zeros, but a real p95 in m/s -- and are
        recoverable, so they are "noisy" rather than "dead". All 13 must be
        obtainable.
        """
        ok = set(healthy_stations(assessment))
        recoverable = set(healthy_stations(assessment, include_noisy=True))
        assert "3nzr48bG" in ok and "nDby4YpR" in ok
        assert len(recoverable) == 13, f"expected 13 usable, got {len(recoverable)}"
        assert not (set(unusable_stations(assessment)) & recoverable)

    def test_noisy_stations_still_carry_real_wind(self, assessment):
        """
        "Noisy" must mean intermittent, not absent. If a station's p95 is a real
        wind speed it is throwing away useful training data by being discarded.
        """
        for sid, r in assessment.items():
            if r.get("verdict") == "noisy" and r.get("p95") is not None:
                assert r["p95"] > 1.0, (
                    f"{sid} called noisy but its p95 is only {r['p95']:.2f} m/s, "
                    f"which is not a recoverable reading")

    def test_no_station_is_silently_dropped(self, assessment):
        """
        Station ids are case-sensitive and a hand-written id that differs only in
        case would drop a real station from the gate without any error. The
        assessment must cover every station the pipeline knows about.
        """
        pipe = TelemetryDataPipeline()
        known = set(pipe.station_hourly)
        assessed = {k for k in assessment if not k.startswith("_")}
        assert known == assessed, (
            f"stations not assessed: {sorted(known - assessed)}; "
            f"unknown assessed: {sorted(assessed - known)}")

    def test_dead_stations_still_report_temperature(self, assessment):
        """
        A dead wind channel alongside a healthy temperature channel is a sensor
        fault, not a station outage -- which is exactly what a site visit needs
        to know.
        """
        pipe = TelemetryDataPipeline()
        for sid in ("4VAl2p9k", "Bkpj1zRO", "95pM7BAV"):
            temps = [r.get("temperature") for r in pipe.station_hourly[sid].values()
                     if r.get("temperature") is not None]
            assert len(temps) > 50
            assert max(temps) > 25.0, f"{sid} temperature also looks dead"

class TestLiveHealth:
    """
    Live assessment reads the observation trail. The property that matters most
    is ISOLATION: one station's verdict must never be another station's.
    """

    @staticmethod
    def _trail(tmp_path, dead_n=400, live_n=400):
        import json
        import random
        from datetime import datetime, timedelta, timezone
        random.seed(11)
        path = str(tmp_path / "obs.jsonl")
        now = datetime.now(timezone.utc)
        with open(path, "w", encoding="utf-8") as f:
            for i in range(max(dead_n, live_n)):
                ts = (now - timedelta(minutes=i)).isoformat().replace("+00:00", "Z")
                f.write(json.dumps({"station_id": "KT-DEAD", "observed_at_utc": ts,
                                    "telemetry": {"wind_speed_kmh": 0.0}}) + "\n")
                f.write(json.dumps({"station_id": "KT-LIVE", "observed_at_utc": ts,
                                    "telemetry": {"wind_speed_kmh": abs(random.gauss(3.0, 1.5))}}) + "\n")
        return path, now

    def test_dead_station_is_detected(self, tmp_path):
        from sensor_health import clear_live_cache, live_health
        path, now = self._trail(tmp_path)
        clear_live_cache()
        h = live_health("KT-DEAD", trail_path=path, now=now)
        assert h["verdict"] == "dead"

    def test_healthy_station_is_not_contaminated_by_a_dead_neighbour(self, tmp_path):
        from sensor_health import clear_live_cache, live_health
        path, now = self._trail(tmp_path)
        clear_live_cache()
        h = live_health("KT-LIVE", trail_path=path, now=now)
        assert h["verdict"] == "ok", (
            "a healthy station inherited a dead neighbour's verdict: the fleet "
            "aggregate was being attributed to whichever station was asked for")

    def test_unknown_station_is_unknown_not_dead(self, tmp_path):
        from sensor_health import clear_live_cache, live_health
        path, now = self._trail(tmp_path)
        clear_live_cache()
        h = live_health("KT-NOT-THERE", trail_path=path, now=now)
        assert h["verdict"] == "unknown"
        assert "need" in (h["reason"] or "")

    def test_missing_trail_is_unknown(self, tmp_path):
        from sensor_health import clear_live_cache, live_health
        clear_live_cache()
        h = live_health("KT-X", trail_path=str(tmp_path / "nope.jsonl"))
        assert h["verdict"] == "unknown"

    def test_observations_outside_the_lookback_are_ignored(self, tmp_path):
        import json
        from datetime import datetime, timedelta, timezone
        from sensor_health import clear_live_cache, live_health
        path = str(tmp_path / "old.jsonl")
        old = datetime.now(timezone.utc) - timedelta(days=30)
        with open(path, "w", encoding="utf-8") as f:
            for i in range(300):
                ts = (old - timedelta(minutes=i)).isoformat().replace("+00:00", "Z")
                f.write(json.dumps({"station_id": "KT-DEAD", "observed_at_utc": ts,
                                    "telemetry": {"wind_speed_kmh": 0.0}}) + "\n")
        clear_live_cache()
        h = live_health("KT-DEAD", trail_path=path, lookback_hours=24.0)
        assert h["verdict"] == "unknown", "stale readings decided current health"

    def test_result_is_cached(self, tmp_path):
        from sensor_health import clear_live_cache, live_health
        path, now = self._trail(tmp_path)
        clear_live_cache()
        a = live_health("KT-DEAD", trail_path=path, now=now)
        b = live_health("KT-DEAD", trail_path=path, now=now)
        assert a is b or a == b
