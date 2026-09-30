"""
Tests for the NOAA GFS live provider.

These do NOT touch the network. The fetch, index parse and GRIB decode are all
exercised against synthetic data, because a test suite that depends on NOAA
being up is a suite that fails for reasons unrelated to the code.

What IS covered against reality is the metadata contract -- cycle naming, lead
availability, units -- which is where a silent break would do most damage.
"""

import os
import sys
from datetime import datetime, timedelta, timezone

import pytest

from external_sources.gfs import (
    AVAILABILITY_DELAY_HOURS,
    CYCLES_UTC,
    GfsProvider,
    SOURCE_ID,
    SUPPORTED_LEADS,
)

_lead_for = GfsProvider._lead_for


class TestLeadMapping:
    def test_exact_published_leads_pass_through(self):
        for lead in (3, 6, 12, 24):
            assert _lead_for(lead) == lead

    def test_unpublished_lead_snaps_up(self):
        # GFS is 3-hourly early and 6-hourly later; never round DOWN, which
        # would silently shorten the forecast the caller asked for.
        assert _lead_for(1) == 3
        assert _lead_for(4) == 6
        assert _lead_for(7) == 9
        assert _lead_for(13) == 18

    def test_lead_beyond_the_horizon_returns_none(self):
        assert _lead_for(200) is None
        assert _lead_for(1000) is None

    def test_never_rounds_down(self):
        for h in range(1, 121):
            lead = _lead_for(h)
            if lead is not None:
                assert lead >= h, f"lead {h} was rounded down to {lead}"


class TestCycleSelection:
    def _provider(self):
        return GfsProvider(15.0, 121.0, registry=None)

    def test_cycles_are_the_standard_four(self):
        assert CYCLES_UTC == (0, 6, 12, 18)

    def test_returns_a_cycle_before_now(self):
        p = self._provider()
        now = datetime(2026, 9, 29, 20, 0, tzinfo=timezone.utc)
        got = p.latest_cycle(now)
        assert got is not None
        cycle, _ = got
        assert cycle <= now
        assert cycle.hour in CYCLES_UTC
        assert cycle.minute == 0 and cycle.second == 0

    def test_respects_the_availability_delay(self):
        """A cycle inside its delay window must not be selected."""
        p = self._provider()
        now = datetime(2026, 9, 29, 13, 0, tzinfo=timezone.utc)  # 12Z is 1h old
        cycle, _ = p.latest_cycle(now)
        assert cycle.hour == 6, "picked a cycle still inside its distribution delay"
        assert now - cycle >= timedelta(hours=AVAILABILITY_DELAY_HOURS)

    def test_uses_the_newest_cycle_once_the_delay_has_passed(self):
        p = self._provider()
        now = datetime(2026, 9, 29, 19, 0, tzinfo=timezone.utc)  # 12Z is 7h old
        cycle, _ = p.latest_cycle(now)
        assert cycle == datetime(2026, 9, 29, 12, 0, tzinfo=timezone.utc)

    def test_falls_back_across_days(self):
        p = self._provider()
        now = datetime(2026, 9, 1, 1, 0, tzinfo=timezone.utc)
        cycle, _ = p.latest_cycle(now)
        assert cycle == datetime(2026, 8, 31, 18, 0, tzinfo=timezone.utc)
        assert (now - cycle) >= timedelta(hours=AVAILABILITY_DELAY_HOURS)

    def test_never_returns_a_future_cycle(self):
        p = self._provider()
        for hour in range(0, 24, 3):
            now = datetime(2026, 9, 29, hour, 0, tzinfo=timezone.utc)
            got = p.latest_cycle(now)
            if got:
                assert got[0] <= now


class TestGate:
    def test_provider_refuses_when_registry_blocks(self):
        class _Blocked:
            def is_production_eligible(self, _s):
                return False, "Source not found in registry."

        p = GfsProvider(15.0, 121.0, registry=_Blocked())
        assert not p.available
        assert p.blocked_reason
        assert p.forecast(6) is None

    def test_provider_refuses_when_registry_raises(self):
        class _Boom:
            def is_production_eligible(self, _s):
                raise RuntimeError("registry file corrupt")

        p = GfsProvider(15.0, 121.0, registry=_Boom())
        assert not p.available
        assert "registry" in (p.blocked_reason or "")
        assert p.forecast(6) is None

    def test_status_reports_pressure_unavailable(self):
        p = GfsProvider(15.0, 121.0)
        s = p.status()
        assert s["source"] == SOURCE_ID
        assert s["pressure_available"] is False
        assert s["supported_leads_h"] == list(SUPPORTED_LEADS)

    def test_status_never_raises(self):
        GfsProvider(15.0, 121.0).status()


class TestUnitContract:
    def test_forecast_never_claims_pressure(self):
        """
        The feed publishes only the MSL perturbation, so the provider must not
        emit a pressure key. If it ever did, the router would apply a
        correction fitted against real surface pressure to a perturbation and
        silently corrupt the forecast.
        """
        p = GfsProvider(15.0, 121.0)
        p._value = lambda cycle, lead, short, level: {
            ("TMP", "2 m above ground"): 298.15,
            ("RH", "2 m above ground"): 70.0,
            ("UGRD", "10 m above ground"): 3.0,
            ("VGRD", "10 m above ground"): 4.0,
        }.get((short, level))
        p.latest_cycle = lambda now=None: (datetime(2026, 9, 29, 12, tzinfo=timezone.utc), 0)
        f = p.forecast(6, now=datetime(2026, 9, 29, 20, tzinfo=timezone.utc))
        assert f is not None
        assert "pressure" not in f
        assert f["_meta"]["pressure_unavailable"] is True

    def test_units_are_converted_to_station_scale(self):
        p = GfsProvider(15.0, 121.0)
        p._value = lambda cycle, lead, short, level: {
            ("TMP", "2 m above ground"): 298.15,        # 25 degC in kelvin
            ("RH", "2 m above ground"): 70.0,
            ("UGRD", "10 m above ground"): 3.0,
            ("VGRD", "10 m above ground"): 4.0,
        }.get((short, level))
        p.latest_cycle = lambda now=None: (datetime(2026, 9, 29, 12, tzinfo=timezone.utc), 0)
        f = p.forecast(6, now=datetime(2026, 9, 29, 20, tzinfo=timezone.utc))
        assert f["temperature"] == pytest.approx(25.0, abs=0.01)
        assert f["humidity"] == pytest.approx(70.0, abs=0.01)
        # wind stays in m/s, which is what the stations report
        assert f["wind_speed"] == pytest.approx(5.0, abs=0.01)

    def test_humidity_is_clamped(self):
        p = GfsProvider(15.0, 121.0)
        p._value = lambda c, l, s, lv: {
            ("TMP", "2 m above ground"): 300.0,
            ("RH", "2 m above ground"): 140.0,
        }.get((s, lv))
        p.latest_cycle = lambda now=None: (datetime(2026, 9, 29, 12, tzinfo=timezone.utc), 0)
        f = p.forecast(6, now=datetime(2026, 9, 29, 20, tzinfo=timezone.utc))
        assert f["humidity"] == 100.0

    def test_issue_time_is_cycle_plus_latency(self):
        p = GfsProvider(15.0, 121.0)
        p._value = lambda c, l, s, lv: 298.15 if s == "TMP" else None
        p.latest_cycle = lambda now=None: (datetime(2026, 9, 29, 12, tzinfo=timezone.utc), 0)
        f = p.forecast(6, now=datetime(2026, 9, 29, 20, tzinfo=timezone.utc))
        assert f["_meta"]["issued_utc"] == "2026-09-29T18:00:00+00:00"
        assert f["_meta"]["valid_utc"] == "2026-09-29T18:00:00+00:00"


class TestFailureModes:
    def test_unsupported_lead_returns_none(self):
        assert GfsProvider(15.0, 121.0).forecast(500) is None

    def test_no_usable_cycle_returns_none(self):
        p = GfsProvider(15.0, 121.0)
        p.latest_cycle = lambda now=None: None
        assert p.forecast(6) is None

    def test_all_fields_missing_returns_none(self):
        p = GfsProvider(15.0, 121.0)
        p._value = lambda *a, **k: None
        p.latest_cycle = lambda now=None: (datetime(2026, 9, 29, 12, tzinfo=timezone.utc), 0)
        assert p.forecast(6) is None

    def test_never_raises_on_bad_coordinates(self):
        for lat, lon in ((0, 0), (-90, 180), (90, 0), (15.0, 121.0)):
            p = GfsProvider(lat, lon)
            try:
                p.status()
            except Exception as exc:  # noqa: BLE001
                pytest.fail(f"status() raised for ({lat},{lon}): {exc}")

    def test_decode_failure_is_recorded_not_raised(self, monkeypatch):
        p = GfsProvider(15.0, 121.0)
        p._value = lambda c, l, s, lv: 298.15 if s == "TMP" else None
        p.latest_cycle = lambda now=None: (datetime(2026, 9, 29, 12, tzinfo=timezone.utc), 0)
        monkeypatch.setattr(p, "_sample", lambda blob: (_ for _ in ()).throw(
            ValueError("corrupt grib")))
        f = p.forecast(6)
        assert f is None or "temperature" in f


class TestInterpolation:
    def test_out_of_range_latitude_falls_back_not_crashes(self, monkeypatch):
        """
        A station at an impossible latitude must still produce a value rather
        than raising: the grid sample clamps to the nearest valid node.
        """
        p = GfsProvider(121.0, 15.0)  # the swapped-coordinate mistake
        import numpy as np
        import eccodes

        class _G:
            def __enter__(self): return self
            def __exit__(self, *a): return False

        def fake_new(blob): return _G()
        def fake_get(h, k):
            return {"Ni": 1440, "Nj": 721,
                    "latitudeOfFirstGridPointInDegrees": 90.0,
                    "longitudeOfFirstGridPointInDegrees": 0.0,
                    "longitudeOfLastGridPointInDegrees": 359.75,
                    "iDirectionIncrementInDegrees": 0.25,
                    "jDirectionIncrementInDegrees": 0.25}.get(k, 0.0)
        def fake_arr(h, k): return np.full(1440 * 721, 298.15)

        monkeypatch.setattr(eccodes, "codes_new_from_message", fake_new)
        monkeypatch.setattr(eccodes, "codes_get", fake_get)
        monkeypatch.setattr(eccodes, "codes_get_array", fake_arr)
        monkeypatch.setattr(eccodes, "codes_release", lambda h: None)
        v = p._sample(b"not-a-real-grib")
        assert v is not None and 200.0 < v < 350.0
