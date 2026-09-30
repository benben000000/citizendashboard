"""
Tests for the wind sensor gate inside the inference path.

The gate's contract is narrow and must be exact:

  * a dead anemometer must not yield a learned wind forecast
  * the OTHER variables must be completely unaffected -- a dead wind sensor is
    not a reason to withhold a temperature forecast
  * an inconclusive health check must not suppress anything
  * omitting station_id must preserve the previous behaviour exactly
"""

import json
import os
import tempfile
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

import sensor_health
from inference import LNNServerlessPredictor


def make_sequence(wind, n=24, seed=5):
    rng = np.random.default_rng(seed)
    X = np.empty((n, 8), dtype=np.float32)
    X[:, 0] = 29.0 + rng.normal(0, 0.3, n)
    X[:, 1] = 33.0 + rng.normal(0, 0.3, n)
    X[:, 2] = 78.0 + rng.normal(0, 2.0, n)
    X[:, 3] = 1005.0 + rng.normal(0, 0.4, n)
    X[:, 4] = wind
    X[:, 5] = 0.0
    X[:, 6] = 0.0
    X[:, 7] = 0.0
    return X, np.ones(n, dtype=np.float32)


@pytest.fixture(scope="module")
def predictor():
    return LNNServerlessPredictor(horizon_hours=6)


@pytest.fixture()
def dead_trail():
    """A trail in which KT-DEAD-1 is dead and KT-OK-1 is healthy."""
    import random
    random.seed(19)
    d = tempfile.mkdtemp(prefix="windgate")
    path = os.path.join(d, "obs.jsonl")
    now = datetime.now(timezone.utc)
    with open(path, "w", encoding="utf-8") as f:
        for i in range(400):
            ts = (now - timedelta(minutes=i)).isoformat().replace("+00:00", "Z")
            f.write(json.dumps({"station_id": "KT-DEAD-1", "observed_at_utc": ts,
                                "telemetry": {"wind_speed_kmh": 0.0}}) + "\n")
            f.write(json.dumps({"station_id": "KT-OK-1", "observed_at_utc": ts,
                                "telemetry": {"wind_speed_kmh": abs(random.gauss(3.0, 1.5))}}) + "\n")
    old = sensor_health.DEFAULT_TRAIL
    sensor_health.DEFAULT_TRAIL = path
    sensor_health.clear_live_cache()
    yield path
    sensor_health.DEFAULT_TRAIL = old
    sensor_health.clear_live_cache()


class TestGateQuarantines:
    def test_dead_sensor_sets_a_quarantine_status(self, predictor, dead_trail):
        X, dt = make_sequence(0.0)
        out = predictor.predict_from_observed_sequence(
            telemetry_sequence=X, dt_sequence=dt, horizon_hours=6,
            station_id="KT-DEAD-1")
        assert out["wind_speed_status"].startswith("QUARANTINED_SENSOR")
        assert out["wind_sensor_health"]["verdict"] == "dead"

    def test_published_wind_falls_back_to_the_observation(self, predictor, dead_trail):
        X, dt = make_sequence(0.0)
        out = predictor.predict_from_observed_sequence(
            telemetry_sequence=X, dt_sequence=dt, horizon_hours=6,
            station_id="KT-DEAD-1")
        assert out["wind_speed_kmh"] == pytest.approx(0.0, abs=0.01)

    def test_other_variables_are_unaffected(self, predictor, dead_trail):
        """A dead anemometer must not withhold temperature, humidity or pressure."""
        X, dt = make_sequence(0.0)
        out = predictor.predict_from_observed_sequence(
            telemetry_sequence=X, dt_sequence=dt, horizon_hours=6,
            station_id="KT-DEAD-1")
        assert out["temperature_c"] is not None
        assert out["relative_humidity_pct"] is not None
        assert out["pressure_hpa"] is not None
        assert out["chance_of_rain_pct"] is not None
        assert 20.0 < out["temperature_c"] < 40.0

    def test_healthy_sensor_is_left_alone(self, predictor, dead_trail):
        X, dt = make_sequence(3.0)
        out = predictor.predict_from_observed_sequence(
            telemetry_sequence=X, dt_sequence=dt, horizon_hours=6,
            station_id="KT-OK-1")
        assert out["wind_speed_status"] == "OK"
        assert out["wind_speed_kmh"] > 0.0


class TestGateIsFailSafe:
    def test_omitting_station_id_preserves_old_behaviour(self, predictor, dead_trail):
        X, dt = make_sequence(0.0)
        out = predictor.predict_from_observed_sequence(
            telemetry_sequence=X, dt_sequence=dt, horizon_hours=6)
        assert out["wind_speed_status"] == "OK"
        assert out["wind_sensor_health"] is None

    def test_unknown_station_does_not_suppress(self, predictor, dead_trail):
        X, dt = make_sequence(2.0)
        out = predictor.predict_from_observed_sequence(
            telemetry_sequence=X, dt_sequence=dt, horizon_hours=6,
            station_id="KT-NEVER-SEEN")
        assert out["wind_speed_status"] == "OK"
        assert out["wind_speed_kmh"] > 0.0

    def test_health_check_raising_does_not_fail_the_forecast(self, predictor, dead_trail,
                                                             monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("trail unreadable")
        monkeypatch.setattr(sensor_health, "live_health", boom)
        import inference
        monkeypatch.setattr(inference, "live_health", boom)
        X, dt = make_sequence(2.0)
        out = predictor.predict_from_observed_sequence(
            telemetry_sequence=X, dt_sequence=dt, horizon_hours=6,
            station_id="KT-OK-1")
        assert out["wind_speed_status"] == "OK"
        assert out["temperature_c"] is not None

    def test_missing_trail_file_does_not_suppress(self, predictor, tmp_path, monkeypatch):
        monkeypatch.setattr(sensor_health, "DEFAULT_TRAIL", str(tmp_path / "nope.jsonl"))
        sensor_health.clear_live_cache()
        X, dt = make_sequence(2.0)
        out = predictor.predict_from_observed_sequence(
            telemetry_sequence=X, dt_sequence=dt, horizon_hours=6,
            station_id="KT-WHATEVER")
        assert out["wind_speed_status"] == "OK"
        assert out["wind_speed_kmh"] > 0.0


class TestNoWindForecastIsFabricated:
    def test_dead_station_does_not_report_a_calm_forecast_as_real(self, predictor,
                                                                  dead_trail):
        """
        The dangerous outcome is a reader seeing "0.0 m/s, calm" and believing it.
        A dead sensor must be visibly quarantined, not silently zero.
        """
        X, dt = make_sequence(0.0)
        out = predictor.predict_from_observed_sequence(
            telemetry_sequence=X, dt_sequence=dt, horizon_hours=6,
            station_id="KT-DEAD-1")
        assert out["wind_speed_status"] != "OK"
        assert out["wind_sensor_health"] is not None
        assert out["wind_sensor_health"]["verdict"] in ("dead", "absent")
