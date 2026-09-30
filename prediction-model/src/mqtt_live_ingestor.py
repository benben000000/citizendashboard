"""Long-running AWS IoT MQTT telemetry ingestor.

This is the supported replacement for the archived mqtt_pinn_live_streamer.py.
It subscribes to physical-station telemetry and atomically updates the dashboard
cache. Run it on a persistent worker/VM/container, never in a Vercel function.
"""

from __future__ import annotations

import json
import math
import os
import signal
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from awscrt import io, mqtt
from awsiot import mqtt_connection_builder
from dataset import compute_noaa_heat_index
from dotenv import load_dotenv
from inference import LNNServerlessPredictor
from prediction_audit import get_audit, get_observation_audit, variable_entry


ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env.mqtt.local")

ENDPOINT = os.environ["MQTT_ENDPOINT"]
PORT = int(os.getenv("MQTT_PORT", "8883"))
TOPIC = os.getenv("MQTT_TOPIC", "kloudtrack/+/data")
CLIENT_ID = os.getenv("MQTT_CLIENT_ID", "kloudtrack-dashboard-ingestor")
CA_PATH = os.environ["MQTT_CA_PATH"]
CERT_PATH = os.environ["MQTT_CERT_PATH"]
PRIVATE_KEY_PATH = os.environ["MQTT_PRIVATE_KEY_PATH"]
CACHE_PATH = Path(os.getenv("MQTT_LIVE_CACHE_PATH", str(ROOT / "prediction-model/data/mqtt_live_predictions.json")))
STATION_CACHE_MAX_AGE_SECONDS = int(os.getenv("MQTT_STATION_CACHE_MAX_AGE_SECONDS", "900"))
HISTORY_PATH = Path(os.getenv("MQTT_OBSERVATION_HISTORY_PATH", str(ROOT / "prediction-model/data/mqtt_observation_history.json")))
SEQUENCE_LENGTH = int(os.getenv("MQTT_PREDICTION_SEQUENCE_LENGTH", "24"))
PREDICTION_INTERVAL_SECONDS = int(os.getenv("MQTT_PREDICTION_INTERVAL_SECONDS", "300"))
NO_MESSAGE_TIMEOUT_SECONDS = int(os.getenv("MQTT_NO_MESSAGE_TIMEOUT_SECONDS", "180"))
HISTORY_MAX_AGE_SECONDS = int(os.getenv("MQTT_HISTORY_MAX_AGE_SECONDS", "3600"))
MAX_SEQUENCE_GAP_SECONDS = int(os.getenv("MQTT_MAX_SEQUENCE_GAP_SECONDS", "180"))

STOP = threading.Event()
CACHE_LOCK = threading.Lock()
LAST_MESSAGE_AT = time.monotonic()


def number(payload: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def load_cache() -> dict[str, Any]:
    try:
        with CACHE_PATH.open("r", encoding="utf-8") as source:
            data = json.load(source)
            return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def write_cache(cache: dict[str, Any]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=CACHE_PATH.parent) as temp:
        json.dump(cache, temp, ensure_ascii=False, indent=2)
        temp.write("\n")
        temp_path = Path(temp.name)
    temp_path.replace(CACHE_PATH)


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as source:
            data = json.load(source)
            return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent) as temp:
        json.dump(data, temp, ensure_ascii=False, indent=2)
        temp.write("\n")
        temp_path = Path(temp.name)
    temp_path.replace(path)


def is_recent(entry: Any, now: datetime) -> bool:
    if not isinstance(entry, dict):
        return False
    value = entry.get("timestamp")
    if not isinstance(value, str):
        return False
    try:
        observed_at = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return (now - observed_at).total_seconds() <= STATION_CACHE_MAX_AGE_SECONDS


def parse_timestamp(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def observed_features(raw: dict[str, float | None]) -> list[float] | None:
    required = ("temperature_c", "humidity_pct", "pressure_hpa", "wind_speed_kmh", "wind_direction_deg", "rain_mm")
    if any(raw.get(key) is None for key in required):
        return None
    temperature = float(raw["temperature_c"])
    humidity = float(raw["humidity_pct"])
    wind_direction = math.radians(float(raw["wind_direction_deg"]))
    return [
        temperature,
        float(compute_noaa_heat_index(temperature, humidity)),
        humidity,
        float(raw["pressure_hpa"]),
        float(raw["wind_speed_kmh"]),
        math.sin(wind_direction),
        math.cos(wind_direction),
        float(raw["rain_mm"]),
    ]


def record_observation(station_id: str, timestamp: str, raw: dict[str, float | None]) -> list[dict[str, Any]]:
    features = observed_features(raw)
    history = load_json(HISTORY_PATH)
    stations = history.setdefault("stations", {})
    now_epoch = parse_timestamp(timestamp)
    station_history = [
        entry for entry in stations.get(station_id, [])
        if now_epoch is not None
        and (entry_epoch := parse_timestamp(entry.get("timestamp"))) is not None
        and 0 <= now_epoch - entry_epoch <= HISTORY_MAX_AGE_SECONDS
    ]
    if features is not None:
        last_epoch = parse_timestamp(station_history[-1].get("timestamp")) if station_history else None
        if now_epoch is not None and last_epoch is not None and now_epoch - last_epoch > MAX_SEQUENCE_GAP_SECONDS:
            station_history = []
        station_history.append({"timestamp": timestamp, "features": features, "water_level_m": raw.get("water_level_m")})
        del station_history[:-SEQUENCE_LENGTH]
    stations[station_id] = station_history
    write_json(HISTORY_PATH, history)
    return station_history


# Variables the audit trail tracks individually, mapped to the key the
# predictor emits and to the field the ingestor caches.
AUDITED_VARIABLES = (
    ("temperature", "temperature_c", "temperature_c"),
    ("humidity", "relative_humidity_pct", "humidity_pct"),
    ("pressure", "pressure_hpa", "pressure_hpa"),
    ("wind_speed", "wind_speed_kmh", "wind_speed_kmh"),
)


def audit_prediction(station_id: str, timestamp: str, prediction: dict[str, Any] | None,
                     error: str | None) -> None:
    """
    Append one record to the prediction audit trail.

    Best effort by construction: every failure path is swallowed so that a
    missing audit line can never cost a forecast. The trail records the model
    bundle, the policy version and the commit behind the numbers, which is the
    only way to tell later whether a published value came from the artifact
    currently in the repo.
    """
    try:
        forecast = (prediction or {}).get("forecast") or {}
        if not isinstance(forecast, dict):
            forecast = {}
        variables = {}
        for name, model_key, _cache_key in AUDITED_VARIABLES:
            if model_key in forecast:
                variables[name] = variable_entry(
                    forecast[model_key], producer="lln",
                    nwp_raw=None, nwp_corrected=None)
        try:
            horizon = int(str((prediction or {}).get("horizon", "1h")).rstrip("h"))
        except (TypeError, ValueError):
            horizon = 1
        get_audit().record(
            station_id=station_id,
            horizon_hours=horizon,
            forecast=forecast,
            origin_timestamp=(prediction or {}).get("generated_at") or timestamp,
            device_id=station_id,
            variables=variables,
            nwp={"applied": False,
                 "reason": "router not wired into the live ingestor yet"},
            sensor_health=(forecast or {}).get("wind_sensor_health"),
            extra={"prediction_status": "READY" if prediction else "UNAVAILABLE",
                   "horizon_label": (prediction or {}).get("horizon"),
                   "prediction_error": error,
                   "chance_of_rain_pct": forecast.get("chance_of_rain_pct"),
                   "source": (prediction or {}).get("source")},
        )
    except Exception as exc:  # noqa: BLE001 - never break the ingestor
        print(f"Audit write skipped: {type(exc).__name__}: {exc}")


def audit_observation(station_id: str, timestamp: str, raw: dict[str, Any]) -> None:
    """
    Append the observation that just arrived to the observation trail.

    This is the other half of the audit pair. Predictions cannot be scored
    without the truth they were scored against, and the live cache is
    overwritten in place, so without this a forecast could never be verified
    after the fact.
    """
    try:
        get_observation_audit().record(
            station_id=station_id,
            observed_at_utc=timestamp,
            device_id=station_id,
            telemetry=raw,
        )
    except Exception as exc:  # noqa: BLE001 - never break the ingestor
        print(f"Observation audit write skipped: {type(exc).__name__}: {exc}")


def predict(station_history: list[dict[str, Any]], timestamp: str,
            horizon_hours: int = 1,
            station_id: str | None = None) -> dict[str, Any] | None:
    if len(station_history) < SEQUENCE_LENGTH:
        return None
    features = np.asarray([entry["features"] for entry in station_history], dtype=np.float32)
    if features.shape != (SEQUENCE_LENGTH, 8) or not np.isfinite(features).all():
        return None
    observation_times = [parse_timestamp(entry.get("timestamp")) for entry in station_history]
    if any(value is None for value in observation_times):
        return None
    deltas_seconds = [observation_times[idx] - observation_times[idx - 1] for idx in range(1, len(observation_times))]
    if any(delta <= 0 or delta > MAX_SEQUENCE_GAP_SECONDS for delta in deltas_seconds):
        return None
    dt_hours = np.asarray([deltas_seconds[0], *deltas_seconds], dtype=np.float32) / 3600.0
    latest_water = station_history[-1].get("water_level_m")
    predictor = get_predictor(horizon_hours)
    result = predictor.predict_from_observed_sequence(
        telemetry_sequence=features,
        dt_sequence=dt_hours,
        forecast_origin_timestamp=timestamp,
        horizon_hours=horizon_hours,
        current_water_level=float(latest_water) if latest_water is not None else None,
        feature_names=["temperature", "heat_index", "humidity", "pressure", "wind_speed", "wind_sin", "wind_cos", "precipitation"],
        station_id=station_id,
    )
    return {
        "generated_at": timestamp,
        "source": "mqtt_observed_sequence",
        "horizon": f"{horizon_hours}h",
        "sequence_length": SEQUENCE_LENGTH,
        "max_sequence_gap_seconds": MAX_SEQUENCE_GAP_SECONDS,
        "forecast": result,
    }


# Horizons published operationally. Each has its own trained bundle under
# prediction-model/data/bundles/h<H>/, so each needs its own loaded model. The
# cost is trivial -- ~7-18 ms per inference and ~50 ms to load all five -- so
# there is no reason to serve a single horizon and hide the rest.
OPERATIONAL_HORIZONS: tuple[int, ...] = (1, 3, 6, 12, 24)

PREDICTORS: dict[int, LNNServerlessPredictor] = {}


def get_predictor(horizon_hours: int = 1) -> LNNServerlessPredictor:
    """Loaded model for a horizon, cached for the process lifetime."""
    predictor = PREDICTORS.get(horizon_hours)
    if predictor is None:
        predictor = LNNServerlessPredictor(horizon_hours=horizon_hours)
        PREDICTORS[horizon_hours] = predictor
    return predictor


def on_message(topic: str, payload: bytes, **_: Any) -> None:
    global LAST_MESSAGE_AT
    LAST_MESSAGE_AT = time.monotonic()
    try:
        message = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        print(f"Ignored non-JSON MQTT payload on {topic}")
        return
    if not isinstance(message, dict):
        return

    parts = topic.split("/")
    if len(parts) != 3 or parts[0] != "kloudtrack" or parts[2] != "data":
        print(f"Ignored unexpected MQTT topic: {topic}")
        return

    station_id = parts[1]
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    raw = {
        "temperature_c": number(message, "temperature", "temp", "temp_c"),
        "humidity_pct": number(message, "humidity", "hum", "rh"),
        "pressure_hpa": number(message, "pressure", "pres", "baro"),
        "wind_speed_kmh": number(message, "wind_speed", "wind", "wind_kmh"),
        "wind_direction_deg": number(message, "wind_direction", "wind_dir", "wind_direction_deg", "direction"),
        "water_level_m": number(message, "water_level", "water", "level_m"),
        "rain_mm": number(message, "rain", "rain_mm", "precipitation"),
    }

    with CACHE_LOCK:
        cache = load_cache()
        existing = cache.get("stations", {})
        stations = {
            key: entry for key, entry in existing.items()
            if is_recent(entry, datetime.now(timezone.utc))
        } if isinstance(existing, dict) else {}
        cache["stations"] = stations
        station_history = record_observation(station_id, timestamp, raw)
        prior = stations.get(station_id, {})
        prior_h1 = prior.get("operational_prediction") if isinstance(prior, dict) else None
        prior_generated = parse_timestamp(prior_h1.get("generated_at")) if isinstance(prior_h1, dict) else None
        should_predict = prior_generated is None or (datetime.now(timezone.utc).timestamp() - prior_generated) >= PREDICTION_INTERVAL_SECONDS

        # One prediction per published horizon. A horizon that fails is recorded
        # as absent rather than dropped, so a partial failure is visible instead
        # of silently shortening the forecast.
        prior_all = prior.get("operational_predictions") if isinstance(prior, dict) else None
        predictions: dict[str, Any] = dict(prior_all) if isinstance(prior_all, dict) else {}
        prediction_error = None
        errors_by_horizon: dict[str, str] = {}
        if should_predict:
            for horizon in OPERATIONAL_HORIZONS:
                key = f"{horizon}h"
                try:
                    result = predict(station_history, timestamp, horizon,
                                     station_id)
                except Exception as error:  # noqa: BLE001
                    errors_by_horizon[key] = f"{type(error).__name__}: {error}"
                    continue
                if result is None:
                    continue
                predictions[key] = result
            if errors_by_horizon and not predictions:
                prediction_error = "; ".join(f"{k}: {v}" for k, v in errors_by_horizon.items())

        # `operational_prediction` stays as the 1h entry so existing consumers
        # keep working while callers migrate to the per-horizon map.
        operational_prediction = predictions.get("1h") or prior_h1
        stations[station_id] = {
            "operational_predictions": predictions,
            "timestamp": timestamp,
            "station_id": station_id,
            "topic": topic,
            "qc_status": "RAW",
            "raw_telemetry": raw,
            "operational_prediction": operational_prediction,
            "prediction_status": "READY" if predictions else "WAITING_FOR_COMPLETE_SEQUENCE",
            "prediction_error": prediction_error,
            "prediction_errors_by_horizon": errors_by_horizon,
            "published_horizons": sorted(predictions),
        }
        cache["last_updated"] = timestamp
        cache["total_active_stations"] = len(stations)
        write_cache(cache)
        audit_observation(station_id, timestamp, raw)
        if should_predict:
            for key, prediction in predictions.items():
                try:
                    audit_prediction(station_id, timestamp, prediction,
                                     errors_by_horizon.get(key))
                except Exception as exc:  # noqa: BLE001
                    print(f"Audit write skipped for {key}: {type(exc).__name__}: {exc}")
    print(f"Received {station_id} at {timestamp}")


def main() -> None:
    global LAST_MESSAGE_AT
    required_paths = [CA_PATH, CERT_PATH, PRIVATE_KEY_PATH]
    missing = [path for path in required_paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"MQTT credential file(s) not found: {', '.join(missing)}")

    event_loop_group = io.EventLoopGroup(1)
    host_resolver = io.DefaultHostResolver(event_loop_group)
    bootstrap = io.ClientBootstrap(event_loop_group, host_resolver)

    while not STOP.is_set():
        connection = None
        try:
            print(f"Connecting to AWS IoT at {ENDPOINT}:{PORT} as {CLIENT_ID}", flush=True)
            connection = mqtt_connection_builder.mtls_from_path(
                endpoint=ENDPOINT,
                port=PORT,
                cert_filepath=CERT_PATH,
                pri_key_filepath=PRIVATE_KEY_PATH,
                ca_filepath=CA_PATH,
                client_bootstrap=bootstrap,
                client_id=CLIENT_ID,
                clean_session=False,
                keep_alive_secs=30,
            )
            connection.connect().result(timeout=20)
            connection.subscribe(topic=TOPIC, qos=mqtt.QoS.AT_LEAST_ONCE, callback=on_message)[0].result(timeout=20)
            LAST_MESSAGE_AT = time.monotonic()
            print(f"MQTT connected: {ENDPOINT}:{PORT}, subscribed to {TOPIC} as {CLIENT_ID}")
            while not STOP.wait(1):
                if time.monotonic() - LAST_MESSAGE_AT > NO_MESSAGE_TIMEOUT_SECONDS:
                    raise TimeoutError(
                        f"No MQTT messages received for {NO_MESSAGE_TIMEOUT_SECONDS}s; reconnecting"
                    )
        except Exception as error:
            print(f"MQTT connection failed ({error}); retrying in 10 seconds")
            STOP.wait(10)
        finally:
            if connection is not None:
                try:
                    connection.disconnect().result(timeout=10)
                except Exception:
                    pass


if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda *_: STOP.set())
    signal.signal(signal.SIGTERM, lambda *_: STOP.set())
    main()
