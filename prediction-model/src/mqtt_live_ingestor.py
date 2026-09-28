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

STOP = threading.Event()
CACHE_LOCK = threading.Lock()


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
    station_history = stations.setdefault(station_id, [])
    if features is not None:
        station_history.append({"timestamp": timestamp, "features": features, "water_level_m": raw.get("water_level_m")})
        del station_history[:-SEQUENCE_LENGTH]
    write_json(HISTORY_PATH, history)
    return station_history


def predict(station_history: list[dict[str, Any]], timestamp: str) -> dict[str, Any] | None:
    if len(station_history) < SEQUENCE_LENGTH:
        return None
    features = np.asarray([entry["features"] for entry in station_history], dtype=np.float32)
    if features.shape != (SEQUENCE_LENGTH, 8) or not np.isfinite(features).all():
        return None
    latest_water = station_history[-1].get("water_level_m")
    predictor = get_predictor()
    result = predictor.predict_from_observed_sequence(
        telemetry_sequence=features,
        forecast_origin_timestamp=timestamp,
        horizon_hours=1,
        current_water_level=float(latest_water) if latest_water is not None else None,
        feature_names=["temperature", "heat_index", "humidity", "pressure", "wind_speed", "wind_sin", "wind_cos", "precipitation"],
    )
    return {"generated_at": timestamp, "source": "mqtt_observed_sequence", "sequence_length": SEQUENCE_LENGTH, "forecast": result}


PREDICTOR: LNNServerlessPredictor | None = None


def get_predictor() -> LNNServerlessPredictor:
    global PREDICTOR
    if PREDICTOR is None:
        PREDICTOR = LNNServerlessPredictor(horizon_hours=1)
    return PREDICTOR


def on_message(topic: str, payload: bytes, **_: Any) -> None:
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
        prior_prediction = prior.get("operational_prediction") if isinstance(prior, dict) else None
        prior_generated = parse_timestamp(prior_prediction.get("generated_at")) if isinstance(prior_prediction, dict) else None
        should_predict = prior_generated is None or (datetime.now(timezone.utc).timestamp() - prior_generated) >= PREDICTION_INTERVAL_SECONDS
        operational_prediction = prior_prediction
        prediction_error = None
        if should_predict:
            try:
                operational_prediction = predict(station_history, timestamp)
            except Exception as error:
                prediction_error = str(error)
        stations[station_id] = {
            "timestamp": timestamp,
            "station_id": station_id,
            "topic": topic,
            "qc_status": "RAW",
            "raw_telemetry": raw,
            "operational_prediction": operational_prediction,
            "prediction_status": "READY" if operational_prediction else "WAITING_FOR_COMPLETE_SEQUENCE",
            "prediction_error": prediction_error,
        }
        cache["last_updated"] = timestamp
        cache["total_active_stations"] = len(stations)
        write_cache(cache)
    print(f"Received {station_id} at {timestamp}")


def main() -> None:
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
            print(f"MQTT connected: {ENDPOINT}:{PORT}, subscribed to {TOPIC} as {CLIENT_ID}")
            STOP.wait()
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
