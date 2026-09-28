"""Long-running AWS IoT MQTT telemetry ingestor.

This is the supported replacement for the archived mqtt_pinn_live_streamer.py.
It subscribes to physical-station telemetry and atomically updates the dashboard
cache. Run it on a persistent worker/VM/container, never in a Vercel function.
"""

from __future__ import annotations

import json
import os
import signal
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from awscrt import io, mqtt
from awsiot import mqtt_connection_builder
from dotenv import load_dotenv


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
        "water_level_m": number(message, "water_level", "water", "level_m"),
        "rain_mm": number(message, "rain", "rain_mm", "precipitation"),
    }

    with CACHE_LOCK:
        cache = load_cache()
        stations = cache.setdefault("stations", {})
        stations[station_id] = {
            "timestamp": timestamp,
            "station_id": station_id,
            "topic": topic,
            "qc_status": "RAW",
            "raw_telemetry": raw,
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
