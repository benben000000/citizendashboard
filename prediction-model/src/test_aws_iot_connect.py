import os
import sys
import time
import json
from awscrt import io, mqtt, auth, http
from awsiot import mqtt_connection_builder

# Ensure UTF-8 output on Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MQTT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "mqtt")
CA_PATH = os.path.abspath(os.path.join(MQTT_DIR, "AmazonRootCA1.pem"))
CERT_PATH = os.path.abspath(os.path.join(MQTT_DIR, "e55604370d2fcea8679a2dc02a52eeb033fe201f6f20b89eb33d512777dbc19e-certificate.pem.crt"))
KEY_PATH = os.path.abspath(os.path.join(MQTT_DIR, "e55604370d2fcea8679a2dc02a52eeb033fe201f6f20b89eb33d512777dbc19e-private.pem.key"))

ENDPOINT = "a68bn74ibyvu1-ats.iot.ap-southeast-1.amazonaws.com"
PORT = 8883
CLIENT_ID = "kloudtrack-pinn-listener-local"

print(f"Connecting via AWS IoT SDK v2 to: {ENDPOINT}:{PORT}")
print(f"  CA: {CA_PATH}")
print(f"  Cert: {CERT_PATH}")
print(f"  Key: {KEY_PATH}")

def on_connection_interrupted(connection, error, **kwargs):
    print(f"⚠️ Connection interrupted. error: {error}")

def on_connection_resumed(connection, return_code, session_present, **kwargs):
    print(f"✅ Connection resumed. return_code: {return_code} session_present: {session_present}")

def on_message_received(topic, payload, dup, qos, retain, **kwargs):
    print(f"\n📥 [MQTT PACKET RECEIVED] Topic: {topic}")
    try:
        data = json.loads(payload.decode("utf-8"))
        print(f"   Parsed JSON: {json.dumps(data, indent=2)}")
    except Exception:
        print(f"   Raw: {payload.decode('utf-8', errors='ignore')}")

def main():
    event_loop_group = io.EventLoopGroup(1)
    host_resolver = io.DefaultHostResolver(event_loop_group)
    client_bootstrap = io.ClientBootstrap(event_loop_group, host_resolver)

    print("Building AWS IoT MQTT Connection...")
    mqtt_connection = mqtt_connection_builder.mtls_from_path(
        endpoint=ENDPOINT,
        port=PORT,
        cert_filepath=CERT_PATH,
        pri_key_filepath=KEY_PATH,
        client_bootstrap=client_bootstrap,
        ca_filepath=CA_PATH,
        on_connection_interrupted=on_connection_interrupted,
        on_connection_resumed=on_connection_resumed,
        client_id=CLIENT_ID,
        clean_session=False,
        keep_alive_secs=30,
    )

    print("Connecting to AWS IoT Core...")
    connect_future = mqtt_connection.connect()
    connect_future.result(timeout=15)
    print("🎉 SUCCESS! Connected to AWS IoT Core Broker!")

    # Subscribe to telemetry topics
    topics = ["kloudtrack/+/data", "kloudtrack/#"]
    for t in topics:
        print(f"📡 Subscribing to topic '{t}'...")
        subscribe_future, packet_id = mqtt_connection.subscribe(
            topic=t,
            qos=mqtt.QoS.AT_LEAST_ONCE,
            callback=on_message_received,
        )
        subscribe_result = subscribe_future.result(timeout=10)
        print(f"  ✓ Subscribed to '{t}' with QoS: {subscribe_result['qos']}")

    print("\n🎧 Passive Read-Only Listener Active! Listening for live incoming packets (15s)...")
    time.sleep(15)

    print("Disconnecting gracefully...")
    disconnect_future = mqtt_connection.disconnect()
    disconnect_future.result(timeout=10)
    print("Disconnected cleanly.")

if __name__ == "__main__":
    main()
