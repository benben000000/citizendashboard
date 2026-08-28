import os
import sys
import time
from awscrt import io, mqtt
from awsiot import mqtt_connection_builder

MQTT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "mqtt")
CA_PATH = os.path.abspath(os.path.join(MQTT_DIR, "AmazonRootCA1.pem"))
CERT_PATH = os.path.abspath(os.path.join(MQTT_DIR, "e55604370d2fcea8679a2dc02a52eeb033fe201f6f20b89eb33d512777dbc19e-certificate.pem.crt"))
KEY_PATH = os.path.abspath(os.path.join(MQTT_DIR, "e55604370d2fcea8679a2dc02a52eeb033fe201f6f20b89eb33d512777dbc19e-private.pem.key"))
ENDPOINT = "a68bn74ibyvu1-ats.iot.ap-southeast-1.amazonaws.com"

# Candidate Client IDs
candidate_client_ids = [
    "e55604370d2fcea8679a2dc02a52eeb033fe201f6f20b89eb33d512777dbc19e",
    "KT-6CBD47DC5194",
    "KT-4049D3215788",
    "KT-4C31325C7BCC",
    "kloudtrack-listener",
    "kloudtrack-subscriber",
    "kloudtrack-client",
]

def try_connect(port, client_id):
    print(f"Testing port={port}, client_id='{client_id}'...")
    event_loop_group = io.EventLoopGroup(1)
    host_resolver = io.DefaultHostResolver(event_loop_group)
    client_bootstrap = io.ClientBootstrap(event_loop_group, host_resolver)

    try:
        if port == 443:
            mqtt_connection = mqtt_connection_builder.mtls_with_alpn(
                endpoint=ENDPOINT,
                port=443,
                cert_filepath=CERT_PATH,
                pri_key_filepath=KEY_PATH,
                client_bootstrap=client_bootstrap,
                ca_filepath=CA_PATH,
                client_id=client_id,
                clean_session=True,
                keep_alive_secs=30,
            )
        else:
            mqtt_connection = mqtt_connection_builder.mtls_from_path(
                endpoint=ENDPOINT,
                port=port,
                cert_filepath=CERT_PATH,
                pri_key_filepath=KEY_PATH,
                client_bootstrap=client_bootstrap,
                ca_filepath=CA_PATH,
                client_id=client_id,
                clean_session=True,
                keep_alive_secs=30,
            )

        connect_future = mqtt_connection.connect()
        connect_future.result(timeout=6)
        print(f"🎉 SUCCESS! Connected with port={port}, client_id='{client_id}'!")
        mqtt_connection.disconnect().result(timeout=5)
        return True
    except Exception as e:
        print(f"  Failed: {e}")
        return False

def main():
    print("--- Testing Port 443 with ALPN ---")
    for cid in candidate_client_ids:
        if try_connect(443, cid):
            return

    print("\n--- Testing Port 8883 ---")
    for cid in candidate_client_ids:
        if try_connect(8883, cid):
            return

if __name__ == "__main__":
    main()
