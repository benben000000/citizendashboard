import os
from awscrt import io, mqtt
from awsiot import mqtt_connection_builder

MQTT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "mqtt")
CA1_PATH = os.path.abspath(os.path.join(MQTT_DIR, "AmazonRootCA1.pem"))
CA3_PATH = os.path.abspath(os.path.join(MQTT_DIR, "AmazonRootCA3.pem"))
CERT_PATH = os.path.abspath(os.path.join(MQTT_DIR, "e55604370d2fcea8679a2dc02a52eeb033fe201f6f20b89eb33d512777dbc19e-certificate.pem.crt"))
KEY_PATH = os.path.abspath(os.path.join(MQTT_DIR, "e55604370d2fcea8679a2dc02a52eeb033fe201f6f20b89eb33d512777dbc19e-private.pem.key"))
ENDPOINT = "a68bn74ibyvu1-ats.iot.ap-southeast-1.amazonaws.com"

def test_ca(ca_file, port):
    print(f"Testing with {os.path.basename(ca_file)} on port {port}...")
    event_loop_group = io.EventLoopGroup(1)
    host_resolver = io.DefaultHostResolver(event_loop_group)
    client_bootstrap = io.ClientBootstrap(event_loop_group, host_resolver)

    try:
        mqtt_connection = mqtt_connection_builder.mtls_from_path(
            endpoint=ENDPOINT,
            port=port,
            cert_filepath=CERT_PATH,
            pri_key_filepath=KEY_PATH,
            client_bootstrap=client_bootstrap,
            ca_filepath=ca_file,
            client_id="KT-6CBD47DC5194",
            clean_session=True,
            keep_alive_secs=30,
        )
        connect_future = mqtt_connection.connect()
        connect_future.result(timeout=6)
        print(f"🎉 SUCCESS with {os.path.basename(ca_file)} on port {port}!")
        mqtt_connection.disconnect().result(timeout=5)
        return True
    except Exception as e:
        print(f"  Failed: {e}")
        return False

if __name__ == "__main__":
    test_ca(CA1_PATH, 8883)
    test_ca(CA3_PATH, 8883)
    test_ca(CA1_PATH, 443)
    test_ca(CA3_PATH, 443)
