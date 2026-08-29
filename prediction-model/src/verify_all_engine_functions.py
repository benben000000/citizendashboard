import urllib.request
import json
import time

base_url = 'http://localhost'

endpoints = [
    ('/weather', 'AWS Weather Dashboard Page'),
    ('/water-level', 'WLMS River Stage Dashboard Page'),
    ('/prediction', 'PINN-LNN Continuous Nowcast Page'),
    ('/api/telemetry/dashboard', 'Live Telemetry Dashboard API'),
    ('/api/water-level/dashboard', 'Live Water Level Dashboard API'),
    ('/api/prediction/station/3nzr48bG?horizon=24h', 'LNN Prediction Engine API (Balanga 24h)'),
    ('/api/prediction/station/3nzr48bG?horizon=6h', 'LNN Prediction Engine API (Balanga 6h)'),
    ('/api/telemetry/station/3nzr48bG/parameter/temperature?interval=15', 'Real 24h Temperature History API'),
    ('/api/telemetry/station/3nzr48bG/parameter/humidity?interval=15', 'Real 24h Humidity History API'),
    ('/api/telemetry/station/3nzr48bG/parameter/pressure?interval=15', 'Real 24h Pressure History API'),
    ('/api/telemetry/station/3nzr48bG/parameter/uvIndex?interval=15', 'Real 24h UV Index History API'),
    ('/api/telemetry/processed', 'Processed Telemetry Downstream API'),
    ('/api/telemetry/station/3nzr48bG/processed?history=true', 'Single-Station Processed API'),
]

print("===============================================================")
print("   SYSTEM-WIDE ENGINE & API VERIFICATION SUITE")
print("===============================================================")

all_pass = True
for path, desc in endpoints:
    t0 = time.time()
    try:
        req = urllib.request.Request(
            f"{base_url}{path}",
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        with urllib.request.urlopen(req, timeout=8) as res:
            elapsed_ms = (time.time() - t0) * 1000
            status = res.status
            content = res.read().decode("utf-8")
            is_json = "application/json" in res.headers.get("Content-Type", "")
            
            detail = ""
            if is_json:
                data = json.loads(content)
                if "data" in data:
                    d = data["data"]
                    if isinstance(d, list):
                        detail = f"[{len(d)} items]"
                    elif isinstance(d, dict):
                        detail = f"[{len(d.keys())} keys]"
                elif "forecast" in data:
                    detail = f"[{len(data.get('forecast', []))} forecast steps]"
            else:
                detail = f"[{len(content)} bytes HTML]"
            
            print(f"  [PASS] {status} ({elapsed_ms:5.1f}ms) : {desc} -> {path} {detail}")
    except Exception as e:
        all_pass = False
        print(f"  [FAIL] ERROR         : {desc} -> {path} ERROR: {e}")

print("===============================================================")
print("OVERALL STATUS: ALL SYSTEMS OPERATIONAL (100% PASS)" if all_pass else "OVERALL STATUS: SOME FAILED")
print("===============================================================")
