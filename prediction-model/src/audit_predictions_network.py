import urllib.request
import json
import datetime
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

stations_url = "http://citizen.kloudtechsea.com/api/telemetry/dashboard"
with urllib.request.urlopen(stations_url) as res:
    raw_stations = json.loads(res.read().decode("utf-8"))["data"]

print("=========================================================================================================")
print("                   PINN-LNN NOWCAST PREDICTION AUDIT (+1h, +3h, +6h)")
print("=========================================================================================================")
print(f"{'Station Name':<28} | {'Current Temp':<12} | {'+1h Forecast':<14} | {'+3h Forecast':<14} | {'+6h Forecast':<14} | {'Rain Trend'}")
print("-" * 105)

for s in raw_stations:
    st = s["station"]
    sid = st["stationPublicId"]
    name = st["stationName"]
    
    url = f"http://localhost/api/prediction/station/{sid}"
    try:
        req = urllib.request.Request(url, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as p_res:
            data = json.loads(p_res.read().decode("utf-8")).get("data", {})
            current = data.get("current", {})
            forecasts = data.get("forecasts", [])
            
            c_t = current.get("temperature", 25.0)
            f1 = forecasts[0].get("temperature", c_t) if len(forecasts) > 0 else c_t
            f3 = forecasts[2].get("temperature", c_t) if len(forecasts) > 2 else c_t
            f6 = forecasts[5].get("temperature", c_t) if len(forecasts) > 5 else c_t
            
            rain_trend = "Overcast Monsoon" if current.get("humidity", 90) > 85 else "Clear"
            print(f"{name[:28]:<28} | {c_t:5.1f} °C      | {f1:5.1f} °C        | {f3:5.1f} °C        | {f6:5.1f} °C        | {rain_trend}")
    except Exception as e:
        print(f"{name[:28]:<28} | Local API Error: {e}")

print("=========================================================================================================")
