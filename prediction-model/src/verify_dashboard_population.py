import urllib.request
import json
import sys

# Ensure UTF-8 output on Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

print("=" * 115)
print("🌐 CHECKING LOCALHOST LIVE WEATHER & WATER LEVEL API POPULATION")
print("=" * 115)

try:
    req_w = urllib.request.urlopen("http://localhost/api/telemetry/dashboard", timeout=5)
    data_w = json.loads(req_w.read().decode("utf-8")).get("data", [])
    print(f"\n🌤️ WEATHER PAGE TELEMETRY (Total Stations: {len(data_w)})")
    print(f"{'Station Name':<30} | {'Station ID':<10} | {'Temp':<8} | {'Humidity':<10} | {'Pressure':<11} | {'Heat Index':<10} | {'Wind'}")
    print("-" * 115)
    for item in data_w:
        st = item.get("station", {})
        tel = item.get("telemetry", {})
        print(f"{st.get('stationName', 'Unknown')[:30]:<30} | {st.get('stationPublicId', ''):<10} | {tel.get('temperature', 0):<8.1f}°C | {tel.get('humidity', 0):<10.1f}% | {tel.get('pressure', 0):<11.1f}hPa | {tel.get('heatIndex', 0):<10.1f}°C | {tel.get('windSpeed', 0):.1f} km/h")
except Exception as e:
    print("Error querying weather API:", e)

try:
    req_wl = urllib.request.urlopen("http://localhost/api/water-level/dashboard", timeout=5)
    data_wl = json.loads(req_wl.read().decode("utf-8")).get("data", [])
    print(f"\n🌊 WATER LEVEL PAGE TELEMETRY (Total Stations: {len(data_wl)})")
    print(f"{'Station Name':<30} | {'Station ID':<10} | {'Calc Water Level':<18} | {'Min Range':<10} | {'Max Range':<10} | {'Sample Int'}")
    print("-" * 115)
    for item in data_wl:
        st = item.get("station", {})
        wl = item.get("waterLevel", {})
        print(f"{st.get('stationName', 'Unknown')[:30]:<30} | {st.get('stationPublicId', ''):<10} | {wl.get('calculatedWaterLevel', 0):<18.1f}cm | {wl.get('minimum', 0):<10.1f}cm | {wl.get('maximum', 0):<10.1f}cm | {wl.get('sampleInterval', 0)}s (15m)")
except Exception as e:
    print("Error querying water level API:", e)

print("=" * 115)
