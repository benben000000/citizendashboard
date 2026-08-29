import urllib.request
import json
import sys
from datetime import datetime, timezone
import dateutil.parser

# Ensure UTF-8 output on Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

url_w = 'http://citizen.kloudtechsea.com/api/telemetry/dashboard'
url_wl = 'http://citizen.kloudtechsea.com/api/water-level/dashboard'

print("=" * 135)
print("🔍 LIVE AUDIT OF PROD KLOUDTRACK STATIONS (citizen.kloudtechsea.com)")
print(f"🕒 Current Audit Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S PST')}")
print("=" * 135)

try:
    req = urllib.request.urlopen(url_w, timeout=10)
    data = json.loads(req.read().decode('utf-8'))
    stations = data.get('data', [])
    print(f"Total Weather Stations Reported: {len(stations)}\n")
    print(f"{'Station Name':<30} | {'Station ID':<10} | {'Temp':<7} | {'Hum':<6} | {'Press':<8} | {'Wind':<6} | {'HI':<6} | {'Recorded At':<20} | {'Anomaly & Health Status'}")
    print("-" * 135)
    
    now_utc = datetime.now(timezone.utc)

    for item in stations:
        st = item.get('station', {})
        tel = item.get('telemetry', {})
        name = st.get('stationName', 'Unknown')[:28]
        sid = st.get('stationPublicId', 'Unknown')
        temp = tel.get('temperature', 0) or 0
        hum = tel.get('humidity', 0) or 0
        pres = tel.get('pressure', 0) or 0
        wind = tel.get('windSpeed', 0) or 0
        hi = tel.get('heatIndex', 0) or 0
        rec_at = tel.get('recordedAt', '')

        anomalies = []
        
        # Check staleness
        try:
            rec_dt = dateutil.parser.parse(rec_at)
            age_hours = (now_utc - rec_dt).total_seconds() / 3600.0
            if age_hours > 24:
                anomalies.append(f"STALE({age_hours/24:.0f}d old)")
            elif age_hours > 2:
                anomalies.append(f"DELAYED({age_hours:.1f}h)")
        except Exception:
            anomalies.append("NO_TIMESTAMP")

        # Check Absurd Telemetry Values
        if temp < 16.0 or temp > 43.0:
            anomalies.append(f"ABSURD_TEMP({temp}°C)")
        if hum < 20.0 or hum > 100.0:
            anomalies.append(f"ABSURD_HUM({hum}%)")
        if pres < 970.0 or pres > 1030.0:
            anomalies.append(f"ABSURD_PRES({pres}hPa)")
        if hi > 52.0:
            anomalies.append(f"ABSURD_HI({hi}°C)")
        if wind < 0 or wind > 180:
            anomalies.append(f"ABSURD_WIND({wind}km/h)")
            
        status_str = "⚠️ " + ", ".join(anomalies) if anomalies else "✅ ONLINE_HEALTHY"
        print(f"{name:<30} | {sid:<10} | {temp:<7.1f} | {hum:<6.1f} | {pres:<8.1f} | {wind:<6.1f} | {hi:<6.1f} | {rec_at[:19]:<20} | {status_str}")

except Exception as e:
    print("Error fetching prod weather telemetry:", e)

print("\n" + "=" * 135)
print("🌊 LIVE WATER LEVEL STATIONS AUDIT")
print("=" * 135)

try:
    req_wl = urllib.request.urlopen(url_wl, timeout=10)
    data_wl = json.loads(req_wl.read().decode('utf-8'))
    wl_stations = data_wl.get('data', [])
    print(f"Total Water Level Stations Reported: {len(wl_stations)}\n")
    print(f"{'Station Name':<30} | {'Station ID':<10} | {'Calc Level':<10} | {'Min':<8} | {'Max':<8} | {'Recorded At':<20} | {'Status'}")
    print("-" * 135)
    
    for item in wl_stations:
        st = item.get('station', {})
        wl = item.get('waterLevel', {})
        name = st.get('stationName', 'Unknown')[:28]
        sid = st.get('stationPublicId', 'Unknown')
        calc_lvl = wl.get('calculatedWaterLevel', 0) or 0
        min_lvl = wl.get('minimum', 0) or 0
        max_lvl = wl.get('maximum', 0) or 0
        rec_at = wl.get('recordedAt', '')

        wl_anomalies = []
        if calc_lvl < 0 or calc_lvl > 2000:
            wl_anomalies.append(f"ABSURD_LEVEL({calc_lvl}cm)")
        if min_lvl < 0 or max_lvl < 0:
            wl_anomalies.append("NEGATIVE_RANGE")
        
        status_str = "⚠️ " + ", ".join(wl_anomalies) if wl_anomalies else "✅ ONLINE_HEALTHY"
        print(f"{name:<30} | {sid:<10} | {calc_lvl:<10.1f} | {min_lvl:<8.1f} | {max_lvl:<8.1f} | {rec_at[:19]:<20} | {status_str}")

except Exception as e:
    print("Error fetching prod water level telemetry:", e)
