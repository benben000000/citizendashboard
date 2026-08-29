"""
Comprehensive audit of all 23 weather stations across all metrics:
Temperature, Humidity, Pressure, Heat Index, Wind Speed, Wind Direction, Precipitation.
"""

import sys
import json
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def audit_all_stations():
    print("=" * 135)
    print("🔍 COMPREHENSIVE STATION-BY-STATION TELEMETRY & STATS AUDIT (ALL 23 STATIONS)")
    print("=" * 135)

    req = urllib.request.urlopen("http://localhost/api/telemetry/dashboard", timeout=8)
    data = json.loads(req.read().decode("utf-8")).get("data", [])

    print(f"{'Station Name':<30} | {'ID':<9} | {'Temp':<7} | {'Hum':<6} | {'Pres':<9} | {'HI':<7} | {'Wind Dir':<9} | {'Wind Spd':<9} | {'Precip':<7} | {'Audit Issues / Flags'}")
    print("-" * 135)

    for item in data:
        st = item.get("station", {})
        tel = item.get("telemetry", {})
        name = st.get("stationName", "Unknown")[:29]
        sid = st.get("stationPublicId", "unknown")
        
        t = tel.get("temperature")
        h = tel.get("humidity")
        p = tel.get("pressure")
        hi = tel.get("heatIndex")
        wdir = tel.get("windDirection")
        wspd = tel.get("windSpeed")
        precip = tel.get("precipitation")

        issues = []
        if wspd == 0 or wspd is None:
            issues.append("⚠️ Calm/Zero Wind (0 km/h)")
        if t is None or t == 0:
            issues.append("❌ Missing Temp")
        if h is None or h == 0:
            issues.append("❌ Missing Hum")
        if p is None or p == 0:
            issues.append("❌ Missing Pres")

        issue_str = ", ".join(issues) if issues else "✅ ALL STATS ACTIVE"

        t_str = f"{t:.1f}°C" if t is not None else "N/A"
        h_str = f"{h:.1f}%" if h is not None else "N/A"
        p_str = f"{p:.1f}hPa" if p is not None else "N/A"
        hi_str = f"{hi:.1f}°C" if hi is not None else "N/A"
        wdir_str = f"{wdir}°" if wdir is not None else "N/A"
        wspd_str = f"{wspd:.1f}km/h" if wspd is not None else "N/A"
        precip_str = f"{precip:.1f}mm" if precip is not None else "N/A"

        print(f"{name:<30} | {sid:<9} | {t_str:<7} | {h_str:<6} | {p_str:<9} | {hi_str:<7} | {wdir_str:<9} | {wspd_str:<9} | {precip_str:<7} | {issue_str}")

    print("=" * 135)

if __name__ == "__main__":
    audit_all_stations()
