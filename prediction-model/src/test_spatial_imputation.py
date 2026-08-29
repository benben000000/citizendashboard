import math
import json
import sys
import urllib.request

# Ensure UTF-8 output on Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def haversine_km(loc1, loc2):
    lon1, lat1 = loc1
    lon2, lat2 = loc2
    r = 6371
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))

req = urllib.request.urlopen('http://citizen.kloudtechsea.com/api/telemetry/dashboard')
data = json.loads(req.read().decode('utf-8')).get('data', [])

healthy = []
faulty = []

for item in data:
    st = item['station']
    tel = item['telemetry']
    t = tel.get('temperature', 0) or 0
    h = tel.get('humidity', 0) or 0
    p = tel.get('pressure', 0) or 0
    if 16 <= t <= 43 and 20 <= h <= 100 and 970 <= p <= 1030:
        healthy.append(item)
    else:
        faulty.append(item)

print("=" * 115)
print(f"📡 REAL-TIME SPATIAL IDW NEIGHBOR RECONSTRUCTION AUDIT")
print(f"🟢 Healthy Stations: {len(healthy)} | 🔴 Faulty / Down Stations: {len(faulty)}")
print("=" * 115)

for item in faulty:
    st = item['station']
    tel = item['telemetry']
    name = st['stationName']
    loc = st['location']
    
    tot_w = 0
    w_t, w_h, w_p, w_w = 0, 0, 0, 0
    neighbors = []
    
    for h in healthy:
        h_st = h['station']
        h_tel = h['telemetry']
        h_loc = h_st['location']
        d_km = haversine_km(loc, h_loc)
        w = math.exp(-(d_km ** 2) / (2 * 25 ** 2)) + 0.001
        tot_w += w
        w_t += h_tel['temperature'] * w
        w_h += h_tel['humidity'] * w
        w_p += h_tel['pressure'] * w
        w_w += h_tel['windSpeed'] * w
        neighbors.append((d_km, h_st['stationName']))
        
    calc_t = w_t / tot_w
    calc_h = w_h / tot_w
    calc_p = w_p / tot_w
    calc_w = w_w / tot_w
    calc_hi = calc_t + (calc_h / 100.0) * 5.5
    
    neighbors.sort(key=lambda x: x[0])
    top_neighbors_str = ", ".join([f"{n[1]} ({n[0]:.1f} km)" for n in neighbors[:3]])
    
    print(f"\n🚨 FAULTY SENSOR: {name}")
    print(f"   • Raw Corrupted Telemetry: Temp={tel.get('temperature')}°C | Pressure={tel.get('pressure')}hPa | Humidity={tel.get('humidity')}%")
    print(f"   • Nearest Healthy Donors: {top_neighbors_str}")
    print(f"   ✨ SPATIAL RECONSTRUCTED: Temp={calc_t:.1f}°C | Hum={calc_h:.1f}% | Pres={calc_p:.1f}hPa | HI={calc_hi:.1f}°C | Wind={calc_w:.1f} km/h")
    print(f"   • Status: Gated & Imputed (Reverts automatically to sensor when online)")

print("\n" + "=" * 115)
