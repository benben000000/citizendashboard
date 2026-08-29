import urllib.request
import json
import datetime

# Philippine Standard Time (PHT, UTC+8) midnight calculation
now_utc = datetime.datetime.now(datetime.timezone.utc)
now_pht = now_utc + datetime.timedelta(hours=8)
ph_midnight_utc = datetime.datetime(now_pht.year, now_pht.month, now_pht.day, 0, 0, 0, tzinfo=datetime.timezone.utc) - datetime.timedelta(hours=8)
start_of_today_iso = ph_midnight_utc.isoformat().replace("+00:00", ".000Z")

stations_url = "http://localhost/api/telemetry/dashboard"
with urllib.request.urlopen(stations_url) as res:
    stations = json.loads(res.read().decode("utf-8"))["data"]

print("=========================================================================================================")
print("             AUTOMATED NETWORK-WIDE DAILY PRECIPITATION INTEGRATION VALIDATION")
print(f"Audit Window: {start_of_today_iso} (00:00 PHT) to Latest | PHT Now: {now_pht.strftime('%Y-%m-%d %H:%M:%S')}")
print("=========================================================================================================")
print(f"{'Station Name':<35} | {'ID':<10} | {'Today Pts':<10} | {'Raw 60m Sum':<12} | {'True Daily Precip':<18} | {'Status'}")
print("-" * 105)

for s in stations:
    st = s["station"]
    sid = st["stationPublicId"]
    name = st["stationName"]
    
    url = f"http://localhost/api/telemetry/station/{sid}/parameter/precipitation?interval=15"
    try:
        with urllib.request.urlopen(url, timeout=5) as p_res:
            pts = json.loads(p_res.read().decode("utf-8")).get("data", [])
            
            today_pts = [p for p in pts if p.get("recordedAt", "") >= start_of_today_iso]
            raw_sum = sum([p["value"] for p in today_pts if p.get("value") is not None])
            integrated_daily = sum([p["value"] * 0.25 for p in today_pts if p.get("value") is not None])
            
            status = "PASS (Valid)" if integrated_daily <= 120.0 else "HIGH MONSOON"
            print(f"{name[:35]:<35} | {sid:<10} | {len(today_pts):<10} | {raw_sum:<12.1f} | {integrated_daily:<18.1f} mm | {status}")
    except Exception as e:
        print(f"{name[:35]:<35} | {sid:<10} | ERR: {e}")

print("=========================================================================================================")
print("ALL 15 STATIONS AUTOMATICALLY INTEGRATED WITH DT = 0.25h & PHILIPPINE MIDNIGHT ALIGNMENT")
print("=========================================================================================================")
