import urllib.request
import json
import datetime

stations_url = 'http://citizen.kloudtechsea.com/api/telemetry/dashboard'
with urllib.request.urlopen(stations_url) as res:
    stations = json.loads(res.read().decode('utf-8'))['data']

now_utc = datetime.datetime.now(datetime.timezone.utc)
now_pht = now_utc + datetime.timedelta(hours=8)
ph_midnight_utc = datetime.datetime(now_pht.year, now_pht.month, now_pht.day, 0, 0, 0, tzinfo=datetime.timezone.utc) - datetime.timedelta(hours=8)
start_of_today_iso = ph_midnight_utc.isoformat().replace('+00:00', '.000Z')

print("=========================================================================================================")
print("                   FULL NETWORK PRECIPITATION RAW ANOMALY AUDIT")
print("=========================================================================================================")
for s in stations:
    st = s['station']
    sid = st['stationPublicId']
    name = st['stationName']
    
    url = f'http://citizen.kloudtechsea.com/api/telemetry/station/{sid}/parameter/precipitation?interval=15'
    try:
        req = urllib.request.Request(url, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=5) as p_res:
            pts = json.loads(p_res.read().decode('utf-8')).get('data', [])
            today_pts = [p for p in pts if p.get('recordedAt', '') >= start_of_today_iso and p.get('value') is not None]
            
            vals = [p['value'] for p in today_pts]
            max_val = max(vals) if vals else 0
            spikes = [p for p in today_pts if p['value'] > 100.0]
            
            raw_sum = sum(vals)
            raw_integrated = raw_sum * 0.25
            
            clean_vals = [v for v in vals if v <= 100.0]
            clean_integrated = sum(clean_vals) * 0.25
            
            print(f"{name[:30]:<30} ({sid}): Points={len(today_pts):2} | Max={max_val:6.1f} mm/h | Spikes={len(spikes)} | Raw Integrated={raw_integrated:6.1f} mm | Clean Integrated={clean_integrated:6.1f} mm")
            for sp in spikes:
                print(f"   --> SPIKE ANOMALY: {sp['recordedAt']} -> Value: {sp['value']} mm/h (REBOOT/ADC OVERFLOW GLITCH)")
    except Exception as e:
        print(f"{name[:30]:<30} ({sid}): ERR -> {e}")
print("=========================================================================================================")
