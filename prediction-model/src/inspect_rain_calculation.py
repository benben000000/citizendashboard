import urllib.request
import json
import datetime

url = 'http://citizen.kloudtechsea.com/api/telemetry/station/95pM7BAV/parameter/precipitation?interval=15'
with urllib.request.urlopen(url) as res:
    data = json.loads(res.read().decode('utf-8'))['data']

print("=========================================================================================")
print("PRECIPITATION TIME-SERIES DEEP DIVE: BALANGA (95pM7BAV)")
print("=========================================================================================")
print(f"{'Timestamp (UTC)':<24} | {'Timestamp (PHT UTC+8)':<22} | {'Sensor Val (mm)':<15}")
print("-" * 70)

today_pht_sum = 0
today_pht_max = 0
today_count = 0

for p in data:
    utc = p['recordedAt']
    val = p.get('value', 0) or 0
    dt_utc = datetime.datetime.fromisoformat(utc.replace('Z', '+00:00'))
    dt_pht = dt_utc + datetime.timedelta(hours=8)
    
    # Check if recorded in today's PHT (2026-08-29)
    if dt_pht.date() == datetime.date(2026, 8, 29):
        today_pht_sum += val
        today_count += 1
        if val > today_pht_max:
            today_pht_max = val

    if val > 0:
        print(f"{utc:<24} | {dt_pht.strftime('%Y-%m-%d %H:%M:%S'):<22} | {val:<15.4f}")

print("-" * 70)
print(f"Total points today (PHT Aug 29): {today_count}")
print(f"Raw Direct Sum: {today_pht_sum:.2f} mm")
print(f"Max 15-min reading: {today_pht_max:.2f} mm")
print("=========================================================================================")
