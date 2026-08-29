import urllib.request
import json

url = 'http://citizen.kloudtechsea.com/api/telemetry/dashboard'
with urllib.request.urlopen(url) as res:
    raw_data = json.loads(res.read().decode('utf-8'))['data']

print("=========================================================================================")
print("UPSTREAM SENSOR HARDWARE AUDIT: RAINFALL ACCUMULATION & COUNTER STATUS")
print("=========================================================================================")
for s in raw_data:
    st = s['station']
    tel = s.get('telemetry') or {}
    p = tel.get('precipitation')
    hp = tel.get('hourlyPrecip')
    rec = tel.get('recordedAt')
    print(f"{st['stationName']:<35} ({st['stationPublicId']}) | Today Precip: {p} mm | Hourly: {hp} mm | Recorded: {rec}")

print("=========================================================================================")
