import urllib.request
import json

url = 'http://citizen.kloudtechsea.com/api/water-level/dashboard'
with urllib.request.urlopen(url) as res:
    data = json.loads(res.read().decode('utf-8'))['data']

print("=========================================================================================")
print("                   WATER LEVEL SENSOR (WLMS) HARDWARE AUDIT")
print("=========================================================================================")
for s in data:
    st = s['station']
    tel = s.get('telemetry') or {}
    lvl = tel.get('waterLevel')
    dist = tel.get('distance')
    rec = tel.get('recordedAt')
    print(f"{st['stationName']:<35} ({st['stationPublicId']}) | Level: {lvl} m | Sensor Dist: {dist} cm | Recorded: {rec}")
print("=========================================================================================")
