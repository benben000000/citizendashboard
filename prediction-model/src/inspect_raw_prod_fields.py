"""
Raw inspection of upstream production telemetry.
"""

import sys
import json
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def inspect():
    req = urllib.request.urlopen('http://citizen.kloudtechsea.com/api/telemetry/dashboard', timeout=8)
    raw = json.loads(req.read().decode('utf-8'))
    stations = raw.get('data', [])

    print("=" * 120)
    print("EXACT RAW UPSTREAM PRODUCTION TELEMETRY FROM CITIZEN.KLOUDTECHSEA.COM")
    print("=" * 120)

    for s in stations:
        st = s.get('station', {})
        tel = s.get('telemetry', {})
        name = st.get('stationName', 'Unknown')
        sid = st.get('stationPublicId', st.get('id', 'unknown'))
        print(f"Station: {name} ({sid})")
        print(f"   Raw Telemetry: {json.dumps(tel, indent=2)}")
        print("-" * 120)

if __name__ == "__main__":
    inspect()
