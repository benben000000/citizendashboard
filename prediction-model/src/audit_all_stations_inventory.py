"""
Audit all Station IDs and Names across the entire project and API
"""

import os
import re
import json
import urllib.request

def fetch_json(url):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "StationAudit/1.0"})
        with urllib.request.urlopen(req, timeout=6) as res:
            return json.loads(res.read().decode("utf-8"))
    except Exception as e:
        return {"error": str(e)}

def main():
    print("=" * 80)
    print("      COMPREHENSIVE STATION INVENTORY & REGISTRY AUDIT")
    print("=" * 80)

    # 1. API Telemetry Dashboard
    dash_data = fetch_json("http://citizen.kloudtechsea.com/api/telemetry/dashboard")
    api_stations = {}
    if isinstance(dash_data, dict) and "data" in dash_data:
        for item in dash_data["data"]:
            st = item.get("station", {})
            sid = st.get("stationPublicId")
            if sid:
                api_stations[sid] = {
                    "name": st.get("stationName"),
                    "lat": st.get("latitude"),
                    "lon": st.get("longitude"),
                    "location": st.get("location"),
                    "category": "AWS" if "AWS" in st.get("stationName", "") else "WLMS"
                }

    print(f"\n1. Remote Live API (http://citizen.kloudtechsea.com/api/telemetry/dashboard):")
    print(f"   Total Active Stations in Dashboard: {len(api_stations)}")
    for sid, info in sorted(api_stations.items(), key=lambda x: x[1]['name']):
        print(f"   • [{sid}] {info['name']} ({info['category']}) @ ({info['lat']}, {info['lon']})")

    # 2. Search codebase for any hardcoded or additional station IDs/names
    code_stations = set()
    for root, dirs, files in os.walk("."):
        if any(d in root for d in [".git", "node_modules", ".next", ".venv", "artifacts"]):
            continue
        for f in files:
            if f.endswith((".ts", ".tsx", ".json", ".js", ".py")):
                path = os.path.join(root, f)
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as fp:
                        content = fp.read()
                        matches = re.findall(r'([0-9A-Za-z]{8})', content)
                        # Look for matches that look like station IDs
                        for m in matches:
                            if m in api_stations:
                                code_stations.add(m)
                except Exception:
                    pass

    # 3. Check WLMS and Water Level station lists in src/
    print("\n2. Checking Water Level & Weather Station Lists in Codebase:")
    water_level_service_path = os.path.join("src", "services", "water-level.service.ts")
    if os.path.exists(water_level_service_path):
        with open(water_level_service_path, "r", encoding="utf-8") as f:
            content = f.read()
            print(f"   - water-level.service.ts length: {len(content)} chars")

    telemetry_service_path = os.path.join("src", "services", "telemetry.service.ts")
    if os.path.exists(telemetry_service_path):
        with open(telemetry_service_path, "r", encoding="utf-8") as f:
            content = f.read()
            print(f"   - telemetry.service.ts length: {len(content)} chars")

    prediction_service_path = os.path.join("src", "services", "prediction.service.ts")
    if os.path.exists(prediction_service_path):
        with open(prediction_service_path, "r", encoding="utf-8") as f:
            content = f.read()
            # check profiles
            profiles = re.findall(r'\"([0-9A-Za-z]+)\":\s*\{', content)
            print(f"   - PINN Profiles defined in prediction.service.ts: {len(profiles)} profiles ({', '.join(profiles[:6])}...)")

    # 4. Check for any other stations mentioned in documentation or project files
    doc_stations = set()
    for root, dirs, files in os.walk("prediction-model"):
        for f in files:
            if f.endswith((".md", ".txt", ".json", ".py")):
                path = os.path.join(root, f)
                with open(path, "r", encoding="utf-8", errors="ignore") as fp:
                    for line in fp:
                        for sid in api_stations:
                            if sid in line:
                                doc_stations.add(sid)

    print(f"\n3. Cross-Reference Summary:")
    print(f"   - Stations in API Telemetry Stream: {len(api_stations)}")
    print(f"   - Stations configured in PINN Prediction Service: {len(profiles)}")
    print(f"   - Stations documented in Prediction Model: {len(doc_stations)}")

if __name__ == "__main__":
    main()
