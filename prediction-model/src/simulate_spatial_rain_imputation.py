import math

# Station metadata with coordinates and today's rain observations
stations = {
    "95pM7BAV": {"name": "Doña Maria AWS - Balanga", "loc": [120.5369, 14.6764], "raw_rain": 78.9, "status": "ONLINE"},
    "2Dpo5DAK": {"name": "1Bataan Command Center", "loc": [120.5401, 14.6853], "raw_rain": 4.2, "status": "CLOGGED/SHIELDED"},
    "lMAZe9b3": {"name": "Abucay AWS - Bataan", "loc": [120.4959, 14.7191], "raw_rain": 0.0, "status": "OFFLINE"},
    "QgbGldAY": {"name": "Pag-asa Bagac AWS", "loc": [120.3951, 14.5982], "raw_rain": 0.0, "status": "ONLINE_SHADOW"},
    "nDbyYbR1": {"name": "Sabang Morong AWS", "loc": [120.2642, 14.6834], "raw_rain": 0.0, "status": "OFFLINE"},
    "rqAkmpKG": {"name": "Barretto AWS - Olongapo", "loc": [120.2635, 14.8519], "raw_rain": 1.2, "status": "ONLINE"},
    "3nzr48bG": {"name": "Calumpit AWS - Bulacan", "loc": [120.7631, 14.9152], "raw_rain": 18.4, "status": "ONLINE"},
    "wkAWLzlm": {"name": "Lazatin AWS - San Fernando", "loc": [120.6869, 15.0345], "raw_rain": 64.2, "status": "ONLINE"},
}

def haversine_km(loc1, loc2):
    lon1, lat1 = loc1
    lon2, lat2 = loc2
    return math.hypot((lon2 - lon1) * 111.0, (lat2 - lat1) * 111.0)

def is_trans_ridge(loc1, loc2):
    # Traverses Bataan mountain ridge between West (<120.42) and East (>120.48)
    lon1, lat1 = loc1
    lon2, lat2 = loc2
    return ((lon1 < 120.42 and lon2 > 120.48) or (lon2 < 120.42 and lon1 > 120.48))

print("==========================================================================================")
print("       SPATIAL PRECIPITATION RECONSTRUCTION FOR OFFLINE & CLOGGED GAUGES")
print("==========================================================================================")
print(f"{'Station Name':<28} | {'Raw Rain':<10} | {'Status':<16} | {'Spatially Reconstructed':<22} | {'Primary Source'}")
print("-" * 105)

online_healthy = {k: v for k, v in stations.items() if v["status"] in ["ONLINE", "ONLINE_SHADOW"]}

for sid, st in stations.items():
    if st["status"] == "ONLINE" or st["status"] == "ONLINE_SHADOW":
        print(f"{st['name']:<28} | {st['raw_rain']:<10.1f} | {st['status']:<16} | {st['raw_rain']:<22.1f} mm | Direct Hardware Sensor")
        continue

    # Calculate Topographic IDW from healthy online stations
    weights = []
    values = []
    sources = []
    
    for h_id, h_st in online_healthy.items():
        d = haversine_km(st["loc"], h_st["loc"])
        # Topographic barrier penalty (4x distance if across 1,388m mountain divide)
        if is_trans_ridge(st["loc"], h_st["loc"]):
            d_eff = d * 4.0
        else:
            d_eff = d
            
        w = 1.0 / math.pow(max(1.0, d_eff), 2)
        weights.append(w)
        values.append(h_st["raw_rain"])
        sources.append((h_st["name"], d, w))

    total_w = sum(weights)
    reconstructed_rain = sum(w * v for w, v in zip(weights, values)) / total_w
    top_source = sorted(sources, key=lambda x: x[2], reverse=True)[0]
    
    print(f"{st['name']:<28} | {st['raw_rain']:<10.1f} | {st['status']:<16} | {reconstructed_rain:<22.1f} mm | {top_source[0]} ({top_source[1]:.1f} km)")

print("==========================================================================================")
