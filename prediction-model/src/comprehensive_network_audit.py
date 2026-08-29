import urllib.request
import json
import datetime
import math

print("=============================================================================================================")
print("                   COMPREHENSIVE METEOROLOGICAL NETWORK AUDIT & SENSOR VALIDATION")
print("                     WMO-No. 8 & PAGASA Central Luzon Meteorological Physical Benchmarks")
print("=============================================================================================================")

# 1. Fetch Upstream Raw Dashboard Telemetry
upstream_url = "http://citizen.kloudtechsea.com/api/telemetry/dashboard"
try:
    with urllib.request.urlopen(upstream_url, timeout=10) as res:
        upstream_data = json.loads(res.read().decode("utf-8"))["data"]
except Exception as e:
    print(f"Error fetching upstream data: {e}")
    upstream_data = []

# 2. Fetch Processed Dashboard Telemetry
processed_url = "http://localhost/api/telemetry/processed"
try:
    with urllib.request.urlopen(processed_url, timeout=10) as res:
        processed_data = json.loads(res.read().decode("utf-8"))["data"]
except Exception as e:
    print(f"Error fetching processed data: {e}")
    processed_data = []

processed_map = {s["stationId"]: s for s in processed_data}

now_utc = datetime.datetime.now(datetime.timezone.utc)
now_pht = now_utc + datetime.timedelta(hours=8)
current_hour_pht = now_pht.hour + now_pht.minute / 60.0
is_night = current_hour_pht < 6.0 or current_hour_pht >= 18.0

print(f"Audit Timestamp: {now_pht.strftime('%Y-%m-%d %H:%M:%S')} PHT (UTC+8) | Solar State: {'Night (Solar = 0)' if is_night else 'Day'}")
print(f"Total Stations Audited: {len(upstream_data)}")
print("=" * 109)

audit_results = []

for item in upstream_data:
    st = item["station"]
    raw_t = item.get("telemetry") or {}
    sid = st["stationPublicId"]
    name = st["stationName"]
    proc = processed_map.get(sid, {}).get("liveMetrics", {})
    qc_flags = processed_map.get(sid, {}).get("qualityControl", {})
    
    # 1. Timestamps & Liveness
    raw_rec = raw_t.get("recordedAt")
    stale_hours = 0
    liveness_status = "ONLINE"
    if raw_rec:
        try:
            rec_dt = datetime.datetime.fromisoformat(raw_rec.replace("Z", "+00:00"))
            stale_hours = (now_utc - rec_dt).total_seconds() / 3600.0
            if stale_hours > 24:
                liveness_status = "OFFLINE (>24h)"
            elif stale_hours > 2:
                liveness_status = "STALE (>2h)"
        except Exception:
            liveness_status = "INVALID_TS"
    else:
        liveness_status = "NO_DATA"

    # 2. Temperature Validation
    raw_temp = raw_t.get("temperature")
    proc_temp = proc.get("temperature")
    temp_flag = "VALID"
    temp_note = ""
    if raw_temp is None:
        temp_flag = "ABSURD (Missing)"
        temp_note = "Sensor offline, reconstructed"
    elif raw_temp > 45.0 or raw_temp < 15.0:
        temp_flag = "ABSURD (Spike/Dropout)"
        temp_note = f"Raw={raw_temp}°C exceeds physical tropical limit (15-43°C)"
    elif raw_temp > 35.0:
        temp_flag = "SLIGHTLY ABSURD"
        temp_note = f"High for overcast Habagat monsoon ({raw_temp}°C)"
    
    # 3. Humidity Validation
    raw_hum = raw_t.get("humidity")
    proc_hum = proc.get("humidity")
    hum_flag = "VALID"
    hum_note = ""
    if raw_hum is None:
        hum_flag = "ABSURD (Missing)"
    elif raw_hum < 20.0 or raw_hum > 100.0:
        hum_flag = "ABSURD"
        hum_note = f"Raw={raw_hum}% outside physical relative humidity bounds"
    elif raw_hum < 50.0 and is_night:
        hum_flag = "SLIGHTLY ABSURD"
        hum_note = f"Unusually dry for monsoon night ({raw_hum}%)"

    # 4. Barometric Pressure Validation
    raw_pres = raw_t.get("pressure")
    proc_pres = proc.get("pressure")
    pres_flag = "VALID"
    pres_note = ""
    if raw_pres is None:
        pres_flag = "ABSURD (Missing)"
    elif raw_pres < 950.0 or raw_pres > 1040.0:
        pres_flag = "ABSURD"
        pres_note = f"Raw={raw_pres} hPa outside sea-level atmospheric envelope (970-1030 hPa)"
    elif raw_pres < 1000.0 or raw_pres > 1018.0:
        pres_flag = "SLIGHTLY ABSURD"
        pres_note = f"Moderate pressure departure ({raw_pres} hPa)"

    # 5. Precipitation Validation (Hourly & Today's)
    raw_precip = raw_t.get("precipitation")
    raw_hourly = raw_t.get("hourlyPrecip")
    precip_flag = "VALID"
    precip_note = ""
    if raw_hourly is not None and raw_hourly > 150.0:
        precip_flag = "ABSURD"
        precip_note = f"Hourly rate {raw_hourly} mm exceeds cloudburst ceiling"
    elif raw_hourly is not None and raw_hourly > 60.0:
        precip_flag = "SLIGHTLY ABSURD"
        precip_note = f"Torrential rain band detected ({raw_hourly} mm/h)"

    # 6. Solar / Light / UV Validation
    raw_uv = raw_t.get("uvIndex")
    raw_light = raw_t.get("lightIntensity")
    solar_flag = "VALID"
    solar_note = ""
    if is_night:
        if raw_uv and raw_uv > 0:
            solar_flag = "ABSURD"
            solar_note = f"UV index {raw_uv} detected at night (Sun is below horizon)"
        elif raw_light and raw_light > 500:
            solar_flag = "SLIGHTLY ABSURD"
            solar_note = f"Artificial street lighting detected ({raw_light} lux)"

    # 7. Wind Validation
    raw_wind = raw_t.get("windSpeed")
    wind_flag = "VALID"
    wind_note = ""
    if raw_wind is not None and (raw_wind < 0 or raw_wind > 200.0):
        wind_flag = "ABSURD"
        wind_note = f"Wind speed {raw_wind} km/h invalid"

    # 8. Heat Index Validation
    raw_hi = raw_t.get("heatIndex")
    proc_hi = proc.get("heatIndex")

    audit_results.append({
        "stationId": sid,
        "stationName": name,
        "liveness": liveness_status,
        "staleHours": stale_hours,
        "rawTemp": raw_temp,
        "procTemp": proc_temp,
        "tempFlag": temp_flag,
        "tempNote": temp_note,
        "rawHum": raw_hum,
        "procHum": proc_hum,
        "humFlag": hum_flag,
        "humNote": hum_note,
        "rawPres": raw_pres,
        "procPres": proc_pres,
        "presFlag": pres_flag,
        "presNote": pres_note,
        "rawHourly": raw_hourly,
        "precipFlag": precip_flag,
        "precipNote": precip_note,
        "rawWind": raw_wind,
        "windFlag": wind_flag,
        "rawUv": raw_uv,
        "rawLight": raw_light,
        "solarFlag": solar_flag,
        "solarNote": solar_note,
        "isSpatialEstimate": qc_flags.get("isSpatialEstimate", False),
    })

# Print Full Detailed Matrix
print(f"{'Station Name':<30} | {'Status':<14} | {'Raw Temp':<9} | {'Proc Temp':<9} | {'Raw Hum':<8} | {'Raw Pres':<10} | {'Rain Rate':<10} | {'QC Audit Summary'}")
print("-" * 120)

for r in audit_results:
    qc_issues = []
    if "ABSURD" in r["tempFlag"]: qc_issues.append("Temp " + r["tempFlag"])
    if "ABSURD" in r["humFlag"]: qc_issues.append("Hum " + r["humFlag"])
    if "ABSURD" in r["presFlag"]: qc_issues.append("Pres " + r["presFlag"])
    if "ABSURD" in r["precipFlag"]: qc_issues.append("Precip " + r["precipFlag"])
    if "ABSURD" in r["solarFlag"]: qc_issues.append("Solar " + r["solarFlag"])
    if r["liveness"] != "ONLINE": qc_issues.append(r["liveness"])
    
    summary = "[PASS] Physical WMO" if len(qc_issues) == 0 else "[FLAG] " + ", ".join(qc_issues)
    raw_t_str = f"{r['rawTemp']:.1f}C" if r['rawTemp'] is not None else "None"
    proc_t_str = f"{r['procTemp']:.1f}C" if r['procTemp'] is not None else "None"
    raw_h_str = f"{r['rawHum']:.0f}%" if r['rawHum'] is not None else "None"
    raw_p_str = f"{r['rawPres']:.1f}" if r['rawPres'] is not None else "None"
    rain_str = f"{r['rawHourly']:.1f} mm/h" if r['rawHourly'] is not None else "0.0"

    print(f"{r['stationName'][:30]:<30} | {r['liveness']:<14} | {raw_t_str:<9} | {proc_t_str:<9} | {raw_h_str:<8} | {raw_p_str:<10} | {rain_str:<10} | {summary}")

print("=============================================================================================================")
