import json, os, sys, time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(r"C:\\Ben File\\beta-citizen-prediction")
DATA = ROOT / "prediction-model" / "data"
AUDIT = DATA / "prediction_audit.jsonl"
LIVE = DATA / "mqtt_live_predictions.json"
EVENTS = DATA / "ingestor_events.jsonl"
MAX_RECORDS = 40000
CACHE_SECONDS = 3
_cache = {"at": 0.0, "payload": None}

def read_jsonl_tail(path, limit):
    rows = []
    try: size = path.stat().st_size
    except FileNotFoundError: return rows
    if size == 0: return rows
    want = min(size, limit * 1400)
    start = max(0, size - want)
    try:
        with path.open("rb") as h:
            h.seek(start)
            raw = h.read()
    except OSError: return rows
    lines = raw.decode("utf-8", errors="replace").split("\n")
    if start > 0 and lines: lines.pop(0)
    for line in lines:
        line = line.strip()
        if not line: continue
        try: rows.append(json.loads(line))
        except ValueError: continue
    return rows[-limit:]

def num(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None

def build_payload():
    now = time.time()
    if _cache["payload"] is not None and now - _cache["at"] < CACHE_SECONDS:
        return _cache["payload"]
    raw = read_jsonl_tail(Path("prediction-model/data/prediction_audit.jsonl"), MAX_RECORDS)
    rows = []
    stations = set()
    horizons = {}
    pipe = get_telemetry_pipeline()
    for rec in read_jsonl_tail(Path("prediction-model/data/prediction_audit.jsonl"), MAX_RECORDS):
        fc = rec.get("forecast_raw") or {}
        vars = rec.get("variables") or {}
        def prod(k): return (vars.get(k) or {}).get("producer")
        sid = rec.get("station_id"); hz = num(rec.get("horizon_hours"))
        if sid: stations.add(sid)
        if hz is not None: horizons[hz] = horizons.get(hz, 0) + 1
        rows.append({
            "ts": rec.get("recorded_at_utc"), "origin": fc.get("forecast_origin_timestamp") or rec.get("origin_timestamp_utc"),
            "valid": fc.get("target_timestamp"), "station": sid, "hz": horizon,
            "temperature_c": num(fc.get("temperature_c")), "heat_index_c": num(fc.get("heat_index_c")),
            "humidity_pct": num(fc.get("relative_humidity_pct")), "pressure_hpa": num(fc.get("pressure_hpa")),
            "pressure_tendency": fc.get("pressure_tendency"),
            "wind_speed_kmh": num(fc.get("wind_speed_kmh")), "wind_direction_deg": num(fc.get("wind_direction_deg")),
            "chance_of_rain_pct": num(fc.get("chance_of_rain_pct")),
            "expected_precipitation_mm": num(fc.get("expected_precipitation_mm")),
            "uv_index": num(fc.get("uv_index")), "light_intensity": num(fc.get("light_intensity")),
            "prod_temp": prod("temperature"), "prod_hum": prod("humidity"),
            "prod_press": prod("pressure"), "prod_wind": prod("wind_speed"),
            "trusted": (fc.get("input_quality_gate") or {}).get("learned_output_trusted"),
            "rain_src": fc.get("rain_probability_source"),
        })
    events = []
    for event in read_jsonl_tail(EVENTS, 80):
        det = event.get("reason") or event.get("error")
        if not det and event.get("missing_fields"): det = "missing: " + ", ".join(event["missing_fields"])
        events.append({"at": event.get("recorded_at_utc"), "kind": event.get("kind"), "station": event.get("station_id"), "detail": det})
    live = None
    try:
        live = json.loads(LIVE.read_text(encoding="utf-8"))
        live = {"last_updated": live.get("last_updated"), "active_stations": live.get("total_active_stations")}
    except: pass
    payload = {"generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": str(AUDIT), "row_count": len(rows), "station_count": len(stations),
        "stations": sorted(stations), "by_horizon": {str(k): v for k, v in sorted(horizons.items())},
        "live": live, "rows": rows, "events": list(reversed(events))}
    _cache["at"] = time.time(); _cache["payload"] = payload
    return payload
_cache = {"at": 0.0, "payload": None}

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def _send(self, code, body, ctype):
        b = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b))); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if urlparse(self.path).path == "/api/logs":
            try: self._send(200, json.dumps(build_payload()), "application/json")
            except Exception as e: self._send(500, json.dumps({"error": str(e)}), "application/json"); return
        if self.path in ("/", "/index.html"): self._send(200, PAGE, "text/html; charset=utf-8"); return
        self._send(404, "not found", "text/plain")

PAGE = """<!doctype html><html><head><meta charset='utf-8'><title>prediction model logs</title><style>
body{font-family:ui-monospace,Consolas,monospace;font-size:12px;margin:10px;background:#fff;color:#000}
h1{font-size:16px;margin:0 0 6px}
.bar{font-size:12px;margin-bottom:8px;padding:5px 7px;border:1px solid #999;background:#f4f4f4}
#live{color:#080;font-weight:bold}
canvas{border:1px solid #999;background:#fcfcfc;display:block;margin-bottom:8px}
select{font:inherit;padding:2px}
table{border-collapse:collapse;width:100%}
td,th{border:1px solid #bbb;padding:2px 5px;white-space:nowrap;text-align:left}
th{position:sticky;top:0;background:#e4e4e4;font-size:11px}
tr:nth-child(even){background:#f7f7f7}
.wrap{max-height:58vh;overflow:auto;border:1px solid #999}
.ev{max-height:13vh;overflow:auto;border:1px solid #999;margin-top:8px}
.ev table{font-size:11px}
th.f{position:sticky;left:0;background:#e4e4e4;z-index:2}
td.f{position:sticky;left:0;background:#f0f0f0;z-index:1}
tr:nth-child(even) td.f{background:#e9e9e9}
</style></head><body>
<h1>prediction model &mdash; logs, trends, and process events</h1>
<div class='bar'>server <b id='gen'>-</b> | rows <b id='rc'>-</b> | stations <b id='sc'>-</b> | horizons <b id='hz'>-</b> | live cache age <b id='age'>-</b> | new this poll <span id='live'>0</span> | auto-refresh <select id='iv' onchange='setPoll(this.value)'><option value='3000'>3s</option><option value='10000'>10s</option><option value='30000'>30s</option><option value='60000'>60s</option></select> | station <select id='sel' onchange='draw()'></select> <button onclick='load()'>refresh</button></div>
<canvas id='cv' width='1500' height='250'></canvas>
<div class='wrap'><table><thead><tr>
<th class='f'>recorded UTC</th><th>station</th><th>hz</th><th>temp C</th><th>heat idx C</th>
<th>humid %</th><th>press hPa</th><th>press tend</th><th>wind km/h</th><th>wind deg</th>
<th>rain %</th><th>precip mm</th><th>uv</th><th>light</th><th>temp prod</th>
<th>hum prod</th><th>press prod</th><th>wind prod</th><th>trusted</th><th>rain src</th><th>valid at UTC</th>
</tr></thead><tbody id='tb'></tbody></table></div>
<div class='ev'><table><thead><tr><th>at UTC</th><th>kind</th><th>station</th><th>detail</th></tr></thead>
<tbody id='eb'></tbody></table></div>
<script>
var DATA=null, seen={}, timer=null;
var COLS=['ts','station','hz','temperature_c','heat_index_c','humidity_pct','pressure_hpa',
'pressure_tendency','wind_speed_kmh','wind_direction_deg','chance_of_rain_pct',
'expected_precipitation_mm','uv_index','light_intensity','prod_temp','prod_hum','prod_press',
'prod_wind','trusted','rain_src','valid'];
function esc(v){if(v===null||v===undefined||v==='')return '';return String(v).replace(/&/g,'&').replace(/</g,'<');}
function fmt(v){if(v===null||v===undefined)return '';var s=String(v);return s.indexOf('T')>0?s.replace('T',' ').slice(0,19):s;}
function ts(v){if(!v)return 0;var t=Date.parse(v);return isNaN(t)?0:t;}
function setPoll(ms){if(timer)clearInterval(timer);timer=setInterval(load,parseInt(ms,10));}
function draw(){
  var cv=document.getElementById('cv'), g=cv.getContext('2d');
  g.clearRect(0,0,cv.width,cv.height);
  if(!DATA)return;
  var sel=document.getElementById('sel').value;
  var rows=DATA.rows.filter(function(r){return !sel||r.station===sel;});
  var series=[{k:'temperature_c',c:'#c00',l:'temp C'},{k:'wind_speed_kmh',c:'#06c',l:'wind km/h'},
    {k:'chance_of_rain_pct',c:'#0a0',l:'rain %'},{k:'pressure_hpa',c:'#888',l:'press hPa'}];
  var pts=rows.filter(function(r){return r.hz===1&&r[series[0].k]!==null;}).sort(function(a,b){return ts(a.ts)-ts(b.ts);});
  pts=pts.slice(-1200);
  if(pts.length<2){g.fillStyle='#666';g.font='13px monospace';g.fillText('collecting +1h points for the trend...',14,26);return;}
  var t0=ts(pts[0].ts),t1=ts(pts[pts.length-1].ts);if(t1<=t0)t1=t0+1;
  var all=[],i,s,vals,mn,mx;
  for(s=0;s<series.length;s++){
    vals=pts.map(function(r){return r[series[s].k];}).filter(function(v){return v!==null&&v!==undefined;});
    if(vals.length)all.push({mn:Math.min.apply(null,vals),mx:Math.max.apply(null,vals)});
  }
  if(!all.length)return;
  var gmn=Math.min.apply(null,all.map(function(a){return a.mn;}));
  var gmx=Math.max.apply(null,all.map(function(a){return a.mx;}));
  if(gmx===gmn)gmx=gmn+1;
  g.strokeStyle='#ccc';g.beginPath();g.moveTo(70,cv.height-24);g.lineTo(cv.width-14,cv.height-24);g.stroke();
  g.font='11px monospace';g.fillStyle='#333';
  g.fillText(gmx.toFixed(2),6,16);g.fillText(gmn.toFixed(2),6,cv.height-28);
  for(s=0;s<series.length;s++){
    g.strokeStyle=series[s].c;g.lineWidth=1.4;g.beginPath();
    var started=false;
    for(i=0;i<pts.length;i++){
      var v=pts[i][series[s].k];
      if(v===null||v===undefined)continue;
      var x=70+(ts(pts[i].ts)-t0)/(t1-t0)*(cv.width-90);
      var y=cv.height-24-(v-gmn)/(gmx-gmn)*(cv.height-46);
      if(!started){g.moveTo(x,y);started=true;}else g.lineTo(x,y);
    }
    g.stroke();g.fillStyle=series[s].c;g.fillText(series[s].l,80+s*130,16);
  }
  g.fillStyle='#333';g.fillText(fmt(pts[0].ts)+'  ->  '+fmt(pts[pts.length-1].ts)+'   ('+pts.length+' +1h points)',70,cv.height-8);
}
function load(){
  fetch('/api/logs',{cache:'no-store'}).then(function(r){return r.json();}).then(function(d){
    if(d.error){document.getElementById('gen').textContent='ERROR '+d.error;return;}
    DATA=d;var added=0,i;
    for(i=0;i<d.rows.length;i++){
      var o=d.rows[i],k=o.station+'|'+o.hz+'|'+o.ts;
      if(seen[k]===undefined){seen[k]=1;added++;}
    }
    document.getElementById('gen').textContent=d.generated_at_utc;
    document.getElementById('rc').textContent=d.row_count;
    document.getElementById('sc').textContent=d.station_count;
    document.getElementById('hz').textContent=JSON.stringify(d.by_horizon);
    var a='-',t;
    if(d.live&&d.live.last_updated){t=Date.parse(d.live.last_updated);if(!isNaN(t))a=((Date.now()-t)/1000).toFixed(0)+'s';}
    document.getElementById('age').textContent=a;
    document.getElementById('live').textContent=added;
    var sel=document.getElementById('sel'),cur=sel.value;
    if(sel.options.length!==d.stations.length){
      sel.innerHTML='<option value="">all stations</option>';
      for(i=0;i<d.stations.length;i++){var op=document.createElement('option');op.value=d.stations[i];op.text=d.stations[i];sel.appendChild(op);}
      sel.value=cur;
    }
    var h='',r2,j;
    for(i=d.rows.length-1;i>=0;i--){
      r2=d.rows[i];h+='<tr>';
      for(j=0;j<COLS.length;j++){
        h+='<td'+(j===0?' class="f"':'')+'>'+esc(fmt(r2[COLS[j]]))+'</td>';
      }
      h+='</tr>';
    }
    document.getElementById('tb').innerHTML=h;
    var e='',ev;
    for(i=0;i<(d.events||[]).length;i++){
      ev=d.events[i];
      e+='<tr><td>'+esc(fmt(ev.at))+'</td><td>'+esc(ev.kind)+'</td><td>'+esc(ev.station)+'</td><td>'+esc(ev.detail)+'</td></tr>';
    }
    document.getElementById('eb').innerHTML=e;
    draw();
  }).catch(function(e){document.getElementById('gen').textContent='fetch failed '+e;});
}
load();setPoll(3000);
</script></body></html>"""


if __name__ == "__main__":
    port = 8090
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print("prediction logs: http://127.0.0.1:%d" % port)
    print("audit trail    : %s" % AUDIT)
    print("Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("")
        server.server_close()
