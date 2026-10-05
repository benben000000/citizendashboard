import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";

const CAP = 2000000;
const COLS: [string, string][] = [
  ["recorded_at_utc", "recorded UTC"],
  ["station_id", "station"],
  ["hz", "hz"],
  ["temperature_c", "temp C"],
  ["heat_index_c", "heat idx C"],
  ["relative_humidity_pct", "humid %"],
  ["pressure_hpa", "press hPa"],
  ["pressure_tendency", "press tend"],
  ["wind_speed_kmh", "wind km/h"],
  ["wind_direction_deg", "wind dir"],
  ["chance_of_rain_pct", "rain %"],
  ["expected_precipitation_mm", "precip mm"],
  ["uv_index", "uv uncal"],
  ["light_intensity", "light uncal"],
  ["prod_temp", "temp prod"],
  ["prod_hum", "hum prod"],
  ["prod_press", "press prod"],
  ["prod_wind", "wind prod"],
  ["trusted", "trusted"],
  ["rain_src", "rain src"],
  ["valid_at", "valid at UTC"],
];

const n = (v: any) => (typeof v === "number" && isFinite(v) ? v : null);
const e = (v: any) => (v === null || v === undefined ? "" : String(v));

export async function GET() {
  try {
    const file = path.join(process.cwd(), "prediction-model", "data", "prediction_audit.jsonl");
    const size = fs.statSync(file).size;
    const start = Math.max(0, size - CAP);
    const buf = Buffer.alloc(size - start);
    const h = fs.openSync(file, "r");
    fs.readSync(h, buf, 0, buf.length, start);
    fs.closeSync(h);
    const parts = buf.toString("utf-8").split("\n");
    if (start > 0) parts.shift();
    const rows: any[] = [];
    for (const line of parts) {
      if (!line.trim()) continue;
      let r: any;
      try { r = JSON.parse(line); } catch { continue; }
      const fr = r.forecast_raw || {};
      const v = r.variables || {};
      const src = (k: string) => (v[k] && v[k].producer) || null;
      const row: any = {};
      row.recorded_at_utc = e(r.recorded_at_utc).replace("T", " ").slice(0, 19);
      row.station_id = e(r.station_id);
      row.hz = n(r.horizon_hours);
      row.temperature_c = n(fr.temperature_c);
      row.heat_index_c = n(fr.heat_index_c);
      row.relative_humidity_pct = n(fr.relative_humidity_pct);
      row.pressure_hpa = n(fr.pressure_hpa);
      row.pressure_tendency = e(fr.pressure_tendency);
      row.wind_speed_kmh = n(fr.wind_speed_kmh);
      row.wind_direction_deg = n(fr.wind_direction_deg);
      row.chance_of_rain_pct = n(fr.chance_of_rain_pct);
      row.expected_precipitation_mm = n(fr.expected_precipitation_mm);
      row.uv_index = n(fr.uv_index);
      row.light_intensity = n(fr.light_intensity);
      row.prod_temp = e(src("temperature"));
      row.prod_hum = e(src("humidity"));
      row.prod_press = e(src("pressure"));
      row.prod_wind = e(src("wind_speed"));
      row.trusted = e(fr.input_quality_gate && fr.input_quality_gate.learned_output_trusted);
      row.rain_src = e(fr.rain_probability_source);
      row.valid_at = e(fr.target_timestamp).replace("T", " ").slice(0, 19);
      rows.push(row);
    }
    rows.reverse();
    let s = "<!doctype html><html><head><meta charset='utf-8'><title>prediction model logs</title><style>"
      + "body{font-family:ui-monospace,Consolas,monospace;font-size:12px;margin:12px}"
      + "table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:2px 6px;white-space:nowrap;text-align:left}"
      + "th{position:sticky;top:0;background:#eee}.s{max-height:88vh;overflow:auto}</style></head><body>"
      + "<h1 style='font-size:16px'>prediction model logs - LIVE</h1><div>server "
      + new Date().toISOString().replace("T", " ").slice(0, 19)
      + " | rows " + rows.length + (start > 0 ? " | PARTIAL WINDOW" : "") + " | press F5 to refresh</div>"
      + "<div class='s'><table><thead><tr>";
    for (const c of COLS) s += "<th>" + c[1] + "</th>";
    s += "</tr></thead><tbody>";
    for (const r of rows.slice(0, 800)) {
      s += "<tr>";
      for (const [k] of COLS) s += "<td>" + e(r[k]).replace(/&/g, "&amp;").replace(/</g, "&lt;") + "</td>";
      s += "</tr>";
    }
    s += "</tbody></table></div></body></html>";
    return new NextResponse(s, { headers: { "content-type": "text/html; charset=utf-8", "cache-control": "no-store" } });
  } catch (err) {
    return NextResponse.json({ success: false, error: String(err) }, { status: 500 });
  }
}
