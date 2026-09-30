"""
Fetch station telemetry history from the KloudTrack history API.

WHY THIS REPLACES THE PER-PARAMETER FETCH
-----------------------------------------
An earlier version walked `/api/telemetry/station/{id}/parameter/{param}` once
per field, which is 7 requests per station per window. That is both slow and
abusive: a burst of them took the endpoint down for the whole account and it took
a while to recover.

The documented history endpoint returns the FULL weather reading in one request:

    GET {base}/telemetry/station/{stationId}/history
        ?interval=<minutes> & take=<n> & startDate= & endDate= &
        filterOutliers=<true|false>

so the whole fleet is 16 requests instead of 448. `interval=60` gives hourly
resolution, which is what the model trains on, and the response already carries
wind direction and heat index -- two fields the per-parameter path served only
unreliably.

RATE LIMIT
----------
20 requests per minute per account. The default pacing is 3.5s, which stays
under that even for a full fleet fetch. Every response is cached to disk, so a
re-run costs no requests at all.

OUTLIER FILTERING
-----------------
The API defaults `filterOutliers` to true, which removes readings that deviate
from expected sensor ranges. This fetcher asks for `false` and lets the pipeline
apply its own documented physical-bounds quarantine instead. Two reasons: the
rejection then has a recorded reason instead of vanishing upstream, and
diagnostics need the unfiltered values -- a pressure of 846 hPa was being
dropped without explanation, and that is exactly the kind of silent loss this
project has been removing.

SECRETS
-------
The API key is read from KLOUDTRACK_API_KEY in the environment (or .env.local).
It is never written to a source file, a log, or a committed data artifact.
"""

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(HERE, "..", "data"))

DEFAULT_BASE = "https://api.kloudtechsea.com/api/v1"
ENDPOINT = "/telemetry/station/{station}/history"
USER_AGENT = "Kloudtrack-Audit/4.0"

MODEL_STATIONS = [
    "lMAZe9b3", "QgbGldAY", "Rjz2dbXW", "4VAl2p9k", "nDby4YpR", "03pqkGAj",
    "3nzr8bGo", "nDbyYbR1", "rqAkmpKG", "Bkpj1zRO", "wkAWLzlm", "1Zb102pg",
    "3nzr48bG", "VEpdDpBK", "2Dpo5DAK", "95pM7BAV",
]

# Station ids are case-SENSITIVE and this list once carried "wkAWlzlm" (lowercase
# l) instead of "wkAWLzlm". The API returned 404, the client treated 4xx as
# "genuinely absent" and returned an empty list WITHOUT writing a cache file, and
# the station silently vanished from the refetch -- 15 of 16 stations, reported as
# an apparent hardware outage rather than a typo in our own roster. The station
# was healthy: 50,000 rows, 835 hourly bins, and the highest wind calibration
# factor in the fleet.
#
# validate_roster() now fails loudly on any id that does not match the canonical
# index, so this class of bug cannot present itself as a dead sensor again.

CSV_FIELDS = [
    "station_id", "station_name", "location", "recorded_at",
    "temperature", "heat_index", "humidity", "pressure",
    "wind_speed", "wind_direction", "precipitation",
    "uv_index", "light_intensity",
]

CACHE_DIR = os.path.join(DATA_DIR, "kloudtrack_history_cache")


def load_key():
    """API key from the environment, falling back to .env.local."""
    key = os.environ.get("KLOUDTRACK_API_KEY")
    if key:
        return key.strip()
    env = os.path.join(DATA_DIR, "..", "..", ".env.local")
    try:
        with open(os.path.abspath(env), encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("KLOUDTRACK_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        pass
    return None


def validate_roster(stations, strict=True):
    """
    Fail loudly when MODEL_STATIONS disagrees with the canonical station index.

    A mistyped id 404s, and a 404 is indistinguishable from a dead station unless
    something checks. That is exactly how a lowercase "l" for an uppercase "L"
    turned one healthy station into an apparent hardware outage.
    """
    index_path = os.path.join(DATA_DIR, "station_index.json")
    try:
        with open(index_path, encoding="utf-8") as f:
            canonical = {s["station_id"] for s in json.load(f)}
    except (OSError, ValueError, KeyError, TypeError):
        print(f"  [warn] could not read {index_path}; roster not validated")
        return []

    known = {s for s in stations if s in canonical}
    unknown = sorted(set(stations) - canonical)
    # Case-insensitive match catches the exact failure mode without needing a
    # second source of truth.
    by_fold = {s.lower(): s for s in canonical}
    for bad in list(unknown):
        guess = by_fold.get(bad.lower())
        if guess:
            print(f"  [ERROR] roster id {bad!r} differs from the canonical id "
                  f"{guess!r} by case only. Station ids are case-sensitive.")
        else:
            print(f"  [ERROR] roster id {bad!r} is not in station_index.json")

    missing = sorted(canonical - set(stations))
    if missing:
        print(f"  [warn] {len(missing)} station(s) in the index are absent from "
              f"MODEL_STATIONS: {', '.join(missing)}")

    if unknown and strict:
        # The offending ids belong in the exception message, not only on stdout.
        # A caller that captures the exception without the console output (cron,
        # CI, a wrapper) must still learn which id is wrong and what it should be.
        detail = "; ".join(
            f"{bad!r} -> {by_fold.get(bad.lower())!r} (case only)"
            if bad.lower() in by_fold else f"{bad!r} (not in station_index.json)"
            for bad in unknown)
        raise SystemExit(
            f"Roster validation failed for {len(unknown)} station id(s): {detail}. "
            f"Fix MODEL_STATIONS before fetching; a wrong id looks exactly like "
            f"a dead station and the cache will not record the difference.")
    return sorted(canonical - set(stations))


def station_meta():
    meta = {}
    src = os.path.join(DATA_DIR, "weather_telemetry.csv")
    try:
        with open(src, newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                sid = (row.get("station_id") or "").strip()
                if sid and sid not in meta:
                    meta[sid] = (row.get("station_name") or "",
                                row.get("location") or "")
    except OSError:
        pass
    return meta


class HistoryClient:
    def __init__(self, key, base=DEFAULT_BASE, delay=3.5, timeout=90, retries=4):
        if not key:
            raise SystemExit(
                "KLOUDTRACK_API_KEY is not set. Export it, or add it to .env.local. "
                "It is never read from source.")
        self.key = key
        self.base = base.rstrip("/")
        self.delay = delay
        self.timeout = timeout
        self.retries = retries
        self.calls = 0
        self.cache_hits = 0
        self._last = 0.0
        os.makedirs(CACHE_DIR, exist_ok=True)

    def _pace(self):
        gap = time.time() - self._last
        if gap < self.delay:
            time.sleep(self.delay - gap)
        self._last = time.time()

    def history(self, station, start, end, interval=60, take=5000,
                filter_outliers=False):
        """Full weather history for one station, as a list of reading dicts."""
        tag = f"{station}__{start}__{end}__i{interval}__f{int(bool(filter_outliers))}"
        path = os.path.join(CACHE_DIR, tag + ".json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self.cache_hits += 1
                    return json.load(f)
            except (OSError, ValueError):
                pass

        url = (self.base + ENDPOINT.format(station=station)
               + f"?interval={interval}&take={take}"
               + f"&startDate={start}&endDate={end}"
               + f"&filterOutliers={'true' if filter_outliers else 'false'}")
        last = None
        for attempt in range(self.retries):
            self._pace()
            self.calls += 1
            try:
                req = urllib.request.Request(
                    url, headers={"x-kloudtrack-key": self.key,
                                  "User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    payload = json.loads(r.read().decode("utf-8", "replace"))
                if not payload.get("success", True):
                    last = RuntimeError(payload.get("message", "success=false"))
                    self._pace()
                    time.sleep(min(30.0, (2 ** attempt) * 3.0))
                    continue
                readings = (payload.get("data") or {}).get("telemetry") or []
                try:
                    with open(path, "w", encoding="utf-8", newline="") as f:
                        json.dump(readings, f)
                except OSError:
                    pass
                return readings
            except urllib.error.HTTPError as e:
                last = e
                if e.code in (400, 401, 403, 404):
                    # Bad request, auth, or genuinely absent. Retrying is waste.
                    return []
                self._pace()
                time.sleep(min(30.0, (2 ** attempt) * 3.0))
            except (urllib.error.URLError, ValueError, OSError, TimeoutError) as e:
                last = e
                self._pace()
                time.sleep(min(30.0, (2 ** attempt) * 3.0))
        print(f"    ! {station}: giving up after {self.retries} attempts "
              f"({type(last).__name__}: {str(last)[:60]})")
        return None


def to_rows(station, readings, meta):
    """API reading shape -> the pipeline's CSV schema."""
    out = {}
    name, loc = meta.get(station, ("", ""))
    horizon = datetime.now(timezone.utc) + timedelta(days=1)
    for rec in readings or []:
        ts = rec.get("recordedAt")
        if not ts:
            continue
        try:
            t = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        except ValueError:
            continue
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        if t > horizon:
            continue  # device clock fault; the source CSV has a 2069 row
        wind = rec.get("wind") or {}
        key = t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        out[key] = {
            "station_id": station, "station_name": name, "location": loc,
            "recorded_at": key,
            "temperature": rec.get("temperature"),
            "heat_index": rec.get("heatIndex"),
            "humidity": rec.get("humidity"),
            "pressure": rec.get("pressure"),
            "wind_speed": wind.get("speed"),
            "wind_direction": wind.get("direction"),
            "precipitation": rec.get("precipitation"),
            "uv_index": rec.get("uvIndex"),
            "light_intensity": rec.get("lightIntensity"),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-06-20")
    ap.add_argument("--end", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    ap.add_argument("--out", default=os.path.join(DATA_DIR, "weather_telemetry_current.csv"))
    ap.add_argument("--base", default=os.environ.get("KLOUDTRACK_API_BASE_URL_HISTORY",
                                                      DEFAULT_BASE))
    ap.add_argument("--interval", type=int, default=60,
                    help="aggregation in minutes; 60 = hourly")
    ap.add_argument("--take", type=int, default=5000)
    ap.add_argument("--delay", type=float, default=3.5,
                    help="seconds between requests; the API allows 20/minute")
    ap.add_argument("--stations", default=None)
    ap.add_argument("--no-roster-check", action="store_true",
                    help="Skip validation of station ids against station_index.json. "
                         "Only for deliberately fetching an unknown station.")
    ap.add_argument("--filter-outliers", action="store_true",
                    help="let the API drop out-of-range readings (default: keep them "
                         "and let the pipeline quarantine with a recorded reason)")
    args = ap.parse_args()

    stations = ([s.strip() for s in args.stations.split(",") if s.strip()]
                if args.stations else MODEL_STATIONS)
    validate_roster(stations, strict=not args.no_roster_check)
    client = HistoryClient(load_key(), base=args.base, delay=args.delay)
    meta = station_meta()

    print(f"KloudTrack history fetch  {args.start} -> {args.end}")
    print(f"  base     : {args.base}")
    print(f"  stations : {len(stations)}   requests planned: {len(stations)}")
    print(f"  interval : {args.interval} min   take {args.take}   "
          f"pacing {args.delay}s (limit 20/min)")

    merged = {}
    per_station = {}
    for s in stations:
        readings = client.history(s, args.start, args.end, args.interval,
                                  args.take, args.filter_outliers)
        if readings is None:
            per_station[s] = 0
            continue
        rows = to_rows(s, readings, meta)
        # Key on (station, timestamp), not timestamp alone. Every station reports
        # on the same hourly grid, so merging on the bare timestamp silently
        # collapsed 16 stations into one -- 31,000 readings became 2,448 rows,
        # with the last station fetched overwriting every other.
        for k, v in rows.items():
            merged[(s, k)] = v
        per_station[s] = len(rows)
        span = ""
        if rows:
            ks = sorted(rows)
            span = f"  {ks[0][:10]} -> {ks[-1][:10]}"
        print(f"  {s:<10} {len(rows):>6} rows{span}")

    total = 0
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for key in sorted(merged):
            r = merged[key]
            w.writerow({c: ("" if r.get(c) is None else r.get(c))
                        for c in CSV_FIELDS})
            total += 1

    print(f"\n  API calls made       : {client.calls}")
    print(f"  cache hits           : {client.cache_hits}")
    print(f"  stations with data   : {sum(1 for v in per_station.values() if v)}/"
          f"{len(stations)}")
    print(f"  rows written         : {total:,}")
    print(f"  written              : {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
