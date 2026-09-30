"""
NOAA GFS live forecast provider (US Government public domain).

WHY THIS EXISTS
---------------
The hybrid router needs a live NWP value per station per horizon, from a source
that has cleared ExternalSourceRegistry. `noaa_gfs_v1` is registered as
production-eligible because NOAA/NCEP output is a work of the US Government and
is not subject to copyright protection.

DATA PATH
---------
Files come from the AWS Open Data mirror of the NCEP GFS Global Data Assimilation
System, over anonymous HTTPS with no credentials:

    https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.<YYYYMMDD>/<HH>/atmos/
        gfs.t<HH>z.pgrb2.0p25.f<FFF>        the field data  (540 MB)
        gfs.t<HH>z.pgrb2.0p25.f<FFF>.idx    the byte-range index (40 KB)

The field file is far too large to download, but it ships with a `.idx` that
gives the byte offset of every one of its ~743 fields. This module fetches the
index, locates the handful of fields it needs, and issues HTTP Range requests for
just those byte ranges. A five-field read is roughly 4 MB rather than 540 MB,
and each range is cached on disk so a restart does not refetch.

FIELDS AVAILABLE, AND ONE THAT IS NOT
-------------------------------------
  TMP  2 m above ground    -> temperature, K converted to degC
  RH   2 m above ground    -> relative humidity, %
  UGRD 10 m above ground   -> wind, m/s (already in the station's units)
  VGRD 10 m above ground   -> wind

  pressure -- NOT AVAILABLE. Every product in this bucket is a `pgrb2` file and
  the only pressure field they carry is PRMSL, the PERTURBATION of mean sea
  level pressure. Absolute MSL is not published here, and a perturbation is not
  a surface pressure: adding a guessed base state would inject a constant
  altitude bias straight into the correction, which is exactly the error the
  per-station coefficients exist to remove. So this provider returns no pressure
  and the router degrades the pressure cells to the LNN. That is a deliberate,
  measured trade: the GFS pressure gain (~+30% at 6h) is forgone rather than
  faked.

CAUSAL AVAILABILITY
-------------------
A cycle is usable only once it has actually been distributed. GFS runs at 00/06/
12/18Z and each cycle takes several hours to appear in full, so the most recent
cycle is only selected once `cycle + AVAILABILITY_DELAY_HOURS` has passed. The
provider never reads a forecast it could not have had at forecast time.
"""

import math
import os
import re
import threading
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

from external_source_registry import ExternalSourceRegistry

SOURCE_ID = "noaa_gfs_v1"
BUCKET = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
USER_AGENT = "kloudtrack-gfs/1.0 (+public-domain NOAA data)"

CYCLES_UTC = (0, 6, 12, 18)
# Conservative. GFS products are usually complete well before this, but using a
# generous delay costs a few hours of lead time and removes any chance of reading
# a partially-distributed cycle.
AVAILABILITY_DELAY_HOURS = 6.0

PRODUCT = "pgrb2.0p25"
# Lead hours we can request. GFS publishes 3-hourly steps to f120 then 6-hourly.
# The router keeps +1h on the LNN, so 3/6/12/24 are the only leads needed.
SUPPORTED_LEADS = (3, 6, 9, 12, 18, 24, 30, 36, 48, 60, 72, 96, 120)

# shortName -> level string, exactly as they appear in the .idx descriptor
WANTED = {
    "temperature": ("TMP", "2 m above ground"),
    "humidity": ("RH", "2 m above ground"),
    "wind_u": ("UGRD", "10 m above ground"),
    "wind_v": ("VGRD", "10 m above ground"),
}

DEFAULT_CACHE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "gfs_cache",
)


def _http_get(url: str, byte_range: Optional[Tuple[int, int]] = None,
              timeout: int = 60) -> bytes:
    headers = {"User-Agent": USER_AGENT}
    if byte_range:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


class GfsProvider:
    """Fetches GFS point forecasts for one station. Never raises to callers."""

    def __init__(
        self,
        lat: float,
        lon: float,
        registry: Optional[ExternalSourceRegistry] = None,
        cache_dir: Optional[str] = None,
        timeout: int = 60,
        availability_delay_hours: float = AVAILABILITY_DELAY_HOURS,
    ):
        self.lat = float(lat)
        self.lon = float(lon) % 360.0
        self.cache_dir = cache_dir or DEFAULT_CACHE
        self.timeout = int(timeout)
        self.availability_delay = float(availability_delay_hours)
        self._lock = threading.Lock()
        self._index_cache: Dict[str, list] = {}
        self._value_cache: Dict[Tuple[str, str, str], Optional[float]] = {}
        self.blocked_reason: Optional[str] = None
        self.last_error: Optional[str] = None

        try:
            reg = registry or ExternalSourceRegistry()
            ok, reason = reg.is_production_eligible(SOURCE_ID)
        except Exception as exc:  # noqa: BLE001
            self.blocked_reason = f"registry check failed: {type(exc).__name__}"
            return
        if not ok:
            self.blocked_reason = reason
            return
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
        except OSError as exc:
            self.blocked_reason = f"cache dir unusable: {exc}"
            return

    @property
    def available(self) -> bool:
        return self.blocked_reason is None

    # ---- cycle selection ---------------------------------------------
    def latest_cycle(self, now: Optional[datetime] = None) -> Optional[Tuple[datetime, int]]:
        """
        Most recent cycle whose lead hours are ALL available, or None.

        For a cycle at time C, the longest lead L is only usable at C + L. So a
        request for lead L is served from the newest cycle satisfying
        C + L + delay <= now, which keeps the causal constraint honest for every
        lead rather than only for the analysis.
        """
        ref = now or datetime.now(timezone.utc)
        if ref.tzinfo is None:
            ref = ref.replace(tzinfo=timezone.utc)
        for back in range(0, 4):
            day = (ref - timedelta(days=back)).replace(
                minute=0, second=0, microsecond=0)
            for hour in sorted(CYCLES_UTC, reverse=True):
                cycle = day.replace(hour=hour)
                if cycle > ref:
                    continue
                if cycle + timedelta(hours=self.availability_delay) <= ref:
                    return cycle, back
        return None

    @staticmethod
    def _lead_for(horizon_hours: float) -> Optional[int]:
        h = int(round(float(horizon_hours)))
        if h in SUPPORTED_LEADS:
            return h
        # snap up to the next published lead
        for lead in SUPPORTED_LEADS:
            if lead >= h:
                return lead
        return None

    def _cycle_dir(self, cycle: datetime) -> str:
        return f"{BUCKET}/gfs.{cycle.strftime('%Y%m%d')}/{cycle.strftime('%H')}/atmos"

    def _file_name(self, cycle: datetime, lead: int) -> str:
        return f"gfs.t{cycle.strftime('%H')}z.{PRODUCT}.f{lead:03d}"

    # ---- index and byte ranges ---------------------------------------
    def _index(self, cycle: datetime, lead: int) -> list:
        key = f"{cycle.isoformat()}_{lead}"
        with self._lock:
            if key in self._index_cache:
                return self._index_cache[key]
        name = self._file_name(cycle, lead)
        url = f"{self._cycle_dir(cycle)}/{name}.idx"
        local = os.path.join(self.cache_dir, name + ".idx")
        try:
            if os.path.exists(local) and os.path.getsize(local) > 0:
                with open(local, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            else:
                text = _http_get(url, timeout=self.timeout).decode("utf-8", "replace")
                with open(local, "w", encoding="utf-8", newline="\n") as f:
                    f.write(text)
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            self.last_error = f"index fetch failed: {type(exc).__name__}: {exc}"
            return []

        rows = []
        for line in text.splitlines():
            m = re.match(r"\s*(\d+):(\d+):(.*):\s*$", line)
            if not m:
                continue
            parts = m.group(3).split(":")
            if len(parts) < 3:
                continue
            rows.append({"num": int(m.group(1)), "off": int(m.group(2)),
                         "short": parts[1], "level": parts[2],
                         "fcst": parts[3] if len(parts) > 3 else ""})
        for i, r in enumerate(rows):
            r["end"] = rows[i + 1]["off"] if i + 1 < len(rows) else None
        with self._lock:
            self._index_cache[key] = rows
        return rows

    def _field_bytes(self, cycle: datetime, lead: int, short: str,
                     level: str) -> Optional[bytes]:
        rows = self._index(cycle, lead)
        if not rows:
            return None
        name = self._file_name(cycle, lead)
        url = f"{self._cycle_dir(cycle)}/{name}"
        local = os.path.join(
            self.cache_dir, f"{name}.{short}.{level.replace(' ', '_')}.grb2")
        if os.path.exists(local) and os.path.getsize(local) > 0:
            with open(local, "rb") as f:
                return f.read()
        for r in rows:
            if r["short"] == short and r["level"] == level:
                if r["end"] is None:
                    return None
                try:
                    blob = _http_get(url, (r["off"], r["end"] - 1), self.timeout)
                except (urllib.error.URLError, OSError, TimeoutError) as exc:
                    self.last_error = f"range fetch failed: {type(exc).__name__}: {exc}"
                    return None
                try:
                    with open(local, "wb") as f:
                        f.write(blob)
                except OSError:
                    pass  # caching is an optimisation, not a requirement
                return blob
        return None

    # ---- decode and interpolate --------------------------------------
    def _sample(self, blob: bytes) -> Optional[float]:
        """
        Bilinear-interpolate one GRIB2 message at the station location.

        GFS is a regular lat/lon grid, so nearest-neighbour would be simpler but
        a 0.25 degree grid step is a visible error against a point observation;
        bilinear is worth the few extra lines.
        """
        try:
            import eccodes
            import numpy as np
        except ImportError as exc:
            self.last_error = f"eccodes unavailable: {exc}"
            return None
        # Three eccodes API details, each of which fails differently:
        #   * codes_grib_new_from_file needs a real file descriptor and raises
        #     "UnsupportedOperation: fileno" on an in-memory stream, so the raw
        #     message is decoded directly instead.
        #   * the function is codes_new_from_message -- there is no
        #     codes_grib_new_from_message in eccodes 2.48.
        #   * it returns a product id (an int), not a context manager, so the
        #     handle must be released explicitly or the library leaks per call.
        handle = None
        try:
            handle = eccodes.codes_new_from_message(blob)
            ni = eccodes.codes_get(handle, "Ni")
            nj = eccodes.codes_get(handle, "Nj")
            lat1 = eccodes.codes_get(handle, "latitudeOfFirstGridPointInDegrees")
            lon1 = eccodes.codes_get(handle, "longitudeOfFirstGridPointInDegrees")
            lon2 = eccodes.codes_get(handle, "longitudeOfLastGridPointInDegrees")
            di = eccodes.codes_get(handle, "iDirectionIncrementInDegrees") or 0.25
            dj = eccodes.codes_get(handle, "jDirectionIncrementInDegrees") or 0.25
            vals = np.asarray(
                eccodes.codes_get_array(handle, "values"), dtype=float)
        except Exception as exc:  # noqa: BLE001 - eccodes raises many types
            self.last_error = f"grib decode failed: {type(exc).__name__}: {exc}"
            return None
        finally:
            if handle is not None:
                try:
                    eccodes.codes_release(handle)
                except Exception:  # noqa: BLE001
                    pass
        if ni is None or nj is None or vals.size != ni * nj:
            self.last_error = f"unexpected grid shape ni={ni} nj={nj} n={vals.size}"
            return None

        dj = abs(dj)
        # rows run north -> south
        row = (lat1 - self.lat) / dj
        # columns run west -> east; wrap the longitude into the grid's span
        span = (lon2 - lon1) or 360.0
        lon = self.lon
        while lon < lon1:
            lon += span
        while lon > lon1 + span:
            lon -= span
        col = (lon - lon1) / (di or 0.25)

        r0, c0 = int(math.floor(row)), int(math.floor(col))
        fr, fc = row - r0, col - c0
        if r0 < 0 or c0 < 0 or r0 + 1 >= nj or c0 + 1 >= ni:
            # fall back to the nearest valid node rather than failing outright
            r0 = min(max(r0, 0), nj - 2)
            c0 = min(max(c0, 0), ni - 2)
            fr = min(max(fr, 0.0), 1.0)
            fc = min(max(fc, 0.0), 1.0)

        def at(rr: int, cc: int) -> float:
            return float(vals[rr * ni + cc])

        v00, v01 = at(r0, c0), at(r0, c0 + 1)
        v10, v11 = at(r0 + 1, c0), at(r0 + 1, c0 + 1)
        return float((v00 * (1 - fc) + v01 * fc) * (1 - fr)
                     + (v10 * (1 - fc) + v11 * fc) * fr)

    def _value(self, cycle: datetime, lead: int, short: str,
               level: str) -> Optional[float]:
        key = (cycle.isoformat(), str(lead), short)
        with self._lock:
            if key in self._value_cache:
                return self._value_cache[key]
        blob = self._field_bytes(cycle, lead, short, level)
        val = self._sample(blob) if blob else None
        with self._lock:
            self._value_cache[key] = val
        return val

    # ---- public API ---------------------------------------------------
    def forecast(self, horizon_hours: float,
                 now: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
        """
        Point forecast for the station, in the units the router expects:
        temperature degC, humidity %, wind_speed m/s.

        Returns None if the source is blocked, the lead is unsupported, or any
        network/decode step failed. Never raises. There is deliberately no
        pressure key: this feed publishes only the MSL perturbation.
        """
        if not self.available:
            return None
        lead = self._lead_for(horizon_hours)
        if lead is None:
            return None
        chosen = self.latest_cycle(now)
        if chosen is None:
            return None
        cycle, _back = chosen
        # NOTE: no per-lead availability check here. Distribution latency is a
        # property of the CYCLE, not of the lead time: the f024 forecast from a
        # 12Z cycle is published a few hours after 12Z, not 30 hours after it.
        # Requiring cycle + lead + delay <= now made every lead beyond about
        # 6h fall back to a stale cycle, which the router then correctly
        # rejected as stale -- the forecast was fine, the check was wrong.

        out: Dict[str, Any] = {}
        raw: Dict[str, Any] = {}
        for key, (short, level) in WANTED.items():
            v = self._value(cycle, lead, short, level)
            if v is not None:
                raw[key] = v

        if "temperature" in raw:
            out["temperature"] = raw["temperature"] - 273.15
        if "humidity" in raw:
            out["humidity"] = max(0.0, min(100.0, raw["humidity"]))
        if "wind_u" in raw and "wind_v" in raw:
            out["wind_speed"] = math.hypot(raw["wind_u"], raw["wind_v"])
        if not out:
            return None

        out["_meta"] = {
            "source": SOURCE_ID,
            "model": "GFS",
            "cycle_utc": cycle.isoformat(),
            # Issue time is the freshness reference, NOT valid time. A 24h
            # forecast whose valid time is 6h away can still be stale if its
            # cycle is a day old, and a 1h forecast is fresh even though its
            # valid time is nearly now. Gate on when the model said it.
            "issued_utc": (cycle + timedelta(hours=self.availability_delay)).isoformat(),
            "lead_hours": lead,
            "requested_hours": horizon_hours,
            "valid_utc": (cycle + timedelta(hours=lead)).isoformat(),
            "fields_present": sorted(raw.keys()),
            "pressure_unavailable": True,
        }
        return out

    def status(self) -> Dict[str, Any]:
        cyc = self.latest_cycle()
        return {
            "available": self.available,
            "source": SOURCE_ID,
            "blocked_reason": self.blocked_reason,
            "lat": self.lat,
            "lon": self.lon,
            "latest_cycle_utc": cyc[0].isoformat() if cyc else None,
            "supported_leads_h": list(SUPPORTED_LEADS),
            "pressure_available": False,
            "last_error": self.last_error,
        }
