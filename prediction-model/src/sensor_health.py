"""
Sensor health gate.

WHY
---
Three of sixteen stations report a constant 0.0 m/s wind while their temperature
is perfectly normal (31-33 degC). Those anemometers are dead: the stations are
transmitting, only the wind channel is gone. Including them does real damage in
two directions:

  * training -- the network is shown a target that is identically zero for 19% of
    the fleet and learns "wind is zero" as a legitimate forecast;
  * evaluation -- every producer predicts ~0 against a truth of ~0, so the
    apparent error collapses and the model looks far better than it is. Excluding
    those three stations cut the LNN's wind margin over ECMWF by roughly 60%.

This module detects that condition from the data so the rest of the pipeline can
act on it. It does not repair hardware; that is a site visit.

VERDICTS
--------
  ok      the channel looks like weather
  noisy   high variance, or a large share of exact zeros with nonzero mean,
          consistent with a damped or intermittently disconnected sensor
  dead    effectively constant zero, or constant overall
  absent  the field is missing entirely

The thresholds are deliberately conservative. A station that is merely sheltered
reads low but still varies; calling that "dead" would discard usable data and
quietly shrink the fleet.
"""

import json
import math
import os
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence

# A wind sensor that is dead or barely alive. Thresholds are expressed relative to
# the fleet's typical reading so they do not encode an absolute wind speed that
# would be wrong in a different climate.
#
# Magnitude, not frequency, is the discriminator. One station reports zeros 94% of
# the time and is caught only by asking how large its non-zero values get: its
# 95th percentile is 0.11 m/s, so the handful of "readings" are sensor twitch, not
# weather. A genuinely intermittent station that does see gales has a 95th
# percentile in m/s and is kept.
DEAD_TAIL_FRACTION = 0.25          # p95 below this share of the fleet median = dead
DEAD_ZERO_FRACTION = 0.98
DEAD_VARIANCE_FRACTION = 1e-4      # var/mean^2 below this is a constant
NOISY_ZERO_FRACTION = 0.35
# Gustiness: real wind is intermittent, so variance relative to the mean has a
# floor. Below this the channel is smoother than meteorology allows.
MIN_GUSTINESS = 0.05
# Number of distinct nonzero values. A dead-but-jittering sensor can emit a
# handful; weather cannot.
MIN_DISTINCT_VALUES = 8
MIN_SAMPLES = 50


def _parse_ts(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _finite(values: Iterable[Any]) -> List[float]:
    out = []
    for v in values:
        if v is None or isinstance(v, bool):
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out.append(f)
    return out


def _percentile(values: List[float], pct: float) -> float:
    """Nearest-rank percentile; no interpolation, so a small sample is safe."""
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(pct / 100.0 * (len(s) - 1)))))
    return float(s[k])


def assess_wind(values: Sequence[Any], fleet_median: Optional[float] = None) -> Dict[str, Any]:
    """
    Judge one station's wind channel.

    Returns a verdict plus the statistics it was based on, so the call is
    auditable rather than a bare label.
    """
    v = _finite(values)
    n = len(v)
    report: Dict[str, Any] = {
        "n": n, "verdict": "absent", "mean": None, "std": None,
        "zero_fraction": None, "gustiness": None, "distinct_nonzero": None,
    }
    if n == 0:
        report["reason"] = "no finite wind samples"
        return report
    if n < MIN_SAMPLES:
        report["verdict"] = "absent"
        report["reason"] = f"only {n} samples, below the {MIN_SAMPLES} needed to judge"
        return report

    mean = sum(v) / n
    var = sum((x - mean) ** 2 for x in v) / n
    std = math.sqrt(var)
    zeros = sum(1 for x in v if x == 0.0)
    zf = zeros / n
    nonzero = [x for x in v if x != 0.0]
    distinct = len(set(round(x, 6) for x in nonzero))
    gust = (std / mean) if mean > 1e-9 else 0.0

    report.update({"mean": mean, "std": std, "zero_fraction": zf,
                   "gustiness": gust, "distinct_nonzero": distinct})

    # Constant channel of any value is dead.
    if distinct == 0 or var / (mean * mean) < DEAD_VARIANCE_FRACTION:
        report["verdict"] = "dead"
        report["reason"] = "channel is constant"
        return report
    # Effectively always zero, relative to what the fleet typically sees.
    ref = fleet_median if fleet_median else 0.0
    if zf >= DEAD_ZERO_FRACTION and (ref <= 0 or mean < 0.05 * ref):
        report["verdict"] = "dead"
        report["reason"] = f"{zf * 100:.1f}% exact zeros, mean {mean:.3f} m/s"
        return report
    # Sparse but trivially small: the channel is twitching, not measuring. Uses
    # the 95th percentile so a genuine tail of gales keeps the station.
    p95 = _percentile(v, 95)
    report["p95"] = p95
    if ref > 0 and zf >= 0.5 and p95 < DEAD_TAIL_FRACTION * ref:
        report["verdict"] = "dead"
        report["reason"] = (f"{zf * 100:.1f}% exact zeros and p95 {p95:.3f} m/s, "
                            f"below {DEAD_TAIL_FRACTION:.2f} x fleet median "
                            f"{ref:.2f} m/s")
        return report
    if zf >= NOISY_ZERO_FRACTION:
        report["verdict"] = "noisy"
        report["reason"] = f"{zf * 100:.1f}% exact zeros"
        return report
    if distinct < MIN_DISTINCT_VALUES:
        report["verdict"] = "noisy"
        report["reason"] = f"only {distinct} distinct nonzero values"
        return report
    if gust < MIN_GUSTINESS:
        report["verdict"] = "noisy"
        report["reason"] = f"gustiness {gust:.3f} below {MIN_GUSTINESS}"
        return report

    report["verdict"] = "ok"
    report["reason"] = "channel behaves like weather"
    return report


def assess_fleet(per_station: Dict[str, Sequence[Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Assess every station, using the fleet median as the reference scale so the
    "effectively zero" test is relative to normal conditions for this network.
    """
    medians = []
    for values in per_station.values():
        v = _finite(values)
        if v:
            medians.append(sorted(v)[len(v) // 2])
    fleet_median = float(sorted(medians)[len(medians) // 2]) if medians else 0.0
    out = {sid: assess_wind(vals, fleet_median) for sid, vals in per_station.items()}
    out["_fleet_median_wind"] = {"verdict": "reference", "mean": fleet_median, "n": len(medians)}
    return out


def healthy_stations(assessment: Dict[str, Dict[str, Any]],
                     include_noisy: bool = False) -> List[str]:
    ok = {"ok"} if not include_noisy else {"ok", "noisy"}
    return sorted(sid for sid, r in assessment.items()
                  if not sid.startswith("_") and r.get("verdict") in ok)


def unusable_stations(assessment: Dict[str, Dict[str, Any]]) -> List[str]:
    return sorted(sid for sid, r in assessment.items()
                  if not sid.startswith("_") and r.get("verdict") in ("dead", "absent"))


def summarise(assessment: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    counts = Counter(r.get("verdict") for sid, r in assessment.items()
                     if not sid.startswith("_"))
    return {
        "counts": dict(counts),
        "healthy": healthy_stations(assessment),
        "unusable": unusable_stations(assessment),
        "fleet_median_wind": assessment.get("_fleet_median_wind", {}).get("mean"),
    }

# ---------------------------------------------------------------------------
# Live assessment
#
# The historical assessment above runs over the training CSV. Inference needs
# the same verdict for a station as it behaves RIGHT NOW, judged from the live
# observation trail rather than a file that ends weeks ago. A sensor that dies
# tomorrow must stop producing a wind forecast today.
# ---------------------------------------------------------------------------

DEFAULT_TRAIL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "observation_audit.jsonl")

# How long a station's live verdict is reused before recomputing it. The trail
# is append-only and cheap to tail, but re-reading it per forecast would be
# wasteful; a stale verdict is also a risk, so this is deliberately short.
LIVE_TTL_SECONDS = 300.0

_LIVE_CACHE: Dict[str, Dict[str, Any]] = {}


def _load_trail(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if isinstance(rec, dict):
                    out.append(rec)
    except OSError:
        return []
    return out


def live_health(station_id: str, trail_path: Optional[str] = None,
                lookback_hours: float = 24.0,
                now: Optional[datetime] = None,
                ttl_seconds: float = LIVE_TTL_SECONDS) -> Dict[str, Any]:
    """
    Verdict for a station from its recent LIVE observations.

    Returns a verdict whose "verdict" key is "unknown" when there is not enough
    data to judge. Callers must treat "unknown" as "carry on as before" -- an
    inconclusive health check must never suppress a forecast.
    """
    ref = now or datetime.now(timezone.utc)
    key = f"{station_id}|{trail_path or DEFAULT_TRAIL}"
    cached = _LIVE_CACHE.get(key)
    if cached and (time.monotonic() - cached["at"]) < ttl_seconds:
        return cached["report"]

    path = trail_path or DEFAULT_TRAIL
    cutoff = ref - timedelta(hours=lookback_hours)
    values: List[float] = []
    fleet: Dict[str, List[float]] = {}
    for rec in _load_trail(path):
        # Only THIS station's readings may be judged for THIS station. Without
        # the filter below the fleet's aggregate verdict is attributed to
        # whichever station was asked for, so a dead anemometer reads "ok"
        # because its neighbours are healthy.
        rec_station = rec.get("station_id")
        if rec_station is None or str(rec_station) != str(station_id):
            continue
        ts = _parse_ts(rec.get("observed_at_utc"))
        if ts is None or ts < cutoff or ts > ref + timedelta(hours=1):
            continue
        w = (rec.get("telemetry") or {}).get("wind_speed_kmh")
        if w is None:
            continue
        try:
            fv = float(w)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(fv):
            continue
        values.append(fv)

    # The fleet median is the reference scale, built from every station, so
    # "effectively zero" means zero relative to normal conditions for this
    # network rather than an absolute wind speed.
    for rec in _load_trail(path):
        ts = _parse_ts(rec.get("observed_at_utc"))
        if ts is None or ts < cutoff or ts > ref + timedelta(hours=1):
            continue
        w = (rec.get("telemetry") or {}).get("wind_speed_kmh")
        if w is None:
            continue
        try:
            fv = float(w)
        except (TypeError, ValueError):
            continue
        if math.isfinite(fv):
            fleet.setdefault(rec.get("station_id") or "", []).append(fv)

    if len(values) < MIN_SAMPLES:
        report = {"verdict": "unknown", "n": len(values), "mean": None,
                  "reason": f"only {len(values)} live samples, need {MIN_SAMPLES}"}
    else:
        medians = [sorted(v)[len(v) // 2] for v in fleet.values() if v]
        ref_median = float(sorted(medians)[len(medians) // 2]) if medians else 0.0
        report = assess_wind(values, ref_median)
        report["source"] = "live_observation_trail"
        report["lookback_hours"] = lookback_hours
    report["station_id"] = station_id
    _LIVE_CACHE[key] = {"at": time.monotonic(), "report": report}
    return report


def clear_live_cache() -> None:
    _LIVE_CACHE.clear()
