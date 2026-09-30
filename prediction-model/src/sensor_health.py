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


# ---------------------------------------------------------------------------
# Continuous station health
#
# WHY A SCHEDULED CHECK
# --------------------
# assess_wind() answers "is this wind channel behaving", but only for a window
# someone remembered to open. Nothing re-evaluated it, so a station that went
# quiet, or an anemometer that died after the last run, stayed "ok" on the last
# verdict indefinitely. That is the same class of failure as a training corpus
# that quietly rotted: a correct check that is never run.
#
# `scheduled_station_health()` is the runnable form. It joins two signals that
# are individually insufficient:
#
#   liveness   is the station reporting at all, and how completely
#   wind       is the channel behaving, judged against the fleet
#
# A dead anemometer and a dead station look identical in a wind-only check --
# both are a run of zeros -- and the two have opposite fixes: re-site the
# sensor, or drive to the station. Liveness separates them. Conversely, 95pM7BAV
# reports 93% exact zeros and a p95 of 0.047 m/s, so a zero-FREQUENCY rule would
# have called it dead; the p95 test against the fleet median is what catches it,
# and that reasoning is inherited unchanged from assess_wind() above.
#
# An inconclusive result is reported as "unknown" and never escalated. A health
# check that pages on its own inability to decide is a health check people
# disable.
# ---------------------------------------------------------------------------

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
_RANK = {PASS: 0, WARN: 1, FAIL: 2}

# Liveness, from the hourly reporting grid. A healthy station reports at least
# once an hour, so a silent day is already a fault; three silent days means the
# device is off and someone has to reach it. These are deliberately the same
# numbers the corpus monitor uses, so the two cannot disagree.
SILENCE_WARN_HOURS = 24.0
SILENCE_FAIL_HOURS = 72.0

# Coverage of the window the station was actually present for. Separates "quiet
# at the end" (high coverage, long silence) from "intermittent throughout"
# (low coverage), which are different faults.
COVERAGE_WARN = 0.90
COVERAGE_FAIL = 0.60

# A hole strictly inside the station's own record. Trailing silence is already
# scored above, so this only judges the interior.
GAP_WARN_HOURS = 6.0
GAP_FAIL_HOURS = 24.0

STATION_HEALTH_SCHEMA = "kloudtrack.station_health.v1"


def _health_rank(verdict: str) -> int:
    return _RANK.get(verdict, 0)


def _health_worst(*verdicts: str) -> str:
    out = PASS
    for v in verdicts:
        if _health_rank(v) > _health_rank(out):
            out = v
    return out


def assess_station_liveness(last_observed: Optional[datetime],
                            now: Optional[datetime] = None,
                            bins_present: Optional[int] = None,
                            bins_expected: Optional[int] = None,
                            max_internal_gap_hours: Optional[float] = None,
                            rows: Optional[int] = None,
                            silence_warn_hours: float = SILENCE_WARN_HOURS,
                            silence_fail_hours: float = SILENCE_FAIL_HOURS,
                            coverage_warn: float = COVERAGE_WARN,
                            coverage_fail: float = COVERAGE_FAIL,
                            gap_warn_hours: float = GAP_WARN_HOURS,
                            gap_fail_hours: float = GAP_FAIL_HOURS) -> Dict[str, Any]:
    """
    Liveness for one station, as a plain verdict plus the numbers behind it.

    Pure: no I/O, no clock read unless `now` is omitted. Everything is
    optional so a caller holding only a timestamp still gets an answer.
    """
    ref = now or datetime.now(timezone.utc)
    report: Dict[str, Any] = {
        "rows": rows,
        "last_observed": None,
        "silence_hours": None,
        "silent": None,
        "coverage": None,
        "max_internal_gap_hours": max_internal_gap_hours,
        "reasons": [],
    }
    if last_observed is None:
        report["verdict"] = FAIL
        report["silent"] = True
        report["reason"] = "no observation on record"
        report["reasons"].append(
            "station has never been observed; this is either a device that is "
            "off or an id that does not match anything upstream")
        return report

    if last_observed.tzinfo is None:
        last_observed = last_observed.replace(tzinfo=timezone.utc)
    report["last_observed"] = last_observed.astimezone(timezone.utc).isoformat()
    silence = (ref - last_observed.astimezone(timezone.utc)).total_seconds() / 3600.0
    report["silence_hours"] = round(silence, 3)

    if silence < 0:
        report["verdict"] = FAIL
        report["reason"] = f"last observation is {abs(silence):.1f} h in the future"
        report["silent"] = False
        return report

    verdicts = []
    if silence >= silence_fail_hours:
        verdicts.append(FAIL)
    elif silence >= silence_warn_hours:
        verdicts.append(WARN)
    report["silent"] = silence >= silence_warn_hours
    if silence >= silence_warn_hours:
        report["reasons"].append(
            f"silent for {silence:.1f} h (last observation "
            f"{report['last_observed']})")

    if bins_expected:
        coverage = bins_present / bins_expected
        report["coverage"] = round(coverage, 6)
        if coverage <= coverage_fail:
            verdicts.append(FAIL)
            report["reasons"].append(
                f"covers only {coverage * 100:.1f}% of its own span "
                f"({bins_present}/{bins_expected} hourly bins)")
        elif coverage <= coverage_warn:
            verdicts.append(WARN)
            report["reasons"].append(
                f"covers {coverage * 100:.1f}% of its own span "
                f"({bins_present}/{bins_expected} hourly bins)")

    gap = max_internal_gap_hours
    if gap is not None:
        if gap >= gap_fail_hours:
            verdicts.append(FAIL)
            report["reasons"].append(
                f"{gap:.1f} h hole inside the record; intermittent, not silent")
        elif gap >= gap_warn_hours:
            verdicts.append(WARN)
            report["reasons"].append(
                f"{gap:.1f} h hole inside the record")

    report["verdict"] = _health_worst(*verdicts) if verdicts else PASS
    report["reason"] = (report["reasons"][0] if report["reasons"]
                        else "reporting normally")
    return report


def scheduled_station_health(csv_path: Optional[str] = None,
                             now: Optional[datetime] = None,
                             stations: Optional[Iterable[str]] = None,
                             lookback_hours: Optional[float] = None,
                             include_wind: bool = True,
                             wind_column: str = "wind_speed",
                             thresholds: Optional[Dict[str, Any]] = None,
                             freshness_report: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Re-evaluate every station and emit a machine-readable status per station.

    `freshness_report` lets a caller that has already run the freshness monitor
    pass its result in and skip a second pass over the CSV. `csv_path` is the
    fallback. One of the two must be supplied.

    `lookback_hours` restricts the wind verdict to the most recent N hours, so a
    channel that failed after the last training run is caught; leave it None to
    judge the whole corpus the way assess_fleet() does.
    """
    if freshness_report is None and csv_path is None:
        raise ValueError("scheduled_station_health needs csv_path or freshness_report")
    ref = now or datetime.now(timezone.utc)
    cfg = {"silence_warn_hours": SILENCE_WARN_HOURS,
           "silence_fail_hours": SILENCE_FAIL_HOURS,
           "coverage_warn": COVERAGE_WARN,
           "coverage_fail": COVERAGE_FAIL,
           "gap_warn_hours": GAP_WARN_HOURS,
           "gap_fail_hours": GAP_FAIL_HOURS}
    cfg.update(thresholds or {})

    # Imported here, not at module scope: the corpus scanner is a heavier
    # dependency than this module otherwise carries, and nothing above needs it.
    import check_data_freshness as cdf

    if freshness_report is None:
        expected = list(stations) if stations is not None else None
        freshness_report = cdf.check_csv(csv_path, now=ref, expected_stations=expected)

    wind_by_station: Dict[str, List[Any]] = {}
    fleet_median: Optional[float] = None
    if include_wind:
        wind_by_station, fleet_median = _recent_wind(
            csv_path or freshness_report["source"]["path"],
            ref, lookback_hours, wind_column)

    out: Dict[str, Dict[str, Any]] = {}
    for sid, entry in sorted(freshness_report["stations"].items()):
        last = _parse_ts(entry.get("last"))
        live = assess_station_liveness(
            last, now=ref,
            bins_present=entry.get("bins_present"),
            bins_expected=entry.get("bins_expected"),
            max_internal_gap_hours=entry.get("max_internal_gap_hours"),
            rows=entry.get("rows"),
            silence_warn_hours=cfg["silence_warn_hours"],
            silence_fail_hours=cfg["silence_fail_hours"],
            coverage_warn=cfg["coverage_warn"],
            coverage_fail=cfg["coverage_fail"],
            gap_warn_hours=cfg["gap_warn_hours"],
            gap_fail_hours=cfg["gap_fail_hours"])
        status: Dict[str, Any] = {
            "station_id": sid,
            "name": entry.get("name") or "",
            "liveness": live,
            "verdict": live["verdict"],
            "reasons": list(live.get("reasons") or []),
        }

        if include_wind and sid in wind_by_station:
            # Same call, same reference scale, same reasoning as assess_fleet():
            # magnitude, not zero frequency, is what separates a twitching
            # sensor from a real one.
            wind = assess_wind(wind_by_station[sid], fleet_median)
            wind["source"] = "csv" if csv_path or freshness_report else "unknown"
            if lookback_hours:
                wind["lookback_hours"] = lookback_hours
            status["wind"] = wind
            if wind["verdict"] == "dead":
                status["verdict"] = _health_worst(status["verdict"], FAIL)
                status["reasons"].append(
                    f"wind channel dead: {wind.get('reason')}")
                if live["verdict"] != FAIL:
                    status["reasons"].append(
                        "station is still transmitting, so this is a sensor "
                        "fault, not an outage")
            elif wind["verdict"] == "absent":
                status["reasons"].append(
                    f"wind channel not assessable: {wind.get('reason')}")
            elif wind["verdict"] == "noisy":
                status["verdict"] = _health_worst(status["verdict"], WARN)
                status["reasons"].append(
                    f"wind channel noisy: {wind.get('reason')}")

        if entry.get("notes"):
            status["notes"] = list(entry["notes"])
        out[sid] = status

    counts = Counter(s["verdict"] for s in out.values())
    worst = _health_worst(freshness_report["verdict"],
                          *[s["verdict"] for s in out.values()])
    report = {
        "schema": STATION_HEALTH_SCHEMA,
        "generated_at": cdf.iso(ref),
        "source": freshness_report["source"]["path"],
        "lookback_hours": lookback_hours,
        "fleet_median_wind": fleet_median,
        "thresholds": cfg,
        "counts": {"ok": counts.get(PASS, 0), "warn": counts.get(WARN, 0),
                   "fail": counts.get(FAIL, 0)},
        "verdict": worst,
        "dead_channels": sorted(sid for sid, s in out.items()
                                if s.get("wind", {}).get("verdict") == "dead"),
        "silent_stations": sorted(sid for sid, s in out.items()
                                  if s["liveness"].get("silent")),
        "missing_stations": list(freshness_report.get("missing_stations") or []),
        "stations": out,
    }
    report["summary"] = (
        f"[{report['verdict']}] station health: {report['counts']['ok']} ok, "
        f"{report['counts']['warn']} warn, {report['counts']['fail']} fail"
        + (f"; dead wind channels: {', '.join(report['dead_channels'])}"
           if report["dead_channels"] else "")
        + (f"; not reporting: {', '.join(report['silent_stations'])}"
           if report["silent_stations"] else ""))
    return report


def _recent_wind(csv_path: str, now: datetime,
                 lookback_hours: Optional[float], column: str
                 ) -> tuple:
    """
    Wind samples per station, optionally restricted to a recent window.

    A second pass over the CSV is the cost of a correct recent verdict; the
    freshness report only carries timestamps, and timestamps cannot tell whether
    an anemometer is dead. The scan is streaming, so the memory cost is bounded
    by the number of stations, not by the size of the corpus.
    """
    import csv as _csv
    per: Dict[str, List[float]] = {}
    cutoff = (now - timedelta(hours=lookback_hours)) if lookback_hours else None
    try:
        with open(csv_path, "r", encoding="utf-8", errors="replace", newline="") as f:
            reader = _csv.DictReader(f)
            if not reader.fieldnames or column not in reader.fieldnames:
                return per, None
            for row in reader:
                sid = (row.get("station_id") or "").strip()
                if not sid:
                    continue
                if cutoff is not None:
                    ts = _parse_ts(row.get("recorded_at"))
                    if ts is None or ts < cutoff or ts > now + timedelta(hours=1):
                        continue
                raw = row.get(column)
                if raw is None or raw == "":
                    continue
                try:
                    fv = float(raw)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(fv):
                    per.setdefault(sid, []).append(fv)
    except OSError:
        return per, None
    medians = [sorted(v)[len(v) // 2] for v in per.values() if v]
    fleet = float(sorted(medians)[len(medians) // 2]) if medians else None
    return per, fleet


def render_station_health(report: Dict[str, Any], max_rows: int = 60) -> str:
    """Human-readable view of `scheduled_station_health`, for an alert body."""
    lines = [f"[{report['verdict']}] STATION HEALTH  {report['source']}",
             "=" * 78,
             f"  generated  {report['generated_at']}",
             f"  window     {report['lookback_hours'] or 'whole corpus'}"
             + (f"   fleet median wind "
                f"{report['fleet_median_wind']:.3f}"
                if report.get("fleet_median_wind") is not None else ""),
             f"  counts     {report['counts']['ok']} ok  "
             f"{report['counts']['warn']} warn  {report['counts']['fail']} fail",
             "",
             f"  {'station':<10}{'live':<6}{'silence':>9}{'cover':>7}"
             f"{'gap':>7}  {'wind':<7}v  reason",
             f"  {'-' * 10}{'-' * 6}{'-' * 9}{'-' * 7}{'-' * 7}  {'-' * 7}"
             f"{'-' * 2}  {'-' * 40}"]
    order = sorted(report["stations"],
                   key=lambda s: (-_health_rank(report["stations"][s]["verdict"]), s))
    for sid in order[:max_rows]:
        s = report["stations"][sid]
        live = s["liveness"]
        sil = ("n/a" if live.get("silence_hours") is None
               else f"{live['silence_hours']:.1f}h")
        cov = ("n/a" if live.get("coverage") is None
               else f"{live['coverage'] * 100:5.1f}%")
        gap = s["liveness"].get("max_internal_gap_hours")
        gap_s = "-" if gap is None else f"{gap:.1f}h"
        wind = s.get("wind", {}).get("verdict", "-")
        reason = (s["reasons"] or ["-"])[0]
        lines.append(f"  {sid:<10}{('yes' if live.get('silent') else 'no'):<6}"
                     f"{sil:>9}{cov:>7}{gap_s:>7}  {wind:<7}{s['verdict']:<2}  "
                     f"{reason[:56]}")
    if len(order) > max_rows:
        lines.append(f"  ... and {len(order) - max_rows} more")
    if report["dead_channels"]:
        lines += ["", "  DEAD WIND CHANNELS"]
        for sid in report["dead_channels"]:
            w = report["stations"][sid].get("wind", {})
            lines.append(f"    {sid:<10} {w.get('reason')}")
    if report["missing_stations"]:
        lines += ["", "  NO DATA AT ALL: " + ", ".join(report["missing_stations"])]
    return "\n".join(lines)
