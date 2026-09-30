"""
Data Quality Report for Weather Telemetry -- the reporting layer that makes
quarantine visible.

WHY
---
``dataset.py`` quarantines rows that fail physical-bounds or completeness
checks and increments counters in ``quarantine_counts``. Those counters are
emitted into ``data_quality_report.json`` and then read by nobody. On the
current corpus 4,921 of 30,470 rows (16.2%) are dropped silently, so a model is
trained on a subset nobody chose deliberately. This module answers the
questions the counters cannot:

  1. What is the loss, as a percentage, per reason?
  2. Is each loss CLUSTERED IN TIME (an outage -- remedy is a site visit / gap
     handling) or SPREAD EVENLY (a per-sensor fault -- remedy is a sensor)?
     These two look identical in a counter and call for opposite fixes.
  3. Is the retained set BIASED relative to the corpus we meant to model? A
     row dropped for a pressure-datum fault is not a random row.
  4. For the pressure rejections specifically: are the rejected values a sensor
     fault, an altitude/datum offset, or corrupt rows? The answer decides
     whether a calibration offset recovers 550 rows or whether they must be
     dropped.

This module is deliberately a REPLICA of the ``dataset.py`` classification
order, not a reimplementation of the training pipeline. It must not import
``dataset.py`` (which pulls in torch and runs the full split/normalization
machinery), and it must never write to any file other than its own output.

Method notes that matter for reading the numbers
-----------------------------------------------
* Time-matched bias. Comparing quarantined rows against the whole retained
  corpus confounds the fault with the weather: if a station faults for 12 days
  in a cool month, the naive comparison says "quarantined rows are colder"
  when really it says "those 12 days were colder". Every bias figure is
  therefore computed twice -- once against the station's whole retained
  history (``station_standardised``) and once against retained rows from the
  SAME station within a +/-72 h window around the quarantined row
  (``time_matched``). The time-matched number is the one to act on.
* Clustering. A run is a maximal set of quarantined rows for one
  (station, reason) that are no more than ``CLUSTER_GAP_HOURS`` apart. A run of
  at least ``OUTAGE_RUN_MIN_HOURS`` marks the cohort ``clustered_outage``;
  anything shorter is ``spread``. Long runs are outages, short scattered ones
  are per-sensor noise.
* Reason precedence. ``dataset.py`` tests completeness BEFORE physical bounds,
  so a row that is both incomplete and out of bounds is counted only as
  ``weather_missing_field``. ``rows_with_hidden_bounds_violations`` measures how
  many rows that hides. Reported as a defect; not fixed here.

Outputs:
  prediction-model/data/data_quality_report_current.json

CLI:
  python report_data_quality.py
  python report_data_quality.py --csv <path> --out <path>
  python report_data_quality.py --no-write
"""

import os
import csv
import json
import math
import hashlib
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SRC_DIR), "data")
DEFAULT_CSV = os.path.join(DATA_DIR, "weather_telemetry_current.csv")
DEFAULT_OUT = os.path.join(DATA_DIR, "data_quality_report_current.json")

SCHEMA = "data_quality_report_current/v1"

# ---------------------------------------------------------------------------
# Contract mirrored from dataset.py. Kept as a literal copy on purpose: if
# dataset.py changes these, the report must change with it, loudly, rather than
# import torch and drift. test_data_quality.py asserts the two agree.
# ---------------------------------------------------------------------------
REQUIRED_FIELDS = (
    "temperature",
    "heat_index",
    "humidity",
    "wind_speed",
    "wind_direction",
    "pressure",
    "precipitation",
)

PHYSICAL_BOUNDS: Dict[str, Tuple[float, float]] = {
    "temperature": (10.0, 50.0),
    "heat_index": (10.0, 70.0),
    "humidity": (10.0, 100.0),
    "wind_speed": (0.0, 180.0),
    "wind_direction": (0.0, 360.0),
    "pressure": (900.0, 1050.0),
    "precipitation": (0.0, 50.0),
}

MIN_VALID_YEAR = 2025
MAX_VALID_YEAR = 2027

NULL_TOKENS = ("nan", "na", "n/a", "null", "none", "-")

# Local solar time offset used for hour-of-day reporting (dataset.py uses
# UTC+8 for the Philippines in its engineered features).
LOCAL_UTC_OFFSET_HOURS = 8

# --- Clustering thresholds -------------------------------------------------
# Telemetry is nominally hourly. Anything within this of the previous
# quarantined row is the same episode.
CLUSTER_GAP_HOURS = 1.5
# A contiguous run this long is an outage, not a flaky reading.
OUTAGE_RUN_MIN_HOURS = 6.0

# --- Bias thresholds -------------------------------------------------------
# Effect size at or above this is a material covariate shift worth disclosing.
MATERIAL_EFFECT_SIZE = 1.0
# Effect size below this is noise; used to call a reason "unbiased".
NEGLIGIBLE_EFFECT_SIZE = 0.5
# A matched control thinner than this cannot localise the baseline: the SD of
# three neighbours is itself unstable and the ratio explodes. Pairs like that
# are reported as under-determined instead of as a huge effect size.
MIN_CONTROL_PAIRS = 8
# A cohort smaller than this cannot support a bias claim at all. 10 dropped
# rows out of 30,000 is a rounding error, and letting its effect size compete
# in the impact ranking would put garbage at number four.
MIN_BIAS_COHORT = 30
# Effect sizes are clamped before ranking. Beyond this a "bias" is a statement
# that the companion field is corrupt, not that the weather shifted, and the
# two must not compete in the same ranking column.
EFFECT_SIZE_CAP = 10.0
# Standard atmosphere, for turning a pressure deficit into an altitude.
STD_ATM_P0_HPA = 1013.25
STD_ATM_EXPONENT = 5.255

# --- Pressure cohort diagnosis thresholds ----------------------------------
# A cohort is a "frozen sensor" if one value accounts for more than this share
# of it: real weather never repeats a value to the hPa.
FROZEN_SHARE_MIN = 0.5
FROZEN_MIN_COHORT = 20
# A cohort is a datum-offset candidate if it is tight (IQR below this fraction
# of the station's own retained IQR) yet not frozen, and sits at least this far
# below the station's retained median in units of the retained IQR.
DATUM_TIGHT_IQR_FRACTION = 0.50
DATUM_MIN_OFFSET_IQRS = 2.0
DATUM_MIN_COHORT = 20

# --- Verdict thresholds ----------------------------------------------------
# Share of the corpus that must survive quarantine for the fleet to look good.
GOOD_LOSS_MAX_PCT = 5.0
FAIR_LOSS_MAX_PCT = 10.0
# A verdict is never better than FAIR while a material bias stands.
BIAS_CAPS_VERDICT = "FAIR"

REASON_CLASS = {
    "weather_missing_field": "completeness",
    "weather_bounds_temperature": "physical_bounds",
    "weather_bounds_heat_index": "physical_bounds",
    "weather_bounds_humidity": "physical_bounds",
    "weather_bounds_wind_speed": "physical_bounds",
    "weather_bounds_wind_direction": "physical_bounds",
    "weather_bounds_pressure": "physical_bounds",
    "weather_bounds_precipitation": "physical_bounds",
    "weather_year_out_of_bounds": "structural",
    "weather_malformed_timestamp": "structural",
    "weather_missing_station_id": "structural",
}

# --- Impact scoring weights (documented in the report so it is auditable) --
IMPACT_W_VOLUME = 0.35
IMPACT_W_BIAS = 0.35
IMPACT_W_SHAPE = 0.15
IMPACT_W_RECOVERABLE = 0.15
# Volume saturates at this share of the corpus; bias saturates at this effect
# size. Beyond these the problem is catastrophic and the extra magnitude adds
# nothing to the ranking.
IMPACT_VOLUME_SATURATION_PCT = 15.0
IMPACT_BIAS_SATURATION = 3.0


# ---------------------------------------------------------------------------
# Small numeric helpers (stdlib only -- this module must not need numpy)
# ---------------------------------------------------------------------------

def percentile(values: Sequence[float], pct: float) -> Optional[float]:
    """Nearest-rank percentile. No interpolation, so it is safe on tiny samples."""
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    k = int(round(pct / 100.0 * (len(s) - 1)))
    return float(s[max(0, min(len(s) - 1, k))])


def median(values: Sequence[float]) -> Optional[float]:
    return percentile(values, 50.0)


def describe(values: Sequence[float], digits: int = 4) -> Dict[str, Any]:
    """mean, sd, p1, p50, p99, min, max. sd is the population sd (n denominator)."""
    vals = [float(v) for v in values if v is not None]
    n = len(vals)
    if n == 0:
        return {"n": 0, "mean": None, "sd": None, "p1": None, "p50": None,
                "p99": None, "min": None, "max": None}
    mean = sum(vals) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in vals) / n)

    def r(v: Optional[float]) -> Optional[float]:
        return None if v is None else round(float(v), digits)

    return {
        "n": n,
        "mean": r(mean),
        "sd": r(sd),
        "p1": r(percentile(vals, 1.0)),
        "p50": r(percentile(vals, 50.0)),
        "p99": r(percentile(vals, 99.0)),
        "min": r(min(vals)),
        "max": r(max(vals)),
    }


def iqr(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return percentile(values, 75.0) - percentile(values, 50.0)


def parse_utc_timestamp(raw: Any) -> Optional[datetime]:
    """Same acceptance set as dataset.parse_utc_timestamp."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%d %H:%M:%S%z"):
        try:
            return datetime.strptime(s, fmt).astimezone(timezone.utc)
        except ValueError:
            continue
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s[:19], fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def parse_field_value(raw: Any) -> Optional[float]:
    """Strict numeric parse. Returns None for absent / NaN / inf / unparseable.

    Mirrors dataset.py: a blank, a null token, or a non-finite float is a
    MISSING observation, never a zero. This is the distinction the old
    ``float(x or default)`` code got wrong.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        s = raw.strip()
        if s == "" or s.lower() in NULL_TOKENS:
            return None
    try:
        f = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f


def in_bounds(field: str, value: float,
              bounds: Dict[str, Tuple[float, float]] = None) -> bool:
    b = (bounds or PHYSICAL_BOUNDS).get(field)
    if b is None:
        return True
    lo, hi = b
    return lo <= value <= hi


def implied_altitude_m(pressure_hpa: float) -> Optional[float]:
    """Station pressure -> elevation via the ISA barometric formula.

    Used to ask 'is this reading a plausible station pressure rather than a
    sea-level pressure?' A 160 hPa deficit is about 1.4 km, which is a
    different order of magnitude from anything the fleet's sites sit at -- so a
    deficit of that size is a datum fault, not a real site altitude.
    """
    if pressure_hpa is None or pressure_hpa <= 0:
        return None
    ratio = pressure_hpa / STD_ATM_P0_HPA
    if ratio <= 0:
        return None
    return (288.15 / 0.0065) * (1.0 - ratio ** (1.0 / STD_ATM_EXPONENT))


def compute_file_sha256(path: str) -> str:
    if not os.path.exists(path):
        return "file_not_found"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 16):
            h.update(chunk)
    return h.hexdigest()


def git_head_commit(repo_dir: str) -> str:
    """Read-only ``git rev-parse HEAD``. Never a write; never raises."""
    try:
        import subprocess
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir,
            stderr=subprocess.DEVNULL, text=True, timeout=10)
        return out.strip() or "unknown"
    except Exception:
        return "unknown"


def _pct(numerator: float, denominator: float, digits: int = 2) -> float:
    if not denominator:
        return 0.0
    return round(100.0 * numerator / denominator, digits)


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None


# ---------------------------------------------------------------------------
# Row ingestion and classification
# ---------------------------------------------------------------------------

class TelemetryRow(dict):
    """A raw CSV row plus its classification. Dict so it stays easy to assert on."""


def read_rows(csv_path: str) -> List[TelemetryRow]:
    """Read and classify every row, in dataset.py's precedence order."""
    out: List[TelemetryRow] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for raw in csv.DictReader(f):
            out.append(classify_row(raw))
    return out


def classify_row(raw: Dict[str, Any],
                 bounds: Dict[str, Tuple[float, float]] = None) -> TelemetryRow:
    """Assign the row a primary quarantine reason, or None if it is ingested.

    Precedence is dataset.py's: timestamp -> year -> station_id -> missing
    fields -> physical bounds. Preserving the order is the whole point; a
    different order would produce different counters and the report would be
    describing a pipeline that does not exist.
    """
    bounds = bounds or PHYSICAL_BOUNDS
    station = (raw.get("station_id") or "").strip() or None
    name = (raw.get("station_name") or "").strip() or None
    dt = parse_utc_timestamp(raw.get("recorded_at"))

    parsed: Dict[str, float] = {}
    missing: List[str] = []
    for field in REQUIRED_FIELDS:
        v = parse_field_value(raw.get(field))
        if v is None:
            missing.append(field)
        else:
            parsed[field] = v

    if dt is None:
        reason = "weather_malformed_timestamp"
    elif dt.year < MIN_VALID_YEAR or dt.year > MAX_VALID_YEAR:
        reason = "weather_year_out_of_bounds"
    elif station is None:
        reason = "weather_missing_station_id"
    elif missing:
        reason = "weather_missing_field"
    else:
        reason = None
        for field in REQUIRED_FIELDS:
            if not in_bounds(field, parsed[field], bounds):
                reason = "weather_bounds_" + field
                break

    # A bounds check that never ran because a completeness check fired first.
    # Only meaningful on the completeness path: on a bounds path the offending
    # field is the trigger, and counting it again here would report the same
    # violations a second time under a different heading.
    hidden_fields: List[str] = []
    if reason == "weather_missing_field":
        hidden_fields = [f for f in REQUIRED_FIELDS
                         if f in parsed and not in_bounds(f, parsed[f], bounds)]

    return TelemetryRow(
        station_id=station,
        station_name=name,
        timestamp=dt,
        reason=reason,
        missing_fields=list(missing),
        parsed=parsed,
        # Out-of-bounds values the reason label does not mention, because the
        # completeness check fired first. Counted so the report can disclose
        # how much the reason labels hide.
        unreported_bounds_violations=hidden_fields,
    )


# ---------------------------------------------------------------------------
# Clustering: outage-shaped or spread?
# ---------------------------------------------------------------------------

def contiguous_runs(timestamps: Sequence[datetime],
                    gap_hours: float = CLUSTER_GAP_HOURS) -> List[List[datetime]]:
    """Maximal groups of timestamps separated by no more than ``gap_hours``."""
    ordered = sorted(timestamps)
    if not ordered:
        return []
    runs: List[List[datetime]] = [[ordered[0]]]
    for t in ordered[1:]:
        if (t - runs[-1][-1]).total_seconds() / 3600.0 <= gap_hours:
            runs[-1].append(t)
        else:
            runs.append([t])
    return runs


def classify_clustering(timestamps: Sequence[datetime],
                        gap_hours: float = CLUSTER_GAP_HOURS,
                        run_min_hours: float = OUTAGE_RUN_MIN_HOURS
                        ) -> Dict[str, Any]:
    """Is a cohort of quarantined rows an outage or per-sensor noise?

    ``verdict`` is ``clustered_outage`` when the longest contiguous run reaches
    ``run_min_hours``; otherwise ``spread``. The supporting numbers are always
    returned so a reader can see the gradient rather than trust the label.
    """
    stamps = sorted(t for t in timestamps if t is not None)
    if not stamps:
        return {
            "rows": 0, "distinct_days": 0, "first": None, "last": None,
            "span_hours": 0.0, "n_runs": 0, "longest_run_hours": 0.0,
            "fill_fraction": 0.0, "verdict": "spread",
            "implied_outage_hours": 0.0, "n_runs_at_or_above_threshold": 0,
            "longest_runs": [],
        }

    runs = contiguous_runs(stamps, gap_hours)
    run_hours = [len(r) for r in runs]
    longest = max(run_hours)
    span_hours = (stamps[-1] - stamps[0]).total_seconds() / 3600.0
    fill = len(stamps) / (span_hours + 1.0) if span_hours >= 0 else 1.0
    long_runs = sorted(run_hours, reverse=True)[:5]
    long_spans = sorted(((r[0], r[-1]) for r in runs),
                        key=lambda p: p[1] - p[0], reverse=True)[:3]

    return {
        "rows": len(stamps),
        "distinct_days": len({t.date() for t in stamps}),
        "first": _iso(stamps[0]),
        "last": _iso(stamps[-1]),
        "span_hours": round(span_hours, 1),
        "n_runs": len(runs),
        "longest_run_hours": longest,
        "top_run_lengths": long_runs,
        "fill_fraction": round(fill, 4),
        "verdict": "clustered_outage" if longest >= run_min_hours else "spread",
        "implied_outage_hours": int(sum(h for h in run_hours if h >= run_min_hours)),
        "n_runs_at_or_above_threshold": int(sum(1 for h in run_hours
                                                 if h >= run_min_hours)),
        "longest_run_windows": [
            {"start": _iso(a), "end": _iso(b),
             "hours": round((b - a).total_seconds() / 3600.0, 1)}
            for a, b in long_spans
        ],
    }


# ---------------------------------------------------------------------------
# Observation continuity
# ---------------------------------------------------------------------------

def continuity_metrics(timestamps: Sequence[datetime]) -> Dict[str, Any]:
    """Longest gap and distinct gap count on the observed hourly grid."""
    stamps = sorted({t for t in timestamps if t is not None})
    if not stamps:
        return {"observed_bins": 0, "span_bins": 0, "longest_gap_hours": 0.0,
                "distinct_gaps": 0, "gaps_over_6h": 0, "missing_bins": 0,
                "largest_gap_window": None, "largest_gaps": []}
    span_bins = int((stamps[-1] - stamps[0]).total_seconds() // 3600) + 1
    gaps: List[Tuple[datetime, datetime, float]] = []
    for a, b in zip(stamps, stamps[1:]):
        h = (b - a).total_seconds() / 3600.0
        if h > 1.0 + 1e-6:
            gaps.append((a, b, h))
    gaps.sort(key=lambda g: g[2], reverse=True)
    return {
        "observed_bins": len(stamps),
        "span_bins": span_bins,
        "coverage_pct": _pct(len(stamps), span_bins),
        "longest_gap_hours": round(gaps[0][2], 1) if gaps else 0.0,
        "distinct_gaps": len(gaps),
        "gaps_over_6h": sum(1 for g in gaps if g[2] > 6.0),
        "missing_bins": span_bins - len(stamps),
        "largest_gap_window": (
            {"from": _iso(gaps[0][0]), "to": _iso(gaps[0][1]),
             "hours": round(gaps[0][2], 1)} if gaps else None),
        "largest_gaps": [
            {"from": _iso(a), "to": _iso(b), "hours": round(h, 1)}
            for a, b, h in gaps[:5]
        ],
    }


# ---------------------------------------------------------------------------
# Training impact: does dropping these rows move the retained distribution?
# ---------------------------------------------------------------------------

def _robust_scale(values: Sequence[float]) -> Optional[float]:
    """IQR of the control values, floored away from zero.

    SD is the wrong normaliser here: control windows around a corrupt row can
    contain a handful of absurd companion readings, and one of those inflates
    the SD until every effect size against it collapses toward zero. IQR is
    unmoved by a few outliers, so the effect size keeps its meaning.
    """
    if len(values) < 4:
        return None
    q1 = percentile(values, 25.0)
    q3 = percentile(values, 75.0)
    if q1 is None or q3 is None:
        return None
    return q3 - q1


def time_matched_bias(quarantined: Sequence[TelemetryRow],
                      retained_by_station: Dict[str, List[TelemetryRow]],
                      fields: Sequence[str] = ("temperature", "humidity",
                                               "wind_speed", "precipitation",
                                               "pressure"),
                      window_hours: float = 72.0,
                      exclude_fields: Sequence[str] = (),
                      bounds: Dict[str, Tuple[float, float]] = None) -> Dict[str, Any]:
    """Compare each quarantined row against retained rows, same station, +/- window.

    This is the number that answers "is the model trained on calmer data than
    reality". The whole-corpus comparison cannot, because the faults are
    multi-day episodes and a multi-day episode is a weather sample too.

    ``exclude_fields`` drops the field whose failure CAUSED the quarantine.
    Leaving it in is circular: a row rejected for reporting 845 hPa will
    always look wildly biased on pressure, and that says nothing about whether
    the rows we kept are representative. What matters is the COMPANION fields
    -- the temperature and humidity that came alongside the bad pressure.
    """
    bounds = bounds or PHYSICAL_BOUNDS
    span = timedelta(hours=window_hours)
    skip = set(exclude_fields)
    deltas: Dict[str, List[float]] = {f: [] for f in fields}
    quants: Dict[str, List[float]] = {f: [] for f in fields}
    controls: Dict[str, List[float]] = {f: [] for f in fields}
    n_controls: Dict[str, List[int]] = {f: [] for f in fields}

    for q in quarantined:
        if q["timestamp"] is None or q["station_id"] is None:
            continue
        pool = retained_by_station.get(q["station_id"], [])
        if not pool:
            continue
        lo_t, hi_t = q["timestamp"] - span, q["timestamp"] + span
        for f in fields:
            if f in skip:
                continue
            qv = q["parsed"].get(f)
            if qv is None:
                continue
            cv = [r["parsed"][f] for r in pool
                  if f in r["parsed"] and lo_t <= r["timestamp"] <= hi_t]
            ctrl = median(cv)
            if ctrl is None or len(cv) < MIN_CONTROL_PAIRS:
                continue
            quants[f].append(qv)
            controls[f].append(ctrl)
            n_controls[f].append(len(cv))
            deltas[f].append(qv - ctrl)

    out: Dict[str, Any] = {}
    for f in fields:
        if f in skip:
            out[f] = {"excluded": "trigger field for this quarantine reason"}
            continue
        n = len(deltas[f])
        if n < 5:
            out[f] = {"n_pairs": n,
                      "note": "too few matched pairs with at least %d control "
                              "rows to compare" % MIN_CONTROL_PAIRS}
            continue
        raw_effect = None
        scale = _robust_scale(controls[f])
        if scale and scale > 1e-9:
            raw_effect = median(deltas[f]) / scale
        elif not scale or scale <= 1e-9:
            # The control channel is flat, so there is no baseline to measure
            # a shift against. That makes a NON-zero delta unquantifiable --
            # but a zero delta needs no baseline to confirm, and reporting it
            # as 'undetermined' would hide the common and benign case.
            mid = median(deltas[f])
            if mid is not None and abs(mid) <= 1e-9:
                raw_effect = 0.0
        c = controls[f]
        c_mean = sum(c) / len(c)
        c_sd = math.sqrt(sum((x - c_mean) ** 2 for x in c) / len(c))
        q1, q3 = percentile(c, 25.0), percentile(c, 75.0)
        beyond = (sum(1 for d in deltas[f] if d > (q3 - q1) / 2.0 or
                      d < -(q3 - q1) / 2.0) / n) if q1 is not None else None
        capped = (raw_effect is not None
                  and abs(raw_effect) > EFFECT_SIZE_CAP)
        out[f] = {
            "n_pairs": n,
            "median_control_rows": int(median(n_controls[f])),
            "quarantined_median": round(median(quants[f]), 4),
            "matched_control_median": round(median(c), 4),
            "median_delta": round(median(deltas[f]), 4),
            "control_spread_sd": round(c_sd, 4),
            "control_iqr": round(scale, 4) if scale is not None else None,
            "control_is_flat": bool(scale is None or scale <= 1e-9),
            "effect_size": (None if raw_effect is None
                            else round(max(-EFFECT_SIZE_CAP,
                                           min(EFFECT_SIZE_CAP, raw_effect)), 4)),
            "effect_size_clamped": capped,
            "pct_pairs_beyond_control_iqr": (
                None if beyond is None else round(100.0 * beyond, 2)),
        }
    return out


def station_standardised_bias(quarantined: Sequence[TelemetryRow],
                              retained_by_station: Dict[str, List[TelemetryRow]],
                              fields: Sequence[str] = ("temperature", "humidity",
                                                       "wind_speed", "precipitation",
                                                       "pressure"),
                              exclude_fields: Sequence[str] = ()
                              ) -> Dict[str, Any]:
    """z-score of quarantined values against each station's own retained history.

    A cross-check on the time-matched result. Large here but small in
    time_matched means the fault rode a weather regime; large in both means a
    real covariate shift.
    """
    skip = set(exclude_fields)
    ref: Dict[Tuple[str, str], Dict[str, float]] = {}
    for st, pool in retained_by_station.items():
        for f in fields:
            if f in skip:
                continue
            vals = [r["parsed"][f] for r in pool if f in r["parsed"]]
            s = describe(vals)
            if s["n"] >= 20 and s["sd"] not in (None, 0.0):
                ref[(st, f)] = {"mean": s["mean"], "sd": s["sd"], "n": s["n"]}

    zs: Dict[str, List[float]] = {f: [] for f in fields}
    for q in quarantined:
        for f in fields:
            r = ref.get((q["station_id"], f))
            if not r or f not in q["parsed"]:
                continue
            z = (q["parsed"][f] - r["mean"]) / r["sd"]
            zs[f].append(max(-EFFECT_SIZE_CAP, min(EFFECT_SIZE_CAP, z)))

    out: Dict[str, Any] = {}
    for f in fields:
        if f in skip:
            out[f] = {"excluded": "trigger field for this quarantine reason"}
            continue
        if not zs[f]:
            out[f] = {"n": 0}
            continue
        s = describe(zs[f])
        out[f] = {"n": s["n"], "mean_z_clamped": s["mean"], "sd_z": s["sd"],
                  "p50_z_clamped": s["p50"], "p99_z_clamped": s["p99"]}
    return out


def companion_corruption_rate(cohort: Sequence[TelemetryRow],
                              trigger_fields: Sequence[str],
                              fields: Sequence[str] = REQUIRED_FIELDS,
                              bounds: Dict[str, Tuple[float, float]] = None
                              ) -> Dict[str, Any]:
    """How often is a COMPANION field (not the trigger) also out of bounds?

    This separates two situations that look identical in a row count. If the
    companions are physically plausible, dropping the row removes a real
    weather sample and the retained set shifts -- a bias to disclose. If the
    companions are corrupt too, the row was never weather; dropping it is
    correct and there is no selection effect to worry about.
    """
    bounds = bounds or PHYSICAL_BOUNDS
    trigger = set(trigger_fields)
    companions = [f for f in fields if f not in trigger]
    n = len(cohort)
    if n == 0:
        return {"rows": 0, "rows_with_corrupt_companion": 0, "rate": None,
                "clean_subset_rows": 0, "clean_subset_pct": None}
    bad_rows = 0
    per_field = Counter()
    for r in cohort:
        hit = False
        for f in companions:
            if f in r["parsed"] and not in_bounds(f, r["parsed"][f], bounds):
                per_field[f] += 1
                hit = True
        if hit:
            bad_rows += 1
    rate = bad_rows / n
    clean = n - bad_rows
    return {
        "rows": n,
        "companion_fields_tested": companions,
        "rows_with_corrupt_companion": bad_rows,
        "rate": round(rate, 4),
        "clean_subset_rows": clean,
        "clean_subset_pct": round(clean / n, 4),
        "by_companion_field": dict(per_field.most_common()),
        "interpretation": (
            "most of this cohort is corrupt row data, not weather; bias is "
            "measured on the clean subset only"
            if rate >= 0.5 else
            "mostly plausible weather, dropped for one bad field"),
    }


def trigger_fields_for_reason(reason: Optional[str]) -> Tuple[str, ...]:
    """Which field's failure is responsible for this reason label."""
    if reason and reason.startswith("weather_bounds_"):
        return (reason[len("weather_bounds_"):],)
    return ()


def max_abs_effect_size(bias: Dict[str, Any]) -> Optional[float]:
    vals = [abs(v["effect_size"]) for v in bias.values()
            if isinstance(v, dict) and v.get("effect_size") is not None]
    return round(max(vals), 4) if vals else None


def _direction_phrase(bias: Dict[str, Any]) -> str:
    """Plain-language direction of a shift, e.g. 'cooler and drier'."""
    warmer = cooler = wetter = drier = windier = calmer = False
    for f, v in bias.items():
        e = v.get("effect_size") if isinstance(v, dict) else None
        if e is None or abs(e) < MATERIAL_EFFECT_SIZE:
            continue
        if f == "temperature":
            warmer, cooler = e > 0, e < 0
        elif f == "humidity":
            wetter, drier = e > 0, e < 0
        elif f == "wind_speed":
            windier, calmer = e > 0, e < 0
    words = []
    if cooler:
        words.append("cooler")
    if warmer:
        words.append("warmer")
    if drier:
        words.append("drier")
    if wetter:
        words.append("wetter")
    if windier:
        words.append("windier")
    if calmer:
        words.append("calmer")
    if not words:
        return "shifted on %s" % (", ".join(
            f for f, v in bias.items()
            if isinstance(v, dict) and v.get("effect_size") is not None) or "covariates")
    if len(words) > 1:
        return ", ".join(words[:-1]) + " and " + words[-1]
    return words[0]


def bias_verdict(effect: Optional[float], clean_subset_rows: int = None) -> str:
    if clean_subset_rows is not None and clean_subset_rows < MIN_BIAS_COHORT:
        return "undetermined_cohort_too_small"
    if effect is None:
        return "undetermined"
    if effect >= MATERIAL_EFFECT_SIZE:
        return "material_selection_bias"
    if effect <= NEGLIGIBLE_EFFECT_SIZE:
        return "no_material_selection_bias"
    return "minor_selection_bias"


# ---------------------------------------------------------------------------
# Pressure bounds investigation
# ---------------------------------------------------------------------------

def diagnose_pressure_cohort(values: Sequence[float],
                             retained_stats: Optional[Dict[str, Any]] = None
                             ) -> Dict[str, Any]:
    """Classify a station's out-of-bounds pressure values.

    ``frozen_sensor``  one value repeated. Weather never repeats a reading to
                       the hPa; a transducer or its transport has stopped.
    ``datum_offset``   a tight band, far below the station's own level, with
                       real sub-hourly variability. This is a station-pressure
                       versus sea-level-pressure error: recoverable with a
                       calibration offset instead of a drop.
    ``corrupt``        magnitudes that no atmosphere produces, and scattered
                       rather than banded.

    ``retained_stats`` must be the same station's IN-BOUNDS pressure summary;
    it is what makes the offset measurable.
    """
    vals = [float(v) for v in values if v is not None]
    n = len(vals)
    stats = describe(vals)
    distinct = len({round(v, 6) for v in vals})
    top_value, top_count = (None, 0)
    if vals:
        top_value, top_count = Counter(vals).most_common(1)[0]
    frozen_share = (top_count / n) if n else 0.0

    # Sub-hourly variability: is anything moving at all?
    deltas = [abs(vals[i + 1] - vals[i]) for i in range(len(vals) - 1)]
    step_sd = None
    if len(deltas) >= 2:
        dm = sum(deltas) / len(deltas)
        step_sd = math.sqrt(sum((d - dm) ** 2 for d in deltas) / len(deltas))

    ret = retained_stats or {}
    ret_median = ret.get("p50")
    ret_p1, ret_p99 = ret.get("p1"), ret.get("p99")
    ret_range = (ret_p99 - ret_p1) if (ret_p1 is not None and ret_p99 is not None) else None
    cohort_iqr = iqr(vals)
    offset = (stats["p50"] - ret_median) if (ret_median is not None
                                             and stats["p50"] is not None) else None
    offset_in_ret_iqrs = None
    if offset is not None and ret_range:
        offset_in_ret_iqrs = round(offset / ret_range, 3) if ret_range else None

    alt = implied_altitude_m(stats["p50"]) if stats["p50"] else None
    ret_alt = implied_altitude_m(ret_median) if ret_median else None
    alt_delta = (alt - ret_alt) if (alt is not None and ret_alt is not None) else None

    out: Dict[str, Any] = {
        "rows": n,
        "distribution": stats,
        "distinct_values": distinct,
        "modal_value": top_value,
        "modal_value_share": round(frozen_share, 4),
        "abs_step_sd": round(step_sd, 4) if step_sd is not None else None,
        "station_retained_median": ret_median,
        "median_offset_from_retained_hpa": round(offset, 3) if offset is not None else None,
        "median_offset_in_retained_ranges": offset_in_ret_iqrs,
        "implied_altitude_m": round(alt, 1) if alt is not None else None,
        "implied_altitude_delta_m": round(alt_delta, 1) if alt_delta is not None else None,
    }

    if n >= FROZEN_MIN_COHORT and frozen_share >= FROZEN_SHARE_MIN:
        out["verdict"] = "frozen_sensor"
        out["explanation"] = (
            "%d of %d rejected readings (%.1f%%) are the single value %r. "
            "A barometer does not repeat a value to the hPa; the channel is "
            "stuck or its transport has stopped. Unrecoverable from data."
            % (top_count, n, 100.0 * frozen_share, top_value))
    elif (n >= DATUM_MIN_COHORT and cohort_iqr is not None and ret_range
          and cohort_iqr <= DATUM_TIGHT_IQR_FRACTION * ret_range
          and offset is not None and offset_in_ret_iqrs is not None
          and abs(offset_in_ret_iqrs) >= DATUM_MIN_OFFSET_IQRS):
        out["verdict"] = "datum_offset"
        out["explanation"] = (
            "Rejected values form a tight band (IQR %.2f hPa, %.1f%% of the "
            "station's own in-bounds range) offset %+.1f hPa from its normal "
            "level, with sub-hourly movement of sd %.3f hPa. A systematic offset "
            "with structure preserved is a datum error -- the station is "
            "reporting station pressure instead of sea-level pressure. "
            "Equivalent to %+.0f m of altitude. Recoverable with a calibration "
            "offset; dropping these rows is the wrong remedy."
            % (cohort_iqr, 100.0 * DATUM_TIGHT_IQR_FRACTION, offset,
               step_sd or 0.0, alt_delta or 0.0))
    else:
        out["verdict"] = "corrupt"
        out["explanation"] = (
            "Rejected values are scattered across %d distinct magnitudes with "
            "no band structure (modal share %.1f%%). Consistent with corrupt "
            "rows or a transport fault, not with a calibration offset."
            % (distinct, 100.0 * frozen_share))

    out["recoverable_by_calibration"] = out["verdict"] == "datum_offset"
    return out


def pressure_population_census(per_station: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-station pressure diagnoses into a population census."""
    by_verdict: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"rows": 0, "stations": []})
    for st, d in sorted(per_station.items()):
        v = by_verdict[d["verdict"]]
        v["rows"] += d["rows"]
        v["stations"].append({
            "station_id": st,
            "rows": d["rows"],
            "median_offset_from_retained_hpa": d["median_offset_from_retained_hpa"],
            "modal_value": d["modal_value"],
            "modal_value_share": d["modal_value_share"],
        })
    total = sum(v["rows"] for v in by_verdict.values())
    return {
        "total_rejected_rows": total,
        "populations": {
            k: {
                "rows": v["rows"],
                "pct_of_rejected": _pct(v["rows"], total),
                "stations": sorted(v["stations"], key=lambda s: -s["rows"]),
            }
            for k, v in sorted(by_verdict.items(), key=lambda kv: -kv[1]["rows"])
        },
    }


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

def build_report(csv_path: str = DEFAULT_CSV,
                 bounds: Dict[str, Tuple[float, float]] = None,
                 window_hours: float = 72.0,
                 git_commit: Optional[str] = None,
                 generated_at: Optional[str] = None) -> Dict[str, Any]:
    """Build the full report. Pure with respect to the filesystem except reading."""
    bounds = bounds or PHYSICAL_BOUNDS
    rows = read_rows(csv_path)

    retained = [r for r in rows if r["reason"] is None]
    quarantined = [r for r in rows if r["reason"] is not None]

    station_names: Dict[str, str] = {}
    station_rows: Dict[str, List[TelemetryRow]] = defaultdict(list)
    for r in rows:
        if r["station_id"]:
            station_rows[r["station_id"]].append(r)
            if r["station_name"]:
                station_names.setdefault(r["station_id"], r["station_name"])

    retained_by_station: Dict[str, List[TelemetryRow]] = {
        st: [r for r in rs if r["reason"] is None]
        for st, rs in station_rows.items()
    }

    report: Dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "code_commit": git_commit if git_commit is not None else git_head_commit(
            os.path.dirname(SRC_DIR)),
        "provenance": {
            # Basename only: prediction-model/data/*.json is scanned by the
            # path-hygiene gate for machine-specific absolute paths.
            "source_file": os.path.basename(csv_path),
            "source_sha256": compute_file_sha256(csv_path),
            "rows_read": len(rows),
            "classifies_in_priority_order": [
                "timestamp parse", "year bounds", "station_id present",
                "field completeness (all 7 required)", "physical bounds "
                "(checked in REQUIRED_FIELDS order)",
            ],
            "bounds_source": "mirrors dataset.PHYSICAL_BOUNDS; "
                             "test_data_quality.py asserts they agree",
        },
        "physical_bounds": {k: list(v) for k, v in bounds.items()},
        "required_fields": list(REQUIRED_FIELDS),
    }

    report["row_accounting"] = _row_accounting(rows, retained, quarantined, bounds)
    report["quarantine_detail"] = _quarantine_detail(
        rows, quarantined, station_rows, station_names)
    report["pressure_bounds_investigation"] = _pressure_investigation(
        rows, station_rows, station_names)
    report["completeness_by_station"] = _completeness(
        rows, station_rows, station_names, bounds)
    report["distributions_by_station"] = _distributions(
        rows, station_rows, station_names, bounds)
    report["continuity_by_station"] = _continuity(
        rows, station_rows, station_names, bounds)
    report["training_impact"] = _training_impact(
        rows, retained, quarantined, retained_by_station, window_hours, bounds)
    report["impact_ranking"] = _impact_ranking(
        report, rows, station_rows, station_names)
    report["data_health_verdict"] = _health_verdict(
        report, rows, station_rows, station_names)
    report["defects_reported_not_fixed"] = _defects(rows, report)
    return report


def _row_accounting(rows, retained, quarantined, bounds) -> Dict[str, Any]:
    n_in = len(rows)
    counts = Counter(r["reason"] for r in quarantined)
    # dataset.py also tallies weather_missing_<field> per missing field within
    # the same row, so those sums exceed weather_missing_field. Say so, rather
    # than let a reader add them up wrongly.
    field_tally = Counter()
    for r in quarantined:
        if r["reason"] == "weather_missing_field":
            for f in r["missing_fields"]:
                field_tally["weather_missing_" + f] += 1

    by_reason = {}
    for reason, n in counts.most_common():
        by_reason[reason] = {
            "rows": n,
            "pct_of_rows_in": _pct(n, n_in),
            "pct_of_quarantined": _pct(n, len(quarantined)),
            "reason_class": REASON_CLASS.get(reason, "other"),
        }

    return {
        "rows_in": n_in,
        "rows_ingested": len(retained),
        "rows_quarantined": len(quarantined),
        "ingested_pct": _pct(len(retained), n_in),
        "quarantined_pct": _pct(len(quarantined), n_in),
        "by_reason": by_reason,
        "field_level_missingness": {
            "note": "Counts are per missing field within the "
                    "weather_missing_field rows. A row missing three fields "
                    "contributes to three entries, so these DO NOT sum to "
                    "rows_quarantined. This mirrors dataset.py, which "
                    "increments one counter per absent field.",
            "rows_with_at_least_one_missing_field":
                by_reason.get("weather_missing_field", {}).get("rows", 0),
            "by_field": {
                k: {"field_occurrences": v,
                    "pct_of_rows_in": _pct(v, n_in),
                    "pct_of_missing_field_rows": _pct(
                        v, by_reason.get("weather_missing_field", {}).get("rows", 0))}
                for k, v in field_tally.most_common()
            },
        },
        "rows_with_unreported_bounds_violations": {
            "note": "Rows the reason label attributes to completeness but which "
                    "ALSO carry at least one out-of-bounds value in a field "
                    "that IS present. dataset.py tests completeness first and "
                    "then returns, so the physical-bounds counters never see "
                    "them and the reason understates how corrupt these rows "
                    "are.",
            "rows": sum(1 for r in quarantined
                        if r["unreported_bounds_violations"]),
            "pct_of_missing_field_rows": _pct(
                sum(1 for r in quarantined if r["unreported_bounds_violations"]),
                by_reason.get("weather_missing_field", {}).get("rows", 0)),
            "by_field": dict(Counter(
                f for r in quarantined for f in r["unreported_bounds_violations"]
            ).most_common()),
        },
    }


def _quarantine_detail(rows, quarantined, station_rows, station_names) -> Dict[str, Any]:
    by_reason: Dict[str, List[TelemetryRow]] = defaultdict(list)
    for r in quarantined:
        by_reason[r["reason"]].append(r)

    total_rows = len(rows)
    detail = {}
    for reason, sel in sorted(by_reason.items(), key=lambda kv: -len(kv[1])):
        per_station = {}
        station_counter = Counter(r["station_id"] for r in sel)
        for st, n in station_counter.most_common():
            st_rows = [r for r in sel if r["station_id"] == st]
            cluster = classify_clustering([r["timestamp"] for r in st_rows])
            cluster["rows"] = n
            cluster["pct_of_station_rows"] = _pct(n, len(station_rows.get(st, [])))
            cluster["pct_of_corpus"] = _pct(n, total_rows)
            per_station[st] = cluster

        clustered_rows = sum(c["rows"] for c in per_station.values()
                             if c["verdict"] == "clustered_outage")
        fleet = classify_clustering([r["timestamp"] for r in sel])

        hod = Counter((r["timestamp"].hour + LOCAL_UTC_OFFSET_HOURS) % 24
                      for r in sel if r["timestamp"])
        hod_total = sum(hod.values())
        top_hod = hod.most_common(3)

        detail[reason] = {
            "rows": len(sel),
            "pct_of_rows_in": _pct(len(sel), total_rows),
            "pct_of_quarantined": _pct(len(sel), len(quarantined)),
            "reason_class": REASON_CLASS.get(reason, "other"),
            "stations_affected_count": len(station_counter),
            "stations_affected": [
                {"station_id": st,
                 "station_name": station_names.get(st),
                 "rows": n,
                 "pct_of_station_rows": _pct(n, len(station_rows.get(st, []))),
                 "verdict": per_station[st]["verdict"],
                 "longest_run_hours": per_station[st]["longest_run_hours"]}
                for st, n in station_counter.most_common()
            ],
            "fleet_time_range": {"first": fleet["first"], "last": fleet["last"],
                                 "span_hours": fleet["span_hours"]},
            "dominant_pattern": ("clustered_outage"
                                 if per_station
                                 and clustered_rows >= 0.5 * len(sel)
                                 else "spread"),
            "pct_of_rows_in_clustered_cohorts": _pct(clustered_rows, len(sel)),
            "clustering_evidence": {
                "n_station_cohorts": len(per_station),
                "n_cohorts_clustered": sum(
                    1 for c in per_station.values()
                    if c["verdict"] == "clustered_outage"),
                "longest_run_hours_fleet_wide": fleet["longest_run_hours"],
                "threshold_hours": OUTAGE_RUN_MIN_HOURS,
                "n_distinct_fleet_wide_runs": fleet["n_runs"],
            },
            "hour_of_day_pht": {
                "note": "A reason concentrated in a few local hours points at a "
                        "diurnal sensor or schedule fault; a flat profile points "
                        "at an outage. Under an even 24 h feed each hour holds "
                        "%.2f%% of the rows." % (100.0 / 24.0),
                "total": hod_total,
                "peak_hour_pht": hod.most_common(1)[0][0] if hod else None,
                "peak_hour_share_pct": _pct(
                    max(hod.values()) if hod else 0, hod_total),
                "peak_hour_uniform_multiple": (
                    round((max(hod.values()) / (hod_total / 24.0)), 2)
                    if hod and hod_total else None),
                "top_hours": [{"hour_pht": h, "rows": c,
                               "pct": _pct(c, hod_total)}
                              for h, c in top_hod],
                "verdict": (
                    "diurnal_concentrated"
                    if hod and hod_total and max(hod.values()) > 2.0 * hod_total / 24.0
                    else "flat"),
            },
            "per_station": per_station,
        }
    return detail


def _pressure_investigation(rows, station_rows, station_names) -> Dict[str, Any]:
    reason = "weather_bounds_pressure"
    sel = [r for r in rows if r["reason"] == reason]
    if not sel:
        return {"status": "no_pressure_bounds_rejections_in_this_corpus",
                "rows": 0}

    per_station = {}
    for st in sorted({r["station_id"] for r in sel if r["station_id"]}):
        vals = [r["parsed"]["pressure"] for r in sel
                if r["station_id"] == st and "pressure" in r["parsed"]]
        retained_p = [r["parsed"]["pressure"] for r in station_rows.get(st, [])
                      if r["reason"] is None and "pressure" in r["parsed"]]
        d = diagnose_pressure_cohort(vals, describe(retained_p))
        d["station_name"] = station_names.get(st)
        d["clustering"] = classify_clustering(
            [r["timestamp"] for r in sel if r["station_id"] == st])
        d["pct_of_station_rows"] = _pct(len(vals), len(station_rows.get(st, [])))
        per_station[st] = d

    census = pressure_population_census(per_station)
    all_vals = sorted(r["parsed"]["pressure"] for r in sel
                      if "pressure" in r["parsed"])
    lo, hi = PHYSICAL_BOUNDS["pressure"]
    fleet_cluster = classify_clustering([r["timestamp"] for r in sel])

    recoverable = census["populations"].get("datum_offset", {}).get("rows", 0)
    conclusions = []
    if recoverable:
        conclusions.append(
            "%d of %d pressure rejections (%.1f%%) are a DATUM OFFSET, not a "
            "sensor fault: a tight band %.0f hPa below the station's normal "
            "level, equivalent to about %.0f m of altitude, with its "
            "sub-hourly structure intact. The remedy is a per-station "
            "calibration offset, which would recover those rows; the remedy "
            "actually applied -- dropping them -- discards them."
            % (recoverable, len(sel), 100.0 * recoverable / len(sel),
               _median_offset(per_station, "datum_offset"),
               _median_alt_delta(per_station, "datum_offset")))
    frozen = census["populations"].get("frozen_sensor", {}).get("rows", 0)
    if frozen:
        conclusions.append(
            "%d rejections are a FROZEN SENSOR: one value repeated verbatim. "
            "That is a dead or disconnected channel and cannot be recovered "
            "from the data; it needs a site visit." % frozen)
    corrupt = census["populations"].get("corrupt", {}).get("rows", 0)
    if corrupt:
        conclusions.append(
            "%d rejections are CORRUPT ROWS: magnitudes no atmosphere "
            "produces (negatives, 10^5-10^6 hPa, scattered 300-2000 hPa). "
            "Correctly dropped." % corrupt)
    if fleet_cluster["verdict"] == "clustered_outage":
        conclusions.append(
            "The rejections are CLUSTERED IN TIME, not scattered: the longest "
            "contiguous run is %d h across the fleet and %.1f%% of rejected "
            "rows sit in station cohorts that qualify as outages. This is a "
            "switching sensor/transport fault, not a noisy channel."
            % (fleet_cluster["longest_run_hours"],
               _pct(census["populations"].get("datum_offset", {}).get("rows", 0)
                    + census["populations"].get("frozen_sensor", {}).get("rows", 0),
                    len(sel))))

    return {
        "rows": len(sel),
        "pct_of_rows_in": _pct(len(sel), len(rows)),
        "bounds_applied": list(PHYSICAL_BOUNDS["pressure"]),
        "rejected_value_distribution": describe(all_vals),
        "rejected_value_percentiles": {
            str(p): percentile(all_vals, p) for p in (1, 5, 10, 25, 50, 75, 90, 99)
        },
        "fraction_below_850_hpa": _pct(sum(1 for v in all_vals if v < 850.0), len(all_vals)),
        "fraction_above_1050_hpa": _pct(sum(1 for v in all_vals if v > hi), len(all_vals)),
        "fleet_clustering": fleet_cluster,
        "population_census": census,
        "per_station": per_station,
        "diagnosis_conclusions": conclusions,
    }


def _median_offset(per_station, verdict) -> float:
    vals = [d["median_offset_from_retained_hpa"] for st, d in per_station.items()
            if d["verdict"] == verdict
            and d["median_offset_from_retained_hpa"] is not None]
    return median(vals) or 0.0


def _median_alt_delta(per_station, verdict) -> float:
    vals = [d["implied_altitude_delta_m"] for st, d in per_station.items()
            if d["verdict"] == verdict
            and d["implied_altitude_delta_m"] is not None]
    return median(vals) or 0.0


def _completeness(rows, station_rows, station_names, bounds) -> Dict[str, Any]:
    """Fraction of EXPECTED hourly bins that carry each required field.

    Expected bins are the station's own first-to-last hour, so a station that
    joined the fleet late is not penalised, but a station that dropped out for
    46 days is charged for every hour of that gap.
    """
    out = {}
    for st, rs in sorted(station_rows.items()):
        stamps = [r["timestamp"] for r in rs if r["timestamp"]]
        if not stamps:
            out[st] = {"station_name": station_names.get(st), "error": "no parseable timestamps"}
            continue
        span_bins = int((max(stamps) - min(stamps)).total_seconds() // 3600) + 1

        bins: Dict[datetime, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list))
        for r in rs:
            if r["timestamp"] is None:
                continue
            h = r["timestamp"].replace(minute=0, second=0, microsecond=0)
            for f in REQUIRED_FIELDS:
                if f in r["parsed"]:
                    bins[h][f].append(r["parsed"][f])

        fields = {}
        for f in REQUIRED_FIELDS:
            present = sum(1 for b in bins.values() if b.get(f))
            usable = sum(1 for b in bins.values()
                         if any(in_bounds(f, v, bounds) for v in b.get(f, ())))
            fields[f] = {
                "expected_bins": span_bins,
                "present_bins": present,
                "present_pct": _pct(present, span_bins),
                "usable_bins": usable,
                "usable_pct": _pct(usable, span_bins),
                "present_but_unusable_bins": present - usable,
            }
        out[st] = {
            "station_name": station_names.get(st),
            "rows_raw": len(rs),
            "expected_bins": span_bins,
            "observed_bins": len(bins),
            "bin_coverage_pct": _pct(len(bins), span_bins),
            "first_hour": _iso(min(bins)) if bins else None,
            "last_hour": _iso(max(bins)) if bins else None,
            "fields": fields,
            "all_seven_fields_usable_pct": _pct(
                sum(1 for b in bins.values()
                    if all(any(in_bounds(f, v, bounds) for v in b.get(f, ()))
                           for f in REQUIRED_FIELDS)),
                span_bins),
        }
    return out


def _distributions(rows, station_rows, station_names, bounds) -> Dict[str, Any]:
    out = {}
    for st, rs in sorted(station_rows.items()):
        fields = {}
        for f in REQUIRED_FIELDS:
            vals = [r["parsed"][f] for r in rs if f in r["parsed"]]
            d = describe(vals)
            oob = sum(1 for v in vals if not in_bounds(f, v, bounds))
            d["out_of_bounds_count"] = oob
            d["out_of_bounds_pct"] = _pct(oob, d["n"])
            d["out_of_bounds_pct_of_all_station_rows"] = _pct(oob, len(rs))
            d["missing_count"] = len(rs) - d["n"]
            d["missing_pct"] = _pct(len(rs) - d["n"], len(rs))
            fields[f] = d
        out[st] = {"station_name": station_names.get(st), "rows_raw": len(rs),
                   "fields": fields}
    return out


def _continuity(rows, station_rows, station_names, bounds) -> Dict[str, Any]:
    """Continuity on the grid the model actually trains on: in-bounds bins."""
    out = {}
    for st, rs in sorted(station_rows.items()):
        usable_hours, all_hours = set(), set()
        for r in rs:
            if r["timestamp"] is None:
                continue
            h = r["timestamp"].replace(minute=0, second=0, microsecond=0)
            all_hours.add(h)
            if r["reason"] is None:
                usable_hours.add(h)
        usable = continuity_metrics(sorted(usable_hours))
        usable["station_name"] = station_names.get(st)
        usable["raw_bins_observed"] = len(all_hours)
        usable["usable_bins"] = len(usable_hours)
        usable["usable_pct_of_raw_bins"] = _pct(len(usable_hours), len(all_hours))
        usable["ingested_vs_raw_pct"] = _pct(len(usable_hours), len(rs))
        out[st] = usable
    return out


BIAS_FIELDS = ("temperature", "humidity", "wind_speed", "precipitation", "pressure")


def _training_impact(rows, retained, quarantined, retained_by_station,
                     window_hours, bounds) -> Dict[str, Any]:
    by_reason: Dict[str, List[TelemetryRow]] = defaultdict(list)
    for r in quarantined:
        by_reason[r["reason"]].append(r)

    def field_stats(sel) -> Dict[str, Any]:
        return {f: describe([r["parsed"][f] for r in sel if f in r["parsed"]])
                for f in BIAS_FIELDS}

    out = {
        "method": {
            "question": "For each quarantine reason, does dropping these rows "
                        "move the distribution of the rows we kept?",
            "station_standardised": "z-score of each quarantined value against "
                                    "the same station's in-bounds history "
                                    "(mean, sd), clamped to +/-%g. Sensitive to "
                                    "the fact that faults are multi-day "
                                    "episodes." % EFFECT_SIZE_CAP,
            "time_matched": "each quarantined row against the median of "
                            "in-bounds rows at the SAME station within "
                            "+/- %g h. This is the figure to act on: it holds "
                            "the weather roughly fixed and asks whether the "
                            "rows that survived still describe it." % window_hours,
            "trigger_field_excluded": "The field whose failure caused the "
                                      "quarantine is excluded from the "
                                      "comparison. Including it is circular: a "
                                      "row rejected for reporting 845 hPa will "
                                      "always look wildly biased on pressure, "
                                      "which says nothing about the kept rows. "
                                      "Only the COMPANION fields can carry a "
                                      "selection effect.",
            "effect_size": "median paired delta divided by the IQR of the "
                           "matched control medians. IQR rather than SD because "
                           "one corrupt companion reading in a control window "
                           "would otherwise flatten every effect size. 0 means "
                           "no shift; >= %.1f is a material covariate shift."
                           % MATERIAL_EFFECT_SIZE,
            "companion_corruption": "Share of the cohort whose companion fields "
                                    "are ALSO out of physical bounds. Those rows "
                                    "were never weather, so the bias is measured "
                                    "on the clean subset and the corrupt share "
                                    "is disclosed separately rather than mixed "
                                    "into the same statistic.",
            "min_cohort_for_bias_claim": "A cohort needs at least %d weather "
                                         "rows before its effect size means "
                                         "anything. Below that the verdict is "
                                         "'undetermined_cohort_too_small' and "
                                         "the reason earns no bias credit in the "
                                         "impact ranking."
                                         % MIN_BIAS_COHORT,
            "caveat": "Correlation with the kept rows is not proof of causal "
                      "bias: a sensor that fails in the rain is failing "
                      "BECAUSE of the weather. Even so, the retained set is "
                      "not a random sample of the period, and a model fitted "
                      "on it under-represents those conditions at inference "
                      "time regardless of cause.",
        },
        "retained_reference": field_stats(retained),
        "by_reason": {},
    }

    for reason, sel in sorted(by_reason.items(), key=lambda kv: -len(kv[1])):
        trigger = trigger_fields_for_reason(reason)
        corruption = companion_corruption_rate(sel, trigger, bounds=bounds)
        # Measure the bias on the rows that are actually WEATHER. A row whose
        # companion fields are also out of bounds was never a weather sample,
        # so including it would mix two populations and report a spurious
        # shift. Its share is disclosed instead.
        clean = [r for r in sel if not any(
            f in r["parsed"] and not in_bounds(f, r["parsed"][f], bounds)
            for f in corruption["companion_fields_tested"])]
        tm = time_matched_bias(clean, retained_by_station, fields=BIAS_FIELDS,
                               window_hours=window_hours,
                               exclude_fields=trigger, bounds=bounds)
        ss = station_standardised_bias(clean, retained_by_station,
                                       fields=BIAS_FIELDS,
                                       exclude_fields=trigger)
        effect = max_abs_effect_size(tm)
        verdict = bias_verdict(effect, len(clean))
        out["by_reason"][reason] = {
            "rows": len(sel),
            "pct_of_rows_in": _pct(len(sel), len(rows)),
            "trigger_field": trigger[0] if trigger else None,
            "companion_corruption": corruption,
            "bias_measured_on_rows": len(clean),
            "quarantined": field_stats(sel),
            "time_matched": tm,
            "station_standardised": ss,
            "max_abs_effect_size": effect,
            "max_abs_effect_size_supports_claim":
                verdict in ("material_selection_bias", "minor_selection_bias"),
            "bias_verdict": verdict,
            "bias_direction": _direction_phrase(tm) if effect is not None else None,
            "biased_fields": sorted(
                f for f, v in tm.items()
                if isinstance(v, dict) and v.get("effect_size") is not None
                and abs(v["effect_size"]) >= MATERIAL_EFFECT_SIZE),
        }
    return out


def _impact_ranking(report, rows, station_rows, station_names) -> List[Dict[str, Any]]:
    """Rank problems by IMPACT, not by row count.

    score = 100 * (0.35*volume + 0.35*bias + 0.15*shape + 0.15*recoverable)

    volume      share of the corpus lost, saturating at 15%.
    bias        worst time-matched effect size, saturating at 3.0.
    shape       1.0 when the loss is outage-clustered, 0.0 when spread. A
                clustered loss is worse than an equal scattered one because it
                removes contiguous history, which costs whole 24 h windows and
                therefore far more training rows than the count suggests.
    recoverable 1.0 when a diagnosis says a calibration offset would win the
                rows back. Recoverable damage is high-impact and cheap.

    The weights live in module constants and are echoed into the report so the
    ranking can be re-derived rather than trusted.
    """
    accounting = report["row_accounting"]
    detail = report["quarantine_detail"]
    impact = report["training_impact"]
    pressure = report["pressure_bounds_investigation"]
    census = pressure.get("population_census", {}).get("populations", {})
    recoverable_rows = sum(
        d["rows"] for d in pressure.get("per_station", {}).values()
        if d.get("recoverable_by_calibration"))

    findings = []
    for reason, d in detail.items():
        entry = impact["by_reason"].get(reason, {})
        effect = entry.get("max_abs_effect_size")
        verdict = entry.get("bias_verdict", "undetermined")
        volume = min(1.0, d["pct_of_rows_in"] / IMPACT_VOLUME_SATURATION_PCT)
        # Only a verdict that actually claims a bias earns bias credit. An
        # under-determined cohort and a correctly-dropped corrupt row both
        # score zero, so 10 garbage rows cannot outrank a 4,000-row outage.
        if verdict in ("material_selection_bias", "minor_selection_bias"):
            bias = min(1.0, (effect or 0.0) / IMPACT_BIAS_SATURATION)
        else:
            bias = 0.0
        shape = 1.0 if d["dominant_pattern"] == "clustered_outage" else 0.0
        rec = 0.0
        if reason == "weather_bounds_pressure":
            rec = _pct(recoverable_rows, d["rows"]) / 100.0
        score = round(100.0 * (IMPACT_W_VOLUME * volume + IMPACT_W_BIAS * bias
                               + IMPACT_W_SHAPE * shape
                               + IMPACT_W_RECOVERABLE * rec), 1)
        findings.append({
            "problem": reason,
            "reason_class": d["reason_class"],
            "rows": d["rows"],
            "pct_of_rows_in": d["pct_of_rows_in"],
            "stations_affected": d["stations_affected_count"],
            "dominant_pattern": d["dominant_pattern"],
            "bias_verdict": verdict,
            "bias_measured_on_rows": entry.get("bias_measured_on_rows"),
            "companion_corruption_rate":
                entry.get("companion_corruption", {}).get("rate"),
            "max_abs_effect_size": effect,
            "biased_fields": entry.get("biased_fields", []),
            "recoverable_rows_if_calibrated": recoverable_rows
            if reason == "weather_bounds_pressure" else 0,
            "impact_components": {
                "volume": round(volume, 4), "bias": round(bias, 4),
                "outage_clustered": shape, "recoverable": round(rec, 4),
            },
            "impact_score": score,
            "recommended_action": _recommend(
                reason, d, entry, pressure, census, recoverable_rows),
        })

    findings.sort(key=lambda f: -f["impact_score"])
    for i, f in enumerate(findings, 1):
        f["rank"] = i
    report["impact_method"] = {
        "weights": {"volume": IMPACT_W_VOLUME, "bias": IMPACT_W_BIAS,
                    "outage_clustered": IMPACT_W_SHAPE,
                    "recoverable": IMPACT_W_RECOVERABLE},
        "volume_saturation_pct": IMPACT_VOLUME_SATURATION_PCT,
        "bias_saturation_effect_size": IMPACT_BIAS_SATURATION,
        "note": "Ranked by consequence, not by row count. A small loss that "
                "biases the training set outranks a large loss that does not.",
    }
    return findings


def _recommend(reason, detail, impact_entry, pressure, census, recoverable_rows):
    clustered = detail["dominant_pattern"] == "clustered_outage"
    bias = impact_entry.get("bias_verdict", "undetermined")
    worst_station = detail["stations_affected"][0] if detail["stations_affected"] else {}

    if reason == "weather_bounds_pressure":
        pop = census.get("datum_offset", {}).get("rows", 0)
        if pop:
            return (
                "CALIBRATE, DO NOT DROP. Derive a per-station pressure offset "
                "for the datum-offset stations (start with %s at %+.1f hPa) and "
                "re-ingest those %d rows; the sub-hourly structure is intact so "
                "the corrected series is usable. Confirm against the station's "
                "elevation metadata, since the whole diagnosis is that it is "
                "reporting station pressure rather than sea-level pressure. "
                "THEN, separately, raise a site visit for the frozen-sensor "
                "stations: no data-side fix exists for a channel that reports "
                "one value forever."
                % (worst_station.get("station_id"),
                   _median_offset(pressure.get("per_station", {}), "datum_offset"),
                   pop))
        return ("Site visit: the rejected values are frozen or corrupt, not "
                "recoverable from data.")
    if reason == "weather_missing_field":
        if clustered:
            return ("Availability problem, not a bounds problem, and NOT a "
                    "Recover the channel where the vendor still logs the raw "
                    "payload; until then, disclose the gap in every training "
                    "report rather than reporting a clean row count. A "
                    "%d-hour contiguous run also destroys every 24 h window "
                    "that touches it, so the real training loss is much "
                    "larger than the row count."
                    % (detail["clustering_evidence"]["longest_run_hours_fleet_wide"]))
        return "Investigate per-field ingestion for this station."
    if bias == "undetermined_cohort_too_small":
        return ("Too small to matter: %d rows, of which only %d are weather "
                "rather than corrupt. Correctly dropped; fix the ingestor and "
                "stop counting it as a bias."
                % (detail["rows"],
                   impact_entry.get("bias_measured_on_rows", 0)))
    if (impact_entry.get("companion_corruption", {}).get("rate") or 0) >= 0.5:
        return ("Mostly corrupt rows, correctly dropped: the companion fields "
                "are out of bounds too, so these were never weather. No "
                "selection bias to disclose. The upstream ingestor still needs "
                "fixing -- it is emitting rows with garbage in most fields at "
                "once, which is a transport fault, not a sensor one.")
    if bias == "material_selection_bias":
        return ("Disclose the bias. The kept set is not a random sample of the "
                "corpus (%s). Either add a selection weight, or restrict the "
                "scored period to a window where the fault is inactive, or "
                "keep the rows and fix the sensor. Do not report a validation "
                "score on this split without saying what was removed."
                % ", ".join(impact_entry.get("biased_fields", []) or ["covariates"]))
    if bias == "no_material_selection_bias":
        return ("Volume problem, not a bias problem: the kept rows describe "
                "the same weather as the dropped ones. Reduce it by recovering "
                "coverage, and disclose the reduction.")
    return "Investigate."


def _health_verdict(report, rows, station_rows, station_names) -> Dict[str, Any]:
    accounting = report["row_accounting"]
    impact = report["training_impact"]
    loss = accounting["quarantined_pct"]

    material = [(r, e) for r, e in impact["by_reason"].items()
                if e["bias_verdict"] == "material_selection_bias"]
    minor = [(r, e) for r, e in impact["by_reason"].items()
             if e["bias_verdict"] == "minor_selection_bias"]
    corrupt = [(r, e) for r, e in impact["by_reason"].items()
               if (e.get("companion_corruption", {}).get("rate") or 0) >= 0.5]
    too_small = [(r, e) for r, e in impact["by_reason"].items()
                 if e["bias_verdict"] == "undetermined_cohort_too_small"]
    clustered = [r for r, d in report["quarantine_detail"].items()
                 if d["dominant_pattern"] == "clustered_outage"]
    pressure = report["pressure_bounds_investigation"]
    recoverable = sum(d["rows"] for d in pressure.get("per_station", {}).values()
                      if d.get("recoverable_by_calibration"))

    reasons: List[str] = []
    if loss > GOOD_LOSS_MAX_PCT:
        reasons.append(
            "%.2f%% of the corpus (%d of %d rows) is quarantined before it "
            "reaches the model, against a %.1f%% budget for a clean feed."
            % (loss, accounting["rows_quarantined"], accounting["rows_in"],
               GOOD_LOSS_MAX_PCT))
    for r, e in sorted(material, key=lambda kv: -(kv[1]["max_abs_effect_size"] or 0)):
        reasons.append(
            "DROPPING %r IS BIASED. Its %d weather rows are %s than the hours "
            "around them at the same station (worst time-matched effect size "
            "%+.2f x control IQR on %s), and they are removed. The retained "
            "set is therefore systematically %s than the period it was drawn "
            "from."
            % (r, e.get("bias_measured_on_rows", 0),
               e.get("bias_direction") or "shifted",
               e["max_abs_effect_size"],
               ", ".join(e["biased_fields"]) or "companion covariates",
               e.get("bias_direction") or "shifted"))
    for r, e in sorted(minor, key=lambda kv: -(kv[1]["max_abs_effect_size"] or 0)):
        reasons.append(
            "Dropping %r shifts the retained set by a small but non-zero "
            "amount (worst effect size %+.2f on %s). Worth disclosing; not "
            "disqualifying." % (r, e["max_abs_effect_size"],
                                ", ".join(e["biased_fields"]) or "companions"))
    if corrupt:
        reasons.append(
            "%d quarantine reason(s) (%s) are mostly corrupt row data rather "
            "than weather -- their companion fields are out of bounds too. "
            "Those drops are correct and are not counted as a selection bias; "
            "quoting their raw effect sizes would overstate the problem."
            % (len(corrupt), ", ".join("'%s'" % r for r, _ in corrupt)))
    if too_small:
        reasons.append(
            "%d reason(s) are too small to characterise against a minimum of %d "
            "weather rows and earn no bias credit either way: %s."
            % (len(too_small), MIN_BIAS_COHORT,
               ", ".join("'%s' (%d rows, %d weather)"
                         % (r, e["rows"], e.get("bias_measured_on_rows", 0))
                         for r, e in sorted(too_small, key=lambda kv: -kv[1]["rows"]))))
    if clustered:
        reasons.append(
            "%d of %d quarantine reasons are outage-clustered rather than "
            "spread: the loss is contiguous, so the 24 h windows straddling "
            "each episode are lost too and the true training loss exceeds the "
            "row count." % (len(clustered),
                            len(report["quarantine_detail"])))
    if recoverable:
        reasons.append(
            "%d pressure rejections are a recoverable datum offset being "
            "discarded as if they were garbage. The current remedy is the "
            "expensive one." % recoverable)
    worst = _worst_continuity(report)
    if worst:
        reasons.append(
            "Station %s has a %g-hour hole in its ingested hourly grid (%s). "
            "No forecast window can be built across it."
            % (worst["station_id"], worst["longest_gap_hours"],
               worst["station_name"] or worst["station_id"]))

    if loss > FAIR_LOSS_MAX_PCT or material:
        verdict = "POOR"
    elif loss > GOOD_LOSS_MAX_PCT or minor or clustered:
        verdict = "FAIR"
    else:
        verdict = "GOOD"

    # The direction of the shift has to be derived from the measurement, not
    # remembered. A hardcoded "warmer and drier" is exactly the kind of claim
    # that goes stale the moment the corpus changes.
    retained_shape = _retained_shape_phrase(impact)
    summary = (
        "POOR: %.1f%% of rows are quarantined and the retained set is not a "
        "random sample of the corpus. A model fitted on it has inherited the "
        "selection bias of the filter, and every score computed on it is a "
        "score on a %s period than the fleet actually experiences. %d rows are "
        "additionally recoverable with a calibration offset that has not been "
        "applied." % (loss, retained_shape, recoverable)
    ) if verdict == "POOR" else (
        "FAIR: the retained set is broadly representative but %.1f%% of rows "
        "are quarantined, mostly in contiguous outage episodes." % loss)

    return {
        "verdict": verdict,
        "scale": {
            "GOOD": "quarantined <= %.1f%% and no bias above the negligible "
                    "threshold and no outage-clustered reason"
                    % GOOD_LOSS_MAX_PCT,
            "FAIR": "quarantined <= %.1f%%, or only minor bias, or only "
                    "outage-clustered loss" % FAIR_LOSS_MAX_PCT,
            "POOR": "quarantined > %.1f%%, or at least one material selection "
                    "bias" % FAIR_LOSS_MAX_PCT,
        },
        "quarantined_pct": loss,
        "reasons": reasons,
        "summary": summary,
        "disclosure": "Any performance figure computed on this corpus is a "
                      "figure on the RETAINED set and inherits the biases "
                      "listed above. It is not a figure on the fleet.",
        "worst_continuity": worst,
    }


def _retained_shape_phrase(impact: Dict[str, Any]) -> str:
    """Describe the retained set by INVERTING what was dropped.

    The bias is measured on the dropped cohort, but the consequence is for the
    set we kept. If the rows we threw away were cooler and drier, the rows we
    kept are warmer and more humid -- and saying it the other way round is the
    kind of sign error that survives review because it sounds plausible.
    """
    moved: Dict[str, float] = {}
    for entry in impact.get("by_reason", {}).values():
        if entry.get("bias_verdict") != "material_selection_bias":
            continue
        for f, v in entry.get("time_matched", {}).items():
            e = v.get("effect_size") if isinstance(v, dict) else None
            if e is None or abs(e) < MATERIAL_EFFECT_SIZE:
                continue
            # Row counts are small relative to the loss, so weight by rows but
            # cap the influence of any one reason.
            moved[f] = moved.get(f, 0.0) + e * min(1.0, entry.get("rows", 0) / 200.0)
    if not moved:
        return "no systematically different"
    warmer = moved.get("temperature", 0) < 0
    wetter = moved.get("humidity", 0) < 0
    windier = moved.get("wind_speed", 0) < 0
    words = []
    if warmer:
        words.append("warmer")
    if wetter:
        words.append("more humid")
    if windier:
        words.append("windier")
    if not words:
        return "no systematically different"
    if len(words) == 1:
        return words[0]
    return ", ".join(words[:-1]) + " and " + words[-1]


def _worst_continuity(report) -> Optional[Dict[str, Any]]:
    cont = report.get("continuity_by_station", {})
    candidates = [c for c in cont.values()
                  if c.get("longest_gap_hours", 0) > 0]
    if not candidates:
        return None
    w = max(candidates, key=lambda c: c["longest_gap_hours"])
    for st, c in cont.items():
        if c is w:
            return {"station_id": st, "station_name": c.get("station_name"),
                    "longest_gap_hours": c.get("longest_gap_hours"),
                    "missing_bins": c.get("missing_bins")}
    return None


def _defects(rows, report) -> List[Dict[str, Any]]:
    """Defects found in the pipeline. Reported, deliberately NOT fixed here."""
    out: List[Dict[str, Any]] = []
    hidden = report["row_accounting"]["rows_with_unreported_bounds_violations"]
    if hidden["rows"]:
        out.append({
            "file": "prediction-model/src/dataset.py",
            "severity": "medium",
            "what": "Completeness is tested before physical bounds and the row "
                    "is then dropped, so a row that is both incomplete and out "
                    "of bounds is counted only as weather_missing_field.",
            "evidence": "%d of the %d weather_missing_field rows (%s) also "
                        "carry an out-of-bounds value in a field that IS "
                        "present: %s. Those violations are never counted by "
                        "any weather_bounds_* counter."
                        % (hidden["rows"],
                           report["row_accounting"]["by_reason"].get(
                               "weather_missing_field", {}).get("rows", 0),
                           "%.1f%%" % hidden["pct_of_missing_field_rows"],
                           ", ".join("%s=%d" % kv for kv in
                                     hidden["by_field"].items())),
            "consequence": "The physical-bounds counters under-report, and the "
                           "remedy is chosen from the wrong bucket: a corrupt "
                           "row is filed as a coverage gap, so an operator "
                           "chases a sensor outage that does not exist.",
            "fix_not_applied": "Reorder the checks, or record BOTH reasons in "
                               "one pass, and publish the overlap explicitly. "
                               "NOT changed here: dataset.py is out of scope for "
                               "this report.",
        })
    out.append({
        "file": "prediction-model/src/dataset.py",
        "severity": "high",
        "what": "Quarantine counters are emitted into "
                "data_quality_report.json and are not read by any gate.",
        "evidence": "%d rows (%s%%) are quarantined from the current corpus "
                    "with no operator-visible consequence."
                    % (report["row_accounting"]["rows_quarantined"],
                       report["row_accounting"]["quarantined_pct"]),
        "consequence": "A model can be trained on a deliberately biased subset "
                       "and reported as if it were trained on the corpus.",
        "fix_not_applied": "Gate training on this report's verdict, and require "
                           "the bias section to be disclosed in the model card.",
    })
    return out


# ---------------------------------------------------------------------------
# Operator summary
# ---------------------------------------------------------------------------

def print_summary(report: Dict[str, Any], stream=None) -> None:
    w = stream.write if stream else print
    a = report["row_accounting"]
    v = report["data_health_verdict"]
    w("\n" + "=" * 92)
    w("DATA QUALITY REPORT -- %s" % report["provenance"]["source_file"])
    w("=" * 92)
    w("rows in %d | ingested %d (%.2f%%) | quarantined %d (%.2f%%)"
      % (a["rows_in"], a["rows_ingested"], a["ingested_pct"],
         a["rows_quarantined"], a["quarantined_pct"]))
    w("-" * 92)
    w("%-32s %8s %8s %8s  %-16s %s"
      % ("reason", "rows", "%corpus", "%quar", "pattern", "bias"))
    for reason, d in a["by_reason"].items():
        det = report["quarantine_detail"].get(reason, {})
        imp = report["training_impact"]["by_reason"].get(reason, {})
        w("%-32s %8d %7.2f%% %7.2f%%  %-16s %s"
          % (reason, d["rows"], d["pct_of_rows_in"], d["pct_of_quarantined"],
             det.get("dominant_pattern", "?"),
             imp.get("bias_verdict", "?")))
    w("-" * 92)
    w("VERDICT: %s" % v["verdict"])
    for r in v["reasons"]:
        w("  * %s" % r)
    w("-" * 92)
    w("TOP PROBLEMS BY IMPACT")
    for f in report["impact_ranking"][:5]:
        w("  %d. %-32s score %5.1f  (%s rows, %s, %s)"
          % (f["rank"], f["problem"], f["impact_score"], f["rows"],
             f["dominant_pattern"], f["bias_verdict"]))
    p = report["pressure_bounds_investigation"]
    if p.get("rows"):
        w("-" * 92)
        w("PRESSURE BOUNDS: %d rejected (%.2f%% of corpus)"
          % (p["rows"], p["pct_of_rows_in"]))
        for c in p.get("diagnosis_conclusions", []):
            w("  * %s" % c)
    w("=" * 92 + "\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Report quarantine, completeness, continuity and training "
                    "impact for a weather telemetry CSV.")
    ap.add_argument("--csv", default=DEFAULT_CSV,
                    help="telemetry CSV (default: weather_telemetry_current.csv)")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="output JSON path")
    ap.add_argument("--no-write", action="store_true",
                    help="build and print the report without writing JSON")
    ap.add_argument("--window-hours", type=float, default=72.0,
                    help="half-width in hours of the time-matched control window")
    ap.add_argument("--git-commit", default=None,
                    help="override the recorded commit (read-only git by default)")
    ap.add_argument("--no-git", action="store_true",
                    help="do not shell out to git at all")
    args = ap.parse_args(argv)

    if not os.path.exists(args.csv):
        raise SystemExit("telemetry CSV not found: %s" % args.csv)

    report = build_report(
        csv_path=args.csv,
        window_hours=args.window_hours,
        git_commit="not_recorded" if args.no_git else args.git_commit,
    )
    print_summary(report)

    if not args.no_write:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        print("Report written to: %s" % os.path.basename(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
