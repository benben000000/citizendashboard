"""
Data freshness and coverage monitor for the KloudTrack training corpus.

THE FAILURE THIS EXISTS TO CATCH
--------------------------------
`weather_telemetry.csv` is the committed release input. It stopped at
2026-08-26 while the live feed carried on to 2026-09-30. Thirty-five days of
training data rotted away, and every model trained in that window was trained
on a corpus that was quietly degrading -- with no alert, because nothing was
looking. Age is the cheapest possible health check and it is the one that was
missing.

WHAT IT REPORTS
---------------
  corpus    newest observation, age in hours and days, span, and how much of
            the span is actually covered
  station   last observation, row count, bin coverage, whether it has gone
            silent, and how long it has been quiet
  gaps      hourly bins expected vs present, so a station that dropped out for
            three hours is distinguishable from one that is uniformly sparse
  verdict   PASS / WARN / FAIL against configurable thresholds, as JSON and as
            an alert body

WHY THE GAP STRUCTURE IS REPORTED PER STATION
---------------------------------------------
A single coverage fraction cannot tell a dropout from sparse reporting. A
station that is silent from last Tuesday and one that has always reported every
third hour both score 33%. They are completely different faults: one is a
device that died, the other is a device that never worked. So each station
carries both its coverage over its own span AND the largest hole strictly
inside that span, plus the trailing silence, which sits outside the span by
construction.

WHY IMPLAUSIBLE TIMESTAMPS ARE SET ASIDE
-----------------------------------------
The committed corpus contains two rows stamped 2069-12-31 -- a device clock
fault, already quarantined downstream by the year-range check in dataset.py.
Taking the raw maximum timestamp as "newest observation" makes the corpus look
15,798 days in the FUTURE and silently destroys the age calculation, which is
the one number this tool exists to produce. Implausible rows are therefore
excluded from the age, counted, and reported with their raw maximum so nothing
is hidden.

ROSTER
------
Coverage is only meaningful against an expected station list. By default the
roster is read from `station_index.json` beside the CSV; pass `--expect-stations`
to check against the fetch roster, which is what catches a station that has
dropped out of the corpus entirely. A rostered station with zero rows is a
FAIL, not an omission from the table.

EXIT CODES
----------
  0  PASS     corpus is fresh and every station is reporting
  1  WARN     something needs attention within a day
  2  FAIL     do not train or release on this corpus
  3  ERROR    the corpus could not be read at all

Pure standard library. It has to run on a schedule next to a live feed, where
the dependency set is not something we control.
"""

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass, asdict, fields
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(HERE, "..", "data"))
DEFAULT_CSV = os.path.join(DATA_DIR, "weather_telemetry.csv")
DEFAULT_CURRENT_CSV = os.path.join(DATA_DIR, "weather_telemetry_current.csv")
STATION_INDEX = os.path.join(DATA_DIR, "station_index.json")
STATION_COORDS = os.path.join(DATA_DIR, "station_coords.json")

SCHEMA = "kloudtrack.data_freshness.v1"

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
_RANK = {PASS: 0, WARN: 1, FAIL: 2}

# Timestamps at or after this are the raw maximum. Reported, never used for age.
RAW_TS = "%Y-%m-%dT%H:%M:%SZ"

# How many internal gaps to name per station in the JSON. Enough to show a
# recurring pattern without turning the report into a gap log.
MAX_REPORTED_GAPS = 5


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
#
# The reasoning for every default is in RECOMMENDED_THRESHOLDS_NOTES below and
# is spelled out rather than left as a bare number, because a threshold nobody
# can justify is a threshold nobody will trust when it first fires.

@dataclass(frozen=True)
class FreshnessThresholds:
    """
    Every knob the verdict depends on. Frozen so a report can carry the exact
    configuration that produced it and a reader can tell whether a FAIL came
    from a tightened threshold or from worse data.
    """

    # --- corpus age -------------------------------------------------------
    # The refetch is 16 requests at 3.5s pacing, and every response is cached,
    # so a daily refresh costs about a minute. Missing three daily runs is a
    # broken job, not weather.
    warn_age_hours: float = 72.0            # 3 days
    fail_age_hours: float = 168.0           # 7 days

    # --- per-station silence ---------------------------------------------
    # A healthy station reports on the hourly grid, so one missed day is
    # already abnormal. 24h is the "go and look" line; 72h is "the device is
    # off and someone has to drive to it".
    warn_station_silence_hours: float = 24.0
    fail_station_silence_hours: float = 72.0

    # --- coverage over the station's own span -----------------------------
    # Both corpora are effectively dense once binned, so 90% is generous. The
    # point is to catch a station that is nominally reporting but has a quarter
    # of its history missing.
    warn_station_coverage: float = 0.90
    fail_station_coverage: float = 0.60

    # --- holes strictly inside the span -----------------------------------
    # A dropout is different in kind from a trailing silence: trailing silence
    # is covered by the silence rule, so the gap rule only judges the interior.
    # 6 hours is past any plausible comms retry cycle.
    warn_gap_hours: float = 6.0
    fail_gap_hours: float = 24.0

    # --- late starters ---------------------------------------------------
    # Reported and surfaced, but deliberately NOT part of the verdict. A
    # station commissioned in August is a fact about the fleet, not a fault,
    # and a rule that fires every day for a permanent condition trains people
    # to ignore the dashboard. It matters for training -- a late station has no
    # rows in the earliest split -- so it is called out as its own list.
    late_start_note_hours: float = 24.0

    # --- clock faults -----------------------------------------------------
    # More than two days ahead is a device fault, not NTP slop. Such rows are
    # excluded from every age calculation and reported.
    max_future_skew_hours: float = 48.0

    # --- coverage policy --------------------------------------------------
    # A rostered station with no rows at all is the failure this monitor was
    # written for: one of sixteen stations answering with nothing, in silence.
    fail_on_missing_station: bool = True

    # --- binning ----------------------------------------------------------
    bin_minutes: int = 60

    def __post_init__(self) -> None:
        if self.bin_minutes <= 0:
            raise ValueError("bin_minutes must be positive")
        if self.fail_age_hours < self.warn_age_hours:
            raise ValueError("fail_age_hours must not be below warn_age_hours")
        if self.fail_station_silence_hours < self.warn_station_silence_hours:
            raise ValueError("fail_station_silence_hours must not be below the warn value")
        if self.fail_station_coverage > self.warn_station_coverage:
            raise ValueError("fail_station_coverage must not exceed warn_station_coverage")
        if self.fail_gap_hours < self.warn_gap_hours:
            raise ValueError("fail_gap_hours must not be below warn_gap_hours")
        for name in (f.name for f in fields(self)):
            v = getattr(self, name)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v < 0:
                raise ValueError(f"{name} must not be negative")


RECOMMENDED_THRESHOLDS_NOTES = {
    "warn_age_hours": "3 daily refresh runs missed. The refetch is 16 cached "
                      "requests, so this is a broken job, not weather.",
    "fail_age_hours": "A third of a 100-day training window is gone and the "
                      "seasonal signal with it. The corpus that triggered this "
                      "was 34.7 days old -- 35 of 102 days missing.",
    "warn_station_silence_hours": "One missed day on an hourly feed. Enough to "
                                  "open a ticket, not to page anyone.",
    "fail_station_silence_hours": "Three missed days: the device is off. Three "
                                  "stations in the refetch crossed this line "
                                  "between 2026-08-30 and 2026-09-18 and nothing "
                                  "noticed.",
    "warn_station_coverage": "More than a tenth of the station's own history is "
                             "missing.",
    "fail_station_coverage": "The station is contributing a scatter of samples, "
                             "which trains worse than not training on it at all.",
    "warn_gap_hours": "Past any plausible comms retry cycle; the middle of the "
                      "record is missing rather than its tail.",
    "fail_gap_hours": "A full day missing from the interior of the record.",
    "late_start_note_hours": "Reported, not scored. A permanently late station "
                             "would otherwise hold the whole dashboard in WARN.",
    "max_future_skew_hours": "Beyond two days ahead is a device clock fault, not "
                             "NTP slop. The committed corpus has 2 rows at "
                             "2069-12-31 that would otherwise make the corpus "
                             "look 15,798 days in the future.",
    "fail_on_missing_station": "A rostered station answering with nothing is the "
                               "exact failure that went unnoticed: 1 of 16.",
    "bin_minutes": "60 matches the grid the pipeline resamples to, so the gap "
                  "structure here is the same gap structure the model sees. A "
                  "narrower bin would call ordinary sub-hourly sparsity a gap.",
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def iso(dt: Optional[datetime]) -> Optional[str]:
    """UTC, second resolution, Z-suffixed. The only timestamp form emitted."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime(RAW_TS)


def parse_ts(value: Any) -> Optional[datetime]:
    """
    Tolerant ISO-8601 parse. Returns None rather than raising, because a single
    unparseable row must not take down a monitor that exists to report damage.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def worst(*verdicts: str) -> str:
    """The most severe verdict in the set. Order is FAIL > WARN > PASS."""
    out = PASS
    for v in verdicts:
        if _RANK.get(v, 0) > _RANK[out]:
            out = v
    return out


def verdict_rank(verdict: str) -> int:
    return _RANK.get(verdict, 0)


def _grade(value: float, warn: float, fail: float, higher_is_worse: bool = True) -> str:
    """
    Threshold test shared by age, silence and gap hours.

    Boundaries are inclusive of the threshold on the warn side: at exactly the
    warn value the station is a WARN, at exactly the fail value it is a FAIL.
    A threshold that only trips on the far side of the number is a threshold
    that surprises people, so the tests are written out rather than inferred.
    """
    if higher_is_worse:
        if value >= fail:
            return FAIL
        if value >= warn:
            return WARN
        return PASS
    if value <= fail:
        return FAIL
    if value <= warn:
        return WARN
    return PASS


def _hours(delta_seconds: float) -> float:
    return round(delta_seconds / 3600.0, 3)


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------

class _StationScan:
    """Per-station accumulator. Deliberately tiny: the corpus is 750k rows."""

    __slots__ = ("rows", "implausible", "first", "last", "name", "bins")

    def __init__(self) -> None:
        self.rows = 0
        self.implausible = 0
        self.first: Optional[datetime] = None
        self.last: Optional[datetime] = None
        self.name = ""
        self.bins: Set[int] = set()

    def add(self, ts: datetime, bin_index: int) -> None:
        self.rows += 1
        self.bins.add(bin_index)
        if self.first is None or ts < self.first:
            self.first = ts
        if self.last is None or ts > self.last:
            self.last = ts


def scan_telemetry_csv(path: str, bin_minutes: int = 60,
                       now: Optional[datetime] = None,
                       max_future_skew_hours: float = 48.0) -> Dict[str, Any]:
    """
    One streaming pass over a telemetry CSV.

    Returns a scan dict holding datetimes and bin sets -- the internal form.
    `analyse()` turns it into the JSON-safe report.

    Rows stamped more than `max_future_skew_hours` ahead of `now` are counted as
    implausible and excluded from first/last/bins, so a device clock fault
    cannot masquerade as a fresh corpus. They are never dropped silently: the
    count and the raw maximum come back in the scan.
    """
    ref = now or datetime.now(timezone.utc)
    horizon = ref + timedelta(hours=max_future_skew_hours)
    bin_seconds = int(bin_minutes) * 60

    scan: Dict[str, Any] = {
        "path": os.path.abspath(path),
        "bin_minutes": int(bin_minutes),
        "rows": 0,
        "malformed_rows": 0,
        "blank_station_rows": 0,
        "unusable_rows": 0,
        "implausible_rows": 0,
        "implausible_max": None,
        "newest_raw": None,
        "stations": {},
    }
    st: Dict[str, _StationScan] = {}
    _implausible_max: Optional[datetime] = None
    _newest_raw: Optional[datetime] = None

    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        fields_present = reader.fieldnames or []
        if "station_id" not in fields_present or "recorded_at" not in fields_present:
            raise ValueError(
                f"{path}: expected 'station_id' and 'recorded_at' columns, "
                f"found {fields_present}")
        for row in reader:
            scan["rows"] += 1
            sid = (row.get("station_id") or "").strip()
            if not sid:
                scan["blank_station_rows"] += 1
                scan["unusable_rows"] += 1
                continue
            raw = row.get("recorded_at")
            ts = parse_ts(raw)
            if ts is None:
                scan["malformed_rows"] += 1
                scan["unusable_rows"] += 1
                continue

            if _newest_raw is None or ts > _newest_raw:
                _newest_raw = ts
            if ts > horizon:
                # Device clock fault. dataset.py quarantines these by year
                # range; here they must not be allowed to set the corpus age.
                scan["implausible_rows"] += 1
                if _implausible_max is None or ts > _implausible_max:
                    _implausible_max = ts
                s = st.get(sid)
                if s is not None:
                    s.implausible += 1
                continue

            s = st.get(sid)
            if s is None:
                s = st[sid] = _StationScan()
            if not s.name:
                s.name = (row.get("station_name") or "").strip()
            s.add(ts, int(ts.timestamp()) // bin_seconds)

    scan["stations"] = st
    scan["implausible_max"] = _implausible_max
    scan["newest_raw"] = _newest_raw
    scan["station_count"] = len(st)
    return scan


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _bin_time(bin_index: int, bin_seconds: int) -> datetime:
    return datetime.fromtimestamp(bin_index * bin_seconds, tz=timezone.utc)


def _internal_gaps(bins: Sequence[int], bin_seconds: int) -> List[Dict[str, Any]]:
    """
    Runs of MISSING bins strictly between the station's first and last report.

    This is the discriminator between a dropout and a sparse station, and it
    is computed only over the interior. A station that stopped reporting last
    Tuesday has no internal gap at all -- its loss is trailing silence, which
    is measured against `now` instead.
    """
    gaps: List[Dict[str, Any]] = []
    for a, b in zip(bins, bins[1:]):
        missing = b - a - 1
        if missing <= 0:
            continue
        gaps.append({
            "start": iso(_bin_time(a + 1, bin_seconds)),
            "end": iso(_bin_time(b - 1, bin_seconds)),
            "missing_bins": missing,
            "hours": _hours(missing * bin_seconds),
        })
    gaps.sort(key=lambda g: g["missing_bins"], reverse=True)
    return gaps


def load_roster(path: Optional[str]) -> Tuple[List[str], str]:
    """
    Expected station list. Returns (ids, source) so the report records where
    the expectation came from -- a missing station is only a failure if
    something said it should be there.
    """
    if not path or not os.path.exists(path):
        return [], "none"
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return [], "none"
    if isinstance(raw, dict):
        raw = raw.get("stations", [])
    if not isinstance(raw, list):
        return [], "none"
    ids = []
    for entry in raw:
        if isinstance(entry, str):
            sid = entry.strip()
        elif isinstance(entry, dict):
            sid = str(entry.get("station_id") or "").strip()
        else:
            continue
        if sid:
            ids.append(sid)
    return sorted(set(ids)), os.path.basename(path)


def _station_names() -> Dict[str, str]:
    """Display names from station_coords.json. Best effort, never required."""
    try:
        with open(STATION_COORDS, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {k: str(v.get("name") or "") for k, v in raw.items()
            if isinstance(v, dict) and v.get("name")}


def analyse(scan: Dict[str, Any],
            thresholds: Optional[FreshnessThresholds] = None,
            now: Optional[datetime] = None,
            expected_stations: Optional[Iterable[str]] = None,
            roster_source: str = "explicit") -> Dict[str, Any]:
    """
    Turn a scan into the JSON-safe report. No I/O, no clock reads, so it is
    fully testable and the same report can be produced for a past `now`.
    """
    t = thresholds or FreshnessThresholds()
    ref = now or datetime.now(timezone.utc)
    bin_seconds = t.bin_minutes * 60
    st: Dict[str, _StationScan] = scan.get("stations", {})

    plausible_first: Optional[datetime] = None
    plausible_last: Optional[datetime] = None
    for s in st.values():
        if s.first is not None and (plausible_first is None or s.first < plausible_first):
            plausible_first = s.first
        if s.last is not None and (plausible_last is None or s.last > plausible_last):
            plausible_last = s.last

    roster = sorted({str(s).strip() for s in (expected_stations or []) if str(s).strip()})
    names = _station_names()

    # ---- roster resolution, case-insensitively ----------------------------
    # Station ids are opaque mixed-case strings, and a roster that differs from
    # the corpus by a single letter produces a report that says a perfectly
    # healthy station has vanished. That is not a hypothetical: the refetch
    # roster asked the API for 'wkAWlzlm' where the station is 'wkAWLzlm', and
    # the monitor must not repeat the mistake that hid it. An exact match wins;
    # a case-insensitive match is reported as a MISMATCH and counts as present,
    # because calling a station absent on a letter is exactly the kind of false
    # alarm that gets a real outage dismissed.
    observed_by_fold: Dict[str, str] = {}
    for sid in st:
        observed_by_fold.setdefault(sid.lower(), sid)
    case_mismatches: List[Dict[str, str]] = []
    missing: List[str] = []
    for sid in roster:
        if sid in st:
            continue
        twin = observed_by_fold.get(sid.lower())
        if twin is not None:
            case_mismatches.append({"roster_id": sid, "corpus_id": twin})
        else:
            missing.append(sid)
    mismatch_fold = {m["roster_id"] for m in case_mismatches}
    alias_of = {m["corpus_id"]: m["roster_id"] for m in case_mismatches}

    # ---- stations ---------------------------------------------------------
    # A roster entry that only differs by case is folded onto the corpus
    # station rather than given a phantom row of its own, so the table shows
    # real stations only and the alias is a note.
    stations: Dict[str, Dict[str, Any]] = {}
    for sid in sorted((set(st) | set(roster)) - mismatch_fold):
        s = st.get(sid)
        entry: Dict[str, Any] = {"name": (s.name if s and s.name else names.get(sid, ""))}
        reasons: List[str] = []
        notes: List[str] = []

        alias = alias_of.get(sid)
        if alias:
            notes.append(f"roster spells this station {alias!r}; matched by case")

        if s is None or s.rows == 0:
            entry.update({
                "rows": 0, "bins_present": 0, "bins_expected": 0, "coverage": None,
                "first": None, "last": None, "age_hours": None,
                "silence_hours": None, "silent": True,
                "late_start_hours": None, "max_internal_gap_hours": None,
                "gaps": [], "in_roster": sid in roster,
                "implausible_rows": s.implausible if s else 0,
            })
            entry["verdict"] = FAIL if t.fail_on_missing_station else WARN
            reasons.append("rostered station has no rows in the corpus"
                           if sid in roster else "station has no rows")
            entry["reasons"] = reasons
            entry["notes"] = notes
            stations[sid] = entry
            continue

        bins = sorted(s.bins)
        expected_bins = bins[-1] - bins[0] + 1
        coverage = len(bins) / expected_bins if expected_bins else 0.0
        gaps = _internal_gaps(bins, bin_seconds)
        max_gap = gaps[0]["hours"] if gaps else 0.0
        age = _hours((ref - s.last).total_seconds())
        silence = age

        coverage_verdict = _grade(coverage, t.warn_station_coverage,
                                  t.fail_station_coverage, higher_is_worse=False)
        silence_verdict = _grade(silence, t.warn_station_silence_hours,
                                 t.fail_station_silence_hours)
        gap_verdict = _grade(max_gap, t.warn_gap_hours, t.fail_gap_hours)
        verdict = worst(coverage_verdict, silence_verdict, gap_verdict)

        if silence_verdict != PASS:
            reasons.append(f"silent for {silence:.1f} h "
                           f"(last observation {iso(s.last)})")
        if coverage_verdict != PASS:
            reasons.append(f"span coverage {coverage * 100:.1f}% of {expected_bins} "
                           f"hourly bins")
        if gap_verdict != PASS and gaps:
            g = gaps[0]
            reasons.append(f"internal gap of {g['hours']:.1f} h at {g['start']}")
        if s.implausible:
            notes.append(f"{s.implausible} implausible future-dated rows excluded")
        if plausible_first is not None:
            late = _hours((s.first - plausible_first).total_seconds())
            if late > t.late_start_note_hours:
                # Noted, not scored. See FreshnessThresholds for why.
                notes.append(f"started {late / 24.0:.1f} d after the fleet's first "
                             f"observation ({iso(s.first)})")
        entry.update({
            "rows": s.rows,
            "bins_present": len(bins),
            "bins_expected": expected_bins,
            "bins_missing": expected_bins - len(bins),
            "coverage": round(coverage, 6),
            "first": iso(s.first),
            "last": iso(s.last),
            "age_hours": age,
            "age_days": round(age / 24.0, 3),
            "silence_hours": silence,
            "silent": silence_verdict != PASS,
            "late_start_hours": (_hours((s.first - plausible_first).total_seconds())
                                 if plausible_first is not None else None),
            "max_internal_gap_hours": max_gap,
            "gaps": gaps[:MAX_REPORTED_GAPS],
            "implausible_rows": s.implausible,
            "in_roster": sid in roster or sid in alias_of,
            "verdict": verdict,
            "reasons": reasons,
            "notes": notes,
        })
        stations[sid] = entry

    # ---- corpus -----------------------------------------------------------
    unexpected = sorted(sid for sid in st
                        if roster and sid not in roster and sid not in alias_of)

    corpus_reasons: List[str] = []
    age_hours: Optional[float] = None
    age_days: Optional[float] = None
    age_verdict = FAIL
    if plausible_last is None:
        corpus_verdict = FAIL
        corpus_reasons.append("no usable observation timestamps in the corpus")
    else:
        age_hours = _hours((ref - plausible_last).total_seconds())
        age_days = round(age_hours / 24.0, 3)
        v = _grade(age_hours, t.warn_age_hours, t.fail_age_hours)
        age_verdict = v
        if v != PASS:
            corpus_reasons.append(
                f"corpus is {age_days:.1f} days old "
                f"(newest observation {iso(plausible_last)}, "
                f"WARN {t.warn_age_hours / 24.0:.1f} d / "
                f"FAIL {t.fail_age_hours / 24.0:.1f} d)")
        corpus_verdict = v
        if plausible_last > ref:
            corpus_reasons.append("newest observation is in the future")
            corpus_verdict = FAIL
    if scan.get("implausible_rows"):
        corpus_reasons.append(
            f"{scan['implausible_rows']} rows dated implausibly far ahead "
            f"(raw max {iso(scan.get('implausible_max'))}); excluded from the age")
        corpus_verdict = worst(corpus_verdict, WARN)
    if missing and t.fail_on_missing_station:
        corpus_reasons.append(
            f"{len(missing)} rostered station(s) returned no rows: "
            f"{', '.join(missing)}")
        corpus_verdict = FAIL
    if case_mismatches:
        for m in case_mismatches:
            corpus_reasons.append(
                f"roster id {m['roster_id']!r} differs from corpus id "
                f"{m['corpus_id']!r} by case only; matched, but the two sources "
                f"disagree on the identifier and a case-sensitive lookup will miss it")
        corpus_verdict = worst(corpus_verdict, WARN)
    if scan.get("malformed_rows") or scan.get("blank_station_rows"):
        corpus_reasons.append(
            f"{scan['malformed_rows']} malformed and "
            f"{scan['blank_station_rows']} station-id-less rows skipped")

    observed = [e for e in stations.values() if e["rows"] > 0]
    covs = [e["coverage"] for e in observed if e.get("coverage") is not None]
    span_hours = (_hours((plausible_last - plausible_first).total_seconds())
                  if plausible_first and plausible_last else 0.0)

    report: Dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": iso(ref),
        "source": {
            "path": scan.get("path"),
            "rows": scan.get("rows", 0),
            "usable_rows": scan.get("rows", 0) - scan.get("unusable_rows", 0),
            "malformed_rows": scan.get("malformed_rows", 0),
            "blank_station_rows": scan.get("blank_station_rows", 0),
            "implausible_rows": scan.get("implausible_rows", 0),
            "implausible_max": iso(scan.get("implausible_max")),
            "newest_raw_observation": iso(scan.get("newest_raw")),
            "bin_minutes": t.bin_minutes,
            "roster_source": roster_source,
            "roster_size": len(roster),
        },
        "thresholds": asdict(t),
        "corpus": {
            "newest_observation": iso(plausible_last),
            "oldest_observation": iso(plausible_first),
            "age_hours": age_hours,
            "age_days": age_days,
            "age_verdict": age_verdict,
            "span_hours": span_hours,
            "stations_observed": len(observed),
            "stations_expected": len(roster),
            "mean_station_coverage": round(sum(covs) / len(covs), 6) if covs else None,
            "min_station_coverage": round(min(covs), 6) if covs else None,
            "verdict": corpus_verdict,
            "reasons": corpus_reasons,
        },
        "stations": stations,
        "missing_stations": missing,
        "id_case_mismatches": case_mismatches,
        "unexpected_stations": unexpected,
        "late_starters": sorted(sid for sid, e in stations.items()
                                if (e.get("late_start_hours") or 0.0) > t.late_start_note_hours),
        "counts": {
            "ok": sum(1 for e in stations.values() if e["verdict"] == PASS),
            "warn": sum(1 for e in stations.values() if e["verdict"] == WARN),
            "fail": sum(1 for e in stations.values() if e["verdict"] == FAIL),
        },
    }
    report["verdict"] = worst(corpus_verdict,
                              *[e["verdict"] for e in stations.values()])
    report["findings"] = _findings(report)
    report["summary"] = _summary_line(report)
    return report


def _summary_line(report: Dict[str, Any]) -> str:
    """One sentence, suitable as an alert subject or first line of a body."""
    c = report["corpus"]
    v = report["verdict"]
    src = os.path.basename(report["source"]["path"] or "corpus")
    if c["age_days"] is None:
        return f"[{v}] {src}: no usable telemetry timestamps"
    fresh = (f"newest {c['newest_observation']}, {c['age_days']:.1f} d old")
    detail = ""
    n_fail = report["counts"]["fail"]
    n_warn = report["counts"]["warn"]
    if n_fail or n_warn:
        detail = f"; {n_fail} station(s) FAIL, {n_warn} WARN"
    miss = report["missing_stations"]
    if miss:
        detail += f"; missing: {', '.join(miss)}"
    return f"[{v}] {src}: {fresh}{detail}"


def _findings(report: Dict[str, Any]) -> List[str]:
    """Ordered, human sentences. Worst first, capped so an alert stays readable."""
    out: List[str] = []
    c = report["corpus"]
    out.extend(c["reasons"])
    silent = sorted(((e["silence_hours"], sid) for sid, e in report["stations"].items()
                     if e.get("silent") and e.get("silence_hours") is not None),
                    reverse=True)
    for hrs, sid in silent[:6]:
        e = report["stations"][sid]
        out.append(f"{sid} has not reported since {e['last']} ({hrs / 24.0:.1f} d)")
    gapped = sorted(((e["max_internal_gap_hours"], sid) for sid, e in report["stations"].items()
                     if (e.get("max_internal_gap_hours") or 0) > 0),
                    reverse=True)
    for hrs, sid in gapped[:3]:
        e = report["stations"][sid]
        g = (e["gaps"] or [{}])[0]
        out.append(f"{sid} has an internal gap of {hrs:.1f} h at {g.get('start')}")
    sparse = sorted(((e["coverage"], sid) for sid, e in report["stations"].items()
                     if e.get("coverage") is not None and e["coverage"] < 1.0),
                    key=lambda x: x[0])
    for cov, sid in sparse[:3]:
        e = report["stations"][sid]
        out.append(f"{sid} covers {cov * 100:.1f}% of its own span "
                   f"({e['bins_present']}/{e['bins_expected']} hourly bins)")
    if report["late_starters"]:
        n = len(report["late_starters"])
        out.append(f"{n} station(s) started after the fleet's first observation "
                   f"({', '.join(report['late_starters'])}); they contribute "
                   f"nothing to the earliest part of a chronological split")
    if report["unexpected_stations"]:
        out.append("stations present but not on the roster: "
                   + ", ".join(report["unexpected_stations"]))
    return out


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _fmt_age(hours: Optional[float]) -> str:
    if hours is None:
        return "   n/a"
    if hours < 48:
        return f"{hours:6.1f}h"
    return f"{hours / 24.0:6.2f}d"


def render_text(report: Dict[str, Any], max_rows: int = 40) -> str:
    """
    Human-readable summary. Written to be pasted into an alert body: verdict
    first, the numbers that produced it, then what to do.
    """
    src = report["source"]
    c = report["corpus"]
    t = report["thresholds"]
    lines: List[str] = []

    lines.append(f"[{report['verdict']}] DATA FRESHNESS  {os.path.basename(src['path'] or '')}")
    lines.append("=" * 78)
    lines.append(f"  generated    {report['generated_at']}")
    lines.append(f"  rows         {src['rows']:,} usable {src['usable_rows']:,}   "
                 f"bin {src['bin_minutes']} min")
    lines.append(f"  stations     {c['stations_observed']} observed"
                 + (f" / {c['stations_expected']} rostered ({src['roster_source']})"
                    if c["stations_expected"] else ""))
    if src["implausible_rows"]:
        lines.append(f"  implausible  {src['implausible_rows']:,} rows, raw max "
                     f"{src['implausible_max']}  (excluded from the age)")

    lines.append("")
    lines.append("  CORPUS")
    lines.append(f"    newest observation   {c['newest_observation']}")
    lines.append(f"    oldest observation   {c['oldest_observation']}")
    lines.append(f"    corpus age           {_fmt_age(c['age_hours'])}"
                 f"   (WARN {t['warn_age_hours'] / 24:.1f} d, "
                 f"FAIL {t['fail_age_hours'] / 24:.1f} d)   {c['age_verdict']}")
    if c["verdict"] != c["age_verdict"]:
        lines.append(f"    corpus verdict       {c['verdict']}"
                     f"   (driven by coverage, not by age)")
    if c["mean_station_coverage"] is not None:
        lines.append(f"    station coverage     mean {c['mean_station_coverage'] * 100:.1f}%"
                     f"   min {c['min_station_coverage'] * 100:.1f}%")

    # Worst first: an alert body should open on the station that is broken, not
    # on whichever id happens to sort first.
    order = sorted(report["stations"],
                   key=lambda s: (-verdict_rank(report["stations"][s]["verdict"]),
                                  -(report["stations"][s].get("silence_hours") or 0.0),
                                  s))
    truncated = max(0, len(order) - max_rows)
    lines.append("")
    lines.append(f"  STATIONS, worst first  ({len(order)} total; rows / hourly bins "
                 f"present of expected / last seen /")
    lines.append(f"  silence / largest internal gap)")
    lines.append(f"    {'station':<10}{'rows':>8}{'bins':>13}{'cover':>7}"
                 f"  {'last seen':<21}{'silence':>9}{'gap':>7}  v")
    for sid in order[:max_rows]:
        e = report["stations"][sid]
        if e["rows"]:
            bins = f"{e['bins_present']}/{e['bins_expected']}"
            cover = f"{e['coverage'] * 100:5.1f}%"
            last = (e["last"] or "")[:19]
            sil = f"{e['silence_hours']:.1f}h"
            gap = f"{e['max_internal_gap_hours']:.1f}h"
        else:
            bins, cover, last, sil, gap = "-", "  n/a", "never", "-", "-"
        lines.append(f"    {sid:<10}{e['rows']:>8,}{bins:>13}{cover:>7}"
                     f"  {last:<21}{sil:>9}{gap:>7}  {e['verdict']}")
    if truncated:
        lines.append(f"    ... and {truncated} more; see --json for the full list")

    findings = report.get("findings") or []
    if findings:
        lines.append("")
        lines.append("  FINDINGS")
        for i, f in enumerate(findings, 1):
            lines.append(f"    {i:>2}. {f}")

    lines.append("")
    lines.append("  ACTION")
    if report["verdict"] == FAIL:
        lines.append("    Do not train or release on this corpus.")
        lines.append("    Refetch, then re-run this check:")
        lines.append("      python prediction-model/src/fetch_current_telemetry.py")
        lines.append("      python prediction-model/src/check_data_freshness.py")
        if report["missing_stations"]:
            lines.append("    Stations that returned nothing are an upstream or "
                         "hardware fault, not a")
            lines.append("    pipeline bug -- investigate the device and the API "
                         "response for it.")
    elif report["verdict"] == WARN:
        lines.append("    Corpus is usable but aging. Refetch within the day.")
    else:
        lines.append("    Corpus is current. No action.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience entry points
# ---------------------------------------------------------------------------

def check_csv(path: str,
              thresholds: Optional[FreshnessThresholds] = None,
              now: Optional[datetime] = None,
              expected_stations: Optional[Iterable[str]] = None,
              roster_path: Optional[str] = None) -> Dict[str, Any]:
    """Scan + analyse. The one call a scheduler needs."""
    t = thresholds or FreshnessThresholds()
    if expected_stations is not None:
        roster, source = sorted({str(s).strip() for s in expected_stations
                                 if str(s).strip()}), "explicit"
    else:
        if roster_path is None:
            roster_path = os.path.join(os.path.dirname(os.path.abspath(path)),
                                       "station_index.json")
        roster, source = load_roster(roster_path)
    scan = scan_telemetry_csv(path, t.bin_minutes, now, t.max_future_skew_hours)
    return analyse(scan, t, now, roster, source)


def write_report(report: Dict[str, Any], path: str) -> str:
    path = os.path.abspath(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=2, sort_keys=False)
        f.write("\n")
    return path


def _add_threshold_args(ap: argparse.ArgumentParser) -> None:
    d = FreshnessThresholds()
    ap.add_argument("--warn-age-hours", type=float, default=d.warn_age_hours,
                    help="WARN when the corpus is older than this (default %(default)s h)")
    ap.add_argument("--fail-age-hours", type=float, default=d.fail_age_hours,
                    help="FAIL when the corpus is older than this (default %(default)s h)")
    ap.add_argument("--warn-station-silence-hours", type=float,
                    default=d.warn_station_silence_hours)
    ap.add_argument("--fail-station-silence-hours", type=float,
                    default=d.fail_station_silence_hours)
    ap.add_argument("--warn-station-coverage", type=float,
                    default=d.warn_station_coverage)
    ap.add_argument("--fail-station-coverage", type=float,
                    default=d.fail_station_coverage)
    ap.add_argument("--warn-gap-hours", type=float, default=d.warn_gap_hours)
    ap.add_argument("--fail-gap-hours", type=float, default=d.fail_gap_hours)
    ap.add_argument("--max-future-skew-hours", type=float,
                    default=d.max_future_skew_hours,
                    help="rows dated further ahead than this are implausible and "
                         "are excluded from the age (default %(default)s h)")
    ap.add_argument("--bin-minutes", type=int, default=d.bin_minutes,
                    help="bin width for the gap structure (default %(default)s)")
    ap.add_argument("--no-fail-on-missing-station", action="store_true",
                    help="downgrade a rostered station with no rows to WARN")


def thresholds_from_args(args: argparse.Namespace) -> FreshnessThresholds:
    return FreshnessThresholds(
        warn_age_hours=args.warn_age_hours,
        fail_age_hours=args.fail_age_hours,
        warn_station_silence_hours=args.warn_station_silence_hours,
        fail_station_silence_hours=args.fail_station_silence_hours,
        warn_station_coverage=args.warn_station_coverage,
        fail_station_coverage=args.fail_station_coverage,
        warn_gap_hours=args.warn_gap_hours,
        fail_gap_hours=args.fail_gap_hours,
        late_start_note_hours=FreshnessThresholds.late_start_note_hours,
        max_future_skew_hours=args.max_future_skew_hours,
        fail_on_missing_station=not args.no_fail_on_missing_station,
        bin_minutes=args.bin_minutes,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Report telemetry corpus age, per-station coverage and gap "
                    "structure, and a PASS/WARN/FAIL verdict.")
    ap.add_argument("csv", nargs="?", default=DEFAULT_CSV,
                    help=f"telemetry CSV (default: {DEFAULT_CSV})")
    ap.add_argument("--json", dest="json_out", default=None,
                    help="write the machine-readable report here; '-' for stdout")
    ap.add_argument("--expect-stations", default=None,
                    help="comma-separated roster the corpus must cover")
    ap.add_argument("--roster", dest="roster_path", default=None,
                    help="station roster JSON (default: station_index.json "
                         "beside the CSV)")
    ap.add_argument("--now", default=None,
                    help="override the reference time (ISO-8601); for reproducible runs")
    ap.add_argument("--max-rows", type=int, default=40,
                    help="cap the printed station table (default %(default)s)")
    ap.add_argument("--quiet", action="store_true",
                    help="print nothing; rely on the exit code and --json")
    _add_threshold_args(ap)
    args = ap.parse_args(argv)

    now = parse_ts(args.now) if args.now else None
    if args.now and now is None:
        print(f"error: --now {args.now!r} is not a valid ISO-8601 timestamp",
              file=sys.stderr)
        return 3
    if not os.path.exists(args.csv):
        print(f"error: corpus not found: {args.csv}", file=sys.stderr)
        return 3

    try:
        thresholds = thresholds_from_args(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    expected = None
    if args.expect_stations:
        expected = [s.strip() for s in args.expect_stations.split(",") if s.strip()]

    try:
        report = check_csv(args.csv, thresholds, now, expected, args.roster_path)
    except (OSError, ValueError) as exc:
        print(f"error: could not read {args.csv}: {exc}", file=sys.stderr)
        return 3

    json_to_stdout = args.json_out == "-"
    if json_to_stdout:
        json.dump(report, sys.stdout, indent=2)
        sys.stdout.write("\n")
    elif args.json_out:
        print(f"json: {write_report(report, args.json_out)}")
    if not args.quiet and not json_to_stdout:
        print(render_text(report, args.max_rows))

    return {"PASS": 0, "WARN": 1, "FAIL": 2}[report["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
