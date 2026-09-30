"""
Diagnose why a station is missing from the telemetry corpus.

THE CASE THIS WAS BUILT FOR
---------------------------
`wkAWlzlm` returned zero rows from the KloudTrack history API during the
refetch. One of sixteen stations, gone, with no error message anywhere: the
fetcher treats a 4xx as "genuinely absent, retrying is waste", returns an empty
list, and moves on. Sixteen requests, fifteen answers, one silence.

SILENCE HAS CAUSES AND THEY ARE NOT THE SAME FAULT
---------------------------------------------------
This script separates them with evidence rather than assuming the worst:

  NOT_REQUESTED       the id in the request roster does not exist in any
                      catalogue, and a different spelling of the same id does.
                      Nothing was ever asked for. The station is fine.
  ABSENT_FROM_API     the id is correct and the station was real, but the API
                      answered with nothing. Upstream or hardware.
  API_RETURNED_EMPTY  the API answered successfully with an empty payload, so
                      the fetcher cached an empty list. Upstream or hardware.
  FILTERED_BY_PIPELINE the API returned readings and the transform dropped
                      them. A bug on our side.
  NEWLY_SILENT        the station was healthy, then stopped. Hardware.
  PRESENT_AND_CURRENT  the station is in the refetch and reporting.

The distinction matters. "Absent from the API" and "never asked for" look
identical in the output CSV and call for opposite responses: one is a site
visit, the other is a one-character edit.

HOW THE CLASSIFICATION IS EVIDENCED
-----------------------------------
`HistoryClient.history()` writes a cache file for every SUCCESSFUL response,
including an empty one, and writes nothing for an HTTP 4xx or for a run that
exhausts its retries. So the presence and content of a cache file separates
"the API answered" from "the request never landed" without a single network
call. When a cache file does exist, the fetcher's own `to_rows` transform is
replayed against it -- the real function, imported, not a reimplementation --
so a pipeline-side drop is distinguishable from an empty payload.

The cache directory is opened read-only. Nothing here writes to it.

Pure standard library apart from the fetcher import, and no network access.
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Set

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import check_data_freshness as cdf  # noqa: E402
import fetch_current_telemetry as fetch  # noqa: E402

DATA_DIR = os.path.abspath(os.path.join(HERE, "..", "data"))
CACHE_DIR = fetch.CACHE_DIR
STATION_INDEX = os.path.join(DATA_DIR, "station_index.json")
STATION_COORDS = os.path.join(DATA_DIR, "station_coords.json")
DIURNAL_PROFILES = os.path.join(DATA_DIR, "..", "config", "station_diurnal_profiles.json")
TRAIL = os.path.join(DATA_DIR, "observation_audit.jsonl")

DEFAULT_FOCUS = "wkAWlzlm"

# Classified states, worst first. `status` is the headline; `detail` explains.
NOT_REQUESTED = "NOT_REQUESTED"
ABSENT_FROM_API = "ABSENT_FROM_API"
API_RETURNED_EMPTY = "API_RETURNED_EMPTY"
FILTERED_BY_PIPELINE = "FILTERED_BY_PIPELINE"
NEWLY_SILENT = "NEWLY_SILENT"
PRESENT_AND_CURRENT = "PRESENT_AND_CURRENT"
UNKNOWN = "UNKNOWN"

_ORDER = {NOT_REQUESTED: 0, UNKNOWN: 1, ABSENT_FROM_API: 2, API_RETURNED_EMPTY: 3,
          FILTERED_BY_PIPELINE: 4, NEWLY_SILENT: 5, PRESENT_AND_CURRENT: 6}

# Files whose contents establish what the canonical spelling of a station id is.
ID_CATALOGUES = [
    ("weather_telemetry.csv", "committed corpus"),
    ("station_index.json", "station index"),
    ("station_coords.json", "station coordinates"),
    ("config/station_diurnal_profiles.json", "diurnal profiles"),
]


# ---------------------------------------------------------------------------
# Evidence collectors. Each one answers "does this source know the station?"
# ---------------------------------------------------------------------------

def _catalogue_ids(corpus_csv: str) -> Dict[str, Set[str]]:
    """Every station id each catalogue file mentions, by exact spelling."""
    out: Dict[str, Set[str]] = {}

    seen: Set[str] = set()
    with open(corpus_csv, "r", encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            sid = (row.get("station_id") or "").strip()
            if sid:
                seen.add(sid)
    out["committed corpus"] = seen

    for name, label in ID_CATALOGUES[1:]:
        path = os.path.join(DATA_DIR, name)
        ids: Set[str] = set()
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, ValueError):
            out[label] = set()
            continue
        if isinstance(raw, dict):
            ids = {str(k) for k in raw}
        elif isinstance(raw, list):
            for entry in raw:
                if isinstance(entry, dict) and entry.get("station_id"):
                    ids.add(str(entry["station_id"]))
                elif isinstance(entry, str):
                    ids.add(entry)
        out[label] = ids
    return out


def _fold(ids: Set[str]) -> Dict[str, List[str]]:
    folded: Dict[str, List[str]] = {}
    for sid in ids:
        folded.setdefault(sid.lower(), []).append(sid)
    return folded


def corpus_presence(path: str, station: str) -> Dict[str, Any]:
    """
    Does a CSV carry this station, and how far does it run?

    One streaming pass, exact-id match. The caller resolves spelling first, so
    a case-insensitive miss here is a real absence.
    """
    out = {"path": os.path.abspath(path), "present": False, "rows": 0,
           "first": None, "last": None, "exists": os.path.exists(path)}
    if not out["exists"]:
        return out
    first: Optional[datetime] = None
    last: Optional[datetime] = None
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            if (row.get("station_id") or "").strip() != station:
                continue
            out["rows"] += 1
            ts = cdf.parse_ts(row.get("recorded_at"))
            if ts is None:
                continue
            if first is None or ts < first:
                first = ts
            if last is None or ts > last:
                last = ts
    out["present"] = out["rows"] > 0
    out["first"] = cdf.iso(first)
    out["last"] = cdf.iso(last)
    return out


def _cache_files(station: str) -> List[str]:
    """
    Cache files for a station. The tag format is
    `{station}__{start}__{end}__i{interval}__f{outliers}`, so the match is on the
    leading component followed by the separator, not a substring: a lookup for
    'wkAWlzlm' must not pick up a file belonging to 'wkAWLzlm' or to
    'wkAWLzlm2'.

    The exact spelling is tried first. If nothing matches, the same
    case-insensitive comparison is retried and the spelling actually on disk is
    returned, because Windows will happily resolve 'wkAWlzlm__x.json' to a file
    named 'wkAWLzlm__x.json' and a case-sensitive listdir match would then miss
    a cache file that is really there. The on-disk spelling is reported so the
    reader can see when this happened.
    """
    try:
        names = sorted(os.listdir(CACHE_DIR))
    except OSError:
        return []
    exact = [n for n in names if n.startswith(f"{station}__") and n.endswith(".json")]
    if exact:
        return exact
    folded = station.lower()
    return [n for n in names
            if n.lower().startswith(f"{folded}__") and n.lower().endswith(".json")]


def cache_evidence(station: str) -> Dict[str, Any]:
    """
    What the fetcher's own cache says about this station.

    `cached` True with an empty list means the API answered and had nothing.
    `cached` False means the request never completed successfully, because a
    successful response is always cached -- including an empty one.
    """
    files = _cache_files(station)
    ev: Dict[str, Any] = {
        "cache_dir": CACHE_DIR,
        "files": files,
        "cached": bool(files),
        "on_disk_spelling": None,
        "readings": None,
        "readable": False,
        "note": "",
    }
    if not files:
        ev["note"] = ("no cache file: the request never completed successfully "
                      "(an HTTP 4xx, or all retries exhausted, returns without "
                      "caching)")
        return ev
    ev["on_disk_spelling"] = files[-1].split("__", 1)[0]
    try:
        with open(os.path.join(CACHE_DIR, files[-1]), "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as exc:
        ev["note"] = f"cache file unreadable: {exc}"
        return ev
    if not isinstance(data, list):
        ev["note"] = f"cache file holds {type(data).__name__}, expected a list"
        return ev
    ev["readable"] = True
    ev["readings"] = len(data)
    stamps = sorted(str(r.get("recordedAt") or "") for r in data if isinstance(r, dict))
    ev["first"] = stamps[0] if stamps else None
    ev["last"] = stamps[-1] if stamps else None
    if not data:
        ev["note"] = ("cached empty payload: the API answered successfully with "
                      "no telemetry for this station")
    return ev


def replay_transform(station: str, ev: Dict[str, Any]) -> Dict[str, Any]:
    """
    Re-run the fetcher's own `to_rows` over the cached payload.

    This is the only way to tell "the API had nothing" from "the transform threw
    it all away" -- and it uses the real function rather than a copy, so the
    two cannot drift apart.
    """
    out = {"attempted": False, "rows": 0, "dropped": 0}
    if not ev.get("readable") or not ev.get("files"):
        return out
    try:
        with open(os.path.join(CACHE_DIR, ev["files"][-1]), "r", encoding="utf-8") as f:
            readings = json.load(f)
    except (OSError, ValueError):
        return out
    out["attempted"] = True
    out["rows"] = len(fetch.to_rows(station, readings, {}))
    out["dropped"] = len(readings) - out["rows"]
    return out


def index_entry(station: str) -> Optional[Dict[str, Any]]:
    try:
        with open(STATION_INDEX, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return None
    for entry in raw if isinstance(raw, list) else []:
        if isinstance(entry, dict) and entry.get("station_id") == station:
            return entry
    return None


def coords_entry(station: str) -> Optional[Dict[str, Any]]:
    try:
        with open(STATION_COORDS, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return None
    if isinstance(raw, dict) and isinstance(raw.get(station), dict):
        return raw[station]
    return None


def trail_evidence(station: str, trail_path: str = TRAIL) -> Dict[str, Any]:
    """
    Live MQTT evidence for this station.

    The live feed uses a different identifier namespace -- `KT-` device ids, not
    the 8-character station ids -- so a zero count here is expected and means
    nothing about the station. It is reported to stop anyone reading the absence
    as a second confirmation that the station is dead.
    """
    ev: Dict[str, Any] = {"path": trail_path, "exists": os.path.exists(trail_path),
                          "records": 0, "first": None, "last": None,
                          "distinct_live_ids": 0, "namespace_match": False}
    if not ev["exists"]:
        return ev
    counts: Dict[str, int] = {}
    first: Optional[str] = None
    last: Optional[str] = None
    with open(trail_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if not isinstance(rec, dict):
                continue
            sid = rec.get("station_id")
            if sid is None:
                continue
            sid = str(sid)
            counts[sid] = counts.get(sid, 0) + 1
            if sid == station:
                ev["records"] += 1
                ts = rec.get("observed_at_utc")
                if isinstance(ts, str):
                    if first is None or ts < first:
                        first = ts
                    if last is None or ts > last:
                        last = ts
    ev["distinct_live_ids"] = len(counts)
    ev["first"] = first
    ev["last"] = last
    ev["namespace_match"] = station in counts
    return ev


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def diagnose(focus: str = DEFAULT_FOCUS,
             corpus_csv: str = cdf.DEFAULT_CSV,
             refetch_csv: str = cdf.DEFAULT_CURRENT_CSV,
             now: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Build the full evidence picture for one station and classify the cause.
    No network, no writes. Everything is read-only.
    """
    ref = now or datetime.now(timezone.utc)
    report: Dict[str, Any] = {
        "schema": "kloudtrack.station_coverage.v1",
        "generated_at": cdf.iso(ref),
        "focus_requested": focus,
        "sources": {
            "corpus": os.path.abspath(corpus_csv),
            "refetch": os.path.abspath(refetch_csv),
            "cache_dir": CACHE_DIR,
            "station_index": STATION_INDEX,
            "live_trail": TRAIL,
        },
    }

    # ---- which spelling is real? ------------------------------------------
    catalogues = _catalogue_ids(corpus_csv)
    all_known: Set[str] = set()
    for ids in catalogues.values():
        all_known |= ids
    folded_known = _fold(all_known)

    request_roster = list(fetch.MODEL_STATIONS)
    roster_folded = _fold(set(request_roster))

    exact = focus if focus in all_known else None
    twin = None if exact else next(iter(folded_known.get(focus.lower(), [])), None)

    report["request_roster"] = {
        "source": "fetch_current_telemetry.MODEL_STATIONS",
        "size": len(request_roster),
        "contains_focus": focus in request_roster,
        "case_variants_of_focus": sorted(roster_folded.get(focus.lower(), [])),
    }
    report["id_resolution"] = {
        "requested": focus,
        "canonical": exact or twin,
        "matched_exactly": exact is not None,
        "matched_by_case": exact is None and twin is not None,
        "catalogues": {label: len(ids) for label, ids in catalogues.items()},
        "known_spellings": {label: sorted(ids)
                            for label, ids in catalogues.items()
                            if any(i.lower() == focus.lower() for i in ids)},
    }
    # Per-catalogue verdict on this exact spelling, which is what pins the bug.
    report["id_resolution"]["catalogues_containing_requested_spelling"] = sorted(
        label for label, ids in catalogues.items() if focus in ids)
    report["id_resolution"]["catalogues_containing_canonical_spelling"] = sorted(
        label for label, ids in catalogues.items()
        if report["id_resolution"]["canonical"] in ids)

    station = report["id_resolution"]["canonical"] or focus
    report["station_id"] = station
    report["station_name"] = (coords_entry(station) or {}).get("name", "")

    # ---- evidence ---------------------------------------------------------
    ev_cache = cache_evidence(station)
    ev_replay = replay_transform(station, ev_cache)
    evidence: Dict[str, Any] = {
        "request_roster": {
            "contains": station in request_roster,
            "spelling_matches_canonical": station in request_roster,
        },
        "fetch_cache": ev_cache,
        "transform_replay": ev_replay,
        "refetch_csv": corpus_presence(refetch_csv, station),
        "committed_corpus": corpus_presence(corpus_csv, station),
        "station_index": index_entry(station),
        "station_coords": coords_entry(station),
        "live_trail": trail_evidence(station),
    }
    report["evidence"] = evidence

    # ---- classify ---------------------------------------------------------
    # Not requested wins over everything. If no catalogue has ever heard of the
    # id we asked for, then nothing was asked for, and reporting an API outage
    # would send a technician to a healthy station.
    if exact is None and twin is None and not any(
            focus.lower() == sid.lower() for sid in request_roster):
        # Nothing anywhere has ever heard of this id. Reporting an API outage
        # for it would name a station that may not exist.
        report["status"] = UNKNOWN
        report["verdict"] = "UNKNOWN"
        report["detail"] = (
            f"no catalogue, roster, corpus or cache file mentions {focus!r} "
            f"under any spelling, so there is no evidence the station exists. "
            f"This is not a coverage failure; it is an unrecognised identifier.")
        report["recommended_action"] = (
            "check the id against the provider's station list before treating "
            "it as a missing station")
        report["not_a_hardware_fault"] = None
    elif twin is not None and station not in request_roster:
        report["status"] = NOT_REQUESTED
        report["verdict"] = "UPSTREAM_REQUEST_BUG"
        report["detail"] = (
            f"the fetch roster asked the API for {focus!r}, which exists in no "
            f"catalogue; the station is {station!r}"
            + (f" ({report['station_name']})" if report["station_name"] else "")
            + ". The request 404'd, HistoryClient treated a 4xx as genuinely "
              "absent and returned an empty list without caching it, and the "
              "station vanished from the refetch with no error. The station "
              "itself is healthy.")
        report["recommended_action"] = (
            f"correct the station id in the fetch roster from {focus!r} to "
            f"{station!r} and refetch. No hardware work, no site visit.")
        report["not_a_hardware_fault"] = True
    elif evidence["refetch_csv"]["present"]:
        report["status"] = PRESENT_AND_CURRENT
        report["verdict"] = "OK"
        report["detail"] = (f"{station} is in the refetch with "
                            f"{evidence['refetch_csv']['rows']} rows through "
                            f"{evidence['refetch_csv']['last']}.")
        report["recommended_action"] = "none"
        report["not_a_hardware_fault"] = None
    elif not ev_cache["cached"]:
        report["status"] = ABSENT_FROM_API
        report["verdict"] = "UPSTREAM_OR_HARDWARE"
        report["detail"] = (
            f"the id {station!r} is correct and was requested, but the fetch "
            f"cached nothing, which means the request never completed "
            f"successfully: an HTTP 4xx (absent, auth, bad request) or all "
            f"retries exhausted. "
            + ("The station was reporting normally in the committed corpus, so "
               "it has stopped or been withdrawn upstream."
               if evidence["committed_corpus"]["present"] else
               "The station has no history in the committed corpus either."))
        report["recommended_action"] = (
            "check the raw API response for this station, then the device and "
            "its uplink. Do not paper over it in the pipeline: a station that "
            "stops reporting is a site visit.")
        report["not_a_hardware_fault"] = None
    elif ev_cache["readable"] and ev_cache["readings"] == 0:
        report["status"] = API_RETURNED_EMPTY
        report["verdict"] = "UPSTREAM_OR_HARDWARE"
        report["detail"] = (
            f"the API answered successfully and cached an empty payload for "
            f"{station!r}. The station is reachable and the request is well "
            f"formed; the account simply has no telemetry for it in that "
            f"window.")
        report["recommended_action"] = (
            "confirm with the provider whether the device is commissioned and "
            "reporting; an empty-but-successful response usually means the "
            "station was decommissioned or never finished onboarding.")
        report["not_a_hardware_fault"] = None
    elif ev_replay["attempted"] and ev_replay["rows"] == 0:
        report["status"] = FILTERED_BY_PIPELINE
        report["verdict"] = "PIPELINE_BUG"
        report["detail"] = (
            f"the API returned {ev_cache['readings']} readings and the "
            f"fetcher's own transform produced 0 rows. The loss is on our side, "
            f"not upstream: every reading was future-dated, unparseable, or "
            f"missing a timestamp.")
        report["recommended_action"] = (
            "inspect the cached payload's recordedAt values. A transform that "
            "discards 100% of one station is a filter that is too aggressive, "
            "not a quiet device.")
        report["not_a_hardware_fault"] = False
    elif evidence["committed_corpus"]["present"]:
        report["status"] = NEWLY_SILENT
        report["verdict"] = "UPSTREAM_OR_HARDWARE"
        report["detail"] = (
            f"{station} reported until "
            f"{evidence['committed_corpus']['last']} in the committed corpus "
            f"and is absent from the refetch. It went quiet between the two "
            f"fetches.")
        report["recommended_action"] = "site visit; the device stopped reporting"
        report["not_a_hardware_fault"] = None
    else:
        report["status"] = UNKNOWN
        report["verdict"] = "UNKNOWN"
        report["detail"] = ("no source has any record of this station under any "
                            "spelling, so there is nothing to diagnose against.")
        report["recommended_action"] = ("confirm the id against the provider's "
                                        "station list")
        report["not_a_hardware_fault"] = None

    report["fleet"] = _fleet(corpus_csv, refetch_csv, ref, report["station_id"])
    return report


def _fleet(corpus_csv: str, refetch_csv: str, now: datetime, focus: str) -> Dict[str, Any]:
    """
    Fleet-wide context. One station's absence is only interpretable next to the
    rest: three others here have also stopped reporting inside the refetch
    window, which is a different problem with a different owner.
    """
    out: Dict[str, Any] = {}
    for label, path in (("refetch", refetch_csv), ("committed_corpus", corpus_csv)):
        if not os.path.exists(path):
            out[label] = {"error": "file not found", "path": os.path.abspath(path)}
            continue
        try:
            rep = cdf.check_csv(path, now=now, expected_stations=fetch.MODEL_STATIONS)
        except (OSError, ValueError) as exc:
            out[label] = {"error": str(exc), "path": os.path.abspath(path)}
            continue
        out[label] = {
            "verdict": rep["verdict"],
            "summary": rep["summary"],
            "stations_observed": rep["corpus"]["stations_observed"],
            "stations_expected": rep["corpus"]["stations_expected"],
            "newest_observation": rep["corpus"]["newest_observation"],
            "age_hours": rep["corpus"]["age_hours"],
            "age_verdict": rep["corpus"]["age_verdict"],
            "missing_stations": rep["missing_stations"],
            "id_case_mismatches": rep["id_case_mismatches"],
            "counts": rep["counts"],
            "late_starters": rep["late_starters"],
            "silent_stations": sorted(
                sid for sid, e in rep["stations"].items()
                if e.get("silent") and e.get("silence_hours") is not None),
            "focus_verdict": rep["stations"].get(focus, {}).get("verdict"),
            "focus_silence_hours": rep["stations"].get(focus, {}).get("silence_hours"),
        }
    return out


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render(report: Dict[str, Any]) -> str:
    L: List[str] = []
    st = report["station_id"]
    ev = report["evidence"]
    L.append("=" * 78)
    L.append(f"STATION COVERAGE DIAGNOSIS  {report['focus_requested']}")
    L.append("=" * 78)
    L.append(f"  verdict   {report['verdict']}")
    L.append(f"  status    {report['status']}")
    if report["station_name"]:
        L.append(f"  station   {st}  {report['station_name']}")
    else:
        L.append(f"  station   {st}")

    L.append("")
    L.append("  ID RESOLUTION")
    r = report["id_resolution"]
    L.append(f"    requested            {r['requested']!r}")
    L.append(f"    canonical            {r['canonical']!r}"
             f"   ({'exact match' if r['matched_exactly'] else 'MATCHED BY CASE ONLY'})")
    L.append(f"    catalogues with the requested spelling: "
             f"{', '.join(r['catalogues_containing_requested_spelling']) or 'none'}")
    L.append(f"    catalogues with the canonical spelling:  "
             f"{', '.join(r['catalogues_containing_canonical_spelling']) or 'none'}")

    L.append("")
    L.append("  EVIDENCE")
    rr = ev["request_roster"]
    L.append(f"    fetch roster contains station  {'yes' if rr['contains'] else 'NO'}")
    c = ev["fetch_cache"]
    L.append(f"    fetch cache            {c['note'] or (c['files'][-1] if c['files'] else '-')}")
    if c["readable"]:
        L.append(f"      cached readings      {c['readings']}"
                 f"   {c['first']} -> {c['last']}")
    rp = ev["transform_replay"]
    if rp["attempted"]:
        L.append(f"      transform replay     {rp['rows']} rows survive "
                 f"({rp['dropped']} dropped by the fetcher's own to_rows)")
    rf = ev["refetch_csv"]
    L.append(f"    refetch CSV            "
             f"{rf['rows']:,} rows  {rf['first'] or 'none'} -> {rf['last'] or 'none'}")
    cc = ev["committed_corpus"]
    L.append(f"    committed corpus       "
             f"{cc['rows']:,} rows  {cc['first'] or 'none'} -> {cc['last'] or 'none'}")
    ix = ev["station_index"]
    if ix:
        L.append(f"    station_index.json     {ix['hours']} hours  "
                 f"{ix['first']} -> {ix['last']}")
    else:
        L.append("    station_index.json     no entry")
    t = ev["live_trail"]
    L.append(f"    live MQTT trail        {t['records']} records for this id; "
             f"{t['distinct_live_ids']} distinct live ids use a different "
             f"('KT-') namespace,")
    L.append(f"                          so an absence there is expected and is "
             f"not evidence about this station")

    L.append("")
    L.append("  FINDING")
    for line in _wrap(report["detail"], 72):
        L.append(f"    {line}")
    L.append("")
    L.append("  ACTION")
    for line in _wrap(report["recommended_action"], 72):
        L.append(f"    {line}")
    if report.get("not_a_hardware_fault") is True:
        L.append("")
        L.append("    This is NOT a hardware fault and NOT a dead station. Do not")
        L.append("    send anyone to the site; fix the identifier.")

    L.append("")
    L.append("  FLEET CONTEXT")
    for label in ("committed_corpus", "refetch"):
        f = report["fleet"].get(label) or {}
        if "error" in f:
            L.append(f"    {label:<18} {f['error']}")
            continue
        L.append(f"    {label:<18} {f['verdict']}  "
                 f"{f['stations_observed']}/{f['stations_expected']} stations  "
                 f"newest {f['newest_observation']}")
        if f["missing_stations"]:
            L.append(f"    {'':<18} no rows: {', '.join(f['missing_stations'])}")
        if f["id_case_mismatches"]:
            for m in f["id_case_mismatches"]:
                L.append(f"    {'':<18} id case mismatch: roster {m['roster_id']!r} "
                         f"vs corpus {m['corpus_id']!r}")
        if f["silent_stations"]:
            silent = f["silent_stations"]
            # The committed corpus is stale as a whole, so naming sixteen silent
            # stations says nothing sixteen times. Say it once.
            shown = (", ".join(silent[:6]) + f", ... all {len(silent)}"
                     if len(silent) > 6 else ", ".join(silent))
            L.append(f"    {'':<18} not reporting: {shown}")
        if f["late_starters"]:
            L.append(f"    {'':<18} late starters: {', '.join(f['late_starters'])}")
    return "\n".join(L)


def _wrap(text: str, width: int) -> List[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines or [""]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Determine why a station is missing from the telemetry "
                    "corpus: never requested, absent upstream, empty response, "
                    "or filtered by the pipeline.")
    ap.add_argument("station", nargs="?", default=DEFAULT_FOCUS,
                    help=f"station id to investigate (default: {DEFAULT_FOCUS})")
    ap.add_argument("--corpus", default=cdf.DEFAULT_CSV,
                    help="committed corpus CSV")
    ap.add_argument("--refetch", default=cdf.DEFAULT_CURRENT_CSV,
                    help="refetched corpus CSV")
    ap.add_argument("--now", default=None, help="override the reference time")
    ap.add_argument("--json", dest="json_out", default=None,
                    help="write the machine-readable report here ('-' for stdout)")
    ap.add_argument("--quiet", action="store_true", help="suppress the text report")
    args = ap.parse_args(argv)

    now = cdf.parse_ts(args.now) if args.now else None
    if args.now and now is None:
        print(f"error: --now {args.now!r} is not valid ISO-8601", file=sys.stderr)
        return 3

    try:
        report = diagnose(args.station, args.corpus, args.refetch, now)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    if args.json_out:
        if args.json_out == "-":
            json.dump(report, sys.stdout, indent=2)
            sys.stdout.write("\n")
        else:
            path = cdf.write_report(report, args.json_out)
            if not args.quiet:
                print(f"json: {path}")
    if not args.quiet and args.json_out != "-":
        print(render(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
