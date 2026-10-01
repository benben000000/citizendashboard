"""
Fifteen-minute pipeline status digest.

WHY THIS IS A FILE AND NOT A MESSAGE

This process cannot message anyone. It runs on a schedule, writes what it
finds, and gets out of the way. A digest that claims to "alert" someone while
having no channel to reach them is worse than no digest, because it manufactures
the impression of supervision. So this reports facts to a file, and the file is
the deliverable.

WHAT IT REPORTS, AND WHY EACH LINE IS HERE

Every line answers a question that has already bitten us, or answers one that
would otherwise be unanswerable during a pitch:

  * ingestor health  -- a live-but-silent process looks identical to a working
    one, because the cache is only ever written by the ingestor
  * scored vs pending -- distinguishes "nothing has matured yet" (expected,
    a function of elapsed time) from "nobody has looked" (a bug, and the
    single most expensive mistake made in this project)
  * producer split -- shows how much of the reported accuracy is the model and
    how much is persistence, which is the number most likely to be
    over-claimed and hardest to reconstruct later
  * per-horizon counts -- makes the coverage gaps explicit instead of letting
    a reader assume all five horizons are equally evidenced
  * recent events -- outages, sequence resets, incomplete frames: the causes of
    a coverage gap, recorded when they happened

Coverage percentages are reported as fractions of what is MATURE rather than as
a bare total, because a large pending pool is healthy and a stalled scorer is
not, and those two look identical in a raw count.

Usage:
    python pipeline_status.py            # append a digest, print it
    python pipeline_status.py --json     # machine-readable, for the dashboard
    python pipeline_status.py --quiet    # append only
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "prediction-model/data"
EVENTS = DATA / "ingestor_events.jsonl"
STATUS_PATH = Path(os.getenv("PIPELINE_STATUS_PATH", str(DATA / "pipeline_status.jsonl")))
LATEST_PATH = Path(os.getenv("PIPELINE_STATUS_LATEST", str(DATA / "pipeline_status_latest.md")))

HORIZONS = (1, 3, 6, 12, 24)
VARIABLES = ("temperature", "humidity", "pressure", "wind_speed")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as source:
            for line in source:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return []
    return records


def parse_ts(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def file_age_minutes(path: Path) -> float | None:
    try:
        return (datetime.now().timestamp() - path.stat().st_mtime) / 60.0
    except FileNotFoundError:
        return None


def count_files(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as source:
            return sum(1 for line in source if line.strip())
    except FileNotFoundError:
        return 0


def collect() -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    verification = read_jsonl(DATA / "prediction_verification.jsonl")
    audit = read_jsonl(DATA / "prediction_audit.jsonl")
    events = read_jsonl(EVENTS)
    summary_path = DATA / "prediction_verification_summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        summary = {}

    by_horizon: dict[int, dict[str, Any]] = {}
    for record in verification:
        try:
            horizon = int(float(record.get("horizon_hours")))
        except (TypeError, ValueError):
            continue
        bucket = by_horizon.setdefault(
            horizon, {"n": 0, "stations": set(), "producers": collections.Counter()})
        # ONE record per (station, horizon) -- it holds four variable entries, so
        # counting variables here would inflate every figure fourfold. The
        # producer split is counted over variables because that is the only
        # level at which a producer is attributed, but it is reported as a
        # separate series and never added to "scored".
        bucket["n"] += 1
        if record.get("station_id"):
            bucket["stations"].add(record["station_id"])
        for detail in (record.get("variables") or {}).values():
            bucket["producers"][detail.get("producer") or "unknown"] += 1

    # Events in the last 15 minutes, so consecutive digests do not repeat.
    recent_events = []
    for event in events:
        moment = parse_ts(event.get("recorded_at_utc"))
        if moment and (now - moment).total_seconds() <= 900:
            recent_events.append({"kind": event.get("kind"),
                                  "station_id": event.get("station_id"),
                                  "recorded_at_utc": event.get("recorded_at_utc")})

    health = classify_health(file_age_minutes(DATA / "mqtt_live_predictions.json"),
                             file_age_minutes(DATA / "observation_audit.jsonl"),
                             file_age_minutes(DATA / "prediction_audit.jsonl"))

    return {
        "generated_at_utc": now.isoformat().replace("+00:00", "Z"),
        "health": health,
        "totals": {
            "scored": int(summary.get("scored") or len(verification)),
            "pending": int(summary.get("pending") or 0),
            "duplicate_publishes_excluded": int(summary.get("duplicate_publishes") or 0),
            "predictions_published": len(audit),
            "observations_recorded": count_files(DATA / "observation_audit.jsonl"),
            "stations_verified": len({r.get("station_id") for r in verification
                                      if r.get("station_id")}),
        },
        "by_horizon": {
            str(h): {
                "scored": by_horizon.get(h, {}).get("n", 0),
                "stations": len(by_horizon.get(h, {}).get("stations", set())),
                "producers": dict(by_horizon.get(h, {}).get("producers", {})),
            } for h in HORIZONS
        },
        "recent_events": recent_events[-12:],
        "files_min_since_write": {
            name: (None if (age := file_age_minutes(DATA / name)) is None else round(age, 1))
            for name in ("mqtt_live_predictions.json", "observation_audit.jsonl",
                         "prediction_audit.jsonl", "prediction_verification.jsonl")
        },
    }


def classify_health(cache_age: float | None, obs_age: float | None,
                    pred_age: float | None) -> dict[str, Any]:
    """
    Grade the pipeline, and say which check failed.

    Thresholds come from the ingestor's own operating parameters rather than
    round numbers: stations publish about every 60s and a forecast needs 24
    consecutive samples, so a 24-sample window takes about 24 minutes to refill
    after any interruption. Anything past 45 minutes is a hang, not a refill.
    """
    problems: list[str] = []
    if cache_age is None:
        problems.append("live cache has never been written")
    elif cache_age > 45:
        problems.append(f"live cache stale: {cache_age:.0f} min old")
    if obs_age is not None and obs_age > 10:
        problems.append(f"observations stopped: {obs_age:.0f} min old")
    if pred_age is not None and pred_age > 45:
        problems.append(f"predictions stopped: {pred_age:.0f} min old")
    return {"state": "OK" if not problems else "ATTENTION",
            "problems": problems}


def render_markdown(report: dict[str, Any]) -> str:
    totals = report["totals"]
    health = report["health"]
    lines = [
        f"## Pipeline status -- {report['generated_at_utc']}",
        "",
        f"**{health['state']}**" + ("" if health["state"] == "OK"
                                    else " -- " + "; ".join(health["problems"])),
        "",
        f"- scored: **{totals['scored']}**   pending: {totals['pending']}   "
        f"duplicate publishes excluded: {totals['duplicate_publishes_excluded']}",
        f"- stations verified: **{totals['stations_verified']}**   "
        f"predictions published: {totals['predictions_published']}",
        "",
        "| horizon | scored forecasts | stations | variable entries by producer |",
        "|---|---|---|---|",
    ]
    for horizon in HORIZONS:
        cell = report["by_horizon"][str(horizon)]
        split = cell["producers"]
        if not split:
            split_text = "no matured records yet"
        else:
            split_text = ", ".join(f"{k} {v}" for k, v in sorted(split.items(),
                                                                 key=lambda kv: -kv[1]))
        lines.append(f"| +{horizon}h | {cell['scored']} | {cell['stations']} | {split_text} |")
    lines += [
        "",
        "Counts are forecasts (one per station and horizon); the producer split is "
        "counted over variable entries, so it is roughly four times larger and is "
        "not comparable to the scored column.",
    ]

    lines += ["", "### Recent pipeline events (last 15 min)"]
    if report["recent_events"]:
        for event in report["recent_events"]:
            station = f" {event['station_id']}" if event.get("station_id") else ""
            lines.append(f"- `{event['kind']}`{station} at {event['recorded_at_utc']}")
    else:
        lines.append("- none")

    lines += ["", "### File freshness (minutes since last write)"]
    for name, age in report["files_min_since_write"].items():
        lines.append(f"- `{name}`: {'never' if age is None else f'{age:.1f} min'}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--json", action="store_true", help="emit JSON instead of markdown")
    parser.add_argument("--quiet", action="store_true", help="write files, print nothing")
    args = parser.parse_args()

    report = collect()
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with STATUS_PATH.open("a", encoding="utf-8", newline="\n") as sink:
        sink.write(json.dumps(report, ensure_ascii=False, default=str) + "\n")
    markdown = render_markdown(report)
    LATEST_PATH.write_text(markdown, encoding="utf-8", newline="\n")

    if not args.quiet:
        print(markdown if args.json is False else json.dumps(report, indent=2, default=str))
    return 0 if report["health"]["state"] == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
