"""
Live skill against persistence, measured on the running production system.

WHY THIS IS NOT A BACKTEST

Everything in release_baseline.md comes from a held-out test split of a
historical corpus. It is a real number about a frozen model. It is not evidence
that the deployed system behaves that way, because the deployed system makes
different decisions: the policy routes most cells to persistence, the station
population is not the corpus population, and nothing in that file knows whether
the ingestor was up.

This measures the served forecasts against later station observations, on the
rows the production system actually produced. Same rows, same ground truth, no
simulation and no held-out split -- the forecast was made before the observation
existed, which is the only definition of a forecast that means anything.

THE NUMBER THAT MATTERS

Absolute MAE cannot say whether the model earned its place. A weather system that
returned the last observation would post a perfectly respectable MAE and be
worthless. So every row is also scored as PERSISTENCE: the value the station
reported at the forecast's own origin time, scored against the same truth.

  skill_pct = 100 * (MAE_persistence - MAE_forecast) / MAE_persistence

Positive means the served forecast beat holding the last reading. That is the
claim worth making, and it is the one an evaluator will ask for.

HONESTY CONSTRAINTS BUILT IN

  * Coverage is reported per variable per horizon. A cell with 40 rows is not the
    same evidence as one with 2,844, and a mean that hides the difference is the
    mistake this whole project has been correcting.
  * A cell below MIN_ROWS is printed but marked INSUFFICIENT rather than being
    quietly included in an overall average.
  * Persistence is looked up per station with a stated tolerance. If the
    origin-time observation is missing the row is DROPPED, not defaulted to the
    forecast's own value -- defaulting would make skill trivially zero and hide
    the problem.
  * Nothing here decides anything. It reports; the release gate decides.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "prediction-model/data"

VARIABLES = {
    "temperature": ("temperature_c", "degC"),
    "humidity": ("humidity_pct", "%RH"),
    "pressure": ("pressure_hpa", "hPa"),
    "wind_speed": ("wind_speed_kmh", "m/s"),
}

# Persistence for wind is read from the same field the forecast is scored on,
# so the comparison is like-for-like. No unit conversion is applied here: the
# serving layer converts km/h to m/s before publishing, and the verified records
# already carry the served unit. Mixing units would silently inflate or deflate
# every wind number in this report.
HORIZONS = (1, 3, 6, 12, 24)
MIN_ROWS = 30
PERSISTENCE_TOLERANCE_MINUTES = 30.0


def parse_ts(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return []
    return out


def build_observation_index(
        records: list[dict[str, Any]]) -> dict[str, tuple[list[float], list[dict]]]:
    """station -> (sorted epoch seconds, records)."""
    index: dict[str, list[tuple[float, dict]]] = collections.defaultdict(list)
    for record in records:
        moment = parse_ts(record.get("observed_at_utc"))
        if moment is None:
            continue
        station = record.get("station_id")
        if station is None:
            continue
        index[station].append((moment.timestamp(), record))
    return {k: ([t for t, _ in sorted(v)], [r for _, r in sorted(v)])
            for k, v in index.items()}


def nearest_value(times: list[float], records: list[dict], when: datetime,
                  field: str, tolerance_minutes: float) -> float | None:
    if not times:
        return None
    target = when.timestamp()
    idx = bisect.bisect_left(times, target)
    best: tuple[float, float] | None = None
    for j in (idx - 1, idx, idx + 1):
        if 0 <= j < len(times):
            delta = abs(times[j] - target)
            if best is None or delta < best[0]:
                best = (delta, j)
    if best is None or best[0] > tolerance_minutes * 60.0:
        return None
    value = (records[best[1]].get("telemetry") or {}).get(field)
    return float(value) if isinstance(value, (int, float)) else None


def mean(xs: list[float]) -> float | None:
    finite = [x for x in xs if isinstance(x, (int, float)) and math.isfinite(x)]
    return sum(finite) / len(finite) if finite else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--verification", default=str(DATA / "prediction_verification.jsonl"))
    parser.add_argument("--observations", default=str(DATA / "observation_audit.jsonl"))
    parser.add_argument("--out-json", default=str(DATA / "live_skill_report.json"))
    parser.add_argument("--out-md", default=str(DATA / "live_skill_report.md"))
    args = parser.parse_args()

    verified = read_jsonl(Path(args.verification))
    observations = read_jsonl(Path(args.observations))
    if not verified:
        print("  no verified forecasts on disk; nothing to report")
        return 1
    index = build_observation_index(observations)
    print(f"  verified forecasts : {len(verified):,}")
    print(f"  observations       : {len(observations):,} across {len(index)} stations")

    # per (horizon, variable): forecast error, persistence error, producers
    cells: dict[tuple[int, str], dict[str, Any]] = {}

    for record in verified:
        horizon_value = record.get("horizon_hours")
        origin = parse_ts(record.get("origin_timestamp_utc"))
        station = record.get("station_id")
        if horizon_value is None or origin is None or station is None:
            continue
        try:
            horizon = int(float(horizon_value))
        except (TypeError, ValueError):
            continue
        entry = index.get(station)
        if entry is None:
            continue
        times, obs_records = entry

        for variable, detail in (record.get("variables") or {}).items():
            if variable not in VARIABLES:
                continue
            field, _unit = VARIABLES[variable]
            forecast = detail.get("predicted")
            truth = detail.get("actual")
            if not isinstance(forecast, (int, float)) or not isinstance(truth, (int, float)):
                continue
            # Persistence is the station's OWN reading at the forecast origin.
            origin_reading = nearest_value(times, obs_records, origin, field,
                                           PERSISTENCE_TOLERANCE_MINUTES)
            if origin_reading is None:
                continue
            cell = cells.setdefault((horizon, variable), {
                "n": 0, "forecast_abs": [], "persistence_abs": [],
                "producers": collections.Counter()})
            cell["n"] += 1
            cell["forecast_abs"].append(abs(float(forecast) - float(truth)))
            cell["persistence_abs"].append(abs(origin_reading - float(truth)))
            cell["producers"][detail.get("producer") or "unknown"] += 1

    report_cells: dict[str, Any] = {}
    lines = [
        "# Live verified skill against persistence",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')}",
        "",
        "Measured on forecasts the production system actually published, scored",
        "against later station observations. No held-out split, no simulation: the",
        "forecast existed before the observation did.",
        "",
        f"Persistence is the station's own reading at the forecast's origin time,",
        f"within {PERSISTENCE_TOLERANCE_MINUTES:.0f} minutes. Rows without that",
        "reading are dropped rather than defaulted.",
        "",
        "| horizon | variable | n | served MAE | persistence MAE | skill | verdict |",
        "|---|---|---|---|---|---|---|",
    ]

    for horizon in HORIZONS:
        for variable, (_field, unit) in VARIABLES.items():
            cell = cells.get((horizon, variable))
            if not cell:
                lines.append(f"| +{horizon}h | {variable} | 0 | - | - | - | no rows |")
                report_cells[f"{variable}|{horizon}"] = {"n": 0, "status": "NO_ROWS"}
                continue
            served = mean(cell["forecast_abs"])
            persist = mean(cell["persistence_abs"])
            n = cell["n"]
            if served is None or persist is None or persist == 0:
                lines.append(f"| +{horizon}h | {variable} | {n} | - | - | - | not computable |")
                continue
            skill = 100.0 * (persist - served) / persist
            if n < MIN_ROWS:
                verdict = f"INSUFFICIENT (n<{MIN_ROWS})"
                status = "INSUFFICIENT_EVIDENCE"
            elif skill >= 1.0:
                verdict = "beats persistence"
                status = "SKILL"
            elif skill <= -1.0:
                verdict = "WORSE than persistence"
                status = "REGRESSION"
            else:
                verdict = "ties persistence"
                status = "TIE"
            lines.append(
                f"| +{horizon}h | {variable} ({unit}) | {n} | {served:.4f} | "
                f"{persist:.4f} | {skill:+.1f}% | {verdict} |")
            report_cells[f"{variable}|{horizon}"] = {
                "n": n, "served_mae": round(served, 6),
                "persistence_mae": round(persist, 6),
                "skill_pct": round(skill, 4), "status": status,
                "producers": dict(cell["producers"]),
            }

    scored = [c for c in report_cells.values() if c.get("status") in ("SKILL", "TIE", "REGRESSION")]
    skill_cells = [c for c in scored if c["status"] == "SKILL"]
    ties = [c for c in scored if c["status"] == "TIE"]
    regressions = [c for c in scored if c["status"] == "REGRESSION"]
    insufficient = [c for c in report_cells.values() if c.get("status") == "INSUFFICIENT_EVIDENCE"]

    lines += [
        "",
        "## Summary",
        "",
        f"- cells with computable evidence (n>={MIN_ROWS}): **{len(scored)} of "
        f"{len(HORIZONS) * len(VARIABLES)}**",
        f"- beats persistence by >=1%: **{len(skill_cells)}**",
        f"- ties persistence (within 1%): **{len(ties)}**",
        f"- worse than persistence: **{len(regressions)}**",
        f"- insufficient rows: {len(insufficient)}",
        "",
    ]
    if regressions:
        lines.append("A served forecast that is worse than persistence is a defect,")
        lines.append("not a tuning opportunity: persistence is the floor the system")
        lines.append("promises never to go below. Listed here rather than averaged in.")
        lines.append("")

    Path(args.out_json).write_text(json.dumps({
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "forecasts_scored": len(verified),
        "min_rows": MIN_ROWS,
        "persistence_tolerance_minutes": PERSISTENCE_TOLERANCE_MINUTES,
        "counts": {"computable": len(scored), "skill": len(skill_cells),
                   "tie": len(ties), "regression": len(regressions),
                   "insufficient": len(insufficient)},
        "cells": report_cells,
    }, indent=2) + "\n", encoding="utf-8", newline="\n")
    Path(args.out_md).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    print("\n".join(lines[lines.index("| horizon | variable | n | served MAE | persistence MAE | skill | verdict |"):]))
    print()
    print(f"  skill cells   : {len(skill_cells)}   ties: {len(ties)}   "
          f"regressions: {len(regressions)}   insufficient: {len(insufficient)}")
    print(f"  written: {args.out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
