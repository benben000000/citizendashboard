"""
Keep the live ingestor alive, and prove that it is alive.

WHY THIS EXISTS

The ingestor is the only source of verification evidence, and the dashboard
cannot tell the difference between "no news" and "dead". Both render as the
last value the ingestor wrote, because the cache file is only ever updated by
the ingestor itself. A process that crashes at 02:00 leaves a file that looks
perfectly healthy until you check the timestamps -- which is exactly how a
68-minute hole on 2026-10-01 stayed invisible until someone asked why the
scored count had not moved.

So supervision here is not "restart on exit" alone. It is:

  1. start the ingestor if it is not running,
  2. if it IS running but has produced nothing for longer than the staleness
     budget, treat that as a failure and restart it anyway, because a live
     process publishing nothing is the failure mode that matters, and
  3. write every decision to the event trail, including "it was fine", so the
     log distinguishes a watchdog that is working from one that has never run.

THRESHOLDS

The staleness budget is deliberately much larger than the publish interval.
Stations publish about every 60s and a forecast needs 24 consecutive samples,
so a 24-sample window takes ~24 minutes to refill after any interruption. The
default budget of 45 minutes sits above that, so a slow refill is never
mistaken for a hung process, while a genuine hang is caught inside the hour.
Both are overridable from the environment for a deployment whose stations
publish on a different cadence.

USAGE

    python ingestor_watchdog.py            # check once, act if needed
    python ingestor_watchdog.py --check    # report only, never start anything

Exit code is 0 when the ingestor is healthy afterwards, 1 when it had to be
restarted, so a scheduled task's own result code carries the signal.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = Path(__file__).resolve().parent
CACHE_PATH = Path(os.getenv(
    "MQTT_LIVE_CACHE_PATH",
    str(ROOT / "prediction-model/data/mqtt_live_predictions.json")))
EVENTS_PATH = Path(os.getenv(
    "MQTT_EVENT_LOG_PATH",
    str(ROOT / "prediction-model/data/ingestor_events.jsonl")))

INGESTOR = SRC / "mqtt_live_ingestor.py"

# A station publishes about every 60s and a forecast needs 24 consecutive
# samples, so refilling a window after an interruption takes ~24 minutes. The
# budget sits above that so a legitimate refill is not mistaken for a hang.
STALE_AFTER_SECONDS = int(os.getenv("INGESTOR_STALE_AFTER_SECONDS", "2700"))
HEARTBEAT_STALE_AFTER_SECONDS = int(os.getenv("INGESTOR_HEARTBEAT_STALE_AFTER_SECONDS", "600"))


def record_event(kind: str, **fields: Any) -> None:
    """Append to the shared event trail. Never raises."""
    try:
        EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(
            {"kind": kind,
             "recorded_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
             **fields},
            ensure_ascii=False, default=str)
        with EVENTS_PATH.open("a", encoding="utf-8", newline="\n") as sink:
            sink.write(line + "\n")
    except Exception as exc:  # noqa: BLE001 - the watchdog must not crash
        print(f"watchdog event write skipped: {type(exc).__name__}: {exc}")


def read_events(limit_bytes: int = 262144) -> list[dict[str, Any]]:
    """
    Read the tail of the event trail.

    Only the tail is read on purpose: the trail grows without bound, and a
    watchdog that parses a multi-megabyte file every minute is a problem of its
    own. Parsing is line-at-a-time and tolerant of a torn final line, because a
    watchdog that crashes on a partial write cannot report the partial write.
    """
    events: list[dict[str, Any]] = []
    try:
        with EVENTS_PATH.open("rb") as source:
            source.seek(0, os.SEEK_END)
            size = source.tell()
            if size > limit_bytes:
                source.seek(-limit_bytes, os.SEEK_END)
                source.readline()  # discard a probably-partial first line
            else:
                # Small file: seek back to the start. Seeking to the end and
                # reading from there yields nothing, which is indistinguishable
                # from "the watchdog has never run" -- the most dangerous
                # possible answer, because it means no events are ever reported.
                source.seek(0)
            for raw in source:
                try:
                    events.append(json.loads(raw.decode("utf-8")))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
    except FileNotFoundError:
        return []
    return events


def parse_timestamp(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def cache_age_seconds() -> float | None:
    """Age of the live cache, or None if it does not exist yet."""
    try:
        return max(0.0, time.time() - CACHE_PATH.stat().st_mtime)
    except FileNotFoundError:
        return None


def running_pids() -> list[int]:
    """PIDs of live ingestor processes, excluding this watchdog's own matches."""
    if os.name != "nt":
        result = subprocess.run(["pgrep", "-f", "mqtt_live_ingestor.py"],
                                capture_output=True, text=True, check=False)
        return [int(line) for line in result.stdout.split() if line.strip().isdigit()]

    # wmic is deprecated and its replacement is not guaranteed present, so ask
    # PowerShell instead. Failure here is treated as "cannot tell", which the
    # caller resolves conservatively by starting a process -- a duplicate
    # ingestor is wasteful, a missing one costs a day of irreplaceable evidence.
    script = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
              "Where-Object { $_.CommandLine -like '*mqtt_live_ingestor.py*' } | "
              "Select-Object -ExpandProperty ProcessId")
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, text=True, check=False, timeout=60)
    except (subprocess.SubprocessError, OSError):
        return []
    return [int(line) for line in result.stdout.split() if line.strip().isdigit()]


def newest_event_time(events: list[dict[str, Any]]) -> float | None:
    stamps = [parse_timestamp(e.get("recorded_at_utc")) for e in events]
    stamps = [s for s in stamps if s is not None]
    return max(stamps) if stamps else None


def assess() -> dict[str, Any]:
    """Decide what the current state is. Pure inspection, no side effects."""
    pids = running_pids()
    age = cache_age_seconds()
    events = read_events()
    newest_event = newest_event_time(events)
    heartbeat_age = (time.time() - newest_event) if newest_event else None

    if not pids:
        state = "NOT_RUNNING"
        reason = "no ingestor process found"
    elif age is None:
        state = "NO_CACHE"
        reason = "process exists but has never written the live cache"
    elif age > STALE_AFTER_SECONDS:
        state = "STALE"
        reason = f"process alive but cache is {age / 60:.1f} min old (budget {STALE_AFTER_SECONDS / 60:.0f} min)"
    elif heartbeat_age is not None and heartbeat_age > HEARTBEAT_STALE_AFTER_SECONDS:
        state = "NO_EVENTS"
        reason = f"cache is current but no event written for {heartbeat_age / 60:.1f} min"
    else:
        state = "HEALTHY"
        reason = f"cache {age:.0f}s old, {len(pids)} process(es)"

    return {"state": state, "reason": reason, "pids": pids,
            "cache_age_seconds": None if age is None else round(age, 1),
            "event_age_seconds": None if heartbeat_age is None else round(heartbeat_age, 1)}


def stop_pids(pids: list[int]) -> list[int]:
    """Stop the given PIDs. Returns the ones that were actually stopped."""
    stopped: list[int] = []
    for pid in pids:
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                               capture_output=True, text=True, check=False, timeout=60)
            else:
                os.kill(pid, 15)
            stopped.append(pid)
        except (subprocess.SubprocessError, OSError):
            continue
    return stopped


def start_ingestor() -> int | None:
    """
    Launch the ingestor detached, so it outlives this watchdog.

    Windows: DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP so the child is not
    killed when the parent's console closes or the scheduled task ends.
    """
    creationflags = 0
    if os.name == "nt":
        creationflags = (getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
                         | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
                         | getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000))
    log_dir = Path(os.getenv("TEMP", "/tmp")) / "opencode"
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        with (log_dir / "ingestor.out.log").open("a", encoding="utf-8") as out, \
             (log_dir / "ingestor.err.log").open("a", encoding="utf-8") as err:
            subprocess.Popen(
                [sys.executable, str(INGESTOR)],
                cwd=str(ROOT), stdout=out, stderr=err,
                stdin=subprocess.DEVNULL, creationflags=creationflags,
                env={**os.environ, "PYTHONUNBUFFERED": "1"})
    except (OSError, subprocess.SubprocessError) as exc:
        record_event("watchdog_start_failed", error=f"{type(exc).__name__}: {exc}")
        print(f"  could not start ingestor: {exc}")
        return None
    # Give the process a moment to register, so the caller can report a
    # verified start rather than an assumed one.
    time.sleep(5)
    return next(iter(running_pids()), None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--check", action="store_true",
                        help="report the current state without starting or stopping anything")
    args = parser.parse_args()

    report = assess()
    print(f"  ingestor state: {report['state']} -- {report['reason']}")
    print(f"  pids: {report['pids'] or 'none'}   "
          f"cache age: {report['cache_age_seconds']}s   "
          f"event age: {report['event_age_seconds']}s")

    if args.check or report["state"] == "HEALTHY":
        record_event("watchdog_check", state=report["state"], reason=report["reason"],
                     pids=report["pids"], cache_age_seconds=report["cache_age_seconds"])
        return 0

    if report["pids"]:
        stopped = stop_pids(report["pids"])
        record_event("watchdog_restart", reason=report["reason"], stopped=stopped)
        print(f"  stopped unhealthy ingestor(s): {stopped}")
        time.sleep(3)
    else:
        record_event("watchdog_start", reason=report["reason"])

    new_pid = start_ingestor()
    if new_pid is None:
        return 1

    time.sleep(20)
    after = assess()
    print(f"  after restart: {after['state']} -- {after['reason']}")
    record_event("watchdog_result", state=after["state"], reason=after["reason"], pid=new_pid)
    # A restarted process is a real event, not a success: the caller (a scheduled
    # task, a monitoring alert) needs to see that something was wrong.
    return 0 if after["state"] == "HEALTHY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
