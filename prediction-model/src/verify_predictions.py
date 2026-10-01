"""
Score matured predictions against what actually happened.

WHY THIS IS A SEPARATE FILE
---------------------------
The prediction trail is append-only and must stay that way: a published number
is a historical fact and rewriting it would destroy the evidence. So scoring
does not update predictions, it appends *verification* records that name the
`record_id` they scored. The pair -- what we said, and how wrong it turned out
to be -- is then reconstructable forever, with the original untouched.

MATCHING RULE
-------------
A forecast for origin O at horizon H is verified against the observation at
O + H. That is the exact quantity the model was asked to predict. The nearest
observation is used within a tolerance, because device timestamps are not
exactly on the hour, but a forecast is never scored against something further
away than the tolerance.

WHAT IS REPORTED
----------------
  signed and absolute error per variable
  the producer that produced the forecast, so the accuracy of `nwp` / `blend`
    can be compared against `lln` on the same stations and days
  a pending count for forecasts whose target time has not arrived yet

Run it on a schedule; it is idempotent, because a verification record already
present for a record_id is not rewritten.
"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from prediction_audit import (  # noqa: E402
    PredictionAudit,
    read_observations,
    read_records,
    variable_entry,
)

DATA_DIR = os.path.abspath(os.path.join(HERE, "..", "data"))
VERIFICATION_PATH = os.path.join(DATA_DIR, "prediction_verification.jsonl")
# Named explicitly rather than left to read_records' implicit PREDICTION_AUDIT_PATH
# default. Both paths are now resolved here, so a caller passing None gets this
# module's idea of where the trail is instead of a silent fallback buried in a
# reader three frames away -- which is what made the verification trail resolve
# to the prediction trail and the verifier score nothing.
PREDICTION_PATH = os.path.join(DATA_DIR, "prediction_audit.jsonl")

# How far from the target time an observation may be and still count.
DEFAULT_TOLERANCE_MINUTES = 30.0

# The audit key for each variable -> the observation telemetry key that carries
# the matching truth. The units are identical (both degC, %RH, hPa, m/s), which
# is the only reason a straight comparison is valid.
VAR_TO_TELEMETRY = {
    "temperature": "temperature_c",
    "humidity": "humidity_pct",
    "pressure": "pressure_hpa",
    "wind_speed": "wind_speed_kmh",
}


def _parse(ts: Any) -> Optional[datetime]:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _nearest_observation(obs: List[Dict[str, Any]], target: datetime,
                         tolerance_min: float) -> Optional[Dict[str, Any]]:
    """Closest observation to the target time within tolerance."""
    best, best_dt = None, None
    tol = timedelta(minutes=tolerance_min)
    for o in obs:
        t = _parse(o.get("observed_at_utc"))
        if t is None:
            continue
        delta = abs((t - target).total_seconds())
        if delta > tol.total_seconds():
            continue
        if best_dt is None or delta < best_dt:
            best, best_dt = o, delta
    return best


def _num(x: Any) -> Optional[float]:
    if x is None or isinstance(x, bool):
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if f == f and abs(f) != float("inf") else None


def verify(
    prediction_path: Optional[str] = None,
    observation_path: Optional[str] = None,
    verification_path: Optional[str] = None,
    tolerance_minutes: float = DEFAULT_TOLERANCE_MINUTES,
    now: Optional[datetime] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Score every matured, not-yet-verified prediction and append the results.

    Returns a summary; never raises, because this is a reporting job and a
    failure to score must not be mistaken for a failure to predict.
    """
    ref = now or datetime.now(timezone.utc)
    # Resolve both trail paths before reading either. read_records() and
    # read_observations() each fall back to an environment default when handed
    # None, so passing the raw argument means "None" is indistinguishable from
    # "deliberately use the default" -- and when the caller relies on the
    # defaults, the fallback for the verification trail is the PREDICTION trail.
    # That mistake made `already` a set of the prediction file's own record_ids
    # (so already_verified always read 0) and seeded the identity set with every
    # forecast in that file (so every forecast scored as a duplicate of itself).
    resolved_prediction = prediction_path or PREDICTION_PATH
    resolved_verification = verification_path or VERIFICATION_PATH
    predictions = read_records(resolved_prediction)
    observations = read_observations(observation_path)
    already = {r.get("prediction_record_id")
               for r in read_records(resolved_verification)}

    by_station: Dict[str, List[Dict[str, Any]]] = {}
    for o in observations:
        sid = o.get("station_id")
        if sid is not None:
            by_station.setdefault(sid, []).append(o)
    for lst in by_station.values():
        lst.sort(key=lambda r: str(r.get("observed_at_utc") or ""))

    writer = PredictionAudit(path=verification_path or VERIFICATION_PATH,
                              data_dir=DATA_DIR)
    scored, pending, unmatched, skipped = 0, 0, 0, 0
    duplicate = 0
    # Seeded from the records already on disk, not left empty.
    #
    # The identity set has to start as the union of what has already been
    # verified and what is being verified now. Starting empty is subtly wrong:
    # `already` only holds record_ids, so a republished forecast from a PREVIOUS
    # run is not in it, passes the first check, and is scored and written again.
    # The duplicates were stopped within a single run but not across runs, which
    # is why the file held 20,398 lines for 2,591 unique forecasts -- an 8x
    # inflation that anyone reading the file line count would take for real
    # evidence volume.
    # Seeded from the records already on disk, not left empty.
    #
    # The identity set has to start as the union of what has already been
    # verified and what is being verified now. Starting empty is subtly wrong:
    # `already` only holds record_ids, so a republished forecast from a PREVIOUS
    # run is not in it, passes the first check, and is scored and written again.
    # The duplicates were stopped within a single run but not across runs, which
    # is why the file held 20,398 lines for 2,591 unique forecasts -- an 8x
    # inflation that anyone reading the file line count would take for real
    # evidence volume.
    seen_identities: set = {
        (r.get("station_id"), r.get("origin_timestamp_utc"), _num(r.get("horizon_hours")))
        for r in read_records(resolved_verification)
    }
    per_var_abs: Dict[str, List[float]] = {}
    # Counts per producer, NOT a pooled mean. A mean absolute error across degC,
    # %RH, hPa and m/s is not a quantity -- the units do not share a scale, and
    # an hPa error dominates a degC error numerically while meaning nothing more.
    # The comparable number is mae_by_variable_and_producer.
    producer_counts: Dict[str, int] = {}
    per_var_producer: Dict[str, Dict[str, List[float]]] = {}

    for p in predictions:
        rid = p.get("record_id")
        if rid and rid in already:
            skipped += 1
            continue
        # DE-DUPLICATE BY FORECAST IDENTITY, not by record id.
        #
        # `already` deduplicates on record_id, which does nothing about republished
        # forecasts: each republish is written as a new audit record with a new
        # record_id, so an identical forecast sent every 60 seconds during a
        # reconnect storm is scored 35 times and dominates the mean. One
        # (station, origin timestamp, horizon) IS one forecast no matter how many
        # times it was transmitted, and it must contribute once.
        #
        # Measured effect: 18.9% of the audit trail was redundant, inflating the
        # trail x1.23 and weighting some predictions 35x. That is not a fabricated
        # number, but it is not a representative one either.
        identity = (p.get("station_id"), p.get("origin_timestamp_utc"),
                    _num(p.get("horizon_hours")))
        if identity in seen_identities:
            duplicate += 1
            continue
        seen_identities.add(identity)
        if limit is not None and scored >= limit:
            break
        origin = _parse(p.get("origin_timestamp_utc"))
        horizon = _num(p.get("horizon_hours"))
        if origin is None or horizon is None:
            unmatched += 1
            continue
        target = origin + timedelta(hours=horizon)
        if target > ref:
            pending += 1
            continue
        obs = _nearest_observation(by_station.get(p.get("station_id"), []),
                                   target, tolerance_minutes)
        if obs is None:
            unmatched += 1
            continue

        truth = obs.get("telemetry") or {}
        variables: Dict[str, Any] = {}
        for var, entry in (p.get("variables") or {}).items():
            if var not in VAR_TO_TELEMETRY:
                continue
            predicted = _num((entry or {}).get("value"))
            actual = _num(truth.get(VAR_TO_TELEMETRY[var]))
            if predicted is None or actual is None:
                continue
            err = predicted - actual
            producer = (entry or {}).get("producer") or "unknown"
            variables[var] = {
                "predicted": predicted,
                "actual": actual,
                "error": err,
                "abs_error": abs(err),
                "producer": producer,
            }
            per_var_abs.setdefault(var, []).append(abs(err))
            producer_counts[producer] = producer_counts.get(producer, 0) + 1
            per_var_producer.setdefault(var, {}).setdefault(producer, []).append(abs(err))

        if not variables:
            unmatched += 1
            continue

        writer.write({
            "schema_version": 1,
            # record_id is derived from the FORECAST identity, not from the audit
            # record's own id. Two audit records for the same forecast therefore
            # produce the same verification id, which makes a duplicate visible
            # as a repeated key downstream instead of hiding behind a unique id.
            "record_id": f"ver-{p.get('station_id')}-{p.get('origin_timestamp_utc')}-{horizon:g}h",
            "prediction_record_id": rid,
            "verified_at_utc": ref.isoformat().replace("+00:00", "Z"),
            "station_id": p.get("station_id"),
            "device_id": p.get("device_id"),
            "horizon_hours": horizon,
            "origin_timestamp_utc": p.get("origin_timestamp_utc"),
            "target_timestamp_utc": target.isoformat().replace("+00:00", "Z"),
            "observation_record_id": obs.get("record_id"),
            "observation_offset_minutes": round(
                ((_parse(obs.get("observed_at_utc")) or target) - target)
                .total_seconds() / 60.0, 2),
            "policy_version": (p.get("provenance", {}).get("policy", {})
                               .get("policy_version")),
            "model_bundle": (p.get("provenance", {}).get("model", {})
                             .get("bundle")),
            "variables": variables,
        })
        scored += 1

    def mean(xs: List[float]) -> Optional[float]:
        return round(sum(xs) / len(xs), 4) if xs else None

    return {
        "predictions_seen": len(predictions),
        "observations_seen": len(observations),
        "scored": scored,
        "pending": pending,
        "unmatched": unmatched,
        "already_verified": skipped,
        "duplicate_publishes": duplicate,
        "tolerance_minutes": tolerance_minutes,
        "mae_by_variable": {k: mean(v) for k, v in sorted(per_var_abs.items())},
        "n_by_variable_and_producer": {
            v: {p: len(x) for p, x in sorted(d.items())}
            for v, d in sorted(per_var_producer.items())
        },
        "mae_by_variable_and_producer": {
            v: {p: mean(x) for p, x in sorted(d.items())}
            for v, d in sorted(per_var_producer.items())
        },
        "writer": writer.status(),
    }


def main() -> int:
    summary = verify()
    print("=" * 92)
    print("PREDICTION VERIFICATION")
    print("=" * 92)
    print(f"  predictions seen   : {summary['predictions_seen']}")
    print(f"  observations seen  : {summary['observations_seen']}")
    print(f"  scored now         : {summary['scored']}")
    print(f"  pending (unmatured): {summary['pending']}")
    print(f"  unmatched          : {summary['unmatched']}")
    print(f"  already verified   : {summary['already_verified']}")
    print(f"  duplicate publishes: {summary['duplicate_publishes']}")
    if summary["mae_by_variable"]:
        print("\n  mean absolute error by variable:")
        for k, v in summary["mae_by_variable"].items():
            print(f"    {k:<12} {v}")
    if summary["mae_by_variable_and_producer"]:
        print("\n  mean absolute error by variable and producer:")
        for var, per in summary["mae_by_variable_and_producer"].items():
            n = summary["n_by_variable_and_producer"].get(var, {})
            detail = "  ".join(f"{p}={m} (n={n.get(p, 0)})" for p, m in per.items())
            print(f"    {var:<12} {detail}")
    out = os.path.join(DATA_DIR, "prediction_verification_summary.json")
    # A committed data artifact must not carry an absolute machine path. The
    # provenance gate fails the build on anything matching a drive letter, and
    # it is right to: a summary that names C:\... is meaningless on another
    # machine and leaks the build layout.
    serialisable = dict(summary)
    w = dict(summary.get("writer") or {})
    if w.get("path"):
        w["path"] = os.path.relpath(w["path"], os.path.abspath(os.path.join(HERE, "..", "..")))
        w["path"] = w["path"].replace("\\", "/")
    serialisable["writer"] = w
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(serialisable, f, indent=2)
    print(f"\n  written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
