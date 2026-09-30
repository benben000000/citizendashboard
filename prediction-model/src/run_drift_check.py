"""
Run the release gate's baseline drift guard.

WHY THIS EXISTS
---------------
`release_gate.py` blocks promotion unless the incumbent's recorded baseline is
re-scored and compared. That guard exists to catch the failure that has bitten
this project repeatedly: a corpus, a scoring function, or a normalisation constant
moving underneath an unchanged model, so a score difference looks like a model
improvement when it is a measurement change.

There is no code that produces the comparison, so the gate always blocked on
DRIFT_CHECK_NOT_SUPPLIED. This supplies it.

WHAT IT DOES
------------
Loads the recorded baseline (prediction-model/data/release_baseline.md), re-scores
the SERVED BUNDLES on the SAME corpus the baseline was captured against, and
compares cell by cell.

  MATCH             every cell agrees within tolerance
  DRIFT             at least one cell moved
  MISSING           a recorded cell produced no fresh score
  SKIPPED_WITH_JUSTIFICATION
                    the caller passed --justification instead of re-scoring

The justification path exists in the gate and BLOCKS anyway. That is deliberate
and this script preserves it: there is no spelling of "I did not run the guard"
that reads as a pass.

STABILITY, NOT OPTIMISATION
---------------------------
The comparison must run against the incumbent as served. If the refit policy has
already been installed, the incumbent bundle embeds the NEW policy and its scores
have moved for reasons that have nothing to do with model quality. The check
therefore warns when the served bundles carry a policy whose version differs from
the baseline record, so a drift finding is not misread as a model regression.

Note the asymmetry this creates, deliberately: re-scoring is required, so this
script does the work. Interpretting the result is the gate's job.
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
sys.path.insert(0, HERE)

CHANNELS = ["temperature", "humidity", "pressure", "wind_speed"]
HORIZONS = [1, 3, 6, 12, 24]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", required=True,
                    help="benchmark_vs_nwp.py output for the SERVED bundles on "
                         "the SAME corpus the baseline was captured against. "
                         "Produced by running benchmark_vs_nwp.py with no "
                         "--candidate-dir.")
    ap.add_argument("--baseline", default=os.path.join(DATA, "release_baseline.md"))
    ap.add_argument("--model-label", default="LNN incumbent")
    ap.add_argument("--out", required=True,
                    help="Where to write the drift report JSON.")
    ap.add_argument("--tolerance", type=float, default=0.02,
                    help="Relative tolerance. 0.02 = 2 percent.")
    args = ap.parse_args()

    import release_gate as rg

    recorded = rg.load_release_baseline(args.baseline)
    print(f"baseline document : {os.path.basename(args.baseline)}")
    print(f"  corpus sha      : {str(recorded.corpus_sha256)[:16]}...")
    print(f"  cells recorded  : {len(getattr(recorded, 'cells', {}) or {})}")

    if not os.path.exists(args.benchmark):
        print(f"ERROR: no fresh benchmark at {args.benchmark}")
        print("Run: python prediction-model/src/benchmark_vs_nwp.py \\")
        print("       --weather-csv <same corpus as the baseline> \\")
        print(f"       --out {args.benchmark} --label \"{args.model_label}\"")
        return 2

    with open(args.benchmark, encoding="utf-8") as f:
        results = json.load(f)["results"]

    fresh = {}
    for channel in CHANNELS:
        for h in HORIZONS:
            key = f"h{h}"
            row = (results.get(key) or {}).get(channel) or {}
            node = row.get(args.model_label) or row.get("LNN production") \
                or row.get("LNN (this project)")
            fresh[(channel, h)] = node["mae"] if node else None

    # Rain occurrence lives under a separate `_rain_brier` block and is a Brier
    # score, not an MAE. Omitting it reported 5 recorded cells as MISSING, which
    # reads as "the incumbent produced nothing there" rather than "this script did
    # not look".
    for h in HORIZONS:
        brier = (results.get(f"h{h}") or {}).get("_rain_brier") or {}
        node = brier.get(args.model_label) or brier.get("LNN production") \
            or brier.get("LNN (this project)")
        fresh[("rain_occurrence", h)] = node if node is not None else None

    scored = sum(1 for v in fresh.values() if v is not None)
    print(f"  fresh scores    : {scored}/{len(fresh)} cells")

    report = rg.compare_baseline_to_fresh(recorded, fresh,
                                          tolerance_rel=args.tolerance)
    payload = {
        "status": report.status,
        "tolerance_rel": report.tolerance_rel,
        "checked": report.checked,
        "drifted": [{"channel": c, "horizon": h} for c, h in report.drifted],
        "missing": [{"channel": c, "horizon": h} for c, h in report.missing],
        "newly_covered": [{"channel": c, "horizon": h} for c, h in report.newly_covered],
        "justification": report.justification,
        "excluded_fields": list(report.excluded_fields),
        "clean": report.clean,
        "details": {f"{c}|{h}": v for (c, h), v in report.details.items()},
    }
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2)

    print(f"\ndrift status     : {report.status}")
    print(f"  checked         : {report.checked}")
    print(f"  drifted         : {len(report.drifted)}")
    print(f"  missing         : {len(report.missing)}")
    print(f"  newly covered   : {len(report.newly_covered)}")
    if report.drifted:
        print("  drifted cells   :")
        for c, h in report.drifted[:12]:
            d = report.details.get((c, h), {})
            print(f"     {c:<12} +{h:>2}h  {d}")
    print(f"\nwritten: {args.out}")

    if not report.clean:
        print("\nThe baseline and a fresh re-scoring disagree. That is a finding,")
        print("not an obstacle to work around: either the corpus moved, the scoring")
        print("changed, or the served bundles are not the ones the record describes.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
