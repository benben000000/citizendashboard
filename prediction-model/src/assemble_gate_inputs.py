"""
Assemble the release gate's single decision-input bundle.

WHY THIS IS A SEPARATE STEP

`release_gate.py --inputs` takes ONE JSON document, but the decision inputs are
produced by two tools that each write their own file:

  * `build_promotion_evidence.py` writes the paired challenger/incumbent cells,
  * `run_drift_check.py` writes the incumbent re-scoring used by the drift guard.

Hand-merging them is how a stale or half-populated bundle reaches the gate, and
the gate's failure modes are deliberately quiet: a missing key inside a cell is
"the metric is missing", not a default and not a dropped cell. So the merge is
done here, explicitly, and refuses to emit a bundle that is missing an input the
gate requires.

The drift guard is the part worth being careful about. The gate does not read the
drift report's verdict; it RE-COMPUTES the comparison from
`fresh_incumbent_metrics`. Passing the report alone raises GateInputError, and
passing a report claiming MATCH without the fresh numbers to back it is exactly
the forgery the guard exists to prevent. This tool therefore copies the fresh
incumbent metrics out of the benchmark JSON, keyed the way the gate parses them
(`<channel>|<horizon>`), and never writes the report's own status into the
bundle.

Usage:
    python assemble_gate_inputs.py \
        --evidence prediction-model/data/promotion_evidence.json \
        --benchmark prediction-model/data/nwp_benchmark_incumbent.json \
        --model-label "LNN incumbent" \
        --out prediction-model/data/gate_inputs.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "prediction-model/data"

# The gate parses "<channel>|<horizon>". The benchmark nests as
# results.<hN>.<channel>.<model label>, so the two have to be transposed here.
#
# The rain channel is the exception that matters: the gate and the baseline call
# it "rain_occurrence", but the benchmark stores it under "_rain_brier", because
# for a probability the right score is the Brier score and NOT mean absolute
# error. Feeding rain |p - y| produces a confident meaningless number, and that
# mistake has been made in this repo before. The key is remapped rather than
# aliased so the metric identity is explicit at the point of the mapping.
BENCHMARK_CHANNELS = {
    "temperature": "temperature",
    "humidity": "humidity",
    "pressure": "pressure",
    "wind_speed": "wind_speed",
    "rain_occurrence": "_rain_brier",
}

# Metrics the benchmark reports per channel. Brier is read as "brier" and never
# as "mae": the two are not interchangeable and averaging |p - y| for a
# probability would be a different, wrong quantity wearing the right name.
METRIC_KEYS = ("mae", "brier")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_fresh_metrics(benchmark: dict[str, Any], model_label: str) -> tuple[dict[str, float], list[str]]:
    """
    Pull `<channel>|<horizon>` -> score out of a benchmark document.

    Returns the metrics and the list of channels that could not be resolved, so
    the caller can refuse rather than emit a bundle with a hole in it. A missing
    channel is never filled with a default: the gate treats an absent key as
    "metric missing" and would mark the cell NOT_RECHECKED, which surfaces as
    BASELINE_DRIFT and blocks promotion for a reason that has nothing to do with
    the candidate.
    """
    results = benchmark.get("results") or {}
    fresh: dict[str, float] = {}
    seen_channels: set[str] = set()
    for horizon_key, block in results.items():
        if not isinstance(block, dict):
            continue
        try:
            horizon = int(str(horizon_key).lstrip("hH"))
        except ValueError:
            continue
        for channel, benchmark_key in BENCHMARK_CHANNELS.items():
            cell = block.get(benchmark_key)
            if not isinstance(cell, dict):
                continue
            entry = cell.get(model_label)
            seen_channels.add(channel)
            # The Brier block stores a bare number per model, while the MAE blocks
            # store a dict of named metrics. Both are legitimate shapes in this
            # benchmark format and only one of them is what the other channels
            # use, so both are accepted here rather than assuming uniformity --
            # assuming it produced a bundle that silently omitted every rain cell
            # and surfaced to the operator only as BASELINE_DRIFT.
            if isinstance(entry, (int, float)):
                fresh[f"{channel}|{horizon}"] = float(entry)
            elif isinstance(entry, dict):
                for metric in METRIC_KEYS:
                    value = entry.get(metric)
                    if isinstance(value, (int, float)):
                        fresh[f"{channel}|{horizon}"] = float(value)
                        break
    unresolved = sorted(set(BENCHMARK_CHANNELS) - seen_channels)
    return fresh, unresolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--evidence", required=True, help="paired evidence JSON")
    parser.add_argument("--benchmark", required=True, help="fresh incumbent benchmark JSON")
    parser.add_argument("--model-label", required=True,
                        help="model key inside the benchmark JSON; a wrong value yields "
                             "an empty comparison, which is a block, not a warning")
    parser.add_argument("--out", required=True)
    parser.add_argument("--tolerance", type=float, default=None,
                        help="drift relative tolerance; the gate default is used when omitted")
    args = parser.parse_args()

    evidence = load_json(Path(args.evidence))
    benchmark = load_json(Path(args.benchmark))

    if not isinstance(evidence, dict):
        print(f"  evidence document is not a JSON object: {args.evidence}")
        return 2

    bundle = dict(evidence)
    fresh, unresolved = extract_fresh_metrics(benchmark, args.model_label)

    if not fresh:
        print(f"  NO FRESH METRICS extracted from {args.benchmark} under label "
              f"{args.model_label!r}.")
        print("  Refusing to emit a bundle: the drift guard would then have nothing to")
        print("  compare and the gate would block with no indication of why.")
        print("  Check --model-label against the benchmark's model keys.")
        available = sorted({
            key
            for block in (benchmark.get("results") or {}).values()
            if isinstance(block, dict)
            for cell in block.values() if isinstance(cell, dict)
            for key in cell
        })
        print(f"  available model keys: {available}")
        return 1

    if unresolved:
        # Reported loudly rather than tolerated. An unresolved channel becomes a
        # NOT_RECHECKED cell, and the gate reports that as BASELINE_DRIFT -- a
        # failure that looks like the incumbent changed when in fact the input
        # bundle was incomplete.
        print(f"  WARNING: {len(unresolved)} channel(s) not found in the benchmark: "
              f"{unresolved}")
        print("  Those cells will be NOT_RECHECKED and the gate will report "
              "BASELINE_DRIFT.")
        print("  Expected benchmark keys: "
              f"{sorted(set(BENCHMARK_CHANNELS.values()))}")

    drift_source = os.path.relpath(Path(args.benchmark), ROOT).replace("\\", "/")
    bundle["fresh_incumbent_metrics"] = fresh
    bundle["drift_source"] = drift_source
    bundle["drift_model_label"] = args.model_label
    if args.tolerance is not None:
        bundle["drift_tolerance_rel"] = args.tolerance

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bundle, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"  evidence  : {args.evidence} "
          f"({len(evidence.get('challenger') or {})} challenger cells)")
    print(f"  fresh     : {len(fresh)} incumbent metrics from {drift_source}")
    print(f"  corpus    : {str(evidence.get('corpus_sha256'))[:16]}")
    print(f"  written   : {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
