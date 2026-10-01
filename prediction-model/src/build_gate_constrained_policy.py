"""
Build a policy whose model routes are authorised by the release gate.

WHY THIS TOOL EXISTS

`refit_policy.py` and `release_gate.py` answer different questions, and on this
evidence they disagree. Both are correct about their own question:

  refit: "would serving the model beat serving persistence?" -- a point estimate
         of mean MAE, with no significance test.
  gate:  "is the challenger better than what is currently SERVED, at 2 sigma,
          with both chronological halves agreeing?"

The disagreement is real and it is not noise. On the 2026-10-01 retrain, the
refit routed the model to 8 cells; 6 of those the gate had rejected. Deploying
the refit directly would have shipped temperature at 1h/3h/6h/12h, humidity at
12h and pressure at 24h from a model the gate had refused.

TWO DISTINCT CAUSES, AND ONLY ONE IS A DEFECT

1. Raw vs calibrated. Where the live policy serves persistence, the gate scores
   the candidate's RAW head while the deployed candidate would serve a
   CALIBRATED one. The calibration is doing the work:

       temperature|h1   persistence 0.558   raw 0.582 (worse)   calibrated 0.551
       temperature|h3   persistence 0.999   raw 2.830 (far worse)  calibrated 0.982
       humidity|h12     persistence 3.474   raw 3.684 (worse)   calibrated 3.038

   The gate is measuring a configuration that would never ship. This is a real
   defect in how the evidence is assembled, and it should be fixed at source in
   build_promotion_evidence.py rather than papered over here.

2. Different questions. Where the live policy already serves a MODEL
   (temperature|h6, temperature|h12), the gate demands the challenger beat that
   model, not merely persistence. A higher bar, correctly applied. And a point
   estimate with no significance test will happily route on a 0.4% mean
   improvement -- pressure|h24 is persistence 1.117 vs raw 1.113 -- that cannot
   clear 2 sigma on any honest sample.

This tool takes the refit's MEASURED CALIBRATION and the gate's ROUTING
AUTHORITY, and never overrides the gate. A cell the gate did not promote is
routed to persistence regardless of what the refit preferred, so the failure
mode is a model that is not used, never a model that was never approved.

Usage:
    python build_gate_constrained_policy.py \
        --refit prediction-model/data/inference_policy_candidate.json \
        --verdict prediction-model/data/promotion_verdict_full_retrain.json \
        --out prediction-model/data/inference_policy_gated.json
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

MODEL_SOURCE = "learned_model"
PERSISTENCE = "persistence_fallback"

# The incumbent's rain blend per horizon, used verbatim for any horizon the gate
# did not promote. Taken from the served policy rather than hard-coded, so this
# tool cannot quietly become the source of a rain weighting.
RAIN_CHANNEL = "rain_occurrence"


def load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def promoted_cells(verdict: dict[str, Any]) -> dict[tuple[str, int], str]:
    """(channel, horizon) -> verdict, for cells the gate actually decided."""
    out: dict[tuple[str, int], str] = {}
    for key, cell in (verdict.get("cells") or {}).items():
        channel, _, horizon = str(key).partition("|h")
        try:
            out[(channel, int(horizon))] = cell.get("verdict")
        except ValueError:
            continue
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--refit", required=True,
                        help="policy from refit_policy.py (source of calibration)")
    parser.add_argument("--verdict", required=True,
                        help="release gate verdict JSON (source of routing authority)")
    parser.add_argument("--served", default=None,
                        help="the currently served policy; supplies the rain blend for "
                             "horizons the gate did not promote. Defaults to the live "
                             "inference_policy.json.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    refit = load(Path(args.refit))
    verdict = load(Path(args.verdict))
    served = load(Path(args.served or (DATA / "inference_policy.json")))

    cells = promoted_cells(verdict)
    if not cells:
        print("  verdict document has no decided cells; refusing to build a policy "
              "from an empty authority.")
        return 1

    if not verdict.get("promotion_decision_computed"):
        print(f"  verdict is {verdict.get('recommendation')} with "
              f"promotion_decision_computed=false. Integrity blockers mean the gate "
              f"did NOT evaluate promotion; treating that as an authority to route to "
              f"a model would be exactly the mistake this tool exists to prevent.")
        return 1

    promoted = {k for k, v in cells.items() if v == "PROMOTE"}
    out = json.loads(json.dumps(refit))  # deep copy, keeps the measured calibration
    out["policy_version"] = "gated-refit"
    out["regenerated_by"] = "build_gate_constrained_policy.py"
    out["regeneration_note"] = (
        "Calibration measured by refit_policy.py; model routing authorised solely by "
        "release_gate.py. Cells the gate did not promote are forced to persistence, "
        "even where the refit preferred the model."
    )
    out["gate_authority"] = {
        "verdict_document": os.path.relpath(Path(args.verdict), ROOT).replace("\\", "/"),
        "recommendation": verdict.get("recommendation"),
        "promoted_cells": sorted(f"{c}|h{h}" for c, h in promoted),
        "refit_wanted_model_but_gate_rejected": [],
    }

    downgraded: list[str] = []
    authorised: list[str] = []

    for h_key, h_block in (out.get("horizons") or {}).items():
        sources = h_block.setdefault("selected_sources", {})
        for var in list(sources):
            if sources[var] != MODEL_SOURCE:
                continue
            cell = (var, int(h_key))
            if cells.get(cell) == "PROMOTE":
                authorised.append(f"{var}|h{h_key}")
            else:
                # The gate is the authority. A rejection, an absent cell, and a
                # verdict that never evaluated this cell are all "not authorised".
                sources[var] = PERSISTENCE
                downgraded.append(f"{var}|h{h_key} ({cells.get(cell) or 'NOT_GATED'})")

        # Rain is a blend, not a source. Only a promoted horizon may keep any
        # model weight; everything else reverts to the incumbent's blend.
        incumbent_rain = (served.get("horizons", {}).get(h_key) or {})
        if cells.get((RAIN_CHANNEL, int(h_key))) != "PROMOTE":
            h_block["rain_model_weight"] = incumbent_rain.get("rain_model_weight")
            h_block["rain_persistence_weight"] = incumbent_rain.get("rain_persistence_weight")
        else:
            authorised.append(f"{RAIN_CHANNEL}|h{h_key} "
                              f"(model={h_block.get('rain_model_weight')})")

    out["gate_authority"]["refit_wanted_model_but_gate_rejected"] = sorted(downgraded)

    out_path = Path(args.out)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8", newline="\n")

    print(f"  verdict            : {verdict.get('recommendation')} "
          f"(decision computed: {verdict.get('promotion_decision_computed')})")
    print(f"  promoted cells     : {len(promoted)}")
    print(f"  model routes kept  : {len(authorised)}")
    for item in sorted(authorised):
        print(f"      keep    {item}")
    print(f"  routes downgraded  : {len(downgraded)}")
    for item in sorted(downgraded):
        print(f"      force   {item} -> persistence")
    print(f"  written            : {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
