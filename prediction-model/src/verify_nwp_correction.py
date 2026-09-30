"""
End-to-end verification of the shipped NWP correction artifact.

This closes the loop that the fitter cannot. The fitter reports test-split
numbers computed from coefficients held in memory while it was fitting them.
This script instead loads prediction-model/data/nwp_correction.json through the
same NwpCorrection/select path that inference will use, applies it to the test
split, and reports what the shipped artifact actually achieves.

If the artifact were incomplete, mis-keyed, or keyed to the wrong source, the
routed numbers would silently collapse back to the LNN. That failure mode is
invisible in the fitter's own output, which is why this check exists.

TEST SPLIT ONLY. Nothing here selects anything.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from dataset import TelemetryDataPipeline, build_forecast_windows, DEFAULT_SEQ_LEN  # noqa: E402
from inference import LNNServerlessPredictor  # noqa: E402
from nwp_correction import (  # noqa: E402
    CLAMP, NwpCorrection, artifact_path_for, select,
)

DATA_DIR = os.path.abspath(os.path.join(HERE, "..", "data"))
CACHE = os.path.join(DATA_DIR, "nwp_benchmark_cache.json")
BUNDLES = os.path.join(DATA_DIR, "bundles")
SEQ_CLIP_LO = [10, 10, 10, 900, 0, -1, -1, 0]
SEQ_CLIP_HI = [50, 70, 100, 1050, 180, 1, 1, 150]

VARS = {
    "temperature": ("temperature_2m", "target_temperature", "temperature_c", 1.0),
    "humidity": ("relative_humidity_2m", "target_humidity", "relative_humidity_pct", 1.0),
    "pressure": ("surface_pressure", "target_pressure", "pressure_hpa", 1.0),
    "wind_speed": ("wind_speed_10m", "target_wind_speed", "wind_speed_kmh", 1.0 / 3.6),
}


def main():
    # Verify a specific source's artifact, so the licence-clean GFS build and
    # the stronger ECMWF benchmark build can each be checked independently.
    want = os.environ.get("VERIFY_NWP_SOURCE", "")
    path = artifact_path_for(want) if want else None
    corr = NwpCorrection.load(path) if path else NwpCorrection.load()
    if corr is None or not corr.available:
        print("NO ARTIFACT -- the correction layer is inert and every forecast "
              "would fall back to the LNN.")
        return 1

    raw = json.load(open(CACHE, "r", encoding="utf-8"))
    best = {}
    for k, v in raw.items():
        if k.startswith("_"):
            continue
        b = f"{v['station_id']}|{v['model']}"
        if b not in best or len(v.get("time", [])) > len(best[b]["time"]):
            best[b] = v
    source = corr.source
    pipe = TelemetryDataPipeline()

    print("=" * 96)
    print("END-TO-END VERIFICATION OF THE SHIPPED ARTIFACT   (test split)")
    print("=" * 96)
    print(f"  source        : {source}")
    print(f"  coefficients  : {corr.n_coefficients()}")
    print(f"  routes        : {len(corr.artifact.get('routing', {}))} horizons")
    print(f"  fit protocol  : {corr.fit_report}")

    horizons = [int(x) for x in os.environ.get("VERIFY_HORIZONS", "1,3,6,12,24").split(",")]
    rows_out = {}
    deltas, used_counts = [], {}

    for h in horizons:
        res = build_forecast_windows(pipeline=pipe, split="test", horizon=h,
                                     seq_len=DEFAULT_SEQ_LEN, return_metadata=True)
        X = res[0].detach().cpu().numpy().astype(np.float64)
        dt = res[1].detach().cpu().numpy().astype(np.float64)
        meta = res[6]
        X_raw = np.clip(X * pipe.norm_stds + pipe.norm_means, SEQ_CLIP_LO, SEQ_CLIP_HI)

        lnn = LNNServerlessPredictor(bundle_dir=os.path.join(BUNDLES, f"h{h}"))
        model_out = {v: np.array([lnn.predict_from_observed_sequence(
            telemetry_sequence=X_raw[i], dt_sequence=dt[i], horizon_hours=h).get(VARS[v][2])
            for i in range(len(meta))], float) for v in VARS}

        idx_cache = {}
        print(f"\n+{h}h   n={len(meta)}")
        print(f"  {'variable':<12}{'LNN now':>10}{'routed':>10}{'change':>10}{'persist':>10}  used")
        for v, (fld, tk, _lk, sc) in VARS.items():
            tgt = np.array([m[tk] for m in meta], float)
            org = np.array([m[f"origin_{v}"] for m in meta], float)
            nwp = np.full(len(meta), np.nan)
            for i, m in enumerate(meta):
                rec = best.get(f"{m['station_id']}|{source}")
                if rec is None:
                    continue
                ix = idx_cache.get(id(rec))
                if ix is None:
                    ix = {t: j for j, t in enumerate(rec["time"])}
                    idx_cache[id(rec)] = ix
                j = ix.get(m["target_timestamp"][:13] + ":00")
                if j is not None and rec[fld][j] is not None:
                    nwp[i] = rec[fld][j] * sc

            routed = np.full(len(meta), np.nan)
            used = {}
            for i, m in enumerate(meta):
                prod, c, w = corr.plan(h, v, nwp_value=nwp[i],
                                       station_id=m["station_id"])
                val, who = select(prod, w, c, model_out[v][i], org[i])
                if val is not None:
                    routed[i] = val
                used[who] = used.get(who, 0) + 1

            ok = np.isfinite(routed) & np.isfinite(model_out[v]) & np.isfinite(tgt)
            model_mae = np.abs(model_out[v][ok] - tgt[ok]).mean()
            rt_mae = np.abs(routed[ok] - tgt[ok]).mean()
            pe_mae = np.abs(org[ok] - tgt[ok]).mean()
            d = (model_mae - rt_mae) / model_mae * 100
            deltas.append(d)
            for k, n in used.items():
                used_counts[k] = used_counts.get(k, 0) + n
            rows_out[f"{h}h.{v}"] = {"lnn": float(model_mae), "routed": float(rt_mae),
                                     "persistence": float(pe_mae), "change_pct": float(d)}
            top = max(used, key=used.get)
            print(f"  {v:<12}{model_mae:>10.3f}{rt_mae:>10.3f}{d:>9.1f}%{pe_mae:>10.3f}"
                  f"  {top} {used[top]}/{len(meta)}")

    print("\n" + "=" * 96)
    print("VERDICT")
    print("=" * 96)
    print(f"  mean change vs the LNN that ships today : {np.mean(deltas):+.1f}%")
    print(f"  cells improved                          : "
          f"{sum(1 for d in deltas if d > 0.05)}/{len(deltas)}")
    print(f"  cells unchanged                         : "
          f"{sum(1 for d in deltas if abs(d) <= 0.05)}/{len(deltas)}")
    print(f"  cells made WORSE                        : "
          f"{sum(1 for d in deltas if d < -0.05)}/{len(deltas)}")
    print(f"  producers actually used                 : {used_counts}")

    regress = [(k, v) for k, v in rows_out.items() if v["change_pct"] < -0.05]
    if regress:
        print("\n  REGRESSIONS (investigate before shipping):")
        for k, v in sorted(regress, key=lambda kv: kv[1]["change_pct"]):
            print(f"    {k:<22} {v['lnn']:.3f} -> {v['routed']:.3f} "
                  f"({v['change_pct']:+.1f}%)")

    out = os.path.join(DATA_DIR, "nwp_correction_verification.json")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"source": source, "cells": rows_out,
                   "mean_change_pct": float(np.mean(deltas)),
                   "producers_used": used_counts}, f, indent=2)
    print(f"\n  written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
