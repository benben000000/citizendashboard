"""
Aggregate the seed sweep: is the leaky floor's effect larger than seed noise?

THE QUESTION
------------
The single-seed comparison showed the leaky floor changing MAE by single-digit
percentages in both directions. Before that is called an effect, the spread
across seeds has to be smaller. This scores every (floor, seed, horizon) on the
same test split and reports mean, spread, and the floor-to-floor delta against
the seed noise it sits inside.

WHY THIS DOES NOT CALL THE PROMOTION RULE
-----------------------------------------
`train_predictive_quality.py` computes paired 2-sigma confidence intervals against
the incumbent. This script deliberately does NOT reuse that machinery. It scores
the two arms against each other on the same rows, which answers "does the floor
matter" but says nothing about whether either arm beats production. Promotion is a
separate decision made on the incumbent comparison.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(ROOT, "prediction-model", "data")

sys.path.insert(0, os.path.join(ROOT, "prediction-model", "src"))

HORIZONS = [1, 3, 6, 12, 24]
VARS = ["temperature", "humidity", "pressure", "wind_speed"]
ORIGIN_COLS = ["origin_temperature", "origin_humidity", "origin_pressure",
               "origin_wind_speed", "origin_wind_u", "origin_wind_v"]


def load_sweep_dirs(base):
    found = {}
    for tag in sorted(os.listdir(base)):
        full = os.path.join(base, tag)
        if not os.path.isdir(full):
            continue
        floor, _, seed = tag.rpartition("_s")
        if not floor:
            continue
        if all(os.path.exists(os.path.join(full, f"candidate_h{h}h.pt")) for h in HORIZONS):
            found.setdefault(floor, {})[int(seed)] = full
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", default=os.path.join(DATA, "seed_sweep"))
    ap.add_argument("--weather-csv", default=os.path.join(
        DATA, "weather_telemetry_current.csv"))
    ap.add_argument("--out", default=os.path.join(DATA, "seed_sweep_analysis.json"))
    args = ap.parse_args()

    import torch
    from model import GarciaWeatherLNNFeatured
    from dataset import (TelemetryDataPipeline, build_forecast_windows,
                         build_feature_augmented_forecast_windows, DEFAULT_SEQ_LEN)

    arms = load_sweep_dirs(args.sweep_dir)
    if not arms:
        print(f"no complete arms under {args.sweep_dir}")
        return 1
    for f, seeds in arms.items():
        print(f"  arm {f:<6} seeds {sorted(seeds)}")

    pipe = TelemetryDataPipeline(weather_csv=args.weather_csv)
    print(f"  corpus {os.path.basename(args.weather_csv)}")
    print(f"  test   {pipe.test_start} .. {pipe.time_range_max}")

    cache = {}
    for h in HORIZONS:
        res = build_forecast_windows(pipeline=pipe, split="test", horizon=h,
                                     seq_len=DEFAULT_SEQ_LEN, return_metadata=True)
        aug = build_feature_augmented_forecast_windows(
            pipe, split="test", horizon=h, seq_len=DEFAULT_SEQ_LEN)
        cache[h] = (res[0], aug[1], res[1], res[6])
        print(f"  +{h:>2}h  windows={len(res[6])}")

    results = {f: {v: {h: {} for h in HORIZONS} for v in VARS} for f in arms}
    zeros = {f: {h: 0 for h in HORIZONS} for f in arms}

    for floor, seeds in arms.items():
        for seed, d in seeds.items():
            for h in HORIZONS:
                ck = torch.load(os.path.join(d, f"candidate_h{h}h.pt"),
                                map_location="cpu", weights_only=False)
                man = ck.get("manifest", {}) or {}
                mdl = GarciaWeatherLNNFeatured(
                    input_dim=int(man.get("input_dim", 8)),
                    context_dim=int(man.get("context_dim", 75)),
                    hidden_dim=int(man.get("hidden_dim", 32)),
                    use_two_stage_precipitation=True)
                mdl.load_state_dict(ck["model_state_dict"])
                mdl.eval()
                tele, ctx, dt, meta = cache[h]
                origin = torch.tensor(
                    np.column_stack([[m[c] for m in meta] for c in ORIGIN_COLS]),
                    dtype=torch.float32)
                with torch.no_grad():
                    out = mdl(tele, ctx, dt, origin_weather=origin)
                ws = out["wind_speed"].squeeze(-1).numpy()
                zeros[floor][h] += int((ws == 0).sum())
                for v, key in (("temperature", "temperature"),
                               ("humidity", "humidity"),
                               ("pressure", "pressure"),
                               ("wind_speed", "wind_speed")):
                    pred = out[key].squeeze(-1).numpy()
                    tgt = np.array([m[f"target_{v}"] for m in meta], dtype=float)
                    results[floor][v][h][seed] = float(np.mean(np.abs(pred - tgt)))

    print("\n" + "=" * 96)
    print("SEED SWEEP RESULT  (MAE; lower is better)")
    print("=" * 96)
    print(f"{'var':<12}{'hz':>4}{'persist':>10}{'relu mean':>11}{'leaky mean':>12}"
          f"{'delta':>9}{'relu sd':>9}{'leaky sd':>10}{'verdict':>12}")

    out_rows = {}
    for v in VARS:
        for h in HORIZONS:
            tele, ctx, dt, meta = cache[h]
            origin = np.column_stack([[m[c] for m in meta] for c in ORIGIN_COLS])
            persist = float(np.mean(np.abs(origin[:, 3 if v == "wind_speed" else
                                                {"temperature": 0, "humidity": 1,
                                                 "pressure": 2}[v]] - 0)
                                    )) if False else None
            # recompute persistence properly from metadata
            key = f"origin_{v}"
            persist = float(np.mean(np.abs(
                np.array([m[key] for m in meta], float)
                - np.array([m[f"target_{v}"] for m in meta], float))))

            r = [results["relu"][v][h][s] for s in sorted(results["relu"][v][h])]
            l = [results["leaky"][v][h][s] for s in sorted(results["leaky"][v][h])]
            if not r or not l:
                continue
            rm, lm = float(np.mean(r)), float(np.mean(l))
            rs, ls = float(np.std(r, ddof=1)) if len(r) > 1 else 0.0, \
                     float(np.std(l, ddof=1)) if len(l) > 1 else 0.0
            delta = (rm - lm) / rm * 100
            noise = max(rs, ls)
            # The floor only counts as a real effect if it clears the seed spread.
            verdict = "leaky better" if delta > noise else (
                "relu better" if -delta > noise else "within noise")
            print(f"{v:<12}{h:>4}{persist:>10.4f}{rm:>11.4f}{lm:>12.4f}"
                  f"{delta:>+8.1f}%{rs:>9.4f}{ls:>10.4f}{verdict:>12}")
            out_rows[f"{v}_h{h}"] = {
                "persistence": persist, "relu_mean": rm, "leaky_mean": lm,
                "delta_pct": delta, "relu_sd": rs, "leaky_sd": ls,
                "noise_band": noise, "verdict": verdict,
                "relu": r, "leaky": l,
            }
        print()

    print("exact-zero wind predictions across all seeds")
    for f in arms:
        tot = sum(zeros[f].values())
        print(f"  {f:<8} {tot}")

    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"rows": out_rows,
                   "zeros": zeros,
                   "arms": {k: sorted(v) for k, v in arms.items()}},
                  f, indent=2)
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
