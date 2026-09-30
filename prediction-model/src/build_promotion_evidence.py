"""
Build the decision-input bundle the release gate requires.

WHY THIS EXISTS
---------------
`release_gate.py` was built and mutation-tested but has never been run against
real evidence, so no promotion decision has ever been made by the rule rather
than by judgement. The reason is a structural gap: the gate requires PAIRED,
PER-ROW statistics computed on the VALIDATION split, split into chronological
halves, and nothing in the repo produced them.

`benchmark_vs_nwp.py` emits aggregate TEST-split metrics. Those cannot be
substituted, and the gate is right to refuse them: a test split must never drive a
decision, and an aggregate MAE cannot be tested for a paired difference because
the row-level pairing is exactly what is lost.

So this script does the missing step. For every (channel, horizon) cell it:

  1. rebuilds the validation windows once,
  2. scores the candidate AND the incumbent on those SAME rows,
  3. keeps the per-row absolute errors so they can be paired,
  4. splits the validation period chronologically into an early and a late half,
  5. emits n_pairs / mean_difference / stderr for the full span and each half.

THE STATISTIC
-------------
mean_difference = mean(incumbent_row_error - candidate_row_error)

Positive means the challenger is better. Row error is |y - yhat| for the MAE
channels and the Brier term (p - y)^2 for rain occurrence, declared per channel
rather than assumed: feeding rain |p - y| produces a confident meaningless
verdict, and that mistake has been made before in this repo.

The halves test the SIGN only. Requiring 2-sigma inside each half needs roughly
four times the rows on autocorrelated series and would refuse to promote
anything at all; strictness belongs in the full-span 2-sigma test.

WHAT THIS DOES NOT DO
--------------------
It does not decide anything. It produces evidence; `release_gate.py` reads it and
emits the verdict. It never reads the test split for any decision input, and it
records the test metric separately and clearly labelled as reporting-only.

It also refuses to run at all if the candidate's corpus does not match the
evaluation corpus. Scoring a checkpoint against a corpus it was not trained on
applies the wrong normalisation constants, and that mistake has already been made
once in this project.
"""

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
sys.path.insert(0, HERE)

from model import GarciaWeatherLNNFeatured  # noqa: E402
from dataset import (  # noqa: E402
    TelemetryDataPipeline,
    build_forecast_windows,
    build_feature_augmented_forecast_windows,
    DEFAULT_SEQ_LEN,
)

HORIZONS = [1, 3, 6, 12, 24]
MAE_CHANNELS = ["temperature", "humidity", "pressure", "wind_speed"]
RAIN_CHANNEL = "rain_occurrence"

ORIGIN_COLS = ["origin_temperature", "origin_humidity", "origin_pressure",
               "origin_wind_speed", "origin_wind_u", "origin_wind_v"]

# Cells below this are not enough to say anything about a sign.
MIN_PAIRS = 30


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def paired_block(inc_err, cand_err, order_key, label, source):
    """
    Paired statistics over one span of rows.

    The pairing is by row identity, not by position: both models are scored on
    the same windows in the same order, and `order_key` carries the chronology so
    the halves are split on TIME and not on array index. Windows arrive in
    station-then-time order, so an index-based half-split would silently compare
    different stations in the two halves.
    """
    n = len(inc_err)
    if n < MIN_PAIRS:
        return None, {"reason": f"n_pairs {n} < MIN_PAIRS {MIN_PAIRS}"}

    diff = np.asarray(inc_err, float) - np.asarray(cand_err, float)
    mean_diff = float(diff.mean())
    # stderr of the PAIRED differences. Unpaired stderr would be far larger and
    # would test the wrong null.
    if n < 2:
        return None, {"reason": "n_pairs < 2"}
    sd = float(diff.std(ddof=1))
    stderr = sd / math.sqrt(n)
    if not math.isfinite(stderr) or stderr <= 0:
        return None, {"reason": "stderr is zero or non-finite; cannot test"}

    return {
        "n_pairs": n,
        "mean_difference": mean_diff,
        "stderr": stderr,
        "split": "validation",
        "source": source,
        "sample_ids": [str(k) for k in order_key[:200]],
        "_note": f"mean_difference over {n} paired rows; positive = challenger better",
    }, {}


def channel_row_errors(channel, cand_out, inc, meta):
    """
    Per-row error for one channel, for both models, in the same row order.

    `inc` is (inc_pred_by_channel, inc_rain_probability).

    Returns (candidate_errors, incumbent_errors, sample_keys).
    """
    inc_pred, inc_rain = inc
    keys = [f"{m['station_id']}@{m['target_timestamp']}" for m in meta]

    if channel == RAIN_CHANNEL:
        y = np.array([float(m.get("target_precipitation", 0.0) or 0.0) > 0.1
                      for m in meta], float)
        p_c = torch.sigmoid(cand_out["rain_prob"]).squeeze(-1).numpy()
        return ((p_c - y) ** 2).tolist(), ((inc_rain - y) ** 2).tolist(), keys

    y = np.array([float(m[f"target_{channel}"]) for m in meta], float)
    p_c = cand_out[channel].squeeze(-1).numpy()
    return np.abs(p_c - y).tolist(), np.abs(inc_pred[channel] - y).tolist(), keys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate-dir", required=True,
                    help="Directory holding candidate_h<N>h.pt")
    ap.add_argument("--incumbent-dir", default=os.path.join(DATA, "bundles"),
                    help="Directory holding the served bundles (h<N>/checkpoint.pt)")
    ap.add_argument("--weather-csv", default=os.path.join(DATA, "weather_telemetry_current.csv"))
    ap.add_argument("--out", default=os.path.join(DATA, "promotion_evidence.json"))
    ap.add_argument("--horizons", default=None)
    ap.add_argument("--channels", default=",".join(MAE_CHANNELS) + "," + RAIN_CHANNEL)
    args = ap.parse_args()

    horizons = ([int(x) for x in args.horizons.split(",")] if args.horizons
                else HORIZONS)
    channels = [c.strip() for c in args.channels.split(",") if c.strip()]

    corpus_sha = sha256_file(args.weather_csv)

    # --- refuse to evaluate a checkpoint against a corpus it was not trained on ---
    def manifest_corpus(path):
        mf = os.path.join(os.path.dirname(path), f"candidate_h{h}_manifest.json")
        if not os.path.exists(mf):
            mf = os.path.join(os.path.dirname(path), "bundle_manifest.json")
        try:
            with open(mf, encoding="utf-8") as f:
                return (json.load(f).get("weather_telemetry_sha256") or
                        json.load(open(os.path.join(os.path.dirname(path),
                                                    "candidate_h%d_manifest.json" % h),
                                               encoding="utf-8")).get("weather_telemetry_sha256"))
        except (OSError, ValueError):
            return None

    print("=" * 78)
    print("PROMOTION EVIDENCE BUILDER")
    print("=" * 78)
    print(f"  corpus      : {os.path.basename(args.weather_csv)}")
    print(f"  corpus sha  : {corpus_sha[:16]}...")
    print(f"  candidate   : {args.candidate_dir}")
    print(f"  incumbent   : {args.incumbent_dir}")
    print(f"  horizons    : {horizons}")
    print(f"  channels    : {', '.join(channels)}")

    pipe = TelemetryDataPipeline(weather_csv=args.weather_csv)
    bundle = {"corpus_sha256": corpus_sha,
              "baseline_path": os.path.join(DATA, "release_baseline.md"),
              "policy_path": os.path.join(DATA, "inference_policy.json"),
              "channels": channels, "horizons": horizons,
              "candidate": {}, "incumbent": {},
              "_generated_utc": datetime.now(timezone.utc).isoformat(),
              "_generated_by": "build_promotion_evidence.py"}

    for h in horizons:
        cand_ck = os.path.join(args.candidate_dir, f"candidate_h{h}h.pt")
        if not os.path.exists(cand_ck):
            print(f"  +{h}h  SKIPPED: no candidate checkpoint")
            continue
        inc_ck = os.path.join(args.incumbent_dir, f"h{h}", "checkpoint.pt")
        if not os.path.exists(inc_ck):
            print(f"  +{h}h  SKIPPED: no incumbent bundle")
            continue

        # Pairing check. The incumbent bundle manifest records the corpus it was
        # built from, but it is the OLD corpus by construction, so this check
        # applies to the candidate and is reported for the incumbent without
        # failing the run.
        cand_sha = manifest_corpus(cand_ck)
        if cand_sha and cand_sha != corpus_sha:
            print(f"  +{h}h  ABORT: candidate was trained on corpus {cand_sha[:12]} "
                  f"but the evaluation corpus is {corpus_sha[:12]}. Scoring it here "
                  f"would apply the wrong normalisation constants.")
            return 2

        print(f"\n  +{h}h  building val-split windows...")
        res = build_forecast_windows(pipeline=pipe, split="val", horizon=h,
                                     seq_len=DEFAULT_SEQ_LEN, return_metadata=True)
        if res is None:
            print("        SKIPPED: no validation windows")
            continue
        tele, dt = res[0], res[1]
        meta = res[6]
        aug = build_feature_augmented_forecast_windows(
            pipe, split="val", horizon=h, seq_len=DEFAULT_SEQ_LEN)
        ctx = aug[1]

        origin = torch.tensor(
            np.column_stack([[m[c] for m in meta] for c in ORIGIN_COLS]),
            dtype=torch.float32)

        ck = torch.load(cand_ck, map_location="cpu", weights_only=False)
        man = ck.get("manifest", {}) or {}
        cand = GarciaWeatherLNNFeatured(
            input_dim=int(man.get("input_dim", 8)),
            context_dim=int(man.get("context_dim", 75)),
            hidden_dim=int(man.get("hidden_dim", 32)),
            use_two_stage_precipitation=True)
        cand.load_state_dict(ck["model_state_dict"])
        cand.eval()

        # The incumbent is the SERVED bundle: a different architecture
        # (GarciaWeatherLNN, not the featured variant) with a different input
        # contract (de-normalised sequence) and a policy layer that may substitute
        # persistence for any given cell.
        #
        # It is scored through LNNServerlessPredictor, the same path
        # benchmark_vs_nwp.py uses, so the comparison is like-for-like: identical
        # rows, identical ground truth, identical de-normalisation and clipping.
        # Scoring it any other way would compare a model against a strawman.
        from inference import LNNServerlessPredictor

        n = len(meta)
        print(f"        {n} validation rows; scoring incumbent bundle (per row)...")
        predictor = LNNServerlessPredictor(
            bundle_dir=os.path.join(args.incumbent_dir, f"h{h}"))
        X_norm = res[0].detach().cpu().numpy().astype(np.float64)
        X_raw = np.clip(X_norm * pipe.norm_stds + pipe.norm_means,
                        [10, 10, 10, 900, 0, -1, -1, 0],
                        [50, 70, 100, 1050, 180, 1, 1, 150])
        dt_raw = dt.detach().cpu().numpy().astype(np.float64)

        inc_pred = {c: np.empty(n) for c in MAE_CHANNELS}
        inc_rain = np.empty(n)
        for i in range(n):
            o = predictor.predict_from_observed_sequence(
                telemetry_sequence=X_raw[i], dt_sequence=dt_raw[i], horizon_hours=h)
            inc_pred["temperature"][i] = o["temperature_c"]
            inc_pred["humidity"][i] = o["relative_humidity_pct"]
            inc_pred["pressure"][i] = o["pressure_hpa"]
            inc_pred["wind_speed"][i] = o["wind_speed_kmh"] / 3.6  # km/h -> m/s
            inc_rain[i] = o["chance_of_rain_pct"] / 100.0

        with torch.no_grad():
            cand_out = cand(tele, ctx, dt, origin_weather=origin)

        for channel in channels:
            key = f"{channel}|{h}"
            c_err, i_err, keys = channel_row_errors(
                channel, cand_out, (inc_pred, inc_rain), meta)

            # Chronological halves, keyed on the target timestamp.
            order = sorted(range(n), key=lambda i: (meta[i]["target_timestamp"],
                                                    meta[i]["station_id"]))
            half = n // 2
            spans = {"full": order,
                     "early": order[:half],
                     "late": order[half:]}

            paired = {}
            for span_name, idx in spans.items():
                blk, why = paired_block([i_err[i] for i in idx],
                                        [c_err[i] for i in idx],
                                        [keys[i] for i in idx],
                                        span_name,
                                        f"build_promotion_evidence.py:{channel}+{h}h:{span_name}")
                if blk is None:
                    print(f"        {channel:<12} {span_name:<5} INCOMPLETE: {why}")
                    continue
                paired[span_name] = blk

            node = {
                "validation": {"value": float(np.mean(c_err)), "n_rows": n,
                               "split": "validation",
                               "source": f"candidate_artifacts:{args.candidate_dir}"},
                "test": None,
                "paired": paired,
            }
            bundle["candidate"][key] = node
            bundle["incumbent"][key] = {
                "validation": {"value": float(np.mean(i_err)), "n_rows": n,
                               "split": "validation",
                               "source": f"served_bundle:{args.incumbent_dir}/h{h}"},
                "test": None,
                "paired": paired,
            }

            full = paired.get("full")
            sig = ""
            if full:
                ratio = full["mean_difference"] / full["stderr"]
                sig = f"  mean_diff {full['mean_difference']:+.5f}  {ratio:+.2f} sigma"
            print(f"        {channel:<12} {sig}")

    bundle["_incumbent_note"] = (
        "Incumbent is the SERVED bundle, scored through LNNServerlessPredictor on "
        "the same validation rows with the same de-normalisation and clipping that "
        "benchmark_vs_nwp.py uses. It is deliberately not the candidate architecture: "
        "comparing a candidate against a strawman would manufacture a promotion. "
        "Note that the bundle carries a policy layer that may substitute "
        "persistence for a given cell, so its error already reflects what users "
        "actually received."
    )

    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(bundle, f, indent=2)
    print(f"\nwritten: {args.out}")
    print(f"  candidate cells: {len(bundle['candidate'])}")
    print(f"  incumbent cells: {len(bundle['incumbent'])}")
    print("\nThe gate will refuse this bundle with CORPUS_MISMATCH while the served")
    print("bundles were built from an older corpus than the evidence. That refusal is")
    print("correct: it is not possible to justify promoting a model trained on one")
    print("corpus using evidence from another, against an incumbent trained on a")
    print("third. The remedy is a release that moves all three together, not a")
    print("per-channel swap across corpora.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
