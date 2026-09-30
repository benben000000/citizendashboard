"""
Diagnostic: is the deployed model calibration-limited or training-limited?

For every test window this measures FOUR things on the same rows:

  persistence        the origin observation at t0 (what the policy currently
                     serves at most horizons)
  deployed policy    what the dashboard actually shows (model + policy gate)
  raw learned head   the network's own output BEFORE the policy gate, taken from
                     diagnostics.raw_learned_predictions
  bias-corrected head the learned head after an affine correction fitted on TRAIN

The gap between "raw learned head" and "persistence" is the HEADROOM: the skill
the network already has but the operational policy throws away. If that gap is
large, recalibration plus policy change is the fix. If the head is no better
than persistence, the problem is training and no calibration will help.
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

DATA_DIR = os.path.abspath(os.path.join(HERE, "..", "data"))
HORIZONS = [1, 3, 6, 12, 24]

VARS = ["temperature", "humidity", "pressure", "wind_speed"]
POLICY_KEY = {
    "temperature": "temperature_c",
    "humidity": "relative_humidity_pct",
    "pressure": "pressure_hpa",
    "wind_speed": "wind_speed_kmh",
}
HEAD_KEY = {
    "temperature": "temperature_c",
    "humidity": "relative_humidity_pct",
    "pressure": "pressure_hpa",
    "wind_speed": "wind_speed_kmh",
}
ORIGIN_KEY = {
    "temperature": "origin_temperature",
    "humidity": "origin_humidity",
    "pressure": "origin_pressure",
    "wind_speed": "origin_wind_speed",
}
TARGET_KEY = {v: f"target_{v}" for v in VARS}
UNITS = {"temperature": "degC", "humidity": "%", "pressure": "hPa", "wind_speed": "km/h"}


def mae(p, t):
    p, t = np.asarray(p, float), np.asarray(t, float)
    m = np.isfinite(p) & np.isfinite(t)
    if m.sum() == 0:
        return float("nan"), float("nan")
    e = np.abs(p[m] - t[m])
    return float(e.mean()), float((t[m] - p[m]).mean())


def raw_window(pipe, arr, dt):
    return np.clip(arr * pipe.norm_stds + pipe.norm_means,
                   [10, 10, 10, 900, 0, -1, -1, 0], [50, 70, 100, 1050, 180, 1, 1, 150])


def run_h(pipe, h):
    te = build_forecast_windows(pipeline=pipe, split="test", horizon=h,
                                seq_len=DEFAULT_SEQ_LEN, return_metadata=True)
    tr = build_forecast_windows(pipeline=pipe, split="train", horizon=h,
                                seq_len=DEFAULT_SEQ_LEN, return_metadata=True)
    Xte = te[0].detach().cpu().numpy().astype(np.float64)
    Xtr = tr[0].detach().cpu().numpy().astype(np.float64)
    dte = te[1].detach().cpu().numpy().astype(np.float64)
    dtr = tr[1].detach().cpu().numpy().astype(np.float64)
    mte, mtr = te[6], tr[6]

    p = LNNServerlessPredictor(bundle_dir=os.path.join(DATA_DIR, "bundles", f"h{h}"))

    def collect(X, dt, meta, n_max=None):
        policy = {v: [] for v in VARS}
        head = {v: [] for v in VARS}
        idx = range(len(X)) if n_max is None else range(min(n_max, len(X)))
        for i in idx:
            o = p.predict_from_observed_sequence(
                telemetry_sequence=raw_window(pipe, X[i], dt[i]), dt_sequence=dt[i],
                horizon_hours=h)
            for v in VARS:
                policy[v].append(o[POLICY_KEY[v]])
                head[v].append(o["diagnostics"]["raw_learned_predictions"][HEAD_KEY[v]])
        return ({v: np.array(policy[v]) for v in VARS},
                {v: np.array(head[v]) for v in VARS})

    print(f"  collecting train heads (n={len(Xtr)}) ...", flush=True)
    _, tr_head = collect(Xtr, dtr, mtr)
    print("  collecting test  heads ...", flush=True)
    te_policy, te_head = collect(Xte, dte, mte)

    truth = {v: np.array([m[TARGET_KEY[v]] for m in mte], float) for v in VARS}
    persist = {v: np.array([m[ORIGIN_KEY[v]] for m in mte], float) for v in VARS}
    tr_truth = {v: np.array([m[TARGET_KEY[v]] for m in mtr], float) for v in VARS}

    out = {}
    print(f"\n  {'variable':<12}{'persist':>9}{'policy':>9}{'HEAD':>9}{'head corr':>11}"
          f"{'head+cal':>10}{'HEAD skill':>12}{'h+c skill':>11}")
    print("  " + "-" * 82)
    for v in VARS:
        t = truth[v]
        pm, pb = mae(persist[v], t)
        plm, _ = mae(te_policy[v], t)
        hm, hb = mae(te_head[v], t)

        # affine calibration of the head, fitted on TRAIN
        a, b = np.polyfit(tr_head[v], tr_truth[v], 1)
        cal = a * te_head[v] + b
        cm, cb = mae(cal, t)

        skill_raw = (pm - hm) / pm * 100
        skill_cal = (pm - cm) / pm * 100
        print(f"  {v:<12}{pm:>9.3f}{plm:>9.3f}{hm:>9.3f}{hb:>+11.3f}"
              f"{cm:>10.3f}{skill_raw:>+11.1f}%{skill_cal:>+10.1f}%   (a={a:.3f} b={b:+.3f})")
        out[v] = {"persist": pm, "policy": plm, "head": hm, "head_bias": hb,
                  "head_cal": cm, "cal_bias": cb,
                  "head_skill_pct": skill_raw, "head_cal_skill_pct": skill_cal,
                  "a": float(a), "b": float(b)}
    return out


def main():
    pipe = TelemetryDataPipeline()
    print("=" * 96)
    print("HEADROOM DIAGNOSTIC — is the deployed model calibration-limited or training-limited?")
    print("=" * 96)
    print("HEAD = the network's own output before the operational policy gate.")
    print("If HEAD is no better than persistence, recalibration cannot help.\n")

    all_out = {}
    for h in HORIZONS:
        print(f"{'=' * 96}\n+{h}h\n{'-' * 96}")
        try:
            all_out[f"h{h}"] = run_h(pipe, h)
        except Exception as exc:  # noqa: BLE001
            print(f"  failed: {exc}")

    print(f"\n{'=' * 96}")
    print("VERDICT INPUT — mean head skill (calibrated) across the four variables")
    print("=" * 96)
    for h in HORIZONS:
        d = all_out.get(f"h{h}")
        if not d:
            continue
        raw = np.mean([d[v]["head_skill_pct"] for v in VARS])
        cal = np.mean([d[v]["head_cal_skill_pct"] for v in VARS])
        print(f"  +{h:<3} raw head {raw:>+7.1f}%   calibrated head {cal:>+7.1f}%")

    with open(os.path.join(DATA_DIR, "headroom_diagnostic.json"), "w",
              encoding="utf-8", newline="\n") as f:
        json.dump(all_out, f, indent=2, default=float)
    print(f"\nwritten: {os.path.join(DATA_DIR, 'headroom_diagnostic.json')}")


if __name__ == "__main__":
    main()
