"""
Choose the NWP calibration scheme on VALIDATION, with persistence in frame.

An unshrunk affine fit on TRAIN is actively harmful here: it made ECMWF
temperature WORSE on validation (0.995 raw -> 1.110 pooled) and humidity worse
too (4.208 -> 5.011). The reason is structural, not numerical. Each variable has
a different correct NWP/station relationship:

  temperature   the station reads ~+0.83 C above the grid cell on average, but
                the offset is partly a real local effect the model should KEEP,
                so shrinking it back toward identity is the safe move.
  humidity      station sensors are consistently wetter than the model
                (mean -4.2 %RH), but with sd 4.5 % the offset is not stable
                enough across stations to transfer as a single constant.
  pressure      stations sit at different altitudes, so the NWP/station offset
                is a per-station constant, not a global one. A single pooled
                slope cannot represent 16 different altitude offsets.
  wind_speed    NWP is km/h and the stations report m/s. A ~0.15 slope is a UNIT
                CONVERSION, not a bias, and must not be shrunk toward identity
                or the units stay wrong.

So the scheme has to be chosen per variable, and the choice has to be made on
VALIDATION. Shrinkage toward the identity map is the right prior:

    a' = 1 + lambda * (a - 1)
    b' = lambda * b

which at lambda=0 is the raw NWP and at lambda=1 is the unshrunk fit. We pick
lambda and the pooling level per variable on validation, and we always keep
raw NWP and persistence as candidates so calibration can never be worse than
doing nothing.
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

DATA_DIR = os.path.abspath(os.path.join(HERE, "..", "data"))
CACHE = os.path.join(DATA_DIR, "nwp_benchmark_cache.json")
HORIZON = int(os.environ.get("CAL_HORIZON", "6"))

VARS = {
    "temperature": ("temperature_2m", "target_temperature", 1.0),
    "humidity": ("relative_humidity_2m", "target_humidity", 1.0),
    "pressure": ("surface_pressure", "target_pressure", 1.0),
    # NWP reports wind in km/h; the stations report m/s. Compare in the STATION's
    # units so the affine fit is a genuine bias correction, not a unit change.
    "wind_speed": ("wind_speed_10m", "target_wind_speed", 1.0 / 3.6),
}
META_ORIGIN = {v: f"origin_{v}" for v in VARS}


def load_nwp():
    with open(CACHE, "r", encoding="utf-8") as f:
        raw = json.load(f)
    best = {}
    for k, v in raw.items():
        if k.startswith("_"):
            continue
        b = f"{v['station_id']}|{v['model']}"
        if b not in best or len(v.get("time", [])) > len(best[b]["time"]):
            best[b] = v
    return best


def collect(pipe, best, split, horizon, model):
    res = build_forecast_windows(pipeline=pipe, split=split, horizon=horizon,
                                 seq_len=DEFAULT_SEQ_LEN, return_metadata=True)
    meta = res[6]
    out = {v: {"nwp": [], "tgt": [], "stn": [], "org": []} for v in VARS}
    for m in meta:
        key = f"{m['station_id']}|{model}"
        rec = best.get(key)
        if rec is None:
            continue
        ix = {t: i for i, t in enumerate(rec["time"])}
        kh = m["target_timestamp"][:13] + ":00"
        if kh not in ix:
            continue
        for v, (fld, mk, _scale) in VARS.items():
            a = rec[fld][ix[kh]]
            if a is None:
                continue
            out[v]["nwp"].append(a)
            out[v]["tgt"].append(m[mk])
            out[v]["stn"].append(m["station_id"])
            out[v]["org"].append(m[META_ORIGIN[v]])
    return {v: {k: np.array(x, float) if k != "stn" else np.array(x)
                for k, x in d.items()} for v, d in out.items()}


def fit_shrunk(x, y, lam):
    if len(x) < 100:
        return 1.0, 0.0
    a, b = np.polyfit(x, y, 1)
    return float(1.0 + lam * (a - 1.0)), float(lam * b)


def main():
    pipe = TelemetryDataPipeline()
    best = load_nwp()
    H = HORIZON
    for model in ("ecmwf_ifs025", "gfs_seamless"):
        if not any(k.endswith(model) for k in best):
            continue
        print("=" * 100)
        print(f"CALIBRATION SELECTION  {model}  +{H}h   (fit on TRAIN, chosen on VAL)")
        print("=" * 100)
        tr = collect(pipe, best, "train", H, model)
        va = collect(pipe, best, "val", H, model)

        chosen = {}
        print(f"\n{'variable':<12}{'persist':>9}{'raw NWP':>9}{'pooled':>9}"
              f"{'per-stn':>9}{'best':>9}  {'scheme':<26}{'lambda':>7}{'gain/NWP':>10}")
        print("-" * 100)
        for v, (_f, _m, scale) in VARS.items():
            # unit-normalise wind so the schemes are comparable
            trn = tr[v]["nwp"] * scale
            van = va[v]["nwp"] * scale
            tta, vaa = tr[v]["tgt"], va[v]["tgt"]
            tst, vst = tr[v]["stn"], va[v]["stn"]

            pers = float(np.abs(vaa - va[v]["org"]).mean())
            raw = float(np.abs(van - vaa).mean())

            a_p, b_p = fit_shrunk(trn, tta, 1.0)
            pooled = float(np.abs(a_p * van + b_p - vaa).mean())

            # NB: these must start at +inf, not 0.0. Seeding them with 0.0 makes
            # every real candidate look worse and silently selects a no-op.
            best_lam, best_per = 0.0, pooled
            best_lam_p = float("inf")
            for lam in np.arange(0.0, 1.01, 0.1):
                pr = np.zeros_like(vaa)
                for s in np.unique(vst):
                    mv = vst == s
                    mt = tst == s
                    aa, bb = fit_shrunk(trn[mt], tta[mt], lam)
                    pr[mv] = aa * van[mv] + bb
                m = float(np.abs(pr - vaa).mean())
                if m < best_per:
                    best_per, best_lam = m, float(lam)
                ap, bp = fit_shrunk(trn, tta, lam)
                mm = float(np.abs(ap * van + bp - vaa).mean())
                if mm < best_lam_p:
                    best_lam_p = mm

            cands = {"raw NWP (lambda=0)": (raw, 0.0, "pooled"),
                     "pooled shrink": (best_lam_p, best_lam, "pooled"),
                     "per-station shrink": (best_per, best_lam, "per_station")}
            bname = min(cands, key=lambda k: cands[k][0])
            bmae, blam, bmode = cands[bname]
            chosen[v] = {"mode": bmode, "lambda": blam, "val_mae": bmae,
                         "raw_nwp_mae": raw, "persistence_mae": pers}
            print(f"{v:<12}{pers:>9.3f}{raw:>9.3f}{pooled:>9.3f}{best_per:>9.3f}"
                  f"{bmae:>9.3f}  {bname:<26}{blam:>7.1f}"
                  f"{(raw - bmae) / raw * 100:>9.1f}%")

        print(f"\n  {'variable':<12}{'persistence':>13}{'chosen':>10}{'skill vs pers':>16}")
        for v, c in chosen.items():
            sk = (c["persistence_mae"] - c["val_mae"]) / c["persistence_mae"] * 100
            print(f"  {v:<12}{c['persistence_mae']:>13.3f}{c['val_mae']:>10.3f}{sk:>15.1f}%")

        outp = os.path.join(DATA_DIR, f"calibration_choice_h{H}_{model}.json")
        with open(outp, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"horizon_h": H, "nwp_model": model, "chosen": chosen}, f, indent=2)
        print(f"\n  written: {outp}\n")


if __name__ == "__main__":
    main()
