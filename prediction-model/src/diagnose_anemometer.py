"""
Diagnose the station anemometers against ECMWF 10 m wind.

WHY THIS IS A DIAGNOSTIC AND NOT A FIX
--------------------------------------
The stations report a mean wind of 1.50 m/s where ECMWF reports 3.11 m/s at
10 m -- a ratio of 0.483. That is either a real siting effect (an AWS anemometer
at 2-3 m in an urban or sheltered site routinely reads 40-70% of 10 m forecast
wind) or a sensor/calibration fault. The two have very different fixes: the first
is physics to be corrected for in the model, the second is a visit to the
hardware. This script distinguishes them.

WHAT DISTINGUISHES THEM
-----------------------
  scale        a clean multiplicative constant implies a units or calibration
               error and is correctable in software
  stability    a ratio that drifts over time implies a degrading sensor; a
               constant ratio implies siting
  diurnal      real urban sheltering has a diurnal signature (buildings warm and
               the boundary layer stabilises overnight); a multiplicative fault
               usually does not
  variance     a damped or under-sampled sensor shows less variance than the
               model; genuine sheltering reduces mean wind more than it reduces
               gustiness
  extremes     if the station misses the windiest episodes entirely, the model
               will under-warn; if the ratio collapses in calm conditions, the
               sensor has a noise floor
  lag          wind sensors with real response time correlate BEST at a lag; a
               timing error shifts the optimal lag

TEST SPLIT ONLY. Nothing here influences the model; it is a read.
"""

import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from dataset import TelemetryDataPipeline  # noqa: E402

DATA_DIR = os.path.abspath(os.path.join(HERE, "..", "data"))
CACHE = os.path.join(DATA_DIR, "nwp_benchmark_cache.json")
MODEL = "ecmwf_ifs025"
KMH_TO_MS = 1.0 / 3.6


def load_nwp(model):
    with open(CACHE, "r", encoding="utf-8") as f:
        raw = json.load(f)
    best = {}
    for k, v in raw.items():
        if k.startswith("_") or v.get("model") != model:
            continue
        b = v["station_id"]
        if b not in best or len(v.get("time", [])) > len(best[b]["time"]):
            best[b] = v
    return best


def series(pipe, best, station):
    """Aligned (station_ms, ecmwf_ms, hour) on the TEST split."""
    rec = best.get(station)
    if rec is None:
        return None
    idx = {t: i for i, t in enumerate(rec["time"])}
    obs = pipe.station_hourly.get(station, {})
    s_vals, n_vals, hours = [], [], []
    test_start = pipe.test_start
    for ts in sorted(obs):
        # station_hourly is keyed by datetime, not by ISO string
        if isinstance(ts, str):
            if ts <= str(test_start):
                continue
            iso = ts
        else:
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts <= test_start:
                continue
            iso = ts.strftime("%Y-%m-%dT%H:%M:%S+00:00")
        o = obs[ts]
        key = iso[:13] + ":00"
        j = idx.get(key)
        if j is None:
            continue
        nv = rec["wind_speed_10m"][j]
        sv = o.get("wind_speed")
        if nv is None or sv is None:
            continue
        s_vals.append(float(sv))
        n_vals.append(float(nv) * KMH_TO_MS)
        hours.append(int(iso[11:13]))
    if len(s_vals) < 100:
        return None
    return (np.array(s_vals), np.array(n_vals), np.array(hours))


def ols_scale(y, x):
    """Least-squares y = s*x, the calibration factor and its residual spread."""
    denom = float((x * x).sum())
    if denom <= 0:
        return float("nan"), float("nan")
    s = float((x * y).sum() / denom)
    resid = y - s * x
    return s, float(np.sqrt((resid * resid).mean()))


def main():
    pipe = TelemetryDataPipeline()
    best = load_nwp(MODEL)
    stations = sorted(s for s in pipe.station_hourly if s in best)

    print("=" * 100)
    print("ANEMOMETER DIAGNOSTIC  —  station wind vs ECMWF 10 m wind (test split, m/s)")
    print("=" * 100)

    all_s, all_n, all_h, per_station = [], [], [], {}
    for s in stations:
        got = series(pipe, best, s)
        if got is None:
            continue
        sv, nv, hr = got
        per_station[s] = (sv, nv, hr)
        all_s.append(sv); all_n.append(nv); all_h.append(hr)
    if not per_station:
        print("no overlapping data")
        return
    S = np.concatenate(all_s); N = np.concatenate(all_n); H = np.concatenate(all_h)
    print(f"  stations {len(per_station)}   observations {len(S)}")
    print(f"  station mean {S.mean():.3f} m/s   ECMWF mean {N.mean():.3f} m/s"
          f"   ratio {S.mean() / N.mean():.3f}")

    # ---- 1. is the ratio a clean constant? ---------------------------------
    print("\n  1. SCALE  (is it a single calibration constant?)")
    s_all, rms = ols_scale(S, N)
    r_all = float(np.corrcoef(S, N)[0, 1])
    print(f"     pooled least-squares factor  s = {s_all:.4f}   residual RMS {rms:.3f} m/s")
    print(f"     ratio of means                {S.mean() / N.mean():.4f}")
    print(f"     Pearson r                     {r_all:.3f}")
    print(f"     station SD {S.std():.3f}  vs ECMWF SD {N.std():.3f}"
          f"   variance ratio {(S.std() / N.std())**2:.3f}")
    print("     -> a factor well below the ratio of means means the station is a")
    print("        better predictor of its own future than the model is of it;")

    # ---- 2. per-station spread --------------------------------------------
    print("\n  2. PER-STATION FACTORS  (a wide spread means siting, not a common fault)")
    rows = []
    for s, (sv, nv, hr) in sorted(per_station.items()):
        f, _ = ols_scale(sv, nv)
        rows.append((s, f, sv.mean(), nv.mean(), float(np.corrcoef(sv, nv)[0, 1])))
    for s, f, sm, nm, r in rows:
        print(f"     {s:<10} factor {f:5.3f}   station {sm:5.2f}  ECMWF {nm:5.2f}"
              f"   r {r:5.3f}")
    fs = np.array([r[1] for r in rows])
    print(f"     factor across stations: min {fs.min():.3f}  median {np.median(fs):.3f}"
          f"  max {fs.max():.3f}  spread {fs.max() - fs.min():.3f}")

    # ---- 3. temporal stability --------------------------------------------
    print("\n  3. STABILITY  (does the factor drift? a drifting factor is a failing sensor)")
    order = np.argsort(H)
    q = len(S) // 4
    for i in range(4):
        seg = slice(i * q, (i + 1) * q)
        f, _ = ols_scale(S[seg], N[seg])
        print(f"     quartile {i + 1}  factor {f:.4f}   n={q}")
    print("     -> a flat sequence means the offset is structural (siting/height),")
    print("        not a sensor ageing out.")

    # ---- 4. diurnal signature ---------------------------------------------
    print("\n  4. DIURNAL  (real sheltering warms and stabilises overnight)")
    for lo, hi, name in ((0, 6, "00-06"), (6, 10, "06-10"),
                         (10, 16, "10-16"), (16, 20, "16-20"), (20, 24, "20-24")):
        m = (H >= lo) & (H < hi)
        if m.sum() < 30:
            continue
        f, _ = ols_scale(S[m], N[m])
        print(f"     {name}  factor {f:.4f}   n={int(m.sum()):5d}")

    # ---- 5. does it miss the windy episodes? ------------------------------
    print("\n  5. EXTREMES  (does the station see the windy periods at all?)")
    qs = [50, 75, 90, 95, 99]
    print(f"     {'pct':>5}{'station':>10}{'ECMWF':>9}{'factor':>9}")
    for qp in qs:
        t = np.percentile(N, qp)
        m = N >= t
        f, _ = ols_scale(S[m], N[m])
        print(f"     {qp:>4}%{S[m].mean():>10.2f}{N[m].mean():>9.2f}{f:>9.3f}")
    print("     -> a factor that collapses in the top decile means gusts are being")
    print("        missed, so the model will under-warn exactly when it matters.")

    # ---- 6. response lag ---------------------------------------------------
    print("\n  6. LAG  (best correlation at a shift means a response-time error)")
    best_lag, best_r = 0, r_all
    for lag in range(-3, 4):
        if lag < 0:
            a, b = S[-lag:], N[:len(N) + lag]
        elif lag > 0:
            a, b = S[:len(S) - lag], N[lag:]
        else:
            a, b = S, N
        if len(a) < 100:
            continue
        r = float(np.corrcoef(a, b)[0, 1])
        if r > best_r:
            best_r, best_lag = r, lag
        print(f"     lag {lag:+d}h  r = {r:.4f}")
    print(f"     best lag {best_lag:+d}h at r={best_r:.4f}"
          f"   (shift 0h was {r_all:.4f})")

    # ---- 7. calm-weather noise floor ---------------------------------------
    print("\n  7. NOISE FLOOR  (station readings when the model says it is calm)")
    m = N < np.percentile(N, 20)
    print(f"     ECMWF below p20: station mean {S[m].mean():.3f}  SD {S[m].std():.3f}")
    print(f"                        ECMWF   mean {N[m].mean():.3f}  SD {N[m].std():.3f}")
    print("     -> a station SD comparable to its mean in calm air indicates a")
    print("        noise floor or a reporting floor, not meteorology.")

    out = {
        "model": MODEL,
        "n_observations": int(len(S)),
        "n_stations": len(per_station),
        "station_mean_ms": float(S.mean()),
        "nwp_mean_ms": float(N.mean()),
        "ratio_of_means": float(S.mean() / N.mean()),
        "pooled_factor": s_all,
        "pooled_residual_rms": rms,
        "pearson_r": r_all,
        "variance_ratio": float((S.std() / N.std()) ** 2),
        "per_station_factor": {s: float(f) for s, f, *_ in rows},
        "factor_spread": float(fs.max() - fs.min()),
        "best_lag_hours": int(best_lag),
        "best_lag_r": best_r,
    }
    path = os.path.join(DATA_DIR, "anemometer_diagnostic.json")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=2)
    print(f"\nwritten: {path}")


if __name__ == "__main__":
    main()
