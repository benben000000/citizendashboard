"""
Does neighbouring-station telemetry carry signal the local window does not?

This is the decisive cheap experiment. If a neighbouring station's current
reading does NOT improve the prediction of this station's next-hour value beyond
what the station's own history already provides, then a spatial-context model
cannot help and retraining with neighbour inputs is not worth the compute.

For each station we compare, over the TEST split:

    persistence      predict t+1h from this station's own t
    neighbour-x      predict t+1h from the x-th nearest station's t
    persistence      MAE of this station's own t -> t+1h  (the thing we must beat)
    neighbour-x      MAE of neighbour's t -> this station's t+1h

A neighbour is only useful if its MAE is LOWER than persistence, i.e. if a
distant station's current reading predicts this station's future better than this
station's own current reading does. That is a strong statement and rarely true
for persistence, but advection can make it true during moving weather.

We also report the "upwind" special case: the neighbour whose bearing points
toward the station, which is the one physically capable of carrying an advancing
signal.
"""

import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from dataset import TelemetryDataPipeline  # noqa: E402

DATA_DIR = os.path.abspath(os.path.join(HERE, "..", "data"))


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def bearing(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


HORIZON = int(os.environ.get("SIGNAL_HORIZON", "1"))
TENDENCY_H = 3  # hours of neighbour history used for the tendency feature


def main():
    pipe = TelemetryDataPipeline()
    coords = json.load(open(os.path.join(DATA_DIR, "station_coords.json"), encoding="utf-8"))
    stations = sorted(pipe.station_hourly)
    lat = {s: float(coords[s]["lat"]) for s in stations}
    lon = {s: float(coords[s]["lon"]) for s in stations}

    # nearest neighbours, nearest first
    nbrs = {}
    for s in stations:
        d = sorted(
            ((haversine(lat[s], lon[s], lat[o], lon[o]), o) for o in stations if o != s)
        )
        nbrs[s] = [o for _, o in d]

    H = HORIZON
    print("=" * 92)
    print(f"NEIGHBOUR SIGNAL TEST at +{H}h — can a neighbour beat our own persistence?")
    print("=" * 92)
    print(f"TEST split only. MAE of predicting t+{H}h from t. Lower is better.")
    print(f"Neighbour features tested: raw value, and {TENDENCY_H}h tendency.\n")

    VARS = ["temperature", "humidity", "pressure", "precipitation"]
    K = 3
    rows = []

    for s in stations:
        hours = sorted(h for h in pipe.station_hourly[s] if h > pipe.test_start)
        for step in range(TENDENCY_H, len(hours) - H):
            t0 = hours[step - 1]
            t1 = hours[step - 1 + H]
            own = pipe.station_hourly[s].get(t1)
            if own is None:
                continue
            rec = {"station": s}
            for v in VARS:
                p_now = pipe.station_hourly[s][t0][v]
                p_prev = pipe.station_hourly[s][hours[step - 1 - TENDENCY_H]][v]
                rec[f"pers_{v}"] = abs(own[v] - p_now)
                rec[f"owntend_{v}"] = abs(own[v] - (p_now + (p_now - p_prev)))
                for k in range(K):
                    o = nbrs[s][k]
                    r_now = pipe.station_hourly[o].get(t0)
                    r_prev = pipe.station_hourly[o].get(hours[step - 1 - TENDENCY_H])
                    if r_now is None or r_prev is None:
                        rec[f"nbr{k}_{v}"] = None
                        rec[f"nbrtend{k}_{v}"] = None
                    else:
                        rec[f"nbr{k}_{v}"] = abs(own[v] - r_now[v])
                        rec[f"nbrtend{k}_{v}"] = abs(own[v] - (r_now[v] + (r_now[v] - r_prev[v])))
            rows.append(rec)

    print(f"observations: {len(rows)} station-hours at +{H}h\n")
    print(f"{'variable':<15}{'persistence':>13}{'own tend':>11}{'nbr1':>10}{'nbr tend1':>11}"
          f"{'best gain':>12}")
    print("-" * 72)
    summary = {"horizon_h": H, "n_observations": len(rows), "variables": {}}
    for v in VARS:
        pers = np.array([r[f"pers_{v}"] for r in rows], float)
        own_t = np.array([r[f"owntend_{v}"] for r in rows], float)
        n1 = np.array([x for x in (r[f"nbr0_{v}"] for r in rows) if x is not None], float)
        nt1 = np.array([x for x in (r[f"nbrtend0_{v}"] for r in rows) if x is not None], float)
        cands = {"own tendency": own_t.mean(), "neighbour1": n1.mean(),
                 "neighbour1 tendency": nt1.mean()}
        best_name = min(cands, key=cands.get)
        gain = (pers.mean() - cands[best_name]) / pers.mean() * 100
        print(f"{v:<15}{pers.mean():>13.3f}{own_t.mean():>11.3f}{n1.mean():>10.3f}"
              f"{nt1.mean():>11.3f}{gain:>+11.1f}%  ({best_name})")
        summary["variables"][v] = {
            "persistence": float(pers.mean()),
            "own_tendency": float(own_t.mean()),
            "neighbour1": float(n1.mean()),
            "neighbour1_tendency": float(nt1.mean()),
            "best_gain_pct": float(gain), "best_source": best_name,
        }

    print("\nInterpretation")
    print("-" * 72)
    for v, s in summary["variables"].items():
        g = s["best_gain_pct"]
        verdict = "USEFUL" if g > 1.0 else ("marginal" if g > 0 else "NOT useful")
        print(f"  {v:<15} best non-persistence source is {g:+.1f}% vs persistence -> {verdict}")

    out = os.path.join(DATA_DIR, f"neighbour_signal_h{H}.json")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
