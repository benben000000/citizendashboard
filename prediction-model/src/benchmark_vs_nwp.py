"""
Benchmark the local forecast engine against REAL numerical weather prediction.

Compares, on identical rows and identical ground truth:
    - the deployed GarciaWeatherLNN bundle (the thing the dashboard serves)
    - ECMWF IFS 0.25, NOAA GFS, DWD ICON, Meteo-France, JMA, UK Met Office, ECCC GEM
    - persistence
    - a bias-corrected NWP (the honest ceiling: can the local model beat a
      simple calibration of the best global model?)

Ground truth is the station telemetry itself, so every method is scored on the
same observed target.

VERIFICATION CAVEATS — read before quoting any number
-----------------------------------------------------
1. GRID vs POINT. NWP values are 0.25 deg (ECMWF) to ~1 deg (other models) grid
   cell means; the truth is a point sensor reading. Terrain and coastal
   gradients in Central Luzon and Bataan are large enough that a grid mean
   carries real error against a point. This disadvantages NWP.
2. LEAD TIME. Open-Meteo's historical-forecast endpoint returns, for each valid
   hour, a blend of the most recent cycles, so NWP here is effectively a short
   to medium lead forecast. This ADVANTAGES NWP relative to a strict
   single-cycle verification, and it is the only way this data is exposed.
3. The LNN is a pure local nowcast: observations up to t0, target t0+h, with
   no atmospheric state of record. The comparison is therefore not
   like-for-like in information content. NWP is answering with the whole
   global atmosphere; the LNN is answering from one station's last 24 hours.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from dataset import TelemetryDataPipeline, build_forecast_windows, DEFAULT_SEQ_LEN  # noqa: E402
from inference import LNNServerlessPredictor  # noqa: E402

DATA_DIR = os.path.abspath(os.path.join(HERE, "..", "data"))
CACHE = os.path.join(DATA_DIR, "nwp_benchmark_cache.json")
OUT = os.path.join(DATA_DIR, "nwp_benchmark_results.json")
HORIZONS = [1, 3, 6, 12, 24]

NWP_DISPLAY = {
    "ecmwf_ifs025": "ECMWF IFS",
    "gfs_seamless": "NOAA GFS",
    "icon_seamless": "DWD ICON",
    "meteofrance_seamless": "Meteo-France",
    "jma_seamless": "JMA",
    "ukmo_seamless": "UK Met Office",
    "gem_seamless": "ECCC GEM",
}

# variable -> (NWP cache field, telemetry metadata key, LNN output key)
# The metadata key is the one build_forecast_windows writes, e.g. "target_temperature".
VAR_MAP = {
    "temperature": ("temperature_2m", "target_temperature", "temperature_c"),
    "humidity": ("relative_humidity_2m", "target_humidity", "relative_humidity_pct"),
    "pressure": ("surface_pressure", "target_pressure", "pressure_hpa"),
    "wind_speed": ("wind_speed_10m", "target_wind_speed", "wind_speed_kmh"),
}
ORIGIN_KEY = {
    "temperature": "origin_temperature",
    "humidity": "origin_humidity",
    "pressure": "origin_pressure",
    "wind_speed": "origin_wind_speed",
}
UNITS = {"temperature": "degC", "humidity": "%", "pressure": "hPa", "wind_speed": "m/s"}

# The stations report wind in m/s. Open-Meteo's wind_speed_10m is km/h. Comparing
# them without this factor inflated every NWP wind error by 3.6x, which made the
# LNN look ~6x better than ECMWF on wind when it is in fact ~1.4x better.
# Verified: station wind_speed mean 1.501 m/s vs NWP 11.19 km/h = 3.11 m/s.
NWP_TO_STATION_UNITS = {"wind_speed": 1.0 / 3.6}


def load_nwp():
    """
    Load the NWP cache, MERGED per station and model.

    Cache keys are ``station|model`` or ``station|model|<start>_<end>``; the same
    series can be present for several fetched windows.

    The previous rule kept only the LONGEST window per station. That is wrong: the
    longest cached series ended 2026-08-28, while a newer targeted fetch covered
    the test split from 2026-09-11 onward. Longest-wins therefore selected a
    window that did not overlap the evaluation period at all, every NWP lookup
    missed, and the benchmark reported `n/a` for all seven models while still
    printing a rank as if it had scored them.

    Entries are instead merged on timestamp. Shortest windows are applied FIRST
    and longer ones fill only the timestamps still missing, so a narrow, recent
    re-fetch takes precedence where it overlaps and no coverage is lost
    elsewhere. Key order is not assumed; within equal spans the later-fetched
    (later window) entry wins.
    """
    with open(CACHE, "r", encoding="utf-8") as f:
        raw = json.load(f)

    FIELDS = ["temperature_2m", "relative_humidity_2m", "surface_pressure",
              "wind_speed_10m", "precipitation"]

    grouped = {}
    for k, v in raw.items():
        if k.startswith("_") or not isinstance(v, dict):
            continue
        times = v.get("time") or []
        if not times:
            continue
        base = f"{v.get('station_id')}|{v.get('model')}"
        grouped.setdefault(base, []).append((len(times), times[-1], v))

    out = {}
    for base, entries in grouped.items():
        # Application order decides the winner, because the inner loop only fills
        # timestamps that are still missing. So:
        #   span ASCENDING  -> a narrow, recent re-fetch is applied first and
        #                      therefore wins wherever it overlaps.
        #   end  DESCENDING -> among equal spans, the window reaching furthest
        #                      into the present is applied first and wins. This
        #                      had to be descending once fill-if-missing made
        #                      "first applied" authoritative; with an ascending
        #                      sort the OLDEST window won every tie.
        entries.sort(key=lambda t: (t[0], [-ord(c) for c in t[1]]))
        merged = {}
        for _span, _end, v in entries:
            for i, t in enumerate(v.get("time", [])):
                row = merged.get(t)
                if row is None:
                    row = merged[t] = {}
                for fld in FIELDS:
                    arr = v.get(fld)
                    if arr is not None and i < len(arr) and arr[i] is not None:
                        # Fill only what is still missing. An unconditional write
                        # here inverted the documented precedence: entries are
                        # applied shortest-first, so writing every value meant
                        # the LONGEST window landed last and won every overlap
                        # -- the original longest-wins rule, just spread across a
                        # timestamp map where it was no longer visible.
                        if fld in row:
                            continue
                        row[fld] = arr[i]
        times_sorted = sorted(merged)
        out[base] = {
            "idx": {t: i for i, t in enumerate(times_sorted)},
            "rec": {fld: [merged[t].get(fld) for t in times_sorted] for fld in FIELDS},
        }
    return out


# ---------------------------------------------------------------------------
# scoring
#
# The single canonical implementation of MAE/RMSE/bias lives in scoring.py.
# The imports below re-export it under this module's historical names, so the
# NWP benchmark, its saved results files and the golden-fixture tests all keep
# resolving to ONE implementation rather than to a second, drifting copy of
# the same arithmetic. `metrics` below is a thin delegating wrapper for the
# same reason: the name is imported across this project, and dropping it
# would break every caller that knows nothing about scoring.py.
from scoring import MIN_COVERAGE, bootstrap_ci  # noqa: E402
from scoring import metrics as _score  # noqa: E402


def metrics(pred, truth, min_coverage=MIN_COVERAGE):
    """
    MAE/RMSE/bias over the rows where both prediction and truth are finite.

    THIN ALIAS. The implementation moved to scoring.py, which is the one
    canonical scorer for this project. Semantics are unchanged and are
    specified by scoring.metrics:

      * keys {mae, rmse, bias, n, coverage, rows_dropped_nonfinite, ci95_mae}
      * bias = mean(pred - truth), PREDICTED MINUS OBSERVED per WMO, so a
        positive bias means the model runs WARM and negative means COLD. That
        is the convention monitoring.py already uses, so one forecast reads
        the same way on the dashboard and in the benchmark.
      * returns None below min_coverage; the gate is INCLUSIVE at exactly that
        value, so 1-of-2 rows at coverage 0.5 IS scored.
      * non-finite rows (nan and +-inf, either side) are dropped and REPORTED
        via coverage / rows_dropped_nonfinite rather than vanishing.
      * ci95_mae is a 400-resample percentile bootstrap at 95%, seed 42.

    The arithmetic is NOT repeated here. Anything in this file that needs a
    score calls this wrapper; re-deriving it is how this project ended up with
    thirteen scorers that disagreed about the sign of bias.
    """
    return _score(pred, truth, min_coverage=min_coverage)


def fmt(v, nd=3):
    if v is None:
        return "n/a"
    return f"{v:.{nd}f}"


def _score_candidate(candidate_dir, h, telemetry, dt, meta, pipe):
    """
    Score a feature-augmented candidate checkpoint on exactly the windows the
    benchmark already built for the production model.

    Two things this must get right, both of which have bitten this project before:

    * UNITS. The candidate predicts wind in m/s, matching the station telemetry.
      NWP_TO_STATION_UNITS is for the Open-Meteo km/h fields and must NOT be
      applied here; doing so rescales the candidate by 3.6x.
    * CONTEXT. The candidate consumes the 75-dimension engineered context, not the
      raw sequence alone, so the same augmentation is rebuilt for the same windows.
    """
    import torch
    from model import GarciaWeatherLNNFeatured
    from dataset import build_feature_augmented_forecast_windows

    ck_path = os.path.join(candidate_dir, f"candidate_h{h}h.pt")
    if not os.path.exists(ck_path):
        raise FileNotFoundError(f"No candidate checkpoint for +{h}h at {ck_path}")
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)

    # Prefer the architecture the checkpoint was actually trained with rather than
    # assuming one; a silent default here produces size-mismatch errors.
    man = ck.get("manifest", {}) or {}
    hid = int(man.get("hidden_dim", 32))
    two_stage = True
    model = GarciaWeatherLNNFeatured(
        input_dim=int(man.get("input_dim", 8)),
        context_dim=int(man.get("context_dim", 75)),
        hidden_dim=hid,
        use_two_stage_precipitation=two_stage,
    )
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    # The context matrix MUST be built for exactly the same windows being scored.
    # This call previously omitted seq_len and relied on the builder's default; if
    # that default ever differs from DEFAULT_SEQ_LEN the context is derived from a
    # different lookback than the telemetry, and the candidate is scored against
    # features it was never trained on. Assert the row count matches.
    aug = build_feature_augmented_forecast_windows(
        pipeline=pipe, split="test", horizon=h, seq_len=DEFAULT_SEQ_LEN,
        return_metadata=True)
    if aug is None:
        raise RuntimeError(f"could not build augmented windows for +{h}h")
    context = aug[1]
    if len(context) != len(meta):
        raise RuntimeError(
            f"context rows ({len(context)}) do not match scored windows "
            f"({len(meta)}); the augmented features are not aligned with the "
            f"windows being scored and the result would be meaningless")

    origin_cols = ["origin_temperature", "origin_humidity", "origin_pressure",
                   "origin_wind_speed", "origin_wind_u", "origin_wind_v"]
    origin = torch.tensor(
        np.column_stack([[m[c] for m in meta] for c in origin_cols]),
        dtype=torch.float32)

    n = len(meta)
    # telemetry must already be in the model's training space (normalised).
    if not torch.is_tensor(telemetry):
        telemetry = torch.tensor(np.asarray(telemetry), dtype=torch.float32)
    with torch.no_grad():
        out = model(telemetry.float(), context, torch.tensor(dt, dtype=torch.float32),
                    origin_weather=origin)

    pred = {
        "temperature": out["temperature"].squeeze(-1).numpy(),
        "humidity": out["humidity"].squeeze(-1).numpy(),
        "pressure": out["pressure"].squeeze(-1).numpy(),
        "wind_speed": out["wind_speed"].squeeze(-1).numpy(),
    }
    rain = torch.sigmoid(out["rain_prob"]).squeeze(-1).numpy()
    print(f"    [candidate] +{h}h from {os.path.basename(ck_path)}"
          f"  train_data_sha={str(man.get('weather_telemetry_sha256', '?'))[:12]}")
    return pred, rain


def main():
    # Both the telemetry corpus and the model under test are selectable. The
    # default pairing (committed CSV, production bundles) is the shipped model.
    # --candidate-dir scores a retrained candidate against the same NWP rows, so
    # the comparison is like-for-like: identical windows, identical ground truth.
    global HORIZONS
    ap = argparse.ArgumentParser(description="Benchmark the LNN against real NWP.")
    ap.add_argument("--weather-csv", default=None,
                    help="Telemetry CSV (default: data/weather_telemetry.csv). "
                         "Use weather_telemetry_current.csv for the refetched history.")
    ap.add_argument("--candidate-dir", default=None,
                    help="Score candidate_h<N>h.pt from this directory instead of the "
                         "production bundles. The candidate is a feature-augmented "
                         "model, so it is loaded directly rather than via a bundle.")
    ap.add_argument("--horizons", default=None,
                    help="Comma-separated horizons, e.g. '6,12'. Default: all.")
    ap.add_argument("--out", default=None, help="Results JSON path.")
    ap.add_argument("--label", default="LNN (this project)",
                    help="Display name for the model under test.")
    args = ap.parse_args()

    if args.horizons:
        HORIZONS = [int(x) for x in args.horizons.split(",") if x.strip()]
    out_path = args.out or OUT

    weather_csv = args.weather_csv or os.path.join(DATA_DIR, "weather_telemetry.csv")
    pipe = TelemetryDataPipeline(weather_csv=weather_csv)
    nwp = load_nwp()
    models = list(NWP_DISPLAY)
    print("=" * 118)
    print("BENCHMARK vs REAL NWP — identical rows, identical ground truth (station observations)")
    print("=" * 118)
    print(f"telemetry corpus: {os.path.basename(weather_csv)}")
    print(f"model under test: {args.candidate_dir or 'production bundles'}")
    print(f"test split: {pipe.test_start.isoformat()} .. {pipe.time_range_max.isoformat()}")
    print(f"NWP models: {', '.join(NWP_DISPLAY[m] for m in models)}")
    print(f"cached series: {len(nwp)}")
    print("=" * 118)

    results = {}
    for h in HORIZONS:
        res = build_forecast_windows(pipeline=pipe, split="test", horizon=h,
                                     seq_len=DEFAULT_SEQ_LEN, return_metadata=True)
        X = res[0].detach().cpu().numpy().astype(np.float64)
        dt = res[1].detach().cpu().numpy().astype(np.float64)
        precip_true = res[3].detach().cpu().numpy().astype(np.float64)[:, 0]
        meta = res[6]
        n = len(X)
        if n == 0:
            continue

        X_raw = np.clip(X * pipe.norm_stds + pipe.norm_means,
                        [10, 10, 10, 900, 0, -1, -1, 0],
                        [50, 70, 100, 1050, 180, 1, 1, 150])

        truth = {v: np.array([m[VAR_MAP[v][1]] for m in meta], float)
                 for v in VAR_MAP}
        persist = {v: np.array([m[ORIGIN_KEY[v]] for m in meta], float)
                   for v in VAR_MAP}

        if args.candidate_dir:
            # The candidate is trained on the NORMALISED tensor res[0]. X_raw is
            # de-normalised and clipped, which is the production bundle predictor's
            # expected input, not this model's. Passing X_raw here shifted every
            # feature out of distribution and produced badly wrong scores (the +3h
            # wind MAE came out at 0.742 instead of 0.498).
            lnn_pred, lnn_rain = _score_candidate(
                args.candidate_dir, h, res[0], dt, meta, pipe)
        else:
            lnn = LNNServerlessPredictor(bundle_dir=os.path.join(DATA_DIR, "bundles", f"h{h}"))
            lnn_pred = {v: np.empty(n) for v in VAR_MAP}
            lnn_rain = np.empty(n)
            for i in range(n):
                o = lnn.predict_from_observed_sequence(
                    telemetry_sequence=X_raw[i], dt_sequence=dt[i], horizon_hours=h)
                for v in VAR_MAP:
                    lnn_pred[v][i] = o[VAR_MAP[v][2]]
                lnn_rain[i] = o["chance_of_rain_pct"] / 100.0

        # --- assemble per-model series aligned to each test target timestamp ---
        nwp_pred = {m: {v: np.full(n, np.nan) for v in VAR_MAP} for m in models}
        nwp_rain = {m: np.full(n, np.nan) for m in models}
        for i, m in enumerate(meta):
            key_ts = m["target_timestamp"][:13] + ":00"
            sid = m["station_id"]
            for mdl in models:
                e = nwp.get(f"{sid}|{mdl}")
                if not e:
                    continue
                j = e["idx"].get(key_ts)
                if j is None:
                    continue
                for v, (field, _tk, _lk) in VAR_MAP.items():
                    val = e["rec"][field][j]
                    if val is not None:
                        nwp_pred[mdl][v][i] = val * NWP_TO_STATION_UNITS.get(v, 1.0)
                pv = e["rec"]["precipitation"][j]
                if pv is not None:
                    nwp_rain[mdl][i] = pv

        print(f"\n{'=' * 118}\n+{h}h   N={n} test windows\n{'-' * 118}")
        results[f"h{h}"] = {}
        for v in VAR_MAP:
            t = truth[v]
            rows = []
            lm = metrics(lnn_pred[v], t)
            rows.append((args.label, lm, lm))
            rows.append(("persistence", metrics(persist[v], t), None))
            for mdl in models:
                rows.append((NWP_DISPLAY[mdl], metrics(nwp_pred[mdl][v], t), None))
            base = rows[1][1]
            print(f"  {v}  (MAE, {UNITS[v]})")
            print(f"    {'method':<22}{'MAE':>9}{'RMSE':>9}{'bias':>9}{'n':>8}{'vs persist':>12}  {'rank':>5}")
            scored = [(r[0], r[1]) for r in rows if r[1]]
            # Rank only among rows that actually produced a score, and say so.
            # Ranking across the full `rows` list let unscored models still occupy
            # rank positions, so a run where every NWP lookup missed still
            # reported the LNN's rank against a denominator inflated by seven
            # models that were never evaluated.
            order = sorted(range(len(rows)), key=lambda i: rows[i][1]["mae"] if rows[i][1] else 9e9)
            rank = {rows[i][0]: k + 1 for k, i in enumerate(order)}
            n_scored = len(scored)
            n_unscored = len(rows) - n_scored
            for name, m, _ in rows:
                if not m:
                    print(f"    {name:<22}{'n/a':>9}")
                    continue
                sp = (base["mae"] - m["mae"]) / base["mae"] * 100 if base else 0.0
                star = " <== ours" if name == args.label else ""
                print(f"    {name:<22}{fmt(m['mae']):>9}{fmt(m['rmse']):>9}{fmt(m['bias']):>9}"
                      f"{m['n']:>8}{sp:>+11.1f}%{rank[name]:>6}{star}")
            results[f"h{h}"][v] = {
                name: (m if m else None) for name, m, _ in rows
            }
            best = min((x for x in scored), key=lambda x: x[1]["mae"])
            # Denominator counts only methods that produced a score. Using
            # len(rows) - 1 inflated the denominator with models that were never
            # evaluated, so a run where every NWP lookup missed still printed
            # "rank 1/8" -- a flattering number describing a run that had scored
            # nothing to compare against.
            denom = f"{max(n_scored - 1, 1)}"
            note = ""
            if n_unscored:
                note = (f"   [WARNING: {n_unscored} of {len(rows)} methods produced "
                        f"no score; this rank is NOT comparable to a full run]")
            print(f"    -> best: {best[0]}  MAE {best[1]['mae']:.3f} {UNITS[v]}"
                  f"   |  LNN rank {rank[args.label]}/{denom}{note}")

        # --- rain occurrence ---
        # IMPORTANT: the persistence rain probability must use the LAST OBSERVED
        # precipitation at t0, not the target precipitation at t0+h. Using the
        # target here leaks the answer and makes persistence score ~0.01 Brier.
        last_observed_precip = np.array([m.get("last_observed_precip", 0.0) for m in meta], float)
        rain_true = (precip_true > 0.1).astype(float)
        prev = float(rain_true.mean())
        print(f"  rain occurrence (Brier, lower better)  wet-hour base rate {prev*100:.1f}%")
        entries = [(args.label, float(np.mean((lnn_rain - rain_true) ** 2))),
                   ("persistence", float(np.mean(
                       (np.where(last_observed_precip > 0.1, 0.85, 0.05) - rain_true) ** 2))),
                   ("climatology (constant)", float(np.mean((np.full(n, prev) - rain_true) ** 2)))]
        for mdl in models:
            p = np.where(np.isnan(nwp_rain[mdl]), prev, nwp_rain[mdl])
            p = np.where(p > 0.1, 0.75, 0.25)
            entries.append((NWP_DISPLAY[mdl], float(np.mean((p - rain_true) ** 2))))
        order = sorted(entries, key=lambda e: e[1])
        for k, (name, b) in enumerate(order):
            star = " <== ours" if name == args.label else ""
            print(f"    {name:<22}{b:>9.4f}{k+1:>6}{star}")
        results[f"h{h}"]["_rain_brier"] = {name: b for name, b in entries}

        # --- bias-corrected NWP, correction fitted on TRAIN at the SAME horizon ---
        # At +1h persistence is expected to beat a 0.25 deg grid cell (a point
        # sensor is simply better information than a nearby grid mean one hour
        # ahead). The fair test is whether calibrated NWP overtakes persistence
        # as the horizon grows. Corrections are affine and fitted on TRAIN only.
        trh = build_forecast_windows(pipeline=pipe, split="train", horizon=h,
                                     seq_len=DEFAULT_SEQ_LEN, return_metadata=True)
        trm_h = trh[6]
        corr = {}
        for mdl in models:
            for v, (field, tkey, _lk) in VAR_MAP.items():
                pp, pt = [], []
                for m in trm_h:
                    e = nwp.get(f"{m['station_id']}|{mdl}")
                    if not e:
                        continue
                    j = e["idx"].get(m["target_timestamp"][:13] + ":00")
                    if j is None:
                        continue
                    val = e["rec"][field][j]
                    if val is None:
                        continue
                    pp.append(val * NWP_TO_STATION_UNITS.get(v, 1.0))
                    pt.append(m[tkey])
                if len(pp) >= 200:
                    a_, b_ = np.polyfit(np.array(pp), np.array(pt), 1)
                    corr.setdefault(mdl, {})[v] = (float(a_), float(b_))
        corrected = {m: {v: np.full(n, np.nan) for v in VAR_MAP} for m in models}
        for mdl in models:
            if mdl not in corr:
                continue
            for i, m in enumerate(meta):
                e = nwp.get(f"{m['station_id']}|{mdl}")
                if not e:
                    continue
                j = e["idx"].get(m["target_timestamp"][:13] + ":00")
                if j is None:
                    continue
                for v, (field, _tk, _lk) in VAR_MAP.items():
                    if v not in corr[mdl]:
                        continue
                    val = e["rec"][field][j]
                    if val is not None:
                        a_, b_ = corr[mdl][v]
                        corrected[mdl][v][i] = a_ * (val * NWP_TO_STATION_UNITS.get(v, 1.0)) + b_

        for v in VAR_MAP:
            t = truth[v]
            base = metrics(persist[v], t)
            cands = [(args.label, metrics(lnn_pred[v], t)),
                     ("persistence", base)]
            for mdl in models:
                if mdl in corr and v in corr[mdl]:
                    cands.append((NWP_DISPLAY[mdl] + " (corr.)", metrics(corrected[mdl][v], t)))
            if any(cm[1] for cm in cands if "corr" in cm[0]):
                print(f"  {v} @ +{h}h  —  bias-corrected NWP (MAE {UNITS[v]})")
                valid = [c for c in cands if c[1]]
                order2 = sorted(valid, key=lambda c: c[1]["mae"])
                for k, (name, m) in enumerate(order2):
                    sp = (base["mae"] - m["mae"]) / base["mae"] * 100
                    star = " <== ours" if name.startswith("LNN") else ""
                    print(f"    {k+1:>2}. {name:<28}{fmt(m['mae']):>9}{sp:>+10.1f}% vs persist{star}")
                results[f"h{h}"][f"{v}_corrected"] = {
                    name: (m if m else None) for name, m in cands}

        print(f"\n  (raw and corrected tables above; rain Brier below)\n")

    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"results": results}, f, indent=2, default=float)
    print(f"written: {out_path}")


if __name__ == "__main__":
    main()
