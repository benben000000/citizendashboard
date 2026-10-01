"""
Refit the operational inference policy from measured skill.

WHY THIS EXISTS
---------------
The shipped policy routes every continuous variable to `persistence_fallback`
at +1h, +3h and +24h. Measurement (diagnose_headroom.py) shows the learned
heads are already BETTER than persistence at almost every horizon — the policy
is discarding skill the network already has. This script rebuilds the policy
from evidence.

PROCEDURE (no test data is touched at any point)
------------------------------------------------
1. TRAIN split  -> fit the affine calibration (a, b) per variable per horizon.
2. VALIDATION   -> choose, per variable per horizon, between
                     persistence | raw head | calibrated head
                   and choose the shrinkage factor lambda that trades the
                   calibration toward identity.
3. TEST split   -> never read. Used only afterwards, by benchmark_vs_nwp.py,
                   to report the result.

SHRINKAGE
---------
An unshrunk affine fit is unstable at long horizons. One pressure fit produced
a=0.835, b=+165.6 hPa, which was WORSE than no correction. The applied
correction is therefore a_shrunk = 1 + lambda*(a-1), b_shrunk = lambda*b, with
lambda chosen on validation from a grid. lambda=0 recovers the raw head.

SWITCHING MARGIN
----------------
A source is only switched away from persistence when it beats persistence on
validation by more than MIN_MARGIN, so a lucky validation fluctuation cannot
flip the production routing.
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

from dataset import TelemetryDataPipeline, build_forecast_windows, DEFAULT_SEQ_LEN  # noqa: E402
from inference import LNNServerlessPredictor  # noqa: E402

DATA_DIR = os.path.abspath(os.path.join(HERE, "..", "data"))
POLICY_PATH = os.path.join(DATA_DIR, "inference_policy.json")
OUT_PATH = os.path.join(DATA_DIR, "inference_policy_refit.json")
HORIZONS = [1, 3, 6, 12, 24]

VARS = ["temperature", "humidity", "pressure", "wind_speed"]
HEAD_KEY = {
    "temperature": "temperature_c",
    "humidity": "relative_humidity_pct",
    "pressure": "pressure_hpa",
    "wind_speed": "wind_speed_kmh",
}
TARGET_KEY = {v: f"target_{v}" for v in VARS}
ORIGIN_KEY = {v: f"origin_{v}" for v in VARS}

LAMBDA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
MIN_MARGIN = 0.01          # require a >1% validation improvement to switch
MIN_CALIBRATION_SAMPLES = 400


def mae(p, t):
    p, t = np.asarray(p, float), np.asarray(t, float)
    m = np.isfinite(p) & np.isfinite(t)
    if m.sum() == 0:
        return float("inf")
    return float(np.abs(p[m] - t[m]).mean())


def collect(pipe, predictor, split, h, cap=None):
    res = build_forecast_windows(pipeline=pipe, split=split, horizon=h,
                                 seq_len=DEFAULT_SEQ_LEN, return_metadata=True)
    X = res[0].detach().cpu().numpy().astype(np.float64)
    dt = res[1].detach().cpu().numpy().astype(np.float64)
    meta = res[6]
    n = len(X) if cap is None else min(cap, len(X))
    heads = {v: np.empty(n) for v in VARS}
    persist = {v: np.empty(n) for v in VARS}
    truth = {v: np.empty(n) for v in VARS}
    for i in range(n):
        m = meta[i]
        raw = np.clip(X[i] * pipe.norm_stds + pipe.norm_means,
                      [10, 10, 10, 900, 0, -1, -1, 0], [50, 70, 100, 1050, 180, 1, 1, 150])
        o = predictor.predict_from_observed_sequence(
            telemetry_sequence=raw, dt_sequence=dt[i], horizon_hours=h)
        for v in VARS:
            heads[v][i] = o["diagnostics"]["raw_learned_predictions"][HEAD_KEY[v]]
            persist[v][i] = m[ORIGIN_KEY[v]]
            truth[v][i] = m[TARGET_KEY[v]]
    return heads, persist, truth


def clamp_var(v, x):
    if v == "temperature":
        return min(60.0, max(-20.0, x))
    if v == "humidity":
        return min(100.0, max(0.0, x))
    if v == "pressure":
        return min(1100.0, max(850.0, x))
    if v == "wind_speed":
        return max(0.0, x)
    return x


POLICY_VARS = ["temperature", "humidity", "pressure", "wind_speed", "wind_direction"]


def _write_neutral_measurement_policy() -> str:
    """
    Write a policy that routes nothing, for measurement only.

    Every cell is persistence and the rain blend is fully model, so the served
    routing decisions cannot leak into a measurement whose whole purpose is to
    decide routing. `fitted_model_commit` is left "unknown" so the serving
    provenance guard skips itself: this policy is explicitly not fitted for any
    weights and makes no claim to be, which is exactly why it is safe to use
    against a candidate.

    Written to a fixed filename rather than a temp path so a failed refit leaves
    an inspectable artefact instead of nothing.
    """
    neutral = {
        "policy_version": "refit-measurement-neutral",
        "generated_at": "1970-01-01T00:00:00+00:00",
        "policy_code_commit": "unknown",
        "fitted_model_commit": "unknown",
        "regenerated_by": "refit_policy.py (_write_neutral_measurement_policy)",
        "regeneration_note": (
            "Measurement scaffold only. Not a deployable policy: it routes every "
            "cell to persistence and asserts no fitted weights."
        ),
        "horizons": {},
    }
    for h in HORIZONS:
        neutral["horizons"][str(h)] = {
            "selected_sources": {v: "persistence_fallback" for v in POLICY_VARS},
            "rain_model_weight": 1.0,
            "rain_persistence_weight": 0.0,
            "operational_rain_threshold": 0.25,
            "calibration_code_commit": "unknown",
        }
    path = os.path.join(DATA_DIR, "refit_neutral_policy.json")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(neutral, handle, indent=2)
        handle.write("\n")
    return path


def main():
    import argparse

    ap = argparse.ArgumentParser(
        description="Refit the inference policy from measured validation skill.")
    ap.add_argument("--weather-csv", default=None,
                    help="Telemetry corpus to refit against. Defaults to "
                         "data/weather_telemetry.csv. This MUST match the corpus "
                         "the model under test was trained on: the refit chooses "
                         "between persistence and the model, so evaluating a "
                         "candidate against a different corpus silently selects "
                         "on out-of-distribution data.")
    ap.add_argument("--bundle-dir", default=None,
                    help="Bundle directory to evaluate (h1/ h3/ ...). Defaults to "
                         "the served bundles. Point this at a candidate's bundles "
                         "to refit FOR that candidate.")
    ap.add_argument("--out", default=None,
                    help="Where to write the refit policy. Defaults to "
                         "data/inference_policy_refit.json. The live policy is "
                         "never overwritten; it is copied over deliberately.")
    ap.add_argument("--measure-neutral", action="store_true", default=True,
                    help="Measure with a routing-neutral policy instead of the "
                         "served one. Required whenever --bundle-dir points at "
                         "weights the served policy was not fitted for, which is "
                         "every real refit FOR a candidate. Measurement is "
                         "unaffected: collect() reads the head output from "
                         "diagnostics['raw_learned_predictions'], not the "
                         "policy-served value.")
    ap.add_argument("--measure-served", dest="measure_neutral", action="store_false",
                    help="Opt out and measure through the served policy. Fails "
                         "closed against a candidate the served policy was not "
                         "fitted for; retained for reproducing historical refits.")
    args = ap.parse_args()
    measure_neutral = args.measure_neutral

    weather_csv = args.weather_csv or os.path.join(DATA_DIR, "weather_telemetry.csv")
    bundle_root = args.bundle_dir or os.path.join(DATA_DIR, "bundles")
    out_path = args.out or os.path.join(DATA_DIR, "inference_policy_refit.json")

    import hashlib as _hashlib

    def _sha256_file(path):
        h = _hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    corpus_sha256 = _sha256_file(weather_csv)
    _water = os.path.join(DATA_DIR, "water_level_telemetry.csv")
    water_sha256 = _sha256_file(_water) if os.path.exists(_water) else None
    print(f"corpus sha256  : {corpus_sha256[:16]}...")
    print(f"water  sha256  : {str(water_sha256)[:16]}...")

    pipe = TelemetryDataPipeline(weather_csv=weather_csv)
    print(f"corpus        : {os.path.basename(weather_csv)}")
    print(f"bundle root   : {os.path.relpath(bundle_root, DATA_DIR)}")
    print(f"policy written: {os.path.basename(out_path)}")
    with open(POLICY_PATH, "r", encoding="utf-8") as f:
        base_policy = json.load(f)

    # Which policy do the predictors load while MEASURING?
    #
    # The measurement must not run through the served policy. Serving applies
    # fitted_model_commit, which compares the policy's fitted weights against
    # the bundle's weights and refuses to load on a mismatch -- correct for
    # serving, and fatal here: refitting a policy FOR a candidate is impossible
    # if the incumbent policy forbids loading that candidate. The tool could only
    # ever refit for the weights already deployed, which is the one case needing
    # no refit.
    #
    # This was a live bug, not a theoretical one: --bundle-dir already documented
    # "point this at a candidate's bundles to refit FOR that candidate", and that
    # path raised the fail-closed error on the first horizon.
    #
    # A neutral measurement policy is safe because collect() reads the head output
    # from diagnostics["raw_learned_predictions"], not the policy-served value. The
    # head-vs-persistence comparison the refit exists to make is therefore
    # unaffected by what the policy routes, and there is no circularity.
    measure_policy_path = None
    if measure_neutral:
        measure_policy_path = _write_neutral_measurement_policy()
        print(f"measuring with neutral policy (not the served one): "
              f"{os.path.basename(measure_policy_path)}")
    # Which weights is this policy fitted against? Taken from the bundle
    # checkpoint, per horizon, and asserted consistent across horizons.
    fitted_model_commit = None
    for _h in HORIZONS:
        _ck = os.path.join(bundle_root, f"h{_h}", "checkpoint.pt")
        if not os.path.exists(_ck):
            continue
        import torch as _torch
        _c = _torch.load(_ck, map_location="cpu", weights_only=False)
        _cm = ((_c.get("manifest") or {}).get("code_commit")) or "unknown"
        if fitted_model_commit is None:
            fitted_model_commit = _cm
        elif fitted_model_commit != _cm:
            raise SystemExit(
                f"bundles span more than one weights commit "
                f"({fitted_model_commit[:12]} vs {_cm[:12]} at h{_h}); a single "
                f"policy cannot be fitted against both")
    print(f"fitted against  : {str(fitted_model_commit)[:16]}...")

    horizons_out = {}
    report = {}

    for h in HORIZONS:
        predictor = LNNServerlessPredictor(
            bundle_dir=os.path.join(bundle_root, f"h{h}"),
            policy_path=measure_policy_path)
        print(f"\n+{h}h  collecting TRAIN ...", flush=True)
        tr_head, tr_pers, tr_truth = collect(pipe, predictor, "train", h)
        print(f"     collecting VAL   ...", flush=True)
        va_head, va_pers, va_truth = collect(pipe, predictor, "val", h)

        base_h = base_policy["horizons"].get(str(h), {})
        selected = dict(base_h.get("selected_sources", {}))
        calib = {}

        print(f"     {'variable':<12}{'val persist':>12}{'val head':>10}{'val cal':>10}"
              f"{'chosen':>26}{'a':>8}{'b':>10}")
        print("     " + "-" * 78)
        for v in VARS:
            tr_ok = int(np.isfinite(tr_head[v]).sum())
            if tr_ok < MIN_CALIBRATION_SAMPLES:
                selected[v] = "persistence_fallback"
                print(f"     {v:<12}{'insufficient calibration data':>44}")
                continue

            a_raw, b_raw = np.polyfit(tr_head[v], tr_truth[v], 1)

            best = None
            for lam in LAMBDA_GRID:
                a = 1.0 + lam * (a_raw - 1.0)
                b = lam * b_raw
                cand = np.array([clamp_var(v, a * x + b) for x in va_head[v]])
                e = mae(cand, va_truth[v])
                if best is None or e < best[0]:
                    best = (e, lam, a, b)

            e_pers = mae(va_pers[v], va_truth[v])
            e_head = mae(va_head[v], va_truth[v])
            e_cal, lam, a, b = best

            improved = (e_pers - e_cal) / e_pers > MIN_MARGIN
            if improved and e_cal <= e_head:
                source, a_out, b_out = "learned_model", a, b
            elif e_pers - e_head > e_pers * MIN_MARGIN:
                source, a_out, b_out = "learned_model", 1.0, 0.0
            else:
                source, a_out, b_out = "persistence_fallback", None, None

            if source == "learned_model" and a_out is not None:
                calib[v] = {"a": round(float(a_out), 6), "b": round(float(b_out), 6),
                            "lambda": float(lam), "fitted_on": "train",
                            "selected_on": "validation"}
            selected[v] = source
            print(f"     {v:<12}{e_pers:>12.3f}{e_head:>10.3f}{e_cal:>10.3f}"
                  f"{source:>26}{(a_out if a_out is not None else 1.0):>8.3f}"
                  f"{(b_out if b_out is not None else 0.0):>+10.3f}")

        # Wind direction and heat index are not scored by this refit; keep the
        # existing conservative routing rather than asserting skill we did not
        # measure.
        selected.setdefault("wind_direction", "persistence_fallback")
        selected["heat_index"] = "derived_from_selected_temp_and_humidity"

        entry = dict(base_h)
        entry["selected_sources"] = selected
        if calib:
            entry["calibration"] = calib
        else:
            entry.pop("calibration", None)
        horizons_out[str(h)] = entry
        report[f"h{h}"] = {"selected_sources": selected, "calibration": calib}

    new_policy = {
        "policy_version": "2.0.0",
        "policy_code_commit": base_policy.get("policy_code_commit"),
        # A shipped policy with no generation timestamp is undatable and
        # unauditable: you cannot tell when the coefficients were fitted, or
        # whether a bundle predates the policy it claims to carry. This was
        # hard-coded empty, which propagated into all five bundle policies.
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "regenerated_by": "prediction-model/src/refit_policy.py",
        "regeneration_note": (
            "Sources and calibration coefficients refitted from measured skill. "
            "Calibration fitted on TRAIN, source selection and shrinkage chosen on "
            "VALIDATION. The TEST split was not read during refitting."
        ),
        # The corpus this policy was ACTUALLY refitted against, not the one the
        # previous policy was fitted on. Carrying the old hashes forward made a
        # refit-on-new-data policy claim to describe old data, which the release
        # gate correctly refuses with POLICY_CORPUS_DIVERGED. A policy that
        # misstates its own training data is worse than no policy: it makes the
        # provenance record a lie while appearing to satisfy it.
        # The weights this policy was fitted AGAINST. inference.py compares this
        # to the bundle's checkpoint commit and fails closed on a mismatch, which is
        # the failure worth catching: a policy fitted for model A must not be
        # served with model B. It is deliberately NOT policy_code_commit, which
        # records when the policy was fitted and legitimately differs.
        "fitted_model_commit": fitted_model_commit,
        "dataset_hashes": {
            "weather_telemetry_sha256": corpus_sha256,
            # Computed from the file, not inherited. Inheriting it let a `null`
            # propagate: a policy with water_telemetry_sha256 = null fails the
            # provenance gate on a water hash it could never have checked.
            # Key name matters: verify_provenance.py reads
            # `water_level_telemetry_sha256`. Writing `water_telemetry_sha256`
            # instead put a correct hash under a key the gate never looks at, so
            # it compared None against the real hash and reported a mismatch on a
            # value that was right.
            "water_level_telemetry_sha256": water_sha256,
            "weather_csv_path": os.path.basename(weather_csv),
        },
        "horizons": horizons_out,
    }

    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(new_policy, f, indent=2)
    with open(os.path.join(DATA_DIR, "policy_refit_report.json"), "w",
              encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=2)

    print(f"\nwritten: {out_path}")
    print(f"written: {os.path.join(DATA_DIR, 'policy_refit_report.json')}")
    print("\nReview the table above, then copy the refit policy over the live one:")
    print(f"  copy inference_policy_refit.json -> inference_policy.json")
    print(f"  then re-run generate_bundles.py so bundle policy hashes match.")


if __name__ == "__main__":
    main()
