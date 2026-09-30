# Operations Runbook

**Component:** KloudTrack prediction engine — `prediction-model/`
**Scope:** retrain, re-benchmark, regenerate bundles, roll back, and read the audit trails.
**Companion documents:** [model-architecture.md](model-architecture.md) ·
[known-limitations.md](known-limitations.md)

All commands run from the repository root on Windows PowerShell.

---

## 0. Before you touch anything

Three rules. Each of them corresponds to a defect that produced a wrong number which was
believed (see [known-limitations.md §6](known-limitations.md#6-bug-history--seven-defects-each-of-which-produced-a-wrong-number-that-was-believed)).

1. **`rc != 0` is never ignorable.** A training run can leave complete, plausible artifacts
   on disk and still exit non-zero because a late stage failed. Read the exit code before
   reading the artifacts.
2. **Check for `n/a` and `null` in every benchmark table.** A benchmark that prints a rank
   next to missing scores has silently failed. This exact shape shipped once.
3. **Verify independently-seeded runs produce different standard deviations.** If two
   independently-trained arms report identical variance to six decimals, they are the same
   model and the comparison measured nothing.

Working interpreter:

```powershell
$PY = ".\.venv\Scripts\python.exe"
```

---

## 1. Prerequisites

```powershell
& $PY -m pip install --upgrade pip
& $PY -m pip install -r prediction-model\requirements.txt
```

CI targets Python 3.11 on `ubuntu-latest`. Local runs on other versions can produce
different float behaviour; if a number shifts unexpectedly, check the interpreter version
before investigating the model.

Environment:

| Variable | Purpose |
|---|---|
| `KLOUDTRACK_API_KEY` | station history API. Never written to source, logs, or committed artifacts. Falls back to `.env.local`. |
| `MQTT_TOPIC` | live ingest topic, default `kloudtrack/+/data` |
| `MQTT_BROKER` / credentials | live ingest connection |

---

## 2. Pre-flight: data freshness

**Run this before any refetch.** A rostered station that answers with nothing is silent
data loss, and one such station (`wkAWlzlm`) is currently on the roster.

```powershell
& $PY prediction-model\src\check_data_freshness.py `
    --source prediction-model\data\weather_telemetry.csv
```

It emits a verdict of `PASS` / `WARN` / `FAIL` plus the exact threshold configuration that
produced it, so you can tell a tightened threshold from worse data.

**Refetching station history** (16 cached requests at 3.5 s pacing, ~1 minute):

```powershell
& $PY prediction-model\src\fetch_current_telemetry.py
```

> If a station is missing after the refetch, check `data\kloudtrack_history_cache\` for a
> file named `<station>__<start>__<end>__i60__f0.json`. A missing file means the endpoint
> returned zero rows. Do not treat HTTP 200 as data.

---

## 3. Retrain

### 3.1 Full five-horizon run

```powershell
& $PY prediction-model\src\train_predictive_quality.py `
    --horizons 1,3,6,12,24 `
    --epochs 60 `
    --patience 12 `
    --lr 1e-3 `
    --output-dir prediction-model\data\candidate_artifacts
```

| Flag | Default | Notes |
|---|---|---|
| `--horizons` | all five | comma-separated. **A run without `1` skips the promotion audit** and records `promotion_audit: None` with a note. That is expected, not a failure. |
| `--epochs` | 60 | |
| `--patience` | 12 | early stopping |
| `--lr` | 1e-3 | |
| `--seed` | 42 | change it when measuring noise |
| `--output-dir` | `data/candidate_artifacts` | one directory per run; never overwrite a prior run |
| `--weather-csv` | committed CSV | use `weather_telemetry_current.csv` for a refetched corpus |
| `--commit` | auto | explicit commit hash for candidate provenance |

**Never write into a directory that already holds a run you care about.** Two runs
silently overwriting each other is the same class of failure as two arms training the same
model: you get one result where you believe you have two.

### 3.2 Single horizon (fast iteration)

```powershell
& $PY prediction-model\src\train_predictive_quality.py --horizons 6 --epochs 20
```

### 3.3 Post-run checklist

```
[ ] exit code == 0
[ ] promotion_audit present in the run output (or explicitly None with a note)
[ ] all five horizons present in horizon_evaluations
[ ] output-dir contains candidate_h1h.pt … candidate_h24h.pt
[ ] candidate manifest records the corpus SHA-256
[ ] per-seed std devs differ from any prior run at the same seed
```

### 3.4 Seed sweep (only when sizing a noise band)

```powershell
& $PY prediction-model\src\run_seed_sweep.py --seeds 101,202,303 --epochs 60
& $PY prediction-model\src\analyze_seed_sweep.py
```

Cost: roughly an hour per arm per seed. **Verify the arms differ before spending the
compute.** The harness constructs both arms and asserts the presence of one marker and the
absence of the other, and it launches the trainer from the arm directory. If per-seed
standard deviations come back identical, the arms are not independent — stop and fix
before trusting anything in the summary.

---

## 4. Refit the operational policy

Only after retraining, and only after the benchmark below.

```powershell
& $PY prediction-model\src\refit_policy.py
```

Regenerates `data/inference_policy.json`:

- Calibration coefficients fitted on **TRAIN**.
- Source selection and shrinkage strength chosen on **VALIDATION**.
- **TEST is not read during refit.**

The file's `regeneration_note` records this. If a future change makes that sentence false,
the policy is no longer auditable and the change must not ship.

**Inspect the diff before accepting it.** A refit that flips many cells at once is a signal,
not a routine update:

```powershell
git diff -- prediction-model/data/inference_policy.json
```

Cells that move from `persistence_fallback` to `learned_model` are the interesting ones and
must be justified by the validation split in the same commit.

---

## 5. Re-benchmark

### 5.1 Fetch NWP ground truth (first run only, then cached)

```powershell
& $PY prediction-model\src\fetch_nwp_benchmark.py
```

Seven models: ECMWF IFS, NOAA GFS, DWD ICON, Meteo-France, JMA, UK Met Office, ECCC GEM.

### 5.2 Run the benchmark

Production bundles (the shipped model — this is the default and usually what you want):

```powershell
& $PY prediction-model\src\benchmark_vs_nwp.py `
    --out prediction-model\data\nwp_benchmark_production.json
```

A retrained candidate, scored on identical windows and identical ground truth:

```powershell
& $PY prediction-model\src\benchmark_vs_nwp.py `
    --candidate-dir prediction-model\data\candidate_artifacts `
    --out prediction-model\data\nwp_benchmark_candidate.json
```

Narrow to the horizons you changed:

```powershell
& $PY prediction-model\src\benchmark_vs_nwp.py --horizons 6,12
```

| Flag | Meaning |
|---|---|
| `--weather-csv` | corpus; defaults to the committed `weather_telemetry.csv` |
| `--candidate-dir` | score a candidate instead of the production bundles |
| `--horizons` | comma-separated, default all five |
| `--out` | results JSON path |
| `--label` | display name for the model under test |

### 5.3 Sanity checks on the output — do not skip

```
[ ] exit code == 0
[ ] no method reads n/a or null in any horizon
[ ] every method has the same n within a horizon
[ ] wind is in m/s and NWP wind is between 1.9 and 2.5 at +6h
      -> a value near 8-10 means the km/h conversion is missing again
[ ] LNN and persistence differ where the policy selects learned_model,
    and are identical where it selects persistence_fallback
```

The last check is the most informative. **If `LNN production` and `persistence` differ in a
cell the policy routes to `persistence_fallback`, something is bypassing the policy gate**
and every downstream number is suspect.

### 5.4 Independent baseline comparison

```powershell
& $PY prediction-model\src\benchmark_independent.py
& $PY prediction-model\src\evaluate_downscale_test.py
```

> `benchmark_independent.score` returns bias as `mean(truth − pred)`, the **opposite sign**
> to the canonical scorer. Not a reportable bias figure.

### 5.5 Headroom diagnostic (before opening any gate)

```powershell
& $PY prediction-model\src\diagnose_headroom.py
```

Writes `data/headroom_diagnostic.json`. Answers one question: is the cell calibration-limited
or training-limited? If `head_skill_pct` is at or below zero, no amount of calibration will
help and the gate should stay shut.

### 5.6 Independent benchmark targets (the numbers of record)

From `data/nwp_benchmark_production.json`, test split, committed corpus, production
bundles. Wind in m/s.

| Channel | +6h | +12h |
|---|---|---|
| Temperature — LNN vs persistence | 1.4772 vs 1.6987 (**+13.0%**) | 1.6474 vs 2.1619 (**+23.8%**) |
| Temperature — ECMWF | **1.2740** (ECMWF wins) | **1.2880** (ECMWF wins) |
| Wind speed — LNN vs persistence | 1.2168 vs 1.3192 (**+7.8%**) | 1.2428 vs 1.5114 (**+17.8%**) |
| Wind speed — ECMWF / GFS | 1.9340 / 2.3933 | 1.9186 / 2.4092 |
| Pressure @ +1h — LNN vs ECMWF | 0.3942 vs 2.2194 | — |
| Rain occurrence Brier | 0.2209 (persistence 0.2456) | 0.2364 (persistence 0.2646) |

Rain ranks and margins, because the Brier absolute value alone hides both the decay and the
+24h failure:

| Horizon | Rank | vs best NWP (DWD ICON) | vs persistence | vs climatology |
|---|---|---|---|---|
| +1h | 1 / 10 | +43.2% | +15.9% | +44.0% |
| +3h | 1 / 10 | +23.0% | +12.8% | +24.1% |
| +6h | 1 / 10 | +9.0% | +10.1% | +9.3% |
| +12h | 1 / 10 | +2.1% | +10.7% | +2.5% |
| +24h | **3 / 10** | **−2.7%** | +13.5% | **−4.6%** |

Reference: +24h temperature 1.4059; +24h rain Brier 0.2506, which is **worse** than
climatology (0.2396) and DWD ICON (0.2440). Do not report +24h rain as a win.

**A regression is any of:** wind MAE rising above persistence at a cell the policy routes to
`learned_model`; rain Brier rising above persistence; NWP wind returning to km/h magnitudes;
any method printing `n/a`.

---

## 6. Regenerate artifacts and bundles

### 6.1 Manifests

```powershell
& $PY prediction-model\src\generate_manifests.py --output-dir prediction-model\data
& $PY prediction-model\src\generate_manifests.py --check-only     # verify without writing
```

### 6.2 Bundles

```powershell
& $PY prediction-model\src\generate_bundles.py
& $PY prediction-model\src\generate_bundles.py --check-only        # verify without writing
```

Produces one hash-verified bundle per horizon under `data/bundles/h{1,3,6,12,24}/`:

```
checkpoint.pt            from data/lnn_weather_water_h<h>.pt
inference_policy.json    horizon-scoped active policy with full provenance
bundle_manifest.json     hashes, commit triple, feature schema
README.md
```

**Order matters.** `generate_bundles.py` reads `data/inference_policy.json`. Refit the policy
(§4) *before* bundling, or the bundles will carry a stale policy with a fresh timestamp —
which is exactly the silent-inconsistency failure mode the manifest exists to prevent.

`--check-only` verifies without writing and is the right command for a read-only validation
environment.

### 6.3 Verify provenance

```powershell
& $PY prediction-model\src\verify_provenance.py
& $PY prediction-model\src\verify_provenance.py --allow-commit <sha>   # historical acknowledgement
```

Checks the commit triple and dataset hashes. **Prefer `--allow-commit` over editing an
artifact.** Editing an artifact to make provenance pass destroys the only evidence that it
needed to be overridden.

`--allow-commit` is an acknowledgement that an artifact legitimately predates the current
commit. It is not a way to make a mismatched bundle load.

---

## 7. Rollback

Bundles and checkpoints are content-addressed and the previous set is intact on disk. Roll
back is a file operation, not a rebuild.

### 7.1 What to restore, in order

| Step | Path | Notes |
|---|---|---|
| 1 | `data/inference_policy.json` | **First.** This alone reverts source selection, calibration and rain blend. |
| 2 | `data/bundles/h{1,3,6,12,24}/` | whole directories, including `bundle_manifest.json` |
| 3 | `data/lnn_weather_water_h*.pt` | only if the checkpoints themselves are the problem |
| 4 | `data/lnn_trained_weights_h*.json` | standalone (MF-2) weights, if used |

Policy first, because it is the smallest, most auditable and most reversible change. **Reverting
only the bundles without the policy leaves the manifest hashes referring to a policy that is
no longer on disk**, and `inference.py` will fail closed on the hash check — which is correct
behaviour but wastes an incident's time.

### 7.2 Verify the rollback

```powershell
& $PY prediction-model\src\generate_bundles.py --check-only
& $PY prediction-model\src\verify_provenance.py
& $PY prediction-model\src\smoke_test.py
```

### 7.3 Rollback triggers

Roll back when **any** of these hold:

- The live API raises rather than returning a forecast (fail-closed behaviour firing).
- A served value violates a physical bound.
- A benchmark shows a `learned_model` cell regressing to worse-than-persistence.
- `verify_provenance.py` or the bundle hash check fails on the deployed set.
- A refit changed more cells than the validation evidence supports.

**Do not** roll back to make a benchmark number look better. That is the failure this
subsystem's audit trail exists to prevent.

### 7.4 Note the asymmetry

Rollback does not restore confidence. If a defect reached users, the audit trails
(§9) still contain every forecast issued under it. Rolling back stops further exposure; it
does not un-issue what was already sent. Record the incident window from
`prediction_audit.jsonl` before you start rolling back, because that file is the only
record of who was told what.

---

## 8. Thresholds

### 8.1 Candidate promotion

From `src/train_predictive_quality.py`:

| Constant | Value | Meaning |
|---|---|---|
| `PROMOTION_MIN_RELATIVE_IMPROVEMENT` | `0.02` | candidate must beat persistence by ≥2% relative |
| `MIN_HORIZONS_FOR_PROMOTION` | `3` | and must do so on ≥3 evaluated horizons |

A target is promoted only when **both** conditions hold. Otherwise the baseline is retained.
This is why humidity, which has +19.74% at +12h, is still served by persistence at every
horizon: one strong horizon is not three. **Changing this rule is a policy decision with
evidence requirements, not a configuration tweak.**

### 8.2 Data freshness

From `src/check_data_freshness.py`:

| Knob | WARN | FAIL |
|---|---|---|
| corpus age | 72 h | 168 h |
| per-station silence | 24 h | 72 h |
| station coverage over its own span | 0.90 | 0.60 |
| interior gap | 6 h | 24 h |
| clock skew (future rows) | — | 48 h excluded |
| rostered station with no rows | — | **FAIL** (`fail_on_missing_station: True`) |

Late station start (>24 h) is **reported but deliberately not part of the verdict** — a
station commissioned in August is a fact about the fleet, not a fault.

### 8.3 Monitoring

From `src/monitoring.py`:

| Constant | Value |
|---|---|
| `CANONICAL_HORIZONS` | `[1, 3, 6, 12, 24]` |
| `HEAVY_RAIN_THRESHOLDS` | `[2.5, 5.0, 10.0]` mm/h |
| `MIN_RELIABLE_SAMPLES` | 30 |
| skill alarm | MAE skill `< −0.20` with `n ≥ 30` |
| calibration alarm | rain ECE `> 0.25` with `n ≥ 30` |

Sample counts below `MIN_RELIABLE_SAMPLES` are flagged unreliable and must not trigger an
alarm. A score computed over 12% of rows is a score of whichever rows happened to be present,
and it is indistinguishable from a full-coverage score by the number alone.

### 8.4 Rain operational thresholds

From `inference_policy.json`, per horizon:

| Horizon | model weight | persistence weight | operational threshold |
|---|---|---|---|
| +1h | 0.80 | 0.20 | 0.25 |
| +3h | 0.60 | 0.40 | 0.10 |
| +6h | 0.45 | 0.55 | 0.10 |
| +12h | 0.45 | 0.55 | 0.10 |
| +24h | 0.40 | 0.60 | 0.10 |

### 8.5 CI gates

`prediction-model/ci/prediction-model.yml`, mirrored at
`.github/workflows/prediction-model.yml`. Fifteen gates:

| Gate | Command |
|---|---|
| 1 Clean checkout | — |
| 2 Dependencies | `pip install -r prediction-model/requirements.txt` |
| 3 Static checks | `python -m py_compile prediction-model/src/*.py` |
| 4 Canonical contract | `test_canonical_contract.py` |
| 5 Inference policy integration | `test_inference_contract.py` |
| 6 Provenance | `test_provenance.py` |
| 7 Smoke test | `smoke_test.py` |
| 8 Monitoring & drift | `test_monitoring.py` + `monitoring.py` |
| 9 Predictive quality | `test_predictive_quality.py` + `generate_manifests.py --check-only` |
| 10 Path hygiene | rejects absolute machine paths (`C:\…`) in artifacts |
| 11 Deleted artifact references | — |
| 12 Working-tree cleanliness | read-only test verification |
| 13 Bundle provenance | `verify_provenance.py` |
| 14 Five-horizon artifact presence | — |
| 15 Real-data validation | `validate.py --horizons 1 3 6 12 24` |

**Gate 12 matters more than it looks.** It exists because a validation run that writes
artifacts makes the working tree dirty, which makes the next run's provenance check fail, which
makes the failure look like a provenance problem. `validate.py` is read-only with respect to
committed artifacts; if it is dirtying the tree, something regressed.

**Gate 10 matters for portability.** Artifacts under `prediction-model/data/` must contain
repo-relative paths only. An absolute path means a developer's directory layout has leaked
into a committed artifact.

---

## 9. Reading the audit trails

Two append-only JSONL files. Both are append-only by design: they are evidence, and evidence
that can be rewritten is not evidence.

| File | One record per | Purpose |
|---|---|---|
| `data/prediction_audit.jsonl` | issued forecast | what we told a user |
| `data/observation_audit.jsonl` | received observation | what actually happened |

### 9.1 Prediction record

```json
{
  "schema_version": 1,
  "recorded_at_utc": "2026-09-30T01:53:55.654288Z",
  "origin_timestamp_utc": "2026-09-30T01:53:55.432805Z",
  "station_id": "KT-6CBD47DC5194",
  "horizon_hours": 1.0,
  "provenance": {
    "model":  { "bundle": "h1", "checkpoint_trained_at": null, "seed": null },
    "policy": { "policy_version": "2.0.0",
                "policy_code_commit": "cf0a37e2…",
                "policy_regenerated_by": "prediction-model/src/refit_policy.py" },
    "nwp":    { "applied": false, "reason": "router not wired into the live ingestor yet" }
  },
  "variables": {
    "temperature": { "value": 34.08, "producer": "lln", "nwp_raw": null, "nwp_corrected": null }
  },
  "sensor_health": null,
  "forecast_raw": { "policy_version": "2.0.0", "rain_model_weight": 0.8, "…": "…" }
}
```

Two traps:

1. **`producer` is not `policy_source`.** The `lln` label appears even for variables the
   policy is serving from persistence, because the network ran and its output was gated.
   To know what was served, read `provenance.policy.policy_version` and look that horizon up
   in `inference_policy.json`. **Do not infer the source from `producer`.**
2. **`nwp.applied: false` is the normal live state.** The NWP router is not wired into the
   live ingestor; NWP enters through offline benchmarking and correction only.

### 9.2 Scoring live predictions

```powershell
& $PY prediction-model\src\verify_predictions.py
& $PY prediction-model\src\prediction_audit.py
```

Writes `data/prediction_verification_summary.json`:

```json
{
  "predictions_seen": 81, "observations_seen": 237,
  "scored": 0, "pending": 81, "unmatched": 0, "already_verified": 0,
  "tolerance_minutes": 30.0,
  "mae_by_variable": {}, "mae_by_variable_and_producer": {}
}
```

Reading it:

- **`scored: 0` with `pending: 81`** is normal for a freshly deployed system. A prediction
  cannot be scored until the target hour's observation has arrived.
- **`unmatched > 0`** means observations arrived with no corresponding issued forecast.
  Check the ingest, not the model.
- **`mae_by_variable_and_producer` is the field that answers "was it the model or the
  policy?"** — read it before attributing any live MAE to the network.

Prediction cannot be scored without the observation trail. The verifier is
`tolerance_minutes: 30` matching; widen it only with a reason recorded in the incident.

### 9.3 Incident reconstruction

Given a start and end time:

1. Filter `prediction_audit.jsonl` on `recorded_at_utc` to get every forecast issued in the
   window. Record the count — this is who was affected.
2. Extract the distinct `provenance.policy.policy_version` values. More than one means the
   window spans a policy change and you have two different systems to reason about.
3. Filter `observation_audit.jsonl` over the same window for the ground truth.
4. Run `verify_predictions.py` over the window for the realised error.
5. Confirm the policy version against `data/inference_policy.json` history. **The policy file
   records only the current version**, so step 2 is the authoritative record of what was live.

An audit write failure is logged and swallowed by design — losing an audit line may cost you
an auditable forecast, but it must never cost a live user their forecast. A gap in the trail
means a write failed; check the ingestor logs for the exception before concluding the system
was healthy.

---

## 10. Failure playbooks

### 10.1 Inference raises at startup

Expected fail-closed behaviour. Read the message:

| Message | Cause | Action |
|---|---|---|
| "Operational inference policy not found" | no `inference_policy.json` | run §4, then §6 |
| "Requested horizon N is not supported" | horizon not in the policy table | never substitute another horizon; add it in §4 |
| "policy commit does not match model commit" | policy and bundle from different runs | regenerate bundles (§6.2) or roll back both (§7) |
| "Bundle policy hash mismatch" | tampered or truncated policy | roll back (§7); do not regenerate over the top of a corruption |
| "Malformed operational inference policy" | invalid JSON | roll back (§7) |

**All of these should raise.** If inference returns a value anyway, something has changed the
fail-closed contract and that is a more serious incident than the raise.

### 10.2 Benchmark prints `n/a` for an NWP model

Stop the run. This is the `load_nwp()` window-selection failure mode: the cache merged to a
window set with no overlap with the evaluation period. Refetch (§5.1) and re-run. **Never
report a ranking produced alongside missing scores.**

### 10.3 NWP wind numbers jump 3.6×

Missing km/h → m/s conversion. `NWP_TO_STATION_UNITS` applies to Open-Meteo fields only.
The candidate model already predicts m/s — applying the factor to it rescales it by 3.6×.

### 10.4 Wind forecasts read 0.0 for most windows

Check `_nonnegative` is in use and that no outer `clamp(min=0)` wraps it. The floor must be
inside the residual addition.

### 10.5 A candidate scores far worse than a previous run

Check, in order: corpus SHA-256, whether the run was fed normalised input
([known-limitations §6.3](known-limitations.md#63-candidate-feature-space-scored-on-the-wrong-input-space)),
whether `--seed` changed, and whether both runs really used the same windows. A 30%+ swing
is far more often a harness difference than a model finding.

### 10.6 `promotion_audit` missing from a run output

The run did not evaluate promotion. Either a partial-horizon run (expected, and now recorded
explicitly as `None` with a note) or the arity defect has returned. If `rc == 0` and the key
is simply absent, treat it as the defect.

### 10.7 Two sweep arms score identically

The arms trained the same model. Verify the trainer is launched from the arm directory. A
correct run produces differing per-seed standard deviations; identical variance across
independently-seeded arms is impossible and is the tell.

---

## 11. Full release sequence

```powershell
$PY = ".\.venv\Scripts\python.exe"

# 0. Data
& $PY prediction-model\src\check_data_freshness.py --source prediction-model\data\weather_telemetry.csv
& $PY prediction-model\src\fetch_current_telemetry.py

# 1. Train (rc MUST be 0; promotion_audit MUST be present)
& $PY prediction-model\src\train_predictive_quality.py --horizons 1,3,6,12,24 --epochs 60

# 2. Re-benchmark production bundles BEFORE touching the policy
& $PY prediction-model\src\fetch_nwp_benchmark.py
& $PY prediction-model\src\benchmark_vs_nwp.py --out prediction-model\data\nwp_benchmark_production.json

# 3. Headroom diagnostic — evidence for or against opening any gate
& $PY prediction-model\src\diagnose_headroom.py

# 4. Refit the policy (train-fit, validation-selects, test untouched)
& $PY prediction-model\src\refit_policy.py
git diff -- prediction-model/data/inference_policy.json

# 5. Artifacts and bundles
& $PY prediction-model\src\generate_manifests.py --output-dir prediction-model\data
& $PY prediction-model\src\generate_bundles.py

# 6. Verify (read-only)
& $PY prediction-model\src\verify_provenance.py
& $PY prediction-model\src\generate_bundles.py --check-only
& $PY prediction-model\src\smoke_test.py
& $PY prediction-model\src\test_canonical_contract.py
& $PY prediction-model\src\test_inference_contract.py
& $PY prediction-model\src\test_provenance.py
& $PY prediction-model\src\test_predictive_quality.py
& $PY prediction-model\src\test_wind_head_gradient.py
& $PY prediction-model\src\test_wind_sensor_gate.py
& $PY prediction-model\src\test_scoring_golden_fixtures.py
& $PY prediction-model\src\validate.py --horizons 1 3 6 12 24

# 7. Confirm the working tree is clean after read-only validation (CI gate 12)
git status --porcelain
```

**If step 7 shows modifications to committed artifacts, stop.** `validate.py` is read-only
with respect to those artifacts. A dirty tree means something wrote where it should not have,
and shipping from that state means shipping artifacts whose provenance no longer matches
their contents.

---

## 12. Command index

| Task | Command |
|---|---|
| Data freshness verdict | `check_data_freshness.py --source <csv>` |
| Refetch station history | `fetch_current_telemetry.py` |
| Field / bounds / availability audit | `audit_data_availability.py` |
| Train candidate | `train_predictive_quality.py --horizons 1,3,6,12,24` |
| Train single horizon | `train_predictive_quality.py --horizons 6 --epochs 20` |
| Seed sweep | `run_seed_sweep.py --seeds 101,202,303` |
| Analyse sweep | `analyze_seed_sweep.py` |
| Refit policy | `refit_policy.py` |
| Fetch NWP | `fetch_nwp_benchmark.py` |
| Benchmark production | `benchmark_vs_nwp.py --out prediction-model/data/nwp_benchmark_production.json` |
| Benchmark candidate | `benchmark_vs_nwp.py --candidate-dir <dir> --out <path>` |
| Independent baselines | `benchmark_independent.py` |
| Downscale evaluation | `evaluate_downscale_test.py` |
| Headroom diagnostic | `diagnose_headroom.py` |
| Anemometer diagnostic | `diagnose_anemometer.py` |
| Sensor health | `sensor_health.py` |
| Monitoring / drift | `monitoring.py --output prediction-model/data/monitoring_report.json` |
| Manifests | `generate_manifests.py [--check-only]` |
| Bundles | `generate_bundles.py [--check-only]` |
| Provenance | `verify_provenance.py [--allow-commit <sha>]` |
| Audit trails | `prediction_audit.py`, `verify_predictions.py` |
| Canonical validation | `validate.py --horizons 1 3 6 12 24` |
| Smoke test | `smoke_test.py` |
| NWP correction fit / verify | `fit_nwp_correction.py`, `verify_nwp_correction.py` |
| NWP calibration selection | `select_nwp_calibration.py` |