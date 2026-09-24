# Autonomous Fix Prompt: Repair Prediction-Model Monitoring and Five-Horizon Release Gates

You are an autonomous senior ML/software engineer working on:

`https://github.com/benben000000/citizendashboard`

Work from a fresh clone or clean worktree of the latest `origin/main`. The current audited head is approximately:

```text
e698d4a51728172811c60804b541dcc4f5fb5084
```

Do not stop at analysis or a plan. Implement the fix, add regression tests, run all verification gates, review the diff, commit, and push only if all gates pass.

## Objective

Fix the prediction-model monitoring engine so it correctly reads the committed prediction log, evaluates all five forecast horizons, reports valid metrics, rejects incompatible schemas, and cannot silently produce misleading zero/default metrics.

The supported horizons are:

```text
1h, 3h, 6h, 12h, 24h
```

## Confirmed defect

The current prediction log uses these fields:

```text
station_id
origin_timestamp
target_timestamp
horizon_hours
actual_lead_hours
actual_rain
actual_precip_mm
actual_temp
actual_humidity
actual_pressure
actual_wind_speed
actual_wind_dir_deg
actual_heat_index
pred_temp
pred_humidity
pred_pressure
pred_wind_speed
pred_wind_dir_deg
derived_heat_index
mf1_rain_prob
hybrid_rain_prob
mf1_precip_mm
persist_temp
persist_humidity
persist_pressure
persist_wind_speed
persist_rain
actual_water_level
mf1_water_level
persist_water
```

However, `prediction-model/src/monitoring.py` currently expects obsolete names such as:

```text
horizon
true_temp
true_rh
true_p
true_ws
true_precip
pred_rain_prob
orig_temp
orig_rh
orig_p
orig_ws
orig_precip
```

The current implementation also silently defaults missing columns to zero. This causes all rows to be assigned to +1h and produces misleading metrics.

## Scope restrictions

1. Modify only prediction-model code, tests, documentation, and directly related CI configuration.
2. Do not modify dashboard UI or unrelated application code.
3. Do not delete raw telemetry:
   - `prediction-model/data/weather_telemetry.csv`
   - `prediction-model/data/water_level_telemetry.csv`
4. Preserve model bundles, canonical checkpoints, manifests, scorecards, and inference policy unless regeneration is required.
5. Do not commit temporary scripts, local environments, secrets, credentials, or machine-specific paths.
6. Do not weaken tests or make monitoring pass through default values.

## Phase 1: Baseline and inventory

1. Fetch `origin/main`.
2. Record the starting SHA and worktree status.
3. Confirm the archive branch remains available:

   ```text
   cleanup/archive-before-remediation-20260922
   ```

4. Inspect:
   - `prediction-model/src/monitoring.py`;
   - `prediction-model/data/test_predictions_log.csv`;
   - existing monitoring/provenance/inference tests;
   - `.github/workflows/prediction-model.yml`.
5. Run the current monitoring command and record the defective output for comparison.

## Phase 2: Add an explicit prediction-log schema contract

Implement a single authoritative schema definition in `monitoring.py` or a small canonical helper module.

Define required fields and their types, for example:

```python
PREDICTION_LOG_REQUIRED_COLUMNS = {
    "station_id": "string",
    "origin_timestamp": "timestamp",
    "target_timestamp": "timestamp",
    "horizon_hours": "integer",
    "actual_lead_hours": "float",
    "actual_rain": "float_or_binary",
    "actual_precip_mm": "float",
    "actual_temp": "float",
    "actual_humidity": "float",
    "actual_pressure": "float",
    "actual_wind_speed": "float",
    "pred_temp": "float",
    "pred_humidity": "float",
    "pred_pressure": "float",
    "pred_wind_speed": "float",
    "mf1_rain_prob": "float",
    "hybrid_rain_prob": "float",
    "mf1_precip_mm": "float",
    "persist_temp": "float",
    "persist_humidity": "float",
    "persist_pressure": "float",
    "persist_wind_speed": "float",
    "persist_rain": "numeric_with_explicit_unit",
}
```

Required behavior:

1. Validate that every required column exists before reading rows.
2. Fail with a clear error listing missing columns.
3. Do not silently substitute zero for a missing required field.
4. Reject unsupported or ambiguous schema versions.
5. Add a schema version to monitoring output, such as:

   ```json
   "prediction_log_schema_version": "1.0"
   ```

6. Validate numeric fields for parseability and finite values.
7. Validate `horizon_hours` against `[1, 3, 6, 12, 24]`.
8. Validate `actual_lead_hours` against the accepted horizon tolerance.
9. Report malformed-row counts explicitly and fail if malformed rows exceed the documented threshold.

## Phase 3: Correct the mapping

Replace all obsolete field lookups with the current CSV schema.

The canonical mapping must be equivalent to:

```python
record = {
    "station_id": row["station_id"],

    "horizon_hours": int(float(row["horizon_hours"])),

    "true_temperature": float(row["actual_temp"]),
    "pred_temperature": float(row["pred_temp"]),
    "persist_temperature": float(row["persist_temp"]),

    "true_humidity": float(row["actual_humidity"]),
    "pred_humidity": float(row["pred_humidity"]),
    "persist_humidity": float(row["persist_humidity"]),

    "true_pressure": float(row["actual_pressure"]),
    "pred_pressure": float(row["pred_pressure"]),
    "persist_pressure": float(row["persist_pressure"]),

    "true_wind_speed": float(row["actual_wind_speed"]),
    "pred_wind_speed": float(row["pred_wind_speed"]),
    "persist_wind_speed": float(row["persist_wind_speed"]),

    "true_precip_mm": float(row["actual_precip_mm"]),
    "pred_precip_mm": float(row["mf1_precip_mm"]),

    "true_rain_prob": float(row["actual_rain"]),
    "pred_rain_prob": float(row["hybrid_rain_prob"]),
    "persist_rain_prob": float(row["persist_rain"]),
}
```

Do not assume `persist_rain` is a precipitation amount. Determine its meaning from the producing validator and document it explicitly.

If `persist_rain` is binary or a rain probability, use it only for rain-occurrence metrics. Do not use it as `persist_precip_mm` unless the source contract proves it is measured in millimeters.

For precipitation amount persistence, use the correct source field. If no valid persistence amount exists in the log, mark that metric unavailable and update the report rather than inventing a value.

Use `hybrid_rain_prob` for operational rain-probability monitoring and retain `mf1_rain_prob` as a diagnostic model-only metric where useful.

## Phase 4: Ensure all five horizons are evaluated

The monitoring engine must:

1. Group rows by `horizon_hours`.
2. Evaluate each supported horizon independently.
3. Produce results for exactly:

   ```text
   horizon_1h
   horizon_3h
   horizon_6h
   horizon_12h
   horizon_24h
   ```

4. Fail if any required horizon has zero valid rows unless an explicit `--allow-missing-horizon` option is supplied.
5. Report per-horizon:
   - sample count;
   - station count;
   - valid-row count;
   - malformed-row count;
   - temperature MAE/RMSE/bias;
   - humidity MAE/RMSE/bias;
   - pressure MAE/RMSE/bias;
   - wind-speed MAE/RMSE/bias;
   - rain Brier score;
   - rain precision/recall/F1/CSI where calculable;
   - precipitation amount metrics where valid;
   - persistence comparison;
   - data-quality summary.

6. Do not default rows to +1h when the horizon field is missing or invalid.

## Phase 5: Add regression tests

Add or update tests, preferably in:

```text
prediction-model/src/test_monitoring.py
```

Required tests:

### Test 1: Current production schema parses

Use the committed `test_predictions_log.csv` or a small fixture with the exact current header. Confirm parsing succeeds.

### Test 2: All five horizons are present

Confirm monitoring output contains:

```python
{"horizon_1h", "horizon_3h", "horizon_6h", "horizon_12h", "horizon_24h"}
```

### Test 3: Correct sample counts

Compare each horizon’s monitoring count with an independent CSV count.

### Test 4: Correct temperature mapping

Construct a one-row fixture where:

```text
actual_temp = 25
pred_temp = 27
```

and verify MAE is 2, not a default-derived value.

### Test 5: Correct rain mapping

Construct a fixture with a known `actual_rain` and `hybrid_rain_prob`, then independently calculate Brier score and compare.

### Test 6: Missing columns fail closed

Remove `actual_temp`, `horizon_hours`, and `hybrid_rain_prob` individually. Confirm a clear schema error is raised.

### Test 7: Invalid horizon fails closed

Use horizon `2` or an empty value and confirm it is rejected rather than assigned to 1h.

### Test 8: Malformed numeric values fail closed or are counted explicitly

Use `NaN`, `Inf`, and nonnumeric text. Confirm behavior matches the documented malformed-row policy.

### Test 9: Current CSV is not silently defaulted

Run monitoring against the committed log and assert:

- all five horizons exist;
- no required-field fallback occurred;
- Brier score is not trivially zero due to missing-field defaults;
- temperature metrics are finite and plausible.

### Test 10: Read-only behavior

Run monitoring without `--output` and verify no tracked file changes.

## Phase 6: Improve monitoring CLI and output

Add explicit CLI behavior:

```bash
python prediction-model/src/monitoring.py \
  --data-dir prediction-model/data \
  --predictions-log prediction-model/data/test_predictions_log.csv \
  --horizons 1 3 6 12 24 \
  --require-all-horizons
```

If these arguments do not exist, implement them or document the equivalent interface.

The JSON report must include:

```json
{
  "monitoring_version": "...",
  "prediction_log_schema_version": "...",
  "input_log_sha256": "...",
  "evaluated_horizons_hours": [1, 3, 6, 12, 24],
  "missing_horizons_hours": [],
  "malformed_row_count": 0,
  "horizons_performance": {}
}
```

The command must return a nonzero exit code when:

- required columns are missing;
- a required horizon is absent under `--require-all-horizons`;
- invalid values exceed the allowed malformed-row threshold;
- the input file is empty;
- the input schema version is incompatible.

## Phase 7: Update CI gates

Update `.github/workflows/prediction-model.yml` so the monitoring gate executes the corrected five-horizon command and tests.

The CI gate must include:

```bash
python prediction-model/src/test_monitoring.py
python prediction-model/src/monitoring.py \
  --data-dir prediction-model/data \
  --predictions-log prediction-model/data/test_predictions_log.csv \
  --horizons 1 3 6 12 24 \
  --require-all-horizons
```

The CI must verify:

1. Five horizons are present.
2. Monitoring output is valid JSON if an output path is used.
3. No required fields are missing.
4. No malformed rows are silently discarded.
5. Tests leave the working tree clean.

Do not rely only on a process exit code from `monitoring.py`; assert the output structure and horizon set.

If `prediction-model/ci/prediction-model.yml` duplicates `.github/workflows/prediction-model.yml`, either remove the duplicate if it is not used, or document why both are retained. Avoid two divergent CI definitions.

## Phase 8: Validate the entire repository

Use the project virtual environment and run:

```bash
python -m py_compile prediction-model/src/*.py
python prediction-model/src/test_canonical_contract.py
python prediction-model/src/test_inference_contract.py
python prediction-model/src/test_monitoring.py
python prediction-model/src/test_provenance.py
python prediction-model/src/smoke_test.py
python prediction-model/src/monitoring.py \
  --data-dir prediction-model/data \
  --predictions-log prediction-model/data/test_predictions_log.csv \
  --horizons 1 3 6 12 24 \
  --require-all-horizons
python prediction-model/src/generate_manifests.py --check-only
python prediction-model/src/verify_provenance.py
git diff --check
git status --short
```

Run monitoring in a clean detached worktree of the final commit as a final reproducibility check.

Confirm the final monitoring output includes all five horizons and plausible metrics. Do not accept output that reports only +1h.

## Artifact and provenance requirements

Do not retrain or regenerate model artifacts unless the monitoring fix changes scorecard semantics or provenance-bearing content.

If scorecards or manifests are regenerated:

1. Record the generation commit.
2. Update implementation/artifact/model-weight provenance consistently.
3. Update hashes.
4. Run the provenance verifier again.
5. Commit the intended artifacts only.

## Required final report

Report:

- starting commit;
- final commit;
- files changed;
- whether any files were deleted;
- archive reference;
- exact monitoring bug fixed;
- old versus new CSV field mapping;
- monitoring schema version;
- input log SHA-256;
- sample count per horizon;
- malformed-row count;
- temperature, rain, and precipitation metrics per horizon;
- monitoring test results;
- canonical/inference/provenance/smoke test results;
- manifest check-only result;
- provenance result;
- CI workflow result or blocker;
- `git diff --check` result;
- final GO/NO-GO recommendation.

## Release gate

Recommend **GO** only if:

- monitoring maps the current CSV schema correctly;
- all five horizons are independently evaluated;
- missing fields fail closed;
- no required metrics are silently based on zero defaults;
- regression tests pass;
- monitoring output is schema-versioned and reproducible;
- the working tree remains clean after tests;
- provenance passes;
- all existing model and inference gates pass;
- CI uses the corrected monitoring command.

Otherwise recommend **NO-GO**, identify every failed gate, and do not push.

Execute the entire fix. Do not stop at a plan.
