# Next Required Action: Train, Evaluate, and Conditionally Promote the New Forecasting Models

## Objective

The repository now contains next-phase model logic and a 67-test quality suite, but it does not yet contain new trained evidence proving that the next-phase models outperform the current candidate.

The next required action is therefore an **evidence-generation release cycle**:

1. train the new target-specific models;
2. evaluate them across all five horizons;
3. compare them against strong local baselines and the current candidate;
4. measure rolling-origin and regime-specific performance;
5. generate uncertainty and anomaly metrics;
6. regenerate provenance-correct artifacts;
7. promote only targets that pass strict gates.

Current repository tip:

```text
88120003ee50e10bd3f440a4b1eb5f2062550544
```

Current candidate artifacts were generated at the previous implementation commit:

```text
4ecb09f18c3ef8fa0fa8286c76a43ad104690fcf
```

The next release must regenerate artifacts from the new tip and must not reuse the old scorecard as evidence for the new models.

## Scope

The next evidence cycle covers:

- residual temperature forecasting;
- residual humidity and pressure evaluation where implemented;
- vector wind-direction forecasting;
- calm-wind fallback behavior;
- hurdle precipitation occurrence and amount forecasting;
- rolling-origin evaluation;
- quantile and interval evaluation;
- anomaly-event evaluation;
- compact ensemble comparison if individual models justify it;
- target-specific policy generation;
- strict artifact provenance.

The following remain unchanged unless new evidence passes their gates:

- raw telemetry;
- canonical MF-1 and MF-2 pipelines;
- active baseline bundles;
- UV blocked status;
- luminosity daylight-only beta status;
- target-specific rollback behavior.

# Step 1 — Create a clean evidence-generation worktree

## Actions

1. Fetch `origin/main`.
2. Confirm `origin/main` equals:

```text
88120003ee50e10bd3f440a4b1eb5f2062550544
```

3. Create a detached clean worktree or dedicated experiment branch.
4. Verify the repository has no uncommitted changes.
5. Confirm the raw telemetry files exist.
6. Record SHA-256 hashes for:
   - `prediction-model/data/weather_telemetry.csv`;
   - `prediction-model/data/water_level_telemetry.csv`.
7. Run the existing 67 predictive-quality tests and the complete release gates before training.

## Gate

Stop if the worktree is dirty, the raw hashes differ unexpectedly, or any baseline test fails.

# Step 2 — Freeze the experiment configuration

## Required configuration

Create one versioned configuration object for the evidence cycle containing:

```text
code_commit
raw_weather_sha256
raw_water_sha256
feature_schema_hash
label_schema_hash
horizons: [1, 3, 6, 12, 24]
rolling_fold_count
embargo_hours
calibration_periods
final_test_period
model_family
training_seed
optimizer
learning_rate
batch_size
epoch_limit
early_stopping_patience
loss_configuration
quantile_levels
calm_wind_threshold
rain_threshold
heavy_rain_threshold
anomaly_thresholds
```

Use fixed seeds for Python, NumPy, PyTorch, and any tree-model libraries. Record the actual environment versions.

## Gate

No hyperparameter, threshold, feature, or policy change may be made after the final test period is opened.

# Step 3 — Freeze the benchmark and folds

## Required folds

Use at least three non-overlapping rolling-origin folds plus one final untouched test period.

Each fold must contain:

```text
training interval
calibration interval
validation interval
test interval
embargo interval
```

For every sample, record:

```text
issue_timestamp_utc
target_timestamp_utc
horizon_hours
fold_id
split_name
station_id
label_quality_status
```

## Checks

Verify:

- no train/test timestamp overlap;
- no calibration/test overlap;
- no future data in features;
- no target duplication across folds;
- embargo enforcement;
- identical evaluation rows across model families.

## Gate

If folds are not disjoint, discard the run and rebuild the split before looking at metrics.

# Step 4 — Train the required model candidates

Train each model separately for 1h, 3h, 6h, 12h, and 24h. Do not silently substitute the old candidate model.

## 4.1 Required references

Evaluate and save predictions for:

- persistence;
- damped persistence;
- hourly climatology;
- weekly climatology;
- autoregression;
- ridge regression;
- gradient-boosted trees;
- canonical MF-1;
- current MF-1-FEATURED candidate.

## 4.2 Residual temperature model

Train:

```text
predicted_temperature = baseline_temperature + learned_residual
```

Evaluate at least these baselines:

- persistence;
- damped persistence;
- seasonal/autoregressive temperature baseline.

The model must predict residuals, not only absolute temperature. Save the baseline type and residual normalization in the manifest.

Generate p10, p50, and p90 outputs if the quantile path is enabled.

## 4.3 Residual humidity and pressure models

If the implementation is enabled, apply the same residual structure to humidity and pressure. Otherwise record them as not trained in the scorecard rather than implying they were evaluated.

Enforce physical bounds and report any clipped or invalid output.

## 4.4 Vector wind-direction model

Train the model on vector components:

```text
u = wind_speed * cos(direction)
v = wind_speed * sin(direction)
```

Recover direction with `atan2`. Evaluate with a documented calm-wind rule. For calm conditions, compare:

- persistence direction;
- vector model direction;
- fallback policy.

Save vector normalization and calm-wind handling in the manifest.

## 4.5 Hurdle precipitation model

Train two linked components:

```text
rain_probability = P(precipitation > threshold)
amount = amount conditional on rain
```

Evaluate the combined amount forecast and the occurrence probability separately.

Report:

- dry samples;
- rainy samples;
- heavy-rain samples;
- onset transitions;
- cessation transitions.

## 4.6 Compact ensemble

Do not train an ensemble automatically. Train it only if the individual models show complementary errors.

Candidate members may include:

- residual autoregression;
- ridge;
- gradient boosting;
- residual neural model;
- vector wind model;
- hurdle precipitation model;
- current MF-1-FEATURED candidate.

Use out-of-fold predictions to fit blending weights. Fit weights only on calibration data.

# Step 5 — Evaluate each fold and target

## Continuous targets

For temperature, humidity, pressure, and wind speed report:

- MAE;
- RMSE;
- bias;
- median absolute error;
- skill versus persistence;
- skill versus climatology;
- p10/p50/p90 coverage;
- interval width;
- weighted interval score.

## Rain occurrence

Report:

- Brier score;
- Brier skill score;
- log loss;
- expected calibration error;
- reliability bins;
- ROC-AUC;
- precision-recall AUC;
- critical success index;
- recall at fixed false-alarm rates;
- rain onset lead time.

## Precipitation amount

Report separately:

- all-sample MAE and RMSE;
- rainy-sample MAE and RMSE;
- heavy-rain MAE and bias;
- exceedance recall;
- false alarms;
- occurrence/amount joint score.

## Wind direction

Report:

- circular MAE;
- circular RMSE;
- vector-component error;
- calm-wind prevalence;
- error for calm, low, normal, and high wind;
- persistence comparison.

## Anomalies

Report:

- event recall;
- precision;
- false alarms per day;
- missed events;
- warning lead time;
- severity-stratified performance;
- seasonal performance.

## Statistical confidence

Use block bootstrap or time-clustered resampling. Report:

- mean metric;
- confidence interval;
- lower confidence bound;
- worst fold;
- worst regime;
- sample count;
- quarantined count.

# Step 6 — Run the local-station information-ceiling analysis

## Required ablations

Evaluate the same folds and targets for:

1. recent raw station values;
2. raw values plus lag features;
3. plus seasonal features;
4. plus trend and volatility features;
5. plus quality indicators;
6. current MF-1-FEATURED;
7. residual target-specific models;
8. vector wind model;
9. hurdle precipitation model;
10. compact ensemble.

## Decision rule

For each target and horizon classify the result:

```text
IMPROVED
NON_INFERIOR
NO_EVIDENCE
INFORMATION_LIMITED
```

Use `INFORMATION_LIMITED` when no valid local-only model beats the strongest baseline across the independent folds.

Do not increase model capacity merely because a target is information-limited.

# Step 7 — Apply strict promotion gates

Promotion occurs independently by target and horizon.

## General gate

A model must satisfy all of the following:

- positive mean skill or documented non-inferiority;
- lower confidence bound passes the threshold;
- improvement in at least two of three rolling folds;
- no critical worst-fold regression;
- no critical worst-regime regression;
- no physical-bound failures;
- no calibration failure;
- no leakage;
- complete artifact provenance;
- acceptable local compute and latency;
- rollback path passes.

## Target-specific gates

### Temperature

Do not promote unless residual temperature improves the 1h, 3h, and 6h weaknesses without materially degrading 12h or 24h.

### Wind direction

Do not promote unless vector prediction beats persistence in circular error and passes calm-wind handling.

### Rain occurrence

Promote only if Brier improvement remains after log-loss, reliability, precision-recall, and event-recall checks.

### Precipitation amount

Promote only if rainy and heavy-rain metrics pass. Dry-majority performance is insufficient.

### Wind speed

Retain conditional promotion only if the candidate remains positive in rolling and worst-regime analysis.

### Uncertainty

Do not promote intervals unless empirical coverage is within the declared tolerance and interval width remains useful.

### Anomalies

Do not promote anomaly alerts without reviewed labels, event metrics, and a fixed false-alarm budget.

# Step 8 — Generate the next evidence artifacts

## Required artifacts

Create a new versioned artifact namespace containing:

- per-horizon checkpoints;
- per-horizon calibration files;
- per-horizon prediction logs;
- rolling-fold scorecard;
- baseline comparison report;
- target-specific metrics;
- uncertainty metrics;
- anomaly metrics;
- information-ceiling report;
- model-selection report;
- target-specific policy;
- bundle manifests;
- complete provenance manifest.

Do not overwrite the previous candidate artifacts until the new evidence is reviewed.

## Required manifest fields

```text
code_commit
parent_baseline_commit
training_seed
training_config_hash
feature_schema_hash
label_schema_hash
raw_data_hashes
fold_definitions
model_family
horizon_hours
target_name
baseline_name
metrics
confidence_intervals
calibration_metrics
artifact_hashes
policy_decision
limitations
```

# Step 9 — Reconcile scorecard, policy, inference, and bundles

Before promotion, verify exact agreement between:

- scorecard selected source;
- target-specific policy;
- bundle manifest;
- inference source selection;
- checkpoint metadata;
- calibration artifact;
- model and policy hashes.

For each target/horizon, create an explicit decision:

```text
PROMOTE_CANDIDATE
KEEP_BASELINE
KEEP_PERSISTENCE
RESEARCH_ONLY
BLOCKED
INFORMATION_LIMITED
```

The scorecard must explain every decision.

# Step 10 — Run complete release verification

Run:

```text
python -m py_compile prediction-model/src/*.py
python prediction-model/src/test_canonical_contract.py
python prediction-model/src/test_inference_contract.py
python prediction-model/src/test_monitoring.py
python prediction-model/src/test_predictive_quality.py
python prediction-model/src/test_provenance.py
python prediction-model/src/smoke_test.py
python prediction-model/src/generate_bundles.py --check-only
python prediction-model/src/verify_provenance.py
git diff --check
git status --short --branch
```

Also run:

- independent raw-data hash verification;
- independent checkpoint hash verification;
- rolling-fold disjointness verification;
- scorecard/policy equality verification;
- simulated drift and rollback verification;
- path-hygiene checks;
- deleted-artifact reference checks.

## Gate

Any failure blocks commit and promotion. Fix the cause, regenerate affected artifacts, and rerun the complete release cycle.

# Step 11 — Commit and regenerate at the final release identity

## Sequence

1. Commit source and test changes.
2. Record the new commit.
3. Regenerate all next-phase artifacts from that exact release identity.
4. Normalize prediction logs before hashing.
5. Stage only intended source, tests, manifests, scorecards, and artifacts.
6. Run staged diff and whitespace checks.
7. Commit the artifact regeneration.
8. Run strict provenance again against the final repository tip.
9. Verify that all candidate manifests use the documented commit convention.
10. Confirm active bundles remain unchanged unless target-specific promotion passed.

# Step 12 — Push decision

Push to `origin/main` only when:

- all tests pass;
- all provenance checks pass;
- all hashes pass;
- candidate artifacts are generated from the documented release identity;
- no promoted target fails a critical regime;
- no raw telemetry changed;
- active policy matches scorecard;
- rollback passes;
- the commit contains no temporary or machine-specific files.

If any quality gate fails, push the implementation only if the repository policy permits research-only changes and the scorecard clearly marks the candidate as non-production. Do not push a failed operational promotion as production-ready.

# Expected result of this action

The next release must produce one of these honest outcomes:

```text
A. Candidate promotion succeeds for specific targets and horizons.
B. Candidate remains research-only while selected baselines stay active.
C. Some targets are marked information-limited.
D. The experiment is rejected because of leakage, calibration, instability, or weak evidence.
```

All four outcomes are acceptable. The unacceptable outcome is promoting a model based on incomplete or single-split evidence.

## Definition of done

This next action is complete only when:

1. all five horizons are trained for every enabled model family;
2. rolling-origin evaluation has at least three folds;
3. strong local baselines are scored on identical rows;
4. residual temperature results are reported;
5. vector wind results are reported;
6. hurdle precipitation results are reported;
7. uncertainty metrics are reported;
8. anomaly event metrics are reported where labels exist;
9. information-limited targets are identified;
10. per-target and per-horizon promotion decisions are explicit;
11. scorecard, policy, inference, and bundles agree;
12. artifacts have independent hashes;
13. strict provenance passes at the final release identity;
14. all tests pass from a clean checkout;
15. active bundles change only for targets that pass every gate.
