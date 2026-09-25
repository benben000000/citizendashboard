# Final Implementation Plan for All Remaining Prediction-Model Recommendations

## 1. Purpose

The repository has a healthy release foundation, but the latest scorecard still represents the existing candidate rather than complete evidence for the newly implemented next-phase models.

This plan executes every remaining recommendation required before broader promotion:

1. Generate a new evidence scorecard from the next-phase models.
2. Evaluate at least three rolling-origin periods.
3. Improve temperature with residual forecasting.
4. Improve wind direction with vector forecasting.
5. Evaluate precipitation with a hurdle model.
6. Validate uncertainty intervals.
7. Measure anomaly-event performance.
8. Run local-station information-ceiling experiments.
9. Keep target-specific policy and rollback.
10. Close artifact provenance under an explicit release convention.
11. Promote only target/horizon combinations that pass strict evidence gates.

## 2. Current repository state

Latest GitHub `origin/main`:

```text
319cc0152814bd4de77d114cd90ca08381534b71
```

Latest artifact-generation commit:

```text
319cc0152814bd4de77d114cd90ca08381534b71
```

Current candidate artifacts identify the previous implementation/release identity:

```text
99a3fc290bcf9a96c4e3f98575f5c7e8b487bac6
```

The repository currently passes:

- Python compilation;
- 15 canonical tests;
- 11 inference tests;
- monitoring tests;
- 67 predictive-quality tests;
- 14 provenance tests;
- smoke tests;
- five-horizon bundle checks;
- provenance verification;
- artifact hash verification.

Current performance:

- rain occurrence beats persistence at all horizons;
- temperature loses to persistence at 1h, 3h, and 6h;
- temperature beats persistence at 12h and 24h;
- wind direction loses to persistence at all horizons;
- UV remains blocked;
- luminosity remains daylight-only beta.

## 3. Non-negotiable principles

1. Do not claim the next-phase models are better until they are trained and evaluated.
2. Do not tune on the final test period.
3. Do not replace active baselines globally.
4. Do not delete raw telemetry.
5. Do not alter the canonical pipeline without preserving its tests and artifacts.
6. Do not use external weather inputs in the production model under the local-station-only constraint.
7. Do not hide failures with aggregate averages.
8. Do not accept a model that wins only one period or one regime.
9. Do not promote an artifact with ambiguous provenance.
10. Treat `INFORMATION_LIMITED` as a valid scientific result.

# Phase 0 — Protect the repository and establish the experiment branch

## Goal

Make the entire effort reversible and ensure the new evidence cannot overwrite the current release unexpectedly.

## Actions

1. Fetch `origin/main`.
2. Record the current head.
3. Confirm the archive branch remains available:

```text
cleanup/archive-before-remediation-20260922
```

4. Create a dedicated experiment branch from `origin/main`.
5. Confirm the worktree is clean.
6. Confirm raw telemetry files exist.
7. Compute and record raw weather and water hashes.
8. Run the current 67-test baseline suite.
9. Preserve the existing candidate artifacts as the comparison baseline.

## Gate

Stop if any baseline test fails, raw hashes unexpectedly change, or the worktree is dirty.

# Phase 1 — Define one explicit artifact-release convention

## Goal

Remove ambiguity between implementation commit, artifact-generation commit, and final repository tip.

## Required manifest fields

Every new artifact must contain:

```text
code_commit
artifact_commit
parent_code_commit
release_commit
training_config_hash
feature_schema_hash
label_schema_hash
raw_weather_sha256
raw_water_sha256
artifact_sha256
seed
horizon_hours
target_name
model_family
```

## Convention

Use the following process:

1. Commit all source and test changes.
2. Record that code commit as `code_commit` and `parent_code_commit`.
3. Generate all artifacts from that code commit.
4. Stage the artifacts.
5. Create the artifact-release commit.
6. Record the artifact-release commit as `artifact_commit` and `release_commit`.
7. Regenerate only metadata that must refer to the artifact-release commit, without changing model values.
8. Create the final release commit if required by the chosen convention.
9. Make the verifier check the documented relationship rather than silently allowing arbitrary historical commits.

The scorecard must clearly show whether the final repository tip is the artifact-release commit or its child. Do not describe artifacts as exact-final-HEAD if they were generated from a parent.

## Gate

Strict verification must pass and the release report must explain every commit field.

# Phase 2 — Freeze data, labels, features, and configuration

## Goal

Ensure all models use the same reproducible data contract.

## Required data contract

For each sample store:

```text
issue_timestamp_utc
target_timestamp_utc
horizon_hours
station_id
fold_id
split_name
label_quality_status
feature_schema_hash
```

## Required configuration

Record:

```text
horizons = [1, 3, 6, 12, 24]
seed
model family
training epochs
batch size
learning rate
early-stopping patience
loss function
quantile levels
rain threshold
heavy-rain threshold
calm-wind threshold
embargo duration
rolling-fold count
```

## Label checks

Audit:

- UTC alignment;
- hourly cadence;
- missing labels;
- duplicate timestamps;
- unit conversions;
- impossible values;
- frozen sensors;
- reset periods;
- rain accumulation semantics;
- heat-index formula;
- UV calibration;
- luminosity daylight status.

## Causality checks

Change values after the issue timestamp and prove that historical feature rows do not change. Remove all future rows and prove the as-of feature vector remains identical.

## Gate

Any leakage, unknown unit, ambiguous label, or invalid timestamp blocks the affected target.

# Phase 3 — Build the rolling-origin benchmark

## Goal

Produce evidence that survives changes in weather regime and time period.

## Required structure

Create at least three non-overlapping folds plus one final untouched test period:

```text
Fold 1: train -> calibration -> validation -> test
Fold 2: train -> calibration -> validation -> test
Fold 3: train -> calibration -> validation -> test
Final: untouched test period
```

Use chronological splits and an embargo to prevent overlapping target windows.

## Required baselines

Evaluate every model on identical rows:

- persistence;
- damped persistence;
- hourly climatology;
- weekly climatology;
- seasonal persistence;
- autoregression;
- ridge regression;
- gradient-boosted trees;
- canonical MF-1;
- canonical MF-2;
- current MF-1-FEATURED candidate.

## Required reporting

For every target and horizon report:

- sample count;
- quarantine count;
- MAE;
- RMSE;
- bias;
- skill against persistence;
- skill against climatology;
- mean across folds;
- confidence interval;
- lower confidence bound;
- worst fold;
- worst regime.

## Gate

A model cannot be promoted on a single fixed test split or a single favorable period.

# Phase 4 — Train and evaluate residual temperature models

## Goal

Resolve the current temperature weakness at 1h, 3h, and 6h.

## Model form

```text
forecast_temperature = baseline_temperature + learned_residual
```

Train residual models against:

- persistence;
- damped persistence;
- seasonal/autoregressive baseline.

Use only issue-time features:

- recent temperature lags;
- temperature slope;
- rolling volatility;
- humidity and pressure history;
- wind history;
- time-of-day;
- day-of-year;
- sensor-quality indicators.

## Required outputs

- p10;
- p50;
- p90;
- point forecast;
- baseline forecast;
- predicted residual;
- physical-bound status.

## Evaluation

Compare against persistence at every horizon and rolling fold.

## Promotion gate

Promote only if:

- 1h, 3h, and 6h improve or meet the predefined non-inferiority threshold;
- 12h and 24h do not materially regress;
- at least two of three rolling folds improve;
- the lower confidence bound is acceptable;
- no critical regime fails;
- intervals pass calibration;
- latency and artifact size are acceptable.

If the gate fails, keep baseline temperature.

# Phase 5 — Train and evaluate vector wind-direction models

## Goal

Resolve the current wind-direction weakness.

## Model form

Transform direction into vector components:

```text
u = speed * cos(direction)
v = speed * sin(direction)
```

Train the model on `u` and `v`. Recover direction with `atan2`.

## Required handling

Evaluate separately for:

- calm wind;
- low wind;
- normal wind;
- high wind;
- calm-to-windy transitions;
- windy-to-calm transitions.

Use persistence fallback below the documented calm threshold unless the vector model proves superior.

## Metrics

- circular MAE;
- circular RMSE;
- vector-component MAE;
- direction bias;
- error by wind-speed regime;
- invalid-vector count.

## Promotion gate

Promote only if the vector model beats persistence in circular error across rolling folds and does not produce physically invalid wind vectors.

If it fails, retain persistence direction.

# Phase 6 — Train and evaluate hurdle precipitation models

## Goal

Improve precipitation amount quality without damaging rain occurrence calibration.

## Model form

```text
rain_probability = P(precipitation > threshold)
conditional_amount = amount | rain
combined_amount = rain_probability * conditional_amount
```

## Required slices

Report results for:

- all samples;
- dry samples;
- rainy samples;
- heavy-rain samples;
- onset transitions;
- cessation transitions;
- each horizon;
- each rolling fold.

## Metrics

Occurrence:

- Brier score;
- Brier skill score;
- log loss;
- reliability;
- precision-recall AUC;
- critical success index;
- event recall.

Amount:

- MAE;
- RMSE;
- bias;
- heavy-rain bias;
- exceedance recall;
- false alarms.

## Promotion gate

The hurdle model must improve or maintain occurrence calibration and improve conditional amount performance. Dry-majority accuracy alone is insufficient.

# Phase 7 — Add and validate uncertainty intervals

## Goal

Make predictions useful for risk and anomaly decisions.

## Targets

Generate intervals for:

- temperature;
- humidity;
- pressure;
- wind speed;
- precipitation amount.

Use disjoint calibration data. Apply conformal calibration by target and horizon when sample sizes support it.

## Metrics

- empirical coverage;
- interval width;
- weighted interval score;
- sharpness;
- coverage by weather regime;
- coverage during extremes;
- calibration drift.

## Gate

Do not promote intervals unless declared coverage is achieved within tolerance and the intervals are not unreasonably broad.

# Phase 8 — Evaluate anomaly events

## Goal

Measure operational anomaly usefulness instead of relying on point-error averages.

## Event labels

Create reviewed labels for:

- extreme heat or cold;
- rapid temperature change;
- pressure drop;
- heavy rain;
- rapid wind increase;
- wind-direction shift;
- humidity excursion;
- sensor failure or drift.

Separate weather anomalies from sensor anomalies.

## Metrics

- event recall;
- precision;
- false alarms per day;
- missed events;
- warning lead time;
- performance by severity;
- performance by season;
- performance by horizon.

## Gate

Do not promote anomaly alerts without reviewed labels, declared thresholds, and a false-alarm budget.

# Phase 9 — Run the local-station information-ceiling analysis

## Goal

Determine which targets can realistically be improved using only local telemetry.

## Ablations

Evaluate:

1. raw recent values;
2. lagged values;
3. seasonal features;
4. trend and volatility features;
5. sensor-quality features;
6. current candidate;
7. residual temperature model;
8. vector wind model;
9. hurdle precipitation model;
10. compact ensemble.

## Classification

For every target/horizon assign:

```text
IMPROVED
NON_INFERIOR
NO_EVIDENCE
INFORMATION_LIMITED
```

Use `INFORMATION_LIMITED` when no valid local-only model beats the strongest baseline across independent folds.

# Phase 10 — Evaluate a compact ensemble only when justified

## Goal

Reduce variance without adding unjustified complexity.

Use out-of-fold predictions to estimate blend weights on calibration data only.

Candidate members:

- autoregression;
- ridge;
- gradient boosting;
- residual neural models;
- vector wind model;
- hurdle precipitation model;
- current candidate.

## Gate

Use the ensemble only if it improves point skill, interval quality, or worst-regime robustness enough to justify its compute, latency, and maintenance cost.

# Phase 11 — Produce the new evidence scorecard

## Required files

Create a new versioned artifact namespace rather than overwriting the previous candidate until review is complete.

Required artifacts:

- per-horizon model checkpoints;
- calibration files;
- prediction logs;
- rolling-origin scorecard;
- baseline comparison report;
- uncertainty report;
- anomaly report;
- information-ceiling report;
- model-selection report;
- target-specific policy;
- bundle manifests;
- provenance manifest.

## Required scorecard fields

```text
code_commit
artifact_commit
release_commit
parent_baseline_commit
raw_data_hashes
feature_schema_hash
label_schema_hash
training_seed
fold_definitions
model_family
horizon_hours
target_name
baseline_name
point_metrics
interval_metrics
calibration_metrics
anomaly_metrics
confidence_intervals
worst_fold
worst_regime
policy_decision
limitations
```

# Phase 12 — Reconcile policy, inference, and bundles

## Goal

Ensure the scorecard decision is the decision used in live inference.

For every target and horizon verify equality across:

- scorecard selected source;
- inference policy;
- bundle manifest;
- checkpoint metadata;
- calibration artifact;
- inference output;
- model and policy hashes.

Possible decisions:

```text
PROMOTE_CANDIDATE
KEEP_BASELINE
KEEP_PERSISTENCE
RESEARCH_ONLY
BLOCKED
INFORMATION_LIMITED
```

## Current default policy until gates pass

```text
temperature: baseline
humidity: baseline
pressure: baseline
wind_speed: candidate, conditional
wind_direction: persistence
rain_occurrence: candidate, conditional
precipitation_amount: candidate, conditional
heat_index: derived from selected outputs
uv_index: blocked
light_intensity: daylight beta
```

# Phase 13 — Validate monitoring and rollback

## Monitoring signals

Track:

- feature drift;
- missingness;
- frozen sensors;
- target bias;
- rolling MAE;
- rolling Brier score;
- interval coverage;
- anomaly false alarms;
- fallback frequency;
- selected policy source;
- artifact hashes.

## Response

```text
mild drift -> warning
calibration drift -> recalibration
target degradation -> target-specific fallback
hash/provenance failure -> rollback
```

Simulate each condition and prove rollback works without source edits.

# Phase 14 — Run final release gates

## Required commands

Run from a clean checkout:

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

- independent raw-data hashes;
- independent artifact hashes;
- rolling-fold disjointness;
- policy/scorecard equality;
- rollback simulation;
- path-hygiene checks;
- deleted-reference checks.

Any failure blocks promotion.

# Phase 15 — Final artifact and commit sequence

1. Commit source and test changes.
2. Record the code commit.
3. Train and evaluate all horizons from that code commit.
4. Generate candidate artifacts and scorecards.
5. Normalize prediction logs before hashing.
6. Generate policies and bundles.
7. Verify independent hashes.
8. Commit the artifact-release set.
9. Update and verify release metadata under the documented convention.
10. Run strict provenance against the final release state.
11. Confirm active bundles changed only for passing target/horizon decisions.
12. Review `git diff`, `git status`, and `git diff --check`.
13. Push only after all gates pass.

# Phase 16 — Promotion decision matrix

## Temperature

- Promote only if residual forecasting improves 1h, 3h, and 6h.
- Keep baseline if any critical rolling period fails.

## Wind direction

- Promote only if vector direction beats persistence across rolling periods.
- Keep persistence if the information ceiling is reached.

## Rain occurrence

- Continue conditional candidate use if calibration and event metrics remain positive.
- Revert to baseline if Brier improvement is not reliable across folds.

## Precipitation amount

- Promote only if rainy and heavy-rain metrics improve.
- Do not rely on all-sample dry-majority metrics.

## Wind speed

- Keep candidate conditional until worst-regime and rolling evidence passes.

## Uncertainty

- Promote only calibrated intervals.
- Keep point forecasts research-only if intervals fail.

## Anomalies

- Promote only reviewed event detectors with acceptable false-alarm rates.

## UV and luminosity

- Keep UV blocked until calibration evidence passes.
- Keep luminosity daylight beta until night and seasonal behavior are validated.

# Phase 17 — Failure handling

## If temperature remains worse than persistence

Retain baseline temperature and investigate label alignment, residual definition, seasonal features, sensor drift, and regime-specific failure. Do not increase model size without evidence.

## If wind direction remains worse

Retain persistence, improve calm-wind handling, and mark the target information-limited if vector modeling fails.

## If rain gains disappear

Reject promotion and keep the current policy only if the existing candidate remains valid on the frozen benchmark.

## If heavy-rain samples are too sparse

Report the limitation. Use training-only weighting if justified, but do not oversell confidence.

## If intervals fail during extremes

Use regime-aware calibration, widen them honestly, or block uncertainty promotion.

## If anomaly labels are insufficient

Do not promote anomaly alerts. Continue data-labeling work.

## If provenance becomes ambiguous

Stop release and regenerate under the documented convention.

## If compute exceeds local limits

Use one horizon at a time, cache features, reduce ensemble members, and use early stopping. Never skip the final untouched test.

# Phase 18 — Definition of done

The full recommendation set is complete only when:

1. the current candidate is frozen and reproducible;
2. at least three rolling-origin folds are evaluated;
3. strong local baselines are included;
4. labels and causal features are audited;
5. residual temperature results are reported;
6. vector wind-direction results are reported;
7. hurdle precipitation results are reported;
8. uncertainty intervals are evaluated;
9. anomaly-event metrics are reported where labels exist;
10. information-limited targets are identified;
11. compact ensemble value is measured;
12. every target/horizon has an explicit policy decision;
13. scorecard, policy, inference, and bundles agree;
14. monitoring and rollback simulations pass;
15. artifact hashes pass independently;
16. provenance passes under the documented convention;
17. all tests pass from a clean checkout;
18. active bundles change only for target/horizon combinations that pass every gate;
19. the final report clearly states what improved, what did not, and what remains impossible under local-only data.

## Final recommendation

The repository should now move from framework completion to evidence completion. The priority is not adding more tests or more model names. The priority is to train the next-phase models, evaluate them honestly across rolling periods, produce a complete scorecard, and promote only proven target/horizon combinations.

The likely outcome is target-specific rather than universal:

- rain occurrence may remain a candidate success;
- temperature may improve only at selected horizons;
- wind direction may remain persistence or information-limited;
- UV may remain blocked;
- luminosity may remain beta;
- some targets may not justify neural complexity.

That outcome is acceptable and scientifically preferable to a broad unsupported promotion claim.
