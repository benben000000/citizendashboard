# Complete Prediction-Model Implementation Plan

## 1. Objective

Improve the current repository from a tested local forecasting research system into the strongest defensible local-station prediction system possible under the current constraint:

> Use only the local station’s historical telemetry and modest local compute to produce calibrated 1h, 3h, 6h, 12h, and 24h forecasts for weather variables and anomalies.

This plan does not promise global state-of-the-art weather forecasting. A single station cannot observe approaching weather systems outside its measurement history. The plan instead defines how to maximize local-station skill without hiding failures behind aggregate metrics or unsupported promotion claims.

## 2. Current baseline and release status

Current GitHub tip:

```text
4ecb09f18c3ef8fa0fa8286c76a43ad104690fcf
```

The repository currently has:

- canonical MF-1 and MF-2 pipelines;
- `MF-1-FEATURED` candidate modeling;
- five horizons: 1h, 3h, 6h, 12h, and 24h;
- candidate checkpoints, calibration artifacts, prediction logs, and manifests;
- active baseline bundles;
- target-specific inference policy;
- anomaly detector;
- monitoring;
- model comparison and predictive-quality scorecards;
- 46 passing predictive-quality tests;
- passing canonical, inference, monitoring, provenance, bundle, and smoke gates.

Current candidate behavior:

- rain occurrence: better than persistence at every tested horizon;
- temperature: better than persistence at 12h and 24h, worse at 1h, 3h, and 6h;
- wind direction: worse than persistence at all horizons;
- wind speed: conditionally promoted;
- heat index: derived from selected temperature and humidity outputs;
- UV: blocked by calibration quality;
- luminosity: daylight-only beta.

Current active policy is therefore appropriately conservative. The candidate must not replace every baseline bundle.

## 3. Required work order

The work must be performed in this order:

1. Preserve the repository archive and create a clean worktree.
2. Close exact artifact provenance.
3. Freeze the benchmark and target definitions.
4. Expand baseline and rolling-origin evaluation.
5. Audit labels and feature causality.
6. Implement target-specific quality models.
7. Add uncertainty and anomaly evaluation.
8. Add compact ensembles only if justified.
9. Strengthen inference policy and rollback.
10. Regenerate all artifacts from the final code tip.
11. Run independent verification.
12. Promote only target/horizon combinations that pass every gate.

No architecture experiment should be accepted before provenance, target definitions, and benchmark splits are frozen.

# Phase 0 — Repository protection and experiment branch

## Goal

Make the entire quality program reversible and prevent accidental changes to the active model.

## Actions

1. Fetch `origin/main`.
2. Record the starting commit.
3. Verify the existing archive branch:

```text
cleanup/archive-before-remediation-20260922
```

4. Create a dedicated implementation branch from the latest `origin/main`.
5. Confirm raw telemetry and canonical source files are present.
6. Run the existing 46-test suite before modifications.
7. Save the baseline results as an external local audit record, not as a machine-specific repository artifact.

## Files that must remain protected

- `prediction-model/data/weather_telemetry.csv`;
- `prediction-model/data/water_level_telemetry.csv`;
- `prediction-model/src/dataset.py`;
- `prediction-model/src/model.py`;
- `prediction-model/src/inference.py`;
- `prediction-model/src/train.py`;
- `prediction-model/src/train_standalone.py`;
- `prediction-model/src/train_and_evaluate_canonical.py`;
- `prediction-model/src/validate.py`;
- all canonical contract and smoke tests;
- current active bundles;
- raw-data and artifact manifests.

## Gate

Stop if the working tree is not clean, raw telemetry hashes change unexpectedly, or the archive cannot be identified.

# Phase 1 — Close exact provenance

## Goal

Ensure every candidate artifact can be traced to the exact code and data used to generate it.

## Current defect

The newest repository tip is:

```text
4ecb09f18c3ef8fa0fa8286c76a43ad104690fcf
```

The candidate manifests currently identify the previous artifact commit:

```text
39fe267e83e97f4d7a159f36aaaecf2e0c995c89
```

The artifact hashes and repository verifier pass, but strict final-HEAD identity is not closed.

## Files

- `prediction-model/src/train_predictive_quality.py`;
- `prediction-model/src/verify_provenance.py`;
- `prediction-model/src/generate_bundles.py`;
- `prediction-model/data/candidate_artifacts/*`;
- `prediction-model/data/bundles/*`;
- scorecard and policy JSON files;
- provenance tests.

## Implementation

1. Define the release identity convention.
2. Make every manifest record:
   - implementation commit;
   - training-data hashes;
   - feature-schema hash;
   - training configuration hash;
   - seed;
   - artifact hashes;
   - generation timestamp.
3. Normalize prediction-log line endings before hashing.
4. Make `verify_provenance.py` fail on stale artifact identity unless an explicitly documented historical-artifact mode is requested.
5. Regenerate all candidate horizons from the final intended release worktree.
6. Regenerate active bundles only when policy promotion is justified.
7. Re-run independent hash verification from Git object contents.
8. Commit only after the generated artifact fields and final release convention agree.

## Self-reference rule

If exact commit equality is impossible because generating artifacts changes the commit hash, implement one of these explicit conventions:

- `implementation_commit` equals the code commit, while `artifact_commit` equals the artifact commit, with both accepted and documented;
- create a release commit that contains code, then generate artifacts, then create a dedicated artifact-release commit and define the artifact commit as the release identity;
- use a signed release manifest that records the parent code commit and artifact commit separately.

Do not silently claim exact final-HEAD provenance when the artifact was generated from a parent commit.

## Gate

The provenance verifier must pass in strict mode and the report must state exactly what each commit field means.

# Phase 2 — Freeze the local-station benchmark

## Goal

Make every future comparison scientifically valid and resistant to split tuning.

## Files

- `prediction-model/src/dataset.py`;
- `prediction-model/src/validate.py`;
- `prediction-model/src/train_predictive_quality.py`;
- `prediction-model/src/test_predictive_quality.py`;
- scorecard schemas and documentation.

## Implementation

Define three time roles:

- training period;
- calibration/validation periods;
- untouched rolling test periods.

Create at least three non-overlapping rolling test windows. Keep one final period untouched until all model and policy decisions are frozen.

Do not use random row splits. Preserve temporal ordering and use an embargo when overlapping windows could share information.

Every evaluation row must store:

```text
issue_timestamp_utc
target_timestamp_utc
horizon_hours
station_id
split_name
feature_schema_hash
label_quality_status
```

## Baselines to add

The current persistence comparison is necessary but insufficient. Add:

- persistence;
- damped persistence;
- hourly climatology;
- weekly climatology;
- seasonal persistence;
- autoregression;
- ridge regression;
- gradient-boosted trees;
- MF-1;
- MF-2;
- MF-1-FEATURED;
- compact target-specific models;
- compact ensemble.

## Metrics

### Continuous variables

- MAE;
- RMSE;
- bias;
- median absolute error;
- skill relative to persistence;
- skill relative to climatology;
- p10/p50/p90 interval coverage and width;
- weighted interval score.

### Rain occurrence

- Brier score;
- Brier skill score;
- log loss;
- expected calibration error;
- reliability curve;
- ROC-AUC;
- precision-recall AUC;
- critical success index;
- recall at fixed false-alarm rates;
- rain onset lead time.

### Precipitation amount

Report all samples, rainy samples, and heavy-rain samples separately:

- MAE;
- RMSE;
- bias;
- logarithmic error where appropriate;
- heavy-rain recall or exceedance skill.

### Wind direction

- circular MAE;
- circular RMSE;
- calm-wind coverage;
- error conditioned on wind speed;
- vector-component MAE.

### Anomalies

- event recall;
- precision;
- false alarms per day;
- missed events;
- warning lead time;
- performance by severity and season.

## Statistical reporting

Add block bootstrap or time-clustered confidence intervals. Report mean, lower confidence bound, worst period, and worst regime. Do not rely on one average score.

## Gate

A candidate can be called improved only when it wins or is statistically non-inferior across multiple rolling periods and does not fail a critical regime.

# Phase 3 — Audit labels, sensor quality, and feature causality

## Goal

Prevent model changes from compensating for defective labels or leakage.

## Files

- `prediction-model/src/dataset.py`;
- `prediction-model/src/feature_engineering.py` if present;
- `prediction-model/src/anomaly_detector.py`;
- `prediction-model/src/test_canonical_contract.py`;
- `prediction-model/src/test_predictive_quality.py`.

## Label audit

For every target and horizon, verify:

- exact UTC alignment;
- hourly cadence;
- lead-time tolerance;
- duplicate timestamps;
- missing labels;
- unit conversion;
- impossible values;
- frozen sensors;
- sensor reset periods;
- suspicious step changes;
- rain threshold definition;
- precipitation accumulation interval;
- calm-wind threshold;
- heat-index formula;
- UV calibration state;
- luminosity daylight status.

## Feature audit

For each feature, record:

```text
name
source columns
lookback
latest allowed timestamp
transformation
unit
missing-value rule
```

Add causal mutation tests:

1. change values after the issue timestamp;
2. rebuild the feature row;
3. assert that the as-of feature row is unchanged.

Also run truncation tests that remove all future rows and compare the resulting feature vector.

## Gate

Any future-sensitive feature, unreviewed label, or ambiguous unit blocks the associated model from promotion.

# Phase 4 — Implement target-specific quality models

## Goal

Fix the specific weaknesses shown by the current scorecard rather than enlarging the shared model indiscriminately.

## 4.1 Temperature, humidity, and pressure residual models

### Current problem

Temperature loses to persistence at 1h, 3h, and 6h.

### Implementation

Implement:

```text
forecast = baseline forecast + learned residual
```

Use baseline candidates:

- persistence;
- damped persistence;
- seasonal persistence;
- autoregression.

Train the model to predict the residual. Add quantile outputs and target-specific scaling.

### Gate

Temperature must beat or meet non-inferiority against the strongest baseline at every horizon selected for promotion. If it does not, retain the baseline policy.

## 4.2 Wind direction vector model

### Current problem

The candidate loses to persistence at every horizon.

### Implementation

Predict vector components:

```text
u = speed * cos(direction)
v = speed * sin(direction)
```

Recover direction with `atan2`. Evaluate direction separately for:

- calm wind;
- low wind;
- normal wind;
- high wind.

Use persistence in calm conditions unless the vector model clearly improves the result.

### Gate

The vector model must beat persistence in circular error on rolling periods and must not introduce physically impossible vector magnitudes.

## 4.3 Rain occurrence

### Current state

The candidate already beats persistence in Brier score at every tested horizon.

### Implementation

Preserve the candidate as the reference. Add:

- class prevalence checks;
- log loss;
- reliability analysis;
- precision-recall analysis;
- critical success index;
- fixed false-alarm threshold tables;
- event-level onset lead time.

Tune thresholds on calibration data only.

### Gate

Do not accept a lower Brier score if it is caused only by predicting the majority class or if heavy-rain event recall is unacceptable.

## 4.4 Precipitation amount hurdle model

### Implementation

Separate:

```text
P(rain) × amount conditional on rain
```

Use a nonnegative amount head or a transformed positive regression. Score rainy and heavy-rain samples separately.

### Gate

The candidate must improve amount skill without worsening rain occurrence calibration or heavy-rain behavior.

## 4.5 Heat index

### Implementation

Continue deriving heat index from the selected temperature and humidity forecasts. Test mixed-source discontinuities and physical plausibility.

### Gate

Heat-index error and risk-category accuracy must be reported separately from temperature and humidity.

## 4.6 UV and luminosity

UV remains blocked until calibration evidence passes. Luminosity remains daylight-only beta until night-time behavior, seasonal coverage, and sensor quality are validated.

# Phase 5 — Add uncertainty and calibrated forecasts

## Goal

Make predictions useful for anomaly risk and operational decisions.

## Files

- `prediction-model/src/model.py`;
- `prediction-model/src/train_predictive_quality.py`;
- `prediction-model/src/validate.py`;
- calibration artifact generation;
- `prediction-model/src/inference.py`.

## Implementation

Produce p10, p50, and p90 for:

- temperature;
- humidity;
- pressure;
- wind speed;
- precipitation amount.

Retain calibrated rain probabilities. Use a disjoint calibration set. Apply conformal calibration by target and horizon where sample sizes permit.

## Metrics

- empirical coverage;
- interval width;
- weighted interval score;
- sharpness;
- coverage by horizon;
- coverage by rain/dry regime;
- coverage during extremes;
- calibration drift.

## Gate

An 80% interval must achieve coverage within a predefined tolerance and remain narrower than a trivial broad interval. If coverage fails during extremes, document the limitation or implement regime-aware calibration.

# Phase 6 — Use compact ensembles only when justified

## Goal

Reduce variance without supercomputer-scale training.

## Candidate members

- persistence or residual autoregression;
- ridge regression;
- gradient-boosted trees;
- compact temporal neural network;
- current MF-1-FEATURED candidate;
- vector wind model where applicable;
- hurdle precipitation model where applicable.

Use 3–5 seeds or bootstrap members. Use out-of-fold predictions for stacking. Fit blend weights on calibration data only.

Track:

- skill gain;
- calibration gain;
- inference latency;
- memory;
- training time;
- artifact size;
- failure frequency.

## Gate

The ensemble must improve the best single model or materially improve uncertainty quality. Otherwise keep the simpler model.

# Phase 7 — Improve anomaly detection

## Goal

Detect meaningful weather and sensor anomalies rather than ordinary forecast errors.

## Files

- `prediction-model/src/anomaly_detector.py`;
- `prediction-model/src/monitoring.py`;
- `prediction-model/src/validate.py`;
- predictive-quality tests;
- scorecard schema.

## Implementation

Create reviewed labels for:

- extreme heat or cold;
- abrupt temperature change;
- pressure drop;
- heavy rain;
- rapid wind increase;
- wind-direction shift;
- humidity excursion;
- sensor failure or drift.

Separate weather anomalies from sensor anomalies. Derive exceedance probabilities from forecast distributions. Evaluate event-level behavior.

## Gate

No anomaly target can be promoted without reviewed labels, a fixed false-alarm budget, recall, precision, and warning lead time.

# Phase 8 — Prove the local-station information ceiling

## Goal

Determine whether further model complexity can plausibly improve performance under the local-only constraint.

## Required ablations

Run the same rolling benchmark for:

1. recent raw station values;
2. raw values plus lag features;
3. plus seasonal features;
4. plus trends and volatility;
5. plus quality indicators;
6. current MF-1-FEATURED;
7. target-specific models;
8. compact ensemble.

An external-information oracle may be used for diagnosis, but external stations, NWP, radar, satellite, and reanalysis may not enter the production model under the current constraint.

## Decision

If no valid local-only model beats persistence for a target and horizon across independent periods, mark it:

```text
INFORMATION_LIMITED
```

Retain the strongest baseline. Do not endlessly increase model size.

# Phase 9 — Strengthen the current target-specific policy

## Files

- `prediction-model/src/validate.py`;
- `prediction-model/src/inference.py`;
- `prediction-model/src/generate_bundles.py`;
- `prediction-model/src/verify_provenance.py`;
- `prediction-model/data/inference_policy.json`;
- candidate scorecards and bundle manifests.

## Policy rules

Each target and horizon must have an independently evaluated source:

- candidate;
- baseline;
- persistence;
- derived formula;
- blocked.

Promotion requires:

- positive mean skill;
- lower confidence bound above the non-inferiority threshold;
- no critical worst-regime failure;
- physical-bound compliance;
- calibration pass;
- complete provenance;
- rollback pass;
- latency and memory pass.

The policy must be identical between scorecard, bundle manifest, and live inference.

# Phase 10 — Monitoring and rollback

## Files

- `prediction-model/src/monitoring.py`;
- `prediction-model/src/inference.py`;
- `prediction-model/src/generate_bundles.py`;
- active and candidate policies.

## Monitor

- feature drift;
- missingness drift;
- frozen-sensor behavior;
- target bias;
- rolling MAE and Brier score;
- interval coverage;
- rain-event recall;
- anomaly false alarms;
- fallback frequency;
- policy-source usage;
- artifact and policy hashes.

## Response policy

```text
mild drift -> warning
calibration drift -> recalibration
severe target degradation -> target-specific baseline fallback
integrity/provenance failure -> bundle rollback
```

## Gate

Rollback must be tested from a clean checkout and must not require editing source code manually.

# Phase 11 — Release and CI integration

## Required CI gates

Add or preserve CI checks for:

- compile;
- canonical contracts;
- inference contracts;
- monitoring schema;
- predictive-quality tests;
- label and feature audits;
- rolling scorecard schema;
- candidate artifact hashes;
- active bundle consistency;
- strict provenance;
- path hygiene;
- deleted-file reference checks;
- clean generated-artifact check;
- smoke inference;
- rollback simulation.

## Release sequence

1. Freeze code and experiment configuration.
2. Regenerate all five candidate horizons.
3. Generate scorecards and calibration artifacts.
4. Verify hashes independently.
5. Run strict provenance.
6. Review target-specific policy.
7. Run complete tests.
8. Commit the final artifact set.
9. Re-run provenance at the final release identity.
10. Push only if all gates pass.

# Phase 12 — Target-specific promotion plan

## Immediate status

Keep the following policy until new evidence changes it:

```text
temperature: baseline
humidity: baseline
pressure: baseline
wind_speed: candidate, conditional
wind_direction: persistence
rain_occurrence: candidate, conditional
precipitation_amount: candidate, conditional
heat_index: derived from selected temperature and humidity
uv_index: blocked
light_intensity: daylight beta
```

## Promotion rules

### Temperature

Do not promote until the residual model beats persistence at 1h, 3h, and 6h on rolling periods without failing 12h or 24h.

### Wind direction

Do not promote until the vector model beats persistence in circular error with calm-wind handling.

### Rain occurrence

Promote only if Brier improvement remains after reliability, precision-recall, and event-level checks.

### Precipitation amount

Promote only if rainy and heavy-rain performance is acceptable.

### Wind speed

Retain conditional promotion only if the result survives worst-regime and rolling-period checks.

### UV

Keep blocked until sensor calibration is repaired and independently verified.

# Phase 13 — Failure handling

## Temperature remains worse than persistence

Retain persistence or baseline. Investigate residual specification, seasonal structure, sensor drift, and target alignment. Do not enlarge the network without evidence.

## Rain improvement is caused by class imbalance

Reject the claim. Require Brier, log loss, reliability, precision-recall, event recall, and heavy-rain metrics.

## Wind direction remains poor

Use vector components, calm-wind conditioning, and persistence fallback. Treat direction as information-limited if no local-only model improves it.

## Rare events are too sparse

Use training-only event weighting or focal objectives, but preserve untouched event-inclusive tests. Never duplicate test events into training.

## Sensor drift dominates

Quarantine affected labels, mark the period, recalibrate the sensor, and fall back to the baseline.

## Uncertainty intervals fail during storms

Use regime-aware calibration, widen intervals honestly, or mark storm intervals unsupported.

## Candidate wins only one period

Reject promotion. Treat it as unstable evidence.

## Compute becomes excessive

Reduce architecture size, train one horizon at a time, use early stopping, cache features, and reduce ensemble members.

## Local-only signal is insufficient

Document the information ceiling. Retain the strongest baseline. Do not claim that more neural layers solve unobserved atmospheric state.

# Phase 14 — Definition of done

The implementation is complete only when all of the following are true:

1. Exact artifact and data provenance is closed under the documented release convention.
2. At least three rolling test periods are scored.
3. Strong local baselines are included.
4. Feature mutation and leakage tests pass.
5. Target labels and sensor quality are audited.
6. Temperature has a residual-model experiment.
7. Wind direction has a vector-model experiment.
8. Precipitation amount has a hurdle-model experiment.
9. Rain occurrence has calibration and event metrics.
10. Continuous targets have calibrated intervals or an explicit reason they are not ready.
11. Anomaly labels and event metrics are documented.
12. Worst-period and worst-regime results are included.
13. The policy is target-specific and matches live inference.
14. Monitoring and rollback are tested.
15. Only passing target/horizon combinations are promoted.
16. Raw telemetry remains unchanged.
17. All tests pass from a clean checkout.
18. The final release has no stale manifests or unexplained artifact commits.

## Final recommendation

The repository has moved from a basic prediction pipeline to a controlled candidate lifecycle. The next stage is not a blanket candidate rollout. It is a disciplined quality program focused on the current weaknesses:

1. close provenance;
2. freeze stronger rolling benchmarks;
3. audit labels and causality;
4. build residual temperature forecasting;
5. build vector wind direction;
6. build hurdle precipitation;
7. add calibrated uncertainty;
8. evaluate anomalies as events;
9. use compact ensembles only when they earn their complexity;
10. promote targets individually.

If these steps fail to beat persistence for a target, mark that target information-limited and retain the baseline. That is the correct scientific outcome under the local-station-only constraint.
