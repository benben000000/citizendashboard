# Repository-Grounded Best-in-Class Local-Station Implementation Plan

## Purpose

This plan is based on the actual prediction-model state in the current GitHub repository, not on a blank-slate design.

Current audited remote tip:

```text
39fe267e83e97f4d7a159f36aaaecf2e0c995c89
```

The repository already contains a substantial research and release framework. The next work must preserve that framework and improve forecast quality through controlled, measurable changes.

The realistic target is:

> Become best-in-class for the declared local station, telemetry schema, and 1h–24h horizons under a local-station-only constraint.

The project must not claim to be the best global weather model. The repository does not contain the spatial observations, radar, satellite, NWP fields, reanalysis volume, or compute budget needed for that claim.

## Current repository state

### What is already implemented

The repository currently has:

- canonical MF-1 and MF-2 model pipelines;
- strict real-data telemetry ingestion;
- raw weather and water telemetry preservation;
- five horizons: 1h, 3h, 6h, 12h, and 24h;
- feature-augmented candidate model `MF-1-FEATURED`;
- 75-dimensional candidate context;
- target-specific candidate policy;
- candidate checkpoints, manifests, calibration files, and prediction logs;
- a 20-artifact candidate scorecard inventory;
- active baseline bundles for all five horizons;
- candidate rollback logic;
- anomaly detector integration;
- heat-index derivation;
- UV quarantine;
- daylight-only luminosity beta status;
- monitoring and path-hygiene checks;
- exact artifact hashes for checkpoints, calibration files, and prediction logs;
- clean-worktree, provenance, bundle, inference, smoke, and predictive-quality tests.

### Current verification status

The latest clean detached audit passed:

| Gate | Result |
|---|---:|
| Python compilation | PASS |
| Canonical tests | PASS — 15 |
| Inference tests | PASS — 11 |
| Monitoring tests | PASS — 10 |
| Predictive-quality tests | PASS — 31 |
| Provenance tests | PASS — 14 |
| Smoke tests | PASS |
| Active bundle check | PASS — all 5 horizons |
| Repository provenance verifier | PASS |
| Whitespace check | PASS |
| Clean worktree | PASS |

### Current model performance

The latest candidate scorecard reports:

| Horizon | Candidate temperature MAE | Persistence temperature MAE | Candidate rain Brier | Persistence rain Brier | Candidate wind-direction error | Persistence wind-direction error |
|---:|---:|---:|---:|---:|---:|---:|
| 1h | 0.5007°C | 0.4926°C | 0.1363 | 0.1470 | 40.85° | 39.42° |
| 3h | 1.3001°C | 0.9554°C | 0.1837 | 0.1914 | 50.31° | 46.95° |
| 6h | 1.7042°C | 1.4413°C | 0.1782 | 0.2246 | 56.21° | 53.20° |
| 12h | 1.6119°C | 1.8301°C | 0.1944 | 0.2483 | 58.03° | 57.05° |
| 24h | 1.2359°C | 1.2104°C | 0.2174 | 0.2728 | 54.44° | 54.33° |

The current operational policy is appropriately conservative:

| Target | Current source |
|---|---|
| Temperature | Baseline |
| Humidity | Baseline |
| Pressure | Baseline |
| Wind speed | Candidate |
| Wind direction | Persistence |
| Rain occurrence | Candidate |
| Precipitation amount | Candidate |
| Heat index | Derived from selected temperature and humidity |
| UV index | Blocked |
| Luminosity | Daylight-only beta |

## What must not be changed casually

The following are protected foundations:

- `prediction-model/data/weather_telemetry.csv`;
- `prediction-model/data/water_level_telemetry.csv`;
- canonical dataset, model, training, inference, validation, and test files;
- current active baseline bundles;
- the archive branch `cleanup/archive-before-remediation-20260922`;
- raw-data hashes and manifest provenance;
- target-specific rollback behavior;
- UV quarantine behavior.

Quality improvements must be additive and reversible. Every new candidate must be evaluated beside the current baseline and current `MF-1-FEATURED` candidate.

## Known release-integrity gap

The current candidate manifests contain:

```text
implementation_commit = 5411a7370a5e8c3a09865bc400dc491d21cf4434
artifact_commit = 5411a7370a5e8c3a09865bc400dc491d21cf4434
model_weights_commit = 5411a7370a5e8c3a09865bc400dc491d21cf4434
```

The repository tip is:

```text
39fe267e83e97f4d7a159f36aaaecf2e0c995c89
```

The repository verifier accepts the historical artifact commit, and all file hashes are correct. However, strict final-HEAD provenance is not yet exact. Close this issue before the next quality experiment so that future scorecards have a clean identity.

## Phase 1 — Close provenance before changing model quality

### Files

- `prediction-model/src/train_predictive_quality.py`;
- `prediction-model/src/verify_provenance.py`;
- `prediction-model/data/candidate_artifacts/*.json`;
- `prediction-model/data/candidate_artifacts/*.pt`;
- `prediction-model/data/candidate_artifacts/*.csv`.

### Work

1. Define one release convention for code commit, artifact commit, and model-weights commit.
2. Ensure the trainer records the exact release identity intended by the convention.
3. Ensure prediction-log hashing uses normalized LF content consistently.
4. Regenerate all five horizons from a clean release worktree.
5. Independently verify checkpoint, calibration, and prediction-log hashes.
6. Run provenance verification without a historical-commit override.
7. Commit the corrected artifact set.
8. Recheck the final tip and document the convention if commit self-reference makes exact equality impossible.

### Gate

Do not begin architecture comparison until the candidate artifact lineage is unambiguous. A model-quality experiment with stale or ambiguous artifacts is not admissible evidence.

## Phase 2 — Build a locked local-station benchmark inside the current pipeline

### Files to extend

- `prediction-model/src/dataset.py`;
- `prediction-model/src/validate.py`;
- `prediction-model/src/train_predictive_quality.py`;
- `prediction-model/src/test_predictive_quality.py`;
- `prediction-model/data/candidate_artifacts/predictive_quality_scorecard.json`.

### Work

The current scorecard compares the candidate with persistence, but the next benchmark must add stronger local-only references:

- seasonal persistence;
- hourly climatology;
- weekly climatology;
- damped persistence;
- autoregression;
- ridge regression;
- gradient-boosted trees;
- current canonical MF-1;
- current MF-2;
- current MF-1-FEATURED.

Use the same target rows, horizons, and untouched test periods for every model. Add at least three rolling-origin evaluation periods. Preserve one final test period that is never used for feature or model selection.

Add the following to the scorecard:

- absolute metrics;
- skill relative to persistence;
- skill relative to seasonal climatology;
- confidence intervals;
- worst rolling period;
- worst regime;
- sample counts;
- quarantine counts;
- station segment counts where station identifiers permit segmentation.

### Gate

A candidate may not be called improved because it wins one aggregate split. It must win or meet non-inferiority gates on multiple rolling periods.

## Phase 3 — Audit current labels and features before adding capacity

### Files to inspect and extend

- `prediction-model/src/dataset.py`;
- `prediction-model/src/anomaly_detector.py`;
- `prediction-model/src/feature_engineering.py` if present;
- `prediction-model/src/test_canonical_contract.py`;
- `prediction-model/src/test_predictive_quality.py`.

### Work

Create a target audit for:

- exact target timestamp alignment;
- 1h, 3h, 6h, 12h, and 24h lead-time tolerance;
- unit consistency;
- missing labels;
- duplicate timestamps;
- impossible values;
- frozen sensors;
- sensor resets;
- suspicious step changes;
- target leakage through derived features;
- rain threshold definition;
- precipitation accumulation window;
- calm-wind definition;
- heat-index derivation;
- UV calibration status;
- luminosity day/night status.

For every engineered feature, record:

```text
feature_name
source_columns
lookback_window
latest_allowed_timestamp
transformation
missing_value_rule
```

Add mutation tests that alter values after the forecast timestamp and verify that the feature vector does not change.

### Gate

No target-specific model is promoted until its labels and features pass the as-of contract. If a target has insufficient trustworthy labels, mark it information-limited or blocked rather than compensating with a larger network.

## Phase 4 — Replace the current shared candidate with target-specific objectives

### Files to extend

- `prediction-model/src/model.py`;
- `prediction-model/src/train_predictive_quality.py`;
- `prediction-model/src/validate.py`;
- `prediction-model/src/inference.py`;
- `prediction-model/src/test_inference_contract.py`;
- `prediction-model/src/test_predictive_quality.py`.

### 4.1 Temperature, humidity, and pressure: residual models

The current candidate loses to persistence for temperature at 1h, 3h, and 6h. Do not respond by simply enlarging the feature-augmented network.

Implement a residual path:

```text
forecast = strong_local_baseline + learned_residual
```

Compare residuals over:

- persistence;
- damped persistence;
- seasonal persistence;
- autoregression.

Use target-specific scaling and quantile outputs. The model should learn when to deviate from persistence, not relearn persistence from scratch.

### 4.2 Wind direction: vector prediction

The current candidate loses to persistence at every horizon. Replace scalar direction regression with:

```text
u = speed * cos(direction)
v = speed * sin(direction)
direction = atan2(v, u)
```

Evaluate direction only above a documented calm-wind threshold and report calm conditions separately. Keep persistence as the fallback in calm conditions.

### 4.3 Rain occurrence: calibrated classification

Retain the current candidate rain module as a reference, but add:

- class-frequency diagnostics;
- log loss;
- Brier skill score;
- reliability curves;
- precision-recall AUC;
- critical success index;
- recall at fixed false-alarm rates;
- rain onset lead time.

Thresholds must be selected on calibration data only.

### 4.4 Precipitation amount: hurdle model

Split the target into:

```text
P(rain) × amount_given_rain
```

Evaluate all rows, rainy rows, and heavy-rain rows separately. Do not allow dry-majority performance to hide poor heavy-rain behavior.

### 4.5 Heat index

Keep the existing derived NOAA calculation. Validate that it uses the selected temperature and humidity sources and does not create discontinuities when the policy mixes baseline and candidate outputs.

### Gate

A target-specific model must beat or meet non-inferiority against the strongest local baseline on all promoted rolling periods. The global candidate must not be promoted if a target-specific model fails.

## Phase 5 — Add calibrated probabilistic forecasts

### Files to extend

- `prediction-model/src/model.py`;
- `prediction-model/src/train_predictive_quality.py`;
- calibration JSON artifacts;
- `prediction-model/src/validate.py`;
- `prediction-model/src/inference.py`.

### Work

Add p10, p50, and p90 outputs for:

- temperature;
- humidity;
- pressure;
- wind speed;
- precipitation amount.

Keep calibrated rain probabilities. Use a disjoint calibration period and retain conformal calibration by target and horizon where sample size permits.

Report:

- empirical coverage;
- interval width;
- weighted interval score;
- sharpness;
- coverage by horizon;
- coverage by rain/dry regime;
- coverage during extremes;
- calibration drift.

### Gate

An interval is not production-ready because it exists. It must achieve declared coverage without becoming so wide that it is useless.

## Phase 6 — Use compact local ensembles

### Files to extend

- `prediction-model/src/train_predictive_quality.py`;
- `prediction-model/src/model.py`;
- `prediction-model/src/validate.py`;
- candidate manifest and scorecard schemas.

### Work

Train a small, controlled ensemble of:

- persistence or residual autoregression;
- ridge regression;
- gradient-boosted tree;
- compact temporal neural model;
- current MF-1-FEATURED model.

Use 3–5 seeds where compute allows. Use out-of-fold predictions for any stacking layer. Fit blend weights on calibration data, never on final test data.

Track:

- training time;
- peak memory;
- model size;
- inference latency;
- skill gain;
- calibration gain;
- failure rate.

### Gate

An ensemble must outperform its best member or improve uncertainty quality enough to justify its additional complexity. Otherwise retain the simpler model.

## Phase 7 — Improve anomaly detection using the existing detector

### Files to extend

- `prediction-model/src/anomaly_detector.py`;
- `prediction-model/src/validate.py`;
- `prediction-model/src/monitoring.py`;
- `prediction-model/src/test_predictive_quality.py`.

### Work

Use forecast distributions and historical station baselines to define anomaly probabilities for:

- extreme temperature;
- abrupt temperature change;
- pressure drops;
- heavy precipitation;
- rapid wind increase;
- wind-direction shifts;
- humidity excursions;
- sensor failures.

Separate weather anomalies from sensor anomalies. Create reviewed event labels. Report event recall, false alarms per day, missed events, and warning lead time.

### Gate

No operational anomaly claim without reviewed labels and an explicit false-alarm budget.

## Phase 8 — Prove the local-station information ceiling

### Files to extend

- `prediction-model/src/train_predictive_quality.py`;
- scorecard and experiment registry artifacts.

### Work

Run local-only ablations:

1. recent raw station values;
2. raw values plus lag features;
3. plus seasonal features;
4. plus engineered trends and volatility;
5. plus quality indicators;
6. current candidate architecture;
7. compact ensemble.

Do not add external stations, radar, satellite, NWP, or reanalysis to the production feature set. External information may be used only in a diagnostic oracle experiment to demonstrate what information is missing.

### Decision rule

If no valid local-only model beats persistence for a target and horizon across rolling periods, label that target/horizon:

```text
INFORMATION_LIMITED
```

Do not continue increasing model capacity indefinitely. The correct remedy would require relaxing the local-only constraint.

## Phase 9 — Strengthen promotion gates in the existing policy system

### Files to extend

- `prediction-model/src/validate.py`;
- `prediction-model/src/inference.py`;
- `prediction-model/src/verify_provenance.py`;
- `prediction-model/data/inference_policy.json`;
- `prediction-model/data/candidate_artifacts/predictive_quality_scorecard.json`.

### Work

Retain the existing target-specific policy model, but require for each promoted target:

- positive mean skill;
- confidence interval or block-bootstrap non-inferiority;
- no critical worst-regime failure;
- physical-bound compliance;
- acceptable calibration;
- complete artifact provenance;
- rollback success;
- latency and memory compliance.

Use the current policy structure as the deployment safety mechanism. Do not change all targets to candidate merely because the candidate artifact exists.

### Expected policy during the next iteration

```text
temperature: baseline until short-horizon regression is resolved
humidity: baseline until target-specific evidence improves
pressure: baseline until target-specific evidence improves
wind_speed: candidate only if rolling gates remain positive
wind_direction: persistence until vector model improves
rain_occurrence: candidate if calibration and event gates pass
precipitation_amount: candidate only if rainy/heavy-rain gates pass
heat_index: derived from selected temperature and humidity
uv_index: blocked
light_intensity: daylight beta
```

## Phase 10 — Add monitoring-based rollback

### Files

- `prediction-model/src/monitoring.py`;
- `prediction-model/src/inference.py`;
- `prediction-model/src/generate_bundles.py`;
- active bundle policies;
- candidate policy artifacts.

### Work

Monitor:

- feature drift;
- sensor missingness;
- target bias;
- rolling MAE and Brier score;
- interval coverage;
- rain-event recall;
- anomaly false alarms;
- fallback frequency;
- policy-source usage.

Define automatic actions:

```text
mild drift -> warning
calibration drift -> recalibration
severe target degradation -> target-specific baseline fallback
integrity or provenance failure -> full bundle rollback
```

### Gate

A candidate is not operationally ready until the monitoring system can detect degradation and inference can fall back without code changes.

## Phase 11 — Compute and repository discipline

### Rules

- Do not delete canonical files or raw telemetry.
- Do not commit local virtual environments or temporary logs.
- Store every experiment’s seed, configuration, code commit, data hashes, and output hashes.
- Use compact models and early stopping.
- Prefer residual learning to large direct networks.
- Train one horizon at a time if memory is limited.
- Cache features with schema and data hashes.
- Save resumable checkpoints.
- Record training time and inference latency.
- Require an improvement before accepting added complexity.
- Keep all candidates reversible and separate from active bundles until promotion.

## Phase 12 — Execution order for the next repository changes

### Change set 1: provenance closure

Fix the exact release-identity convention and regenerate the current candidate artifacts. This is a prerequisite for trustworthy comparisons.

### Change set 2: benchmark expansion

Add seasonal persistence, autoregression, ridge, tree, rolling-origin splits, uncertainty intervals, and worst-regime tables to the existing scorecard.

### Change set 3: label and leakage audit

Add target-alignment assertions, feature mutation tests, sensor-quality reports, and rain/precipitation definition checks.

### Change set 4: target-specific quality models

Implement residual temperature forecasting, vector wind direction, hurdle precipitation, and calibrated classification.

### Change set 5: compact ensemble

Add only a small ensemble if the target-specific models demonstrate complementary errors.

### Change set 6: anomaly and uncertainty release

Add reviewed anomaly labels, event metrics, quantile outputs, conformal intervals, and calibration monitoring.

### Change set 7: conditional rollout

Promote only target/horizon combinations that pass all gates. Keep baseline and persistence fallback policies.

## Hard stop conditions

Stop model expansion and document the information limit when any of these is true:

- labels remain unreliable after the data audit;
- no local-only model beats persistence after rolling evaluation;
- gains disappear outside the tuned evaluation period;
- uncertainty coverage fails during important regimes;
- anomaly labels cannot be reviewed;
- model complexity increases without measurable skill gain;
- training or inference exceeds local resource limits;
- target-specific results conflict with the global promotion claim;
- exact provenance cannot be established.

## Definition of success

This project can claim best-in-class **for the defined local-station task** only when:

1. the candidate beats strong local baselines across at least three untouched rolling periods;
2. every promoted target has positive or statistically defensible non-inferior skill;
3. temperature no longer materially loses to persistence at promoted horizons;
4. wind direction uses vector prediction and meets its circular-error gate;
5. rain occurrence and precipitation amount pass event and heavy-rain gates;
6. anomaly alerts have reviewed labels and a false-alarm budget;
7. intervals have verified coverage and useful sharpness;
8. model and policy artifacts have complete provenance;
9. monitoring detects drift and triggers fallback;
10. the system meets local compute and latency limits.

## Final recommendation

The repository is ready for a **quality-improvement phase**, not a global model replacement.

The first implementation task should be the benchmark and label audit, not a larger neural network. The second should be target-specific modeling for the current weak points: temperature and wind direction. The third should be calibrated uncertainty and anomaly evaluation. Only after those gates pass should the candidate policy be expanded.

If the local telemetry cannot beat persistence for a target after these steps, retain the baseline and mark that target information-limited. Achieving better performance would then require relaxing the local-only constraint, not adding more layers.

## References

[1]: https://deepmind.google/blog/graphcast-ai-model-for-faster-and-more-accurate-global-weather-forecasting/ "GraphCast: AI model for faster and more accurate global weather forecasting"
[2]: https://www.nature.com/articles/s41586-024-08252-9 "Probabilistic weather forecasting with machine learning"
[3]: https://sites.research.google/gr/weatherbench/ "WeatherBench 2 benchmark for data-driven global weather models"
[4]: https://mediatum.ub.tum.de/doc/1597509/1597509.pdf "WeatherBench: A benchmark data set for data-driven weather forecasting"
