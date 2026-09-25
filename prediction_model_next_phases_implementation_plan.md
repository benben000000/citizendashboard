# Next-Phases Prediction-Model Implementation Plan

## Current starting point

The repository is currently at:

```text
4ecb09f18c3ef8fa0fa8286c76a43ad104690fcf
```

The release foundation is now passing:

- 53 predictive-quality tests;
- canonical contract tests;
- inference tests;
- monitoring tests;
- provenance tests;
- smoke tests;
- all five bundle checks;
- strict exact-HEAD artifact provenance;
- independent artifact hashes;
- clean-worktree and whitespace checks.

The next phases should therefore focus on **forecasting quality**, not additional release scaffolding.

The current quality facts are:

- rain occurrence beats persistence at all five horizons;
- temperature loses to persistence at 1h, 3h, and 6h;
- temperature beats persistence at 12h and 24h;
- wind direction loses to persistence at all five horizons;
- wind speed remains conditionally promoted;
- UV remains blocked;
- luminosity remains daylight-only beta.

## Next-phase objective

Improve the weakest target/horizon combinations while preserving the current target-specific fallback policy.

The next release must not globally promote the candidate. It must produce evidence for each target and horizon independently.

## Phase N1 — Freeze the current candidate as the comparison baseline

### Purpose

Prevent the next experiments from overwriting the current evidence.

### Files and artifacts

- `prediction-model/data/candidate_artifacts/predictive_quality_scorecard.json`;
- `prediction-model/data/candidate_artifacts/model_comparison_report.json`;
- `prediction-model/data/candidate_artifacts/candidate_h*h_manifest.json`;
- `prediction-model/data/inference_policy.json`;
- `prediction-model/src/verify_provenance.py`.

### Actions

1. Record the current candidate scorecard as the `MF-1-FEATURED` baseline.
2. Record current per-horizon metrics, policy decisions, data hashes, seed, and configuration.
3. Do not alter the current active bundles.
4. Create a new experiment namespace for all next-phase artifacts.
5. Require every experiment to include a parent baseline commit and baseline artifact hashes.

### Gate

The current candidate can be reproduced exactly from a clean checkout. If not, stop and repair reproducibility before model changes.

## Phase N2 — Implement rolling-origin evaluation

### Purpose

The current scorecard is not sufficient evidence for best-in-class quality unless the gains survive multiple time periods.

### Files

- `prediction-model/src/dataset.py`;
- `prediction-model/src/validate.py`;
- `prediction-model/src/train_predictive_quality.py`;
- `prediction-model/src/test_predictive_quality.py`;
- `prediction-model/data/candidate_artifacts/predictive_quality_scorecard.json`.

### Design

Create at least three non-overlapping rolling test windows:

```text
train window A -> calibration A -> test window A
train window B -> calibration B -> test window B
train window C -> calibration C -> test window C
```

Keep one final untouched test period that is not used for selecting features, hyperparameters, losses, or policy weights.

Avoid random row splits. Preserve chronological order. Apply an embargo when adjacent windows could share overlapping input history or target rows.

### Required reporting

For every model, target, and horizon report:

- sample count;
- MAE;
- RMSE;
- bias;
- skill versus persistence;
- skill versus seasonal climatology;
- mean across periods;
- lower confidence bound;
- worst period;
- worst regime;
- quarantine count;
- missing-label count.

### Gate

A model is eligible for promotion only if it wins or meets non-inferiority across multiple rolling periods. One favorable period is not sufficient.

## Phase N3 — Add stronger local-only baselines

### Purpose

Determine whether the neural candidate is actually better than simple local forecasting methods.

### Models to add

Implement and evaluate:

- damped persistence;
- hourly climatology;
- weekly climatology;
- seasonal persistence;
- autoregression;
- ridge regression;
- gradient-boosted trees;
- current MF-1;
- current MF-2;
- current MF-1-FEATURED.

### Implementation requirements

Each baseline must use the same:

- training rows;
- calibration rows;
- test rows;
- horizon definitions;
- label-quality filters;
- evaluation metrics.

### Gate

Do not promote the neural model if a simpler model performs better with lower compute and lower operational risk.

## Phase N4 — Implement residual temperature forecasting

### Problem

The candidate temperature forecast is worse than persistence at 1h, 3h, and 6h.

### Files

- `prediction-model/src/model.py`;
- `prediction-model/src/train_predictive_quality.py`;
- `prediction-model/src/validate.py`;
- `prediction-model/src/inference.py`;
- `prediction-model/src/test_predictive_quality.py`;
- `prediction-model/src/test_inference_contract.py`.

### Model design

Use:

```text
forecast = baseline_temperature + learned_residual
```

Evaluate three baseline choices:

1. persistence;
2. damped persistence;
3. seasonal/autoregressive baseline.

The neural or tree model predicts only the residual. Include recent temperature slope, rolling volatility, time-of-day, day-of-year, and sensor-quality features that are available at issue time.

Add quantile outputs for p10, p50, and p90 after the point forecast is stable.

### Training controls

- use the same frozen rolling splits;
- use early stopping;
- record seed and configuration;
- compare against the exact current candidate;
- avoid tuning on the final test period;
- save the residual baseline configuration in the manifest.

### Acceptance gates

Temperature promotion requires:

- improvement over persistence at 1h, 3h, and 6h;
- no material regression at 12h and 24h;
- improvement or non-inferiority in at least three rolling periods;
- no unacceptable worst-regime failure;
- no physical-bound violations;
- acceptable latency and model size.

If the residual model fails, retain the current baseline temperature policy.

## Phase N5 — Implement vector wind-direction forecasting

### Problem

The current candidate loses to persistence for wind direction at all horizons.

### Files

- `prediction-model/src/model.py`;
- `prediction-model/src/train_predictive_quality.py`;
- `prediction-model/src/validate.py`;
- `prediction-model/src/inference.py`;
- `prediction-model/src/test_predictive_quality.py`;
- `prediction-model/src/test_inference_contract.py`.

### Model design

Replace scalar direction regression with vector prediction:

```text
u = wind_speed * cos(direction)
v = wind_speed * sin(direction)
```

Train the model to predict `u` and `v`. Recover direction with:

```text
direction = atan2(v, u)
```

Evaluate wind direction separately for:

- calm wind;
- low wind;
- ordinary wind;
- high wind.

Use persistence or a deterministic fallback under a documented calm threshold.

### Metrics

- circular MAE;
- circular RMSE;
- vector-component MAE;
- wind-speed-conditioned direction error;
- calm-wind coverage;
- directional bias.

### Acceptance gates

The vector model must:

- beat persistence in circular error across rolling periods;
- avoid invalid vector magnitudes;
- remain stable during calm-to-windy transitions;
- pass inference and policy consistency tests.

If it fails, keep persistence for direction.

## Phase N6 — Implement hurdle precipitation amount forecasting

### Problem

Precipitation amount and rain occurrence are currently treated as related but require separate evaluation.

### Files

- `prediction-model/src/model.py`;
- `prediction-model/src/train_predictive_quality.py`;
- `prediction-model/src/validate.py`;
- `prediction-model/src/inference.py`;
- scorecard and calibration artifacts;
- predictive-quality tests.

### Model design

Use a two-stage forecast:

```text
P(rain) × amount_given_rain
```

The occurrence head predicts rain probability. The amount head predicts a positive amount conditional on rain. Use a log or other documented positive transformation if needed.

### Required evaluation slices

- all samples;
- rainy samples;
- heavy-rain samples;
- dry-to-rain transitions;
- rain cessation;
- each horizon;
- each rolling test period.

### Metrics

- Brier score for occurrence;
- log loss;
- reliability;
- precision-recall AUC;
- amount MAE;
- amount RMSE;
- heavy-rain bias;
- exceedance recall;
- false-alarm rate.

### Gate

The hurdle model must improve precipitation amount without degrading rain occurrence calibration or heavy-rain event recall.

## Phase N7 — Add calibrated uncertainty

### Purpose

A point prediction is not enough for anomaly decisions or operational use.

### Files

- `prediction-model/src/model.py`;
- `prediction-model/src/train_predictive_quality.py`;
- `prediction-model/src/validate.py`;
- calibration artifacts;
- `prediction-model/src/inference.py`;
- predictive-quality tests.

### Implementation

Add p10, p50, and p90 for:

- temperature;
- humidity;
- pressure;
- wind speed;
- precipitation amount.

Use a disjoint calibration period. Apply conformal calibration by target and horizon where sample sizes allow.

### Metrics

- empirical coverage;
- interval width;
- weighted interval score;
- sharpness;
- coverage by weather regime;
- coverage during high-error events;
- coverage drift over time.

### Gate

An 80% interval must achieve declared coverage within tolerance while remaining useful. If intervals are too wide or fail during important regimes, the target remains research-only.

## Phase N8 — Upgrade anomaly evaluation

### Purpose

Measure whether the system detects meaningful anomalies rather than only improving average error.

### Files

- `prediction-model/src/anomaly_detector.py`;
- `prediction-model/src/validate.py`;
- `prediction-model/src/monitoring.py`;
- `prediction-model/src/test_predictive_quality.py`.

### Labels

Create reviewed event labels for:

- extreme heat or cold;
- rapid temperature change;
- pressure drop;
- heavy rain;
- rapid wind increase;
- wind-direction shift;
- humidity excursion;
- sensor failure or drift.

Separate weather anomalies from sensor anomalies.

### Metrics

- event recall;
- precision;
- false alarms per day;
- missed events;
- median warning lead time;
- performance by severity;
- performance by season and horizon.

### Gate

No anomaly feature is promoted without reviewed labels and a declared false-alarm budget.

## Phase N9 — Run local-station information-ceiling experiments

### Purpose

Determine whether further model complexity can improve the forecast under the local-only constraint.

### Ablations

Run all models under the same rolling benchmark:

1. recent raw station values;
2. raw values plus lags;
3. plus seasonal features;
4. plus trends and volatility;
5. plus quality indicators;
6. current candidate;
7. target-specific models;
8. compact ensemble.

An external-information oracle may be run only for diagnosis. It must not enter the production feature set.

### Decision rule

If no valid local-only model beats persistence across independent periods, mark the target/horizon:

```text
INFORMATION_LIMITED
```

Do not increase model capacity indefinitely.

## Phase N10 — Add compact ensemble only if justified

### Candidate members

- residual autoregression;
- ridge regression;
- gradient-boosted trees;
- compact neural temporal model;
- current MF-1-FEATURED;
- residual temperature model;
- vector wind model;
- hurdle precipitation model.

Use 3–5 members or seeds. Fit blend weights on calibration data only. Use out-of-fold predictions for stacking.

### Gate

The ensemble must improve point skill, uncertainty quality, or worst-regime robustness enough to justify additional compute and operational complexity.

## Phase N11 — Strengthen target-specific inference policy

### Files

- `prediction-model/src/inference.py`;
- `prediction-model/src/validate.py`;
- `prediction-model/src/generate_bundles.py`;
- `prediction-model/src/verify_provenance.py`;
- `prediction-model/data/inference_policy.json`;
- bundle manifests.

### Policy requirements

The scorecard, policy, bundle manifest, and live inference output must agree on:

- target source;
- horizon;
- model family;
- fallback;
- calibration artifact;
- policy version;
- commit identity;
- artifact hashes.

### Promotion policy

Keep the current policy unless new evidence passes:

- positive mean skill;
- lower confidence bound above the threshold;
- no critical worst-regime failure;
- physical plausibility;
- calibration gate;
- latency and memory gate;
- rollback gate;
- strict provenance gate.

## Phase N12 — Monitoring and rollback validation

### Monitoring

Add or retain monitoring for:

- feature drift;
- sensor missingness;
- frozen sensors;
- target bias;
- rolling MAE;
- rolling Brier score;
- interval coverage;
- rain-event recall;
- anomaly false alarms;
- fallback rate;
- policy-source usage.

### Response

```text
mild drift -> warning
calibration drift -> recalibration
severe target degradation -> target-specific baseline fallback
provenance or hash failure -> rollback
```

### Gate

Simulate degradation and prove that inference falls back without manual source-code edits.

## Phase N13 — Regenerate and release the next candidate

### Release sequence

1. Freeze feature schema and model configuration.
2. Freeze rolling split definitions.
3. Train all five horizons.
4. Generate candidate checkpoints.
5. Generate calibration artifacts.
6. Generate prediction logs.
7. Generate rolling scorecards.
8. Generate target-specific policy.
9. Generate bundle manifests.
10. Verify every artifact hash independently.
11. Run strict provenance.
12. Run all 53 tests plus new tests.
13. Run smoke and rollback tests.
14. Review policy against scorecard.
15. Commit the complete artifact set.
16. Re-run provenance at the final release identity.
17. Push only if every gate passes.

## Required new tests

Add tests for:

- rolling-origin split disjointness;
- final-test isolation;
- residual-baseline correctness;
- vector wind-direction wrapping;
- calm-wind fallback;
- hurdle precipitation output validity;
- rain threshold calibration;
- heavy-rain metric presence;
- quantile monotonicity p10 <= p50 <= p90;
- interval coverage calculation;
- anomaly event-label integrity;
- target-policy/scorecard equality;
- rollback under simulated drift;
- exact artifact identity after final commit.

## Next-release promotion policy

Until new experiments pass, keep:

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

## Hard stop conditions

Stop and document the limitation if:

- temperature still cannot beat persistence at its target horizons;
- wind direction still cannot beat persistence after vector modeling;
- rain improvement disappears on rolling periods;
- heavy-rain behavior is unacceptable;
- uncertainty intervals are miscalibrated;
- anomaly labels cannot be reviewed;
- model gains require final-test tuning;
- compute exceeds local limits;
- provenance becomes ambiguous;
- the candidate wins only on aggregate but fails a critical regime.

## Definition of done for the next phases

The next phases are complete only when:

1. the current candidate is frozen as a reproducible baseline;
2. three or more rolling test periods are evaluated;
3. strong local baselines are present;
4. label and feature causality audits pass;
5. residual temperature forecasting is implemented and tested;
6. vector wind-direction forecasting is implemented and tested;
7. hurdle precipitation forecasting is implemented and tested;
8. rain event metrics and calibration are reported;
9. continuous-target intervals are calibrated or explicitly blocked;
10. anomaly events have reviewed labels and event metrics;
11. information-limited targets are identified honestly;
12. compact ensembles are accepted only if they earn their complexity;
13. policy, scorecard, bundles, and inference agree;
14. monitoring and rollback pass simulation;
15. all artifacts have strict provenance;
16. all tests pass from a clean checkout;
17. only target/horizon combinations that pass evidence gates are promoted.

## Final implementation recommendation

The next engineering work should begin with **rolling-origin evaluation and baseline expansion**, followed immediately by the **residual temperature model** and **vector wind-direction model**. These are the clearest current weaknesses.

Rain should be improved through calibration and heavy-rain evaluation rather than blindly replacing the current candidate, because its Brier results are already favorable.

The candidate should remain target-specific. If a target cannot beat persistence with correct labels, causal features, target-specific objectives, and rolling evaluation, the repository should mark it information-limited and retain the simpler baseline.
