# Autonomous Predictive-Quality Improvement Prompt: Weather Forecasting and Telemetry Anomaly Detection

You are an autonomous senior machine-learning engineer, time-series scientist, and production-quality software engineer working on:

`https://github.com/benben000000/citizendashboard`

Work from a fresh clone or clean worktree of the latest `origin/main`. Execute the work end to end. Do not stop at a plan. Inspect the real telemetry, implement the improvements, run experiments, add tests, regenerate provenance-correct artifacts, compare against strong baselines, commit, and push only when every release gate passes.

## Primary goal

Improve the **quality of the predictions themselves**, not only the repository workflow.

The final system should produce accurate, calibrated, and operationally honest predictions for:

- temperature;
- relative humidity;
- barometric pressure;
- precipitation amount;
- rain versus no rain;
- wind speed;
- wind direction;
- UV index;
- heat index;
- luminosity/light intensity;
- water level as an internal beta experiment;
- weather anomalies and telemetry anomalies.

The system must distinguish between:

1. Forecasting a future value.
2. Predicting the probability of an event.
3. Detecting that a current or recent observation is abnormal.
4. Detecting that a sensor is malfunctioning or drifting.

Do not treat anomaly detection as another ordinary regression target. Implement it as a separately evaluated task with labels, thresholds, or a clearly documented unsupervised method.

## Non-negotiable scientific rules

1. Use real telemetry only for canonical training and evaluation.
2. Do not use future observations as input features.
3. Preserve chronological train/calibration/test separation.
4. Preserve the existing embargo between splits.
5. Fit normalization and calibration only on the training/calibration partitions.
6. Do not tune on the final test set.
7. Compare every new model against persistence, climatology, seasonal, and appropriate statistical baselines.
8. Report results by horizon, station, variable, and regime where sample sizes permit.
9. Do not claim improvement when only one metric improves and other operational metrics degrade materially.
10. Do not create synthetic labels and present them as measured ground truth.
11. If a target cannot be reliably predicted because the sensor is uncalibrated or coverage is insufficient, mark it as blocked or conditional rather than forcing a model.
12. Keep water-level forecasting explicitly beta and not for life-safety use.
13. Do not modify dashboard UI or unrelated application code.
14. Preserve raw telemetry:
    - `prediction-model/data/weather_telemetry.csv`
    - `prediction-model/data/water_level_telemetry.csv`

## Current repository context

The current pipeline already includes:

- MF-1 GarciaWeatherLNN/PyTorch forecasting;
- MF-2 standalone recurrent forecasting;
- five forecast horizons: 1h, 3h, 6h, 12h, and 24h;
- an eight-feature canonical input contract;
- horizon-specific inference policy;
- persistence fallback;
- hybrid rain probability blending;
- model-policy bundles;
- scorecards and provenance manifests;
- monitoring and drift tooling;
- an experimental two-stage precipitation evaluator;
- canonical contract, inference, provenance, and smoke tests.

Treat the current implementation as the baseline. Do not replace it with an untested architecture merely because it is newer or more complex.

## Phase 0: Establish a real-data baseline

Before changing model code:

1. Fetch `origin/main` and record the starting commit.
2. Verify the archive branch remains available:

   ```text
   cleanup/archive-before-remediation-20260922
   ```

3. Inventory the raw telemetry schema.
4. Compute and record raw dataset SHA-256 hashes.
5. Record:
   - row counts;
   - station counts;
   - time ranges;
   - sampling intervals;
   - duplicate timestamps;
   - missingness by field;
   - out-of-order timestamps;
   - invalid numeric values;
   - physical-bound violations;
   - station coverage over time;
   - rain prevalence;
   - zero-inflation of precipitation;
   - wind calm prevalence;
   - UV daytime/nighttime distributions;
   - luminosity daytime/nighttime distributions;
   - temperature and humidity ranges;
   - heat-index derivability.
6. Confirm which variables are available as real future labels.
7. Confirm whether UV and luminosity are sufficiently calibrated and covered for supervised forecasting.
8. Run the existing canonical pipeline and save the complete baseline scorecard.

Create an auditable data-feasibility report. The report must state, for each target, one of:

```text
FEASIBLE
FEASIBLE_WITH_LIMITATIONS
BLOCKED_BY_SENSOR_QUALITY
BLOCKED_BY_INSUFFICIENT_COVERAGE
DERIVED_TARGET_ONLY
BETA_ONLY
```

## Phase 1: Correct target construction and label quality

Audit and, where necessary, improve the target-window construction in `prediction-model/src/dataset.py`.

For every sample and horizon:

1. Confirm the forecast origin timestamp.
2. Confirm the target timestamp.
3. Confirm the target station.
4. Confirm the requested lead time.
5. Confirm the accepted lead-time tolerance.
6. Reject targets from another station.
7. Reject targets that precede the origin.
8. Reject targets with an invalid or untrusted sensor value.
9. Preserve target provenance metadata.
10. Record how many rows were rejected for each reason.

Required target definitions:

### Temperature

Predict future temperature in Celsius. Report MAE, RMSE, bias, and skill versus persistence and climatology.

### Relative humidity

Predict future relative humidity as a bounded value. Evaluate both raw error and physically valid range violations.

### Pressure

Predict pressure in hPa and pressure tendency. Evaluate continuous pressure error and categorical rising/steady/falling accuracy.

### Wind speed

Predict nonnegative wind speed. Evaluate MAE, RMSE, bias, calm-wind performance, and high-wind performance.

### Wind direction

Do not use ordinary degree MSE. Represent direction using circular components:

```text
u = speed × cos(direction)
v = speed × sin(direction)
```

Evaluate:

- circular angular error;
- mean absolute direction error for noncalm wind;
- calm-wind classification;
- direction accuracy conditioned on wind speed;
- wraparound correctness near 0°/360°.

When wind speed is below the calm threshold, direction should be reported as unavailable or calm rather than scored as an arbitrary angle.

### Rain occurrence

Predict calibrated probability of rain. Define the event threshold explicitly, for example:

```text
actual_precipitation_mm >= 0.1
```

Evaluate:

- Brier score;
- log loss;
- reliability curve;
- expected calibration error;
- precision;
- recall/POD;
- F1;
- false-alarm ratio;
- CSI;
- PR-AUC;
- ROC-AUC only as a secondary metric;
- event metrics at operational thresholds.

### Precipitation amount

Use a two-stage design only if it improves the baseline:

1. Rain occurrence probability.
2. Conditional precipitation amount given rain.

Evaluate dry and rainy subsets separately. Report MAE, RMSE, bias, quantile loss where applicable, and heavy-rain metrics at 2.5, 5.0, and 10.0 mm/h.

Do not use a binary persistence flag as a millimeter amount baseline.

### Heat index

Prefer deriving heat index from the operational temperature and humidity forecasts using the documented NOAA formula rather than training a separate unconstrained head. Evaluate:

- heat-index MAE;
- risk-category accuracy;
- category confusion matrix;
- errors near category boundaries;
- physical consistency with temperature and humidity.

### UV index

First determine whether UV labels are usable. The current audit has indicated possible nighttime calibration defects. Do not train a single unconstrained UV model across corrupted nighttime values.

If daytime UV is sufficiently valid:

1. Restrict supervised UV forecasting to a documented daylight regime.
2. Add local-time and solar-position features only if derived without future leakage.
3. Model nighttime UV as a physically constrained zero/near-zero regime only if the sensor evidence supports it.
4. Report daytime metrics separately from nighttime handling.
5. Report calibration and physical-bound violations.

If the calibration defect remains unresolved, keep UV marked:

```text
BLOCKED_BY_SENSOR_CALIBRATION
```

and do not claim accurate UV prediction.

### Luminosity/light intensity

Determine whether luminosity is:

- a calibrated physical measurement;
- a daylight proxy;
- station-specific sensor output;
- or an unreliable secondary signal.

If usable:

1. Evaluate daylight and nighttime separately.
2. Use station-aware normalization or station embeddings only within training data.
3. Report MAE, RMSE, bias, daytime accuracy, nighttime false-positive rate, and physical consistency.
4. Avoid comparing raw lux values across stations without calibration.

If not usable across stations, support station-specific or daylight-only evaluation and mark the global target as conditional.

### Anomalies

Implement anomaly detection as a dedicated output, not only as forecast residuals.

Support at least three anomaly classes:

1. **Physical anomaly:** a real weather/telemetry event outside expected behavior.
2. **Sensor anomaly:** missing, stuck, impossible, duplicated, discontinuous, or implausible sensor values.
3. **Forecast anomaly:** a material deviation between forecast and observed value after the target becomes available.

Where explicit labels exist, use supervised evaluation. Otherwise implement an unsupervised or semi-supervised method using training-only normal data, such as:

- robust rolling median and MAD;
- seasonal residual thresholds;
- multivariate Mahalanobis distance with robust covariance;
- isolation forest or autoencoder only as a secondary comparison;
- change-point or stuck-sensor detection.

Anomaly output must include:

- anomaly type;
- affected variable;
- station;
- timestamp;
- severity score;
- threshold or model version;
- explanation features;
- whether the anomaly is physical, sensor-related, or forecast-related;
- whether it is actionable.

Do not label every rare weather event as a sensor failure. Keep physical and sensor anomalies separate.

## Phase 2: Improve the feature representation without leakage

Evaluate the following features only when they can be constructed from data available at forecast origin:

- lagged values at multiple intervals;
- rolling means, medians, standard deviations, minima, and maxima;
- first differences and trends;
- rate of change of pressure;
- rain persistence and recent rain duration;
- dry-spell duration;
- wind vector components;
- calm-wind indicator;
- hour-of-day sine/cosine;
- day-of-year sine/cosine;
- local station identity or station embedding;
- station elevation or static metadata if available;
- daylight indicator;
- solar-position features if computed from timestamp and station location without future telemetry;
- missingness indicators;
- sensor-quality indicators;
- recent anomaly flags.

For each feature:

1. Document its source.
2. Document its availability time.
3. Prove it does not use future values.
4. Add a unit test for leakage.
5. Record whether it is shared across stations or station-specific.

Do not add external weather forecast inputs unless they are explicitly authorized, versioned, available at inference time, and included in provenance.

## Phase 3: Evaluate stronger model architectures

Do not assume a larger neural network is automatically better. Compare candidate models under the same split and seed protocol.

At minimum compare:

1. Persistence baseline.
2. Climatology/seasonal baseline.
3. Linear or ridge regression with lagged features.
4. Gradient-boosted trees or another strong tabular baseline.
5. Current MF-1 model.
6. Current MF-2 model.
7. A multi-task sequence model with shared temporal encoding and target-specific heads.
8. A probabilistic model for rain and precipitation amount.

The multi-task model should use:

- shared temporal representation;
- target-specific continuous heads;
- circular wind head;
- rain probability head;
- conditional precipitation head;
- optional daylight-only UV/luminosity heads;
- derived heat-index output;
- uncertainty or quantile heads only when validated.

Use target-specific loss weighting. Compare:

- fixed weights;
- normalized uncertainty weighting;
- gradient-balancing methods.

Do not let a high-volume target such as temperature dominate all other targets.

## Phase 4: Improve probabilistic calibration and uncertainty

For rain probability:

1. Fit calibration only on the calibration split.
2. Compare isotonic calibration, Platt scaling, and beta calibration where sample sizes permit.
3. Freeze calibration parameters before test evaluation.
4. Report pre- and post-calibration Brier score, log loss, reliability, and ECE.
5. Preserve the calibrated model and calibration metadata in the bundle.

For continuous targets, implement uncertainty only after validation:

1. Use quantile regression or conformal residual intervals.
2. Fit on training/calibration data only.
3. Evaluate coverage on untouched test data.
4. Report nominal coverage, observed coverage, interval width, and coverage by station and horizon.
5. Mark intervals unavailable when calibration sample sizes are insufficient.

Do not expose weather confidence intervals merely because a model returns a variance-like value.

## Phase 5: Improve training and validation protocol

Use deterministic training:

- Record every random seed.
- Record dependency versions.
- Record feature schema.
- Record training configuration.
- Record early-stopping criteria.
- Record learning-rate schedule.
- Record model architecture.
- Record calibration procedure.
- Record dataset hashes.

Use chronological splits with embargo. Add rolling-origin backtesting so the model is tested across multiple historical periods rather than one split only.

For each horizon and target, report:

- sample count;
- station count;
- train/calibration/test date ranges;
- missingness and quarantine counts;
- baseline metrics;
- model metrics;
- skill versus persistence;
- skill versus climatology;
- station-level metrics;
- regime-level metrics;
- calibration metrics;
- confidence intervals for metric estimates where practical.

## Phase 6: Define measurable quality gates

A new model may replace the current production/research default only if it passes all applicable gates.

### Continuous targets

Require:

- lower or non-inferior test MAE versus persistence;
- lower or non-inferior RMSE;
- no unacceptable bias;
- no material degradation at any important station;
- physical-bound compliance;
- improvement confirmed across rolling-origin periods.

### Rain occurrence

Require:

- improved or non-inferior Brier score;
- improved or non-inferior calibration error;
- operational F1/recall not materially worse;
- threshold chosen only on calibration data;
- no unacceptable false-alarm increase.

### Wind direction

Require:

- circular-error improvement or non-inferiority;
- correct calm-wind handling;
- no wraparound regression.

### Precipitation amount

Require:

- rainy-hour and overall performance reported separately;
- heavy-rain recall and CSI not materially worse;
- no dry-hour explosion;
- no unsupported uncertainty claims.

### UV and luminosity

Require:

- sensor-feasibility status;
- daylight/nighttime or station-specific metrics;
- physical-bound checks;
- explicit blocked status if calibration is not adequate.

### Anomaly detection

Require, where labels exist:

- precision;
- recall;
- F1;
- false-alarm rate;
- detection delay;
- station-level breakdown.

For unlabeled anomaly detection, report a validation protocol based on injected or independently reviewed cases, clearly labeling the evaluation limitations.

## Phase 7: Improve inference outputs

Update the operational inference response to include, when supported:

- forecast values by target;
- forecast horizon;
- target timestamp;
- calibrated rain probability;
- rain/no-rain alert using the frozen threshold;
- wind direction and calm status;
- heat-index value and category;
- UV status and value only when the target is feasible;
- luminosity status and value only when the target is feasible;
- anomaly status for the input sequence;
- sensor-quality warnings;
- policy version;
- bundle version;
- implementation/artifact/model-weight commits;
- uncertainty status;
- model status.

If a target is blocked, return an explicit status such as:

```json
{
  "uv_index": {
    "status": "BLOCKED_BY_SENSOR_CALIBRATION",
    "value": null
  }
}
```

Do not silently return a model output for a blocked target.

## Phase 8: Add quality-focused tests

Add tests for:

1. No future leakage in every engineered feature.
2. Correct target timestamp and station alignment.
3. Correct circular wind-direction loss and wraparound.
4. Calm-wind direction handling.
5. Rain probability calibration and threshold separation.
6. Two-stage precipitation behavior.
7. Heat-index formula consistency.
8. UV blocked status when the sensor audit fails.
9. Luminosity daylight/nighttime handling.
10. Physical bounds for all outputs.
11. Anomaly-type separation between sensor and physical events.
12. All five forecast horizons.
13. Per-station scorecard presence.
14. Model-policy bundle hash validation.
15. Exact provenance and dataset hash validation.
16. Read-only test behavior.
17. Monitoring schema and five-horizon coverage.

## Phase 9: Generate research artifacts and scorecards

Generate provenance-correct artifacts for every candidate model that is actually evaluated:

- model checkpoint or bundle;
- training manifest;
- feature manifest;
- data-quality report;
- anomaly detector manifest;
- calibration artifact;
- validation scorecard;
- per-station scorecard;
- per-regime scorecard;
- prediction log;
- monitoring report;
- model registry entry.

Every artifact must include:

- implementation commit;
- artifact commit;
- model-weight commit where relevant;
- dataset hashes;
- feature schema;
- target schema;
- horizon;
- training seed;
- configuration;
- model status;
- uncertainty status;
- limitations.

## Phase 10: CI and release gates

Update the prediction-model CI workflow to run:

```bash
python -m py_compile prediction-model/src/*.py
python prediction-model/src/test_canonical_contract.py
python prediction-model/src/test_inference_contract.py
python prediction-model/src/test_monitoring.py
python prediction-model/src/test_predictive_quality.py
python prediction-model/src/test_provenance.py
python prediction-model/src/smoke_test.py
python prediction-model/src/generate_manifests.py --check-only
python prediction-model/src/monitoring.py --require-all-horizons
python prediction-model/src/verify_provenance.py
git diff --check
git diff --exit-code prediction-model/data/
```

The CI must fail when:

- a target is trained on unusable labels;
- a supported horizon lacks metrics;
- any required target silently falls back to zeros;
- future leakage is detected;
- rain calibration is missing;
- UV is scored despite a blocked calibration status;
- anomaly output lacks type/severity/provenance;
- model outputs violate physical bounds;
- per-station metrics are absent where sample size is sufficient;
- tracked artifacts change during read-only tests;
- provenance or bundle hashes fail.

## Required final report

Report:

- starting commit;
- final commit;
- exact push result or blocker;
- archive reference;
- dataset hashes;
- quarantine counts;
- target-feasibility decisions;
- feature schema and leakage checks;
- training seeds and configurations;
- baseline and candidate models;
- complete metrics for all five horizons;
- metrics for temperature, humidity, pressure, precipitation amount, rain probability, wind speed, wind direction, heat index, UV, luminosity, and anomalies;
- per-station and per-regime metrics;
- calibration metrics;
- uncertainty coverage and interval widths where available;
- blocked targets and reasons;
- model-selection decision;
- artifact and bundle provenance;
- tests and CI gates;
- remaining limitations;
- operational GO/NO-GO recommendation.

## Release decision

Recommend **GO for research evaluation** only when:

- the new model is evaluated against strong baselines;
- no target uses leaked or synthetic labels;
- all claims are supported by test metrics;
- blocked sensor targets remain explicitly blocked;
- anomaly detection is separated from forecasting;
- scorecards are complete and provenance-correct.

Recommend **GO for operational deployment** only when:

- the model improves or safely matches the baseline across the required metrics;
- calibration is validated;
- station-level failures are understood;
- inference outputs include status and uncertainty metadata;
- monitoring is schema-correct and covers all horizons;
- model-policy bundles validate by hash;
- CI passes from a clean checkout;
- the final working tree is clean;
- the push succeeds.

Otherwise report **NO-GO** and list the failed quality gates. Do not hide a weak target behind an aggregate score.

Do not stop at a plan. Execute the complete predictive-quality improvement cycle through real-data analysis, model development, evaluation, artifact generation, testing, commit, provenance verification, and final release decision.

## References

[1]: https://github.com/benben000000/citizendashboard "Citizen Dashboard repository"
[2]: https://github.com/benben000000/citizendashboard/tree/main/prediction-model "Citizen Dashboard prediction-model directory"
