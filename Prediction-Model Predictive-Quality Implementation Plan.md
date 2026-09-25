# Prediction-Model Predictive-Quality Implementation Plan

## Executive objective

The project must move from a **validated research pipeline with policy-based inference** to a **measurably more accurate multi-horizon weather and telemetry forecasting system**.

The primary success condition is not the number of tests, features, or neural-network layers. The system is successful only when a newly trained model demonstrates better or safely equivalent performance than the current eight-feature baseline on untouched real telemetry, across the required forecast horizons and important stations.

The required forecast horizons are:

```text
1 hour, 3 hours, 6 hours, 12 hours, and 24 hours
```

The main target variables are:

| Target | Required outcome | Current constraint |
|---|---|---|
| Temperature | Accurate continuous forecast | Fully feasible |
| Relative humidity | Accurate bounded continuous forecast | Fully feasible |
| Pressure | Accurate value and rising/steady/falling tendency | Fully feasible |
| Precipitation amount | Accurate zero-inflated amount forecast | Requires two-stage evaluation |
| Rain occurrence | Calibrated probability and rain/no-rain decision | Fully feasible with threshold trade-offs |
| Wind speed | Accurate nonnegative forecast | Must evaluate calm and high-wind regimes |
| Wind direction | Accurate circular forecast | Direction is undefined during calm wind |
| Heat index | Physically consistent derived forecast | Should be derived from selected temperature/humidity |
| UV index | Accurate daylight forecast only if calibration is valid | Currently blocked by sensor calibration |
| Luminosity | Accurate daylight/nighttime forecast if sensor scale is usable | Conditional and station-dependent |
| Water level | Internal beta research forecast | Not for life-safety use |
| Anomalies | Separate sensor, physical-weather, and forecast-residual detection | Requires labels or a defensible review protocol |

## What must be achieved

The implementation must achieve the following concrete outcomes:

1. **Feature-augmented training is real, not only tested.** The leakage-safe engineered features currently defined in `dataset.py` must be integrated into a model-training path and used to train new checkpoints.
2. **The current model remains a protected baseline.** No new model replaces it until the new model is evaluated against it on untouched data.
3. **Every target receives an appropriate model and metric.** Continuous variables, circular direction, binary rain occurrence, precipitation amount, derived heat index, and anomalies must not be evaluated with one generic metric.
4. **Accuracy is measured per horizon and station.** Aggregate results alone are insufficient.
5. **Rain probabilities are calibrated.** A high F1 score without reliable probability quality is not enough for operational use.
6. **Anomaly detection has an evaluation protocol.** The system must report false alarms, missed events, detection delay, and station differences where labels or reviewed cases exist.
7. **UV and luminosity are handled honestly.** The system must either produce validated forecasts under a restricted regime or remain explicitly blocked/conditional.
8. **The final inference API serves only the model and policy that passed the scorecard.** Bundle hashes and provenance must match.
9. **The model must improve real test performance, not merely pass structural tests.**

## Current baseline to preserve

The current repository contains a functioning baseline that must be frozen before experimentation:

- Canonical eight-feature sequence input.
- MF-1 GarciaWeatherLNN/PyTorch model.
- MF-2 standalone recurrent model.
- Horizon-specific policy and persistence fallback.
- Hybrid rain probability blending.
- Five-horizon scorecards.
- Corrected five-horizon monitoring.
- Model-policy bundles.
- Provenance verification.
- Anomaly detector implementation.
- Zero-leakage feature extraction tests.

Create an immutable baseline record containing:

- baseline commit;
- raw weather SHA-256;
- raw water SHA-256;
- feature schema;
- training seed;
- model configuration;
- split dates;
- embargo duration;
- checkpoint hashes;
- all five-horizon metrics;
- station counts;
- quarantine counts;
- calibration metrics;
- limitations.

The baseline record must be generated before training any candidate model.

# Workstream 1: Establish target and data quality

## Goal

Create trustworthy labels and a target-feasibility matrix so the candidate model is not optimized against corrupted or misaligned data.

## Implementation

Audit and update the canonical dataset construction in:

```text
prediction-model/src/dataset.py
prediction-model/src/audit_data_availability.py
```

For every sample, validate:

- station identity;
- forecast-origin timestamp;
- target timestamp;
- requested horizon;
- actual lead time;
- station-local continuity;
- missingness;
- physical bounds;
- duplicate timestamps;
- future leakage;
- target provenance.

Create a target-quality table with:

| Field | Required result |
|---|---|
| `target_name` | Exact target identifier |
| `feasibility_status` | Feasible, conditional, blocked, derived, or beta |
| `valid_sample_count` | Count after quarantine |
| `quarantine_count` | Count and reasons |
| `station_count` | Number of contributing stations |
| `date_range` | Earliest and latest usable target |
| `unit` | Canonical unit |
| `label_rule` | Exact event or numeric definition |
| `known_limitations` | Calibration and coverage limitations |

## Completion criteria

This workstream is complete when:

- Every target has an explicit feasibility status.
- No target contains future observations in its input sequence.
- Target timestamps and stations are verified.
- UV remains blocked unless the calibration defect is resolved.
- Water level remains beta-only.
- The report is reproducible from the raw telemetry hashes.

# Workstream 2: Integrate leakage-safe engineered features

## Goal

Use the existing zero-leakage features in actual model training and inference, while preserving compatibility with the eight-feature baseline.

## Candidate feature groups

Features must be computed only from observations available at forecast origin:

- current values at `t0`;
- lagged temperature, humidity, pressure, wind speed, and precipitation;
- rolling means, standard deviations, minima, and maxima;
- pressure changes over 1-hour and 3-hour windows;
- pressure tendency category;
- rain-persistence duration;
- dry-spell duration;
- wind-vector components;
- calm-wind indicator;
- hour-of-day sine/cosine;
- day-of-year sine/cosine;
- daylight indicator;
- station identity or station embedding if validated;
- missingness and sensor-quality indicators;
- recent anomaly indicators only when computed from past data.

## Required architecture decision

Do not silently change the canonical eight-feature model. Implement a separate feature-augmented model family, for example:

```text
MF-1-FEATURED
MF-2-FEATURED
```

The baseline and candidate must be trainable and evaluable through the same interface.

The feature-augmented model must declare:

- exact feature schema;
- feature order;
- feature units;
- normalization parameters;
- feature availability time;
- model input dimension;
- seed;
- training configuration.

## Training and inference requirements

1. Add a feature builder that accepts an observed history and forecast-origin timestamp.
2. Add a dataset path that returns the engineered features without future values.
3. Add a candidate model input contract.
4. Add candidate inference support behind an explicit model-family or bundle selection.
5. Keep the baseline model available for rollback.
6. Store the feature schema and normalization inside the candidate bundle.
7. Add a test that corrupting future records does not change features at `t0`.
8. Add a test that candidate inference rejects missing or misordered features.

## Completion criteria

This workstream is complete only when:

- At least one candidate model is trained using the engineered features.
- Candidate checkpoints are generated for all five horizons.
- Candidate inference uses the same feature builder as candidate training.
- Feature schemas and hashes are present in manifests.
- Baseline and candidate predictions are both available on the same test rows.

# Workstream 3: Build appropriate target heads

## Goal

Use target-specific modeling rather than forcing every target through the same output and loss design.

## Required target designs

### Continuous variables

Temperature, humidity, pressure, and wind speed should use target-specific continuous heads. Apply physical constraints where appropriate:

- humidity should remain within a documented valid range;
- wind speed must be nonnegative;
- pressure should remain within the learned and physical operating range.

Report raw and constrained outputs separately during evaluation so clipping cannot hide model failure.

### Wind direction

Represent wind direction through circular components or a von Mises-style probabilistic head. Evaluate angular error modulo 360 degrees.

For calm wind:

- classify calm first;
- do not score an arbitrary direction;
- report direction accuracy only for noncalm samples.

### Rain occurrence

Use a probability head trained with a binary objective. Evaluate calibration separately from event classification.

The operational threshold must be selected using calibration data only.

### Precipitation amount

Use a two-stage architecture:

1. Rain occurrence probability.
2. Conditional amount given rain.

Compare at least:

- current single-head amount model;
- log-transformed amount regression;
- conditional rainy-sample regression;
- Tweedie or Gamma-style objective if stable in the environment.

Do not promote the two-stage model unless it improves rainy-hour and heavy-rain performance without materially degrading dry-hour behavior.

### Heat index

Compute heat index from the operational temperature and humidity outputs. Do not train an independent head unless there is a demonstrated reason and a consistency test.

### UV index

If calibration remains invalid, do not train or score a global UV model. If daytime-only data becomes reliable:

- train and evaluate only for the valid daylight regime;
- include daylight status in the target contract;
- report daytime performance separately;
- keep nighttime behavior physically constrained and documented.

### Luminosity

Evaluate luminosity by station and daylight regime. If cross-station scales are not comparable, use station-specific normalization or mark the global target conditional.

## Completion criteria

Each target has:

- an appropriate output head;
- a documented loss;
- a documented metric set;
- physical constraints;
- a target-specific scorecard section;
- a promotion or blocked decision.

# Workstream 4: Train candidate models correctly

## Goal

Train baseline and feature-augmented candidates under identical, reproducible conditions.

## Required candidates

At minimum train and evaluate:

1. Persistence.
2. Climatology or seasonal baseline.
3. Ridge or linear regression with engineered features.
4. Gradient-boosted tree baseline where appropriate.
5. Current eight-feature MF-1.
6. Current eight-feature MF-2.
7. Feature-augmented MF-1.
8. Feature-augmented MF-2.
9. Two-stage precipitation candidate.
10. Calibrated rain-probability variants.

The tree and linear baselines are important. A neural model should not be called improved if a simpler model performs better.

## Reproducibility configuration

For every candidate, record:

- random seed;
- Python version;
- package versions;
- model family;
- input schema;
- input dimension;
- hidden dimensions;
- number of layers;
- optimizer;
- learning rate;
- batch size;
- loss weights;
- epochs;
- early-stopping patience;
- checkpoint selection rule;
- train/calibration/test dates;
- embargo duration;
- dataset hashes.

## Completion criteria

Training is complete when every candidate has:

- a checkpoint or serialized model;
- a manifest;
- training history;
- calibration artifact if applicable;
- predictions for every test horizon;
- no synthetic labels;
- no empty validation split;
- a reproducible command.

# Workstream 5: Evaluate forecast accuracy

## Goal

Determine whether the candidate actually improves real forecasting quality.

## Required evaluation dimensions

Evaluate every candidate by:

- horizon;
- station;
- target;
- rain/dry regime;
- daylight/nighttime regime;
- calm/noncalm wind regime;
- heavy-rain regime;
- data-quality status;
- rolling-origin period.

## Metrics

### Temperature, humidity, pressure, wind speed

Report:

- MAE;
- RMSE;
- mean bias;
- median absolute error;
- 95% bootstrap confidence interval;
- persistence skill;
- climatology skill;
- physical-bound violations.

### Pressure tendency

Report:

- rising/steady/falling accuracy;
- macro-F1;
- confusion matrix;
- class support;
- performance by horizon.

### Wind direction

Report:

- circular MAE;
- median angular error;
- percentage within 10°, 22.5°, 45°, and 90°;
- noncalm sample count;
- calm classification metrics.

### Rain occurrence

Report:

- Brier score;
- log loss;
- expected calibration error;
- reliability bins;
- precision;
- recall/POD;
- F1;
- CSI;
- false-alarm ratio;
- PR-AUC;
- selected threshold;
- threshold selection dataset.

### Precipitation amount

Report:

- overall MAE/RMSE/bias;
- dry-hour MAE/RMSE/bias;
- rainy-hour MAE/RMSE/bias;
- 2.5, 5.0, and 10.0 mm/h event precision, recall, and CSI;
- persistence amount baseline;
- climatology amount baseline;
- amount skill versus each baseline.

If persistence amount is not present in the prediction log, add it correctly or mark the comparison unavailable. Never use a binary rain flag as millimeters.

### Heat index

Report:

- MAE;
- RMSE;
- category accuracy;
- category macro-F1;
- boundary-error rate;
- consistency with temperature and humidity.

### UV and luminosity

Report only within the validated feasible regime. Include:

- sample counts;
- day/night or daylight classification;
- station breakdown;
- MAE/RMSE/bias;
- calibration or bound violations;
- blocked/conditional status.

### Anomaly detection

Separate the metrics for:

- sensor anomalies;
- physical weather anomalies;
- forecast-residual anomalies.

Report:

- precision;
- recall;
- F1;
- false-positive rate;
- detection delay;
- station breakdown;
- severity calibration;
- reviewed-label coverage.

If no independent labels exist, create a reviewed evaluation set or clearly mark anomaly accuracy as unvalidated. Synthetic injections may test detector mechanics but cannot be treated as real-world accuracy.

# Workstream 6: Calibrate and select the operational policy

## Goal

Select the candidate and policy using calibration data without contaminating the final test set.

For each horizon:

1. Select model versus persistence per target using calibration data.
2. Select rain blend weights using calibration data.
3. Select rain thresholds using the documented operational objective.
4. Fit probability calibration using calibration data.
5. Freeze all choices before test evaluation.
6. Evaluate once on untouched test data.

The policy must state:

- model family;
- target source selection;
- rain blend weights;
- rain threshold;
- calibration method;
- calibration sample count;
- feature schema;
- bundle version;
- provenance.

# Workstream 7: Improve anomaly detection quality

## Goal

Make anomaly outputs useful and distinguish sensor faults from genuine weather events.

Implement and validate three independent pathways:

### Sensor anomalies

Detect:

- missing values;
- NaN/Inf;
- impossible bounds;
- stuck values;
- repeated timestamps;
- abrupt discontinuities;
- invalid units;
- cross-field contradictions.

### Physical weather anomalies

Detect unusual but potentially real events using:

- robust seasonal residuals;
- station-specific baselines;
- multivariate robust distance;
- change-point signals;
- rainfall and pressure event logic.

### Forecast residual anomalies

After the target becomes available, compute residuals against the forecast and flag material deviations using calibration-derived thresholds.

Every anomaly record must include:

- anomaly type;
- affected variable;
- station;
- timestamp;
- severity;
- threshold version;
- evidence;
- actionability;
- model or detector version.

# Workstream 8: Promote only demonstrably better models

## Candidate promotion rules

A candidate may replace the baseline only if all applicable conditions hold:

1. It improves or safely matches aggregate performance.
2. It improves or safely matches performance at each important station.
3. It does not materially degrade any high-priority horizon.
4. It improves or safely matches rain calibration.
5. It does not increase false alarms beyond the agreed operational tolerance.
6. It handles physical bounds correctly.
7. It improves wind-direction circular error or remains non-inferior.
8. It improves rainy-hour and heavy-rain precipitation performance or is rejected.
9. Its anomaly detector has validated or explicitly limited accuracy claims.
10. Its input schema, artifacts, and policy are reproducible.

A candidate must be rejected when:

- it only improves training loss;
- it improves aggregate metrics while failing a key station;
- it improves Brier score but harms operational alert performance beyond tolerance;
- it uses invalid UV labels;
- it uses synthetic anomaly labels as if they were real labels;
- it depends on unavailable features at inference time;
- it has stale or ambiguous provenance.

## Suggested starting promotion thresholds

These are initial engineering gates, not universal scientific truths. They must be reviewed against sample sizes and operational costs:

| Area | Initial gate |
|---|---|
| Temperature/humidity/pressure/wind speed | Candidate MAE no worse than baseline and at least 3% better on the aggregate test set, with no important station worse by more than 10% |
| Rain Brier score | At least 3% relative improvement or statistically non-inferior result |
| Rain calibration error | No degradation; preferably at least 5% relative improvement |
| Rain recall/F1 | No more than 2 percentage-point degradation at the operational threshold |
| Wind circular error | At least 3% improvement or non-inferior result |
| Precipitation rainy-hour MAE | Improvement required before promotion of the two-stage amount head |
| Heavy-rain CSI | No degradation at 2.5, 5.0, and 10.0 mm/h thresholds |
| Physical-bound violations | Zero unexplained violations |
| Anomaly detector | Validated metrics required before operational claims |

Use confidence intervals or paired bootstrap tests where practical. Do not treat tiny metric differences as meaningful without uncertainty analysis.

# Workstream 9: Update inference and bundles

## Goal

Serve the promoted model consistently and retain rollback capability.

Create separate bundles for baseline and candidate models, each containing:

- checkpoint;
- model manifest;
- feature schema;
- normalization;
- inference policy;
- calibration artifact;
- anomaly detector configuration;
- dataset hashes;
- implementation commit;
- artifact commit;
- model-weight commit;
- bundle hash;
- limitations;
- model status.

Inference must:

- select only a validated bundle;
- reject schema mismatch;
- reject hash mismatch;
- reject unsupported horizon;
- expose model and policy version;
- expose anomaly results;
- expose uncertainty status;
- expose blocked target statuses;
- preserve the baseline bundle for rollback.

# Workstream 10: Validation, CI, and release

Run from a clean checkout:

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

Add a dedicated candidate-training and evaluation command that:

1. Trains all candidate families for all five horizons.
2. Generates predictions on identical test rows.
3. Generates the complete scorecard.
4. Generates the policy only from calibration data.
5. Generates bundles only after evaluation passes.
6. Leaves the repository clean during read-only checks.

## Final deliverables

The completed implementation must produce:

- baseline scorecard;
- candidate scorecard;
- per-horizon metrics;
- per-station metrics;
- per-regime metrics;
- anomaly scorecard;
- target-feasibility report;
- calibration report;
- feature manifest;
- model manifests;
- model-policy bundles;
- prediction logs;
- monitoring report;
- reproducibility commands;
- updated model registry;
- updated README;
- final provenance verification output.

## Definition of done

The project is done only when all of the following are true:

1. A feature-augmented candidate has been trained on real data.
2. The candidate was evaluated against the protected baseline and strong non-neural baselines.
3. Scores exist for all five horizons.
4. Scores exist for all feasible targets.
5. Station-level and regime-level performance has been reviewed.
6. Rain probability calibration is reported.
7. Wind direction is evaluated circularly.
8. Heat index is physically consistent.
9. UV is either validated in a restricted regime or remains blocked.
10. Luminosity is either validated by regime/station or remains conditional.
11. Precipitation amount includes valid amount baselines.
12. Anomaly detection has a labeled or explicitly limited evaluation protocol.
13. No target uses future leakage or synthetic labels as canonical truth.
14. The promoted bundle matches the policy and provenance hashes.
15. Inference serves the promoted candidate and supports rollback.
16. All tests and CI gates pass.
17. The final worktree is clean.
18. The final commit is pushed successfully.

## Final decision format

The final report must state one of:

```text
GO — candidate model promoted for research or operational use
CONDITIONAL GO — candidate retained for research only; specific gates remain open
NO-GO — candidate rejected or release blocked
```

The report must include:

- starting commit and final commit;
- exact push result or blocker;
- archive reference;
- baseline and candidate model names;
- all training seeds and configurations;
- dataset hashes;
- quarantine counts;
- full metrics for every horizon and feasible target;
- station-level worst cases;
- calibration results;
- anomaly-detection validation results;
- UV/luminosity feasibility decisions;
- precipitation baseline results;
- bundle and policy provenance;
- tests and CI results;
- unresolved limitations;
- the reason for the final GO, CONDITIONAL GO, or NO-GO decision.

Do not report success merely because the tests pass. The decisive evidence must be a real-data, untouched-test comparison showing whether prediction quality improved.
