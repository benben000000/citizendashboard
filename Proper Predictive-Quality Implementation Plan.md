# Proper Predictive-Quality Implementation Plan

## 1. Current repository state

The latest GitHub `main` commit is:

```text
b9812096e1e72009ea9d696a5f3d7936748b8df6
```

The latest implementation includes:

- corrected five-horizon monitoring;
- zero-leakage feature extraction;
- a 75-feature augmented context vector;
- a feature-augmented neural candidate model;
- ridge and climatology baselines;
- rain calibration and threshold selection on validation data;
- dedicated anomaly detection;
- 21 predictive-quality tests;
- a 15-gate CI workflow.

The following checks pass on a clean detached worktree:

| Check | Result |
|---|---:|
| Python compilation | PASS |
| Canonical contract tests | PASS — 15 tests |
| Inference tests | PASS — 11 tests |
| Monitoring tests | PASS — 10 tests |
| Predictive-quality tests | PASS — 21 tests |
| Provenance tests | PASS — 14 tests |
| Smoke test | PASS |
| Provenance verifier | PASS |

The new real-data trainer also executes across all five horizons and 15 stations. A one-epoch trial produced the following result:

| Horizon | Candidate temperature MAE | Persistence temperature MAE | Candidate rain Brier | Persistence rain Brier | Candidate wind-direction error | Persistence wind-direction error |
|---:|---:|---:|---:|---:|---:|---:|
| 1h | 0.5166°C | 0.4926°C | 0.2321 | 0.1713 | 63.10° | 39.42° |
| 3h | 1.0675°C | 0.9554°C | 0.2354 | 0.2270 | 82.38° | 46.95° |
| 6h | 1.4219°C | 1.4413°C | 0.2365 | 0.2695 | 72.61° | 53.20° |
| 12h | 1.6717°C | 1.8301°C | 0.2434 | 0.3000 | 84.00° | 57.05° |
| 24h | 1.2058°C | 1.2104°C | 0.2327 | 0.3328 | 67.30° | 54.33° |

The trial correctly returned:

```text
CONDITIONAL GO — candidate retained for research; operational gates pending
```

The first three promotion gates failed because the candidate was worse than persistence at 1h for temperature, rain Brier score, and wind-direction error. This is not a repository failure. It is evidence that the candidate model needs further modeling work before promotion.

A second important gap is artifact lifecycle. The new trainer currently writes:

```text
predictive_quality_scorecard.json
model_comparison_report.json
```

but does not save reusable candidate checkpoints, calibration artifacts, prediction logs, or model-policy bundles. Therefore the candidate cannot yet be promoted through production inference even if its metrics improve.

## 2. Final goal

The project goal is to produce an accurate, calibrated, reproducible, and operationally honest forecasting system for:

- temperature;
- relative humidity;
- pressure;
- precipitation amount;
- rain probability and rain/no-rain decisions;
- wind speed;
- wind direction;
- heat index;
- UV index where calibration permits;
- luminosity where calibration and station comparability permit;
- water level as an internal beta target;
- sensor, physical-weather, and forecast-residual anomalies.

The required horizons are:

```text
1h, 3h, 6h, 12h, 24h
```

The success criterion is a real-data improvement on untouched test data, not merely passing structural tests or reducing training loss.

## 3. Promotion standard

The existing eight-feature MF-1 and MF-2 pipeline remains the protected baseline. The new feature-augmented candidate must be compared against:

1. persistence;
2. climatology or seasonal mean;
3. ridge regression using the 75 engineered features;
4. gradient-boosted trees or another strong tabular baseline;
5. current eight-feature MF-1;
6. current eight-feature MF-2;
7. feature-augmented neural MF-1;
8. feature-augmented neural MF-2;
9. two-stage precipitation candidate;
10. calibrated rain-probability variants.

A candidate may not replace the baseline until the complete scorecard shows that it improves or safely matches the baseline across the required targets and horizons.

## 4. Workstream A — Freeze the baseline

### Goal

Create a permanent, reproducible reference before changing training or policy.

### Implementation

Create a baseline manifest containing:

- baseline commit;
- weather telemetry SHA-256;
- water telemetry SHA-256;
- exact eight-feature schema;
- training seeds;
- model configuration;
- train, validation, calibration, and test date ranges;
- embargo duration;
- checkpoint hashes;
- all five-horizon predictions;
- all baseline metrics;
- station-level metrics;
- rain calibration metrics;
- quarantine counts;
- known limitations.

The baseline must be generated from the clean checkout at `b9812096` or its final parent after the implementation changes are committed.

### Acceptance criteria

This workstream passes only when the baseline can be regenerated without modifying tracked artifacts during read-only checks and the baseline scorecard contains all five horizons.

## 5. Workstream B — Make the feature-augmented path production-complete

### Goal

Turn the 75-feature candidate from an in-memory experiment into a reproducible model family that can be trained, evaluated, bundled, and served.

### Implementation

Add a distinct candidate family, for example:

```text
MF-1-FEATURED
MF-2-FEATURED
```

Do not silently change the input dimension of the protected eight-feature model.

The candidate bundle must contain:

- candidate checkpoint;
- exact 75-feature schema and order;
- feature units;
- normalization means and standard deviations;
- input dimension;
- model configuration;
- target schema;
- seed;
- calibration artifact;
- anomaly-detector configuration;
- dataset hashes;
- implementation commit;
- artifact commit;
- model-weight commit;
- checkpoint hash;
- policy hash;
- bundle version;
- model status;
- limitations.

### Required code changes

Update:

```text
prediction-model/src/train_predictive_quality.py
prediction-model/src/model.py
prediction-model/src/inference.py
prediction-model/src/dataset.py
```

The trainer must save at least:

```text
candidate_h{horizon}h.pt
candidate_h{horizon}h_manifest.json
candidate_h{horizon}h_calibration.json
candidate_h{horizon}h_predictions.csv
```

The trainer must not report a candidate as deployable unless those files exist and their hashes are recorded.

### Acceptance criteria

Run a one-epoch execution first. Then run the configured training schedule. Confirm that all five candidate checkpoints load successfully in a fresh Python process and that inference uses the same feature builder used during training.

## 6. Workstream C — Fix the candidate training objective

### Goal

Improve the candidate where the first trial is weak instead of promoting the current result.

### Current evidence

The one-epoch trial shows:

- 1h temperature is worse than persistence;
- rain Brier is worse than persistence at 1h, 3h, and 6h;
- wind-direction error is worse than persistence at every tested horizon;
- temperature is better than persistence at 6h, 12h, and 24h;
- rainy precipitation amount is better than persistence at 3h, 6h, 12h, and 24h;
- physical-bound violations are zero.

This means the candidate is not uniformly bad, but it is not ready for promotion.

### Required changes

#### Temperature, humidity, pressure, and wind speed

Use target-specific losses with normalized target scales. Prevent high-volume targets from dominating the shared representation. Compare fixed loss weights with learned or uncertainty-based weights, but keep the chosen method deterministic.

#### Wind direction

Do not train direction as ordinary degree regression. Predict wind vector components and evaluate only noncalm samples for angular error. Add a separate calm-wind classification or gate. Compare against a persistence vector baseline.

Investigate why the current candidate produces 63° error at 1h versus 39° for persistence. Do not promote until this is corrected or the candidate is replaced by a safer per-target policy.

#### Rain occurrence

Separate probability quality from thresholded classification. Use validation-only calibration. Compare:

- raw candidate probability;
- temperature-scaled or Platt-calibrated probability;
- isotonic calibration when calibration sample support is sufficient;
- hybrid candidate/persistence probability;
- ridge probability;
- persistence probability.

The rain threshold must be selected on validation data only. Report Brier score, log loss, ECE, reliability bins, recall, precision, F1, CSI, and false-alarm rate.

#### Precipitation amount

Keep occurrence and amount separate. Compare single-head and two-stage candidates. Evaluate dry hours, rainy hours, and heavy-rain thresholds separately. Add a real millimeter persistence amount baseline. Never use a binary rain flag as a millimeter amount.

#### Heat index

Derive heat index from the selected temperature and humidity forecasts. Test formula consistency and category accuracy. Do not promote an unconstrained heat-index head if the derived output is more reliable.

## 7. Workstream D — Add the missing strong baseline

### Goal

Avoid assuming the neural candidate is best.

Implement a gradient-boosted tree baseline or an equivalent strong tabular baseline using the same 75-feature matrix. It must use identical chronological splits and target definitions.

For each horizon, compare:

- persistence;
- climatology;
- ridge;
- tree model;
- current MF-1/MF-2;
- feature-augmented neural model.

### Acceptance criteria

A neural candidate is not promoted if a simpler baseline performs better on the target and horizon being considered, unless the neural model provides a separate validated operational advantage such as calibration, multi-target consistency, or lower maintenance cost.

## 8. Workstream E — Run statistically valid training and evaluation

### Goal

Replace the one-epoch trial with a reproducible final experiment.

### Required schedule

Run:

1. one-epoch pipeline smoke test;
2. short hyperparameter sweep on validation data;
3. selected training schedule with early stopping;
4. rolling-origin evaluation;
5. untouched test evaluation exactly once for final selection.

The experiment must record:

- seed;
- epochs;
- learning rate;
- batch size;
- optimizer;
- early-stopping rule;
- loss weights;
- feature schema;
- normalization;
- split dates;
- embargo;
- station counts;
- dataset hashes;
- candidate checkpoint hash.

Do not use the final test set to select epochs, thresholds, blend weights, calibration methods, or features.

## 9. Workstream F — Build the complete scorecard

### Goal

Make promotion decisions from complete evidence rather than a few headline metrics.

For every target and horizon, report:

- sample count;
- station count;
- train/calibration/test date ranges;
- quarantine count;
- MAE;
- RMSE;
- bias;
- persistence skill;
- climatology skill;
- confidence interval;
- physical-bound violations;
- station-level worst case;
- regime-level metrics.

### Target metrics

#### Temperature, humidity, pressure, and wind speed

Report MAE, RMSE, bias, median absolute error, confidence interval, persistence skill, and physical violations.

#### Wind direction

Report circular MAE, median angular error, percentage within 10°, 22.5°, 45°, and 90°, noncalm sample count, and calm classification metrics.

#### Rain occurrence

Report Brier score, log loss, ECE, reliability bins, precision, recall, F1, CSI, false-alarm ratio, PR-AUC, threshold, and calibration method.

#### Precipitation amount

Report overall, dry-hour, rainy-hour, and heavy-rain metrics. Include a true amount persistence baseline or explicitly mark it unavailable.

#### Heat index

Report MAE, RMSE, category accuracy, macro-F1, boundary errors, and consistency with temperature/humidity.

#### UV

Keep the target blocked while calibration is invalid. If calibration is repaired, evaluate daylight separately and do not infer nighttime skill from invalid labels.

#### Luminosity

Evaluate by station and daylight regime. Keep the target conditional if sensor scales are not comparable.

#### Anomalies

Separate sensor anomalies, physical weather anomalies, and forecast residual anomalies. Report precision, recall, F1, false-positive rate, detection delay, and reviewed-label coverage. If independent labels are unavailable, report anomaly accuracy as unvalidated.

## 10. Workstream G — Define promotion gates that match the actual goal

The candidate must pass all applicable gates before promotion.

| Gate | Requirement |
|---|---|
| Continuous target accuracy | Candidate MAE is at least 3% better than persistence or statistically non-inferior, with no important station worse by more than 10% |
| Rain probability | Brier and log loss improve or are non-inferior; calibration error does not materially worsen |
| Rain decisions | Recall, F1, CSI, and false-alarm rate remain within agreed operational tolerance |
| Wind direction | Circular error improves or is non-inferior to persistence, with correct calm handling |
| Precipitation amount | Rainy-hour and heavy-rain performance do not degrade |
| Heat index | Derived result remains physically consistent and improves or matches the baseline |
| UV | No operational promotion while calibration is blocked |
| Luminosity | No global promotion without station/daylight validation |
| Anomalies | Real-label or reviewed-case evaluation exists before operational claims |
| Physical validity | Zero unexplained output-bound violations |
| Reproducibility | Candidate checkpoints, calibration, bundles, and manifests reload successfully |
| Provenance | All hashes and commits validate |

The first trial fails the temperature, rain Brier, and wind-direction gates. Those failures must remain visible in the next scorecard.

## 11. Workstream H — Integrate policy and inference only after promotion

### Goal

Keep the baseline as the rollback model and expose the candidate only when its scorecard passes.

Add policy fields for:

- model family;
- target-specific source;
- rain blend weights;
- rain threshold;
- calibration method;
- feature schema;
- candidate bundle version;
- anomaly detector version;
- blocked-target statuses;
- uncertainty status.

Inference must reject:

- wrong feature dimension;
- wrong feature order;
- missing normalization;
- wrong checkpoint hash;
- wrong policy hash;
- unsupported horizon;
- UV requests when the target is blocked.

The baseline bundle must remain available for rollback.

## 12. Workstream I — CI and release execution

Add the candidate experiment to CI only as a bounded smoke or validation job. Do not make CI retrain a long full model unless runtime and artifact handling are explicitly controlled.

The release workflow must run:

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

The full candidate experiment must be run from a controlled training command and must produce all five checkpoints, manifests, predictions, calibration artifacts, and scorecards before any promotion commit.

## 13. Exact execution order

### Step 1 — Baseline freeze

Record commit, raw hashes, baseline metrics, split configuration, and artifact hashes.

### Step 2 — Trainer completion

Add checkpoint, manifest, calibration, and prediction-log outputs to `train_predictive_quality.py`.

### Step 3 — Candidate model correction

Fix the candidate loss design and wind-direction handling. Improve rain probability calibration. Add the strong tree baseline.

### Step 4 — Short experiment

Run one epoch to confirm the path. Run a short validation-only sweep. Do not inspect final test results for model selection during the sweep.

### Step 5 — Selected training

Train the selected candidate for all five horizons with the fixed seed and early stopping.

### Step 6 — Complete evaluation

Generate predictions, scorecards, station/regime breakdowns, calibration reports, and anomaly reports on untouched test data.

### Step 7 — Promotion decision

Apply every gate. If any required gate fails, keep the candidate research-only and retain the baseline policy.

### Step 8 — Bundle and inference integration

Only after promotion gates pass, build candidate bundles, update policy, and test rollback to the baseline bundle.

### Step 9 — Provenance and CI

Regenerate provenance-correct artifacts at the final implementation commit. Run every test and the clean-worktree check.

### Step 10 — Commit and push

Commit only the intended prediction-model changes and artifacts. Push to `origin/main` only if the final GO gates pass.

## 14. Definition of done

The implementation is complete only when:

1. A feature-augmented candidate is trained on real telemetry.
2. Candidate checkpoints exist for 1h, 3h, 6h, 12h, and 24h.
3. Candidate checkpoints reload in a fresh process.
4. Candidate inference uses the same feature builder as training.
5. Baseline, ridge, tree, and candidate metrics are in one scorecard.
6. The final test set is untouched during model and policy selection.
7. Temperature, humidity, pressure, wind speed, wind direction, rain probability, and precipitation amount have target-specific metrics.
8. Heat index is physically consistent.
9. UV remains blocked or is validated only in a proven feasible regime.
10. Luminosity remains conditional or is validated by station and daylight regime.
11. Anomaly detection has a real or reviewed evaluation protocol.
12. The candidate passes all required promotion gates.
13. Calibration artifacts and model-policy bundles exist.
14. Inference validates bundle and policy hashes.
15. Provenance verification passes.
16. All tests and CI gates pass.
17. The repository remains clean after read-only tests.
18. The final commit is pushed successfully.

## 15. Current decision

Based on the latest real-data trial:

```text
Current status: CONDITIONAL GO for research only
Operational promotion: NO-GO
```

The immediate next implementation task is not to add more tests. It is to make the candidate trainer artifact-complete, then improve the candidate where the real scorecard failed:

1. 1h temperature versus persistence;
2. 1h–6h rain Brier score;
3. wind-direction circular error at every horizon.

Only after those metrics are improved or shown to be non-inferior should the candidate be considered for bundle creation and operational inference.

## References

[1]: https://github.com/benben000000/citizendashboard "Citizen Dashboard repository"
[2]: https://github.com/benben000000/citizendashboard/tree/main/prediction-model "Citizen Dashboard prediction-model directory"
