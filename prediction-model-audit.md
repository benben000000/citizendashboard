# Prediction Model Audit — `benben000000/citizendashboard`

**Scope.** This audit covers only the code and evidence under `prediction-model/`. It does not assess the dashboard UI, deployment configuration, or unrelated application code. The review was performed against repository commit `c1b2e75` (2026-09-21).

## Executive conclusion

The repository contains a promising prototype of a recurrent continuous-time model, but the current evidence does **not** support calling the prediction system production-ready. The largest problem is not the choice of Liquid Neural Network (LNN) architecture. It is the evaluation design and the separation between code paths.

The documented PyTorch model is a multi-task sequence model that emits rain probability, precipitation, and water level. However, its training pipeline currently fails at runtime because it passes an unsupported keyword to `TelemetryDataset`. Even if that call is corrected, the data pipeline trains on targets aligned to the same timestamps as the inputs rather than future timestamps. The water-level target is not joined to gauge observations; it is synthesized from rainfall. The validation set is generated from the same deterministic source windows as the training set, so it is not an independent temporal holdout.

The repository also contains a separate hand-written `ContinuousLNNCell` trainer and several adaptive/PINN-style engines. The headline scorecards appear to come from those other paths, not from the PyTorch `WeatherWaterLNN` path described in the README. The reported rain performance is weak even on its own terms: the included scorecard reports **12.99% precision**, **22.42% F1**, and **51.07% accuracy**. The water-level metric is not a real gauge-forecast metric because the main dataset builder creates a synthetic water target.

**Bottom line:** treat this as a research/demo prototype. Do not use the current scores to justify flood-warning decisions without rebuilding the dataset, defining a true forecasting task, separating train/validation/test periods, and validating against observed gauge values at each forecast horizon.

## What the model is intended to be

The principal PyTorch implementation is `WeatherWaterLNN` in `prediction-model/src/model.py`. It has:

- a four-feature input: temperature, heat index, wind speed, and pressure;
- an encoder that maps four inputs to a 32-dimensional hidden state;
- a custom CfC-like recurrent cell whose decay depends on `dt`;
- three output heads: rain probability, precipitation volume, and water level.

The cell is continuous-time in form, but the model is not a complete physics-informed hydrological model. The implementation contains no explicit conservation residual, rainfall-runoff equation, calibrated rating curve, watershed topology, or observed water-level coupling. The physics/PINN terminology is used more strongly in the separate adaptive scripts than in the core PyTorch model.

There are at least three materially different model families in the directory:

1. `model.py` / `train.py` / `inference.py`: PyTorch `WeatherWaterLNN` with a CfC-like cell.
2. `train_standalone.py`: a separate hand-written `ContinuousLNNCell` with manually updated weights.
3. `station_adaptive_pinn_lnn.py` and related benchmark scripts: station-specific heuristic/physics-coupled engines with hard-coded coefficients, online adaptation, and external forecast inputs.

These should be versioned and evaluated as separate models. At present they are presented as one prediction model, which makes the reported metrics difficult to attribute or reproduce.

## Findings by severity

### Critical — the documented PyTorch training command is broken

`prediction-model/src/train.py:27–29` calls:

```python
TelemetryDataset(num_samples=600, seq_len=24)
```

But `prediction-model/src/dataset.py:109–111` defines the constructor with `max_samples`, not `num_samples`. A clean run therefore raises `TypeError: unexpected keyword argument 'num_samples'` before training starts. The README directs users to the standalone trainer, but the repository also describes the PyTorch pipeline as the main LNN training path. The model reported in documentation cannot currently be reproduced through its own training script.

**Impact.** There is no reliable chain from source code to the saved scorecards. Fixing the keyword alone is not sufficient because the evaluation design has additional validity problems below.

**Required fix.** Correct the argument name, add a smoke test that loads one batch and runs one optimizer step, and pin the exact dataset/model/checkpoint used for every published metric.

### Critical — the task is not actually future forecasting

`dataset.py:76–95` creates a 24-step input window and assigns targets from the same 24 rows. For example, precipitation at row `t` is used as the target at the same row `t`. There is no shift such as `y[t + horizon]`, and there is no forecast-origin/target-time pair.

The model therefore learns an in-window reconstruction or contemporaneous classification task, not a one-hour, three-hour, or 24-hour-ahead forecast. `inference.py:63–76` then takes the final recurrent output and labels the sequence as a lead-horizon prediction. Repeating or perturbing current conditions inside the inference sequence does not repair the missing future-target definition.

**Impact.** Accuracy from this setup cannot be interpreted as probability of rain at a future lead time, expected rainfall over a future interval, or future river stage.

**Required fix.** Define a forecast origin `t0`, construct targets at `t0 + h`, and evaluate each horizon independently. For an hourly model, the input may be the preceding 24 observed hours and the target may be the next hour; for 24-hour prediction, use either a direct 24-hour target or an explicitly recursive rollout with uncertainty accumulation.

### Critical — water-level labels are synthetic, not observed gauge targets

The main dataset loader reads the weather CSV but never joins the supplied `water_csv_path` to the weather rows. Instead, `dataset.py:84–87` creates:

```python
base_water = 3.2
accum = np.convolve(precips, np.exp(-np.arange(6) / 3.0), mode="same") * 0.06
water_levels = base_water + accum
```

The water head is consequently trained and validated against a fixed baseline plus a rainfall convolution. It is not trained against `water_level_telemetry.csv`, despite that file being listed as a primary input in the README. The included water data inventory contains 43,883 rows from one water station, while the weather data contains 756,156 rows across 16 stations; no temporal/station join is implemented in this loader.

**Impact.** A reported water MAE such as `0.013 m` or `0.0621 m` is not evidence of river-stage forecast accuracy. It primarily measures how well the model reproduces the synthetic rule or a similarly constructed proxy.

**Required fix.** Join weather and gauge data using station/catchment identity and timestamp tolerance. Define the actual gauge target, preserve missingness, and report results against held-out observed water levels. If only one gauge is available, state that the model is single-gauge and do not generalize the result to 16 weather stations.

### Critical — validation leakage and non-independent splits

`train.py:27–29` creates a training dataset and a validation dataset by reading the same source and taking the first 600 and first 150 generated windows respectively. In `dataset.py:74–76`, window starts are deterministic and ordered. Thus, the validation set is a subset of the same beginning-of-file windows used by training, rather than a later time period or disjoint station set.

The standalone trainer has a related problem. `train_standalone.py:101–124` shuffles a selected balanced set and then slices the shuffled set into train and validation portions. This can be acceptable for an IID classification experiment, but it is not a valid temporal forecast split; nearby or duplicate conditions from the same stations can appear on both sides.

**Impact.** Metrics are optimistic and do not measure generalization to a future storm, a new reporting period, or an unseen station.

**Required fix.** Split by time before windowing. Use a chronological train/validation/test split, for example 60/20/20, with a final untouched test period. For spatial generalization, add a station-held-out test. Report confidence intervals by event and by station rather than only pooled row-level metrics.

### High — station identity and temporal continuity are discarded

`dataset.py:49–60` reduces every weather row to five numeric values and discards `station_id`, `recorded_at`, and location. `dataset.py:76` then constructs windows over the raw file order. The inventory shows 16 stations with large per-station blocks, so windows can be station-local only by accident and can cross station boundaries at block transitions. The loader does not sort by timestamp, verify sampling intervals, or reset recurrent state at station changes.

**Impact.** The model can learn file ordering and pooled averages instead of station dynamics. Its `dt` input is hard-coded to one hour at `dataset.py:89`, so the advertised irregular-telemetry capability is not exercised.

**Required fix.** Parse timestamps, sort within station, calculate actual elapsed time, reject or mask gaps, and build windows separately per station or include a station embedding. Never allow a window to cross stations.

### High — rain metrics are operationally poor and threshold-dependent

The scorecard in `prediction-model/data/validation_scorecard.json` reports:

| Metric | Reported value |
|---|---:|
| Recall/POD | 81.89% |
| Precision | 12.99% |
| F1 | 22.42% |
| Accuracy | 51.07% |

The positive label is a hard threshold on precipitation (`precip > 0.1`) in `dataset.py:81–82`, while the classification threshold is 0.5 in the standalone validator. No calibration curve, precision-recall curve, false-alarm rate by event, lead-time analysis, or cost-sensitive threshold selection is included.

**Impact.** The model may issue many false rain alarms. High recall alone is not sufficient for public warning use, and pooled row metrics can hide poor performance on rare heavy-rain events.

**Required fix.** Report event-based detection, false alarm ratio, critical success index, Brier score, calibration, and performance by intensity class and lead horizon. Select the warning threshold using an explicit operational cost function.

### High — the uncertainty bands are asserted, not estimated

The validation script creates horizon bands with `0.08 * sqrt(horizon)` and marks every horizon as “PASS (Physically Bounded)” in `validate.py:153–164`. This is a hand-written formula, not a confidence interval estimated from residuals, an ensemble, conformal prediction, or a probabilistic likelihood.

**Impact.** The bands should not be presented as forecast uncertainty or coverage guarantees. There is no evidence that 80%, 90%, or 95% of observations fall inside them.

**Required fix.** Estimate uncertainty on a held-out period and report empirical coverage and interval width by horizon, station, and event regime. A simple conformal interval would be more defensible than a fixed square-root rule.

### High — “PINN” and physics claims are not supported by the core loss

The PyTorch training loss in `train.py:62–67` is a weighted sum of binary cross-entropy, precipitation MSE, and water-level Huber loss. There is no physics residual in this training path. The custom CfC transition is a neural recurrence, not by itself a physical constraint.

The adaptive scripts do contain hand-coded rain affinity, water decay, and station coefficients, but these are separate engines and are not demonstrated to be calibrated from independent physical measurements. Calling the complete system a PINN is therefore misleading unless the physical equations, parameters, constraints, and residual terms are made explicit and trained/evaluated in one reproducible path.

**Required fix.** Either describe the PyTorch model as a continuous-time recurrent neural network, or implement and document actual physics residuals with units, parameter sources, and ablation results showing what the physics term contributes.

### High — inference does not match training or the stated horizon semantics

`inference.py:20–22` loads a checkpoint if present, but the repository contains no `.pt`, `.pth`, or `.onnx` artifact in the cloned tree. The script therefore falls back to an untrained model unless an external checkpoint is supplied. It also predicts a 24-step trajectory from a sequence assembled from current values and a sinusoidal temperature perturbation (`inference.py:87–89`), while the training data uses constant `dt=1` and same-step targets.

The output water level is reconstructed by adding the difference between neural outputs to the current gauge level (`inference.py:66`), then clamped at 0.5 m (`inference.py:82`). This makes the displayed result partly a post-processing rule rather than a directly trained stage forecast.

**Impact.** A user can obtain plausible-looking predictions from an untrained or mismatched model, and the output label “24h” does not establish 24-hour forecast validity.

**Required fix.** Fail closed when a checkpoint is missing. Store model version, feature schema, normalization constants, training date, and horizon in the checkpoint. Make inference consume the same feature builder used during training and test the exact serialized artifact against a golden input/output fixture.

### Medium — normalization is hard-coded and not fitted per training split

`dataset.py:15–17` uses fixed means and standard deviations described as being calculated from historical records, but the code does not compute or persist them from the training split. This can create distribution shifts across stations and periods, and it prevents a reproducible preprocessing contract unless those values are versioned with the checkpoint.

**Required fix.** Fit preprocessing only on the training period, persist it with the model artifact, and validate ranges and missing-value handling at inference time.

### Medium — no reproducible experiment controls

The training paths do not consistently set random seeds, record the exact source files and hashes, preserve split indices, or report per-horizon/per-station metrics. The standalone trainer uses random shuffling, and the PyTorch path uses randomized initialization and a shuffled loader. The saved JSON files contain metrics but not a complete reproducibility manifest.

**Required fix.** Add seeds, dataset hashes, split manifests, environment/package versions, model config, and a single command that recreates the reported artifact and scorecard.

## What is valid or promising

The CfC-like recurrent update is compact and computationally inexpensive. Its use of `dt` in the decay equation is a reasonable architectural direction for irregularly sampled telemetry. The model also separates rain classification, rainfall amount, and water-level outputs, which is a sensible multi-task structure if the labels are defined correctly.

The repository has useful raw telemetry and several benchmark utilities. The saved scorecard also honestly exposes poor precision rather than reporting accuracy alone. These are good foundations for a proper experiment, but they do not compensate for invalid target construction and split leakage.

## Recommended rebuild order

1. **Choose one canonical model.** Start with `WeatherWaterLNN`; quarantine the standalone and adaptive engines until their inputs, outputs, and metrics are separated.
2. **Build a canonical table.** Preserve station ID, timestamp, weather variables, observed gauge level, and data-quality flags. Sort and resample per station.
3. **Define forecast origins and horizons.** Create future targets at +1 h, +3 h, +6 h, +12 h, and +24 h. Do not use contemporaneous targets for a forecast claim.
4. **Split chronologically before windowing.** Keep a final storm/event period untouched. Add station-held-out evaluation if cross-station generalization is a product requirement.
5. **Train against observed labels.** Rain labels should come from a clearly documented gauge/radar rule; water labels should come from gauge observations, not a rainfall convolution.
6. **Benchmark against simple baselines.** At minimum compare persistence, climatology/seasonal baseline, and a rainfall-runoff or gradient-boosted baseline. The LNN should beat these on the same test set.
7. **Calibrate and evaluate operationally.** Report precision, recall, false-alarm ratio, CSI, Brier score, MAE/RMSE, event detection, interval coverage, and errors by station and horizon.
8. **Make the artifact safe.** Package preprocessing and model weights together, fail if weights are absent, and expose the exact model version and forecast origin in every prediction.

## Final assessment

The prediction code is best classified as an **early research prototype with several disconnected experimental implementations**. The architecture is not the main blocker. The current blocker is that the training labels, validation split, inference horizon, and reported scorecards do not form a valid, reproducible future-forecasting experiment.

A defensible next milestone is not a larger network or another benchmark script. It is one clean end-to-end experiment in which an observed weather window predicts a genuinely future, observed rain/gauge outcome on a time-separated test period.

## References

[1]: https://github.com/benben000000/citizendashboard "Repository audited"
[2]: https://github.com/benben000000/citizendashboard/blob/c1b2e75/prediction-model/src/model.py "PyTorch WeatherWaterLNN model"
[3]: https://github.com/benben000000/citizendashboard/blob/c1b2e75/prediction-model/src/dataset.py "Dataset construction and preprocessing"
[4]: https://github.com/benben000000/citizendashboard/blob/c1b2e75/prediction-model/src/train.py "PyTorch training loop"
[5]: https://github.com/benben000000/citizendashboard/blob/c1b2e75/prediction-model/src/inference.py "Inference and horizon output logic"
[6]: https://github.com/benben000000/citizendashboard/blob/c1b2e75/prediction-model/src/train_standalone.py "Standalone LNN trainer and validator"
[7]: https://github.com/benben000000/citizendashboard/blob/c1b2e75/prediction-model/src/validate.py "Validation scorecard and uncertainty-band logic"
[8]: https://github.com/benben000000/citizendashboard/blob/c1b2e75/prediction-model/data/validation_scorecard.json "Included validation scorecard"
[9]: https://github.com/benben000000/citizendashboard/blob/c1b2e75/prediction-model/README.md "Prediction-model README and stated architecture"
