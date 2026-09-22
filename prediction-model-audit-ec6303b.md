# Prediction Model Audit — Commit `ec6303b`

## Verdict

The newest commit materially improves the evaluation pipeline and resolves the previous validator-alignment, hidden-state, timestamp-quarantine, normalization, and baseline-comparison findings in the code path that generated the new scorecard.

The result is now a more credible **out-of-sample diagnostic**. It is still not a successful forecasting model and should not be treated as production-ready. The new scorecard shows that both model families underperform simple baselines, and one model family is still trained on a different target definition from the one used during evaluation.

## What is now resolved

The latest implementation now:

- loads canonical future-window sequences for calibration and test;
- resets recurrent state for every window;
- separates validation calibration data from the final test data for conformal bands;
- compares the models with persistence and climatology baselines;
- quarantines timestamps outside the declared date range, including the 2069 records;
- fits normalization statistics from the training partition;
- records forecast horizon, split method, test sample count, and model-family attribution;
- reports independent test coverage instead of only calibration-set coverage;
- preserves the `RESEARCH_PROTOTYPE` status.

These are real fixes, not merely documentation changes.

## Remaining findings

### Critical — the horizon is still measured in rows, not hours

`dataset.py` defines `horizons` as hours, but the window builder uses:

```python
future_rows = split_rows[i + seq_len: i + seq_len + max_horizon]
target_idx = max_horizon - 1
```

The latest scorecard labels the test as `forecast_horizon_hours: 1`, but the data has approximately **one-minute median sampling intervals** across stations. Therefore, the target for `horizon=1` is generally the next telemetry row, approximately one minute later, not one hour later.

This is the most important remaining semantic defect. The report is now genuinely out-of-sample, but it is not yet a genuine **+1-hour** evaluation.

**Fix:** select the first future observation at or after `origin_timestamp + timedelta(hours=horizon)`, subject to a tolerance window. Do not interpret row offsets as hours. Store both the requested horizon and the actual elapsed lead time in metadata, then report the actual lead-time distribution.

### Critical — MF-2 is evaluated on future windows but trained on same-row labels

`train_standalone.py` still loads individual rows, labels rain from that row's precipitation, and updates the model using `forward_step` on that same row. It does not train on the 24-observation input windows and future targets used by the new validator.

The validator's MF-2 runner now unrolls a future-window input and evaluates a future target. That is a good evaluation design, but the weights are from a different task. The scorecard therefore tests MF-2 under a distribution/target contract it was not trained to solve.

**Fix:** either train MF-2 using the same canonical future-window dataset and target timestamps as validation, or label MF-2 explicitly as a same-row diagnostic and exclude it from future-forecast claims.

### High — both models lose to persistence on the reported test

The committed scorecard reports the following at the nominal +1h horizon:

| Model | Accuracy | Recall | Precision | F1 | CSI | Brier |
|---|---:|---:|---:|---:|---:|---:|
| MF-1 PyTorch | 92.17% | 0.00% | 0.00% | 0.00% | 0.00% | 0.0705 |
| MF-2 standalone | 70.00% | 57.45% | 14.44% | 23.08% | 13.04% | 0.1627 |
| Persistence | 92.50% | 53.19% | 52.08% | 52.63% | 35.71% | 0.0750 |
| Climatology | 92.17% | 0.00% | 0.00% | 0.00% | 0.00% | 0.0739 |

MF-1 predicts no positive rain cases on the test set. MF-2 produces an **85.56% false-alarm ratio**. Persistence is better on every operational classification metric shown except recall.

**Interpretation:** the pipeline is now honest enough to demonstrate that the current models do not add predictive value for this test set. This is a useful result, but it is not a model success.

### High — water-level prediction is much worse than persistence

The scorecard reports:

| Model | MAE | RMSE |
|---|---:|---:|
| MF-1 PyTorch | 0.8146 m | 0.8619 m |
| MF-2 standalone | 0.8059 m | 0.8451 m |
| Persistence | **0.0007 m** | **0.0027 m** |
| Climatology | 0.2985 m | 0.3460 m |

The model errors are hundreds of times larger than the persistence error on the matched gauge subset. The model should not be used to issue water-stage warnings in its current form.

The small persistence error may partly reflect a very short effective horizon, which reinforces the row-versus-hour finding. After correcting the horizon selection, the baseline comparison must be rerun.

### High — conformal intervals fail their target coverage

The latest independent test coverage is:

| Target coverage | Test coverage | Calibration samples | Test samples |
|---|---:|---:|---:|
| 80% | 41.5% | 32 | 41 |
| 90% | 51.2% | 32 | 41 |
| 95% | 78.0% | 32 | 41 |

The code now correctly evaluates coverage independently, but the result shows that the intervals are not calibrated for this test distribution. They should not be presented as 80%, 90%, or 95% reliable prediction intervals.

The calibration set contains only 32 gauge windows and the test set only 41 gauge windows, so the estimates are also statistically unstable.

**Fix:** correct the horizon first, increase the gauge sample, calculate intervals per horizon, and report uncertainty with confidence bounds. If coverage remains poor, do not publish the nominal coverage as an operational guarantee.

### Medium — the test contains no moderate or heavy rain events

The test intensity breakdown contains 41 trace events and 6 light events, with zero moderate and zero heavy events. The current test cannot support claims about heavy-rain or flood-event detection.

**Fix:** create event-stratified test periods that include meaningful moderate/heavy precipitation and report event-level metrics separately from row-level metrics.

### Medium — MF-1 and MF-2 are still not one canonical training contract

The registry correctly separates model families, but the committed checkpoint and JSON weights are generated by different training designs. MF-1 uses PyTorch sequence training. MF-2 uses a hand-written recurrent cell and same-row balanced updates. The validator now evaluates both against the same future-window test samples, but only MF-1 is close to being trained with that contract, and even MF-1 currently uses row offsets for the nominal hour horizon.

**Fix:** define a model-family-specific training manifest containing: feature window, target timestamp rule, sampling interval, horizon, normalization artifact, split boundaries, and exact test command. Reject evaluation when those contracts do not match.

## Static verification

The updated Python files compile successfully with `py_compile`. The latest repository commit is:

`ec6303b7af56f9db09b2f276664aa35edab5d173`

## Final assessment

The claim “all findings are resolved” is **partially true at the pipeline-integrity level**. The code now has a defensible structure for an independent future-window evaluation, and the previous methodological issues are much better documented and controlled.

It is **not true at the model-performance level**. The nominal +1h target is currently about one minute ahead, MF-2 is trained on a different task, both models underperform persistence, the water-level errors are unacceptable, and the uncertainty bands fail their target coverage.

The next fix should focus on one issue only: implement true time-based target selection, retrain both model families under the same contract, and regenerate the scorecard. Do not tune the architecture or claim production readiness until the models beat persistence on the corrected multi-hour test set.

## References

[1]: https://github.com/benben000000/citizendashboard/commit/ec6303b7af56f9db09b2f276664aa35edab5d173 "Latest prediction-model fix commit"
[2]: https://github.com/benben000000/citizendashboard/blob/ec6303b7af56f9db09b2f276664aa35edab5d173/prediction-model/src/dataset.py "Latest dataset and future-target implementation"
[3]: https://github.com/benben000000/citizendashboard/blob/ec6303b7af56f9db09b2f276664aa35edab5d173/prediction-model/src/train_standalone.py "Latest standalone model trainer"
[4]: https://github.com/benben000000/citizendashboard/blob/ec6303b7af56f9db09b2f276664aa35edab5d173/prediction-model/src/validate.py "Latest independent future-window validator"
[5]: https://github.com/benben000000/citizendashboard/blob/ec6303b7af56f9db09b2f276664aa35edab5d173/prediction-model/data/validation_scorecard.json "Latest committed validation scorecard"
[6]: https://github.com/benben000000/citizendashboard/blob/ec6303b7af56f9db09b2f276664aa35edab5d173/prediction-model/MODEL_REGISTRY.md "Latest model-family registry"
