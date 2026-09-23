# Garcia Weather Telemetry Forecast Engine
## Detailed Prediction-Model and Scorecard Implementation Checklist

**Prepared by:** Manus AI  
**Scope:** Prediction-model implementation, validation, provenance, and product-readiness requirements  
**Current baseline:** GitHub `main` at `b5039f83e082558094557eb77328af13bdec1c1c`  
**Current status:** Research prototype; not approved for autonomous operational or life-safety use

## 1. Executive decision

The requested direction is technically feasible in stages, but it is not already implemented in the current GitHub prediction model. The existing pipeline is primarily a rain-occurrence and water-level research pipeline. It has an eight-feature weather input contract, but it does not independently forecast and score all of the proposed commercial weather variables.

The first commercially defensible version should therefore be built as a **weather-monitoring and probabilistic guidance product**, not as a universal numerical weather-prediction system. Current observations, trend detection, historical charts, and probability-based rain guidance can be delivered before every forecast target demonstrates positive skill over persistence. A learned forecast should be used only where validation shows it adds value. Persistence should remain the fallback where it performs better.

Water-level forecasting should remain internal or beta. The current scorecard contains only approximately 198–221 water-gauge test samples per horizon, and persistence is substantially better at short horizons. Flood prediction should not be part of the core commercial claim until the data and event record are materially stronger.

## 2. Status legend and required evidence

Use the following status values in implementation issues and pull requests:

- **Not implemented:** No production-quality code or scorecard support exists.
- **Partially implemented:** Some inputs, calculations, or research outputs exist, but the contract or validation is incomplete.
- **Implemented, research-only:** The capability exists but has not passed the commercial or operational gates.
- **Blocked by data:** The code can be designed, but the current repository lacks the required target data or coverage.
- **Impossible to claim now:** The requested claim cannot be made honestly from the available evidence. This does not necessarily mean the feature can never be built.
- **Complete:** The implementation, tests, provenance artifacts, and scorecard gates all pass.

Every completed item must provide four kinds of evidence:

1. A code path that can be run from a clean checkout.
2. A test that fails when the contract is violated.
3. A scorecard entry using an untouched test set.
4. A provenance record linking the result to the exact code and dataset hashes.

## 3. Product boundary checklist

### 3.1 Rename and define the product

- [ ] **Define the product as “Garcia Weather Telemetry Forecast Engine.”**

  **Why:** This accurately describes the near-term value: converting customer station measurements into monitoring, trends, short-term guidance, and probabilistic rain information. It avoids promising flood prediction or perfect numerical forecasts.

  **How:** Update the prediction-model README, model registry, product-facing documentation, and any dashboard copy that currently presents water-level forecasting as a primary capability. Use one canonical product description so the model repository and application do not diverge.

  **Positive:** The positioning is supportable with the current telemetry pipeline. It allows the product to deliver value through monitoring and alerts while forecast skill improves.

  **Negative:** The product message will be narrower than a flood or all-weather prediction platform. Sales material must explain that probability and trend guidance are not certainty.

- [ ] **Remove flood prediction from the core commercial claim.**

  **Why:** Water-level test samples are sparse, the gauge coverage is limited, and persistence currently beats the learned models at most short horizons. A flood claim would overstate both validation strength and operational safety.

  **How:** Mark water-level output as `INTERNAL_EXPERIMENT` or `BETA`, keep it out of the default commercial forecast contract, and require an explicit feature flag for any water-level display. Preserve the research scorecard for internal development.

  **Positive:** This reduces safety and reputation risk while preserving the research investment.

  **Negative:** Some customers may expect flood-related functionality. That capability must be presented as conditional and non-life-safety beta functionality.

- [ ] **Define the initial customer promise around observations, trends, and probabilistic guidance.**

  **Why:** Current-condition monitoring does not require the learned model to beat persistence. It is useful immediately when sensor quality, freshness, and alert semantics are reliable.

  **How:** Specify current observations, rising/falling trends, hourly outlooks, rain probability, confidence labels, heat-risk guidance, historical charts, downloads, and alerts as separate product capabilities. Do not use forecast accuracy as the only value metric.

  **Positive:** This creates a realistic v1 product while the forecasting research continues.

  **Negative:** The system needs strong telemetry freshness and quality monitoring. A dashboard showing bad current data can still harm trust even if the forecast is labelled experimental.

### 3.2 Define commercial and non-commercial outputs

- [ ] **Create an explicit output taxonomy.**

  **Why:** The current code mixes rain, precipitation amount, and water-level outputs. Commercial consumers need to know which outputs are supported, beta, or prohibited.

  **How:** Define three classes in the model registry:

  | Output class | Examples | Permitted use |
  |---|---|---|
  | Core | Current observations, trends, temperature outlook, humidity outlook, pressure trend, wind outlook, rain probability | Commercial monitoring and planning, with confidence labels |
  | Secondary/beta | Rain amount ranges, UV, luminosity, water level | Opt-in research or beta use after target-specific validation |
  | Prohibited claim | Flood evacuation trigger, guaranteed rain/no-rain statement, life-safety forecast | Not permitted from this model |

  **Positive:** Documentation, APIs, and UI can enforce the same boundary.

  **Negative:** More output categories require more metadata and product logic.

- [ ] **Keep water level as an internal experiment or beta module.**

  **Why:** This follows the data limitations and prevents an unsupported flood claim.

  **How:** Keep the code and internal scorecard, but exclude water-level fields from the default weather-only response schema. Add a clear `experimental: true` and `not_for_life_safety: true` marker if it is exposed to testers.

  **Positive:** The research can continue without confusing the commercial product boundary.

  **Negative:** Maintaining a beta module increases testing and documentation work.

## 4. Data and telemetry contract checklist

### 4.1 Audit available target data before changing the model

- [ ] **Inventory every telemetry field by station, timestamp, unit, missingness, and time range.**

  **Why:** A forecast target is not implementable merely because a variable appears in a product plan. The repository currently has weather telemetry, but the existing canonical target and scorecard do not cover every proposed variable.

  **How:** Add a deterministic data-audit command that emits field presence, row counts, station coverage, UTC range, missingness, outlier counts, unit assumptions, and source hashes. Store the audit as a provenance-correct artifact or generate it reproducibly from the raw files.

  **Positive:** This prevents designing heads for unavailable or unreliable variables.

  **Negative:** The audit may show that some planned features cannot be implemented without new hardware or external data.

- [ ] **Confirm whether UV index and luminosity/solar irradiance exist as measured fields.**

  **Why:** The current model schema has no UV or luminosity target. Nighttime zeros cannot be used to create a credible daylight forecast score.

  **How:** Search raw telemetry schemas and station metadata. If fields are absent, mark UV and luminosity as blocked by data rather than deriving them from unsupported assumptions. If available, document units, sensor calibration, daylight flags, and station coverage.

  **Positive:** A real target enables an honest derived or learned forecast.

  **Negative:** Sparse or uncalibrated solar sensors may make a scorecard unreliable even when the columns exist.

- [ ] **Verify wind-direction source coverage and circular consistency.**

  **Why:** Wind direction must not be scored with ordinary absolute error. A transition from 359° to 1° is a 2° error, not a 358° error.

  **How:** Record direction in degrees, convert to unit-vector components using `u = speed * cos(theta)` and `v = speed * sin(theta)` or use normalized directional components, and reconstruct direction with `atan2`. Handle calm-wind cases explicitly because direction is undefined or unstable when speed is near zero.

  **Positive:** Circular treatment is physically meaningful and avoids a known metric failure.

  **Negative:** It adds target transformation, calm-wind rules, and specialized metrics.

- [ ] **Define units and resampling semantics for every target.**

  **Why:** Temperature, pressure, humidity, wind speed, rain amount, UV, and luminosity have different aggregation rules. Incorrect aggregation can invalidate the scorecard.

  **How:** Document whether each hourly target uses the last valid value, mean, maximum, minimum, vector average, or sum. Continue summing incremental precipitation within the hour. Preserve raw values and record the resampling rule in manifests.

  **Positive:** The target contract becomes reproducible and auditable.

  **Negative:** Different users may expect different hourly semantics; the product must expose the chosen definition.

- [ ] **Define observed-label rules for every horizon.**

  **Why:** A target must be observed at the true `t0 + h` timestamp. Row offsets or synthetic targets can create leakage or false skill.

  **How:** Build target windows by timestamp, enforce the requested lead-time tolerance, record actual lead distributions, and quarantine windows with missing future labels. Never score an imputed label as an observed outcome.

  **Positive:** Horizon comparisons become trustworthy.

  **Negative:** Strict label availability reduces sample counts, especially for long horizons and sparse variables.

### 4.2 Preserve the existing ingestion protections

- [ ] **Keep strict UTC parsing.**
- [ ] **Keep quarantine counts by reason.**
- [ ] **Keep raw source telemetry unchanged.**
- [ ] **Keep source SHA-256 hashes in every regenerated manifest.**
- [ ] **Keep normalization fitted only on the training split.**
- [ ] **Keep chronological splits and the 48-hour embargo.**

These protections are already part of the remediation direction and must not be weakened when adding new targets. Their positive effect is prevention of leakage and synthetic-data contamination. Their negative effect is that the usable sample count will be lower than the raw row count.

## 5. Model architecture checklist

### 5.1 Decide whether to extend MF-1 or create a weather-only model

- [ ] **Create a weather-only forecast family rather than continuing to make water level a mandatory output.**

  **Why:** The commercial scope is weather telemetry. A multi-task water head can distract training and makes the core contract appear to support flood forecasting.

  **How:** Introduce a weather-only model configuration with target heads for the variables that have verified labels. Keep the existing WeatherWaterLNN as an internal research configuration until migration is complete.

  **Positive:** The core model directly matches the commercial product.

  **Negative:** This creates a new artifact family and requires retraining and scorecard migration.

- [ ] **Use separate target heads for continuous variables and probabilistic/event variables.**

  **Why:** Temperature, humidity, pressure, and wind speed are continuous. Rain occurrence is probabilistic. Rain amount is nonnegative and often zero-inflated.

  **How:** Use regression heads for continuous variables, a classification or calibrated probability head for rain occurrence, and a two-stage rain-amount head for occurrence plus positive amount. Use appropriate losses and masks for each target.

  **Positive:** Each output receives a loss and metric suited to its statistical behavior.

  **Negative:** Multi-task weighting becomes a tuning problem. Losses must not be tuned on the final test set.

- [ ] **Forecast wind direction with circular components.**

  **Why:** Ordinary degree regression is discontinuous at 0/360 degrees.

  **How:** Predict vector components, reconstruct direction with `atan2`, and score circular MAE. Include a calm-wind mask or a separate calm classification rule.

  **Positive:** Correct physical treatment and interpretable direction errors.

  **Negative:** Direction is not meaningful during calm conditions, so the scorecard must report both valid-direction coverage and directional error.

- [ ] **Derive heat index from predicted temperature and humidity.**

  **Why:** Heat index is a deterministic derived quantity. Predicting it independently can create internal inconsistency.

  **How:** Forecast temperature and relative humidity first, then apply a documented heat-index formula to the predictions. Score the derived result against observed heat index or a recomputed reference under the same formula.

  **Positive:** The product outputs remain physically consistent.

  **Negative:** Heat-index error inherits temperature and humidity error and can be sensitive near formula applicability boundaries.

- [ ] **Treat UV and luminosity as separate derived or beta targets only after data approval.**

  **Why:** Solar quantities depend on daylight, geometry, cloud attenuation, sensor calibration, and location. They cannot be honestly supported from absent or unverified telemetry.

  **How:** First implement a clear-sky baseline using station latitude, longitude, timestamp, and solar geometry. Add cloud attenuation only if cloud-related or irradiance observations support it. Score only daylight records.

  **Positive:** A physics-informed baseline provides a meaningful comparator.

  **Negative:** It requires station metadata and may still be inadequate without cloud or solar observations.

### 5.2 Training and reproducibility

- [ ] **Set deterministic seeds for every model family and data-loader path.**
- [ ] **Record package versions, hardware, training configuration, feature schema, target schema, split boundaries, and dataset hashes.**
- [ ] **Save the best checkpoint using validation-only selection.**
- [ ] **Reject empty real training or calibration splits.**
- [ ] **Run training for all horizons: 1h, 3h, 6h, 12h, and 24h.**
- [ ] **Prevent target leakage across station and time boundaries.**
- [ ] **Test checkpoint loading from a clean environment.**

**Why:** Multi-output weather forecasting is especially vulnerable to accidental target leakage and incomparable retraining results.

**How:** Extend the existing canonical manifests and provenance verifier to include target names, target units, target resampling rules, loss weights, model dimensions, and every forecast head.

**Positive:** Results can be reproduced and challenged.

**Negative:** Provenance metadata becomes more complex and must be updated whenever the model contract changes.

## 6. Hybrid forecast strategy checklist

### 6.1 Per-variable model selection

- [ ] **Implement a validation-only skill gate for every forecast target and horizon.**

  **Why:** The note correctly recommends using the learned model only when it has positive skill over persistence. The current learned rain model loses to persistence on thresholded F1 across the tested horizons.

  **How:** For every target and horizon, compute a predefined skill score such as `1 - model_error / persistence_error` for continuous variables. For rain, use multiple criteria: Brier skill, calibration, and decision metrics such as F1/CSI. The gate must be fitted on the calibration split and frozen before test evaluation.

  **Positive:** The deployed forecast is objectively better than a weak fallback for the chosen metric.

  **Negative:** A gate optimized for one metric may worsen another. The product must declare which metric controls each output.

- [ ] **Define fallback behavior for every target.**

  **Why:** A hybrid system is only safe if it has a deterministic behavior when the learned model is unavailable or underperforms.

  **How:** Use persistence for temperature, humidity, pressure, and wind speed when it wins. Use the clear-sky baseline for solar variables when appropriate. Use a calibrated climatology fallback when persistence is undefined. Return the model, fallback, selected metric, and validation skill in the forecast metadata.

  **Positive:** The service remains available and transparent.

  **Negative:** The output may vary by variable and horizon, which requires clear customer-facing explanation.

### 6.2 Calibrated rain blending

- [ ] **Implement per-horizon rain blending.**

  **Why:** MF-1 has useful 1-hour probability information according to Brier score even though thresholded F1 trails persistence. A blend may retain probability quality while improving decisions.

  **How:** On the calibration split only, fit `p_final = w_h * p_model + (1 - w_h) * p_persistence` separately for each horizon. Constrain `w_h` to `[0, 1]`. Select the weight using a declared objective, preferably a proper probabilistic metric such as Brier score or log loss, with threshold metrics reported separately. Freeze `w_h` before evaluating the untouched test set.

  **Positive:** The product can use model information without blindly replacing a stronger persistence forecast.

  **Negative:** A blend can hide model weakness if not reported transparently. Weights must not be selected on the final test set.

- [ ] **Calibrate probability outputs and report reliability.**

  **Why:** A probability forecast should mean what it says. Thresholded F1 alone does not establish calibration.

  **How:** Compare uncalibrated, Platt-scaled, isotonic, and beta-calibrated alternatives only on calibration data, subject to sample-size rules. Report reliability diagrams, Brier score, expected calibration error, confidence intervals, and threshold sensitivity on the untouched test set.

  **Positive:** Customers can receive probability and confidence labels rather than false certainty.

  **Negative:** Calibration can overfit small calibration sets, especially at long horizons or rare rain events.

- [ ] **Separate decision thresholds from probability calibration.**

  **Why:** A probability forecast can have a good Brier score but a poor F1 at threshold 0.5. These are different problems.

  **How:** Keep the probability output unchanged after calibration. Select an alert threshold on the calibration set based on an explicitly declared objective, such as CSI, F1, recall at a maximum false-alarm rate, or cost-weighted utility. Report test results at both the fixed 0.5 threshold and the frozen operational threshold.

  **Positive:** Alert behavior can match operational costs.

  **Negative:** Optimizing thresholds for one customer or event type may not generalize to another.

## 7. Weather-only scorecard checklist

### 7.1 Scorecard structure

- [ ] **Create a separate weather-only scorecard.**

  **Why:** The existing scorecard validates rain and water level. It cannot support accuracy claims for temperature, humidity, pressure, wind, UV, or luminosity.

  **How:** Add a validator that produces one result block for each target and each horizon. Keep the existing research scorecard for rain/water regression comparisons, but do not call it the commercial weather scorecard.

  **Positive:** Product claims map directly to measured evidence.

  **Negative:** The scorecard becomes larger and requires target-specific missing-data handling.

- [ ] **Evaluate all horizons consistently.**

  **Required horizons:** 1h, 3h, 6h, 12h, and 24h.

  **Why:** A model can be useful at 1 hour and harmful at 24 hours. A single aggregate score hides this behavior.

  **How:** Use the same train, calibration, and untouched test date boundaries for every model and target where labels exist. Report sample counts and actual lead distributions for every block.

- [ ] **Compare against persistence, climatology, and a simple machine-learning baseline.**

  **Why:** Persistence is currently a strong baseline. A learned model should not be considered commercially useful merely because it beats climatology.

  **How:** Use last observation persistence, train-split climatology, and a simple baseline such as linear regression, logistic regression, gradient boosting, or a persistence-plus-trend model. Freeze baseline definitions before test evaluation.

  **Positive:** The scorecard prevents inflated claims.

  **Negative:** Strong baselines may make the learned model look less impressive, but that is scientifically correct.

### 7.2 Required metrics by target

- [ ] **Temperature:** report MAE, RMSE, bias, persistence skill, calibration or interval coverage, and sample count.

  **Why:** Temperature has direct operational value and is intuitive for customers.

  **How:** Use the observed target at each horizon. Report results overall and by station. If the target is hourly temperature, document whether it is last-value, mean, or another aggregation.

- [ ] **Relative humidity:** report MAE, RMSE, bias, persistence skill, and sample count.

  **Why:** Humidity supports comfort, heat-risk, and operational planning.

  **How:** Enforce valid percentage bounds and preserve missingness masks. Do not score imputed labels as observations.

- [ ] **Pressure:** report MAE, RMSE, bias, persistence skill, and pressure-tendency accuracy.

  **Why:** Pressure level and pressure tendency have different operational meanings.

  **How:** Define a tendency interval and direction rule before testing. Report both continuous error and the accuracy/precision/recall of rising, falling, and stable categories.

- [ ] **Wind speed:** report MAE, RMSE, persistence skill, and recall for predefined strong-wind thresholds.

  **Why:** Average wind error does not show whether dangerous or operationally important wind events are detected.

  **How:** Choose thresholds from domain requirements or a pre-registered percentile. Report event counts, POD, FAR, CSI, and confidence intervals.

- [ ] **Wind direction:** report circular MAE, valid-direction sample count, and calm-wind coverage.

  **Why:** Ordinary MAE is invalid near the 0/360 boundary.

  **How:** Use vector components and circular reconstruction. Exclude or separately classify calm-wind samples under a documented speed threshold.

- [ ] **Rain occurrence:** report F1, CSI, POD, FAR, Brier score, reliability, calibration error, threshold, event prevalence, and confidence intervals.

  **Why:** Rain has both probabilistic and decision uses. One metric cannot represent both.

  **How:** Report fixed 0.5 results, calibrated probability results, and frozen operational-threshold results. Include persistence, climatology, simple ML, learned model, and hybrid blend.

- [ ] **Rain amount:** report rainy-hour MAE, overall MAE, RMSE, R², bias, quantile coverage, and interval width.

  **Why:** Rain occurrence and rain amount are different tasks. A model can detect rain without estimating amount well.

  **How:** Use a two-stage target contract. Report performance on all hours and only observed rainy hours. Do not claim exact future millimetres when uncertainty intervals are wide.

- [ ] **Heat index:** report MAE, RMSE, bias, and persistence skill.

  **Why:** Heat index is a derived customer-facing risk indicator.

  **How:** Derive it from forecast temperature and humidity using one documented formula, then evaluate the derived forecast.

- [ ] **UV and luminosity:** report daylight-only MAE, RMSE, bias, clear-sky-index error, coverage, and a clear-sky/persistence baseline.

  **Why:** Nighttime zeros can produce a deceptively good score.

  **How:** Use a daylight mask based on station location and solar geometry. Score only valid daylight records and report daylight sample counts.

### 7.3 Scorecard governance

- [ ] **Include positive and negative skill flags.**

  **Why:** The result must explicitly identify where the model loses.

  **How:** Add fields such as `beats_persistence`, `skill_vs_persistence`, `calibration_pass`, `coverage_meets_nominal`, and `operational_recommendation` per target and horizon.

- [ ] **Include data sufficiency warnings.**

  **Why:** A score computed on very few samples is not equivalent to a score computed on thousands.

  **How:** Define minimum test and calibration sample thresholds. Attach `small_sample_warning` and confidence intervals where thresholds are not met.

- [ ] **Keep final test data untouched.**

  **Why:** Hybrid weights, thresholds, calibration parameters, and model selection must not be tuned on the final test set.

  **How:** Use train for fitting, calibration/validation for model selection and calibration, and test exactly once for final reporting.

- [ ] **Make scorecards provenance-correct.**

  **Why:** A result without the exact code and data identity cannot be reliably reproduced.

  **How:** Record commit, raw hashes, feature schema, target schema, split boundaries, embargo, model configuration, calibration method, and package versions. Add a verifier that fails on mismatch.

## 8. Data-dependent feasibility decisions

### 8.1 Temperature, humidity, pressure, and wind speed

**Feasibility:** Likely feasible if future labels exist at adequate station coverage.

**Required next step:** Audit the source files and add target-specific windows and heads.

**Positive:** These variables are common, intuitive, and useful for operations.

**Negative:** Station heterogeneity and sparse periods may require station-aware normalization or per-station evaluation.

### 8.2 Wind direction

**Feasibility:** Feasible if direction or valid vector components are present and calm-wind handling is defined.

**Required next step:** Add circular target encoding and a circular metric.

**Positive:** The requested encoding is technically standard and solves the 359°/1° discontinuity.

**Negative:** Direction quality is poor when wind speed is near zero, and the scorecard will have fewer valid-direction samples.

### 8.3 Rain probability

**Feasibility:** Feasible now, but not yet commercially superior to persistence.

**Required next step:** Add calibration, threshold analysis, and per-horizon blending.

**Positive:** The current 1-hour Brier result suggests useful probability information.

**Negative:** Better probabilities do not automatically yield better alert decisions. At 3–24 hours, larger model improvements are still needed.

### 8.4 Rain amount

**Feasibility:** Partially feasible. A precipitation output exists, but the requested rigorous amount scorecard and quantile coverage do not.

**Required next step:** Implement two-stage amount modeling, rainy-hour metrics, quantile intervals, and calibration.

**Positive:** Incremental precipitation data can support hourly totals.

**Negative:** Rain amount is zero-inflated and heavy-tailed. Exact point forecasts may be unstable, so ranges are safer.

### 8.5 UV and luminosity

**Feasibility:** **Blocked until target data and station metadata are verified.**

**Required next step:** Confirm real UV/luminosity observations, units, calibration, daylight coverage, and location metadata.

**Positive:** A clear-sky baseline can be built if the required metadata exists.

**Negative:** Without measurements or reliable external inputs, the system cannot honestly train or validate these targets.

### 8.6 Water level and flood prediction

**Feasibility:** Water-level research is possible; a robust commercial flood-prediction claim is not currently supported.

**Required next step:** Add more maintained gauges, upstream rainfall coverage, documented flood events, and event-based evaluation.

**Positive:** The existing experimental pipeline can remain useful for research.

**Negative:** Sparse gauge samples and persistence dominance make current flood claims unsafe.

## 9. Testing checklist

- [ ] Add unit tests for every target transformation and unit conversion.
- [ ] Add tests for wind-direction wraparound, including 359° versus 1°.
- [ ] Add tests for calm-wind masking.
- [ ] Add tests proving heat index is derived from forecast temperature and humidity.
- [ ] Add tests proving nighttime UV/luminosity observations are excluded from daylight metrics.
- [ ] Add tests for missing future labels and quarantine behavior.
- [ ] Add tests for exact horizon lead-time tolerance.
- [ ] Add tests that fail if calibration uses final test labels.
- [ ] Add tests that fail if a fallback is selected without a recorded validation rationale.
- [ ] Add checkpoint load tests for every target head and horizon.
- [ ] Run contract tests, inference tests, smoke tests, scorecard tests, and provenance tests in a clean environment.
- [ ] Run `git diff --check`, reference checks, and protected-file checks before commit.

## 10. Release and operational checklist

- [ ] Keep the default commercial API weather-only.
- [ ] Mark water-level responses as beta or internal.
- [ ] Return forecast probability, confidence, horizon, model/fallback source, and validation skill metadata.
- [ ] Do not return a binary rain certainty statement without probability context.
- [ ] Add telemetry freshness and sensor-quality indicators to every forecast response.
- [ ] Define alert suppression behavior when current data is stale or required inputs are missing.
- [ ] Log which model or fallback produced each forecast.
- [ ] Version calibration weights separately from model checkpoints.
- [ ] Re-run the scorecard after each retraining cycle.
- [ ] Do not promote a model when it loses its declared skill gate.
- [ ] Do not use the model for evacuation, life-safety, or autonomous flood triggers.
- [ ] Require a human or customer-defined operational policy for alerts.

## 11. Explicitly impossible or unsafe right now

The following statements should be treated as impossible or unsafe to claim from the current repository evidence:

1. **“The system accurately forecasts UV and luminosity.”** This is impossible without verified target observations, units, daylight coverage, and station metadata.
2. **“The system provides reliable flood prediction.”** The current gauge sample size and persistence comparison do not support this claim.
3. **“The learned rain model is better than persistence at every horizon.”** The current scorecard shows the opposite for thresholded F1.
4. **“A 0.5 rain threshold is operationally optimal.”** The current scorecard does not establish that threshold.
5. **“The model has strong accuracy for temperature, humidity, pressure, wind, UV, or luminosity.”** The current scorecard does not evaluate those forecast targets.
6. **“Nighttime zero scores prove UV or luminosity accuracy.”** That would be a metric artifact, not evidence of daylight forecasting skill.
7. **“A calibrated interval is safe for life-safety decisions.”** Statistical coverage on a research test split does not establish operational safety.
8. **“The current water-level scorecard is sufficient for a flood product.”** It is not. More gauges, upstream rainfall, event labels, and event-based metrics are required.
9. **“A hybrid forecast is implemented.”** It is not implemented until the per-horizon weights, calibration method, frozen selection rules, and test results are stored in the scorecard.
10. **“The current GitHub main contains the full remediation.”** It does not; the remediation remains in local commits and was not successfully published.

## 12. Recommended implementation order

### Phase 0: Freeze the product boundary

Complete the product taxonomy, remove water/flood claims from the core contract, and document the current research-only status. This phase is low risk and should happen before changing model architecture.

### Phase 1: Audit data availability

Produce the field, unit, missingness, station, and daylight audit. Decide which targets are feasible. Do not implement UV or luminosity heads before this phase passes.

### Phase 2: Build the weather-only target contract

Add timestamp-correct targets for temperature, humidity, pressure, wind speed, direction components, rain occurrence, and rain amount where labels exist. Add target-specific quarantine and lead-time assertions.

### Phase 3: Build baseline-first validation

Implement persistence, climatology, trend, and simple machine-learning baselines for every target. Generate a baseline-only scorecard before training a larger model.

### Phase 4: Implement the multi-target weather model

Add continuous heads, circular wind-direction handling, derived heat index, and a two-stage rain head. Keep water level outside the default commercial model.

### Phase 5: Implement calibration and hybrid selection

Calibrate rain probabilities and per-horizon blend weights using validation data only. Implement model-versus-persistence skill gates for continuous targets.

### Phase 6: Generate the strict weather-only scorecard

Report all requested metrics across all five horizons, including sample counts, confidence intervals, daylight masks, rainy-hour subsets, calibration results, and positive/negative skill flags.

### Phase 7: Independent review and provenance

Run tests, inspect the scorecard, verify hashes and commit identity, review the diff, and preserve an archive before any cleanup. Do not push if any provenance or scorecard gate fails.

### Phase 8: Limited release

Release monitoring, trends, and probabilistic guidance first. Keep learned forecasts behind a research or beta flag until the declared skill gates pass. Do not release flood prediction as a core capability.

## 13. Definition of done

The weather-only implementation is complete only when all of the following are true:

- The default model contract contains only approved weather targets.
- Each target has verified real labels and documented units.
- Each target has a persistence, climatology, and simple ML baseline.
- The model is evaluated at 1h, 3h, 6h, 12h, and 24h.
- Wind direction uses circular treatment.
- Heat index is derived consistently from forecast temperature and humidity.
- UV and luminosity are either properly scored during daylight or explicitly marked blocked by missing data.
- Rain probabilities are calibrated and the blend weights are selected on calibration data only.
- Rain amount has separate rainy-hour and overall metrics plus interval coverage.
- The scorecard records positive and negative skill against persistence.
- Water level is excluded from the core commercial claim.
- The provenance gate passes at the final commit.
- Tests, smoke checks, scorecard checks, and diff checks pass.
- The product documentation does not claim more than the evidence supports.

## References

[1]: https://github.com/benben000000/citizendashboard "Garcia citizen dashboard repository and prediction-model implementation"
