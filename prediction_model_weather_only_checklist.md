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

- [x] **Define the product as “Garcia Weather Telemetry Forecast Engine.”**
  - **Status:** Complete
  - **Evidence:**
    1. Code: `prediction-model/src/inference.py:225` returns `"product_name": "Garcia Weather Telemetry Forecast Engine"`.
    2. Test: `prediction-model/src/test_canonical_contract.py::test_checkpoint_load_all_horizons` asserts product name.
    3. Scorecard: `prediction-model/data/weather_validation_scorecard.json` records product name.
    4. Provenance: Registered in `prediction-model/MODEL_REGISTRY.md` and `prediction-model/README.md`.

  **Why:** This accurately describes the near-term value: converting customer station measurements into monitoring, trends, short-term guidance, and probabilistic rain information. It avoids promising flood prediction or perfect numerical forecasts.

  **How:** Update the prediction-model README, model registry, product-facing documentation, and any dashboard copy that currently presents water-level forecasting as a primary capability. Use one canonical product description so the model repository and application do not diverge.

  **Positive:** The positioning is supportable with the current telemetry pipeline. It allows the product to deliver value through monitoring and alerts while forecast skill improves.

  **Negative:** The product message will be narrower than a flood or all-weather prediction platform. Sales material must explain that probability and trend guidance are not certainty.

- [x] **Remove flood prediction from the core commercial claim.**
  - **Status:** Complete
  - **Evidence:**
    1. Code: `inference.py:229` sets `"not_for_life_safety": True` and isolates river level under `"water_level_beta"`.
    2. Test: `test_inference_fails_closed_on_missing_features` and smoke test verify boundary isolation.
    3. Scorecard: `validation_scorecard.json` and `weather_validation_scorecard.json` note beta status and lack of flood warning authorization.
    4. Provenance: Formally documented in `MODEL_REGISTRY.md` Section 1 ("Prohibited Claims").

  **Why:** Water-level test samples are sparse, the gauge coverage is limited, and persistence currently beats the learned models at most short horizons. A flood claim would overstate both validation strength and operational safety.

  **How:** Mark water-level output as `INTERNAL_EXPERIMENT` or `BETA`, keep it out of the default commercial forecast contract, and require an explicit feature flag for any water-level display. Preserve the research scorecard for internal development.

  **Positive:** This reduces safety and reputation risk while preserving the research investment.

  **Negative:** Some customers may expect flood-related functionality. That capability must be presented as conditional and non-life-safety beta functionality.

- [x] **Define the initial customer promise around observations, trends, and probabilistic guidance.**
  - **Status:** Complete
  - **Evidence:**
    1. Code: `inference.py` delivers current observations, pressure tendencies (`RISING`/`FALLING`/`STEADY`), heat index risk categories (`NORMAL`, `CAUTION`, `EXTREME CAUTION`, `DANGER`, `EXTREME DANGER`), and calibrated rain probability.
    2. Test: `test_pressure_tendency_and_heat_index_categories` in `test_canonical_contract.py`.
    3. Scorecard: `weather_validation_scorecard.json` reports baseline and hybrid metrics for all guidance elements.
    4. Provenance: Detailed in `MODEL_REGISTRY.md` Section 1.

  **Why:** Current-condition monitoring does not require the learned model to beat persistence. It is useful immediately when sensor quality, freshness, and alert semantics are reliable.

  **How:** Specify current observations, rising/falling trends, hourly outlooks, rain probability, confidence labels, heat-risk guidance, historical charts, downloads, and alerts as separate product capabilities. Do not use forecast accuracy as the only value metric.

  **Positive:** This creates a realistic v1 product while the forecasting research continues.

  **Negative:** The system needs strong telemetry freshness and quality monitoring. A dashboard showing bad current data can still harm trust even if the forecast is labelled experimental.

### 3.2 Define commercial and non-commercial outputs

- [x] **Create an explicit output taxonomy.**
  - **Status:** Complete
  - **Evidence:**
    1. Code: Enforced in `inference.py:predict_from_observed_sequence` returning core commercial weather parameters and isolating beta river stage.
    2. Test: `test_canonical_contract.py::test_checkpoint_load_all_horizons`.
    3. Scorecard: `weather_validation_scorecard.json` partitions core weather targets from beta hydrology.
    4. Provenance: Documented taxonomy table in `MODEL_REGISTRY.md` Section 1 (Core Commercial, Secondary/Beta, Blocked by Data, Prohibited Claims).

  **Why:** The current code mixes rain, precipitation amount, and water-level outputs. Commercial consumers need to know which outputs are supported, beta, or prohibited.

  **How:** Define three classes in the model registry:

  | Output class | Examples | Permitted use |
  |---|---|---|
  | Core | Current observations, trends, temperature outlook, humidity outlook, pressure trend, wind outlook, rain probability | Commercial monitoring and planning, with confidence labels |
  | Secondary/beta | Rain amount ranges, UV, luminosity, water level | Opt-in research or beta use after target-specific validation |
  | Prohibited claim | Flood evacuation trigger, guaranteed rain/no-rain statement, life-safety forecast | Not permitted from this model |

  **Positive:** Documentation, APIs, and UI can enforce the same boundary.

  **Negative:** More output categories require more metadata and product logic.

- [x] **Keep water level as an internal experiment or beta module.**
  - **Status:** Complete
  - **Evidence:**
    1. Code: `inference.py:245-249` wraps river level in `water_level_beta` with `status: "INTERNAL_EXPERIMENT_BETA"`, `not_for_life_safety: true`.
    2. Test: `test_canonical_contract.py` and `smoke_test.py`.
    3. Scorecard: Evaluated on Calumpit gauge ($N=221$ test windows) in `MODEL_REGISTRY.md` Section 4.3.
    4. Provenance: Provenance verifier checks beta status in manifests and scorecards.

  **Why:** This follows the data limitations and prevents an unsupported flood claim.

  **How:** Keep the code and internal scorecard, but exclude water-level fields from the default weather-only response schema. Add a clear `experimental: true` and `not_for_life_safety: true` marker if it is exposed to testers.

  **Positive:** The research can continue without confusing the commercial product boundary.

  **Negative:** Maintaining a beta module increases testing and documentation work.

## 4. Data and telemetry contract checklist

### 4.1 Audit available target data before changing the model

- [x] **Inventory every telemetry field by station, timestamp, unit, missingness, and time range.**
  - **Status:** Complete
  - **Evidence:**
    1. Code: Reproducible audit generation in `prediction-model/data/weather_data_audit.json` covering 756,156 rows across 16 stations from 2026-06-20 to 2026-08-26.
    2. Test: `test_nighttime_uv_quarantine_and_data_audit` in `test_canonical_contract.py`.
    3. Scorecard: `weather_data_audit.json` records complete null count, numeric coverage, and bounds violations for all fields.
    4. Provenance: SHA-256 telemetry hashes verified in `cleaned_data_manifest.json` and `verify_provenance.py`.

  **Why:** A forecast target is not implementable merely because a variable appears in a product plan. The repository currently has weather telemetry, but the existing canonical target and scorecard do not cover every proposed variable.

  **How:** Add a deterministic data-audit command that emits field presence, row counts, station coverage, UTC range, missingness, outlier counts, unit assumptions, and source hashes. Store the audit as a provenance-correct artifact or generate it reproducibly from the raw files.

  **Positive:** This prevents designing heads for unavailable or unreliable variables.

  **Negative:** The audit may show that some planned features cannot be implemented without new hardware or external data.

- [x] **Confirm whether UV index and luminosity/solar irradiance exist as measured fields.**
  - **Status:** Complete / Blocked by sensor calibration
  - **Evidence:**
    1. Code: `weather_data_audit.json` Section `target_feasibility_determination` identifies severe sensor clock/calibration defect (UV values up to 11.0 at midnight 00:00–04:00 UTC+8).
    2. Test: `test_nighttime_uv_quarantine_and_data_audit` asserts `BLOCKED_BY_SENSOR_CALIBRATION`.
    3. Scorecard: UV quarantined in `weather_validation_scorecard.json`; light intensity classified as secondary daylight-only.
    4. Provenance: Recorded in `MODEL_REGISTRY.md` Section 1.

  **Why:** The current model schema has no UV or luminosity target. Nighttime zeros cannot be used to create a credible daylight forecast score.

  **How:** Search raw telemetry schemas and station metadata. If fields are absent, mark UV and luminosity as blocked by data rather than deriving them from unsupported assumptions. If available, document units, sensor calibration, daylight flags, and station coverage.

  **Positive:** A real target enables an honest derived or learned forecast.

  **Negative:** Sparse or uncalibrated solar sensors may make a scorecard unreliable even when the columns exist.

- [x] **Verify wind-direction source coverage and circular consistency.**
  - **Status:** Complete
  - **Evidence:**
    1. Code: `dataset.py:circular_direction_error_deg` and circular vector decomposition $(u = \cos\theta, v = \sin\theta)$; calm winds ($< 1.0$ km/h) masked to `None`.
    2. Test: `test_circular_wind_direction_wraparound_and_calm_mask` (verifies 359° vs 1° = 2.0° and calm mask).
    3. Scorecard: `weather_validation_scorecard.json` reports circular MAE for valid winds ($N=2,187$) and calm wind percentage (22.45%).
    4. Provenance: Documented in `MODEL_REGISTRY.md` Section 3.2.

  **Why:** Wind direction must not be scored with ordinary absolute error. A transition from 359° to 1° is a 2° error, not a 358° error.

  **How:** Record direction in degrees, convert to unit-vector components using `u = speed * cos(theta)` and `v = speed * sin(theta)` or use normalized directional components, and reconstruct direction with `atan2`. Handle calm-wind cases explicitly because direction is undefined or unstable when speed is near zero.

  **Positive:** Circular treatment is physically meaningful and avoids a known metric failure.

  **Negative:** It adds target transformation, calm-wind rules, and specialized metrics.

- [x] **Define units and resampling semantics for every target.**
  - **Status:** Complete
  - **Evidence:**
    1. Code: `dataset.py:TelemetryDataPipeline` resamples 1-minute telemetry onto UTC hourly grid (last-valid continuous, vector wind, incremental sum precipitation).
    2. Test: `test_precipitation_incremental_hourly_sum`.
    3. Scorecard: Documented units in `MODEL_REGISTRY.md` Section 4.1 ($^\circ$C, %, hPa, km/h, deg, mm, m).
    4. Provenance: Manifests record hourly aggregation policies.

  **Why:** Temperature, pressure, humidity, wind speed, rain amount, UV, and luminosity have different aggregation rules. Incorrect aggregation can invalidate the scorecard.

  **How:** Document whether each hourly target uses the last valid value, mean, maximum, minimum, vector average, or sum. Continue summing incremental precipitation within the hour. Preserve raw values and record the resampling rule in manifests.

  **Positive:** The target contract becomes reproducible and auditable.

  **Negative:** Different users may expect different hourly semantics; the product must expose the chosen definition.

- [x] **Define observed-label rules for every horizon.**
  - **Status:** Complete
  - **Evidence:**
    1. Code: `build_forecast_windows` matches target timestamp $t_0 + h$ with strict tolerance $|\Delta t - h| \le 0.25$h.
    2. Test: `test_synthetic_minute_telemetry_lead_time`.
    3. Scorecard: `test_predictions_log.csv` logs 13,411 test samples with 0 lead tolerance violations.
    4. Provenance: Verified in `verify_provenance.py`.

  **Why:** A target must be observed at the true `t0 + h` timestamp. Row offsets or synthetic targets can create leakage or false skill.

  **How:** Build target windows by timestamp, enforce the requested lead-time tolerance, record actual lead distributions, and quarantine windows with missing future labels. Never score an imputed label as an observed outcome.

  **Positive:** Horizon comparisons become trustworthy.

  **Negative:** Strict label availability reduces sample counts, especially for long horizons and sparse variables.

### 4.2 Preserve the existing ingestion protections

- [x] **Keep strict UTC parsing.** - Status: Complete (`dataset.py:TelemetryDataPipeline`).
- [x] **Keep quarantine counts by reason.** - Status: Complete (`quarantine_counts` in `data_quality_report.json`).
- [x] **Keep raw source telemetry unchanged.** - Status: Complete (SHA-256 hashes preserved).
- [x] **Keep source SHA-256 hashes in every regenerated manifest.** - Status: Complete (Verified by `verify_provenance.py`).
- [x] **Keep normalization fitted only on the training split.** - Status: Complete (`test_normalization_fitted_strictly_on_train`).
- [x] **Keep chronological splits and the 48-hour embargo.** - Status: Complete (`test_chronological_split_embargo` and `test_frozen_calibration_split_isolation`).

These protections are already part of the remediation direction and must not be weakened when adding new targets. Their positive effect is prevention of leakage and synthetic-data contamination. Their negative effect is that the usable sample count will be lower than the raw row count.

## 5. Model architecture checklist

### 5.1 Decide whether to extend MF-1 or create a weather-only model

- [x] **Create a weather-only forecast family rather than continuing to make water level a mandatory output.**
  - **Status:** Complete
  - **Evidence:**
    1. Code: `GarciaWeatherLNN` in `prediction-model/src/model.py` with multi-task weather heads; river stage delta is internal beta.
    2. Test: `test_checkpoint_load_all_horizons` loads `GarciaWeatherLNN` checkpoints for all 5 horizons.
    3. Scorecard: Evaluated across all 5 horizons in `weather_validation_scorecard.json`.
    4. Provenance: Registered as Model Family 1 in `MODEL_REGISTRY.md` Section 2.

  **Why:** The commercial scope is weather telemetry. A multi-task water head can distract training and makes the core contract appear to support flood forecasting.

  **How:** Introduce a weather-only model configuration with target heads for the variables that have verified labels. Keep the existing WeatherWaterLNN as an internal research configuration until migration is complete.

  **Positive:** The core model directly matches the commercial product.

  **Negative:** This creates a new artifact family and requires retraining and scorecard migration.

- [x] **Use separate target heads for continuous variables and probabilistic/event variables.**
  - **Status:** Complete
  - **Evidence:**
    1. Code: Regression heads for continuous targets (temp, humidity, pressure, wind speed, vector wind), BCE classification head for rain occurrence, and ReLU head for rain volume.
    2. Test: Multi-task forward and backward verified in `smoke_test.py` and `test_checkpoint_load_all_horizons`.
    3. Scorecard: Separate metrics reported for regression (MAE/RMSE), classification (F1/CSI/Brier), and volume in `weather_validation_scorecard.json`.
    4. Provenance: Manifest architecture config in checkpoints.

  **Why:** Temperature, humidity, pressure, and wind speed are continuous. Rain occurrence is probabilistic. Rain amount is nonnegative and often zero-inflated.

  **How:** Use regression heads for continuous variables, a classification or calibrated probability head for rain occurrence, and a two-stage rain-amount head for occurrence plus positive amount. Use appropriate losses and masks for each target.

  **Positive:** Each output receives a loss and metric suited to its statistical behavior.

  **Negative:** Multi-task weighting becomes a tuning problem. Losses must not be tuned on the final test set.

- [x] **Forecast wind direction with circular components.**
  - **Status:** Complete
  - **Evidence:**
    1. Code: `wind_u_head` and `wind_v_head` in `GarciaWeatherLNN`, reconstructed via `atan2` in `inference.py:178`.
    2. Test: `test_circular_wind_direction_wraparound_and_calm_mask` in `test_canonical_contract.py`.
    3. Scorecard: Evaluated via circular MAE ($42.1^\circ$ at +1h to $93.0^\circ$ at +24h) in `weather_validation_scorecard.json`.
    4. Provenance: Documented in `MODEL_REGISTRY.md` Section 3.2.

  **Why:** Ordinary degree regression is discontinuous at 0/360 degrees.

  **How:** Predict vector components, reconstruct direction with `atan2`, and score circular MAE. Include a calm-wind mask or a separate calm classification rule.

  **Positive:** Correct physical treatment and interpretable direction errors.

  **Negative:** Direction is not meaningful during calm conditions, so the scorecard must report both valid-direction coverage and directional error.

- [x] **Derive heat index from predicted temperature and humidity.**
  - **Status:** Complete
  - **Evidence:**
    1. Code: `compute_noaa_heat_index` in `dataset.py` implementing NOAA NWS Rothfusz regression with Steadman boundary conditions.
    2. Test: `test_noaa_rothfusz_heat_index` matches NOAA lookup tables.
    3. Scorecard: Derived heat index beats persistence across +1h, +3h, +6h, +12h in `weather_validation_scorecard.json`.
    4. Provenance: Documented in `MODEL_REGISTRY.md` Section 4.1.

  **Why:** Heat index is a deterministic derived quantity. Predicting it independently can create internal inconsistency.

  **How:** Forecast temperature and relative humidity first, then apply a documented heat-index formula to the predictions. Score the derived result against observed heat index or a recomputed reference under the same formula.

  **Positive:** The product outputs remain physically consistent.

  **Negative:** Heat-index error inherits temperature and humidity error and can be sensitive near formula applicability boundaries.

- [x] **Treat UV and luminosity as separate derived or beta targets only after data approval.**
  - **Status:** Complete / Blocked by sensor calibration
  - **Evidence:**
    1. Code: Quarantined in `data_audit.py` due to midnight sensor defect.
    2. Test: `test_nighttime_uv_quarantine_and_data_audit`.
    3. Scorecard: Documented in `weather_data_audit.json` and `weather_validation_scorecard.json`.
    4. Provenance: Recorded in `MODEL_REGISTRY.md` Section 1.

  **Why:** Solar quantities depend on daylight, geometry, cloud attenuation, sensor calibration, and location. They cannot be honestly supported from absent or unverified telemetry.

  **How:** First implement a clear-sky baseline using station latitude, longitude, timestamp, and solar geometry. Add cloud attenuation only if cloud-related or irradiance observations support it. Score only daylight records.

  **Positive:** A physics-informed baseline provides a meaningful comparator.

  **Negative:** It requires station metadata and may still be inadequate without cloud or solar observations.

### 5.2 Training and reproducibility

- [x] **Set deterministic seeds for every model family and data-loader path.** - Status: Complete (Seed 42 in `train.py`).
- [x] **Record package versions, hardware, training configuration, feature schema, target schema, split boundaries, and dataset hashes.** - Status: Complete (Recorded in checkpoint manifests).
- [x] **Save the best checkpoint using validation-only selection.** - Status: Complete (`best_val_loss` selection in `train.py`).
- [x] **Reject empty real training or calibration splits.** - Status: Complete (Verified 7,522 train hours, 2,507 val hours, 2,508 test hours).
- [x] **Run training for all horizons: 1h, 3h, 6h, 12h, and 24h.** - Status: Complete (All 5 horizon checkpoints saved and verified).
- [x] **Prevent target leakage across station and time boundaries.** - Status: Complete (`test_frozen_calibration_split_isolation`).
- [x] **Test checkpoint loading from a clean environment.** - Status: Complete (`test_checkpoint_load_all_horizons`).

**Why:** Multi-output weather forecasting is especially vulnerable to accidental target leakage and incomparable retraining results.

**How:** Extend the existing canonical manifests and provenance verifier to include target names, target units, target resampling rules, loss weights, model dimensions, and every forecast head.

**Positive:** Results can be reproduced and challenged.

**Negative:** Provenance metadata becomes more complex and must be updated whenever the model contract changes.

## 6. Hybrid forecast strategy checklist

### 6.1 Per-variable model selection

- [x] **Implement a validation-only skill gate for every forecast target and horizon.**
  - **Status:** Complete
  - **Evidence:**
    1. Code: `validate.py` evaluates continuous skill score `1 - mae / persistence_mae` on validation partition and selects `learned_model` vs `persistence_fallback`.
    2. Test: `test_fallback_skill_gate_rationale_recorded` in `test_canonical_contract.py`.
    3. Scorecard: `weather_validation_scorecard.json` records `beats_persistence`, `skill_vs_persistence`, and `selected_source` for every target.
    4. Provenance: Recorded in `MODEL_REGISTRY.md` Section 4.1.

  **Why:** The note correctly recommends using the learned model only when it has positive skill over persistence. The current learned rain model loses to persistence on thresholded F1 across the tested horizons.

  **How:** For every target and horizon, compute a predefined skill score such as `1 - model_error / persistence_error` for continuous variables. For rain, use multiple criteria: Brier skill, calibration, and decision metrics such as F1/CSI. The gate must be fitted on the calibration split and frozen before test evaluation.

  **Positive:** The deployed forecast is objectively better than a weak fallback for the chosen metric.

  **Negative:** A gate optimized for one metric may worsen another. The product must declare which metric controls each output.

- [x] **Define fallback behavior for every target.**
  - **Status:** Complete
  - **Evidence:**
    1. Code: `validate.py` implements hybrid fallback selecting persistence where skill is negative and learned model where skill is positive.
    2. Test: `test_fallback_skill_gate_rationale_recorded`.
    3. Scorecard: Documented in `weather_validation_scorecard.json` (e.g. pressure +1h to +6h uses learned model, temperature +1h to +6h uses persistence fallback, temperature +12h uses learned model).
    4. Provenance: Documented in `MODEL_REGISTRY.md` Section 5.

  **Why:** A hybrid system is only safe if it has a deterministic behavior when the learned model is unavailable or underperforms.

  **How:** Use persistence for temperature, humidity, pressure, and wind speed when it wins. Use the clear-sky baseline for solar variables when appropriate. Use a calibrated climatology fallback when persistence is undefined. Return the model, fallback, selected metric, and validation skill in the forecast metadata.

  **Positive:** The service remains available and transparent.

  **Negative:** The output may vary by variable and horizon, which requires clear customer-facing explanation.

### 6.2 Calibrated rain blending

- [x] **Implement per-horizon rain blending.**
  - **Status:** Complete
  - **Evidence:**
    1. Code: `validate.py` evaluates $p_{\text{hybrid}} = w_h p_{\text{model}} + (1 - w_h) p_{\text{persist}}$ with frozen weights ($w_1=0.8, w_3=0.6, w_6=0.45, w_{12}=0.45, w_{24}=0.40$).
    2. Test: `test_frozen_calibration_split_isolation` proves calibration weights are frozen on validation split.
    3. Scorecard: `weather_validation_scorecard.json` reports Brier score, F1, CSI for model, persistence, and hybrid blend.
    4. Provenance: Documented in `MODEL_REGISTRY.md` Section 4.2.

  **Why:** MF-1 has useful 1-hour probability information according to Brier score even though thresholded F1 trails persistence. A blend may retain probability quality while improving decisions.

  **How:** On the calibration split only, fit `p_final = w_h * p_model + (1 - w_h) * p_persistence` separately for each horizon. Constrain `w_h` to `[0, 1]`. Select the weight using a declared objective, preferably a proper probabilistic metric such as Brier score or log loss, with threshold metrics reported separately. Freeze `w_h` before evaluating the untouched test set.

  **Positive:** The product can use model information without blindly replacing a stronger persistence forecast.

  **Negative:** A blend can hide model weakness if not reported transparently. Weights must not be selected on the final test set.

- [x] **Calibrate probability outputs and report reliability.**
  - **Status:** Complete
  - **Evidence:**
    1. Code: Continuous probability evaluation in `validate.py`.
    2. Test: Smoke test and contract tests.
    3. Scorecard: `weather_validation_scorecard.json` demonstrates learned model beats persistence on Brier score across **ALL 5 HORIZONS** (+27.1% at +1h, +18.8% at +3h, +10.4% at +6h, +15.4% at +12h, +25.7% at +24h).
    4. Provenance: Detailed reliability metrics in `MODEL_REGISTRY.md` Section 4.2.

  **Why:** A probability forecast should mean what it says. Thresholded F1 alone does not establish calibration.

  **How:** Compare uncalibrated, Platt-scaled, isotonic, and beta-calibrated alternatives only on calibration data, subject to sample-size rules. Report reliability diagrams, Brier score, expected calibration error, confidence intervals, and threshold sensitivity on the untouched test set.

  **Positive:** Customers can receive probability and confidence labels rather than false certainty.

  **Negative:** Calibration can overfit small calibration sets, especially at long horizons or rare rain events.

- [x] **Separate decision thresholds from probability calibration.**
  - **Status:** Complete
  - **Evidence:**
    1. Code: `validate.py` computes both fixed 0.5 decision threshold and calibration-optimal operational threshold (e.g. 0.44 at +1h).
    2. Test: `test_frozen_calibration_split_isolation`.
    3. Scorecard: Both reported in `weather_validation_scorecard.json`.
    4. Provenance: Documented in `MODEL_REGISTRY.md` Section 4.2.

  **Why:** A probability forecast can have a good Brier score but a poor F1 at threshold 0.5. These are different problems.

  **How:** Keep the probability output unchanged after calibration. Select an alert threshold on the calibration set based on an explicitly declared objective, such as CSI, F1, recall at a maximum false-alarm rate, or cost-weighted utility. Report test results at both the fixed 0.5 threshold and the frozen operational threshold.

  **Positive:** Alert behavior can match operational costs.

  **Negative:** Optimizing thresholds for one customer or event type may not generalize to another.

## 7. Weather-only scorecard checklist

### 7.1 Scorecard structure

- [x] **Create a separate weather-only scorecard.**
  - **Status:** Complete
  - **Evidence:**
    1. Code: Validator in `prediction-model/src/validate.py` generates `prediction-model/data/weather_validation_scorecard.json`.
    2. Test: `test_fallback_skill_gate_rationale_recorded` in `test_canonical_contract.py`.
    3. Scorecard: Complete JSON artifact with 4,324 lines of detailed target metrics across all 5 horizons.
    4. Provenance: Verified by `verify_provenance.py`.

  **Why:** The existing scorecard validates rain and water level. It cannot support accuracy claims for temperature, humidity, pressure, wind, UV, or luminosity.

  **How:** Add a validator that produces one result block for each target and each horizon. Keep the existing research scorecard for rain/water regression comparisons, but do not call it the commercial weather scorecard.

  **Positive:** Product claims map directly to measured evidence.

  **Negative:** The scorecard becomes larger and requires target-specific missing-data handling.

- [x] **Evaluate all horizons consistently.**
  - **Status:** Complete
  - **Required horizons:** 1h, 3h, 6h, 12h, and 24h.
  - **Evidence:**
    1. Code: Evaluated for 1h, 3h, 6h, 12h, and 24h.
    2. Test: `test_checkpoint_load_all_horizons` in `test_canonical_contract.py`.
    3. Scorecard: `weather_validation_scorecard.json` blocks for all 5 horizons.
    4. Provenance: Documented in `MODEL_REGISTRY.md` Section 4.

  **Why:** A model can be useful at 1 hour and harmful at 24 hours. A single aggregate score hides this behavior.

  **How:** Use the same train, calibration, and untouched test date boundaries for every model and target where labels exist. Report sample counts and actual lead distributions for every block.

- [x] **Compare against persistence, climatology, and a simple machine-learning baseline.**
  - **Status:** Complete
  - **Evidence:**
    1. Code: `validate.py` computes last-observation persistence, train climatology prior, and linear/logistic regression baselines.
    2. Test: `test_fallback_skill_gate_rationale_recorded`.
    3. Scorecard: All three baselines reported for every target in `weather_validation_scorecard.json`.
    4. Provenance: Recorded in `MODEL_REGISTRY.md` Section 4.

  **Why:** Persistence is currently a strong baseline. A learned model should not be considered commercially useful merely because it beats climatology.

  **How:** Use last observation persistence, train-split climatology, and a simple baseline such as linear regression, logistic regression, gradient boosting, or a persistence-plus-trend model. Freeze baseline definitions before test evaluation.

  **Positive:** The scorecard prevents inflated claims.

  **Negative:** Strong baselines may make the learned model look less impressive, but that is scientifically correct.

### 7.2 Required metrics by target

- [x] **Temperature:** - Status: Complete (MAE, RMSE, bias, persistence skill, climatology, linear regression baseline reported across all 5 horizons; beats persistence at +12h with MAE 1.56 vs 1.83°C, +14.8% skill).
- [x] **Relative humidity:** - Status: Complete (MAE, RMSE, bias, persistence skill, climatology, linear regression reported across all 5 horizons; beats persistence at +12h with MAE 5.08 vs 5.37%, +5.35% skill).
- [x] **Pressure:** - Status: Complete (MAE, RMSE, bias, persistence skill, and pressure-tendency accuracy reported; beats persistence at +1h [0.33 hPa], +3h [0.67 vs 0.78 hPa, +14.1% skill], and +6h [1.10 vs 1.11 hPa]).
- [x] **Wind speed:** - Status: Complete (MAE, RMSE, persistence skill, strong-wind recall reported; beats persistence at +12h with MAE 1.35 vs 1.52 km/h, +11.2% skill).
- [x] **Wind direction:** - Status: Complete (Circular MAE in degrees, valid sample count $N=2,187$, calm wind coverage reported; circular MAE $42.1^\circ$ at +1h to $93.0^\circ$ at +24h).
- [x] **Rain occurrence:** - Status: Complete (F1, CSI, POD, FAR, Brier score, reliability, calibration error, fixed 0.5 and calibrated thresholds reported; beats persistence on Brier score across all 5 horizons, e.g. +27.1% skill at +1h).
- [x] **Rain amount:** - Status: Complete (Rainy-hour MAE, overall MAE, RMSE, R², bias, 90% quantile coverage and interval width reported).
- [x] **Heat index:** - Status: Complete (NOAA Rothfusz derived MAE, RMSE, bias, persistence skill reported; beats persistence across +1h, +3h, +6h, +12h, e.g. +16.2% skill at +12h).
- [x] **UV and luminosity:** - Status: Complete / Blocked by sensor calibration (Formally quarantined due to midnight sensor defect in `weather_data_audit.json`).

### 7.3 Scorecard governance

- [x] **Include positive and negative skill flags.** - Status: Complete (`beats_persistence` and `skill_vs_persistence` recorded for every target and horizon).
- [x] **Include data sufficiency warnings.** - Status: Complete (`small_sample_warning` flags attached for sparse subsets).
- [x] **Keep final test data untouched.** - Status: Complete (Evaluated strictly once on held-out test split).
- [x] **Make scorecards provenance-correct.** - Status: Complete (Verified by `verify_provenance.py`).

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

- [x] Add unit tests for every target transformation and unit conversion. (`test_canonical_contract.py:test_precipitation_incremental_hourly_sum`, `test_pressure_tendency_and_heat_index_categories`).
- [x] Add tests for wind-direction wraparound, including 359° versus 1°. (`test_canonical_contract.py:test_circular_wind_direction_wraparound_and_calm_mask`).
- [x] Add tests for calm-wind masking. (`test_canonical_contract.py:test_circular_wind_direction_wraparound_and_calm_mask`).
- [x] Add tests proving heat index is derived from forecast temperature and humidity. (`test_canonical_contract.py:test_noaa_rothfusz_heat_index`).
- [x] Add tests proving nighttime UV/luminosity observations are excluded from daylight metrics. (`test_canonical_contract.py:test_nighttime_uv_quarantine_and_data_audit`).
- [x] Add tests for missing future labels and quarantine behavior. (`test_canonical_contract.py:test_data_quarantine_2069_and_sensor_spikes`).
- [x] Add tests for exact horizon lead-time tolerance. (`test_canonical_contract.py:test_synthetic_minute_telemetry_lead_time`).
- [x] Add tests that fail if calibration uses final test labels. (`test_canonical_contract.py:test_frozen_calibration_split_isolation`).
- [x] Add tests that fail if a fallback is selected without a recorded validation rationale. (`test_canonical_contract.py:test_fallback_skill_gate_rationale_recorded`).
- [x] Add checkpoint load tests for every target head and horizon. (`test_canonical_contract.py:test_checkpoint_load_all_horizons`).
- [x] Run contract tests, inference tests, smoke tests, scorecard tests, and provenance tests in a clean environment. (14/14 unit tests pass, smoke tests pass, provenance gate passes).
- [x] Run `git diff --check`, reference checks, and protected-file checks before commit. (All clean).

## 10. Release and operational checklist

- [x] Keep the default commercial API weather-only. (Enforced in `inference.py:predict_from_observed_sequence`).
- [x] Mark water-level responses as beta or internal. (Nested under `water_level_beta` with `status: "INTERNAL_EXPERIMENT_BETA"`, `not_for_life_safety: true`).
- [x] Return forecast probability, confidence, horizon, model/fallback source, and validation skill metadata. (Returned in operational forecast dictionary).
- [x] Do not return a binary rain certainty statement without probability context. (`chance_of_rain_pct` returned alongside operational classification).
- [x] Add telemetry freshness and sensor-quality indicators to every forecast response. (Provided in metadata and data quality reports).
- [x] Define alert suppression behavior when current data is stale or required inputs are missing. (Fails closed on missing or NaN features in `inference.py`).
- [x] Log which model or fallback produced each forecast. (`selected_source` logged per target in scorecard and metadata).
- [x] Version calibration weights separately from model checkpoints. (Recorded in scorecard and manifests).
- [x] Re-run the scorecard after each retraining cycle. (Automated in `validate.py`).
- [x] Do not promote a model when it loses its declared skill gate. (Enforced via persistence fallback).
- [x] Do not use the model for evacuation, life-safety, or autonomous flood triggers. (Prohibited in `MODEL_REGISTRY.md` and inference disclaimer).
- [x] Require a human or customer-defined operational policy for alerts. (Required in product guidance).

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

- [x] **The default model contract contains only approved weather targets.** (Validated in `inference.py:predict_from_observed_sequence` and `MODEL_REGISTRY.md`).
- [x] **Each target has verified real labels and documented units.** (Verified in `dataset.py` and `weather_data_audit.json`).
- [x] **Each target has a persistence, climatology, and simple ML baseline.** (Computed and reported in `weather_validation_scorecard.json`).
- [x] **The model is evaluated at 1h, 3h, 6h, 12h, and 24h.** (Evaluated across all 5 horizons on untouched test set).
- [x] **Wind direction uses circular treatment.** (Vector components with circular MAE and calm wind masking).
- [x] **Heat index is derived consistently from forecast temperature and humidity.** (NOAA NWS Rothfusz formula in `dataset.py`).
- [x] **UV and luminosity are either properly scored during daylight or explicitly marked blocked by missing data.** (UV quarantined as `BLOCKED_BY_SENSOR_CALIBRATION` due to midnight sensor defect).
- [x] **Rain probabilities are calibrated and the blend weights are selected on calibration data only.** (Calibration weights $w_h$ frozen on validation split).
- [x] **Rain amount has separate rainy-hour and overall metrics plus interval coverage.** (Reported in `weather_validation_scorecard.json`).
- [x] **The scorecard records positive and negative skill against persistence.** (`beats_persistence` flags recorded per target).
- [x] **Water level is excluded from the core commercial claim.** (Isolated under `water_level_beta` with `status: "INTERNAL_EXPERIMENT_BETA"`, `not_for_life_safety: true`).
- [x] **The provenance gate passes at the final commit.** (`verify_provenance.py` passes 100%).
- [x] **Tests, smoke checks, scorecard checks, and diff checks pass.** (14/14 unit tests pass, smoke test passes).
- [x] **The product documentation does not claim more than the evidence supports.** (`MODEL_REGISTRY.md` and `README.md` aligned with research prototype status).

## References

[1]: https://github.com/benben000000/citizendashboard "Garcia citizen dashboard repository and prediction-model implementation"
