# Major Prediction-Model Improvement Plan

## Executive summary

The repository is now structurally mature: it has target-specific routing, rolling-fold metadata, uncertainty reports, anomaly reports, information-ceiling analysis, 75 passing predictive-quality tests, and passing provenance and bundle gates.

However, the model is not yet a universal GO because:

- temperature loses to persistence at 1h, 3h, and 6h;
- wind direction loses to persistence at every horizon;
- humidity and pressure are classified information-limited;
- UV is blocked by sensor calibration;
- luminosity is still beta;
- headline point metrics have not materially improved after the recent framework work;
- the latest reports describe the limitations, but do not yet demonstrate a major forecast-skill breakthrough.

This plan is designed to pursue a **major improvement**, not merely add more tests or reports.

The most important conclusion is:

> It is not realistic to make every target GO while using only one local weather station and no spatial or external atmospheric information.

A single station observes local history but cannot directly observe an approaching weather system before it reaches the station. Short-horizon persistence can therefore beat a sophisticated model, especially for temperature and wind direction. To make all major targets GO, the data constraint must be expanded to include spatial context such as nearby stations, radar, satellite, numerical weather predictions, or at minimum a second local station.

The plan therefore has two tracks:

- **Track A — Local-only maximum:** improve the existing system without external data, while accepting that some targets may remain information-limited.
- **Track B — Major-improvement route:** add spatial or external predictors and build a multi-source nowcasting system. This is the only credible route toward making all target families GO.

# 1. Target definition

## GO means more than passing tests

A target/horizon is GO only when it satisfies all of these:

1. beats or is statistically non-inferior to the strongest baseline;
2. passes at least three rolling-origin periods;
3. passes the untouched final test period;
4. improves or maintains worst-regime performance;
5. has calibrated uncertainty;
6. has no unresolved data leakage;
7. has valid physical bounds;
8. has complete provenance;
9. has tested rollback;
10. has acceptable latency and operational complexity.

A green repository test suite alone does not make a weak forecast GO.

## Current target-specific goal

The objective is not to force every target into a candidate model. The objective is to make every target **operationally GO** by assigning the best proven source:

```text
candidate model
baseline model
persistence
derived formula
external-data model
blocked until calibration
```

If the requirement is that every target must be predicted by a newly trained ML candidate, the current local-only data is insufficient for an honest guarantee.

# 2. Phase 0 — Freeze the current baseline and define success thresholds

## Goal

Prevent major-model experiments from being judged against moving targets.

## Actions

1. Freeze the current scorecard and active policy.
2. Record current metrics per target, horizon, fold, and regime.
3. Record the current raw-data hashes.
4. Store all experiment configurations and seeds.
5. Define minimum improvement thresholds before training:

### Suggested thresholds

- continuous-target MAE: at least 5% improvement over the strongest baseline, or statistically non-inferior with a meaningful operational advantage;
- rain Brier score: at least 5% relative improvement plus calibration stability;
- wind-direction circular MAE: at least 5% improvement over persistence outside calm-wind conditions;
- uncertainty: declared 80% coverage within tolerance and useful interval width;
- anomalies: recall and false-alarm budget defined per event class;
- worst-fold regression: no more than the approved tolerance;
- inference latency: fixed budget agreed before training.

6. Define a final untouched test period that is never used for model selection.

## Gate

No model is called improved merely because it passes a structural test.

# 3. Phase 1 — Repair the data limitation first

## Why this is the highest-leverage step

The current information-ceiling report says the model is limited for short-horizon temperature, humidity, pressure, and wind direction. More layers cannot reconstruct atmospheric state that the station never observed.

## Track A: local-only data expansion

If external data is prohibited, improve the local dataset by:

1. extending the history over multiple complete years;
2. adding more nearby sensors owned by the project;
3. adding station-health and calibration logs;
4. preserving raw, timestamped, immutable source files;
5. recording sensor relocation and maintenance events;
6. adding high-frequency data if the sensors support it;
7. retaining missingness and quality flags instead of silently filling them.

## Track B: spatial and external data expansion

To pursue a major improvement across all targets, add:

- nearby weather stations;
- regional pressure and temperature observations;
- radar precipitation or rain-gauge grids;
- satellite cloud and infrared features;
- numerical weather prediction fields;
- lightning or storm-track data where legally and technically available;
- elevation, land-cover, coastline, and station metadata.

Use time-lagged spatial context only. At issue time, the feature builder must use only data that would have been available in real time.

## Data ingestion requirements

Create separate raw zones for:

```text
local_station_raw
nearby_station_raw
radar_raw
satellite_raw
nwp_raw
metadata_raw
```

For each source record:

- source name;
- retrieval time;
- valid time;
- issue time for forecasts;
- spatial coordinates or grid index;
- source version;
- unit;
- missingness;
- quality status;
- checksum.

## Gate

Do not train with external data until a time-availability audit proves the model could have accessed every feature at inference time.

# 4. Phase 2 — Build a proper nowcasting feature cube

## Goal

Move from one-station sequence modeling to a causal, multi-source feature representation.

## Feature groups

### Local station features

- recent values and lags;
- rolling trends;
- rolling volatility;
- first and second differences;
- rain persistence;
- dry-spell duration;
- wind vector history;
- sensor-quality indicators;
- time-of-day and seasonal harmonics.

### Spatial station features

- nearby station values;
- spatial gradients;
- upwind/downwind features;
- pressure differences;
- temperature and humidity gradients;
- recent regional trend;
- station consensus and disagreement.

### Radar and precipitation features

- precipitation intensity near the station;
- rain-cell distance;
- rain-cell motion vector;
- precipitation accumulation;
- echo-top or intensity change;
- storm approach direction.

### Satellite features

- cloud-cover fraction;
- cloud-top temperature;
- cloud-motion features;
- infrared brightness;
- daylight and solar geometry.

### NWP features

- temperature;
- humidity;
- pressure;
- wind vector;
- precipitation probability;
- precipitation amount;
- radiation or cloud variables;
- forecast spread where available.

## Feature-causality tests

For every feature:

1. declare its valid time;
2. declare its retrieval delay;
3. remove future records;
4. rebuild the feature row;
5. assert the as-of feature vector is unchanged.

## Gate

Any future-sensitive or revised-after-issue feature blocks promotion.

# 5. Phase 3 — Create a stronger model architecture

## Goal

Use a model appropriate for multi-source local nowcasting while remaining feasible on local compute.

## Recommended architecture

Use a compact mixture of specialized components rather than one oversized shared network:

```text
local encoder
spatial-station encoder
radar/satellite encoder
NWP encoder
        ↓
causal fusion layer
        ↓
target-specific forecast heads
        ↓
calibration and policy layer
```

The model should include:

- causal temporal attention or gated temporal convolution;
- spatial attention over nearby stations;
- vector wind representation;
- hurdle precipitation head;
- quantile or distributional heads;
- target-specific residual heads;
- mixture or gating weights by horizon and regime.

## Local-compute discipline

- train one horizon at a time;
- cache feature cubes;
- use compact hidden dimensions;
- use early stopping;
- use mixed precision where stable;
- use 3–5 seeds for finalists only;
- keep a tree baseline for every target;
- measure CPU latency and memory;
- avoid large global architectures unless they provide measurable gain.

## Gate

A more complex model must beat a simpler model enough to justify its operational cost.

# 6. Phase 4 — Solve temperature forecasting

## Current weakness

The candidate loses to persistence at 1h, 3h, and 6h.

## Model strategy

Use a residual mixture:

```text
forecast = persistence or NWP baseline + learned residual
```

Train separate residual components for:

- local diurnal persistence;
- regional advection;
- NWP correction;
- storm-transition regimes.

Add a regime gate based on:

- recent temperature trend;
- pressure trend;
- wind direction and speed;
- nearby station gradient;
- cloud/radar evidence;
- NWP disagreement.

## Required tests

Compare:

- local persistence;
- damped persistence;
- autoregression;
- local-only residual model;
- spatial residual model;
- NWP-corrected residual model;
- compact ensemble.

## GO gate

Temperature becomes GO only if it improves 1h, 3h, and 6h across rolling folds while maintaining 12h and 24h performance. If external data is not allowed and this fails, retain baseline and classify the target as information-limited.

# 7. Phase 5 — Solve wind direction

## Current weakness

Wind direction is worse than persistence at every tested horizon.

## Model strategy

Predict vector components:

```text
u = speed × cos(direction)
v = speed × sin(direction)
```

Use spatial pressure gradients, nearby station vectors, NWP wind vectors, and storm-motion features where permitted.

Use a mixture of:

- persistence in calm conditions;
- local vector residual model;
- spatial vector model;
- NWP vector correction.

## Evaluation

Report:

- circular MAE;
- circular RMSE;
- vector-component MAE;
- error by wind-speed regime;
- calm-wind fallback rate;
- error during wind shifts;
- storm-transition performance.

## GO gate

Wind direction becomes GO only if the selected route beats persistence outside calm conditions and does not degrade operational behavior during calm conditions.

# 8. Phase 6 — Solve precipitation and rain events

## Current state

Rain occurrence is already stronger than persistence. The goal is to make the result robust, calibrated, and useful for amount prediction.

## Model strategy

Use a hurdle system:

```text
P(rain) × conditional amount
```

Add radar and spatial station features where available. Use separate heads for:

- rain occurrence;
- rain onset;
- rain cessation;
- conditional amount;
- heavy-rain exceedance.

## GO gate

Rain is GO only if:

- Brier and log loss improve or remain non-inferior;
- reliability is stable;
- heavy-rain recall passes;
- false-alarm rate is acceptable;
- amount prediction improves on rainy samples;
- event lead time is useful.

# 9. Phase 7 — Solve humidity and pressure

## Current limitation

Humidity and pressure are currently information-limited under local-only data.

## Model strategy

Humidity:

- dew-point and vapor-pressure features;
- temperature-humidity coupling;
- nearby station gradients;
- NWP humidity correction;
- rain and cloud context.

Pressure:

- regional pressure gradients;
- temporal pressure tendency;
- nearby station network;
- NWP synoptic fields;
- storm-track indicators.

## GO gate

Do not force candidate promotion. These targets become GO only when they beat their strongest baseline across rolling periods with valid spatial or external information.

# 10. Phase 8 — Fix UV and luminosity data quality

## UV

1. Physically calibrate the UV sensor.
2. Compare against a trusted reference during daylight.
3. Remove or quarantine nighttime artifacts.
4. Define the valid daylight operating range.
5. Rebuild labels only after calibration.
6. Train a daylight-only model first.
7. Evaluate seasonal and cloud-regime behavior.

UV remains blocked until calibration and forecast quality both pass.

## Luminosity

1. Validate sensor saturation and nighttime offsets.
2. Add solar geometry and daylight state.
3. Separate daylight and nighttime models.
4. Evaluate cloud transitions.
5. Validate seasonal behavior.

Promote only after the daylight beta passes a separate release gate.

# 11. Phase 9 — Improve uncertainty and anomaly quality

## Uncertainty

Use:

- conformal calibration;
- quantile heads;
- regime-aware calibration;
- ensemble spread;
- NWP spread if available.

Report:

- coverage;
- interval width;
- weighted interval score;
- coverage during extremes;
- coverage drift.

## Anomaly events

Separate:

- physical weather anomalies;
- sensor failures;
- forecast residual anomalies.

Use reviewed event labels and evaluate:

- recall;
- precision;
- false alarms per day;
- warning lead time;
- severity;
- season;
- horizon.

The current perfect-looking anomaly summary must be reviewed for sample count and label provenance before being treated as strong evidence. A `0.0 FA/day` result with very few labeled events is not sufficient by itself.

# 12. Phase 10 — Establish a champion/challenger evaluation system

## Models

Champion:

- current target-specific production route.

Challengers:

- local-only residual model;
- spatial model;
- radar-enhanced model;
- NWP-corrected model;
- compact ensemble.

## Evaluation protocol

1. Train on historical folds.
2. Calibrate on a disjoint period.
3. Evaluate on rolling tests.
4. Preserve one final untouched test.
5. Compare paired errors on identical rows.
6. Bootstrap differences by time block.
7. Evaluate worst regimes.
8. Generate target-specific decisions.

## Gate

No challenger replaces a champion because of one average metric or one test period.

# 13. Phase 11 — Expand the scorecard and evidence reports

## Required reports

For each target/horizon include:

- point metrics;
- baseline metrics;
- skill scores;
- rolling-fold metrics;
- confidence intervals;
- worst fold;
- worst regime;
- interval coverage;
- interval width;
- event metrics;
- missing and quarantined counts;
- sample counts;
- selected source;
- limitations;
- artifact hashes.

## Required warning

If an anomaly, uncertainty, or rare-event result has low sample count, mark it:

```text
LOW_EVIDENCE
```

Do not convert a perfect score from a tiny sample into a production claim.

# 14. Phase 12 — Enforce operational policy and rollback

The policy must remain target-specific:

```text
temperature: baseline until 1h/3h/6h gates pass
humidity: baseline until spatial/NWP model passes
pressure: baseline until spatial/NWP model passes
wind_speed: candidate only where rolling gates pass
wind_direction: persistence until vector model passes
rain_occurrence: candidate where calibration and event gates pass
precipitation_amount: candidate only where rainy/heavy-rain gates pass
heat_index: derived from selected temperature/humidity
uv_index: blocked until sensor calibration passes
light_intensity: daylight beta until full validation
```

Test rollback under:

- feature drift;
- sensor missingness;
- calibration failure;
- target degradation;
- artifact mismatch;
- policy mismatch;
- provenance failure.

# 15. Phase 13 — Provenance and release closure

## Required convention

Use separate fields for:

```text
parent_code_commit
code_commit
artifact_commit
release_commit
```

Artifacts generated from code commit `X` must not claim they were generated from a later commit `Y` unless they were actually regenerated under `Y`.

The release report must explicitly say whether the final GitHub tip is:

- the code commit;
- the artifact commit;
- a metadata-only child commit.

## Release sequence

1. Commit source and tests.
2. Record the code commit.
3. Train all five horizons.
4. Generate reports and policies.
5. Verify hashes.
6. Create artifact release commit.
7. Verify metadata convention.
8. Run all tests.
9. Run strict provenance.
10. Review policy and scorecard.
11. Push only after all promotion gates pass.

# 16. Phase 14 — Major-improvement acceptance gates

## Local-only route

The local-only route can be called complete only if it:

- improves selected targets;
- honestly marks information-limited targets;
- maintains safe fallback;
- passes calibration and anomaly evidence gates;
- provides reproducible reports.

It cannot honestly guarantee all targets will beat persistence.

## Multi-source route

The multi-source route can make a credible attempt at all-target GO only if:

- spatial or external inputs are available in real time;
- all input availability is causal;
- the new data survives missing-source failures;
- every target beats its baseline across rolling folds;
- uncertainty is calibrated;
- anomaly events have enough labels;
- production latency and cost are acceptable;
- rollback is tested.

# 17. Execution order

Execute in this order:

1. Freeze current baseline and final test period.
2. Audit the sample counts and provenance of the new reports.
3. Repair any scorecard or metric claims based on low evidence.
4. Expand local station history and quality metadata.
5. Decide whether the local-only constraint remains mandatory.
6. If mandatory, complete local-only residual/vector/hurdle models and accept information-limited outcomes.
7. If a major all-target improvement is required, add spatial stations and NWP/radar/satellite inputs.
8. Build the causal feature cube.
9. Train champion/challenger models.
10. Evaluate rolling folds and final holdout.
11. Calibrate uncertainty.
12. Evaluate anomalies.
13. Generate target-specific policy.
14. Run rollback and provenance gates.
15. Promote only passing target/horizon combinations.

# 18. Definition of done

The major-improvement program is complete only when:

1. the data limitation is addressed or explicitly accepted;
2. every target has a strong baseline;
3. every feature has a causal availability contract;
4. at least three rolling-origin folds are evaluated;
5. temperature is improved at 1h, 3h, and 6h or marked information-limited honestly;
6. wind direction beats persistence or is marked information-limited honestly;
7. rain occurrence and amount pass event and calibration gates;
8. humidity and pressure are evaluated with sufficient information;
9. UV calibration is repaired or remains blocked;
10. luminosity passes daylight and nighttime validation or remains beta;
11. uncertainty intervals have adequate coverage and useful width;
12. anomaly metrics have sufficient labeled events;
13. champion/challenger comparisons are complete;
14. scorecards include low-evidence warnings;
15. target-specific inference and rollback work;
16. provenance and hashes are unambiguous;
17. all tests pass from a clean checkout;
18. only proven target/horizon combinations are promoted.

## Final realistic conclusion

The fastest route to a major improvement is not a larger neural network. It is:

1. improve the information available at issue time;
2. build causal spatial and regional features;
3. use target-specific residual and vector models;
4. evaluate with rolling-origin and worst-regime discipline;
5. calibrate uncertainty;
6. promote per target and horizon.

If the system must remain local-station-only, the correct goal is **best possible local forecasting with explicit information-limited targets**, not forcing every target to GO. If the business requirement is that every target must become GO, the spatial/external-data route is required.
