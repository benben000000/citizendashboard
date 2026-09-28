# Prediction-Model Next Implementation Plan

## Objective

Improve forecast accuracy using spatial and external data **without weakening operational safety, provenance, or commercial-use compliance**.

The current repository is structurally healthy, but the predictive breakthrough is incomplete:

- Rain probability improves over persistence.
- Temperature improves only at selected horizons.
- Wind direction does not improve consistently.
- Humidity is still persistence-routed.
- UV and luminosity remain limited or quarantined.
- Spatial/external ingestion is not implemented in `origin/main`.
- Documentation mentions Himawari-9 and radar, but there is no production source registry, license gate, or causal external feature cube.

The implementation must therefore proceed in controlled phases. **No external source is allowed into training or inference until its rights are explicitly approved.**

---

# 1. Non-negotiable acceptance rules

## 1.1 Scientific rules

1. Use real telemetry only.
2. Preserve raw local telemetry and archive history.
3. Split data chronologically.
4. Keep the 48-hour embargo.
5. Never use future observations, revised products, or post-issue source data.
6. Compare every candidate against the existing persistence and climatology baselines.
7. Promote per target and per horizon, never globally by default.
8. Mark targets **Information Limited** when improvement is not demonstrated.
9. Report confidence intervals, sample counts, worst regimes, and missing-source performance.
10. Do not claim world-class performance from a single test period or a small anomaly sample.

## 1.2 Data-rights rules

A source may enter production only when its registry decision is one of:

```text
APPROVED_FREE_COMMERCIAL
APPROVED_WITH_EXPLICIT_NO_COST_PERMISSION
```

The review must explicitly cover:

- commercial use;
- model training;
- commercial inference;
- derived features;
- caching;
- attribution;
- rate limits;
- redistribution restrictions;
- terms version or review date.

The following states are blocked from production:

```text
UNKNOWN_BLOCKED
PENDING_REVIEW
RESEARCH_ONLY
NOT_ALLOWED
```

Technical accessibility is not evidence of legal eligibility.

## 1.3 Operational rules

1. External data is never fetched during live model inference.
2. External-source failures must degrade to a verified local-only model.
3. License revocation or expiry must disable the source at the policy layer.
4. Any artifact hash or provenance mismatch must fail closed.
5. Every prediction must identify the source mode used.
6. Rollback must work independently for each target and horizon.

---

# 2. Current baseline to protect

Create a frozen baseline record from remote commit:

```text
9b58618900940d43ef65b093ca9527e2e6ff4d43
```

Record:

- test counts;
- scorecard metrics;
- inference policy;
- bundle hashes;
- provenance result;
- raw-data hashes;
- current operational routing.

Current baseline routing includes:

| Target | Current behavior |
|---|---|
| Rain | Learned/persistence blend by horizon |
| Temperature | Learned only at selected horizons; persistence elsewhere |
| Pressure | Learned only at 3h |
| Wind speed | Learned only at 12h |
| Wind direction | Persistence fallback |
| Humidity | Persistence fallback |
| Heat index | Derived from selected temperature and humidity |
| UV | Quarantined/limited |
| Luminosity | Limited/beta |

No experiment may overwrite this baseline. All new work must be candidate-only until promotion.

---

# 3. Phase 0 — Establish a clean implementation branch

## Goal

Prevent the divergent local branch from being confused with the release branch.

## Actions

1. Create a feature branch from the current remote head.
2. Preserve the archive branch and raw data.
3. Do not modify dashboard UI or unrelated services.
4. Record the baseline commit in the implementation notes.
5. Create a separate candidate artifact directory if needed.

## Deliverables

```text
prediction-model/docs/multisource_implementation_status.md
prediction-model/data/candidate_artifacts/baseline_release_reference.json
```

## Exit criteria

- Clean checkout passes the existing suite.
- Baseline hashes are recorded.
- No unrelated files are changed.

---

# 4. Phase 1 — Implement the license and source registry

## Goal

Make data-rights review an executable release gate rather than a document-only practice.

## Files to add

```text
prediction-model/data/external_source_registry.json
prediction-model/src/external_source_registry.py
prediction-model/src/test_external_source_registry.py
```

## Minimum registry schema

Each source record must contain:

```json
{
  "source_id": "provider_product_version",
  "provider": "...",
  "product": "...",
  "source_url": "https://...",
  "license_url": "https://...",
  "terms_version_or_date": "...",
  "commercial_use_allowed": false,
  "training_use_allowed": false,
  "commercial_inference_allowed": false,
  "derived_features_allowed": false,
  "raw_caching_allowed": false,
  "redistribution_allowed": false,
  "attribution_required": false,
  "rate_limits": "...",
  "permission_reference": "...",
  "decision": "UNKNOWN_BLOCKED",
  "reviewed_at_utc": "...",
  "review_due_utc": "..."
}
```

## Gate behavior

The registry module must:

- reject missing records;
- reject unknown decisions;
- reject expired reviews;
- reject contradictory fields;
- reject production use when training or commercial inference is false;
- return a structured reason for every denial;
- expose a deterministic `assert_production_eligible(source_id)` function.

## Tests

Test:

- missing source;
- unknown source;
- research-only source;
- personal-use-only source;
- commercial-use source;
- explicit permission source;
- expired approval;
- missing license URL;
- contradictory permission fields;
- source revocation.

## Exit criteria

The release pipeline fails if any training manifest references a source that is not approved.

---

# 5. Phase 2 — Build immutable source manifests and cache contracts

## Goal

Preserve source provenance and support reproducible historical replay.

## Files to add

```text
prediction-model/src/external_source_contract.py
prediction-model/src/external_cache.py
prediction-model/src/test_external_source_contract.py
```

## Required fields per object

- source ID;
- product version;
- valid time UTC;
- issue time UTC, if forecast data;
- retrieval time UTC;
- request parameters;
- response checksum;
- raw-object path;
- license decision;
- license registry hash;
- quality flags;
- latency;
- revision status;
- parser version.

## Caching rules

- Raw data is append-only.
- Raw files are never silently replaced.
- Revisions receive new object IDs.
- The feature builder must select only the version available by issue time.
- Production inference reads prepared, verified files only.

## Exit criteria

A replay from a fixed manifest produces the same normalized values and feature hash.

---

# 6. Phase 3 — Audit and select the first eligible source

## Goal

Choose the highest-value source with the lowest rights and implementation risk.

## Candidate order

### Candidate A — Project-owned or explicitly licensed nearby stations

Preferred because they provide regional gradients without image processing or complex product interpretation.

### Candidate B — Explicitly licensed radar/QPE

High value for rain onset, cessation, and heavy-rain anomalies, but only if the exact product is commercially eligible.

### Candidate C — Himawari-9 derived features

Potential value for cloud growth and convection, but product distribution and commercial training rights must be confirmed for the exact data path.

### Candidate D — PAGASA NWP or other approved NWP

Potentially valuable at 6h–24h, but issue-time replay and usage rights must be confirmed.

## Mandatory decision

Do not implement source-specific production adapters for a source whose status is `UNKNOWN_BLOCKED`.

If no candidate passes the gate, stop at local-only modeling and document the blocker. Do not substitute RainViewer or another source merely because it is easy to access.

## Deliverable

```text
prediction-model/docs/source_license_decisions.md
```

---

# 7. Phase 4 — Integrate approved nearby stations

## Goal

Improve temperature, pressure, humidity, wind, and rain context with approved regional telemetry.

## Adapter

```text
prediction-model/src/external_sources/nearby_stations.py
```

## Feature design

For each nearby station:

- distance;
- bearing;
- elevation difference;
- recent values;
- lags;
- rolling mean;
- rolling standard deviation;
- local trend;
- missingness;
- quality flag;
- telemetry age.

Regional aggregates:

- inverse-distance weighted values;
- nearest valid station;
- regional mean and median;
- station disagreement;
- gradient relative to the target station;
- regional wind-vector mean;
- regional wind-vector dispersion.

## Safety

- All station features must be timestamped in UTC.
- Only observations available at the issue time are eligible.
- Future or corrected observations must not leak into historical training.
- Station outages must remain explicit missingness rather than imputed certainty.

## Evaluation

Compare:

```text
local-only baseline
local + nearby-station candidate
```

Use identical rolling-origin folds and untouched final test data.

## Promotion criteria

Promote per target/horizon only when:

- the primary metric improves over persistence;
- improvement is present in a majority of rolling folds;
- no severe degradation occurs in the worst fold;
- missing-source fallback remains safe;
- uncertainty calibration does not deteriorate materially;
- the improvement is not explained by leakage.

---

# 8. Phase 5 — Integrate approved radar/QPE

## Goal

Improve precipitation nowcasting and rain anomaly detection.

## Adapter

```text
prediction-model/src/external_sources/radar.py
```

## Initial feature set

At multiple radii around the target station:

- mean reflectivity;
- maximum reflectivity;
- rainy-pixel fraction;
- rain-cell distance;
- rain-cell bearing;
- intensity trend;
- recent accumulation;
- motion estimate;
- product latency;
- quality and coverage flags.

## Target order

1. rain occurrence;
2. rain onset;
3. rain cessation;
4. precipitation amount;
5. heavy-rain anomaly.

## Important restriction

RainViewer is not automatically acceptable for commercial production. It must remain blocked unless explicit permission covers the intended use. A public endpoint is not a commercial license.

## Promotion criteria

Radar must improve event-based metrics such as:

- Brier score;
- log loss;
- reliability;
- precision/recall at operational thresholds;
- onset lead time;
- false alarms per day;
- heavy-rain recall.

---

# 9. Phase 6 — Integrate Himawari-9 only if eligible

## Goal

Add cloud and convection context using numeric derived features before attempting deep image models.

## Adapter

```text
prediction-model/src/external_sources/himawari.py
```

## Initial features

- infrared brightness-temperature summaries;
- cloud-top cooling rate;
- cloud-cover fraction;
- visible reflectance where daylight-valid;
- water-vapor summaries;
- cloud motion;
- spatial gradients;
- solar-geometry flags;
- source quality and latency.

## Required validation

- exact product license review;
- historical availability check;
- issue-time replay;
- cloud and daylight missingness tests;
- causal feature mutation test;
- source-ablation evaluation.

## Promotion targets

- rain onset;
- convection/heavy-rain anomaly;
- temperature residuals;
- UV and luminosity only after separate sensor and daylight validation.

Do not add raw-image deep learning until numeric derived features demonstrate value and the data-rights record is complete.

---

# 10. Phase 7 — Integrate approved NWP as a residual model

## Goal

Use an approved forecast product as a regional prior rather than blindly copying it.

## Adapter

```text
prediction-model/src/external_sources/nwp.py
```

## Model form

```text
local forecast = approved NWP forecast + learned local residual
```

## Required fields

- forecast issue time;
- valid time;
- lead time;
- model cycle;
- grid coordinates;
- interpolation method;
- revision status;
- source quality;
- license registry hash.

## Target order

- temperature;
- humidity;
- pressure;
- wind vector;
- precipitation probability.

## Gate

No NWP feature is eligible when only the later analysis or revised value is available. Training must replay the original operational availability.

---

# 11. Phase 8 — Build the causal feature cube

## Goal

Unify local and approved external data without future leakage.

## Row identity

```text
station_id
issue_time_utc
horizon_hours
```

## Feature metadata

Every feature group must include:

- source ID;
- valid time;
- issue time;
- retrieval time;
- availability cutoff;
- quality flag;
- missingness flag;
- license registry hash;
- raw object hash;
- parser version.

## Hard causal rule

A feature is eligible only if:

```text
valid_time <= issue_time
and
operationally_available_time <= issue_time + allowed_ingestion_delay
```

## Tests

- shift future source rows and verify features do not change;
- remove future observations and verify historical features remain identical;
- compare original replay with regenerated replay;
- test source revisions;
- test missing and late data;
- test timezone and UTC boundaries.

---

# 12. Phase 9 — Target-specific model refinement

## Temperature

Use residual correction against persistence and the approved regional prior. Promote only at horizons where the candidate beats persistence consistently.

## Wind direction

Predict unit-vector components:

```text
u = cos(direction)
v = sin(direction)
```

Use circular error for scoring and persistence in calm or low-speed conditions. Do not promote based on ordinary MAE of degrees.

## Humidity

Use bounded residual or transformed modeling. Compare against persistence and climatology. Keep persistence if station telemetry contains insufficient predictive signal.

## Pressure

Use regional gradients and approved NWP residuals if eligible. Validate sensor calibration before increasing model complexity.

## Rain

Retain the hurdle formulation:

```text
P(rain) × conditional amount
```

Use radar/satellite only after rights and causal replay pass.

## UV and luminosity

Treat them as separate workstreams. Do not infer UV quality from rain or temperature improvement. Retain quarantine/beta status until calibration and daylight/nighttime validation are complete.

---

# 13. Phase 10 — Evaluation and scorecard regeneration

## Required ablations

1. local-only;
2. local + nearby stations;
3. local + radar;
4. local + Himawari-derived features;
5. local + approved NWP;
6. all approved sources;
7. source-missing variants;
8. latency-degraded variants.

## Required metrics

### Continuous targets

- MAE;
- RMSE;
- median absolute error;
- bias;
- worst-fold error;
- residual calibration;
- interval coverage and width.

### Rain and anomalies

- Brier score;
- log loss;
- reliability;
- precision;
- recall;
- F1;
- false alarms per day;
- lead time;
- heavy-event recall;
- event sample count.

### Wind direction

- circular MAE;
- circular median error;
- calm-condition performance;
- non-calm performance;
- vector-component error.

## Scorecard additions

Add source-aware fields to the scorecard:

```text
source_set
source_registry_hash
source_missing_rate
source_latency_p95
source_ablation_delta
worst_fold_delta
confidence_interval
sample_count
promotion_decision
information_status
```

## Promotion rule

A target/horizon may be promoted only when its candidate is better than persistence under the predefined metric and passes all operational gates. Otherwise route it to persistence and label it **Information Limited**.

---

# 14. Phase 11 — Provenance, release, and rollback

## Manifest additions

Every candidate bundle must include:

```text
source_registry_hash
source_license_decisions
source_raw_hashes
source_feature_schema_hash
source_parser_versions
source_ablation_results
source_missing_behavior
```

## Release sequence

1. Run source-license gate.
2. Run data-quality gate.
3. Run causal replay gate.
4. Run target/horizon scorecard.
5. Run missing-source tests.
6. Run uncertainty and anomaly evidence checks.
7. Build candidate bundles.
8. Verify provenance.
9. Review target-specific promotion decisions.
10. Publish only approved routes.
11. Retain previous bundle for rollback.

## Rollback triggers

- license expiry or revocation;
- source schema change;
- source latency breach;
- missingness above threshold;
- provenance mismatch;
- metric regression;
- calibration drift;
- unexpected anomaly false-alarm increase.

Rollback must be target-specific and horizon-specific.

---

# 15. Test and CI additions

Add release gates for:

1. registry schema;
2. unknown-source rejection;
3. expired-license rejection;
4. training-manifest license agreement;
5. raw-source hash agreement;
6. source revision handling;
7. causal feature replay;
8. future-mutation rejection;
9. missing-source fallback;
10. late-source fallback;
11. source-latency limits;
12. target-specific policy agreement;
13. candidate bundle provenance;
14. source-aware scorecard completeness;
15. rollback behavior.

Existing tests must continue to pass unchanged.

---

# 16. What must not be done

Do not:

- add RainViewer because it is convenient;
- treat public visibility as commercial permission;
- use Himawari imagery without checking the exact product license;
- use PAGASA data without product-specific rights review;
- train on future radar mosaics or revised analyses;
- replace missing source values with future observations;
- claim global model superiority from local telemetry;
- remove persistence because a learned model looks better on one fold;
- modify the dashboard to hide information-limited targets;
- delete raw telemetry or archive branches;
- silently change the active policy during experiments;
- mix research-only artifacts with production bundles.

---

# 17. Implementation order

The exact order is:

1. Freeze baseline and create feature branch.
2. Add source registry schema and executable license gate.
3. Add immutable external manifest and cache contracts.
4. Audit candidate sources and record decisions.
5. If no source is eligible, stop and retain local-only operation.
6. Integrate the first eligible nearby-station source.
7. Build causal feature cube.
8. Add source-aware tests.
9. Run source ablation and regenerate scorecard.
10. Promote only proven target/horizon routes.
11. Evaluate approved radar/QPE.
12. Evaluate approved Himawari-derived features.
13. Evaluate approved NWP residuals.
14. Harden provenance, monitoring, and rollback.
15. Reclassify information-limited targets based on evidence.

---

# 18. Definition of done

The next phase is complete only when:

- the clean baseline remains reproducible;
- the source registry is committed;
- blocked sources fail closed;
- at least one external source is explicitly approved, or the project records that none is eligible;
- raw source objects and manifests are immutable;
- all feature timestamps are causal;
- external-source failures do not break inference;
- source-aware ablations are complete;
- every target/horizon has a persistence comparison;
- scorecard metrics include confidence and sample counts;
- no candidate is promoted without measurable benefit;
- provenance verifies code, data, source registry, policy, and artifacts;
- rollback works per target and horizon;
- raw telemetry and archive branches remain intact;
- no dashboard or unrelated service changes are introduced.

## Final decision rule

If spatial/external data improves a target while remaining license-eligible, causal, reproducible, and operationally reliable, promote that target/horizon only.

If it does not beat persistence, keep the fallback and label the target **Information Limited**. That is a valid scientific outcome, not a failure of the release process.
