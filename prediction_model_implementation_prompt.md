# Implementation Prompt: License-Gated Spatial Expansion for `prediction-model`

You are implementing the next accuracy-improvement phase of the `prediction-model` repository.

Work only in the prediction-model module. Do not modify the dashboard UI, unrelated services, deployment configuration, or unrelated application behavior.

## Repository baseline

Start from and preserve the current remote baseline:

```text
origin/main
commit: 9b58618900940d43ef65b093ca9527e2e6ff4d43
```

Before making changes:

1. Fetch the latest `origin/main`.
2. Confirm the actual current HEAD.
3. Create a feature branch from the current remote HEAD.
4. Record the baseline commit and current scorecard hashes.
5. Run the existing prediction-model test and provenance suites.
6. Stop if the baseline is not clean or the existing gates fail.

## Mission

Implement a safe, reproducible, license-gated path for improving forecasting accuracy with approved spatial/external data while preserving the local-only fallback.

The goal is not to force every target onto a learned model. The goal is to measurably improve the targets and horizons for which external information provides real signal.

Every target and horizon must continue to be evaluated against persistence and climatology baselines.

If a target cannot beat persistence with sufficient evidence, keep the fallback and mark it `Information Limited`.

# Non-negotiable constraints

## Data and scientific integrity

- Use real telemetry and approved real external data only.
- Never fabricate, synthesize, or backfill training observations.
- Preserve all raw local telemetry.
- Preserve the archive branch.
- Use chronological splits only.
- Preserve the existing 48-hour embargo.
- Prevent future leakage from observations, revised products, analyses, forecasts, or source revisions.
- Use strict UTC timestamps.
- Compare every candidate with the existing persistence and climatology baselines.
- Report sample counts, confidence intervals, worst folds, and worst regimes.
- Never promote globally when only one target/horizon improves.

## License and commercial-use integrity

No source may enter production training, validation, inference, or commercial output unless its registry decision is one of:

```text
APPROVED_FREE_COMMERCIAL
APPROVED_WITH_EXPLICIT_NO_COST_PERMISSION
```

The license review must explicitly verify:

- commercial use;
- model training;
- commercial inference;
- derived-feature creation;
- raw-data caching;
- redistribution restrictions;
- attribution requirements;
- rate limits;
- terms version or review date.

The following are blocked:

```text
UNKNOWN_BLOCKED
PENDING_REVIEW
RESEARCH_ONLY
NOT_ALLOWED
```

Do not use RainViewer, Himawari-9, PAGASA radar, PAGASA NWP, nearby stations, or any other external source merely because it is publicly accessible or technically easy to download. The exact product, access path, and terms must be approved first.

If no external source passes the commercial-use gate, do not add one. Keep the local-only system operational and document the blocker.

## Operational safety

- Never fetch external data during live model inference.
- Inference must use prepared, verified artifacts only.
- Every external-source failure must degrade to a verified local-only model.
- License expiry, revocation, schema change, or provenance mismatch must fail closed or trigger rollback.
- Preserve target-specific and horizon-specific rollback capability.
- Include the source mode and fallback mode in prediction metadata.
- Do not silently substitute a blocked source.

# Required implementation phases

## Phase 0 — Baseline and branch safety

Create a baseline record without modifying existing production artifacts.

Record:

- current remote commit;
- current inference policy hash;
- current model bundle hashes;
- current candidate artifact hashes;
- raw-data hashes;
- current test counts;
- current operational decisions.

Add or update documentation:

```text
prediction-model/docs/multisource_implementation_status.md
```

Do not overwrite the baseline artifacts.

## Phase 1 — Executable source registry and license gate

Add:

```text
prediction-model/data/external_source_registry.json
prediction-model/src/external_source_registry.py
prediction-model/src/test_external_source_registry.py
```

The registry must support records with at least:

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

Implement a deterministic fail-closed API such as:

```python
assert_production_eligible(source_id)
```

It must reject:

- unknown source IDs;
- missing license records;
- unknown decisions;
- research-only decisions;
- expired reviews;
- missing license URLs;
- contradictory permission fields;
- sources without training permission;
- sources without commercial inference permission.

Tests must cover approved, blocked, unknown, expired, contradictory, and revoked sources.

The release path must fail if any training manifest references a non-approved source.

## Phase 2 — Immutable external-source contracts

Add:

```text
prediction-model/src/external_source_contract.py
prediction-model/src/external_cache.py
prediction-model/src/test_external_source_contract.py
```

Every raw external object must record:

- source ID;
- provider and product version;
- valid time UTC;
- issue time UTC when applicable;
- retrieval time UTC;
- request parameters;
- response checksum;
- raw-object path;
- license decision;
- license-registry hash;
- source quality;
- latency;
- revision status;
- parser version.

Raw data must be append-only. Revisions must receive new object identities. Never silently replace an old raw object.

The feature builder must select only data that was operationally available by the issue time.

## Phase 3 — Source audit and selection

Add:

```text
prediction-model/docs/source_license_decisions.md
```

Audit candidate sources in this order:

1. project-owned nearby stations;
2. explicitly commercially licensed open-government nearby stations;
3. explicitly commercially licensed radar/QPE;
4. explicitly commercially licensed NWP;
5. explicitly commercially licensed Himawari-9-derived products.

For each source, record:

- exact product name;
- provider;
- exact access URL;
- exact terms URL;
- commercial training status;
- commercial inference status;
- derived-feature status;
- caching status;
- attribution;
- rate limits;
- review date;
- decision;
- reason for decision.

Do not implement a production adapter for a source marked `UNKNOWN_BLOCKED`, `PENDING_REVIEW`, `RESEARCH_ONLY`, or `NOT_ALLOWED`.

## Phase 4 — Approved nearby-station adapter

Only implement this phase if at least one nearby-station source is approved.

Add:

```text
prediction-model/src/external_sources/__init__.py
prediction-model/src/external_sources/nearby_stations.py
prediction-model/src/test_nearby_stations.py
```

Features may include:

- distance;
- bearing;
- elevation difference;
- recent values;
- causal lags;
- rolling mean;
- rolling volatility;
- local trend;
- missingness;
- quality flag;
- telemetry age;
- inverse-distance weighted regional aggregates;
- nearest valid station;
- station disagreement;
- temperature, humidity, pressure, and wind gradients;
- regional wind-vector mean and dispersion.

Rules:

- use UTC only;
- require valid time to be no later than issue time;
- preserve missingness explicitly;
- do not use future corrected values;
- record source metadata for every feature group.

Evaluate `local-only` versus `local + nearby stations` using the existing chronological rolling-origin process.

## Phase 5 — Approved radar/QPE adapter

Only implement this phase if the exact radar product is commercially approved.

Add:

```text
prediction-model/src/external_sources/radar.py
prediction-model/src/test_radar.py
```

Initial features:

- mean reflectivity;
- maximum reflectivity;
- rainy-pixel fraction;
- rain-cell distance;
- rain-cell bearing;
- intensity trend;
- recent accumulation;
- cell-motion estimate;
- coverage quality;
- source latency.

Prioritize:

1. rain occurrence;
2. rain onset;
3. rain cessation;
4. precipitation amount;
5. heavy-rain anomaly detection.

Evaluate Brier score, log loss, reliability, precision, recall, false alarms per day, onset lead time, and heavy-event recall.

## Phase 6 — Approved Himawari-9-derived features

Only implement this phase if the exact Himawari-9 product and distribution path pass the license gate.

Add:

```text
prediction-model/src/external_sources/himawari.py
prediction-model/src/test_himawari.py
```

Start with numeric derived features, not raw-image deep learning:

- infrared brightness-temperature summaries;
- cloud-top cooling rate;
- cloud-cover fraction;
- visible reflectance where daylight-valid;
- water-vapor summaries;
- cloud motion;
- spatial gradients;
- solar-geometry flags;
- quality and latency flags.

Evaluate rain onset, convection anomalies, temperature residuals, and only later UV/luminosity.

Do not treat a publicly viewable image as automatically commercially eligible.

## Phase 7 — Approved NWP residual adapter

Only implement this phase if the exact NWP product is approved for commercial training and inference.

Add:

```text
prediction-model/src/external_sources/nwp.py
prediction-model/src/test_nwp.py
```

Use the NWP product as a regional prior and learn local residuals:

```text
local_forecast = approved_nwp_forecast + learned_local_residual
```

Record issue time, valid time, lead time, model cycle, grid coordinates, interpolation method, revision status, source quality, and license-registry hash.

Never train against a later analysis or revised value when it was not available operationally at issue time.

# Causal feature cube

Implement a unified feature representation with row identity:

```text
station_id
issue_time_utc
horizon_hours
```

Every feature group must carry:

- source ID;
- valid time;
- issue time;
- retrieval time;
- availability cutoff;
- quality flag;
- missingness flag;
- license-registry hash;
- raw-object hash;
- parser version.

Enforce:

```text
valid_time <= issue_time
operationally_available_time <= issue_time + allowed_ingestion_delay
```

Add tests that:

- mutate future source rows and verify features do not change;
- remove future observations and verify historical features remain identical;
- replay a fixed manifest and reproduce the feature hash;
- handle revised source objects;
- handle missing and late data;
- handle UTC boundaries.

# Target-specific modeling

## Temperature

Use residual correction against persistence and any approved regional prior. Promote only at horizons where the candidate consistently beats persistence.

## Wind direction

Predict vector components:

```text
u = cos(direction)
v = sin(direction)
```

Score with circular metrics. Retain persistence for calm or low-speed conditions when it is safer.

## Humidity

Use bounded residual or transformed modeling. Compare against persistence and climatology. Keep the fallback if local/spatial inputs do not add reliable signal.

## Pressure

Use regional pressure gradients and approved NWP residuals if available. Validate sensor quality before increasing model complexity.

## Rain

Retain the hurdle approach:

```text
P(rain) × conditional_amount
```

Use approved radar/satellite inputs only after license and causal gates pass.

## UV and luminosity

Treat each as a separate workstream. Do not claim improvement from rain or temperature performance. Retain quarantine or beta status until dedicated calibration and daylight/nighttime tests pass.

# Evaluation and scorecard

Run these ablations:

1. local-only;
2. local + nearby stations;
3. local + radar;
4. local + Himawari-derived features;
5. local + approved NWP;
6. all approved sources;
7. source-missing variants;
8. latency-degraded variants.

For continuous targets report:

- MAE;
- RMSE;
- median absolute error;
- bias;
- worst-fold error;
- calibration;
- interval coverage and width.

For rain/anomalies report:

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

For wind direction report:

- circular MAE;
- circular median error;
- calm-condition error;
- non-calm error;
- vector-component error.

Add source-aware scorecard fields:

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

Promote a target/horizon only when:

- the predefined primary metric beats persistence;
- the improvement appears in a majority of rolling folds;
- the worst fold is not materially degraded;
- source-missing behavior is safe;
- uncertainty calibration is not materially worse;
- no leakage is detected;
- provenance and license gates pass.

Otherwise keep the existing fallback and mark the target/horizon `Information Limited`.

# Provenance and release

Update candidate manifests and bundles with:

```text
source_registry_hash
source_license_decisions
source_raw_hashes
source_feature_schema_hash
source_parser_versions
source_ablation_results
source_missing_behavior
```

Add release gates for:

1. source-registry schema;
2. unknown-source rejection;
3. expired-license rejection;
4. training-manifest license agreement;
5. raw-source hash agreement;
6. source-revision handling;
7. causal replay;
8. future-mutation rejection;
9. missing-source fallback;
10. late-source fallback;
11. source-latency limits;
12. policy/scorecard agreement;
13. candidate-bundle provenance;
14. source-aware scorecard completeness;
15. target-specific rollback.

Never modify active production routing until all candidate artifacts are verified.

# Required validation commands

Use the repository’s configured Python environment. At minimum, run:

```bash
cd prediction-model
python -m py_compile src/*.py
PYTHONPATH=src python src/test_canonical_contract.py
PYTHONPATH=src python src/test_inference_contract.py
PYTHONPATH=src python src/test_monitoring.py
PYTHONPATH=src python src/test_predictive_quality.py
PYTHONPATH=src python src/test_provenance.py
PYTHONPATH=src python src/smoke_test.py
PYTHONPATH=src python src/generate_bundles.py --check-only
PYTHONPATH=src python src/verify_provenance.py
```

Also run all new source registry, source contract, adapter, causal-replay, fallback, scorecard, and rollback tests.

Run from a clean checkout or detached worktree before reporting completion.

# Required final report

Before stopping, report:

1. exact starting commit;
2. exact ending commit;
3. files changed;
4. sources audited;
5. source license decisions;
6. sources actually used;
7. sources blocked and why;
8. tests run and results;
9. scorecard deltas against persistence;
10. target/horizon promotion decisions;
11. information-limited targets;
12. provenance verification result;
13. rollback verification result;
14. known limitations;
15. recommended next phase.

Do not claim the task is complete if:

- the source license gate is only documentation;
- an external source is used without explicit approval;
- causal replay is missing;
- persistence comparisons are missing;
- the existing provenance suite fails;
- raw telemetry or archive history was altered;
- dashboard or unrelated services were modified.

## Definition of done

The implementation is complete only when:

- the baseline remains reproducible;
- the source registry is executable;
- blocked sources fail closed;
- at least one source is explicitly approved, or the blocker is documented;
- raw source objects are immutable and hashed;
- feature construction is causal;
- external-source failure falls back safely;
- source ablations are complete;
- every target/horizon has persistence comparisons;
- scorecards include confidence and sample counts;
- no target/horizon is promoted without measurable evidence;
- provenance verifies code, data, source registry, policy, and artifacts;
- target/horizon rollback works;
- raw telemetry and archive branches remain intact;
- all existing and new tests pass;
- no dashboard or unrelated service changes are present.

If the data cannot prove improvement, preserve the fallback and state clearly that the target is `Information Limited`.
