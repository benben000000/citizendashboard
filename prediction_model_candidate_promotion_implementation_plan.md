# Candidate Promotion and Operational Rollout Implementation Plan

## Executive decision

The repository is structurally healthy at GitHub `main` commit `0153a7957460cf28cae528e152e997fed9b2165d`. The latest clean-worktree audit passed compilation, canonical tests, inference tests, monitoring tests, 28 predictive-quality tests, provenance tests, smoke tests, and whitespace checks.

The remaining work is not primarily a test-suite problem. It is a **candidate lifecycle and release-integrity problem**. The feature-augmented candidate can be trained and its artifacts can be generated locally, but the candidate artifacts are not committed to GitHub, the production bundles still point to the previous eight-feature model, and the release scorecard can report operational `GO` under a permissive five-percent non-inferiority rule even when the candidate is worse than persistence for some targets and horizons.

The objective of this plan is to make the candidate lifecycle complete, make the release decision scientifically defensible, and promote the candidate only if the committed artifacts and operational policy are mutually consistent.

> **Target release state:** a provenance-complete model bundle, a reproducible five-horizon scorecard, a policy that selects each target source explicitly, a tested rollback path, and an honest `GO`, `CONDITIONAL GO`, or `NO-GO` decision.

## Current baseline and known gaps

The protected baseline remains the existing `GarciaWeatherLNN` bundle with the canonical eight-feature schema. The current candidate is `MF-1-FEATURED` with a 75-feature context vector. The latest implementation includes candidate checkpoint creation, candidate manifests, calibration artifacts, prediction logs, candidate loading in inference, manifest tamper checks, and heat-index tests.

The current gaps are:

| Gap | Consequence | Required resolution |
|---|---|---|
| Candidate artifacts are generated only in local output directories | GitHub cannot reproduce or serve the candidate | Regenerate and commit approved artifacts under a documented canonical path |
| Production bundles remain the previous baseline bundles | The candidate is not actually deployed | Create candidate bundle artifacts only after release gates pass |
| Bundle provenance contains incomplete fields such as null commit identifiers | Model identity cannot be independently verified | Populate implementation, artifact, weight, checkpoint, policy, and data hashes |
| The scorecard combines research and operational decisions | A permissive research result can be misread as production approval | Separate research status from operational status |
| Non-inferiority uses a simple five-percent tolerance | Small regressions may be accepted without uncertainty analysis | Add paired uncertainty intervals and target-specific operational gates |
| UV is blocked and luminosity is beta-only | Those targets cannot be claimed as production-ready | Preserve explicit blocked and conditional statuses |
| Candidate policy and committed production bundles are not yet aligned | Inference may load a different model than the scorecard evaluates | Generate and verify one candidate policy/bundle set |

## Definition of success

The implementation is complete only when all of the following conditions are satisfied:

1. The candidate has been trained on the real telemetry using a declared seed and configuration.
2. Candidate checkpoints, manifests, calibration artifacts, prediction logs, and scorecards exist for 1h, 3h, 6h, 12h, and 24h.
3. Every candidate checkpoint reloads in a clean process.
4. The feature schema used by inference is identical to the feature schema used by training.
5. The candidate scorecard compares persistence, climatology, ridge, tree, baseline neural, and candidate neural results.
6. The final test set is not used to select features, epochs, thresholds, blend weights, or calibration parameters.
7. The policy is generated from calibration data and points to the exact candidate artifacts evaluated in the scorecard.
8. Every bundle has valid hashes and commit identifiers.
9. The production policy is updated only if the operational gates pass.
10. A baseline rollback is tested and remains available.
11. UV remains blocked until sensor calibration is validated.
12. Luminosity remains daylight-only beta unless its validation gates pass.
13. The repository passes all tests, provenance checks, and clean-worktree checks after artifact regeneration.

## Phase 1 — Freeze the release baseline

### Purpose

Create an immutable reference so the candidate cannot be promoted by comparing against a moving target.

### Implementation

Record the following before regenerating candidate artifacts:

- starting commit;
- raw weather telemetry SHA-256;
- raw water telemetry SHA-256;
- canonical eight-feature schema;
- current production bundle hashes;
- current inference policy hash;
- split dates and embargo duration;
- seed and package versions;
- baseline metrics for all five horizons;
- station counts and quarantine counts;
- current UV and luminosity status.

Create a release workspace from a detached clean checkout of `origin/main`. Do not train from a dirty development branch.

### Acceptance criteria

The baseline manifest must be reproducible and must identify the exact production checkpoint and policy. The baseline artifacts must not be overwritten during candidate experimentation.

## Phase 2 — Make artifact paths canonical

### Purpose

Ensure every future training run writes to a predictable, reviewable location without contaminating raw data or creating machine-specific paths.

### Required artifact layout

Use a dedicated research and release directory such as:

```text
prediction-model/data/candidate_artifacts/
  baseline_manifest.json
  predictive_quality_scorecard.json
  model_comparison_report.json
  candidate_h1h.pt
  candidate_h1h_manifest.json
  candidate_h1h_calibration.json
  candidate_h1h_predictions.csv
  candidate_h3h.pt
  candidate_h3h_manifest.json
  candidate_h3h_calibration.json
  candidate_h3h_predictions.csv
  candidate_h6h.pt
  candidate_h6h_manifest.json
  candidate_h6h_calibration.json
  candidate_h6h_predictions.csv
  candidate_h12h.pt
  candidate_h12h_manifest.json
  candidate_h12h_calibration.json
  candidate_h12h_predictions.csv
  candidate_h24h.pt
  candidate_h24h_manifest.json
  candidate_h24h_calibration.json
  candidate_h24h_predictions.csv
```

The final production bundle should remain separate:

```text
prediction-model/data/bundles/h1/
prediction-model/data/bundles/h3/
prediction-model/data/bundles/h6/
prediction-model/data/bundles/h12/
prediction-model/data/bundles/h24/
```

The candidate artifact directory records evaluation. The bundle directory records what inference is authorized to serve. They must not be confused.

### Required changes

Update `train_predictive_quality.py` so that it:

- creates the canonical output directory;
- writes all five horizon artifacts there;
- records relative filenames rather than machine-specific absolute paths;
- records artifact hashes after writing;
- fails if any expected artifact is missing;
- emits a machine-readable summary of all generated files.

Update `.gitignore` only if necessary. Do not ignore canonical candidate artifacts that are intentionally part of the release record.

### Acceptance criteria

A clean run must produce exactly the expected artifact classes for all five horizons. A second run with the same seed and inputs must either reproduce byte-identical deterministic artifacts or document which files are expected to vary and why.

## Phase 3 — Complete manifest and provenance fields

### Purpose

Make it possible to prove that the model evaluated is the model loaded by inference.

Each candidate manifest must include:

```json
{
  "model_family": "MF-1-FEATURED",
  "horizon_hours": 1,
  "feature_schema": [],
  "input_dimension": 8,
  "context_dimension": 75,
  "seed": 42,
  "model_config": {},
  "training_config": {},
  "weather_telemetry_sha256": "...",
  "water_telemetry_sha256": "...",
  "checkpoint_filename": "candidate_h1h.pt",
  "checkpoint_sha256": "...",
  "calibration_filename": "candidate_h1h_calibration.json",
  "calibration_sha256": "...",
  "predictions_filename": "candidate_h1h_predictions.csv",
  "implementation_commit": "...",
  "artifact_commit": "...",
  "model_weights_commit": "...",
  "status": "CANDIDATE_RESEARCH"
}
```

The exact field names may follow the repository’s existing manifest contract, but all three commit identities and all relevant content hashes must be represented consistently.

The provenance verifier must reject:

- missing candidate checkpoint hash;
- missing calibration hash;
- missing data hash;
- stale implementation commit;
- mismatched feature schema;
- mismatched policy or bundle hash;
- a candidate manifest that claims production status without an approved bundle.

### Acceptance criteria

For every candidate horizon, recompute the checkpoint and calibration hashes independently and compare them with the manifest. Run provenance verification from a clean checkout at the final artifact commit.

## Phase 4 — Strengthen the promotion decision

### Purpose

Prevent a permissive research threshold from being reported as production approval.

Replace the current combined final decision with two separate decisions:

```text
research_decision
operational_decision
```

Use these values:

```text
GO
CONDITIONAL_GO
NO_GO
```

### Research decision

The candidate may receive `research_decision = GO` when:

- all five horizons were evaluated;
- candidate artifacts are complete;
- all tests and provenance checks pass;
- no future leakage is detected;
- target feasibility statuses are explicit;
- the candidate can be reloaded and scored reproducibly.

This does not authorize production inference.

### Operational decision

The candidate may receive `operational_decision = GO` only when all critical target gates pass. A five-percent tolerance may be used as a screening rule, but it is not sufficient by itself. Add paired bootstrap confidence intervals or another documented uncertainty method.

At minimum:

| Target | Operational requirement |
|---|---|
| Temperature | Candidate must improve or be statistically non-inferior by horizon; repeated short-horizon regressions are not acceptable |
| Humidity and pressure | Candidate must not materially degrade the baseline and must remain physically valid |
| Rain probability | Brier score and log loss must improve or be non-inferior; calibration error must not materially worsen |
| Precipitation amount | Rainy-hour and heavy-rain metrics must be reported; absence of valid heavy-rain skill prevents a strong operational claim |
| Wind direction | Candidate must improve or be non-inferior to persistence using circular error; otherwise retain persistence for this target |
| Heat index | Derived from selected temperature and humidity predictions and checked for physical consistency |
| UV | Remains blocked while sensor calibration is defective |
| Luminosity | Remains beta and daylight-only unless station/daylight validation passes |
| Anomalies | Operational accuracy claims require real or reviewed labels |

If only some targets pass, use target-specific policy selection rather than promoting the whole candidate model indiscriminately. For example, the candidate may serve rain probability while persistence remains the source for wind direction.

### Acceptance criteria

The scorecard must clearly state why each target is promoted, retained on the baseline, conditional, or blocked. A global `GO` must be impossible when a critical target fails its release gate.

## Phase 5 — Run the final real-data experiment

### Purpose

Generate the evidence used for the release decision.

### Training configuration

Record and freeze:

- seed;
- epochs;
- learning rate;
- batch size;
- optimizer;
- early-stopping patience;
- model dimensions;
- feature schema;
- loss weights;
- split dates;
- calibration dates;
- test dates;
- embargo duration;
- telemetry hashes;
- Python and package versions.

Run the pipeline for all five horizons. Use the same test rows for all candidate and baseline comparisons.

### Required comparisons

The scorecard must include:

- persistence;
- climatology;
- ridge;
- gradient-boosted tree;
- current canonical model;
- feature-augmented candidate;
- target-specific hybrid or fallback policy where applicable.

### Required evaluation slices

Report results by:

- horizon;
- station;
- rain/dry regime;
- daylight/nighttime regime;
- calm/noncalm wind regime;
- heavy-rain regime;
- rolling-origin period.

### Acceptance criteria

The final scorecard must contain all five horizons and must identify the worst station and worst regime for each critical target. No target may be considered improved from an aggregate score alone.

## Phase 6 — Build the candidate policy and production bundles

### Purpose

Connect the evaluated candidate to inference without replacing the baseline until approval is explicit.

For each horizon, create a candidate bundle containing:

- candidate checkpoint;
- candidate manifest;
- candidate calibration artifact;
- candidate policy;
- prediction schema;
- bundle manifest;
- checkpoint hash;
- policy hash;
- implementation commit;
- artifact commit;
- model weights commit;
- data hashes;
- model status.

Initially set:

```text
model_status: CANDIDATE_RESEARCH
```

Do not copy candidate artifacts into the active production bundle path until `operational_decision = GO`.

When promotion is approved, update the policy per target and horizon. The policy must support mixed-source decisions, such as:

```text
1h temperature: baseline or candidate
1h rain probability: candidate
1h wind direction: persistence
UV: blocked
luminosity: daylight beta
```

This is safer than declaring the entire candidate model operational when only some targets improve.

### Acceptance criteria

Load every candidate bundle through the inference API. Verify that a tampered checkpoint, policy, feature schema, or hash fails closed. Test rollback by loading the previous production bundle and producing a valid forecast.

## Phase 7 — Validate heat index and derived targets

### Purpose

Ensure derived weather indicators are physically consistent rather than independently optimized in a way that conflicts with their source variables.

Heat index must be derived from the selected temperature and humidity forecasts using the documented NOAA/Rothfusz implementation already present in the repository. The scorecard must report:

- heat-index MAE;
- heat-index RMSE;
- risk-category accuracy;
- boundary errors;
- physical consistency with temperature and humidity;
- bound violations.

UV remains blocked because the current sensor calibration defect prevents a defensible global forecast. Luminosity remains daylight-only beta and must be reported by station.

### Acceptance criteria

A heat-index prediction cannot be promoted if it conflicts with the selected temperature and humidity values. UV and luminosity status must remain visible in the final operational policy.

## Phase 8 — Commit canonical artifacts and regenerate provenance

### Purpose

Make the release reproducible from GitHub rather than from a temporary local directory.

Use this sequence:

```bash
git fetch origin main --prune
git worktree add --detach /tmp/prediction-release origin/main
cd /tmp/prediction-release/prediction-model
python src/train_predictive_quality.py --output-dir data/candidate_artifacts
python src/verify_provenance.py
python src/test_canonical_contract.py
python src/test_inference_contract.py
python src/test_monitoring.py
python src/test_predictive_quality.py
python src/test_provenance.py
python src/smoke_test.py
```

Review:

```bash
git status --short
git diff --stat
git diff --check
git diff -- prediction-model/data/candidate_artifacts
```

Commit the approved artifacts only after confirming that no local paths, credentials, caches, or temporary files are present.

After the commit changes the implementation hash, regenerate the artifacts again. Run provenance verification again at the final commit. This second regeneration is required because the artifact commit identity changes when the code and artifacts are committed.

### Acceptance criteria

The final GitHub tree must contain the intended candidate artifacts, complete manifests, and the scorecard. The final provenance verifier must pass at the exact final commit.

## Phase 9 — Release gates and push decision

### Mandatory gates

The final release must pass:

```bash
python -m py_compile prediction-model/src/*.py
python prediction-model/src/test_canonical_contract.py
python prediction-model/src/test_inference_contract.py
python prediction-model/src/test_monitoring.py
python prediction-model/src/test_predictive_quality.py
python prediction-model/src/test_provenance.py
python prediction-model/src/smoke_test.py
python prediction-model/src/verify_provenance.py
git diff --check
git status --short
```

The final worktree must be clean after all read-only checks. The scorecard must report separate research and operational decisions. The production bundle must not be changed when the operational decision is `NO_GO` or `CONDITIONAL_GO`.

Push to `origin/main` only when:

- all required tests pass;
- final provenance passes;
- candidate artifacts are committed;
- active bundle and policy hashes match;
- rollback succeeds;
- operational decision is `GO`, or the user explicitly requests a research-only artifact release;
- no protected workflow blocks the push.

## Expected final report

The final report must include:

- starting commit;
- final commit;
- archive reference;
- changed and retained files;
- deleted files, if any;
- dataset hashes;
- quarantine counts;
- training seed and configuration;
- full five-horizon scorecard;
- station and regime worst cases;
- rain calibration metrics;
- precipitation amount metrics;
- wind-direction circular metrics;
- heat-index metrics;
- UV and luminosity feasibility status;
- anomaly validation status;
- candidate artifact paths and hashes;
- bundle and policy hashes;
- provenance verification result;
- test results;
- exact push result or blocker;
- separate research and operational decisions.

The final recommendation must use one of these forms:

```text
GO — candidate promoted to active production for the approved targets and horizons
CONDITIONAL GO — candidate committed for research or target-specific use; production promotion remains limited
NO-GO — candidate retained as research-only; baseline remains active
```

## Immediate next actions

The next implementation session should proceed in this order:

1. Add the canonical `candidate_artifacts` output path and expected-file check.
2. Add complete commit and hash fields to all candidate manifests.
3. Split the global scorecard decision into research and operational decisions.
4. Add uncertainty-aware target gates and target-specific fallback selection.
5. Run the final five-horizon real-data experiment.
6. Review the scorecard and decide whether any targets qualify for promotion.
7. Generate candidate bundles with `CANDIDATE_RESEARCH` status.
8. Promote only approved targets and horizons, leaving the baseline available for rollback.
9. Regenerate artifacts at the final commit.
10. Run the complete release gate and push only after the final status is clean.

## References

[1]: https://github.com/benben000000/citizendashboard "Citizen Dashboard repository"
[2]: https://github.com/benben000000/citizendashboard/tree/main/prediction-model "Citizen Dashboard prediction-model directory"
