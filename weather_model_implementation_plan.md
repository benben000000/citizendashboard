# Weather Telemetry Prediction Model — Implementation Plan

**Repository:** `benben000000/citizendashboard`  
**Baseline commit:** `9b58618900940d43ef65b093ca9527e2e6ff4d43`  
**Source audit:** [Second audit report](</home/ubuntu/citizendashboard/model_audit_report_reevaluation.md>)  
**Objective:** make the evaluation reproducible and honest, restore release integrity, establish a real anomaly benchmark, and only then decide whether model changes are justified.

## Target end state

The release pipeline should be able to answer, from a clean checkout:

1. **Which exact code, weights, policy, data, and test rows produced each metric?**
2. **Can another machine reproduce every headline metric from committed artifacts?**
3. **Are anomaly metrics calculated from reviewed labels rather than report defaults?**
4. **Does the candidate beat the selected baseline consistently enough to be promoted?**
5. **Does the complete test and provenance suite pass without manual overrides?**

Until all five answers are yes, keep the system in `CANDIDATE_RESEARCH` / `CONDITIONAL_GO` status and retain target-specific persistence fallbacks.

## Priority order

| Priority | Workstream | Why it comes first |
|---|---|---|
| P0 | Provenance and artifact lineage | The current release gate fails and the evaluation artifacts point to a different commit. |
| P0 | Full prediction-log publication and replay | The current 500-row logs cannot reproduce scorecard metrics based on thousands of rows. |
| P0 | Real anomaly labels and evaluation | The current anomaly report writes perfect metrics as constants. |
| P1 | Dependency-complete CI | Neural and AWS-related tests cannot run in the audited environment. |
| P1 | Report/schema consistency | Candidate scorecard and operational monitoring report different metrics without explicit model/policy identity. |
| P2 | Forecast-quality and uncertainty improvements | Retraining before fixing evaluation validity would produce another non-auditable result. |
| P2 | Extreme-event benchmark expansion | Zero 10 mm/h CSI means the candidate is not suitable for hazardous-rain alerts. |

---

## Phase 0 — Freeze the baseline and create a release branch

**Purpose:** prevent another artifact/code mismatch while the fixes are implemented.

### Changes

- Create a dedicated branch, for example `fix/prediction-model-audit-findings`.
- Record the current baseline in a small machine-readable file such as `prediction-model/data/release_baseline.json`:
  - repository HEAD;
  - parent commit;
  - raw weather and water hashes;
  - current candidate artifact hashes;
  - scorecard generation timestamp;
  - known failing gates.
- Mark the existing scorecard, anomaly report, uncertainty report, and 500-row candidate logs as **historical candidate artifacts** until regenerated.
- Do not alter metric values manually in JSON files.

### Acceptance criteria

- The branch starts from the audited HEAD.
- A clean `git status` is possible before implementation begins.
- The baseline file identifies the current artifacts as non-promotable.

---

## Phase 1 — Repair release provenance and artifact identity

**Primary files:**

- `prediction-model/src/verify_provenance.py`
- `prediction-model/src/generate_manifests.py`
- `prediction-model/src/generate_bundles.py`
- `prediction-model/src/train_predictive_quality.py`
- `prediction-model/data/inference_policy.json`
- `prediction-model/data/bundles/h{1,3,6,12,24}/`
- `prediction-model/data/candidate_artifacts/`

### Design decision

Use explicit, non-ambiguous commit roles:

- `source_commit`: commit containing the model/evaluator code;
- `artifact_commit`: commit that contains the generated artifact bytes;
- `release_commit`: the release tag or commit being evaluated;
- `parent_source_commit`: optional historical lineage only.

The verifier may accept a parent source commit only when the manifest explicitly declares the two-commit release procedure and the relationship is verified. It must not rely on a broad, manually maintained allowlist that silently admits stale artifacts.

### Changes

1. Replace ambiguous `implementation_commit`/`code_commit` comparisons with the explicit fields above while preserving backward-compatible reads during migration.
2. Make the generator fail if the working tree is dirty, unless an explicit `--allow-dirty` flag is used for local experiments.
3. Add a release command that:
   - checks out the intended source commit;
   - generates artifacts in a temporary output directory;
   - computes all hashes;
   - writes the release manifest;
   - runs verification against the intended source/release identity;
   - copies artifacts only after all checks pass.
4. Make `verify_provenance.py` validate:
   - current release identity;
   - every artifact hash;
   - source/artifact/policy/model identity agreement;
   - prediction-log population hash and row count;
   - report schema version and generator version.
5. Remove the need for `--allow-commit` in the normal release path.

### Verification command

```bash
python prediction-model/src/verify_provenance.py
```

### Acceptance criteria

- The command exits 0 on a clean checkout.
- No candidate manifest references `9311d3f` unless that commit is explicitly the intended release source and is validated as such.
- All five horizon bundles and all candidate artifacts agree on source, artifact, release, model-family, horizon, and data hashes.
- The provenance tests pass without modifying tracked files.

---

## Phase 2 — Publish the complete prediction population and build a replay evaluator

**Primary files:**

- `prediction-model/src/train_predictive_quality.py`
- `prediction-model/src/monitoring.py`
- `prediction-model/src/verify_provenance.py`
- `prediction-model/src/test_predictive_quality.py`
- `prediction-model/data/candidate_artifacts/candidate_h*h_predictions.csv`
- Add: `prediction-model/src/replay_scorecard.py`
- Add: `prediction-model/src/test_replay_scorecard.py`

### Problem to fix

The scorecard reports thousands of test samples, but each committed candidate CSV contains only 500 rows. The artifact is therefore not sufficient to reproduce its own metrics.

### Changes

1. Remove the production/evaluation truncation that limits candidate prediction logs to 500 rows. A `--max-samples` option may remain for smoke tests, but it must be prohibited in release mode.
2. Write every test-row prediction for each horizon, including:
   - origin and target timestamps;
   - station ID;
   - split name;
   - model family and policy ID;
   - feature-schema hash;
   - label quality status;
   - all truth, prediction, and baseline values used by the metrics.
3. Add to each manifest:
   - `prediction_row_count`;
   - `test_population_sha256`;
   - `prediction_log_sha256`;
   - minimum/maximum target timestamp;
   - station count and per-station counts;
   - metric population definition.
4. Implement `replay_scorecard.py` so it reads only the committed prediction logs and regenerates the scorecard metrics without loading model weights.
5. Compare the replayed values against the stored scorecard with a documented floating-point tolerance.
6. Keep the 500-row mode only as a clearly labeled `SMOKE_SAMPLE`, never as a release artifact.

### Acceptance criteria

- Candidate CSV row counts exactly equal the scorecard sample counts: 2,820, 2,784, 2,731, 2,633, and 2,443, unless the scorecard is intentionally regenerated on a different population.
- Replay reproduces temperature MAE, rain Brier, wind-direction error, precipitation MAE, CSI, and confidence intervals within the declared tolerance.
- A missing, truncated, or altered prediction log causes provenance/replay verification to fail.
- The scorecard and operational monitoring log either become the same canonical population or clearly declare different model/policy IDs and purposes.

### Verification commands

```bash
python prediction-model/src/replay_scorecard.py \
  --artifact-dir prediction-model/data/candidate_artifacts \
  --check-against-scorecard

python prediction-model/src/test_replay_scorecard.py
```

---

## Phase 3 — Replace hard-coded anomaly metrics with reviewed-event evaluation

**Primary files:**

- `prediction-model/src/anomaly_detector.py`
- `prediction-model/src/train_predictive_quality.py`
- `prediction-model/src/test_predictive_quality.py`
- Add: `prediction-model/data/anomaly_reviewed_events.jsonl`
- Add: `prediction-model/src/prepare_anomaly_labels.py`
- Add: `prediction-model/src/evaluate_anomaly_report.py`
- Add: `prediction-model/src/test_anomaly_report.py`

### Label dataset design

Create a versioned reviewed-event file with one record per event:

```json
{
  "event_id": "evt-000001",
  "station_id": "03pqkGAj",
  "affected_variable": "precipitation",
  "event_type": "heavy_rain",
  "timestamp": "2026-08-19T01:00:00+00:00",
  "label": "positive",
  "review_source": "telemetry_review_v1",
  "reviewed_at": "2026-09-28T00:00:00+00:00",
  "label_version": "1.0.0"
}
```

The label set must include reviewed negative monitoring periods, not only positive events. Split labels chronologically and ensure the event labels do not overlap the training/calibration period used to tune thresholds.

### Changes

1. Remove the fixed values currently written for:
   - `heavy_rain_recall`;
   - `extreme_heat_recall`;
   - `rapid_temp_change_recall`;
   - `empirical_false_alarms_per_day`;
   - `false_alarm_rate_per_day`.
2. For every horizon and event type, run actual detector outputs against reviewed labels using `evaluate_anomaly_events_with_budget()` or a stricter event-matching implementation.
3. Report:
   - reviewed event count;
   - monitoring days;
   - detected anomaly count;
   - TP, FP, FN, TN where defined;
   - recall, precision, F1, CSI where appropriate;
   - false alarms/day with confidence interval;
   - per-station and per-event-type metrics;
   - threshold version and label version.
4. Make empty-label behavior fail closed. Do not convert an empty reviewed set into 100% recall.
5. Align the detector version in `anomaly_detector.py`, calibration files, manifests, and reports. The current source version `1.0.0` and artifact version `2.0.0` must not coexist silently.
6. Change anomaly report status to `INSUFFICIENT_LABELS` when sample support is below the declared minimum.

### Acceptance criteria

- No anomaly performance number is emitted unless it comes from the reviewed-event evaluator.
- The report contains non-zero denominators and complete confusion/event counts.
- Re-running the evaluator on the committed label file reproduces the report exactly.
- A test with an empty reviewed-label file fails or returns `INSUFFICIENT_LABELS`, never perfect recall.
- The false-alarm budget is evaluated on a declared monitoring period and cannot be passed by default constants.

### Verification command

```bash
python prediction-model/src/evaluate_anomaly_report.py \
  --events prediction-model/data/anomaly_reviewed_events.jsonl \
  --artifact-dir prediction-model/data/candidate_artifacts \
  --check-against-report
```

---

## Phase 4 — Make dependencies and CI deterministic

**Primary files:**

- `prediction-model/requirements.txt`
- Add: `prediction-model/requirements-lock.txt` or a supported lock mechanism
- `prediction-model/requirements-aws.txt` if AWS probes remain separate
- `.github/workflows/prediction-model.yml`
- `prediction-model/ci/prediction-model.yml`
- `prediction-model/src/test_aws_ca.py`
- `prediction-model/src/test_aws_iot_connect.py`
- `prediction-model/src/test_aws_iot_probe.py`

### Changes

1. Pin and record the tested Python and PyTorch versions. The current requirements use open-ended `torch>=2.0.0`; release CI should use a reproducible tested version.
2. Separate required offline model dependencies from optional live AWS probe dependencies.
3. Mark AWS connectivity tests as optional/integration tests. They should skip with a clear reason when credentials or `awscrt` are unavailable, not fail ordinary model CI at import time.
4. Add a dependency smoke step that imports NumPy, PyTorch, the dataset, model, inference, monitoring, and anomaly modules.
5. Make the workflow run the exact same commands locally and in CI.
6. Keep the current 15-gate workflow, but add explicit replay and anomaly-report gates after artifact generation and before provenance acceptance.
7. Make CI fail if any report is generated with a status of `PASS` while its evidence is missing or insufficient.

### Acceptance criteria

- A clean Python 3.11 environment installs all required offline dependencies from the lock file.
- Canonical, inference, predictive-quality, replay, anomaly, monitoring, and provenance tests run without import errors.
- AWS probes are clearly reported as skipped when not configured, rather than silently omitted or treated as model failures.
- CI fails on stale artifacts, incomplete logs, hard-coded anomaly metrics, or scorecard replay mismatches.

---

## Phase 5 — Align scorecard, monitoring, and policy identities

**Primary files:**

- `prediction-model/src/monitoring.py`
- `prediction-model/src/train_predictive_quality.py`
- `prediction-model/src/inference.py`
- `prediction-model/data/inference_policy.json`
- `prediction-model/data/test_predictions_log.csv`
- `prediction-model/data/candidate_artifacts/model_comparison_report.json`

### Changes

1. Add a common `evaluation_identity` block to every report:
   - `model_family`;
   - `model_version`;
   - `policy_id`;
   - `source_commit`;
   - `artifact_commit`;
   - `data_hashes`;
   - `prediction_log_sha256`;
   - `prediction_row_count`;
   - `test_start`, `test_end`, and `horizons`.
2. Rename reports so “candidate research scorecard” and “operational monitoring scorecard” cannot be confused.
3. State whether each metric is:
   - raw candidate model;
   - hybrid selected policy;
   - persistence fallback;
   - operational monitoring output.
4. Ensure the dashboard-facing policy uses only metrics from the selected operational identity, while research artifacts remain available for comparison.
5. Add a test that rejects a report when model/policy/log identities disagree.

### Acceptance criteria

- The +1h monitoring values and candidate scorecard values are either identical because they use the same identity, or are explicitly labeled as different evaluation products.
- Every displayed metric can be traced to one immutable prediction log and one policy.
- The UI and exported benchmark report do not present candidate research metrics as operational production metrics.

---

## Phase 6 — Improve forecast quality only after evaluation is fixed

**Primary files:**

- `prediction-model/src/train_predictive_quality.py`
- `prediction-model/src/dataset.py`
- `prediction-model/src/model.py`
- `prediction-model/src/test_predictive_quality.py`
- `prediction-model/data/candidate_artifacts/`

### Changes

1. Preserve the current target-specific routing until new results are valid:
   - persistence for short-horizon temperature, humidity, pressure, wind speed, and direction where it wins;
   - candidate/blend only where skill is positive and stable;
   - rain probability as probabilistic guidance rather than a guaranteed binary alert.
2. Add rolling-origin, per-station, and per-regime confidence intervals to promotion gates. A single aggregate test split is not enough for promotion.
3. Add explicit event-weighted evaluation for precipitation thresholds at 1, 2.5, 5, 10, and 25 mm/h, with support counts and confidence intervals.
4. Treat zero or near-zero extreme-event support as `INSUFFICIENT_SUPPORT`, not as a passing result.
5. Recalibrate uncertainty separately by horizon. If observed coverage is outside the declared tolerance, set status to `WARNING` and prevent operational publication.
6. Investigate candidate bias at 3h and 6h, where temperature skill is materially negative, before changing architecture.
7. Expand training/evaluation data across more weather regimes and more independent events before attempting hazardous-rain promotion.

### Promotion thresholds

Use explicit, pre-registered gates such as:

- candidate must be non-inferior to persistence for every promoted continuous target;
- positive skill must hold across the required rolling folds, not only the aggregate test set;
- rain probability must improve Brier score and meet a declared false-alarm/recall tradeoff;
- extreme-rain metrics require a minimum reviewed-event count and non-zero lower confidence bound where applicable;
- uncertainty coverage must be within a declared tolerance for each promoted horizon;
- no promoted model may have unresolved provenance or replay failures.

Do not choose the numerical thresholds after seeing the final test results. Store them in a versioned promotion policy.

---

## Phase 7 — Release validation and rollback

### Clean release procedure

```bash
# 1. Clean checkout and environment
 git status --short
 python --version
 python -m pip install -r prediction-model/requirements-lock.txt

# 2. Static and contract checks
 python -m py_compile prediction-model/src/*.py
 python prediction-model/src/test_canonical_contract.py
 python prediction-model/src/test_inference_contract.py
 python prediction-model/src/test_predictive_quality.py
 python prediction-model/src/test_monitoring.py

# 3. Generate complete artifacts in a temporary release directory
 python prediction-model/src/train_predictive_quality.py \
   --output-dir /tmp/citizendashboard-release-artifacts \
   --release-mode

# 4. Replay every scorecard from committed/generated prediction logs
 python prediction-model/src/replay_scorecard.py \
   --artifact-dir /tmp/citizendashboard-release-artifacts \
   --check-against-scorecard

# 5. Evaluate reviewed anomaly events
 python prediction-model/src/evaluate_anomaly_report.py \
   --events prediction-model/data/anomaly_reviewed_events.jsonl \
   --artifact-dir /tmp/citizendashboard-release-artifacts \
   --check-against-report

# 6. Generate/check bundles and provenance
 python prediction-model/src/generate_manifests.py --check-only
 python prediction-model/src/verify_provenance.py

# 7. Run operational monitoring against the declared log
 python prediction-model/src/monitoring.py \
   --data-dir prediction-model/data \
   --predictions-log /tmp/citizendashboard-release-artifacts/operational_predictions.csv \
   --horizons 1 3 6 12 24 \
   --require-all-horizons
```

### Release gate

A release is **blocked** if any of these occur:

- provenance verification fails;
- prediction-log count does not equal scorecard count;
- scorecard replay differs from stored metrics beyond tolerance;
- anomaly report contains missing labels, default constants, or insufficient support while claiming `PASS`;
- uncertainty report claims nominal coverage that fails its tolerance;
- full required test suite has import errors or failures;
- extreme-rain promotion lacks minimum event support;
- the operational policy and displayed metrics use different evaluation identities.

### Rollback

Keep the existing five-horizon canonical bundles as the rollback path. A new candidate may be promoted only after the new release passes the full gates. If any post-release monitoring identity, metric schema, or anomaly budget check fails, route the affected target/horizon back to its last verified persistence or canonical bundle policy.

## Deliverables by phase

| Phase | Required deliverables |
|---|---|
| 0 | Baseline/release record and historical-artifact designation. |
| 1 | Provenance schema, clean release generator, passing verifier, regenerated artifacts. |
| 2 | Full prediction logs, replay evaluator, population hashes, replay tests. |
| 3 | Reviewed anomaly labels, label-preparation process, measured anomaly report, anomaly tests. |
| 4 | Locked dependencies, optional AWS test handling, green deterministic CI. |
| 5 | Shared evaluation identity schema and aligned monitoring/policy reports. |
| 6 | Valid multi-fold quality scorecards, calibrated uncertainty report, extreme-event benchmark. |
| 7 | Reproducible release runbook, pass/fail gates, and tested rollback procedure. |

## Recommended execution order

Implement **Phases 0–3 before retraining**. These phases address the validity of the evidence rather than the model architecture. Implement **Phase 4 in parallel** so the fixes can be tested in a clean environment. Implement **Phase 5 before any dashboard or operational claim changes**. Only after those gates pass should Phase 6 investigate model improvements. Finish with Phase 7 and do not promote the candidate until the complete release gate is green.
