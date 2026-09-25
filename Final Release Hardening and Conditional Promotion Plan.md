# Final Release Hardening and Conditional Promotion Plan

## Executive objective

The prediction-model implementation is now in a strong research-release state at GitHub commit `ba274016cd6028f7d357ef663a9b104725c32f29`.

The latest repository audit passed the complete test and provenance suite. Candidate checkpoints, manifests, calibration artifacts, prediction logs, scorecards, and rollback code are committed. The scorecard now separates research and operational decisions.

The remaining work is release hardening. The candidate artifacts were generated at the previous implementation commit, the active production policy still contains incomplete metadata, and the scorecard’s artifact inventory lists only one horizon even though all five are present.

The final target is:

```text
Research decision: GO
Operational decision: CONDITIONAL_GO until every release gate passes
Active production replacement: only after explicit target-level promotion gates pass
```

The candidate must not replace the current baseline globally merely because the automated test suite passes. Promotion must remain target-specific and evidence-based.

## Current state

### Repository

| Item | Current state |
|---|---|
| Final GitHub commit | `ba274016cd6028f7d357ef663a9b104725c32f29` |
| Candidate artifacts | Committed for all five horizons |
| Candidate model family | `MF-1-FEATURED` |
| Candidate context dimension | 75 |
| Candidate seed | 42 |
| Candidate status | `CANDIDATE_RESEARCH` |
| Research decision | `GO` |
| Operational decision | `CONDITIONAL_GO` |
| Active production model | Existing `GarciaWeatherLNN` baseline |
| UV status | Blocked by sensor calibration |
| Luminosity status | Daylight-only beta |
| Archive branch | Present on GitHub |

### Verified gates

The latest clean detached worktree passed:

- Python compilation;
- 15 canonical contract tests;
- 11 inference tests;
- 10 monitoring tests;
- 31 predictive-quality tests;
- 14 provenance tests;
- smoke tests;
- provenance verification;
- whitespace checks;
- clean-worktree checks.

### Remaining release gaps

| Gap | Why it matters | Resolution |
|---|---|---|
| Candidate artifacts reference commit `0153a795...`, not final commit `ba274016...` | The artifact identity is not exact at the final implementation tip | Regenerate artifacts at the final commit and commit the regenerated outputs |
| Active H1 policy contains null metadata fields | Inference policy identity cannot be independently audited | Regenerate or repair all active policies and validate their hashes |
| Scorecard lists only the H24 artifact group in its generated-artifact field | The scorecard understates the actual five-horizon artifact set | Populate the field from the full manifest summary |
| Operational decision is conditional | The candidate still has short-horizon temperature and wind-direction weaknesses | Use target-specific promotion and retain baseline/persistence where needed |
| Active bundles remain baseline bundles | This is correct until conditional gates pass | Do not replace them globally; promote only approved target sources |

## Phase 1 — Create a clean final-release worktree

### Goal

Ensure all release work begins from the exact GitHub commit that will be audited and pushed.

### Implementation

Create a detached worktree from the current remote main:

```bash
git fetch origin main --prune
git worktree add --detach /tmp/prediction-final-release origin/main
cd /tmp/prediction-final-release
```

Confirm:

```bash
git rev-parse HEAD
```

The result must be:

```text
ba274016cd6028f7d357ef663a9b104725c32f29
```

Do not train or regenerate artifacts from a dirty branch. Do not use local development files that are absent from this worktree.

### Acceptance criteria

The worktree is detached at the exact final implementation commit, contains the protected raw telemetry, and has no untracked or modified files before regeneration.

## Phase 2 — Fix the candidate artifact regeneration contract

### Goal

Ensure all candidate artifacts explicitly describe the exact implementation commit at which they were generated.

### Required artifact fields

Every candidate manifest must contain:

```text
implementation_commit
artifact_commit
model_weights_commit
weather_telemetry_sha256
water_telemetry_sha256
checkpoint_sha256
calibration_sha256
predictions_sha256
model_family
horizon_hours
input_dimension
context_dimension
feature_schema
seed
training_config
status
```

The implementation, artifact, and model-weight commit fields must equal the final code commit unless the project intentionally documents a multi-commit artifact workflow. For this release, use the exact final commit for all three fields.

The prediction-log hash must be added if it is not already present. This prevents a scorecard from referencing a changed prediction file.

### Required trainer changes

Update `prediction-model/src/train_predictive_quality.py` so that:

1. It obtains the current Git commit from the release worktree.
2. It writes that commit into every candidate manifest.
3. It computes hashes only after all files are complete.
4. It records the prediction-log hash.
5. It fails if any horizon is missing.
6. It emits one complete manifest summary for all five horizons.
7. It writes relative repository artifact paths only.
8. It never writes temporary absolute paths into canonical JSON artifacts.

### Acceptance criteria

After regeneration, all five candidate manifests must report:

```text
implementation_commit = ba274016cd6028f7d357ef663a9b104725c32f29
artifact_commit = ba274016cd6028f7d357ef663a9b104725c32f29
model_weights_commit = ba274016cd6028f7d357ef663a9b104725c32f29
```

## Phase 3 — Regenerate all candidate artifacts at the final commit

### Goal

Replace the stale candidate artifacts with provenance-correct artifacts generated from the final code.

### Required command

Run the complete real-data training and evaluation pipeline:

```bash
cd /tmp/prediction-final-release/prediction-model
. /home/ubuntu/citizendashboard/prediction-model/.venv/bin/activate
PYTHONUNBUFFERED=1 python src/train_predictive_quality.py \
  --epochs 10 \
  --seed 42 \
  --output-dir data/candidate_artifacts
```

The run must produce all five horizon groups:

```text
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

It must also regenerate:

```text
baseline_manifest.json
candidate_manifest_summary.json
model_comparison_report.json
predictive_quality_scorecard.json
```

### Acceptance criteria

The pipeline must complete successfully on real telemetry. Every horizon must have nonzero train, calibration, and test samples. No raw telemetry file may be modified. No synthetic target labels may be introduced.

## Phase 4 — Make the scorecard inventory complete

### Goal

Ensure the scorecard reports the full five-horizon artifact set rather than only the last loop iteration.

### Required scorecard field

The scorecard must list all generated artifacts, for example:

```json
{
  "artifacts_generated": [
    "candidate_h1h.pt",
    "candidate_h1h_manifest.json",
    "candidate_h1h_calibration.json",
    "candidate_h1h_predictions.csv",
    "candidate_h3h.pt",
    "candidate_h3h_manifest.json",
    "candidate_h3h_calibration.json",
    "candidate_h3h_predictions.csv",
    "candidate_h6h.pt",
    "candidate_h6h_manifest.json",
    "candidate_h6h_calibration.json",
    "candidate_h6h_predictions.csv",
    "candidate_h12h.pt",
    "candidate_h12h_manifest.json",
    "candidate_h12h_calibration.json",
    "candidate_h12h_predictions.csv",
    "candidate_h24h.pt",
    "candidate_h24h_manifest.json",
    "candidate_h24h_calibration.json",
    "candidate_h24h_predictions.csv"
  ]
}
```

The preferred implementation is to build this list during the horizon loop and write it after the loop. Do not construct it from the final value of the loop variable.

### Acceptance criteria

The scorecard artifact list contains all five horizons, and the manifest summary contains matching hashes and file sizes for every listed file.

## Phase 5 — Repair active production policy metadata

### Goal

Make the existing baseline bundles independently auditable before any candidate promotion occurs.

The active bundles are currently the baseline model and must remain active while the candidate is conditional. However, the active policy metadata must not contain null identity fields.

For each active bundle under:

```text
prediction-model/data/bundles/h1/
prediction-model/data/bundles/h3/
prediction-model/data/bundles/h6/
prediction-model/data/bundles/h12/
prediction-model/data/bundles/h24/
```

ensure that `bundle_manifest.json` and `inference_policy.json` agree on:

```text
model_family
model_status
bundle_version
implementation_commit
artifact_commit
model_weights_commit
checkpoint_sha256
policy_sha256
feature_schema
horizon_hours
```

The policy hash must be computed over the final policy file according to one documented convention. Avoid a circular hash field unless the repository already defines a canonical procedure for excluding the hash field during computation.

### Acceptance criteria

For every active horizon:

- policy and bundle model family agree;
- status is `ACTIVE_PRODUCTION`;
- checkpoint hash matches the checkpoint file;
- policy hash matches the policy file;
- implementation and artifact commits are present;
- schema and horizon agree;
- inference loads successfully.

## Phase 6 — Preserve target-specific conditional promotion

### Goal

Avoid globally promoting a candidate whose quality is mixed across targets and horizons.

The candidate’s current evidence indicates:

- rain probability improves against persistence;
- precipitation amount shows useful candidate behavior but must be reviewed by regime;
- temperature improves at longer horizons but is worse at 1h, 3h, and 6h;
- wind direction is slightly worse than persistence at every horizon;
- heat index is derived and physically checked;
- UV remains blocked;
- luminosity remains daylight beta.

The policy should therefore support target-specific sources. A candidate can be promoted for a target only when that target passes its own gate.

### Required policy behavior

The candidate policy must allow entries such as:

```text
temperature: baseline for 1h, 3h, and 6h; candidate only where approved
rain occurrence: candidate where calibration and event metrics pass
precipitation amount: candidate only where rainy-hour and heavy-rain metrics pass
wind direction: persistence until candidate improves or is statistically non-inferior
heat index: derived from selected temperature and humidity
UV: blocked
luminosity: daylight beta
```

Do not label the entire candidate model `ACTIVE_PRODUCTION` if one target is still using baseline or persistence. Use target-specific policy metadata.

### Acceptance criteria

The operational scorecard must state the source used for every target and horizon. The inference response must expose those source selections and the bundle version.

## Phase 7 — Strengthen the operational release gate

### Goal

Prevent a five-percent tolerance from being interpreted as demonstrated accuracy improvement.

Keep the two decision levels:

```text
research_decision
operational_decision
```

Use the following principles:

- `research_decision = GO` means artifacts, tests, reproducibility, and feasibility checks pass.
- `operational_decision = GO` requires critical target gates to pass.
- A target may remain on baseline or persistence without blocking candidate use for a different target.
- A global operational `GO` is prohibited when a critical target has a material regression.

### Recommended target gates

| Target | Gate |
|---|---|
| Temperature | Candidate must be better or statistically non-inferior at each promoted horizon; repeated short-horizon regressions require baseline retention |
| Rain probability | Brier score and calibration must improve or meet a documented non-inferiority interval |
| Precipitation amount | Rainy-hour and heavy-rain results must be reported and must not materially degrade |
| Wind direction | Candidate must improve or meet a confidence-bound non-inferiority rule; otherwise persistence remains selected |
| Heat index | Derived value must be consistent with selected temperature and humidity |
| UV | Must remain blocked until sensor calibration is validated |
| Luminosity | Must remain daylight beta until station and daylight validation passes |
| Anomalies | Operational quality claims require reviewed or real labels |

At minimum, add paired bootstrap intervals or another uncertainty estimate for candidate-versus-baseline differences. A raw five-percent threshold may remain as a preliminary screen but cannot be the only basis for production promotion.

### Acceptance criteria

The scorecard must not report global operational `GO` if the candidate is worse at several critical horizons and no uncertainty analysis supports non-inferiority. It must report the reason each target is candidate, baseline, persistence, blocked, or beta.

## Phase 8 — Validate candidate and baseline inference paths

### Goal

Prove that the evaluated candidate artifacts and active baseline bundles both load correctly and that rollback works.

Run tests for:

1. candidate checkpoint loading;
2. candidate manifest hash validation;
3. candidate calibration loading;
4. candidate feature-schema validation;
5. candidate prediction output fields;
6. active baseline bundle loading;
7. target-specific policy selection;
8. candidate-to-baseline rollback;
9. tampered checkpoint rejection;
10. tampered policy rejection;
11. unsupported horizon rejection;
12. blocked UV request rejection or explicit blocked status.

### Acceptance criteria

A clean process must load each of the five candidate horizons. A clean process must load each of the five active baseline bundles. Rollback must return valid forecasts and must expose the baseline bundle identity.

## Phase 9 — Commit and regenerate at the final artifact commit

### Goal

Satisfy the exact-HEAD provenance requirement after code and artifact changes are committed.

Use this order:

```bash
git add prediction-model/src prediction-model/data/candidate_artifacts prediction-model/data/bundles
 git diff --cached --check
git diff --cached --stat
git commit -m "Finalize candidate release provenance and policy metadata"
```

Record the new commit. Because the commit hash changes after committing, rerun the complete candidate artifact generation at that new commit. Stage the regenerated artifacts and create a second artifact-only commit if required by the project convention.

Then run:

```bash
python prediction-model/src/verify_provenance.py
```

The final artifact manifests must reference the exact final commit. Do not stop at an earlier commit that is merely accepted by an allowlist.

### Acceptance criteria

The final commit and all candidate manifest commit fields are identical. The provenance verifier passes without a historical-commit override. The candidate summary hashes match the files in the final tree.

## Phase 10 — Run final release gates

Run from a fresh detached checkout of the final commit:

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

Also run independent artifact checks:

```bash
for h in 1 3 6 12 24; do
  sha256sum prediction-model/data/candidate_artifacts/candidate_h${h}h.pt
  sha256sum prediction-model/data/candidate_artifacts/candidate_h${h}h_calibration.json
  jq '.checkpoint_sha256,.calibration_sha256' \
    prediction-model/data/candidate_artifacts/candidate_h${h}h_manifest.json
done
```

### Acceptance criteria

All commands pass. The final worktree is clean. The scorecard contains all five horizons. All candidate and active bundle hashes match. No raw telemetry file changed.

## Phase 11 — Push decision

Push the final commit to `origin/main` only if:

- exact-HEAD provenance passes;
- candidate artifacts are complete;
- active policy metadata is complete;
- candidate and baseline inference paths load;
- rollback passes;
- scorecard and policy agree;
- operational status is honestly represented;
- no protected workflow blocks the push.

If operational promotion remains conditional, push the research artifacts and target-specific policy only. Do not replace active baseline bundles globally.

## Final report requirements

The final report must state:

- starting commit;
- final commit;
- archive branch;
- changed files;
- retained canonical files;
- candidate artifact paths;
- all five checkpoint hashes;
- all five calibration hashes;
- weather and water dataset hashes;
- quarantine counts;
- seed and training configuration;
- complete five-horizon metrics;
- target-specific source policy;
- heat-index results;
- UV and luminosity statuses;
- anomaly validation status;
- active baseline bundle hashes;
- candidate bundle hashes, if promoted;
- rollback test result;
- provenance result;
- complete test results;
- exact push result or blocker;
- separate research and operational recommendations.

Use one of these final recommendations:

```text
GO — candidate promoted for the approved targets and horizons
CONDITIONAL GO — candidate committed for research or target-specific use; baseline remains active for other targets
NO-GO — candidate retained as research-only; baseline remains active
```

## Expected outcome for the current repository

The most likely correct outcome after exact-HEAD regeneration is:

```text
Research decision: GO
Operational decision: CONDITIONAL_GO
Candidate rain policy: eligible if final calibration and event gates remain valid
Candidate temperature policy: limited to horizons that pass target-level gates
Wind direction: retain persistence unless uncertainty analysis supports promotion
Heat index: derived from selected temperature and humidity
UV: blocked
Luminosity: daylight beta
Baseline bundles: remain active until target-specific promotion is approved
```

## References

[1]: https://github.com/benben000000/citizendashboard "Citizen Dashboard repository"
[2]: https://github.com/benben000000/citizendashboard/tree/main/prediction-model "Citizen Dashboard prediction-model directory"
