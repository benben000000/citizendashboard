# Next-Step Implementation Plan: Exact-HEAD Provenance Closure and Target-Specific Rollout

## Objective

The prediction-model repository is now at GitHub commit:

```text
5411a7370a5e8c3a09865bc400dc491d21cf4434
```

The current repository state is structurally healthy:

- all canonical, inference, monitoring, predictive-quality, provenance, and smoke tests pass;
- all five candidate artifact groups are committed;
- candidate checkpoint, calibration, and prediction-log hashes match their manifests;
- all five active baseline bundles pass `generate_bundles.py --check-only`;
- the scorecard reports `research_decision = GO`;
- the scorecard reports `operational_decision = CONDITIONAL_GO`;
- target-specific rollout is implemented;
- UV remains blocked and luminosity remains daylight beta.

The next objective is to close the remaining provenance gap and prepare a controlled target-specific rollout without replacing the baseline globally.

> **Next release target:** regenerate every candidate artifact at the exact final implementation commit, verify the final commit without a historical-commit exception, and roll out only the approved candidate targets while retaining baseline and persistence fallbacks.

## Current release policy

The current policy should remain unchanged during provenance closure:

| Target | Current source |
|---|---|
| Temperature | Baseline |
| Humidity | Baseline |
| Pressure | Baseline |
| Wind speed | Candidate |
| Wind direction | Persistence |
| Rain occurrence | Candidate |
| Precipitation amount | Candidate, subject to final scorecard review |
| Heat index | Derived from selected temperature and humidity |
| UV index | Blocked by sensor calibration |
| Luminosity | Daylight-only beta |

The candidate must not be promoted globally. The next release is a **target-specific rollout**, not a full model replacement.

## Phase 1 — Create the exact final release workspace

### Goal

Regenerate artifacts from the exact commit that will be published and audited.

### Steps

From a clean local clone:

```bash
git fetch origin main --prune
git worktree add --detach /tmp/prediction-next-release origin/main
cd /tmp/prediction-next-release
printf 'HEAD='; git rev-parse HEAD
```

Confirm that the output is exactly:

```text
HEAD=5411a7370a5e8c3a09865bc400dc491d21cf4434
```

Confirm that the worktree is clean:

```bash
git status --short
```

Confirm that both protected raw telemetry files are present and unchanged:

```bash
sha256sum prediction-model/data/weather_telemetry.csv
sha256sum prediction-model/data/water_level_telemetry.csv
```

The hashes must match the hashes recorded in the current candidate manifests.

### Acceptance criteria

The release workspace is detached at the exact remote `main` tip, contains no modifications, and contains the protected raw telemetry.

## Phase 2 — Verify the trainer’s exact-commit behavior

### Goal

Ensure the candidate trainer records the exact final implementation commit in every generated artifact.

### Required manifest fields

Every horizon manifest must include the same final commit in:

```text
implementation_commit
artifact_commit
model_weights_commit
```

For this release, the required value is:

```text
5411a7370a5e8c3a09865bc400dc491d21cf4434
```

Every manifest must also include:

- horizon;
- model family;
- input dimension;
- context dimension;
- feature schema;
- seed;
- training configuration;
- weather hash;
- water hash;
- checkpoint hash;
- calibration hash;
- prediction-log hash;
- candidate status;
- relative filenames.

### Implementation check

Before running the full training command, inspect the trainer’s commit and hashing paths:

```bash
rg -n 'implementation_commit|artifact_commit|model_weights_commit|predictions_sha256|git rev-parse|sha256' \
  prediction-model/src/train_predictive_quality.py
```

If the trainer still obtains a parent or historical commit rather than `HEAD`, fix it before regenerating artifacts.

### Acceptance criteria

A one-horizon dry run or manifest test confirms that the generated manifest uses the exact current `HEAD` and includes all required hashes.

## Phase 3 — Regenerate all five candidate artifact groups

### Goal

Replace the candidate artifacts generated at `4f11b0d5...` with artifacts generated at `5411a737...`.

### Required command

Use the declared environment and real telemetry:

```bash
cd /tmp/prediction-next-release/prediction-model
. /home/ubuntu/citizendashboard/prediction-model/.venv/bin/activate
PYTHONUNBUFFERED=1 python src/train_predictive_quality.py \
  --epochs 10 \
  --seed 42 \
  --output-dir data/candidate_artifacts
```

The command must generate all five horizons:

```text
1h, 3h, 6h, 12h, 24h
```

It must regenerate:

```text
baseline_manifest.json
candidate_manifest_summary.json
model_comparison_report.json
predictive_quality_scorecard.json
```

It must also regenerate the 20 horizon-specific files:

```text
five checkpoints
five manifests
five calibration artifacts
five prediction logs
```

### Data integrity requirements

The run must:

- use real telemetry only;
- preserve the raw telemetry files;
- preserve the declared train, calibration, and test split rules;
- preserve seed `42` unless a new seed is intentionally approved;
- produce nonzero samples for all required splits;
- retain UV quarantine status;
- retain luminosity daylight-beta status.

### Acceptance criteria

Every candidate artifact is generated successfully. All manifests report the exact final commit. The scorecard reports all five horizons and all 20 horizon-specific artifacts.

## Phase 4 — Validate exact hashes independently

### Goal

Verify the generated files without relying only on the trainer’s internal checks.

For every horizon, independently compute and compare:

```bash
for h in 1 3 6 12 24; do
  sha256sum data/candidate_artifacts/candidate_h${h}h.pt
  sha256sum data/candidate_artifacts/candidate_h${h}h_calibration.json
  sha256sum data/candidate_artifacts/candidate_h${h}h_predictions.csv
  jq '{implementation_commit,artifact_commit,model_weights_commit,checkpoint_sha256,calibration_sha256,predictions_sha256}' \
    data/candidate_artifacts/candidate_h${h}h_manifest.json
done
```

Verify that:

- checkpoint file hash equals manifest checkpoint hash;
- calibration file hash equals manifest calibration hash;
- prediction-log hash equals manifest prediction hash;
- all three commit fields equal `5411a737...`;
- weather and water hashes are consistent across manifests;
- horizons are exactly 1, 3, 6, 12, and 24.

### Acceptance criteria

No hash mismatch exists. No stale `4f11b0d5...` commit remains in candidate manifests, summary files, or scorecards unless it is explicitly recorded as historical lineage rather than current artifact identity.

## Phase 5 — Validate the final scorecard and policy

### Goal

Confirm that exact-HEAD regeneration did not change the target-specific rollout decision unexpectedly.

### Required scorecard decisions

The scorecard must contain:

```text
research_decision
operational_decision
operational_target_status
target_specific_source_policy
```

The expected policy remains:

```text
temperature: baseline
humidity: baseline
pressure: baseline
wind_speed: candidate
wind_direction: persistence
precipitation_occurrence: candidate
precipitation_amount: candidate, subject to final metrics
heat_index: derived_noaa
uv_index: blocked
light_intensity: daylight_beta
```

### Required metric review

Review all five horizons for:

- temperature MAE versus persistence;
- rain Brier score versus persistence;
- rain calibration error;
- precipitation rainy-hour and heavy-rain metrics;
- wind-direction circular error versus persistence;
- physical-bound violations;
- station-level worst cases;
- sample counts;
- quarantine counts.

If the regenerated scorecard materially changes the decision, do not force the prior rollout policy. Reassess the target-specific source selection from the new evidence.

### Acceptance criteria

The scorecard and policy agree exactly. The scorecard does not claim global candidate promotion. Critical targets with regressions remain on baseline or persistence.

## Phase 6 — Run all release gates at the exact final commit

### Goal

Prove that the final exact-HEAD artifact tree is valid.

Run:

```bash
cd /tmp/prediction-next-release/prediction-model
. /home/ubuntu/citizendashboard/prediction-model/.venv/bin/activate
python -m py_compile src/*.py
PYTHONPATH=src python src/test_canonical_contract.py
PYTHONPATH=src python src/test_inference_contract.py
PYTHONPATH=src python src/test_monitoring.py
PYTHONPATH=src python src/test_predictive_quality.py
PYTHONPATH=src python src/test_provenance.py
PYTHONPATH=src python src/smoke_test.py
PYTHONPATH=src python src/generate_bundles.py --check-only
PYTHONPATH=src python src/verify_provenance.py
git diff --check
git status --short
```

The provenance command must be run without an historical commit override. The final test result must verify exact `HEAD` identity.

### Acceptance criteria

All gates pass. The worktree may contain only the intentionally regenerated candidate artifacts before staging. After staging and committing, a fresh detached checkout must be clean after all read-only checks.

## Phase 7 — Commit exact-HEAD artifacts

### Goal

Make the final candidate artifacts reproducible from GitHub.

Review the staged changes:

```bash
git add prediction-model/data/candidate_artifacts

git diff --cached --check
git diff --cached --stat
git diff --cached --name-status
```

Confirm that no following items are staged:

- raw telemetry replacements;
- virtual environments;
- temporary logs;
- local reports outside the canonical artifact directory;
- credentials;
- machine-specific paths;
- unrelated dashboard files.

Commit the regenerated artifacts:

```bash
git commit -m "Regenerate candidate artifacts at exact release head"
```

### Important provenance rule

If the artifact commit changes the implementation `HEAD`, the manifests must be regenerated again at the new commit. The final sequence is therefore:

1. regenerate at current code tip;
2. commit artifacts;
3. record the new final commit;
4. regenerate again at that new final commit if commit identity is embedded in artifacts;
5. commit the second regeneration;
6. run provenance verification on the final tip.

Do not accept an allowlist-based pass as the final state when exact-HEAD regeneration is required.

## Phase 8 — Prepare target-specific operational rollout

### Goal

Promote only the targets that pass their gates while preserving the baseline everywhere else.

The rollout must not copy the candidate checkpoint into every active bundle automatically. Instead, update the policy and bundle metadata per target and horizon.

### Required rollout behavior

Candidate may be selected for:

- wind speed;
- rain occurrence;
- precipitation amount, if final amount metrics remain acceptable.

Baseline or persistence must remain selected for:

- temperature where candidate is weaker;
- humidity;
- pressure;
- wind direction.

Derived or blocked targets remain:

- heat index from selected temperature and humidity;
- UV blocked;
- luminosity daylight beta.

### Bundle requirements

Each active bundle must expose:

- model family;
- model status;
- bundle version;
- horizon;
- implementation commit;
- artifact commit;
- model weights commit;
- checkpoint hash;
- policy hash;
- target-specific source policy;
- rollback bundle identity.

If the candidate is not approved for a target, the active bundle must continue using the baseline or persistence source for that target.

### Acceptance criteria

Inference returns the same target-specific source policy represented in the scorecard. A rollback call restores the baseline bundle and produces valid forecasts.

## Phase 9 — Final clean-checkout verification

### Goal

Ensure the release can be reconstructed from the public repository without local state.

Create a fresh detached worktree from the final remote commit and run:

```bash
git worktree add --detach /tmp/prediction-final-verification origin/main
cd /tmp/prediction-final-verification/prediction-model
PYTHONPATH=src python src/test_provenance.py
PYTHONPATH=src python src/generate_bundles.py --check-only
PYTHONPATH=src python src/verify_provenance.py
git status --short
git diff --check
```

### Acceptance criteria

The fresh checkout passes all checks without relying on files from the original worktree. The final manifests identify the final commit exactly.

## Phase 10 — Push and operational decision

### Push requirements

Push only when:

- exact-HEAD provenance passes;
- all five candidate artifacts are present;
- all independent hashes match;
- active bundle checks pass;
- scorecard and policy agree;
- target-specific rollout is explicit;
- rollback passes;
- UV and luminosity limitations remain visible;
- the final worktree is clean.

### Expected outcome

The expected outcome is:

```text
research_decision: GO
operational_decision: CONDITIONAL_GO
```

The candidate should be used selectively for approved targets. The baseline should remain active for temperature, humidity, pressure, and wind direction until those targets demonstrate improvement or statistically defensible non-inferiority.

If exact-HEAD artifacts cannot be regenerated cleanly, use:

```text
NO-GO for rollout
```

and retain the current baseline bundles.

## Final report

The completion report must include:

- starting commit;
- final commit;
- exact candidate artifact commit fields;
- all five checkpoint hashes;
- all five calibration hashes;
- all five prediction-log hashes;
- weather and water hashes;
- quarantine counts;
- seed and training configuration;
- full five-horizon scorecard;
- target-specific source policy;
- active bundle metadata;
- rollback result;
- test counts and results;
- bundle check result;
- exact provenance result;
- push result or blocker;
- final research decision;
- final operational decision.

Use one final status:

```text
GO — target-specific rollout approved
CONDITIONAL_GO — exact artifacts committed, but some targets remain on baseline or persistence
NO-GO — provenance or quality gates failed; baseline remains active
```

## References

[1]: https://github.com/benben000000/citizendashboard "Citizen Dashboard repository"
[2]: https://github.com/benben000000/citizendashboard/tree/main/prediction-model "Citizen Dashboard prediction-model directory"
