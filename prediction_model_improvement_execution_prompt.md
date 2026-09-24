# Autonomous Execution Prompt: Improve and Harden the Prediction Model

You are an autonomous senior ML/software engineer responsible for improving the prediction-model pipeline in:

`https://github.com/benben000000/citizendashboard`

Work directly from a fresh clone or clean worktree of `origin/main`. Execute the work completely; do not stop at a plan. Implement the changes, add tests, regenerate required artifacts, review the diff, commit, and push only when all release gates pass.

The current model is a five-horizon weather forecasting pipeline with horizons:

```text
1h, 3h, 6h, 12h, 24h
```

The current architecture includes:

- MF-1 GarciaWeatherLNN/PyTorch weather model.
- MF-2 standalone recurrent model.
- Real weather and water telemetry.
- Horizon-specific inference policy.
- Persistence fallback.
- Hybrid rain-probability blending.
- Scorecards and provenance manifests.
- Beta-only water-level forecasting.
- Explicitly unavailable weather uncertainty intervals.

## Main objective

Improve the current implementation so that it is:

1. **Operationally consistent** — the API cannot bypass the validated policy.
2. **Reproducible** — every model, policy, scorecard, and manifest has unambiguous provenance.
3. **Testable** — tests are deterministic, comprehensive, and do not modify tracked artifacts.
4. **Observable** — performance is monitored by horizon, station, variable, event regime, and data quality.
5. **Honest** — documentation and API output claim only capabilities supported by current evidence.
6. **Extensible** — precipitation modeling, uncertainty, packaging, rollback, and CI can be improved without breaking the canonical contract.

## Repository and safety requirements

1. Preserve the archive branch:

   ```text
   cleanup/archive-before-remediation-20260922
   ```

2. Record the starting commit before editing.
3. Do not delete raw telemetry:
   - `prediction-model/data/weather_telemetry.csv`
   - `prediction-model/data/water_level_telemetry.csv`
4. Preserve canonical source and test files unless a reviewed replacement is required.
5. Do not modify dashboard UI or unrelated application code.
6. Do not commit:
   - virtual environments;
   - temporary scripts;
   - secrets;
   - credentials;
   - local audit logs;
   - machine-specific paths;
   - unreviewed generated outputs.
7. Do not weaken tests or provenance rules to obtain a passing result.
8. Keep water-level forecasting explicitly beta and not for life-safety use.
9. Keep UV forecasting blocked until sensor calibration is validated.

## Phase 0: Baseline and inventory

Before editing:

1. Fetch `origin/main`.
2. Record:
   - starting commit;
   - branch status;
   - Python version;
   - dependency versions;
   - archive branch reference.
3. Inventory every file under `prediction-model/`.
4. Identify canonical files, generated artifacts, legacy files, and runtime outputs.
5. Run the current tests and record baseline results.
6. Run the current provenance verifier.
7. Run `git status --short` and confirm the worktree is clean.

Create a temporary audit report outside the repository if needed. Do not commit it unless intentionally converted into a project artifact.

## Phase 1: Separate read-only validation from artifact generation

The current test process can modify tracked manifests. Correct this.

### Required design

Separate artifact generation from test validation:

```bash
python prediction-model/src/generate_manifests.py --output-dir prediction-model/data
```

must be the explicit artifact-generation path.

Tests such as:

```bash
python prediction-model/src/test_canonical_contract.py
python prediction-model/src/test_provenance.py
```

must be read-only with respect to tracked repository files.

### Implementation requirements

1. Refactor manifest-generation helpers to accept an explicit output directory or output paths.
2. In tests, use a temporary directory for generated comparison artifacts.
3. Never overwrite files in `prediction-model/data/` during tests.
4. Add a test or shell gate that records the repository status before tests and verifies it is unchanged afterward.
5. Add CI behavior equivalent to:

   ```bash
   git diff --exit-code
   ```

   after all read-only tests.
6. If a test intentionally needs regenerated artifacts, make that explicit in its name and documentation.

## Phase 2: Make model-policy packaging unambiguous

The current model and inference policy are separate artifacts. Improve this without breaking backwards compatibility.

### Required bundle structure

Create a reproducible model bundle for each supported horizon or a clearly documented shared bundle:

```text
prediction-model/data/bundles/h1/
├── checkpoint.pt
├── inference_policy.json
├── bundle_manifest.json
└── README.md
```

Repeat for 3h, 6h, 12h, and 24h, or document and justify a shared bundle.

### Bundle manifest requirements

Each `bundle_manifest.json` must include:

- `bundle_version`;
- `horizon_hours`;
- `implementation_commit`;
- `artifact_commit`;
- `checkpoint_sha256`;
- `policy_sha256`;
- raw weather dataset SHA-256;
- raw water dataset SHA-256;
- feature schema;
- normalization means and standard deviations or their manifest reference;
- model family;
- model dimensions;
- random seed;
- training configuration;
- training timestamp;
- model status;
- uncertainty status;
- water-level safety status.

### Inference requirements

Update `prediction-model/src/inference.py` so that:

1. It can load a validated bundle.
2. It validates all bundle hashes before serving.
3. It rejects a mismatched checkpoint and policy.
4. It rejects a mismatched horizon.
5. It rejects unsupported horizons.
6. It fails closed if a required manifest field is missing.
7. It preserves the existing eight-feature input contract.
8. It preserves NaN/Inf rejection.
9. It exposes a helper such as:

   ```python
   predictor.supported_horizons()
   ```

10. It exposes the active bundle and policy version in the response.
11. It keeps `research_projected_sequence()` explicitly research-only and separate from operational inference.

Maintain compatibility with the existing checkpoint/policy paths only through a documented fallback loader that performs the same validation.

## Phase 3: Strengthen provenance semantics

Choose and implement one explicit provenance model. Prefer the following two-commit-aware design:

```json
{
  "implementation_commit": "...",
  "artifact_commit": "...",
  "model_weights_commit": "..."
}
```

### Required rules

1. `implementation_commit` identifies the source code used to generate the artifact.
2. `artifact_commit` identifies the Git commit containing the artifact.
3. `model_weights_commit` identifies the commit containing the exact weights if different.
4. The verifier must check all three where applicable.
5. Do not silently accept arbitrary recent commits.
6. Historical overrides must be explicit, for example:

   ```bash
   python prediction-model/src/verify_provenance.py --allow-commit <sha>
   ```

7. The default verifier must clearly report whether it is checking:
   - exact final-tree provenance; or
   - an explicitly supported implementation/artifact provenance model.
8. Add tests for:
   - stale implementation commit;
   - stale artifact commit;
   - mismatched model-policy commit;
   - mismatched checkpoint hash;
   - mismatched policy hash;
   - missing provenance field;
   - explicit historical override.

If strict final-tree provenance is chosen instead, regenerate all artifacts after the final commit until every artifact matches the final SHA. Document the chosen approach in `README.md` and `MODEL_REGISTRY.md`.

## Phase 4: Add API-level policy integration tests

The tests must use the committed policy/bundle, not only temporary custom policies.

For every supported horizon:

1. Load the default operational bundle.
2. Build a valid 24-step eight-feature observed sequence.
3. Call `predict_from_observed_sequence()`.
4. Verify the returned policy version and provenance.
5. Verify selected sources match the policy.
6. Verify the API uses persistence for variables marked `persistence_fallback`.
7. Verify the API uses model values for variables marked `learned_model`.
8. Independently recompute rain blending:

   ```text
   expected = model_weight * raw_model_probability
           + persistence_weight * persistence_probability
   ```

9. Compare the independent value with the API response.
10. Verify operational alert classification using the frozen threshold.
11. Verify unsupported horizons fail closed.
12. Verify missing bundle files fail closed.
13. Verify stale bundle hashes fail closed.
14. Verify weather uncertainty is explicitly reported unavailable.
15. Verify water-level output remains beta-only.

## Phase 5: Improve monitoring and evaluation

Add a monitoring/evaluation module that can run on new labeled telemetry without changing the frozen policy automatically.

Suggested path:

```text
prediction-model/src/monitoring.py
```

### Required monitoring dimensions

Report metrics by:

- horizon;
- station;
- target variable;
- rain versus dry regime;
- heavy-rain regime;
- calm versus windy regime;
- data-quality/quarantine status;
- time window.

### Required metrics

For continuous variables:

- sample count;
- MAE;
- RMSE;
- bias;
- persistence MAE;
- skill versus persistence;
- climatology MAE;
- missingness rate;
- quarantine rate.

For rain occurrence:

- Brier score;
- calibration error;
- F1;
- precision;
- recall/POD;
- false-alarm ratio;
- CSI;
- confusion matrix;
- reliability bins.

For precipitation amount:

- overall MAE/RMSE/bias;
- dry-hour MAE/RMSE/bias;
- rainy-hour MAE/RMSE/bias;
- heavy-rain precision, recall, and CSI at 2.5, 5.0, and 10.0 mm/h;
- persistence and climatology baselines.

### Monitoring safeguards

1. Do not automatically change the inference policy based on one evaluation run.
2. Require minimum sample counts.
3. Report confidence intervals or bootstrap uncertainty for monitoring summaries where practical.
4. Make policy updates explicit and versioned.
5. Add a drift report for:
   - feature distributions;
   - missingness;
   - quarantine rate;
   - station coverage;
   - rain prevalence.

## Phase 6: Improve precipitation modeling without breaking the current contract

Do not replace the existing model blindly. First add an experimental path behind an explicit feature/configuration flag.

### Recommended architecture

Evaluate a two-stage precipitation design:

1. Rain occurrence head:
   - binary probability;
   - BCE or focal loss if justified;
   - calibration evaluation.

2. Conditional rain-amount head:
   - trained on rainy samples;
   - log1p, Gamma, Tweedie, or quantile objective;
   - explicit zero-rain handling.

Compare it against the current amount head using untouched test data.

### Acceptance requirements

The new precipitation approach may replace the current one only if it improves or provides a justified trade-off on:

- overall MAE;
- rainy-hour MAE;
- heavy-rain recall;
- heavy-rain CSI;
- bias;
- calibration;
- no material degradation in dry-hour behavior.

If it does not win, retain it as an experimental model and document the result.

## Phase 7: Add weather uncertainty only when validated

Do not claim intervals before validation.

If implementing intervals:

1. Fit residual/conformal quantiles on calibration data only.
2. Evaluate on untouched test data.
3. Do this separately per target and horizon.
4. Report:
   - nominal level;
   - observed coverage;
   - coverage confidence interval;
   - full interval width;
   - sample count;
   - station breakdown.
5. Monitor coverage after deployment.
6. Fail closed or mark unavailable when sample counts are insufficient.

Until these gates pass, return and document:

```json
{
  "weather_uncertainty": {
    "status": "UNAVAILABLE"
  }
}
```

Do not describe unavailable intervals as confidence bands.

## Phase 8: CI and release automation

Add or update CI so the prediction-model release pipeline performs:

```text
1. Clean checkout
2. Dependency installation
3. Static checks
4. Canonical contract tests
5. Inference policy integration tests
6. Provenance tests
7. Smoke test
8. Monitoring/evaluation schema tests
9. Path hygiene check
10. Deleted-reference check
11. Working-tree cleanliness check
12. Exact/bundle provenance verification
13. Five-horizon artifact presence check
14. Optional real-data validation job
```

The release must fail if:

- tests fail;
- tracked files change during read-only tests;
- a bundle hash is wrong;
- a policy/checkpoint mismatch exists;
- machine-specific paths are committed;
- a protected raw telemetry file is missing;
- a supported horizon lacks artifacts;
- documentation claims unsupported uncertainty;
- active scripts reference deleted artifacts.

## Required command sequence

Run and record commands equivalent to:

```bash
git fetch origin main --prune
git status --short --branch
python -m pip install -r prediction-model/requirements.txt
python prediction-model/src/test_canonical_contract.py
python prediction-model/src/test_inference_contract.py
python prediction-model/src/test_provenance.py
python prediction-model/src/smoke_test.py
python prediction-model/src/verify_provenance.py
python prediction-model/src/validate.py --horizons 1 3 6 12 24
git diff --check
git status --short --branch
```

Add the new monitoring, bundle, and read-only-test commands to the project README.

## Artifact regeneration and commit protocol

1. Implement source changes.
2. Run unit and contract tests without modifying tracked artifacts.
3. Train or regenerate artifacts using explicit generation commands.
4. Review generated artifact diffs.
5. Commit source and intended artifacts.
6. Reconcile implementation/artifact/model-weight provenance according to the chosen policy.
7. Run the exact final verification.
8. Run tests again from a fresh clean worktree.
9. Confirm the fresh worktree remains clean after tests.
10. Push to `origin/main` only if all gates pass.

## Required final report

The final response must include:

- starting commit;
- final commit;
- exact push result or blocker;
- archive reference;
- files changed and deleted, grouped by reason;
- canonical files retained;
- chosen provenance model;
- implementation commit;
- artifact commit;
- model weights commit;
- bundle hashes;
- raw telemetry hashes;
- training seeds and configurations;
- all five-horizon metrics;
- precipitation baseline and heavy-rain metrics;
- per-station monitoring results if generated;
- uncertainty status and any valid coverage/width metrics;
- test commands and pass/fail results;
- confirmation that read-only tests leave the worktree clean;
- `git diff --check` result;
- operational GO/NO-GO recommendation;
- unresolved risks and follow-up work.

## Release recommendation gate

Recommend **GO** only if:

- the API cannot bypass the policy;
- model and policy bundles validate by hash;
- provenance is unambiguous;
- read-only tests do not modify tracked files;
- all tests pass;
- all five horizons are present and validated;
- no machine-specific paths remain;
- no active scripts reference deleted files;
- documentation matches actual behavior;
- uncertainty claims are honest;
- the final worktree is clean;
- the push succeeds.

Otherwise recommend **NO-GO**, list every failed gate, and do not push.

Do not stop at a plan. Execute the entire improvement cycle through implementation, testing, artifact regeneration, commit, provenance verification, and final release decision.
