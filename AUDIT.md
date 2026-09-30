# KloudTrack Prediction System — Audit & Remediation Report

**Scope:** `prediction-model/` (Python forecasting engine) and `src/` (Next.js dashboard)
**Branch:** `fix/prediction-model-audit-findings`
**Status of this document:** findings verified against source; remediation applied and re-verified
**Test baseline:** 281/281 Python tests pass; `tsc --noEmit` clean; `next build` succeeds

---

## 1. Executive summary

The repository contains **two prediction systems that were never reconciled**.

1. A Python forecasting engine (`prediction-model/`) with real data governance:
   chronological splitting, embargo, train-only normalisation, SHA-256 provenance
   manifests, frozen inference policy, and fail-closed bundle verification.
2. A TypeScript reimplementation inside the Next.js app
   (`src/services/prediction.service.ts`) that the **website actually served** — a
   different input dimensionality, a different hidden size, hardcoded normalisation,
   and weights matching no artifact in the repository. No audited scorecard applied to it.

The Python engine's own forecasts were being written to disk with explicit safety
metadata (`not_for_life_safety: true`, `model_status: RESEARCH_PROTOTYPE`). The
dashboard read that file for raw sensor values and **discarded the forecast block**,
then rendered its own numbers — including imperative flood directives such as
`DANGER: FLOODED / DO NOT PASS` — for a model that explicitly declares it must not
be used for life safety.

Separately, the report generator contained a function that produced evaluation
results from `np.random.normal(...)` and published them with a verdict reading
*"verified with paired bootstrap tests on identical evaluation rows"*.

**Direction taken:** make the validated Python engine the production model, make the
app fail closed when it is unavailable, and remove every fabricated value.

---

## 2. What the model actually does

### 2.1 Honest characterisation

`CfCCell` (`prediction-model/src/model.py:36-71`) implements:

```
decay     = exp(-dt * softplus(ff_time([x, h])))
gate      = sigmoid(ff_gate([x, h]))
candidate = tanh(ff_state([x, h]))
h_next    = decay * h + (1 - decay) * gate * candidate
```

A genuine Closed-form Continuous cell solves `h(t) = e^{-At}h(0) + A^{-1}(e^{-At} - I)f(θ·x)`.
There is **no `A` matrix and no `A^{-1}`**, and `dt` scales a *rate*, not a timescale.
The implemented cell is an exponentially-weighted moving average with a learned,
input-dependent rate. That is a reasonable architecture; describing it as a
Neural ODE / CfC in the README and working paper is not accurate.

| | |
|---|---|
| Architecture | `GarciaWeatherLNN` (MF-1), 8 features → 8 heads, hidden 32 |
| Parameters | 11,433 (production), 17,529 (candidate) |
| Input window | 24 hourly observations; horizons {1, 3, 6, 12, 24}h |
| Training data | ~7,522 station-hours, 16 stations, ~2 months |
| Heads evaluated | final hidden state only (`model.py:301-305`) |
| Output form | origin-anchored deltas (`model.py:308-317`) |

11k parameters on 7.5k samples is an overfitting regime; near-persistence output
structure and validation-based checkpoint selection mask it rather than remove it.

### 2.2 Does it beat persistence?

From `prediction-model/MODEL_REGISTRY.md:84-115` (untouched test split):

| Target | Horizons where the learned model beats persistence |
|---|---|
| Temperature | 1 of 5 (+12h only) |
| Humidity | 1 of 5 |
| Pressure | 3 of 5, but +0.73% / +0.89% at 1h/6h — noise |
| Wind speed | 1 of 5 |
| Heat index | 4 of 5, but +1.3%–1.9% at 1h–6h |
| **Rain probability** | **5 of 5, +19% to +36% Brier skill** |

**At +24h every target routes to `PERSISTENCE_FALLBACK`.** The operational policy is
doing the correct thing, but it means the "neural forecast" is for most
variable/horizon pairs the last observed value.

The one genuine, shippable contribution is **calibrated rain probability** — hybrid
blending that beats persistence on Brier score at all five horizons, with weights fit
on the validation split and frozen before test.

**Anomalous row group:** at +6h, +12h and +24h the "Calibrated Hybrid Blend" and
"Persistence" rows report byte-identical F1, accuracy, POD, precision and CSI
(64.78 / 73.05 / 65.86 / 63.75 / 47.91 at +6h) while Brier differs. Identical to four
decimals across six discrete metrics means the blended probability never crosses the
0.5 decision threshold differently from persistence — the "hybrid" is relabelled
persistence for any user who thresholds it. **Not investigated; flagged.**

### 2.3 Production bundle behaviour (measured)

Verified directly against `prediction-model/data/bundles/h1`:

| Input | `chance_of_rain_pct` | Alert |
|---|---|---|
| 24h dry diurnal window | 3.6 | false |
| 24h window, sustained heavy rain in last 6h | 92.4 | **true** |

All continuous variables routed to `persistence_fallback`, as the registry documents.
The production bundle is well calibrated.

### 2.4 Candidate model is not fit to promote (new finding)

`GarciaWeatherLNNFeatured` — the `candidate_artifacts/` model — returns:

| Input | `chance_of_rain_pct` |
|---|---|
| 24h diurnal window, **zero precipitation throughout** | **57.1** |
| 24h window, sustained heavy rain | 99.3 |

A 57% rain probability in completely dry conditions means the occurrence head is
saturated and barely discriminates "no rain" from "heavy rain". The candidate
remains `CANDIDATE_RESEARCH` and unpromoted, which is the correct status.
**It must not be promoted on the strength of its training-run artifacts.**

### 2.5 River stage is a beta capability

From `MODEL_REGISTRY.md:170-181`, N=1 gauge, N=221 samples:

- Model loses to persistence at +1h, +3h, +6h, +12h.
- At +1h: model MAE 0.0338 m vs persistence **0.0146 m** — 2.3× worse.
- Conformal intervals validated at +1h only; coverage 90.5% at 80% nominal, i.e.
  intervals ~4.5× the error width. Honest, but operationally useless.

---

## 3. Findings and remediation

### CRITICAL

#### C-1 — Fabricated evaluation results · **FIXED**
`train_predictive_quality.py:2212-2255` — `run_champion_challenger_evaluation` never
loaded data or a model. It drew error vectors from `np.random.default_rng(seed).normal(...)`,
fed them to a real `ChampionChallengerEvaluator`, and wrote
`champion_challenger_report.json` with the verdict *"verified with paired bootstrap
tests on identical evaluation rows."* Confidence intervals included.

**Fix:** rewritten to compute paired champion/challenger comparisons from the real
logged evaluation rows in `candidate_artifacts/candidate_h{h}h_predictions.csv`
(truth, candidate prediction, and persistence baseline per row). Targets with no
logged baseline — wind direction, rain occurrence — are reported as
`NOT_EVALUATED` with an explicit reason instead of being simulated. Report version
bumped to 2.0.0 with `synthetic_data_used: false`.

#### C-2 — Hardcoded "PASS" gates and decisions · **FIXED**
`train_predictive_quality.py:1666-1689` — promotion gates 5-10 were literal
`"status": "PASS"`. `all_passed` was computed and never used. Anomaly metrics
asserted `empirical_false_alarms_per_day: 0.0`, `within_false_alarm_budget: True`,
`heavy_rain_recall: 1.0` for every horizon. `operational_decision` was the constant
`"CONDITIONAL_GO"`. The `TelemetryAnomalyDetector` is imported and never called.

**Fix:**
- Gates 5, 6, 8, 9, 10 now verify observable state (horizon presence, station count,
  artifact existence on disk, blocked-target status in the calibration file).
- Gate 7 is `NOT_EVALUATED` with a reason — the detector is genuinely not run.
- `NOT_EVALUATED` is not a pass and blocks promotion.
- Anomaly metrics report `NOT_EVALUATED` with `None` values instead of 1.0 recall.
- `operational_target_status` is **derived** from measured per-horizon skill with a
  2% relative-improvement threshold across ≥3 horizons; the decision follows the gates.

#### C-3 — Candidate inference path crashed on every call · **FIXED**
`inference.py:512` used `origin_dt` 130 lines before it was assigned, and referenced
`timezone`, which was never imported. Both branches of the ternary raised, so
`GarciaWeatherLNNFeatured` could not be served at all. The candidate artifacts existed
and the model class auto-detected them, so this would have fired immediately on load.

**Fix:** origin parsing hoisted before its use point, `timezone` imported, unparseable
origins now fail closed rather than silently substituting `now`, and the no-origin
branch synthesises a contiguous hourly grid. **Verified end-to-end** — the candidate
now loads and forecasts.

#### C-4 — Shipped model ≠ validated model · **FIXED**
The website served a ~1,000-line unvalidated TypeScript forecaster.

| | TS (shipped) | Python (audited) |
|---|---|---|
| hidden_dim | 8 | 32 |
| input features | 4 | 8 |
| normalisation | hardcoded `[28.5, 33.0, 10.0, 1008.0]` | train-fitted |
| weights | `pinn_lnn_3h_online_weights.json` | `bundles/h{1,3,6,12,24}/checkpoint.pt` |

`NORM_MEANS`/`NORM_STDS` were hardcoded while the loaded weights file carried its own
`means`/`stds` that were then ignored. `DEFAULT_LNN_WEIGHTS` matched no artifact in
the repository (verified by value search).

**Fix:** new `src/services/forecast.service.ts` reads the real engine output from
`prediction-model/data/mqtt_live_predictions.json` with a staleness gate, and
`prediction.service.ts` sources its weather overview from that payload. When no
validated forecast exists, the response is observation-only and says so.

#### C-5 — Safety metadata discarded; imperative flood directives shipped · **FIXED**
`mqtt_live_predictions.json:32-33` carries `model_status: "RESEARCH_PROTOTYPE"` and
`not_for_life_safety: true`. `grep` across `src/` found **zero** references to either.
`PredictionPublicDTO` had no field to carry them. Meanwhile `en.ts:191` rendered
`"DANGER: FLOODED / DO NOT PASS"` and `en.ts:207` `"FLASH FLOOD SURGE FROM MOUNTAINS!"`.

**Fix:**
- `ModelGovernance`, `ModelInputQuality`, `ModelSourceSelection`,
  `ModelForecastProvenance` added to the DTO and populated from the engine response.
- New `model-governance-banner.tsx` renders the notice **above** the forecast hero,
  including model status, horizon, freshness, checkpoint hash, per-variable source
  policy, uncertainty-unavailable notice, and blocked-target reasons.
- Safety copy rewritten in `en.ts` and `fil.ts` from directives to reported
  observations, with explicit redirection to PAGASA / local government.
- `hasError` — previously set, passed, and never rendered — now surfaces an alert.

#### C-6 — Silent imputation of missing sensor data · **FIXED**
`dataset.py:292-298` used `float(row.get(x) or <default>)`. Two defects:
a station reporting empty/NaN for a month produced 720 fabricated `28.5 °C`
observations that **passed every physical-bounds check**; and `or` conflates a
legitimate zero with a missing value, so a calm `wind_speed = 0.0` was silently
rewritten to `10.0` km/h.

**Fix:** strict per-field parsing. Empty, `"NaN"`, non-numeric, and infinite values
are quarantined and counted per field. Zeros are preserved. No imputation.

#### C-7 — Fabricated telemetry shipped in exports · **FIXED**
`export.service.ts:243-247` emitted `dopplerRadarDBZ: 35.0`,
`microburstProbabilityPct`, `convectiveBuoyancyJkg: 1200.0` (constant),
`inferenceLatencyUs: 52.4` (constant) into downloadable CSV/XLSX. The system has no
radar; `radar_qpe_v1` and `rainviewer_v1` are `UNKNOWN_BLOCKED` in the licence
registry.

**Fix:** fields removed from the schema. Replaced with governance columns
(`modelStatus`, `notForLifeSafety`, `rainProbabilitySource`, `inputQualityFlag`,
`learnedOutputTrusted`) so a reader can tell what produced the row.

### HIGH

#### H-1 — Water level presented as a flood warning system · **ADDRESSED**
Loses to persistence at 4 of 5 horizons. **Fix:** stage-derived `riskLevel` is
labelled a research estimate in the UI; `waterBeta` notice rendered from
`modelProvenance`; copy rewritten from "SAFE TO PASS" to "No elevated river stage
indicated".

#### H-2 — Two monitoring triggers could never fire · **FIXED**
- `monitoring.py:844` read `drift_report["status"]`, but `evaluate_drift` never emitted
  that key — drift status was permanently `NORMAL`.
- `monitoring.py:862` read `calibration_error_ece`; the emitter produces
  `expected_calibration_error` — the recalibration branch was unreachable.

**Fix:** `evaluate_drift` now compares live feature means against the
`train_fitted_normalization` baseline it already loaded but never used, emitting a real
status (`NORMAL` / `WARNING` / `DRIFT_DETECTED` / `NO_BASELINE`) with per-feature
z-scores. ECE key corrected.

#### H-3 — Divergent bounds and unit mismatch · **FIXED**
`dataset.py` bounded precipitation at 50 mm **per 1-minute record**;
`anomaly_detector.py` at 150 mm treating the column as **per hour** — but
`audit_sequence` is fed the hourly-summed series. Heat index 70 vs 75; water level
absent from the detector.

**Fix:** detector bounds renamed `HOURLY_PHYSICAL_BOUNDS` with the semantic documented
and water level added; dataset per-minute bounds left as-is and commented.

#### H-4 — Anomaly detection was advisory, not a gate · **FIXED**
`inference.py` ran the detector on every request and reported the result, but a
`QUARANTINED` sequence still produced a forecast.

**Fix:** `input_quality_gate` in the response. Quarantined input suppresses all
learned outputs and falls back to the last valid observation for every target. Also
fixed the severity formula, which mapped a 60 °C reading (physically impossible,
limit 50 °C) to severity 0.75 → non-blocking `WARNING`; bound violations and stuck
sensors now floor at the quarantine threshold. **Verified:** 60 °C sequence now
returns `QUARANTINED` with `persistence_fallback_input_quarantined`.

#### H-5 — Manifest recorded config that did not match the code · **PARTIALLY FIXED**
`training_config` claimed `early_stopping_patience: 3` (no patience counter exists)
and `batch_size: 32` (loader uses 64). `baseline_manifest` hardcoded Aug-1/Aug-21/Aug-26
boundaries and `embargo_duration_hours: 24`, contradicting the actual 48 h embargo.
`get_git_commit()` falls back to a hardcoded SHA, so a wrong commit can be stamped
into provenance silently.

**Fixed:** the fabricated gates/decisions that consumed these. **Not fixed:** the
`training_config` literals and the commit fallback — see §5.

#### H-6 — Unauthenticated compute-amplification endpoint · **FIXED**
`/api/benchmark/export` had no session check, ran 7-horizon inference across up to 23
stations with `maxDuration = 60`, and was reachable from a floating button in the root
layout.

**Fix:** requires a valid portal session; returns 503 when the portal is unconfigured
rather than exposing the route. The export modal handles 401/503 with a clear message.

#### H-7 — Portal credentials committed to the repository · **FIXED**
`portal-auth.ts` had `PORTAL_ADMIN_PASSWORD || "Kloudtrack2026!"` and
`PORTAL_SECRET_SALT || "kloudtrack_portal_salt_2026"`. The session "signature" was
`base64url(payload + ":" + salt)` — an encoding, not a MAC, forgeable by anyone who
could read the source. `checkCredentials` also accepted the literal username `"admin"`
regardless of the configured user.

**Fix:** real HMAC-SHA256 signature, `timingSafeEqual` comparison, no committed
defaults (fails closed), exact-username match. `createSessionToken` returns `null` when
unconfigured and login reports 503.

> **Action required:** set `PORTAL_ADMIN_USER`, `PORTAL_ADMIN_PASSWORD`,
> `PORTAL_SECRET_SALT` in the deployment environment. The portal is unreachable until
> they are set. Generate with
> `node -e "console.log(require('crypto').randomBytes(32).toString('base64url'))"`.

#### H-8 — Flood risk structurally pinned to "normal" (unit bug) · **FIXED**
`stations.json` `referenceThreshold` is in **centimetres** (780/500/450);
`prediction.service.ts:876` converts levels to **metres**; `:896` then computed
`780 × 0.7 = 546` and compared it against a value of ~3.5. `riskLevel` could never
exceed `normal`.

**Fix:** thresholds normalised to metres at a single documented point.

### MEDIUM

| # | Finding | Status |
|---|---|---|
| M-1 | Machine-specific absolute path written into the committed live artifact (13 occurrences) — caught by the repo's own path-hygiene gate | **FIXED**: `inference.py` writes repo-relative paths; artifact cleaned; ingestor restarted to pick up the fix |
| M-2 | `nwp.py:191` `atan2(wind_u, wind_v)` — u/v swapped, transposing every wind direction | **FIXED** |
| M-3 | 75-feature normalisation fit on ≥3-hour windows while training requires 24 | **FIXED**: complete windows only; cache key now schema-aware |
| M-4 | `ResidualWeatherModel` quantiles fit in-sample but labelled `"conformal_residual_quantiles_p10_p50_p90"` | **FIXED**: relabelled `in_sample_fitted_residual_quantiles_p10_p50_p90`, `is_split_conformal: false`, verdict corrected |
| M-5 | `confidenceScore: 0.94` hardcoded, never computed | **FIXED**: derived from freshness; labelled "data freshness only, not forecast accuracy" |
| M-6 | `confidencePct: 98.6` on kriged/spatial values | **FIXED**: derived from contributor count and distance, ceiling 85% |
| M-7 | `isSpatialEstimate` / `estimateSource` threaded to the UI but never rendered | **FIXED**: "Estimated" badge shown at the point of display |
| M-8 | Station identity bridge: 0 of 23 stations matched the MQTT cache; ~2 matched by luck | **FIXED**: new `mqtt-station-map.ts`; **23/23 resolve, 15/23 to currently-live devices** |
| M-9 | `.env.example` documented `KLOUDTRACK_API_KEY`; code reads `KLOUDTRACK_API_TOKEN` | **FIXED**: `.env.example` rewritten, all used variables documented |
| M-10 | AWS IoT private key present in `mqtt/` (gitignored) | **Verify never committed; rotate if so** |

---

## 4. What remains

| Item | Why not done |
|---|---|
| Delete the ~1,000-line TS forecaster (`computeLnnMultiHorizonForecast`) | Still used by `benchmark-export.service.ts` and the portal export for **historical backfill**, where no live model output exists. Removing it requires deciding what those exports should emit instead. |
| `training_config` literals (patience, batch size) and `get_git_commit()` fallback | Requires a full training re-run to regenerate honest artifacts. The artefacts are committed and provenance-gated; regenerating them invalidates bundle hashes. |
| Investigate identical discrete rain metrics at +6h/+12h/+24h | Needs a fresh `validate.py` run; noted as an open question. |
| 30 root-level planning docs, several byte-identical duplicates | Documentation cleanup, no code impact. `MODEL_REGISTRY.md` and the three audit reports are the sources of truth. |
| Promote the candidate model | **Should not happen.** Its rain head returns 57% in dry conditions (§2.4). |
| Radar / NWP / satellite features | All sources are `UNKNOWN_BLOCKED` in the licence registry. Correctly returning empty. Do not enable without a rights review. |

---

## 5. Verification

```
281 passed                          # pytest prediction-model/src
tsc --noEmit                        # clean
next lint                           # 1 pre-existing warning, 0 errors
next build                          # succeeds, 14 routes
```

Directly measured behaviour after remediation:

```
candidate bundle load   : OK (previously raised on every call)
production bundle, dry  : chance_of_rain_pct = 3.6,  alert = false
production bundle, heavy: chance_of_rain_pct = 92.4, alert = true
impossible input (60 °C): sensor_quality_flag = QUARANTINED, learned_output_trusted = false
                          source = persistence_fallback_input_quarantined
station identity bridge : 23/23 resolve; 15/23 to currently-live devices
```

### Test fixture change worth reviewing

`test_inference_contract.py` used `[row] * 24` fixtures — 24 identical hourly readings.
That is a stuck-sensor signature, so the anomaly gate correctly quarantines it and the
tests began asserting quarantine behaviour instead of the policy they were written
for. A `_realistic_seq()` helper now builds a diurnal walk with the **last row
preserved**, so each test's intent (control the origin observation) is unchanged.
Worth a second opinion: this changed test fixtures, not just production code.

---

## 6. Recommendation

Ship **calibrated rain probability plus persistence guidance**, honestly labelled.

The engineering in `prediction-model/` is genuinely good — the data governance is
professional and the failure modes are mostly fail-closed. The problem was never the
pipeline; it was that a validated engine was disconnected from the product, and a
report generator that would have caught the disconnection was itself generating
synthetic numbers.

Priorities:

1. **Set the portal secrets** in the deployment environment (H-7).
2. **Re-run training** to regenerate artifacts with honest `training_config`, then
   re-run `verify_provenance.py` (H-5).
3. **Decide the backfill story** for exports, then delete the TS forecaster.
4. **Retire the planning-doc pile**, keeping `MODEL_REGISTRY.md` and the audits.

Do not promote the candidate model, and do not enable external sources before a
licence review.
