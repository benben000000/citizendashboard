# Model Architecture

**Component:** KloudTrack prediction engine — Continuous-Time CfC/LNN surface meteorology forecaster
**Canonical code:** `prediction-model/src/`
**Code commit of record:** `cf0a37e239fd6cc5a3a43affb6fe69148ebba7bf`
**Policy version of record:** `2.0.0` (`prediction-model/data/inference_policy.json`)
**Status:** Research prototype. Not approved for autonomous flood warnings, emergency operations, or life-safety triggers.

This document describes how the system is built. It does **not** claim it is competitive
with operational numerical weather prediction. See
[known-limitations.md](known-limitations.md) before quoting any accuracy number from this
document, and [operations-runbook.md](operations-runbook.md) before changing anything.

---

## 0. Read this first: the system is a policy, not a network

The single most important structural fact about this system is that **the neural network
is not what most users receive.** A per-variable, per-horizon *source-selection policy*
sits between the network and the API. For each of the 20 benchmarked
(horizon × variable) cells, the policy independently chooses either the learned model or
the last observation ("persistence").

Measured directly from `inference_policy.json` at policy version 2.0.0:

| Horizon | temperature | humidity | pressure | wind_speed |
|---|---|---|---|---|
| +1h | persistence | persistence | persistence | persistence |
| +3h | persistence | persistence | persistence | persistence |
| +6h | **learned_model** | persistence | persistence | **learned_model** |
| +12h | **learned_model** | persistence | persistence | **learned_model** |
| +24h | persistence | persistence | persistence | **learned_model** |

- **15 of the 20** benchmarked cells are served by `persistence_fallback`.
- **5 of the 20** use `learned_model`.
- Including `wind_direction` (a fifth continuous variable that is not MAE-benchmarked
  against NWP), it is **20 of 25** cells on persistence.

> **Note on a commonly-quoted figure.** The number "13 of 20" has circulated for this
> policy. It does not match the file. Read straight from `inference_policy.json`
> v2.0.0 the count is **15 of 20** (or 20 of 25 including wind direction). Any downstream
> summary that says 13 is quoting a stale policy revision.

**Consequence for anyone reading accuracy numbers.** When a report says "the model
scores 1.677 %RH MAE on humidity," the truthful statement is: *the served forecast was
the last observed humidity reading, and persistence happens to score 1.677 %RH on this
test split.* It is not evidence that a network learned humidity structure. Humidity is
served by persistence at **every** horizon. The same applies to pressure at every horizon,
to temperature at +1h/+3h/+24h, and to wind direction everywhere.

The network is still evaluated and still produces values in those cells — they are
simply discarded by the policy before reaching the API. The measured size of what is
discarded is in [known-limitations.md §2](known-limitations.md#2-discarded-headroom).

Rain occurrence is the one channel where the learned model participates at **every**
horizon, via a hybrid blend rather than a hard switch (§5).

---

## 1. System data flow

```
 AWS stations (16)
      │
      │  1-minute MQTT / REST telemetry
      ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │ LAYER 1  INGEST + AUDIT            mqtt_live_ingestor.py         │
 │   • subscribe kloudtrack/+/data                                │
 │   • parse, timestamp, atomic cache update                      │
 │   • append to prediction_audit.jsonl   (what we said)          │
 │   • append to observation_audit.jsonl  (what actually happened)│
 └───────────────────────────┬─────────────────────────────────────┘
                             │  hourly-grid observations
                             ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │ LAYER 2  DATASET + WINDOWS        dataset.py                   │
 │   • outlier quarantine (physical bounds, year range)           │
 │   • resample 1-min → hourly bins (last-valid / circular / sum) │
 │   • 24-step input window [t-23 … t0]                            │
 │   • target = observation at t0 + h,  h ∈ {1,3,6,12,24}         │
 │   • chronological 60/20/20 split + 48 h embargo                 │
 │   • normalisation fitted on TRAIN only                         │
 └───────────────────────────┬─────────────────────────────────────┘
                             │  normalised tensors + metadata
                             ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │ LAYER 3  MODEL                    model.py                      │
 │   GarciaWeatherLNNFeatured (CfC/LNN + 75 context features)     │
 │   → per-variable residual heads, zero-initialised              │
 └───────────────────────────┬─────────────────────────────────────┘
                             │  residual deltas + rain logit
                             ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │ LAYER 4  SOURCE SELECTION         inference_policy.json         │
 │   • per-variable: learned_model | persistence_fallback          │
 │   • per-variable: affine calibration (a, b) + shrinkage (λ)    │
 │   • rain: hybrid blend, model weight w / persistence 1-w       │
 │   FAIL-CLOSED: unknown horizon or commit mismatch → raise      │
 └───────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │ LAYER 5  API                     src/app/api/prediction/...    │
 │   GET /api/prediction/station/[stationId]                      │
 │   GET /api/prediction/mqtt/live                                 │
 │   every response stamped with policy_version + policy_commit   │
 └─────────────────────────────────────────────────────────────────┘
```

---

## 2. Layer 1 — Ingest and the audit pair

`src/mqtt_live_ingestor.py` subscribes to `kloudtrack/+/data` (override with `MQTT_TOPIC`),
maintains an atomic on-disk cache, and writes two append-only trails:

| Trail | File | Contains |
|---|---|---|
| Prediction audit | `data/prediction_audit.jsonl` | one record per issued forecast: origin, target, per-variable `producer`, policy version/commit, NWP status, sensor health |
| Observation audit | `data/observation_audit.jsonl` | one record per received observation |

The observation trail is not optional bookkeeping. **A prediction cannot be scored without
it** — the score is `prediction_forecast − observation_actual`, and only the second half is
written by the ingestor. A live record looks like:

```json
{
  "schema_version": 1,
  "recorded_at_utc": "2026-09-30T01:53:55.654288Z",
  "origin_timestamp_utc": "2026-09-30T01:53:55.432805Z",
  "station_id": "KT-6CBD47DC5194",
  "horizon_hours": 1.0,
  "provenance": {
    "model":  { "bundle": "h1", "seed": null },
    "policy": { "policy_version": "2.0.0",
                "policy_code_commit": "cf0a37e2…",
                "policy_regenerated_by": "prediction-model/src/refit_policy.py" },
    "nwp":    { "applied": false, "reason": "router not wired into the live ingestor yet" }
  },
  "variables": { "temperature": { "value": 34.08, "producer": "lln", "nwp_raw": null } }
}
```

Two things to note when reading these records:

1. **`producer` is not `policy_source`.** The field records which subsystem emitted the
   value, and the label `lln` appears even for variables the policy is serving from
   persistence, because the network ran and its output was then gated. To determine what
   was actually served, read `provenance.policy.policy_version` and cross-reference
   `inference_policy.json` for that horizon. Do not infer the source from `producer`.
2. **`nwp.applied: false` is the normal live state.** The NWP correction router is not
   wired into the live ingestor; NWP enters through the offline benchmark and the offline
   correction path only.

Audit writes never raise into the forecast path. A failed audit append is logged and
swallowed — losing an audit line may cost you an auditable forecast, but it must never
cost a live user their forecast.

---

## 3. Layer 2 — Dataset, windows, and splits

Implemented in `src/dataset.py`.

### 3.1 Quarantine

Raw 1-minute telemetry is filtered before binning. Counts land in
`data/data_quality_report.json`. From the run of record:

| Reason | Rows quarantined |
|---|---|
| weather_bounds_temperature | 6,634 |
| weather_bounds_pressure | 16,133 |
| weather_bounds_wind_speed | 183 |
| weather_bounds_humidity | 1 |
| weather_bounds_precipitation | 2 |
| weather_year_out_of_bounds | 2 |
| water_physical_bounds | 12 |

22,955 weather rows and 12 water rows quarantined out of 756,156 / 43,883. The year-range
rule is what removes the 2069 timestamp anomaly.

### 3.2 Hourly resampling

| Variable | Rule |
|---|---|
| temperature, humidity, pressure, wind speed | last valid observation in the hour |
| wind direction | circular vector components `(u = cosθ, v = sinθ)`; speeds below 1.0 km/h masked as calm |
| precipitation | sum of valid tipping-bucket minute increments, mm |
| heat index | derived, NOAA Rothfusz regression |
| river stage | last valid gauge observation, collocated Calumpit WLMS `O3z0j5bG` |

### 3.3 The canonical forecast sample

- **Input:** 24 hourly observations ending at origin `t0`, i.e. `[t-23 … t0]`.
- **Target:** the observation at `t0 + h`, for `h ∈ {1, 3, 6, 12, 24}`.
- **Lead verification:** `|actual_lead − h| ≤ 0.25 h`, enforced per window.
- **Windows never span stations** and **never span a split boundary**.
- A window is rejected (and counted) if the target hour is missing, the target precedes
  the origin, the 24-step history is incomplete, or the lead is out of tolerance.

Every rejection reason is counted in a `rejections` dict that travels with the metadata, so
a shrinking window count is always attributable rather than mysterious.

### 3.4 Splits

Strategy `chronological_60_20_20_with_48h_embargo`, from `data/data_quality_report.json`:

| Split | Range (UTC) |
|---|---|
| Train | 2026-06-20T10:00 → 2026-08-01T18:00 |
| Validation | 2026-08-03T18:00 → 2026-08-13T22:00 |
| Test | 2026-08-15T22:00 → 2026-08-26T02:00 |

The embargo is `seq_len + max_horizon` = 24 + 24 = 48 hours, so no window's input history
or target can reach across a split boundary. Normalisation means and standard deviations
are fitted on TRAIN only and frozen into `cleaned_data_manifest.json`; validation chooses
policy, test is read exactly once for reporting.

Test window counts per horizon: 2,273 (+1h), 2,243 (+3h), 2,199 (+6h), 2,120 (+12h),
1,984 (+24h). The decline is the 24-step history requirement plus the end of the record —
not selective dropping.

---

## 4. Layer 3 — The CfC/LNN model

Implemented in `src/model.py`. The production class is `GarciaWeatherLNNFeatured`.

### 4.1 The CfC cell

`CfCCell` is a closed-form continuous-time recurrent cell — no numerical ODE solver:

```
w     = softplus(W_time · [x ; h])
decay = exp(−dt · w)                      # dt = elapsed hours between observations
gate  = σ(W_gate · [x ; h])
cand  = tanh(W_state · [x ; h])
h'    = decay · h + (1 − decay) · gate · cand
```

The `dt` term is what makes the model continuous-time: a gap of 1 minute and a gap of
3 hours produce different decays, so the hidden state carries real elapsed-time
semantics rather than step-count semantics. This matters because the ingest is irregular
1-minute data binned to hours with holes.

### 4.2 Feature path

```
telemetry_seq [B, 24, 8]  ──► seq_encoder (Linear → SiLU → Linear) ──► CfC cell × 24 ──► h
context_feats [B, 75]     ──► context_encoder (Linear → SiLU → Linear) ──► c
                                                     fusion(cat[h, c]) ──► fused
```

The 8 canonical sequence features are `temperature, heat_index, humidity, pressure,
wind_speed, wind_sin, wind_cos, precipitation` — normalised with train-fitted statistics.
The 75 context features are engineered, zero-leakage descriptors computed **at origin
`t0` only**. Nothing in the context vector is derived from the target hour.

### 4.3 Zero-initialised residual heads — the persistence prior

All five continuous-variable heads are **residual** heads whose final layer is
zero-initialised:

```python
nn.init.zeros_(self.temp_head[-1].weight);    nn.init.zeros_(self.temp_head[-1].bias)
nn.init.zeros_(self.rh_head[-1].weight);      nn.init.zeros_(self.rh_head[-1].bias)
nn.init.zeros_(self.pressure_head[-1].weight); nn.init.zeros_(self.pressure_head[-1].bias)
nn.init.zeros_(self.ws_head[-1].weight);      nn.init.zeros_(self.ws_head[-1].bias)
nn.init.zeros_(self.wdir_head[-1].weight);    nn.init.zeros_(self.wdir_head[-1].bias)
nn.init.constant_(self.rain_occurrence_head[-1].bias, -2.2)   # ≈ 10% empirical base rate
```

Each head emits a **delta**, which is added back to the origin observation:

```
pred_temp = clamp(t₀ + Δtemp,  −10, 60)
pred_rh   = clamp(rh₀ + Δrh,     0, 100)
pred_p    = clamp(p₀  + Δp,    850, 1090)
pred_ws   = _nonnegative(ws₀ + Δws, max_value=250.0)
```

At initialisation the network is **exactly persistence** — every prediction equals the
last observation. Training departs from persistence only insofar as it reduces loss.

This is a deliberate, load-bearing choice. Weather at 1–24 h is dominated by persistence;
starting from a zero residual means the optimiser has to *earn* every departure, and a run
that fails to learn anything degrades to persistence rather than to noise. It also means
the h1 cells that serve persistence are not hiding a network that scored badly — they are
cells where the network's earned residual was not good enough to justify itself, and the
policy correctly declined to use it.

The rain head is the exception: it is initialised to the empirical base-rate log-odds
(−2.2 ⇒ σ(−2.2) ≈ 0.10), because a zero logit would mean 50% rain, which is wrong by
construction.

### 4.4 The wind floor — a leaky floor, not a ReLU

Wind speed is physically non-negative, so the output needs a floor. It is deliberately
**not** `F.relu`, and the reason is a measured failure documented in the source:

> `F.relu` has exactly zero gradient on its negative side, which turns the floor into a
> one-way trap. The wind head starts zero-initialised (a persistence prior) and learns a
> negative residual because wind decays; relu then clamps those windows to a hard 0.0 and
> `smooth_l1` contributes nothing for them. They can never be pushed back up. Measured on
> the +6h candidate, `d_ws` averaged **−5.75 m/s** and **70% of test windows** sat on the
> dead side of the relu.

The replacement:

```python
_WIND_FLOOR_SLOPE = 0.01

def _nonnegative(x, max_value=None):
    x = F.leaky_relu(x, negative_slope=_WIND_FLOOR_SLOPE)   # leak keeps gradient alive
    if max_value is not None:
        x = torch.clamp(x, max=max_value)                   # physical ceiling only
    return x
```

The upper bound does use `clamp` because 250 km/h is a physical ceiling that real readings
never approach, so saturating there costs nothing. The lower bound is a leak so a window
that overshoots negative can recover. `softplus` would also preserve gradient but cannot
represent an exact 0, which would silently shift the persistence prior off the calm-wind
case. Note also that the floor lives *inside* the residual addition — wrapping
`_nonnegative(...)` in an outer `clamp(min=0)` would reintroduce the dead gradient the fix
exists to remove.

Regression test: `src/test_wind_head_gradient.py`.

### 4.5 Rain: two-stage occurrence × amount

```
rain_logit   = rain_occurrence_head(fused)          → rain_prob = σ(rain_logit)
cond_amount  = softplus(conditional_rain_head(fused))
precip_mm    = rain_prob × cond_amount
```

Occurrence and magnitude are separated so that the ~57% of hours with zero precipitation
do not wash out the heavy-rain response. The occurrence head is the component that carries
the system's only 1st-of-10 skill result — a result whose margin decays sharply with lead
time and does not survive +24h (§7.5, and
[known-limitations.md §7](known-limitations.md#7-rain-occurrence-the-strongest-result-and-its-decay)).

---

## 5. Layer 4 — The source-selection policy layer

`data/inference_policy.json`, version 2.0.0. Structure:

```jsonc
{
  "policy_version": "2.0.0",
  "policy_code_commit": "cf0a37e2…",       // must match the bundle's model commit
  "dataset_hashes": { "weather_telemetry_sha256": "86ce9064…", … },
  "horizons": {
    "6": {
      "selected_sources": { "temperature": "learned_model", "humidity": "persistence_fallback", … },
      "rain_model_weight": 0.45,
      "rain_persistence_weight": 0.55,
      "operational_rain_threshold": 0.10,
      "calibration_code_commit": "cf0a37e2…",
      "calibration": { "temperature": { "a": 1.0, "b": -0.0, "lambda": 0.0 }, … }
    }, …
  }
}
```

### 5.1 Selection rules

`selected_sources` names one of:

| Value | Meaning |
|---|---|
| `learned_model` | serve the network output for this variable at this horizon |
| `persistence_fallback` | serve the origin observation; the network output is computed and discarded |
| `derived_from_selected_temp_and_humidity` | recompute the NOAA Rothfusz heat index from whichever temp and humidity were actually selected |

Selection is **measured, not asserted**. `src/refit_policy.py` regenerates the file:
calibration coefficients are fitted on TRAIN, and source selection plus shrinkage strength
are chosen on VALIDATION. The regeneration note in the file records that the TEST split
was not read during refitting.

### 5.2 Affine calibration and shrinkage

Where a variable is served by the learned model, the residual is calibrated:

```
calibrated = persistence + a · (learned − persistence)          # affine on the residual
final      = (1 − λ) · calibrated + λ · persistence             # shrinkage toward persistence
```

`λ = 0` keeps the full calibrated residual; `λ = 1` is pure persistence. Coefficients of
record:

| Horizon | Variable | a | b | λ |
|---|---|---|---|---|
| +6h | temperature | 1.0 | −0.0 | 0.0 |
| +6h | wind_speed | 0.686043 | 0.553712 | 0.75 |
| +12h | temperature | 0.933945 | 1.642678 | 1.0 |
| +12h | wind_speed | 0.761377 | 0.400885 | 0.5 |
| +24h | wind_speed | 0.674093 | 0.498611 | 0.75 |

`λ = 1.0` at +12h temperature is the shrinkage optimiser selecting *pure persistence* for
that cell in its calibrated form. See
[known-limitations.md §3](known-limitations.md#3-the-24h-temperature-anomaly) for why that
particular cell is the subject of an open investigation.

### 5.3 Rain blend

Rain is never a hard switch. Every horizon blends the network occurrence probability with
the persistence rain flag:

| Horizon | model weight | persistence weight | operational threshold |
|---|---|---|---|
| +1h | 0.80 | 0.20 | 0.25 |
| +3h | 0.60 | 0.40 | 0.10 |
| +6h | 0.45 | 0.55 | 0.10 |
| +12h | 0.45 | 0.55 | 0.10 |
| +24h | 0.40 | 0.60 | 0.10 |

The trend is monotone and deliberate: the learned occurrence signal is worth most at short
lead and degrades toward climatology, so its weight falls as the horizon grows and the
persistence signal takes over.

### 5.4 Fail-closed behaviour

Inference refuses to run rather than guess:

- No `inference_policy.json` → raise. ("Inference fails closed without a valid
  operational policy.")
- Requested horizon not present in the `horizons` table → raise. It will not silently
  substitute the 1h policy or invent defaults.
- `policy_code_commit` ≠ bundle model commit → raise.
- Bundle `policy_sha256` ≠ policy file hash → raise (tamper/corruption detection).
- Any of `bundle_manifest.json`, `checkpoint.pt`, `inference_policy.json` missing → raise.

---

## 6. Layer 5 — Bundles and the API

### 6.1 Bundle layout

`src/generate_bundles.py` packages one self-contained, hash-verified bundle per horizon
under `data/bundles/h{1,3,6,12,24}/`:

```
checkpoint.pt            # lnn_weather_water_h<h>.pt
inference_policy.json    # horizon-scoped active policy with full provenance
bundle_manifest.json     # hashes + commit triple + feature schema
README.md
```

`inference.py` verifies `policy_sha256` against the manifest before loading, so a
tampered or truncated policy file is a hard failure, not a silent fallback. Verify an
existing bundle set without regenerating with `--check-only`.

### 6.2 The two-commit provenance contract

Each bundle carries three commits:

| Field | Meaning |
|---|---|
| `implementation_commit` | the code that produced the bundle |
| `artifact_commit` | the commit the artifact was recorded against |
| `model_weights_commit` | the code commit the weights were trained under |

`src/verify_provenance.py` checks these and the dataset hashes. An explicit historical
commit can be acknowledged with `--allow-commit <sha>` rather than by silently editing
artifacts.

### 6.3 API surface

| Route | Handler | Notes |
|---|---|---|
| `GET /api/prediction/station/[stationId]` | `src/app/api/prediction/station/[stationId]/route.ts` | per-station forecast, horizon parameter |
| `GET /api/prediction/mqtt/live` | `src/app/api/prediction/mqtt/live/route.ts` | latest live forecast snapshot |

Every forecast response carries `policy_version`, `policy_code_commit`,
`rain_probability_source`, `rain_model_weight`, `rain_persistence_weight`,
`rain_operational_threshold`, `model_status: RESEARCH_PROTOTYPE`, and
`not_for_life_safety: true`. A consumer that ignores these fields is consuming a number
whose provenance it cannot reconstruct — see §0.

---

## 7. Measured accuracy on the test split

Source of record: `data/nwp_benchmark_production.json` (production bundles, committed
corpus, test split 2026-08-15T22:00 → 2026-08-26T02:00). Seven NWP models: ECMWF IFS,
NOAA GFS, DWD ICON, Meteo-France, JMA, UK Met Office, ECCC GEM. All wind figures in m/s.

**These are the numbers to quote. Other scorecards in the repository are stale — see
[known-limitations.md §8](known-limitations.md#8-stale-artifacts-that-will-mislead-you).**

### 7.1 Temperature (MAE, °C)

| | +1h | +3h | +6h | +12h | +24h |
|---|---|---|---|---|---|
| **LNN production** | 0.5937 | 1.1368 | **1.4772** | **1.6474** | 1.4059 |
| persistence | 0.5936 | 1.1367 | 1.6987 | 2.1619 | 1.4059 |
| ECMWF IFS | **1.2624** | **1.2668** | 1.2740 | 1.2880 | 1.3327 |
| NOAA GFS | 1.6749 | 1.6827 | 1.6961 | 1.7214 | 1.7802 |
| DWD ICON | 1.3181 | 1.3229 | 1.3280 | 1.3299 | 1.3584 |
| Meteo-France | 1.3489 | 1.3532 | 1.3541 | 1.3650 | 1.4099 |
| JMA | 1.3006 | 1.3003 | 1.3063 | 1.3076 | 1.3546 |
| UK Met Office | 1.2275 | 1.2304 | 1.2357 | 1.2282 | 1.2469 |
| ECCC GEM | 1.6243 | 1.6292 | 1.6408 | 1.6448 | 1.7140 |

At +6h the served forecast beats persistence by **13.0%**; at +12h by **23.8%**. It
**loses to ECMWF at both** (+6h: 1.477 vs 1.274; +12h: 1.647 vs 1.288). ECMWF also has the
lowest MAE of any method at +1h and +3h, where the policy serves persistence. Do not
describe this system as beating ECMWF on temperature.

### 7.2 Wind speed (MAE, m/s)

| | +1h | +3h | +6h | +12h | +24h |
|---|---|---|---|---|---|
| **LNN production** | 0.8800 | 1.1013 | **1.2168** | **1.2428** | 1.1800 |
| persistence | 0.8798 | 1.1012 | 1.3192 | 1.5114 | 1.1784 |
| ECMWF IFS | 1.9224 | 1.9275 | 1.9340 | 1.9186 | 1.9233 |
| NOAA GFS | 2.3596 | 2.3717 | 2.3933 | 2.4092 | 2.4585 |
| DWD ICON | 2.1183 | 2.1239 | 2.1371 | 2.1362 | 2.1608 |
| Meteo-France | 1.9952 | 2.0039 | 2.0146 | 2.0069 | 2.0108 |
| JMA | 2.0747 | 2.0814 | 2.0957 | 2.1094 | 2.1681 |
| UK Met Office | 2.4196 | 2.4333 | 2.4530 | 2.4673 | 2.4759 |
| ECCC GEM | 2.1045 | 2.1068 | 2.1076 | 2.0834 | 2.0903 |

This is the clearest win. At +6h the served forecast is **7.8%** better than persistence;
at +12h it is **17.8%** better. It beats **all seven NWP models at both horizons**, by a
wide margin: at +12h it is 35% better than ECMWF and 48% better than GFS.

Two caveats that must travel with this table. First, the margins are only this large
because an earlier units bug inflated every NWP wind figure by 3.6× — the honest framing
is "clearly better than every NWP model here," not "an order of magnitude better." Second,
NWP models were evaluated against the same station anemometers that contribute three
dead-sensor stations to the corpus; see
[known-limitations.md §4](known-limitations.md#4-instrument-defects-in-the-ground-truth).

### 7.3 Pressure (MAE, hPa)

| | +1h | +3h | +6h | +12h | +24h |
|---|---|---|---|---|---|
| **LNN production** | **0.3942** | **0.9311** | **1.3153** | **0.7967** | **1.0746** |
| persistence | 0.3943 | 0.9312 | 1.3153 | 0.7966 | 1.0746 |
| ECMWF IFS | 2.2194 | 2.2173 | 2.2191 | 2.2205 | 2.1985 |
| NOAA GFS | 2.5121 | 2.5099 | 2.5114 | 2.5114 | 2.4872 |
| DWD ICON | 2.3112 | 2.3058 | 2.3051 | 2.2968 | 2.2682 |
| Meteo-France | 2.3646 | 2.3614 | 2.3622 | 2.3632 | 2.3400 |
| JMA | 2.5392 | 2.5369 | 2.5333 | 2.5303 | 2.5014 |
| UK Met Office | 2.6293 | 2.6269 | 2.6315 | 2.6268 | 2.5879 |
| ECCC GEM | 2.4504 | 2.4474 | 2.4481 | 2.4450 | 2.4164 |

Pressure beats all seven NWP models at all five horizons. **This is not a modelling
result.** The policy serves persistence for pressure at every horizon, so the served value
*is* the origin observation and the MAE equals persistence to within ~1e-4 hPa at every
horizon. The real finding is that **persistence beats a 0.25° grid cell on surface pressure
at these stations** — surface pressure over terrain has strong elevation structure that a
coarse model cell smooths away, and the station samples its own terrain exactly.

The pressure head itself contributes essentially nothing: with the gate open it scores
1.3153 hPa at +6h against persistence's 1.3153 hPa.

### 7.4 Humidity (MAE, %RH)

| | +1h | +3h | +6h | +12h | +24h |
|---|---|---|---|---|---|
| **LNN production** | 1.6774 | 3.2533 | 4.8844 | 6.3619 | 4.4134 |
| persistence | 1.6765 | 3.2532 | 4.8843 | 6.3622 | 4.4133 |
| ECMWF IFS | 5.6760 | 5.6994 | 5.7046 | 5.7217 | 5.9472 |
| NOAA GFS | 7.1604 | 7.2024 | 7.2528 | 7.3511 | 7.6617 |
| DWD ICON | 6.2439 | 6.2804 | 6.3046 | 6.3465 | 6.5561 |
| Meteo-France | 5.4015 | 5.4331 | 5.4549 | 5.5217 | 5.7753 |
| JMA | 10.4410 | 10.4752 | 10.5086 | 10.4986 | 10.6949 |
| UK Met Office | 6.7297 | 6.7697 | 6.8108 | 6.7477 | 6.8734 |
| ECCC GEM | 5.0046 | 5.0377 | 5.0643 | 5.1055 | 5.3814 |

The served forecast matches persistence to within **0.1% at every horizon**, because it
*is* persistence at every horizon. The apparent 3–4× margin over ECMWF and GFS measures
station-versus-grid humidity disagreement, not forecast skill.

### 7.5 Rain occurrence (Brier score, lower is better)

Wet-hour base rate runs 39.8–43.1% across horizons.

| | +1h | +3h | +6h | +12h | +24h |
|---|---|---|---|---|---|
| **LNN production** | **0.1373** | **0.1859** | **0.2209** | **0.2364** | 0.2506 |
| persistence | 0.1633 | 0.2133 | 0.2456 | 0.2646 | 0.2895 |
| climatology (constant) | 0.2453 | 0.2448 | 0.2435 | 0.2424 | **0.2396** |
| DWD ICON | 0.2416 | 0.2415 | 0.2428 | 0.2415 | 0.2440 |
| ECMWF IFS | 0.2515 | 0.2522 | 0.2546 | 0.2540 | 0.2560 |
| NOAA GFS | 0.2539 | 0.2538 | 0.2533 | 0.2521 | 0.2535 |
| JMA | 0.2592 | 0.2609 | 0.2630 | 0.2625 | 0.2669 |
| Meteo-France | 0.2671 | 0.2660 | 0.2660 | 0.2660 | 0.2699 |
| ECCC GEM | 0.2686 | 0.2691 | 0.2717 | 0.2710 | 0.2722 |
| UK Met Office | 0.2961 | 0.2968 | 0.2987 | 0.2991 | 0.3007 |

**This is the system's one unambiguous result — with a sharply decaying margin, which must
travel with it.** Rank and margin against the strongest field in each horizon:

| Horizon | Rank | vs best NWP (DWD ICON) | vs persistence | vs climatology |
|---|---|---|---|---|
| +1h | **1 / 10** | +43.2% | +15.9% | +44.0% |
| +3h | **1 / 10** | +23.0% | +12.8% | +24.1% |
| +6h | **1 / 10** | +9.0% | +10.1% | +9.3% |
| +12h | **1 / 10** | +2.1% | +10.7% | +2.5% |
| +24h | **3 / 10** | **−2.7%** | +13.5% | **−4.6%** |

Three things must be said together, and the third is the one most often dropped:

1. From +1h through +12h the served forecast ranks **1st of 10** — ahead of persistence,
   climatology and all seven NWP models. It is also the only channel where the learned model
   participates at every horizon.
2. **The NWP margin decays from 43% to 2%.** The +12h win is real but thin: 0.2364 against
   DWD ICON's 0.2415 is a margin inside the run-to-run noise of any calibration change.
   Quoting "+43% better than ECMWF on rain" without the horizon attached is a
   misrepresentation of this table.
3. **At +24h the claim fails outright.** The served forecast ranks **3rd of 10**:
   climatology (0.2396) and DWD ICON (0.2440) both beat it (0.2506). The +24h window set is
   also the smallest (n = 1,984). Neither "beats climatology at every horizon" nor "beats
   all seven NWP models at every horizon" is supported by this data. Both must not be said.

---

## 8. What this system is, stated plainly

- It is a **per-variable, per-horizon source-selection layer** with a CfC/LNN behind it and
  a hybrid rain blend.
- Its **rain occurrence** result ranks 1st of 10 from +1h through +12h, ahead of persistence,
  climatology and all seven NWP models. The NWP margin decays from 43% at +1h to 2% at +12h,
  and at +24h it ranks **3rd** — climatology and DWD ICON both beat it.
- Its **wind** result beats every NWP model at +6h and +12h, and beats persistence by 7.8%
  and 17.8%. The margin is measured against a ground truth containing three dead anemometers.
- Its **temperature** result beats persistence by 13.0% at +6h and 23.8% at +12h, and
  **loses to ECMWF at both**.
- Its **pressure** and **humidity** results are persistence results wearing the model's
  clothes.
- It is **not** competitive with ECMWF or GFS as a general-purpose weather forecast, and
  this document should never be cited as saying so.

Companion documents:
[known-limitations.md](known-limitations.md) ·
[operations-runbook.md](operations-runbook.md)