# Garcia Weather Telemetry Forecast Engine: Model Registry

This document formally distinguishes, versions, and records the evaluation scorecards and operational governance for the forecast models in `prediction-model/`, establishing strict scientific standards per `prediction_model_weather_only_checklist.md` and repository audit protocols.

---

## 1. Executive Product Definition & Boundary

The primary product is defined as the **Garcia Weather Telemetry Forecast Engine**. It is a **surface weather monitoring, trend analysis, and probabilistic guidance system** designed to convert station telemetry into real-time operational weather insights.

> [!IMPORTANT]
> **Status: RESEARCH_PROTOTYPE**
> **Not approved for autonomous flood alarms, emergency evacuation triggers, or life-safety operations.**
> Hydrological river-stage forecasting remains an **internal/beta research capability** (`not_for_life_safety: true`). Flood prediction claims are explicitly excluded from the commercial product core until gauge density and multi-event historical records are materially expanded.

### Explicit Output Taxonomy

| Output Class | Variables / Capabilities | Permitted Operational Use | Governing Rule |
|---|---|---|---|
| **Core Commercial** | Current observations, rising/falling trends, temperature outlook, relative humidity outlook, barometric pressure tendency, wind speed & circular direction, calibrated rain probability, derived NOAA heat index | Commercial monitoring, agricultural and industrial planning, weather dashboards, with calibrated probability labels (conformal intervals apply strictly to beta water level; weather prediction intervals are unavailable) | Deployed via hybrid guidance: learned model where skill is positive, persistence fallback where persistence error is lower |
| **Secondary / Beta** | Rain accumulation ranges (mm), light intensity (lux daylight cycle), river stage stage delta (meters) | Opt-in research and beta testing only; requires explicit beta flag | River stage carries mandatory `INTERNAL_EXPERIMENT_BETA` and `not_for_life_safety: true` metadata |
| **Blocked by Data** | UV Index | Quarantined / not scored | Blocked due to raw sensor calibration defect (reporting up to 11.0 index during nighttime 00:00–04:00 local time) |
| **Prohibited Claims** | Flood evacuation triggers, autonomous flood warnings, guaranteed rain/no-rain binary statements, unvalidated weather confidence bands | **STRICTLY PROHIBITED** | Not supported by current telemetry evidence or model validation |

---

## 2. Model Family Registry

| Model ID | Name | Architecture | Framework | Training & Evaluation Contract | Current Status | Primary Artifacts |
|---|---|---|---|---|---|---|
| **MF-1** | `GarciaWeatherLNN` | Continuous-Time CfC Recurrent Neural Network with Multi-Task Surface Meteorology Heads | PyTorch | Multi-task gradient descent on canonical hourly future windows ($t_0 + h$) | `RESEARCH_PROTOTYPE` | `data/lnn_weather_water.pt`, `data/lnn_weather_water_h{1,3,6,12,24}.pt` |
| **MF-2** | `ContinuousLNNCell` | Analytical ODE Closed-Form Continuous Cell | Pure Python / Zero-dependency | Unrolled 24-step sequence training on canonical hourly future windows ($t_0 + h$) | `RESEARCH_PROTOTYPE` | `data/lnn_trained_weights.json`, `data/lnn_trained_weights_h{1,3,6,12,24}.json` |
| **MF-3** | `StationAdaptivePINN` | Diurnal/Heuristic physics-coupled engine | Pure Python | Online parameter adaptation & diurnal priors | `EXPERIMENTAL_BASELINE` | Configs in `config/` |

---

## 3. Canonical Telemetry & Target Contract

Both **MF-1** and **MF-2** operate under a verified, leak-free forecast data contract:

### 3.1 Data Cleaning & Outlier Quarantine
- **Timezone-aware UTC**: All timestamps parsed into timezone-aware standard UTC.
- **Date Quarantine**: Valid collection bounds $[2025, 2027]$ strictly enforced; timestamp anomalies (year 2069 clock bug, 2 rows) quarantined.
- **Physical Sensor Bounds Quarantine**:
  - Temperature: $10.0$ to $50.0^\circ$C (6,634 rows quarantined)
  - Heat Index: $10.0$ to $70.0^\circ$C (5,853 rows quarantined)
  - Humidity: $10.0$ to $100.0\%$ (sensor dropouts $< 10\%$ quarantined)
  - Wind Speed: $0.0$ to $180.0$ km/h (183 rows quarantined)
  - Pressure: $900.0$ to $1050.0$ hPa (16,134 rows quarantined)
  - Precipitation: $0.0$ to $50.0$ mm/min incremental (434 rows quarantined, including extreme spikes $> 7000$ mm/min)
  - Water Gauge: $0.0$ to $15.0$ m (12 rows quarantined, e.g. $-57.73$m spike)
- **Deterministic Data Audit**: Documented in `data/weather_data_audit.json` with verified SHA-256 hashes:
  - `weather_telemetry.csv`: `86ce906453aa071e12726e91da6c96c516285cf0bc7c3e3fb51d8d111f3bea4a`
  - `water_level_telemetry.csv`: `acca3ffed16592209dd25ba562a1057ff95ecb4f285e521567347acb6099c872`

### 3.2 Hourly Resampling Policy
Raw telemetry is approximately 1-minute irregular telemetry resampled onto an explicit UTC hourly grid:
- **Temperature, Humidity, Pressure, Wind Speed**: Last valid observation within the hourly bin.
- **Wind Direction**: Vector average $(u = \cos\theta, v = \sin\theta)$ reconstructed via circular `atan2`; calm winds ($< 1.0$ km/h) are explicitly masked.
- **Precipitation**: Sum of tipping-bucket minute increments within the hour, yielding total hourly precipitation volume in mm.
- **Water Gauge Stage**: Last valid observation within the hour from collocated gauge station (Calumpit WLMS, `O3z0j5bG`) matched to Calumpit AWS (`3nzr48bG`).
- **Completeness**: 12,537 station-hours across 16 stations and 2,690 water gauge hours.

### 3.3 Forecast Sample Specification
For forecast horizon $h \in \{1, 3, 6, 12, 24\}$ hours:
- **Forecast Origin $t_0$**: Final observation timestamp in the input window.
- **Input Window**: Preceding 24 consecutive hourly observations ending at $t_0$ ($[t_{-23}, \dots, t_0]$).
- **Target Timestamp**: Exactly $t_0 + h$ hours on the hourly grid.
- **Elapsed Lead Time Validation**: Verified with tolerance $|\text{actual\_lead\_hours} - h| \le 0.25$ hours (0 tolerance violations across 13,411 test samples).

### 3.4 Chronological Splitting with Embargo
- **Splits**: Chronological partitioning (60% Train: June 20 to August 1; 20% Calibration/Val: August 3 to August 13; 20% Test: August 15 to August 26, 2026).
- **Embargo**: 48-hour embargo ($\text{seq\_len} + \text{max\_horizon}$) between train and val, and between val and test, preventing any boundary context leakage.
- **Normalization**: Feature means and standard deviations fitted exclusively on the clean training partition and persisted in all model checkpoints.

---

## 4. Multi-Horizon Weather Telemetry Scorecard

Evaluated on the **untouched test split** (2,820 sequence windows across 16 stations). All hybrid blend weights and operational decision thresholds were fitted exclusively on the held-out calibration split and frozen prior to test evaluation.

### 4.1 Surface Meteorology Metrics

| Horizon | Variable | Target Unit | Model MAE | Persistence MAE | Climatology MAE | Skill vs Persistence | Gate Selection |
|---|---|---|---:|---:|---:|---:|:---:|
| **+1h** | Temperature | $^\circ$C | 0.51 | 0.49 | 1.61 | -3.10% | `PERSISTENCE_FALLBACK` |
| | Relative Humidity | % | 1.47 | 1.40 | 9.84 | -4.84% | `PERSISTENCE_FALLBACK` |
| | Pressure | hPa | **0.33** | 0.33 | 3.16 | **+0.73%** | `LEARNED_MODEL` |
| | Wind Speed | km/h | 0.85 | 0.85 | 2.30 | -0.87% | `PERSISTENCE_FALLBACK` |
| | Heat Index (NOAA) | $^\circ$C | **1.49** | 1.52 | 4.09 | **+1.94%** | `LEARNED_DERIVED` |
| | Wind Direction | deg (circ) | **42.1$^\circ$** (N=2,187) | -- | -- | Circular MAE | `CIRCULAR_VECTOR` |
| **+3h** | Temperature | $^\circ$C | 1.00 | 0.96 | 1.61 | -4.10% | `PERSISTENCE_FALLBACK` |
| | Relative Humidity | % | 2.96 | 2.80 | 9.85 | -5.74% | `PERSISTENCE_FALLBACK` |
| | Pressure | hPa | **0.67** | 0.78 | 3.16 | **+14.12%** | `LEARNED_MODEL` |
| | Wind Speed | km/h | 1.08 | 1.06 | 2.30 | -1.75% | `PERSISTENCE_FALLBACK` |
| | Heat Index (NOAA) | $^\circ$C | **2.62** | 2.66 | 4.09 | **+1.50%** | `LEARNED_DERIVED` |
| | Wind Direction | deg (circ) | **71.9$^\circ$** (N=2,187) | -- | -- | Circular MAE | `CIRCULAR_VECTOR` |
| **+6h** | Temperature | $^\circ$C | 1.48 | 1.44 | 1.62 | -2.71% | `PERSISTENCE_FALLBACK` |
| | Relative Humidity | % | 4.88 | 4.23 | 9.86 | -15.42% | `PERSISTENCE_FALLBACK` |
| | Pressure | hPa | **1.10** | 1.11 | 3.17 | **+0.89%** | `LEARNED_MODEL` |
| | Wind Speed | km/h | 1.32 | 1.29 | 2.30 | -2.25% | `PERSISTENCE_FALLBACK` |
| | Heat Index (NOAA) | $^\circ$C | **3.81** | 3.86 | 4.10 | **+1.30%** | `LEARNED_DERIVED` |
| | Wind Direction | deg (circ) | **87.7$^\circ$** (N=2,187) | -- | -- | Circular MAE | `CIRCULAR_VECTOR` |
| **+12h** | Temperature | $^\circ$C | **1.56** | 1.83 | 1.63 | **+14.80%** | `LEARNED_MODEL` |
| | Relative Humidity | % | **5.08** | 5.37 | 9.87 | **+5.35%** | `LEARNED_MODEL` |
| | Pressure | hPa | 0.79 | 0.69 | 3.18 | -14.99% | `PERSISTENCE_FALLBACK` |
| | Wind Speed | km/h | **1.35** | 1.52 | 2.30 | **+11.21%** | `LEARNED_MODEL` |
| | Heat Index (NOAA) | $^\circ$C | **4.02** | 4.80 | 4.11 | **+16.20%** | `LEARNED_DERIVED` |
| | Wind Direction | deg (circ) | **81.9$^\circ$** (N=2,187) | -- | -- | Circular MAE | `CIRCULAR_VECTOR` |
| **+24h** | Temperature | $^\circ$C | 1.23 | 1.21 | 1.64 | -1.57% | `PERSISTENCE_FALLBACK` |
| | Relative Humidity | % | 4.00 | 3.83 | 9.89 | -4.41% | `PERSISTENCE_FALLBACK` |
| | Pressure | hPa | 0.96 | 0.94 | 3.18 | -1.82% | `PERSISTENCE_FALLBACK` |
| | Wind Speed | km/h | 1.26 | 1.24 | 2.30 | -1.53% | `PERSISTENCE_FALLBACK` |
| | Heat Index (NOAA) | $^\circ$C | 3.29 | 3.17 | 4.12 | -3.67% | `PERSISTENCE_FALLBACK` |
| | Wind Direction | deg (circ) | **93.0$^\circ$** (N=2,187) | -- | -- | Circular MAE | `CIRCULAR_VECTOR` |

---

### 4.2 Rain Occurrence & Probabilistic Scorecard

### 4.2 Rain Occurrence & Probabilistic Scorecard

The operational system evaluates both continuous **probability quality** (Brier score, reliability, calibration error) and discrete **event classification** (F1, recall/POD, precision, CSI):

#### Probability Quality vs. Event Classification Tradeoff
- **Probability Quality (Brier Score)**: Hybrid probability blending consistently minimizes Brier score across **ALL 5 HORIZONS**:
  - +1h: Brier **0.1219** vs Persistence 0.1713 (**+28.84% skill**)
  - +3h: Brier **0.1772** vs Persistence 0.2270 (**+21.94% skill**)
  - +6h: Brier **0.2183** vs Persistence 0.2695 (**+18.99% skill**)
  - +12h: Brier **0.2144** vs Persistence 0.3000 (**+28.53% skill**)
  - +24h: Brier **0.2118** vs Persistence 0.3328 (**+36.4% skill**)
- **Event Classification at 0.5 Fixed Threshold**: While hybrid blending minimizes probability error for risk outlooks, persistence achieves comparable or slightly higher F1/recall on short horizons (1h/3h). Operational users desiring hard alerts should utilize the calibrated operational decision threshold ($T_{op}$) or persistence baseline.

| Horizon | Model / Strategy | Brier Score | Brier Skill vs Persist | Decision F1 | Accuracy | Recall (POD) | Precision | CSI |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **+1h** | **GarciaWeatherLNN** | **0.1249** | **+27.09%** | 76.28% | 82.52% | 73.19% | 79.64% | 61.66% |
| | **Calibrated Hybrid Blend ($w_m=0.8$)** | **0.1219** | **+28.84%** | **77.42%** | **83.01%** | 76.54% | 78.33% | **63.16%** |
| | Persistence | 0.1713 | 0.00% | 77.55% | 82.87% | 77.73% | 77.37% | 63.33% |
| | Logistic Regression | 0.1516 | +11.50% | 61.68% | 77.84% | 46.88% | 90.14% | 44.59% |
| | Climatology Prior | 0.2778 | -62.17% | 0.00% | 61.95% | 0.00% | 0.00% | 0.00% |
| **+3h** | **GarciaWeatherLNN** | **0.1844** | **+18.77%** | 59.33% | 73.01% | 51.74% | 69.51% | 42.18% |
| | **Calibrated Hybrid Blend ($w_m=0.6$)** | **0.1772** | **+21.94%** | **68.75%** | **76.81%** | 68.34% | 69.17% | **52.38%** |
| | Persistence | 0.2270 | 0.00% | 70.33% | 77.30% | 70.73% | 69.93% | 54.24% |
| | Logistic Regression | 0.2033 | +10.44% | 25.50% | 66.42% | 15.11% | 81.63% | 14.61% |
| | Climatology Prior | 0.2777 | -22.33% | 0.00% | 61.96% | 0.00% | 0.00% | 0.00% |
| **+6h** | **GarciaWeatherLNN** | **0.2415** | **+10.39%** | 9.87% | 63.16% | 5.37% | 61.11% | 5.09% |
| | **Calibrated Hybrid Blend ($w_m=0.45$)** | **0.2183** | **+18.99%** | **64.78%** | **73.05%** | 65.86% | 63.75% | **47.91%** |
| | Persistence | 0.2695 | 0.00% | 64.78% | 73.05% | 65.86% | 63.75% | 47.91% |
| | Logistic Regression | 0.2249 | +16.55% | 6.32% | 63.09% | 3.31% | 70.83% | 3.26% |
| | Climatology Prior | 0.2751 | -2.08% | 0.00% | 62.36% | 0.00% | 0.00% | 0.00% |
| **+12h** | **GarciaWeatherLNN** | **0.2538** | **+15.40%** | 0.00% | 62.59% | 0.00% | 0.00% | 0.00% |
| | **Calibrated Hybrid Blend ($w_m=0.45$)** | **0.2144** | **+28.53%** | **61.01%** | **70.00%** | 62.74% | 59.37% | **43.89%** |
| | Persistence | 0.3000 | 0.00% | 61.01% | 70.00% | 62.74% | 59.37% | 43.89% |
| | Logistic Regression | 0.2337 | +22.10% | 0.20% | 62.59% | 0.10% | 50.00% | 0.10% |
| | Climatology Prior | 0.2736 | +8.80% | 0.00% | 62.59% | 0.00% | 0.00% | 0.00% |
| **+24h** | **GarciaWeatherLNN** | **0.2473** | **+25.69%** | 0.00% | 63.65% | 0.00% | 0.00% | 0.00% |
| | **Calibrated Hybrid Blend ($w_m=0.4$)** | **0.2118** | **+36.36%** | **56.64%** | **66.72%** | 59.80% | 53.80% | **39.51%** |
| | Persistence | 0.3328 | 0.00% | 56.64% | 66.72% | 59.80% | 53.80% | 39.51% |
| | Logistic Regression | 0.2373 | +28.70% | 0.00% | 63.65% | 0.00% | 0.00% | 0.00% |
| | Climatology Prior | 0.2668 | +19.83% | 0.00% | 63.65% | 0.00% | 0.00% | 0.00% |

---

### 4.3 Beta River Stage Hydrological Metrics (Calumpit Gauge, N=221)

> [!CAUTION]
> **BETA RESEARCH MODULE ONLY — NOT FOR LIFE SAFETY**
> Evaluated at the single collocated river gauge in Calumpit, Bulacan (`O3z0j5bG`).

| Horizon | Model Family | MAE (m) | RMSE (m) | Bias (m) | Persistence MAE (m) | Climatology MAE (m) | Beats Persistence? |
|---|---|---:|---:|---:|---:|---:|:---:|
| **+1h** | **MF-1 PyTorch** | 0.0338 m | 0.0428 m | -0.0065 m | **0.0146 m** | 1.2640 m | **NO** |
| | **MF-2 Standalone** | 0.1253 m | 0.1798 m | 0.0624 m | **0.0146 m** | 1.2640 m | **NO** |
| **+3h** | **MF-1 PyTorch** | 0.1040 m | 0.1332 m | -0.0182 m | **0.0422 m** | 1.2660 m | **NO** |
| | **MF-2 Standalone** | 0.1244 m | 0.1674 m | 0.0378 m | **0.0422 m** | 1.2660 m | **NO** |
| **+6h** | **MF-1 PyTorch** | 0.1684 m | 0.2198 m | -0.0241 m | **0.0767 m** | 1.2688 m | **NO** |
| | **MF-2 Standalone** | 0.1273 m | 0.1689 m | 0.0366 m | **0.0767 m** | 1.2688 m | **NO** |
| **+12h** | **MF-1 PyTorch** | 0.2422 m | 0.3155 m | 0.0450 m | **0.1111 m** | 1.2720 m | **NO** |
| | **MF-2 Standalone** | 0.1409 m | 0.1716 m | 0.0322 m | **0.1111 m** | 1.2720 m | **NO** |
| **+24h** | **MF-1 PyTorch** | **0.1327 m** | **0.1624 m** | 0.0312 m | 0.1495 m | 1.2793 m | **YES** |
| | **MF-2 Standalone** | 0.1605 m | 0.1915 m | 0.0676 m | 0.1495 m | 1.2793 m | **NO** |

#### Conformal Uncertainty Coverage (Horizon +1h, N=221)
- **Nominal 80.0%**: Observed Coverage = **90.50%** (Full interval width = 0.154 m, 95% CI [85.91%, 93.70%]) -> **PASSES**
- **Nominal 90.0%**: Observed Coverage = **95.02%** (Full interval width = 0.189 m, 95% CI [91.31%, 97.20%]) -> **PASSES**
- **Nominal 95.0%**: Observed Coverage = **99.10%** (Full interval width = 0.214 m, 95% CI [96.76%, 99.75%]) -> **PASSES**

---

### 4.4 Precipitation Amount Scorecard

Evaluated on the test split (N=2,820 sequence windows across 16 stations):
- **Baselines Evaluated**: Persistence baseline ($y_{t_0}$), Climatology prior baseline, and model skill vs. persistence.
- **Subsets**: Dry-hour ($< 0.1$ mm) and Rainy-hour ($\ge 0.1$ mm) subsets evaluated separately.
- **Heavy Rain Event Detection**: Threshold metrics evaluated at $2.5$ mm (moderate), $5.0$ mm (heavy), and $10.0$ mm (very heavy).
- **Uncertainty Status**: Conformal prediction intervals for precipitation volume are explicitly **UNAVAILABLE**. No fabricated coverage claims are made.

---

## 5. Operational Deployment Recommendation

1. **Deploy Hybrid Weather Guidance**:
   - For barometric pressure (+1h, +3h, +6h) and derived heat index (+1h to +12h), the learned model outperforms persistence and is selected by policy.
   - For short-term temperature, humidity, and wind speed (+1h to +6h), operational persistence is selected as fallback.
   - At +12h, the learned model captures diurnal cycle shifts and outperforms persistence in temperature (MAE 1.56°C vs 1.83°C) and wind speed (MAE 1.35 vs 1.52 km/h).
2. **Deploy Calibrated Rain Probabilities**:
   - The learned model's continuous probability outputs are superior to persistence across all horizons (Brier score improvements of 10.4% to 36.4%).
   - Customer-facing interfaces should display the **calibrated probability** rather than binary rain/no-rain claims. Note: Surface weather variables do NOT have validated confidence intervals (uncertainty intervals unavailable; conformal intervals apply ONLY to the beta river stage experiment).
3. **Quarantine Solar UV Index**:
   - UV index output is marked `BLOCKED_BY_SENSOR_CALIBRATION` and withheld from automated scoring due to raw hardware calibration defects.
4. **Isolate River Level as Beta**:
   - River level delta forecasts must retain explicit metadata: `status: "INTERNAL_EXPERIMENT_BETA"`, `not_for_life_safety: true`.
   - Never trigger automated sirens, alerts, or evacuation directives from this module.

---

## 6. Model-Policy Packaging & Provenance Architecture

To eliminate operational drift and prevent API bypass of the validated inference policy, model weights and policies are packaged into deterministic, verified bundles:

### 6.1 Bundle Structure

```plaintext
prediction-model/data/bundles/
├── h1/   (checkpoint.pt, inference_policy.json, bundle_manifest.json, README.md)
├── h3/   (checkpoint.pt, inference_policy.json, bundle_manifest.json, README.md)
├── h6/   (checkpoint.pt, inference_policy.json, bundle_manifest.json, README.md)
├── h12/  (checkpoint.pt, inference_policy.json, bundle_manifest.json, README.md)
└── h24/  (checkpoint.pt, inference_policy.json, bundle_manifest.json, README.md)
```

Each `bundle_manifest.json` contains:
- `bundle_version`
- `horizon_hours`
- `implementation_commit` (source code commit SHA)
- `artifact_commit` (git commit containing the bundle artifacts)
- `checkpoint_sha256` (cryptographic SHA-256 hash of `checkpoint.pt`)
- `policy_sha256` (cryptographic SHA-256 hash of `inference_policy.json`)
- `raw_weather_dataset_sha256` & `raw_water_dataset_sha256`
- `feature_schema` (canonical 8 features)
- `model_family` (`GarciaWeatherLNN`)

### 6.2 Fail-Closed Cryptographic Verification
During inference initialization, `LNNServerlessPredictor` validates all hashes against `bundle_manifest.json`. Any missing files, horizon mismatches, or tampered bytes immediately raise a fail-closed exception.

---

## 7. Experimental Two-Stage Precipitation & Monitoring

### 7.1 Two-Stage Precipitation Architecture (Experimental)
A decoupled two-stage precipitation architecture is available in `model.py` behind the explicit flag `use_two_stage_precipitation=True`:
1. **Occurrence Head**: Binary classifier logit for $P(\text{Rain} > 0)$.
2. **Conditional Amount Head**: Non-negative regression for $E[\text{Rain} \mid \text{Rain} > 0]$ via softplus activation.
3. **Expected Value**: $E[Y] = P(Y > 0) \times E[Y \mid Y > 0]$.

**Comparative Evaluation Results (Test Partition, N=908):**
- Baseline Single-Head MAE: **1.6998 mm** (Dry-hour: 0.0869 mm, Rainy-hour: 3.2638 mm)
- Experimental Two-Stage MAE: **2.0637 mm** (Dry-hour: 0.3020 mm, Rainy-hour: 3.7719 mm)
- **Decision**: Retain baseline single-head as production default (`use_two_stage_precipitation=False`). The two-stage architecture remains fully integrated for research into custom Tweedie/focal loss functions.

### 7.2 Post-Deployment Monitoring & Drift Detection
The monitoring suite (`prediction-model/src/monitoring.py`) evaluates model telemetry across:
- **Dimensions**: Horizon, station, rain/dry regime, heavy-rain regime (2.5, 5.0, 10.0 mm/h), calm/windy regime.
- **Continuous Metrics**: MAE, RMSE, bias, persistence MAE, skill score, climatology MAE, missingness.
- **Event Metrics**: Brier score, ECE calibration error, F1, precision, POD/recall, FAR, CSI, confusion matrix.
- **Safeguards**: Bootstrap 95% confidence intervals, minimum sample count requirements, and feature distribution drift detection. Monitoring never mutates the frozen operational policy automatically.
