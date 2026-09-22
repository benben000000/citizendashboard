# Prediction Model Registry

This document formally distinguishes, versions, and records the evaluation scorecards for the distinct model families present in the `prediction-model/` directory, establishing strict scientific governance per the system audits (`prediction-model-audit.md`, `prediction-model-audit-followup.md`, and `prediction-model-audit-ec6303b.md`).

---

## Registry Overview

| Model ID | Name | Architecture | Framework | Training & Evaluation Contract | Current Status | Primary Artifact |
|---|---|---|---|---|---|---|
| **MF-1** | `WeatherWaterLNN` | Continuous-Time CfC Recurrent Neural Network | PyTorch | Multi-task gradient descent on canonical hourly future windows ($t_0 + h$) | `RESEARCH_PROTOTYPE` | `data/lnn_weather_water.pt`, `data/lnn_weather_water_h*.pt` |
| **MF-2** | `ContinuousLNNCell` | Analytical ODE Closed-Form Continuous Cell | Pure Python / Zero-dependency | Unrolled 24-step sequence training on canonical hourly future windows ($t_0 + h$) | `RESEARCH_PROTOTYPE` | `data/lnn_trained_weights.json`, `data/lnn_trained_weights_h*.json` |
| **MF-3** | `StationAdaptivePINN` | Diurnal/Heuristic physics-coupled engine | Pure Python | Online parameter adaptation & diurnal priors | `EXPERIMENTAL_BASELINE` | Configs in `config/` |

---

## Canonical Forecast Data Contract & Governance

Both **MF-1** and **MF-2** are trained and evaluated under a unified, canonical forecast data contract:

### 1. Data Cleaning & Outlier Quarantine
- **Timezone-aware UTC**: All timestamps parsed into standard UTC.
- **Date Quarantine**: Strictly enforces valid collection bounds $[2025, 2027]$. Records outside this range (specifically the year 2069 sensor clock anomaly, 2 rows) are quarantined and tracked.
- **Physical Sensor Bounds**: Sensor spikes outside physical limits are quarantined:
  - Temperature: $10.0$ to $50.0^\circ$C (6,634 rows quarantined)
  - Heat Index: $10.0$ to $70.0^\circ$C
  - Wind Speed: $0.0$ to $180.0$ km/h (183 rows quarantined)
  - Pressure: $900.0$ to $1050.0$ hPa (16,134 rows quarantined)
  - Precipitation: $0.0$ to $300.0$ mm (432 rows quarantined)
  - Water Gauge: $0.0$ to $15.0$ m (12 rows quarantined, e.g. $-57.73$m spike)
- **Data Quality Manifest**: Recorded in `data/cleaned_data_manifest.json` with SHA-256 hashes:
  - `weather_telemetry.csv`: `86ce906453aa071e12726e91da6c96c516285cf0bc7c3e3fb51d8d111f3bea4a`
  - `water_level_telemetry.csv`: `acca3ffed16592209dd25ba562a1057ff95ecb4f285e521567347acb6099c872`

### 2. Hourly Resampling Policy
Raw telemetry is approximately 1-minute irregular telemetry. It is resampled onto an explicit UTC hourly grid:
- **Temperature, Heat Index, Wind Speed, Pressure**: Last valid observation within the hourly bin.
- **Precipitation**: Sum of tipping-bucket minute increments within the hour, yielding total hourly precipitation volume in mm.
- **Water Gauge Stage**: Last valid observation within the hour from collocated gauge station (Calumpit WLMS, `O3z0j5bG`) matched to Calumpit AWS (`3nzr48bG`).
- **Completeness**: Total resampled corpus yields 12,537 station-hours across 16 stations and 2,690 water gauge hours.

### 3. Exact Forecast Sample Specification
For forecast horizon $h \in \{1, 3, 6, 12, 24\}$ hours:
- **Forecast Origin $t_0$**: Final observation timestamp in the input window.
- **Input Window**: Preceding 24 consecutive hourly observations ending at $t_0$ ($[t_{-23}, \dots, t_0]$).
- **Target Timestamp**: Exactly $t_0 + h$ hours on the hourly grid.
- **Elapsed Lead Time Validation**: Verified with tolerance $|\text{actual\_lead\_hours} - h| \le 0.25$ hours. Resolves previous row-offset defect (where $h=1$ targeted the next telemetry row $\approx 1$ minute ahead).
- **Recurrent State Reset**: Hidden state is re-initialized to $h=0$ at the start of every sequence window to guarantee zero cross-window or cross-station state leakage.

### 4. Chronological Splitting with Embargo
- **Splits**: Chronological partitioning (60% Train: June 20 to August 1; 20% Calibration/Val: August 3 to August 13; 20% Test: August 15 to August 26, 2026).
- **Embargo**: 48-hour embargo ($\text{seq\_len} + \text{max\_horizon}$) between train and val, and between val and test, preventing any boundary context leakage.
- **Normalization**: Feature means ($[27.91, 32.22, 1.59, 1005.75]$) and standard deviations ($[3.20, 7.03, 3.62, 4.51]$) are fitted exclusively on the clean training partition and persisted in all model artifacts.

---

## Out-of-Sample Validation Scorecard

Evaluated on the **untouched test split** (2,820 sequence windows across 16 stations, including 221 collocated river gauge observations).

### Multi-Horizon Rain Forecasting Metrics (Threshold = 0.5)

| Horizon | Model / Baseline | Accuracy | Precision | Recall (POD) | F1-Score | CSI | Brier Score |
|---|---|---:|---:|---:|---:|---:|---:|
| **+1h** | **MF-1 PyTorch** | 81.28% | 78.12% | 70.55% | 74.14% | 58.91% | **0.1385** |
| | **MF-2 Standalone** | 73.40% | **92.61%** | 32.71% | 48.35% | 31.88% | 0.1873 |
| | **Persistence** | **82.87%** | 77.37% | 77.73% | **77.55%** | **63.33%** | 0.1713 |
| | **Logistic Regression** | 77.84% | 90.14% | 46.88% | 61.68% | 44.59% | 0.1516 |
| | **Climatology** | 61.95% | 0.00% | 0.00% | 0.00% | 0.00% | 0.2778 |
| **+3h** | **MF-1 PyTorch** | 73.64% | 70.70% | 52.41% | 60.20% | 43.06% | **0.1923** |
| | **MF-2 Standalone** | 63.51% | **93.88%** | 4.34% | 8.30% | 4.33% | 0.2471 |
| | **Persistence** | **77.30%** | 69.93% | **70.73%** | **70.33%** | **54.24%** | 0.2270 |
| | **Logistic Regression** | 66.42% | 81.63% | 15.11% | 25.50% | 14.61% | 0.2033 |
| | **Climatology** | 61.96% | 0.00% | 0.00% | 0.00% | 0.00% | 0.2777 |
| **+6h** | **MF-1 PyTorch** | 67.41% | 59.75% | 41.15% | 48.73% | 32.22% | 0.2497 |
| | **MF-2 Standalone** | 62.61% | 70.59% | 1.17% | 2.30% | 1.16% | 0.2625 |
| | **Persistence** | **73.05%** | 63.75% | **65.86%** | **64.78%** | **47.91%** | 0.2695 |
| | **Logistic Regression** | 63.09% | **70.83%** | 3.31% | 6.32% | 3.26% | **0.2249** |
| | **Climatology** | 62.36% | 0.00% | 0.00% | 0.00% | 0.00% | 0.2751 |
| **+12h** | **MF-1 PyTorch** | 65.70% | 55.65% | 41.02% | 47.22% | 30.91% | 0.2664 |
| | **MF-2 Standalone** | 62.59% | 0.00% | 0.00% | 0.00% | 0.00% | 0.2679 |
| | **Persistence** | **70.00%** | 59.37% | **62.74%** | **61.01%** | **43.89%** | 0.3000 |
| | **Logistic Regression** | 62.59% | 50.00% | 0.10% | 0.20% | 0.10% | **0.2337** |
| | **Climatology** | 62.59% | 0.00% | 0.00% | 0.00% | 0.00% | 0.2736 |
| **+24h** | **MF-1 PyTorch** | **68.32%** | **59.16%** | 41.44% | 48.74% | 32.22% | 0.2516 |
| | **MF-2 Standalone** | 63.65% | 0.00% | 0.00% | 0.00% | 0.00% | 0.2668 |
| | **Persistence** | 66.72% | 53.80% | **59.80%** | **56.64%** | **39.51%** | 0.3328 |
| | **Logistic Regression** | 63.65% | 0.00% | 0.00% | 0.00% | 0.00% | **0.2373** |
| | **Climatology** | 63.65% | 0.00% | 0.00% | 0.00% | 0.00% | 0.2668 |

---

### Multi-Horizon River Stage Hydrological Metrics (Calumpit Gauge, N=221 test samples)

| Horizon | Model Family | MAE (m) | RMSE (m) | Bias (m) | Persistence MAE (m) | Climatology MAE (m) | Beats Persistence? |
|---|---|---:|---:|---:|---:|---:|:---:|
| **+1h** | **MF-1 PyTorch** | 0.0507 m | 0.0573 m | 0.0147 m | **0.0146 m** | 1.2640 m | **NO** |
| | **MF-2 Standalone** | 0.1253 m | 0.1798 m | 0.0624 m | **0.0146 m** | 1.2640 m | **NO** |
| **+3h** | **MF-1 PyTorch** | 0.1847 m | 0.2372 m | -0.0376 m | **0.0422 m** | 1.2660 m | **NO** |
| | **MF-2 Standalone** | 0.1244 m | 0.1674 m | 0.0378 m | **0.0422 m** | 1.2660 m | **NO** |
| **+6h** | **MF-1 PyTorch** | 0.3974 m | 0.4554 m | -0.0532 m | **0.0767 m** | 1.2688 m | **NO** |
| | **MF-2 Standalone** | 0.1273 m | 0.1689 m | 0.0366 m | **0.0767 m** | 1.2688 m | **NO** |
| **+12h** | **MF-1 PyTorch** | 0.4751 m | 0.5669 m | 0.0916 m | **0.1111 m** | 1.2720 m | **NO** |
| | **MF-2 Standalone** | 0.1409 m | 0.1716 m | 0.0322 m | **0.1111 m** | 1.2720 m | **NO** |
| **+24h** | **MF-1 PyTorch** | **0.1459 m** | **0.1664 m** | 0.0455 m | 0.1495 m | 1.2793 m | **YES** |
| | **MF-2 Standalone** | 0.1605 m | 0.1915 m | 0.0676 m | 0.1495 m | 1.2793 m | **NO** |

---

### Conformal Prediction Coverage Evaluation (Test Partition, N=221)

Quantiles calibrated on held-out calibration split ($N_{\text{calib}}=202$), evaluated on independent test set ($N_{\text{test}}=221$).

| Nominal Coverage | Metric | MF-1 PyTorch | MF-2 Standalone | Assessment |
|---|---|---:|---:|:---:|
| **80.0%** | Observed Coverage | **95.48%** | **83.26%** | **PASSES (Meets Nominal)** |
| | Interval Full Width | 0.183 m | 0.475 m | Tight (< 0.5m) |
| | 95% Wilson CI | [91.87%, 97.52%] | [77.78%, 87.60%] | Coverage target satisfied |
| **90.0%** | Observed Coverage | **99.10%** | **91.86%** | **PASSES (Meets Nominal)** |
| | Interval Full Width | 0.232 m | 0.711 m | Tight (< 0.75m) |
| | 95% Wilson CI | [96.76%, 99.75%] | [87.49%, 94.79%] | Coverage target satisfied |
| **95.0%** | Observed Coverage | **100.0%** | **95.02%** | **PASSES (Meets Nominal)** |
| | Interval Full Width | 0.283 m | 0.819 m | Tight (< 0.85m) |
| | 95% Wilson CI | [98.29%, 100.0%] | [91.31%, 97.20%] | Coverage target satisfied |

---

### Rain Intensity Breakdown (Horizon +1h)

The resampled test set contains genuine precipitation events across all severity categories:

| Intensity Class | Range (mm/h) | Test Sample Count | MF-1 POD | MF-2 POD | Persistence POD |
|---|---|---:|---:|---:|---:|
| **Dry** | $< 0.1$ | 1,747 | 12.14% FA | 1.60% FA | 13.97% FA |
| **Trace** | $0.1$ to $0.5$ | 320 | 45.62% | 15.62% | 59.38% |
| **Light** | $0.5$ to $2.5$ | 437 | 79.86% | 23.11% | 84.44% |
| **Moderate** | $2.5$ to $7.5$ | 204 | 83.82% | 62.75% | 88.73% |
| **Heavy** | $> 7.5$ | 112 | 81.25% | 64.29% | 83.93% |

---

## Governance & Operational Recommendation

> [!CAUTION]
> **OPERATIONAL RECOMMENDATION: DO NOT DEPLOY FOR AUTONOMOUS OPERATION.**
> **Current Classification: `RESEARCH_PROTOTYPE`**
> 
> 1. **Rain Forecast**: MF-1 achieves strong probabilistic calibration (Brier Score 0.1385 at +1h, outperforming Persistence 0.1713 and Logistic Regression 0.1516) and 74.14% F1, but operational persistence remains ahead at 77.55% F1 at +1h.
> 2. **River Stage Forecast**: Conditioned delta-stage formulation lowered water MAE to 0.0507m at +1h and 0.1459m at +24h (beating Persistence 0.1495m at +24h); however, Persistence at +1h (0.0146m) remains much tighter.
> 3. **Uncertainty Calibration**: Conformal prediction intervals satisfy nominal coverage targets across all tested levels (95.48% at nominal 80%, 99.10% at nominal 90%, 100.0% at nominal 95%) with narrow practical widths (< 0.28m).
> 4. **Safety Decision**: Because persistence remains superior across short multi-hour horizons on this low-dynamic regime, models remain designated strictly as `RESEARCH_PROTOTYPE` and must not be used for life-safety or automated flood evacuation triggers.
