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

## Out-of-Sample Validation Scorecard (Commit `ec6303b`)

Evaluated on the **untouched test split** (2,820 sequence windows across 16 stations, including 221 collocated river gauge observations).

### Multi-Horizon Rain Forecasting Metrics (Threshold = 0.5)

| Horizon | Model / Baseline | Accuracy | Precision | Recall (POD) | F1-Score | CSI | Brier Score |
|---|---|---:|---:|---:|---:|---:|---:|
| **+1h** | **MF-1 PyTorch** | 65.82% | **79.14%** | 13.79% | 23.49% | 13.31% | 0.2540 |
| | **MF-2 Standalone** | 61.95% | 0.00% | 0.00% | 0.00% | 0.00% | 0.2713 |
| | **Persistence** | **82.87%** | 77.37% | **77.73%** | **77.55%** | **63.33%** | **0.1713** |
| | **Climatology** | 61.95% | 0.00% | 0.00% | 0.00% | 0.00% | 0.2778 |
| **+3h** | **MF-1 PyTorch** | 64.04% | **77.55%** | 9.92% | 17.58% | 9.64% | 0.2592 |
| | **MF-2 Standalone** | 62.11% | 0.00% | 0.00% | 0.00% | 0.00% | 0.2713 |
| | **Persistence** | **77.41%** | 70.36% | **70.27%** | **70.31%** | **54.22%** | **0.2259** |
| | **Climatology** | 62.11% | 0.00% | 0.00% | 0.00% | 0.00% | 0.2783 |
| **+6h** | **MF-1 PyTorch** | 63.49% | **76.09%** | 8.49% | 15.28% | 8.29% | 0.2612 |
| | **MF-2 Standalone** | 62.14% | 0.00% | 0.00% | 0.00% | 0.00% | 0.2713 |
| | **Persistence** | **73.12%** | 64.83% | **64.79%** | **64.81%** | **47.94%** | **0.2688** |
| | **Climatology** | 62.14% | 0.00% | 0.00% | 0.00% | 0.00% | 0.2778 |
| **+12h** | **MF-1 PyTorch** | 64.64% | **77.78%** | 11.66% | 20.28% | 11.28% | 0.2589 |
| | **MF-2 Standalone** | 62.67% | 0.00% | 0.00% | 0.00% | 0.00% | 0.2713 |
| | **Persistence** | **70.83%** | 61.16% | **60.92%** | **61.04%** | **43.93%** | **0.2917** |
| | **Climatology** | 62.67% | 0.00% | 0.00% | 0.00% | 0.00% | 0.2762 |
| **+24h** | **MF-1 PyTorch** | 64.47% | **73.28%** | 12.87% | 21.89% | 12.28% | 0.2599 |
| | **MF-2 Standalone** | 62.63% | 0.00% | 0.00% | 0.00% | 0.00% | 0.2713 |
| | **Persistence** | **67.62%** | 56.55% | **56.70%** | **56.63%** | **39.50%** | **0.3238** |
| | **Climatology** | 62.63% | 0.00% | 0.00% | 0.00% | 0.00% | 0.2783 |

---

### Multi-Horizon River Stage Hydrological Metrics (Calumpit Gauge, N=221 test samples)

| Horizon | Model Family | MAE (m) | RMSE (m) | Bias (m) | Persistence MAE (m) | Climatology MAE (m) | Beats Persistence? |
|---|---|---:|---:|---:|---:|---:|:---:|
| **+1h** | **MF-1 PyTorch** | 1.1702 m | 1.2155 m | -1.1702 m | **0.0146 m** | 1.2640 m | **NO** |
| | **MF-2 Standalone** | 1.1452 m | 1.1721 m | -1.1452 m | **0.0146 m** | 1.2640 m | **NO** |
| **+3h** | **MF-1 PyTorch** | 1.2594 m | 1.3007 m | -1.2594 m | **0.0422 m** | 1.2640 m | **NO** |
| | **MF-2 Standalone** | 1.1497 m | 1.1764 m | -1.1497 m | **0.0422 m** | 1.2640 m | **NO** |
| **+6h** | **MF-1 PyTorch** | 1.1787 m | 1.2238 m | -1.1787 m | **0.0767 m** | 1.2640 m | **NO** |
| | **MF-2 Standalone** | 1.1951 m | 1.2205 m | -1.1951 m | **0.0767 m** | 1.2640 m | **NO** |
| **+12h** | **MF-1 PyTorch** | 1.2650 m | 1.3061 m | -1.2650 m | **0.1111 m** | 1.2640 m | **NO** |
| | **MF-2 Standalone** | 1.2697 m | 1.2932 m | -1.2697 m | **0.1111 m** | 1.2640 m | **NO** |
| **+24h** | **MF-1 PyTorch** | 1.4282 m | 1.4646 m | -1.4282 m | **0.1495 m** | 1.2640 m | **NO** |
| | **MF-2 Standalone** | 1.1670 m | 1.1927 m | -1.1670 m | **0.1495 m** | 1.2640 m | **NO** |

---

### Conformal Prediction Coverage Evaluation (Test Partition, N=221)

Quantiles calibrated on held-out calibration split ($N_{\text{calib}}=204$), evaluated on independent test set ($N_{\text{test}}=221$).

| Nominal Coverage | Metric | MF-1 PyTorch | MF-2 Standalone | Assessment |
|---|---|---:|---:|:---:|
| **80.0%** | Observed Coverage | **32.13%** | **42.53%** | **FAILS (Under-coverage)** |
| | Interval Full Width | 2.051 m | 2.328 m | Wide |
| | 95% Wilson CI | [26.32%, 38.54%] | [36.20%, 49.13%] | Upper bound below 80% |
| **90.0%** | Observed Coverage | **57.01%** | **63.35%** | **FAILS (Under-coverage)** |
| | Interval Full Width | 2.486 m | 2.537 m | Wide |
| | 95% Wilson CI | [50.42%, 63.37%] | [56.82%, 69.42%] | Upper bound below 90% |
| **95.0%** | Observed Coverage | **71.49%** | **75.57%** | **FAILS (Under-coverage)** |
| | Interval Full Width | 2.758 m | 2.659 m | Wide |
| | 95% Wilson CI | [65.21%, 77.04%] | [69.50%, 80.76%] | Upper bound below 95% |

---

### Rain Intensity Breakdown (Horizon +1h)

The resampled test set contains genuine precipitation events across all severity categories:

| Intensity Class | Range (mm/h) | Test Sample Count | MF-1 POD | Persistence POD |
|---|---|---:|---:|---:|
| **Dry** | $< 0.1$ | 1,747 | 2.23% False Alarm | 13.97% False Alarm |
| **Trace** | $0.1$ to $0.5$ | 320 | 10.00% | 59.38% |
| **Light** | $0.5$ to $2.5$ | 437 | 15.56% | 84.44% |
| **Moderate** | $2.5$ to $7.5$ | 204 | 16.18% | 88.73% |
| **Heavy** | $> 7.5$ | 112 | 13.39% | 83.93% |

---

## Governance & Operational Recommendation

> [!CAUTION]
> **OPERATIONAL RECOMMENDATION: DO NOT DEPLOY FOR AUTONOMOUS OPERATION.**
> **Current Classification: `RESEARCH_PROTOTYPE`**
> 
> 1. **Rain Forecast**: While MF-1 achieves high precision (79.14% at +1h), its recall (POD) is low (13.79%), resulting in an F1-score of 23.49% versus **77.55% for Persistence**.
> 2. **River Stage Forecast**: Both MF-1 (MAE 1.17m) and MF-2 (MAE 1.15m) significantly underperform Persistence (MAE 0.0146m). The models exhibit severe negative bias and cannot be used for flood warnings.
> 3. **Uncertainty Calibration**: Conformal intervals fail independent test coverage (nominal 80% gives 32–43%, nominal 90% gives 57–63%, nominal 95% gives 71–76%) due to non-exchangeable distribution shifts between calibration and test periods.
