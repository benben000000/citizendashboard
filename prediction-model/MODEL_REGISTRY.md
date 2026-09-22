# Prediction Model Registry

This document formally distinguishes and versions the three distinct model families present in the `prediction-model/` directory, resolving model conflation and establishing strict governance per the system audits (`prediction-model-audit.md` and `prediction-model-audit-followup.md`).

---

## Registry Overview

| Model ID | Name | Architecture | Framework | Training Method | Current Status | Primary Artifact |
|---|---|---|---|---|---|---|
| **MF-1** | `WeatherWaterLNN` | Continuous-Time CfC Recurrent Neural Network | PyTorch | Multi-task gradient descent (AdamW) | Research Prototype | `lnn_weather_water.pt` |
| **MF-2** | `ContinuousLNNCell` | Analytical ODE Closed-Form Continuous Cell | Pure Python / Zero-dependency | Chronological balanced gradient updates | Research Prototype | `lnn_trained_weights.json` |
| **MF-3** | `StationAdaptivePINN` | Diurnal/Heuristic physics-coupled engine | Pure Python | Online parameter adaptation & diurnal priors | Experimental Baseline | Configs in `config/` |

---

## Canonical Evaluation Protocol (Audit Alignment)

All reported benchmarks and scorecards in `validation_scorecard.json` adhere to the canonical evaluation protocol:
1. **Data Quality Quarantine**:
   - Date outliers outside valid collection bounds ($2025 \le \text{year} \le 2027$, e.g., the year 2069 sensor anomaly) are quarantined.
   - Physical sensor spikes failing physical limits (temperature 10–50°C, heat index 10–70°C, wind 0–180 km/h, pressure 900–1050 hPa) are rejected prior to window construction.
2. **Train-Fitted Normalization**:
   - Feature means and standard deviations are fitted strictly on the chronological training split ($t \le t_{\text{train\_end}}$) and versioned with model artifacts.
3. **Future-Forecasting Sequence Windows**:
   - Input is a 24-hour sequence window $[t_{-23}, \dots, t_0]$ preceding forecast origin $t_0$.
   - Target is the observed condition at future lead time $t_0 + h$ (where $h \in \{1, 3, 6, 12, 24\}$ hours).
4. **Strict State Isolation**:
   - Recurrent hidden state is re-initialized to $h = 0$ for every sequence window, guaranteeing zero temporal or cross-station leakage.
5. **Operational Baseline Benchmarks**:
   - **Persistence Baseline**: Predicts the condition observed at forecast origin $t_0$ ($\hat{y}_{t_0+h} = y_{t_0}$).
   - **Climatology Baseline**: Predicts the long-term training distribution mean ($\hat{y}_{t_0+h} = \bar{y}_{\text{train}}$).
6. **Split Conformal Uncertainty**:
   - Conformal residual quantiles ($q_{0.80}, q_{0.90}, q_{0.95}$) are fitted on the **calibration set** (`split="val"`).
   - Empirical coverage and interval widths are independently evaluated on the **untouched test set** (`split="test"`).

---

## Model Family 1: `WeatherWaterLNN` (PyTorch CfC/LNN)

### Description
The canonical deep learning model in the repository. Implements a Closed-form Continuous-time (CfC) neural ODE recurrent cell in PyTorch, processing irregularly sampled telemetry with explicit elapsed-time delta ($\Delta t$) conditioning.

### Source Files
- Architecture: [`src/model.py`](file:///c:/Ben%20File/beta-citizen-prediction/prediction-model/src/model.py)
- Dataset Pipeline: [`src/dataset.py`](file:///c:/Ben%20File/beta-citizen-prediction/prediction-model/src/dataset.py)
- Training Loop: [`src/train.py`](file:///c:/Ben%20File/beta-citizen-prediction/prediction-model/src/train.py)
- Operational & Scenario Inference: [`src/inference.py`](file:///c:/Ben%20File/beta-citizen-prediction/prediction-model/src/inference.py)
- Export Pipeline: [`src/export_onnx.py`](file:///c:/Ben%20File/beta-citizen-prediction/prediction-model/src/export_onnx.py)

### Input Schema
- **Sequence Length**: 24 historical observation hours ($t_{-23}, \dots, t_0$)
- **Features ($X \in \mathbb{R}^{24 \times 4}$)**:
  1. `temperature`: Ambient temperature in Celsius (train-fitted normalization)
  2. `heat_index`: Apparent temperature in Celsius (train-fitted normalization)
  3. `wind_speed`: Wind velocity in km/h (train-fitted normalization)
  4. `pressure`: Barometric pressure in hPa (train-fitted normalization)
- **Time Delta ($\Delta t \in \mathbb{R}^{24 \times 1}$)**: Elapsed continuous hours between consecutive telemetry observations.

### Target Schema (Future Forecasting)
- Target is evaluated at forecast horizon $t_0 + h$ ($h \in \{1, 3, 6, 12, 24\}$ hours):
  1. `rain_prob`: Binary rain classification ($1.0$ if precipitation $> 0.1$ mm/h at $t_0 + h$, else $0.0$)
  2. `precip_mm`: Precipitation volume in mm at $t_0 + h$ (ReLU activated, $\ge 0$)
  3. `water_level`: River stage in meters at $t_0 + h$ from matched gauge observation (masked when unobserved)

### Loss Function
$$\mathcal{L}_{\text{total}} = \lambda_{\text{rain}} \mathcal{L}_{\text{BCE}} + \lambda_{\text{precip}} \mathcal{L}_{\text{MSE}} + \lambda_{\text{water}} \mathcal{L}_{\text{Huber}} \odot M_{\text{gauge}}$$
Where $M_{\text{gauge}}$ is a binary mask: $1$ if observed water gauge reading exists, $0$ if missing.

### Artifacts & Reproducibility
- Checkpoint: `prediction-model/data/lnn_weather_water.pt`
- Includes reproducibility manifest with: random seed, dataset SHA256 hashes, train-fitted normalization stats, architecture parameters, environment versions, and untouched test split metrics.

---

## Model Family 2: `ContinuousLNNCell` (Standalone Analytical ODE)

### Description
A zero-dependency, pure Python implementation of the continuous-time CfC recurrence. Does not require PyTorch or GPU acceleration; designed for edge / constrained deployment and lightweight benchmarking.

### Source Files
- Trainer & Model: [`src/train_standalone.py`](file:///c:/Ben%20File/beta-citizen-prediction/prediction-model/src/train_standalone.py)
- Comprehensive Validator: [`src/validate.py`](file:///c:/Ben%20File/beta-citizen-prediction/prediction-model/src/validate.py)
- Weights JSON: [`data/lnn_trained_weights.json`](file:///c:/Ben%20File/beta-citizen-prediction/prediction-model/data/lnn_trained_weights.json)
- Output Scorecard: [`data/validation_scorecard.json`](file:///c:/Ben%20File/beta-citizen-prediction/prediction-model/data/validation_scorecard.json)

### Split & Evaluation Strategy
- **Split Strategy**: Chronological split (first 80% by timestamp for training, last 20% for validation/testing).
- **Rain Metrics Evaluated**:
  - Probability of Detection (Recall / POD)
  - Precision
  - F1-Score
  - False Alarm Ratio (FAR)
  - Critical Success Index (CSI)
  - Brier Score
  - Threshold sensitivity analysis ($0.3$ to $0.7$)
  - Detection rate broken down by precipitation intensity class (trace, light, moderate, heavy)
  - Per-station breakdown across all 16 weather stations
- **Water Metrics**:
  - Evaluated exclusively against **real river gauge observations** from Calumpit WLMS (`O3z0j5bG`) matched to Calumpit AWS weather station (`3nzr48bG`).
  - Benchmarked directly against Persistence and Climatology.
- **Uncertainty Bands**:
  - Conformal prediction intervals ($q_{0.80}, q_{0.90}, q_{0.95}$) calibrated on `split="val"`, with empirical coverage independently evaluated on `split="test"`.

---

## Model Family 3: `StationAdaptivePINN` (Heuristic Physical Baseline)

### Description
A station-specific heuristic engine combining diurnal temperature profiles, empirical physical decay curves, and online parameter tuning. This engine was historically called "PINN" in documentation, but does not train via physics residuals backpropagated through a differential equation solver.

### Source Files
- Implementation: `src/station_adaptive_pinn_lnn.py`
- Configuration: `config/station_diurnal_profiles.json`

### Classification & Usage
- **Status**: Experimental heuristic baseline.
- Not linked to canonical `WeatherWaterLNN` checkpoints or inference endpoints.
- Physics terminology ("PINN") in this family refers to hand-parameterized empirical physical bounds, not gradient-optimized physics loss.

---

## Scorecard Attribution & Governance

- Any metric published in `validation_scorecard.json` **must** document the exact Model Family (`MF-1`, `MF-2`, or `MF-3`) that produced it and the comparative baselines.
- **MF-1** and **MF-2** are research prototypes and must not be designated "PRODUCTION READY" or used as an autonomous flood warning system without domain expert calibration.
