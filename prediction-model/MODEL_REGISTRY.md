# Prediction Model Registry

This document formally distinguishes and versions the three distinct model families present in the `prediction-model/` directory, resolving the model conflation identified in the system audit (`prediction-model-audit.md`).

---

## Registry Overview

| Model ID | Name | Architecture | Framework | Training Method | Current Status | Primary Artifact |
|---|---|---|---|---|---|---|
| **MF-1** | `WeatherWaterLNN` | Continuous-Time CfC Recurrent Neural Network | PyTorch | Multi-task gradient descent (AdamW) | Research Prototype | `lnn_weather_water.pt` |
| **MF-2** | `ContinuousLNNCell` | Analytical ODE Closed-Form Continuous Cell | Pure Python / Zero-dependency | Heuristic / Perceptron numerical updates | Research Prototype | `lnn_trained_weights.json` |
| **MF-3** | `StationAdaptivePINN` | Diurnal/Heuristic physics-coupled engine | Pure Python | Online parameter adaptation & heuristic physics | Experimental Baseline | Configs in `config/` |

---

## Model Family 1: `WeatherWaterLNN` (PyTorch CfC/LNN)

### Description
The canonical deep learning model in the repository. Implements a Closed-form Continuous-time (CfC) neural ODE recurrent cell in PyTorch, processing irregularly sampled telemetry with explicit elapsed-time delta ($\Delta t$) conditioning.

### Source Files
- Architecture: [`src/model.py`](file:///c:/Ben%20File/beta-citizen-prediction/prediction-model/src/model.py)
- Dataset Pipeline: [`src/dataset.py`](file:///c:/Ben%20File/beta-citizen-prediction/prediction-model/src/dataset.py)
- Training Loop: [`src/train.py`](file:///c:/Ben%20File/beta-citizen-prediction/prediction-model/src/train.py)
- Production Inference: [`src/inference.py`](file:///c:/Ben%20File/beta-citizen-prediction/prediction-model/src/inference.py)
- Export Pipeline: [`src/export_onnx.py`](file:///c:/Ben%20File/beta-citizen-prediction/prediction-model/src/export_onnx.py)

### Input Schema
- **Sequence Length**: 24 historical observation hours ($t_{-23}, \dots, t_0$)
- **Features ($X \in \mathbb{R}^{24 \times 4}$)**:
  1. `temperature`: Ambient temperature in Celsius (normalized by training mean/std)
  2. `heat_index`: Perceived heat index in Celsius (normalized)
  3. `wind_speed`: Wind speed in km/h (normalized)
  4. `pressure`: Barometric pressure in hPa (normalized)
- **Time Delta ($\Delta t \in \mathbb{R}^{24 \times 1}$)**: Elapsed hours between consecutive telemetry observations.

### Target Schema (Future Forecasting)
- Target is constructed at forecast horizon $t_0 + h$ (where $h \in \{1, 3, 6, 12, 24\}$ hours):
  1. `rain_prob`: Binary rain classification ($1.0$ if precipitation $> 0.1$ mm/h at $t_0 + h$, else $0.0$)
  2. `precip_mm`: Precipitation volume in mm at $t_0 + h$ (ReLU activated, $\ge 0$)
  3. `water_level`: River stage in meters at $t_0 + h$ from matched gauge observation (masked when unobserved)

### Loss Function
$$\mathcal{L}_{\text{total}} = \lambda_{\text{rain}} \mathcal{L}_{\text{BCE}} + \lambda_{\text{precip}} \mathcal{L}_{\text{MSE}} + \lambda_{\text{water}} \mathcal{L}_{\text{Huber}} \odot M_{\text{gauge}}$$
Where $M_{\text{gauge}}$ is a binary mask: $1$ if observed water gauge reading exists, $0$ if missing.

### Artifacts & Reproducibility
- Checkpoint: `prediction-model/data/lnn_weather_water.pt`
- Includes reproducibility manifest with: random seed, dataset SHA256 hashes, normalization parameters, architecture parameters, and environment versions.

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
- **Split Strategy**: Chronological split (first 80% by timestamp for training, last 20% for validation).
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
  - Synthetic convolution labels are discarded.
- **Uncertainty Bands**:
  - Empirical conformal prediction intervals at 80%, 90%, and 95% coverage computed from held-out residual quantiles.
  - Replaced arbitrary $\pm 0.08 \sqrt{h}$ heuristic.

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

- Any metric published in `validation_scorecard.json` **must** document the exact Model Family (`MF-1`, `MF-2`, or `MF-3`) that produced it.
- **MF-1** and **MF-2** are research prototypes and must not be designated "PRODUCTION READY" or used as an autonomous flood warning system without domain expert calibration.
