# KloudTrack Continuous-Time Weather & Hydrological Model (CfC/LNN)

This module implements continuous-time recurrent neural network architectures for weather telemetry and hydrological stage forecasting using **Closed-form Continuous-time (CfC) / Liquid Neural Networks (LNNs)**.

> [!IMPORTANT]
> **Status: Research Prototype**
> The models and scorecards in this module are research prototypes developed for experimental evaluation. They are **not** certified for autonomous flood warning or emergency response decisions. Hydrological validation is currently single-gauge (Calumpit WLMS, station `O3z0j5bG`). See [MODEL_REGISTRY.md](file:///c:/Ben%20File/beta-citizen-prediction/prediction-model/MODEL_REGISTRY.md) and [prediction-model-audit.md](file:///c:/Ben%20File/beta-citizen-prediction/prediction-model-audit.md) for detailed evaluation criteria and known limitations.

---

## Model Families

To resolve historical conflation between different experimental implementations, the codebase explicitly separates three distinct model families. For detailed specifications, schemas, and governance, see [MODEL_REGISTRY.md](file:///c:/Ben%20File/beta-citizen-prediction/prediction-model/MODEL_REGISTRY.md).

| Model ID | Name | Architecture | Framework | Training Method | Primary Artifact |
|---|---|---|---|---|---|
| **MF-1** | `WeatherWaterLNN` | Continuous-Time CfC Recurrent Neural Network | PyTorch | Multi-task gradient descent (AdamW) | `lnn_weather_water.pt` |
| **MF-2** | `ContinuousLNNCell` | Analytical ODE Closed-Form Continuous Cell | Pure Python (Zero-dependency) | Chronologically split numerical updates | `lnn_trained_weights.json` |
| **MF-3** | `StationAdaptivePINN` | Diurnal/Heuristic physics-coupled engine | Pure Python | Online parameter adaptation & diurnal priors | Configs in `config/` |

---

## Key Capabilities & Formulation

- **Continuous-Time Recurrence**: Rather than assuming fixed 1-hour intervals, the recurrent hidden state evolves according to an analytical closed-form solution to a continuous-time ordinary differential equation (ODE):
  $$\frac{dh}{dt} = -\left[\frac{1}{\tau} + f(x, h)\right] h + A \cdot f(x, h)$$
  conditioned on the actual measured elapsed time $\Delta t$ between telemetry readings.
- **Future Forecasting Targets**: The model is trained to predict future conditions at horizon $t_0 + h$ (where $h \in \{1, 3, 6, 12, 24\}$ hours) given a 24-hour historical observation window preceding forecast origin $t_0$, avoiding contemporaneous target leakage.
- **Chronological Split**: Data is partitioned strictly by timestamp (train: first 60%–80%, validation: subsequent 20%, test: final held-out period) per station before windowing to ensure realistic temporal generalization.
- **Real River Gauge Target Matching**: Water level training and evaluation targets are joined from real gauge telemetry (`water_level_telemetry.csv`, Calumpit WLMS) to the nearest weather station (`3nzr48bG`, Calumpit AWS). Samples without valid gauge observations are masked during loss calculation.
- **Empirical Conformal Uncertainty**: Uncertainty intervals are calculated via conformal prediction on held-out residuals at 80%, 90%, and 95% empirical coverage, replacing arbitrary static formulas.

---

## Telemetry Features & Schema

The model ingests 4 normalized weather channels along with elapsed continuous-time step $\Delta t$:
1. **Temperature ($^\circ$C)**: Ambient dry-bulb temperature
2. **Heat Index ($^\circ$C)**: Calculated apparent temperature
3. **Wind Speed (km/h)**: Surface anemometer velocity
4. **Atmospheric Pressure (hPa)**: Barometric surface pressure
5. **Elapsed Time $\Delta t$ (hours)**: Continuous delta between consecutive measurements

### Multi-Task Output Heads
1. **Chance of Rain**: Probability $[0.0, 1.0]$ via Sigmoid activation
2. **Precipitation Volume**: Expected accumulation in mm via ReLU activation ($\ge 0$)
3. **River Stage**: Projected water level in meters at target horizon $t_0 + h$

---

## Directory Structure

```plaintext
prediction-model/
├── MODEL_REGISTRY.md                 # Formal catalog and governance of model families
├── data/
│   ├── weather_telemetry.csv         # 756,156 historical telemetry rows across 16 stations
│   ├── water_level_telemetry.csv     # 43,883 river gauge telemetry rows (Calumpit WLMS)
│   ├── dataset_summary.json          # Telemetry manifest and station inventory
│   ├── lnn_trained_weights.json      # Model Family 2 trained weights
│   ├── lnn_weather_water.pt          # Model Family 1 PyTorch checkpoint with manifest
│   └── validation_scorecard.json     # Comprehensive validation metrics & conformal bands
├── docs/
│   ├── technical-whitepaper.md       # Technical paper & model documentation
│   ├── pagasa-validation-report.md   # 72h continuous benchmark against PAGASA ground truth
│   ├── model-card.md                 # Model card specification
│   ├── compliance-and-fair-usage.md  # Fair usage & community safety guidelines
│   └── system-architecture.md        # System architecture diagram
├── src/
│   ├── dataset.py                    # Chronological splitting, real gauge join, normalization
│   ├── model.py                      # PyTorch CfCCell and WeatherWaterLNN (Model Family 1)
│   ├── train.py                      # PyTorch training pipeline with reproducibility controls
│   ├── train_standalone.py           # Zero-dependency standalone trainer (Model Family 2)
│   ├── validate.py                   # Scorecard suite (FAR, CSI, Brier, conformal bands)
│   ├── inference.py                  # Fail-closed serverless predictor
│   ├── export_onnx.py                # Exports PyTorch weights to ONNX format
│   └── fetch_dataset.py              # Ingests telemetry data
└── README.md
```

---

## Quick Start & Verification

### 1. Verify Pipeline (Smoke Test)
Run a fast end-to-end smoke test (loads 1 mini-batch, executes forward pass, computes multi-task loss, runs backward pass, and verifies optimizer step):
```bash
python prediction-model/src/train.py --smoke
```

### 2. Train PyTorch CfC/LNN (Model Family 1)
Trains the deep continuous-time recurrent model with chronological train/validation splits and masked real gauge loss:
```bash
python prediction-model/src/train.py
```

### 3. Train Standalone Cell (Model Family 2)
Trains the lightweight zero-dependency continuous cell:
```bash
python prediction-model/src/train_standalone.py
```

### 4. Run Comprehensive Validation Suite
Evaluates both MF-1 (PyTorch) and MF-2 (Standalone) models against Persistence and Climatology baselines on canonical future-window test sequences, computing out-of-sample conformal test coverage:
```bash
python prediction-model/src/validate.py
```

### 5. Run Fail-Closed Inference
Produces lead-horizon predictions with checkpoint manifest validation and fail-closed safety:
```bash
python prediction-model/src/inference.py
```
