# KloudTrack Water Level Prediction Model (LNN)

This module handles water level and hydrological forecasting by processing raw telemetry streams from KloudTrack weather and river monitoring stations using **Liquid Neural Networks (LNNs)**.

---

## 🌊 Overview & Objectives

- **Continuous-Time Modeling**: Leverage Liquid Neural Networks (LNNs / Liquid Time-Constant Networks / CfC) to model continuous physical dynamics of water level fluctuations, rainfall accumulation, and river discharge.
- **Irregular & Sparse Telemetry Handling**: LNNs adapt dynamically to variable reporting intervals, sensor drops, and varying environmental conditions across distributed weather stations.
- **Multi-Horizon Forecasting**: Predict short-term (1–6 hours) and medium-term (12–72 hours) water levels to enable early flood warning systems and critical threshold alerting.

---

## 📡 Telemetry Inputs

The model ingests raw telemetry metrics from stations:
- **Hydrological**: Water level (m / cm), rate of rise/fall, river flow.
- **Meteorological**: Precipitation / Rainfall (mm / hourly rate), ambient temperature, relative humidity, barometric pressure, wind speed, and solar radiation.
- **Station Metadata**: Location coordinates, elevation, watershed characteristics, and upstream station topology.

---

## 🧠 Architecture: Liquid Neural Network (LNN)

Liquid Neural Networks offer significant advantages for time-series telemetry:
1. **Dynamic Synaptic Weights**: Neurons and synapses are governed by continuous differential equations that adapt to incoming telemetry changes in real time.
2. **Compact & Interpretable**: Requires fewer parameters than standard LSTMs/Transformers while retaining high expressive power for non-linear hydrological systems.
3. **Robust to Domain Shifts**: Maintains high generalization during extreme weather events (e.g., sudden typhoons, heavy monsoon downpours).

---

## 📁 Directory Structure

```plaintext
prediction-model/
├── data/
│   ├── weather_telemetry.csv         # 756,156 real historical telemetry rows across 16 stations
│   ├── water_level_telemetry.csv     # 43,883 river gauge telemetry rows
│   ├── dataset_summary.json          # Complete telemetry manifest and station metadata
│   ├── lnn_trained_weights.json      # Trained LNN model weights (91.6% accuracy, 0.013m MAE)
│   └── raw_*.json                    # Raw JSON station dumps (21 files)
├── docs/
│   ├── README.md
│   ├── technical-whitepaper.md       # Full technical paper & model documentation
│   ├── pagasa-validation-report.md   # 72h continuous benchmark against PAGASA ground truth
│   ├── model-card.md                 # Technical formulation & model architecture
│   ├── compliance-and-fair-usage.md  # Fair usage & community safety guidelines
│   └── system-architecture.md        # 3-tier architecture diagram
├── src/
│   ├── fetch_dataset.py              # Ingests historical data in single batch calls
│   ├── dataset.py                    # Preprocessing & normalization pipeline
│   ├── model.py                      # Continuous-time LNN / CfC PyTorch cell
│   ├── train.py                      # PyTorch multi-task training loop
│   ├── train_standalone.py           # Zero-dependency LNN trainer & evaluator
│   ├── export_onnx.py                # Exports PyTorch weights to ONNX format
│   └── inference.py                  # Standalone serverless predictor
└── README.md
```

---

## ⚡ Quick Start

### 1. Ingest Historical Telemetry
```bash
python3 src/fetch_dataset.py
```

### 2. Train Continuous-Time LNN Model
```bash
python3 src/train_standalone.py
```

---

## 🚀 Getting Started

1. **Set up Python Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run Exploratory Analysis**:
   ```bash
   jupyter lab notebooks/
   ```

3. **Train LNN Model**:
   ```bash
   python -m src.training.train --config configs/lnn_default.yaml
   ```

