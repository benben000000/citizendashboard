# System Architecture — LNN Prediction Model

## 1. Overview

The prediction model pipeline transforms raw time-series telemetry from KloudTrack weather and hydrological monitoring stations into accurate, continuous-time water level forecasts using **Liquid Neural Networks (LNNs)**.

---

## 2. High-Level Flow

```text
[ Weather & River Stations ] (Rainfall, Water Level, Temp, Pressure, Humidity)
             │
             ▼
[ Telemetry Ingestion Layer ] (KloudTrack API / Message Queue)
             │
             ▼
[ Preprocessing & Feature Pipeline ] (Missing value handling, Resampling, Feature scaling)
             │
             ▼
[ Liquid Neural Network (LNN) Engine ] (Continuous-time differential ODE solvers / CfC)
             │
             ▼
[ Model Serving & Alerting API ] (Forecast endpoints, Early flood threshold warnings)
             │
             ▼
[ KloudTrack Frontend / Clients ] (Dashboards, Map visualizations, SMS/Webhook Alerts)
```

---

## 3. Core Components

### 3.1 Telemetry Ingestion
- Ingests multi-sensor metrics (Water level, precipitation rates, temperature, humidity, atmospheric pressure).
- Buffers incoming streaming data and handles asynchronous sensor delivery.

### 3.2 Preprocessing & Feature Engineering
- **Irregular Time-Step Handling**: Dynamic delta-time ($\Delta t$) computation for continuous-time input.
- **Hydrological Lag Features**: Cumulative rainfall windows (1h, 3h, 6h, 24h), rate of rise/fall ($\Delta h/\Delta t$).
- **Spatial Topology**: Upstream station metrics linked to downstream water level sensors.

### 3.3 LNN Forecasting Core
- Built with continuous-time Liquid Neural Network / Closed-form Continuous-time (CfC) layers.
- Outputs multi-horizon forecasts:
  - **Short-term**: 1 to 6 hours (tactical alert window)
  - **Medium-term**: 12 to 72 hours (strategic disaster preparedness)

### 3.4 Inference & Serving Layer
- Lightweight model inference engine with caching.
- Exposes REST / gRPC endpoints for client applications and alert dispatchers.
