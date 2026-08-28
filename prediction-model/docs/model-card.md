# Model Card: Liquid Neural Network (LNN / CfC) for Hydrological & Rain Forecasting

## 1. Overview
- **Model Name**: KloudTrack Continuous-Time LNN (`WeatherWaterLNN`)
- **Version**: 1.0.0
- **Model Type**: Closed-form Continuous-time (CfC) Neural ODE Network
- **Architecture**: 32-hidden units CfC continuous dynamical cell with dual prediction heads.
- **Inference Runtime**: Serverless CPU execution (< 3ms per trajectory inference).

---

## 2. Input Telemetry Features
The model ingests 4 primary meteorological telemetry channels captured at weather stations:

| Channel | Physical Unit | Description | Normalization |
| :--- | :--- | :--- | :--- |
| **`temperature`** | °C | Ambient dry-bulb air temperature | $\mu=28.5, \sigma=4.5$ |
| **`heat_index`** | °C | Apparent thermal feeling combining humidity & heat | $\mu=33.0, \sigma=6.5$ |
| **`wind_speed`** | km/h | Anemometer atmospheric airflow velocity | $\mu=10.0, \sigma=8.0$ |
| **`pressure`** | hPa | Barometric atmospheric surface pressure | $\mu=1008.0, \sigma=6.0$ |
| **$\Delta t$** | hours | Elapsed continuous time delta between measurements | Dynamic continuous scaling |

---

## 3. Predicted Targets

1. **Chance of Rain & Precipitation**:
   - `chance_of_rain_pct`: Probability $[0, 100\%]$ of rain event occurring within target horizon.
   - `precipitation_mm`: Expected precipitation volume in millimeters ($mm$).
2. **Hydrological River Water Level**:
   - `predicted_water_level_m`: Predicted river height in meters ($m$).
   - `flood_risk_level`: Multi-tier alert classification (**Safe**, **Advisory / Alert**, **Warning**, **Critical**).

---

## 4. Mathematical Formulation: Closed-form Continuous-time (CfC)
Rather than discrete recurrence (LSTM/GRU), the model state $h(t)$ evolves according to:

$$\frac{dh(t)}{dt} = - \left( \frac{1}{\tau} + f(x(t), h(t)) \right) h(t) + A \cdot f(x(t), h(t))$$

The closed-form analytical solution computed by `CfCCell` at time $t + \Delta t$:

$$h(t + \Delta t) \approx \exp\left(-\Delta t \cdot w_\tau(x, h)\right) \odot h(t) + \left(1 - \exp\left(-\Delta t \cdot w_\tau(x, h)\right)\right) \odot \sigma(W_g [x, h]) \odot \tanh(W_s [x, h])$$

---

## 5. Loss Function
The model trains using a combined multi-task objective:

$$\mathcal{L}_{\text{total}} = \lambda_{\text{rain}} \mathcal{L}_{\text{BCE}}(\hat{p}, p) + \lambda_{\text{precip}} \mathcal{L}_{\text{MSE}}(\hat{v}, v) + \lambda_{\text{water}} \mathcal{L}_{\text{Huber}}(\hat{H}, H)$$

---

## 6. Responsible AI & Fair Usage
- **Decision Support**: This model provides predictive guidance for citizens and local disaster risk management authorities (LDRRMO). Official emergency evacuation orders should always be confirmed with PAGASA and local government command centers.
- **Fair Access**: Lightweight CPU architecture ensures zero expensive GPU infrastructure is required to deliver sub-second public alerts.
