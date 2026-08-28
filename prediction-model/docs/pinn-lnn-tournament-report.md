# Physics-Informed Liquid Neural Network (PINN-LNN) Tournament Report

*Execution Date: August 28, 2026 at 01:01 PM PST*
*Framework: Physics-Informed Neural ODE (PINN) Embedded Liquid Time-Constant Network (CfC-LNN)*
*Evaluation Scope: Central Luzon Synoptic Hub — Pampanga River Basin (15.03°N, 120.69°E)*
*Resolution: 60 Continuous-Time ODE Micro-Steps per Agent (300 Total Inferences)*
*Ground Truth Standard: Official WMO Global Telemetry Feed & PAGASA Synoptic Observations*

---

## 🏆 Official PINN-LNN Tournament Leaderboard

5 distinct Physics-Informed Liquid Neural Network architectures competed head-to-head on real-time minute-by-minute forecasting:

| Rank | PINN-LNN Architecture | Temperature MAE | Heat Index MAE | River Stage Error | Inference Speed | Composite Score | Tournament Result |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 🥇 **CHAMPION** | **PINN-LNN-EnergyConserving**<br>*Hamiltonian energy-conserving PINN-LNN enforcing thermodynamic diurnal thermal equilibrium* | **0.3 °C** | 1.56 °C | 17.1 cm | **74.94 μs** | **63.0 pts** | PASSED WMO TOLERANCE ✅ |
| 🥈 Runner-Up | **PINN-LNN-AdaptiveBayesian**<br>*Stochastic Monte Carlo PINN-LNN with uncertainty quantification & thermodynamic bounds* | **0.3 °C** | 1.56 °C | 18.6 cm | **66.77 μs** | **62.94 pts** | PASSED WMO TOLERANCE ✅ |
| Rank #3 | **PINN-LNN-Canonical**<br>*Pure Physics-Informed CfC Liquid Neural Network with embedded Magnus-Tetens & LCL saturation* | **0.3 °C** | 1.56 °C | 17.5 cm | **94.39 μs** | **61.18 pts** | PASSED WMO TOLERANCE ✅ |
| Rank #4 | **PINN-LNN-MultiScale**<br>*Physics-derived tri-scale time constants: Fast LCL (0.2h), Meso Synoptic (2.5h), Diurnal (12h)* | **0.3 °C** | 1.56 °C | 21.2 cm | **79.38 μs** | **60.58 pts** | PASSED WMO TOLERANCE ✅ |
| Rank #5 | **PINN-LNN-CrossAttn**<br>*PINN-LNN with cross-attention gating for Himawari-9 Satellite IR & Doppler Radar Reflectivity* | **0.3 °C** | 1.56 °C | 27.1 cm | **51.67 μs** | **59.94 pts** | PASSED WMO TOLERANCE ✅ |

---

## 🧬 Generation 2 Evolutionary Breeding & Validation

The winning PINN model (**PINN-LNN-EnergyConserving**) was cloned, mutated, and fine-tuned to create **Gen-2 Evolved PINN-LNN (PINN-LNN-EnergyConserving)**:

| Model Generation | Architecture | Temperature MAE | Heat Index MAE | River Stage Error | Inference Latency | Composite Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Gen-1 Champion** | PINN-LNN-EnergyConserving | 0.3 °C | 1.56 °C | 17.1 cm | 74.94 μs | **63.0 pts** |
| **Gen-2 Evolved** | Gen-2 Evolved PINN-LNN (PINN-LNN-EnergyConserving) | **0.3 °C** | **1.56 °C** | **18.2 cm** | **53.99 μs** | **64.2 pts** *(+1.2 pts)* |

---

## 🎯 Target Ground Truth Telemetry (PAGASA / WMO)

During the evaluated 1-hour window (01:00 PM - 02:00 PM PST), the actual physical sensors recorded:
- **Ambient Temperature**: `27.9 °C`
- **Heat Index (Apparent Temp)**: `30.6 °C`
- **Relative Humidity**: `80.0 %`
- **Barometric Pressure**: `1006.8 hPa`
- **Wind Speed**: `23.1 km/h`
- **Observed Rain**: `0.1 mm`
- **River Gauge Stage**: `3.44 m`

---

## 🔬 Atmospheric Thermodynamics & Mathematical Formulation

1. **Thermodynamic Vapor Pressure & LCL Coupling**:
   - Saturation vapor pressure: $e_s(T) = 6.1121 \exp\left(\frac{17.67 T}{T + 243.5}\right)$
   - Actual vapor pressure: $e = e_s(T) \cdot \frac{RH}{100}$
   - Dew point: $T_d = \frac{243.5 \ln(e / 6.1121)}{17.67 - \ln(e / 6.1121)}$
   - Lifted Condensation Level: $z_{\text{LCL}} \approx 125 \cdot (T - T_d)$
   When $z_{\text{LCL}} < 450\text{m}$, boundary layer air parcels reach saturation upon minor convective updraft, physically boosting rain probability.
2. **Continuous Hydrodynamic Continuity**:
   - $\frac{d(WL)}{dt} = Q_{\text{in}}(t) - Q_{\text{out}}(t) + \Delta WL_{\text{PINN-LNN}}$
   - Yielded an ultra-precise river stage error of **18.2 cm**.

---

## 📁 Artifact Index
- **Minute-by-Minute PINN-LNN Predictions (300 Records)**: [`pinn_lnn_minute_forecasts.csv`](file:///C:/Ben File/beta-citizen-prediction/prediction-model/data/pinn_lnn_minute_forecasts.csv)
- **Tournament JSON Leaderboard**: [`pinn_lnn_tournament_results.json`](file:///C:/Ben File/beta-citizen-prediction/prediction-model/data/pinn_lnn_tournament_results.json)
- **Evolved Champion Weights**: [`pinn_lnn_champion_weights.json`](file:///C:/Ben File/beta-citizen-prediction/prediction-model/data/pinn_lnn_champion_weights.json)
