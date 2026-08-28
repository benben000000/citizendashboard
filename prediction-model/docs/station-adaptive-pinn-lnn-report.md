# KloudTrack Per-Station Adaptive PINN-LNN Calibration Report

*Execution Date: August 28, 2026 at 04:47 PM PST*
*Framework: Per-Station Adaptive Physics-Informed Liquid Neural Network (PINN-LNN)*
*Scope: All 23 Field Telemetry Stations in Central Luzon & Bataan Peninsula*
*Micro-Resolutions: 60 Continuous-Time ODE Steps per Station (1,380 Inferences Total)*

---

## 📊 Complete 23-Station Calibration Scorecard

| Station ID | Station Name | Microclimate Type | Elevation | Baseline Water | Peak Forecast | Temp MAE | Speed (Latency) | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `KT-6CBD47DC5194` | **Old Cabcaben Pier - Bataan** | `COASTAL_MARINE` | 4.0 m | 1.85 m | **5.157 m** | **1.57 °C** | **34.39 μs** | CALIBRATED & ACTIVE ✅ |
| `KT-CC380371FE68` | **Dinalupihan** | `LOWLAND_VALLEY` | 28.0 m | 2.4 m | **5.678 m** | **1.48 °C** | **37.59 μs** | CALIBRATED & ACTIVE ✅ |
| `KT-B850AD182EC8` | **Dona Maria** | `URBAN_PLAIN` | 16.0 m | 2.1 m | **5.301 m** | **1.49 °C** | **38.29 μs** | CALIBRATED & ACTIVE ✅ |
| `KT-A86039DC5194` | **Pag Asa Orani** | `COASTAL_PLAIN` | 12.0 m | 2.3 m | **5.887 m** | **1.73 °C** | **48.07 μs** | CALIBRATED & ACTIVE ✅ |
| `KT-8CEE47DC5194` | **1Bataan Command Center** | `REGIONAL_HUB` | 22.0 m | 2.0 m | **5.105 m** | **1.45 °C** | **38.24 μs** | CALIBRATED & ACTIVE ✅ |
| `KT-E0B89EF7A608` | **General Natividad** | `OROGRAPHIC_FOOTHILL` | 75.0 m | 3.1 m | **8.334 m** | **1.5 °C** | **34.32 μs** | CALIBRATED & ACTIVE ✅ |
| `KT-4049D3215788` | **Calumpit WLMS** | `RIVER_CONFLUENCE` | 6.0 m | 3.44 m | **6.562 m** | **1.58 °C** | **43.48 μs** | CALIBRATED & ACTIVE ✅ |
| `KT-4C31325C7BCC` | **Calumpit AWS** | `RIVER_BASIN` | 7.0 m | 3.42 m | **7.068 m** | **1.6 °C** | **34.77 μs** | CALIBRATED & ACTIVE ✅ |
| `KT-245EAD182EC8` | **Bongabon** | `OROGRAPHIC_FOOTHILL` | 92.0 m | 2.8 m | **8.391 m** | **1.88 °C** | **37.71 μs** | CALIBRATED & ACTIVE ✅ |
| `KT-3CCCAC182EC8` | **Pag-Asa Bagac** | `COASTAL_MARINE` | 15.0 m | 1.95 m | **5.283 m** | **1.73 °C** | **32.31 μs** | CALIBRATED & ACTIVE ✅ |
| `KT-D032325C7BCC` | **Población Mariveles** | `DEEP_HARBOR_COAST` | 8.0 m | 1.7 m | **4.89 m** | **1.57 °C** | **31.02 μs** | CALIBRATED & ACTIVE ✅ |
| `KT-D831325C7BCC` | **Abucay AWS** | `COASTAL_PLAIN` | 14.0 m | 2.2 m | **5.362 m** | **1.37 °C** | **33.1 μs** | CALIBRATED & ACTIVE ✅ |
| `KT-A80A1B29E748` | **Avida Asten AWS** | `URBAN_MICROCLIMATE` | 18.0 m | 1.5 m | **4.468 m** | **0.97 °C** | **38.25 μs** | CALIBRATED & ACTIVE ✅ |
| `KT-B82DB21C0610` | **San Jose City** | `CENTRAL_PLAIN` | 85.0 m | 2.9 m | **7.704 m** | **2.14 °C** | **31.5 μs** | CALIBRATED & ACTIVE ✅ |
| `KT-5C74AC182EC8` | **San Luis AWS** | `WETLAND_BASIN` | 10.0 m | 3.25 m | **6.478 m** | **1.73 °C** | **30.33 μs** | CALIBRATED & ACTIVE ✅ |
| `KT-20FCA4182EC8` | **Lazatin AWS** | `CENTRAL_PLAIN` | 20.0 m | 2.5 m | **6.281 m** | **1.24 °C** | **36.79 μs** | CALIBRATED & ACTIVE ✅ |
| `KT-184AAD182EC8` | **Baretto AWS** | `COASTAL_BAY` | 5.0 m | 1.8 m | **5.031 m** | **1.58 °C** | **31.99 μs** | CALIBRATED & ACTIVE ✅ |
| `KT-EC4FAD182EC8` | **Old Cabalan AWS** | `MOUNTAIN_PASS` | 110.0 m | 2.2 m | **8.277 m** | **2.08 °C** | **30.93 μs** | CALIBRATED & ACTIVE ✅ |
| `KT-183017F7A608` | **Sabang Morong AWS** | `COASTAL_MARINE` | 6.0 m | 1.9 m | **5.024 m** | **1.79 °C** | **37.2 μs** | CALIBRATED & ACTIVE ✅ |
| `KT-94AD8332A7B0` | **Wawa Limay AWS** | `COASTAL_ESTUARY` | 4.0 m | 2.05 m | **6.514 m** | **1.88 °C** | **40.64 μs** | CALIBRATED & ACTIVE ✅ |
| `KT-BC25B61815AC` | **Alasas AWS** | `CENTRAL_PLAIN` | 15.0 m | 2.6 m | **5.916 m** | **1.48 °C** | **34.17 μs** | CALIBRATED & ACTIVE ✅ |
| `KT-3C50AD182EC8` | **Sapang Buho AWS** | `RIVER_WATERSHED` | 60.0 m | 3.0 m | **6.727 m** | **1.49 °C** | **31.62 μs** | CALIBRATED & ACTIVE ✅ |
| `KT-8050AD182EC8` | **Popolon AWS** | `RIVER_WATERSHED` | 68.0 m | 3.05 m | **7.663 m** | **1.93 °C** | **31.91 μs** | CALIBRATED & ACTIVE ✅ |

---

## 🔬 Microclimate Categorization & Physical Coupling

1. **Coastal Marine Stations (e.g. Old Cabcaben Pier, Mariveles, Sabang Morong)**:
   - Embedded with semi-diurnal tidal damping and high boundary layer humidity (RH > 82%).
2. **River Basin & Confluence Gauges (e.g. Calumpit WLMS, San Luis, Sapang Buho)**:
   - Governed by river catchment continuity: d(WL)/dt = Qin - Qout with flood wave lag tau = 8.0h.
3. **Orographic Foothills (e.g. Bongabon, General Natividad, Old Cabalan Pass)**:
   - Enhanced convective updraft multipliers (1.35x) and low-altitude LCL saturation triggers (LCL < 400m).
4. **Urban & Central Plains (e.g. 1Bataan Command Center, San Jose City, Avida Asten)**:
   - Fast thermal dissipation time constants and hypsometric barometric compensation.

---

## 📁 Generated Data Artifacts

- **Per-Station Profiles JSON**: [`station_pinn_profiles.json`](file:///C:/Ben File/beta-citizen-prediction/prediction-model/data/station_pinn_profiles.json)
- **Minute-by-Minute 1,380 Forecasts CSV**: [`station_adaptive_minute_forecasts.csv`](file:///C:/Ben File/beta-citizen-prediction/prediction-model/data/station_adaptive_minute_forecasts.csv)
