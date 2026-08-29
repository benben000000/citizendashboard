# KloudTrack Developer Architecture & Maintenance Guide

**Project:** Citizen Weather, Water Level & PINN-LNN Nowcast Dashboard  
**Architecture:** Next.js App Router (TypeScript) + In-Memory SWR Cache + Physics-Informed Liquid Neural Network (PINN-LNN) Engine  
**Version:** 2.4-Production  
**Last Updated:** August 2026  

---

## 1. System Architecture Overview

The system provides real-time hydrometeorological intelligence across 23 weather stations and 13 water level monitoring stations in Central Luzon and Bataan.

```
+-----------------------------------------------------------------------------------+
|                               CITIZEN DASHBOARD CLIENT                            |
|        /weather (Weather)    /water-level (Hydraulic)    /prediction (Nowcast)    |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|                             NEXT.JS API ROUTE LAYER                               |
|   /api/telemetry/dashboard    /api/water-level/dashboard    /api/prediction/...   |
+-----------------------------------------------------------------------------------+
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
+------------------------------------+          +------------------------------------+
|       TELEMETRY & HYDRAULIC        |          |         PINN-LNN PREDICTION        |
|          SERVICE LAYER             |          |            SERVICE LAYER           |
|  • In-Memory SWR Cache             |          |  • Closed-Form Continuous-Time LNN |
|  • Physics QC Bounds (3-Tier)      |          |  • Magnus-Tetens Psychrometrics    |
|  • Spatial IDW Gaussian Imputation |          |  • Saint-Venant 1D Hydrodynamics   |
+------------------------------------+          +------------------------------------+
                 │                                               │
                 └───────────────────────┬───────────────────────┘
                                         ▼
+-----------------------------------------------------------------------------------+
|                               DATA INGESTION BUS                                  |
|   1. Live Upstream REST: http://citizen.kloudtechsea.com/api                      |
|   2. AWS IoT Core MQTT mTLS: prediction-model/data/mqtt_live_predictions.json      |
+-----------------------------------------------------------------------------------+
```

---

## 2. Directory Structure & Key Files

```
beta-citizen-prediction/
├── docs/                                   # System documentation & specs
│   ├── DEVELOPER_MAINTENANCE_GUIDE.md      # This maintenance guide
│   ├── TECHNICAL_SPECIFICATIONS.md         # Technical and math specifications
│   ├── VALIDATION_SHEETS.md                # WMO / PAGASA compliance sheets
│   └── COMMERCIAL_RIGHTS_IP_CLEARANCE.md   # IP, attribution & fair-use clearance
├── src/
│   ├── app/                                # Next.js App Router routes & API endpoints
│   │   ├── [locale]/                       # Localized frontend pages (/weather, /water-level, /prediction)
│   │   └── api/                            # Backend JSON endpoints
│   │       ├── telemetry/dashboard/        # Batched weather telemetry endpoint
│   │       ├── water-level/dashboard/      # Batched water level endpoint
│   │       └── prediction/                 # LNN continuous nowcast endpoints
│   ├── components/features/                # Core UI features (Weather, Water Level, Prediction, Map)
│   ├── lib/
│   │   ├── config/cache.config.ts          # Cache TTL and HTTP header policies
│   │   ├── constants/stations.json         # Station registry & public IDs
│   │   ├── kloudtrack/client.ts            # Server-side HTTP client for Kloudtrack API
│   │   └── utils/
│   │       ├── cache.ts                    # In-Memory LRU Cache with SWR support
│   │       └── weather.ts                  # Weather condition parser & convective rain logic
│   ├── services/
│   │   ├── telemetry.service.ts            # Weather telemetry transformation & Spatial IDW
│   │   ├── water-level.service.ts          # Water level telemetry transformation & trajectory
│   │   └── prediction.service.ts           # PINN-LNN multi-horizon continuous nowcasting
│   └── types/                              # TypeScript interface definitions
└── prediction-model/                       # Python PINN-LNN Training & MQTT Daemon
    ├── data/                               # Segregated CSV logs and live MQTT state cache
    │   ├── raw_mqtt_telemetry.csv          # Raw uncleaned sensor readings
    │   ├── denoised_pinn_telemetry.csv     # Cleaned continuous-time physical state
    │   └── mqtt_live_predictions.json      # Segregated real-time broadcast cache
    └── src/
        ├── mqtt_pinn_live_streamer.py      # MQTT subscriber & PINN broadcaster daemon
        └── validate_wmo_pagasa_alignment.py# Automated validation audit suite
```

---

## 3. Data Processing & Fault Mitigation Pipeline

### 3.1 3-Tier Physics Quality Control (QC)
Sensors deployed in tropical environments can suffer from electrical faults, dead batteries, or ADC drift. The system filters incoming data through strict thermodynamic bounds before presenting it to users:

| Metric | Valid Physical Range | Anomaly Action |
| :--- | :---: | :--- |
| **Ambient Temperature** | $16.0^\circ\text{C} \le T \le 43.0^\circ\text{C}$ | Reject spikes ($> 43^\circ\text{C}$) or zero-dropouts ($0^\circ\text{C}$); trigger Spatial IDW. |
| **Relative Humidity** | $20.0\% \le \text{RH} \le 100.0\%$ | Clamp out-of-range floats; flag sensor saturation. |
| **Surface Barometric Pressure** | $970.0\text{ hPa} \le P \le 1030.0\text{ hPa}$ | Reject sensor drift ($< 970\text{ hPa}$ or $> 1030\text{ hPa}$); replace via spatial neighbors. |
| **Ultrasonic Water Stage** | $50.0\text{ cm} \le h \le 1200.0\text{ cm}$ | Denoise ultrasonic ripple jitter using continuous LNN ODE state decay. |

### 3.2 Haversine Gaussian Spatial Inverse Distance Weighting (IDW)
When a weather station is down, disconnected, or reporting absurd telemetry (e.g. Barretto AWS $108^\circ\text{C}$), `telemetry.service.ts` automatically reconstructs its probable local microclimate from surrounding active stations:

1. **Calculate Great-Circle Distance ($d_j$)** to all healthy stations:
   $$d_j = 2 R \arcsin\left(\sqrt{\sin^2\left(\frac{\Delta \phi}{2}\right) + \cos\phi_1 \cos\phi_2 \sin^2\left(\frac{\Delta \lambda}{2}\right)}\right)$$
   *(Memoized in `TelemetryService.distanceCache` for $O(1)$ constant-time lookup).*
2. **Compute Gaussian Spatial Weights ($w_j$)** with meso-scale radius $L = 25\text{ km}$:
   $$w_j = \exp\left(-\frac{d_j^2}{2 L^2}\right) + 0.001$$
3. **Impute Target State**:
   $$\hat{T} = \frac{\sum w_j T_j}{\sum w_j}, \quad \hat{\text{RH}} = \frac{\sum w_j \text{RH}_j}{\sum w_j}, \quad \hat{P} = \frac{\sum w_j P_j}{\sum w_j}, \quad \hat{R} = \frac{\sum w_j R_j}{\sum w_j}$$

---

## 4. Mechanical Rain Gauge Delay & Psychrometric Saturation Detection

Mechanical tipping bucket rain gauges require a physical water accumulation threshold ($0.2\text{ mm}$ per tip) to trigger a sensor event. During sudden tropical convective downpours, a 5–10 minute mechanical reporting lag can occur.

To ensure citizens receive immediate, accurate weather warnings:
- The system evaluates **thermodynamic psychrometric saturation**:
  When **$\text{RH} \ge 94\%$** and **$T \le 26.8^\circ\text{C}$** (evaporative cooling), the Lifted Condensation Level drops ($\text{LCL} \le 80\text{ m}$), indicating active cloud-to-ground precipitation.
- The system automatically triggers the convective rain status (**🌧️ Moderate/Heavy Rain**) and estimates the real-time precipitation rate without waiting for mechanical bucket delay.

---

## 5. Adding or Modifying Stations

To register a new weather or water level station:

1. **Update Station Registry (`src/lib/constants/stations.json`)**:
   ```json
   {
     "weather": {
       "stationIdToFetch": [
         { "stationId": "NEW_STATION_ID", "contactNumber": "", "email": "", "location": "city-slug" }
       ]
     },
     "waterLevel": {
       "stationIdToFetch": [
         { "stationId": "NEW_WLMS_ID", "contactNumber": "", "email": "", "location": "city-slug", "referenceThreshold": 780 }
       ]
     }
   }
   ```
2. **Add Station Metadata Defaults (`src/lib/constants/default-stations.ts`)**:
   Provide station name, city, province, and geographical coordinates `[longitude, latitude]`.

---

## 6. Development & Maintenance Commands

### 6.1 Running the Frontend Dev Server
```bash
npm run dev
# Starts Next.js on port 80 (http://localhost)
```

### 6.2 Running the Python MQTT Live Streamer Daemon
```bash
.venv\Scripts\python.exe prediction-model/src/mqtt_pinn_live_streamer.py
# Ingests live MQTT telemetry, performs LNN state updates, and logs to CSV
```

### 6.3 Running Automated Validation Audits
```bash
.venv\Scripts\python.exe prediction-model/src/validate_wmo_pagasa_alignment.py
# Audits Weather, Water Level, and Predictions against WMO & PAGASA benchmarks
```

### 6.4 Running Latency & Performance Profiler
```bash
.venv\Scripts\python.exe prediction-model/src/benchmark_performance.py
# Measures API endpoint response times and SWR cache hit latencies
```
