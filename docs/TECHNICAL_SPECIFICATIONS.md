# KloudTrack Technical Specifications

**Document:** Complete System Technical Specifications  
**Application:** Citizendashboard (Weather, Water Level, and Nowcast Prediction)  
**Version:** 2.4-Production  

---

## 1. Technical Stack & Dependencies

| Layer | Technology | Version | Purpose |
| :--- | :--- | :---: | :--- |
| **Frontend Framework** | Next.js (App Router) | 14.2.x | Server-Side Rendering (SSR), API route handlers, and static optimization |
| **Language** | TypeScript / JavaScript | 5.x | Strict type safety across telemetry DTOs and prediction matrices |
| **Styling & Design** | Vanilla CSS + TailwindCSS | 3.4.x | Pixel-perfect responsive glassmorphism UI |
| **Icons & Visuals** | Lucide React | 0.4.x | Optimized vector iconography |
| **Charting Engine** | Recharts | 2.12.x | Responsive 24-hour summary curves and flood margin rendering |
| **Localization** | next-intl | 3.x | English and Filipino (Tagalog) bilingual support |
| **Backend AI Runtime** | Python (PyTorch / NumPy) | 3.11.x | Closed-Form Continuous-Time Liquid Neural Network ODE engine |
| **Ingestion Bus** | AWS IoT Core MQTT mTLS | Protocol v3.1.1 | Sub-second telemetry packet streaming |

---

## 2. Mathematical Formulation of the PINN-LNN Engine

### 2.1 Continuous-Time Liquid Neural ODE
The hidden state trajectory $\mathbf{h}(t) \in \mathbb{R}^8$ is governed by the continuous-time ordinary differential equation:
$$\frac{d\mathbf{h}(t)}{dt} = -\left[\frac{1}{\boldsymbol{\tau}} + \sigma(\mathbf{W}_{\text{in}} \mathbf{x}(t) + \mathbf{b}_{\text{in}})\right] \odot \mathbf{h}(t) + \mathbf{A} \odot \tanh(\mathbf{W}_{\text{in}} \mathbf{x}(t) + \mathbf{W}_{\text{rec}} \mathbf{h}(t) + \mathbf{b}_h)$$

Using the closed-form analytical solution (CfC), the state update for an arbitrary continuous lead time $\Delta t \ge 0$ is computed in a single forward evaluation without numerical discretization drift:
$$\mathbf{h}(t + \Delta t) = e^{-\frac{\Delta t}{\boldsymbol{\tau}}} \odot \mathbf{h}(t) + \left(1 - e^{-\frac{\Delta t}{\boldsymbol{\tau}}}\right) \odot \tanh\left(\mathbf{W}_{\text{in}} \mathbf{x}(t) + \mathbf{W}_{\text{rec}} \mathbf{h}(t) + \mathbf{b}_h\right)$$

### 2.2 Thermodynamic Saturation & Lifted Condensation Level (LCL)
The weather prediction module computes the saturation vapor pressure $e_s(T)$ via the **Magnus-Tetens relation**:
$$e_s(T) = 6.1121 \exp\left(\frac{17.67 \cdot T}{T + 243.5}\right) \quad [\text{hPa}]$$
The dew point temperature $T_d$ and cloud base height $z_{\text{LCL}}$ are derived as:
$$T_d \approx T - \left(\frac{100 - \text{RH}}{5}\right), \quad z_{\text{LCL}} \approx 125 \cdot (T - T_d) \quad [\text{meters}]$$

When $\text{RH} \ge 94\%$ and $z_{\text{LCL}} \le 80\text{ m}$, the model triggers thermodynamic convective condensation, predicting sudden rain bursts and localized cooling.

### 2.3 Saint-Venant 1D Hydrodynamic Continuity
The river stage elevation $H(t)$ at flood choke points (such as Calumpit Gatbuca Bridge) is governed by 1D open-channel mass conservation:
$$\frac{\partial A}{\partial t} + \frac{\partial Q}{\partial x} = q_{\text{lateral}}(t)$$
where $q_{\text{lateral}}(t) = C_r \cdot \text{Rainfall}(t - t_{\text{lag}})$ represents precipitation catchment runoff.

---

## 3. Caching & Performance Architecture

| Parameter | Value | Description |
| :--- | :---: | :--- |
| **In-Memory Cache TTL (Telemetry)** | `300s` (5 min) | Station telemetry cache duration |
| **In-Memory Cache TTL (Predictions)** | `30s` (0.5 min) | Continuous nowcast rollout cache |
| **Stale-While-Revalidate (SWR) Window**| `600s` (10 min) | Returns stale cache instantly ($0\text{ ms}$) during background fetch |
| **LRU Memory Cap** | `500 entries` | Maximum entries before oldest item eviction |
| **Spatial Distance Matrix** | Static Memoized | Haversine distance computed once per station pair |
| **API Response Latency Target** | **$< 20\text{ ms}$** | Measured average latency: **`13.7 – 17.6 ms`** |
| **Client Memory Footprint** | **$< 35\text{ MB}$** | Lightweight JavaScript bundle with package tree-shaking |

---

## 4. API Endpoint Specifications

### 4.1 `GET /api/telemetry/dashboard`
- **Description**: Returns all 23 weather stations with 3-tier QC filtering and Spatial IDW neighbor reconstruction.
- **Cache-Control**: `public, s-maxage=60, stale-while-revalidate=30`
- **Output**: Array of `{ station: StationPublicInfo, telemetry: TelemetryMetrics }`

### 4.2 `GET /api/water-level/dashboard`
- **Description**: Returns hydraulic water level stage data for all 13 monitoring stations.
- **Cache-Control**: `public, s-maxage=60, stale-while-revalidate=30`
- **Output**: Array of `{ station: StationPublicInfo, waterLevel: WaterLevelMetrics }`

### 4.3 `GET /api/prediction/station/[stationId]?horizon=[1h|3h|6h|12h|24h|48h|72h]`
- **Description**: Returns continuous-time LNN nowcast predictions, sudden rain burst advisories, and rolling weather projections.
- **Output**: `{ station, summary, forecast, history, weatherForecast, suddenRainBurst }`
