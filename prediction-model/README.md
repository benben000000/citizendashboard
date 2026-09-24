# Garcia Weather Telemetry Forecast Engine (Continuous-Time CfC/LNN)

The **Garcia Weather Telemetry Forecast Engine** converts real-time automated weather station telemetry into surface meteorology monitoring, trend detection, and probabilistic guidance using **Closed-form Continuous-time (CfC) / Liquid Neural Networks (LNNs)**.

> [!IMPORTANT]
> **Status: RESEARCH_PROTOTYPE**
> **Product Scope:** Surface weather telemetry monitoring, trends, and probabilistic guidance.
> **Not approved for autonomous flood warnings, emergency operations, or life-safety triggers.**
> Hydrological river-stage forecasting remains an **internal/beta research capability** (`not_for_life_safety: true`). Flood prediction claims are explicitly excluded from the commercial product core. See [MODEL_REGISTRY.md](file:///c:/Ben%20File/beta-citizen-prediction/prediction-model/MODEL_REGISTRY.md) for full benchmark scorecards, output taxonomy, and methodology.

---

## 1. Product Capabilities & Output Taxonomy

| Output Class | Variables | Permitted Operational Use |
|---|---|---|
| **Core Commercial** | Current observations, rising/falling trends, temperature outlook, relative humidity outlook, barometric pressure tendency, wind speed & circular direction, calibrated rain probability, derived NOAA heat index | Commercial monitoring, agricultural/industrial planning, dashboard displays with calibrated probabilities. (Weather prediction intervals are unavailable; conformal prediction intervals apply ONLY to the beta water-level experiment.) |
| **Secondary / Beta** | Rain accumulation ranges (mm), light intensity (lux daylight cycle), river stage delta (meters) | Opt-in research and beta testing only (`not_for_life_safety: true`) |
| **Blocked by Data** | UV Index | Quarantined / not scored due to raw telemetry calibration defect (reporting up to 11.0 index at midnight) |
| **Prohibited Claims** | Flood evacuation triggers, autonomous flood warnings, guaranteed rain statements, uncalibrated weather confidence bands | **STRICTLY PROHIBITED** |

---

## 2. Model Families

| Model ID | Name | Architecture | Framework | Training & Evaluation Contract | Primary Artifacts |
|---|---|---|---|---|---|
| **MF-1** | `GarciaWeatherLNN` | Continuous-Time CfC Recurrent Neural Network with Multi-Task Surface Meteorology Heads | PyTorch | Multi-task gradient descent on canonical hourly future windows ($t_0 + h$) | `data/lnn_weather_water.pt`, `data/lnn_weather_water_h*.pt` |
| **MF-2** | `ContinuousLNNCell` | Analytical ODE Closed-Form Continuous Cell | Pure Python (Zero-dependency) | Unrolled 24-step sequence training on canonical hourly future windows ($t_0 + h$) | `data/lnn_trained_weights.json`, `data/lnn_trained_weights_h*.json` |
| **MF-3** | `StationAdaptivePINN` | Diurnal/Heuristic physics-coupled engine | Pure Python | Online parameter adaptation & diurnal priors | Configs in `config/` |

---

## 3. Canonical Forecasting Contract

All models adhere to strict scientific governance preventing data leakage and benchmark contamination:
1. **Raw Ingestion & Outlier Quarantine**:
   - Timezone-aware UTC timestamp parsing.
   - Outliers outside $[2025, 2027]$ quarantined (including the 2069 timestamp anomaly).
   - Sensor spikes violating physical bounds are quarantined and tracked.
   - Provenance hashes and counts saved to `data/cleaned_data_manifest.json` and `data/weather_data_audit.json`.
2. **Documented Hourly Grid**:
   - Irregular 1-minute telemetry is resampled into hourly bins.
   - Temperature, Humidity, Pressure, Wind Speed: last valid observation within the hour.
   - Wind Direction: circular vector components $(u=\cos\theta, v=\sin\theta)$; calm winds ($< 1.0$ km/h) are masked.
   - Precipitation: sum of valid tipping-bucket minute increments in the hour (volume in mm).
   - River stage: last valid gauge observation within the hour from collocated station (Calumpit WLMS, `O3z0j5bG`).
3. **Forecast Sample Definition**:
   - Input: preceding 24 hourly observations ending at origin $t_0$ ($[t_{-23}, \dots, t_0]$).
   - Target: observation at future timestamp $t_0 + h$ (where $h \in \{1, 3, 6, 12, 24\}$ hours).
   - Elapsed lead time verified within tolerance ($|\text{lead} - h| \le 0.25$h).
4. **Chronological 60/20/20 Partitioning with 48h Embargo**:
   - Train (60%), Validation/Calibration (20%), and Test (20%) split strictly by timestamp.
   - 48-hour embargo ($\text{seq\_len} + \text{max\_horizon}$) between splits prevents boundary context leakage.
   - Normalization parameters fitted strictly on the training partition.
5. **Frozen Calibration Split & Operational Policy**:
   - Operational inference policy (`data/inference_policy.json`) generated deterministically by validation.
   - Hybrid rain blend weights and operational alert thresholds are fitted exclusively on the calibration split and frozen prior to test evaluation.
   - Variable sources governed per horizon (`learned_model` vs `persistence_fallback`).

---

## 4. Multi-Horizon Scorecard Highlights

Evaluated on the **untouched test partition** (2,820 sequence windows, 13,411 sample records across 16 stations):

- **Rain Probability Quality vs. Event Classification**:
  - **Probability Quality (Brier Score)**: Hybrid probability blending consistently minimizes Brier score across **ALL 5 HORIZONS**:
    - +1h: Brier **0.1219** vs Persistence 0.1713 (**+28.8% skill**)
    - +3h: Brier **0.1772** vs Persistence 0.2270 (**+21.9% skill**)
    - +6h: Brier **0.2183** vs Persistence 0.2695 (**+19.0% skill**)
    - +12h: Brier **0.2144** vs Persistence 0.3000 (**+28.5% skill**)
    - +24h: Brier **0.2118** vs Persistence 0.3328 (**+36.4% skill**)
  - **Event Classification Tradeoff**: While hybrid blending minimizes probability error, persistence achieves comparable or slightly higher F1/recall on short horizons (1h/3h). Operational alert thresholds ($T_{op}$) are calibrated on held-out validation data to optimize decision utility.
- **Precipitation Amount Baselines**:
  - Scorecards report full baselines: persistence MAE/RMSE, climatology MAE/RMSE, model skill score vs. persistence, and dry-hour vs. rainy-hour breakdowns.
  - Heavy-rain threshold events (2.5, 5.0, 10.0 mm/h) are evaluated with precision, POD/recall, and CSI.
  - **Precipitation Uncertainty Status**: Conformal prediction intervals are **UNAVAILABLE** for precipitation and surface weather variables; intervals apply strictly to the beta water-level experiment.
- **Surface Weather Accuracy**:
  - Barometric Pressure: beats persistence at +1h (0.33 hPa), +3h (0.67 vs 0.78 hPa, **+14.1% skill**), and +6h (1.10 vs 1.11 hPa).
  - Derived Heat Index (NOAA Rothfusz): beats persistence at +1h (1.49 vs 1.52°C), +3h (2.62 vs 2.66°C), +6h (3.81 vs 3.86°C), and +12h (4.02 vs 4.80°C, **+16.2% skill**).
  - 12h Diurnal Cycle: captures diurnal shifts, beating persistence in temperature (MAE 1.56 vs 1.83°C, **+14.8% skill**) and wind speed (MAE 1.35 vs 1.52 km/h, **+11.2% skill**).
- **River Stage Delta (Beta)**:
  MAE of 0.0338 m at +1h and 0.1327 m at +24h (beating persistence 0.1495 m at +24h). Conformal intervals meet nominal coverage targets (90.5% at 80% nominal, 95.0% at 90% nominal, 99.1% at 95% nominal).

---

## 5. Quick Start & Execution Commands

All commands are runnable from the repository root:

```bash
# 1. Audit raw telemetry field coverage, units, missingness, and sensor bounds
python prediction-model/src/audit_data_availability.py

# 2. Train GarciaWeatherLNN for a specific horizon (e.g. +1h)
python prediction-model/src/train.py --horizon 1 --epochs 15

# 3. Master orchestrator: train all models and evaluate all 5 horizons
python prediction-model/src/train_and_evaluate_canonical.py

# 4. Explicit artifact generation (separated from read-only test validation)
python prediction-model/src/generate_manifests.py --output-dir prediction-model/data
python prediction-model/src/generate_bundles.py

# 5. Run independent validation suite, generate scorecards and inference policy
python prediction-model/src/validate.py --horizons 1 3 6 12 24

# 6. Run operational monitoring, regime evaluation, and drift detection
python prediction-model/src/monitoring.py --output prediction-model/data/monitoring_report.json

# 7. Evaluate experimental two-stage precipitation architecture vs baseline
python prediction-model/src/evaluate_two_stage_precipitation.py

# 8. Run contract, inference, and provenance test suites (100% read-only)
python prediction-model/src/test_canonical_contract.py
python prediction-model/src/test_inference_contract.py
python prediction-model/src/test_provenance.py
python prediction-model/src/smoke_test.py

# 9. Verify two-commit provenance architecture and bundle hash integrity
python prediction-model/src/verify_provenance.py
# (Or allow an explicit historical commit override)
python prediction-model/src/verify_provenance.py --allow-commit <sha>

# 10. Run operational inference from bundles or checkpoints
python prediction-model/src/inference.py
```

---

## 6. Directory Structure

```plaintext
prediction-model/
├── requirements.txt                       # Pinned runtime dependencies
├── MODEL_REGISTRY.md                      # Formal catalog, methodology, and scorecards
├── FILE_CLASSIFICATION.md                 # File retention registry & archive map
├── data/
│   ├── weather_telemetry.csv              # Raw weather station telemetry (756,156 rows)
│   ├── water_level_telemetry.csv          # Raw river gauge telemetry (43,883 rows)
│   ├── weather_data_audit.json            # Deterministic data availability audit
│   ├── cleaned_data_manifest.json         # Data cleaning manifest with hashes & counts
│   ├── data_quality_report.json           # Comprehensive quarantine audit report
│   ├── inference_policy.json              # Versioned operational inference policy
│   ├── lnn_weather_water_h*.pt            # MF-1 PyTorch checkpoints per horizon
│   ├── lnn_trained_weights_h*.json        # MF-2 Standalone weights per horizon
│   ├── weather_validation_scorecard.json  # Dedicated commercial weather scorecard
│   ├── validation_scorecard.json          # Multi-horizon evaluation scorecard
│   ├── test_predictions_log.csv          # Per-sample predictions & metadata log
│   └── bundles/                           # Operational model-policy bundles
│       ├── h1/                            # Bundle for +1h forecast (checkpoint + policy + manifest)
│       ├── h3/                            # Bundle for +3h forecast
│       ├── h6/                            # Bundle for +6h forecast
│       ├── h12/                           # Bundle for +12h forecast
│       └── h24/                           # Bundle for +24h forecast
├── src/
│   ├── audit_data_availability.py         # Target data and sensor calibration audit
│   ├── dataset.py                         # Canonical data pipeline & hourly resampler
│   ├── model.py                           # PyTorch CfCCell, GarciaWeatherLNN, TwoStagePrecipitationHead
│   ├── train.py                           # MF-1 PyTorch multi-task training pipeline
│   ├── train_standalone.py                # MF-2 standalone training pipeline
│   ├── train_and_evaluate_canonical.py    # Master end-to-end orchestrator
│   ├── generate_manifests.py              # Explicit artifact generation tool
│   ├── generate_bundles.py                # Model-policy packaging & manifest bundler
│   ├── monitoring.py                      # Operational monitoring, regimes & drift engine
│   ├── evaluate_two_stage_precipitation.py # Experimental precipitation head comparison
│   ├── validate.py                        # Multi-horizon validator & conformal evaluator
│   ├── inference.py                       # Bundle-aware serverless inference engine
│   ├── test_canonical_contract.py         # Automated unit tests for contract and isolation
│   ├── test_inference_contract.py         # Unit tests for 5-horizon bundle & policy contract
│   ├── test_provenance.py                 # Unit tests for two-commit provenance & path hygiene
│   └── smoke_test.py                      # Smoke test for data and model steps
└── README.md
```
