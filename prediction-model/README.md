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
| **Core Commercial** | Current observations, rising/falling trends, temperature outlook, relative humidity outlook, barometric pressure tendency, wind speed & circular direction, calibrated rain probability, derived NOAA heat index | Commercial monitoring, agricultural/industrial planning, dashboard displays with confidence and probability labels |
| **Secondary / Beta** | Rain accumulation ranges (mm), light intensity (lux daylight cycle), river stage delta (meters) | Opt-in research and beta testing only (`not_for_life_safety: true`) |
| **Blocked by Data** | UV Index | Quarantined / not scored due to raw telemetry calibration defect (reporting up to 11.0 index at midnight) |
| **Prohibited Claims** | Flood evacuation triggers, autonomous flood warnings, guaranteed rain statements | **STRICTLY PROHIBITED** |

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
5. **Frozen Calibration Split & Hybrid Selection**:
   - Hybrid blend weights and operational alert thresholds are fitted exclusively on the calibration split and frozen prior to test evaluation.

---

## 4. Multi-Horizon Scorecard Highlights

Evaluated on the **untouched test partition** (2,820 sequence windows, 13,411 sample records across 16 stations):

- **Rain Probability Calibration (Brier Score)**:
  `GarciaWeatherLNN` outperforms persistence across **ALL 5 HORIZONS**:
  - +1h: Brier **0.1249** vs Persistence 0.1713 (**+27.1% skill**)
  - +3h: Brier **0.1844** vs Persistence 0.2270 (**+18.8% skill**)
  - +6h: Brier **0.2415** vs Persistence 0.2695 (**+10.4% skill**)
  - +12h: Brier **0.2538** vs Persistence 0.3000 (**+15.4% skill**)
  - +24h: Brier **0.2473** vs Persistence 0.3328 (**+25.7% skill**)
- **Calibrated Rain Hybrid Blend**:
  Achieves 77.4% F1 at +1h (matching persistence 77.5%) while reducing Brier score to 0.1219.
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

# 4. Run independent validation suite and generate scorecards
python prediction-model/src/validate.py --horizons 1 3 6 12 24

# 5. Run full unit test suite (11 unit tests)
pytest prediction-model/src/test_canonical_contract.py

# 6. Verify end-to-end artifact provenance and hash integrity
python prediction-model/src/verify_provenance.py

# 7. Run operational inference on observed 8-feature sequence
python prediction-model/src/inference.py
```

---

## 6. Directory Structure

```plaintext
prediction-model/
├── requirements.txt                  # Pinned runtime dependencies
├── MODEL_REGISTRY.md                 # Formal catalog, methodology, and scorecards
├── FILE_CLASSIFICATION.md            # File retention registry & archive map
├── data/
│   ├── weather_telemetry.csv         # Raw weather station telemetry (756,156 rows)
│   ├── water_level_telemetry.csv     # Raw river gauge telemetry (43,883 rows)
│   ├── weather_data_audit.json       # Deterministic data availability audit
│   ├── cleaned_data_manifest.json    # Data cleaning manifest with hashes & counts
│   ├── data_quality_report.json      # Comprehensive quarantine audit report
│   ├── lnn_weather_water_h*.pt       # MF-1 PyTorch checkpoints per horizon
│   ├── lnn_trained_weights_h*.json   # MF-2 Standalone weights per horizon
│   ├── weather_validation_scorecard.json # Dedicated commercial weather scorecard
│   ├── validation_scorecard.json     # Multi-horizon evaluation scorecard
│   └── test_predictions_log.csv     # Per-sample predictions & metadata log
├── src/
│   ├── audit_data_availability.py    # Target data and sensor calibration audit
│   ├── dataset.py                    # Canonical data pipeline & hourly resampler
│   ├── model.py                      # PyTorch CfCCell, WeatherWaterLNN, GarciaWeatherLNN
│   ├── train.py                      # MF-1 PyTorch multi-task training pipeline
│   ├── train_standalone.py           # MF-2 standalone training pipeline
│   ├── train_and_evaluate_canonical.py # Master end-to-end orchestrator
│   ├── validate.py                   # Multi-horizon validator & conformal evaluator
│   ├── inference.py                  # Serverless inference engine (operational & research APIs)
│   ├── test_canonical_contract.py    # Automated unit tests for contract and isolation
│   └── smoke_test.py                 # Smoke test for data and model steps
└── README.md
```
