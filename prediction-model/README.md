# KloudTrack Continuous-Time Weather & Hydrological Model (CfC/LNN)

This module implements continuous-time recurrent neural network architectures for weather telemetry and hydrological stage forecasting using **Closed-form Continuous-time (CfC) / Liquid Neural Networks (LNNs)**.

> [!IMPORTANT]
> **Status: RESEARCH_PROTOTYPE**
> The models and scorecards in this repository are research prototypes evaluated against strict out-of-sample benchmarks. They are **not** certified for autonomous flood warnings, emergency operations, or life-safety decisions. On the latest independent multi-horizon evaluation, both models underperform operational persistence baselines. See [MODEL_REGISTRY.md](file:///c:/Ben%20File/beta-citizen-prediction/prediction-model/MODEL_REGISTRY.md) for full benchmark scorecards and methodology.

---

## Model Families

To prevent architectural conflation, the repository maintains three clearly separated model families governed by the canonical data contract in [MODEL_REGISTRY.md](file:///c:/Ben%20File/beta-citizen-prediction/prediction-model/MODEL_REGISTRY.md):

| Model ID | Name | Architecture | Framework | Training & Evaluation Contract | Primary Artifact |
|---|---|---|---|---|---|
| **MF-1** | `WeatherWaterLNN` | Continuous-Time CfC Recurrent Neural Network | PyTorch | Multi-task gradient descent on canonical hourly future windows ($t_0 + h$) | `data/lnn_weather_water.pt`, `data/lnn_weather_water_h*.pt` |
| **MF-2** | `ContinuousLNNCell` | Analytical ODE Closed-Form Continuous Cell | Pure Python (Zero-dependency) | Unrolled 24-step sequence training on canonical hourly future windows ($t_0 + h$) | `data/lnn_trained_weights.json`, `data/lnn_trained_weights_h*.json` |
| **MF-3** | `StationAdaptivePINN` | Diurnal/Heuristic physics-coupled engine | Pure Python | Online parameter adaptation & diurnal priors | Configs in `config/` |

---

## Canonical Forecasting Contract

All model families operate on a strictly defined future-forecasting data contract:
1. **Raw Ingestion & Quarantine**:
   - Timezone-aware UTC timestamp parsing.
   - Outliers outside $[2025, 2027]$ are quarantined (specifically the 2069 timestamp bug).
   - Sensor spikes violating physical limits (e.g. pressure $< 900$ or $> 1050$ hPa, precipitation $> 300$ mm, water level $< 0$ or $> 15$ m) are quarantined and audited.
   - Provenance hashes and counts saved to `data/cleaned_data_manifest.json` and `data/data_quality_report.json`.
2. **Documented Hourly Grid**:
   - Irregular 1-minute telemetry is resampled into hourly bins.
   - Temperature, Heat Index, Wind Speed, Pressure: last valid observation within the hour.
   - Precipitation: sum of valid tipping-bucket minute increments in the hour (volume in mm).
   - River stage: last valid gauge observation within the hour from collocated station (Calumpit WLMS, `O3z0j5bG`).
3. **Forecast Sample Definition**:
   - Input: preceding 24 hourly observations ending at origin $t_0$.
   - Target: observation at future timestamp $t_0 + h$ (where $h \in \{1, 3, 6, 12, 24\}$ hours).
   - Elapsed lead time verified within tolerance ($|\text{lead} - h| \le 0.25$h), eliminating the previous row-offset defect.
4. **Chronological 60/20/20 Partitioning with 48h Embargo**:
   - Train (60%), Validation/Calibration (20%), and Test (20%) split strictly by timestamp.
   - 48-hour embargo ($\text{seq\_len} + \text{max\_horizon}$) between splits prevents boundary context leakage.
   - Normalization parameters fitted strictly on the training partition.
5. **Independent Validation & Conformal Bands**:
   - Conformal prediction quantiles calibrated on the validation split.
   - Empirical coverage independently measured on the untouched test split.

---

## Benchmark Scorecard Summary

Evaluated on untouched test partition (2,820 sequence windows, 221 gauge-matched samples):

```plaintext
===================================================================================================================
EXECUTIVE SCORECARD SUMMARY ACROSS HORIZONS (WITH STRONGER BASELINES)
===================================================================================================================
Horizon  | MF-1 F1   | MF-2 F1   | LogReg F1  | Rec-3h F1  | Persist F1  | MF-1 W-MAE  | MF-2 W-MAE  | Persist W-MAE
-------------------------------------------------------------------------------------------------------------------
+01h     | 74.1%     | 48.4%     | 61.7%      | 76.2%      | 77.5%       | 0.0507m     | 0.1253m     | 0.0146m      
+03h     | 60.2%     | 8.3%      | 25.5%      | 70.2%      | 70.3%       | 0.1847m     | 0.1244m     | 0.0422m      
+06h     | 48.7%     | 2.3%      | 6.3%       | 66.9%      | 64.8%       | 0.3974m     | 0.1273m     | 0.0767m      
+12h     | 47.2%     | 0.0%      | 0.2%       | 61.9%      | 61.0%       | 0.4751m     | 0.1409m     | 0.1111m      
+24h     | 48.7%     | 0.0%      | 0.0%       | 58.6%      | 56.6%       | 0.1459m     | 0.1605m     | 0.1495m      
===================================================================================================================
Recommendation: DO NOT DEPLOY (RESEARCH_PROTOTYPE ONLY). Models do not demonstrate statistically significant, robust superiority over operational persistence baselines across multi-hour horizons.
```

- **Rain Forecasting**: MF-1 achieves 74.1% F1 at +1h and ~47–60% at longer horizons; however, operational persistence (77.5% at +1h) and recent-window heuristics remain higher.
- **River Stage Forecasting**: Water stage MAE has improved dramatically with delta formulation (MF-1 MAE 0.051m at +1h, 0.146m at +24h vs previous ~1.2m), but persistence remains very competitive on this low-dynamic tidal sequence (0.015m at +1h).
- **Conformal Uncertainty & Calibration**: Conformal intervals and calibration are fully audited across all test partitions. Models remain designated as `RESEARCH_PROTOTYPE`.

---

## Directory Structure & Key Files

```plaintext
prediction-model/
├── requirements.txt                  # Pinned runtime dependencies
├── MODEL_REGISTRY.md                 # Formal catalog, methodology, and scorecards
├── data/
│   ├── weather_telemetry.csv         # Raw weather station telemetry (756,156 rows)
│   ├── water_level_telemetry.csv     # Raw river gauge telemetry (43,883 rows)
│   ├── cleaned_data_manifest.json    # Data cleaning manifest with hashes & counts
│   ├── data_quality_report.json      # Comprehensive quarantine audit report
│   ├── lnn_weather_water_h*.pt       # MF-1 PyTorch checkpoints per horizon
│   ├── lnn_trained_weights_h*.json   # MF-2 Standalone weights per horizon
│   ├── validation_scorecard.json     # Independent multi-horizon scorecard
│   └── test_predictions_log.csv     # Per-sample predictions & metadata log
├── src/
│   ├── dataset.py                    # Canonical data pipeline & hourly resampler
│   ├── model.py                      # PyTorch CfCCell and WeatherWaterLNN (MF-1)
│   ├── train.py                      # MF-1 training pipeline
│   ├── train_standalone.py           # MF-2 standalone training pipeline
│   ├── train_and_evaluate_canonical.py # Master end-to-end orchestrator
│   ├── validate.py                   # Multi-horizon validator & conformal evaluator
│   ├── test_canonical_contract.py    # Unit tests for contract and isolation
│   └── smoke_test.py                 # Smoke test for data and model steps
└── README.md
```

---

## Reproducible Commands

All commands should be executed from the repository root using the project environment:

### 1. Run Canonical Contract Unit Tests
Validates synthetic 1-minute lead time, quarantine filters, train-only normalization, and 48h embargo:
```bash
python prediction-model/src/test_canonical_contract.py
```

### 2. Run Smoke Test
Verifies mini-batch forward/backward steps for MF-1 and standalone updates for MF-2:
```bash
python prediction-model/src/smoke_test.py
```

### 3. Train Model Family 1 (PyTorch WeatherWaterLNN)
```bash
# Train for specific horizon (e.g. +1h)
python prediction-model/src/train.py --horizon 1 --epochs 20

# Train for +3h, +6h, +12h, +24h
python prediction-model/src/train.py --horizon 3
python prediction-model/src/train.py --horizon 6
python prediction-model/src/train.py --horizon 12
python prediction-model/src/train.py --horizon 24
```

### 4. Train Model Family 2 (ContinuousLNNCell)
```bash
python prediction-model/src/train_standalone.py --horizon 1 --epochs 12
```

### 5. Run Full Multi-Horizon Independent Validation
Runs out-of-sample evaluation on untouched test split and produces `validation_scorecard.json`:
```bash
python prediction-model/src/validate.py --horizons 1 3 6 12 24
```

### 6. Run Complete Master Pipeline (Train + Validate All)
```bash
python prediction-model/src/train_and_evaluate_canonical.py
```
