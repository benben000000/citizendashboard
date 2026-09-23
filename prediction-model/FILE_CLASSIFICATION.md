# Prediction Model File Classification & Retention Registry
This document records the classification of all files in prediction-model/, documenting which files are retained for the canonical production/research pipeline, which are retained for active external integrations (dashboard, MQTT, paper docs), and which legacy files were archived into git reference branch cleanup/archive-before-remediation-20260922.
**Archive Reference Branch**: cleanup/archive-before-remediation-20260922 (pushed to GitHub)
**Starting Commit**: 5039f83e082558094557eb77328af13bdec1c1c

## Summary
- **Total Tracked Files Analyzed**: 169
- **Retained Files (Cleaned Tree)**: 60
- **Removed Legacy Files (Archived)**: 109

## 1. Retained Files

| File Path | Category | Purpose / Justification |
|---|---|---|
| prediction-model/MODEL_REGISTRY.md | Canonical Runtime / Test / Artifact | Official model registry and evaluation scorecard |
| prediction-model/README.md | Canonical Runtime / Test / Artifact | Master documentation and reproduction guide |
| prediction-model/config/station_diurnal_profiles.json | External Integration / Active Service | MF-3 station diurnal physical profiles |
| prediction-model/data/champion_lnn_weights.json | External Integration / Active Service | Reference champion weights cited in audit report |
| prediction-model/data/cleaned_data_manifest.json | Canonical Runtime / Test / Artifact | Data cleaning provenance manifest with SHA-256 |
| prediction-model/data/data_quality_report.json | Canonical Runtime / Test / Artifact | Comprehensive quarantine audit report |
| prediction-model/data/denoised_pinn_telemetry.csv | External Integration / Active Service | Denoised telemetry stream used by export.service.ts |
| prediction-model/data/detailed_audit_report_data.json | External Integration / Active Service | Detailed audit metrics JSON cited in audit report |
| prediction-model/data/lnn_trained_weights.json | Canonical Runtime / Test / Artifact | MF-2 weights default (+1h) |
| prediction-model/data/lnn_trained_weights_h1.json | Canonical Runtime / Test / Artifact | MF-2 Standalone weights (+1h horizon) |
| prediction-model/data/lnn_trained_weights_h12.json | Canonical Runtime / Test / Artifact | MF-2 Standalone weights (+12h horizon) |
| prediction-model/data/lnn_trained_weights_h24.json | Canonical Runtime / Test / Artifact | MF-2 Standalone weights (+24h horizon) |
| prediction-model/data/lnn_trained_weights_h3.json | Canonical Runtime / Test / Artifact | MF-2 Standalone weights (+3h horizon) |
| prediction-model/data/lnn_trained_weights_h6.json | Canonical Runtime / Test / Artifact | MF-2 Standalone weights (+6h horizon) |
| prediction-model/data/lnn_weather_water.pt | Canonical Runtime / Test / Artifact | MF-1 checkpoint default (+1h) |
| prediction-model/data/lnn_weather_water_h1.pt | Canonical Runtime / Test / Artifact | MF-1 PyTorch checkpoint (+1h horizon) |
| prediction-model/data/lnn_weather_water_h12.pt | Canonical Runtime / Test / Artifact | MF-1 PyTorch checkpoint (+12h horizon) |
| prediction-model/data/lnn_weather_water_h24.pt | Canonical Runtime / Test / Artifact | MF-1 PyTorch checkpoint (+24h horizon) |
| prediction-model/data/lnn_weather_water_h3.pt | Canonical Runtime / Test / Artifact | MF-1 PyTorch checkpoint (+3h horizon) |
| prediction-model/data/lnn_weather_water_h6.pt | Canonical Runtime / Test / Artifact | MF-1 PyTorch checkpoint (+6h horizon) |
| prediction-model/data/mqtt_live_predictions.json | External Integration / Active Service | Live MQTT stream prediction cache used by water-level.service.ts and telemetry.service.ts |
| prediction-model/data/pinn_lnn_3h_online_weights.json | External Integration / Active Service | Online adaptive weights used by prediction.service.ts |
| prediction-model/data/raw_mqtt_telemetry.csv | External Integration / Active Service | Live raw telemetry feed used by export.service.ts |
| prediction-model/data/segregated/clean_consolidated_2024_2026.csv | External Integration / Active Service | Historical consolidated backtest telemetry |
| prediction-model/data/segregated/september_2026_backtest_1h.csv | External Integration / Active Service | Historical September backtest dataset |
| prediction-model/data/test_predictions_log.csv | Canonical Runtime / Test / Artifact | Test set predictions log with lead times (13,411 rows) |
| prediction-model/data/validation_scorecard.json | Canonical Runtime / Test / Artifact | Official multi-horizon evaluation scorecard |
| prediction-model/data/weather_data_audit.json | Canonical Runtime / Test / Artifact | Deterministic target data availability & sensor calibration audit |
| prediction-model/data/weather_validation_scorecard.json | Canonical Runtime / Test / Artifact | Commercial weather-only multi-horizon scorecard |
| prediction-model/data/water_level_telemetry.csv | Canonical Runtime / Test / Artifact | Raw source water level gauge telemetry (ground truth) |
| prediction-model/data/weather_telemetry.csv | Canonical Runtime / Test / Artifact | Raw source weather telemetry (ground truth) |
| prediction-model/docs/Garcia_PINN_LNN_Scientific_Audit.md | Historical Research Documentation | Scientific audit paper of Garcia PINN-LNN |
| prediction-model/docs/Garcia_PINN_LNN_Working_Paper.html | Historical Research Documentation | HTML working paper manuscript |
| prediction-model/docs/Garcia_PINN_LNN_Working_Paper.md | Historical Research Documentation | Markdown working paper manuscript |
| prediction-model/docs/Garcia_PINN_LNN_Working_Paper.pdf | Historical Research Documentation | Published working paper PDF artifact |
| prediction-model/docs/README.md | Historical Research Documentation | Docs overview |
| prediction-model/docs/figures/fig1_system_architecture.svg | Historical Research Documentation | Paper figure 1: System architecture |
| prediction-model/docs/figures/fig2_pinn_lnn_cell.svg | Historical Research Documentation | Paper figure 2: PINN-LNN cell schematic |
| prediction-model/docs/figures/fig3_tournament_benchmark.svg | Historical Research Documentation | Paper figure 3: Tournament benchmarks |
| prediction-model/logs/continuous_3h_validation_log.json | External Integration / Active Service | Continuous 3h validation log cited in maintenance docs |
| prediction-model/requirements.txt | Canonical Runtime / Test / Artifact | Project dependency specification |
| prediction-model/data/inference_policy.json | Canonical Runtime / Test / Artifact | Versioned operational inference policy (Finding 1) |
| prediction-model/src/audit_data_availability.py | Canonical Runtime / Test / Artifact | Target data availability & sensor calibration audit |
| prediction-model/src/benchmark_performance.py | External Integration / Active Service | Performance benchmark referenced in developer guide |
| prediction-model/src/dataset.py | Canonical Runtime / Test / Artifact | Canonical data pipeline & hourly resampler |
| prediction-model/src/generate_comprehensive_audit_outputs.py | External Integration / Active Service | Historical comprehensive audit output generator |
| prediction-model/src/generate_september_backtest_csv.py | Intentionally Retained Legacy (Disabled) | Historical backtest script; fails closed with deprecation RuntimeError |
| prediction-model/src/inference.py | Canonical Runtime / Test / Artifact | Serverless inference engine (operational & research APIs) |
| prediction-model/src/model.py | Canonical Runtime / Test / Artifact | PyTorch CfCCell and WeatherWaterLNN (MF-1) |
| prediction-model/src/mqtt_pinn_live_streamer.py | Intentionally Retained Legacy (Disabled) | Historical MQTT edge daemon; fails closed with deprecation RuntimeError |
| prediction-model/src/simulate_spatial_rain_imputation.py | External Integration / Active Service | Spatial IDW sensor reconstruction engine |
| prediction-model/src/smoke_test.py | Canonical Runtime / Test / Artifact | Automated end-to-end smoke test |
| prediction-model/src/station_adaptive_pinn_lnn.py | Intentionally Retained Legacy (Disabled) | Historical StationAdaptivePINN; fails closed with deprecation RuntimeError |
| prediction-model/src/test_aws_ca.py | External Integration / Active Service | AWS Root CA certificate validation utility |
| prediction-model/src/test_aws_iot_connect.py | External Integration / Active Service | AWS IoT Core mTLS connection tester |
| prediction-model/src/test_aws_iot_probe.py | External Integration / Active Service | AWS IoT endpoint topic probe utility |
| prediction-model/src/test_canonical_contract.py | Canonical Runtime / Test / Artifact | Automated unit tests for data contract and fail-closed inference |
| prediction-model/src/test_inference_contract.py | Canonical Runtime / Test / Artifact | Automated unit tests for operational inference policy contract |
| prediction-model/src/test_provenance.py | Canonical Runtime / Test / Artifact | Automated unit tests for exact-HEAD provenance and path hygiene |
| prediction-model/src/test_spatial_imputation.py | External Integration / Active Service | Spatial IDW sensor reconstruction unit test |
| prediction-model/src/train.py | Canonical Runtime / Test / Artifact | MF-1 PyTorch training pipeline |
| prediction-model/src/train_and_evaluate_all_models.py | Intentionally Retained Legacy (Disabled) | Historical benchmark script; fails closed with deprecation RuntimeError |
| prediction-model/src/train_and_evaluate_canonical.py | Canonical Runtime / Test / Artifact | Canonical end-to-end master orchestrator |
| prediction-model/src/train_standalone.py | Canonical Runtime / Test / Artifact | MF-2 Standalone ContinuousLNNCell trainer |
| prediction-model/src/validate.py | Canonical Runtime / Test / Artifact | Independent validator and conformal evaluator |
| prediction-model/src/validate_wmo_pagasa_alignment.py | External Integration / Active Service | PAGASA/WMO groundtruth validation script |

## 2. Updated and Resolved Legacy References (Finding 4)

The cleanup process removed obsolete experimental artifacts. Four surviving scripts originally referenced these deleted files. They are outside the canonical training/evaluation pipeline and have been remediated so they fail closed immediately with a documented legacy deprecation message referencing the archive branch:

| Script Path | Prior Deleted Artifact Reference | Remediated Status | Archive Location |
|---|---|---|---|
| `prediction-model/src/train_and_evaluate_all_models.py` | `data/audit_and_benchmark_metrics.json` | Fails closed with `RuntimeError` upon invocation | `cleanup/archive-before-remediation-20260922` |
| `prediction-model/src/generate_september_backtest_csv.py` | `data/pinn_lnn_champion_weights.json` | Fails closed with `RuntimeError` upon invocation | `cleanup/archive-before-remediation-20260922` |
| `prediction-model/src/mqtt_pinn_live_streamer.py` | `data/pinn_lnn_champion_weights.json` | Fails closed with `RuntimeError` upon invocation | `cleanup/archive-before-remediation-20260922` |
| `prediction-model/src/station_adaptive_pinn_lnn.py` | `data/station_pinn_profiles.json`, `data/station_adaptive_minute_forecasts.csv` | Fails closed with `RuntimeError` upon invocation | `cleanup/archive-before-remediation-20260922` |

No active canonical script references any deleted artifact.

## 3. Removed Legacy Files (Safely Preserved in Archive Branch)

| File Path | Category | Reason for Removal |
|---|---|---|
| prediction-model/data/audit_and_benchmark_metrics.json | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/data/comparison_24h_lnn_vs_pagasa.csv | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/data/comparison_24h_lnn_vs_pagasa.json | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/data/dataset_summary.json | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/data/live_1hour_audit_log.csv | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/data/live_1hour_audit_log.json | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/data/lnn_24hour_minute_predictions.csv | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/data/lnn_24hour_validation_log.json | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/data/mqtt_live_stream.csv | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/data/multi_agent_minute_forecasts.csv | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/data/multi_agent_tournament_results.json | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/data/multi_horizon_benchmark_results.json | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/data/multi_station_3h_prediction.csv | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/data/multi_station_3h_prediction.json | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/data/multimodal_live_snapshot.json | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/data/multimodal_prediction_log.csv | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/data/multimodal_prediction_log.json | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/data/pinn_lnn_champion_weights.json | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/data/pinn_lnn_evolution_history.json | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/data/pinn_lnn_iterative_minute_logs.csv | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/data/pinn_lnn_minute_forecasts.csv | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/data/pinn_lnn_tournament_results.json | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/data/prediction_results_24h.csv | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/data/prediction_results_24h.json | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/data/prediction_results_72h.csv | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/data/prediction_results_72h.json | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/data/segregated/2026/telemetry_2026.csv | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/data/segregated/2_month_telemetry_and_predictions_2026-08-01_to_2026-09-20_1h.csv | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/data/segregated/segregation_summary.json | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/data/station_adaptive_minute_forecasts.csv | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/data/station_pagasa_wmo_comparison.csv | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/data/station_pagasa_wmo_comparison.json | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/data/station_pinn_profiles.json | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/data/training_history_2026.json | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/data/validation_report.json | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/logs/benchmark_1h_matrix_results.json | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/logs/benchmark_48h_fidelity_results.json | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/logs/continuous_24h_summary.json | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/logs/continuous_24h_validation_log.json | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/logs/continuous_3h_summary.json | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/logs/continuous_validation_log.json | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/logs/groundtruth_48h_validation.json | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/logs/live_1h_prediction_summary.json | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/logs/live_1h_prediction_vs_actual_log.json | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/logs/rain_burst_timeline.json | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/src/audit_1hour_realtime_prediction.py | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/src/audit_all_stations_inventory.py | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/src/audit_all_weather_stats.py | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/src/audit_precipitation_hardware.py | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/src/audit_predictions_network.py | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/src/audit_raw_precip_spikes.py | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/src/audit_realtime_telemetry.py | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/src/audit_wlms_hardware.py | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/src/benchmark_1h_side_by_side_matrix.py | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/src/benchmark_24h_pagasa.py | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/src/benchmark_48h_realworld_fidelity.py | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/src/benchmark_all_stations_3h.py | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/src/benchmark_cross_model_comparison.py | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/src/benchmark_diurnal_soft_prior.py | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/src/benchmark_pagasa.py | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/src/benchmark_stations_vs_pagasa.py | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/src/benchmark_step2_dynamic_tau.py | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/src/capture_portal_screenshots.py | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/src/compile_pdf.py | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/src/comprehensive_network_audit.py | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/src/continuous_24h_lnn_benchmark.py | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/src/continuous_24h_network_validator.py | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/src/continuous_3h_realtime_validator_trainer.py | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/src/continuous_network_validator.py | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/src/display_audit_summary.py | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/src/evaluate_multi_horizon_benchmarks.py | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/src/export_onnx.py | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/src/fetch_dataset.py | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/src/fix_git_path.py | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/src/generate_24h_prediction_log.py | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/src/generate_2month_backtest_csv.py | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/src/generate_audit_report_pdf.py | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/src/generate_master_documentation_pdf.py | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/src/generate_working_paper_pdf.py | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/src/github_device_pusher.py | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/src/inspect_flood_stages.py | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/src/inspect_live_prod_stations.py | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/src/inspect_rain_calculation.py | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/src/inspect_raw_prod_fields.py | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/src/inspect_station_breakdown.py | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/src/iteration1_audit.py | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/src/iteration2_audit.py | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/src/iteration3_audit.py | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/src/live_1h_prediction_telemetry_matcher.py | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/src/log_station_timeseries.py | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/src/multi_agent_lnn_tournament.py | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/src/pagasa_wmo_groundtruth_validator.py | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/src/pinn_lnn_iterative_loop.py | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/src/pinn_lnn_tournament.py | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/src/qa_test_dashboard.py | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/src/qa_test_multi_horizon.py | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/src/qa_test_pdf_pages.py | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/src/qa_test_station_switch.py | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/src/run_ml_audit_and_backtest.py | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/src/satellite_radar_fetcher.py | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/src/segregate_and_clean_dataset.py | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/src/train_lnn_2026.py | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/src/validate_aug29_bulletins.py | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/src/validate_live_event_stats.py | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/src/validate_spatial_estimate_wmo.py | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/src/verify_all_engine_functions.py | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/src/verify_all_stations_precipitation.py | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/src/verify_dashboard_population.py | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
| prediction-model/src/verify_portal_exports.py | Legacy Experiment / Obsolete Log | Obsolete experimental/testing artifact from pre-canonical iterations. Not referenced by active code. Safely archived in branch cleanup/archive-before-remediation-20260922. |
