"""
Master Pipeline Orchestrator: Train and Evaluate All Canonical Forecast Models.

Executes:
  1. Data quality audit and cleaned manifest output.
  2. Training MF-1 (PyTorch WeatherWaterLNN) across all horizons [1, 3, 6, 12, 24].
  3. Training MF-2 (Standalone ContinuousLNNCell) across all horizons [1, 3, 6, 12, 24].
  4. Running comprehensive independent validation suite (validate.py).
"""

import os
import sys
import time

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dataset import get_telemetry_pipeline, DEFAULT_HORIZONS
from train import train_mf1_model
from train_standalone import train_mf2_model
from validate import run_full_validation


def main():
    print("=" * 80)
    print("CITIZEN PREDICTION MODEL: CANONICAL FORECAST TRAINING & EVALUATION SUITE")
    print("=" * 80)
    start_time = time.time()

    # 1. Initialize Pipeline & Generate Data Quality Manifest
    print("\n>>> STEP 1: Ingesting Data & Generating Quality Report...")
    pipeline = get_telemetry_pipeline()
    report = pipeline.generate_data_quality_report()
    print("Data Quality Report:")
    print(f"  Raw weather rows: {report['raw_counts']['weather_telemetry_rows']}")
    print(f"  Raw water rows:   {report['raw_counts']['water_level_telemetry_rows']}")
    print(f"  Quarantine reasons: {report['quarantine_counts_by_reason']}")
    print(f"  Total valid station-hours: {report['resampled_hourly_summary']['total_station_hours']}")

    horizons = [1, 3, 6, 12, 24]

    # 2. Train Model Family 1 (PyTorch WeatherWaterLNN) for each horizon
    print("\n>>> STEP 2: Training Model Family 1 (PyTorch WeatherWaterLNN)...")
    for h in horizons:
        t0 = time.time()
        print(f"\n--- Training MF-1 for Horizon +{h}h ---")
        train_mf1_model(
            horizon=h,
            epochs=15,
            batch_size=32,
            lr=0.003,
            seed=42,
            max_train_samples=None,  # Full canonical training set
        )
        print(f"Completed MF-1 (+{h}h) in {time.time() - t0:.1f}s")

    # 3. Train Model Family 2 (Standalone ContinuousLNNCell) for each horizon
    print("\n>>> STEP 3: Training Model Family 2 (Standalone ContinuousLNNCell)...")
    for h in horizons:
        t0 = time.time()
        print(f"\n--- Training MF-2 for Horizon +{h}h ---")
        train_mf2_model(
            horizon=h,
            epochs=12,
            lr=0.015,
            seed=42,
            max_train_samples=1500,  # Stratified representative subset for fast pure-python training
        )
        print(f"Completed MF-2 (+{h}h) in {time.time() - t0:.1f}s")

    # 4. Run Independent Validation Suite
    print("\n>>> STEP 4: Running Independent Validation & Conformal Evaluation...")
    scorecard = run_full_validation(horizons=horizons)

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"ALL PIPELINE STAGES COMPLETED IN {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("=" * 80)


if __name__ == "__main__":
    main()
