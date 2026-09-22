"""
Smoke Test Suite for Prediction Model Pipeline.

Verifies:
  1. Canonical dataset loader initializes without error.
  2. MF-1 (PyTorch WeatherWaterLNN): 1 batch forward pass, loss computation, backward pass, optimizer step.
  3. MF-2 (ContinuousLNNCell): 1 window unroll, loss computation, and standalone gradient update step.
  4. Data quality manifest generation.
"""

import os
import sys

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import get_telemetry_pipeline, TelemetryDataset
from model import WeatherWaterLNN
from train_standalone import ContinuousLNNCell


def run_smoke_test():
    print("=" * 60)
    print("Running Prediction Model Smoke Test...")
    print("=" * 60)

    # 1. Dataset & Pipeline
    print("1. Initializing Telemetry Pipeline...")
    pipeline = get_telemetry_pipeline()
    report = pipeline.generate_data_quality_report()
    assert report["resampled_hourly_summary"]["total_station_hours"] > 0, "No station hours found!"
    print(f"   Success: {report['resampled_hourly_summary']['total_station_hours']} station-hours available.")

    # 2. Build small train dataset
    print("2. Building canonical training dataset (horizon=1, 16 samples)...")
    ds = TelemetryDataset(split="train", horizon=1, max_samples=16, pipeline=pipeline)
    assert len(ds) >= 2, f"Dataset too small for smoke test: {len(ds)}"
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    telemetry = batch["telemetry"]
    dt = batch["dt"]
    rain_prob = batch["rain_prob"]
    precip_mm = batch["precip_mm"]
    water_level = batch["water_level"]
    has_water = batch["has_water"]
    print(f"   Batch shapes: telemetry={telemetry.shape}, dt={dt.shape}, rain={rain_prob.shape}")

    in_features = telemetry.shape[-1]

    # 3. Model Family 1: PyTorch WeatherWaterLNN
    print("3. Testing MF-1 (WeatherWaterLNN) Forward + Backward...")
    device = torch.device("cpu")
    model = WeatherWaterLNN(input_dim=in_features, hidden_dim=16).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.01)
    bce_loss_fn = nn.BCELoss()
    mse_loss_fn = nn.MSELoss()

    optimizer.zero_grad()
    p_rain, p_precip, p_water = model(telemetry, dt)
    loss = bce_loss_fn(p_rain[:, -1, :], rain_prob) + mse_loss_fn(p_precip[:, -1, :], precip_mm)
    loss.backward()
    optimizer.step()
    print(f"   MF-1 step successful! Loss: {loss.item():.4f}")

    # 4. Model Family 2: Standalone ContinuousLNNCell
    print("4. Testing MF-2 (ContinuousLNNCell) Standalone Unroll + Update...")
    cell = ContinuousLNNCell(in_features=in_features, hidden_dim=8, seed=42)
    sample_feat = telemetry[0].numpy()
    sample_dt = dt[0].numpy()
    target_rain_val = float(rain_prob[0, 0])
    target_water_val = float(water_level[0, 0])
    has_w = bool(has_water[0, 0] > 0.5)

    hist_h, pred_r, pred_w = cell.unroll_window(sample_feat, sample_dt)
    assert len(hist_h) == 24, f"Expected 24 unrolled steps, got {len(hist_h)}"

    # Perform 1 gradient update
    err_r = pred_r - target_rain_val
    cell.b_rain -= 0.01 * err_r
    for j in range(cell.hidden_dim):
        cell.W_rain[j] -= 0.01 * err_r * hist_h[-1][j]
    print(f"   MF-2 step successful! Pred rain: {pred_r:.4f}, Target: {target_rain_val}")

    print("=" * 60)
    print("ALL SMOKE TESTS PASSED!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    run_smoke_test()
