"""
Training Script for Continuous-Time CfC/LNN Weather & Hydrological Model.
Optimizes multi-task loss:
  L_total = lambda_rain * L_bce + lambda_precip * L_precip + lambda_water * L_water

Note: This trains the PyTorch WeatherWaterLNN model (Model Family 1).
      See MODEL_REGISTRY.md for the other model families.

Audit fixes applied:
  - num_samples -> max_samples keyword fix (Critical)
  - Chronological train/val split via dataset.py (Critical)
  - Water loss masked to real gauge observations only (Critical)
  - Future-horizon targets, not same-step reconstruction (Critical)
  - Reproducibility: seeds, hashes, manifest in checkpoint (Medium)
"""

import os
import sys
import hashlib
import json
import random
import platform
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import WeatherWaterLNN
from dataset import TelemetryDataset, FEATURE_MEANS, FEATURE_STDS

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
DEFAULT_SEED = 42


def set_reproducibility_seed(seed: int = DEFAULT_SEED):
    """Set all random seeds for reproducible training runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _dataset_file_hash(path: str) -> str:
    """SHA-256 hash of a dataset file for reproducibility tracking."""
    if not os.path.exists(path):
        return "FILE_NOT_FOUND"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_reproducibility_manifest(seed: int, epoch: int, val_loss: float,
                                     water_rmse: float, model_config: dict,
                                     data_dir: str, horizon: int) -> dict:
    """Build a complete reproducibility manifest saved with the checkpoint."""
    weather_csv = os.path.join(data_dir, "weather_telemetry.csv")
    water_csv = os.path.join(data_dir, "water_level_telemetry.csv")
    return {
        "training_date": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "model_config": model_config,
        "forecast_horizon_hours": horizon,
        "feature_schema": ["temperature", "heat_index", "wind_speed", "pressure"],
        "normalization": {
            "means": FEATURE_MEANS.tolist(),
            "stds": FEATURE_STDS.tolist(),
        },
        "dataset_hashes": {
            "weather_telemetry_sha256": _dataset_file_hash(weather_csv),
            "water_level_telemetry_sha256": _dataset_file_hash(water_csv),
        },
        "split_method": "chronological_60_20_20",
        "environment": {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "platform": platform.platform(),
        },
        "best_epoch": epoch,
        "best_val_loss": val_loss,
        "best_water_rmse": water_rmse,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_lnn_model(
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 0.003,
    seed: int = DEFAULT_SEED,
    horizon: int = 1,
    save_path: str = "lnn_weather_water.pt",
):
    print("=" * 60)
    print("Initializing Continuous-Time CfC/LNN Training Pipeline...")
    print(f"Forecast horizon: +{horizon}h ahead")
    print("=" * 60)

    set_reproducibility_seed(seed)

    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    if save_path == "lnn_weather_water.pt":
        save_path = os.path.join(data_dir, "lnn_weather_water.pt")

    # Prepare datasets with chronological split (Fix 9) and future targets (Fix 11)
    train_dataset = TelemetryDataset(max_samples=600, seq_len=24, split="train", horizon=horizon)
    val_dataset = TelemetryDataset(max_samples=150, seq_len=24, split="val", horizon=horizon)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = {"input_dim": 4, "hidden_dim": 32}
    model = WeatherWaterLNN(**model_config).to(device)

    # Losses & Optimizer
    bce_loss_fn = nn.BCELoss()
    mse_loss_fn = nn.MSELoss(reduction="none")
    huber_loss_fn = nn.SmoothL1Loss(reduction="none")
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_epoch = 0
    best_water_rmse = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0

        for batch in train_loader:
            telemetry = batch["telemetry"].to(device)
            dt = batch["dt"].to(device)
            target_rain = batch["rain_prob"].to(device)
            target_precip = batch["precip_mm"].to(device)
            target_water = batch["water_level"].to(device)
            has_water = batch["has_water"].to(device)

            optimizer.zero_grad()

            pred_rain, pred_precip, pred_water = model(telemetry, dt)

            # Use final hidden state output for the forecast
            # pred_rain shape: [batch, seq_len, 1] -> take last step
            pred_rain_final = pred_rain[:, -1, :]
            pred_precip_final = pred_precip[:, -1, :]
            pred_water_final = pred_water[:, -1, :]

            loss_rain = bce_loss_fn(pred_rain_final, target_rain)
            loss_precip = mse_loss_fn(pred_precip_final, target_precip).mean()

            # Water loss: masked to real gauge observations only (Fix 10)
            water_loss_per_sample = huber_loss_fn(pred_water_final, target_water)
            water_mask_sum = has_water.sum()
            if water_mask_sum > 0:
                loss_water = (water_loss_per_sample * has_water).sum() / water_mask_sum
            else:
                loss_water = torch.tensor(0.0, device=device)

            # Combined multi-task loss
            loss = 1.0 * loss_rain + 0.2 * loss_precip + 2.0 * loss_water
            loss.backward()

            # Gradient clipping for ODE numerical stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()

        scheduler.step()
        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0.0
        val_water_rmse_sum = 0.0
        val_water_count = 0

        with torch.no_grad():
            for batch in val_loader:
                telemetry = batch["telemetry"].to(device)
                dt = batch["dt"].to(device)
                target_rain = batch["rain_prob"].to(device)
                target_precip = batch["precip_mm"].to(device)
                target_water = batch["water_level"].to(device)
                has_water = batch["has_water"].to(device)

                pred_rain, pred_precip, pred_water = model(telemetry, dt)

                pred_rain_final = pred_rain[:, -1, :]
                pred_precip_final = pred_precip[:, -1, :]
                pred_water_final = pred_water[:, -1, :]

                loss_rain = bce_loss_fn(pred_rain_final, target_rain)
                loss_precip = mse_loss_fn(pred_precip_final, target_precip).mean()

                water_loss_per_sample = huber_loss_fn(pred_water_final, target_water)
                water_mask_sum = has_water.sum()
                if water_mask_sum > 0:
                    loss_water = (water_loss_per_sample * has_water).sum() / water_mask_sum
                    # Water RMSE on gauge-matched samples only
                    water_sq_err = ((pred_water_final - target_water) ** 2 * has_water).sum()
                    val_water_rmse_sum += water_sq_err.item()
                    val_water_count += int(water_mask_sum.item())
                else:
                    loss_water = torch.tensor(0.0, device=device)

                val_loss = 1.0 * loss_rain + 0.2 * loss_precip + 2.0 * loss_water
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        avg_water_rmse = (val_water_rmse_sum / val_water_count) ** 0.5 if val_water_count > 0 else float("nan")

        if epoch % 5 == 0 or epoch == epochs:
            water_str = f"{avg_water_rmse:.3f}m" if val_water_count > 0 else "N/A (no gauge data)"
            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] "
                f"| Train Loss: {avg_train_loss:.4f} "
                f"| Val Loss: {avg_val_loss:.4f} "
                f"| Water RMSE: {water_str}"
            )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            best_water_rmse = avg_water_rmse
            manifest = _build_reproducibility_manifest(
                seed, epoch, avg_val_loss, avg_water_rmse, model_config, data_dir, horizon,
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": avg_val_loss,
                    "water_rmse": avg_water_rmse,
                    "manifest": manifest,
                },
                save_path,
            )

    print("=" * 60)
    print(f"Training Complete! Best model saved to: {os.path.abspath(save_path)}")
    print(f"Best epoch: {best_epoch}, Val loss: {best_val_loss:.4f}, Water RMSE: {best_water_rmse:.3f}m")
    print("=" * 60)
    return model


# ---------------------------------------------------------------------------
# Smoke test — verifies the full pipeline loads and runs one step
# ---------------------------------------------------------------------------
def smoke_test():
    """Loads one batch and runs one optimizer step. Raises on any failure."""
    print("Running smoke test...")
    set_reproducibility_seed(0)
    ds = TelemetryDataset(max_samples=10, seq_len=8, split="train", horizon=1)
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    model = WeatherWaterLNN(input_dim=4, hidden_dim=16)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCELoss()
    mse = nn.MSELoss()

    batch = next(iter(loader))
    pred_rain, pred_precip, pred_water = model(batch["telemetry"], batch["dt"])
    loss = bce(pred_rain[:, -1, :], batch["rain_prob"]) + mse(pred_precip[:, -1, :], batch["precip_mm"])
    loss.backward()
    optimizer.step()
    print(f"Smoke test PASSED - loss={loss.item():.4f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--smoke":
        smoke_test()
    else:
        train_lnn_model(epochs=25, batch_size=32, horizon=1)
