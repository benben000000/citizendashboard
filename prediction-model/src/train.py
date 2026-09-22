"""
Training Pipeline for Continuous-Time CfC/LNN Weather & Hydrological Forecasting (Model Family 1).

Trains PyTorch WeatherWaterLNN strictly on canonical future-window forecast samples:
  - Input: [batch, seq_len, 4] normalized hourly telemetry.
  - dt: [batch, seq_len, 1] actual elapsed hours between measurements.
  - Targets: rain_prob, precip_mm, and water_level at future horizon t0 + h.
  - Partitioning: Train split only for parameter updates; Validation split only for checkpoint selection.
  - Test split is NEVER inspected during training (evaluated strictly by independent validator).
  - Checkpoint includes complete reproducibility manifest with hashes, normalization, and parameters.
"""

import os
import sys
import argparse
import random
import platform
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dataset import (
    TelemetryDataset,
    get_telemetry_pipeline,
    compute_file_sha256,
    DATA_DIR,
    WEATHER_CSV_PATH,
    WATER_CSV_PATH,
    DEFAULT_SEQ_LEN,
)
from model import WeatherWaterLNN

DEFAULT_SEED = 42


def set_reproducibility_seed(seed: int = DEFAULT_SEED):
    """Seed all pseudo-random number generators for deterministic reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_checkpoint_manifest(
    model_family: str,
    seed: int,
    horizon: int,
    epoch: int,
    val_loss: float,
    val_water_rmse: float,
    model_config: dict,
    pipeline,
) -> dict:
    """Construct complete metadata manifest saved inside the checkpoint."""
    # Attempt to get git commit hash
    git_commit = "unknown"
    try:
        import subprocess
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(DATA_DIR),
            text=True
        ).strip()
    except Exception:
        pass

    return {
        "model_family": model_family,
        "model_status": "RESEARCH_PROTOTYPE",
        "code_commit": git_commit,
        "training_date": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "forecast_horizon_hours": horizon,
        "feature_schema": ["temperature", "heat_index", "wind_speed", "pressure"],
        "input_sequence_length_hours": DEFAULT_SEQ_LEN,
        "resampling_rule": "UTC hourly bins: last valid temp/hi/ws/pressure, sum of precip volume (mm), last valid water stage (m)",
        "model_config": model_config,
        "split_boundaries": {
            "strategy": "chronological_60_20_20_with_48h_embargo",
            "train_start": pipeline.time_range_min.isoformat() if pipeline.time_range_min else None,
            "train_end": pipeline.train_end.isoformat() if pipeline.train_end else None,
            "val_start": pipeline.val_start.isoformat() if pipeline.val_start else None,
            "val_end": pipeline.val_end.isoformat() if pipeline.val_end else None,
            "test_start": pipeline.test_start.isoformat() if pipeline.test_start else None,
            "test_end": pipeline.time_range_max.isoformat() if pipeline.time_range_max else None,
        },
        "normalization": {
            "means": pipeline.norm_means.tolist() if isinstance(pipeline.norm_means, np.ndarray) else list(pipeline.norm_means),
            "stds": pipeline.norm_stds.tolist() if isinstance(pipeline.norm_stds, np.ndarray) else list(pipeline.norm_stds),
            "source": "train_split_fitted",
        },
        "dataset_hashes": {
            "weather_telemetry_sha256": compute_file_sha256(WEATHER_CSV_PATH),
            "water_level_telemetry_sha256": compute_file_sha256(WATER_CSV_PATH),
        },
        "environment": {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "platform": platform.platform(),
        },
        "training_metrics": {
            "best_epoch": epoch,
            "best_val_loss": round(val_loss, 4),
            "best_val_water_rmse_meters": round(val_water_rmse, 4) if not np.isnan(val_water_rmse) else None,
        },
    }


def train_mf1_model(
    horizon: int = 1,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 0.003,
    seed: int = DEFAULT_SEED,
    max_train_samples: int = None,
    save_path: str = None,
):
    """
    Train Model Family 1 (PyTorch WeatherWaterLNN) on canonical future-window dataset.
    """
    print("=" * 70)
    print(f"Training Model Family 1 (PyTorch WeatherWaterLNN) | Horizon: +{horizon}h")
    print("=" * 70)

    set_reproducibility_seed(seed)
    pipeline = get_telemetry_pipeline()

    if save_path is None:
        save_path = os.path.join(DATA_DIR, f"lnn_weather_water_h{horizon}.pt")

    # Load canonical train and validation splits
    print(f"Loading canonical datasets (horizon={horizon}h, seq_len={DEFAULT_SEQ_LEN})...")
    train_dataset = TelemetryDataset(
        split="train",
        horizon=horizon,
        seq_len=DEFAULT_SEQ_LEN,
        max_samples=max_train_samples,
        pipeline=pipeline,
    )
    val_dataset = TelemetryDataset(
        split="val",
        horizon=horizon,
        seq_len=DEFAULT_SEQ_LEN,
        max_samples=max_train_samples // 3 if max_train_samples else None,
        pipeline=pipeline,
    )

    print(f"  Training samples:   {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")

    if len(train_dataset) == 0:
        raise RuntimeError(f"No valid training samples found for horizon +{horizon}h!")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = {"input_dim": 4, "hidden_dim": 32}
    model = WeatherWaterLNN(**model_config).to(device)

    # Multi-task loss functions
    bce_loss_fn = nn.BCELoss()
    mse_loss_fn = nn.MSELoss(reduction="none")
    huber_loss_fn = nn.SmoothL1Loss(reduction="none")
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_epoch = 0
    best_water_rmse = float("inf")
    best_model_state = None

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

            # Target is evaluated at forecast origin t0 + h (final step in window)
            pred_rain_final = pred_rain[:, -1, :]
            pred_precip_final = pred_precip[:, -1, :]
            pred_water_final = pred_water[:, -1, :]

            loss_rain = bce_loss_fn(pred_rain_final, target_rain)
            loss_precip = mse_loss_fn(pred_precip_final, target_precip).mean()

            # Water loss: only backpropagate when a real gauge observation is present
            water_diff = huber_loss_fn(pred_water_final, target_water)
            water_count = has_water.sum()
            if water_count > 0:
                loss_water = (water_diff * has_water).sum() / water_count
            else:
                loss_water = torch.tensor(0.0, device=device)

            total_loss = loss_rain + 0.1 * loss_precip + 0.5 * loss_water
            total_loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += total_loss.item()

        scheduler.step()
        avg_train_loss = total_train_loss / max(1, len(train_loader))

        # Model validation on held-out validation split ONLY
        model.eval()
        total_val_loss = 0.0
        val_water_sq_sum = 0.0
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

                water_diff = huber_loss_fn(pred_water_final, target_water)
                water_count = has_water.sum()
                if water_count > 0:
                    loss_water = (water_diff * has_water).sum() / water_count
                    for i in range(len(has_water)):
                        if has_water[i].item() > 0.5:
                            diff = pred_water_final[i].item() - target_water[i].item()
                            val_water_sq_sum += diff ** 2
                            val_water_count += 1
                else:
                    loss_water = torch.tensor(0.0, device=device)

                val_loss = loss_rain + 0.1 * loss_precip + 0.5 * loss_water
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / max(1, len(val_loader))
        avg_water_rmse = (val_water_sq_sum / val_water_count) ** 0.5 if val_water_count > 0 else float("nan")

        if epoch % 5 == 0 or epoch == epochs:
            w_str = f"{avg_water_rmse:.4f}m" if not np.isnan(avg_water_rmse) else "N/A"
            print(f"  Epoch [{epoch:02d}/{epochs:02d}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Water RMSE: {w_str}")

        # Checkpoint selection strictly on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            best_water_rmse = avg_water_rmse
            best_model_state = model.state_dict()

    # Save best checkpoint with complete manifest
    manifest = build_checkpoint_manifest(
        model_family="MF-1: PyTorch WeatherWaterLNN",
        seed=seed,
        horizon=horizon,
        epoch=best_epoch,
        val_loss=best_val_loss,
        val_water_rmse=best_water_rmse,
        model_config=model_config,
        pipeline=pipeline,
    )

    checkpoint = {
        "epoch": best_epoch,
        "model_state_dict": best_model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": best_val_loss,
        "water_rmse": best_water_rmse,
        "manifest": manifest,
    }
    torch.save(checkpoint, save_path)
    # Also save to default path if horizon == 1
    if horizon == 1:
        default_path = os.path.join(DATA_DIR, "lnn_weather_water.pt")
        torch.save(checkpoint, default_path)

    print("=" * 70)
    print(f"MF-1 Training Complete! Saved checkpoint -> {save_path}")
    print(f"Best Epoch: {best_epoch} | Best Val Loss: {best_val_loss:.4f}")
    print("=" * 70)
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MF-1 WeatherWaterLNN on canonical future windows.")
    parser.add_argument("--horizon", type=int, default=1, help="Forecast horizon in hours (default: 1)")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs (default: 20)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=0.003, help="Learning rate (default: 0.003)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed (default: 42)")
    parser.add_argument("--max_samples", type=int, default=None, help="Cap training samples (for smoke test)")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save checkpoint")
    args = parser.parse_args()

    train_mf1_model(
        horizon=args.horizon,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        max_train_samples=args.max_samples,
        save_path=args.save_path,
    )
