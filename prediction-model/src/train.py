"""
Training Script for LNN Weather & Hydrological Forecast Model.
Optimizes multi-task loss:
  L_total = lambda_rain * L_bce + lambda_precip * L_precip + lambda_water * L_water
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import WeatherWaterLNN
from dataset import TelemetryDataset


def train_lnn_model(
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 0.003,
    save_path: str = "lnn_weather_water.pt"
):
    print("=" * 60)
    print("Initializing Liquid Neural Network (LNN / CfC) Training Pipeline...")
    print("=" * 60)

    # Prepare datasets
    train_dataset = TelemetryDataset(num_samples=600, seq_len=24)
    val_dataset = TelemetryDataset(num_samples=150, seq_len=24)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WeatherWaterLNN(input_dim=4, hidden_dim=32).to(device)

    # Losses & Optimizer
    bce_loss_fn = nn.BCELoss()
    mse_loss_fn = nn.MSELoss()
    huber_loss_fn = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0

        for batch in train_loader:
            telemetry = batch["telemetry"].to(device)
            dt = batch["dt"].to(device)
            target_rain = batch["rain_prob"].to(device)
            target_precip = batch["precip_mm"].to(device)
            target_water = batch["water_level"].to(device)

            optimizer.zero_grad()

            pred_rain, pred_precip, pred_water = model(telemetry, dt)

            loss_rain = bce_loss_fn(pred_rain, target_rain)
            loss_precip = mse_loss_fn(pred_precip, target_precip)
            loss_water = huber_loss_fn(pred_water, target_water)

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
        val_water_rmse = 0.0

        with torch.no_grad():
            for batch in val_loader:
                telemetry = batch["telemetry"].to(device)
                dt = batch["dt"].to(device)
                target_rain = batch["rain_prob"].to(device)
                target_precip = batch["precip_mm"].to(device)
                target_water = batch["water_level"].to(device)

                pred_rain, pred_precip, pred_water = model(telemetry, dt)

                loss_rain = bce_loss_fn(pred_rain, target_rain)
                loss_precip = mse_loss_fn(pred_precip, target_precip)
                loss_water = huber_loss_fn(pred_water, target_water)

                val_loss = 1.0 * loss_rain + 0.2 * loss_precip + 2.0 * loss_water
                total_val_loss += val_loss.item()
                val_water_rmse += torch.sqrt(mse_loss_fn(pred_water, target_water)).item()

        avg_val_loss = total_val_loss / len(val_loader)
        avg_water_rmse = val_water_rmse / len(val_loader)

        if epoch % 5 == 0 or epoch == epochs:
            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] "
                f"| Train Loss: {avg_train_loss:.4f} "
                f"| Val Loss: {avg_val_loss:.4f} "
                f"| Water RMSE: {avg_water_rmse:.3f}m"
            )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": avg_val_loss,
                    "water_rmse": avg_water_rmse,
                },
                save_path,
            )

    print("=" * 60)
    print(f"Training Complete! Best model saved to: {os.path.abspath(save_path)}")
    print("=" * 60)
    return model


if __name__ == "__main__":
    train_lnn_model(epochs=25, batch_size=32)
