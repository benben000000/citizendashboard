"""
Training Script for Continuous-Time CfC/LNN Weather & Hydrological Model.
Optimizes multi-task loss:
  L_total = lambda_rain * L_bce + lambda_precip * L_precip + lambda_water * L_water

Note: This trains the PyTorch WeatherWaterLNN model (Model Family 1).
See MODEL_REGISTRY.md for model family details.

Audit fixes applied:
  - max_samples keyword fix (Fix 1)
  - Train-fitted normalization parameters persisted in checkpoint (Fix 2)
  - Reproducibility controls: seeds, SHA256 hashes, environment versions (Fix 3)
  - Untouched test split evaluation recorded in checkpoint manifest (Fix 4)
  - Chronological 60/20/20 train/val/test splits without temporal leakage (Fix 9)
  - Real gauge observations joined for water level, missing masked (Fix 10)
  - Future-forecasting targets at t0 + h (Fix 11)
"""

import sys
import os
import random
import hashlib
import platform
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import TelemetryDataset, FEATURE_MEANS, FEATURE_STDS
from model import WeatherWaterLNN

DEFAULT_SEED = 42


def set_reproducibility_seed(seed: int = DEFAULT_SEED):
    """Seed all pseudo-random number generators for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _dataset_file_hash(filepath: str) -> str:
    """Compute SHA256 hex digest of a dataset file for provenance tracking."""
    if not os.path.exists(filepath):
        return "file_not_found"
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _build_reproducibility_manifest(seed: int, epoch: int, val_loss: float,
                                     water_rmse: float, model_config: dict,
                                     data_dir: str, horizon: int,
                                     norm_means: np.ndarray,
                                     norm_stds: np.ndarray) -> dict:
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
            "means": norm_means.tolist() if isinstance(norm_means, np.ndarray) else list(norm_means),
            "stds": norm_stds.tolist() if isinstance(norm_stds, np.ndarray) else list(norm_stds),
            "source": "train_split_fitted",
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


def evaluate_test_split(model: nn.Module, test_loader: DataLoader, device: torch.device) -> dict:
    """
    Evaluates model on the untouched test split and returns operational metrics.
    """
    model.eval()
    bce_loss_fn = nn.BCELoss()
    mse_loss_fn = nn.MSELoss(reduction="none")
    huber_loss_fn = nn.SmoothL1Loss(reduction="none")

    tp, fp, tn, fn = 0, 0, 0, 0
    brier_sum = 0.0
    water_errors = []
    water_sq_errors = []
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for batch in test_loader:
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
            loss_water = (huber_loss_fn(pred_water_final, target_water) * has_water).sum() / (has_water.sum() + 1e-6)
            loss = loss_rain + 0.1 * loss_precip + 0.5 * loss_water
            total_loss += loss.item() * len(telemetry)
            total_count += len(telemetry)

            for i in range(len(telemetry)):
                p = float(pred_rain_final[i, 0].item())
                y = float(target_rain[i, 0].item())
                brier_sum += (p - y) ** 2
                cls_pred = 1 if p >= 0.5 else 0
                if cls_pred == 1 and y == 1.0:
                    tp += 1
                elif cls_pred == 1 and y == 0.0:
                    fp += 1
                elif cls_pred == 0 and y == 0.0:
                    tn += 1
                else:
                    fn += 1

                if has_water[i, 0].item() > 0.5:
                    w_pred = float(pred_water_final[i, 0].item())
                    w_true = float(target_water[i, 0].item())
                    water_errors.append(abs(w_pred - w_true))
                    water_sq_errors.append((w_pred - w_true) ** 2)

    acc = (tp + tn) / total_count * 100.0 if total_count > 0 else 0.0
    rec = tp / (tp + fn) * 100.0 if (tp + fn) > 0 else 0.0
    prec = tp / (tp + fp) * 100.0 if (tp + fp) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    far = fp / (tp + fp) * 100.0 if (tp + fp) > 0 else 0.0
    csi = tp / (tp + fp + fn) * 100.0 if (tp + fp + fn) > 0 else 0.0
    brier = brier_sum / total_count if total_count > 0 else 0.0
    water_mae = sum(water_errors) / len(water_errors) if water_errors else None
    water_rmse = (sum(water_sq_errors) / len(water_sq_errors)) ** 0.5 if water_sq_errors else None

    return {
        "test_samples": total_count,
        "test_loss": round(total_loss / total_count, 4) if total_count > 0 else 0.0,
        "accuracy": round(acc, 2),
        "recall_pod": round(rec, 2),
        "precision": round(prec, 2),
        "f1_score": round(f1, 2),
        "false_alarm_ratio": round(far, 2),
        "critical_success_index": round(csi, 2),
        "brier_score": round(brier, 4),
        "water_gauge_samples": len(water_errors),
        "water_mae_meters": round(water_mae, 4) if water_mae is not None else None,
        "water_rmse_meters": round(water_rmse, 4) if water_rmse is not None else None,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_lnn_model(
    epochs: int = 25,
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

    # Prepare datasets: fit normalization on train split, pass to val and test (Fix 2 & 9)
    train_dataset = TelemetryDataset(max_samples=800, seq_len=24, split="train", horizon=horizon)
    norm_means = train_dataset.norm_means
    norm_stds = train_dataset.norm_stds

    val_dataset = TelemetryDataset(max_samples=200, seq_len=24, split="val", horizon=horizon,
                                   norm_means=norm_means, norm_stds=norm_stds)
    test_dataset = TelemetryDataset(max_samples=200, seq_len=24, split="test", horizon=horizon,
                                    norm_means=norm_means, norm_stds=norm_stds)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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

            # Use final hidden state output for the forecast at horizon t0 + h
            pred_rain_final = pred_rain[:, -1, :]
            pred_precip_final = pred_precip[:, -1, :]
            pred_water_final = pred_water[:, -1, :]

            loss_rain = bce_loss_fn(pred_rain_final, target_rain)
            loss_precip = mse_loss_fn(pred_precip_final, target_precip).mean()

            # Water loss: only penalize when a real gauge observation is present
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
        avg_train_loss = total_train_loss / len(train_loader)

        # Validation (chronological holdout)
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

                water_diff = huber_loss_fn(pred_water_final, target_water)
                water_count = has_water.sum()
                if water_count > 0:
                    loss_water = (water_diff * has_water).sum() / water_count
                    for i in range(len(has_water)):
                        if has_water[i].item() > 0.5:
                            diff_val = pred_water_final[i].item() - target_water[i].item()
                            val_water_rmse_sum += diff_val ** 2
                            val_water_count += 1
                else:
                    loss_water = torch.tensor(0.0, device=device)

                val_loss = loss_rain + 0.1 * loss_precip + 0.5 * loss_water
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
                norm_means, norm_stds,
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

    # ------------------------------------------------------------------
    # Untouched Test Split Evaluation (Fix 4)
    # ------------------------------------------------------------------
    print("\nEvaluating on UNTOUCHED TEST SPLIT (Out-of-sample holdout)...")
    checkpoint = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate_test_split(model, test_loader, device)

    print("=" * 60)
    print("UNTOUCHED TEST SPLIT EVALUATION (MF-1 PyTorch WeatherWaterLNN):")
    print(f"  Test Samples:               {test_metrics['test_samples']}")
    print(f"  Accuracy:                   {test_metrics['accuracy']}%")
    print(f"  Recall (POD):               {test_metrics['recall_pod']}%")
    print(f"  Precision:                  {test_metrics['precision']}%")
    print(f"  F1-Score:                   {test_metrics['f1_score']}%")
    print(f"  False Alarm Ratio (FAR):    {test_metrics['false_alarm_ratio']}%")
    print(f"  Critical Success Index:     {test_metrics['critical_success_index']}%")
    print(f"  Brier Score:                {test_metrics['brier_score']}")
    if test_metrics["water_mae_meters"] is not None:
        print(f"  Water Stage MAE:            {test_metrics['water_mae_meters']:.4f} m ({test_metrics['water_gauge_samples']} gauge samples)")
        print(f"  Water Stage RMSE:           {test_metrics['water_rmse_meters']:.4f} m")
    else:
        print("  Water Stage:                N/A (no gauge observations in test period)")
    print("=" * 60)

    # Re-save checkpoint with test evaluation included in manifest
    manifest["test_evaluation"] = test_metrics
    checkpoint["manifest"] = manifest
    torch.save(checkpoint, save_path)

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
        train_lnn_model(epochs=20, batch_size=32, horizon=1)
