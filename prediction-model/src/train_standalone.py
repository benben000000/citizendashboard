"""
Standalone Continuous-Time LNN (CfC) Trainer (Model Family 2).

Trains standalone ContinuousLNNCell strictly on canonical future-window forecast samples:
  - Consumes exact same seq_len (24) hourly input windows as MF-1.
  - Resets recurrent hidden state h=0 at the start of each window sample (no cross-window leakage).
  - Evaluates loss at future horizon t0 + h.
  - Masks river-stage water loss when real gauge target is unobserved.
  - Evaluates validation loss on chronological validation split for checkpointing.
  - Saves weights and full metadata manifest in lnn_trained_weights.json.
"""

import os
import sys
import math
import random
import json
import argparse
import platform
from datetime import datetime, timezone

import numpy as np

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dataset import (
    get_telemetry_pipeline,
    build_forecast_windows,
    compute_file_sha256,
    DATA_DIR,
    WEATHER_CSV_PATH,
    WATER_CSV_PATH,
    DEFAULT_SEQ_LEN,
)

DEFAULT_SEED = 42


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, x))))


def tanh(x: float) -> float:
    return math.tanh(max(-20.0, min(20.0, x)))


class ContinuousLNNCell:
    """Standalone continuous-time recurrent liquid cell."""

    def __init__(self, in_features: int = 4, hidden_dim: int = 8, seed: int = DEFAULT_SEED):
        random.seed(seed)
        self.in_features = in_features
        self.hidden_dim = hidden_dim

        scale = math.sqrt(2.0 / (in_features + hidden_dim))
        self.W_in = [[random.uniform(-scale, scale) for _ in range(hidden_dim)] for _ in range(in_features)]
        self.W_rec = [[random.uniform(-scale, scale) for _ in range(hidden_dim)] for _ in range(hidden_dim)]
        self.b_h = [0.0] * hidden_dim
        self.tau = [1.5] * hidden_dim

        self.W_rain = [random.uniform(-scale, scale) for _ in range(hidden_dim)]
        self.b_rain = 0.0

        self.W_water = [random.uniform(-scale, scale) for _ in range(hidden_dim)]
        self.b_water = 2.50  # Prior centered around mean river stage (2.5m)

    def forward_step(self, x: list, h_prev: list, dt: float = 1.0):
        """Single-step continuous ODE transition."""
        h_next = []
        for j in range(self.hidden_dim):
            in_sum = sum(x[i] * self.W_in[i][j] for i in range(self.in_features))
            rec_sum = sum(h_prev[k] * self.W_rec[k][j] for k in range(self.hidden_dim))
            act = tanh(in_sum + rec_sum + self.b_h[j])

            decay = math.exp(-dt / max(0.1, self.tau[j]))
            h_j = decay * h_prev[j] + (1.0 - decay) * act
            h_next.append(h_j)

        rain_logit = self.b_rain + sum(h_next[j] * self.W_rain[j] for j in range(self.hidden_dim))
        rain_prob = sigmoid(rain_logit)

        water_pred = self.b_water + sum(h_next[j] * self.W_water[j] for j in range(self.hidden_dim))
        return h_next, rain_prob, water_pred

    def unroll_window(self, telemetry_arr: np.ndarray, dt_arr: np.ndarray):
        """Unroll entire sequence window starting from fresh hidden state h=0."""
        seq_len = telemetry_arr.shape[0]
        h = [0.0] * self.hidden_dim
        history_h = []
        pred_rain = 0.0
        pred_water = self.b_water

        for t in range(seq_len):
            x = telemetry_arr[t].tolist()
            dt_val = float(dt_arr[t, 0])
            h, pred_rain, pred_water = self.forward_step(x, h, dt=dt_val)
            history_h.append(h)

        return history_h, pred_rain, pred_water

    def copy_weights(self):
        """Deep copy weights for checkpointing."""
        return {
            "W_in": [row[:] for row in self.W_in],
            "W_rec": [row[:] for row in self.W_rec],
            "b_h": self.b_h[:],
            "tau": self.tau[:],
            "W_rain": self.W_rain[:],
            "b_rain": self.b_rain,
            "W_water": self.W_water[:],
            "b_water": self.b_water,
        }

    def load_weights(self, d: dict):
        self.W_in = [row[:] for row in d["W_in"]]
        self.W_rec = [row[:] for row in d["W_rec"]]
        self.b_h = d["b_h"][:]
        self.tau = d["tau"][:]
        self.W_rain = d["W_rain"][:]
        self.b_rain = d["b_rain"]
        self.W_water = d["W_water"][:]
        self.b_water = d["b_water"]


def train_mf2_model(
    horizon: int = 1,
    epochs: int = 20,
    lr: float = 0.015,
    seed: int = DEFAULT_SEED,
    max_train_samples: int = None,
    save_path: str = None,
):
    """
    Train Model Family 2 (ContinuousLNNCell) on canonical future windows.
    """
    print("=" * 70)
    print(f"Training Model Family 2 (ContinuousLNNCell) | Horizon: +{horizon}h")
    print("=" * 70)

    random.seed(seed)
    np.random.seed(seed)
    pipeline = get_telemetry_pipeline()

    if save_path is None:
        save_path = os.path.join(DATA_DIR, f"lnn_trained_weights_h{horizon}.json")

    # Load canonical future-window tensors
    print(f"Loading canonical dataset windows (horizon={horizon}h, seq_len={DEFAULT_SEQ_LEN})...")
    train_res = build_forecast_windows(
        pipeline=pipeline,
        split="train",
        horizon=horizon,
        seq_len=DEFAULT_SEQ_LEN,
        max_samples=max_train_samples,
    )
    val_res = build_forecast_windows(
        pipeline=pipeline,
        split="val",
        horizon=horizon,
        seq_len=DEFAULT_SEQ_LEN,
        max_samples=max_train_samples // 3 if max_train_samples else None,
    )

    if train_res is None or val_res is None:
        raise RuntimeError("Failed to build canonical windows for MF-2 training!")

    train_telemetry, train_dt, train_rain, train_precip, train_water, train_has_water = train_res
    val_telemetry, val_dt, val_rain, val_precip, val_water, val_has_water = val_res

    num_train = train_telemetry.shape[0]
    num_val = val_telemetry.shape[0]
    print(f"  Training samples:   {num_train}")
    print(f"  Validation samples: {num_val}")

    model = ContinuousLNNCell(in_features=4, hidden_dim=8, seed=seed)

    best_val_loss = float("inf")
    best_epoch = 0
    best_water_rmse = float("inf")
    best_weights = None

    indices = list(range(num_train))

    for epoch in range(1, epochs + 1):
        random.shuffle(indices)
        train_loss = 0.0

        for idx in indices:
            telemetry_arr = train_telemetry[idx].numpy()
            dt_arr = train_dt[idx].numpy()
            target_rain = float(train_rain[idx, 0])
            target_water = float(train_water[idx, 0])
            has_water = bool(train_has_water[idx, 0] > 0.5)

            # Unroll window with fresh hidden state h=0
            history_h, pred_rain, pred_water = model.unroll_window(telemetry_arr, dt_arr)
            h_final = history_h[-1]
            h_prev = history_h[-2] if len(history_h) > 1 else [0.0] * model.hidden_dim
            x_final = telemetry_arr[-1].tolist()
            dt_final = float(dt_arr[-1, 0])

            # Loss computation
            err_rain = pred_rain - target_rain
            loss_rain = -(target_rain * math.log(max(1e-7, pred_rain)) + (1.0 - target_rain) * math.log(max(1e-7, 1.0 - pred_rain)))

            if has_water:
                err_water = pred_water - target_water
                loss_water = 0.5 * (err_water ** 2)
                d_water = 0.1 * err_water
            else:
                loss_water = 0.0
                d_water = 0.0

            total_loss = loss_rain + 0.5 * loss_water
            train_loss += total_loss

            # Gradient updates for output heads
            d_rain = err_rain
            model.b_rain -= lr * d_rain
            for j in range(model.hidden_dim):
                model.W_rain[j] -= lr * d_rain * h_final[j]

            if has_water:
                model.b_water -= lr * d_water
                for j in range(model.hidden_dim):
                    model.W_water[j] -= lr * d_water * h_final[j]

            # Backpropagation into final cell state
            dh = [d_rain * model.W_rain[j] + (d_water * model.W_water[j] if has_water else 0.0)
                  for j in range(model.hidden_dim)]

            for j in range(model.hidden_dim):
                decay = math.exp(-dt_final / max(0.1, model.tau[j]))
                d_act = dh[j] * (1.0 - decay) * (1.0 - h_final[j] ** 2)
                model.b_h[j] -= lr * d_act * 0.05
                for i in range(model.in_features):
                    model.W_in[i][j] -= lr * d_act * x_final[i] * 0.05
                for k in range(model.hidden_dim):
                    model.W_rec[k][j] -= lr * d_act * h_prev[k] * 0.05

        # Validation on held-out validation split ONLY
        val_loss_sum = 0.0
        val_water_sq_sum = 0.0
        val_water_count = 0

        for v_idx in range(num_val):
            v_telemetry = val_telemetry[v_idx].numpy()
            v_dt = val_dt[v_idx].numpy()
            v_rain_true = float(val_rain[v_idx, 0])
            v_water_true = float(val_water[v_idx, 0])
            v_has_water = bool(val_has_water[v_idx, 0] > 0.5)

            _, v_pred_rain, v_pred_water = model.unroll_window(v_telemetry, v_dt)

            v_l_rain = -(v_rain_true * math.log(max(1e-7, v_pred_rain)) + (1.0 - v_rain_true) * math.log(max(1e-7, 1.0 - v_pred_rain)))
            if v_has_water:
                v_err_w = v_pred_water - v_water_true
                v_l_water = 0.5 * (v_err_w ** 2)
                val_water_sq_sum += v_err_w ** 2
                val_water_count += 1
            else:
                v_l_water = 0.0

            val_loss_sum += v_l_rain + 0.5 * v_l_water

        avg_val_loss = val_loss_sum / max(1, num_val)
        avg_val_rmse = math.sqrt(val_water_sq_sum / val_water_count) if val_water_count > 0 else float("nan")

        if epoch % 5 == 0 or epoch == epochs:
            w_str = f"{avg_val_rmse:.4f}m" if not math.isnan(avg_val_rmse) else "N/A"
            print(f"  Epoch [{epoch:02d}/{epochs:02d}] Train Loss: {train_loss/num_train:.4f} | Val Loss: {avg_val_loss:.4f} | Val Water RMSE: {w_str}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            best_water_rmse = avg_val_rmse
            best_weights = model.copy_weights()

    # Git commit hash
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

    # Build manifest identical to MF-1
    manifest = {
        "model_family": "MF-2: ContinuousLNNCell (Standalone)",
        "model_status": "RESEARCH_PROTOTYPE",
        "code_commit": git_commit,
        "training_date": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "forecast_horizon_hours": horizon,
        "feature_schema": ["temperature", "heat_index", "wind_speed", "pressure"],
        "input_sequence_length_hours": DEFAULT_SEQ_LEN,
        "resampling_rule": "UTC hourly bins: last valid temp/hi/ws/pressure, sum of precip volume (mm), last valid water stage (m)",
        "model_config": {"in_features": 4, "hidden_dim": model.hidden_dim},
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
            "numpy_version": np.__version__,
            "platform": platform.platform(),
        },
        "training_metrics": {
            "best_epoch": best_epoch,
            "best_val_loss": round(best_val_loss, 4),
            "best_val_water_rmse_meters": round(best_water_rmse, 4) if not math.isnan(best_water_rmse) else None,
        },
    }

    output_payload = {
        "manifest": manifest,
        "hidden_dim": model.hidden_dim,
        **best_weights,
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    # Save to default path if horizon == 1
    if horizon == 1:
        default_path = os.path.join(DATA_DIR, "lnn_trained_weights.json")
        with open(default_path, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=2)

    print("=" * 70)
    print(f"MF-2 Training Complete! Saved weights -> {save_path}")
    print(f"Best Epoch: {best_epoch} | Best Val Loss: {best_val_loss:.4f}")
    print("=" * 70)
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MF-2 ContinuousLNNCell on canonical future windows.")
    parser.add_argument("--horizon", type=int, default=1, help="Forecast horizon in hours (default: 1)")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs (default: 20)")
    parser.add_argument("--lr", type=float, default=0.015, help="Learning rate (default: 0.015)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed (default: 42)")
    parser.add_argument("--max_samples", type=int, default=None, help="Cap training samples (for smoke test)")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save weights")
    args = parser.parse_args()

    train_mf2_model(
        horizon=args.horizon,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        max_train_samples=args.max_samples,
        save_path=args.save_path,
    )
