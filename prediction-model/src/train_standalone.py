"""
Balanced Continuous-Time LNN (CfC) Trainer & Optimizer (Model Family 2).

This is the standalone (zero-dependency on PyTorch) ContinuousLNNCell trainer.
See MODEL_REGISTRY.md for model family details.

Audit fixes applied:
  - Chronological train/val split (not random shuffle) (Critical)
  - Random seed for reproducibility (Medium)
  - Real water-level gauge targets where available (Critical)
  - Rain threshold changed to > 0.1mm to match dataset.py (consistency)
  - Dataset hash and split info recorded in saved weights (Medium)
"""

import os
import csv
import math
import random
import json
import hashlib
import platform
from datetime import datetime, timezone
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
WEATHER_CSV = os.path.join(DATA_DIR, "weather_telemetry.csv")
WATER_CSV = os.path.join(DATA_DIR, "water_level_telemetry.csv")

MEANS = [28.5, 33.0, 10.0, 1008.0]
STDS = [4.5, 6.5, 8.0, 6.0]

DEFAULT_SEED = 42


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, x))))


def tanh(x: float) -> float:
    return math.tanh(max(-20.0, min(20.0, x)))


def _file_hash(path: str) -> str:
    """SHA-256 hash of a file for reproducibility."""
    if not os.path.exists(path):
        return "FILE_NOT_FOUND"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ContinuousLNNCell:
    def __init__(self, in_features=4, hidden_dim=8):
        self.in_features = in_features
        self.hidden_dim = hidden_dim

        scale = math.sqrt(2.0 / (in_features + hidden_dim))
        # Input to hidden weights
        self.W_in = [[random.uniform(-scale, scale) for _ in range(hidden_dim)] for _ in range(in_features)]
        self.W_rec = [[random.uniform(-scale, scale) for _ in range(hidden_dim)] for _ in range(hidden_dim)]
        self.b_h = [0.0] * hidden_dim

        # Decay gate
        self.tau = [1.5] * hidden_dim

        # Output heads
        self.W_rain = [random.uniform(-scale, scale) for _ in range(hidden_dim)]
        self.b_rain = 0.0

        self.W_water = [random.uniform(-scale, scale) for _ in range(hidden_dim)]
        self.b_water = 3.45

    def forward_step(self, x, h_prev, dt=1.0):
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


def _load_water_gauge_lookup():
    """Load real water-level observations for gauge-matched validation."""
    lookup = {}
    if not os.path.exists(WATER_CSV):
        return lookup
    with open(WATER_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts = row.get("recorded_at", "")[:16]  # Minute precision
                wl_m = row.get("water_level_m")
                wl_cm = row.get("water_level_cm")
                if wl_m is not None and wl_m != "":
                    wl = float(wl_m)
                elif wl_cm is not None and wl_cm != "":
                    wl = float(wl_cm) / 100.0
                else:
                    continue
                lookup[ts] = wl
            except (ValueError, TypeError):
                continue
    return lookup


def load_chronological_dataset(samples_per_class=4000, train_frac=0.8, seed=DEFAULT_SEED):
    """
    Load dataset with CHRONOLOGICAL split (not random shuffle).

    The data is sorted by timestamp. The first train_frac of time-ordered
    data is used for training, and the remaining for validation. Within
    each split, we still do balanced sampling for rain/dry.

    Returns (train_data, val_data) where each is a list of tuples:
        (feat, target_rain, target_water, precip, has_real_water)
    """
    random.seed(seed)

    if not os.path.exists(WEATHER_CSV):
        return [], []

    # Load real water gauge data
    water_gauge = _load_water_gauge_lookup()

    # Load all rows with timestamps for chronological ordering
    all_rows = []
    with open(WEATHER_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts = row.get("recorded_at", "")
                t = float(row.get("temperature") or 28.5)
                hi = float(row.get("heat_index") or 33.0)
                ws = float(row.get("wind_speed") or 10.0)
                p = float(row.get("pressure") or 1008.0)
                precip = float(row.get("precipitation") or 0.0)

                feat = [
                    (t - MEANS[0]) / STDS[0],
                    (hi - MEANS[1]) / STDS[1],
                    (ws - MEANS[2]) / STDS[2],
                    (p - MEANS[3]) / STDS[3],
                ]

                # Real water target from gauge (if available)
                ts_key = ts[:16]
                real_water = water_gauge.get(ts_key)

                if precip > 0.1:
                    rain_target = 1.0
                    water_t = real_water if real_water is not None else 3.45
                    has_real = real_water is not None
                    all_rows.append((ts, feat, rain_target, water_t, precip, has_real, "rain"))
                else:
                    rain_target = 0.0
                    water_t = real_water if real_water is not None else 3.45
                    has_real = real_water is not None
                    all_rows.append((ts, feat, rain_target, water_t, 0.0, has_real, "dry"))
            except Exception:
                continue

    # Sort by timestamp (chronological order)
    all_rows.sort(key=lambda x: x[0])

    # Chronological split
    split_idx = int(len(all_rows) * train_frac)
    train_rows = all_rows[:split_idx]
    val_rows = all_rows[split_idx:]

    def _balance_and_format(rows, max_per_class):
        rain = [(r[1], r[2], r[3], r[4], r[5]) for r in rows if r[6] == "rain"]
        dry = [(r[1], r[2], r[3], r[4], r[5]) for r in rows if r[6] == "dry"]
        random.shuffle(rain)
        random.shuffle(dry)
        selected = rain[:max_per_class] + dry[:max_per_class]
        random.shuffle(selected)
        return selected

    train_data = _balance_and_format(train_rows, samples_per_class)
    val_data = _balance_and_format(val_rows, samples_per_class // 2)

    return train_data, val_data


def train_balanced(seed=DEFAULT_SEED):
    print("=" * 70)
    print("Training Balanced Continuous-Time CfC/LNN Model (Model Family 2)")
    print("=" * 70)

    random.seed(seed)

    train_data, val_data = load_chronological_dataset(samples_per_class=4000, seed=seed)
    if not train_data:
        print("ERROR: Dataset not found.")
        return

    print(f"Chronological Split (train first 80% by time, val last 20%)")
    print(f"   Training Set:   {len(train_data):,} balanced samples")
    print(f"   Validation Set: {len(val_data):,} balanced samples")

    model = ContinuousLNNCell(in_features=4, hidden_dim=8)
    lr = 0.015
    epochs = 20

    for epoch in range(1, epochs + 1):
        h = [0.0] * model.hidden_dim
        train_loss = 0.0

        for feat, target_rain, target_water, _, has_real_water in train_data:
            h, pred_rain, pred_water = model.forward_step(feat, h, dt=1.0)

            # Weighted BCE loss
            eps = 1e-7
            bce = -(target_rain * math.log(max(eps, pred_rain)) + (1.0 - target_rain) * math.log(max(eps, 1.0 - pred_rain)))

            # Water loss: only use real gauge targets
            if has_real_water:
                mse_water = (pred_water - target_water) ** 2
            else:
                mse_water = 0.0

            loss = bce + 0.8 * mse_water
            train_loss += loss

            # Gradients for Output Heads
            grad_rain = pred_rain - target_rain
            for j in range(model.hidden_dim):
                model.W_rain[j] -= lr * grad_rain * h[j]
                model.W_in[0][j] -= lr * grad_rain * 0.01
                model.W_in[3][j] -= lr * grad_rain * 0.01
            model.b_rain -= lr * grad_rain

            if has_real_water:
                grad_water = 2.0 * (pred_water - target_water)
                for j in range(model.hidden_dim):
                    model.W_water[j] -= lr * grad_water * h[j] * 0.05
                model.b_water -= lr * grad_water * 0.02

        # Validation
        tp = fp = tn = fn = 0
        mae_sum = 0.0
        water_count = 0
        h_val = [0.0] * model.hidden_dim

        for feat, target_rain, target_water, _, has_real_water in val_data:
            h_val, pred_rain, pred_water = model.forward_step(feat, h_val, dt=1.0)
            p_class = 1 if pred_rain >= 0.5 else 0
            t_class = int(target_rain)

            if p_class == 1 and t_class == 1:
                tp += 1
            elif p_class == 1 and t_class == 0:
                fp += 1
            elif p_class == 0 and t_class == 0:
                tn += 1
            elif p_class == 0 and t_class == 1:
                fn += 1

            if has_real_water:
                mae_sum += abs(pred_water - target_water)
                water_count += 1

        acc = (tp + tn) / len(val_data) * 100.0 if val_data else 0.0
        rec = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
        prec = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        mae = mae_sum / water_count if water_count > 0 else float("nan")

        if epoch % 5 == 0 or epoch == epochs:
            water_str = f"{mae:.3f}m (n={water_count})" if water_count > 0 else "N/A (no gauge data)"
            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] "
                f"| Loss: {train_loss/len(train_data):.4f} "
                f"| Acc: {acc:.1f}% "
                f"| Recall: {rec:.1f}% "
                f"| Precision: {prec:.1f}% "
                f"| F1: {f1:.1f}% "
                f"| Water MAE: {water_str}"
            )

    # Save balanced weights with reproducibility manifest
    weights_path = os.path.join(DATA_DIR, "lnn_trained_weights.json")
    weights = {
        "model_family": "ContinuousLNNCell (Model Family 2 — standalone trainer)",
        "model_status": "RESEARCH_PROTOTYPE",
        "hidden_dim": model.hidden_dim,
        "W_in": model.W_in,
        "W_rec": model.W_rec,
        "b_h": model.b_h,
        "tau": model.tau,
        "W_rain": model.W_rain,
        "b_rain": model.b_rain,
        "W_water": model.W_water,
        "b_water": model.b_water,
        "validation_accuracy": f"{acc:.1f}%",
        "validation_recall": f"{rec:.1f}%",
        "validation_precision": f"{prec:.1f}%",
        "validation_f1": f"{f1:.1f}%",
        "validation_mae_water": f"{mae:.3f}m" if water_count > 0 else "N/A",
        "water_gauge_matched_samples": water_count,
        "split_method": "chronological_80_20",
        "reproducibility": {
            "seed": seed,
            "training_date": datetime.now(timezone.utc).isoformat(),
            "weather_csv_sha256": _file_hash(WEATHER_CSV),
            "water_csv_sha256": _file_hash(WATER_CSV),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "train_samples": len(train_data),
            "val_samples": len(val_data),
        },
    }
    with open(weights_path, "w", encoding="utf-8") as f:
        json.dump(weights, f, indent=2)

    print("=" * 70)
    print(f"Training Complete! Saved -> {weights_path}")
    print("=" * 70)


if __name__ == "__main__":
    train_balanced(seed=DEFAULT_SEED)
