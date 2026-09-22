"""
Balanced Continuous-Time LNN (CfC) Trainer & Optimizer (Model Family 2).

This is the standalone (zero-dependency on PyTorch) ContinuousLNNCell trainer.
See MODEL_REGISTRY.md for model family details.

Audit fixes applied:
  - Date outlier quarantine (rejects records with year != 2025-2027)
  - Physical bounds sensor check (rejects corrupt hardware spikes)
  - Train-fitted normalization computed strictly on training split
  - Chronological split (not random shuffle)
  - Seed for reproducibility
  - Real water gauge targets with masking
  - Dataset hashes and normalization in saved weights
  - RESEARCH_PROTOTYPE status
"""

import os
import csv
import math
import random
import json
import hashlib
import platform
from datetime import datetime, timezone

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
WEATHER_CSV = os.path.join(DATA_DIR, "weather_telemetry.csv")
WATER_CSV = os.path.join(DATA_DIR, "water_level_telemetry.csv")

DEFAULT_SEED = 42

# Allowed collection date bounds
MIN_VALID_YEAR = 2025
MAX_VALID_YEAR = 2027

# Physically plausible sensor ranges
PHYSICAL_BOUNDS = {
    "temperature": (10.0, 50.0),
    "heat_index": (10.0, 70.0),
    "wind_speed": (0.0, 180.0),
    "pressure": (900.0, 1050.0),
    "precipitation": (0.0, 300.0),
}


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, x))))


def tanh(x):
    return math.tanh(max(-20.0, min(20.0, x)))


def _file_hash(filepath: str) -> str:
    if not os.path.exists(filepath):
        return "file_not_found"
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
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
                ts = row.get("recorded_at", "")
                if not (ts.startswith("2026") or ts.startswith("2025") or ts.startswith("2027")):
                    continue
                ts_key = ts[:16]
                wl_m = row.get("water_level_m")
                wl_cm = row.get("water_level_cm")
                if wl_m is not None and wl_m != "":
                    wl = float(wl_m)
                elif wl_cm is not None and wl_cm != "":
                    wl = float(wl_cm) / 100.0
                else:
                    continue
                lookup[ts_key] = wl
            except (ValueError, TypeError):
                continue
    return lookup


def load_chronological_dataset(samples_per_class=4000, train_frac=0.8, seed=DEFAULT_SEED):
    """
    Load dataset with CHRONOLOGICAL split (not random shuffle).
    Applies data quality filtering and fits normalization strictly on train split.
    """
    random.seed(seed)

    if not os.path.exists(WEATHER_CSV):
        return [], [], [27.2, 31.0, 2.0, 1004.5], [2.8, 6.3, 3.0, 4.6]

    water_gauge = _load_water_gauge_lookup()

    all_raw_rows = []
    with open(WEATHER_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts = row.get("recorded_at", "")
                if not (ts.startswith("2026") or ts.startswith("2025") or ts.startswith("2027")):
                    continue
                t = float(row.get("temperature") or 28.5)
                hi = float(row.get("heat_index") or 33.0)
                ws = float(row.get("wind_speed") or 10.0)
                p = float(row.get("pressure") or 1008.0)
                precip = float(row.get("precipitation") or 0.0)

                # Physical bounds check
                if not (
                    PHYSICAL_BOUNDS["temperature"][0] <= t <= PHYSICAL_BOUNDS["temperature"][1] and
                    PHYSICAL_BOUNDS["heat_index"][0] <= hi <= PHYSICAL_BOUNDS["heat_index"][1] and
                    PHYSICAL_BOUNDS["wind_speed"][0] <= ws <= PHYSICAL_BOUNDS["wind_speed"][1] and
                    PHYSICAL_BOUNDS["pressure"][0] <= p <= PHYSICAL_BOUNDS["pressure"][1] and
                    PHYSICAL_BOUNDS["precipitation"][0] <= precip <= PHYSICAL_BOUNDS["precipitation"][1]
                ):
                    continue

                ts_key = ts[:16]
                real_water = water_gauge.get(ts_key)
                all_raw_rows.append((ts, [t, hi, ws, p], precip, real_water))
            except Exception:
                continue

    # Chronological sort
    all_raw_rows.sort(key=lambda x: x[0])

    # Fit normalization strictly on training split
    split_idx = int(len(all_raw_rows) * train_frac)
    train_raw = all_raw_rows[:split_idx]
    val_raw = all_raw_rows[split_idx:]

    train_feats = [r[1] for r in train_raw]
    means = [sum(f[i] for f in train_feats) / len(train_feats) for i in range(4)]
    stds = [math.sqrt(sum((f[i] - means[i]) ** 2 for f in train_feats) / len(train_feats)) for i in range(4)]
    stds = [max(1e-4, s) for s in stds]

    def _normalize_and_label(raw_list):
        formatted = []
        for ts, raw_f, precip, real_water in raw_list:
            norm_f = [(raw_f[i] - means[i]) / stds[i] for i in range(4)]
            rain_target = 1.0 if precip > 0.1 else 0.0
            water_target = real_water if real_water is not None else 3.45
            has_real = real_water is not None
            label = "rain" if precip > 0.1 else "dry"
            formatted.append((ts, norm_f, rain_target, water_target, precip, has_real, label))
        return formatted

    train_formatted = _normalize_and_label(train_raw)
    val_formatted = _normalize_and_label(val_raw)

    def _balance_and_format(rows, max_per_class):
        rain = [(r[1], r[2], r[3], r[4], r[5]) for r in rows if r[6] == "rain"]
        dry = [(r[1], r[2], r[3], r[4], r[5]) for r in rows if r[6] == "dry"]
        random.shuffle(rain)
        random.shuffle(dry)
        selected = rain[:max_per_class] + dry[:max_per_class]
        random.shuffle(selected)
        return selected

    train_data = _balance_and_format(train_formatted, samples_per_class)
    val_data = _balance_and_format(val_formatted, samples_per_class // 2)

    return train_data, val_data, means, stds


def train_balanced(seed=DEFAULT_SEED):
    print("=" * 70)
    print("Training Balanced Continuous-Time CfC/LNN Model (Model Family 2)")
    print("=" * 70)

    train_data, val_data, norm_means, norm_stds = load_chronological_dataset(
        samples_per_class=4000, train_frac=0.8, seed=seed
    )
    print(f"Chronological Split (train first 80% by time, val last 20%)")
    print(f"   Training Set:   {len(train_data):,} balanced samples")
    print(f"   Validation Set: {len(val_data):,} balanced samples")
    print(f"   Fitted Means:   {[round(m, 2) for m in norm_means]}")
    print(f"   Fitted Stds:    {[round(s, 2) for s in norm_stds]}")

    model = ContinuousLNNCell(in_features=4, hidden_dim=8)
    lr = 0.02
    epochs = 20

    for epoch in range(1, epochs + 1):
        train_loss = 0.0
        h = [0.0] * model.hidden_dim

        for feat, target_rain, target_water, precip, has_real_water in train_data:
            h, pred_rain, pred_water = model.forward_step(feat, h, dt=1.0)

            # Rain loss & gradient
            err_rain = pred_rain - target_rain
            loss_rain = - (target_rain * math.log(max(1e-7, pred_rain)) + (1.0 - target_rain) * math.log(max(1e-7, 1.0 - pred_rain)))
            d_rain = err_rain

            # Water loss & gradient
            if has_real_water:
                err_water = pred_water - target_water
                d_water = err_water * 0.1
                loss_water = 0.5 * (err_water ** 2)
            else:
                d_water = 0.0
                loss_water = 0.0

            total_sample_loss = loss_rain + 0.5 * loss_water
            train_loss += total_sample_loss

            # Update heads
            model.b_rain -= lr * d_rain
            for j in range(model.hidden_dim):
                model.W_rain[j] -= lr * d_rain * h[j]

            if has_real_water:
                model.b_water -= lr * d_water
                for j in range(model.hidden_dim):
                    model.W_water[j] -= lr * d_water * h[j]

            # Hidden state gradients
            dh = [d_rain * model.W_rain[j] + (d_water * model.W_water[j] if has_real_water else 0.0)
                  for j in range(model.hidden_dim)]

            for j in range(model.hidden_dim):
                decay = math.exp(-1.0 / max(0.1, model.tau[j]))
                d_act = dh[j] * (1.0 - decay) * (1.0 - h[j] ** 2)
                model.b_h[j] -= lr * d_act * 0.05
                for i in range(model.in_features):
                    model.W_in[i][j] -= lr * d_act * feat[i] * 0.05

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

    # Save balanced weights with reproducibility manifest and train-fitted normalization
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
        "normalization": {
            "means": norm_means,
            "stds": norm_stds,
            "source": "train_split_fitted",
        },
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
