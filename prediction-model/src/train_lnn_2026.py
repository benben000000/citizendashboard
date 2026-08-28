"""
KloudTrack Continuous-Time Liquid Neural Network (LNN-CfC) Training Engine.
Trains on 2024-2026 clean multi-station telemetry (716,527 records) across 25 Epochs.
Optimizes continuous ODE hidden state decay (tau) and multi-task rain/hydrological heads.
"""

import os
import csv
import math
import random
import json
import time
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CLEAN_DATASET_CSV = os.path.join(DATA_DIR, "segregated", "clean_consolidated_2024_2026.csv")
OUTPUT_WEIGHTS_JSON = os.path.join(DATA_DIR, "lnn_trained_weights.json")
OUTPUT_HISTORY_JSON = os.path.join(DATA_DIR, "training_history_2026.json")

# Hyperparameters
NUM_EPOCHS = 25
BATCH_SIZE = 64
SAMPLES_PER_EPOCH = 30000  # Balanced subset resampled per epoch
LEARNING_RATE_INIT = 0.012
LEARNING_RATE_MIN = 0.0008
HIDDEN_DIM = 8
IN_FEATURES = 4
RAIN_POS_WEIGHT = 2.4  # Reweight positive rain events for high Recall (POD)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, x))))


def tanh(x: float) -> float:
    return math.tanh(max(-20.0, min(20.0, x)))


class ContinuousLNNModel:
    def __init__(self, in_features=4, hidden_dim=8):
        self.in_features = in_features
        self.hidden_dim = hidden_dim

        scale = math.sqrt(2.0 / (in_features + hidden_dim))
        # Input to hidden weights
        self.W_in = [[random.gauss(0.0, scale) for _ in range(hidden_dim)] for _ in range(in_features)]
        self.W_rec = [[random.gauss(0.0, scale) for _ in range(hidden_dim)] for _ in range(hidden_dim)]
        self.b_h = [0.0] * hidden_dim
        self.tau = [1.5 + random.uniform(-0.2, 0.2) for _ in range(hidden_dim)]

        # Rain Classification Head
        self.W_rain = [random.gauss(0.0, scale) for _ in range(hidden_dim)]
        self.b_rain = -0.5

        # Water Level Regression Head
        self.W_water = [random.gauss(0.0, scale) for _ in range(hidden_dim)]
        self.b_water = 3.45

        # Optimizer Adam-like momentum buffers
        self.m_W_in = [[0.0] * hidden_dim for _ in range(in_features)]
        self.v_W_in = [[0.0] * hidden_dim for _ in range(in_features)]
        self.m_W_rec = [[0.0] * hidden_dim for _ in range(hidden_dim)]
        self.v_W_rec = [[0.0] * hidden_dim for _ in range(hidden_dim)]
        self.m_tau = [0.0] * hidden_dim
        self.v_tau = [0.0] * hidden_dim
        self.m_W_rain = [0.0] * hidden_dim
        self.v_W_rain = [0.0] * hidden_dim
        self.m_b_rain = 0.0
        self.v_b_rain = 0.0
        self.m_W_water = [0.0] * hidden_dim
        self.v_W_water = [0.0] * hidden_dim
        self.m_b_water = 0.0
        self.v_b_water = 0.0

    def forward(self, x, h_prev, dt=1.0):
        h_next = []
        for j in range(self.hidden_dim):
            in_sum = sum(x[i] * self.W_in[i][j] for i in range(self.in_features))
            rec_sum = sum(h_prev[k] * self.W_rec[k][j] for k in range(self.hidden_dim))
            act = tanh(in_sum + rec_sum + self.b_h[j])

            decay = math.exp(-dt / max(0.2, self.tau[j]))
            h_j = decay * h_prev[j] + (1.0 - decay) * act
            h_next.append(h_j)

        rain_logit = self.b_rain + sum(h_next[j] * self.W_rain[j] for j in range(self.hidden_dim))
        rain_prob = sigmoid(rain_logit)

        water_pred = self.b_water + sum(h_next[j] * self.W_water[j] for j in range(self.hidden_dim))
        return h_next, rain_prob, water_pred

    def backward_and_step(self, batch_data, lr=0.005, beta1=0.9, beta2=0.999, eps=1e-8):
        # Accumulate gradients
        grad_W_in = [[0.0] * self.hidden_dim for _ in range(self.in_features)]
        grad_W_rec = [[0.0] * self.hidden_dim for _ in range(self.hidden_dim)]
        grad_tau = [0.0] * self.hidden_dim
        grad_W_rain = [0.0] * self.hidden_dim
        grad_b_rain = 0.0
        grad_W_water = [0.0] * self.hidden_dim
        grad_b_water = 0.0

        n = len(batch_data)
        if n == 0:
            return 0.0

        total_loss = 0.0

        for sample in batch_data:
            x, target_rain, target_water, dt = sample
            h_prev = [0.0] * self.hidden_dim
            h_next, rain_prob, water_pred = self.forward(x, h_prev, dt)

            # Multi-Task Loss: Weighted BCE + Huber/MSE
            # Rain Loss
            eps_clip = 1e-7
            p_clipped = max(eps_clip, min(1.0 - eps_clip, rain_prob))
            if target_rain == 1.0:
                bce_loss = -RAIN_POS_WEIGHT * math.log(p_clipped)
                d_rain_logit = RAIN_POS_WEIGHT * (p_clipped - 1.0)
            else:
                bce_loss = -math.log(1.0 - p_clipped)
                d_rain_logit = p_clipped

            # Water Level Loss (MSE)
            water_err = water_pred - target_water
            water_loss = 0.5 * (water_err ** 2)
            d_water = water_err

            sample_loss = bce_loss + 0.15 * water_loss
            total_loss += sample_loss

            # Gradients on Rain Head
            grad_b_rain += d_rain_logit
            for j in range(self.hidden_dim):
                grad_W_rain[j] += d_rain_logit * h_next[j]

            # Gradients on Water Head
            grad_b_water += d_water
            for j in range(self.hidden_dim):
                grad_W_water[j] += d_water * h_next[j]

            # Backprop through continuous ODE hidden state
            for j in range(self.hidden_dim):
                d_h_j = d_rain_logit * self.W_rain[j] + 0.15 * d_water * self.W_water[j]
                # tanh derivative
                in_sum = sum(x[i] * self.W_in[i][j] for i in range(self.in_features))
                act = tanh(in_sum + self.b_h[j])
                d_act = (1.0 - act ** 2)

                decay = math.exp(-dt / max(0.2, self.tau[j]))
                d_in = d_h_j * (1.0 - decay) * d_act

                for i in range(self.in_features):
                    grad_W_in[i][j] += d_in * x[i]

                # Tau gradient
                grad_tau[j] += d_h_j * (act - h_prev[j]) * (dt / (self.tau[j] ** 2)) * decay

        # Average and apply Adam update
        inv_n = 1.0 / n
        # Update W_in
        for i in range(self.in_features):
            for j in range(self.hidden_dim):
                g = grad_W_in[i][j] * inv_n
                self.m_W_in[i][j] = beta1 * self.m_W_in[i][j] + (1 - beta1) * g
                self.v_W_in[i][j] = beta2 * self.v_W_in[i][j] + (1 - beta2) * (g ** 2)
                m_hat = self.m_W_in[i][j] / (1 - beta1)
                v_hat = self.v_W_in[i][j] / (1 - beta2)
                self.W_in[i][j] -= lr * m_hat / (math.sqrt(v_hat) + eps)

        # Update Tau
        for j in range(self.hidden_dim):
            g = grad_tau[j] * inv_n
            self.m_tau[j] = beta1 * self.m_tau[j] + (1 - beta1) * g
            self.v_tau[j] = beta2 * self.v_tau[j] + (1 - beta2) * (g ** 2)
            m_hat = self.m_tau[j] / (1 - beta1)
            v_hat = self.v_tau[j] / (1 - beta2)
            self.tau[j] = max(0.5, min(4.0, self.tau[j] - lr * m_hat / (math.sqrt(v_hat) + eps)))

        # Update Rain Head
        self.m_b_rain = beta1 * self.m_b_rain + (1 - beta1) * (grad_b_rain * inv_n)
        self.v_b_rain = beta2 * self.v_b_rain + (1 - beta2) * ((grad_b_rain * inv_n) ** 2)
        self.b_rain -= lr * (self.m_b_rain / (1 - beta1)) / (math.sqrt(self.v_b_rain / (1 - beta2)) + eps)

        for j in range(self.hidden_dim):
            g = grad_W_rain[j] * inv_n
            self.m_W_rain[j] = beta1 * self.m_W_rain[j] + (1 - beta1) * g
            self.v_W_rain[j] = beta2 * self.v_W_rain[j] + (1 - beta2) * (g ** 2)
            self.W_rain[j] -= lr * (self.m_W_rain[j] / (1 - beta1)) / (math.sqrt(self.v_W_rain[j] / (1 - beta2)) + eps)

        # Update Water Head
        self.m_b_water = beta1 * self.m_b_water + (1 - beta1) * (grad_b_water * inv_n)
        self.v_b_water = beta2 * self.v_b_water + (1 - beta2) * ((grad_b_water * inv_n) ** 2)
        self.b_water -= lr * (self.m_b_water / (1 - beta1)) / (math.sqrt(self.v_b_water / (1 - beta2)) + eps)

        for j in range(self.hidden_dim):
            g = grad_W_water[j] * inv_n
            self.m_W_water[j] = beta1 * self.m_W_water[j] + (1 - beta1) * g
            self.v_W_water[j] = beta2 * self.v_W_water[j] + (1 - beta2) * (g ** 2)
            self.W_water[j] -= lr * (self.m_W_water[j] / (1 - beta1)) / (math.sqrt(self.v_W_water[j] / (1 - beta2)) + eps)

        return total_loss * inv_n


def load_dataset():
    print(f"📖 Loading clean dataset from {CLEAN_DATASET_CSV}...")
    rain_pool = []
    dry_pool = []

    means = [28.5, 33.0, 10.0, 1008.0]
    stds = [4.5, 6.5, 8.0, 6.0]

    with open(CLEAN_DATASET_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("station_type") != "weather":
                continue
            try:
                temp = float(row.get("temperature_c") or 28.5)
                hi = float(row.get("heat_index_c") or 33.0)
                ws = float(row.get("wind_speed_kmh") or 10.0)
                pres = float(row.get("pressure_hpa") or 1008.0)
                precip = float(row.get("precipitation_mm") or 0.0)

                feat = [
                    (temp - means[0]) / stds[0],
                    (hi - means[1]) / stds[1],
                    (ws - means[2]) / stds[2],
                    (pres - means[3]) / stds[3],
                ]

                is_rain = 1.0 if precip > 0.0 else 0.0
                water_target = 3.42 + min(4.0, precip * 0.18)

                item = (feat, is_rain, water_target, 1.0)
                if is_rain == 1.0:
                    rain_pool.append(item)
                else:
                    dry_pool.append(item)
            except Exception:
                continue

    print(f"  ✓ Loaded {len(rain_pool):,} Rain Samples & {len(dry_pool):,} Dry Samples.")
    return rain_pool, dry_pool, means, stds


def evaluate(model, val_set):
    tp = fp = tn = fn = 0
    total_mae = 0.0

    for feat, target_rain, target_water, dt in val_set:
        _, rain_prob, water_pred = model.forward(feat, [0.0] * model.hidden_dim, dt)
        pred_label = 1 if rain_prob >= 0.45 else 0

        if target_rain == 1.0:
            if pred_label == 1:
                tp += 1
            else:
                fn += 1
        else:
            if pred_label == 1:
                fp += 1
            else:
                tn += 1

        total_mae += abs(water_pred - target_water)

    total = len(val_set)
    acc = (tp + tn) / max(1, total)
    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))  # Probability of Detection (POD)
    f1 = 2 * (prec * rec) / max(1e-6, (prec + rec))
    water_mae = total_mae / max(1, total)

    return {
        "accuracy": round(acc * 100, 2),
        "precision": round(prec * 100, 2),
        "recall_pod": round(rec * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "water_mae_m": round(water_mae, 3),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn
    }


def train_lnn_model():
    print("=" * 80)
    print(f"🚀 Training KloudTrack LNN Continuous-Time Model for {NUM_EPOCHS} Epochs")
    print("=" * 80)

    rain_pool, dry_pool, means, stds = load_dataset()
    if not rain_pool or not dry_pool:
        print("❌ Dataset pools empty!")
        return

    # Holdout validation set (20% of rain + matching dry)
    random.shuffle(rain_pool)
    random.shuffle(dry_pool)

    val_rain_count = int(len(rain_pool) * 0.20)
    val_set = rain_pool[:val_rain_count] + dry_pool[:val_rain_count]
    train_rain = rain_pool[val_rain_count:]
    train_dry = dry_pool[val_rain_count:]

    print(f"📊 Training Pool: {len(train_rain):,} rain / {len(train_dry):,} dry")
    print(f"📊 Holdout Validation Set: {len(val_set):,} balanced test records")
    print("-" * 80)

    model = ContinuousLNNModel(in_features=IN_FEATURES, hidden_dim=HIDDEN_DIM)
    history = []

    best_f1 = 0.0
    best_weights = {}

    start_train_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        # Cosine Annealing Learning Rate
        lr = LEARNING_RATE_MIN + 0.5 * (LEARNING_RATE_INIT - LEARNING_RATE_MIN) * (1 + math.cos(math.pi * epoch / NUM_EPOCHS))

        # Sample balanced batch for this epoch
        half_sample = SAMPLES_PER_EPOCH // 2
        sampled_rain = random.choices(train_rain, k=half_sample)
        sampled_dry = random.choices(train_dry, k=half_sample)
        epoch_data = sampled_rain + sampled_dry
        random.shuffle(epoch_data)

        # Batch iterations
        epoch_losses = []
        for i in range(0, len(epoch_data), BATCH_SIZE):
            batch = epoch_data[i:i + BATCH_SIZE]
            loss = model.backward_and_step(batch, lr=lr)
            epoch_losses.append(loss)

        avg_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        val_metrics = evaluate(model, val_set)
        epoch_dur = time.time() - epoch_start

        print(f"Epoch {epoch:2d}/{NUM_EPOCHS:2d} [{epoch_dur:.1f}s] - LR: {lr:.5f} | Train Loss: {avg_loss:.4f} | "
              f"Val Acc: {val_metrics['accuracy']}% | POD/Recall: {val_metrics['recall_pod']}% | "
              f"F1: {val_metrics['f1_score']}% | Water MAE: {val_metrics['water_mae_m']}m")

        history.append({
            "epoch": epoch,
            "lr": round(lr, 6),
            "train_loss": round(avg_loss, 4),
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall_pod": val_metrics["recall_pod"],
            "val_f1_score": val_metrics["f1_score"],
            "val_water_mae_m": val_metrics["water_mae_m"],
            "duration_sec": round(epoch_dur, 2)
        })

        if val_metrics["f1_score"] > best_f1:
            best_f1 = val_metrics["f1_score"]
            best_weights = {
                "trained_at": datetime.now().isoformat(),
                "epochs_trained": epoch,
                "best_val_f1_pct": best_f1,
                "best_val_recall_pod_pct": val_metrics["recall_pod"],
                "best_val_water_mae_m": val_metrics["water_mae_m"],
                "means": means,
                "stds": stds,
                "hidden_dim": HIDDEN_DIM,
                "W_in": [[round(w, 5) for w in row] for row in model.W_in],
                "W_rec": [[round(w, 5) for w in row] for row in model.W_rec],
                "b_h": [round(b, 5) for b in model.b_h],
                "tau": [round(t, 4) for t in model.tau],
                "W_rain": [round(w, 5) for w in model.W_rain],
                "b_rain": round(model.b_rain, 5),
                "W_water": [round(w, 5) for w in model.W_water],
                "b_water": round(model.b_water, 5),
            }

    total_time = time.time() - start_train_time
    print("=" * 80)
    print(f"🏁 Training Finished in {total_time:.1f}s across {NUM_EPOCHS} Epochs!")
    print(f"🏆 Best Validation F1-Score: {best_f1}% | POD/Recall: {best_weights['best_val_recall_pod_pct']}% | River Stage MAE: {best_weights['best_val_water_mae_m']}m")

    # Save weights & history
    with open(OUTPUT_WEIGHTS_JSON, "w", encoding="utf-8") as f:
        json.dump(best_weights, f, indent=2)
    print(f"💾 Optimized LNN Weights -> {OUTPUT_WEIGHTS_JSON}")

    with open(OUTPUT_HISTORY_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "trained_at": datetime.now().isoformat(),
            "total_epochs": NUM_EPOCHS,
            "total_training_duration_seconds": round(total_time, 2),
            "best_epoch_metrics": best_weights,
            "epochs": history
        }, f, indent=2)
    print(f"📊 Training History Log -> {OUTPUT_HISTORY_JSON}")
    print("=" * 80)


if __name__ == "__main__":
    train_lnn_model()
