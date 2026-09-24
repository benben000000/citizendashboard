"""
Evaluation and Comparison Script for Experimental Two-Stage Precipitation Head.

Compares:
  - Baseline Single-Head (current production GarciaWeatherLNN)
  - Experimental Two-Stage Architecture (occurrence logit + conditional amount)
on held-out telemetry observations across standard meteorological metrics:
  - overall MAE/RMSE/bias
  - dry-hour MAE/RMSE/bias
  - rainy-hour MAE/RMSE/bias
  - heavy-rain precision, recall (POD), and CSI at 2.5, 5.0, and 10.0 mm/h.

Usage:
  python prediction-model/src/evaluate_two_stage_precipitation.py
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

DATA_DIR = os.path.join(os.path.dirname(SRC_DIR), "data")

from dataset import load_real_telemetry_sequences
from model import GarciaWeatherLNN, TwoStagePrecipitationHead


def evaluate_precipitation_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute meteorological evaluation metrics for precipitation amount."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    n = len(y_true)

    err = y_pred - y_true
    overall_mae = float(np.mean(np.abs(err)))
    overall_rmse = float(np.sqrt(np.mean(err ** 2)))
    overall_bias = float(np.mean(err))

    # Dry vs Rainy
    dry_mask = y_true < 0.1
    rainy_mask = y_true >= 0.1

    dry_mae = float(np.mean(np.abs(err[dry_mask]))) if dry_mask.any() else 0.0
    dry_rmse = float(np.sqrt(np.mean(err[dry_mask] ** 2))) if dry_mask.any() else 0.0
    dry_bias = float(np.mean(err[dry_mask])) if dry_mask.any() else 0.0

    rainy_mae = float(np.mean(np.abs(err[rainy_mask]))) if rainy_mask.any() else 0.0
    rainy_rmse = float(np.sqrt(np.mean(err[rainy_mask] ** 2))) if rainy_mask.any() else 0.0
    rainy_bias = float(np.mean(err[rainy_mask])) if rainy_mask.any() else 0.0

    # Heavy rain thresholds (2.5, 5.0, 10.0 mm/h)
    heavy_metrics = {}
    for thresh in [2.5, 5.0, 10.0]:
        t_true = y_true >= thresh
        t_pred = y_pred >= thresh

        tp = int(np.sum(t_true & t_pred))
        fp = int(np.sum(~t_true & t_pred))
        fn = int(np.sum(t_true & ~t_pred))
        tn = int(np.sum(~t_true & ~t_pred))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

        heavy_metrics[f"{thresh}_mmh"] = {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "csi": round(csi, 4),
        }

    return {
        "sample_count": n,
        "rainy_samples": int(np.sum(rainy_mask)),
        "dry_samples": int(np.sum(dry_mask)),
        "overall_mae_mm": round(overall_mae, 4),
        "overall_rmse_mm": round(overall_rmse, 4),
        "overall_bias_mm": round(overall_bias, 4),
        "dry_hour_mae_mm": round(dry_mae, 4),
        "dry_hour_rmse_mm": round(dry_rmse, 4),
        "dry_hour_bias_mm": round(dry_bias, 4),
        "rainy_hour_mae_mm": round(rainy_mae, 4),
        "rainy_hour_rmse_mm": round(rainy_rmse, 4),
        "rainy_hour_bias_mm": round(rainy_bias, 4),
        "heavy_rain_regimes": heavy_metrics,
    }


def run_comparison(output_json: str = None) -> dict:
    """Run comparative evaluation on held-out test data."""
    weather_csv = os.path.join(DATA_DIR, "weather_telemetry.csv")
    water_csv = os.path.join(DATA_DIR, "water_level_telemetry.csv")

    test_seqs = load_real_telemetry_sequences(
        weather_csv_path=weather_csv,
        water_csv_path=water_csv,
        split="test",
        horizons=[1],
        max_sequences=1000,
    )
    if test_seqs is None:
        print("No test sequences available for comparison.")
        return {}

    telemetry, dt, rain_prob, precip_mm, water_level, has_water = test_seqs
    y_trues = precip_mm.numpy().flatten()

    # Load baseline model
    ckpt_path = os.path.join(DATA_DIR, "lnn_weather_water.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m_config = ckpt.get("manifest", {}).get("model_config", {"input_dim": 8, "hidden_dim": 32})

    baseline_model = GarciaWeatherLNN(**m_config, use_two_stage_precipitation=False)
    baseline_model.load_state_dict(ckpt["model_state_dict"])
    baseline_model.eval()

    # Experimental model with two-stage precipitation head
    experimental_model = GarciaWeatherLNN(**m_config, use_two_stage_precipitation=True)
    experimental_model.encoder.load_state_dict(baseline_model.encoder.state_dict())
    experimental_model.cfc_cell.load_state_dict(baseline_model.cfc_cell.state_dict())
    experimental_model.eval()

    with torch.no_grad():
        _, b_precip, _ = baseline_model(telemetry, dt)
        b_preds = b_precip[:, -1, 0].numpy().flatten()

        _, e_precip, _ = experimental_model(telemetry, dt)
        e_preds = e_precip[:, -1, 0].numpy().flatten()

    b_metrics = evaluate_precipitation_metrics(y_trues, b_preds)
    e_metrics = evaluate_precipitation_metrics(y_trues, e_preds)

    comparison = {
        "status": "EXPERIMENTAL_ARCHITECTURE_EVALUATED",
        "baseline_single_head": b_metrics,
        "experimental_two_stage_head": e_metrics,
        "recommendation": (
            "Retain baseline as default production model. Experimental two-stage precipitation "
            "architecture remains available behind use_two_stage_precipitation=True for further tuning "
            "with domain-specific loss objectives (e.g. Tweedie/focal loss)."
        ),
    }

    print("=" * 80)
    print("EXPERIMENTAL TWO-STAGE PRECIPITATION EVALUATION REPORT")
    print("=" * 80)
    print(f"Test samples evaluated: {b_metrics['sample_count']}")
    print(f"Baseline MAE:     {b_metrics['overall_mae_mm']} mm (dry: {b_metrics['dry_hour_mae_mm']} mm, rainy: {b_metrics['rainy_hour_mae_mm']} mm)")
    print(f"Experimental MAE: {e_metrics['overall_mae_mm']} mm (dry: {e_metrics['dry_hour_mae_mm']} mm, rainy: {e_metrics['rainy_hour_mae_mm']} mm)")
    print("=" * 80)

    if output_json:
        os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2)
        print(f"Comparison report saved to: {output_json}")

    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Two-Stage Precipitation Model vs Baseline")
    parser.add_argument("--output", default=None, help="Optional output JSON path for comparison report")
    args = parser.parse_args()
    run_comparison(args.output)
