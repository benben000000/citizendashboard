"""
Inference Process for Continuous-Time CfC/LNN Weather & Hydrological Model.

Takes raw station telemetry (temp, heat_index, wind, pressure) and produces
predictions for chance of rain, rain volume, and projected river water levels.

IMPORTANT: This requires a trained checkpoint file. If no checkpoint is found,
inference will raise FileNotFoundError (fail-closed) rather than silently
returning predictions from an untrained model.

This is Model Family 1 (PyTorch WeatherWaterLNN). See MODEL_REGISTRY.md.
"""

import sys
import json
import os
import numpy as np
import torch

from dataset import normalize_features, FEATURE_MEANS, FEATURE_STDS
from model import WeatherWaterLNN


class LNNServerlessPredictor:
    def __init__(self, model_weights_path: str = "lnn_weather_water.pt"):
        self.device = torch.device("cpu")

        # Check current working directory, then data directory fallback
        if not os.path.exists(model_weights_path):
            fallback_data_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", os.path.basename(model_weights_path)
            )
            if os.path.exists(fallback_data_path):
                model_weights_path = fallback_data_path

        if not os.path.exists(model_weights_path):
            raise FileNotFoundError(
                f"Model checkpoint not found at '{model_weights_path}'. "
                f"Train the model first with `python train.py`. "
                f"Inference will NOT proceed with untrained weights."
            )

        checkpoint = torch.load(model_weights_path, map_location=self.device, weights_only=False)

        # Validate checkpoint structure
        if "model_state_dict" not in checkpoint:
            raise ValueError(
                f"Invalid checkpoint at '{model_weights_path}': missing 'model_state_dict'. "
                f"Re-train the model to produce a valid checkpoint."
            )

        # Load manifest metadata if available
        self.manifest = checkpoint.get("manifest", {})
        model_config = self.manifest.get("model_config", {"input_dim": 4, "hidden_dim": 32})

        # Load normalization constants from checkpoint if available
        norm_info = self.manifest.get("normalization", {})
        if norm_info:
            self._norm_means = np.array(norm_info["means"], dtype=np.float32)
            self._norm_stds = np.array(norm_info["stds"], dtype=np.float32)
        else:
            # Fallback to module-level constants (log warning)
            print("WARNING: Checkpoint has no normalization metadata. Using default constants.")
            self._norm_means = FEATURE_MEANS
            self._norm_stds = FEATURE_STDS

        self.model = WeatherWaterLNN(**model_config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        training_date = self.manifest.get("training_date", "unknown")
        seed = self.manifest.get("seed", "unknown")
        print(f"Loaded model checkpoint: trained={training_date}, seed={seed}")

    def _normalize(self, features: np.ndarray) -> np.ndarray:
        """Normalize using checkpoint-stored constants (not module globals)."""
        return (features - self._norm_means) / self._norm_stds

    def predict(
        self,
        current_temp: float,
        current_heat_index: float,
        current_wind_speed: float,
        current_pressure: float,
        current_water_level: float = 3.2,
        horizon_hours: int = 24,
    ):
        """
        Produces a forecast trajectory from current observed conditions.

        NOTE: This creates a projected input sequence using diurnal perturbation.
        For operational use, feed actual observed historical sequences instead.
        The output label (e.g. "24h") represents the model's internal horizon
        length, NOT a validated 24-hour forecast accuracy claim.
        """
        seq_len = min(72, max(1, horizon_hours))
        features = []
        dt_seq = []

        for h in range(1, seq_len + 1):
            # Diurnal atmospheric perturbation projection
            # NOTE: This is a simplified heuristic, not observed data.
            temp_step = current_temp + _diurnal_perturb(h)
            heat_step = current_heat_index + _diurnal_perturb(h) * 1.2
            wind_step = max(0.5, current_wind_speed + np.sin(h / 3.0) * 1.5)
            press_step = current_pressure - (0.3 if current_wind_speed > 15 else -0.1)

            feat = np.array([temp_step, heat_step, wind_step, press_step], dtype=np.float32)
            features.append(self._normalize(feat))
            dt_seq.append([1.0])

        x_tensor = torch.tensor(np.stack([features]), dtype=torch.float32)
        dt_tensor = torch.tensor(np.stack([dt_seq]), dtype=torch.float32)

        with torch.no_grad():
            rain_prob, precip_mm, water_level = self.model(x_tensor, dt_tensor)

        # Extract predictions for the final horizon target
        final_rain_prob = float(rain_prob[0, -1, 0].item())
        final_precip_mm = float(precip_mm[0, -1, 0].item())
        predicted_water = float(
            current_water_level + (water_level[0, -1, 0].item() - water_level[0, 0, 0].item())
        )

        # Build trajectory points
        trajectory = []
        for step in range(seq_len):
            trajectory.append({
                "hour_offset": step + 1,
                "rain_probability": round(float(rain_prob[0, step, 0].item()) * 100, 1),
                "precipitation_mm": round(float(precip_mm[0, step, 0].item()), 2),
                "predicted_water_level": round(
                    float(
                        current_water_level
                        + (water_level[0, step, 0].item() - water_level[0, 0, 0].item())
                    ),
                    2,
                ),
            })

        return {
            "model_version": self.manifest.get("training_date", "unknown"),
            "model_seed": self.manifest.get("seed", "unknown"),
            "model_status": "RESEARCH_PROTOTYPE",
            "lead_horizon": f"{horizon_hours}h",
            "disclaimer": (
                "This prediction uses projected (not observed) input sequences. "
                "The horizon label does not constitute a validated forecast accuracy claim."
            ),
            "chance_of_rain_pct": round(final_rain_prob * 100, 1),
            "expected_precipitation_mm": round(final_precip_mm, 2),
            "predicted_water_level_m": round(max(0.5, predicted_water), 2),
            "trajectory": trajectory,
        }


def _diurnal_perturb(step_hour: int) -> float:
    """Sinusoidal diurnal temperature perturbation (heuristic only)."""
    return float(np.sin(2 * np.pi * step_hour / 24.0) * 1.5)


if __name__ == "__main__":
    try:
        predictor = LNNServerlessPredictor()
        result = predictor.predict(
            current_temp=29.5,
            current_heat_index=35.2,
            current_wind_speed=12.0,
            current_pressure=1007.5,
            current_water_level=3.45,
            horizon_hours=24,
        )
        print(json.dumps(result, indent=2))
    except FileNotFoundError as e:
        print(f"INFERENCE BLOCKED: {e}")
        sys.exit(1)
