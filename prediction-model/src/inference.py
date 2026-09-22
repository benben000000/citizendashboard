"""
Inference Process for Continuous-Time CfC/LNN Weather & Hydrological Model.

Provides two distinct inference APIs:
  1. `predict_from_observed_sequence`: OPERATIONAL API for evaluating real observed
     historical sequences leading up to forecast origin t0, predicting conditions
     at target timestamp t0 + h.
  2. `research_projected_sequence`: EXPLORATORY API generating synthetic diurnal
     perturbations for exploratory scenario testing (not validated forecast).

IMPORTANT: This requires a trained checkpoint file. If no checkpoint is found,
inference will raise FileNotFoundError (fail-closed) rather than silently
returning predictions from an untrained model.

This is Model Family 1 (PyTorch WeatherWaterLNN). See MODEL_REGISTRY.md.
"""

import sys
import json
import os
from datetime import datetime, timedelta
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
        if norm_info and "means" in norm_info and "stds" in norm_info:
            self._norm_means = np.array(norm_info["means"], dtype=np.float32)
            self._norm_stds = np.array(norm_info["stds"], dtype=np.float32)
        else:
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

    def predict_from_observed_sequence(
        self,
        telemetry_sequence: np.ndarray,
        dt_sequence: np.ndarray = None,
        forecast_origin_timestamp: str = None,
        horizon_hours: int = 1,
    ) -> dict:
        """
        Operational forecast endpoint using actual observed historical telemetry sequence.

        Args:
            telemetry_sequence: Array of shape [seq_len, 4] with raw observations:
                (temperature, heat_index, wind_speed, pressure).
            dt_sequence: Optional array of shape [seq_len, 1] or [seq_len] with elapsed
                hours between measurements. Defaults to 1.0h per step if None.
            forecast_origin_timestamp: ISO timestamp string of the last observation t0.
            horizon_hours: Number of hours ahead to forecast.

        Returns:
            Dictionary containing prediction outcomes, forecast origin timestamp,
            and target timestamp.
        """
        telemetry_arr = np.asarray(telemetry_sequence, dtype=np.float32)
        if telemetry_arr.ndim != 2 or telemetry_arr.shape[1] != 4:
            raise ValueError(f"Expected telemetry_sequence shape [seq_len, 4], got {telemetry_arr.shape}")

        seq_len = telemetry_arr.shape[0]

        if dt_sequence is None:
            dt_arr = np.ones((seq_len, 1), dtype=np.float32)
        else:
            dt_arr = np.asarray(dt_sequence, dtype=np.float32)
            if dt_arr.ndim == 1:
                dt_arr = dt_arr.reshape(-1, 1)

        norm_features = self._normalize(telemetry_arr)
        x_tensor = torch.tensor(norm_features[np.newaxis, :, :], dtype=torch.float32)
        dt_tensor = torch.tensor(dt_arr[np.newaxis, :, :], dtype=torch.float32)

        with torch.no_grad():
            rain_prob, precip_mm, water_level = self.model(x_tensor, dt_tensor)

        final_rain_prob = float(rain_prob[0, -1, 0].item())
        final_precip_mm = float(precip_mm[0, -1, 0].item())
        predicted_water = float(water_level[0, -1, 0].item())

        target_ts = None
        if forecast_origin_timestamp:
            try:
                origin_dt = datetime.fromisoformat(forecast_origin_timestamp.replace("Z", "+00:00"))
                target_dt = origin_dt + timedelta(hours=horizon_hours)
                target_ts = target_dt.isoformat()
            except Exception:
                target_ts = None

        return {
            "api_mode": "observed_sequence_forecast",
            "model_version": self.manifest.get("training_date", "unknown"),
            "model_seed": self.manifest.get("seed", "unknown"),
            "model_status": "RESEARCH_PROTOTYPE",
            "forecast_origin_timestamp": forecast_origin_timestamp,
            "target_timestamp": target_ts,
            "forecast_horizon": f"{horizon_hours}h",
            "chance_of_rain_pct": round(final_rain_prob * 100, 1),
            "expected_precipitation_mm": round(final_precip_mm, 2),
            "predicted_water_level_m": round(max(0.0, predicted_water), 2),
        }

    def research_projected_sequence(
        self,
        current_temp: float,
        current_heat_index: float,
        current_wind_speed: float,
        current_pressure: float,
        current_water_level: float = 3.2,
        horizon_hours: int = 24,
    ) -> dict:
        """
        Exploratory research API: generates a synthetic diurnal perturbation trajectory.

        WARNING: This uses synthetic projections, NOT observed weather sequences.
        It is intended for what-if sensitivity analysis, not operational forecasting.
        """
        seq_len = min(72, max(1, horizon_hours))
        features = []
        dt_seq = []

        for h in range(1, seq_len + 1):
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

        final_rain_prob = float(rain_prob[0, -1, 0].item())
        final_precip_mm = float(precip_mm[0, -1, 0].item())
        predicted_water = float(
            current_water_level + (water_level[0, -1, 0].item() - water_level[0, 0, 0].item())
        )

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
            "api_mode": "research_projected_sequence",
            "model_version": self.manifest.get("training_date", "unknown"),
            "model_seed": self.manifest.get("seed", "unknown"),
            "model_status": "RESEARCH_PROTOTYPE",
            "lead_horizon": f"{horizon_hours}h",
            "disclaimer": (
                "This prediction uses projected (not observed) input sequences. "
                "For operational forecasting, use predict_from_observed_sequence."
            ),
            "chance_of_rain_pct": round(final_rain_prob * 100, 1),
            "expected_precipitation_mm": round(final_precip_mm, 2),
            "predicted_water_level_m": round(max(0.5, predicted_water), 2),
            "trajectory": trajectory,
        }

    def predict(self, *args, **kwargs):
        """Deprecated alias for research_projected_sequence."""
        return self.research_projected_sequence(*args, **kwargs)


def _diurnal_perturb(step_hour: int) -> float:
    """Sinusoidal diurnal temperature perturbation (heuristic only)."""
    return float(np.sin(2 * np.pi * step_hour / 24.0) * 1.5)


if __name__ == "__main__":
    try:
        predictor = LNNServerlessPredictor()
        # Test operational observed sequence forecast API
        dummy_seq = np.array([[28.0, 32.0, 5.0, 1010.0]] * 24, dtype=np.float32)
        op_result = predictor.predict_from_observed_sequence(
            telemetry_sequence=dummy_seq,
            forecast_origin_timestamp="2026-08-01T12:00:00",
            horizon_hours=1,
        )
        print("Operational API Result:")
        print(json.dumps(op_result, indent=2))

        print("\nResearch Scenario API Result:")
        res_result = predictor.research_projected_sequence(
            current_temp=29.5,
            current_heat_index=35.2,
            current_wind_speed=12.0,
            current_pressure=1007.5,
            current_water_level=3.45,
            horizon_hours=24,
        )
        print(json.dumps(res_result, indent=2))
    except FileNotFoundError as e:
        print(f"INFERENCE BLOCKED: {e}")
        sys.exit(1)
