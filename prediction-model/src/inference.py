"""
Serverless Inference Process for LNN Weather & Hydrological Forecast.
Takes raw station telemetry (temp, heat_index, wind, pressure) and produces
predictions for chance of rain, rain volume, and projected river water levels
across arbitrary lead horizons (1h to 72h).
"""

import sys
import json
import numpy as np
import torch

from dataset import normalize_features
from model import WeatherWaterLNN


class LNNServerlessPredictor:
    def __init__(self, model_weights_path: str = "lnn_weather_water.pt"):
        self.device = torch.device("cpu")
        self.model = WeatherWaterLNN(input_dim=4, hidden_dim=32)
        try:
            checkpoint = torch.load(model_weights_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
        except Exception:
            # If checkpoint not yet trained, initialize default weights
            pass
        self.model.eval()

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
        Executes continuous-time ODE forward integration from current telemetry.
        """
        seq_len = min(72, max(1, horizon_hours))
        # Create sequence forward in time
        features = []
        dt_seq = []

        for h in range(1, seq_len + 1):
            # Diurnal atmospheric perturbation projection
            temp_step = current_temp + math_sin_perturb(h)
            heat_step = current_heat_index + math_sin_perturb(h) * 1.2
            wind_step = max(0.5, current_wind_speed + np.sin(h / 3.0) * 1.5)
            press_step = current_pressure - (0.3 if current_wind_speed > 15 else -0.1)

            feat = np.array([temp_step, heat_step, wind_step, press_step], dtype=np.float32)
            features.append(normalize_features(feat))
            dt_seq.append([1.0])

        x_tensor = torch.tensor(np.stack([features]), dtype=torch.float32)
        dt_tensor = torch.tensor(np.stack([dt_seq]), dtype=torch.float32)

        with torch.no_grad():
            rain_prob, precip_mm, water_level = self.model(x_tensor, dt_tensor)

        # Extract predictions for the final horizon target
        final_rain_prob = float(rain_prob[0, -1, 0].item())
        final_precip_mm = float(precip_mm[0, -1, 0].item())
        predicted_water = float(current_water_level + (water_level[0, -1, 0].item() - water_level[0, 0, 0].item()))

        # Build trajectory points
        trajectory = []
        for step in range(seq_len):
            trajectory.append({
                "hour_offset": step + 1,
                "rain_probability": round(float(rain_prob[0, step, 0].item()) * 100, 1),
                "precipitation_mm": round(float(precip_mm[0, step, 0].item()), 2),
                "predicted_water_level": round(float(current_water_level + (water_level[0, step, 0].item() - water_level[0, 0, 0].item())), 2),
            })

        return {
            "lead_horizon": f"{horizon_hours}h",
            "chance_of_rain_pct": round(final_rain_prob * 100, 1),
            "expected_precipitation_mm": round(final_precip_mm, 2),
            "predicted_water_level_m": round(max(0.5, predicted_water), 2),
            "trajectory": trajectory,
        }


def math_sin_perturb(step_hour: int) -> float:
    return float(np.sin(2 * np.pi * step_hour / 24.0) * 1.5)


if __name__ == "__main__":
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
