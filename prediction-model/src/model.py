"""
Garcia Weather Telemetry Forecast Engine: Continuous-Time CfC/LNN Architecture.

Implements the multi-output Continuous-Time Closed-form Continuous (CfC) Neural ODE
architecture for the Garcia Weather Telemetry Forecast Engine.

Capabilities:
  - Surface Meteorology Heads:
      * Temperature (Celsius)
      * Relative Humidity (%)
      * Atmospheric Pressure (hPa)
      * Wind Speed (km/h)
      * Wind Vector Components (u, v) & reconstructed circular angle (degrees)
      * Derived Heat Index (Celsius, deterministic NOAA Rothfusz regression)
  - Hydrometeorological Heads:
      * Rain Occurrence Probability (calibrated [0, 1])
      * Expected Rain Accumulation (hourly volume, mm)
  - Secondary/Beta Module:
      * Hydrological River Stage Delta (meters, Calumpit gauge benchmark)

Status: Research prototype & probabilistic guidance engine. Not for life-safety or automated flood evacuation triggers.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import compute_noaa_heat_index


class CfCCell(nn.Module):
    """
    Closed-form Continuous-time (CfC) Neural ODE Cell.
    Approximates continuous-time ODE solution:
      dh/dt = - [1/tau + f(x, h)] * h + A * f(x, h)
    in closed analytical form without numerical ODE solvers.
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Backbone networks
        self.ff_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.ff_time = nn.Linear(input_size + hidden_size, hidden_size)
        self.ff_state = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, input_size]
        h_prev: [batch, hidden_size]
        dt: [batch, 1] - elapsed continuous time delta
        """
        combined = torch.cat([x, h_prev], dim=-1)

        # Decay gate based on elapsed time dt
        w_tau = F.softplus(self.ff_time(combined))
        decay = torch.exp(-dt * w_tau)

        # Non-linear activations
        gate = torch.sigmoid(self.ff_gate(combined))
        candidate = torch.tanh(self.ff_state(combined))

        # Closed-form continuous hidden state transition
        h_next = decay * h_prev + (1.0 - decay) * gate * candidate
        return h_next


class WeatherWaterLNN(nn.Module):
    """
    Continuous-time Liquid Neural Network Model (Legacy MF-1 Contract).
    Processes sequential weather telemetry across arbitrary lead horizons.
    """
    def __init__(self, input_dim: int = 8, hidden_dim: int = 32):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Input feature projection
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Recurrent continuous-time LNN cell
        self.cfc_cell = CfCCell(hidden_dim, hidden_dim)

        # Output Head 1: Rain Forecast (Probability [0, 1] & Volume [mm])
        self.rain_head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.SiLU(),
            nn.Linear(16, 2)  # [rain_logit, precipitation_mm]
        )

        # Output Head 2: Hydrological Water Level Delta (meters change relative to t0)
        self.water_head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.SiLU(),
            nn.Linear(16, 1)  # [delta_water_level_m]
        )

    def forward(self, telemetry_seq: torch.Tensor, dt_seq: torch.Tensor, initial_water: torch.Tensor = None):
        """
        telemetry_seq: [batch, seq_len, 8] -> (temp, heat_index, humidity, pressure, wind_speed, wind_sin, wind_cos, precip)
        dt_seq: [batch, seq_len, 1] -> (elapsed hours between measurements)
        initial_water: Optional [batch, 1] -> water stage at origin t0 to reconstruct absolute stage
        
        Returns:
          rain_prob: [batch, seq_len, 1] (0.0 to 1.0)
          precipitation_mm: [batch, seq_len, 1] (>= 0)
          water_level: [batch, seq_len, 1] (meters absolute if initial_water provided, else delta)
        """
        batch_size, seq_len, _ = telemetry_seq.shape
        h = torch.zeros(batch_size, self.hidden_dim, device=telemetry_seq.device)

        rain_probs = []
        precip_vols = []
        water_levels = []

        for t in range(seq_len):
            x_t = telemetry_seq[:, t, :]
            dt_t = dt_seq[:, t, :]

            # Project input
            feat = self.encoder(x_t)

            # Continuous ODE state update
            h = self.cfc_cell(feat, h, dt_t)

            # Heads
            rain_out = self.rain_head(h)
            rain_prob = torch.sigmoid(rain_out[:, 0:1])
            precip_mm = F.relu(rain_out[:, 1:2])

            delta_water = self.water_head(h)
            if initial_water is not None:
                water_stage = initial_water + delta_water
            else:
                water_stage = delta_water

            rain_probs.append(rain_prob)
            precip_vols.append(precip_mm)
            water_levels.append(water_stage)

        return (
            torch.stack(rain_probs, dim=1),
            torch.stack(precip_vols, dim=1),
            torch.stack(water_levels, dim=1)
        )


class GarciaWeatherLNN(WeatherWaterLNN):
    """
    Garcia Weather Telemetry Forecast Engine (Continuous-Time CfC/LNN).
    Extends WeatherWaterLNN with full surface meteorology heads:
      - Continuous Weather Heads: Temperature, Humidity, Pressure, Wind Speed, Wind Vector (u, v)
      - Event Head: Rain Probability (sigmoid logit)
      - Rain Amount Head: Non-negative Rain Accumulation (mm)
      - Derived Output: Deterministic NOAA Heat Index from forecast (Temp, RH)
      - Beta Head: Hydrological River Stage Delta (optional research/beta)
    """
    def __init__(self, input_dim: int = 8, hidden_dim: int = 32):
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim)

        # Core Weather Heads
        self.temp_head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.SiLU(),
            nn.Linear(16, 1)  # delta temperature (C)
        )
        self.rh_head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.SiLU(),
            nn.Linear(16, 1)  # delta humidity (%)
        )
        self.pressure_head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.SiLU(),
            nn.Linear(16, 1)  # delta pressure (hPa)
        )
        self.ws_head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.SiLU(),
            nn.Linear(16, 1)  # delta wind speed (km/h)
        )
        self.wdir_head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.SiLU(),
            nn.Linear(16, 2)  # circular unit components (u=cos, v=sin)
        )

    def forward(
        self,
        telemetry_seq: torch.Tensor,
        dt_seq: torch.Tensor,
        initial_water: torch.Tensor = None,
        origin_weather: torch.Tensor = None,
        return_dict: bool = False,
    ):
        """
        telemetry_seq: [batch, seq_len, 8] -> (temp, heat_index, humidity, pressure, wind_speed, wind_sin, wind_cos, precip)
        dt_seq: [batch, seq_len, 1] -> (elapsed hours between measurements)
        initial_water: Optional [batch, 1] -> water stage at origin t0
        origin_weather: Optional [batch, 6] -> (origin temp, humidity, pressure, wind_speed, wind_u, wind_v)
        return_dict: If True, returns full dictionary of multi-target weather forecasts.
                     If False, returns backwards-compatible 3-tuple (rain_prob, precip_mm, water_level).
        """
        batch_size, seq_len, _ = telemetry_seq.shape
        h = torch.zeros(batch_size, self.hidden_dim, device=telemetry_seq.device)

        rain_probs = []
        precip_vols = []
        water_levels = []

        for t in range(seq_len):
            x_t = telemetry_seq[:, t, :]
            dt_t = dt_seq[:, t, :]

            # Project input
            feat = self.encoder(x_t)

            # Continuous ODE state update
            h = self.cfc_cell(feat, h, dt_t)

            # Heads
            rain_out = self.rain_head(h)
            rain_prob = torch.sigmoid(rain_out[:, 0:1])
            precip_mm = F.relu(rain_out[:, 1:2])

            delta_water = self.water_head(h)
            if initial_water is not None:
                water_stage = initial_water + delta_water
            else:
                water_stage = delta_water

            rain_probs.append(rain_prob)
            precip_vols.append(precip_mm)
            water_levels.append(water_stage)

        rain_prob_seq = torch.stack(rain_probs, dim=1)
        precip_mm_seq = torch.stack(precip_vols, dim=1)
        water_level_seq = torch.stack(water_levels, dim=1)

        if not return_dict:
            return rain_prob_seq, precip_mm_seq, water_level_seq

        # Compute final-step continuous weather predictions
        d_temp = self.temp_head(h)
        d_rh = self.rh_head(h)
        d_p = self.pressure_head(h)
        d_ws = self.ws_head(h)
        raw_uv = self.wdir_head(h)
        norm_uv = F.normalize(raw_uv, p=2, dim=-1)

        if origin_weather is not None:
            pred_temp = origin_weather[:, 0:1] + d_temp
            pred_rh = torch.clamp(origin_weather[:, 1:2] + d_rh, 10.0, 100.0)
            pred_p = origin_weather[:, 2:3] + d_p
            pred_ws = F.relu(origin_weather[:, 3:4] + d_ws)
        else:
            pred_temp = d_temp
            pred_rh = d_rh
            pred_p = d_p
            pred_ws = F.relu(d_ws)

        return {
            "rain_prob": rain_prob_seq[:, -1, :],
            "precipitation_mm": precip_mm_seq[:, -1, :],
            "water_level": water_level_seq[:, -1, :],
            "temperature": pred_temp,
            "humidity": pred_rh,
            "pressure": pred_p,
            "wind_speed": pred_ws,
            "wind_u": norm_uv[:, 0:1],
            "wind_v": norm_uv[:, 1:2],
        }

    def predict_weather(
        self,
        telemetry_seq: torch.Tensor,
        dt_seq: torch.Tensor,
        origin_weather: torch.Tensor = None,
        initial_water: torch.Tensor = None,
    ) -> dict:
        """
        Produce complete operational weather forecast dictionary for input sequence window.
        """
        self.eval()
        with torch.no_grad():
            res = self.forward(
                telemetry_seq=telemetry_seq,
                dt_seq=dt_seq,
                initial_water=initial_water,
                origin_weather=origin_weather,
                return_dict=True,
            )

        # Convert tensors to python values / arrays
        temp_val = res["temperature"].squeeze(-1).cpu().numpy()
        rh_val = res["humidity"].squeeze(-1).cpu().numpy()
        p_val = res["pressure"].squeeze(-1).cpu().numpy()
        ws_val = res["wind_speed"].squeeze(-1).cpu().numpy()
        u_val = res["wind_u"].squeeze(-1).cpu().numpy()
        v_val = res["wind_v"].squeeze(-1).cpu().numpy()
        r_prob = res["rain_prob"].squeeze(-1).cpu().numpy()
        p_mm = res["precipitation_mm"].squeeze(-1).cpu().numpy()
        w_lvl = res["water_level"].squeeze(-1).cpu().numpy()

        # Reconstruct circular wind direction (degrees) and derived Heat Index
        wind_dirs = []
        heat_indices = []
        t_arr = np.atleast_1d(temp_val)
        rh_arr = np.atleast_1d(rh_val)
        u_arr = np.atleast_1d(u_val)
        v_arr = np.atleast_1d(v_val)

        for i in range(len(t_arr)):
            deg = math.degrees(math.atan2(float(v_arr[i]), float(u_arr[i]))) % 360.0
            hi = compute_noaa_heat_index(float(t_arr[i]), float(rh_arr[i]))
            wind_dirs.append(round(deg, 2))
            heat_indices.append(round(hi, 2))

        return {
            "temperature": temp_val,
            "humidity": rh_val,
            "pressure": p_val,
            "wind_speed": ws_val,
            "wind_direction_deg": np.array(wind_dirs),
            "wind_u": u_val,
            "wind_v": v_val,
            "heat_index": np.array(heat_indices),
            "rain_probability": r_prob,
            "precipitation_mm": p_mm,
            "water_level_stage": w_lvl,
        }
