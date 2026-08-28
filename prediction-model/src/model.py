"""
Liquid Neural Network (LNN) / Closed-form Continuous-time (CfC) Neural Model
for Weather Station Telemetry & Flood Forecasting.

Inputs:
  - Temperature (C)
  - Heat Index (C)
  - Wind Speed (km/h)
  - Atmospheric Pressure (hPa)
  - Time Delta dt (hours)

Outputs:
  - Chance of Rain (%) & Expected Rain Accumulation (mm)
  - Projected River Water Level (meters) & Flood Risk Stage
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    Complete Continuous-time Liquid Neural Network Model.
    Processes sequential weather telemetry across arbitrary lead horizons.
    """
    def __init__(self, input_dim: int = 4, hidden_dim: int = 32):
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

        # Output Head 2: Hydrological Water Level (Meters)
        self.water_head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.SiLU(),
            nn.Linear(16, 1)  # [predicted_water_level_m]
        )

    def forward(self, telemetry_seq: torch.Tensor, dt_seq: torch.Tensor):
        """
        telemetry_seq: [batch, seq_len, 4] -> (temp, heat_index, wind_speed, pressure)
        dt_seq: [batch, seq_len, 1] -> (elapsed hours between measurements)
        
        Returns:
          rain_prob: [batch, seq_len, 1] (0.0 to 1.0)
          precipitation_mm: [batch, seq_len, 1] (>= 0)
          water_level_m: [batch, seq_len, 1] (meters)
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

            water_level = self.water_head(h)

            rain_probs.append(rain_prob)
            precip_vols.append(precip_mm)
            water_levels.append(water_level)

        return (
            torch.stack(rain_probs, dim=1),
            torch.stack(precip_vols, dim=1),
            torch.stack(water_levels, dim=1)
        )
