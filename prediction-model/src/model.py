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
from collections import defaultdict
from datetime import datetime
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


class TwoStagePrecipitationHead(nn.Module):
    """
    Experimental Two-Stage Precipitation Forecasting Architecture.
    Stage 1: Rain Occurrence (binary classifier logit, evaluated for calibration & BCE).
    Stage 2: Conditional Rain Amount (continuous non-negative volume given rain > 0).
    Expected formulation:
      E[Y] = P(Y > 0) * E[Y | Y > 0]
    """
    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.occurrence_net = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.SiLU(),
            nn.Linear(16, 1)  # logit
        )
        self.conditional_amount_net = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.SiLU(),
            nn.Linear(16, 1)  # log-volume / softplus
        )

    def forward(self, h: torch.Tensor):
        """
        h: [batch, hidden_dim]
        Returns:
          rain_prob: [batch, 1] in [0, 1]
          precip_mm: [batch, 1] non-negative expected volume in mm
        """
        logit = self.occurrence_net(h)
        rain_prob = torch.sigmoid(logit)
        cond_amount = F.softplus(self.conditional_amount_net(h))
        expected_amount = rain_prob * cond_amount
        return rain_prob, expected_amount


class GarciaWeatherLNN(WeatherWaterLNN):
    """
    Garcia Weather Telemetry Forecast Engine (Continuous-Time CfC/LNN).
    Extends WeatherWaterLNN with full surface meteorology heads:
      - Continuous Weather Heads: Temperature, Humidity, Pressure, Wind Speed, Wind Vector (u, v)
      - Event Head: Rain Probability (sigmoid logit)
      - Rain Amount Head: Non-negative Rain Accumulation (mm)
      - Derived Output: Deterministic NOAA Heat Index from forecast (Temp, RH)
      - Beta Head: Hydrological River Stage Delta (optional research/beta)
      - Experimental: Two-Stage Precipitation Architecture (behind use_two_stage_precipitation flag)
    """
    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 32,
        use_two_stage_precipitation: bool = False,
    ):
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim)
        self.use_two_stage_precipitation = use_two_stage_precipitation
        if self.use_two_stage_precipitation:
            self.two_stage_rain_head = TwoStagePrecipitationHead(hidden_dim=hidden_dim)

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
            if self.use_two_stage_precipitation:
                rain_prob, precip_mm = self.two_stage_rain_head(h)
            else:
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


class GarciaWeatherLNNFeatured(nn.Module):
    """
    Candidate Model Family (MF-1-FEATURED): Continuous-Time CfC/LNN with Zero-Leakage Feature Fusion.
    Fuses sequential hourly telemetry dynamics [batch, seq_len, 8] with normalized zero-leakage engineered features [batch, 75] at origin t0.
    Outputs:
      - Continuous Weather Heads: Temperature, Humidity, Pressure, Wind Speed, Wind Vector (u, v).
      - Two-Stage Precipitation: Occurrence logit + Conditional amount (mm).
      - Derived Heat Index (NOAA Rothfusz formula).
      - Hydrological River Stage Delta (for gauge station).
    """
    def __init__(
        self,
        input_dim: int = 8,
        context_dim: int = 75,
        hidden_dim: int = 32,
        use_two_stage_precipitation: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.use_two_stage_precipitation = use_two_stage_precipitation

        # Sequence encoder and CfC continuous recurrent cell
        self.seq_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.cfc_cell = CfCCell(hidden_dim, hidden_dim)

        # Context projection for engineered features
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Fusion layer combining sequence state h and context c
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU()
        )

        # Weather heads
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

        # Precipitation heads
        self.rain_occurrence_head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.SiLU(),
            nn.Linear(16, 1)  # occurrence logit
        )
        self.conditional_rain_head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.SiLU(),
            nn.Linear(16, 1)  # conditional rain amount
        )

        # Hydrological river stage delta
        self.water_head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.SiLU(),
            nn.Linear(16, 1)
        )

        # Initialize delta heads to zero for strict persistence prior at step 0
        nn.init.zeros_(self.temp_head[-1].weight)
        nn.init.zeros_(self.temp_head[-1].bias)
        nn.init.zeros_(self.rh_head[-1].weight)
        nn.init.zeros_(self.rh_head[-1].bias)
        nn.init.zeros_(self.pressure_head[-1].weight)
        nn.init.zeros_(self.pressure_head[-1].bias)
        nn.init.zeros_(self.ws_head[-1].weight)
        nn.init.zeros_(self.ws_head[-1].bias)
        nn.init.zeros_(self.wdir_head[-1].weight)
        nn.init.zeros_(self.wdir_head[-1].bias)
        # Prior log-odds for rain occurrence matching ~10% empirical base rate
        nn.init.constant_(self.rain_occurrence_head[-1].bias, -2.2)

    def forward(
        self,
        telemetry_seq: torch.Tensor,
        context_feats: torch.Tensor,
        dt_seq: torch.Tensor,
        initial_water: torch.Tensor = None,
        origin_weather: torch.Tensor = None,
    ):
        """
        telemetry_seq: [batch, seq_len, 8]
        context_feats: [batch, 75]
        dt_seq: [batch, seq_len, 1]
        initial_water: Optional [batch, 1]
        origin_weather: Optional [batch, 6] -> (temp, humidity, pressure, ws, u, v)
        """
        batch_size, seq_len, _ = telemetry_seq.shape
        h = torch.zeros(batch_size, self.hidden_dim, device=telemetry_seq.device)

        # Unroll sequence through continuous CfC cell
        for t in range(seq_len):
            x_t = telemetry_seq[:, t, :]
            dt_t = dt_seq[:, t, :]
            feat = self.seq_encoder(x_t)
            h = self.cfc_cell(feat, h, dt_t)

        # Project static context
        c = self.context_encoder(context_feats)

        # Fuse sequential state and context
        fused = self.fusion(torch.cat([h, c], dim=-1))

        # Base deltas
        d_temp = self.temp_head(fused)
        d_rh = self.rh_head(fused)
        d_p = self.pressure_head(fused)
        d_ws = self.ws_head(fused)
        uv_raw = self.wdir_head(fused)

        d_water = self.water_head(fused)
        water_stage = (initial_water + d_water) if initial_water is not None else d_water

        # Rain Occurrence and Conditional Amount
        rain_logit = self.rain_occurrence_head(fused)
        rain_prob = torch.sigmoid(rain_logit)
        cond_amount = F.softplus(self.conditional_rain_head(fused))
        precip_mm = rain_prob * cond_amount

        # Absolute weather if origin provided
        if origin_weather is not None:
            t_orig = origin_weather[:, 0:1]
            rh_orig = origin_weather[:, 1:2]
            p_orig = origin_weather[:, 2:3]
            ws_orig = origin_weather[:, 3:4]
            u_orig = origin_weather[:, 4:5]
            v_orig = origin_weather[:, 5:6]
            pred_temp = torch.clamp(t_orig + d_temp, min=-10.0, max=60.0)
            pred_rh = torch.clamp(rh_orig + d_rh, min=0.0, max=100.0)
            pred_p = torch.clamp(p_orig + d_p, min=850.0, max=1090.0)
            pred_ws = torch.clamp(F.relu(ws_orig + d_ws), min=0.0, max=250.0)
            # Origin wind vector residual
            orig_uv = torch.cat([u_orig, v_orig], dim=-1)
            combined_uv = orig_uv + uv_raw
            uv_norm = F.normalize(combined_uv, p=2, dim=-1, eps=1e-6)
        else:
            pred_temp = d_temp
            pred_rh = d_rh
            pred_p = d_p
            pred_ws = F.relu(d_ws)
            uv_norm = F.normalize(uv_raw, p=2, dim=-1, eps=1e-6)

        return {
            "temperature": pred_temp,
            "humidity": pred_rh,
            "pressure": pred_p,
            "wind_speed": pred_ws,
            "wind_u": uv_norm[:, 0:1],
            "wind_v": uv_norm[:, 1:2],
            "rain_prob": rain_prob,
            "rain_logit": rain_logit,
            "precipitation_mm": precip_mm,
            "conditional_amount": cond_amount,
            "water_level": water_stage,
            "delta_water": d_water,
        }


class DecisionStump:
    """Fast axis-aligned binary decision stump in pure NumPy."""
    def __init__(self):
        self.feature_idx = 0
        self.threshold = 0.0
        self.left_value = 0.0
        self.right_value = 0.0

    def fit(self, X: np.ndarray, residuals: np.ndarray, subsample_features: int = 15):
        n_samples, n_features = X.shape
        best_gain = -1e9
        feature_indices = np.random.choice(n_features, min(subsample_features, n_features), replace=False)
        for f_idx in feature_indices:
            x_col = X[:, f_idx]
            thresholds = np.quantile(x_col, [0.1, 0.3, 0.5, 0.7, 0.9])
            for th in thresholds:
                left_mask = x_col <= th
                right_mask = ~left_mask
                n_l, n_r = int(np.sum(left_mask)), int(np.sum(right_mask))
                if n_l < 5 or n_r < 5:
                    continue
                sum_l = float(np.sum(residuals[left_mask]))
                sum_r = float(np.sum(residuals[right_mask]))
                gain = (sum_l ** 2) / n_l + (sum_r ** 2) / n_r
                if gain > best_gain:
                    best_gain = gain
                    self.feature_idx = int(f_idx)
                    self.threshold = float(th)
                    self.left_value = float(sum_l / (n_l + 1e-4))
                    self.right_value = float(sum_r / (n_r + 1e-4))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        left = X[:, self.feature_idx] <= self.threshold
        return np.where(left, self.left_value, self.right_value)


class GradientBoostedWeatherModel:
    """
    Strong Tabular Baseline: Pure NumPy Gradient Boosted Decision Tree Ensemble on 75 Engineered Features.
    Fulfills Workstream D of the Proper Predictive-Quality Implementation Plan.
    Trains an ensemble of gradient-boosted decision stumps with shrinkage for each meteorological target.
    """
    def __init__(self, n_estimators: int = 25, learning_rate: float = 0.1, random_state: int = 42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.models = {}  # target_idx -> list of DecisionStump
        self.base_preds = {}  # target_idx -> float
        self.targets = ["temperature", "humidity", "pressure", "wind_speed", "wind_u", "wind_v", "precipitation_mm", "rain_prob"]

    def fit(self, X: np.ndarray, Y: np.ndarray):
        np.random.seed(self.random_state)
        N, D = X.shape
        _, K = Y.shape
        for k in range(K):
            y_k = Y[:, k].astype(np.float32)
            base = float(np.mean(y_k))
            self.base_preds[k] = base
            curr_pred = np.full(N, base, dtype=np.float32)
            stumps = []
            for _ in range(self.n_estimators):
                residuals = y_k - curr_pred
                stump = DecisionStump()
                stump.fit(X, residuals)
                pred_step = stump.predict(X)
                curr_pred += self.learning_rate * pred_step
                stumps.append(stump)
            self.models[k] = stumps
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        N = X.shape[0]
        K = len(self.targets)
        Y_pred = np.zeros((N, K), dtype=np.float32)
        for k in range(K):
            pred = np.full(N, self.base_preds[k], dtype=np.float32)
            for stump in self.models[k]:
                pred += self.learning_rate * stump.predict(X)
            Y_pred[:, k] = pred

        # Enforce physical constraints identical to Ridge
        Y_pred[:, 0] = np.clip(Y_pred[:, 0], -10.0, 60.0)      # temperature
        Y_pred[:, 1] = np.clip(Y_pred[:, 1], 0.0, 100.0)       # humidity
        Y_pred[:, 2] = np.clip(Y_pred[:, 2], 850.0, 1090.0)    # pressure
        Y_pred[:, 3] = np.clip(Y_pred[:, 3], 0.0, 250.0)       # wind_speed
        # Normalize wind vectors
        uv_norm = np.sqrt(Y_pred[:, 4] ** 2 + Y_pred[:, 5] ** 2) + 1e-6
        Y_pred[:, 4] /= uv_norm
        Y_pred[:, 5] /= uv_norm
        Y_pred[:, 6] = np.clip(Y_pred[:, 6], 0.0, 300.0)       # precipitation_mm
        Y_pred[:, 7] = np.clip(Y_pred[:, 7], 0.0, 1.0)         # rain_prob
        return Y_pred


class RidgeWeatherModel:
    """
    Multi-target regularized linear baseline fit on normalized zero-leakage engineered features [batch, 75].
    Closed-form analytical solution: W = (X^T X + lambda I)^(-1) X^T Y
    """
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.weights = None  # [d + 1, k]
        self.targets = ["temperature", "humidity", "pressure", "wind_speed", "wind_u", "wind_v", "precipitation_mm", "rain_prob"]

    def fit(self, X: np.ndarray, Y: np.ndarray):
        """
        X: [N, D] feature matrix
        Y: [N, K] target matrix
        """
        N, D = X.shape
        X_aug = np.hstack([np.ones((N, 1), dtype=np.float32), X.astype(np.float32)])
        # Regularization matrix (do not regularize bias)
        reg = self.alpha * np.eye(D + 1, dtype=np.float32)
        reg[0, 0] = 0.0
        A = X_aug.T @ X_aug + reg
        B = X_aug.T @ Y.astype(np.float32)
        self.weights = np.linalg.solve(A, B)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        N = X.shape[0]
        X_aug = np.hstack([np.ones((N, 1), dtype=np.float32), X.astype(np.float32)])
        Y_pred = X_aug @ self.weights
        # Enforce physical constraints:
        # Col 0: temp [-10, 60]
        Y_pred[:, 0] = np.clip(Y_pred[:, 0], -10.0, 60.0)
        # Col 1: humidity [0, 100]
        Y_pred[:, 1] = np.clip(Y_pred[:, 1], 0.0, 100.0)
        # Col 2: pressure [850, 1090]
        Y_pred[:, 2] = np.clip(Y_pred[:, 2], 850.0, 1090.0)
        # Col 3: wind_speed >= 0
        Y_pred[:, 3] = np.clip(Y_pred[:, 3], 0.0, 250.0)
        # Col 4, 5: wind_u, wind_v normalized
        uv_norm = np.sqrt(Y_pred[:, 4] ** 2 + Y_pred[:, 5] ** 2) + 1e-6
        Y_pred[:, 4] /= uv_norm
        Y_pred[:, 5] /= uv_norm
        # Col 6: precip_mm >= 0
        Y_pred[:, 6] = np.clip(Y_pred[:, 6], 0.0, 300.0)
        # Col 7: rain_prob [0, 1]
        Y_pred[:, 7] = np.clip(Y_pred[:, 7], 0.0, 1.0)
        return Y_pred


class ClimatologyWeatherModel:
    """
    Station-hour historical average baseline.
    Computes average target values indexed by (station_id, hour_of_day) from training partition.
    """
    def __init__(self):
        self.table = {}  # (station_id, hour) -> dict of mean values
        self.global_mean = {}

    def fit_from_metadata(self, train_metadata: list):
        """Fit climatology tables from train metadata dictionaries."""
        stats = defaultdict(lambda: defaultdict(list))
        global_stats = defaultdict(list)
        for r in train_metadata:
            st = r["station_id"]
            # Extract local Philippine hour (UTC + 8)
            t0 = datetime.fromisoformat(r["origin_timestamp"])
            h = (t0.hour + 8) % 24
            for target, val_key in [
                ("temperature", "target_temperature"),
                ("humidity", "target_humidity"),
                ("pressure", "target_pressure"),
                ("wind_speed", "target_wind_speed"),
                ("wind_u", "target_wind_u"),
                ("wind_v", "target_wind_v"),
                ("precipitation_mm", "actual_precip_mm"),
                ("rain_prob", "actual_rain_prob"),
            ]:
                if val_key in r and r[val_key] is not None:
                    v = float(r[val_key])
                    stats[(st, h)][target].append(v)
                    global_stats[target].append(v)

        self.global_mean = {k: float(np.mean(v)) if v else 0.0 for k, v in global_stats.items()}
        for (st, h), vals in stats.items():
            self.table[(st, h)] = {k: float(np.mean(v)) if v else self.global_mean.get(k, 0.0) for k, v in vals.items()}
        return self

    def predict(self, station_id: str, hour_of_day: int) -> dict:
        return self.table.get((station_id, hour_of_day), self.global_mean)


class PersistenceWeatherModel:
    """
    Canonical zero-skill persistence baseline.
    Predicts t0 + h target values equal to the last valid observation at origin t0.
    """
    @staticmethod
    def predict_from_origin(origin_rec: dict) -> dict:
        temp = float(origin_rec["origin_temperature"])
        rh = float(origin_rec["origin_humidity"])
        p = float(origin_rec["origin_pressure"])
        ws = float(origin_rec["origin_wind_speed"])
        u = float(origin_rec.get("origin_wind_u", 0.0))
        v = float(origin_rec.get("origin_wind_v", 0.0))
        precip = float(origin_rec.get("last_observed_precip", 0.0))
        rain_prob = 1.0 if precip >= 0.1 else 0.0
        hi = compute_noaa_heat_index(temp, rh)
        return {
            "temperature": temp,
            "humidity": rh,
            "pressure": p,
            "wind_speed": ws,
            "wind_u": u,
            "wind_v": v,
            "wind_direction_deg": (math.degrees(math.atan2(v, u)) % 360.0) if ws >= 1.0 else None,
            "heat_index": hi,
            "precipitation_mm": precip,
            "rain_probability": rain_prob,
        }
