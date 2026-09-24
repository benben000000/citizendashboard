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
import math
from datetime import datetime, timedelta
import numpy as np
import torch

from dataset import normalize_features, FEATURE_MEANS, FEATURE_STDS
from model import WeatherWaterLNN


def compute_sha256(filepath: str) -> str:
    """Compute SHA-256 hash of a file."""
    import hashlib
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class LNNServerlessPredictor:
    SUPPORTED_HORIZONS = [1, 3, 6, 12, 24]

    @classmethod
    def supported_horizons(cls) -> list:
        """Return list of canonically validated forecast horizons in hours."""
        return list(cls.SUPPORTED_HORIZONS)

    def get_supported_horizons(self) -> list:
        """Instance helper for supported forecast horizons."""
        return list(self.SUPPORTED_HORIZONS)

    def __init__(
        self,
        model_weights_path: str = None,
        policy_path: str = None,
        bundle_dir: str = None,
        horizon_hours: int = None,
        device: str = "cpu",
    ):
        self.device = torch.device(device)

        if horizon_hours is None:
            # Auto-detect horizon from model_weights_path or bundle_dir if present
            import re
            detected_h = None
            if bundle_dir:
                m = re.search(r"h(\d+)", os.path.basename(bundle_dir.rstrip("/\\")))
                if m:
                    detected_h = int(m.group(1))
            if detected_h is None and model_weights_path:
                m = re.search(r"_h(\d+)\.pt", os.path.basename(model_weights_path))
                if m:
                    detected_h = int(m.group(1))
            self.horizon_hours = detected_h if detected_h is not None else 1
        else:
            self.horizon_hours = int(horizon_hours)

        if self.horizon_hours not in self.SUPPORTED_HORIZONS:
            raise ValueError(
                f"Unsupported horizon {self.horizon_hours}h. Supported horizons are: {self.SUPPORTED_HORIZONS}"
            )

        repo_data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
        )

        # Check default bundle location if bundle_dir not explicitly passed
        if bundle_dir is None and model_weights_path is None:
            candidate_bundle = os.path.join(repo_data_dir, "bundles", f"h{self.horizon_hours}")
            if os.path.exists(candidate_bundle):
                bundle_dir = candidate_bundle

        self.is_bundled = False
        self.bundle_dir = bundle_dir
        self.bundle_manifest = {}
        self.bundle_version = "legacy-unbundled"

        if bundle_dir is not None:
            if not os.path.exists(bundle_dir):
                raise FileNotFoundError(f"Bundle directory not found: '{bundle_dir}'.")

            manifest_path = os.path.join(bundle_dir, "bundle_manifest.json")
            ckpt_path = os.path.join(bundle_dir, "checkpoint.pt")
            pol_path = os.path.join(bundle_dir, "inference_policy.json")

            for p, label in [(manifest_path, "bundle_manifest.json"), (ckpt_path, "checkpoint.pt"), (pol_path, "inference_policy.json")]:
                if not os.path.exists(p):
                    raise FileNotFoundError(f"Missing required bundle artifact '{label}' in '{bundle_dir}'.")

            with open(manifest_path, "r", encoding="utf-8") as f:
                self.bundle_manifest = json.load(f)

            # Validate required manifest fields
            required_manifest_fields = [
                "bundle_version",
                "horizon_hours",
                "implementation_commit",
                "artifact_commit",
                "checkpoint_sha256",
                "policy_sha256",
                "raw_weather_dataset_sha256",
                "raw_water_dataset_sha256",
                "feature_schema",
                "model_family",
            ]
            for rf in required_manifest_fields:
                if rf not in self.bundle_manifest:
                    raise ValueError(f"Bundle manifest '{manifest_path}' missing required field: '{rf}'.")

            # Validate horizon match
            b_h = self.bundle_manifest.get("horizon_hours")
            if b_h != self.horizon_hours:
                raise ValueError(
                    f"Bundle horizon mismatch: bundle is packaged for +{b_h}h, "
                    f"but predictor was initialized for +{self.horizon_hours}h."
                )

            # Cryptographic hash validation (fail-closed)
            actual_ckpt_hash = compute_sha256(ckpt_path)
            expected_ckpt_hash = self.bundle_manifest.get("checkpoint_sha256")
            if actual_ckpt_hash != expected_ckpt_hash:
                raise ValueError(
                    f"Bundle checkpoint hash mismatch (tampered or corrupted): "
                    f"expected {expected_ckpt_hash}, got {actual_ckpt_hash}."
                )

            actual_pol_hash = compute_sha256(pol_path)
            expected_pol_hash = self.bundle_manifest.get("policy_sha256")
            if actual_pol_hash != expected_pol_hash:
                raise ValueError(
                    f"Bundle policy hash mismatch (tampered or corrupted): "
                    f"expected {expected_pol_hash}, got {actual_pol_hash}."
                )

            model_weights_path = ckpt_path
            policy_path = pol_path
            self.is_bundled = True
            self.bundle_version = self.bundle_manifest.get("bundle_version", "1.0.0")

        else:
            # Fallback legacy loading
            if model_weights_path is None:
                model_weights_path = f"lnn_weather_water_h{self.horizon_hours}.pt" if self.horizon_hours != 1 else "lnn_weather_water.pt"

            if not os.path.exists(model_weights_path):
                fallback_data_path = os.path.join(repo_data_dir, os.path.basename(model_weights_path))
                if os.path.exists(fallback_data_path):
                    model_weights_path = fallback_data_path

            if not os.path.exists(model_weights_path):
                raise FileNotFoundError(
                    f"Model checkpoint not found at '{model_weights_path}'. "
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
        model_config = self.manifest.get("model_config", {"input_dim": 8, "hidden_dim": 32})

        # Load normalization constants from checkpoint if available
        norm_info = self.manifest.get("normalization", {})
        if norm_info and "means" in norm_info and "stds" in norm_info:
            self._norm_means = np.array(norm_info["means"], dtype=np.float32)
            self._norm_stds = np.array(norm_info["stds"], dtype=np.float32)
        else:
            print("WARNING: Checkpoint has no normalization metadata. Using default constants.")
            self._norm_means = FEATURE_MEANS
            self._norm_stds = FEATURE_STDS

        from model import GarciaWeatherLNN
        state_dict = checkpoint["model_state_dict"]
        has_weather_heads = any(k.startswith("temp_head") for k in state_dict.keys())

        if has_weather_heads:
            self.model = GarciaWeatherLNN(**model_config)
        else:
            self.model = WeatherWaterLNN(**model_config)

        self.model.load_state_dict(state_dict)
        self.model.eval()

        training_date = self.manifest.get("training_date", "unknown")
        seed = self.manifest.get("seed", "unknown")
        print(f"Loaded model checkpoint: trained={training_date}, seed={seed}, weather_heads={has_weather_heads}")

        # Locate and load operational inference policy (fail-closed)
        if policy_path is None:
            default_policy_name = "inference_policy.json"
            candidate_policy = os.path.join(repo_data_dir, default_policy_name)
            if os.path.exists(candidate_policy):
                policy_path = candidate_policy
            elif os.path.exists(default_policy_name):
                policy_path = default_policy_name
        elif not os.path.isabs(policy_path) and not os.path.exists(policy_path):
            fallback_pol = os.path.join(repo_data_dir, os.path.basename(policy_path))
            if os.path.exists(fallback_pol):
                policy_path = fallback_pol

        if not os.path.exists(policy_path):
            raise FileNotFoundError(
                f"Operational inference policy not found at '{policy_path}'. "
                f"Validation must be executed first to generate inference_policy.json. "
                f"Inference fails closed without a valid operational policy."
            )

        try:
            with open(policy_path, "r", encoding="utf-8") as f:
                self.policy = json.load(f)
        except Exception as e:
            raise ValueError(f"Malformed operational inference policy at '{policy_path}': {e}")

        if not isinstance(self.policy, dict) or "horizons" not in self.policy:
            raise ValueError(f"Invalid operational inference policy at '{policy_path}': missing 'horizons' table.")

        policy_commit = self.policy.get("policy_code_commit")
        model_commit = self.manifest.get("code_commit")
        if policy_commit and model_commit and policy_commit != "unknown" and model_commit != "unknown":
            if policy_commit != model_commit:
                raise ValueError(
                    f"Operational policy commit mismatch (fail-closed): "
                    f"policy commit '{policy_commit}' does not match model commit '{model_commit}'."
                )

    def _normalize(self, features: np.ndarray) -> np.ndarray:
        """Normalize using checkpoint-stored constants (not module globals)."""
        return (features - self._norm_means) / self._norm_stds

    def predict_from_observed_sequence(
        self,
        telemetry_sequence: np.ndarray,
        dt_sequence: np.ndarray = None,
        forecast_origin_timestamp: str = None,
        horizon_hours: int = None,
        current_water_level: float = None,
    ) -> dict:
        """
        Operational forecast endpoint using actual observed historical telemetry sequence.

        Args:
            telemetry_sequence: Array of shape [seq_len, 8] with raw observations:
                (temperature, heat_index, humidity, pressure, wind_speed, wind_sin, wind_cos, precipitation).
            dt_sequence: Optional array of shape [seq_len, 1] or [seq_len] with elapsed
                hours between measurements. Defaults to 1.0h per step if None.
            forecast_origin_timestamp: ISO timestamp string of the last observation t0.
            horizon_hours: Number of hours ahead to forecast. Defaults to predictor.horizon_hours.
            current_water_level: Optional current river stage in meters.

        Returns:
            Dictionary containing prediction outcomes, forecast origin timestamp,
            and target timestamp.
        """
        from dataset import compute_noaa_heat_index

        if horizon_hours is None:
            horizon_hours = self.horizon_hours
        else:
            horizon_hours = int(horizon_hours)

        if horizon_hours not in self.SUPPORTED_HORIZONS:
            raise ValueError(
                f"Requested horizon {horizon_hours}h is not supported in operational inference policy "
                f"(supported: {self.SUPPORTED_HORIZONS}). Fail closed; do not silently invent defaults or substitute 1h policy."
            )

        if horizon_hours != self.horizon_hours:
            raise ValueError(
                f"Horizon mismatch: predictor is initialized for +{self.horizon_hours}h "
                f"(using checkpoint/bundle for {self.horizon_hours}h), but requested forecast horizon is +{horizon_hours}h. "
                f"Initialize an LNNServerlessPredictor with horizon_hours={horizon_hours} (or bundle_dir for h{horizon_hours}) to serve this horizon."
            )

        telemetry_arr = np.asarray(telemetry_sequence, dtype=np.float32)
        if telemetry_arr.ndim != 2:
            raise ValueError(f"Expected telemetry_sequence shape [seq_len, 8], got {telemetry_arr.shape}")

        # Operational forecast strictly requires all 8 canonical features (fail-closed)
        if telemetry_arr.shape[1] != 8:
            raise ValueError(
                f"Operational forecast requires all 8 canonical physical features "
                f"['temperature', 'heat_index', 'humidity', 'pressure', 'wind_speed', 'wind_sin', 'wind_cos', 'precipitation'], "
                f"got {telemetry_arr.shape[1]} features. For exploratory simulation with defaults, use research_projected_sequence."
            )

        if np.isnan(telemetry_arr).any() or np.isinf(telemetry_arr).any():
            raise ValueError("Operational forecast input sequence contains NaN or infinite values (fail-closed).")

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
        init_water_tensor = torch.tensor([[current_water_level]], dtype=torch.float32) if current_water_level is not None else None

        # Build origin weather tensor: [temp, humidity, pressure, wind_speed, wind_u, wind_v]
        orig_temp = float(telemetry_arr[-1, 0])
        orig_rh = float(telemetry_arr[-1, 2])
        orig_p = float(telemetry_arr[-1, 3])
        orig_ws = float(telemetry_arr[-1, 4])
        orig_sin = float(telemetry_arr[-1, 5])
        orig_cos = float(telemetry_arr[-1, 6])
        orig_weather_tensor = torch.tensor([[orig_temp, orig_rh, orig_p, orig_ws, orig_cos, orig_sin]], dtype=torch.float32)

        from model import GarciaWeatherLNN
        is_garcia = isinstance(self.model, GarciaWeatherLNN)

        with torch.no_grad():
            if is_garcia:
                out = self.model(
                    x_tensor,
                    dt_tensor,
                    initial_water=init_water_tensor,
                    origin_weather=orig_weather_tensor,
                    return_dict=True,
                )
                final_rain_prob = float(out["rain_prob"][0, 0].item())
                final_precip_mm = float(out["precipitation_mm"][0, 0].item())
                predicted_water = float(out["water_level"][0, 0].item())
                pred_temp = float(out["temperature"][0, 0].item())
                pred_rh = float(out["humidity"][0, 0].item())
                pred_p = float(out["pressure"][0, 0].item())
                pred_ws = float(out["wind_speed"][0, 0].item())
                u_val = float(out["wind_u"][0, 0].item())
                v_val = float(out["wind_v"][0, 0].item())
                # Circular wind direction
                pred_wind_dir = (math.degrees(math.atan2(v_val, u_val)) + 360.0) % 360.0
                if pred_ws < 1.0:
                    pred_wind_dir = None  # Calm wind
            else:
                rain_prob, precip_mm, water_level = self.model(x_tensor, dt_tensor, initial_water=init_water_tensor)
                final_rain_prob = float(rain_prob[0, -1, 0].item())
                final_precip_mm = float(precip_mm[0, -1, 0].item())
                predicted_water = float(water_level[0, -1, 0].item())
                pred_temp = orig_temp
                pred_rh = orig_rh
                pred_p = orig_p
                pred_ws = orig_ws
                pred_wind_dir = None

        # Check and enforce operational inference policy for requested horizon (fail-closed)
        h_key = str(horizon_hours)
        horizons_table = self.policy.get("horizons", {})
        if h_key not in horizons_table:
            raise ValueError(
                f"Requested horizon {horizon_hours}h is not supported in operational inference policy "
                f"(supported: {list(horizons_table.keys())}). Fail closed; do not silently invent defaults or substitute 1h policy."
            )
        h_policy = horizons_table[h_key]
        selected_sources = h_policy.get("selected_sources", {})
        rain_model_weight = float(h_policy.get("rain_model_weight", 1.0))
        rain_persistence_weight = float(h_policy.get("rain_persistence_weight", 0.0))
        operational_rain_threshold = float(h_policy.get("operational_rain_threshold", 0.5))

        # Origin persistence observations
        last_observed_precip = float(telemetry_arr[-1, 7])
        orig_wind_deg = (math.degrees(math.atan2(orig_sin, orig_cos)) + 360.0) % 360.0 if orig_ws >= 1.0 else None
        persistence_rain_prob = 1.0 if last_observed_precip >= 0.1 else 0.0

        # Continuous weather variable selection according to operational policy
        temp_src = selected_sources.get("temperature", "persistence_fallback")
        rh_src = selected_sources.get("humidity", "persistence_fallback")
        p_src = selected_sources.get("pressure", "persistence_fallback")
        ws_src = selected_sources.get("wind_speed", "persistence_fallback")
        wd_src = selected_sources.get("wind_direction", "persistence_fallback")

        op_temp = pred_temp if temp_src == "learned_model" else orig_temp
        op_rh = pred_rh if rh_src == "learned_model" else orig_rh
        op_p = pred_p if p_src == "learned_model" else orig_p
        op_ws = pred_ws if ws_src == "learned_model" else orig_ws
        op_wind_dir = pred_wind_dir if wd_src == "learned_model" else orig_wind_deg
        if op_ws < 1.0:
            op_wind_dir = None

        # Derived Heat Index & Risk Category from operational values
        derived_hi = compute_noaa_heat_index(op_temp, op_rh)
        if derived_hi < 27.0:
            hi_risk = "NORMAL"
        elif derived_hi < 32.0:
            hi_risk = "CAUTION"
        elif derived_hi < 41.0:
            hi_risk = "EXTREME CAUTION"
        elif derived_hi < 54.0:
            hi_risk = "DANGER"
        else:
            hi_risk = "EXTREME DANGER"

        # Rain probability hybrid blending
        blended_rain_prob = max(0.0, min(1.0, (rain_model_weight * final_rain_prob) + (rain_persistence_weight * persistence_rain_prob)))
        rain_operational_alert = bool(blended_rain_prob >= operational_rain_threshold)

        # Pressure Tendency from operational pressure vs origin
        dp = op_p - orig_p
        if dp > 0.5:
            p_tendency = "RISING"
        elif dp < -0.5:
            p_tendency = "FALLING"
        else:
            p_tendency = "STEADY"

        target_ts = None
        if forecast_origin_timestamp:
            try:
                origin_dt = datetime.fromisoformat(forecast_origin_timestamp.replace("Z", "+00:00"))
                target_dt = origin_dt + timedelta(hours=horizon_hours)
                target_ts = target_dt.isoformat()
            except Exception:
                target_ts = None

        provenance_dict = {
            "implementation_commit": self.bundle_manifest.get("implementation_commit") or self.policy.get("policy_code_commit", "unknown"),
            "artifact_commit": self.bundle_manifest.get("artifact_commit") or self.manifest.get("artifact_commit", "unknown"),
            "model_weights_commit": self.bundle_manifest.get("model_weights_commit") or self.manifest.get("code_commit", "unknown"),
            "bundle_version": self.bundle_version,
            "checkpoint_sha256": self.bundle_manifest.get("checkpoint_sha256", "legacy-unbundled"),
            "policy_sha256": self.bundle_manifest.get("policy_sha256", "legacy-unbundled"),
        }

        return {
            "api_mode": "observed_sequence_forecast",
            "product_name": "Garcia Weather Telemetry Forecast Engine",
            "bundle_version": self.bundle_version,
            "active_bundle_horizon": f"{self.horizon_hours}h",
            "active_bundle_path": self.bundle_dir if self.bundle_dir else None,
            "model_version": self.manifest.get("training_date", "unknown"),
            "model_seed": self.manifest.get("seed", "unknown"),
            "model_status": self.manifest.get("model_status", "RESEARCH_PROTOTYPE"),
            "not_for_life_safety": True,
            "forecast_origin_timestamp": forecast_origin_timestamp,
            "target_timestamp": target_ts,
            "forecast_horizon": f"{horizon_hours}h",
            # Core Operational Weather Forecast (Policy-Governed)
            "temperature_c": round(op_temp, 2),
            "relative_humidity_pct": round(op_rh, 1),
            "pressure_hpa": round(op_p, 2),
            "pressure_tendency": p_tendency,
            "wind_speed_kmh": round(op_ws, 2),
            "wind_direction_deg": round(op_wind_dir, 1) if op_wind_dir is not None else None,
            "heat_index_c": round(derived_hi, 2),
            "heat_index_risk_category": hi_risk,
            "chance_of_rain_pct": round(blended_rain_prob * 100, 1),
            "expected_precipitation_mm": round(final_precip_mm, 2),
            # Operational Policy & Calibration Metadata
            "selected_source_by_variable": {
                "temperature": temp_src,
                "humidity": rh_src,
                "pressure": p_src,
                "wind_speed": ws_src,
                "wind_direction": wd_src,
                "heat_index": selected_sources.get("heat_index", "derived_from_selected_temp_and_humidity"),
            },
            "rain_probability_source": "hybrid_blend" if (rain_model_weight > 0 and rain_persistence_weight > 0) else ("learned_model" if rain_model_weight >= 1.0 else "persistence"),
            "rain_model_weight": rain_model_weight,
            "rain_persistence_weight": rain_persistence_weight,
            "rain_operational_threshold": operational_rain_threshold,
            "rain_operational_alert": rain_operational_alert,
            "policy_version": self.policy.get("policy_version", "unknown"),
            "policy_code_commit": self.policy.get("policy_code_commit", "unknown"),
            "model_code_commit": self.manifest.get("code_commit", "unknown"),
            "provenance": provenance_dict,
            # Weather Uncertainty (Explicitly Unavailable)
            "weather_uncertainty": {
                "status": "UNAVAILABLE",
                "reason": "Conformal prediction intervals apply ONLY to the internal beta water-level experiment. Weather prediction intervals are unavailable.",
            },
            # Diagnostics (Preserving raw model predictions & origin observations)
            "diagnostics": {
                "raw_learned_predictions": {
                    "temperature_c": round(pred_temp, 2),
                    "relative_humidity_pct": round(pred_rh, 1),
                    "pressure_hpa": round(pred_p, 2),
                    "wind_speed_kmh": round(pred_ws, 2),
                    "wind_direction_deg": round(pred_wind_dir, 1) if pred_wind_dir is not None else None,
                    "rain_probability": round(final_rain_prob, 4),
                    "precipitation_mm": round(final_precip_mm, 2),
                },
                "persistence_observations": {
                    "temperature_c": round(orig_temp, 2),
                    "relative_humidity_pct": round(orig_rh, 1),
                    "pressure_hpa": round(orig_p, 2),
                    "wind_speed_kmh": round(orig_ws, 2),
                    "wind_direction_deg": round(orig_wind_deg, 1) if orig_wind_deg is not None else None,
                    "rain_probability": persistence_rain_prob,
                    "precipitation_mm": round(last_observed_precip, 2),
                },
            },
            # Internal / Beta Research Module (Not for life safety)
            "water_level_beta": {
                "predicted_water_level_m": round(max(0.0, predicted_water), 2),
                "status": "INTERNAL_EXPERIMENT_BETA",
                "not_for_life_safety": True,
            },
            # Backwards Compatibility Aliases
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
            hum_step = 75.0 - _diurnal_perturb(h) * 3.0
            press_step = current_pressure - (0.3 if current_wind_speed > 15 else -0.1)
            wind_step = max(0.5, current_wind_speed + np.sin(h / 3.0) * 1.5)
            wind_sin = float(np.sin(np.radians(45.0)))
            wind_cos = float(np.cos(np.radians(45.0)))
            precip_step = 0.0

            feat = np.array([temp_step, heat_step, hum_step, press_step, wind_step, wind_sin, wind_cos, precip_step], dtype=np.float32)
            features.append(self._normalize(feat))
            dt_seq.append([1.0])

        x_tensor = torch.tensor(np.stack([features]), dtype=torch.float32)
        dt_tensor = torch.tensor(np.stack([dt_seq]), dtype=torch.float32)
        init_water_t = torch.tensor([[current_water_level]], dtype=torch.float32)

        with torch.no_grad():
            rain_prob, precip_mm, water_level = self.model(x_tensor, dt_tensor, initial_water=init_water_t)

        final_rain_prob = float(rain_prob[0, -1, 0].item())
        final_precip_mm = float(precip_mm[0, -1, 0].item())
        predicted_water = float(water_level[0, -1, 0].item())

        trajectory = []
        for step in range(seq_len):
            trajectory.append({
                "hour_offset": step + 1,
                "rain_probability": round(float(rain_prob[0, step, 0].item()) * 100, 1),
                "precipitation_mm": round(float(precip_mm[0, step, 0].item()), 2),
                "predicted_water_level": round(
                    float(water_level[0, step, 0].item()),
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
        # Test operational observed sequence forecast API with 8 canonical features
        dummy_seq = np.array([[28.0, 32.0, 75.0, 1010.0, 5.0, 0.0, 1.0, 0.0]] * 24, dtype=np.float32)
        op_result = predictor.predict_from_observed_sequence(
            telemetry_sequence=dummy_seq,
            forecast_origin_timestamp="2026-08-01T12:00:00",
            horizon_hours=1,
        )
        print("Operational API Result (8-feature input):")
        print(json.dumps(op_result, indent=2))

        # Demonstrate operational forecast strictly fails closed if live features are missing
        dummy_seq_4 = np.array([[28.0, 32.0, 5.0, 1010.0]] * 24, dtype=np.float32)
        try:
            predictor.predict_from_observed_sequence(
                telemetry_sequence=dummy_seq_4,
                forecast_origin_timestamp="2026-08-01T12:00:00",
                horizon_hours=1,
            )
            print("ERROR: Missing features did not fail closed!")
        except ValueError as e:
            print(f"\n[PASS] Operational API correctly failed closed on missing features: {e}")

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
