"""
Autonomous Telemetry Anomaly Detection and Quality Assessment Engine.

Implements dedicated anomaly detection distinguishing:
  1. Physical Anomalies: real weather extremes outside expected climatological bounds.
  2. Sensor Anomalies: stuck sensors, discontinuous spikes, impossible rates of change, out-of-bounds readings.
  3. Forecast Anomalies: material deviation between operational forecasts and subsequent ground-truth observations (> 3-sigma).

Adheres to scientific rules:
  - Does NOT treat anomaly detection as an ordinary regression target.
  - Distinguishes sensor malfunction from rare extreme weather.
  - Computes robust rolling statistics (Median & Median Absolute Deviation / MAD).
  - Produces structured, actionable diagnostics with severity scoring in [0.0, 1.0].
"""

import math
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# Physical bounds for Philippine tropical surface meteorology
PHYSICAL_BOUNDS = {
    "temperature": (10.0, 50.0),       # Celsius
    "humidity": (10.0, 100.0),         # Relative humidity %
    "pressure": (900.0, 1050.0),       # hPa
    "wind_speed": (0.0, 180.0),        # km/h
    "precipitation": (0.0, 150.0),     # mm/h
    "heat_index": (10.0, 75.0),        # Celsius
}

# Maximum physically plausible 1-hour rates of change
MAX_1H_DELTA = {
    "temperature": 8.0,                # Celsius/hour
    "humidity": 40.0,                  # %/hour
    "pressure": 15.0,                  # hPa/hour (severe tropical cyclone eyewall limit)
    "wind_speed": 60.0,                # km/h/hour
}

# Physical extreme weather thresholds (genuine meteorological events)
PHYSICAL_EXTREMES = {
    "temperature_high": 42.0,          # Extreme heatwave (Celsius)
    "temperature_low": 15.0,           # Extreme cool wave (Celsius)
    "pressure_low": 985.0,             # Deep tropical depression / typhoon (hPa)
    "wind_speed_high": 65.0,           # Gale force / tropical storm (km/h)
    "precipitation_intense": 30.0,     # Intense torrential rainfall (mm/h)
    "heat_index_danger": 45.0,         # Danger / Extreme Danger heat index (Celsius)
}


class AnomalyRecord:
    """Structured representation of a detected anomaly."""

    def __init__(
        self,
        anomaly_type: str,
        affected_variable: str,
        timestamp: Optional[str],
        severity_score: float,
        explanation: str,
        is_actionable: bool,
        station_id: Optional[str] = None,
        raw_value: Optional[float] = None,
        threshold_version: str = "1.0.0",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        if anomaly_type not in ("physical", "sensor", "forecast"):
            raise ValueError(f"Invalid anomaly_type '{anomaly_type}'. Must be 'physical', 'sensor', or 'forecast'.")

        self.anomaly_type = anomaly_type
        self.affected_variable = affected_variable
        self.timestamp = timestamp
        self.severity_score = max(0.0, min(1.0, float(severity_score)))
        self.explanation = explanation
        self.is_actionable = bool(is_actionable)
        self.station_id = station_id
        self.raw_value = raw_value
        self.threshold_version = threshold_version
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "anomaly_type": self.anomaly_type,
            "affected_variable": self.affected_variable,
            "station": self.station_id,
            "timestamp": self.timestamp,
            "severity_score": round(self.severity_score, 4),
            "threshold_version": self.threshold_version,
            "explanation": self.explanation,
            "is_actionable": self.is_actionable,
            "raw_value": round(self.raw_value, 4) if self.raw_value is not None else None,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return (
            f"<AnomalyRecord type={self.anomaly_type} var={self.affected_variable} "
            f"severity={self.severity_score:.2f} actionable={self.is_actionable}>"
        )


class TelemetryAnomalyDetector:
    """
    Multi-faceted detector evaluating physical limits, stuck sensors,
    discontinuous spikes, robust distribution outliers, and forecast residuals.
    """

    def __init__(
        self,
        stuck_sensor_min_steps: int = 6,
        robust_z_threshold: float = 3.5,
        forecast_residual_sigma: float = 3.0,
    ):
        self.stuck_sensor_min_steps = stuck_sensor_min_steps
        self.robust_z_threshold = robust_z_threshold
        self.forecast_residual_sigma = forecast_residual_sigma
        self.version = "1.0.0"

    def detect_sensor_anomalies(
        self,
        values: np.ndarray,
        variable_name: str,
        timestamps: Optional[List[str]] = None,
        station_id: Optional[str] = None,
    ) -> List[AnomalyRecord]:
        """
        Detect sensor malfunctions:
          1. Sensor bound violations (below min or above max capability).
          2. Stuck sensor (constant / zero-variance series over window).
          3. Discontinuous jump (impossible 1-hour rate of change).
        """
        anomalies = []
        arr = np.asarray(values, dtype=np.float64)
        n = len(arr)
        if n == 0:
            return anomalies

        bounds = PHYSICAL_BOUNDS.get(variable_name)
        max_delta = MAX_1H_DELTA.get(variable_name)

        # 1. Physical Sensor Bound Violations
        if bounds:
            low_b, high_b = bounds
            for idx in range(n):
                val = arr[idx]
                if np.isnan(val):
                    continue
                if val < low_b or val > high_b:
                    ts = timestamps[idx] if timestamps and idx < len(timestamps) else None
                    sev = min(1.0, abs(val - (low_b if val < low_b else high_b)) / max(1.0, high_b - low_b) + 0.5)
                    anomalies.append(
                        AnomalyRecord(
                            anomaly_type="sensor",
                            affected_variable=variable_name,
                            timestamp=ts,
                            severity_score=sev,
                            explanation=f"Sensor bound violation: {val:.2f} is outside physical limit [{low_b}, {high_b}]",
                            is_actionable=True,
                            station_id=station_id,
                            raw_value=val,
                            threshold_version=self.version,
                            metadata={"check": "sensor_bounds", "bounds": [low_b, high_b]},
                        )
                    )

        # 2. Stuck Sensor Detection (identical values or near-zero variance over consecutive steps)
        # Note: precipitation naturally stays 0.0 for days, so stuck sensor check applies to continuous variables
        if variable_name in ("temperature", "humidity", "pressure") and n >= self.stuck_sensor_min_steps:
            consecutive_count = 1
            for idx in range(1, n):
                if not np.isnan(arr[idx]) and not np.isnan(arr[idx - 1]) and abs(arr[idx] - arr[idx - 1]) < 1e-5:
                    consecutive_count += 1
                else:
                    consecutive_count = 1

                if consecutive_count >= self.stuck_sensor_min_steps:
                    ts = timestamps[idx] if timestamps and idx < len(timestamps) else None
                    val = float(arr[idx])
                    anomalies.append(
                        AnomalyRecord(
                            anomaly_type="sensor",
                            affected_variable=variable_name,
                            timestamp=ts,
                            severity_score=min(1.0, 0.5 + 0.08 * (consecutive_count - self.stuck_sensor_min_steps)),
                            explanation=f"Stuck {variable_name} sensor: {consecutive_count} consecutive identical readings ({val:.2f}) with zero variance",
                            is_actionable=True,
                            station_id=station_id,
                            raw_value=val,
                            threshold_version=self.version,
                            metadata={"check": "stuck_sensor", "consecutive_steps": consecutive_count},
                        )
                    )
                    break

        # 3. Discontinuous Jump / Rate-of-Change Violation
        if max_delta and n >= 2:
            for idx in range(1, n):
                v_curr = arr[idx]
                v_prev = arr[idx - 1]
                if np.isnan(v_curr) or np.isnan(v_prev):
                    continue
                delta = abs(v_curr - v_prev)
                if delta > max_delta:
                    ts = timestamps[idx] if timestamps and idx < len(timestamps) else None
                    sev = min(1.0, delta / max_delta * 0.7)
                    anomalies.append(
                        AnomalyRecord(
                            anomaly_type="sensor",
                            affected_variable=variable_name,
                            timestamp=ts,
                            severity_score=sev,
                            explanation=f"Discontinuous jump: delta {delta:.2f} exceeds physical rate limit {max_delta:.2f}/h",
                            is_actionable=True,
                            station_id=station_id,
                            raw_value=v_curr,
                            threshold_version=self.version,
                            metadata={"check": "rate_of_change", "delta": delta, "max_delta": max_delta},
                        )
                    )

        return anomalies

    def detect_physical_anomalies(
        self,
        values: np.ndarray,
        variable_name: str,
        timestamps: Optional[List[str]] = None,
        station_id: Optional[str] = None,
    ) -> List[AnomalyRecord]:
        """
        Detect genuine extreme weather anomalies:
          1. Severe tropical weather thresholds (deep low pressure, gale wind, torrential rain).
          2. Robust Median / MAD outliers on plausible physical variations.
        """
        anomalies = []
        arr = np.asarray(values, dtype=np.float64)
        n = len(arr)
        if n == 0:
            return anomalies

        valid_mask = ~np.isnan(arr)
        if not np.any(valid_mask):
            return anomalies

        # 1. Deterministic Extreme Weather Thresholds
        for idx in range(n):
            val = arr[idx]
            if np.isnan(val):
                continue
            ts = timestamps[idx] if timestamps and idx < len(timestamps) else None

            if variable_name == "pressure" and val < PHYSICAL_EXTREMES["pressure_low"]:
                anomalies.append(
                    AnomalyRecord(
                        anomaly_type="physical",
                        affected_variable=variable_name,
                        timestamp=ts,
                        severity_score=min(1.0, (PHYSICAL_EXTREMES["pressure_low"] - val) / 20.0 + 0.6),
                        explanation=f"Deep barometric low ({val:.1f} hPa): consistent with intense tropical depression / typhoon eyewall",
                        is_actionable=True,
                        station_id=station_id,
                        raw_value=val,
                        threshold_version=self.version,
                        metadata={"extreme_category": "tropical_cyclone_barometric_low"},
                    )
                )
            elif variable_name == "wind_speed" and val >= PHYSICAL_EXTREMES["wind_speed_high"]:
                anomalies.append(
                    AnomalyRecord(
                        anomaly_type="physical",
                        affected_variable=variable_name,
                        timestamp=ts,
                        severity_score=min(1.0, (val - PHYSICAL_EXTREMES["wind_speed_high"]) / 50.0 + 0.6),
                        explanation=f"Severe wind event ({val:.1f} km/h): gale force winds exceeding PAGASA storm criteria",
                        is_actionable=True,
                        station_id=station_id,
                        raw_value=val,
                        threshold_version=self.version,
                        metadata={"extreme_category": "gale_force_wind"},
                    )
                )
            elif variable_name == "precipitation" and val >= PHYSICAL_EXTREMES["precipitation_intense"]:
                anomalies.append(
                    AnomalyRecord(
                        anomaly_type="physical",
                        affected_variable=variable_name,
                        timestamp=ts,
                        severity_score=min(1.0, (val - PHYSICAL_EXTREMES["precipitation_intense"]) / 40.0 + 0.7),
                        explanation=f"Torrential rainfall ({val:.1f} mm/h): intense cloudburst exceeding emergency warning threshold",
                        is_actionable=True,
                        station_id=station_id,
                        raw_value=val,
                        threshold_version=self.version,
                        metadata={"extreme_category": "torrential_rainfall"},
                    )
                )
            elif variable_name == "temperature" and val >= PHYSICAL_EXTREMES["temperature_high"]:
                anomalies.append(
                    AnomalyRecord(
                        anomaly_type="physical",
                        affected_variable=variable_name,
                        timestamp=ts,
                        severity_score=min(1.0, (val - PHYSICAL_EXTREMES["temperature_high"]) / 6.0 + 0.5),
                        explanation=f"Extreme heatwave ({val:.1f} C): surface temperature exceeding dangerous seasonal threshold",
                        is_actionable=True,
                        station_id=station_id,
                        raw_value=val,
                        threshold_version=self.version,
                        metadata={"extreme_category": "extreme_heat"},
                    )
                )

        # 2. Robust MAD (Median Absolute Deviation) Outliers over the window
        valid_vals = arr[valid_mask]
        if len(valid_vals) >= 12:
            med = float(np.median(valid_vals))
            mad = float(np.median(np.abs(valid_vals - med)))
            # Standard normal scaling factor for MAD: 1.4826
            scale = 1.4826 * mad
            if scale > 1e-4:
                for idx in range(n):
                    val = arr[idx]
                    if np.isnan(val):
                        continue
                    robust_z = abs(val - med) / scale
                    if robust_z >= self.robust_z_threshold:
                        # Make sure not already flagged under deterministic extremes
                        ts = timestamps[idx] if timestamps and idx < len(timestamps) else None
                        already_flagged = any(
                            a.affected_variable == variable_name and a.timestamp == ts for a in anomalies
                        )
                        if not already_flagged:
                            sev = min(1.0, (robust_z - self.robust_z_threshold) / 4.0 + 0.4)
                            anomalies.append(
                                AnomalyRecord(
                                    anomaly_type="physical",
                                    affected_variable=variable_name,
                                    timestamp=ts,
                                    severity_score=sev,
                                    explanation=f"Statistical climatological outlier: robust z-score {robust_z:.2f} relative to median {med:.2f}",
                                    is_actionable=False,
                                    station_id=station_id,
                                    raw_value=val,
                                    threshold_version=self.version,
                                    metadata={"check": "robust_mad", "median": med, "mad": mad, "robust_z": robust_z},
                                )
                            )

        return anomalies

    def detect_forecast_anomalies(
        self,
        predicted_value: float,
        observed_value: float,
        variable_name: str,
        expected_std: float,
        timestamp: Optional[str] = None,
        station_id: Optional[str] = None,
    ) -> Optional[AnomalyRecord]:
        """
        Evaluate forecast residual deviation against expected training error.
        Flagged when absolute residual > forecast_residual_sigma * expected_std.
        """
        if np.isnan(predicted_value) or np.isnan(observed_value):
            return None

        residual = abs(observed_value - predicted_value)
        threshold = max(1e-3, self.forecast_residual_sigma * expected_std)

        if residual > threshold:
            sigmas = residual / max(1e-4, expected_std)
            sev = min(1.0, (sigmas - self.forecast_residual_sigma) / 4.0 + 0.5)
            return AnomalyRecord(
                anomaly_type="forecast",
                affected_variable=variable_name,
                timestamp=timestamp,
                severity_score=sev,
                explanation=f"Forecast residual anomaly: |obs - pred| = {residual:.2f} ({sigmas:.1f} sigma deviation, threshold {threshold:.2f})",
                is_actionable=True,
                station_id=station_id,
                raw_value=observed_value,
                threshold_version=self.version,
                metadata={
                    "predicted": predicted_value,
                    "observed": observed_value,
                    "residual": residual,
                    "sigmas": sigmas,
                    "expected_std": expected_std,
                },
            )
        return None

    def audit_sequence(
        self,
        telemetry_sequence: np.ndarray,
        forecast_origin_timestamp: Optional[str] = None,
        station_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive audit of an observed historical input sequence [seq_len, 8].
        Features: (temperature, heat_index, humidity, pressure, wind_speed, wind_sin, wind_cos, precipitation).
        """
        arr = np.asarray(telemetry_sequence, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] < 8:
            return {
                "has_anomaly": False,
                "sensor_quality_flag": "QUARANTINED",
                "sensor_anomalies": [],
                "physical_anomalies": [],
                "total_anomaly_count": 0,
                "summary": "Invalid input shape for sequence audit",
                "actionable": True,
            }

        # Check for NaN / Inf
        if np.isnan(arr).any() or np.isinf(arr).any():
            return {
                "has_anomaly": True,
                "sensor_quality_flag": "QUARANTINED",
                "sensor_anomalies": [
                    {
                        "anomaly_type": "sensor",
                        "affected_variable": "sequence",
                        "station": station_id,
                        "timestamp": forecast_origin_timestamp,
                        "severity_score": 1.0,
                        "threshold_version": self.version,
                        "explanation": "Input sequence contains NaN or Infinite sensor values",
                        "is_actionable": True,
                        "raw_value": None,
                    }
                ],
                "physical_anomalies": [],
                "total_anomaly_count": 1,
                "summary": "Sensor sequence quarantined due to NaN/Inf values",
                "actionable": True,
            }

        temp_col = arr[:, 0]
        rh_col = arr[:, 2]
        p_col = arr[:, 3]
        ws_col = arr[:, 4]
        precip_col = arr[:, 7]

        sensor_anomalies = []
        physical_anomalies = []

        # Audit variables
        cols = [
            ("temperature", temp_col),
            ("humidity", rh_col),
            ("pressure", p_col),
            ("wind_speed", ws_col),
            ("precipitation", precip_col),
        ]

        for var_name, col_data in cols:
            s_anoms = self.detect_sensor_anomalies(col_data, var_name, station_id=station_id)
            sensor_anomalies.extend([a.to_dict() for a in s_anoms])

            p_anoms = self.detect_physical_anomalies(col_data, var_name, station_id=station_id)
            physical_anomalies.extend([a.to_dict() for a in p_anoms])

        total_anom = len(sensor_anomalies) + len(physical_anomalies)
        has_anom = total_anom > 0

        if len(sensor_anomalies) > 0:
            quality_flag = "WARNING" if all(a["severity_score"] < 0.8 for a in sensor_anomalies) else "QUARANTINED"
        elif len(physical_anomalies) > 0:
            quality_flag = "EXTREME_WEATHER"
        else:
            quality_flag = "CLEAN"

        summary_parts = []
        if sensor_anomalies:
            summary_parts.append(f"{len(sensor_anomalies)} sensor anomaly(s)")
        if physical_anomalies:
            summary_parts.append(f"{len(physical_anomalies)} physical weather event(s)")
        summary = "; ".join(summary_parts) if summary_parts else "Input sequence clean and within physical limits"

        actionable = any(a["is_actionable"] for a in sensor_anomalies + physical_anomalies)

        return {
            "has_anomaly": has_anom,
            "sensor_quality_flag": quality_flag,
            "sensor_anomalies": sensor_anomalies,
            "physical_anomalies": physical_anomalies,
            "total_anomaly_count": total_anom,
            "summary": summary,
            "actionable": actionable,
        }

    def detect_distribution_anomalies(
        self,
        observed_value: float,
        p10: float,
        p50: float,
        p90: float,
        variable_name: str,
        timestamp: Optional[str] = None,
        station_id: Optional[str] = None,
    ) -> Optional[AnomalyRecord]:
        """
        Evaluate observation against calibrated forecast distribution [p10, p90].
        Separates physical weather extremes from sensor defects.
        """
        if np.isnan(observed_value) or np.isnan(p10) or np.isnan(p90):
            return None

        # Check physical bounds first
        if variable_name in PHYSICAL_BOUNDS:
            low, high = PHYSICAL_BOUNDS[variable_name]
            if observed_value < low or observed_value > high:
                return AnomalyRecord(
                    anomaly_type="sensor",
                    affected_variable=variable_name,
                    timestamp=timestamp,
                    severity_score=1.0,
                    explanation=f"Out-of-bounds sensor defect: {variable_name}={observed_value:.2f} outside [{low}, {high}]",
                    is_actionable=True,
                    station_id=station_id,
                    raw_value=observed_value,
                    threshold_version=self.version,
                    metadata={"p10": p10, "p50": p50, "p90": p90, "defect": "out_of_bounds"},
                )

        # Check distribution bounds
        width = max(0.1, p90 - p10)
        if observed_value < p10:
            deviation = (p10 - observed_value) / width
            if deviation > 0.5:
                sev = min(1.0, 0.4 + 0.3 * deviation)
                return AnomalyRecord(
                    anomaly_type="physical",
                    affected_variable=variable_name,
                    timestamp=timestamp,
                    severity_score=sev,
                    explanation=f"Climatological low excursion: {observed_value:.2f} below forecast p10 ({p10:.2f})",
                    is_actionable=True,
                    station_id=station_id,
                    raw_value=observed_value,
                    threshold_version=self.version,
                    metadata={"p10": p10, "p50": p50, "p90": p90, "deviation_ratio": deviation},
                )
        elif observed_value > p90:
            deviation = (observed_value - p90) / width
            if deviation > 0.5:
                sev = min(1.0, 0.4 + 0.3 * deviation)
                return AnomalyRecord(
                    anomaly_type="physical",
                    affected_variable=variable_name,
                    timestamp=timestamp,
                    severity_score=sev,
                    explanation=f"Climatological high excursion: {observed_value:.2f} above forecast p90 ({p90:.2f})",
                    is_actionable=True,
                    station_id=station_id,
                    raw_value=observed_value,
                    threshold_version=self.version,
                    metadata={"p10": p10, "p50": p50, "p90": p90, "deviation_ratio": deviation},
                )
        return None

    def evaluate_anomaly_events_with_budget(
        self,
        detected_anomalies: List[Dict[str, Any]],
        reviewed_events: List[Dict[str, Any]],
        total_monitoring_days: float = 30.0,
        false_alarm_budget_per_day: float = 2.0,
    ) -> Dict[str, Any]:
        """
        Evaluate anomaly detection performance against reviewed extreme event labels with an explicit false alarm budget.
        """
        tp = 0
        fn_events = []
        matched_detected = set()

        for ev in reviewed_events:
            ev_ts = ev.get("timestamp")
            ev_var = ev.get("affected_variable")
            matched = False
            for idx, anom in enumerate(detected_anomalies):
                if anom.get("affected_variable") == ev_var and anom.get("timestamp") == ev_ts:
                    matched = True
                    matched_detected.add(idx)
                    break
            if matched:
                tp += 1
            else:
                fn_events.append(ev)

        fp = len(detected_anomalies) - len(matched_detected)
        total_events = len(reviewed_events)
        event_recall = (tp / max(1, total_events)) * 100.0 if total_events > 0 else 100.0
        fa_per_day = fp / max(0.1, total_monitoring_days)
        within_budget = fa_per_day <= false_alarm_budget_per_day

        return {
            "reviewed_events_count": total_events,
            "detected_anomalies_count": len(detected_anomalies),
            "true_positives": tp,
            "false_positives": fp,
            "event_recall_pct": round(event_recall, 2),
            "false_alarms_per_day": round(fa_per_day, 2),
            "false_alarm_budget_per_day": false_alarm_budget_per_day,
            "within_false_alarm_budget": within_budget,
            "missed_events_count": len(fn_events),
            "status": "PASS" if within_budget and event_recall >= 70.0 else "WARNING",
        }
