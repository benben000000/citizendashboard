"""
Gated integration of the NWP-corrected hybrid router into operational forecasts.

THE GATE
--------
This layer is DISABLED by default. It produces exactly the forecast the LNN
produces today unless every one of the following holds:

  1. `enabled` is true (default False, or the NWP_HYBRID_ENABLED env var)
  2. the NWP source is APPROVED in ExternalSourceRegistry
  3. the source-keyed correction artifact exists and is internally consistent
  4. the supplied NWP values are fresh (within max_age_hours) and finite
  5. the station and horizon have a route and a coefficient

Any failure degrades to the LNN output UNCHANGED, with a reason recorded. The
gate never raises into the forecast path and never emits a value the LNN did
not already produce. That is the whole safety contract: the worst case of
turning this on is that it does nothing.

WHY A POST-PROCESSOR
--------------------
The router consumes the LNN's output and the NWP forecast, and returns a
forecast. Wrapping the predictor keeps the correction out of the model's hot
path, lets it be switched off without touching the model, and makes the whole
thing testable by feeding it two dicts.

DATA RIGHTS
-----------
The source id here must match a record in the external source registry. NOAA GFS
is registered and public domain; PAGASA is UNKNOWN_BLOCKED and this layer will
refuse to use it. The registry check is not advisory -- it is the reason this
class exists rather than a direct call into the adapter.
"""

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from external_source_registry import ExternalSourceRegistry
from nwp_correction import NwpCorrection, artifact_path_for, select

# Registry source id -> NWP model key used in the correction artifacts.
#
# These are deliberately different namespaces. The registry id is a data-rights
# identity ("which licence covers this feed"); the model key is a provider's
# internal name for the run ("gfs_seamless"). Assuming they coincide is exactly
# the kind of silent mismatch that ships one model's coefficients against
# another's bias structure, so the mapping is stated here and reviewed rather
# than inferred from string similarity.
#
# A registered source with no entry here cannot be used: there are no
# coefficients fitted for it, and guessing would be unsound.
SOURCE_MODELS = {
    "noaa_gfs_v1": "gfs_seamless",
    # "pagasa_nwp_v1": "<unset>",   # UNKNOWN_BLOCKED in the registry
    # "ecmwf_ifs025": "ecmwf_ifs025",  # benchmarked, but not licence-cleared
}

# Which NWP field carries each variable. The correction module works in the
# station's own units, so the provider is responsible for handing over values
# already in those units (wind in m/s, not km/h). The benchmark pipeline
# divides by 3.6 at ingest; a live provider must do the same or the wind
# correction will be off by a factor of 3.6.
VARIABLE_FIELDS = {
    "temperature": "temperature",
    "humidity": "humidity",
    "pressure": "pressure",
    "wind_speed": "wind_speed",
}

# The LNN output key for each variable.
MODEL_KEYS = {
    "temperature": "temperature_c",
    "humidity": "relative_humidity_pct",
    "pressure": "pressure_hpa",
    "wind_speed": "wind_speed_kmh",
}


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _parse_ts(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


class HybridRouter:
    """
    Applies the NWP bias correction to LNN output, subject to the licence gate.

    Construct once per process. `apply` is cheap and safe to call per forecast.
    """

    def __init__(
        self,
        source_id: str = "noaa_gfs_v1",
        enabled: Optional[bool] = None,
        registry: Optional[ExternalSourceRegistry] = None,
        artifact_path: Optional[str] = None,
        max_age_hours: float = 24.0,
    ):
        self.source_id = source_id
        self.max_age_hours = float(max_age_hours)
        self.enabled = _env_flag("NWP_HYBRID_ENABLED", False) if enabled is None \
            else bool(enabled)
        self.blocked_reason: Optional[str] = None
        self.correction: Optional[NwpCorrection] = None

        # --- gate 2: licence -------------------------------------------------
        if not self.enabled:
            self.blocked_reason = "disabled by configuration"
            return
        try:
            reg = registry or ExternalSourceRegistry()
            ok, reason = reg.is_production_eligible(source_id)
        except Exception as exc:  # noqa: BLE001 - a broken gate must not crash
            self.blocked_reason = f"registry check failed: {type(exc).__name__}"
            return
        if not ok:
            self.blocked_reason = f"source {source_id!r} not production-eligible: {reason}"
            return

        # --- gate 3: artifact ------------------------------------------------
        model_key = SOURCE_MODELS.get(source_id)
        if model_key is None:
            self.blocked_reason = (
                f"no correction coefficients are fitted for {source_id!r}; "
                "register a SOURCE_MODELS mapping and refit first")
            return
        path = artifact_path or artifact_path_for(model_key)
        corr = NwpCorrection.load(path)
        if corr is None or not corr.available:
            self.blocked_reason = f"correction artifact missing or empty at {path}"
            return
        # Compare against the MODEL key: the artifact records which model it was
        # fitted against, while source_id is the registry's data-rights identity
        # for the same feed. Comparing the two directly would always mismatch.
        if corr.source != model_key:
            self.blocked_reason = (
                f"artifact is for model {corr.source!r}, expected {model_key!r} "
                f"(registry source {source_id!r})")
            return
        self.correction = corr

    # ---- introspection -------------------------------------------------
    @property
    def active(self) -> bool:
        """True only when the gate is fully open and the router may be used."""
        return self.correction is not None

    def status(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "active": self.active,
            "source_id": self.source_id,
            "blocked_reason": self.blocked_reason,
            "n_coefficients": self.correction.n_coefficients() if self.correction else 0,
            "correction_version": self.correction.version if self.correction else None,
            "max_age_hours": self.max_age_hours,
        }

    # ---- the NWP feed --------------------------------------------------
    def _nwp_is_fresh(self, nwp_timestamp: Any, now: Optional[datetime]) -> bool:
        """
        `nwp_timestamp` MUST be the forecast's ISSUE time, not its valid time.

        Freshness is about when the model said it, not about when the weather is
        supposed to arrive. A 24h forecast with a valid time six hours away is
        stale if its cycle is a day old, and a 1h forecast is fresh even though
        its valid time is almost now -- the opposite of what a valid-time check
        would conclude.
        """
        if nwp_timestamp is None:
            # No timestamp means we cannot prove freshness, so we do not use it.
            return False
        ts = _parse_ts(nwp_timestamp)
        if ts is None:
            return False
        ref = now or datetime.now(timezone.utc)
        age = (ref - ts).total_seconds() / 3600.0
        # This is a FEED-HEALTH check, not a lead-time limit. A 24h forecast
        # legitimately comes from a cycle that is many hours old; what must not
        # happen is a feed that has stopped advancing. So the bound is generous
        # and the issue time is the reference, which is what makes it meaningful
        # across every lead.
        return -48.0 <= age <= self.max_age_hours

    # ---- the application ------------------------------------------------
    def apply(
        self,
        station_id: str,
        horizon_hours: Any,
        model_output: Dict[str, Any],
        nwp: Optional[Dict[str, Any]] = None,
        nwp_timestamp: Any = None,
        now: Optional[datetime] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Return (forecast, provenance).

        `forecast` is a COPY of model_output with corrected fields written in.
        On any gate failure it is model_output unchanged, and provenance says
        why. The caller's dict is never mutated in place.
        """
        out = dict(model_output or {})
        prov: Dict[str, Any] = {
            "applied": False,
            "source_id": self.source_id,
            "producers": {},
            "reasons": [],
        }

        if not self.active:
            prov["reasons"].append(self.blocked_reason or "router inactive")
            return out, prov
        if not station_id:
            prov["reasons"].append("no station id")
            return out, prov
        if not self._nwp_is_fresh(nwp_timestamp, now):
            prov["reasons"].append("NWP missing or stale")
            return out, prov
        if not isinstance(nwp, dict):
            prov["reasons"].append("no NWP values supplied")
            return out, prov

        prov["applied"] = True
        for variable, field in VARIABLE_FIELDS.items():
            model_key = MODEL_KEYS[variable]
            if model_key not in out:
                continue
            model_value = out[model_key]
            raw = nwp.get(field)
            producer, corrected, weight = self.correction.plan(
                horizon_hours, variable, nwp_value=raw, station_id=station_id)
            value, used = select(producer, weight, corrected, model_value, None)
            prov["producers"][variable] = {
                "producer": used,
                "model_value": model_value,
                "nwp_raw": raw,
                "published": value,
            }
            if used == "lln" or value is None:
                # The router declined this cell; leave the model's own value.
                continue
            if value == model_value:
                continue
            out[model_key] = value
        return out, prov

    def batch_apply(
        self,
        station_id: str,
        forecasts: List[Dict[str, Any]],
        nwp_by_horizon: Optional[Dict[Any, Dict[str, Any]]] = None,
        nwp_timestamps: Optional[Dict[Any, Any]] = None,
        now: Optional[datetime] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Apply across a multi-horizon forecast set. Never raises."""
        nwp_by_horizon = nwp_by_horizon or {}
        nwp_timestamps = nwp_timestamps or {}
        results, provs = [], []
        for f in forecasts:
            h = f.get("horizon_hours", f.get("horizon"))
            try:
                r, p = self.apply(station_id, h, f, nwp_by_horizon.get(h),
                                  nwp_timestamps.get(h), now)
            except Exception as exc:  # noqa: BLE001
                r, p = dict(f or {}), {"applied": False, "error": f"{type(exc).__name__}: {exc}"}
            results.append(r)
            provs.append(p)
        return results, provs
