"""
Deterministic NWP bias-correction and forecast router.

WHY THIS IS ARITHMETIC AND NOT A NETWORK
-----------------------------------------
The measured result was: a 64k-parameter CfC trained as an NWP residual head
selected its best checkpoint at epoch 0-1 at every horizon, and improved on a
plain bias-corrected NWP forecast by 0-5%. Almost all of the available skill is
in the per-station affine correction, not in the learned residual.

Shipping a network to capture 5% would mean shipping a checkpoint, a training
pipeline, per-bundle weight regeneration, and a non-deterministic inference
path, in exchange for a gain smaller than the noise between horizons. So the
correction is expressed here as explicit, inspectable coefficients and the
router as an explicit lookup. Both are unit-testable, auditable, and cheap to
re-derive when the coefficients are refit.

WHAT IT DOES
------------
1. CORRECTION. For each (source, horizon, variable, station) it stores a shrunk
   affine map of the global forecast into local units:

       corrected = a * nwp + b        a = 1 + lambda*(a_raw - 1),  b = lambda*b_raw

   Shrinkage toward the IDENTITY map (a=1, b=0) is essential. An unshrunk fit
   made ECMWF temperature worse than raw ECMWF on validation (0.995 -> 1.110),
   because it transfers train-period structure that does not hold out of period.

2. ROUTING. For each (horizon, variable) it stores which producer to trust and,
   where a blend won on validation, the blend weight:

       producer in {persistence, lnn, nwp, blend}

   NWP loses badly at short lead (at +1h persistence beats ECMWF by 2x on every
   variable) and wins at longer lead, so a single global choice is wrong. The
   router is fit on VALIDATION and never on TEST. The LNN is the default: a
   challenger must beat it by a margin before it is allowed to take over.

3. FAIL-SAFE. Any missing coefficient, unknown key, non-finite input, or
   unavailable NWP degrades to the LNN forecast. This layer can worsen a
   forecast but can never raise, crash, or invent data. A returned value of
   None means "no forecast available" and must never be read as 0.0.

DATA RIGHTS
-----------
Nothing here fetches anything. The caller supplies NWP values it has already
obtained through a source that passed ExternalSourceRegistry. This module is
deliberately provider-agnostic so that swapping NOAA GFS (public domain) for
another source is a configuration change, not a code change.
"""

import json
import math
import os
from typing import Any, Dict, Optional, Tuple

# Physical bounds. Applied after correction so a bad coefficient cannot emit an
# impossible value. These mirror the bounds used elsewhere in the pipeline.
CLAMP = {
    "temperature": (-20.0, 60.0),
    "humidity": (0.0, 100.0),
    "pressure": (850.0, 1100.0),
    "wind_speed": (0.0, None),
}

PRODUCERS = ("persistence", "lln", "nwp", "blend")

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DEFAULT_PATH = os.path.join(DATA_DIR, "nwp_correction.json")


def artifact_path_for(source: str) -> str:
    """
    Path to the frozen artifact for a given NWP source.

    Coefficients are keyed by source because the correction is fitted
    against one model's bias structure. ECMWF and GFS have different
    biases, so their coefficients are not interchangeable and are stored
    in separate files. The generic DEFAULT_PATH is kept as a fallback for
    single-source deployments.
    """
    return os.path.join(DATA_DIR, f"nwp_correction_{source}.json")


def _finite(x: Any) -> bool:
    """True only for a real, finite number. Strings and None are rejected."""
    if x is None or isinstance(x, bool):
        return False
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _as_float(x: Any) -> Optional[float]:
    return float(x) if _finite(x) else None


def select(producer: str, blend_weight: float, nwp_corrected: Any,
           model_value: Any, persistence_value: Any) -> Tuple[Optional[float], str]:
    """
    Choose the value to publish, given a planned producer and the candidates.

    The fallback chain is total: whatever the plan says, the result is the best
    available candidate, never an exception and never a fabricated number. A
    returned None means no candidate existed at all and must be read as "no
    forecast", not as 0.0.

        nwp          -> corrected NWP, else the LNN
        lln          -> the LNN
        persistence  -> persistence, else the LNN
        blend        -> weighted mean of corrected NWP and the LNN
    """
    corrected = _as_float(nwp_corrected)
    model = _as_float(model_value)
    pers = _as_float(persistence_value)
    weight = _as_float(blend_weight)
    if weight is None:
        weight = 0.0
    weight = min(1.0, max(0.0, weight))

    if producer == "nwp" and corrected is not None:
        return corrected, "nwp"
    if producer == "persistence" and pers is not None:
        return pers, "persistence"
    if producer == "blend" and corrected is not None and model is not None:
        return weight * corrected + (1.0 - weight) * model, "blend"

    # Anything the plan asked for that is unavailable degrades to the LNN, then
    # to persistence. The LNN is the incumbent, so it is the safe floor.
    if model is not None:
        return model, "lln"
    if corrected is not None:
        return corrected, "nwp"
    if pers is not None:
        return pers, "persistence"
    return None, "none"


class NwpCorrection:
    """Applies shrunk affine NWP correction and resolves the producer per horizon."""

    def __init__(self, artifact: Dict[str, Any]):
        artifact = artifact if isinstance(artifact, dict) else {}
        self.artifact = artifact
        self.version = artifact.get("version", "unknown")
        self.source = artifact.get("nwp_source", "unknown")
        self.fit_report = artifact.get("fit", {})
        self._coef = {}
        coeffs = artifact.get("coefficients")
        if isinstance(coeffs, dict):
            for src, per_h in coeffs.items():
                if not isinstance(per_h, dict):
                    continue
                for hz, per_st in per_h.items():
                    if not isinstance(per_st, dict):
                        continue
                    for st, per_v in per_st.items():
                        if not isinstance(per_v, dict):
                            continue
                        for var, ab in per_v.items():
                            a = _as_float(ab[0] if isinstance(ab, (list, tuple))
                                          and len(ab) == 2 else None)
                            b = _as_float(ab[1] if isinstance(ab, (list, tuple))
                                          and len(ab) == 2 else None)
                            if a is not None and b is not None:
                                self._coef[(str(src), str(hz), str(st), str(var))] = (a, b)
        self._route = {}
        routing = artifact.get("routing")
        if isinstance(routing, dict):
            for hz, per_v in routing.items():
                if not isinstance(per_v, dict):
                    continue
                for var, r in per_v.items():
                    if isinstance(r, dict) and r.get("producer") in PRODUCERS:
                        self._route[(str(hz), str(var))] = r

    # ---- loading -------------------------------------------------------
    @classmethod
    def load(cls, path: str = DEFAULT_PATH) -> Optional["NwpCorrection"]:
        """Load the artifact. Returns None if absent, so absence is not fatal."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return cls(json.load(f))
        except (OSError, ValueError, TypeError):
            return None

    # ---- introspection -------------------------------------------------
    @property
    def available(self) -> bool:
        return bool(self._coef)

    def n_coefficients(self) -> int:
        return len(self._coef)

    def route_for(self, horizon_h: Any, variable: str) -> Optional[Dict[str, Any]]:
        return self._route.get((str(horizon_h), str(variable)))

    def describe(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "nwp_source": self.source,
            "n_coefficients": self.n_coefficients(),
            "n_routes": len(self._route),
            "fit": self.fit_report,
        }

    # ---- correction ----------------------------------------------------
    def coefficient(self, station_id: str, horizon_h: Any,
                    variable: str) -> Optional[Tuple[float, float]]:
        if station_id is None or variable is None:
            return None
        return self._coef.get((self.source, str(horizon_h), str(station_id),
                               str(variable)))

    def correct(self, nwp_value: Any, station_id: str, horizon_h: Any,
                variable: str) -> Optional[float]:
        """
        Map a raw NWP value into corrected local units.

        Returns None when no coefficient exists or the input is unusable, so the
        caller falls back to its own forecast. Never raises.
        """
        if variable not in CLAMP:
            return None
        ab = self.coefficient(station_id, horizon_h, variable)
        if ab is None:
            return None
        raw = _as_float(nwp_value)
        if raw is None:
            return None
        out = ab[0] * raw + ab[1]
        if not _finite(out):
            return None
        lo, hi = CLAMP[variable]
        if lo is not None and out < lo:
            out = lo
        if hi is not None and out > hi:
            out = hi
        return out

    # ---- routing -------------------------------------------------------
    def plan(self, horizon_h: Any, variable: str, nwp_value: Any = None,
             station_id: Optional[str] = None):
        """
        Work out which producer to use, and the corrected NWP value if any.

        Returns (producer, nwp_corrected, blend_weight). Callers pass the result
        to select() along with their own LNN and persistence values. Splitting
        it this way keeps the correction step (which needs a station) separate
        from the choice step (which does not).
        """
        corrected = None
        if station_id is not None:
            corrected = self.correct(nwp_value, station_id, horizon_h, variable)
        route = self.route_for(horizon_h, variable)
        if not isinstance(route, dict):
            return "lln", corrected, 0.0
        producer = route.get("producer")
        if producer not in PRODUCERS:
            producer = "lln"
        weight = _as_float(route.get("blend_weight"))
        if weight is None:
            weight = 0.0
        return producer, corrected, min(1.0, max(0.0, weight))
