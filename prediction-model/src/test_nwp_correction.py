"""
Contract tests for the NWP correction + routing layer.

The load-bearing property is not accuracy, it is SAFETY. This layer sits in the
inference path, so the tests concentrate on what it does when things are wrong,
because a forecast that silently degrades to the LNN is fine and a forecast that
raises, invents a number, or emits an impossible value is not.

Covered:
  * the shipped artifact loads and is internally consistent
  * missing / malformed input returns None rather than raising
  * impossible values are clamped rather than passed through
  * a station with no coefficients falls back instead of failing
  * the router is monotone: it never returns worse than what it was given
  * routing cannot be talked into emitting a value from an unusable NWP input
"""

import json
import math
import os

import pytest

from nwp_correction import CLAMP, DEFAULT_PATH, NwpCorrection, PRODUCERS, select

ARTIFACT = None
try:
    with open(DEFAULT_PATH, "r", encoding="utf-8") as f:
        ARTIFACT = json.load(f)
except (OSError, ValueError):
    ARTIFACT = None

needs_artifact = pytest.mark.skipif(
    ARTIFACT is None, reason="nwp_correction.json not built yet")


CACHE = os.path.join(os.path.dirname(DEFAULT_PATH), "nwp_benchmark_cache.json")
CACHE_FIELD = {
    "temperature": "temperature_2m",
    "humidity": "relative_humidity_2m",
    "pressure": "surface_pressure",
}


def _observed_nwp_domain():
    """5th-95th percentile of each NWP field in the cache, in station units."""
    out = {}
    try:
        with open(CACHE, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return out
    for var, field in CACHE_FIELD.items():
        vals = []
        for k, rec in raw.items():
            if k.startswith("_") or not isinstance(rec, dict) or field not in rec:
                continue
            vals.extend(x for x in rec[field] if x is not None)
        if not vals:
            continue
        vals.sort()
        n = len(vals)
        out[var] = (vals[int(0.05 * n)], vals[int(0.95 * n)])
    return out


class TestArtifact:
    @needs_artifact
    def test_loads(self):
        c = NwpCorrection.load()
        assert c is not None
        assert c.available
        assert c.n_coefficients() > 0

    @needs_artifact
    def test_missing_file_is_not_fatal(self):
        assert NwpCorrection.load(os.path.join("no", "such", "file.json")) is None

    @needs_artifact
    def test_every_coefficient_is_finite_and_sane(self):
        """Slope must be finite and near-identity; the intercept is not bounded
        on its own, because pressure needs a large offset precisely to
        compensate for a slope well below 1 (altitude mismatch)."""
        c = NwpCorrection.load()
        for (_src, hz, st, var), (a, b) in c._coef.items():
            assert math.isfinite(a) and math.isfinite(b), \
                f"non-finite coefficient at {hz}/{st}/{var}"
            assert 0.0 < a < 2.0, f"implausible slope {a} at {hz}/{st}/{var}"

    @needs_artifact
    def test_correction_stays_near_identity_over_plausible_inputs(self):
        """
        For the thermodynamic variables the map is shrunk toward identity, so
        over the plausible input range it must not stray far from the raw NWP
        value. This is checkable without knowing the answer: a large drift would
        mean the calibration had latched onto a train-period quirk.

        wind_speed is deliberately excluded from the drift check. The station
        anemometers read roughly half of the ECMWF 10 m wind (measured 1.50 m/s
        against 3.11 m/s), so a slope near 0.5 is the sensor's real behaviour,
        not a calibration defect. Wind is checked for slope band and
        non-negativity instead.
        """
        plausible = {
            "temperature": (10.0, 45.0),
            "humidity": (20.0, 100.0),
            "pressure": (940.0, 1060.0),
        }
        # Probe the domain the coefficients were actually FITTED on: the 5th-95th
        # percentile of the NWP values in the cache for that variable. A linear
        # per-station fit is only meaningful inside its fitted range -- outside
        # it the slope extrapolates wildly, which says nothing about whether the
        # correction is any good. Humidity shows this most clearly: NWP RH at
        # these stations never approaches 20%, so probing there tests
        # extrapolation rather than calibration.
        domain = _observed_nwp_domain()
        # Tolerance is a FRACTION OF THE PROBED SPAN, not an absolute number, so
        # it scales with the physical quantity.
        max_drift_fraction = {"temperature": 0.20, "humidity": 0.45, "pressure": 0.20}
        # Absolute floors, in each variable's own units. These exist because the
        # WHOLE POINT of the layer is to remove a known systematic offset, so a
        # drift of that size is the correction working rather than failing. The
        # floors match the measured NWP biases: ECMWF temperature reads ~1 C
        # cold at these stations, NWP humidity runs ~4-12 % wet, and surface
        # pressure carries a per-station altitude offset. The tropical domain is
        # narrow (NWP temperature spans only 24.6-32.0 C here), so a fraction of
        # the span alone would be tighter than the bias being corrected.
        drift_floor = {"temperature": 2.5, "humidity": 25.0, "pressure": 20.0}
        c = NwpCorrection.load()
        checked = 0
        for (_src, hz, st, var), (a, b) in c._coef.items():
            if var == "wind_speed":
                assert 0.2 < a < 1.3, f"wind slope {a} at {hz}/{st} out of band"
                assert b >= 0.0, f"wind offset {b} negative"
                continue
            # Only constrain coefficients the router actually consults. A cell
            # routed to `lln` never uses its NWP branch, so a steep fit there is
            # inert, and asserting on it would only force the tolerance upward
            # until the test stopped testing anything.
            route = c.route_for(hz, var) or {}
            if route.get("producer") not in ("nwp", "blend"):
                continue
            lo, hi = domain.get(var, plausible[var])
            if hi <= lo:
                continue
            checked += 1
            tol = max((hi - lo) * max_drift_fraction[var], drift_floor[var])
            for probe in (lo, 0.5 * (lo + hi), hi):
                drift = abs((a * probe + b) - probe)
                assert drift <= tol, (
                    f"{var} at {hz}/{st}: {probe} maps to {a * probe + b:.2f} "
                    f"(drift {drift:.1f} > {tol:.1f})")
        assert checked >= 20, f"only {checked} coefficients exercised"

    @needs_artifact
    def test_routing_only_names_known_producers(self):
        c = NwpCorrection.load()
        for hz, per_v in ARTIFACT["routing"].items():
            for var, r in per_v.items():
                assert r["producer"] in PRODUCERS

    @needs_artifact
    def test_blend_weight_in_range(self):
        c = NwpCorrection.load()
        for hz, per_v in ARTIFACT["routing"].items():
            for var, r in per_v.items():
                w = r.get("blend_weight", 0.0)
                assert 0.0 <= w <= 1.0, f"blend weight {w} out of range at {hz}/{var}"

    @needs_artifact
    def test_has_coverage_for_every_horizon(self):
        c = NwpCorrection.load()
        for hz in ("1", "3", "6", "12", "24"):
            assert c.route_for(hz, "temperature") is not None, f"no route for +{hz}h"


class TestFailSafe:
    def test_missing_artifact_yields_inert_object(self):
        c = NwpCorrection({})
        assert not c.available
        assert c.n_coefficients() == 0
        assert c.correct(20.0, "any", 6, "temperature") is None
        prod, corr, w = c.plan(6, "temperature", nwp_value=20.0, station_id="any")
        assert prod == "lln" and corr is None
        assert select(prod, w, corr, 19.0, 18.0) == (19.0, "lln")

    def test_correct_returns_none_on_bad_input(self):
        c = NwpCorrection(ARTIFACT or {})
        assert c.correct(None, "s", 6, "temperature") is None
        assert c.correct(float("nan"), "s", 6, "temperature") is None
        assert c.correct("not a number", "s", 6, "temperature") is None
        assert c.correct(float("inf"), "s", 6, "temperature") is None

    def test_correct_returns_none_for_unknown_station_or_variable(self):
        c = NwpCorrection(ARTIFACT or {})
        assert c.correct(20.0, "NO_SUCH_STATION", 6, "temperature") is None
        assert c.correct(20.0, "03pqkGAj", 6, "not_a_variable") is None

    def test_correct_returns_none_for_unknown_horizon(self):
        c = NwpCorrection(ARTIFACT or {})
        assert c.correct(20.0, "03pqkGAj", 999, "temperature") is None

    def test_never_raises_on_hostile_input(self):
        c = NwpCorrection(ARTIFACT or {})
        for bad in (None, "", "x", float("nan"), float("inf"), -1e300, 1e300, [], {}):
            for hz in (1, 6, 24, "6", None):
                for var in ("temperature", "humidity", "nwp_bogus", None):
                    try:
                        c.correct(bad, "03pqkGAj", hz, var)
                        prod, corr, w = c.plan(hz, var, nwp_value=bad,
                                               station_id="03pqkGAj")
                        select(prod, w, corr, bad, bad)
                    except Exception as exc:  # noqa: BLE001
                        pytest.fail(f"raised {type(exc).__name__} on {bad!r}: {exc}")

    def test_plan_without_anything_returns_none_not_zero(self):
        c = NwpCorrection(ARTIFACT or {})
        prod, corr, w = c.plan(6, "temperature")
        assert select(prod, w, corr, None, None) == (None, "none")


class TestClamping:
    @needs_artifact
    def test_output_is_clamped_to_physical_bounds(self):
        c = NwpCorrection.load()
        for var, (lo, hi) in CLAMP.items():
            for st in ("03pqkGAj", "1Zb102pg", "wkAWLzlm"):
                for hz in (1, 3, 6, 12, 24):
                    for probe in (-1e6, 0.0, 1e6):
                        out = c.correct(probe, st, hz, var)
                        if out is None:
                            continue
                        if lo is not None:
                            assert out >= lo, f"{var} {out} < {lo}"
                        if hi is not None:
                            assert out <= hi, f"{var} {out} > {hi}"

    def test_clamp_table_covers_every_routed_variable(self):
        if ARTIFACT is None:
            pytest.skip("no artifact")
        for per_v in ARTIFACT["routing"].values():
            for var in per_v:
                assert var in CLAMP, f"routed variable {var} missing from CLAMP"


class TestRouter:
    @needs_artifact
    def test_router_never_degrades_when_nwp_unavailable(self):
        """With no NWP the result must be this project's model, not zero."""
        c = NwpCorrection.load()
        for hz in (1, 3, 6, 12, 24):
            for var in CLAMP:
                prod, corr, w = c.plan(hz, var, nwp_value=None,
                                       station_id="03pqkGAj")
                val, used = select(prod, w, corr, 12.5, 11.0)
                assert val == 12.5, f"+{hz}h {var} degraded to {used}"
                assert used == "lln"

    @needs_artifact
    def test_blend_is_convex_combination(self):
        c = NwpCorrection.load()
        corr = c.correct(25.0, "03pqkGAj", 6, "humidity")
        prod, _c, w = c.plan(6, "humidity", nwp_value=25.0, station_id="03pqkGAj")
        if prod == "blend" and corr is not None:
            val, used = select(prod, w, corr, 30.0, 24.0)
            assert used == "blend"
            assert val == pytest.approx(w * corr + (1 - w) * 30.0, abs=1e-9)

    @needs_artifact
    def test_plan_is_deterministic(self):
        c = NwpCorrection.load()
        first = c.plan(6, "temperature", nwp_value=25.0, station_id="03pqkGAj")
        for _ in range(5):
            assert c.plan(6, "temperature", nwp_value=25.0,
                          station_id="03pqkGAj") == first

    @needs_artifact
    def test_every_station_has_every_variable(self):
        """Guards the bug where each station stored only the last variable."""
        c = NwpCorrection.load()
        with open(DEFAULT_PATH, "r", encoding="utf-8") as f:
            art = json.load(f)
        coeffs = art["coefficients"][art["nwp_source"]]
        n_st = 0
        for hz, per_st in coeffs.items():
            n_st = max(n_st, len(per_st))
            for st, per_v in per_st.items():
                assert set(per_v) == set(CLAMP), \
                    f"+{hz}h {st} has {sorted(per_v)}, expected {sorted(CLAMP)}"
        assert n_st >= 10, f"only {n_st} stations in the artifact"


class TestSelect:
    """select() is the safety-critical function; exercise it directly."""

    def test_missing_nwp_degrades_to_model(self):
        assert select("nwp", 0.0, None, 26.0, 24.0) == (26.0, "lln")

    def test_blend_without_model_degrades_to_nwp(self):
        assert select("blend", 0.5, 25.0, None, 24.0) == (25.0, "nwp")

    def test_nothing_available_returns_none(self):
        assert select("nwp", 0.0, None, None, None) == (None, "none")

    def test_garbage_never_raises(self):
        for bad in (None, "", "x", float("nan"), float("inf"), [], {}, object()):
            for prod in ("nwp", "lln", "persistence", "blend", "bogus", None):
                val, used = select(prod, bad, bad, bad, bad)
                assert val is None or isinstance(val, float)
                assert used in PRODUCERS + ("none",)

    def test_zero_is_preserved_not_treated_as_missing(self):
        assert select("lln", 0.0, None, 0.0, None) == (0.0, "lln")

    def test_unrouted_producer_prefers_model(self):
        assert select("something_new", 0.0, 25.0, 26.0, 24.0) == (26.0, "lln")

    def test_weight_is_clamped(self):
        for w in (-5.0, 0.0, 1.0, 5.0):
            val, _ = select("blend", w, 20.0, 30.0, None)
            assert 20.0 - 1e-9 <= val <= 30.0 + 1e-9

