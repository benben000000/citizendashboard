"""
Contract tests for the gated NWP hybrid integration.

The gate's job is to be inert unless everything is in order. These tests
concentrate on that, because a router that silently degrades a good forecast is
worse than no router at all. Accuracy is verified separately by
verify_nwp_correction.py against the held-out test split.
"""

import json
import os
from datetime import datetime, timedelta, timezone

import pytest

from nwp_correction import NwpCorrection, artifact_path_for
from nwp_integration import (
    MODEL_KEYS, SOURCE_MODELS, VARIABLE_FIELDS, HybridRouter,
)

SOURCE = "noaa_gfs_v1"
ARTIFACT = artifact_path_for(SOURCE_MODELS[SOURCE])
HAVE_ARTIFACT = os.path.exists(ARTIFACT)
needs_artifact = pytest.mark.skipif(not HAVE_ARTIFACT,
                                    reason="GFS correction artifact not built")

NOW = datetime(2026, 9, 29, 12, 0, tzinfo=timezone.utc)
FRESH = (NOW - timedelta(hours=1)).isoformat()


def model_output(temp=26.0, hum=70.0, pres=1005.0, wind=3.0):
    return {
        "temperature_c": temp,
        "relative_humidity_pct": hum,
        "pressure_hpa": pres,
        "wind_speed_kmh": wind,
        "chance_of_rain_pct": 30.0,
    }


def nwp_values(temp=27.0, hum=72.0, pres=1004.0, wind=3.2):
    return {"temperature": temp, "humidity": hum, "pressure": pres, "wind_speed": wind}


class TestDefaultIsInert:
    def test_disabled_by_default(self):
        r = HybridRouter(source_id=SOURCE, enabled=False)
        assert not r.active
        assert r.blocked_reason

    def test_disabled_router_returns_model_output_unchanged(self):
        r = HybridRouter(source_id=SOURCE, enabled=False)
        base = model_output()
        out, prov = r.apply("03pqkGAj", 6, base, nwp_values(), FRESH, NOW)
        assert out == base
        assert prov["applied"] is False
        assert prov["reasons"]

    def test_does_not_mutate_caller_dict(self):
        r = HybridRouter(source_id=SOURCE, enabled=False)
        base = model_output()
        snapshot = dict(base)
        out, _ = r.apply("03pqkGAj", 6, base, nwp_values(), FRESH, NOW)
        assert base == snapshot, "the caller's dict was mutated in place"
        assert out is not base


class TestLicenceGate:
    def test_unknown_source_is_refused(self):
        r = HybridRouter(source_id="pagasa_nwp_v1", enabled=True)
        assert not r.active
        assert "not production-eligible" in (r.blocked_reason or "")

    def test_registered_source_without_coefficients_is_refused(self):
        """
        A licence that clears the registry but has no fitted coefficients must
        still be refused. Otherwise the layer would load the wrong artifact or
        fall back to raw NWP, which is uncalibrated.
        """
        class _StubRegistry:
            def is_production_eligible(self, _sid):
                return True, "approved"

        r = HybridRouter(source_id="licensed_but_unfitted", enabled=True,
                         registry=_StubRegistry())
        assert not r.active
        assert "no correction coefficients" in (r.blocked_reason or "")

    def test_unregistered_source_is_refused_before_anything_else(self):
        r = HybridRouter(source_id="definitely_not_registered", enabled=True)
        assert not r.active
        assert "not production-eligible" in (r.blocked_reason or "")

    def test_every_mapped_source_has_an_artifact(self):
        for sid, model_key in SOURCE_MODELS.items():
            assert os.path.exists(artifact_path_for(model_key)), \
                f"{sid} maps to {model_key} but its artifact is missing"

    def test_registered_source_passes_when_enabled(self):
        r = HybridRouter(source_id=SOURCE, enabled=True)
        assert r.active, r.blocked_reason

    def test_missing_artifact_blocks(self, tmp_path):
        r = HybridRouter(source_id=SOURCE, enabled=True,
                         artifact_path=str(tmp_path / "nope.json"))
        assert not r.active
        assert "artifact" in (r.blocked_reason or "")

    def test_artifact_source_mismatch_blocks(self, tmp_path):
        with open(ARTIFACT, "r", encoding="utf-8") as f:
            art = json.load(f)
        art["nwp_source"] = "some_other_model"
        bad = tmp_path / "mismatch.json"
        with open(bad, "w", encoding="utf-8") as f:
            json.dump(art, f)
        r = HybridRouter(source_id=SOURCE, enabled=True, artifact_path=str(bad))
        assert not r.active
        assert "expected" in (r.blocked_reason or "")

    @needs_artifact
    def test_status_reports_the_gate(self):
        r = HybridRouter(source_id=SOURCE, enabled=True)
        s = r.status()
        assert s["active"] is True
        assert s["source_id"] == SOURCE
        assert s["n_coefficients"] > 0


class TestStaleness:
    @needs_artifact
    def test_stale_nwp_is_ignored(self):
        r = HybridRouter(source_id=SOURCE, enabled=True)
        old = (NOW - timedelta(hours=48)).isoformat()
        base = model_output()
        out, prov = r.apply("03pqkGAj", 6, base, nwp_values(), old, NOW)
        assert out == base
        assert any("stale" in x for x in prov["reasons"])

    @needs_artifact
    def test_missing_timestamp_is_not_trusted(self):
        r = HybridRouter(source_id=SOURCE, enabled=True)
        base = model_output()
        out, prov = r.apply("03pqkGAj", 6, base, nwp_values(), None, NOW)
        assert out == base
        assert any("stale" in x for x in prov["reasons"])

    @needs_artifact
    def test_unparseable_timestamp_is_not_trusted(self):
        r = HybridRouter(source_id=SOURCE, enabled=True)
        out, prov = r.apply("03pqkGAj", 6, model_output(), nwp_values(),
                            "not-a-time", NOW)
        assert any("stale" in x for x in prov["reasons"])

    @needs_artifact
    def test_fresh_nwp_is_accepted(self):
        r = HybridRouter(source_id=SOURCE, enabled=True)
        _, prov = r.apply("03pqkGAj", 6, model_output(), nwp_values(), FRESH, NOW)
        assert prov["applied"] is True

    @needs_artifact
    def test_absurd_future_timestamp_rejected(self):
        r = HybridRouter(source_id=SOURCE, enabled=True)
        far = (NOW + timedelta(days=30)).isoformat()
        out, prov = r.apply("03pqkGAj", 6, model_output(), nwp_values(), far, NOW)
        assert any("stale" in x for x in prov["reasons"])


class TestBehaviourWhenActive:
    @needs_artifact
    def test_lln_routed_cells_are_left_alone(self):
        r = HybridRouter(source_id=SOURCE, enabled=True)
        base = model_output()
        out, prov = r.apply("03pqkGAj", 1, base, nwp_values(), FRESH, NOW)
        assert out == base, "1h is routed to the LNN and must be untouched"
        assert prov["producers"]["temperature"]["producer"] == "lln"

    @needs_artifact
    def test_nwp_routed_cells_are_replaced(self):
        r = HybridRouter(source_id=SOURCE, enabled=True)
        out, prov = r.apply("03pqkGAj", 6, model_output(), nwp_values(), FRESH, NOW)
        p = prov["producers"]["temperature"]
        assert p["producer"] in ("nwp", "blend")
        assert out["temperature_c"] == pytest.approx(p["published"])
        assert out["temperature_c"] != model_output()["temperature_c"]

    @needs_artifact
    def test_values_stay_inside_physical_bounds(self):
        r = HybridRouter(source_id=SOURCE, enabled=True)
        for probe in (-500.0, 0.0, 1e6):
            out, _ = r.apply("03pqkGAj", 6, model_output(),
                             nwp_values(temp=probe, hum=probe, pres=probe,
                                        wind=probe), FRESH, NOW)
            assert -20.0 <= out["temperature_c"] <= 60.0
            assert 0.0 <= out["relative_humidity_pct"] <= 100.0
            assert 850.0 <= out["pressure_hpa"] <= 1100.0
            assert out["wind_speed_kmh"] >= 0.0

    @needs_artifact
    def test_untouched_keys_survive(self):
        r = HybridRouter(source_id=SOURCE, enabled=True)
        out, _ = r.apply("03pqkGAj", 6, model_output(), nwp_values(), FRESH, NOW)
        assert out["chance_of_rain_pct"] == 30.0
        assert set(out) == set(model_output())

    @needs_artifact
    def test_unknown_station_falls_back(self):
        r = HybridRouter(source_id=SOURCE, enabled=True)
        base = model_output()
        out, _ = r.apply("NO_SUCH_STATION", 6, base, nwp_values(), FRESH, NOW)
        assert out == base

    @needs_artifact
    def test_missing_nwp_field_leaves_that_variable_alone(self):
        r = HybridRouter(source_id=SOURCE, enabled=True)
        partial = {"temperature": 27.0}
        out, prov = r.apply("03pqkGAj", 6, model_output(), partial, FRESH, NOW)
        assert out["pressure_hpa"] == model_output()["pressure_hpa"]
        assert prov["applied"] is True

    @needs_artifact
    def test_never_raises_on_hostile_input(self):
        r = HybridRouter(source_id=SOURCE, enabled=True)
        for bad in (None, "", "x", float("nan"), [], {}, 0):
            for hz in (1, 6, 24, None, "6"):
                try:
                    r.apply("03pqkGAj", hz, model_output(), bad, bad, NOW)
                except Exception as exc:  # noqa: BLE001
                    pytest.fail(f"raised {type(exc).__name__} on {bad!r}: {exc}")

    @needs_artifact
    def test_batch_never_raises_and_preserves_order(self):
        r = HybridRouter(source_id=SOURCE, enabled=True)
        forecasts = [dict(model_output(), horizon_hours=h) for h in (1, 3, 6, 12, 24)]
        res, prov = r.batch_apply(
            "03pqkGAj", forecasts,
            nwp_by_horizon={h: nwp_values() for h in (1, 3, 6, 12, 24)},
            nwp_timestamps={h: FRESH for h in (1, 3, 6, 12, 24)}, now=NOW)
        assert len(res) == len(forecasts) == len(prov)
        assert [f.get("horizon_hours") for f in res] == [1, 3, 6, 12, 24]

    @needs_artifact
    def test_batch_with_no_nwp_is_a_no_op(self):
        r = HybridRouter(source_id=SOURCE, enabled=True)
        forecasts = [dict(model_output(), horizon_hours=h) for h in (1, 6, 24)]
        res, prov = r.batch_apply("03pqkGAj", forecasts, now=NOW)
        assert res == forecasts
        assert all(p["applied"] is False for p in prov)


class TestVariableMapping:
    def test_every_mapped_variable_has_both_keys(self):
        assert set(VARIABLE_FIELDS) == set(MODEL_KEYS)

    @needs_artifact
    def test_model_keys_exist_in_real_output(self):
        """The mapping must match what the predictor actually emits."""
        import numpy as np
        from dataset import TelemetryDataPipeline
        from inference import LNNServerlessPredictor
        from dataset import build_forecast_windows, DEFAULT_SEQ_LEN
        pipe = TelemetryDataPipeline()
        res = build_forecast_windows(pipeline=pipe, split="test", horizon=6,
                                     seq_len=DEFAULT_SEQ_LEN, return_metadata=True)
        X = res[0].detach().cpu().numpy().astype(np.float64)
        dt = res[1].detach().cpu().numpy().astype(np.float64)
        Xr = np.clip(X[0] * pipe.norm_stds + pipe.norm_means,
                     [10, 10, 10, 900, 0, -1, -1, 0],
                     [50, 70, 100, 1050, 180, 1, 1, 150])
        pred = LNNServerlessPredictor(
            bundle_dir=os.path.join(os.path.dirname(ARTIFACT), "bundles", "h6"))
        out = pred.predict_from_observed_sequence(
            telemetry_sequence=Xr, dt_sequence=dt[0], horizon_hours=6)
        for key in MODEL_KEYS.values():
            assert key in out, f"predictor does not emit {key}"
