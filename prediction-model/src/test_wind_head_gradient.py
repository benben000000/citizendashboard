"""
Regression tests for the wind head's dead-gradient trap.

WHAT WENT WRONG
---------------
The wind speed output was floored with `F.relu`:

    pred_ws = F.relu(origin_ws + d_ws)

`ws_head`'s final layer is zero-initialised, so the model starts at a persistence
prior (d_ws = 0). Wind decays, so the head learns a NEGATIVE residual -- measured
at -5.75 m/s mean on the +6h candidate. relu then clamps every window whose
origin+delta went negative to a hard 0.0, and smooth_l1 contributes EXACTLY zero
gradient for those windows. They can never be pushed back up.

The result was a self-reinforcing collapse: on the +6h candidate, 2208 of 3132
test windows (70%) returned exactly 0.0 wind. The head was not inert -- it had
trained hard, and the output gate was eating the result.

These tests pin the fix: the floor must keep the constraint while leaving a
gradient path on the negative side.
"""

import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import (  # noqa: E402
    GarciaWeatherLNN,
    GarciaWeatherLNNFeatured,
    _WIND_FLOOR_SLOPE,
    _nonnegative,
)


class TestNonNegativeFloor:
    """The floor itself must be gradient-preserving."""

    def test_gradient_survives_on_the_negative_side(self):
        x = torch.tensor([-22.0, -1.0, -0.001], requires_grad=True)
        _nonnegative(x).sum().backward()
        assert (x.grad != 0).all(), (
            "wind floor is killing the gradient on negative inputs; the head "
            "cannot recover once it overshoots"
        )

    def test_slope_is_small_so_the_constraint_still_holds(self):
        # A large slope would quietly remove the non-negativity constraint.
        assert 0.0 < _WIND_FLOOR_SLOPE <= 0.05

    def test_positive_values_pass_through_unchanged(self):
        x = torch.tensor([0.0, 0.5, 3.2, 250.0])
        assert torch.allclose(_nonnegative(x), x)

    def test_zero_gradient_case_is_absent_versus_relu(self):
        """Plain relu is the trap; show the difference explicitly."""
        relu_in = torch.tensor([-5.0], requires_grad=True)
        F.relu(relu_in).sum().backward()
        assert relu_in.grad.item() == 0.0, "relu is expected to be dead here"

        leaky_in = torch.tensor([-5.0], requires_grad=True)
        _nonnegative(leaky_in).sum().backward()
        assert leaky_in.grad.item() != 0.0


class TestWindHeadReceivesGradient:
    """A collapsed head must still be trainable."""

    def _model(self):
        torch.manual_seed(0)
        return GarciaWeatherLNNFeatured(
            input_dim=8, context_dim=75, hidden_dim=32,
            use_two_stage_precipitation=True,
        )

    def test_wind_head_grad_is_finite_and_nonzero_at_random_init(self):
        m = self._model()
        m.eval()
        B, S = 16, 6
        tele = torch.randn(B, S, 8) * 0.1 + torch.tensor(
            [28.0, 32.0, 70.0, 1008.0, 2.0, 0.3, 0.9, 0.0])
        ctx = torch.randn(B, 75) * 0.1  # context is [batch, 75], not a sequence
        dt = torch.ones(B, S, 1)
        # Origin wind deliberately 0 so the residual alone decides the output.
        origin = torch.tensor([[28.0, 70.0, 1008.0, 0.0, 0.0, 0.0]]).repeat(B, 1)
        target = torch.full((B, 1), 4.0)

        m.zero_grad()
        out = m(tele, ctx, dt, origin_weather=origin)
        F.smooth_l1_loss(out["wind_speed"], target).backward()

        g = m.ws_head[-1].weight.grad
        assert g is not None, "no gradient reached the wind head at all"
        assert torch.isfinite(g).all(), "wind head gradient is not finite"
        assert g.abs().sum() > 0, "wind head gradient is exactly zero"

    def test_grad_reaches_wind_head_even_when_output_is_far_negative(self):
        """The exact failure mode: pre-activation deeply negative.

        Under the old relu this produced zero gradient and the head was stuck.
        """
        m = self._model()
        m.eval()
        # Drive the head hard negative.
        with torch.no_grad():
            m.ws_head[-1].bias.fill_(-30.0)

        B, S = 8, 6
        tele = torch.randn(B, S, 8) * 0.1
        ctx = torch.randn(B, 75) * 0.1  # context is [batch, 75], not a sequence
        dt = torch.ones(B, S, 1)
        origin = torch.tensor([[28.0, 70.0, 1008.0, 0.0, 0.0, 0.0]]).repeat(B, 1)
        target = torch.full((B, 1), 5.0)

        m.zero_grad()
        out = m(tele, ctx, dt, origin_weather=origin)
        F.smooth_l1_loss(out["wind_speed"], target).backward()

        g = m.ws_head[-1].weight.grad
        assert g is not None
        assert g.abs().sum() > 0, (
            "wind head received zero gradient while its output was deeply "
            "negative -- this is the collapse that produced 70% exact-zero wind"
        )
        # A recovery signal must point the head back UP toward the target.
        assert g.sum() > 0, (
            "gradient must push the head back up when the prediction is below "
            "target; a non-positive direction would keep it collapsed"
        )


class TestNonNegativityStillEnforced:
    """The fix must not quietly remove the physical constraint."""

    def test_output_never_materially_negative(self):
        m = GarciaWeatherLNNFeatured(
            input_dim=8, context_dim=75, hidden_dim=32,
            use_two_stage_precipitation=True)
        m.eval()
        with torch.no_grad():
            m.ws_head[-1].bias.fill_(-40.0)
        B, S = 8, 6
        tele = torch.randn(B, S, 8) * 0.1
        ctx = torch.randn(B, 75) * 0.1  # context is [batch, 75], not a sequence
        dt = torch.ones(B, S, 1)
        origin = torch.tensor([[28.0, 70.0, 1008.0, 0.0, 0.0, 0.0]]).repeat(B, 1)
        out = m(tele, ctx, dt, origin_weather=origin)
        assert (out["wind_speed"] >= -_WIND_FLOOR_SLOPE * 50).all(), (
            "wind speed is materially negative; the constraint is not holding"
        )


class TestBaseClassPath:
    """GarciaWeatherLNN (non-featured) has the same output gate."""

    def test_no_bare_relu_on_wind_outputs(self):
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "model.py"), encoding="utf-8") as f:
            src = f.read()
        for bad in ("pred_ws = F.relu(",
                    "pred_ws = torch.clamp(F.relu("):
            assert bad not in src, (
                f"found `{bad}` in model.py; the wind floor must go through "
                "_nonnegative() to keep the gradient alive"
            )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
