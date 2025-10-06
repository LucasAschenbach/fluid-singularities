import os
import sys
import unittest

import torch


# Make `src` importable
THIS_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from pde.boussinesq2d_selfsimilar import Boussinesq2DSelfSimilarPDE  # noqa: E402
from pde.base import PDEBoundaryData  # noqa: E402


def make_inputs(n: int = 8, device: torch.device | None = None) -> torch.Tensor:
    y = torch.randn(n, 2, device=device)
    y.requires_grad_(True)
    return y


class ZeroModel(torch.nn.Module):
    """Returns zeros for (hOmega, hTheta, hPsi)."""

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        n = feats.shape[0]
        return feats.new_zeros(n, 3)


class FeatureLinearModel(torch.nn.Module):
    """Simple linear map on (q, beta) features."""

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        q = feats[:, 0:1]
        beta = feats[:, 1:2]
        hOmega = q
        hTheta = beta
        hPsi = q - beta
        return torch.cat([hOmega, hTheta, hPsi], dim=1)


class TestBoussinesq2DSelfSimilar(unittest.TestCase):
    def test_zero_model_pde_residual_is_zero(self):
        pde = Boussinesq2DSelfSimilarPDE(lambda_value=0.5)
        model = ZeroModel()
        inputs = make_inputs(n=6)

        loss = pde.residuals(model, inputs)

        self.assertEqual(loss.pde.shape, (inputs.shape[0], 3))
        self.assertTrue(torch.allclose(loss.pde, torch.zeros_like(loss.pde), atol=1e-7))
        self.assertIsNone(loss.bc)

    def test_boundary_residual_uses_provided_points(self):
        def target_fn(points: torch.Tensor) -> torch.Tensor:
            target = torch.zeros(points.shape[0], 3, dtype=points.dtype, device=points.device)
            target[:, 2:3] = points[:, 0:1]  # tie third output to y1 boundary coordinate
            return target

        bc = PDEBoundaryData(target_fn=target_fn)
        pde = Boussinesq2DSelfSimilarPDE(lambda_value=0.0, bc=bc)
        model = FeatureLinearModel()

        inputs = make_inputs(n=4)
        boundary_points = torch.randn_like(inputs)

        loss = pde.residuals(model, inputs, bc_inputs=boundary_points)
        self.assertIsNotNone(loss.bc)

        expected_bc = model(boundary_points) - target_fn(boundary_points)

        self.assertTrue(torch.allclose(loss.bc, expected_bc, atol=1e-7))

    def test_infer_lambda_returns_baseline_for_zero_velocity_gradient(self):
        pde = Boussinesq2DSelfSimilarPDE(lambda_value=0.0)
        model = ZeroModel()
        lam_est = pde.infer_lambda(model)
        self.assertAlmostEqual(lam_est, -3.0, places=6)


if __name__ == "__main__":
    unittest.main()
