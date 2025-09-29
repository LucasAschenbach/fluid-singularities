import os
import sys

import torch
import unittest
import math
from typing import Tuple


# Make `src` importable
THIS_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from pde.euler3d import Euler3DPDE  # noqa: E402
from pde.base import BoundaryData  # noqa: E402


class ModelZero(torch.nn.Module):
    """Returns u=v=w=0, p=const."""

    def __init__(self, p_const: float = 5.0):
        super().__init__()
        self.p_const = p_const

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [N,4] -> [N,4]
        N = x.shape[0]
        u = torch.zeros(N, 3, dtype=x.dtype, device=x.device)
        p = torch.full((N, 1), self.p_const, dtype=x.dtype, device=x.device)
        return torch.cat([u, p], dim=1)


class ModelConst(torch.nn.Module):
    """Returns a constant vector [c1, c2, c3, c4]."""

    def __init__(self, vals=(1.0, -2.0, 3.0, 4.0)):
        super().__init__()
        self.register_buffer("vals", torch.tensor(vals, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = x.shape[0]
        return self.vals.to(x).unsqueeze(0).repeat(N, 1)


class ModelUx(torch.nn.Module):
    """Returns u=(x,0,0), p=0 to test gradients and convection."""

    def forward(self, txyz: torch.Tensor) -> torch.Tensor:
        x = txyz[:, 1:2]
        u = torch.cat([x, torch.zeros_like(x), torch.zeros_like(x)], dim=1)
        p = torch.zeros_like(x)
        return torch.cat([u, p], dim=1)
    

class ModelABC(torch.nn.Module):
    """Returns the ABC flow solution for testing."""

    def __init__(self, A=1.0, B=1.0, C=1.0):
        super().__init__()
        self.A = A
        self.B = B
        self.C = C

    @staticmethod
    def sample_points(
        n: int,
        t_range: Tuple[float, float] = (0.0, 0.0),
        x_range: Tuple[float, float] = (0.0, 2 * math.pi),
        y_range: Tuple[float, float] = (0.0, 2 * math.pi),
        z_range: Tuple[float, float] = (0.0, 2 * math.pi),
        device: torch.device | None = None,
    ) -> torch.Tensor:
        def uni(lo: float, hi: float) -> torch.Tensor:
            return lo + (hi - lo) * torch.rand(n, 1, device=device)

        t = uni(*t_range)
        x = uni(*x_range)
        y = uni(*y_range)
        z = uni(*z_range)
        return torch.cat([t, x, y, z], dim=1)


    def forward(self, txyz: torch.Tensor) -> torch.Tensor:
        x = txyz[:, 1]
        y = txyz[:, 2]
        z = txyz[:, 3]

        u = self.A * torch.sin(z) + self.C * torch.cos(y)
        v = self.B * torch.sin(x) + self.A * torch.cos(z)
        w = self.C * torch.sin(y) + self.B * torch.cos(x)

        vel2 = u * u + v * v + w * w
        p = -0.5 * vel2

        return torch.stack([u, v, w, p], dim=1)


def make_inputs(N=16, device=None):
    t = torch.zeros(N, 1, device=device, requires_grad=True)
    # sample x,y,z for variety
    xyz = torch.randn(N, 3, device=device)
    xyz.requires_grad_(True)
    return torch.cat([t, xyz], dim=1)


class TestEuler3DResiduals(unittest.TestCase):
    def test_zero_solution_pde_residual_is_zero(self):
        pde = Euler3DPDE()
        model = ModelZero(p_const=2.5)
        inputs = make_inputs(N=8)
        loss = pde.residuals(model, inputs)
        self.assertEqual(loss.pde.shape, (inputs.shape[0], 4))
        self.assertTrue(torch.allclose(loss.pde, torch.zeros_like(loss.pde), atol=1e-7))


    def test_dirichlet_bc_masking_on_velocity_only(self):
        model = ModelConst(vals=(1.0, -2.0, 3.0, 4.0))
        inputs = make_inputs(N=5)

        # Target zeros for all outputs, but enforce only on velocity components
        target = torch.zeros(inputs.shape[0], 4, dtype=inputs.dtype)
        mask = torch.zeros_like(target)
        mask[:, 0:3] = 1.0  # enforce u,v,w only
        bc = BoundaryData(target=target, mask=mask)
        pde = Euler3DPDE(bc=bc)

        loss = pde.residuals(model, inputs)
        expected_bc = torch.tensor([1.0, -2.0, 3.0, 0.0], dtype=inputs.dtype).repeat(inputs.shape[0], 1)
        self.assertIsNotNone(loss.bc)
        self.assertTrue(torch.allclose(loss.bc, expected_bc, atol=1e-7))

        # Only BC contributes to total since PDE residual is zero for constant fields
        expected_total = torch.mean(expected_bc**2)
        self.assertTrue(torch.allclose(loss.total(w_pde=1.0, w_bc=1.0), expected_total, atol=1e-7))


    def test_gradient_and_convection_terms_simple_field(self):
        pde = Euler3DPDE()
        model = ModelUx()
        inputs = make_inputs(N=6)

        loss = pde.residuals(model, inputs)
        # For u=(x,0,0), p=0:
        # div(u) = du1/dx = 1
        # conv for i=0: u · grad(u0) = x * 1 = x, others 0
        # du/dt = 0, grad p = 0
        x = inputs[:, 1:2].detach()
        expected = torch.cat([x, torch.zeros_like(x), torch.zeros_like(x), torch.ones_like(x)], dim=1)

        self.assertTrue(torch.allclose(loss.pde, expected, atol=1e-6))
    

    def test_abc_solution_pde_residual_is_zero(self):
        pde = Euler3DPDE()
        model = ModelABC(A=1.0, B=1.0, C=1.0)
        inputs = ModelABC.sample_points(10, device=pde.device)
        inputs.requires_grad_(True)

        loss = pde.residuals(model, inputs)
        self.assertTrue(torch.allclose(loss.pde, torch.zeros_like(loss.pde), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
