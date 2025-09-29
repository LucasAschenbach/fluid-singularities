from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
from _base import ResidualLoss, PDE

import torch


@dataclass
class Euler3DLoss(ResidualLoss):
    """Loss for 3D Euler equations."""
    pass

class Euler3DPDE(PDE):
    """3D incompressible Euler PDE residuals using torch autograd.

    Equations (rho=1):
        - Mass: div(u) = 0
        - Momentum: du/dt + (u · ∇)u + ∇p = 0

    Network maps (t, x, y, z) -> (u, v, w, p).
    """

    def __init__(self, rho: float = 1.0, device: Optional[torch.device] = None) -> None:
        super().__init__(input_dim=4, output_dim=4)
        self.rho = rho
        self.device = device

    @staticmethod
    def _grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)[0]

    def residuals(
        self,
        model: torch.nn.Module,
        txyz: torch.Tensor,
        data_target: Optional[torch.Tensor] = None,
    ) -> Euler3DLoss:
        """Compute PDE residuals at collocation points.

        Args:
            model: network mapping (t,x,y,z)->(u,v,w,p)
            txyz: [N,4] requires_grad=True
            data_target: optional supervised target [N,4]
        Returns:
            EulerLoss with pde, div, and optional data loss (MSE)
        """
        assert txyz.requires_grad, "Input points must require grad for autograd derivatives."

        out = model(txyz)
        u = out[:, :3]  # [N,3]
        p = out[:, 3:4]  # [N,1]

        # unpack inputs
        t = txyz[:, 0:1]
        x = txyz[:, 1:2]
        y = txyz[:, 2:3]
        z = txyz[:, 3:4]

        # time derivative
        du_dt = torch.zeros_like(u)
        for i in range(3):
            du_dt[:, i:i+1] = self._grad(u[:, i:i+1], t)

        # spatial gradients
        grads = []  # each: [N,3] gradient wrt (x,y,z)
        for i in range(3):
            ui = u[:, i:i+1]
            du_dx = self._grad(ui, x)
            du_dy = self._grad(ui, y)
            du_dz = self._grad(ui, z)
            grads.append(torch.cat([du_dx, du_dy, du_dz], dim=1))
        # grads[i][..., j] = d u_i / d [x,y,z]_j

        # divergence
        div_u = grads[0][:, 0:1] + grads[1][:, 1:2] + grads[2][:, 2:2+1]

        # convective term (u · ∇)u
        conv = torch.zeros_like(u)
        for i in range(3):
            # u_x * du_i/dx + u_y * du_i/dy + u_z * du_i/dz
            conv[:, i:i+1] = (
                u[:, 0:1] * grads[i][:, 0:1]
                + u[:, 1:2] * grads[i][:, 1:2]
                + u[:, 2:3] * grads[i][:, 2:3]
            )

        # pressure gradient
        dp_dx = self._grad(p, x)
        dp_dy = self._grad(p, y)
        dp_dz = self._grad(p, z)
        grad_p = torch.cat([dp_dx, dp_dy, dp_dz], dim=1)

        # momentum residual
        mom_res = du_dt + conv + grad_p

        # MSE losses
        pde_loss = torch.cat([mom_res, div_u], dim=1)  # [N,4]

        data_loss = None
        if data_target is not None:
            data_loss = out - data_target

        return Euler3DLoss(pde=pde_loss, data=data_loss)
