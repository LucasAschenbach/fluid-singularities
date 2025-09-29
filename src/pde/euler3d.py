from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from .base import ResidualLoss, PDE, BoundaryData

import torch


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
        super().__init__(input_dim=4, output_dim=4, device=device)
        self.rho = rho

    def residuals(
        self,
        model: torch.nn.Module,
        inputs: torch.Tensor,
        bc: Optional[BoundaryData] = None,
        data: Optional[torch.Tensor] = None,
    ) -> Euler3DLoss:
        """Compute PDE residuals and masked BC residuals.

        Args:
            model: network mapping (t,x,y,z)->(u,v,w,p)
            inputs: [N,4] requires_grad=True
            bc: optional BoundaryData to enforce Dirichlet conditions with an
                optional per-output mask.
            data: DEPRECATED. If provided (and `bc` is None), treated as a
                  fully-specified Dirichlet target with mask=1.
        Returns:
            Euler3DLoss containing PDE residuals and optional BC residual.
        """
        txyz = inputs
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

        pde_residual = torch.cat([mom_res, div_u], dim=1)  # [N,4]

        bc_residual = None
        if bc is not None:
            bc_residual = bc.residual(out)

        return Euler3DLoss(pde=pde_residual, bc=bc_residual)
