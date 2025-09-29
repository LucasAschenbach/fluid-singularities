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
    ) -> Euler3DLoss:
        """Compute PDE residuals and masked BC residuals.

        Args:
            model: network mapping (t,x,y,z)->(u,v,w,p)
            inputs: [N,4] requires_grad=True
            bc: optional BoundaryData to enforce Dirichlet conditions with an
                optional per-output mask.
        Returns:
            Euler3DLoss containing PDE residuals and optional BC residual.
        """
        txyz = inputs
        assert txyz.requires_grad, "Input points must require grad for autograd derivatives."

        out = model(txyz)
        u = out[:, :3]  # [N,3]
        p = out[:, 3:4]  # [N,1]

        # Gradient tensor for u wrt (t,x,y,z): J_u[n, i, k] = d u_i / d[t,x,y,z]_k
        J_u = torch.stack([self._grad(u[:, i:i+1], txyz) for i in range(3)], dim=1)  # [N,3,4]

        # du/dt is column k=0
        du_dt = J_u[:, :, 0]

        # Spatial gradients (x,y,z) are columns k=1..3
        grads = J_u[:, :, 1:4]  # [N,3,3]

        # Divergence: trace of grads
        div_u = torch.diagonal(grads, dim1=1, dim2=2).sum(dim=1, keepdim=True)

        # Convective term (u · ∇)u = grads @ u (batched matmul)
        conv = torch.bmm(grads, u.unsqueeze(2)).squeeze(2)

        # Pressure gradient wrt (x,y,z)
        grad_p = self._grad(p, txyz)[:, 1:4]

        # momentum residual
        mom_res = du_dt + conv + grad_p

        pde_residual = torch.cat([mom_res, div_u], dim=1)  # [N,4]

        bc_residual = None
        if bc is not None:
            bc_residual = bc.residual(out)

        return Euler3DLoss(pde=pde_residual, bc=bc_residual)
