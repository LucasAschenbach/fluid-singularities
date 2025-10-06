from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Callable

import torch

from .base import ResidualLoss, PDE, BoundaryData


@dataclass
class Boussinesq2DLoss(ResidualLoss):
    """Container for 2D Boussinesq (self-similar) residuals and optional BC residuals."""
    pass


class Boussinesq2DSelfSimilarPDE(PDE):
    r"""2D inviscid Boussinesq in self-similar coordinates (steady system in y).

    Unknowns in similarity space y=(y1,y2):
        - Streamfunction: Psi(y)
        - Temperature/buoyancy: Theta(y)
        - Vorticity: Omega(y) (built from NN with symmetry/decay envelope)

    Velocity: u = (∂_{y2} Psi, -∂_{y1} Psi)

    Steady residuals parameterized by λ (derived from the similarity ansatz):
        R_omega = u·∇Omega - (1+λ) y·∇Omega + Omega - ∂_{y1} Theta
        R_theta = u·∇Theta - (1+λ) y·∇Theta + (1-λ) Theta
        R_psi   = Δ Psi - Omega

    Model interface:
        This class expects `model` to map compactified inputs (q, beta) -> (hOmega, hTheta, hPsi),
        i.e. output_dim = 3. We apply symmetry/decay envelopes internally:

            q(y)    = (1 + |y|^2)^(-1/(2*(1+λ)))         # compactified radial feature
            beta(y) = y2 / sqrt(1 + |y|^2)               # bounded vertical feature

            odd_y1  = y1 / sqrt(1 + |y|^2)

            Omega(y) = odd_y1 * hOmega(q,beta) * q(y)             # odd in y1, fixed power-law tail
            Theta(y) = hTheta(q,beta) * q(y)**theta_decay_power   # even in y1 (from inputs)
            Psi(y)   = hPsi(q,beta)   * q(y)**psi_decay_power     # even in y1 (from inputs)

    Notes:
        * inputs must be [N,2] points in y-space with requires_grad=True.
        * bc_inputs (if provided) are passed to `bc.target_fn` for Dirichlet residuals.
    """

    def __init__(
        self,
        lambda_value: float,
        theta_decay_power: float = 0.0,
        psi_decay_power: float = 0.0,
        bc: Optional[BoundaryData] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(input_dim=2, output_dim=3, bc=bc, device=device)
        self.lambda_value = float(lambda_value)
        self.theta_decay_power = float(theta_decay_power)
        self.psi_decay_power = float(psi_decay_power)

    @staticmethod
    def _compactify(y: torch.Tensor, lambda_value: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Transform (y1,y2) into compactified coordinates (r2, q, beta).""" 
        y1, y2 = y[..., 0], y[..., 1]
        r2 = 1.0 + y1 * y1 + y2 * y2
        q = r2 ** (-0.5 / (1.0 + lambda_value))
        beta = y2 / torch.sqrt(r2)
        return r2, q, beta

    @staticmethod
    def _odd_y1_prefactor(r2: torch.Tensor, y1: torch.Tensor) -> torch.Tensor:
        """Odd-in-y1 symmetry prefactor used for vorticity envelope."""
        return y1 / torch.sqrt(r2)

    def _compose_fields(
        self,
        model: torch.nn.Module,
        y: torch.Tensor,
        lambda_value: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """From model outputs (hOmega, hTheta, hPsi) and envelopes, build (Omega, Theta, Psi)."""
        y1, y2 = y[..., 0], y[..., 1]
        r2, q, beta = self._compactify(y, lambda_value)

        feats = torch.stack([q, beta], dim=-1)
        raw = model(feats)

        assert raw.shape[-1] == 3, "Model must output (hOmega, hTheta, hPsi) with last dim = 3."
        hOmega = raw[..., 0]
        hTheta = raw[..., 1]
        hPsi   = raw[..., 2]

        odd_y1 = self._odd_y1_prefactor(r2, y1)

        # Envelopes: symmetry + decay
        Omega = odd_y1 * hOmega * q
        Theta = hTheta * (q ** self.theta_decay_power)
        Psi   = hPsi   * (q ** self.psi_decay_power)
        return Omega, Theta, Psi

    def residuals(
        self,
        model: torch.nn.Module,
        inputs: torch.Tensor,
        bc_inputs: Optional[torch.Tensor] = None,
    ) -> Boussinesq2DLoss:
        """Compute self-similar Boussinesq residuals at collocation points and optional BC residuals."""
        y = inputs
        assert y.ndim == 2 and y.shape[-1] == 2, "inputs must be [N,2] similarity points (y1,y2)."
        assert y.requires_grad, "Input points must require grad for autograd derivatives."

        lam = self.lambda_value
        s = 1.0 + lam

        # Compose physical fields with symmetry/decay envelopes
        Omega, Theta, Psi = self._compose_fields(model, y, lam)

        # Velocity u = (∂y2 Psi, -∂y1 Psi)
        dPsi = self._grad(Psi, y)  # [N,2]
        u1, u2 = dPsi[:, 1], -dPsi[:, 0]  # u = (dPsi/dy2, -dPsi/dy1)

        # Gradients/Laplacian and ∂_{y1} Theta
        dOm = self._grad(Omega, y)  # [N,2]
        dTh = self._grad(Theta, y)  # [N,2]

        # Laplacian of Psi
        dPsi_dy1 = dPsi[:, 0:1]
        dPsi_dy2 = dPsi[:, 1:2]
        d2Psi_dy1 = self._grad(dPsi_dy1, y)[:, 0:1]
        d2Psi_dy2 = self._grad(dPsi_dy2, y)[:, 1:2]
        lap_Psi = (d2Psi_dy1 + d2Psi_dy2).squeeze(-1)  # [N]

        # Dot-products
        y_dot_gradOm = y[:, 0] * dOm[:, 0] + y[:, 1] * dOm[:, 1]
        y_dot_gradTh = y[:, 0] * dTh[:, 0] + y[:, 1] * dTh[:, 1]
        u_dot_gradOm = u1 * dOm[:, 0] + u2 * dOm[:, 1]
        u_dot_gradTh = u1 * dTh[:, 0] + u2 * dTh[:, 1]

        # Residuals
        R_omega = u_dot_gradOm - s * y_dot_gradOm + Omega - dTh[:, 0]
        R_theta = u_dot_gradTh - s * y_dot_gradTh + (1.0 - lam) * Theta
        R_psi   = lap_Psi - Omega

        pde_residual = torch.stack([R_omega, R_theta, R_psi], dim=1)  # [N,3]

        bc_residual: Optional[torch.Tensor] = None
        if self.bc is not None:
            bc_out = model(torch.stack(self._compactify(bc_inputs, lam)[1:], dim=-1))
            bc_residual = self.bc.residual(bc_out, points=bc_inputs)

        return Boussinesq2DLoss(pde=pde_residual, bc=bc_residual)
