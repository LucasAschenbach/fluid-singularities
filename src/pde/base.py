import typing
from dataclasses import dataclass
import torch


@dataclass
class ResidualLoss:
    """Container for PDE and boundary residual losses."""
    pde: torch.Tensor
    bc: typing.Optional[torch.Tensor]

    def total(self, w_pde: float = 1.0, w_bc: float = 1.0) -> torch.Tensor:
        parts = [w_pde * torch.mean(self.pde**2)]
        if self.bc is not None:
            parts.append(w_bc * torch.mean(self.bc**2))
        return sum(parts)


@dataclass
class BoundaryData:
    """Boundary condition specification.

    Only Dirichlet conditions are currently supported. The target is provided
    by `target_fn`, which receives the boundary coordinates and returns the
    desired model outputs at those locations.
    """

    target_fn: typing.Callable[[torch.Tensor], torch.Tensor]
    mask: typing.Optional[torch.Tensor] = None
    type: str = "dirichlet"

    def residual(
        self,
        output: torch.Tensor,
        points: torch.Tensor,
    ) -> torch.Tensor:
        """Compute boundary residual for the provided model `output`."""
        if self.type.lower() != "dirichlet":
            raise NotImplementedError(f"BC type '{self.type}' not supported.")

        target = self.target_fn(points)
        residual = output - target

        if self.mask is None:
            return residual

        return residual * self.mask


class PDE:
    """Base class for PDE definitions."""

    def __init__(
        self,
        input_dim: int = 4,
        output_dim: int = 4,
        bc: typing.Optional[BoundaryData] = None,
        device: typing.Optional[torch.device] = None,
    ) -> None:
        """Initialize PDE with input/output dimensions.
        
        Args:
            input_dim: Input dimension (default 4 for t,x,y,z)
            output_dim: Output dimension (default 4 for u,v,w,p)
            bc: optional BoundaryData to enforce Dirichlet conditions with an
                optional per-output mask.
            device: Optional torch device for tensors
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bc = bc
        self.device = device

    @staticmethod
    def _grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.requires_grad, "Inputs must require grad for autograd derivatives."
        # If `outputs` don't require grad (e.g., constant model), return zeros
        if not outputs.requires_grad:
            return torch.zeros_like(inputs)

        grad = torch.autograd.grad(
            outputs,
            inputs,
            grad_outputs=torch.ones_like(outputs),
            create_graph=True, # needed for higher-order derivatives
            allow_unused=True, # avoid `None` if no dependency on inputs
        )[0]
        return grad

    def residuals(
        self,
        model: torch.nn.Module,
        inputs: torch.Tensor,
        bc_inputs: typing.Optional[torch.Tensor] = None,
    ) -> ResidualLoss:
        """Compute PDE residuals at collocation points and BC residuals.

        Args:
            model: Network mapping inputs (e.g., (t,x,y,z)) to outputs
                   (e.g., (u,v,w,p)).
            inputs: [N, input_dim] with requires_grad=True for autograd.
            bc_inputs: Optional [M, input_dim] boundary sample locations used
                exclusively for boundary-condition residuals. Falls back to
                `inputs` when omitted so existing calls remain valid.
        Returns:
            ResidualLoss with PDE residuals and optional boundary residuals.
        """
        raise NotImplementedError("Subclasses must implement this method.")
