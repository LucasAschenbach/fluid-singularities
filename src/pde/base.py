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

    This provides a structured way to encode boundary conditions (BCs) for
    PINNs. Currently only Dirichlet conditions are supported in a flexible,
    masked form.

    Attributes:
        target: [N, output_dim] tensor with desired values on boundary.
        mask: Optional [N, output_dim] boolean/float mask (1=enforce, 0=ignore)
              allowing selective enforcement per output component and point.
        type: Type of boundary condition. Only "dirichlet" is supported now.
    """

    target: torch.Tensor
    mask: typing.Optional[torch.Tensor] = None
    type: str = "dirichlet"

    def residual(self, output: torch.Tensor) -> torch.Tensor:
        """Compute boundary residual for the provided model `output`.

        For Dirichlet BCs, residual = mask * (output - target), with mask=1 if
        not provided.
        """
        if self.type.lower() != "dirichlet":
            raise NotImplementedError(f"BC type '{self.type}' not supported.")

        if self.mask is None:
            return output - self.target

        # Allow boolean masks; ensure same dtype as output for multiplication
        m = self.mask.to(dtype=output.dtype)
        return (output - self.target) * m


class PDE:
    """Base class for PDE definitions."""

    def __init__(self, input_dim: int = 4, output_dim: int = 4, device: typing.Optional[torch.device] = None) -> None:
        """Initialize PDE with input/output dimensions.
        
        Args:
            input_dim: Input dimension (default 4 for t,x,y,z)
            output_dim: Output dimension (default 4 for u,v,w,p)
            device: Optional torch device for tensors
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device

    @staticmethod
    def _grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)[0]

    def residuals(
        self,
        model: torch.nn.Module,
        inputs: torch.Tensor,
        bc: typing.Optional[BoundaryData] = None,
    ) -> ResidualLoss:
        """Compute PDE residuals at collocation points and BC residuals.

        Args:
            model: Network mapping inputs (e.g., (t,x,y,z)) to outputs
                   (e.g., (u,v,w,p)).
            inputs: [N, input_dim] with requires_grad=True for autograd.
            bc: Optional boundary condition specification to enforce via a
                masked Dirichlet residual on the model outputs.
        Returns:
            ResidualLoss with PDE residuals and optional boundary residuals.
        """
        raise NotImplementedError("Subclasses must implement this method.")
