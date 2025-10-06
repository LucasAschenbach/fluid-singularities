import typing
from dataclasses import dataclass
import torch


@dataclass
class ResidualLoss:
    """Container for PDE and boundary residual losses."""
    pde: torch.Tensor
    bc: typing.Optional[torch.Tensor]
    data: typing.Optional[torch.Tensor] = None

    def total(self, w_pde: float = 1.0, w_bc: float = 1.0, w_data: float = 1.0) -> torch.Tensor:
        parts = [w_pde * torch.mean(self.pde**2)]
        if self.bc is not None:
            parts.append(w_bc * torch.mean(self.bc**2))
        if self.data is not None:
            parts.append(w_data * torch.mean(self.data**2))
        return sum(parts)


@dataclass
class PDEBoundaryData:
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
        model: torch.nn.Module,
        points: torch.Tensor,
    ) -> torch.Tensor:
        """Compute boundary residual for the provided model `output`."""
        if self.type.lower() != "dirichlet":
            raise NotImplementedError(f"BC type '{self.type}' not supported.")

        target = self.target_fn(points)
        pred = model(points)
        residual = pred - target

        if self.mask is None:
            return residual

        return residual * self.mask

@dataclass
class PDEData:
    """Container for PDE-related data."""
    inputs: torch.Tensor
    outputs: torch.Tensor
    
    def residual(self, model: torch.nn.Module) -> torch.Tensor:
        pred = model(self.inputs)
        return pred - self.outputs


class PDE:
    """Base class for PDE definitions."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        bc: typing.Optional[PDEBoundaryData] = None,
        data: typing.Optional[PDEData] = None,
        loss_cls: typing.Type[ResidualLoss] = ResidualLoss,
    ) -> None:
        """Initialize PDE with input/output dimensions.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            bc: optional PDEBoundaryData to enforce Dirichlet conditions with an
                optional per-output mask.
            data: optional PDEData to enforce targets at fixed input locations.
            loss_cls: ResidualLoss subclass used to package residuals.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bc = bc
        self.data = data
        self.loss_cls = loss_cls

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

    def _build_residual_loss(
        self,
        pde_residual: torch.Tensor,
        model: torch.nn.Module,
        bc_inputs: typing.Optional[torch.Tensor],
    ) -> ResidualLoss:
        """Package PDE, BC, and data residuals using the configured loss class."""
        bc_residual = None
        if self.bc is not None and bc_inputs is not None:
            bc_residual = self.bc.residual(model, bc_inputs)
        data_residual = None
        if self.data is not None:
            data_residual = self.data.residual(model)
        
        return self.loss_cls(pde=pde_residual, bc=bc_residual, data=data_residual)

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
