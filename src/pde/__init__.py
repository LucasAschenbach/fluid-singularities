from .base import ResidualLoss, BoundaryData, PDE
from .euler3d import Euler3DPDE, Euler3DLoss
from .boussinesq2d_selfsimilar import Boussinesq2DSelfSimilarPDE, Boussinesq2DLoss
from .domain import sample_rect_interior, sample_rect_boundary

__all__ = [
    "ResidualLoss",
    "BoundaryData",
    "PDE",
    "Euler3DPDE",
    "Euler3DLoss",
    "Boussinesq2DSelfSimilarPDE",
    "Boussinesq2DLoss",
    "sample_rect_interior",
    "sample_rect_boundary",
]
