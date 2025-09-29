from .base import ResidualLoss, BoundaryData, PDE
from .euler3d import Euler3DPDE, Euler3DLoss

__all__ = [
    "ResidualLoss",
    "BoundaryData",
    "PDE",
    "Euler3DPDE",
    "Euler3DLoss",
]
