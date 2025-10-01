"""Sampling utilities for rectangular domains used by PINNs."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch

Bounds = Sequence[Tuple[float, float]]


def _bounds_tensor(domain: Bounds, device: torch.device) -> torch.Tensor:
    return torch.tensor(domain, dtype=torch.float32, device=device)


def sample_rect_interior(domain: Bounds, n: int, device: torch.device) -> torch.Tensor:
    """Sample `n` points uniformly from the interior of a hyper-rectangle."""
    if n <= 0:
        return torch.empty(0, len(domain), device=device)

    bounds = _bounds_tensor(domain, device)
    lows = bounds[:, 0]
    highs = bounds[:, 1]
    rand = torch.rand((n, len(domain)), device=device)
    return lows + rand * (highs - lows)


def sample_rect_boundary(domain: Bounds, n: int, device: torch.device) -> torch.Tensor:
    """Sample `n` points on the boundary of a hyper-rectangle."""
    if n <= 0:
        return torch.empty(0, len(domain), device=device)

    bounds = _bounds_tensor(domain, device)
    pts = sample_rect_interior(domain, n, device)

    dims = len(domain)
    idx = torch.randint(0, dims, (n,), device=device)
    sides = torch.randint(0, 2, (n,), device=device)

    row_index = torch.arange(n, device=device)
    pts[row_index, idx] = torch.where(
        sides.bool(),
        bounds[idx, 1],
        bounds[idx, 0],
    )

    return pts
