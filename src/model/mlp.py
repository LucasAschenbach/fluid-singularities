from __future__ import annotations

from typing import Iterable, List, Optional

import torch
import torch.nn as nn

from .utils import SinActivation, PositionalEncoding


class MLP(nn.Module):
    """Simple MLP for PINNs, mapping (t, x, y, z) -> (u, v, w, p).

    - Optional Fourier positional encoding on inputs.
    - Activation: tanh (default) or sine/gelu/relu.
    """

    def __init__(
        self,
        in_dim: int = 4,
        out_dim: int = 4,
        activation: str = "tanh",
        hidden_layers: Iterable[int] = [128, 128, 128, 128],
        use_positional_encoding: bool = True,
        use_final_activation: bool = False,
    ) -> None:
        super().__init__()

        self.pos_enc: Optional[PositionalEncoding] = None
        if use_positional_encoding:
            self.pos_enc = PositionalEncoding(in_dim, n_freqs=4)
            core_in = self.pos_enc.out_dim
        else:
            core_in = in_dim

        acts = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "sine": SinActivation,
            "sin": SinActivation,
        }
        if activation not in acts:
            raise ValueError(f"Unknown activation '{activation}'")

        layers: List[nn.Module] = []
        prev = core_in
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(acts[activation]())
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        if use_final_activation:
            layers.append(acts[activation]())

        self.net = nn.Sequential(*layers)

    def forward(self, tx: torch.Tensor) -> torch.Tensor:
        if self.pos_enc is not None:
            z = self.pos_enc(tx)
        else:
            z = tx
        return self.net(z)

    @staticmethod
    def split_output(out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split network output into velocity (u) and pressure (p).

        Expects last dim = 4: [u, v, w, p].
        Returns (u: [...,3], p: [...,1]).
        """
        u = out[..., :3]
        p = out[..., 3:4]
        return u, p
