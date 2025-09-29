import torch
import torch.nn as nn

class SinActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return torch.sin(x)


class PositionalEncoding(nn.Module):
    """Fourier positional encoding for PINNs.

    Encodes each input dimension with [sin(2^k x), cos(2^k x)] for k in [0..n_freqs-1].
    """

    def __init__(self, in_dim: int, n_freqs: int = 4):
        super().__init__()
        self.in_dim = in_dim
        self.n_freqs = n_freqs
        # frequencies as 1, 2, 4, ... 2^{n_freqs-1}
        self.register_buffer("freqs", 2.0 ** torch.arange(n_freqs))

    @property
    def out_dim(self) -> int:
        return self.in_dim * (2 * self.n_freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, D]
        # expand: [N, D, F]
        xf = x.unsqueeze(-1) * self.freqs  # type: ignore[operator]
        return torch.cat([torch.sin(xf), torch.cos(xf)], dim=-1).flatten(start_dim=1)
