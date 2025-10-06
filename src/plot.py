from __future__ import annotations

import argparse

from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import torch

from model import MLP
from utils import get_device


@dataclass
class PlotConfig:
    ckpt_path: str
    output_names: list[str]
    domain: list[tuple[float, float]]
    dims: list[int]
    fixed: list[float]
    resolution: int


def plot_mlp(cfg: PlotConfig) -> None:
    """Load a model from a checkpoint and plot its output on a 2D grid."""
    device = get_device()
    ckpt = torch.load(cfg.ckpt_path, map_location=device)

    model_cfg = ckpt.get("config", {})
    input_dim = model_cfg.get("in_dim", 2)    # Default for Boussinesq2DSelfSimilar
    output_dim = model_cfg.get("out_dim", 3)  # Default for Boussinesq2DSelfSimilar
    model = MLP(
        in_dim=input_dim,
        out_dim=output_dim,
        hidden_layers=model_cfg.get("hidden_layers", [128, 128, 128, 128]),
        activation=model_cfg.get("activation", "tanh"),
        use_positional_encoding=model_cfg.get("use_positional_encoding", True),
    ).to(device)
    
    model.load_state_dict(ckpt["model"])
    model.eval()

    assert len(cfg.dims) + len(cfg.fixed) == input_dim, (
        f"Number of plot dimensions ({len(cfg.dims)}) and fixed values ({len(cfg.fixed)}) "
        f"must sum to model input dimension ({input_dim})."
    )

    x1 = torch.linspace(cfg.domain[0][0], cfg.domain[0][1], cfg.resolution, device=device)
    x2 = torch.linspace(cfg.domain[1][0], cfg.domain[1][1], cfg.resolution, device=device)
    grid_x1, grid_x2 = torch.meshgrid(x1, x2, indexing="ij")

    inputs = torch.zeros(cfg.resolution**2, input_dim, device=device)
    inputs[:, cfg.dims[0]] = grid_x1.flatten()
    inputs[:, cfg.dims[1]] = grid_x2.flatten()

    fixed_dims = sorted(list(set(range(input_dim)) - set(cfg.dims)))
    for i, dim_idx in enumerate(fixed_dims):
        inputs[:, dim_idx] = cfg.fixed[i]

    with torch.no_grad():
        outputs = model(inputs)

    outputs = outputs.reshape(cfg.resolution, cfg.resolution, -1).cpu().numpy()
    grid_x1 = grid_x1.cpu().numpy()
    grid_x2 = grid_x2.cpu().numpy()

    num_outputs = outputs.shape[-1]
    fig, axes = plt.subplots(1, num_outputs, figsize=(5 * num_outputs, 4))
    if num_outputs == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        pcm = ax.pcolormesh(grid_x1, grid_x2, outputs[..., i], shading="auto", cmap="viridis")
        fig.colorbar(pcm, ax=ax)
        ax.set_xlabel(f"dim {cfg.dims[0]}")
        ax.set_ylabel(f"dim {cfg.dims[1]}")
        title = cfg.output_names[i] if i < len(cfg.output_names) else f"Output {i+1}"
        ax.set_title(title)

    plt.tight_layout()
    plt.show()


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot model output from a checkpoint.")
    p.add_argument("--ckpt_path", type=str, help="Path to the model checkpoint (.pt file).")
    p.add_argument("--output_names", type=str, nargs="+", default=[], help="Names of the output channels to label the plots.")
    p.add_argument("--domain", type=float, nargs=4, default=[0, 1, -1, 1], help="Domain for the plot (x1_min, x1_max, x2_min, x2_max).")
    p.add_argument("--dims", type=int, nargs=2, default=[0, 1], help="Two domain dimensions to use for plotting (e.g., --dims 0 2).")
    p.add_argument("--fixed", type=float, nargs="*", default=[], help="Fixed values for non-plotted dimensions.")
    p.add_argument("--resolution", type=int, default=100, help="Resolution of the plot grid.")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    cfg = PlotConfig(
        ckpt_path=args.ckpt_path,
        output_names=args.output_names,
        domain=[(args.domain[0], args.domain[1]), (args.domain[2], args.domain[3])],
        dims=args.dims,
        fixed=args.fixed,
        resolution=args.resolution,
    )
    plot_mlp(cfg)


if __name__ == "__main__":
    main()
