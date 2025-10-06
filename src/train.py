from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Optional
from tqdm import tqdm

import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from pde import (
    PDE,
    PDEData,
    Euler3DPDE,
    Boussinesq2DSelfSimilarPDE,
    sample_rect_interior,
    sample_rect_boundary
)
from model import MLP
from utils import get_device, set_seed, mse


@dataclass
class TrainConfig:
    epochs: int = 2000
    batch_size: int = 4096
    bc_batch_size: int = 1024
    steps_per_epoch: int = 15
    lr: float = 1e-3
    w_pde: float = 1.0
    w_bc: float = 1.0
    w_data: float = 1.0
    seed: int = 42
    ckpt: Optional[str] = None
    log_dir: str = "runs"
    run_name: str = "boussinesq2d_selfsimilar"
    ckpt_interval: int = 25
    device: str = get_device()


def save_checkpoint(model: torch.nn.Module, cfg: TrainConfig, log_dir: str, name: str) -> None:
    ckpt_path = os.path.join(log_dir, name)
    # TODO: include input/output dimensions
    torch.save({"model": model.state_dict(), "config": cfg.__dict__}, ckpt_path)


def train(cfg: TrainConfig) -> None:
    # Logging setup
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = cfg.log_dir + f"/{cfg.run_name}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    device = torch.device(cfg.device)
    set_seed(cfg.seed)


    lam0 = 1.90  # a stable-ish value to start; later refine via infer_lambda()
    data_norm = PDEData(
        inputs=torch.tensor([[0.0, 0.0], [8.0, 0.0], [8.0, 8.0], [0.0, 8.0], [-8.0, 8.0], [-8.0, 0.0], [-8.0, -8.0], [0.0, -8.0], [8.0, -8.0]], dtype=torch.get_default_dtype(), device=device),
        outputs=torch.tensor([[1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype=torch.get_default_dtype(), device=device),
    ) # normalizing conditions to fix scale and prevent trivial zero solution
    pde = Boussinesq2DSelfSimilarPDE(
        lambda_value=lam0,
        theta_decay_power=0.0,    # try 1.0 later for extra decay on Theta
        psi_decay_power=0.0,      # often 0 is fine; add mild decay if ∇Psi grows too much
        data=data_norm,
    )
    domain = [(0,1), (-1,1)]  # q,β

    # Model
    # TODO: load from checkpoint if provided
    model = MLP(
        in_dim=pde.input_dim,
        out_dim=pde.output_dim,
        hidden_layers=[128, 128, 128, 128],
        activation="tanh",
        use_positional_encoding=True
    ).to(device)

    opt = optim.Adam(model.parameters(), lr=cfg.lr)

    model.train()
    global_step = 0
    loss = torch.tensor(float("inf"))
    prev_loss_val = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        # Sample collocation points once per epoch
        points_epoch = sample_rect_interior(domain, cfg.batch_size, device)
        bc_points_epoch = None
        if pde.bc is not None:
            bc_points_epoch = sample_rect_boundary(domain, cfg.bc_batch_size, device)

        train_loop = tqdm(range(cfg.steps_per_epoch), desc=f"Epoch {epoch}/{cfg.epochs}")
        for step in train_loop:
            opt.zero_grad()
            points = points_epoch.detach().requires_grad_(True) # create fresh leaf tensor

            residual = pde.residuals(model, points, bc_inputs=bc_points_epoch)
            loss = residual.total(w_pde=cfg.w_pde, w_bc=cfg.w_bc, w_data=cfg.w_data)

            loss.backward()
            opt.step()

            # logging
            train_loop.set_postfix(
                loss=loss.item(),
                pde=mse(residual.pde).item(),
                bc=mse(residual.bc).item() if residual.bc is not None else 0.0,
                data=mse(residual.data).item() if residual.data is not None else 0.0,
            )
            writer.add_scalar("loss/total", loss.item(), global_step)
            writer.add_scalar("loss/pde", mse(residual.pde).item(), global_step)
            if residual.bc is not None:
                writer.add_scalar("loss/bc", mse(residual.bc).item(), global_step)
            if residual.data is not None:
                writer.add_scalar("loss/data", mse(residual.data).item(), global_step)

            global_step += 1

        if epoch % cfg.ckpt_interval == 0 and epoch > 0:
            save_checkpoint(model, cfg, log_dir, f"model_{epoch}.pt")
        if loss.item() < prev_loss_val and epoch > cfg.ckpt_interval:
            save_checkpoint(model, cfg, log_dir, "model_best.pt")
            prev_loss_val = loss.item()
    
    save_checkpoint(model, cfg, log_dir, f"model_{cfg.epochs}.pt")
    print("Training completed.")

    # Simple validation: compare on a grid
    model.eval()
    # TODO: implement proper validation
    writer.close()


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a PINN on a PDE operator.")
    p.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    p.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    p.add_argument("--bc-batch-size", type=int, default=TrainConfig.bc_batch_size)
    p.add_argument("--epoch-steps", type=int, default=TrainConfig.steps_per_epoch)
    p.add_argument("--lr", type=float, default=TrainConfig.lr)
    p.add_argument("--w-pde", type=float, default=TrainConfig.w_pde)
    p.add_argument("--w-bc", type=float, default=TrainConfig.w_bc)
    p.add_argument("--w-data", type=float, default=TrainConfig.w_data)
    p.add_argument("--seed", type=int, default=TrainConfig.seed)
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--log-dir", type=str, default=TrainConfig.log_dir)
    p.add_argument("--run-name", type=str, default=TrainConfig.run_name)
    p.add_argument("--ckpt-interval", type=int, default=TrainConfig.ckpt_interval)
    p.add_argument("--device", type=str, default=TrainConfig.device)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        bc_batch_size=args.bc_batch_size,
        steps_per_epoch=args.epoch_steps,
        lr=args.lr,
        w_pde=args.w_pde,
        w_bc=args.w_bc,
        w_data=args.w_data,
        seed=args.seed,
        ckpt=args.ckpt,
        log_dir=args.log_dir,
        run_name=args.run_name,
        ckpt_interval=args.ckpt_interval,
        device=args.device,
    )
    train(cfg)


if __name__ == "__main__":
    main()
