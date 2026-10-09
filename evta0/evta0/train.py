"""TD-bootstrap training for eVTA0.

Training objective::

    y_t = d_t * z_t + (1 - d_t) * p̂_θ⁻(x_t⁺)
    L(θ) = E[ (p̂_θ(x_t) - y_t)² ]

Terminal outcomes ``z_t`` anchor the regression; non-terminal targets bootstrap
from a slow target network ``θ⁻`` (an EMA copy updated every ``K`` optimizer
steps with mixing coefficient ``α``).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from evta0.data import EVTA0TrajectoryDataset, collate_td_tuples
from evta0.model import EVTA0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(path: str) -> dict:
    """Load a YAML config with ``${ENV_VAR}`` expansion in path strings."""
    with open(path) as handle:
        config = yaml.safe_load(handle)
    return config


def build_dataset(config: dict) -> EVTA0TrajectoryDataset:
    data_cfg = config["data"]
    hdf5_root = data_cfg["hdf5_root"]
    manifest = data_cfg.get("split_manifest")
    holdout_ratio = float(data_cfg.get("holdout_ratio", 0.0))
    if manifest:
        exclude = EVTA0TrajectoryDataset.load_exclusions_from_manifest(manifest)
    elif holdout_ratio > 0:
        exclude = EVTA0TrajectoryDataset.random_exclusions(
            hdf5_root, holdout_ratio, int(config["train"]["seed"])
        )
    else:
        exclude = {}
    return EVTA0TrajectoryDataset(
        hdf5_root=hdf5_root,
        sample_step=int(data_cfg.get("sample_step", 15)),
        context_window=int(config["model"]["context_window"]),
        frame_interval=int(config["model"]["frame_interval"]),
        target_horizon=int(config["model"]["target_horizon"]),
        exclude=exclude,
    )


def train(config: dict, args: argparse.Namespace) -> None:
    train_cfg = config["train"]
    set_seed(int(train_cfg.get("seed", 42)))

    save_dir = Path(args.save_dir or config["output"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(config)
    print(f"[data] {dataset.statistics()}")

    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        drop_last=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=collate_td_tuples,
    )

    model = EVTA0(pretrained_path=config["model"]["pretrained_path"], **{
        key: value
        for key, value in config["model"].items()
        if key != "pretrained_path"
    })
    target_model = model.build_target_model()

    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )

    start_epoch = 0
    global_step = 0
    if args.resume:
        checkpoint = model.load_checkpoint(args.resume)
        if "target_model_state" in checkpoint:
            target_model.load_checkpoint_state(checkpoint["target_model_state"])
        else:
            target_model.load_checkpoint_state(checkpoint["model_state"])
        start_epoch = int(checkpoint.get("current_epoch", -1)) + 1
        print(f"[resume] continuing from epoch {start_epoch}")

    interval = int(train_cfg.get("target_update_interval", 16))    # K
    alpha = float(train_cfg.get("ema_alpha", 0.1))                 # α
    grad_clip = float(train_cfg.get("grad_clip", 0.5))
    history = []

    for epoch in range(start_epoch, int(train_cfg["epochs"])):
        model.train()
        epoch_losses, epoch_preds, epoch_targets = [], [], []
        for batch in tqdm(loader, desc=f"epoch {epoch}"):
            task_prompt = batch["task_prompt"]

            optimizer.zero_grad()
            predictions = model({"obs": batch["obs"], "task_prompt": task_prompt})

            with torch.no_grad():
                bootstrap = target_model({"obs": batch["next_obs"], "task_prompt": task_prompt})
                # Only scalar labels need to reach the model device; image
                # tensors are converted to PIL images inside the model.
                done = batch["done"].to(predictions.device)
                terminal_reward = batch["terminal_reward"].to(predictions.device)
                targets = torch.where(done, terminal_reward, bootstrap)

            loss = torch.nn.functional.mse_loss(predictions.float(), targets.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (param for param in model.parameters() if param.requires_grad), grad_clip
            )
            optimizer.step()

            global_step += 1
            if global_step % interval == 0:
                target_model.ema_update_from(model, alpha)

            epoch_losses.append(loss.item())
            epoch_preds.extend(predictions.detach().float().cpu().tolist())
            epoch_targets.extend(targets.detach().float().cpu().tolist())

        stats = {
            "epoch": epoch,
            "loss": float(np.mean(epoch_losses)),
            "pred_mean": float(np.mean(epoch_preds)),
            "pred_std": float(np.std(epoch_preds)),
            "target_mean": float(np.mean(epoch_targets)),
            "target_std": float(np.std(epoch_targets)),
        }
        history.append(stats)
        print(
            f"[epoch {epoch}] loss={stats['loss']:.4f} "
            f"pred={stats['pred_mean']:.3f}±{stats['pred_std']:.3f} "
            f"target={stats['target_mean']:.3f}±{stats['target_std']:.3f}"
        )

        model.save_checkpoint(
            save_dir / f"checkpoint_epoch_{epoch}.pth",
            current_epoch=epoch,
            global_step=global_step,
            supervision="td_bootstrap_ema_v1",
            target_model_state=target_model.get_checkpoint_state(),
            config=config,
        )
        (save_dir / "training_history.json").write_text(json.dumps(history, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to configs/evta0.yaml")
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from")
    args = parser.parse_args()
    train(load_config(args.config), args)


if __name__ == "__main__":
    main()
