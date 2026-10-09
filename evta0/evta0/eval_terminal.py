"""MSE and Kendall tau-a evaluation at rollout terminal states.

For every rollout in an augmented HDF5 collection (the same format used for
training; see :mod:`evta0.data`), the model scores the exact terminal frame
with the standard history window [T-8, T-6, T-4, T-2, T] (zero-left padded).
Two outcome-quality metrics are then computed:

* **MSE** between the terminal prediction and the true terminal target
  (``rewards[-1]``, i.e. the soft-augmented outcome).
* **Kendall tau-a** over all (success, failure) prediction pairs within each
  task, macro-averaged over the tasks that contain both outcomes.

Usage::

    python -m evta0.eval_terminal \
        --checkpoint /path/to/checkpoint.pth \
        --hdf5-root /path/to/rollout_collection \
        --output results/terminal.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

from evta0.data import frames_to_uint8
from evta0.metrics import kendall_tau_a, mse
from evta0.model import EVTA0
from evta0.protocol import ordered_history_indices


def terminal_window(group, terminal: int, context_window: int, frame_interval: int) -> dict:
    """History window ending exactly at the terminal frame, zero-left padded."""
    requested = ordered_history_indices(terminal, context_window, frame_interval)
    valid = [index for index in requested if index >= 0]
    pad = context_window - len(valid)

    def read(key: str) -> np.ndarray:
        frames = frames_to_uint8(group[key][valid])
        if pad:
            frames = np.concatenate(
                [np.zeros((pad, *frames.shape[1:]), dtype=frames.dtype), frames], axis=0
            )
        return frames

    return {"main_images": read("obs_main_images"), "wrist_images": read("obs_wrist_images")}


@torch.inference_mode()
def evaluate_terminal(
    model: EVTA0,
    hdf5_root: str,
    *,
    context_window: int = 5,
    frame_interval: int = 2,
    batch_size: int = 32,
) -> dict:
    model.eval()

    samples = []  # (suite, task, success, target, window)
    for hdf5_path in sorted(Path(hdf5_root).rglob("*.hdf5")):
        suite = hdf5_path.parent.name
        with h5py.File(hdf5_path, "r") as handle:
            traj_keys = sorted(
                (key for key in handle.keys() if key.startswith("traj_")),
                key=lambda key: int(key.split("_")[-1]),
            )
            for traj_key in traj_keys:
                group = handle[traj_key]
                terminal = int(group["dones"].shape[0]) - 1
                samples.append(
                    {
                        "suite": suite,
                        "task": str(group.attrs["task_prompt"]),
                        "success": bool(group.attrs["success"]),
                        "target": float(np.asarray(group["rewards"][terminal]).item()),
                        "window": terminal_window(group, terminal, context_window, frame_interval),
                    }
                )

    predictions = []
    for start in tqdm(range(0, len(samples), batch_size), desc="terminal batches"):
        part = samples[start : start + batch_size]
        batch = {
            "obs": {
                "main_images": np.stack([s["window"]["main_images"] for s in part]),
                "wrist_images": np.stack([s["window"]["wrist_images"] for s in part]),
            },
            "task_prompt": [s["task"] for s in part],
        }
        preds = model.predict(batch)
        predictions.extend(preds.detach().float().cpu().tolist())

    rows = []
    for sample, prediction in zip(samples, predictions):
        rows.append(
            {
                "suite": sample["suite"],
                "task": sample["task"],
                "success": sample["success"],
                "target": sample["target"],
                "prediction": prediction,
            }
        )

    def summarize(group_rows: list[dict]) -> dict:
        return {
            "mse": mse([row["prediction"] for row in group_rows],
                       [row["target"] for row in group_rows]),
            "num_rollouts": len(group_rows),
            **{
                f"tau_a_{key}": value
                for key, value in kendall_tau_a(group_rows).items()
            },
        }

    report = {
        "overall": summarize(rows),
        "by_suite": {
            suite: summarize([row for row in rows if row["suite"] == suite])
            for suite in dict.fromkeys(row["suite"] for row in rows)
        },
        "rows": rows,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to configs/evta0.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--hdf5-root", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    model_kwargs = {}
    if args.config:
        import yaml

        model_kwargs.update(yaml.safe_load(open(args.config))["model"])
    model = EVTA0(device_map=args.device, **model_kwargs)
    model.load_checkpoint(args.checkpoint)

    report = evaluate_terminal(
        model, args.hdf5_root,
        context_window=getattr(model, "context_window", 5),
        frame_interval=getattr(model, "frame_interval", 2),
        batch_size=args.batch_size,
    )

    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2))
        print(f"Saved report to {args.output}")


if __name__ == "__main__":
    main()
