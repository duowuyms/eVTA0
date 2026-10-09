"""Minimal inference CLI: score one state from five images.

The images must be ordered oldest-to-current, i.e. for the default protocol:
``[t-8, t-6, t-4, t-2, t]``.  Frames before the start of the trajectory should
be supplied as plain black images (zero left padding).

Usage::

    python -m evta0.infer \
        --checkpoint /path/to/checkpoint.pth \
        --task-prompt "put the bowl on the plate" \
        --main-images t_minus_8.png t_minus_6.png t_minus_4.png t_minus_2.png t.png \
        [--wrist-images w1.png w2.png w3.png w4.png w5.png]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from evta0.model import EVTA0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one eVTA0 success-probability prediction")
    parser.add_argument("--task-prompt", required=True)
    parser.add_argument("--main-images", nargs="+", required=True)
    parser.add_argument("--wrist-images", nargs="*", default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None, help="Optional configs/evta0.yaml")
    parser.add_argument("--base-model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    context_window = 5
    frame_interval = 2
    if args.config is not None:
        import yaml

        model_cfg = yaml.safe_load(args.config.read_text())["model"]
        context_window = int(model_cfg.get("context_window", context_window))
        frame_interval = int(model_cfg.get("frame_interval", frame_interval))

    expected = context_window
    if len(args.main_images) != expected:
        offsets = [-(context_window - 1 - i) * frame_interval for i in range(context_window)]
        raise SystemExit(
            f"This protocol expects exactly {expected} main images, offsets {offsets}."
        )
    if args.wrist_images is not None and len(args.wrist_images) not in (0, expected):
        raise SystemExit(f"Provide either zero wrist images or exactly {expected} wrist images.")

    model = EVTA0(
        pretrained_path=args.base_model,
        device_map="cpu" if args.device == "cpu" else args.device,
        context_window=context_window,
        frame_interval=frame_interval,
    )
    model.load_checkpoint(args.checkpoint)

    batch = {
        "task_prompt": args.task_prompt,
        "obs": {"main_images": [Image.open(path).convert("RGB") for path in args.main_images]},
    }
    if args.wrist_images:
        batch["obs"]["wrist_images"] = [Image.open(path).convert("RGB") for path in args.wrist_images]

    model.eval()
    with torch.inference_mode():
        probability = model.predict(batch)

    result = {
        "protocol": "eVTA0",
        "context_window": context_window,
        "frame_interval": frame_interval,
        "task_prompt": args.task_prompt,
        "success_probability": float(probability[0].detach().cpu()),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
