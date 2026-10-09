"""VOC / VROC evaluation on demonstration trajectories.

The evaluation data are LeRobot-format demonstration datasets (one directory
per suite with ``meta/episodes.jsonl`` and ``data/chunk-*/episode_*.parquet``).

Protocol (default settings):

1. Anchor frames of an episode are the grid ``range(s, T, s)`` with stride
   ``s = sample_step`` (default 5).  The first, heavily zero-padded windows
   are skipped and the terminal frame is never appended separately.
2. Each anchor ``t`` is scored twice:
   - **forward**: the model sees [t-8, t-6, t-4, t-2, t] (zero-left padded);
   - **reverse**: the model sees the same frames the reversed trajectory
     would expose, i.e. [t+8, t+6, t+4, t+2, t] relative to original time.
3. VOC = tie-aware Spearman(forward predictions, timeline).
   VROC = tie-aware Spearman(reverse-order predictions, reversed timeline).
4. Per-trajectory values are averaged per suite and overall.

Usage::

    python -m evta0.eval_progress \
        --config configs/evta0.yaml \
        --checkpoint /path/to/checkpoint.pth \
        --data-root /path/to/lerobot--libero_spatial_image@v2.0 \
        --num-episodes 10 \
        --output results/progress_spatial.json
"""

from __future__ import annotations

import argparse
import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from evta0.metrics import summarize_trajectory_metrics, voc, vroc
from evta0.model import EVTA0
from evta0.protocol import ordered_history_indices


# ----------------------------------------------------------------------
# Episode loading.
# ----------------------------------------------------------------------
@dataclass
class Episode:
    suite: str
    episode_index: int
    task_prompt: str
    parquet_path: Path
    df: pd.DataFrame = None


def iter_episodes(data_root: str, limit: int | None = None) -> list[Episode]:
    """List episodes of a LeRobot dataset root (sorted by episode index)."""
    root = Path(data_root)
    suite = root.name
    parquet_paths = {}
    for chunk_dir in sorted(root.glob("data/chunk-*")):
        for parquet_path in chunk_dir.glob("episode_*.parquet"):
            parquet_paths[int(parquet_path.stem.split("_")[-1])] = parquet_path

    episodes = []
    with (root / "meta" / "episodes.jsonl").open() as handle:
        for line in handle:
            meta = json.loads(line)
            episode_index = int(meta["episode_index"])
            parquet_path = parquet_paths.get(episode_index)
            if parquet_path is None:
                continue
            episodes.append(
                Episode(
                    suite=suite,
                    episode_index=episode_index,
                    task_prompt=meta["tasks"][0],
                    parquet_path=parquet_path,
                )
            )
    episodes.sort(key=lambda episode: episode.episode_index)
    if limit is not None:
        episodes = episodes[:limit]
    return episodes


# ----------------------------------------------------------------------
# Window construction.
# ----------------------------------------------------------------------
_MAIN_COLUMN_CANDIDATES = ("observation.images.image", "observation.image", "image")
_WRIST_COLUMN_CANDIDATES = ("observation.images.wrist_image", "observation.images.image2", "wrist_image")


def _find_columns(df: pd.DataFrame) -> tuple[str, str]:
    main_col = next((name for name in _MAIN_COLUMN_CANDIDATES if name in df.columns), None)
    if main_col is None:
        raise KeyError(f"No image column found; columns are {list(df.columns)}")
    # Datasets without a wrist camera (e.g. MetaWorld) duplicate the main
    # camera as the wrist stream.
    wrist_col = next((name for name in _WRIST_COLUMN_CANDIDATES if name in df.columns), main_col)
    return main_col, wrist_col


def decode_image(value) -> np.ndarray:
    """Decode a parquet image cell into a uint8 HWC array."""
    if hasattr(value, "as_py"):
        value = value.as_py()
    if isinstance(value, dict) and "bytes" in value:
        image = np.asarray(Image.open(io.BytesIO(value["bytes"])).convert("RGB"))
    else:
        image = np.asarray(value)
    return image.astype(np.uint8)


def build_window(
    df: pd.DataFrame,
    main_col: str,
    wrist_col: str,
    anchor: int,
    *,
    context_window: int,
    frame_interval: int,
    reverse: bool = False,
) -> dict:
    """Build the zero-left-padded history window at ``anchor``.

    Returns a dict with ``main_images`` and ``wrist_images`` arrays of shape
    ``(w, H, W, 3)``.
    """
    requested = ordered_history_indices(
        anchor, context_window, frame_interval, reverse=reverse
    )
    valid = [index for index in requested if 0 <= index < len(df)]
    pad = context_window - len(valid)

    frame_slice = df.iloc[valid]
    main_images = [decode_image(value) for value in frame_slice[main_col].tolist()]
    wrist_images = [decode_image(value) for value in frame_slice[wrist_col].tolist()]

    if pad:
        main_images = [np.zeros_like(main_images[0])] * pad + main_images
        wrist_images = [np.zeros_like(wrist_images[0])] * pad + wrist_images

    return {
        "main_images": np.stack(main_images),
        "wrist_images": np.stack(wrist_images),
    }


# ----------------------------------------------------------------------
# Scoring.
# ----------------------------------------------------------------------
@torch.inference_mode()
def score_episode(
    model: EVTA0,
    episode: Episode,
    *,
    sample_step: int = 5,
    context_window: int = 5,
    frame_interval: int = 2,
    batch_size: int = 16,
) -> dict:
    """Score one episode; returns forward and reverse predictions per anchor."""
    if episode.df is None:
        episode.df = pd.read_parquet(episode.parquet_path)
    df = episode.df
    main_col, wrist_col = _find_columns(df)

    anchors = list(range(sample_step, len(df), sample_step))
    # The reversed presentation pairs anchor position i with position n-1-i.
    reverse_anchors = anchors[::-1]

    forward_windows, reverse_windows = [], []
    for anchor, reverse_anchor in zip(anchors, reverse_anchors):
        forward_windows.append(
            build_window(df, main_col, wrist_col, anchor,
                         context_window=context_window, frame_interval=frame_interval)
        )
        reverse_windows.append(
            build_window(df, main_col, wrist_col, reverse_anchor, reverse=True,
                         context_window=context_window, frame_interval=frame_interval)
        )

    def predict(windows: list[dict]) -> list[float]:
        predictions = []
        for start in range(0, len(windows), batch_size):
            part = windows[start : start + batch_size]
            batch = {
                "obs": {
                    "main_images": np.stack([w["main_images"] for w in part]),
                    "wrist_images": np.stack([w["wrist_images"] for w in part]),
                },
                "task_prompt": [episode.task_prompt] * len(part),
            }
            preds = model.predict(batch)
            predictions.extend(preds.detach().float().cpu().tolist())
        return predictions

    return {
        "anchors": anchors,
        "forward_predictions": predict(forward_windows),
        "reverse_predictions": predict(reverse_windows),
    }


def evaluate_progress(
    model: EVTA0,
    episodes: list[Episode],
    *,
    sample_step: int = 5,
    context_window: int = 5,
    frame_interval: int = 2,
    batch_size: int = 16,
) -> dict:
    """Run VOC/VROC evaluation over a list of episodes."""
    model.eval()
    trajectory_metrics = []
    for episode in tqdm(episodes, desc="episodes"):
        scored = score_episode(
            model, episode,
            sample_step=sample_step,
            context_window=context_window,
            frame_interval=frame_interval,
            batch_size=batch_size,
        )
        trajectory_voc = voc(scored["forward_predictions"])
        trajectory_vroc = vroc(scored["reverse_predictions"])
        valid = np.isfinite(trajectory_voc) and np.isfinite(trajectory_vroc)
        trajectory_metrics.append(
            {
                "suite": episode.suite,
                "episode_index": episode.episode_index,
                "task_prompt": episode.task_prompt,
                "num_anchors": len(scored["anchors"]),
                "voc": trajectory_voc,
                "vroc": trajectory_vroc,
                "valid": bool(valid),
                "forward_predictions": scored["forward_predictions"],
                "reverse_predictions": scored["reverse_predictions"],
            }
        )
        tqdm.write(
            f"[{episode.suite} ep{episode.episode_index:03d}] "
            f"VOC={trajectory_voc:.4f} VROC={trajectory_vroc:.4f} "
            f"({episode.task_prompt[:50]})"
        )

    overall = summarize_trajectory_metrics(trajectory_metrics)
    by_suite = {}
    for suite in dict.fromkeys(row["suite"] for row in trajectory_metrics):
        by_suite[suite] = summarize_trajectory_metrics(
            [row for row in trajectory_metrics if row["suite"] == suite]
        )
    return {"overall": overall, "by_suite": by_suite, "trajectories": trajectory_metrics}


# ----------------------------------------------------------------------
# CLI.
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to configs/evta0.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", action="append", required=True,
                        help="LeRobot dataset root; repeat once per suite")
    parser.add_argument("--num-episodes", type=int, default=None,
                        help="Limit episodes per suite")
    parser.add_argument("--episode-index", type=int, action="append", default=None,
                        help="Evaluate only these episode indices (per suite)")
    parser.add_argument("--sample-step", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default=None, help="Where to write the JSON report")
    args = parser.parse_args()

    model_kwargs = {}
    if args.config:
        import yaml

        model_kwargs.update(yaml.safe_load(open(args.config))["model"])
    model = EVTA0(device_map=args.device, **model_kwargs)
    model.load_checkpoint(args.checkpoint)

    episodes = []
    for data_root in args.data_root:
        suite_episodes = iter_episodes(data_root, limit=args.num_episodes)
        if args.episode_index is not None:
            suite_episodes = [
                episode for episode in suite_episodes if episode.episode_index in args.episode_index
            ]
        episodes.extend(suite_episodes)

    report = evaluate_progress(
        model, episodes,
        sample_step=args.sample_step,
        context_window=getattr(model, "context_window", 5),
        frame_interval=getattr(model, "frame_interval", 2),
        batch_size=args.batch_size,
    )

    print("\n[Overall]")
    print(json.dumps(report["overall"], indent=2))
    for suite, metrics in report["by_suite"].items():
        print(f"[{suite}] VOC={metrics['mean_voc']:.4f} VROC={metrics['mean_vroc']:.4f} "
              f"({metrics['num_valid_trajectories']} trajs)")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2))
        print(f"Saved report to {args.output}")


if __name__ == "__main__":
    main()
