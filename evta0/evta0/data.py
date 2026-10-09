"""Training data: TD tuples ``(x_t, x_t^+, d_t, z_t)`` from rollout HDF5 files.

Expected HDF5 layout (one file per task, one top-level group per rollout)::

    traj_0/
        obs_main_images    (T, H, W, 3) uint8     third-person camera
        obs_wrist_images   (T, H, W, 3) uint8     wrist camera
        rewards            (T,) float             augmented soft terminal target at [-1]
        dones              (T,) bool
        actions            (T, A)
        attrs: success (bool), task_index (int), task_prompt (str)

``rewards[-1]`` holds the terminal outcome ``z``: exactly 1.0 for successful
rollouts, and a soft value in (0, 1) for failures whose distance to the
nearest successful trajectory cluster was converted to a target (see
:mod:`evta0.augment`).  Non-terminal entries of ``rewards`` are unused.

Each dataset item is one TD tuple:

* ``obs``               -- the window x_t: frames [t-(w-1)s, ..., t], zero-left
                           padded when the trajectory starts mid-window.
* ``next_obs``          -- the window x_t^+: x_t plus the single frame at
                           ``t + s`` (the TD bootstrap input).
* ``done``              -- d_t, True only at the exact terminal frame.
* ``terminal_reward``   -- z_t, the (possibly soft) terminal outcome.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from evta0.protocol import ordered_history_indices


@dataclass
class _TrajectoryMeta:
    hdf5_path: str
    traj_key: str
    length: int
    task_prompt: str
    terminal_reward: float
    success: bool


def frames_to_uint8(frames: np.ndarray) -> np.ndarray:
    """Normalize stored frames (float [0,1] or uint8) to uint8."""
    frames = np.asarray(frames)
    if frames.dtype != np.uint8:
        if np.issubdtype(frames.dtype, np.floating):
            if float(frames.max(initial=0.0)) <= 1.0:
                frames = frames * 255.0
            frames = np.clip(frames, 0, 255)
        frames = frames.astype(np.uint8)
    return frames


def task_name_from_path(hdf5_path: str) -> str:
    """Derive a task prompt from a per-task HDF5 file name.

    Files are named like ``put_the_bowl_on_the_plate_data.hdf5``; the task is
    the stem without the trailing ``_data`` and with underscores replaced by
    spaces.
    """
    stem = Path(hdf5_path).stem
    if stem.endswith("_data"):
        stem = stem[: -len("_data")]
    return stem.replace("_", " ")


@dataclass
class EVTA0TrajectoryDataset(Dataset):
    """Sample TD tuples from augmented rollout HDF5 files.

    Args:
        hdf5_root: directory scanned recursively for ``*.hdf5`` files.
        sample_step: stride between consecutive anchor frames inside one
            trajectory (training-time subsampling; 15 by default).
        context_window: number of frames w in the history window (5).
        frame_interval: temporal stride s between window frames (2).
        target_horizon: how many frames ahead the bootstrap target looks (2,
            i.e. one stride -- x_t^+ is x_t plus the frame at t+s).
        exclude: optional mapping ``{hdf5_path -> {traj_key, ...}}`` of
            rollouts held out of training (e.g. the evaluation split).
    """

    hdf5_root: str
    sample_step: int = 15
    context_window: int = 5
    frame_interval: int = 2
    target_horizon: int = 2
    exclude: dict[str, set[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sample_step < 1 or self.context_window < 1 or self.frame_interval < 1:
            raise ValueError("sample_step / context_window / frame_interval must be >= 1")
        self._files: dict[str, h5py.File] = {}

        self.trajectories: list[_TrajectoryMeta] = []
        root = Path(self.hdf5_root)
        for hdf5_path in sorted(root.rglob("*.hdf5")):
            path_str = str(hdf5_path)
            rel_str = str(hdf5_path.relative_to(root))
            excluded = self.exclude.get(path_str, self.exclude.get(rel_str, set()))
            fallback_task = task_name_from_path(path_str)
            with h5py.File(hdf5_path, "r") as handle:
                traj_keys = sorted(
                    (key for key in handle.keys() if key.startswith("traj_")),
                    key=lambda key: int(key.split("_")[-1]),
                )
                for traj_key in traj_keys:
                    if traj_key in excluded:
                        continue
                    group = handle[traj_key]
                    attrs = group.attrs
                    # Collections store the task prompt under different
                    # attribute names; the file name is the last resort.
                    task_prompt = next(
                        (
                            str(attrs[name])
                            for name in ("task_prompt", "task_description")
                            if name in attrs
                        ),
                        fallback_task,
                    )
                    success = (
                        bool(attrs["success"])
                        if "success" in attrs
                        else float(np.asarray(group["rewards"][-1]).item()) > 0.5
                    )
                    self.trajectories.append(
                        _TrajectoryMeta(
                            hdf5_path=path_str,
                            traj_key=traj_key,
                            length=int(group["dones"].shape[0]),
                            task_prompt=task_prompt,
                            terminal_reward=float(np.asarray(group["rewards"][-1]).item()),
                            success=success,
                        )
                    )
        if not self.trajectories:
            raise ValueError(f"No trajectories found under {self.hdf5_root}")

        # Anchor grid per trajectory: regular sample_step stride plus the
        # exact terminal frame when the grid does not already contain it.
        self._anchors: list[tuple[int, int]] = []
        for traj_idx, traj in enumerate(self.trajectories):
            anchors = list(range(0, traj.length, self.sample_step))
            if traj.length - 1 not in anchors:
                anchors.append(traj.length - 1)
            self._anchors.extend((traj_idx, anchor) for anchor in anchors)

    # ------------------------------------------------------------------
    # Split helpers.
    # ------------------------------------------------------------------
    @staticmethod
    def load_exclusions_from_manifest(manifest_path: str) -> dict[str, set[str]]:
        """Parse a split manifest into the ``exclude`` mapping.

        Two formats are accepted:

        * the CSV trajectory split shipped with the dataset release
          (``file,traj_key,split,success`` -- rows with ``split == "eval"``
          are excluded; ``file`` is relative to the collection root);
        * a JSON manifest with per-task held-out keys
          (``tasks[].excluded_table_eval_traj_keys`` or
          ``tasks[].eval_traj_keys``), reproducing the released split.
        """
        path = Path(manifest_path)
        if path.suffix == ".csv":
            import csv

            exclude: dict[str, set[str]] = {}
            with path.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    if row["split"] == "eval":
                        exclude.setdefault(row["file"], set()).add(row["traj_key"])
            return exclude

        payload = json.loads(path.read_text())
        exclude = {}
        for task in payload.get("tasks", []):
            hdf5_path = task["hdf5_path"]
            excluded = exclude.setdefault(hdf5_path, set())
            excluded.update(task.get("excluded_table_eval_traj_keys", []))
            excluded.update(task.get("eval_traj_keys", []))
        return exclude

    @staticmethod
    def random_exclusions(hdf5_root: str, holdout_ratio: float, seed: int) -> dict[str, set[str]]:
        """Hold out a random fraction of rollouts per task file."""
        rng = np.random.default_rng(seed)
        exclude: dict[str, set[str]] = {}
        for hdf5_path in sorted(Path(hdf5_root).rglob("*.hdf5")):
            with h5py.File(hdf5_path, "r") as handle:
                traj_keys = [key for key in handle.keys() if key.startswith("traj_")]
            num_holdout = int(round(len(traj_keys) * holdout_ratio))
            if num_holdout > 0:
                chosen = rng.choice(len(traj_keys), size=num_holdout, replace=False)
                exclude[str(hdf5_path)] = {traj_keys[i] for i in chosen}
        return exclude

    # ------------------------------------------------------------------
    # Dataset interface.
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._anchors)

    def __getitem__(self, index: int) -> dict:
        traj_idx, anchor = self._anchors[index]
        traj = self.trajectories[traj_idx]
        group = self._hdf5_file(traj.hdf5_path)[traj.traj_key]

        requested = ordered_history_indices(anchor, self.context_window, self.frame_interval)
        window = [idx for idx in requested if 0 <= idx < traj.length]
        pad = self.context_window - len(window)  # zero-left padding count
        next_index = min(anchor + self.target_horizon, traj.length - 1)
        next_window = window + [next_index]  # x_t^+ = x_t with one appended future frame

        item = {
            "obs": self._read_obs(group, window, pad),
            "next_obs": self._read_obs(group, next_window, pad),
            "task_prompt": traj.task_prompt,
            "done": anchor == traj.length - 1,
            "terminal_reward": traj.terminal_reward,
            "traj_id": f"{Path(traj.hdf5_path).stem}::{traj.traj_key}",
        }
        return item

    def _read_obs(self, group, indices: list[int], pad: int) -> dict:
        """Read the frames at valid ``indices`` and prepend ``pad`` zero frames."""
        if pad < 0:
            raise ValueError("More valid frames than the context window allows")

        def read(key: str) -> np.ndarray:
            # h5py requires strictly increasing indices, and the clamped
            # terminal bootstrap repeats the anchor frame -- dedup and reorder.
            unique = sorted(set(indices))
            position = {index: slot for slot, index in enumerate(unique)}
            frames = frames_to_uint8(group[key][unique])[[position[index] for index in indices]]
            if pad:
                zeros = np.zeros((pad, *frames.shape[1:]), dtype=frames.dtype)
                frames = np.concatenate([zeros, frames], axis=0)
            return frames

        return {
            "main_images": read("obs_main_images"),
            "wrist_images": read("obs_wrist_images"),
        }

    def _hdf5_file(self, path: str) -> h5py.File:
        # Open lazily so each DataLoader worker gets its own handle.
        handle = self._files.get(path)
        if handle is None:
            handle = h5py.File(path, "r", swmr=True, libver="latest")
            self._files[path] = handle
        return handle

    def statistics(self) -> dict:
        successes = sum(traj.success for traj in self.trajectories)
        return {
            "num_trajectories": len(self.trajectories),
            "num_successes": successes,
            "num_failures": len(self.trajectories) - successes,
            "num_td_tuples": len(self._anchors),
        }


def collate_td_tuples(batch: list[dict]) -> dict:
    """Collate TD tuples into model-ready batches.

    Images are stacked into tensors of shape ``(B, w, H, W, 3)``; the model
    converts them to PIL images internally.
    """
    def stack(field: str, key: str) -> torch.Tensor:
        return torch.from_numpy(np.stack([item[field][key] for item in batch]))

    return {
        "obs": {
            "main_images": stack("obs", "main_images"),
            "wrist_images": stack("obs", "wrist_images"),
        },
        "next_obs": {
            "main_images": stack("next_obs", "main_images"),
            "wrist_images": stack("next_obs", "wrist_images"),
        },
        "task_prompt": [item["task_prompt"] for item in batch],
        "done": torch.tensor([item["done"] for item in batch], dtype=torch.bool),
        "terminal_reward": torch.tensor([item["terminal_reward"] for item in batch], dtype=torch.float32),
        "traj_id": [item["traj_id"] for item in batch],
    }
