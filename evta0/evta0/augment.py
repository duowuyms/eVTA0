"""Data augmentation: soft terminal targets for failed rollouts.

Training data comes as mixed-quality rollouts with binary terminal outcomes.
A hard 0 target for every failure ignores that some failures come close to
succeeding, so eVTA0 augments the data with *soft* failure targets derived
from a frozen pretrained world model (see the README):

1. **Embed.**  Each trajectory's main-camera frames, uniformly sampled to 64
   per clip, are passed through V-JEPA2; the feature is the mean of the last
   hidden state over time (no L2 normalization).
2. **Cluster successes.**  For every task, the embeddings of up to
   ``reference_episodes_per_task`` successful trajectories are standardized
   and clustered with DBSCAN (eps 0.5, min_samples 2, euclidean).  Each
   cluster's center is the mean of its members mapped back to the original
   space; if DBSCAN finds no cluster, the global mean is used.
3. **Distance.**  A failed rollout's distance is its euclidean distance to
   the nearest success center.
4. **Soft target.**  Distances are min-max normalized within the task's
   failure group and mapped through a sigmoid:

       target = 0.6 * sigmoid(10 * (0.5 - normalized_distance))

   Successful rollouts keep target 1.0; failures without a valid embedding or
   without a success reference keep target 0.0.

The augmented collection is written to a copy of the input tree with
``rewards[-1]`` replaced by the soft target and ``dones[-1]`` forced True.

Usage::

    python -m evta0.augment \
        --input-root /path/to/raw_rollout_hdf5 \
        --output-root /path/to/augmented_hdf5 \
        [--reference-root /path/to/lerobot_demos] \
        [--reference-episodes-per-task 10]

Success references come from the input collection itself by default; pass
``--reference-root`` (a LeRobot dataset directory with ``meta/`` and
``data/chunk-*``) to use external success demonstrations instead.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np
from scipy import special
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from evta0.data import frames_to_uint8, task_name_from_path

DEFAULT_VJEPA2 = "facebook/vjepa2-vitg-fpc64-384"

FRAMES_PER_CLIP = 64
DBSCAN_EPS = 0.5
DBSCAN_MIN_SAMPLES = 2
DISTANCE_METRIC = "euclidean"
SIGMOID_SCALE = 0.6
SIGMOID_STEEPNESS = 10.0
SIGMOID_OFFSET = 0.5
AUGMENTATION_METHOD = "vjepa2_dbscan_euclidean_sigmoid_soft_terminal_v1"


# ----------------------------------------------------------------------
# Clustering and target mapping (pure numpy, easy to test).
# ----------------------------------------------------------------------
def success_cluster_centers(
    embeddings: np.ndarray,
    *,
    eps: float = DBSCAN_EPS,
    min_samples: int = DBSCAN_MIN_SAMPLES,
) -> tuple[np.ndarray, dict]:
    """DBSCAN cluster centers of success embeddings, in the original space."""
    embeddings = np.asarray(embeddings, dtype=np.float64)
    if len(embeddings) == 0:
        return np.zeros((0, 0)), {"num_success_embeddings": 0, "num_centers": 0,
                                  "fallback_to_mean": False, "dbscan_noise_count": 0}

    scaler = StandardScaler()
    scaled = scaler.fit_transform(embeddings)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(scaled).labels_

    centers = []
    for label in sorted(set(labels.tolist()) - {-1}):
        cluster_points = scaled[labels == label]
        centers.append(scaler.inverse_transform(cluster_points.mean(axis=0, keepdims=True)).flatten())

    fallback_to_mean = False
    if not centers:
        centers = [embeddings.mean(axis=0)]
        fallback_to_mean = True

    metadata = {
        "num_success_embeddings": int(len(embeddings)),
        "num_centers": int(len(centers)),
        "fallback_to_mean": bool(fallback_to_mean),
        "dbscan_noise_count": int(np.sum(labels == -1)),
        "dbscan_label_counts": {str(label): int(count)
                                for label, count in sorted(Counter(int(l) for l in labels).items())},
    }
    return np.asarray(centers, dtype=np.float64), metadata


def soft_targets_from_distances(min_distances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Min-max normalize distances and map them through the sigmoid.

    Returns ``(targets, normalized_distances)``.  A constant distance group
    normalizes to 0.5 everywhere.
    """
    min_distances = np.asarray(min_distances, dtype=np.float64)
    dist_min = float(min_distances.min())
    dist_range = float(min_distances.max()) - dist_min
    if dist_range < 1e-6:
        normalized = np.full_like(min_distances, SIGMOID_OFFSET)
    else:
        normalized = (min_distances - dist_min) / dist_range
    targets = SIGMOID_SCALE * special.expit(SIGMOID_STEEPNESS * (SIGMOID_OFFSET - normalized))
    return targets.astype(np.float64), normalized.astype(np.float64)


# ----------------------------------------------------------------------
# Rollout discovery and input validation.
# ----------------------------------------------------------------------
def iter_rollouts(hdf5_root: str):
    """Yield ``(hdf5_path, traj_key, task_prompt, success)`` for every rollout.

    Success is taken from the group's ``success`` attribute when present
    (augmented collections), otherwise from the raw binary ``rewards[-1]``.
    """
    for hdf5_path in sorted(Path(hdf5_root).rglob("*.hdf5")):
        default_task = task_name_from_path(str(hdf5_path))
        with h5py.File(hdf5_path, "r") as handle:
            for traj_key in sorted(
                (key for key in handle.keys() if key.startswith("traj_")),
                key=lambda key: int(key.split("_")[-1]),
            ):
                group = handle[traj_key]
                if "success" in group.attrs:
                    success = bool(group.attrs["success"])
                else:
                    success = float(np.asarray(group["rewards"][-1]).item()) > 0.5
                task_prompt = str(group.attrs["task_prompt"]) if "task_prompt" in group.attrs else default_task
                yield str(hdf5_path), traj_key, task_prompt, success


def validate_raw_outcomes(rollouts: list[tuple], tolerance: float = 1e-8) -> None:
    """Fail early unless terminal rewards are raw 0/1 with zero non-terminals."""
    for hdf5_path, traj_key, _, success in rollouts:
        with h5py.File(hdf5_path, "r") as handle:
            group = handle[traj_key]
            rewards = np.asarray(group["rewards"][:], dtype=np.float64)
            terminal = float(rewards[-1])
            nonterminal_max = float(np.abs(rewards[:-1]).max()) if len(rewards) > 1 else 0.0
            if abs(terminal - float(success)) > tolerance or nonterminal_max > tolerance:
                raise ValueError(
                    f"{hdf5_path}::{traj_key}: expected raw binary terminal rewards "
                    f"(terminal={terminal}, success={success}); refusing to augment "
                    "an already-augmented collection."
                )


# ----------------------------------------------------------------------
# V-JEPA2 embedding.
# ----------------------------------------------------------------------
def sample_frame_indices(length: int, num_frames: int = FRAMES_PER_CLIP) -> np.ndarray:
    if length <= 0:
        return np.zeros(num_frames, dtype=np.int64)
    return np.linspace(0, length - 1, num_frames).round().astype(np.int64)


class VJEPA2Embedder:
    """Embed a trajectory (list of uint8 frames) with a frozen V-JEPA2."""

    def __init__(self, model_path: str = DEFAULT_VJEPA2, device: str = "auto") -> None:
        import torch
        from transformers import VJEPA2Model, VJEPA2VideoProcessor

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
        self.device = device
        self.processor = VJEPA2VideoProcessor.from_pretrained(model_path)
        self.model = VJEPA2Model.from_pretrained(model_path, torch_dtype=dtype).to(device)
        self.model.eval()

    def embed(self, trajectories: list[np.ndarray], batch_size: int = 4) -> np.ndarray:
        """Embed main-camera frame stacks; returns one feature per trajectory."""
        import torch
        from PIL import Image

        features = []
        with torch.no_grad():
            for start in tqdm(range(0, len(trajectories), batch_size),
                              desc="embedding", unit="batch"):
                batch = trajectories[start : start + batch_size]
                videos = [
                    [Image.fromarray(frame) for frame in frames_to_uint8(trajectory)]
                    for trajectory in batch
                ]
                inputs = self.processor(videos=videos, return_tensors="pt")
                pixel_values = inputs["pixel_values_videos"].to(self.device)
                outputs = self.model(pixel_values_videos=pixel_values, skip_predictor=True)
                features.append(outputs.last_hidden_state.float().mean(dim=1).cpu().numpy())
        return np.concatenate(features, axis=0) if features else np.zeros((0, 1), dtype=np.float32)


def read_main_frames(hdf5_path: str, traj_key: str) -> np.ndarray:
    """Load the uniformly sampled main-camera frames of one rollout."""
    with h5py.File(hdf5_path, "r") as handle:
        images = handle[traj_key]["obs_main_images"]
        indices = sample_frame_indices(images.shape[0])
        return frames_to_uint8(np.asarray(images[indices]))


# ----------------------------------------------------------------------
# Reference videos (external LeRobot demonstrations).
# ----------------------------------------------------------------------
def load_reference_videos(
    reference_root: str,
    episodes_per_task: int,
    wanted_tasks: set[str] | None = None,
) -> dict[str, list[np.ndarray]]:
    """Sample up to N success demonstration clips per task from a LeRobot root.

    When ``wanted_tasks`` is given, only those task prompts are loaded.
    """
    import pandas as pd

    from evta0.eval_progress import decode_image

    root = Path(reference_root)
    parquet_paths = {}
    for chunk_dir in sorted(root.glob("data/chunk-*")):
        for parquet_path in chunk_dir.glob("episode_*.parquet"):
            parquet_paths[int(parquet_path.stem.split("_")[-1])] = parquet_path

    episodes_by_task: dict[str, list[int]] = defaultdict(list)
    with (root / "meta" / "episodes.jsonl").open() as handle:
        for line in handle:
            meta = json.loads(line)
            task = meta["tasks"][0]
            if wanted_tasks is None or task in wanted_tasks:
                episodes_by_task[task].append(int(meta["episode_index"]))

    videos: dict[str, list[np.ndarray]] = {}
    for task_prompt, episode_indices in sorted(episodes_by_task.items()):
        clips = []
        for episode_index in sorted(episode_indices):
            if len(clips) >= episodes_per_task:
                break
            parquet_path = parquet_paths.get(episode_index)
            if parquet_path is None:
                continue
            df = pd.read_parquet(parquet_path)
            image_col = next(
                (name for name in ("observation.images.image", "observation.image", "image")
                 if name in df.columns),
                None,
            )
            if image_col is None:
                continue
            frames = [decode_image(df[image_col].iloc[int(idx)])
                      for idx in sample_frame_indices(len(df))]
            clips.append(np.stack(frames))
        if clips:
            videos[task_prompt] = clips
    return videos


# ----------------------------------------------------------------------
# Augmentation driver.
# ----------------------------------------------------------------------
def compute_soft_targets(
    rollouts: list[dict],
    rollout_features: np.ndarray,
    centers_by_task: dict[str, np.ndarray],
) -> tuple[list[dict], dict]:
    """Assign each rollout a terminal target plus distance diagnostics."""
    invalid = np.all(np.asarray(rollout_features) == 0, axis=1)
    failures_by_task: dict[str, list[int]] = defaultdict(list)
    distances_by_task: dict[str, list[float]] = defaultdict(list)

    for index, rollout in enumerate(rollouts):
        centers = centers_by_task.get(rollout["task_prompt"])
        distance = math.nan
        if not invalid[index] and centers is not None and len(centers) > 0:
            distance = float(
                cdist(rollout_features[index].reshape(1, -1), centers, DISTANCE_METRIC).min()
            )
        rollout["valid_embedding"] = not bool(invalid[index])
        rollout["success_min_distance"] = distance
        rollout["normalized_success_distance"] = math.nan
        rollout["distance_group_size"] = 0
        if not rollout["success"] and rollout["valid_embedding"] and not math.isnan(distance):
            failures_by_task[rollout["task_prompt"]].append(index)
            distances_by_task[rollout["task_prompt"]].append(distance)

    for task, indices in failures_by_task.items():
        targets, normalized = soft_targets_from_distances(np.asarray(distances_by_task[task]))
        for offset, index in enumerate(indices):
            rollouts[index]["augmented_terminal_reward"] = float(targets[offset])
            rollouts[index]["normalized_success_distance"] = float(normalized[offset])
            rollouts[index]["distance_group_size"] = len(indices)  # legacy attr name

    for rollout in rollouts:
        if rollout["success"] and rollout["valid_embedding"]:
            target = 1.0
        elif "augmented_terminal_reward" in rollout:
            target = float(rollout["augmented_terminal_reward"])
        else:
            target = 0.0
        rollout["augmented_terminal_reward"] = target
        rollout["terminal_reward_delta"] = target - float(rollout["original_terminal_reward"])
        rollout["terminal_reward_changed"] = abs(rollout["terminal_reward_delta"]) > 1e-8

    diagnostics = {
        "invalid_embedding_count": int(invalid.sum()),
        "failure_groups": {
            task: {"count": len(values),
                   "min_distance": float(np.min(values)),
                   "max_distance": float(np.max(values))}
            for task, values in sorted(distances_by_task.items()) if values
        },
    }
    return rollouts, diagnostics


def write_augmented_collection(output_root: str, rollouts: list[dict], model_path: str) -> None:
    """Copy the input tree and write soft targets back into the copy."""
    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError(f"output root already exists: {output_root}")

    by_path: dict[str, list[dict]] = defaultdict(list)
    for rollout in rollouts:
        by_path[rollout["hdf5_path"]].append(rollout)

    for hdf5_path in sorted(by_path):
        destination = output_root / Path(hdf5_path).parent.name / Path(hdf5_path).name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(hdf5_path, destination)
        with h5py.File(destination, "r+") as handle:
            handle.attrs["augmentation_method"] = AUGMENTATION_METHOD
            handle.attrs["augmentation_model"] = model_path
            for rollout in by_path[hdf5_path]:
                group = handle[rollout["traj_key"]]
                group["rewards"][-1] = float(rollout["augmented_terminal_reward"])
                group["dones"][-1] = True
                group.attrs["pre_terminal_reward"] = float(rollout["original_terminal_reward"])
                group.attrs["augmented_terminal_reward"] = float(rollout["augmented_terminal_reward"])
                group.attrs["augmentation_method"] = AUGMENTATION_METHOD
                group.attrs["augmentation_model"] = model_path
                group.attrs["success_min_distance"] = float(rollout["success_min_distance"])
                group.attrs["normalized_success_distance"] = float(
                    rollout["normalized_success_distance"]
                )
                group.attrs["distance_group_size"] = int(rollout["distance_group_size"])
                group.attrs["valid_embedding"] = bool(rollout["valid_embedding"])
                group.attrs["terminal_reward_delta"] = float(rollout["terminal_reward_delta"])
                group.attrs["terminal_reward_changed"] = bool(rollout["terminal_reward_changed"])
                group.attrs["success"] = bool(rollout["success"])
                group.attrs["task_prompt"] = rollout["task_prompt"]


def augment(input_root: str, output_root: str, *, reference_root: str | None = None,
            reference_episodes_per_task: int = 10, model_path: str = DEFAULT_VJEPA2,
            device: str = "auto", batch_size: int = 4) -> dict:
    raw_rollouts = list(iter_rollouts(input_root))
    validate_raw_outcomes(raw_rollouts)
    print(f"[augment] {len(raw_rollouts)} raw rollouts under {input_root}")

    rollouts = [
        {"hdf5_path": hdf5_path, "traj_key": traj_key, "task_prompt": task_prompt,
         "success": success, "original_terminal_reward": 1.0 if success else 0.0}
        for hdf5_path, traj_key, task_prompt, success in raw_rollouts
    ]

    embedder = VJEPA2Embedder(model_path=model_path, device=device)
    print("[augment] embedding rollouts ...")
    frames = [read_main_frames(r["hdf5_path"], r["traj_key"]) for r in rollouts]
    rollout_features = embedder.embed(frames, batch_size=batch_size)

    # Success references: external LeRobot demos when given, otherwise the
    # input collection's own successful rollouts.
    if reference_root is not None:
        input_tasks = {rollout["task_prompt"] for rollout in rollouts}
        reference_videos = load_reference_videos(
            reference_root, reference_episodes_per_task, wanted_tasks=input_tasks
        )
        print(f"[augment] embedding {sum(len(v) for v in reference_videos.values())} "
              "external reference clips ...")
        centers_by_task, cluster_metadata = {}, {}
        for task_prompt, clips in sorted(reference_videos.items()):
            features = embedder.embed(clips, batch_size=batch_size)
            centers, metadata = success_cluster_centers(features)
            centers_by_task[task_prompt] = centers
            cluster_metadata[task_prompt] = metadata
    else:
        by_task: dict[str, list[int]] = defaultdict(list)
        for index, rollout in enumerate(rollouts):
            if rollout["success"]:
                by_task[rollout["task_prompt"]].append(index)
        centers_by_task, cluster_metadata = {}, {}
        for task_prompt, indices in sorted(by_task.items()):
            indices = indices[:reference_episodes_per_task]
            centers, metadata = success_cluster_centers(rollout_features[indices])
            centers_by_task[task_prompt] = centers
            cluster_metadata[task_prompt] = metadata

    rollouts, diagnostics = compute_soft_targets(rollouts, rollout_features, centers_by_task)

    print("[augment] writing augmented collection ...")
    write_augmented_collection(output_root, rollouts, model_path)

    report = {
        "augmentation_method": AUGMENTATION_METHOD,
        "augmentation_model": model_path,
        "feature_extractor": {"frames_per_clip": FRAMES_PER_CLIP,
                              "pooling": "mean of last hidden state over time"},
        "success_clustering": {"algorithm": "StandardScaler + DBSCAN",
                               "eps": DBSCAN_EPS, "min_samples": DBSCAN_MIN_SAMPLES,
                               "metric": DISTANCE_METRIC},
        "reward_mapping": ("success=1.0; invalid/no-reference failure=0.0; "
                           f"failure={SIGMOID_SCALE}*sigmoid({SIGMOID_STEEPNESS}*"
                           f"({SIGMOID_OFFSET}-normalized_distance))"),
        "reference_source": reference_root or "input collection successes",
        "reference_episodes_per_task": reference_episodes_per_task,
        "num_rollouts": len(rollouts),
        "num_successes": sum(r["success"] for r in rollouts),
        "num_soft_targets": sum(1 for r in rollouts if r["terminal_reward_changed"]
                                and not r["success"]),
        "cluster_metadata_by_task": cluster_metadata,
        "diagnostics": diagnostics,
        "records": [{key: value for key, value in rollout.items()} for rollout in rollouts],
    }
    Path(output_root).joinpath("augmentation_report.json").write_text(json.dumps(report, indent=2))
    print(f"[augment] done: {report['num_soft_targets']} soft failure targets written "
          f"to {output_root}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True,
                        help="Raw rollout HDF5 collection (binary terminal rewards)")
    parser.add_argument("--output-root", required=True,
                        help="Destination for the augmented copy (must not exist)")
    parser.add_argument("--reference-root", default=None,
                        help="Optional LeRobot demo root providing success references")
    parser.add_argument("--reference-episodes-per-task", type=int, default=10)
    parser.add_argument("--model-path", default=DEFAULT_VJEPA2,
                        help="V-JEPA2 checkpoint (hub id or local path)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    augment(
        args.input_root,
        args.output_root,
        reference_root=args.reference_root,
        reference_episodes_per_task=args.reference_episodes_per_task,
        model_path=args.model_path,
        device=args.device,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
