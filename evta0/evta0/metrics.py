"""Evaluation metrics: VOC, VROC, MSE, and Kendall tau-a.

Definitions (matching the released evaluation protocol):

* **VOC** (value-ordering correlation): on an expert demonstration, the
  predicted rewards should increase with task progress.  VOC is the
  tie-aware Spearman rank correlation between the anchor-ordered predictions
  and the timeline ``[0, 1, ..., n-1]``.  A constant prediction sequence has
  no ordering and scores 0.

* **VROC** (value-ordering correlation, reversed): the demonstration is fed
  to the model in reversed temporal order (each window looks "backwards" in
  original time).  A well-behaved success-probability model produces
  decreasing rewards along this reversed presentation; VROC is the same
  tie-aware Spearman formula with the predictions listed in reversed anchor
  order and correlated against ``[n-1, ..., 1, 0]``.

* **MSE**: mean squared error between the terminal reward prediction and the
  true terminal outcome (soft target) over a set of rollouts.

* **Kendall tau-a**: per task, every (success, failure) pair of terminal
  predictions contributes +1 (concordant) when the successful rollout scores
  higher, -1 (discordant) when lower, and 0 for ties.
  ``tau_a = (concordant - discordant) / num_pairs`` (ties stay in the
  denominator).  Per-task values are macro-averaged over the tasks that
  contain both outcomes.
"""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import fmean

import numpy as np


def _average_tie_ranks(values: list[float]) -> list[float]:
    """Ranks with ties assigned their average rank (1-based)."""
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = 0.5 * ((start + 1) + end)
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float:
    left_mean, right_mean = fmean(left), fmean(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    denominator = math.sqrt(
        sum(value * value for value in left_centered)
        * sum(value * value for value in right_centered)
    )
    if denominator == 0.0:
        return 0.0
    value = sum(
        l * r for l, r in zip(left_centered, right_centered)
    ) / denominator
    if not math.isfinite(value):
        raise ValueError("non-finite Spearman correlation")
    return float(value)


def _tie_aware_spearman(values: list[float], target: list[float]) -> float:
    if len(values) != len(target) or len(values) < 2:
        return float("nan")
    if not all(math.isfinite(value) for value in values):
        return float("nan")
    if max(values) - min(values) <= 1e-12:
        return 0.0  # a constant prediction has no ordering
    return _pearson(_average_tie_ranks(values), _average_tie_ranks(target))


def voc(pred_rewards) -> float:
    """Value-ordering correlation of predictions along a demonstration.

    Args:
        pred_rewards: predictions at anchors [t_0 < t_1 < ... < t_{n-1}].
    """
    pred_rewards = [float(value) for value in pred_rewards]
    return _tie_aware_spearman(pred_rewards, list(range(len(pred_rewards))))


def vroc(reverse_pred_rewards) -> float:
    """VOC of predictions along a temporally reversed demonstration.

    Args:
        reverse_pred_rewards: predictions produced while the model consumed
            the reversed trajectory, listed in descending anchor order
            (matching ``eval_progress``'s output order).
    """
    reverse_pred_rewards = [float(value) for value in reverse_pred_rewards]
    n = len(reverse_pred_rewards)
    return _tie_aware_spearman(reverse_pred_rewards, list(reversed(range(n))))


def mse(predictions, targets) -> float:
    """Mean squared error between terminal predictions and outcomes."""
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    return float(np.mean((predictions - targets) ** 2))


def kendall_tau_a(rows: list[dict]) -> dict:
    """Within-task success-vs-failure Kendall tau-a with task macro-average.

    Args:
        rows: list of dicts with keys ``task`` (grouping id), ``success``
            (bool), and ``prediction`` (float terminal reward).

    Returns:
        dict with per-task tau-a values and their macro average.
    """
    grouped: dict = defaultdict(lambda: {"success": [], "failure": []})
    for row in rows:
        outcomes = grouped[row["task"]]
        outcomes["success" if row["success"] else "failure"].append(float(row["prediction"]))

    per_task: dict = {}
    for task in sorted(grouped, key=str):
        successes = grouped[task]["success"]
        failures = grouped[task]["failure"]
        if not successes or not failures:
            per_task[task] = None  # tau-a undefined without both outcomes
            continue
        concordant = discordant = 0
        for success_score in successes:
            for failure_score in failures:
                if success_score > failure_score:
                    concordant += 1
                elif success_score < failure_score:
                    discordant += 1
        pairs = len(successes) * len(failures)
        per_task[task] = (concordant - discordant) / pairs

    valid = [value for value in per_task.values() if value is not None]
    return {
        "per_task": per_task,
        "num_valid_tasks": len(valid),
        "macro_average": float(np.mean(valid)) if valid else float("nan"),
    }


def summarize_trajectory_metrics(trajectory_metrics: list[dict]) -> dict:
    """Aggregate per-trajectory VOC/VROC rows (skipping NaN rows)."""
    valid = [
        row
        for row in trajectory_metrics
        if np.isfinite(row.get("voc", np.nan)) and np.isfinite(row.get("vroc", np.nan))
    ]
    vocs = np.asarray([row["voc"] for row in valid], dtype=np.float64)
    vrocs = np.asarray([row["vroc"] for row in valid], dtype=np.float64)
    return {
        "mean_voc": float(vocs.mean()) if vocs.size else float("nan"),
        "std_voc": float(vocs.std()) if vocs.size else float("nan"),
        "mean_vroc": float(vrocs.mean()) if vrocs.size else float("nan"),
        "std_vroc": float(vrocs.std()) if vrocs.size else float("nan"),
        "num_total_trajectories": len(trajectory_metrics),
        "num_valid_trajectories": len(valid),
    }
