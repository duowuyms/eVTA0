"""Unit tests for terminal-anchored reward combination and history masking.

Run from an RLinf checkout with the overlay applied:
    python -m pytest tests/test_terminal_anchor.py -q
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rlinf.envs.libero.libero_env import LiberoEnv
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.env.history_manager import HistoryManager


class _EnvOutput:
    def __init__(self, rewards, terminations, masks):
        self.rewards = rewards
        self.terminations = terminations
        self.dones = None
        self.truncations = None
        self.env_infos = masks


def _make_env_worker(transform: str, reward_weight: float = 1.0) -> EnvWorker:
    worker = object.__new__(EnvWorker)
    worker.reward_output_transform = transform
    worker.reward_weight = reward_weight
    worker.terminal_success_anchor = True
    return worker


def _masks(valid, success_now, success_before):
    return {
        "reward_valid_mask": torch.tensor(valid, dtype=torch.bool),
        "success_now": torch.tensor(success_now, dtype=torch.bool),
        "success_before": torch.tensor(success_before, dtype=torch.bool),
    }


def test_reward_masks_from_chunk_boundaries():
    success_before = torch.tensor([False, True, False])
    terminations = torch.tensor(
        [[False, True, False], [False, False, False], [False, False, False]]
    )
    truncations = torch.zeros_like(terminations)
    past_term, past_trunc, success_now, already, valid = LiberoEnv._build_reward_masks(
        success_before, terminations, truncations
    )
    assert past_term.tolist() == [True, False, False]
    assert success_now.tolist() == [True, False, False]
    assert already.tolist() == [False, True, False]
    assert valid.tolist() == [False, False, True]  # failure truncation stays valid


def test_terminal_anchor_probability_minus_one():
    worker = _make_env_worker("probability_minus_one")
    # rewards: [num_envs=3, chunk_steps=4]; env0 succeeds at step 1 of chunk 0.
    rewards = torch.zeros(3, 4)
    rewards[0, 0] = 1.0
    terminations = torch.zeros(3, 4, dtype=torch.bool)
    terminations[0, 0] = True
    masks = _masks(
        valid=[False, False, True],
        success_now=[True, False, False],
        success_before=[False, True, False],
    )
    # The train loop applies transform_reward_model_output (p - 1) before
    # compute_bootstrap_rewards, so the anchored combination receives the
    # transformed values already.
    model_output = torch.tensor([[0.5], [0.5], [0.7 - 1.0]])

    adjusted = worker._compute_terminal_anchored_rewards(
        _EnvOutput(rewards, terminations, masks), rewards, model_output
    )
    # env0 (first success): terminal p=1 anchor -> r = 0 everywhere.
    assert adjusted[0].abs().sum() == 0
    # env1 (already successful): exactly zero.
    assert adjusted[1].abs().sum() == 0
    # env2 (valid): every action in the chunk gets p - 1.
    assert torch.allclose(adjusted[2], torch.full((4,), 0.7 - 1.0))


def test_terminal_anchor_requires_model_output_on_valid_chunks():
    worker = _make_env_worker("probability_minus_one")
    rewards = torch.zeros(1, 4)
    masks = _masks(valid=[True], success_now=[False], success_before=[False])
    with pytest.raises(RuntimeError):
        worker._compute_terminal_anchored_rewards(
            _EnvOutput(rewards, None, masks), rewards, None
        )


def test_transform_probability_minus_one():
    worker = _make_env_worker("probability_minus_one")
    probabilities = torch.tensor([0.0, 0.5, 1.0])
    assert torch.allclose(
        worker.transform_reward_model_output(probabilities), torch.tensor([-1.0, -0.5, 0.0])
    )
    with pytest.raises(RuntimeError):
        worker.transform_reward_model_output(torch.tensor([1.5]))


def test_history_manager_valid_mask():
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "model": {
                "history_buffers": {
                    "evta0_context": {
                        "history_size": 5,
                        "history_keys": ["main_images"],
                    }
                }
            }
        }
    )
    manager = HistoryManager(cfg, num_envs=2)
    frames = {"main_images": torch.zeros(2, 3, 4, 4)}
    manager.append_to_history_entries(frames, valid_mask=torch.tensor([True, False]))
    manager.append_to_history_entries(frames, valid_mask=torch.tensor([True, False]))
    assert len(manager.history_entries[0]) == 2
    assert len(manager.history_entries[1]) == 0  # invalid env skipped
    with pytest.raises(ValueError):
        manager.append_to_history_entries(frames, valid_mask=torch.tensor([True]))
