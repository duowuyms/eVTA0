"""Unit tests for the eVTA0 history-window construction in the RLinf overlay.

Run from an RLinf checkout with the overlay applied:
    python -m pytest tests/test_evta0_window.py -q
"""

from __future__ import annotations

import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.workers.env.env_worker import (
    EnvWorker,
    _ordered_raw_action_frame_indices,
    _zero_like_reward_history_item,
)


def _make_env_worker(reward_cfg: dict) -> EnvWorker:
    """Build an EnvWorker with only the attributes the window builder needs."""
    worker = object.__new__(EnvWorker)
    worker.cfg = OmegaConf.create({"reward": reward_cfg})
    worker.train_num_envs_per_stage = 2
    return worker


def test_window_indices_match_evta0_protocol():
    """The RLinf window must equal the evta0 package's ordered_history_indices."""
    try:
        from evta0.protocol import ordered_history_indices
    except ImportError:
        # Reference implementation inline when the evta0 package is absent.
        def ordered_history_indices(anchor, w, s, *, reverse=False):
            direction = 1 if reverse else -1
            return [anchor + direction * o * s for o in range(w - 1, -1, -1)]

    for num_frames in (1, 2, 5, 9, 10, 21):
        for context_window in (1, 3, 5):
            for frame_interval in (1, 2, 4):
                selected = _ordered_raw_action_frame_indices(
                    num_frames, context_window, frame_interval
                )
                expected = ordered_history_indices(
                    num_frames - 1, context_window, frame_interval
                )
                assert selected == [
                    index if index >= 0 else None for index in expected
                ]

    # W5/F2 over a ten-frame chunk: [1, 3, 5, 7, 9] == [t-8, t-6, t-4, t-2, t].
    assert _ordered_raw_action_frame_indices(10, 5, 2) == [1, 3, 5, 7, 9]
    # Short episode: negative indices become explicit left padding.
    assert _ordered_raw_action_frame_indices(2, 5, 2) == [None, None, None, None, 1]


def test_zero_like_padding():
    assert torch.equal(
        _zero_like_reward_history_item(torch.ones(3, 4)), torch.zeros(3, 4)
    )
    assert _zero_like_reward_history_item(np.ones((2, 2), dtype=np.uint8)).sum() == 0
    assert _zero_like_reward_history_item("task") == ""
    assert _zero_like_reward_history_item({"a": 1.5}) == {"a": 0.0}


def test_raw_action_frame_history_input():
    worker = _make_env_worker(
        {
            "model": {
                "history_source": "raw_action_frames",
                "context_window": 5,
                "frame_interval": 2,
                "padding_policy": "zero_left",
                "history_buffer_name": "evta0_context",
                "history_buffers": {
                    "evta0_context": {
                        "history_keys": ["main_images", "task_descriptions"],
                    }
                },
            }
        }
    )
    # Two environments; per-env frame tensors carry the frame id so the
    # selected window is observable.
    frames = []
    for frame_id in range(10):
        main = torch.full((2, 3, 8, 8), float(frame_id))
        frames.append(
            {"main_images": main, "task_descriptions": ["put the bowl on the plate"] * 2}
        )
    worker._raw_action_observation_chunks = [frames]
    observations = frames[-1]

    history_input, history_lengths = worker._build_raw_action_frame_history_input(
        stage_id=0, observations=observations
    )
    buffer = history_input["evta0_context"]
    assert history_lengths["evta0_context"] == [5, 5]
    for env_idx in range(2):
        window = buffer["main_images"][env_idx]
        assert [int(frame[0, 0, 0]) for frame in window] == [1, 3, 5, 7, 9]
        assert buffer["task_descriptions"][env_idx] == ["put the bowl on the plate"] * 5

    # Episode start: the pre-history frames are zero-left padded.
    worker._raw_action_observation_chunks = [frames[:2]]
    history_input, _ = worker._build_raw_action_frame_history_input(
        stage_id=0, observations=frames[1]
    )
    window = history_input["evta0_context"]["main_images"][0]
    values = [int(frame[0, 0, 0]) for frame in window]
    assert values == [0, 0, 0, 0, 1]
    assert window[0].abs().sum() == 0  # zero image, not frame 0
