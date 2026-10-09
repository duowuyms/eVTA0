# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""eVTA0 reward adapter for RLinf's embodied reward worker.

Bridges the released eVTA0 success-probability model (the ``evta0`` package
shipped alongside this overlay) to RLinf's ``BaseRewardModel`` interface, so
a frozen eVTA0 checkpoint can drive policy RL.  The adapter is inference
only; rewards are combined into per-chunk training signals by
``EnvWorker`` (``reward_output_transform`` / ``terminal_success_anchor``).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import torch
from omegaconf import DictConfig

from rlinf.config import torch_dtype_from_precision
from rlinf.models.embodiment.reward.base_reward_model import BaseRewardModel


class EVTA0RewardModel(BaseRewardModel):
    """Run a frozen eVTA0 checkpoint from RLinf's reward worker."""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.checkpoint_path = Path(cfg.checkpoint_path)
        self.evta0_package_path = cfg.get("evta0_package_path", None)
        self.history_buffer_name = cfg.get("history_buffer_name", None)
        self.infer_micro_batch_size = int(cfg.get("infer_micro_batch_size", 4))
        self.interval_reward = float(cfg.get("interval_reward", 0.0))

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"eVTA0 checkpoint not found: {self.checkpoint_path}")

        checkpoint = torch.load(
            self.checkpoint_path, map_location="cpu", weights_only=False
        )
        self.checkpoint_state = checkpoint.get("model_state", checkpoint)
        state = self.checkpoint_state

        model = self._build_model()
        model.load_checkpoint_state(state)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)

        # Keep the eVTA0 module out of this adapter's child-module tree. The
        # reward worker calls adapter.to(device); the eVTA0 model manages its
        # own device placement and must not be recursively moved.
        object.__setattr__(self, "_evta0_model", model)

    # ------------------------------------------------------------------
    # BaseRewardModel interface.
    # ------------------------------------------------------------------
    def forward(
        self, input_data: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> dict[str, Any]:
        raise NotImplementedError("EVTA0RewardModel is an inference-time reward adapter.")

    def train(self, mode: bool = True):
        super().train(False)
        if hasattr(self, "_evta0_model"):
            self._evta0_model.eval()
        return self

    def to(self, *args, **kwargs):
        return self

    # ------------------------------------------------------------------
    # Model construction.
    # ------------------------------------------------------------------
    def _build_model(self):
        if self.evta0_package_path and str(self.evta0_package_path) not in sys.path:
            sys.path.insert(0, str(self.evta0_package_path))
        try:
            from evta0.model import EVTA0
        except ImportError as error:
            raise ImportError(
                "The evta0 package could not be imported. Install it, put it on "
                "PYTHONPATH, or set reward.model.evta0_package_path."
            ) from error

        state = self.checkpoint_state
        dtype = torch_dtype_from_precision(self.cfg.get("precision", "bf16"))
        device_map = self._resolve_device_map(self.cfg.get("device_map", "current_device"))

        return EVTA0(
            pretrained_path=self.cfg.get("model_path", state.get("pretrained_path")),
            goal_prompt=state.get("goal_prompt"),
            torch_dtype=dtype,
            device_map=device_map,
            lora_rank=int(self.cfg.get("lora_rank", 16)),
            lora_alpha=int(self.cfg.get("lora_alpha", 32)),
            lora_dropout=float(self.cfg.get("lora_dropout", 0.05)),
            context_window=int(self.cfg.get("context_window", 5)),
            frame_interval=int(self.cfg.get("frame_interval", 2)),
            target_horizon=int(self.cfg.get("target_horizon", 2)),
            padding_policy=str(self.cfg.get("padding_policy", "zero_left")),
        )

    def _resolve_device_map(self, device_map_cfg: Any) -> Any:
        if device_map_cfg != "current_device":
            return device_map_cfg
        if torch.cuda.is_available():
            return {"": torch.cuda.current_device()}
        return {"": "cpu"}

    # ------------------------------------------------------------------
    # Reward computation.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def compute_reward(self, reward_input: dict[str, Any]) -> torch.Tensor:
        """Score every environment's history window; returns P(success) in [0, 1]."""
        history_input = reward_input.get("history_input", {})
        input_batch_size = self._infer_input_batch_size(reward_input, history_input)
        if input_batch_size == 0:
            return torch.empty(0, dtype=torch.float32)
        reward_valid_mask = reward_input.get("reward_valid_mask")
        if reward_valid_mask is not None:
            reward_valid_mask = torch.as_tensor(reward_valid_mask, dtype=torch.bool).reshape(-1)
            if reward_valid_mask.numel() != input_batch_size:
                raise ValueError(
                    "reward_valid_mask must contain one item per eVTA0 input: "
                    f"expected {input_batch_size}, got {reward_valid_mask.numel()}"
                )

        infer_micro_batch_size = self.infer_micro_batch_size or input_batch_size
        reward_chunks: list[torch.Tensor] = []
        for start in range(0, input_batch_size, infer_micro_batch_size):
            end = min(start + infer_micro_batch_size, input_batch_size)
            reward_chunk = torch.full(
                (end - start,), fill_value=self.interval_reward, dtype=torch.float32
            )
            batch, valid_local_ids = self._build_model_batch(
                reward_input, history_input, start, end,
                reward_valid_mask=reward_valid_mask,
            )
            if valid_local_ids:
                probabilities = self._evta0_model.predict(batch)
                reward_chunk[torch.as_tensor(valid_local_ids, dtype=torch.long)] = (
                    probabilities.detach().float().cpu().view(-1)
                )
            reward_chunks.append(reward_chunk)
        return torch.cat(reward_chunks, dim=0)

    # ------------------------------------------------------------------
    # History plumbing.
    # ------------------------------------------------------------------
    def _select_history_buffer(
        self, history_input: dict[str, dict[str, list[list[Any]]]]
    ) -> dict[str, list[list[Any]]]:
        if self.history_buffer_name:
            return history_input.get(self.history_buffer_name, {})
        if not history_input:
            return {}
        return next(iter(history_input.values()))

    def _infer_input_batch_size(
        self,
        reward_input: dict[str, Any],
        history_input: dict[str, dict[str, list[list[Any]]]],
    ) -> int:
        history_buffer = self._select_history_buffer(history_input)
        for env_sequences in history_buffer.values():
            return len(env_sequences)

        for key in ("main_images", "wrist_images", "states", "dones"):
            value = reward_input.get(key)
            if isinstance(value, torch.Tensor):
                return int(value.shape[0])
            if isinstance(value, (list, tuple)):
                return len(value)
        task_descriptions = reward_input.get("task_descriptions")
        return len(task_descriptions) if isinstance(task_descriptions, list) else 0

    def _build_model_batch(
        self,
        reward_input: dict[str, Any],
        history_input: dict[str, dict[str, list[list[Any]]]],
        start: int,
        end: int,
        reward_valid_mask: torch.Tensor | None = None,
    ) -> tuple[dict[str, Any], list[int]]:
        history_buffer = self._select_history_buffer(history_input)
        main_histories = history_buffer.get("main_images", [])
        wrist_histories = history_buffer.get("wrist_images", [])
        task_histories = history_buffer.get("task_descriptions", [])

        main_batches: list[list[Any]] = []
        wrist_batches: list[list[Any]] = []
        task_prompts: list[str] = []
        valid_local_ids: list[int] = []

        for global_idx in range(start, end):
            if reward_valid_mask is not None and not bool(reward_valid_mask[global_idx]):
                continue
            main_seq = self._sequence_at(main_histories, global_idx)
            if not main_seq:
                continue
            wrist_seq = self._sequence_at(wrist_histories, global_idx)
            task_seq = self._sequence_at(task_histories, global_idx)
            prompt = self._resolve_task_prompt(reward_input, task_seq, global_idx)

            valid_local_ids.append(global_idx - start)
            main_batches.append(main_seq)
            wrist_batches.append(wrist_seq)
            task_prompts.append(prompt)

        batch = {
            "obs": {
                "main_images": main_batches,
                "wrist_images": wrist_batches,
            },
            "task_prompt": task_prompts,
        }
        return batch, valid_local_ids

    def _sequence_at(self, sequences: Any, index: int) -> list[Any]:
        if isinstance(sequences, (list, tuple)) and index < len(sequences):
            value = sequences[index]
            if isinstance(value, list):
                return value
            if isinstance(value, tuple):
                return list(value)
            if value is None:
                return []
            return [value]
        return []

    def _resolve_task_prompt(
        self,
        reward_input: dict[str, Any],
        task_seq: list[Any],
        env_idx: int,
    ) -> str:
        for value in reversed(task_seq):
            if isinstance(value, str) and value:
                return value

        task_descriptions = reward_input.get("task_descriptions")
        if isinstance(task_descriptions, list) and env_idx < len(task_descriptions):
            prompt = task_descriptions[env_idx]
            if isinstance(prompt, str):
                return prompt

        if getattr(self._evta0_model, "task_prompts", None):
            return self._evta0_model.task_prompts[0]
        raise ValueError("Unable to resolve task prompt for EVTA0RewardModel.")
