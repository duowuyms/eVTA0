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

"""Reward models for embodied RL, loaded lazily by model type."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from functools import cache
from importlib import import_module

from rlinf.models.embodiment.reward.base_reward_model import BaseRewardModel

_MODEL_SPECS = {
    "resnet": (
        "rlinf.models.embodiment.reward.resnet_reward_model",
        "ResNetRewardModel",
    ),
    "evta0": (
        "rlinf.models.embodiment.reward.evta0_reward_model",
        "EVTA0RewardModel",
    ),
    "vlm": (
        "rlinf.models.embodiment.reward.vlm_reward_model",
        "VLMRewardModel",
    ),
    "history_vlm": (
        "rlinf.models.embodiment.reward.vlm_reward_model",
        "HistoryVLMRewardModel",
    ),
}
_CLASS_TO_TYPE = {
    class_name: model_type for model_type, (_, class_name) in _MODEL_SPECS.items()
}

__all__ = ["BaseRewardModel", *_CLASS_TO_TYPE]


@cache
def _load_reward_model(reward_model_type: str) -> type[BaseRewardModel]:
    try:
        module_name, class_name = _MODEL_SPECS[reward_model_type]
    except KeyError as error:
        raise ValueError(f"Unsupported reward model type: {reward_model_type}") from error
    return getattr(import_module(module_name), class_name)


class _LazyRewardModelRegistry(Mapping[str, type[BaseRewardModel]]):
    """Mapping-compatible registry that imports only the requested backend."""

    def __getitem__(self, key: str) -> type[BaseRewardModel]:
        if key not in _MODEL_SPECS:
            raise KeyError(key)
        return _load_reward_model(key)

    def __iter__(self) -> Iterator[str]:
        return iter(_MODEL_SPECS)

    def __len__(self) -> int:
        return len(_MODEL_SPECS)


reward_model_registry: Mapping[str, type[BaseRewardModel]] = _LazyRewardModelRegistry()


def __getattr__(name: str):
    """Preserve direct class imports without eagerly importing every backend."""

    if name in _CLASS_TO_TYPE:
        return _load_reward_model(_CLASS_TO_TYPE[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_reward_model_class(reward_model_type: str) -> type[BaseRewardModel]:
    return _load_reward_model(reward_model_type)
