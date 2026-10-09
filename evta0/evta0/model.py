"""eVTA0 model: a multimodal success-probability reward model.

Architecture::

    x_t   = [I, l, O_t, [REWARD]]           # instruction, task, w frames, reward token
    h_t   = F_psi(x_t)_[REWARD]             # Qwen3-VL hidden state of the reward token
    p_t   = sigmoid(g_phi(h_t))             # single lightweight head, output in [0, 1]

``F_psi`` is a Qwen3-VL backbone fine-tuned with LoRA.  ``g_phi`` is a single
linear layer.  The reward-token embedding is learned.  Everything except the
LoRA adapters, the reward-token embedding, and the head is frozen.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

import huggingface_hub

# Some transformers versions still import this legacy top-level symbol from
# `huggingface_hub`.
if not hasattr(huggingface_hub, "is_offline_mode"):
    def _hf_is_offline_mode() -> bool:
        return bool(getattr(huggingface_hub.constants, "HF_HUB_OFFLINE", False))

    huggingface_hub.is_offline_mode = _hf_is_offline_mode

from transformers import AutoProcessor

try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:  # very old transformers
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLForConditionalGeneration,
    )

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

from evta0.protocol import (
    GOAL_PROMPT,
    PADDING_POLICY,
    PROMPT_VERSION,
    PROTOCOL_VERSION,
    REWARD_TOKEN,
    goal_prompt_sha256,
)


class RewardTokenEmbeddingAdapter(nn.Module):
    """Wrap the token embedding so the reward-token vector becomes trainable."""

    def __init__(self, base_embedding: nn.Module, reward_token_id: int) -> None:
        super().__init__()
        self.base_embedding = base_embedding
        self.reward_token_id = int(reward_token_id)
        with torch.no_grad():
            init_vector = base_embedding.weight[self.reward_token_id].detach().float().clone()
        self.reward_embedding = nn.Parameter(init_vector)

    @property
    def weight(self) -> torch.Tensor:
        return self.base_embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeds = self.base_embedding(input_ids)
        reward_mask = input_ids.eq(self.reward_token_id)
        if reward_mask.any():
            reward_vector = self.reward_embedding.to(device=embeds.device, dtype=embeds.dtype)
            embeds = torch.where(
                reward_mask.unsqueeze(-1),
                reward_vector.view(1, 1, -1),
                embeds,
            )
        return embeds


class RewardHead(nn.Module):
    """The single head g_phi: a linear map followed by a sigmoid."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid())

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_states).squeeze(-1)


class EVTA0(nn.Module):
    """Qwen3-VL backbone + LoRA + learned reward token + single reward head."""

    # The backbone manages its own device placement via `device_map`.
    supports_manual_device_move = False

    def __init__(
        self,
        pretrained_path: str = "Qwen/Qwen3-VL-4B-Instruct",
        goal_prompt: str = GOAL_PROMPT,
        torch_dtype: torch.dtype | str = torch.bfloat16,
        device_map: str | dict[str, Any] = "auto",
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        context_window: int = 5,
        frame_interval: int = 2,
        target_horizon: int = 2,
        reward_token: str = REWARD_TOKEN,
        padding_policy: str = PADDING_POLICY,
        protocol_version: str = PROTOCOL_VERSION,
        prompt_version: str = PROMPT_VERSION,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if padding_policy != "zero_left":
            raise ValueError("Only zero_left padding is supported by the eVTA0 protocol.")

        self.pretrained_path = str(pretrained_path)
        self.goal_prompt = goal_prompt
        self.context_window = int(context_window)
        self.frame_interval = int(frame_interval)
        self.target_horizon = int(target_horizon)
        if self.context_window < 1 or self.frame_interval < 1 or self.target_horizon < 1:
            raise ValueError("context_window / frame_interval / target_horizon must be >= 1")
        self.protocol_version = str(protocol_version)
        self.prompt_version = str(prompt_version)
        self.padding_policy = str(padding_policy)
        self.reward_token = reward_token

        # ------------------------------------------------------------------
        # Tokenizer / processor and the special reward token.
        # ------------------------------------------------------------------
        self.processor = AutoProcessor.from_pretrained(self.pretrained_path, trust_remote_code=True)
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        if self.tokenizer.add_special_tokens({"additional_special_tokens": [self.reward_token]}) < 0:
            raise RuntimeError("Failed to register the reward token.")

        # ------------------------------------------------------------------
        # Backbone with LoRA adapters.
        # ------------------------------------------------------------------
        self.vlm = Qwen3VLForConditionalGeneration.from_pretrained(
            self.pretrained_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        self.vlm.resize_token_embeddings(len(self.tokenizer))
        self.reward_token_id = self.tokenizer.convert_tokens_to_ids(self.reward_token)
        self._install_reward_token_adapter()

        if gradient_checkpointing:
            self.vlm.gradient_checkpointing_enable()
        for param in self.vlm.parameters():
            param.requires_grad = False

        self.vlm = get_peft_model(
            self.vlm,
            LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                bias="none",
                task_type="FEATURE_EXTRACTION",
            ),
        )
        # The reward-token embedding stays trainable next to the LoRA adapters.
        self.vlm.get_input_embeddings().reward_embedding.requires_grad_(True)

        # ------------------------------------------------------------------
        # Reward head (kept in fp32 for stable training).
        # ------------------------------------------------------------------
        hidden_size = self.vlm.config.text_config.hidden_size
        self.reward_head = RewardHead(hidden_size)
        self.reward_head.to(
            device=self.device,
            dtype=torch.float32,
        )

    # ------------------------------------------------------------------
    # Device helpers.
    # ------------------------------------------------------------------
    @property
    def device(self) -> torch.device:
        return self.vlm.device if hasattr(self.vlm, "device") else next(self.vlm.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.vlm.parameters()).dtype

    # ------------------------------------------------------------------
    # Forward pass.
    # ------------------------------------------------------------------
    def forward(self, batch: dict) -> torch.Tensor:
        """Score a batch of states.

        Args:
            batch: dict with
                - ``obs.main_images``: images of the w history frames.  A list
                  of ``PIL.Image`` / array per sample, or a stacked array of
                  shape ``(B, w, H, W, 3)``.
                - ``obs.wrist_images``: optional, same layout.
                - ``task_prompt``: str or list[str].

        Returns:
            Tensor of shape ``(B,)`` with success probabilities in [0, 1].
        """
        task_prompt, main_images, wrist_images = self._extract_from_batch(batch)
        messages = self._build_messages(task_prompt, main_images, wrist_images)
        inputs = self._build_model_inputs(messages)

        outputs = self.vlm(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
            logits_to_keep=1,  # only hidden states are used; skip full-vocab logits
        )
        last_hidden_state = outputs.hidden_states[-1]
        pooled = self._pool_reward_token(last_hidden_state, inputs["input_ids"])
        head_param = next(self.reward_head.parameters())
        pooled = pooled.to(device=head_param.device, dtype=head_param.dtype)
        return self.reward_head(pooled)

    def predict(self, batch: dict) -> torch.Tensor:
        """Alias of :meth:`forward` used by the evaluation code."""
        return self.forward(batch)

    # ------------------------------------------------------------------
    # Target network for TD training.
    # ------------------------------------------------------------------
    def build_target_model(self) -> "EVTA0":
        """Create the TD target network: a frozen snapshot of this model."""
        import copy

        target = copy.deepcopy(self)
        for param in target.parameters():
            param.requires_grad = False
        target.eval()
        return target

    @torch.no_grad()
    def ema_update_from(self, online: "EVTA0", alpha: float) -> None:
        """EMA update: ``self <- alpha * online + (1 - alpha) * self``.

        The target network is updated every K optimizer steps with
        ``alpha = 0.1``.
        """
        for target_param, online_param in zip(self.parameters(), online.parameters()):
            target_param.mul_(1.0 - alpha).add_(online_param.detach(), alpha=alpha)

    # ------------------------------------------------------------------
    # Checkpointing.
    # ------------------------------------------------------------------
    def get_checkpoint_state(self) -> dict[str, Any]:
        """Serialize the trainable parts (LoRA, reward token, head) + protocol."""
        state = {
            "reward_head": self.reward_head.state_dict(),
            "goal_prompt": self.goal_prompt,
            "pretrained_path": self.pretrained_path,
            "reward_token": self.reward_token,
            "reward_token_id": self.reward_token_id,
            "reward_token_embedding": self._get_reward_token_embedding().detach().cpu(),
            "context_window": self.context_window,
            "frame_interval": self.frame_interval,
            "target_horizon": self.target_horizon,
            "protocol_version": self.protocol_version,
            "prompt_version": self.prompt_version,
            "goal_prompt_sha256": goal_prompt_sha256(self.goal_prompt),
            "padding_policy": self.padding_policy,
        }
        state["peft_state_dict"] = get_peft_model_state_dict(self.vlm)
        return state

    def load_checkpoint_state(self, state: dict[str, Any]) -> None:
        """Load a checkpoint produced by :meth:`get_checkpoint_state`.

        The stored input protocol must match the runtime configuration
        exactly; a mismatch raises ``ValueError`` instead of silently scoring
        with the wrong frame spacing.
        """
        checkpoint_prompt = str(state.get("goal_prompt", self.goal_prompt))
        checkpoint_prompt_sha = str(
            state.get("goal_prompt_sha256", goal_prompt_sha256(checkpoint_prompt))
        )
        if checkpoint_prompt_sha != goal_prompt_sha256(checkpoint_prompt):
            raise ValueError("Corrupt checkpoint: goal_prompt_sha256 does not match goal_prompt.")

        expected = (
            self.protocol_version,
            self.prompt_version,
            goal_prompt_sha256(self.goal_prompt),
            self.context_window,
            self.frame_interval,
            self.target_horizon,
            self.padding_policy,
        )
        actual = (
            str(state.get("protocol_version")),
            str(state.get("prompt_version")),
            checkpoint_prompt_sha,
            int(state.get("context_window", -1)),
            int(state.get("frame_interval", -1)),
            int(state.get("target_horizon", -1)),
            str(state.get("padding_policy")),
        )
        if actual != expected:
            raise ValueError(
                "Checkpoint input protocol mismatch: "
                f"checkpoint={actual}, runtime={expected}."
            )

        self.reward_head.load_state_dict(state["reward_head"])
        if "peft_state_dict" in state:
            set_peft_model_state_dict(self.vlm, state["peft_state_dict"])
        if "reward_token_embedding" in state:
            self._set_reward_token_embedding(state["reward_token_embedding"])

    def save_checkpoint(self, path, **metadata: Any) -> None:
        """Save an inference/training checkpoint dict to ``path``."""
        from pathlib import Path

        checkpoint = {
            "checkpoint_schema_version": 1,
            "model_state": self.get_checkpoint_state(),
            **metadata,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)

    def load_checkpoint(self, path) -> dict[str, Any]:
        """Load a full checkpoint file; returns the outer metadata dict."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.load_checkpoint_state(checkpoint["model_state"])
        return checkpoint

    # ------------------------------------------------------------------
    # Batch plumbing.
    # ------------------------------------------------------------------
    def _extract_from_batch(self, batch: dict):
        obs = batch.get("obs", batch)
        main_images = obs.get("main_images", obs.get("image"))
        wrist_images = obs.get("wrist_images", obs.get("wrist_image"))
        if main_images is None and "image" in batch:
            main_images = batch["image"]
        task_prompt = batch.get("task_prompt")
        if task_prompt is None:
            raise ValueError("Batch is missing 'task_prompt'.")
        return task_prompt, main_images, wrist_images

    def _build_messages(
        self,
        task_prompt: str | list[str],
        main_images,
        wrist_images,
    ) -> list[list[dict[str, Any]]]:
        task_prompts = [task_prompt] if isinstance(task_prompt, str) else list(task_prompt)
        main_batches = self._normalize_image_batches(main_images, len(task_prompts))
        wrist_batches = (
            self._normalize_image_batches(wrist_images, len(task_prompts))
            if wrist_images is not None
            else [[] for _ in task_prompts]
        )

        messages = []
        for prompt, main_batch, wrist_batch in zip(task_prompts, main_batches, wrist_batches):
            content: list[dict[str, Any]] = [
                {"type": "text", "text": self.goal_prompt},
                {"type": "text", "text": f"Task prompt: {prompt}"},
            ]
            for t in range(len(main_batch)):
                content.append({"type": "image", "image": main_batch[t]})
                if t < len(wrist_batch):
                    content.append({"type": "image", "image": wrist_batch[t]})
            content.append({"type": "text", "text": self.reward_token})
            messages.append([{"role": "user", "content": content}])
        return messages

    def _build_model_inputs(self, messages: list[list[dict[str, Any]]]) -> dict[str, torch.Tensor]:
        texts = [
            self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            for message in messages
        ]
        images = [
            [item["image"] for item in message[0]["content"] if item["type"] == "image"]
            for message in messages
        ]
        inputs = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        return inputs.to(self.vlm.device)

    def _pool_reward_token(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        reward_mask = input_ids.eq(self.reward_token_id)
        if not reward_mask.any(dim=1).all():
            raise ValueError("Reward token is missing from one or more samples.")
        reward_positions = reward_mask.int().argmax(dim=1)
        return hidden_states[
            torch.arange(hidden_states.size(0), device=hidden_states.device),
            reward_positions,
        ]

    def _normalize_image_batches(self, images, batch_size: int) -> list[list[Image.Image]]:
        if isinstance(images, torch.Tensor):
            if images.ndim == 3:
                images = images.unsqueeze(0).unsqueeze(0)
            elif images.ndim == 4:
                images = images.unsqueeze(0)
            if images.ndim != 5:
                raise ValueError(f"Expected an image tensor with 4 or 5 dims, got {tuple(images.shape)}")
            return [[self._to_pil_image(image) for image in sample] for sample in images]

        if batch_size == 1 and images and isinstance(images[0], Image.Image):
            return [[self._to_pil_image(image) for image in images]]

        return [[self._to_pil_image(image) for image in sample] for sample in images]

    def _to_pil_image(self, image) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")

        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        image = np.asarray(image)

        if image.ndim != 3:
            raise ValueError(f"Expected a 3D image array, got shape {image.shape}")
        if image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
            image = np.transpose(image, (1, 2, 0))
        if image.dtype != np.uint8:
            if np.issubdtype(image.dtype, np.floating) and image.max() <= 1.0:
                image = image * 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        return Image.fromarray(image).convert("RGB")

    def _get_reward_token_embedding(self) -> torch.Tensor:
        input_embeddings = self.vlm.get_input_embeddings()
        if isinstance(input_embeddings, RewardTokenEmbeddingAdapter):
            return input_embeddings.reward_embedding
        return input_embeddings.weight[self.reward_token_id]

    def _set_reward_token_embedding(self, embedding: torch.Tensor) -> None:
        input_embeddings = self.vlm.get_input_embeddings()
        with torch.no_grad():
            if isinstance(input_embeddings, RewardTokenEmbeddingAdapter):
                input_embeddings.reward_embedding.copy_(
                    embedding.to(
                        device=input_embeddings.reward_embedding.device,
                        dtype=input_embeddings.reward_embedding.dtype,
                    )
                )
            else:
                input_embeddings.weight[self.reward_token_id].copy_(
                    embedding.to(device=input_embeddings.weight.device, dtype=input_embeddings.weight.dtype)
                )

    def _install_reward_token_adapter(self) -> None:
        input_embeddings = self.vlm.get_input_embeddings()
        if isinstance(input_embeddings, RewardTokenEmbeddingAdapter):
            return
        self.vlm.set_input_embeddings(
            RewardTokenEmbeddingAdapter(input_embeddings, reward_token_id=self.reward_token_id)
        )
