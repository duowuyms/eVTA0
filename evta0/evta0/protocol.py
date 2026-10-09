"""Input protocol shared by training and evaluation.

eVTA0 scores a trajectory state from a short observation history.  With the
default configuration (context window ``w = 5``, temporal stride
``s = 2``) the
model sees the frames

    [t - 8, t - 6, t - 4, t - 2, t]

where ``t`` is the anchor being scored.  Frames that fall before the start of
the trajectory are replaced with zero images ("zero left padding"), so the
model always receives exactly ``w`` frames per camera.
"""

from __future__ import annotations

import hashlib

# Version tags stored inside every eVTA0 checkpoint.  A checkpoint is only
# loadable when these tags and the fields below match the runtime config.
PROTOCOL_VERSION = "eVTA0"
PROMPT_VERSION = "eVTA0"

# Instruction shown to the VLM before the task prompt and the images.
GOAL_PROMPT = (
    "You are a reward model for robotic manipulation. "
    "Given the task instruction and the ordered history of observations, "
    "predict the probability of successful task completion. "
    "Return a scalar reward in [0, 1], where 1 indicates a high probability of "
    "success and 0 indicates a high probability of failure. "
    "The multimodal inputs are provided in the following order: task prompt, "
    "history observation images, reward token."
)

REWARD_TOKEN = "<|reward_token|>"

PADDING_POLICY = "zero_left"


def goal_prompt_sha256(prompt: str) -> str:
    """Hash the exact UTF-8 prompt without whitespace normalization."""
    return hashlib.sha256(str(prompt).encode("utf-8")).hexdigest()


def ordered_history_indices(
    anchor: int,
    context_window: int,
    frame_interval: int = 1,
    *,
    reverse: bool = False,
) -> list[int]:
    """Return oldest-to-newest frame indices of the history window at ``anchor``.

    Forward window (used for VOC and everywhere else):

        [anchor - (w-1)*s, ..., anchor - s, anchor]

    Reverse window (used for VROC): the trajectory is presented in reversed
    time, so the "history" seen from the anchor consists of the original
    *future* frames in reverse temporal order:

        [anchor + (w-1)*s, ..., anchor + s, anchor]

    Indices outside ``[0, trajectory_length)`` must be dropped by the caller
    and zero-padded on the left.
    """
    context_window = int(context_window)
    frame_interval = int(frame_interval)
    if context_window < 1:
        raise ValueError("context_window must be >= 1")
    if frame_interval < 1:
        raise ValueError("frame_interval must be >= 1")
    direction = 1 if reverse else -1
    return [
        int(anchor) + direction * offset * frame_interval
        for offset in range(context_window - 1, -1, -1)
    ]
