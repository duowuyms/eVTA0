# eVTA₀ × RLinf: reward-model integration for policy RL

This overlay plugs the released **eVTA₀** success-probability reward model
into [RLinf](https://github.com/RLinf/RLinf) so eVTA₀ can drive RL-based policy learning
(e.g., GRPO on LIBERO).  It contains only the integration code and the accompanying
training / evaluation configs; the eVTA₀ model itself and its training code
live in the sibling `evta0/` directory.

## Pinned RLinf version

The overlay is built against **upstream RLinf commit
`c69eecbaa7eb6db06aa3318f103305703f3266c5`** (2026-07-31, pyproject version
0.3.0).  Apply it to a checkout of exactly this commit; other versions are
untested.

```bash
git clone https://github.com/RLinf/RLinf.git
cd RLinf && git checkout c69eecbaa7eb6db06aa3318f103305703f3266c5
cp -r /path/to/rlinf-evta0/evta0_rlinf/* .    # overlay (5 files)
```

The overlay touches five files (all Apache-2.0, like RLinf):

| File | Role |
|---|---|
| `rlinf/models/embodiment/reward/evta0_reward_model.py` | **new** — `EVTA0RewardModel`: loads a frozen eVTA₀ checkpoint and serves `P(success | history)` from RLinf's reward worker |
| `rlinf/models/embodiment/reward/__init__.py` | lazy reward-model registry with the `evta0` entry |
| `rlinf/workers/env/env_worker.py` | eVTA₀ reward plumbing: strided history window from raw action frames (size/stride from `reward.model.context_window` / `frame_interval`), `r = P - 1` transform, terminal-success anchoring, reward mixing weights |
| `rlinf/workers/env/history_manager.py` | history append honors a per-environment validity mask |
| `rlinf/envs/libero/libero_env.py` | `chunk_step` emits `reward_valid_mask` / `success_now` / `success_before` |

## How the reward enters training

1. The env worker keeps every low-level frame of each action chunk
   (`history_source: raw_action_frames`).
2. At each chunk boundary it materializes the model's history window —
   sized by `reward.model.context_window` / `frame_interval` (with the
   defaults 5 / 2 this is `[t-8, t-6, t-4, t-2, t]`), zero-left padded
   before the episode start — and sends it to the reward worker.
3. `EVTA0RewardModel.compute_reward` scores each environment's window;
   chunks of episodes that already succeeded or just reached terminal
   success are skipped via `reward_valid_mask`.
4. The env worker maps the probability to the training reward with
   `reward_output_transform: probability_minus_one` — `r = P(success) - 1`,
   a failure-probability penalty that preserves ordering and pairwise
   differences while avoiding a length bias toward long rollouts.
5. `terminal_success_anchor: true` gives the chunk containing a first
   success the environment's terminal outcome (0 under `p - 1`), and
   already-successful episodes receive exactly zero; with
   `env_reward_weight: 0.0` the learned reward is the only training signal.

## Usage

Prerequisites: the pinned RLinf checkout with the overlay, the `evta0`
package (`../evta0`) on the reward worker's `PYTHONPATH` or via
`reward.model.evta0_package_path`, an eVTA₀ checkpoint, a
pi0.5 SFT policy checkpoint, and LIBERO assets.

**Important — reward-model python environment (when using the released
checkpoints).**  The released eVTA₀ checkpoints were trained and serialized
under `transformers 5.8.0` / `peft 0.19.1`: if you load them, the interpreter
that runs the reward worker (`RLINF_REWARD_PYTHON`) **must have
`transformers >= 5.8.0` and `peft >= 0.19.1`** — older versions fail while
loading the model (loading errors / missing classes).  If you train your own
eVTA₀ checkpoint or use another reward model, other versions may work as
long as they can load it.  The policy/env interpreter
(`RLINF_DRIVER_PYTHON`) is independent and uses the openpi stack.

```bash
export EMBODIED_PATH=<rlinf-checkout>/examples/embodiment
export RLINF_DRIVER_PYTHON=<policy/env python>       # openpi stack
export RLINF_REWARD_PYTHON=<reward python>           # transformers 5.8 / peft 0.19 stack
export PYTHONPATH=<rlinf-checkout>

cd <rlinf-checkout>
python -u examples/embodiment/train_embodied_agent.py \
    --config-path /path/to/rlinf-evta0/configs/train \
    --config-name evta0_libero10_grpo \
    +rollout.model.model_path=/path/to/pi05_libero_sft \
    +actor.model.model_path=/path/to/pi05_libero_sft \
    +reward.model.checkpoint_path=/path/to/model/libero/checkpoint.pth \
    +reward.model.evta0_package_path=/path/to/evta0 \
    --cuda-list 0,1,2
```

The example config places the actor on GPU 0, the reward model on GPU 1,
and the env + rollout workers on GPU 2 (one node, three interpreters via
`cluster.node_groups`).  Training and evaluation configs are provided for
all four LIBERO suites — they differ only in the env default
(`env/libero_<suite>`) and the horizon:

| Suite | Train / eval config | Horizon |
|---|---|---|
| libero_spatial | `evta0_liberospatial_grpo` / `evta0_liberospatial_id500` | 240 |
| libero_object | `evta0_liberoobject_grpo` / `evta0_liberoobject_id500` | 240 |
| libero_goal | `evta0_libergoal_grpo` / `evta0_libergoal_id500` | 320 |
| libero_10 | `evta0_libero10_grpo` / `evta0_libero10_id500` | 480 |

Evaluation of a trained checkpoint (fixed initial states, 10 tasks × 50
trials, no reward model):

```bash
python -u evaluations/eval_embodied_agent.py \
    --config-path /path/to/rlinf-evta0/configs/eval \
    --config-name evta0_libero10_id500 \
    runner.ckpt_path=/path/to/full_weights.pt \
    +rollout.model.model_path=/path/to/pi05_libero_sft
```

## Tests

```bash
python -m pytest tests/ -q    # from the RLinf checkout root
```

`test_evta0_window.py` pins the window construction to the `evta0` package's
protocol (`ordered_history_indices`) including zero-left padding;
`test_terminal_anchor.py` covers the reward arbitration and history masking
logic.  Both run on CPU without LIBERO.

## Layout

```
evta0_rlinf/          # files to copy onto the RLinf checkout (5 files)
configs/train/        # GRPO training config with the eVTA₀ reward
configs/eval/         # fixed-initial-state evaluation config
tests/                # unit tests for the overlay logic
```

## Citation

If you find this code useful, please cite:

```bibtex
@article{wu2026evta0,
  title={Demonstration-Free Success-Probability Reward Learning for Generalist Robot Policies},
  author={Wu, Duo and Wang, Haifeng and Lu, Rongwei and Wang, Jinghe and Xiong, Tianyi and Wang, Zhimin and Yu, Chao and Ma, Shuai and Wang, Zhi},
  journal={arXiv preprint arXiv:2609.33653},
  year={2026}
}
```

> NOTE: The code in this directory is organized by Z.ai (智谱) and approved by Duo Wu.
