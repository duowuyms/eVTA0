# eVTA₀

Official code for **["Demonstration-Free Success-Probability Reward Learning
for Generalist Robot Policies"](https://arxiv.org/abs/2609.33653)**.

## Method

eVTA₀ scores a state from its recent observation history.  A VLM backbone
consumes an instruction, the task description, the last `w` observations,
and a learned `[REWARD]` token; a single head maps the token's hidden state
to a success probability:

```
x_t   = [instruction, task prompt, O_t, [REWARD]]   # multimodal input
h_t   = F_psi(x_t)_[REWARD]                         # VLM backbone (LoRA)
p_t   = sigmoid(g_phi(h_t))                         # single linear head -> P(success) in [0, 1]
```

By default (see `configs/evta0.yaml`) the model uses `w = 5` observations
taken every `s = 2` frames on a Qwen3-VL-4B backbone with LoRA rank 16;
checkpoints record their window and stride, and loading validates the match.

Training uses TD-style bootstrapping over tuples `(x_t, x_t^+, d_t, z_t)`
sampled from mixed-quality policy rollouts, with terminal outcomes anchoring
the regression:

```
y_t   = d_t * z_t + (1 - d_t) * p̂_θ⁻(x_t⁺)          # θ⁻ = EMA target network
L(θ)  = E[ (p̂_θ(x_t) − y_t)² ]                      # plain MSE
θ⁻   ← α·θ + (1−α)·θ⁻  every K optimizer steps (defaults: K = 16, α = 0.1)
```

`d_t` marks the terminal frame and `z_t` is the terminal outcome; failed
rollouts receive *soft* targets derived from their distance to successful
trajectory clusters (see [Data augmentation](#data-augmentation)).  Because
a raw success probability would favor longer trajectories, the dense reward
used for policy learning is `r_t = p_t − 1` — a failure-probability penalty
that preserves ordering and pairwise differences.

## Repository layout

```
evta0/
├── protocol.py       # input protocol: goal prompt, history-window indices
├── model.py          # EVTA0 model (VLM + LoRA + reward token + success head)
├── data.py           # HDF5 rollouts -> TD tuples (x_t, x_t^+, d_t, z_t)
├── augment.py        # soft failure targets (V-JEPA2 + DBSCAN + sigmoid)
├── train.py          # TD-bootstrap training
├── metrics.py        # VOC, VROC, MSE, Kendall tau-a
├── eval_progress.py  # VOC/VROC on demonstration trajectories (LeRobot format)
├── eval_terminal.py  # MSE / tau-a on rollout terminal states
└── infer.py          # score one state from a history of images
configs/evta0.yaml    # default configuration
```

## Installation

```bash
pip install -r requirements.txt
```

The Qwen3-VL-4B-Instruct base weights are loaded through the Hugging Face
hub cache or a local path (`model.pretrained_path` in the config).

## Data format

Training and terminal evaluation read augmented rollout collections stored as
one HDF5 file per task with one top-level group per rollout:

```
traj_0/
    obs_main_images    (T, H, W, 3) uint8   third-person camera
    obs_wrist_images   (T, H, W, 3) uint8   wrist camera
    rewards            (T,) float           soft terminal target at index -1
    dones              (T,) bool
    actions            (T, A)
    attrs: success (bool), task_index (int), task_prompt (str)
```

`rewards[-1]` is `1.0` for successes and a soft value in `(0, 1)` for
failures; other entries are unused.  `evta0.eval_progress` additionally reads
LeRobot demonstration datasets (`meta/episodes.jsonl` +
`data/chunk-*/episode_*.parquet`).

## Data augmentation

Training eVTA₀ needs soft targets for failed rollouts, not just hard zeros.
`evta0.augment` turns a raw rollout collection (binary terminal rewards) into
an augmented one: trajectories are embedded with a frozen V-JEPA2 world
model, successful trajectories are clustered per task with DBSCAN, and each
failure's terminal target becomes a sigmoid of its normalized distance to the
nearest success cluster center (successes stay at 1.0):

```
target = 0.6 * sigmoid(10 * (0.5 - normalized_distance))
```

The sigmoid scale, steepness, and offset are options of `evta0.augment`
(defaults shown).  Success references default to the collection's own
successful rollouts; pass `--reference-root <lerobot-dataset>` to use
external success demonstrations instead (up to
`--reference-episodes-per-task` per task).

```bash
python -m evta0.augment \
    --input-root /path/to/raw_rollout_hdf5 \
    --output-root /path/to/augmented_hdf5 \
    --reference-root /path/to/lerobot_demos
```

The output tree mirrors the input with `rewards[-1]` replaced by the soft
target and `dones[-1]` forced True; `augmentation_report.json` records the
per-task clustering metadata and every assignment.

## Training

```bash
python -m evta0.train --config configs/evta0.yaml --save-dir outputs/evta0
```

Edit `configs/evta0.yaml` (`data.hdf5_root`, optionally `data.split_manifest`)
first.  Checkpoints are written per epoch as `checkpoint_epoch_<N>.pth`.  The
released checkpoints correspond to epoch 5 (LIBERO) and epoch 7 (MetaWorld).

## Evaluation

VOC / VROC on demonstrations (one `--data-root` per suite).  By default the
anchor grid is `range(sample_step, T, sample_step)` with `sample_step = 5`
(adjustable via `--sample-step`), and both directions (forward and reversed
presentation) are scored; VOC/VROC use a tie-aware Spearman correlation where
constant prediction sequences score 0.

```bash
python -m evta0.eval_progress \
    --config configs/evta0.yaml \
    --checkpoint outputs/evta0/checkpoint_epoch_5.pth \
    --data-root /path/to/lerobot--libero_spatial_image@v2.0 \
    --num-episodes 10 \
    --output results/progress_spatial.json
```

MSE / Kendall tau-a at rollout terminals:

```bash
python -m evta0.eval_terminal \
    --config configs/evta0.yaml \
    --checkpoint outputs/evta0/checkpoint_epoch_5.pth \
    --hdf5-root /path/to/rollout_collection \
    --output results/terminal.json
```

Single-state inference (images ordered oldest-to-current; with the default
`w = 5, s = 2` this is `[t-8, t-6, t-4, t-2, t]`):

```bash
python -m evta0.infer \
    --checkpoint outputs/evta0/checkpoint_epoch_5.pth \
    --task-prompt "put the bowl on the plate" \
    --main-images t8.png t6.png t4.png t2.png t0.png
```

## Symbol <-> code correspondence

| Symbol | Code |
|---|---|
| `F_psi` (backbone) | `EVTA0.vlm` (Qwen3-VL + LoRA) |
| `[REWARD]` token (learned) | `RewardTokenEmbeddingAdapter` |
| `g_phi` (success head) | `EVTA0.reward_head` |
| `p̂(x_t)` | `EVTA0.forward` |
| target network `θ⁻` | `EVTA0.build_target_model` / `ema_update_from` |
| `(x_t, x_t^+, d_t, z_t)` | `evta0.data.EVTA0TrajectoryDataset` items |
| TD target `y_t`, loss `L(θ)` | `evta0.train` |
| `w`, `s` (window / stride) | `model.context_window`, `model.frame_interval` |
| `K`, `α` (EMA cadence) | `train.target_update_interval`, `train.ema_alpha` |
| soft failure targets (world model + DBSCAN + sigmoid) | `evta0.augment` |
| VOC / VROC | `evta0.metrics.voc` / `vroc`, `evta0.eval_progress` |
| MSE / Kendall tau-a | `evta0.metrics.mse` / `kendall_tau_a`, `evta0.eval_terminal` |

## Checkpoint format

A checkpoint is a dict with a `model_state` entry produced by
`EVTA0.get_checkpoint_state()`, holding the LoRA adapters (`peft_state_dict`),
the reward-token embedding, the success head (`reward_head`), and the input
protocol fields.  Loading validates the protocol (window, stride, prompt
hash); a mismatch raises instead of silently mis-scoring.

## Citation

If you find this code useful, please cite:

```bibtex
@article{wu2026evta0,
  title={Demonstration-Free Success-Probability Reward Learning for Generalist Robot Policies},
  author={Duo Wu and Haifeng Wang and Rongwei Lu and Jinghe Wang and Tianyi Xiong and Zhimin Wang and Chao Yu and Shuai Ma and Zhi Wang},
  journal={arXiv preprint arXiv:2609.33653},
  year={2026},
  url={https://arxiv.org/abs/2609.33653}
}
```
