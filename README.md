# eVTA₀

Hi! This is the official repository of the paper
["Demonstration-Free Success-Probability Reward Learning for Generalist Robot
Policies"](https://arxiv.org/abs/2609.33653).

**Project:** https://duowuyms.github.io/evta0/

**Paper:** https://arxiv.org/abs/2609.33653

## Abstract

Reinforcement learning (RL) enables generalist robot policies to improve
through trial-and-error interaction, yet its effectiveness is fundamentally
constrained by sparse task rewards. Existing general-purpose reward models
typically alleviate this issue by learning task progress from expert
demonstrations, but introduce a distribution mismatch with the mixed-quality
rollouts encountered during policy optimization, making their estimates
unreliable on suboptimal and failed behaviors from which the policy must
learn. In this work, we introduce a demonstration-free reward learning
paradigm where dense reward feedback can be learned directly from sparse task
outcomes and policy experience. We theoretically show that terminal task
outcomes implicitly define dense success-probability feedback at intermediate
timesteps, which can be recursively learned through bootstrapping. Based on
this insight, we introduce eVTA₀, which learns success probabilities from
mixed-quality policy rollouts through temporal-difference-style
bootstrapping, without expert demonstrations or intermediate annotations. We
further introduce RL with Evolving Rewards (RLER), a closed-loop framework
that adapts eVTA₀ using newly collected rollouts as the policy evolves.
Experiments show that eVTA₀ provides more informative rewards than
state-of-the-art reward models and achieves the best average policy
performance across all LIBERO task suites under the same RL training budget,
improving success rates by 5.4%-13.8% over the initial policy. In real-world
manipulation, RLER further improves overall success rates by 20%-26%, with
35%-36% gains under out-of-distribution conditions. Project webpage:
https://duowuyms.github.io/evta0.


## Repository structure

| Directory | Description |
|---|---|
| [`evta0/`](evta0/) | Training and evaluation code for the reward model: TD-bootstrap training, soft-target data augmentation, and the VOC / VROC / MSE / Kendall tau-a evaluation metrics. |
| [`rlinf-evta0/`](rlinf-evta0/) | Integration with the [RLinf](https://github.com/RLinf/RLinf) framework for GRPO policy learning driven by a frozen eVTA₀ reward, pinned to a specific RLinf commit, with training and evaluation configs for the four LIBERO suites. |

## Usage

A typical workflow:

1. Prepare training data: collect mixed-quality rollouts with binary terminal
   outcomes, and run `python -m evta0.augment` to assign soft targets to
   failures (see `evta0/README.md`).
2. Train the reward model with TD bootstrapping:
   `python -m evta0.train --config evta0/configs/evta0.yaml`.
3. Evaluate reward quality: VOC / VROC on demonstrations
   (`evta0.eval_progress`) and MSE / Kendall tau-a on eval rollouts
   (`evta0.eval_terminal`).
4. Train a policy with the reward: apply the `rlinf-evta0/` overlay to the
   pinned RLinf checkout and run GRPO with `r = P(success) - 1`
   (see `rlinf-evta0/README.md`).

## Checkpoints and datasets

The LIBERO / MetaWorld checkpoints as well as the training and evaluation datasets are released
[here](https://huggingface.co/collections/notmuch2/evta0). The released checkpoints were serialized under
transformers 5.8.0 / peft 0.19.1; loading them requires
transformers >= 5.8.0 and peft >= 0.19.1 (see `rlinf-evta0/README.md` for the
reward-worker environment notes).

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@article{wu2026evta0,
  title={Demonstration-Free Success-Probability Reward Learning for Generalist Robot Policies},
  author={Wu, Duo and Wang, Haifeng and Lu, Rongwei and Wang, Jinghe and Xiong, Tianyi and Wang, Zhimin and Yu, Chao and Ma, Shuai and Wang, Zhi},
  journal={arXiv preprint arXiv:2609.33653},
  year={2026}
}
```
