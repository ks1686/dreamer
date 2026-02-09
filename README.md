# Dreamer (Modernized/Portable Fork)

**A portable, reproduction-ready fork of the original Dreamer agent.**

This repository contains a **self-contained environment setup** that allows the original TensorFlow 2.2-based code to run on modern Linux systems (2025/2026) without dependency conflicts.

Original Project: [danijar/dreamer](https://github.com/danijar/dreamer)

## 🚀 Key Differences in This Fork

Running older Deep Learning projects can be difficult due to Python/CUDA version mismatches. This fork solves that by:

1.  **Portable Python 3.8**: Bundles a standalone Python build so you don't need to downgrade your system Python.
2.  **Pinned Dependencies**: `requirements.txt` with exact versions of `tensorflow-gpu==2.2.0`, `dm_control`, and `mujoco` that are known to work together.
3.  **One-Click Setup**: Includes `reproduce_env.sh` to automatically set up the environment.

## 🛠️ Quick Start

### 1. Setup Environment

Simply run the reproduction script. It will download the portable Python and install everything into a local `venv/`.

```bash
bash reproduce_env.sh
```

### 2. Run Training

**Always use the python inside `./venv/bin/`**:

```bash
./venv/bin/python dreamer.py --logdir ./logdir/experiment1 --task dmc_walker_walk
```

### 3. Monitor Progress

Visualize training with TensorBoard:

```bash
./venv/bin/tensorboard --logdir ./logdir
```

### 4. Plot Results

Generate plotting curves:

```bash
./venv/bin/python plotting.py --indir ./logdir --outdir ./plots --xaxis step --yaxis train/return --bins 1000
```

_(Use `--bins 3e4` and `--yaxis test/return` for longer runs)_

---

## 📜 Original README

<details>
<summary>Click to view original README</summary>

# Dream to Control

**NOTE:** Check out the code for [DreamerV2](https://github.com/danijar/dreamerv2), which supports both Atari and DMControl environments.

Fast and simple implementation of the Dreamer agent in TensorFlow 2.

<img width="100%" src="https://imgur.com/x4NUHXl.gif">

If you find this code useful, please reference in your paper:

```
@article{hafner2019dreamer,
  title={Dream to Control: Learning Behaviors by Latent Imagination},
  author={Hafner, Danijar and Lillicrap, Timothy and Ba, Jimmy and Norouzi, Mohammad},
  journal={arXiv preprint arXiv:1912.01603},
  year={2019}
}
```

## Method

![Dreamer](https://imgur.com/JrXC4rh.png)

Dreamer learns a world model that predicts ahead in a compact feature space.
From imagined feature sequences, it learns a policy and state-value function.
The value gradients are backpropagated through the multi-step predictions to
efficiently learn a long-horizon policy.

- [Project website][website]
- [Research paper][paper]
- [Official implementation][code] (TensorFlow 1)

[website]: https://danijar.com/dreamer
[paper]: https://arxiv.org/pdf/1912.01603.pdf
[code]: https://github.com/google-research/dreamer

</details>
