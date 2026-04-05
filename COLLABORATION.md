# Collaboration Guide

This project requires a specific older version of Python (3.8) and TensorFlow (2.2.0). To ensure everyone uses the exact same environment, we have created a self-contained setup script.

## Quick Start

1.  **Clone the repository.**
2.  **Run the setup script:**
    ```bash
    bash reproduce_env.sh
    ```
    This will:
    - Download a portable Python 3.8 to `.tools/` (won't affect your system).
    - Create a virtual environment in `venv/`.
    - Install all dependencies from `requirements.txt`.

## Running the Code

Always use the python inside `venv`:

**Train:**

```bash
./venv/bin/python dreamer.py --logdir ./logdir/experiment1 --task dmc_walker_walk
```

**Plot:**

```bash
./venv/bin/python plotting.py --indir ./logdir --outdir ./plots --xaxis step --yaxis train/return --bins 1000
```

_(Use `--yaxis test/return` and `--bins 3e4` for longer runs)_

**Visualize:**

```bash
./venv/bin/tensorboard --logdir ./logdir
```
