# Container Workflow

This project ships with a GPU-capable Docker image so the repository can be run
in a consistent environment for:

- script-based runs
- the main experiment notebook
- probing and analysis
- diagnostics and figure generation

The image is meant to be used on machines with the NVIDIA container runtime
when GPU training is needed.

## What the image contains

- Python `3.11`
- PyTorch `2.3.0` with CUDA `12.1`
- the full repository installed in editable mode
- JupyterLab
- the notebook kernel `Python (head-trajectories)`
- all runtime dependencies used by the codebase and notebook

The image is suitable for:

- running [run_single_experiment.py](/C:/Users/wwwab/Development/Head%20Trajectories/run_single_experiment.py)
- running [run_probing.py](/C:/Users/wwwab/Development/Head%20Trajectories/run_probing.py)
- running [run_analysis.py](/C:/Users/wwwab/Development/Head%20Trajectories/run_analysis.py)
- running [head_trajectories_experiment_runner.ipynb](/C:/Users/wwwab/Development/Head%20Trajectories/notebooks/head_trajectories_experiment_runner.ipynb)

## Build

From the repository root:

```bash
docker build -t head-trajectories:latest .
```

## Launch an Interactive Shell

```bash
docker run --rm -it \
  --gpus all \
  -v "$PWD:/workspace/head-trajectories" \
  -w /workspace/head-trajectories \
  head-trajectories:latest
```

If you only need CPU work, remove `--gpus all`.

## Run the Notebook

```bash
docker run --rm -it \
  --gpus all \
  -p 8888:8888 \
  -v "$PWD:/workspace/head-trajectories" \
  -w /workspace/head-trajectories \
  head-trajectories:latest \
  jupyter lab --ip 0.0.0.0 --port 8888 --no-browser --allow-root --ServerApp.token=head-trajectories
```

Use this token:

```text
head-trajectories
```

Open the printed Jupyter URL in your browser, then run:

- [head_trajectories_experiment_runner.ipynb](/C:/Users/wwwab/Development/Head%20Trajectories/notebooks/head_trajectories_experiment_runner.ipynb)

## Run the Script Workflow

Single experiment:

```bash
docker run --rm -it \
  --gpus all \
  -v "$PWD:/workspace/head-trajectories" \
  -w /workspace/head-trajectories \
  head-trajectories:latest \
  python run_single_experiment.py --profile wikitext103_15m_preliminary --seed 42
```

Probing:

```bash
docker run --rm -it \
  -v "$PWD:/workspace/head-trajectories" \
  -w /workspace/head-trajectories \
  head-trajectories:latest \
  python run_probing.py --seed 42
```

Analysis:

```bash
docker run --rm -it \
  -v "$PWD:/workspace/head-trajectories" \
  -w /workspace/head-trajectories \
  head-trajectories:latest \
  python run_analysis.py
```

## Notes

- The image installs the repository itself with `pip install -e .`, so changes in your mounted working tree are picked up immediately.
- Large run outputs should be written to mounted directories such as `artifacts/` or `run_exports/`, not stored inside an ephemeral container filesystem.
- Modal jobs already build their own image path in [modal_jobs/](/C:/Users/wwwab/Development/Head%20Trajectories/modal_jobs), so this container is mainly for local, VM, and notebook workflows.
- If you run on a hosted GPU VM, make sure the NVIDIA container toolkit is available before using `--gpus all`.
