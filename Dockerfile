FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    JUPYTER_TOKEN=head-trajectories

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    tini \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/head-trajectories

COPY requirements.txt README.md setup.py MANIFEST.in ./

RUN python -m pip install --upgrade pip && \
    python -m pip install \
      "numpy>=1.24.0" \
      "tiktoken>=0.6.0" \
      "datasets>=2.18.0" \
      "huggingface_hub>=0.22.0" \
      "transformers>=4.38.0" \
      "modal>=0.62.0" \
      "matplotlib>=3.8.0" \
      "scipy>=1.12.0" \
      "tqdm>=4.66.0" \
      "pytest>=8.0.0" \
      "pytest-cov>=4.1.0" \
      "jupyterlab>=4.2.0" \
      "ipykernel>=6.29.0" \
      "pillow>=10.0.0"

COPY . .

RUN python -m pip install -e . && \
    python -m ipykernel install --sys-prefix --name head-trajectories --display-name "Python (head-trajectories)"

RUN mkdir -p /workspace/head-trajectories/artifacts \
    /workspace/head-trajectories/checkpoints \
    /workspace/head-trajectories/results \
    /workspace/head-trajectories/figures \
    /workspace/head-trajectories/run_exports

EXPOSE 8888

ENTRYPOINT ["tini", "--"]
CMD ["bash"]
