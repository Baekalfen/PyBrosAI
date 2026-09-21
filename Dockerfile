FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements-cuda.txt .
RUN python3 -m venv /opt/venv \
    && pip install --upgrade pip \
    && pip install -r requirements-cuda.txt

COPY mario_env.py route_memory.py ppo.py ppo_train.py ppo_demo.py README.md ./

ENTRYPOINT ["python", "ppo_train.py"]
