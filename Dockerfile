FROM --platform=linux/amd64 python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DGLBACKEND=pytorch \
    DGL_DISABLE_GRAPHBOLT=1 \
    PYTHONPATH=/app/paper:/app \
    DEXPOSURE_CACHE_DIR=/tmp/dexposure_live_cache_v2

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    "torch==2.2.1+cpu" \
    "torchdata==0.7.1" \
    "dgl==2.1.0" \
    "torch-geometric==2.6.1" \
    "ogb==1.3.6" \
    "delu==0.0.26" \
    "rtdl-num-embeddings==0.0.11" \
    "rtdl-revisiting-models==0.0.2" \
    "huggingface-hub==0.25.2" \
    "loguru==0.7.3" \
    "numpy==1.26.4" \
    "scipy==1.15.2" \
    "scikit-learn==1.6.1" \
    "pandas>=2.2.3,<3" \
    "PyYAML==6.0.2" \
    "tomli-w==1.2.0" \
    "tqdm==4.67.1" \
    "einops==0.8.1" \
    "loralib==0.1.2" \
    "networkx>=3.0" \
    "pydantic>=2.10.0,<3" \
    "fastapi>=0.115.0" \
    "uvicorn>=0.32.0"

RUN pip install --no-cache-dir \
    "optuna==4.2.1" \
    "xgboost==2.0.3" \
    "setuptools>=70.0"

RUN pip install --no-cache-dir "pillow>=10.0"

COPY lib /app/lib
COPY checkpoints/dexposure-fm-v4 /app/checkpoints/dexposure-fm-v4
COPY checkpoints/dexposure-fm-v3 /app/checkpoints/dexposure-fm-v3
COPY checkpoints/dexposure-fm-release/feature_schema.json /app/checkpoints/dexposure-fm-release/feature_schema.json
COPY checkpoints/LimiX-16M.ckpt /app/checkpoints/LimiX-16M.ckpt
COPY data/mapping /app/data/mapping
COPY outputs/pre2022_clean_replay_20261103.json /app/outputs/pre2022_clean_replay_20261103.json
COPY docs/design/dexposure-claw-case-workbench/assets /app/docs/design/dexposure-claw-case-workbench/assets
COPY paper/dexposure_agent /app/paper/dexposure_agent
COPY paper/experiments /app/paper/experiments
COPY paper/scripts /app/paper/scripts
COPY paper/tests /app/paper/tests

EXPOSE 8080

CMD ["uvicorn", "dexposure_agent.container_service:app", "--host", "0.0.0.0", "--port", "8080"]
