# One image for every process of the engine; docker-compose.yml runs it as worker A, worker B and the coordinator.
FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu

# procps: pkill, used by the crash test (docker compose exec worker-b pkill -9 -f src.worker.server)
RUN apt-get update && apt-get install -y --no-install-recommends procps && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# The model goes into the image, so containers start (and restart) without network access.
# Xet is off: its token endpoint 404s for the legacy "distilgpt2" name; plain HTTP works.
RUN HF_HUB_DISABLE_XET=1 python -c "from huggingface_hub import snapshot_download; \
snapshot_download('distilgpt2', allow_patterns=['*.json', '*.txt', 'model.safetensors'])"
ENV HF_HUB_OFFLINE=1

COPY src ./src
COPY results/reference.json ./results/reference.json
