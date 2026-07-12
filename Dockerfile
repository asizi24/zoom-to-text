# syntax=docker/dockerfile:1.7
# ==============================================================================
# Stage 1: Builder — install Python dependencies into an isolated prefix
# ==============================================================================
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt requirements-gpu.txt ./

# BuildKit cache mount: repeat builds resolve wheels from the local pip cache
# instead of re-downloading. faster-whisper runs on CTranslate2 + onnxruntime —
# torch is NOT needed and is deliberately not installed (~1.7 GB saved).
# ENABLE_GPU=1 (default) adds the cuBLAS/cuDNN wheels (~800 MB) so CTranslate2
# can run on CUDA; build with --build-arg ENABLE_GPU=0 for a lean CPU-only image.
ARG ENABLE_GPU=1
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefix=/install -r requirements.txt \
    && if [ "$ENABLE_GPU" = "1" ]; then \
         pip install --prefix=/install -r requirements-gpu.txt; \
       fi

# ==============================================================================
# Stage 2: Runtime — copy only the installed packages, not the build tools
# ==============================================================================
FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Model weights land here; docker-compose mounts a named volume at this
    # exact path. Must stay in sync with the compose volume target.
    WHISPER_CACHE_DIR=/app/models \
    # CTranslate2 dlopens cuBLAS/cuDNN at runtime; the pip wheels put the .so
    # files under site-packages/nvidia/*/lib, which is not a default search
    # path. Harmless when the GPU wheels are absent (CPU-only build).
    LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib

WORKDIR /app

COPY app/    ./app/
COPY static/ ./static/
# key.json is NOT copied — GCP credentials are injected via environment secrets
# (fly secrets set GOOGLE_APPLICATION_CREDENTIALS_JSON="..." for production)

# Non-root user for security. /app/models must exist and be writable by
# appuser BEFORE the volume is mounted, or the model download fails.
RUN useradd -m -u 1000 appuser \
    && mkdir -p /app/models data/downloads \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Python healthcheck — no curl dependency in the image
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)"

# Single uvicorn worker is deliberate: the pipeline queue and the Whisper model
# slot are in-process state. Scale by raising PIPELINE_CONCURRENCY (RAM
# permitting) or by splitting the worker into its own service — not --workers N.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
