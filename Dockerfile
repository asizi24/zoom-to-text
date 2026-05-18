# ==============================================================================
# Stage 1: Builder — install all Python dependencies system-wide
# ==============================================================================
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .

# torch / torchaudio moved to requirements-heavy.txt — NEVER installed in
# the production light image. The pyannote diarization path (DIARIZATION_PROVIDER
# =pyannote) needs heavy + a home-server. Production stays on text diarization.
RUN pip install --no-cache-dir -r requirements.txt

# ==============================================================================
# Stage 2: Runtime — copy only the installed packages, not the build tools
# ==============================================================================
FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*
# WeasyPrint system libs (libpango, libcairo, libharfbuzz, libgdk-pixbuf,
# fonts-dejavu, shared-mime-info) were removed when weasyprint moved to
# requirements-heavy.txt. Add them back in the heavy/home-server image only.

# Copy system-wide packages from builder (accessible by all users)
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY app/        ./app/
COPY static/     ./static/
COPY extension/  ./extension/
# key.json is NOT copied — GCP credentials are injected via environment secrets
# (fly secrets set GOOGLE_APPLICATION_CREDENTIALS_JSON="..." for production)

RUN mkdir -p data/downloads

# Non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
