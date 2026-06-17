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
    gosu \
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

# Create the non-root user FIRST so the application code can be COPY'd straight
# into place already owned by it. A trailing `RUN chown -R appuser:appuser /app`
# rewrites every file's metadata into a brand-new image layer — effectively
# doubling the size of the app layers on disk. `COPY --chown` sets ownership at
# copy time for free.
RUN useradd -m -u 1000 appuser

# extension/ (the Chrome cookie-helper for yt-dlp) is intentionally NOT copied:
# it's a browser-side concern the backend never imports or serves, so shipping
# it into the API image is pure bloat.
COPY --chown=appuser:appuser app/    ./app/
COPY --chown=appuser:appuser static/ ./static/
# key.json is NOT copied — GCP credentials are injected via environment secrets
# (fly secrets set GOOGLE_APPLICATION_CREDENTIALS_JSON="..." for production)

# data/ holds the SQLite DB + downloaded audio; it must be writable by appuser.
# (The entrypoint additionally re-chowns the /app/data bind-mount at runtime,
# since a host volume arrives root-owned and overrides this.)
RUN mkdir -p data/downloads && chown -R appuser:appuser data

# Entrypoint fixes ownership of the /app/data bind-mount at runtime (a host
# mount overrides the image's chown and arrives root-owned, which made the
# app crash with PermissionError on data/downloads). It runs as root, chowns
# the mounted dir, then drops to appuser via gosu before exec'ing the app.
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
