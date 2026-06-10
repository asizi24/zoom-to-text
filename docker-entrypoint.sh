#!/bin/sh
set -e

# The /app/data bind-mount arrives owned by the host user (root from the
# container's view), which prevents appuser (uid 1000) from creating
# data/downloads — the app then crashes at import time with PermissionError.
# Fix ownership here at runtime, then drop privileges to appuser.
mkdir -p /app/data/downloads
chown -R appuser:appuser /app/data 2>/dev/null || true

# The whisper model cache is a named volume mounted at appuser's HOME; Docker
# creates the mount point root-owned, so appuser can't write the downloaded
# model there. Fix ownership before dropping privileges.
mkdir -p /home/appuser/.cache/faster_whisper
chown -R appuser:appuser /home/appuser/.cache 2>/dev/null || true

# Exec the CMD as appuser so the app never runs as root.
exec gosu appuser "$@"
