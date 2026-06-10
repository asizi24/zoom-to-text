# ============================================================================
#  run_local_gpu.ps1 — run Zoom-to-Text natively on Windows, using the RTX 4070 Ti.
#
#    * Transcription runs on the GPU (Whisper cuda / float16) — ~10-20x faster
#      than the Docker CPU path, and float16 is higher precision than int8, so
#      quality goes UP, not down.
#    * Summary / exam / flashcards / chat run on the host Ollama (mistral-nemo:12b,
#      clean Hebrew + 128k context).
#    * One-click loopback login is enabled (no Resend email needed).
#    * There is NO Docker bind-mount here, so the SQLite "disk I/O error" that
#      bit the container cannot happen in this mode.
#
#  The Docker container and this native server both use port 8000 and ./data, so
#  only one may run at a time. This script stops the container first.
#
#  Usage:  right-click -> Run with PowerShell   (or)   pwsh ./run_local_gpu.ps1
#  Stop:   press Ctrl+C in this window.
# ============================================================================
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

# --- Free port 8000 + release ./data/tasks.db: stop the Docker container -------
$running = docker ps --filter name=zoom_transcriber --format '{{.Names}}' 2>$null
if ($running) {
    Write-Host 'Stopping the Docker container (frees port 8000 + the DB)...' -ForegroundColor Yellow
    docker compose stop | Out-Null
}

# --- Native runtime config (env vars override values in .env) ------------------
$env:LLM_PROVIDER         = 'ollama'                 # summary/exam via local Ollama
$env:OLLAMA_MODEL         = 'mistral-nemo:12b'       # clean Hebrew, 128k context
$env:OLLAMA_BASE_URL      = 'http://localhost:11434' # Ollama is local now (not in Docker)
$env:OLLAMA_NUM_CTX       = '24576'                  # fits 12GB VRAM + leaves room for output
$env:LECTURE_LANGUAGE     = 'he'                     # force Hebrew output (local model drifts to English on tech terms)
$env:WHISPER_DEVICE       = 'cuda'                   # <-- the RTX 4070 Ti
$env:WHISPER_COMPUTE_TYPE = 'float16'                # GPU precision (better than int8)
$env:ENABLE_DEV_LOGIN     = 'true'                   # one-click local login
$env:BASE_URL             = 'http://localhost:8000'

Write-Host ''
Write-Host '  Zoom-to-Text — NATIVE GPU mode' -ForegroundColor Green
Write-Host '  Transcription : GPU (cuda/float16)         LLM : Ollama mistral-nemo:12b' -ForegroundColor Gray
Write-Host '  App           : http://localhost:8000' -ForegroundColor Cyan
Write-Host '  One-click login: http://localhost:8000/api/auth/dev-login' -ForegroundColor Cyan
Write-Host ''

& "$PSScriptRoot\venv\Scripts\python.exe" -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
