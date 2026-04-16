@echo off
echo.
echo ============================================
echo  Zoom Transcriber - Rebuild and Restart
echo ============================================
echo.
cd /d "%~dp0"
echo [1/2] Building updated image (fast - uses cache)...
docker compose up -d --build
echo.
echo [2/2] Waiting for server to be ready...
timeout /t 5 /nobreak >nul
echo.
echo ============================================
echo  Done! Open: http://localhost:8000
echo ============================================
echo.
pause
