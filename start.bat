@echo off
setlocal EnableDelayedExpansion

:: ── 1. Ensure .env exists ────────────────────────────────────────────────────
if not exist .env (
    if not exist .env.example (
        echo ERROR: .env.example not found. Cannot bootstrap environment.
        exit /B 1
    )
    copy .env.example .env > nul
    echo .env created from .env.example
    echo IMPORTANT: Open .env and set OPENAI_API_KEY before re-running start.bat
    exit /B 1
)

:: ── 2. Start infrastructure and wait until all healthchecks pass ─────────────
echo Starting infrastructure (postgres / redis / milvus)...
docker compose up -d --wait
if %ERRORLEVEL% neq 0 (
    echo ERROR: Infrastructure failed to start. Check: docker compose logs
    exit /B 1
)
echo Infrastructure healthy.

:: ── 3. Ensure logs directory exists ─────────────────────────────────────────
if not exist logs mkdir logs

:: ── 4. Start uvicorn in background, capture PID via PowerShell ───────────────
echo Starting uvicorn...
powershell -NoProfile -Command ^
    "$proc = Start-Process -FilePath python ^
        -ArgumentList @('-m','uvicorn','main:app','--host','0.0.0.0','--port','8000','--reload') ^
        -RedirectStandardOutput 'logs\uvicorn.log' ^
        -RedirectStandardError  'logs\uvicorn_err.log' ^
        -WindowStyle Hidden ^
        -PassThru; ^
    $proc.Id | Set-Content -Path '.uvicorn.pid' -NoNewline"

if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to start uvicorn.
    exit /B 1
)

set /p UPID=<.uvicorn.pid
echo.
echo  doc-agent is running
echo  PID  : !UPID!
echo  API  : http://localhost:8000
echo  Docs : http://localhost:8000/docs
echo  Logs : logs\uvicorn.log
echo.
echo  Run stop.bat to shut down.

endlocal
