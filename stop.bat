@echo off
setlocal EnableDelayedExpansion

:: ── 1. Kill uvicorn by saved PID ────────────────────────────────────────────
if exist .uvicorn.pid (
    set /p UPID=<.uvicorn.pid
    echo Stopping uvicorn (PID !UPID!)...
    taskkill /PID !UPID! /F > nul 2>&1
    if %ERRORLEVEL% equ 0 (
        echo uvicorn stopped.
    ) else (
        echo WARNING: Process !UPID! not found ^(may have already exited^).
    )
    del .uvicorn.pid
) else (
    echo .uvicorn.pid not found. Attempting fallback taskkill by window title...
    taskkill /FI "WINDOWTITLE eq uvicorn*" /F > nul 2>&1
    echo Fallback complete ^(process may have already exited^).
)

:: ── 2. Stop infrastructure ───────────────────────────────────────────────────
echo Stopping infrastructure...
docker compose down
if %ERRORLEVEL% neq 0 (
    echo WARNING: docker compose down reported an error.
)

echo.
echo  doc-agent stopped.

endlocal
