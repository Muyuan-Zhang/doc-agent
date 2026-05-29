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
    echo .uvicorn.pid not found. Attempting fallback via WMI process search...
    powershell -NoProfile -Command "Get-WmiObject Win32_Process -Filter 'Name=''python.exe''' | Where-Object { $_.CommandLine -like '*uvicorn*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
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
