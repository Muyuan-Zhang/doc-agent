@echo off
setlocal EnableDelayedExpansion

if exist .uvicorn.pid (
    set /p UPID=<.uvicorn.pid
    echo Stopping uvicorn PID !UPID!...
    taskkill /PID !UPID! /T /F > nul 2>&1
    del .uvicorn.pid
) else (
    echo .uvicorn.pid not found, skipping PID kill.
)

echo Killing any remaining uvicorn processes...
powershell -NoProfile -Command "Get-WmiObject Win32_Process -Filter 'Name=''python.exe''' | Where-Object { $_.CommandLine -like '*uvicorn*' } | ForEach-Object { Write-Host ('Killing PID ' + $_.ProcessId); Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"

echo Stopping infrastructure...
docker compose down
if %ERRORLEVEL% neq 0 (
    echo WARNING: docker compose down reported an error.
)

echo.
echo doc-agent stopped.

endlocal