@echo off
chcp 65001 >nul 2>&1

:restart
echo ============================================================
echo   Lottery Predict API Server
echo ============================================================
echo.

REM ---- Port Config ----
REM Default 8000 for local dev; DevCloud requires 8080
REM Override: start_server.bat 8080
set PORT=8000
if not "%1"=="" set PORT=%1

REM ---- Virtual Environment ----
if exist ".venv\Scripts\activate.bat" (
    echo [INFO] .venv detected, activating...
    call .venv\Scripts\activate.bat
) else (
    echo [WARN] .venv not found, using system Python
)

REM ---- Env File ----
if not exist ".env" (
    echo [WARN] .env not found, using defaults
    echo        Copy .env.example to .env:  copy .env.example .env
    echo.
)

REM ---- Port Check ----
echo [INFO] Checking port %PORT%...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%PORT% " ^| findstr "LISTENING"') do (
    echo [WARN] Port %PORT% in use by PID %%a, releasing...
    taskkill /PID %%a /F >nul 2>&1
    timeout /t 2 /nobreak >nul
)

REM ---- Create Directories ----
echo [INFO] Creating data directories...
for %%d in (data\ssq data\sd data\3d data\qlc data\users model predict output logs config) do (
    if not exist "%%d" mkdir "%%d"
)

echo.
echo ============================================================
echo   Starting server...
echo   URL: http://localhost:%PORT%
echo   Login: admin / (password set in .env)
echo   Stop:  Ctrl+C
echo ============================================================
echo.

cd /d "%~dp0"
python -m uvicorn src.api:app --host 0.0.0.0 --port %PORT%

echo.
echo ============================================================
echo   Server stopped. Restarting in 3s...
echo   Press Ctrl+C twice to exit
echo ============================================================
timeout /t 3 /nobreak >nul
goto restart