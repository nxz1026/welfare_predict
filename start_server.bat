@echo off
chcp 65001 >nul 2>&1

:restart
echo ============================================================
echo   Welfare Predict API Server Startup
echo ============================================================
echo.

set PORT=8000
if not "%1"=="" set PORT=%1

echo Checking if port %PORT% is in use...
netstat -ano | findstr :%PORT% >nul 2>&1
if %ERRORLEVEL%==0 (
    echo   Port %PORT% is in use, releasing...
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :%PORT%') do (
        taskkill /PID %%a /F >nul 2>&1
    )
    timeout /t 2 /nobreak >nul
)

echo Creating required directories...
if not exist data mkdir data
if not exist model mkdir model
if not exist predict mkdir predict
if not exist logs mkdir logs
if not exist output mkdir output
if not exist config mkdir config

echo.
echo Starting server...
echo   URL: http://localhost:%PORT%
echo   Default login: admin / 12333 (change via .env)
echo   Press Ctrl+C to stop
echo ============================================================
echo.

cd /d "%~dp0"
python -m uvicorn src.api:app --host 0.0.0.0 --port %PORT%

echo.
echo ============================================================
echo   Server stopped. Restarting in 3 seconds...
echo   Press Ctrl+C twice to exit completely
echo ============================================================
timeout /t 3 /nobreak >nul
goto restart