@echo off
echo ============================================================
echo   福彩推荐系统 API 服务启动脚本
echo ============================================================
echo.

set PORT=8000
if not "%1"=="" set PORT=%1

echo 检查端口 %PORT% 是否被占用...
netstat -ano | findstr :%PORT% >nul 2>&1
if %ERRORLEVEL%==0 (
    echo   端口 %PORT% 已被占用，正在释放...
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :%PORT%') do (
        taskkill /PID %%a /F >nul 2>&1
    )
    timeout /t 2 /nobreak >nul
)

echo 创建必要目录...
if not exist data mkdir data
if not exist model mkdir model
if not exist predict mkdir predict
if not exist logs mkdir logs
if not exist output mkdir output
if not exist config mkdir config

echo.
echo 正在启动服务...
echo   访问地址: http://localhost:%PORT%
echo   默认账号: admin / caipiao2026
echo   按 Ctrl+C 停止服务
echo ============================================================
echo.

cd /d "%~dp0"
python -m uvicorn src.api:app --host 0.0.0.0 --port %PORT%
