@echo off
echo Starting 福彩推荐系统 API server...
cd /d "%~dp0"
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000