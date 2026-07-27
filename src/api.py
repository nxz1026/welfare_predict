# -*- coding: utf-8 -*-
"""
FastAPI Web 服务 — 福彩推荐系统 Web 界面。

单账号登录 + REST API + 静态前端。
基于 SQLite 持久化会话管理，支持多用户。
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

import numpy as np
from fastapi import FastAPI, Query, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config import LOTTERY_CONFIGS
from src.data_fetcher import load_history
from src.recommendation import RecommendationEngine
from src.unified_pipeline import UnifiedPipeline
from src.strategy_backtest import generate_ranking_report
from src.feature_engineering import compute_hot_cold_features, compute_skip_features
from src.analysis import generate_comprehensive_report
from src.session import create_session as db_create_session, validate_session, delete_session, cleanup_expired

# ============================================================
# 配置
# ============================================================

USERNAME = os.getenv("LOTTERY_USER", "admin")
PASSWORD = os.getenv("LOTTERY_PASS", "change-me-in-production")
SESSION_COOKIE = "lottery_session"
SESSION_HOURS = 12

# ============================================================
# FastAPI 应用
# ============================================================

app = FastAPI(title="福彩推荐系统", version="2.1")

# CORS 中间件（P2-02）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# 静态文件
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
os.makedirs(static_dir, exist_ok=True)


# ============================================================
# 登录验证（基于 SQLite 持久化会话）(P0-02)
# ============================================================


def check_auth(request: Request) -> bool:
    """检查请求是否携带有效会话。"""
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return False
    return validate_session(token) is not None


def require_auth(request: Request):
    """要求已登录，否则返回 401。"""
    if not check_auth(request):
        raise HTTPException(status_code=401, detail="未登录")


def get_current_user(request: Request) -> Optional[str]:
    """获取当前登录用户 ID，未登录返回 None。"""
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return None
    return validate_session(token)


# ============================================================
# 全局异常处理器（隐藏内部路径）（P2-02）
# ============================================================


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    from loguru import logger
    logger.exception(f"Unhandled exception on {request.url}")
    return JSONResponse(
        status_code=500,
        content={"detail": "服务器内部错误，请稍后重试"},
    )


# ============================================================
# 健康检查端点（P3-03）
# ============================================================


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "version": "2.1.0",
    }


# ============================================================
# 登录 API
# ============================================================


class LoginRequest(BaseModel):
    username: str
    password: str


@app.post("/api/login")
async def login(request: Request, data: LoginRequest):
    if data.username == USERNAME and data.password == PASSWORD:
        # 创建持久化会话（SQLite 存储）
        token = db_create_session(user_id=data.username, ttl_seconds=SESSION_HOURS * 3600)
        resp = JSONResponse({"ok": True})
        resp.set_cookie(
            SESSION_COOKIE,
            token,
            max_age=SESSION_HOURS * 3600,
            httponly=True,
            samesite="lax",
        )
        return resp
    raise HTTPException(status_code=401, detail="用户名或密码错误")


@app.post("/api/logout")
async def logout(request: Request):
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        delete_session(token)
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(SESSION_COOKIE)
    return resp


@app.get("/api/me")
async def me(request: Request):
    user_id = get_current_user(request)
    return {
        "logged_in": user_id is not None,
        "username": user_id,
    }


# ============================================================
# 业务 API（需要登录）
# ============================================================


@app.get("/api/recommend/{code}")
async def api_recommend(request: Request, code: str):
    require_auth(request)
    if code not in LOTTERY_CONFIGS:
        raise HTTPException(400, f"未知彩种: {code}")

    df = load_history(code)
    engine = RecommendationEngine(code)
    rec = engine.generate(df)

    return {
        "code": code,
        "name": LOTTERY_CONFIGS[code].name,
        "timestamp": datetime.now().isoformat(),
        "total_periods": len(df),
        "strategies": [
            {
                "name": s.strategy_name,
                "red_balls": s.red_balls,
                "blue_ball": s.blue_ball,
                "analysis": s.analysis,
            }
            for s in rec.strategies
        ],
    }


@app.get("/api/predict/{code}")
async def api_predict(
    request: Request,
    code: str,
    method: str = "xgb",
):
    require_auth(request)
    if code not in LOTTERY_CONFIGS:
        raise HTTPException(400, f"未知彩种: {code}")

    try:
        df = load_history(code)
        pipeline = UnifiedPipeline(code, method=method)
        pred = pipeline.predict(df)
    except FileNotFoundError as e:
        raise HTTPException(404, f"模型未训练，请先执行训练脚本: {code}/{method}")
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        from loguru import logger
        logger.error(f"预测失败: {e}")
        raise HTTPException(500, detail="预测失败，请查看服务日志")

    probas = pred.probabilities
    if probas is not None:
        probas = np.nan_to_num(probas, nan=0.0, posinf=0.0, neginf=0.0)
        probas = probas / probas.sum() if probas.sum() > 0 else probas

    return {
        "code": code,
        "name": LOTTERY_CONFIGS[code].name,
        "method": method,
        "red_balls": pred.red_balls,
        "blue_ball": pred.blue_ball,
        "strategy": pred.strategy_used,
        "probabilities": [float(p) for p in probas] if probas is not None else None,
    }


@app.get("/api/history/{code}")
async def api_history(
    request: Request,
    code: str,
    limit: int = Query(default=30, ge=1, le=200),  # P2-02 参数边界校验
):
    require_auth(request)
    if code not in LOTTERY_CONFIGS:
        raise HTTPException(400, f"未知彩种: {code}")

    df = load_history(code)
    recent = df.tail(limit)

    red_cols = [f"红球_{i+1}" for i in range(LOTTERY_CONFIGS[code].red.sequence_len)]
    records = []
    for _, row in recent.iterrows():
        record = {
            "issue": str(row["期数"]),
            "date": str(row.get("开奖日期", "")),
            "reds": [int(row[c]) for c in red_cols],
        }
        if "蓝球_1" in df.columns:
            record["blue"] = int(row["蓝球_1"])
        records.append(record)

    return {
        "code": code,
        "name": LOTTERY_CONFIGS[code].name,
        "total": len(df),
        "records": records,
    }


@app.get("/api/stats/{code}")
async def api_stats(request: Request, code: str):
    require_auth(request)
    if code not in LOTTERY_CONFIGS:
        raise HTTPException(400, f"未知彩种: {code}")

    df = load_history(code)
    config = LOTTERY_CONFIGS[code]

    # 频率统计
    hot_cold = compute_hot_cold_features(df, config, window=30)
    last_hot = hot_cold.iloc[-1]

    # 遗漏统计
    skip = compute_skip_features(df, config)
    last_skip = skip.iloc[-1]

    # 组装图表数据
    numbers = list(range(1, config.red.num_classes + 1))
    hot_values = [int(last_hot[f"hot_count_{n}"]) for n in numbers]
    skip_values = [int(last_skip[f"skip_{n}"]) for n in numbers]

    return {
        "code": code,
        "name": config.name,
        "total_periods": len(df),
        "numbers": numbers,
        "hot_values": hot_values,
        "skip_values": skip_values,
    }


@app.get("/api/ranking/{code}")
async def api_ranking(
    request: Request,
    code: str,
    window: int = Query(default=200, ge=10, le=1000),
    backtest: int = Query(default=50, ge=5, le=200),
):
    require_auth(request)
    if code not in LOTTERY_CONFIGS:
        raise HTTPException(400, f"未知彩种: {code}")

    report = generate_ranking_report(code, window, backtest)

    performances = {}
    for name, perf in report.performances.items():
        performances[name] = {
            "avg_match": perf.avg_match,
            "blue_match": perf.blue_match_count,
            "roi": perf.roi,
            "total_bets": perf.total_bets,
            "prize_counts": perf.prize_counts,
        }

    return {
        "code": code,
        "name": LOTTERY_CONFIGS[code].name,
        "window": window,
        "total_windows": report.total_windows,
        "random_baseline": report.random_baseline,
        "performances": performances,
    }


@app.get("/api/report/{code}")
async def api_report(request: Request, code: str):
    require_auth(request)
    if code not in LOTTERY_CONFIGS:
        raise HTTPException(400, f"未知彩种: {code}")

    report = generate_comprehensive_report(code, window_size=200, n_backtest=50)
    return {"code": code, "report": report}


# ============================================================
# 前端页面
# ============================================================


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    if not check_auth(request):
        return RedirectResponse("/login")
    return open(os.path.join(static_dir, "index.html"), encoding="utf-8").read()


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    if check_auth(request):
        return RedirectResponse("/")
    return open(os.path.join(static_dir, "login.html"), encoding="utf-8").read()


# 静态资源
app.mount("/static", StaticFiles(directory=static_dir), name="static")


# ============================================================
# 启动
# ============================================================


@app.on_event("startup")
async def startup():
    """启动时清理过期会话并初始化数据库。"""
    from src.session import _get_conn
    _get_conn()  # 确保 sessions 表存在
    cleaned = cleanup_expired()
    if cleaned > 0:
        from loguru import logger
        logger.info(f"清理了 {cleaned} 个过期会话")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
