# -*- coding: utf-8 -*-
"""
FastAPI Web 服务 — 福彩推荐系统 Web 界面。

单账号登录 + REST API + 静态前端。
"""

from __future__ import annotations

import os
import secrets
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Request, HTTPException
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

# ============================================================
# 配置
# ============================================================

USERNAME = os.getenv("LOTTERY_USER", "admin")
PASSWORD = os.getenv("LOTTERY_PASS", "caipiao2026")
SESSION_COOKIE = "lottery_session"
SESSION_HOURS = 12

# ============================================================
# FastAPI 应用
# ============================================================

app = FastAPI(title="福彩推荐系统", version="2.0")

# 静态文件
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
os.makedirs(static_dir, exist_ok=True)


# ============================================================
# 登录验证
# ============================================================

def create_session() -> str:
    return secrets.token_urlsafe(32)


def check_auth(request: Request) -> bool:
    session = request.cookies.get(SESSION_COOKIE)
    return session is not None and session == request.app.state.session


def require_auth(request: Request):
    if not check_auth(request):
        raise HTTPException(status_code=401, detail="未登录")


# ============================================================
# 登录 API
# ============================================================

class LoginRequest(BaseModel):
    username: str
    password: str


@app.post("/api/login")
async def login(request: Request, data: LoginRequest):
    if data.username == USERNAME and data.password == PASSWORD:
        token = create_session()
        request.app.state.session = token
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
    request.app.state.session = None
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(SESSION_COOKIE)
    return resp


@app.get("/api/me")
async def me(request: Request):
    return {"logged_in": check_auth(request), "username": USERNAME if check_auth(request) else None}


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
async def api_predict(request: Request, code: str, method: str = "xgb"):
    require_auth(request)
    if code not in LOTTERY_CONFIGS:
        raise HTTPException(400, f"未知彩种: {code}")

    df = load_history(code)
    pipeline = UnifiedPipeline(code, method=method)
    pred = pipeline.predict(df)

    return {
        "code": code,
        "name": LOTTERY_CONFIGS[code].name,
        "method": method,
        "red_balls": pred.red_balls,
        "blue_ball": pred.blue_ball,
        "strategy": pred.strategy_used,
        "probabilities": [float(p) for p in pred.probabilities] if pred.probabilities is not None else None,
    }


@app.get("/api/history/{code}")
async def api_history(request: Request, code: str, limit: int = 30):
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
async def api_ranking(request: Request, code: str, window: int = 200, backtest: int = 50):
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
    app.state.session = None


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
