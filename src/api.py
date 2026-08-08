# -*- coding: utf-8 -*-
"""
FastAPI Web 服务 — 福彩推荐系统 Web 界面。

单账号登录 + REST API + 静态前端。
基于 SQLite 持久化会话管理，支持多用户。
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

import numpy as np
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, Query, Request, HTTPException
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
PASSWORD = os.getenv("LOTTERY_PASS", "")
SESSION_COOKIE = "lottery_session"
SESSION_HOURS = 12
DEBUG = os.getenv("DEBUG", "true").lower() in ("true", "1", "yes")

# ============================================================
# FastAPI 应用
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时初始化，关闭时清理。"""
    # 安全检查：密码为空时警告但不阻止启动（云环境可能通过其他方式保护）
    if not PASSWORD:
        from loguru import logger
        logger.warning(
            "LOTTERY_PASS 环境变量未设置！登录功能将不可用。"
            "请在环境变量中设置登录密码。参考 .env.example 获取配置说明。"
        )
    from src.bootstrap import bootstrap
    bootstrap()
    cleaned = cleanup_expired()
    if cleaned > 0:
        from loguru import logger
        logger.info(f"清理了 {cleaned} 个过期会话")

    # 启动定时调度器（每日 01:03 BJT 数据同步 + 智能训练）
    scheduler = None
    try:
        from src.scheduler import setup_scheduler
        scheduler = setup_scheduler()
        scheduler.start()
        from loguru import logger
        logger.info("定时调度器已启动")
    except Exception as e:
        from loguru import logger
        logger.warning(f"定时调度器启动失败（定时任务不可用）: {e}")

    yield

    # 关闭调度器
    if scheduler:
        scheduler.shutdown(wait=False)
        from loguru import logger
        logger.info("定时调度器已关闭")

app = FastAPI(title="福彩推荐系统", version="2.1", lifespan=lifespan)

# API v1 路由（P5-24）
v1 = APIRouter(prefix="/api/v1")

# CORS 中间件（P2-02）
# DevCloud 部署时通过 CORS_ORIGINS 环境变量设置允许的源，多个源用逗号分隔
_cors_origins = os.environ.get("CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors_origins.split(",") if o.strip()],
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


def require_auth(request: Request) -> None:
    """登录验证：未登录时返回 401。"""
    if not check_auth(request):
        raise HTTPException(status_code=401, detail="未登录或会话已过期")


def _get_unavailable_methods() -> dict:
    """返回当前环境不可用的训练方法及原因。"""
    unavailable = {}
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        unavailable["lstm"] = "TensorFlow 未安装，LSTM 方法不可用"
        unavailable["stacking"] = "TensorFlow 未安装，Stacking 方法需要 LSTM 基学习器"
    return unavailable


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


@v1.post("/login")
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
            secure=not DEBUG,  # 生产环境启用 secure 标志
        )
        return resp
    raise HTTPException(status_code=401, detail="用户名或密码错误")


@v1.post("/logout")
async def logout(request: Request):
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        delete_session(token)
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(SESSION_COOKIE)
    return resp


@v1.get("/me")
async def me(request: Request):
    user_id = get_current_user(request)
    return {
        "logged_in": user_id is not None,
        "username": user_id,
    }


# ============================================================
# 业务 API（需要登录）
# ============================================================


@v1.get("/recommend/{code}")
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
                "name": s.display_name or s.strategy_name,
                "red_balls": s.red_balls,
                "blue_ball": s.blue_ball,
                "analysis": s.analysis,
            }
            for s in rec.strategies
        ],
    }


class CustomRecommendRequest(BaseModel):
    strategy: str = "热号追踪"
    user_reds: list[int] = []
    user_blue: Optional[int] = None


# 策略映射常量
_ML_METHODS = {
    "XGBoost ML": "xgb", "LSTM ML": "lstm", "Poisson ML": "poisson", "Stacking ML": "stacking",
    "xgb": "xgb", "lstm": "lstm", "poisson": "poisson", "stacking": "stacking",
}
_RULE_METHODS = {
    "热号追踪": "conservative", "冷门博击": "aggressive",
    "和值精选": "balanced", "幸运号码": "mystic",
    "conservative": "conservative", "aggressive": "aggressive",
    "balanced": "balanced", "mystic": "mystic",
}


def _validate_user_numbers(
    user_reds: list[int], user_blue: Optional[int], config
) -> None:
    """验证用户输入的号码，无效时抛出 ValueError。"""
    errors = []
    if user_reds:
        if len(user_reds) > config.red.sequence_len:
            errors.append(f"最多输入 {config.red.sequence_len} 个号码")
        elif len(set(user_reds)) != len(user_reds):
            errors.append("号码不能重复")
        elif any(r < config.red.min_val or r > config.red.max_val for r in user_reds):
            errors.append(f"号码范围: {config.red.min_val}-{config.red.max_val}")
        if config.blue and user_blue is not None and (user_blue < config.blue.min_val or user_blue > config.blue.max_val):
            errors.append(f"蓝球范围: {config.blue.min_val}-{config.blue.max_val}")
        if errors:
            raise ValueError("；".join(errors))


def _generate_custom_recommendation(
    code: str, strategy: str, user_reds: list[int], user_blue: Optional[int]
) -> dict:
    """自定义推荐核心业务逻辑（service 层）。"""
    import numpy as np
    from src.recommendation import RecommendationEngine
    from src.unified_pipeline import UnifiedPipeline
    from src.feature_engineering import compute_hot_cold_features, compute_skip_features

    config = LOTTERY_CONFIGS[code]
    _validate_user_numbers(user_reds, user_blue, config)

    is_ml = strategy in _ML_METHODS
    is_rule = strategy in _RULE_METHODS
    if not is_ml and not is_rule:
        raise ValueError(f"未知策略: {strategy}")

    strategy_display = strategy
    df = load_history(code)

    if is_ml:
        all_reds, blue, reason, analysis = _generate_ml_recommendation(
            code, strategy, user_reds, user_blue, config, df
        )
    else:
        all_reds, blue, reason, analysis = _generate_rule_recommendation(
            code, strategy, user_reds, user_blue, config, df
        )

    response = {
        "code": code,
        "name": config.name,
        "strategy_name": strategy,
        "display_name": strategy_display,
        "recommended_reds": all_reds,
        "recommended_blue": blue,
        "reason": reason,
        "analysis": analysis,
    }

    if user_reds:
        locked = set(user_reds)
        rec_set = set(all_reds)
        hits = locked & rec_set
        response["user_reds"] = user_reds
        response["user_blue"] = user_blue
        response["match_count"] = len(hits)
        response["match_numbers"] = sorted(hits)
        response["locked"] = len(locked)

    return response


def _generate_ml_recommendation(
    code: str, strategy: str, user_reds: list[int], user_blue: Optional[int],
    config, df
) -> tuple:
    """ML 策略推荐逻辑。"""
    import numpy as np
    from src.unified_pipeline import UnifiedPipeline

    ml_method = _ML_METHODS[strategy]
    pipeline = UnifiedPipeline(code, method=ml_method)
    if not hasattr(pipeline, 'is_trained') or not pipeline.is_trained:
        try:
            pipeline.train(df)
        except Exception:
            raise ValueError(f"{strategy} 模型未训练，请先在 AI 预测页面执行训练")

    pred = pipeline.predict(df)
    probas = pred.probabilities
    if probas is not None:
        probas = np.nan_to_num(probas, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        probas = np.ones(config.red.num_classes) / config.red.num_classes

    locked = set(user_reds)
    available = [n for n in range(config.red.min_val, config.red.min_val + config.red.num_classes) if n not in locked]
    available_probas = [(n, probas[n - 1]) for n in available]
    available_probas.sort(key=lambda x: -x[1])
    needed = config.red.sequence_len - len(locked)
    extra = [n for n, _ in available_probas[:needed]]

    all_reds = sorted(user_reds + extra) if user_reds else sorted(extra)
    blue = user_blue if user_blue is not None else (int(pred.blue_ball) if pred.blue_ball else 1)

    top5 = [(n, probas[n - 1]) for n, _ in available_probas[:5]]
    reason_parts = [
        f"【{strategy}】基于历史数据的机器学习概率模型，对 1-33 每个号码计算出现概率。"
    ]
    if top5:
        reason_parts.append(f"模型判断概率最高的前 5 个号码为：{'、'.join(f'#{n}({p*100:.1f}%)' for n, p in top5)}。")
    if user_reds:
        user_probas = [(n, probas[n - 1]) for n in user_reds]
        avg_user_prob = sum(p for _, p in user_probas) / len(user_probas) if user_probas else 0
        reason_parts.append(f"您选择的号码平均概率为 {avg_user_prob*100:.1f}%")
        if extra:
            reason_parts.append(f"AI 为您补充了号码 {'、'.join(f'#{n}' for n in extra)}，与您的选择组合成最终推荐。")
    if user_blue is None:
        blue_probas = probas[config.red.num_classes:] if len(probas) > config.red.num_classes else probas[:config.blue.num_classes]
        best_blue = int(np.argmax(blue_probas)) + 1 if len(blue_probas) > 0 else 1
        blue = best_blue
        reason_parts.append(f"蓝球推荐 #{best_blue}（模型评分最高）。")
    else:
        reason_parts.append(f"蓝球已锁定 #{user_blue}。")

    reason = " ".join(reason_parts)
    analysis = {"method": strategy, "mode": "ML概率"}
    return all_reds, blue, reason, analysis


def _generate_rule_recommendation(
    code: str, strategy: str, user_reds: list[int], user_blue: Optional[int],
    config, df
) -> tuple:
    """规则策略推荐逻辑。"""
    from src.recommendation import RecommendationEngine
    from src.feature_engineering import compute_hot_cold_features, compute_skip_features

    strategy_key = _RULE_METHODS[strategy]
    engine = RecommendationEngine(code)
    rec = engine.generate(df)

    result = None
    for s in rec.strategies:
        if s.strategy_name == strategy_key:
            result = s
            break
    if not result:
        result = rec.strategies[0]

    locked = set(user_reds)
    if user_reds:
        extra = [n for n in result.red_balls if n not in locked]
        needed = config.red.sequence_len - len(locked)
        extra = extra[:needed]
        while len(extra) < needed:
            fallback = [n for n in range(config.red.min_val, config.red.min_val + config.red.num_classes) if n not in locked and n not in extra]
            extra.append(fallback[len(extra) % len(fallback)])
        all_reds = sorted(user_reds + extra)
    else:
        all_reds = result.red_balls

    blue = user_blue if user_blue is not None else int(result.blue_ball or 1)

    hot_cold = compute_hot_cold_features(df, config, window=7)
    skip = compute_skip_features(df, config)
    last_hot = hot_cold.iloc[-1] if len(hot_cold) > 0 else None
    last_skip = skip.iloc[-1] if len(skip) > 0 else None

    if strategy_key == "conservative":
        hot_counts = [(n, int(last_hot[f"hot_count_{n}"])) for n in all_reds] if last_hot is not None else []
        hot_str = "、".join(f"#{n}({c}次)" for n, c in hot_counts[:3]) if hot_counts else ""
        reason = (
            f"【热号追踪】基于近 7 期开奖号码的频率统计，选取出现次数最多的号码。"
            f"其中 {hot_str} 出现最为频繁，说明这些号码近期热度持续。"
            f"红球和值 {sum(all_reds)}，奇偶比 {sum(1 for x in all_reds if x%2)}:{sum(1 for x in all_reds if x%2==0)}，"
            f"符合近期热号分布规律。如果您有自已的心水号码，AI 会锁定您的选择并围绕它们补充热号。"
        )
    elif strategy_key == "aggressive":
        skip_vals = [(n, int(last_skip[f"skip_{n}"])) for n in all_reds] if last_skip is not None else []
        max_skip_all = max(int(last_skip[f"skip_{n}"]) for n in range(config.red.min_val, config.red.min_val + config.red.num_classes)) if last_skip is not None else 0
        skip_str = "、".join(f"#{n}(遗漏{c}期)" for n, c in skip_vals[:3]) if skip_vals else ""
        reason = (
            f"【冷门博击】追踪遗漏值最大的号码，这些号码长时间未开出，"
            f"根据概率回补规律有较大概率出现。当前全池最大遗漏 {max_skip_all} 期，"
            f"推荐中包含 {skip_str} 等深度冷号。"
            f"建议用冷号博击高回报，同时适当搭配热号降低风险。"
        )
    elif strategy_key == "balanced":
        avg_sum = result.analysis.get("target_sum", 102) if result else 102
        reason = (
            f"【和值精选】以红球和值 {avg_sum} 为目标，选择奇偶比例均衡（"
            f"{sum(1 for x in all_reds if x%2)}:{sum(1 for x in all_reds if x%2==0)}）的组合。"
            f"实际和值 {sum(all_reds)}，与目标偏差 {abs(sum(all_reds) - avg_sum)}，属于合理范围。"
            f"这种均衡型组合在历史开奖中覆盖率最高。"
        )
    elif strategy_key == "mystic":
        reason = (
            f"【幸运号码】基于您选择或系统生成的幸运数字组合。"
            f"本策略参考了质数分布（质数 {sum(1 for x in all_reds if x in (2,3,5,7,11,13,17,19,23,29,31))} 个）、"
            f"AC 值等非传统指标，为您提供具有个性化特征的号码组合。"
        )
    else:
        reason = result.confidence_note

    return all_reds, blue, reason, result.analysis


@v1.post("/custom-recommend/{code}")
async def api_custom_recommend(request: Request, code: str, data: CustomRecommendRequest):
    require_auth(request)
    if code not in LOTTERY_CONFIGS:
        raise HTTPException(400, f"未知彩种: {code}")
    try:
        return await asyncio.to_thread(
            _generate_custom_recommendation, code, data.strategy, data.user_reds, data.user_blue
        )
    except ValueError as e:
        raise HTTPException(400, str(e))


@v1.get("/predict/{code}")
async def api_predict(
    request: Request,
    code: str,
    method: str = "xgb",
):
    require_auth(request)
    if code not in LOTTERY_CONFIGS:
        raise HTTPException(400, f"未知彩种: {code}")

    try:
        df = await asyncio.to_thread(load_history, code)
        pipeline = UnifiedPipeline(code, method=method)
        pred = await asyncio.to_thread(pipeline.predict, df)
    except FileNotFoundError as e:
        raise HTTPException(404, f"模型未训练，请先执行训练: {code}/{method}")
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


@v1.post("/train/{code}")
async def api_train(
    request: Request,
    code: str,
    method: str = "xgb",
):
    require_auth(request)
    if code not in LOTTERY_CONFIGS:
        raise HTTPException(400, f"未知彩种: {code}")

    # 检查方法是否可用（LSTM/Stacking 需要 TensorFlow）
    unavailable_methods = _get_unavailable_methods()
    if method in unavailable_methods:
        raise HTTPException(
            400,
            detail=f"训练方法 '{method}' 当前不可用: {unavailable_methods[method]}",
        )

    try:
        df = await asyncio.to_thread(load_history, code)
        pipeline = UnifiedPipeline(code, method=method)
        summary = await asyncio.to_thread(pipeline.train, df)
        return {
            "ok": True,
            "code": summary.code,
            "name": summary.name,
            "method": summary.method,
            "n_samples": summary.n_samples,
            "n_features": summary.n_features,
            "metrics": summary.metrics,
        }
    except ImportError as e:
        from loguru import logger
        logger.warning(f"训练方法不可用: {e}")
        raise HTTPException(400, detail=f"训练方法不可用: {e}")
    except Exception as e:
        from loguru import logger
        logger.error(f"训练失败: {e}")
        raise HTTPException(500, detail=f"训练失败: {e}")


@v1.post("/train/{code}/all")
async def api_train_all(request: Request, code: str):
    """批量训练所有可用方法（xgb + poisson + stacking/lstm 如可用）。"""
    require_auth(request)
    if code not in LOTTERY_CONFIGS:
        raise HTTPException(400, f"未知彩种: {code}")

    unavailable = _get_unavailable_methods()
    methods = ["xgb", "poisson"]
    # stacking/lstm 需要 TensorFlow
    if "stacking" not in unavailable:
        methods.append("stacking")
    if "lstm" not in unavailable:
        methods.append("lstm")

    df = await asyncio.to_thread(load_history, code)
    results = []
    for m in methods:
        try:
            pipeline = UnifiedPipeline(code, method=m)
            summary = await asyncio.to_thread(pipeline.train, df)
            results.append({
                "method": m,
                "ok": True,
                "n_samples": summary.n_samples,
                "n_features": summary.n_features,
                "metrics": summary.metrics,
            })
        except Exception as e:
            from loguru import logger
            logger.error(f"训练 {m} 失败: {e}")
            results.append({"method": m, "ok": False, "error": str(e)})

    return {
        "code": code,
        "name": LOTTERY_CONFIGS[code].name,
        "results": results,
        "trained_methods": [r["method"] for r in results if r["ok"]],
        "failed_methods": [r["method"] for r in results if not r["ok"]],
    }


@v1.get("/train/methods")
async def api_train_methods(request: Request):
    """查询可用的训练方法。"""
    require_auth(request)
    all_methods = ["xgb", "poisson", "stacking", "lstm"]
    unavailable = _get_unavailable_methods()
    return {
        "methods": [
            {
                "id": m,
                "name": {"xgb": "XGBoost", "poisson": "泊松分布", "stacking": "Stacking 集成", "lstm": "LSTM/MLP"}[m],
                "available": m not in unavailable,
                "reason": unavailable.get(m, ""),
            }
            for m in all_methods
        ]
    }


@v1.get("/train/{code}/status")
async def api_train_status(
    request: Request,
    code: str,
    method: str = "xgb",
):
    require_auth(request)
    if code not in LOTTERY_CONFIGS:
        raise HTTPException(400, f"未知彩种: {code}")

    import json
    from pathlib import Path
    from src.config import PATHS
    from src.scheduler import load_train_status

    df = load_history(code)
    if df.empty:
        return {"already_trained": False, "issues_count": 0}
    current_issues = sorted(df["期数"].astype(str).tolist())

    model_dir = Path(PATHS["data"]).parent / "model" / code / method
    summary_path = model_dir / "summary.json"
    if not summary_path.exists():
        return {"already_trained": False, "issues_count": len(current_issues)}

    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)

    trained_issues = sorted(summary.get("trained_issues", []))
    # 训练集用的是最近 N 期（df 降序取前 split_idx 行），
    # 因此比较 current_issues（升序）的后 N 个
    if trained_issues and trained_issues == current_issues[-len(trained_issues):]:
        result = {
            "already_trained": True,
            "n_samples": summary.get("n_samples", 0),
            "issues_count": len(current_issues),
            "method": method,
        }
        # 附加 train_status.json 中的信息
        ts = load_train_status(code)
        if ts:
            result["last_trained_at"] = ts.get("last_trained_at")
            result["trained_methods"] = ts.get("trained_methods", [])
            result["failed_methods"] = ts.get("failed_methods", [])
            result["training_ok"] = ts.get("training_ok", True)
        return result
    return {"already_trained": False, "issues_count": len(current_issues)}


def _run_data_update(code: str) -> dict:
    """下载最新数据并与本地数据增量合并去重。"""
    from src.config import PATHS
    from src.data_fetcher import download_history
    from pathlib import Path
    import pandas as pd

    data_path = Path(PATHS["data"]) / code / "data.csv"

    # 记录已有期号（用于计算新增数量）
    existing_issues: set = set()
    if data_path.exists():
        try:
            existing = pd.read_csv(data_path, encoding="utf-8-sig")
        except Exception:
            try:
                existing = pd.read_csv(data_path, encoding="utf-8")
            except Exception:
                existing = None
        if existing is not None and "期数" in existing.columns:
            existing_issues = set(existing["期数"].astype(str).tolist())

    # 增量下载并合并（merge=True 避免覆盖丢失历史数据）
    meta = download_history(code, merge=True)

    # 读取合并后的数据计算新增期数
    try:
        fresh = pd.read_csv(data_path, encoding="utf-8-sig")
    except Exception:
        try:
            fresh = pd.read_csv(data_path, encoding="utf-8")
        except Exception:
            fresh = pd.DataFrame()

    new_issues = set(fresh["期数"].astype(str).tolist()) - existing_issues if "期数" in fresh.columns and existing_issues else set()
    return {
        "ok": True,
        "code": code,
        "total_issues": len(fresh) if "期数" in fresh.columns else meta.total_issues,
        "new_issues": len(new_issues),
    }


@v1.post("/data/update/{code}")
async def api_data_update(
    request: Request,
    code: str,
):
    require_auth(request)
    if code not in LOTTERY_CONFIGS:
        raise HTTPException(400, f"未知彩种: {code}")

    return await asyncio.to_thread(_run_data_update, code)


@v1.get("/history/{code}")
async def api_history(
    request: Request,
    code: str,
    limit: int = Query(default=30, ge=1, le=200),  # P2-02 参数边界校验
    offset: int = Query(default=0, ge=0),  # P5-25 分页偏移
):
    require_auth(request)
    if code not in LOTTERY_CONFIGS:
        raise HTTPException(400, f"未知彩种: {code}")

    import pandas as pd
    df = load_history(code)
    total = len(df)
    recent = df.iloc[offset: offset + limit]

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
        if "试机号" in df.columns and pd.notna(row.get("试机号")):
            record["try_code"] = str(int(row["试机号"]))
        if "开奖号码" in df.columns and pd.notna(row.get("开奖号码")):
            record["winning_number"] = str(row["开奖号码"])
        records.append(record)

    return {
        "code": code,
        "name": LOTTERY_CONFIGS[code].name,
        "total": total,
        "offset": offset,
        "limit": limit,
        "records": records,
    }


@v1.get("/stats/{code}")
async def api_stats(
    request: Request,
    code: str,
    periods: int = Query(default=7, ge=3, le=200),
):
    require_auth(request)
    if code not in LOTTERY_CONFIGS:
        raise HTTPException(400, f"未知彩种: {code}")

    df = load_history(code)
    config = LOTTERY_CONFIGS[code]
    offset = config.red.min_val

    # 3D 按位置统计（百位/十位/个位）
    if code == "sd":
        recent = df.head(periods)
        pos_names = ["百位", "十位", "个位"]
        hot_by_pos = []
        skip_by_pos = []
        for p in range(3):
            col = f"红球_{p+1}"
            # 频率
            counts = recent[col].value_counts()
            hot_pos = [int(counts.get(d, 0)) for d in range(offset, offset + config.red.num_classes)]
            hot_by_pos.extend(hot_pos)
            # 遗漏
            last_occurrence = {}
            for i in range(len(df)):
                val = df.iloc[i][col]
                if val not in last_occurrence:
                    last_occurrence[val] = i
            skip_pos = [int(last_occurrence.get(d, 100)) for d in range(offset, offset + config.red.num_classes)]
            skip_by_pos.extend(skip_pos)
        return {
            "code": code,
            "name": config.name,
            "total_periods": len(df),
            "numbers": list(range(offset, offset + config.red.num_classes)),
            "hot_values": hot_by_pos,
            "skip_values": skip_by_pos,
            "hot_by_pos": True,
        }

    # 非 3D：原有逻辑
    hot_cold = compute_hot_cold_features(df, config, window=periods)
    last_hot = hot_cold.iloc[-1]
    skip = compute_skip_features(df, config)
    last_skip = skip.iloc[-1]

    numbers = list(range(offset, offset + config.red.num_classes))
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


@v1.get("/ranking/{code}")
async def api_ranking(
    request: Request,
    code: str,
    window: int = Query(default=200, ge=10, le=1000),
    backtest: int = Query(default=50, ge=5, le=200),
):
    require_auth(request)
    if code not in LOTTERY_CONFIGS:
        raise HTTPException(400, f"未知彩种: {code}")

    report = await asyncio.to_thread(generate_ranking_report, code, window, backtest)

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


@v1.get("/report/{code}")
async def api_report(request: Request, code: str):
    require_auth(request)
    if code not in LOTTERY_CONFIGS:
        raise HTTPException(400, f"未知彩种: {code}")

    report = await asyncio.to_thread(generate_comprehensive_report, code, 200, 50)
    return {"code": code, "report": report}


# ============================================================
# 前端页面
# ============================================================


# 启动时缓存 HTML 页面，避免每次请求读磁盘且未关闭句柄
_cached_html: dict = {}


def _load_html(filename: str) -> str:
    """加载并缓存 HTML 文件。"""
    if filename not in _cached_html:
        filepath = os.path.join(static_dir, filename)
        with open(filepath, encoding="utf-8") as f:
            _cached_html[filename] = f.read()
    return _cached_html[filename]


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    resp = HTMLResponse(_load_html("index.html"))
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    resp = HTMLResponse(_load_html("login.html"))
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.get("/favicon.ico")
async def favicon():
    """返回空 favicon 避免 404 报错。"""
    from fastapi.responses import Response
    return Response(content=b"", media_type="image/x-icon")


# 静态资源
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# 注册 API v1 路由（P5-24）
app.include_router(v1)


# ============================================================
# 启动
# ============================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
