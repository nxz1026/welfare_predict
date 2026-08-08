# -*- coding: utf-8 -*-
"""
定时任务模块 — 每日数据同步与智能训练。

功能：
1. 每日 01:03 (BJT) 增量获取所有活跃彩种最新开奖数据
2. 数据更新后自动触发训练（仅训练有新数据的方法）
3. 训练状态持久化到 data/{code}/train_status.json
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from .config import LOTTERY_CONFIGS, PATHS
from .bootstrap import ACTIVE_LOTTERY_CODES

# 每日定时同步时间 (BJT, UTC+8)
SCHEDULED_HOUR = 1
SCHEDULED_MINUTE = 3


def _train_status_path(code: str) -> Path:
    """返回训练状态文件路径。"""
    return Path(PATHS["data"]) / code / "train_status.json"


def load_train_status(code: str) -> dict:
    """加载训练状态。"""
    path = _train_status_path(code)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("读取训练状态失败 {}: {}", code, e)
    return {}


def save_train_status(code: str, status: dict) -> None:
    """保存训练状态。"""
    path = _train_status_path(code)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=2)


def get_available_methods() -> List[str]:
    """返回当前环境可用的训练方法列表。"""
    methods = ["xgb", "poisson"]
    try:
        import tensorflow  # noqa: F401
        methods.extend(["stacking", "lstm"])
    except ImportError:
        pass
    return methods


def scheduled_data_sync() -> Dict[str, dict]:
    """定时数据同步：增量获取所有活跃彩种的最新数据。

    Returns:
        各彩种同步结果 {code: {synced, count, error}}
    """
    from .bootstrap import sync_startup_data
    logger.info("=== 定时数据同步开始 ===")
    results = sync_startup_data()
    synced = sum(1 for v in results.values() if v.get("synced"))
    failed = sum(1 for v in results.values() if not v.get("synced"))
    logger.info("定时数据同步完成: 成功 {} / 失败 {} / 总计 {}", synced, failed, len(results))
    return results


def smart_train_after_sync(sync_results: Dict[str, dict]) -> Dict[str, dict]:
    """智能训练：仅在数据有更新时训练。

    对比 train_status.json 中的 last_trained_issues 与当前数据期号，
    如有新数据则触发所有可用方法的训练。

    Args:
        sync_results: scheduled_data_sync() 的返回值

    Returns:
        各彩种训练结果 {code: {trained_methods, failed_methods, skipped}}
    """
    from .data_fetcher import load_history
    from .unified_pipeline import UnifiedPipeline

    train_results = {}
    methods = get_available_methods()

    for code in ACTIVE_LOTTERY_CODES:
        cfg = LOTTERY_CONFIGS.get(code)
        if cfg is None:
            continue

        # 检查同步是否成功
        sync_info = sync_results.get(code, {})
        if not sync_info.get("synced"):
            logger.info("【{}】数据同步失败，跳过训练", cfg.name)
            train_results[code] = {"skipped": True, "reason": "数据同步失败"}
            continue

        # 加载当前数据
        try:
            df = load_history(code)
        except Exception as e:
            logger.error("【{}】加载数据失败: {}", cfg.name, e)
            train_results[code] = {"skipped": True, "reason": f"加载数据失败: {e}"}
            continue

        if df.empty:
            logger.info("【{}】无数据，跳过训练", cfg.name)
            train_results[code] = {"skipped": True, "reason": "无数据"}
            continue

        # 检查是否有新数据
        current_issues = set(df["期数"].astype(str).tolist())
        status = load_train_status(code)
        last_trained_issues = set(status.get("last_trained_issues", []))

        if current_issues == last_trained_issues and status.get("training_ok"):
            logger.info("【{}】数据无变化，跳过训练", cfg.name)
            train_results[code] = {"skipped": True, "reason": "数据无变化"}
            continue

        new_count = len(current_issues - last_trained_issues)
        logger.info("【{}】检测到 {} 期新数据，开始训练", cfg.name, new_count)

        # 训练所有可用方法
        trained = []
        failed = []
        for m in methods:
            try:
                pipeline = UnifiedPipeline(code, method=m)
                pipeline.train(df)
                trained.append(m)
                logger.info("【{}】{} 训练成功", cfg.name, m)
            except Exception as e:
                failed.append(m)
                logger.error("【{}】{} 训练失败: {}", cfg.name, m, e)

        # 更新训练状态
        new_status = {
            "last_trained_issues": sorted(current_issues),
            "last_trained_at": datetime.utcnow().isoformat(),
            "training_ok": len(trained) > 0,
            "trained_methods": trained,
            "failed_methods": failed,
            "new_issues_count": new_count,
        }
        save_train_status(code, new_status)

        train_results[code] = {
            "skipped": False,
            "trained_methods": trained,
            "failed_methods": failed,
            "new_issues_count": new_count,
        }

    return train_results


def scheduled_job() -> None:
    """定时任务入口：数据同步 + 智能训练。"""
    logger.info("定时任务触发: {}", datetime.now().isoformat())
    sync_results = scheduled_data_sync()
    train_results = smart_train_after_sync(sync_results)

    # 汇总日志
    for code, info in train_results.items():
        cfg = LOTTERY_CONFIGS.get(code)
        name = cfg.name if cfg else code
        if info.get("skipped"):
            logger.info("【{}】训练跳过: {}", name, info.get("reason"))
        else:
            logger.info(
                "【{}】训练完成: 成功 {} / 失败 {}",
                name,
                info.get("trained_methods", []),
                info.get("failed_methods", []),
            )


def setup_scheduler() -> "BackgroundScheduler":
    """创建并配置 APScheduler 定时调度器。"""
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger

    scheduler = BackgroundScheduler(timezone="Asia/Shanghai")
    scheduler.add_job(
        scheduled_job,
        CronTrigger(hour=SCHEDULED_HOUR, minute=SCHEDULED_MINUTE),
        id="daily_data_sync",
        name="每日数据同步与智能训练",
        replace_existing=True,
    )
    logger.info(
        "定时任务已配置: 每日 {:02d}:{:02d} (BJT) 执行数据同步与训练",
        SCHEDULED_HOUR, SCHEDULED_MINUTE,
    )
    return scheduler