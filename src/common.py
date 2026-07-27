# -*- coding: utf-8 -*-
"""
公共接口封装。

为脚本层提供以下能力：
1. 下载历史数据：`get_data_run`
2. 查询最新期号：`get_current_number`
3. 训练模型：`train_pipeline`
4. 预测下一期开奖：`predict_latest`
"""

from __future__ import annotations

from typing import Dict, Optional

from loguru import logger

from .config import LOTTERY_CONFIGS, ensure_runtime_directories, get_lottery_config
from .data_fetcher import download_history, get_current_issue, load_history
from .unified_pipeline import train_unified, predict_unified, UnifiedTrainingSummary, UnifiedPrediction


def get_data_run(
    name: str,
    cq: int = 0,
    start_issue: Optional[int] = None,
    end_issue: Optional[int] = None,
) -> None:
    """下载指定彩票的历史数据。"""

    ensure_runtime_directories()
    code = name.lower().strip()
    if code not in LOTTERY_CONFIGS:
        raise ValueError(f"不支持的彩票类型: {name}")
    use_sequence = bool(cq) and code == "kl8"
    download_history(code, start=start_issue, end=end_issue, use_sequence_order=use_sequence)


def get_current_number(name: str) -> str:
    """返回指定彩票的当前期号。"""

    code = name.lower().strip()
    if code not in LOTTERY_CONFIGS:
        raise ValueError(f"不支持的彩票类型: {name}")
    return get_current_issue(code)


def train_pipeline(
    name: str,
    method: str = "xgb",
    validation_ratio: float = 0.15,
    source: Optional[str] = None,
) -> UnifiedTrainingSummary:
    """高层训练接口，使用统一管线（UnifiedPipeline）。

    Args:
        name: 彩票代码
        method: 训练方法 (xgb/lstm/poisson/stacking)
        validation_ratio: 验证集比例
        source: 数据源（未使用，保留兼容）
    """
    code = name.lower().strip()
    logger.info("开始训练【{}】模型, 方法={}...", LOTTERY_CONFIGS[code].name, method)
    df = load_history(code)
    summary = train_unified(df, code=code, method=method, validation_ratio=validation_ratio)
    logger.success("训练完成: {}", summary)
    return summary


def predict_latest(name: str, method: str = "xgb", source: Optional[str] = None) -> Dict[str, list]:
    """使用统一管线预测下一期号码。

    Args:
        name: 彩票代码
        method: 预测方法 (xgb/lstm/poisson/stacking)
        source: 数据源（未使用，保留兼容）
    """
    code = name.lower().strip()
    cfg = get_lottery_config(code)
    df = load_history(code)
    pred = predict_unified(df, code=code, method=method)
    readable = {"red_balls": [int(b) for b in pred.red_balls], "blue_ball": int(pred.blue_ball) if pred.blue_ball else None}
    logger.info("【{}】预测结果: {}", cfg.name, readable)
    return readable


__all__ = [
    "get_data_run",
    "get_current_number",
    "train_pipeline",
    "predict_latest",
    "download_history",
    "load_history",
]
