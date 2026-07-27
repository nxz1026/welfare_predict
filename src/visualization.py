# -*- coding: utf-8 -*-
"""
可视化模块。

1. 反赌徒谬误对比图：热冷号策略 vs 随机选号
2. 分析小票生成：文本格式，可打印或微信发送
3. 和值分布直方图
4. 遗漏柱状图

ponytail: 用 Matplotlib 静态图 + 文本输出，不引入前端依赖。
升级路径：如果需要交互，可换 ECharts + 简单 HTML。
"""

from __future__ import annotations

from datetime import datetime
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # 无头模式
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from .config import LotteryModelConfig, get_lottery_config
from .feature_engineering import (
    compute_hot_cold_features,
    compute_skip_features,
)
from .recommendation import Recommendation, StrategyResult


# ============================================================
# 反赌徒谬误对比图
# ============================================================

def plot_hot_cold_vs_random(
    df: pd.DataFrame,
    config: LotteryModelConfig,
    save_path: Optional[str] = None,
) -> Optional[bytes]:
    """
    绘制热冷号策略 vs 随机选号的理论对比图。
    """
    hot_cold_df = compute_hot_cold_features(df, config)
    hot_cold = hot_cold_df.iloc[-1] if isinstance(hot_cold_df, pd.DataFrame) else hot_cold_df
    skip_df = compute_skip_features(df, config)
    skip = skip_df.iloc[-1] if isinstance(skip_df, pd.DataFrame) else skip_df

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：热冷号频率
    sorted_idx = np.argsort(hot_cold.values)[::-1]
    colors = ["#e74c3c" if i < 11 else "#3498db" if i > 22 else "#95a5a6"
              for i in range(len(sorted_idx))]

    axes[0].bar(range(len(sorted_idx)), hot_cold.values[sorted_idx], color=colors)
    axes[0].axhline(y=hot_cold.mean(), color="black", linestyle="--", alpha=0.5, label=f"均值: {hot_cold.mean():.2f}")
    axes[0].set_xlabel("号码排序")
    axes[0].set_ylabel("历史出现频率")
    axes[0].set_title("红球历史频率分布")
    axes[0].legend()

    # 右图：遗漏值
    sorted_skip = np.argsort(skip.values)[::-1]
    axes[1].bar(range(len(sorted_skip)), skip.values[sorted_skip], color="#e67e22")
    axes[1].axhline(y=skip.mean(), color="black", linestyle="--", alpha=0.5, label=f"均值: {skip.mean():.1f}")
    axes[1].set_xlabel("号码排序")
    axes[1].set_ylabel("遗漏期数")
    axes[1].set_title("红球历史遗漏分布")
    axes[1].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        logger.info("图表已保存: {}", save_path)

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return buf.read()


def plot_sum_distribution(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> Optional[bytes]:
    """
    绘制和值分布直方图。
    """
    red_cols = ["红球_1", "红球_2", "红球_3", "红球_4", "红球_5", "红球_6"]
    sums = df[red_cols].sum(axis=1).values

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(sums, bins=30, color="#3498db", edgecolor="white", alpha=0.8)
    ax.axvline(x=np.mean(sums), color="red", linestyle="--", alpha=0.7, label=f"均值: {np.mean(sums):.1f}")
    ax.axvline(x=np.median(sums), color="green", linestyle="--", alpha=0.7, label=f"中位数: {np.median(sums):.1f}")

    ax.set_xlabel("和值")
    ax.set_ylabel("期数")
    ax.set_title("双色球历史和值分布")
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return buf.read()


# ============================================================
# 分析小票生成
# ============================================================

def generate_ticket_text(
    rec: Recommendation,
    df: pd.DataFrame,
    strategy_name: Optional[str] = None,
) -> str:
    """
    生成分析小票文本。

    Args:
        rec: 推荐结果
        df: 历史数据
        strategy_name: 指定策略（None 则输出所有策略）

    Returns:
        格式化文本，适合打印或微信发送
    """
    lines = []
    lines.append("=" * 30)
    lines.append("      福彩推荐分析单")
    lines.append("=" * 30)
    lines.append(f"预测期号: {rec.draw_issue}期")
    lines.append(f"生成时间: {rec.timestamp}")
    lines.append(f"历史数据: {len(df)} 期")
    lines.append("-" * 30)

    strategies = rec.strategies
    if strategy_name:
        strategies = [s for s in strategies if s.strategy_name == strategy_name]

    for s in strategies:
        lines.append(f"【{s.strategy_name}】")
        red_str = " ".join(f"{n:02d}" for n in s.red_balls)
        lines.append(f"  红球: {red_str}")
        if s.blue_ball:
            lines.append(f"  蓝球: {s.blue_ball:02d}")
        lines.append(f"  依据: {s.confidence_note}")

        # 分析摘要
        if "hot_numbers" in s.analysis:
            lines.append(f"  热号: {s.analysis['hot_numbers']}")
        if "cold_numbers" in s.analysis:
            lines.append(f"  冷号: {s.analysis['cold_numbers']}")
        if "avg_sum" in s.analysis:
            lines.append(f"  历史均值: {s.analysis['avg_sum']:.0f}")
        if "max_skip_values" in s.analysis:
            top_skip = list(s.analysis["max_skip_values"].items())[:3]
            lines.append(f"  高遗漏: {top_skip}")
        if "odd_even_ratio" in s.analysis:
            lines.append(f"  奇偶比: {s.analysis['odd_even_ratio']}")
        if "lucky_numbers_used" in s.analysis:
            lines.append(f"  幸运数: {s.analysis['lucky_numbers_used']}")
        lines.append("")

    lines.append("-" * 30)
    lines.append(f"免责: {rec.disclaimer}")
    lines.append("=" * 30)

    return "\n".join(lines)


def generate_missing_report(
    df: pd.DataFrame,
    config: LotteryModelConfig,
    top_n: int = 5,
) -> str:
    """
    生成遗漏提醒文本。

    Args:
        df: 历史数据
        config: 配置
        top_n: 显示前 N 个高遗漏号码

    Returns:
        格式化文本
    """
    skip_df = compute_skip_features(df, config)
    skip = skip_df.iloc[-1] if isinstance(skip_df, pd.DataFrame) else skip_df
    sorted_idx = np.argsort(skip.values)[::-1]

    lines = []
    lines.append("=" * 25)
    lines.append("     遗漏提醒")
    lines.append("=" * 25)
    lines.append(f"数据截至: {df.iloc[-1]['期数']}")
    lines.append(f"数据量: {len(df)} 期")
    lines.append("-" * 25)

    for i in range(top_n):
        num = int(sorted_idx[i] + 1)
        skip_val = int(skip.values[sorted_idx[i]])
        lines.append(f"  号码 {num:02d}: 已遗漏 {skip_val} 期")

    lines.append("-" * 25)
    lines.append("注: 遗漏期数不预示未来出现概率")
    lines.append("=" * 25)

    return "\n".join(lines)


# ============================================================
# 策略排行榜（文本）
# ============================================================

def generate_strategy_ranking_text(
    ranking: Dict[str, Dict[str, float]],
) -> str:
    """
    生成策略排行榜文本。

    Args:
        ranking: {"策略名": {"avg_match": x, "roi": y, ...}, ...}

    Returns:
        格式化文本
    """
    lines = []
    lines.append("=" * 45)
    lines.append("           策略长期表现排行")
    lines.append("=" * 45)
    lines.append(f"{'策略':<10} {'平均命中':<10} {'蓝球命中':<10} {'ROI':<10}")
    lines.append("-" * 45)

    sorted_strategies = sorted(ranking.items(), key=lambda x: x[1].get("avg_match", 0), reverse=True)

    for name, metrics in sorted_strategies:
        avg_match = metrics.get("avg_match", 0)
        blue_match = metrics.get("blue_match", 0)
        roi = metrics.get("roi", 0)
        lines.append(f"{name:<10} {avg_match:<10.2f} {blue_match:<10} {roi:<10.2%}")

    lines.append("-" * 45)
    lines.append("注: 历史表现不代表未来结果，彩票本质随机")
    lines.append("=" * 45)

    return "\n".join(lines)


__all__ = [
    "plot_hot_cold_vs_random",
    "plot_sum_distribution",
    "generate_ticket_text",
    "generate_missing_report",
    "generate_strategy_ranking_text",
]
