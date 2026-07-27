# -*- coding: utf-8 -*-
"""
分析报告模块 — 替代原 KL8 专用 analysis.py。

功能：
1. 生成综合分析报告（频率统计、遗漏分析、策略回测）
2. ROI 分析（各策略历史投入产出比）
3. 号码热度排行
4. 面向彩票店的文本报告输出

ponytail: 本模块是 backtest.py + evaluation.py 的高层封装。
升级路径：如果需要更复杂的分析（如号码关联规则、时序聚类），可扩展此处。
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from .config import LotteryModelConfig, get_lottery_config
from .data_fetcher import load_history
from .backtest import BacktestEngine, BacktestReport
from .strategy_backtest import StrategyBacktestEngine, RankingReport
from .feature_engineering import compute_hot_cold_features, compute_skip_features


def generate_frequency_report(df: pd.DataFrame, config: LotteryModelConfig) -> str:
    """
    生成频率统计报告。
    
    Args:
        df: 历史数据
        config: 彩票配置
    
    Returns:
        格式化报告文本
    """
    red_cols = [f"红球_{i+1}" for i in range(config.red.sequence_len)]
    balls = df[red_cols].values.flatten()
    
    # 频率统计
    counter = {}
    for b in balls:
        b_int = int(b)
        counter[b_int] = counter.get(b_int, 0) + 1
    
    total = len(balls)
    expected_freq = total / config.red.num_classes
    
    # 排序
    sorted_nums = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    
    lines = []
    lines.append("=" * 50)
    lines.append("           号码频率统计报告")
    lines.append("=" * 50)
    lines.append(f"统计期数: {len(df)} 期")
    lines.append(f"总号码数: {total} 个")
    lines.append(f"期望频率: {expected_freq:.1f} 次/号")
    lines.append(f"热号阈值: > {expected_freq * 1.2:.1f} 次 (+20%)")
    lines.append(f"冷号阈值: < {expected_freq * 0.8:.1f} 次 (-20%)")
    lines.append("-" * 50)
    lines.append(f"{'排名':<6} {'号码':<8} {'出现次数':<10} {'偏差':<10} {'类型'}")
    lines.append("-" * 50)
    
    for rank, (num, count) in enumerate(sorted_nums[:10], 1):
        deviation = (count - expected_freq) / expected_freq * 100
        num_type = "🔥热号" if deviation > 20 else ("❄️冷号" if deviation < -20 else "➖常温")
        lines.append(f"{rank:<6} {num:<8} {count:<10} {deviation:>+6.1f}%    {num_type}")
    
    lines.append("-" * 50)
    lines.append("后 10 名（冷号）:")
    for rank, (num, count) in enumerate(sorted_nums[-10:], len(sorted_nums) - 9):
        deviation = (count - expected_freq) / expected_freq * 100
        lines.append(f"{rank:<6} {num:<8} {count:<10} {deviation:>+6.1f}%    ❄️")
    
    lines.append("=" * 50)
    return "\n".join(lines)


def generate_missing_report(df: pd.DataFrame, config: LotteryModelConfig) -> str:
    """
    生成遗漏分析报告。
    
    Args:
        df: 历史数据
        config: 彩票配置
    
    Returns:
        格式化报告文本
    """
    skip_df = compute_skip_features(df, config)
    last_skip = skip_df.iloc[-1]
    
    lines = []
    lines.append("=" * 50)
    lines.append("           遗漏值分析报告")
    lines.append("=" * 50)
    lines.append(f"分析期号: {len(df)} 期")
    lines.append("-" * 50)
    lines.append("Top 10 高遗漏号码（可能'该出'了）:")
    lines.append("-" * 50)
    
    sorted_skip = last_skip.sort_values(ascending=False)
    for i, (col, val) in enumerate(sorted_skip.head(10).items()):
        num = int(col.split("_")[1])
        lines.append(f"  {i+1:2d}. 号码 {num:02d}: 已遗漏 {int(val)} 期")
    
    lines.append("-" * 50)
    lines.append("Top 10 低遗漏号码（近期刚出）:")
    lines.append("-" * 50)
    
    for i, (col, val) in enumerate(sorted_skip.tail(10).items()):
        num = int(col.split("_")[1])
        lines.append(f"  {i+1:2d}. 号码 {num:02d}: 已遗漏 {int(val)} 期")
    
    lines.append("=" * 50)
    return "\n".join(lines)


def generate_roi_report(
    code: str = "ssq",
    window_size: int = 200,
    n_backtest: int = 100,
    bet_cost: float = 2.0,
) -> str:
    """
    生成 ROI 分析报告。
    
    Args:
        code: 彩票代码
        window_size: 回测窗口
        n_backtest: 回测期数
        bet_cost: 单注成本
    
    Returns:
        格式化报告文本
    """
    df = load_history(code)
    config = get_lottery_config(code)
    
    engine = StrategyBacktestEngine(config, window_size=window_size, bet_cost=bet_cost)
    report = engine.run(df, n_backtest=n_backtest)
    
    lines = []
    lines.append("=" * 55)
    lines.append("          策略 ROI 分析报告")
    lines.append("=" * 55)
    lines.append(f"彩种: {config.name}")
    lines.append(f"回测窗口: {window_size} 期")
    lines.append(f"回测期数: {report.total_windows} 期")
    lines.append("-" * 55)
    
    for name, perf in report.performances.items():
        lines.append(f"\n【{name}】")
        lines.append(f"  投注期数: {perf.total_bets}")
        lines.append(f"  总成本: ¥{perf.total_cost:.2f}")
        lines.append(f"  总奖金: ¥{perf.total_reward:.2f}")
        lines.append(f"  净利润: ¥{perf.net_profit:.2f}")
        lines.append(f"  ROI: {perf.roi:.2%}")
        lines.append(f"  平均命中: {perf.avg_match:.2f} 个红球")
        lines.append(f"  蓝球命中: {perf.blue_match_count} 次")
        lines.append(f"  奖级分布: {perf.prize_counts}")
    
    lines.append("-" * 55)
    lines.append(f"随机 baseline: {report.random_baseline:.2f} 个红球")
    lines.append("=" * 55)
    return "\n".join(lines)


def generate_comprehensive_report(
    code: str = "ssq",
    window_size: int = 200,
    n_backtest: int = 100,
) -> str:
    """
    生成综合分析报告（频率 + 遗漏 + ROI）。
    
    Args:
        code: 彩票代码
        window_size: 回测窗口
        n_backtest: 回测期数
    
    Returns:
        完整报告文本
    """
    df = load_history(code)
    config = get_lottery_config(code)
    
    lines = []
    lines.append("\n" + "█" * 60)
    lines.append("█" + " " * 20 + "彩票综合分析日报" + " " * 20 + "█")
    lines.append("█" * 60)
    lines.append(f"\n生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"彩种: {config.name}")
    lines.append(f"数据范围: {len(df)} 期")
    
    # 频率报告
    freq_report = generate_frequency_report(df, config)
    lines.append("\n" + freq_report)
    
    # 遗漏报告
    missing_report = generate_missing_report(df, config)
    lines.append("\n" + missing_report)
    
    # ROI 报告
    roi_report = generate_roi_report(code, window_size, n_backtest)
    lines.append("\n" + roi_report)
    
    # 免责
    lines.append("\n" + "=" * 60)
    lines.append("⚠️  免责声明：彩票本质随机，本报告仅供娱乐参考。")
    lines.append("    历史表现不代表未来结果，请理性购彩。")
    lines.append("=" * 60)
    
    return "\n".join(lines)


def save_report(report: str, path: str) -> None:
    """保存报告到文件"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info("报告已保存: {}", path)


__all__ = [
    "generate_frequency_report",
    "generate_missing_report",
    "generate_roi_report",
    "generate_comprehensive_report",
    "save_report",
]
