# -*- coding: utf-8 -*-
"""
策略排行榜 — 历史回测对比四种策略的长期表现。

用 walk-forward 方式回测保守/激进/平衡/玄学四种策略，
统计：平均命中红球数、蓝球命中次数、奖级分布、ROI。

ponytail: 回测不考虑交易成本、奖金浮动。
升级路径：如果需要更精确，可加入个人所得税、奖金池分配模型。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from .config import LotteryModelConfig, get_lottery_config
from .recommendation import (
    RecommendationEngine,
    StrategyResult,
)
from .backtest import calculate_prize, get_prize_money


@dataclass
class StrategyPerformance:
    """单策略回测表现"""
    strategy_name: str
    total_bets: int
    total_cost: float
    total_reward: float
    net_profit: float
    roi: float
    avg_match: float
    blue_match_count: int
    prize_counts: Dict[int, int]


@dataclass
class RankingReport:
    """排行榜报告"""
    window_size: int
    total_windows: int
    performances: Dict[str, StrategyPerformance]
    random_baseline: float


class StrategyBacktestEngine:
    """
    策略回测引擎。

    用 walk-forward 方式：每期用前 N 期数据生成推荐，
    然后与实际开奖对比，统计各策略表现。
    """

    def __init__(
        self,
        config: LotteryModelConfig,
        window_size: int = 200,
        bet_cost: float = 2.0,
    ):
        self.config = config
        self.window_size = window_size
        self.bet_cost = bet_cost

    def run(
        self,
        df: pd.DataFrame,
        n_backtest: Optional[int] = None,
    ) -> RankingReport:
        """
        运行策略回测。

        Args:
            df: 历史数据
            n_backtest: 回测期数，默认回测所有可用期数
        """
        red_cols = [f"红球_{i+1}" for i in range(self.config.red.sequence_len)]

        # 总期数
        total = len(df) - self.window_size
        if n_backtest is None or n_backtest > total:
            n_backtest = total

        logger.info("开始策略回测: 训练窗口={}, 回测期数={}", self.window_size, n_backtest)

        # 统计各策略表现
        stats: Dict[str, Dict] = {}
        engine = RecommendationEngine(self.config.code)

        for i in range(n_backtest):
            train_end = i + self.window_size
            test_idx = train_end

            # 训练数据
            df_train = df.iloc[i:train_end]

            # 实际开奖
            actual_row = df.iloc[test_idx]
            actual_reds = sorted([int(actual_row[c]) for c in red_cols])
            actual_blue = int(actual_row.get("蓝球_1", 0))

            try:
                rec = engine.generate(df_train)
            except Exception as e:
                logger.warning("第 {} 期推荐失败: {}", test_idx, e)
                continue

            for s in rec.strategies:
                if s.strategy_name not in stats:
                    stats[s.strategy_name] = {
                        "total_bets": 0,
                        "total_cost": 0.0,
                        "total_reward": 0.0,
                        "match_sum": 0.0,
                        "blue_match": 0,
                        "prize_counts": {j: 0 for j in range(1, 7)},
                    }

                st = stats[s.strategy_name]
                st["total_bets"] += 1
                st["total_cost"] += self.bet_cost

                # 计算命中
                pred_set = set(s.red_balls)
                actual_set = set(actual_reds)
                match_count = len(pred_set & actual_set)
                st["match_sum"] += match_count

                # 蓝球命中
                blue_match = (s.blue_ball == actual_blue)
                if blue_match:
                    st["blue_match"] += 1

                # 奖级
                prize_level = calculate_prize(match_count, blue_match)
                if prize_level is not None:
                    st["prize_counts"][prize_level] += 1
                    st["total_reward"] += get_prize_money(prize_level)

        # 计算汇总
        performances = {}
        for name, st in stats.items():
            total_bets = st["total_bets"]
            avg_match = st["match_sum"] / total_bets if total_bets > 0 else 0
            net_profit = st["total_reward"] - st["total_cost"]
            roi = net_profit / st["total_cost"] if st["total_cost"] > 0 else 0

            performances[name] = StrategyPerformance(
                strategy_name=name,
                total_bets=total_bets,
                total_cost=st["total_cost"],
                total_reward=st["total_reward"],
                net_profit=net_profit,
                roi=roi,
                avg_match=avg_match,
                blue_match_count=st["blue_match"],
                prize_counts=st["prize_counts"],
            )

        # 随机 baseline
        random_avg = self.config.red.sequence_len * (self.config.red.sequence_len / self.config.red.num_classes)

        report = RankingReport(
            window_size=self.window_size,
            total_windows=n_backtest,
            performances=performances,
            random_baseline=random_avg,
        )

        return report


def generate_ranking_report(
    code: str = "ssq",
    window_size: int = 200,
    n_backtest: Optional[int] = None,
) -> RankingReport:
    """
    便捷函数：生成策略排行榜。

    Args:
        code: 彩票代码
        window_size: 回测窗口
        n_backtest: 回测期数

    Returns:
        RankingReport
    """
    from .data_fetcher import load_history

    df = load_history(code)
    config = get_lottery_config(code)
    engine = StrategyBacktestEngine(config, window_size=window_size)
    return engine.run(df, n_backtest)


__all__ = [
    "StrategyBacktestEngine",
    "StrategyPerformance",
    "RankingReport",
    "generate_ranking_report",
]
