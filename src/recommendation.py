# -*- coding: utf-8 -*-
"""
多策略推荐引擎。

为福彩店老板提供"有理论依据"的号码推荐。
四种策略：保守、激进、平衡、玄学。
输出附带分析摘要（用于小票/微信推送）。

ponytail: 策略逻辑清晰但不够"智能"，
因为彩票本质随机，过度复杂化反而降低可解释性。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from .config import LotteryModelConfig, get_lottery_config
from .feature_engineering import (
    compute_hot_cold_features,
    compute_skip_features,
)


@dataclass
class StrategyResult:
    """单策略推荐结果"""
    strategy_name: str
    red_balls: List[int]  # 6 个红球 (1-based)
    blue_ball: Optional[int]  # 蓝球 (1-based)
    analysis: Dict[str, any]  # 分析摘要
    confidence_note: str  # 置信度说明


@dataclass
class Recommendation:
    """完整推荐"""
    draw_issue: str  # 期号（预测目标）
    timestamp: str  # 推荐生成时间
    strategies: List[StrategyResult]  # 各策略结果
    disclaimer: str  # 免责声明


# ============================================================
# 策略实现
# ============================================================

class ConservativeStrategy:
    """
    保守型策略。
    选近期出现频率最高的号码（热号）。
    """

    name = "保守型"

    def __init__(self, config: LotteryModelConfig):
        self.config = config

    def _select_blue(self, df: pd.DataFrame) -> int:
        """蓝球：选近期热号"""
        recent = df.tail(30)
        blue_counts = {}
        for _, row in recent.iterrows():
            b = int(row.get("蓝球_1", 0))
            if 1 <= b <= self.config.blue.num_classes:
                blue_counts[b] = blue_counts.get(b, 0) + 1
        if blue_counts:
            return max(blue_counts, key=blue_counts.get)
        return random.randint(1, self.config.blue.num_classes)

    def generate(
        self,
        df: pd.DataFrame,
        analysis: Dict[str, any],
    ) -> StrategyResult:
        hot_cold_df = compute_hot_cold_features(df, self.config)
        # 取最后一行的当前热度值 (Series, 33 个值)
        hot_cold = hot_cold_df.iloc[-1] if isinstance(hot_cold_df, pd.DataFrame) else hot_cold_df

        # 取最近 30 期
        recent = df.tail(30)
        hot_recent_df = compute_hot_cold_features(recent, self.config)
        hot_recent = hot_recent_df.iloc[-1] if isinstance(hot_recent_df, pd.DataFrame) else hot_recent_df

        # 综合历史 + 近期（近期权重更高）
        combined_score = 0.3 * hot_cold.values + 0.7 * hot_recent.values

        # 选分数最高的 6 个
        selected_idx = np.argsort(combined_score)[-6:]
        red_balls = sorted([int(i + 1) for i in selected_idx])

        blue_ball = self._select_blue(df)

        return StrategyResult(
            strategy_name=self.name,
            red_balls=red_balls,
            blue_ball=blue_ball,
            analysis={
                "hot_numbers": sorted([int(i+1) for i in np.argsort(hot_recent.values)[-5:]]),
                "cold_numbers": sorted([int(i+1) for i in np.argsort(hot_recent.values)[:5]]),
                "avg_sum": int(analysis.get("avg_sum", 0)),
            },
            confidence_note="基于近 30 期热号统计",
        )


class AggressiveStrategy:
    """
    激进型策略。
    选长期遗漏（skip 值最高）的号码。
    """

    name = "激进型"

    def __init__(self, config: LotteryModelConfig):
        self.config = config

    def generate(
        self,
        df: pd.DataFrame,
        analysis: Dict[str, any],
    ) -> StrategyResult:
        skip_df = compute_skip_features(df, self.config)
        skip = skip_df.iloc[-1] if isinstance(skip_df, pd.DataFrame) else skip_df

        # 选 skip 值最大的 6 个（最"该出"的号码）
        selected_idx = np.argsort(skip.values)[-6:]
        red_balls = sorted([int(i + 1) for i in selected_idx])

        # 蓝球：选遗漏最大的
        blue_skip = {}
        for num in range(1, self.config.blue.num_classes + 1):
            last_seen = None
            for idx in range(len(df) - 1, -1, -1):
                if int(df.iloc[idx].get("蓝球_1", 0)) == num:
                    last_seen = len(df) - 1 - idx
                    break
            if last_seen is not None:
                blue_skip[num] = last_seen
        blue_ball = max(blue_skip, key=blue_skip.get) if blue_skip else random.randint(1, self.config.blue.num_classes)

        return StrategyResult(
            strategy_name=self.name,
            red_balls=red_balls,
            blue_ball=blue_ball,
            analysis={
                "max_skip_values": {int(i+1): int(skip.values[i]) for i in np.argsort(skip.values)[-5:]},
                "target_numbers": sorted([int(i+1) for i in np.argsort(skip.values)[-5:]]),
            },
            confidence_note="基于历史遗漏最大值",
        )


class BalancedStrategy:
    """
    平衡型策略。
    选号和值接近历史均值、奇偶比接近 3:3、AC 值居中的组合。
    """

    name = "平衡型"

    def __init__(self, config: LotteryModelConfig):
        self.config = config

    def generate(
        self,
        df: pd.DataFrame,
        analysis: Dict[str, any],
    ) -> StrategyResult:
        # 从概率均匀分布出发，微调满足约束
        avg_sum = analysis.get("avg_sum", 100)

        # 随机生成 N 组，选和值最接近均值 + 奇偶比 3:3 的
        best_combo = None
        best_score = float('inf')

        for _ in range(1000):
            combo = sorted(random.sample(range(1, self.config.red.num_classes + 1), 6))
            s = sum(combo)
            odd_count = sum(1 for x in combo if x % 2 == 1)

            # 评分：偏离均值的程度 + 奇偶偏离程度
            sum_penalty = abs(s - avg_sum) / 20
            odd_penalty = abs(odd_count - 3)
            score = sum_penalty + odd_penalty

            if score < best_score:
                best_score = score
                best_combo = combo

        red_balls = best_combo or sorted(random.sample(range(1, self.config.red.num_classes + 1), 6))

        # 蓝球随机
        blue_ball = random.randint(1, self.config.blue.num_classes)

        return StrategyResult(
            strategy_name=self.name,
            red_balls=red_balls,
            blue_ball=blue_ball,
            analysis={
                "target_sum": int(avg_sum),
                "actual_sum": sum(red_balls),
                "odd_even_ratio": f"{sum(1 for x in red_balls if x % 2 == 1)}:{sum(1 for x in red_balls if x % 2 == 0)}",
            },
            confidence_note="基于和值/奇偶均衡",
        )


class MysticStrategy:
    """
    玄学策略。
    融入用户提供的"幸运数字"，如果没有则用经典吉利数字。
    """

    name = "玄学型"

    def __init__(self, config: LotteryModelConfig):
        self.config = config

    def generate(
        self,
        df: pd.DataFrame,
        analysis: Dict[str, any],
        lucky_numbers: Optional[List[int]] = None,
    ) -> StrategyResult:
        # 经典吉利数字（中国文化）
        default_lucky = [6, 8, 9, 16, 18, 26, 28, 33]
        lucky = lucky_numbers or default_lucky

        # 过滤掉超出红球范围的
        lucky = [n for n in lucky if 1 <= n <= self.config.red.num_classes]

        # 选 3 个幸运数字 + 3 个随机
        lucky_part = random.sample(lucky, min(3, len(lucky)))
        remaining = [n for n in range(1, self.config.red.num_classes + 1) if n not in lucky_part]
        random_part = random.sample(
            remaining,
            min(3, self.config.red.num_classes - len(lucky_part))
        )

        red_balls = sorted(lucky_part + random_part)

        # 蓝球吉利数字
        blue_ball = random.choice([6, 8, 9, 16])

        return StrategyResult(
            strategy_name=self.name,
            red_balls=red_balls,
            blue_ball=blue_ball,
            analysis={
                "lucky_numbers_used": sorted(set(lucky_part)),
                "elements": self._get_elements(),
            },
            confidence_note="基于幸运数字组合",
        )

    def _get_elements(self) -> Dict[str, int]:
        """五行属性对应"""
        return {
            "金": "4, 9, 14, 19, 24, 29",
            "木": "3, 8, 13, 18, 23, 28, 33",
            "水": "1, 6, 11, 16, 21, 26, 31",
            "火": "2, 7, 12, 17, 22, 27, 32",
            "土": "5, 10, 15, 20, 25, 30",
        }


# ============================================================
# 推荐引擎
# ============================================================

class RecommendationEngine:
    """
    多策略推荐引擎。
    """

    def __init__(self, code: str = "ssq"):
        self.code = code
        self.config = get_lottery_config(code)
        self.strategies = {
            "conservative": ConservativeStrategy(self.config),
            "aggressive": AggressiveStrategy(self.config),
            "balanced": BalancedStrategy(self.config),
            "mystic": MysticStrategy(self.config),
        }

    def _compute_analysis(self, df: pd.DataFrame) -> Dict[str, any]:
        """计算分析数据"""
        red_cols = ["红球_1", "红球_2", "红球_3", "红球_4", "红球_5", "红球_6"]

        sums = []
        for _, row in df.iterrows():
            s = sum(int(row[c]) for c in red_cols)
            sums.append(s)

        return {
            "avg_sum": np.mean(sums) if sums else 100,
            "max_sum": max(sums) if sums else 0,
            "min_sum": min(sums) if sums else 0,
            "total_issues": len(df),
        }

    def generate(
        self,
        df: pd.DataFrame,
        lucky_numbers: Optional[List[int]] = None,
        target_issue: Optional[str] = None,
    ) -> Recommendation:
        """
        生成推荐。

        Args:
            df: 历史数据
            lucky_numbers: 用户幸运数字（仅玄学策略使用）
            target_issue: 目标期号（可选）

        Returns:
            Recommendation 对象
        """
        from datetime import datetime

        analysis = self._compute_analysis(df)

        strategies_results = []

        # 保守
        strategies_results.append(
            self.strategies["conservative"].generate(df, analysis)
        )

        # 激进
        strategies_results.append(
            self.strategies["aggressive"].generate(df, analysis)
        )

        # 平衡
        strategies_results.append(
            self.strategies["balanced"].generate(df, analysis)
        )

        # 玄学
        mystic_result = self.strategies["mystic"].generate(df, analysis, lucky_numbers)
        strategies_results.append(mystic_result)

        return Recommendation(
            draw_issue=target_issue or "下",
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
            strategies=strategies_results,
            disclaimer="本推荐基于历史数据统计，仅供娱乐参考，不保证中奖。",
        )


# ============================================================
# 便捷函数
# ============================================================

def generate_recommendation(
    code: str = "ssq",
    lucky_numbers: Optional[List[int]] = None,
    data_path: Optional[str] = None,
) -> Recommendation:
    """
    便捷函数：生成推荐。

    Args:
        code: 彩票代码
        lucky_numbers: 幸运数字
        data_path: 数据文件路径（可选）

    Returns:
        Recommendation
    """
    from .data_fetcher import load_history

    df = load_history(code, data_path) if data_path else load_history(code)
    engine = RecommendationEngine(code)
    return engine.generate(df, lucky_numbers)


__all__ = [
    "RecommendationEngine",
    "Recommendation",
    "StrategyResult",
    "ConservativeStrategy",
    "AggressiveStrategy",
    "BalancedStrategy",
    "MysticStrategy",
    "generate_recommendation",
]
