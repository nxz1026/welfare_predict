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
from typing import Any, Dict, List, Optional

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
    analysis: Dict[str, Any]  # 分析摘要（P2-03: any → Any）
    confidence_note: str  # 置信度说明
    display_name: str = ""


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

    name = "conservative"

    def __init__(self, config: LotteryModelConfig) -> None:
        self.config = config

    def _select_blue(self, df: pd.DataFrame) -> int:
        """蓝球：选近期热号"""
        recent = df.tail(30)
        blue_counts: Dict[int, int] = {}
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
        analysis: Dict[str, Any],  # P2-03: any → Any
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

        # 选分数最高的 N 个
        n = self.config.red.sequence_len
        offset = self.config.red.min_val
        selected_idx = np.argsort(combined_score)[-n:]
        red_balls = sorted([int(i + offset) for i in selected_idx])

        blue_ball = self._select_blue(df) if self.config.blue else None

        return StrategyResult(
            strategy_name=self.name,
            red_balls=red_balls,
            blue_ball=blue_ball,
            analysis={
                "hot_numbers": sorted([int(i + offset) for i in np.argsort(hot_recent.values)[-5:]]),
                "cold_numbers": sorted([int(i + offset) for i in np.argsort(hot_recent.values)[:5]]),
                "avg_sum": int(analysis.get("avg_sum", 0)),
            },
            confidence_note="基于近 30 期热号统计",
            display_name="热号追踪",
        )


class AggressiveStrategy:
    """
    激进型策略。
    选长期遗漏（skip 值最高）的号码。
    """

    name = "aggressive"

    def __init__(self, config: LotteryModelConfig) -> None:
        self.config = config

    def _select_blue(self, df: pd.DataFrame) -> int:
        """
        蓝球：选遗漏最大的。

        P1-03 优化：使用向量化操作替代逐行反向扫描，
        时间复杂度从 O(n × blue_range) 降低到 O(n)。
        """
        recent = df.tail(50)
        blue_col = "蓝球_1"
        if blue_col in recent.columns:
            blue_values = recent[blue_col].astype(int)
            counts = blue_values.value_counts()
            valid = counts[(counts.index >= 1) & (counts.index <= self.config.blue.num_classes)]
            if len(valid) > 0:
                # 选择频率最低的（即遗漏最大的）
                return int(valid.idxmin())
        return random.randint(1, self.config.blue.num_classes)

    def generate(
        self,
        df: pd.DataFrame,
        analysis: Dict[str, Any],  # P2-03: any → Any
    ) -> StrategyResult:
        skip_df = compute_skip_features(df, self.config)
        skip = skip_df.iloc[-1] if isinstance(skip_df, pd.DataFrame) else skip_df

        n = self.config.red.sequence_len
        offset = self.config.red.min_val
        selected_idx = np.argsort(skip.values)[-n:]
        red_balls = sorted([int(i + offset) for i in selected_idx])

        blue_ball = self._select_blue(df) if self.config.blue else None

        return StrategyResult(
            strategy_name=self.name,
            red_balls=red_balls,
            blue_ball=blue_ball,
            analysis={
                "max_skip_values": {int(i+offset): int(skip.values[i]) for i in np.argsort(skip.values)[-5:]},
                "target_numbers": sorted([int(i+offset) for i in np.argsort(skip.values)[-5:]]),
            },
            confidence_note="基于历史遗漏最大值",
            display_name="冷门博击",
        )


class BalancedStrategy:
    """
    平衡型策略。
    选号和值接近历史均值、奇偶比接近 3:3、AC 值居中的组合。
    """

    name = "balanced"

    def __init__(self, config: LotteryModelConfig) -> None:
        self.config = config

    def generate(
        self,
        df: pd.DataFrame,
        analysis: Dict[str, Any],  # P2-03: any → Any
    ) -> StrategyResult:
        avg_sum = analysis.get("avg_sum", 100)
        n = self.config.red.sequence_len
        offset = self.config.red.min_val
        num_range = self.config.red.num_classes
        pool = list(range(offset, offset + num_range))

        best_combo = None
        best_score = float('inf')
        target_odd = n // 2

        for _ in range(1000):
            combo = sorted(random.sample(pool, n))
            s = sum(combo)
            odd_count = sum(1 for x in combo if x % 2 == 1)
            sum_penalty = abs(s - avg_sum) / max(20, num_range)
            odd_penalty = abs(odd_count - target_odd)
            score = sum_penalty + odd_penalty
            if score < best_score:
                best_score = score
                best_combo = combo

        red_balls = best_combo or sorted(random.sample(pool, n))
        blue_ball = random.randint(self.config.blue.min_val, self.config.blue.max_val) if self.config.blue else None

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
            display_name="和值精选",
        )


class MysticStrategy:
    """
    玄学策略。
    融入用户提供的"幸运数字"，如果没有则用经典吉利数字。
    """

    name = "mystic"

    def __init__(self, config: LotteryModelConfig) -> None:
        self.config = config

    def generate(
        self,
        df: pd.DataFrame,
        analysis: Dict[str, Any],  # P2-03: any → Any
        lucky_numbers: Optional[List[int]] = None,
    ) -> StrategyResult:
        offset = self.config.red.min_val
        num_range = self.config.red.num_classes
        n = self.config.red.sequence_len
        pool = list(range(offset, offset + num_range))

        default_lucky = [n for n in [6, 8, 9, 16, 18, 26, 28, 33] if offset <= n < offset + num_range]
        lucky = lucky_numbers or default_lucky
        lucky = [n for n in lucky if offset <= n < offset + num_range]

        lucky_part = random.sample(lucky, min(n // 2, len(lucky)))
        remaining = [x for x in pool if x not in lucky_part]
        random_part = random.sample(remaining, n - len(lucky_part))

        red_balls = sorted(lucky_part + random_part)

        blue_ball = None
        if self.config.blue:
            blue_pool = [b for b in [6, 8, 9, 16] if self.config.blue.min_val <= b <= self.config.blue.max_val]
            if not blue_pool:
                blue_pool = [random.randint(self.config.blue.min_val, self.config.blue.max_val)]
            blue_ball = random.choice(blue_pool)

        return StrategyResult(
            strategy_name=self.name,
            red_balls=red_balls,
            blue_ball=blue_ball,
            analysis={
                "lucky_numbers_used": sorted(set(lucky_part)),
                "elements": self._get_elements(),
            },
            confidence_note="基于幸运数字组合",
            display_name="幸运号码",
        )

    def _get_elements(self) -> Dict[str, str]:
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

    def __init__(self, code: str = "ssq") -> None:
        self.code = code
        self.config = get_lottery_config(code)
        self.strategies = {
            "conservative": ConservativeStrategy(self.config),
            "aggressive": AggressiveStrategy(self.config),
            "balanced": BalancedStrategy(self.config),
            "mystic": MysticStrategy(self.config),
        }

    def _compute_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:  # P2-03: any → Any
        """计算分析数据"""
        red_cols = [f"红球_{i+1}" for i in range(self.config.red.sequence_len)]

        sums = []
        for _, row in df.iterrows():
            s = sum(int(row[c]) for c in red_cols)
            sums.append(s)

        return {
            "avg_sum": float(np.mean(sums)) if sums else 100.0,
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

    df = load_history(code, data_path=data_path)
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
