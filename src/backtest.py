# -*- coding: utf-8 -*-
"""
回测引擎模块。

用于评估推荐策略的历史表现，计算 ROI、命中率等指标。
核心功能：
1. 单注评估：calculate_prize / evaluate_single_bet
2. 滑动窗口回测：BacktestEngine
3. 多策略对比：StrategyBacktestEngine

P4-03 修复：魔法数字已提取为命名常量。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from loguru import logger

from .config import LotteryModelConfig
from .feature_engineering import build_feature_matrix, build_binary_labels


# ============================================================
# 奖金常量（P4-03：从魔法数字提取为命名常量）
# ============================================================

PRIZE_MONEY: Dict[int, float] = {
    1: 5_000_000.0,   # 一等奖（6+1）
    2: 100_000.0,     # 二等奖（6+0 或 5+1，视彩种而定）
    3: 3_000.0,       # 三等奖（6+0 或 5+0）
    4: 200.0,         # 四等奖（4+1 或 5+0）
    5: 10.0,          # 五等奖（4+0 或 3+1）
    6: 5.0,           # 六等奖（2+1 或 3+0 或 1+1 或 0+1）
}

DEFAULT_BET_COST = 2.0  # 每注投注金额（元）


def get_prize_money(prize_level: int) -> float:
    """获取指定奖级的奖金。"""
    return PRIZE_MONEY.get(prize_level, 0.0)


@dataclass
class BetResult:
    """单次投注结果"""
    red_matches: int          # 红球命中数
    blue_matched: bool        # 蓝球是否命中
    prize_level: Optional[int] = None  # 中奖等级（None=未中奖）
    prize: float = 0.0              # 奖金
    cost: float = DEFAULT_BET_COST   # 投注成本
    profit: float = 0.0             # 利润
    roi: float = 0.0                # 投资回报率
    issue: str = ""                 # 期号


@dataclass
class BacktestReport:
    """回测报告"""
    total_bets: int
    prize_counts: Dict[int, int]
    prize_rates: Dict[int, float]
    total_cost: float
    total_reward: float
    net_profit: float
    roi: float
    avg_match: float
    random_avg_match: float
    match_improvement: float


def calculate_prize(red_match: int, blue_match: bool = False) -> Optional[int]:
    """根据红球命中数和蓝球命中情况确定奖级。

    Returns:
        奖级 1-6，未中奖返回 None
    """
    if red_match == 6:
        return 1 if blue_match else 2
    elif red_match == 5:
        return 3 if blue_match else 4
    elif red_match == 4:
        return 4 if blue_match else 5
    elif red_match == 3:
        return 5 if blue_match else None
    elif red_match in (0, 1, 2):
        return 6 if blue_match else None

    return None


def evaluate_single_bet(
    red_balls: List[int],
    blue_ball: int,
    actual_reds: List[int],
    actual_blue: int,
    issue: str = "",
    cost_per_bet: float = DEFAULT_BET_COST,
) -> BetResult:
    """
    评估单次投注结果。

    Args:
        red_balls: 预测红球
        blue_ball: 预测蓝球
        actual_reds: 实际开奖红球
        actual_blue: 实际开奖蓝球
        issue: 期号
        cost_per_bet: 单注成本

    Returns:
        BetResult
    """
    red_matches = len(set(red_balls) & set(actual_reds))
    blue_matched = blue_ball == actual_blue

    prize = calculate_prize(red_matches, blue_matched)
    prize_value = prize if prize is not None else 0.0
    prize_level = prize

    profit = prize_value - cost_per_bet
    roi = profit / cost_per_bet if cost_per_bet > 0 else 0

    return BetResult(
        red_matches=red_matches,
        blue_matched=blue_matched,
        prize_level=prize_level,
        prize=prize_value,
        cost=cost_per_bet,
        profit=profit,
        roi=roi,
        issue=issue,
    )


class BacktestEngine:
    """
    滑动窗口回测引擎。

    使用历史数据模拟滚动预测和投注，
    计算策略的长期表现。
    """

    def __init__(
        self,
        df: pd.DataFrame,
        config: LotteryModelConfig,
        window_size: int = 200,
        top_k: int = 6,
        bet_cost: float = DEFAULT_BET_COST,
    ):
        self.df = df
        self.config = config
        self.window_size = window_size
        self.top_k = top_k
        self.bet_cost = bet_cost

    def run(self, window_size: Optional[int] = None, n_backtest: Optional[int] = None) -> BacktestReport:
        """
        执行滑动窗口回测。

        Args:
            window_size: 训练窗口大小
            n_backtest: 回测期数

        Returns:
            BacktestReport
        """
        if window_size is not None:
            self.window_size = window_size

        features = build_feature_matrix(self.df, self.config)
        labels = build_binary_labels(self.df, self.config)
        issues = self.df["期数"].values

        X = features.iloc[:-1].values
        y = labels[1:]

        if n_backtest is None:
            n_backtest = len(X) - self.window_size

        logger.info("开始回测: 训练窗口={}, 回测期数={}", self.window_size, n_backtest)

        results: List[BetResult] = []

        for i in range(n_backtest):
            train_start = i
            train_end = i + self.window_size
            test_idx = train_end

            if test_idx >= len(X):
                break

            X_train = X[train_start:train_end]
            y_train = y[train_start:train_end]
            X_next = X[test_idx:test_idx + 1]

            try:
                xgb, lstm, poisson, stacking = self._train_models(X_train, y_train)
                pred = self._predict_next(xgb, lstm, poisson, stacking, X_next)
                predicted_reds = [int(x + 1) for x in pred[0]]
            except Exception as e:
                logger.warning("第 {} 期回测失败: {}", test_idx, e)
                continue

            actual_indices = np.where(y[test_idx] == 1)[0]
            actual_reds = [int(x + 1) for x in actual_indices]
            actual_row = self.df.iloc[test_idx]
            actual_blue = int(actual_row.get("蓝球_1", 0)) if "蓝球_1" in actual_row.index else 0

            result = evaluate_single_bet(
                red_balls=predicted_reds,
                blue_ball=0,
                actual_reds=actual_reds,
                actual_blue=actual_blue,
                issue=str(issues[test_idx + 1]) if test_idx + 1 < len(issues) else "",
                cost_per_bet=self.bet_cost,
            )
            results.append(result)

        return self._generate_report(results)

    def _train_models(self, X_train, y_train):
        """训练所有模型（简化版）"""
        from .modeling import XGBoostPredictor
        from .model_lstm import LSTMPredictor
        from .model_poisson import PoissonPrior

        xgb = XGBoostPredictor(self.config)
        xgb.train(X_train, y_train)

        lstm = LSTMPredictor(self.config)
        lstm.train(X_train, y_train)

        poisson = PoissonPrior(self.config)
        poisson.train(X_train, y_train)

        return xgb, lstm, poisson, None

    def _predict_next(self, xgb, lstm, poisson, stacking, X_next):
        """使用模型预测下一期"""
        probas = []
        for m in [xgb, lstm, poisson]:
            try:
                p = m.predict_proba(X_next)[0]
                probas.append(p)
            except Exception:
                pass

        if probas:
            avg_proba = np.mean(probas, axis=0)
        else:
            avg_proba = np.ones(self.config.red.num_classes) / self.config.red.num_classes

        selected = np.argsort(avg_proba)[-self.top_k:]
        return selected.reshape(1, -1)

    def _generate_report(self, results: List[BetResult]) -> BacktestReport:
        """生成回测报告"""
        total_bets = len(results)

        prize_counts = {i: 0 for i in range(1, 7)}
        total_reward = 0.0

        for r in results:
            if r.prize_level is not None:
                prize_counts[r.prize_level] += 1
                total_reward += get_prize_money(r.prize_level)

        prize_rates = {level: count / total_bets for level, count in prize_counts.items()}

        total_cost = total_bets * self.bet_cost
        net_profit = total_reward - total_cost
        roi = net_profit / total_cost if total_cost > 0 else 0

        avg_match = np.mean([r.red_matches for r in results]) if results else 0

        random_avg_match = self.top_k * (self.config.red.sequence_len / self.config.red.num_classes)
        match_improvement = (avg_match - random_avg_match) / random_avg_match * 100 if random_avg_match > 0 else 0

        report = BacktestReport(
            total_bets=total_bets,
            prize_counts=prize_counts,
            prize_rates=prize_rates,
            total_cost=total_cost,
            total_reward=total_reward,
            net_profit=net_profit,
            roi=roi,
            avg_match=avg_match,
            random_avg_match=random_avg_match,
            match_improvement=match_improvement,
        )

        self._log_report(report)
        return report

    def _log_report(self, report: BacktestReport) -> None:
        """输出回测报告"""
        logger.info("========== 回测报告 ==========")
        logger.info("总投注期数: {}", report.total_bets)
        logger.info("总成本: ¥{:.2f}", report.total_cost)
        logger.info("总奖金: ¥{:.2f}", report.total_reward)
        logger.info("净利润: ¥{:.2f}", report.net_profit)
        logger.info("ROI: {:.2%}", report.roi)
        logger.info("平均命中: {:.2f} 个红球", report.avg_match)
        logger.info("随机 baseline: {:.2f} 个红球", report.random_avg_match)
        logger.info("相比随机提升: {:.1f}%", report.match_improvement)
        logger.info("---")
        for level in range(1, 7):
            logger.info("{}等奖: {} 次 ({:.2%})", level, report.prize_counts[level], report.prize_rates[level])


__all__ = [
    "BacktestEngine",
    "BacktestReport",
    "BetResult",
    "evaluate_single_bet",
    "calculate_prize",
    "PRIZE_MONEY",
    "DEFAULT_BET_COST",
    "get_prize_money",
]
