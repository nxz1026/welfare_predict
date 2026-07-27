# -*- coding: utf-8 -*-
"""
回测引擎 + 评估指标。

使用滚动窗口回测（walk-forward backtest），避免信息泄露。
评估指标：奖级命中率、期望收益、与随机选号的 baseline 对比。

ponytail: 回测不考虑交易成本、奖金浮动。
升级路径：如果需要更精确，可加入个人所得税、奖金池分配模型。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from loguru import logger

from .config import LotteryModelConfig, get_lottery_config
from .feature_engineering import build_feature_matrix, build_binary_labels
from .modeling import XGBoostPredictor
from .model_lstm import LSTMPredictor
from .model_poisson import PoissonPrior
from .model_stacking import StackingMetaLearner


@dataclass
class DrawResult:
    """一期开奖号码"""
    issue: str
    red_balls: list  # 6 个红球 (1-based)
    blue_ball: Optional[int] = None


@dataclass
class BetResult:
    """一期预测结果"""
    issue: str
    predicted_reds: list  # 预测的 6 个红球
    actual_reds: list  # 实际的 6 个红球
    match_count: int  # 命中红球数
    prize_level: Optional[int]  # 奖级（1-6）， None 表示未中奖


@dataclass
class BacktestReport:
    """回测报告"""
    total_bets: int  # 总投注期数
    prize_counts: Dict[int, int]  # 各奖级命中次数
    prize_rates: Dict[int, float]  # 各奖级命中率
    total_cost: float  # 总成本（元）
    total_reward: float  # 总奖金（元）
    net_profit: float  # 净利润
    roi: float  # 投资回报率
    avg_match: float  # 平均命中红球数
    random_avg_match: float  # 随机选号平均命中数
    match_improvement: float  # 相比随机的提升百分比


def calculate_prize(red_match: int, blue_match: bool = False) -> Optional[int]:
    """
    根据红球命中数确定双色球奖级。

    一等奖: 6 红 + 1 蓝
    二等奖: 6 红 + 0 蓝
    三等奖: 5 红 + 1 蓝
    四等奖: 5 红 + 0 蓝 或 4 红 + 1 蓝
    五等奖: 4 红 + 0 蓝 或 3 红 + 1 蓝
    六等奖: 2 红 + 1 蓝 或 1 红 + 1 蓝 或 0 红 + 1 蓝

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
    elif red_match in [0, 1, 2]:
        return 6 if blue_match else None
    return None


def get_prize_money(level: int) -> float:
    """获取奖级的平均奖金（近似值，浮动很大）"""
    prizes = {
        1: 5_000_000.0,  # 一等奖约 500 万（浮动）
        2: 100_000.0,    # 二等奖约 10 万（浮动）
        3: 3_000.0,      # 三等奖固定 3000
        4: 200.0,        # 四等奖固定 200
        5: 10.0,         # 五等奖固定 10
        6: 5.0,          # 六等奖固定 5
    }
    return prizes.get(level, 0.0)


def evaluate_single_bet(
    predicted_reds: List[int],
    actual_reds: List[int],
    issue: str = "",
) -> BetResult:
    """评估一期预测"""
    pred_set = set(predicted_reds)
    actual_set = set(actual_reds)
    match_count = len(pred_set & actual_set)

    # 蓝球不预测，随机（假设蓝球未中）
    prize_level = calculate_prize(match_count, blue_match=False)

    return BetResult(
        issue=issue,
        predicted_reds=sorted(predicted_reds),
        actual_reds=sorted(actual_reds),
        match_count=match_count,
        prize_level=prize_level,
    )


class BacktestEngine:
    """
    回测引擎。

    使用 walk-forward 方式：用前 N 期数据训练，预测第 N+1 期，
    然后窗口前移一期，重新训练。避免信息泄露。
    """

    def __init__(
        self,
        config: LotteryModelConfig,
        window_size: int = 50,  # 训练窗口大小
        bet_cost: float = 2.0,  # 单注成本（元）
        top_k: int = 6,  # 选前 k 个号码
        use_lstm: bool = True,  # 是否使用 LSTM（慢）
        xgb_only: bool = False,  # 快速模式：只用 XGBoost + Poisson
    ):
        self.config = config
        self.window_size = window_size
        self.bet_cost = bet_cost
        self.top_k = top_k
        self.use_lstm = use_lstm
        self.xgb_only = xgb_only

    def _train_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> Tuple[XGBoostPredictor, LSTMPredictor, PoissonPrior, StackingMetaLearner]:
        """训练所有模型"""
        xgb = XGBoostPredictor(self.config, n_estimators=50, max_depth=3, learning_rate=0.1)
        xgb.train(X_train, y_train)

        if self.xgb_only:
            # 快速模式：跳过 LSTM，减少训练时间
            lstm = None
        else:
            lstm = LSTMPredictor(self.config, lstm_units=[32, 16], dropout=0.3,
                                learning_rate=0.001, batch_size=16, epochs=5)
            lstm.train(X_train, y_train)

        poisson = PoissonPrior(self.config)
        poisson.train(X_train, y_train)

        # Stacking
        xgb_proba = xgb.predict_proba(X_train)
        if lstm is not None:
            lstm_proba = lstm.predict_proba(X_train)
        else:
            # 快速模式：用 XGBoost 概率填充
            lstm_proba = xgb_proba.copy()
        poisson_proba = poisson.predict_proba(X_train)

        probas = {'xgboost': xgb_proba, 'lstm': lstm_proba, 'poisson': poisson_proba}
        stacking = StackingMetaLearner(self.config)
        stacking.train(probas, y_train)

        return xgb, lstm, poisson, stacking

    def _predict_next(
        self,
        xgb: XGBoostPredictor,
        lstm: Optional[LSTMPredictor],
        poisson: PoissonPrior,
        stacking: StackingMetaLearner,
        X_next: np.ndarray,
    ) -> np.ndarray:
        """预测下一期"""
        xgb_proba = xgb.predict_proba(X_next)
        if lstm is not None:
            lstm_proba = lstm.predict_proba(X_next)
        else:
            lstm_proba = xgb_proba.copy()
        poisson_proba = poisson.predict_proba(X_next)

        probas = {'xgboost': xgb_proba, 'lstm': lstm_proba, 'poisson': poisson_proba}
        return stacking.predict(probas)

    def run(
        self,
        df: pd.DataFrame,
        n_backtest: Optional[int] = None,
    ) -> BacktestReport:
        """
        运行回测。

        Args:
            df: 历史数据 DataFrame
            n_backtest: 回测期数，默认回测所有可用期数

        Returns:
            BacktestReport
        """
        features = build_feature_matrix(df, self.config)
        labels = build_binary_labels(df, self.config)
        issues = df["期数"].values

        # 时间对齐
        X = features.iloc[:-1].values
        y = labels[1:]

        if n_backtest is None:
            n_backtest = len(X) - self.window_size

        logger.info("开始回测: 训练窗口={}, 回测期数={}", self.window_size, n_backtest)

        results: List[BetResult] = []

        for i in range(n_backtest):
            # 训练集: [i, i + window_size)
            # 测试集: i + window_size
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
                predicted_reds = [int(x + 1) for x in pred[0]]  # 0-based -> 1-based
            except Exception as e:
                logger.warning("第 {} 期回测失败: {}", test_idx, e)
                continue

            # 实际号码
            actual_indices = np.where(y[test_idx] == 1)[0]
            actual_reds = [int(x + 1) for x in actual_indices]

            # 评估
            result = evaluate_single_bet(
                predicted_reds,
                actual_reds,
                issue=str(issues[test_idx + 1]) if test_idx + 1 < len(issues) else "",
            )
            results.append(result)

        # 汇总报告
        return self._generate_report(results)

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

        avg_match = np.mean([r.match_count for r in results]) if results else 0

        # 随机选号的理论平均命中: 6 * (6/33) ≈ 1.09
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
]
