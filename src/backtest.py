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
# P1-06：按彩种分派奖级判定与奖金表
# ============================================================

# --- 双色球 (ssq) 奖金表 ---
SSQ_PRIZE_MONEY: Dict[int, float] = {
    1: 5_000_000.0,   # 一等奖（6+1，浮动奖金取近似值）
    2: 100_000.0,     # 二等奖（6+0，浮动奖金取近似值）
    3: 3_000.0,       # 三等奖（5+1，固定奖金）
    4: 200.0,         # 四等奖（5+0 或 4+1，固定奖金）
    5: 10.0,          # 五等奖（4+0 或 3+1，固定奖金）
    6: 5.0,           # 六等奖（2+1 或 1+1 或 0+1，固定奖金）
}

# --- 大乐透 (dlt) 奖金表 ---
DLT_PRIZE_MONEY: Dict[int, float] = {
    1: 5_000_000.0,   # 一等奖（5+2）
    2: 100_000.0,     # 二等奖（5+1）
    3: 10_000.0,      # 三等奖（5+0 或 4+2）
    4: 3_000.0,       # 四等奖（4+1 或 3+2）
    5: 500.0,         # 五等奖（4+0 或 3+1 或 2+2）
    6: 200.0,         # 六等奖（3+0 或 2+1 或 1+2）
    7: 100.0,         # 七等奖（2+0 或 1+1 或 0+2）
    8: 15.0,          # 八等奖（1+0 或 0+1）
    9: 5.0,           # 九等奖（0+0 但追加）
}

# --- 排列三 (pls) / 福彩3D (sd) 奖金表 ---
DIGIT3_PRIZE_MONEY: Dict[int, float] = {
    1: 1_000.0,       # 直选（3 位全中且顺序一致）
    2: 320.0,         # 组三（3 位中 2 个不同数字）
    3: 160.0,         # 组六（3 位全中但顺序不同）
}

# --- 七星彩 (qxc) 奖金表 ---
QXC_PRIZE_MONEY: Dict[int, float] = {
    1: 5_000_000.0,   # 一等奖（7 位全中）
    2: 50_000.0,      # 二等奖（后 6 位全中）
    3: 3_000.0,       # 三等奖（前 6 位连续中 5+后 1 位）
    4: 500.0,         # 四等奖
    5: 30.0,          # 五等奖
    6: 5.0,           # 六等奖
}

# --- 七乐彩 (qlc) 奖金表 ---
QLC_PRIZE_MONEY: Dict[int, float] = {
    1: 5_000_000.0,   # 一等奖（7+1）
    2: 50_000.0,      # 二等奖（7+0）
    3: 3_000.0,       # 三等奖（6+1）
    4: 500.0,         # 四等奖（6+0 或 5+1）
    5: 50.0,          # 五等奖（5+0 或 4+1）
    6: 10.0,          # 六等奖（4+0 或 3+1）
    7: 5.0,           # 七等奖（3+0 或 2+1 或 1+1 或 0+1）
}

# 彩种 → 奖金表 映射
PRIZE_TABLES: Dict[str, Dict[int, float]] = {
    "ssq": SSQ_PRIZE_MONEY,
    "dlt": DLT_PRIZE_MONEY,
    "pls": DIGIT3_PRIZE_MONEY,
    "qxc": QXC_PRIZE_MONEY,
    "sd": DIGIT3_PRIZE_MONEY,
    "qlc": QLC_PRIZE_MONEY,
}

DEFAULT_BET_COST = 2.0  # 每注投注金额（元）


def get_prize_money(prize_level: int, code: str = "ssq") -> float:
    """获取指定奖级的奖金（按彩种分派）。"""
    table = PRIZE_TABLES.get(code, SSQ_PRIZE_MONEY)
    return table.get(prize_level, 0.0)


# 向后兼容：保留旧常量供外部引用
PRIZE_MONEY = SSQ_PRIZE_MONEY


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


def calculate_prize(
    red_match: int,
    blue_match: bool = False,
    blue_match_count: int = 0,
    code: str = "ssq",
) -> Optional[int]:
    """根据命中情况确定奖级（按彩种分派）。

    Args:
        red_match: 红球/前区命中数
        blue_match: 蓝球是否命中（向后兼容）
        blue_match_count: 蓝球/后区命中数（大乐透等双蓝球玩法使用）
        code: 彩种代码

    Returns:
        奖级编号，未中奖返回 None
    """
    dispatch = {
        "ssq": _calculate_prize_ssq,
        "dlt": _calculate_prize_dlt,
        "pls": _calculate_prize_digit3,
        "sd": _calculate_prize_digit3,
        "qxc": _calculate_prize_qxc,
        "qlc": _calculate_prize_qlc,
    }
    handler = dispatch.get(code, _calculate_prize_ssq)
    return handler(red_match, blue_match, blue_match_count)


def _calculate_prize_ssq(
    red_match: int, blue_match: bool, blue_match_count: int
) -> Optional[int]:
    """双色球奖级判定：6 红 + 1 蓝。"""
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


def _calculate_prize_dlt(
    red_match: int, blue_match: bool, blue_match_count: int
) -> Optional[int]:
    """大乐透奖级判定：5 红 + 2 蓝。"""
    bc = blue_match_count if blue_match_count > 0 else (1 if blue_match else 0)
    if red_match == 5:
        if bc == 2:
            return 1
        elif bc == 1:
            return 2
        else:
            return 3
    elif red_match == 4:
        if bc == 2:
            return 3
        elif bc == 1:
            return 4
        else:
            return 5
    elif red_match == 3:
        if bc == 2:
            return 4
        elif bc == 1:
            return 6
        else:
            return 6
    elif red_match == 2:
        if bc == 2:
            return 6
        elif bc == 1:
            return 7
        else:
            return 7
    elif red_match == 1:
        if bc == 2:
            return 7
        elif bc == 1:
            return 8
        else:
            return 8
    elif red_match == 0:
        if bc == 2:
            return 7
        elif bc == 1:
            return 8
        else:
            return 9
    return None


def _calculate_prize_digit3(
    red_match: int, blue_match: bool, blue_match_count: int
) -> Optional[int]:
    """排列三/福彩3D 奖级判定：3 位数字。"""
    if red_match == 3:
        return 1  # 直选
    elif red_match == 2:
        return 2  # 组三
    elif red_match == 1:
        return 3  # 组六
    return None


def _calculate_prize_qxc(
    red_match: int, blue_match: bool, blue_match_count: int
) -> Optional[int]:
    """七星彩奖级判定：7 位连续匹配。"""
    if red_match == 7:
        return 1
    elif red_match == 6:
        return 2
    elif red_match == 5:
        return 3
    elif red_match == 4:
        return 4
    elif red_match == 3:
        return 5
    elif red_match == 2:
        return 6
    return None


def _calculate_prize_qlc(
    red_match: int, blue_match: bool, blue_match_count: int
) -> Optional[int]:
    """七乐彩奖级判定：7 红 + 1 特殊号。"""
    if red_match == 7:
        return 1 if blue_match else 2
    elif red_match == 6:
        return 3 if blue_match else 4
    elif red_match == 5:
        return 4 if blue_match else 5
    elif red_match == 4:
        return 5 if blue_match else 6
    elif red_match == 3:
        return 6 if blue_match else 7
    elif red_match == 2:
        return 7 if blue_match else None
    elif red_match == 1:
        return 7 if blue_match else None
    elif red_match == 0:
        return 7 if blue_match else None
    return None


def evaluate_single_bet(
    red_balls: List[int],
    blue_ball: int,
    actual_reds: List[int],
    actual_blue: int,
    issue: str = "",
    cost_per_bet: float = DEFAULT_BET_COST,
    code: str = "ssq",
    blue_balls: Optional[List[int]] = None,
    actual_blues: Optional[List[int]] = None,
) -> BetResult:
    """
    评估单次投注结果（支持多彩种）。

    Args:
        red_balls: 预测红球/前区
        blue_ball: 预测蓝球（单蓝球玩法，向后兼容）
        actual_reds: 实际开奖红球/前区
        actual_blue: 实际开奖蓝球（单蓝球玩法，向后兼容）
        issue: 期号
        cost_per_bet: 单注成本
        code: 彩种代码
        blue_balls: 预测蓝球列表（多蓝球玩法，如大乐透）
        actual_blues: 实际开奖蓝球列表（多蓝球玩法）

    Returns:
        BetResult
    """
    red_matches = len(set(red_balls) & set(actual_reds))

    # 多蓝球支持（大乐透等）
    if blue_balls is not None and actual_blues is not None:
        blue_match_count = len(set(blue_balls) & set(actual_blues))
        blue_matched = blue_match_count > 0
    else:
        blue_matched = blue_ball == actual_blue
        blue_match_count = 1 if blue_matched else 0

    prize = calculate_prize(red_matches, blue_matched, blue_match_count, code)
    prize_value = get_prize_money(prize, code) if prize is not None else 0.0

    profit = prize_value - cost_per_bet
    roi = profit / cost_per_bet if cost_per_bet > 0 else 0

    return BetResult(
        red_matches=red_matches,
        blue_matched=blue_matched,
        prize_level=prize,
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
                code=self.config.code,
            )
            results.append(result)

        return self._generate_report(results)

    def _train_models(self, X_train: np.ndarray, y_train: np.ndarray) -> tuple:
        """训练所有模型（简化版）"""
        from .modeling import XGBoostPredictor
        from .model_poisson import PoissonPrior

        xgb = XGBoostPredictor(self.config)
        xgb.train(X_train, y_train)

        # LSTM 需要 TensorFlow，未安装时跳过
        try:
            from .model_lstm import LSTMPredictor
            lstm = LSTMPredictor(self.config)
            lstm.train(X_train, y_train)
        except ImportError:
            logger.warning("TensorFlow 未安装，跳过 LSTM 模型训练")
            lstm = None

        poisson = PoissonPrior(self.config)
        poisson.train(X_train, y_train)

        return xgb, lstm, poisson, None

    def _predict_next(self, xgb, lstm, poisson, stacking, X_next: np.ndarray) -> np.ndarray:
        """使用模型预测下一期"""
        probas = []
        for m in [xgb, lstm, poisson]:
            if m is None:
                continue
            try:
                p = m.predict_proba(X_next)[0]
                probas.append(p)
            except Exception as e:
                logger.warning("模型预测失败: {}", e)

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
                total_reward += get_prize_money(r.prize_level, self.config.code)

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
