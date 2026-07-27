# -*- coding: utf-8 -*-
"""
回测引擎单元测试（P2-05 新增）。

覆盖回测核心函数：
- calculate_prize() 奖级判定
- evaluate_single_bet() 投注评估
- BacktestEngine.run() 完整流程
- StrategyBacktestEngine 四种策略
"""

import numpy as np
import pandas as pd
import pytest

from src.config import LotteryModelConfig, SequenceModelSpec, get_lottery_config
from src.backtest import (
    BetResult,
    BacktestEngine,
    calculate_prize,
    evaluate_single_bet,
)
from src.strategy_backtest import (
    StrategyBacktestEngine,
    StrategyPerformance,
    generate_ranking_report,
)


# ============================================================
# Fixtures
# ============================================================

SSQ_CONFIG = get_lottery_config("ssq")


@pytest.fixture
def sample_ssq_df():
    """生成 50 期的模拟双色球数据。"""
    np.random.seed(42)
    n = 50
    records = []
    for i in range(n):
        reds = sorted(np.random.choice(range(1, 34), size=6, replace=False).tolist())
        blue = np.random.randint(1, 17)
        records.append({
            "期数": f"20250{i+1:03d}",
            "红球_1": reds[0],
            "红球_2": reds[1],
            "红球_3": reds[2],
            "红球_4": reds[3],
            "红球_5": reds[4],
            "红球_6": reds[5],
            "蓝球_1": blue,
        })
    return pd.DataFrame(records)


@pytest.fixture
def winning_numbers():
    """模拟开奖号码。"""
    return {
        "reds": [3, 7, 12, 18, 25, 31],
        "blue": 9,
    }


# ============================================================
# calculate_prize 测试
# ============================================================


class TestCalculatePrize:
    def test_first_prize_all_match(self, winning_numbers):
        """一等奖：6 红全中 + 蓝球中。"""
        bet_reds = winning_numbers["reds"]
        bet_blue = winning_numbers["blue"]
        actual_reds = winning_numbers["reds"]
        actual_blue = winning_numbers["blue"]
        match_count = len(set(bet_reds) & set(actual_reds))
        prize = calculate_prize(match_count, bet_blue == actual_blue)
        assert prize == 1  # 一等奖奖级为 1

    def test_no_prize_no_match(self, winning_numbers):
        """完全未中奖。"""
        bet_reds = [1, 2, 4, 5, 6, 8]
        bet_blue = 16
        actual_reds = winning_numbers["reds"]
        actual_blue = winning_numbers["blue"]
        match_count = len(set(bet_reds) & set(actual_reds))
        prize = calculate_prize(match_count, bet_blue == actual_blue)
        assert prize is None

    def test_sixth_prize_only_blue(self, winning_numbers):
        """六等奖：只中蓝球。"""
        bet_reds = [1, 2, 4, 5, 6, 8]
        bet_blue = winning_numbers["blue"]
        actual_reds = winning_numbers["reds"]
        actual_blue = winning_numbers["blue"]
        match_count = len(set(bet_reds) & set(actual_reds))
        prize = calculate_prize(match_count, bet_blue == actual_blue)
        assert prize == 6  # 六等奖奖级为 6

    def test_third_prize_6_red_no_blue(self, winning_numbers):
        """三等奖：6 红全中但蓝球未中。"""
        bet_reds = winning_numbers["reds"]
        bet_blue = winning_numbers["blue"] + 1 if winning_numbers["blue"] < 16 else 1
        actual_reds = winning_numbers["reds"]
        actual_blue = winning_numbers["blue"]
        match_count = len(set(bet_reds) & set(actual_reds))
        prize = calculate_prize(match_count, bet_blue == actual_blue)
        assert prize == 2  # 6红全中但蓝球未中 = 二等奖


# ============================================================
# evaluate_single_bet 测试
# ============================================================


class TestEvaluateSingleBet:
    def test_result_fields_present(self, sample_ssq_df, winning_numbers):
        """BetResult 字段完整性检查。"""
        result = evaluate_single_bet(
            red_balls=[1, 2, 3, 4, 5, 6],
            blue_ball=7,
            actual_reds=winning_numbers["reds"],
            actual_blue=winning_numbers["blue"],
            cost_per_bet=2.0,
        )
        assert isinstance(result, BetResult)
        assert hasattr(result, 'red_matches')
        assert hasattr(result, 'blue_matched')
        assert hasattr(result, 'prize')
        assert hasattr(result, 'profit')
        assert hasattr(result, 'roi')

    def test_profit_calculation(self, sample_ssq_df, winning_numbers):
        """利润计算正确性。"""
        result = evaluate_single_bet(
            red_balls=[1, 2, 3, 4, 5, 6],
            blue_ball=7,
            actual_reds=winning_numbers["reds"],
            actual_blue=winning_numbers["blue"],
            cost_per_bet=2.0,
        )
        assert result.profit == result.prize - result.cost


# ============================================================
# BacktestEngine 测试
# ============================================================


class TestBacktestEngine:
    def test_run_with_sample_data(self, sample_ssq_df):
        """BacktestEngine 能正常完成小规模回测。"""
        engine = BacktestEngine(sample_ssq_df, SSQ_CONFIG)
        report = engine.run(window_size=20, n_backtest=10)
        assert report is not None
        assert report.total_bets > 0
        assert report.roi is not None

    def test_random_baseline_exists(self, sample_ssq_df):
        """��� baseline �����ڡ�"""
        engine = BacktestEngine(sample_ssq_df, SSQ_CONFIG)
        report = engine.run(window_size=20, n_backtest=10)
        assert report.random_avg_match is not None
        assert report.random_avg_match >= 0


# ============================================================
# StrategyBacktestEngine 测试
# ============================================================


class TestStrategyBacktestEngine:
    def test_all_strategies_have_output(self, sample_ssq_df):
        """四种策略均有输出。"""
        engine = StrategyBacktestEngine(SSQ_CONFIG, window_size=20)
        report = engine.run(sample_ssq_df, n_backtest=10)
        expected_strategies = ["conservative", "aggressive", "balanced", "mystic"]
        for name in expected_strategies:
            assert name in report.performances, f"缺少策略: {name}"
            perf = report.performances[name]
            assert perf.total_bets > 0
            assert perf.roi is not None

    def test_performance_fields_complete(self, sample_ssq_df):
        """StrategyPerformance 字段完整。"""
        engine = StrategyBacktestEngine(SSQ_CONFIG, window_size=20)
        report = engine.run(sample_ssq_df, n_backtest=10)
        for name, perf in report.performances.items():
            assert isinstance(perf, StrategyPerformance)
            assert perf.avg_match >= 0
            assert perf.blue_match_count >= 0
            assert perf.total_bets > 0


# ============================================================
# generate_ranking_report 便捷函数测试
# ============================================================


class TestRankingReport:
    def test_report_structure(self, sample_ssq_df):
        """排名报告结构正确。"""
        report = generate_ranking_report("ssq", window_size=20, n_backtest=10, df=sample_ssq_df)
        assert report.total_windows > 0
        assert len(report.performances) >= 4
        assert report.random_baseline is not None
