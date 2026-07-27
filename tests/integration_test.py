# -*- coding: utf-8 -*-
"""
集成测试脚本。

端到端测试整个流程：数据获取 → 特征工程 → 模型训练 → 预测 → 回测 → 评估。
"""

from __future__ import annotations

import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from loguru import logger

from src.config import get_lottery_config, ensure_runtime_directories
from src.data_fetcher import load_history
from src.data_sources import DataSourceManager
from src.feature_engineering import build_feature_matrix, build_binary_labels
from src.modeling import XGBoostPredictor
from src.model_lstm import LSTMPredictor
from src.model_poisson import PoissonPrior
from src.model_stacking import StackingMetaLearner
from src.backtest import BacktestEngine, calculate_prize


def run_e2e_test(code: str = "ssq", window_size: int = 20):
    """
    端到端测试。

    Args:
        code: 彩票代码
        window_size: 回测训练窗口
    """
    logger.info("=" * 50)
    logger.info("端到端集成测试: {}", code)
    logger.info("=" * 50)

    ensure_runtime_directories()
    cfg = get_lottery_config(code)

    # 1. 数据获取
    logger.info("\n[1/6] 数据获取...")
    try:
        df = load_history(code)
        logger.info("数据加载完成: {} 期", len(df))
    except Exception as e:
        logger.error("数据获取失败: {}", e)
        return False

    if len(df) < window_size + 5:
        logger.warning("数据量不足: {} 期，建议至少 {} 期", len(df), window_size + 5)

    # 2. 特征工程
    logger.info("\n[2/6] 特征工程...")
    features = build_feature_matrix(df, cfg)
    labels = build_binary_labels(df, cfg)
    logger.info("特征矩阵: {}, 标签矩阵: {}", features.shape, labels.shape)

    # 时间对齐
    X = features.iloc[:-1].values
    y = labels[1:]
    logger.info("训练数据: X={}, y={}", X.shape, y.shape)

    # 3. 模型训练
    logger.info("\n[3/6] 模型训练...")

    # XGBoost
    logger.info("训练 XGBoost...")
    xgb = XGBoostPredictor(cfg, n_estimators=50, max_depth=3, learning_rate=0.1)
    xgb.train(X, y)

    # LSTM
    logger.info("训练 LSTM...")
    lstm = LSTMPredictor(cfg, lstm_units=[32, 16], dropout=0.3,
                         learning_rate=0.001, batch_size=8, epochs=20)
    lstm.train(X, y)

    # 泊松先验
    logger.info("训练泊松先验...")
    poisson = PoissonPrior(cfg)
    poisson.train(X, y)

    # Stacking
    logger.info("训练 Stacking...")
    xgb_proba = xgb.predict_proba(X)
    lstm_proba = lstm.predict_proba(X)
    poisson_proba = poisson.predict_proba(X)

    probas = {'xgboost': xgb_proba, 'lstm': lstm_proba, 'poisson': poisson_proba}
    stacking = StackingMetaLearner(cfg)
    stacking.train(probas, y)

    logger.info("所有模型训练完成!")

    # 4. 预测
    logger.info("\n[4/6] 预测...")
    final_pred = stacking.predict(probas)
    pred_balls = sorted([int(x + 1) for x in final_pred[0]])
    logger.info("预测下一期号码: {}", pred_balls)

    # 5. 回测
    logger.info("\n[5/6] 回测...")
    engine = BacktestEngine(cfg, window_size=window_size, bet_cost=2.0)
    report = engine.run(df)

    # 6. 评估总结
    logger.info("\n[6/6] 评估总结...")
    logger.info("=" * 50)
    logger.info("集成测试结果汇总")
    logger.info("=" * 50)
    logger.info("数据量: {} 期", len(df))
    logger.info("特征维度: {}", features.shape[1])
    logger.info("模型: XGBoost + LSTM + Poisson + Stacking")
    logger.info("回测期数: {}", report.total_bets)
    logger.info("平均命中: {:.2f} 个 (随机: {:.2f})", report.avg_match, report.random_avg_match)
    logger.info("提升: {:.1f}%", report.match_improvement)
    logger.info("ROI: {:.2%}", report.roi)

    # 各奖级命中
    logger.info("奖级分布:")
    for level in range(1, 7):
        if report.prize_counts[level] > 0:
            logger.info("  {}等: {} 次 ({:.2%})", level, report.prize_counts[level], report.prize_rates[level])

    logger.info("=" * 50)
    logger.info("端到端测试完成!")
    logger.info("=" * 50)

    return True


def main():
    parser = __import__('argparse').ArgumentParser(description="彩票预测系统集成测试")
    parser.add_argument("--code", default="ssq", help="彩票代码")
    parser.add_argument("--window", type=int, default=20, help="回测窗口大小")
    args = parser.parse_args()

    success = run_e2e_test(args.code, args.window)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
