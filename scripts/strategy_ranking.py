# -*- coding: utf-8 -*-
"""
策略排行榜脚本。

用法:
    python scripts/strategy_ranking.py
    python scripts/strategy_ranking.py --code ssq --window 200 --backtest 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_lottery_config, ensure_runtime_directories
from src.data_fetcher import load_history
from src.strategy_backtest import generate_ranking_report
from src.visualization import generate_strategy_ranking_text


def main():
    parser = argparse.ArgumentParser(description="福彩策略排行榜")
    parser.add_argument("--code", default="ssq", help="彩票代码")
    parser.add_argument("--window", type=int, default=200, help="回测窗口大小")
    parser.add_argument("--backtest", type=int, help="回测期数")
    parser.add_argument("--output", "-o", help="输出文件路径")
    args = parser.parse_args()

    ensure_runtime_directories()

    # 加载数据
    logger.info("加载 {} 历史数据...", args.code)
    df = load_history(args.code)
    logger.info("数据加载完成: {} 期", len(df))

    # 回测
    logger.info("开始策略回测...")
    report = generate_ranking_report(args.code, args.window, args.backtest)

    # 生成排行榜文本
    ranking = {}
    for name, perf in report.performances.items():
        ranking[name] = {
            "avg_match": perf.avg_match,
            "blue_match": perf.blue_match_count,
            "roi": perf.roi,
        }

    ranking_text = generate_strategy_ranking_text(ranking)

    # 输出
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(ranking_text)
        logger.info("排行榜已保存: {}", args.output)
    else:
        print()
        print(ranking_text)

    # 详细数据
    print()
    print("详细数据:")
    for name, perf in report.performances.items():
        print(f"  {name}: 投注{perf.total_bets}期, 成本¥{perf.total_cost:.0f}, "
              f"奖金¥{perf.total_reward:.0f}, 净利润¥{perf.net_profit:.0f}, "
              f"ROI={perf.roi:.2%}")
        print(f"    平均命中: {perf.avg_match:.2f}个红球, 蓝球命中: {perf.blue_match_count}次")
        print(f"    奖级分布: {perf.prize_counts}")

    print()
    print(f"随机 baseline: {report.random_baseline:.2f} 个红球")


if __name__ == "__main__":
    main()
