# -*- coding: utf-8 -*-
"""
一键生成推荐脚本。

用法:
    python scripts/generate_recommendation.py
    python scripts/generate_recommendation.py --code ssq
    python scripts/generate_recommendation.py --lucky 6 8 9 16
    python scripts/generate_recommendation.py --output ticket.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_lottery_config, ensure_runtime_directories
from src.data_fetcher import load_history
from src.recommendation import RecommendationEngine, generate_recommendation
from src.visualization import (
    generate_ticket_text,
    generate_missing_report,
    plot_hot_cold_vs_random,
    plot_sum_distribution,
)


def main():
    parser = argparse.ArgumentParser(description="福彩推荐系统")
    parser.add_argument("--code", default="ssq", help="彩票代码 (ssq/dlt/3d)")
    parser.add_argument("--lucky", nargs="+", type=int, help="幸运数字")
    parser.add_argument("--output", "-o", help="输出文件路径")
    parser.add_argument("--strategy", help="指定策略 (保守型/激进型/平衡型/玄学型)")
    parser.add_argument("--plot", action="store_true", help="生成图表")
    parser.add_argument("--missing", action="store_true", help="生成遗漏报告")
    args = parser.parse_args()

    ensure_runtime_directories()

    # 加载数据
    logger.info("加载 {} 历史数据...", args.code)
    df = load_history(args.code)
    logger.info("数据加载完成: {} 期", len(df))

    # 生成推荐
    logger.info("生成推荐...")
    rec = generate_recommendation(args.code, args.lucky)

    # 生成小票文本
    ticket_text = generate_ticket_text(rec, df, args.strategy)

    # 输出
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(ticket_text)
        logger.info("推荐已保存: {}", args.output)
    else:
        print()
        print(ticket_text)

    # 遗漏报告
    if args.missing:
        config = get_lottery_config(args.code)
        missing_text = generate_missing_report(df, config)
        print()
        print(missing_text)

    # 生成图表
    if args.plot:
        config = get_lottery_config(args.code)
        plot_path = f"output/{args.code}_hot_cold.png"
        plot_hot_cold_vs_random(df, config, save_path=plot_path)
        logger.info("图表已保存: {}", plot_path)

        sum_path = f"output/{args.code}_sum_dist.png"
        plot_sum_distribution(df, save_path=sum_path)
        logger.info("图表已保存: {}", sum_path)


if __name__ == "__main__":
    main()
