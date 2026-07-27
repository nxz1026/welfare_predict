# -*- coding: utf-8 -*-
"""
模型训练脚本（TensorFlow 2.15+ 版本）。

示例：
    python scripts/train.py --name ssq --window-size 5 --red-epochs 60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

# Ensure the project root is on sys.path so `src` imports work and the
# bootstrap shim can be imported early.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# import as early as possible; src.bootstrap is best-effort
try:
    import src.bootstrap  # noqa: F401
except Exception:
    # If bootstrap fails, continue; the bootstrap shim is non-critical
    pass

from src.common import get_data_run, train_pipeline  # noqa: E402
from src.config import LOTTERY_CONFIGS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练彩票模型（支持新旧两种模式）")
    parser.add_argument(
        "--name",
        default="ssq",
        help="彩票类型代码，如 ssq / sd / qlc，默认 ssq",
    )
    parser.add_argument("--window-size", type=int, default=None, help="时间窗口大小（旧模式）")
    parser.add_argument("--batch-size", type=int, default=None, help="训练批大小（旧模式）")
    parser.add_argument("--red-epochs", type=int, default=None, help="红球模型训练轮数（旧模式）")
    parser.add_argument("--blue-epochs", type=int, default=None, help="蓝球模型训练轮数（旧模式）")
    parser.add_argument("--download-data", action="store_true", help="训练前自动下载最新数据")
    parser.add_argument("--source", default=None, help="数据源：fivehundred / local_csv，默认读取配置")
    # 新模式参数
    parser.add_argument(
        "--method",
        default="xgb",
        choices=["xgb", "lstm", "poisson", "stacking"],
        help="训练方法（新模式，默认 xgb）",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="使用旧版 TF LSTM 模式（pipeline.py）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    code = args.name.lower().strip()
    if code not in LOTTERY_CONFIGS:
        raise SystemExit(f"不支持的彩票类型：{args.name}，有效选项：{', '.join(LOTTERY_CONFIGS.keys())}")

    if args.download_data:
        logger.info("开始下载数据...")
        get_data_run(code)

    if args.legacy:
        # 旧模式：TF LSTM
        summary = train_pipeline(
            name=code,
            window_size=args.window_size,
            batch_size=args.batch_size,
            red_epochs=args.red_epochs,
            blue_epochs=args.blue_epochs,
            source=args.source,
        )
        logger.success("训练完成，详情见 model/{}/window_{}/{}\n", summary.code, summary.window_size, "metadata.json")
    else:
        # 新模式：XGBoost / LSTM / Poisson / Stacking
        from src.unified_pipeline import UnifiedPipeline
        from src.data_fetcher import load_history

        df = load_history(code)
        pipeline = UnifiedPipeline(code, method=args.method)
        summary = pipeline.train(df)
        logger.success("训练完成: {} ({} 期数据, {} 特征, 方法={})",
                        summary.name, summary.n_samples, summary.n_features, args.method)


if __name__ == "__main__":
    main()

