# -*- coding:utf-8 -*-
"""
通用 CSV 导入工具。

把外部 CSV 导入到项目的 data/<code>/data.csv，自动做 schema 校验与最小清洗。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import LOTTERY_CONFIGS, PATHS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导入外部 CSV 到项目 data 目录")
    parser.add_argument("--src", required=True, help="源 CSV 文件路径")
    parser.add_argument("--name", required=True, help="彩票类型代码，如 ssq / dlt")
    parser.add_argument("--force", action="store_true", help="覆盖已有 data.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    code = args.name.lower().strip()
    if code not in LOTTERY_CONFIGS:
        raise SystemExit(f"不支持的彩票类型：{args.name}，有效选项：{', '.join(LOTTERY_CONFIGS.keys())}")

    src_path = Path(args.src)
    if not src_path.exists():
        raise SystemExit(f"源文件不存在：{src_path}")

    dest_dir = PATHS["data"] / code
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / "data.csv"

    if dest_path.exists() and not args.force:
        raise SystemExit(f"目标文件已存在：{dest_path}，如需覆盖请加 --force")

    df = pd.read_csv(src_path, encoding="utf-8")
    df.to_csv(dest_path, index=False, encoding="utf-8")
    logger.success("CSV 导入完成：{} -> {} ({} 行)", src_path, dest_path, len(df))


if __name__ == "__main__":
    main()
