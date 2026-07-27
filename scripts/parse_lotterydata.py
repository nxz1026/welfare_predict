# -*- coding: utf-8 -*-
"""
解析 lotterydata 数据集为项目标准 CSV 格式。

数据来源：https://github.com/BEWINDOWEB/lotterydata
- 500.com 源：期号,红1,...,红6,蓝,快乐星期天,奖池,一等奖注数,一等奖奖金,二等奖注数,二等奖奖金,总投注额,开奖日期
- 中彩网 源：序号,开奖日期,期号,红1,...,红6,蓝,销售额,一等奖注数,一等奖奖金,二等奖注数,二等奖奖金,三等奖注数,三等奖奖金,奖池

输出 CSV 格式：期数,红球_1,...,红球_6,蓝球_1,一等奖注数,一等奖奖金,奖池,开奖日期
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger


def parse_500_file(input_path: str, output_path: str) -> int:
    """
    解析 500.com 源文件。

    Returns:
        解析的行数
    """
    rows = []
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for line in reader:
            if len(line) < 8:
                continue
            try:
                issue = line[0].strip()
                reds = [int(line[i].strip()) for i in range(1, 7)]
                blue = int(line[7].strip())
                # 奖池（第 9 列，索引 8）
                pool = line[8].strip() if len(line) > 8 else ""
                # 开奖日期（最后一列）
                date = line[-1].strip() if line else ""

                rows.append({
                    "期数": issue,
                    "红球_1": reds[0],
                    "红球_2": reds[1],
                    "红球_3": reds[2],
                    "红球_4": reds[3],
                    "红球_5": reds[4],
                    "红球_6": reds[5],
                    "蓝球_1": blue,
                    "奖池": pool,
                    "开奖日期": date,
                })
            except (ValueError, IndexError):
                continue

    df = pd.DataFrame(rows)
    df.sort_values("期数", inplace=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    return len(df)


def parse_zhcw_file(input_path: str, output_path: str) -> int:
    """
    解析中彩网源文件。

    Returns:
        解析的行数
    """
    rows = []
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for line in reader:
            if len(line) < 9:
                continue
            try:
                issue = line[2].strip()
                reds = [int(line[i].strip()) for i in range(3, 9)]
                blue = int(line[9].strip())
                # 奖池（最后一列）
                pool = line[-1].strip() if line else ""
                # 开奖日期
                date = line[1].strip() if len(line) > 1 else ""

                rows.append({
                    "期数": issue,
                    "红球_1": reds[0],
                    "红球_2": reds[1],
                    "红球_3": reds[2],
                    "红球_4": reds[3],
                    "红球_5": reds[4],
                    "红球_6": reds[5],
                    "蓝球_1": blue,
                    "奖池": pool,
                    "开奖日期": date,
                })
            except (ValueError, IndexError):
                continue

    df = pd.DataFrame(rows)
    df.sort_values("期数", inplace=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    return len(df)


def main():
    """解析所有数据文件"""
    data_dir = Path(__file__).parent.parent / "data"

    # 双色球 - 500.com 源
    ssq_500 = data_dir / "ssq" / "data.txt"
    ssq_out = data_dir / "ssq" / "data.csv"
    if ssq_500.exists():
        n = parse_500_file(str(ssq_500), str(ssq_out))
        logger.info("双色球(500.com): {} 期 -> {}", n, ssq_out)

    # 双色球 - 中彩网源（用于交叉校验）
    ssq_zhcw = data_dir / "ssq" / "data_zhcw.txt"
    ssq_zhcw_out = data_dir / "ssq" / "data_zhcw.csv"
    if ssq_zhcw.exists():
        n = parse_zhcw_file(str(ssq_zhcw), str(ssq_zhcw_out))
        logger.info("双色球(中彩网): {} 期 -> {}", n, ssq_zhcw_out)


if __name__ == "__main__":
    main()
