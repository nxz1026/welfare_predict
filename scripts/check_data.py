# -*- coding: utf-8 -*-
"""数据一致性校验脚本 - 比对本地数据与官方最新数据。"""
import sys
sys.path.insert(0, ".")
import pandas as pd
from src.data_fetcher import get_current_issue, download_history, _repair_sd_data

def check_code(code: str):
    """检查单个彩种的数据一致性。"""
    from src.config import LOTTERY_CONFIGS, PATHS, DATA_FILE_NAME
    cfg = LOTTERY_CONFIGS[code]
    data_path = PATHS["data"] / code / DATA_FILE_NAME

    # 获取官方最新期号
    try:
        remote_issue = get_current_issue(code)
    except Exception as e:
        print(f"  ❌ 获取官方期号失败: {e}")
        return

    # 读取本地数据
    if not data_path.exists():
        print(f"  ❌ 本地数据文件不存在: {data_path}")
        return

    try:
        df = pd.read_csv(data_path, encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(data_path, encoding="utf-8")

    local_issue = str(df["期数"].iloc[0]) if len(df) > 0 else "无"
    local_date = str(df["开奖日期"].iloc[0]) if "开奖日期" in df.columns and len(df) > 0 else "无"

    # 比对期号
    match = "✅ 一致" if local_issue == str(remote_issue) else f"❌ 不一致（本地={local_issue}, 官方={remote_issue}）"

    print(f"  彩种: {cfg.name} ({code})")
    print(f"  本地最新期: {local_issue}, 日期: {local_date}")
    print(f"  官方最新期: {remote_issue}")
    print(f"  期号比对: {match}")
    print(f"  本地总记录: {len(df)} 期")

    # 显示最新一期数据
    if len(df) > 0:
        print(f"  最新一期数据:")
        row = df.iloc[0]
        if code == "ssq":
            reds = [str(int(row[f"红球_{i}"])) for i in range(1, 7)]
            blue = str(int(row["蓝球_1"]))
            print(f"    红球: {' '.join(reds)}  蓝球: {blue}")
        elif code == "sd":
            digits = [str(int(row[f"红球_{i}"])) for i in range(1, 4)]
            try_code = row.get("试机号", "无")
            print(f"    开奖号: {' '.join(digits)}  试机号: {try_code}")
        elif code == "qlc":
            reds = [str(int(row[f"红球_{i}"])) for i in range(1, 8)]
            blue = str(int(row["蓝球_1"]))
            print(f"    红球: {' '.join(reds)}  特别号: {blue}")
        elif code == "kl8":
            nums = [str(int(row[f"红球_{i}"])) for i in range(1, 21)]
            print(f"    开奖号: {' '.join(nums)}")
    print()

if __name__ == "__main__":
    print("=" * 60)
    print("第一阶段：数据一致性校验")
    print("=" * 60)
    for code in ["ssq", "sd", "qlc", "kl8"]:
        try:
            check_code(code)
        except Exception as e:
            print(f"  ❌ {code} 校验异常: {e}\n")