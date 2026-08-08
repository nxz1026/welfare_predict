# -*- coding: utf-8 -*-
"""验证4种玩法数据一致性"""
import pandas as pd
from src.data_fetcher import get_current_issue, LotteryHttpClient
from src.config import LOTTERY_CONFIGS, NETWORK_CONFIG

client = LotteryHttpClient(
    timeout=NETWORK_CONFIG["timeout"],
    retries=NETWORK_CONFIG["retry_count"],
    backoff_factor=NETWORK_CONFIG.get("backoff_factor", 0.6),
    user_agent=NETWORK_CONFIG["user_agent"],
)

for code in ["ssq", "sd", "qlc", "kl8"]:
    cfg = LOTTERY_CONFIGS[code]
    try:
        online = get_current_issue(code, client)
    except Exception as e:
        online = f"Error: {e}"

    try:
        df = pd.read_csv(f"data/{code}/data.csv", encoding="utf-8-sig")
        local = str(df["期数"].iloc[0])
        dt = df["开奖日期"].iloc[0] if "开奖日期" in df.columns else "N/A"
        total = len(df)
    except Exception as e:
        local = f"Error: {e}"
        dt = "N/A"
        total = 0

    match = "✅" if local == online else "❌"
    print(f"{cfg.name}({code}): 在线={online}, 本地={local}, 日期={dt}, 总={total} {match}")