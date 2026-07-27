# -*- coding: utf-8 -*-
"""
用户历史记录管理。

记录顾客常买号码、推荐历史、中奖情况。
用于个性化推荐 — "您上次买的 06 16 这次被系统采用了"。

ponytail: 用 JSON 文件存储，不引入数据库。
升级路径：如果需要多用户/并发，换 SQLite。
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger


class UserHistory:
    """
    单用户历史记录。
    """

    def __init__(self, user_id: str, storage_dir: str = "data/users"):
        self.user_id = user_id
        self.storage_path = Path(storage_dir) / f"{user_id}.json"
        self.data: Dict = self._load()

    def _load(self) -> Dict:
        if self.storage_path.exists():
            with open(self.storage_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "user_id": self.user_id,
            "created_at": datetime.now().isoformat(),
            "frequently_bought": {},  # "06": 12 (购买次数)
            "recommendations": [],    # 推荐历史
            "wins": [],               # 中奖记录
            "total_spent": 0.0,
            "total_won": 0.0,
        }

    def _save(self):
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def record_purchase(self, numbers: Dict[str, List[int]], cost: float = 2.0):
        """
        记录一次购买。

        Args:
            numbers: {"reds": [1,2,3,4,5,6], "blue": 7}
            cost: 花费（元）
        """
        reds = numbers.get("reds", [])
        blue = numbers.get("blue")

        # 更新红球频次
        for num in reds:
            key = str(num)
            self.data["frequently_bought"][key] = self.data["frequently_bought"].get(key, 0) + 1

        # 更新蓝球频次
        if blue is not None:
            key = f"blue_{blue}"
            self.data["frequently_bought"][key] = self.data["frequently_bought"].get(key, 0) + 1

        self.data["total_spent"] += cost
        self._save()

    def record_recommendation(self, rec_summary: Dict):
        """记录一次推荐"""
        self.data["recommendations"].append({
            "timestamp": datetime.now().isoformat(),
            **rec_summary,
        })
        # 只保留最近 50 条
        if len(self.data["recommendations"]) > 50:
            self.data["recommendations"] = self.data["recommendations"][-50:]
        self._save()

    def record_win(self, level: int, amount: float, issue: str = ""):
        """记录一次中奖"""
        self.data["wins"].append({
            "issue": issue,
            "level": level,
            "amount": amount,
            "timestamp": datetime.now().isoformat(),
        })
        self.data["total_won"] += amount
        self._save()

    def get_frequent_numbers(self, top_n: int = 6) -> Dict:
        """
        获取用户最常买的号码。

        Returns:
            {"reds": [6, 8, 16, ...], "blue": 8}
        """
        red_freq = {}
        blue_freq = {}

        for key, count in self.data["frequently_bought"].items():
            if key.startswith("blue_"):
                num = int(key.split("_")[1])
                blue_freq[num] = count
            else:
                num = int(key)
                red_freq[num] = count

        # 排序取前 N
        top_reds = sorted(red_freq.keys(), key=lambda x: red_freq[x], reverse=True)[:top_n]
        top_blue = max(blue_freq.keys(), key=lambda x: blue_freq[x]) if blue_freq else None

        return {
            "reds": top_reds,
            "blue": top_blue,
            "red_freq": {str(k): red_freq[k] for k in top_reds},
            "blue_freq": blue_freq,
        }

    def get_lucky_numbers(self) -> List[int]:
        """获取用户幸运数字（最常买的号码，用于玄学策略）"""
        freq = self.get_frequent_numbers()
        return freq["reds"]

    def get_stats(self) -> Dict:
        """获取用户统计"""
        total_spent = self.data["total_spent"]
        total_won = self.data["total_won"]
        return {
            "total_spent": total_spent,
            "total_won": total_won,
            "net_profit": total_won - total_spent,
            "roi": (total_won - total_spent) / total_spent if total_spent > 0 else 0,
            "total_purchases": len(self.data["recommendations"]),
            "total_wins": len(self.data["wins"]),
        }


# ============================================================
# 便捷函数
# ============================================================

def get_user(user_id: str) -> UserHistory:
    """获取用户历史记录"""
    return UserHistory(user_id)


def record_purchase_and_recommend(
    user_id: str,
    purchased_numbers: Dict[str, List[int]],
    recommendation: Dict,
    cost: float = 2.0,
):
    """记录购买 + 推荐"""
    user = UserHistory(user_id)
    user.record_purchase(purchased_numbers, cost)
    user.record_recommendation(recommendation)


__all__ = [
    "UserHistory",
    "get_user",
    "record_purchase_and_recommend",
]
