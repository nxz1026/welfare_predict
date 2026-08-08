# -*- coding: utf-8 -*-
"""
数据源抽象层。

支持多种数据源：
1. 在线抓取：500.com（30 期默认，快速验证）
2. API 接入：天行数据（完整历史，需 API key）
3. 本地导入：用户提供 CSV 文件
4. 缓存机制：本地 CSV 缓存 + 增量更新
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from loguru import logger

from .config import LotteryModelConfig, PATHS, LOTTERY_CONFIGS, NETWORK_CONFIG


@dataclass
class FetchResult:
    """数据获取结果"""
    source: str
    total_issues: int
    saved_path: str
    timestamp: str
    is_incremental: bool = False


class DataSource(ABC):
    """数据源基类"""

    @abstractmethod
    def fetch_history(
        self,
        code: str,
        start_issue: Optional[int] = None,
        end_issue: Optional[int] = None,
    ) -> pd.DataFrame:
        """获取历史数据"""
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass


class Web500Source(DataSource):
    """500.com 网页抓取"""

    def __init__(self) -> None:
        from .data_fetcher import LotteryHttpClient
        self.client = LotteryHttpClient(
            timeout=NETWORK_CONFIG["timeout"],
            retries=NETWORK_CONFIG["retry_count"],
            backoff_factor=NETWORK_CONFIG.get("backoff_factor", 0.6),
            user_agent=NETWORK_CONFIG["user_agent"],
        )

    def get_name(self) -> str:
        return "500.com"

    def fetch_history(
        self,
        code: str,
        start_issue: Optional[int] = None,
        end_issue: Optional[int] = None,
    ) -> pd.DataFrame:
        """从 500.com 抓取数据"""
        from .data_fetcher import download_history
        result = download_history(code, start=start_issue, end=end_issue)
        return pd.read_csv(result.saved_path, encoding="utf-8")


class TianyanAPISource(DataSource):
    """天行数据 API"""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("TIANYAN_API_KEY")
        if not self.api_key:
            logger.warning("天行数据 API key 未设置，跳过")

    def get_name(self) -> str:
        return "天行数据"

    def fetch_history(
        self,
        code: str,
        start_issue: Optional[int] = None,
        end_issue: Optional[int] = None,
    ) -> pd.DataFrame:
        """通过天行数据 API 获取"""
        if not self.api_key:
            raise ValueError("天行数据 API key 未设置")

        from .config import LOTTERY_CONFIGS
        cfg = LOTTERY_CONFIGS.get(code)
        if cfg is None:
            raise ValueError(f"未知彩种: {code}")

        # 天行数据接口
        url = "https://api.tianapi.com/txapi/lottery/index"
        params = {
            "key": self.api_key,
            "code": code,
            "num": 100,
        }

        import requests
        resp = requests.get(url, params=params, timeout=30)
        data = resp.json()

        if data.get("code") != 200:
            raise ValueError(f"API 错误: {data.get('msg')}")

        records = data.get("newslist", [])
        if not records:
            raise ValueError("API 返回空数据")

        red_len = cfg.red.sequence_len
        blue_len = cfg.blue.sequence_len if cfg.blue else 0

        # 动态映射 API 字段到标准列名
        rows = []
        for record in records:
            row = {"期数": record.get("expect")}
            for i in range(1, red_len + 1):
                row[f"红球_{i}"] = int(record.get(f"red{i}", 0))
            if blue_len > 0:
                for i in range(1, blue_len + 1):
                    row[f"蓝球_{i}"] = int(record.get(f"blue{i}" if blue_len > 1 else "blue", 0))
            rows.append(row)

        df = pd.DataFrame(rows)
        df.sort_values("期数", inplace=True)
        return df.reset_index(drop=True)


class LocalCSVSource(DataSource):
    """本地 CSV 文件"""

    def __init__(self, file_path: Optional[str] = None) -> None:
        self.file_path = file_path

    def get_name(self) -> str:
        return "本地CSV"

    def fetch_history(
        self,
        code: str,
        start_issue: Optional[int] = None,
        end_issue: Optional[int] = None,
    ) -> pd.DataFrame:
        """从本地 CSV 读取"""
        path = self.file_path or str(PATHS["data"] / code / "data.csv")
        df = pd.read_csv(path, encoding="utf-8")
        df.sort_values("期数", inplace=True)
        return df.reset_index(drop=True)


class DataSourceManager:
    """数据源管理器"""

    def __init__(self) -> None:
        self._sources: Dict[str, DataSource] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """注册默认数据源"""
        web500 = Web500Source()
        self.register(web500)
        self.register(LocalCSVSource())
        # fivehundred 作为 500.com 的别名
        self._sources["fivehundred"] = web500

        # 如果有 API key，注册天行数据
        if os.getenv("TIANYAN_API_KEY"):
            self.register(TianyanAPISource())

    def register(self, source: DataSource) -> None:
        """注册数据源"""
        self._sources[source.get_name()] = source
        logger.debug("注册数据源: {}", source.get_name())

    def get_source(self, name: str) -> DataSource:
        """获取数据源"""
        if name not in self._sources:
            raise ValueError(f"未知数据源: {name}，可选: {list(self._sources.keys())}")
        return self._sources[name]

    def fetch_with_fallback(
        self,
        code: str,
        sources: Optional[list] = None,
    ) -> pd.DataFrame:
        """
        按优先级尝试多个数据源。

        Args:
            code: 彩票代码
            sources: 数据源名称列表，默认 ["500.com", "本地CSV"]
        """
        if sources is None:
            sources = ["500.com", "本地CSV"]

        for source_name in sources:
            try:
                source = self.get_source(source_name)
                df = source.fetch_history(code)
                logger.info("从 {} 获取了 {} 期数据", source_name, len(df))
                return df
            except Exception as e:
                logger.warning("{} 获取失败: {}", source_name, e)
                continue

        raise ValueError(f"所有数据源均失败: {sources}")


def fetch_history(
    code: str,
    start_issue: Optional[int] = None,
    end_issue: Optional[int] = None,
    source: str = "fivehundred",
) -> pd.DataFrame:
    """从指定数据源获取历史数据。"""
    manager = DataSourceManager()
    src = manager.get_source(source)
    return src.fetch_history(code, start_issue, end_issue)


def fetch_latest_issue(code: str) -> str:
    """获取指定彩票的最新期号。"""
    from .data_fetcher import get_current_issue
    return get_current_issue(code)


def get_source(name: str) -> DataSource:
    """按名称获取数据源实例。"""
    manager = DataSourceManager()
    return manager.get_source(name)


__all__ = [
    "DataSource",
    "Web500Source",
    "TianyanAPISource",
    "LocalCSVSource",
    "DataSourceManager",
    "FetchResult",
    "fetch_history",
    "fetch_latest_issue",
    "get_source",
]
