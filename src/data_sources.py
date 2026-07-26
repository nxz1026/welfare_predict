# -*- coding: utf-8 -*-
"""
数据源抽象层。

提供两种数据源：
- ``fivehundred``：从 500.com 抓取历史数据，默认源。
- ``local_csv``：从本地 CSV 文件导入，用于离线/二开/备用数据。

统一接口：fetch_history(code) -> DataFrame
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from loguru import logger

from .config import ALLOWED_DOMAINS, PATHS, LOTTERY_CONFIGS, LotteryModelConfig
from .data_fetcher import (
    download_history as _download_history,
    load_history as _load_history,
    LotteryHttpClient,
)


class DataSourceError(Exception):
    """数据源不可用或返回异常数据时抛出。"""


@dataclass(frozen=True)
class DataSourceContext:
    """描述一次数据获取的上下文。"""

    code: str
    source: str
    path: Optional[Path] = None
    start_issue: Optional[int] = None
    end_issue: Optional[int] = None


def _validate_local_csv_schema(df: pd.DataFrame, cfg: LotteryModelConfig) -> None:
    """校验本地 CSV 是否包含必需字段。"""

    if "期数" not in df.columns:
        raise DataSourceError("本地 CSV 缺少必需字段：期数")
    red_columns = [f"红球_{idx + 1}" for idx in range(cfg.red.sequence_len)]
    missing = [col for col in red_columns if col not in df.columns]
    if missing:
        raise DataSourceError(f"本地 CSV 缺少红球字段：{missing}")
    if cfg.blue:
        blue_columns = [f"蓝球_{idx + 1}" for idx in range(cfg.blue.sequence_len)]
        missing = [col for col in blue_columns if col not in df.columns]
        if missing:
            raise DataSourceError(f"本地 CSV 缺少蓝球字段：{missing}")


class FivehundredSource:
    """默认在线数据源：500.com。"""

    name = "fivehundred"

    def fetch_history(
        self,
        context: DataSourceContext,
        client: Optional[LotteryHttpClient] = None,
    ) -> pd.DataFrame:
        cfg = LOTTERY_CONFIGS[context.code]
        if context.path is None:
            _download_history(
                context.code,
                start=context.start_issue,
                end=context.end_issue,
                client=client,
            )
            return _load_history(context.code)
        path = context.path
        if not path.exists():
            raise DataSourceError(f"数据文件不存在：{path}")
        df = pd.read_csv(path, encoding="utf-8")
        _validate_local_csv_schema(df, cfg)
        return df

    def fetch_latest(self, context: DataSourceContext) -> str:
        from .data_fetcher import get_current_issue
        return get_current_issue(context.code)


class LocalCsvSource:
    """离线数据源：本地 CSV。"""

    name = "local_csv"

    def fetch_history(self, context: DataSourceContext) -> pd.DataFrame:
        cfg = LOTTERY_CONFIGS[context.code]
        path = context.path or (PATHS["data"] / context.code / "data.csv")
        if not path.exists():
            raise DataSourceError(f"本地数据文件不存在：{path}")
        df = pd.read_csv(path, encoding="utf-8")
        _validate_local_csv_schema(df, cfg)
        return df

    def fetch_latest(self, context: DataSourceContext) -> str:
        df = self.fetch_history(context)
        return str(df["期数"].max())


_SOURCES: Dict[str, FivehundredSource | LocalCsvSource] = {
    FivehundredSource.name: FivehundredSource(),
    LocalCsvSource.name: LocalCsvSource(),
}


def get_source(name: str = "fivehundred") -> FivehundredSource | LocalCsvSource:
    if name not in _SOURCES:
        raise DataSourceError(f"未知数据源：{name}，可选：{', '.join(_SOURCES)}")
    return _SOURCES[name]


def fetch_history(
    code: str,
    source: str = "fivehundred",
    path: Optional[Path] = None,
    start_issue: Optional[int] = None,
    end_issue: Optional[int] = None,
) -> pd.DataFrame:
    context = DataSourceContext(
        code=code,
        source=source,
        path=path,
        start_issue=start_issue,
        end_issue=end_issue,
    )
    return get_source(source).fetch_history(context)


def fetch_latest_issue(code: str, source: str = "fivehundred", path: Optional[Path] = None) -> str:
    context = DataSourceContext(code=code, source=source, path=path)
    return get_source(source).fetch_latest(context)


__all__ = [
    "DataSourceContext",
    "DataSourceError",
    "FivehundredSource",
    "LocalCsvSource",
    "fetch_history",
    "fetch_latest_issue",
    "get_source",
]
