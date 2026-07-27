# -*- coding: utf-8 -*-
"""
数据抓取模块，负责从 500.com 拉取彩票历史数据并保存到本地。

特点：
1. 使用带重试的 requests.Session，满足网络安全要求；
2. 输出 Pandas DataFrame，供预处理与训练使用；
3. 支持多种彩种的数据下载（ssq/dlt/pls/qxc/sd/qlc）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from loguru import logger
from requests.adapters import HTTPAdapter
from urllib3 import Retry

from .config import (
    ALLOWED_DOMAINS,
    DATA_FILE_NAME,
    LOTTERY_CONFIGS,
    NETWORK_CONFIG,
    PATHS,
    LotteryModelConfig,
    ensure_runtime_directories,
)


@dataclass
class DownloadResult:
    """描述一次下载操作的元信息。"""

    code: str
    total_issues: int
    saved_path: str
    timestamp: str


class LotteryHttpClient:
    """封装网络访问逻辑，提供带重试与域名校验的 GET 方法。"""

    def __init__(
        self,
        timeout: float,
        retries: int,
        backoff_factor: float,
        user_agent: str,
    ) -> None:
        self._timeout = timeout
        self._session = requests.Session()
        retry_strategy = Retry(
            total=retries,
            backoff_factor=backoff_factor,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET"]),
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)
        self._headers = {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9",
        }

    def get_text(self, url: str) -> str:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if all(allowed not in domain for allowed in ALLOWED_DOMAINS):
            raise ValueError(f"禁止访问域名: {domain}")
        response = self._session.get(url, headers=self._headers, timeout=self._timeout)
        response.raise_for_status()
        response.encoding = "utf-8"
        return response.text


def _build_history_url(config: LotteryModelConfig, start: Optional[int], end: Optional[int]) -> str:
    base = f"https://datachart.500.com/{config.code}/history/"

    # .shtml 页面返回近 30 期；.php 仅返回 1 期（500.com 限制），故 SSQ/DLT 改用 .shtml
    if config.code in {"ssq", "dlt", "qlc"}:
        path = "history.shtml"
    elif config.code in {"qxc", "pls", "sd"}:
        path = "inc/history.php"
    else:
        path = "history.shtml"

    if path.endswith(".shtml"):
        return f"{base}{path}"

    # .php 接口用默认返回（~30 期），limit 过大导致异常
    return f"{base}{path}"


def _parse_issue_list(config: LotteryModelConfig, html: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "lxml")
    rows = []
    if config.code in {"ssq", "dlt"}:
        tbody = soup.find("tbody", attrs={"id": "tdata"})
        if not tbody:
            raise ValueError("未找到开奖号码数据表格 (id=tdata)")
        trs = tbody.find_all("tr")
    else:
        table = soup.find("table", id="tablelist")
        if not table:
            raise ValueError("未找到开奖号码数据表格 (id=tablelist)")
        trs = table.find_all("tr")

    for tr in trs:
        tds = tr.find_all("td")
        if not tds:
            continue
        issue = tds[0].get_text(strip=True)
        if not issue or issue == "期号":
            continue
        # 过滤非数字期号的行（表头/注数/金额等）
        if not issue.isdigit():
            continue
        record = {"期数": issue}
        if config.code == "ssq":
            for idx in range(config.red.sequence_len):
                record[f"红球_{idx + 1}"] = tds[idx + 1].get_text(strip=True)
            record["蓝球_1"] = tds[7].get_text(strip=True)
            # 开奖日期在最后一列 (td[15])
            if len(tds) > 15:
                record["开奖日期"] = tds[15].get_text(strip=True)
        elif config.code == "dlt":
            for idx in range(config.red.sequence_len):
                record[f"红球_{idx + 1}"] = tds[idx + 1].get_text(strip=True)
            for idx in range(config.blue.sequence_len):
                record[f"蓝球_{idx + 1}"] = tds[6 + idx].get_text(strip=True)
            if len(tds) > 14:
                record["开奖日期"] = tds[14].get_text(strip=True)
        elif config.code in {"pls", "sd", "qxc"}:
            digits = tds[1].get_text(strip=True).split(" ")
            for idx, value in enumerate(digits):
                record[f"红球_{idx + 1}"] = int(float(value))
        elif config.code == "qlc":
            for idx in range(config.red.sequence_len):
                record[f"红球_{idx + 1}"] = tds[idx + 1].get_text(strip=True)
        rows.append(record)

    if not rows:
        raise ValueError("解析开奖号码失败，未获取到任何数据")
    df = pd.DataFrame(rows)
    df.sort_values("期数", ascending=False, inplace=True)
    return df.reset_index(drop=True)


def get_current_issue(code: str, client: Optional[LotteryHttpClient] = None) -> str:
    """获取指定彩票的最新期号。"""

    cfg = LOTTERY_CONFIGS[code]
    client = client or LotteryHttpClient(
        timeout=NETWORK_CONFIG["timeout"],
        retries=NETWORK_CONFIG["retry_count"],
        backoff_factor=NETWORK_CONFIG.get("backoff_factor", 0.6),
        user_agent=NETWORK_CONFIG["user_agent"],
    )

    if cfg.code in {"qxc", "pls", "sd"}:
        url = f"https://datachart.500.com/{cfg.code}/history/inc/history.php"
    else:
        url = f"https://datachart.500.com/{cfg.code}/history/history.shtml"

    html = client.get_text(url)
    soup = BeautifulSoup(html, "lxml")
    value = soup.find("div", class_="wrap_datachart").find("input", {"id": "end"})["value"]
    logger.info("【{}】最新期号: {}", cfg.name, value)
    return value


def download_history(
    code: str,
    start: Optional[int] = None,
    end: Optional[int] = None,
    use_sequence_order: bool = False,
    client: Optional[LotteryHttpClient] = None,
) -> DownloadResult:
    """下载历史数据并保存到 data/<code>/data.csv。"""

    ensure_runtime_directories()
    cfg = LOTTERY_CONFIGS[code]
    client = client or LotteryHttpClient(
        timeout=NETWORK_CONFIG["timeout"],
        retries=NETWORK_CONFIG["retry_count"],
        backoff_factor=NETWORK_CONFIG.get("backoff_factor", 0.6),
        user_agent=NETWORK_CONFIG["user_agent"],
    )

    # P2-01: kl8 已迁移，直接抛出明确错误
    if code == "kl8":
        raise NotImplementedError(
            "KL8（快乐8）相关功能已于 2025-10 迁移至独立项目。"
            "请访问 https://github.com/KittenCN/kl8-lottery-analyzer"
        )
    elif use_sequence_order:
        raise NotImplementedError(
            f"{cfg.name} 不支持顺序模式（use_sequence_order），"
            "该功能仅适用于已迁移的 KL8 项目"
        )

    url = _build_history_url(cfg, start, end)
    logger.info("下载【{}】历史数据: {}", cfg.name, url)
    html = client.get_text(url)
    df = _parse_issue_list(cfg, html)

    save_dir = PATHS["data"] / cfg.code
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / DATA_FILE_NAME
    df.to_csv(output_path, index=False, encoding="utf-8")
    meta = DownloadResult(
        code=cfg.code,
        total_issues=len(df),
        saved_path=str(output_path),
        timestamp=datetime.utcnow().isoformat(),
    )
    logger.success("数据下载完成，共 {} 期，保存至 {}", meta.total_issues, output_path)
    (output_path.parent / "download_meta.json").write_text(
        json.dumps(meta.__dict__, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta


def _repair_sd_data():
    """从 3d/ 原始数据重建 sd/ 数据文件（含试机号、开奖号码）。"""
    src = PATHS["data"] / "3d" / DATA_FILE_NAME
    dst = PATHS["data"] / "sd" / DATA_FILE_NAME
    if not src.exists():
        raise FileNotFoundError(f"原始 3D 数据不存在，无法重建: {src}")
    df = pd.read_csv(src, encoding="utf-8")
    out = pd.DataFrame()
    out["期数"] = df["issue"]
    out["红球_1"] = df["红球1"]
    out["红球_2"] = df["红球2"]
    out["红球_3"] = df["红球3"]
    try_code = df["tryCode"].fillna(-1).astype(int)
    out["试机号"] = try_code.replace(-1, "")
    out["开奖号码"] = df["frontWinningNum"]
    out["开奖日期"] = df["openTime"]
    out.to_csv(dst, index=False, encoding="utf-8-sig")
    logger.info("sd 数据已从 3d/ 原始数据重建: {} 期", len(out))
    return out


def load_history(code: str) -> pd.DataFrame:
    """加载本地已下载的历史数据。自动检测损坏并尝试重建。"""
    cfg = LOTTERY_CONFIGS[code]
    path = PATHS["data"] / cfg.code / DATA_FILE_NAME
    if not path.exists():
        raise FileNotFoundError(f"未找到 {cfg.name} 历史数据，请先执行下载: {path}")

    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(path, encoding="utf-8")

    if "期数" not in df.columns:
        # 文件存在但无期数列 = 损坏，尝试重建
        logger.warning("{} 数据损坏（缺失期数列），尝试自动重建...", cfg.name)
        return _repair_data(code, cfg, path)

    red_cols = [f"红球_{i+1}" for i in range(cfg.red.sequence_len)]
    missing_red = [c for c in red_cols if c not in df.columns]
    if missing_red:
        logger.warning("{} 数据缺少 {}，尝试自动重建...", cfg.name, missing_red)
        return _repair_data(code, cfg, path)

    return df


def _repair_data(code: str, cfg, path) -> pd.DataFrame:
    """通用数据重建：sd 用本地备份，其余从网络重新下载。"""
    if code == "sd":
        return _repair_sd_data()
    logger.info("尝试从 500.com 重新下载 {} 数据...", cfg.name)
    try:
        meta = download_history(code)
        df = pd.read_csv(path, encoding="utf-8")
        if "期数" in df.columns:
            logger.success("{} 数据已重新下载: {} 期", cfg.name, meta.total_issues)
            return df
    except Exception as e:
        logger.error("重新下载 {} 数据失败: {}", cfg.name, e)
    raise ValueError(f"{path} 数据损坏且自动修复失败，请手动执行数据更新")


__all__ = [
    "DownloadResult",
    "LotteryHttpClient",
    "download_history",
    "get_current_issue",
    "load_history",
]
