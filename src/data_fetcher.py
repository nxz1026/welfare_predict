# -*- coding: utf-8 -*-
"""
数据抓取模块，负责从 500.com 拉取彩票历史数据并保存到本地。

特点：
1. 使用带重试的 requests.Session，满足网络安全要求；
2. 输出 Pandas DataFrame，供预处理与训练使用；
3. 支持多种彩种的数据下载（ssq/dlt/pls/qxc/sd/qlc/kl8）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
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

# 历史数据缓存：键为 (路径, 修改时间)，值为 DataFrame
_HISTORY_CACHE_MAX = 8
_history_cache: dict = {}


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
        # 编码容错：500.com 等中文网站可能使用 GB2312/GBK
        for encoding in ("utf-8", "gb2312", "gbk", "gb18030"):
            try:
                response.encoding = encoding
                text = response.text
                # 检测乱码：如果常见中文词可正常解码则认为编码正确
                if "期" in text or "红球" in text or "蓝球" in text or len(text) > 100:
                    return text
            except (UnicodeDecodeError, UnicodeError):
                continue
        # 兜底：使用 apparent_encoding
        response.encoding = response.apparent_encoding
        return response.text


def _build_history_url(config: LotteryModelConfig, start: Optional[int], end: Optional[int]) -> str:
    base = f"https://datachart.500.com/{config.code}/history/"

    # 快乐8: 使用趋势图页面（80列遗漏值格式），返回近 30 期
    if config.code == "kl8":
        return "https://datachart.500.com/kl8/"

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


def _parse_kl8_trend_chart(html: str) -> pd.DataFrame:
    """解析快乐8趋势图页面（80列遗漏值格式）。

    趋势图每行包含：期号 + 80列数值。
    value=1 表示该号码在本期出现，>1 为遗漏次数。
    提取所有 value=1 的位置（1-80）作为20个开奖号码。
    趋势图无日期列，日期从期号推导（快乐8每日开奖）。
    """
    soup = BeautifulSoup(html, "lxml")
    rows = []

    # 查找数据行：趋势图的 tbody#tdata 或包含期号数字的 tr
    tbody = soup.find("tbody", attrs={"id": "tdata"})
    if tbody:
        trs = tbody.find_all("tr")
    else:
        # 兜底：查找所有包含数字期号的 tr
        table = soup.find("table")
        trs = table.find_all("tr") if table else soup.find_all("tr")

    for tr in trs:
        tds = tr.find_all("td")
        if not tds:
            continue

        # 第一列是期号
        issue = tds[0].get_text(strip=True)
        if not issue or not issue.isdigit():
            continue

        # 后续80列对应号码1-80
        # 趋势图中 chartBall01 类标记当期开奖号码（恰好20个）
        # 文本值为遗漏次数：1=上期出现（非当期），>1=更早出现
        # 因此必须用 chartBall01 类判断，不能用 text=="1"
        drawn_numbers = []
        for i in range(1, min(len(tds), 81)):
            td = tds[i]
            td_classes = td.get("class", [])

            # chartBall01 类 = 当期开奖号码
            if "chartBall01" in td_classes:
                # 球号 = 位置索引（tds[1]=球号1, tds[80]=球号80）
                drawn_numbers.append(i)

        if len(drawn_numbers) != 20:
            logger.debug(
                "kl8 期 {} 提取到 {} 个号码（期望20），跳过", issue, len(drawn_numbers)
            )
            continue

        record = {"期数": issue}
        for idx, num in enumerate(drawn_numbers[:20]):
            record[f"红球_{idx + 1}"] = num

        rows.append(record)

    if not rows:
        raise ValueError("解析快乐8趋势图失败，未获取到任何有效数据（每期应含20个开奖号码）")

    # 日期推导：快乐8每日开奖，使用"最新期≈当前日期"作为参考点
    # 比从1月1日推算更准确（期号序号不等于当年第几天，存在约10天偏差）
    try:
        latest_issue = max(r["期数"] for r in rows)
        latest_year = int(latest_issue[:4])
        latest_seq = int(latest_issue[4:])
        # 最新期对应最近开奖日（当天21点前取昨天，21点后取今天）
        now = datetime.now()
        ref_date = date(now.year, now.month, now.day)
        if now.hour < 21:
            ref_date -= timedelta(days=1)
        for record in rows:
            issue_year = int(record["期数"][:4])
            issue_seq = int(record["期数"][4:])
            if issue_year == latest_year:
                diff = latest_seq - issue_seq
                record["开奖日期"] = (ref_date - timedelta(days=diff)).isoformat()
            else:
                # 跨年数据：用1月1日近似（偏差较大但可接受）
                base = date(issue_year, 1, 1)
                record["开奖日期"] = (base + timedelta(days=issue_seq - 1)).isoformat()
    except (ValueError, IndexError):
        for record in rows:
            record.setdefault("开奖日期", "")

    df = pd.DataFrame(rows)
    df.sort_values("期数", ascending=False, inplace=True)
    return df.reset_index(drop=True)


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
            # sd（福彩3D）额外提取开奖号码和日期
            if config.code == "sd":
                record["开奖号码"] = " ".join(digits)
                # 尝试从后续 td 提取试机号和开奖日期（表结构可能变化，安全兜底）
                for td in tds[2:]:
                    txt = td.get_text(strip=True)
                    if txt and txt.isdigit() and len(txt) <= 3 and "试机号" not in record:
                        record["试机号"] = txt
                    elif txt and ("-" in txt or "/" in txt) and len(txt) >= 8 and "开奖日期" not in record:
                        record["开奖日期"] = txt
        elif config.code == "kl8":
            # 快乐8使用趋势图格式，不应走到此处；由 _parse_kl8_trend_chart() 处理
            # 兜底：尝试从标准表格格式解析（如有其他数据源）
            numbers_td = tds[1]
            spans = numbers_td.find_all("span")
            if spans:
                nums = [s.get_text(strip=True) for s in spans if s.get_text(strip=True).isdigit()]
            else:
                nums = numbers_td.get_text(strip=True).split()
            for idx in range(min(len(nums), config.red.sequence_len)):
                record[f"红球_{idx + 1}"] = int(nums[idx])
            for td in tds[2:]:
                txt = td.get_text(strip=True)
                if txt and ("-" in txt or "/" in txt) and len(txt) >= 8:
                    record["开奖日期"] = txt
                    break
        elif config.code == "qlc":
            # QLC: td[0]=期号, td[1]=开奖号码(7红+1蓝span.cBlue), td[5]=开奖日期
            numbers_td = tds[1]
            blue_span = numbers_td.find("span", class_="cBlue")
            if blue_span:
                # 提取蓝球（特别号）
                record["蓝球_1"] = blue_span.get_text(strip=True)
                # 提取红球：取 span 之前的文本节点
                red_nums = []
                for child in numbers_td.children:
                    if child.name == "span":
                        break
                    if isinstance(child, str):
                        red_nums.extend(child.strip().split())
            else:
                # 兜底：所有数字，最后一个为蓝球
                all_nums = numbers_td.get_text(strip=True).split()
                red_nums = all_nums[:7]
                record["蓝球_1"] = all_nums[7] if len(all_nums) > 7 else ""
            for idx, num in enumerate(red_nums[: config.red.sequence_len]):
                record[f"红球_{idx + 1}"] = num
            # 开奖日期在 td[5]
            if len(tds) > 5:
                record["开奖日期"] = tds[5].get_text(strip=True)
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

    if cfg.code == "kl8":
        url = "https://datachart.500.com/kl8/"
    elif cfg.code in {"qxc", "pls", "sd"}:
        url = f"https://datachart.500.com/{cfg.code}/history/inc/history.php"
    else:
        url = f"https://datachart.500.com/{cfg.code}/history/history.shtml"

    html = client.get_text(url)
    soup = BeautifulSoup(html, "lxml")
    value = None
    # 快乐8趋势图页面使用 input#to 而非 input#end
    if cfg.code == "kl8":
        end_input = soup.find("input", {"id": "to"})
        if end_input and end_input.get("value"):
            value = end_input["value"]
        else:
            # 兜底：从表格第一行提取期号
            tbody = soup.find("tbody", attrs={"id": "tdata"})
            if tbody:
                first_tr = tbody.find("tr")
                if first_tr:
                    tds = first_tr.find_all("td")
                    if tds:
                        value = tds[0].get_text(strip=True)
            if not value:
                raise ValueError("无法从快乐8趋势图获取最新期号")
    else:
        value = soup.find("div", class_="wrap_datachart").find("input", {"id": "end"})["value"]
    logger.info("【{}】最新期号: {}", cfg.name, value)
    return value


def download_history(
    code: str,
    start: Optional[int] = None,
    end: Optional[int] = None,
    use_sequence_order: bool = False,
    merge: bool = False,
    client: Optional[LotteryHttpClient] = None,
) -> DownloadResult:
    """下载历史数据并保存到 data/<code>/data.csv。

    Args:
        code: 彩种代码
        start: 起始期号（可选）
        end: 结束期号（可选）
        use_sequence_order: 是否使用顺序模式（已废弃）
        merge: 是否与已有数据合并去重。True 时先读取已有数据，
               下载后合并去重再保存，避免覆盖丢失历史数据。
        client: 可选的 HTTP 客户端
    """

    ensure_runtime_directories()
    cfg = LOTTERY_CONFIGS[code]
    client = client or LotteryHttpClient(
        timeout=NETWORK_CONFIG["timeout"],
        retries=NETWORK_CONFIG["retry_count"],
        backoff_factor=NETWORK_CONFIG.get("backoff_factor", 0.6),
        user_agent=NETWORK_CONFIG["user_agent"],
    )

    if use_sequence_order:
        raise NotImplementedError(
            f"{cfg.name} 不支持顺序模式（use_sequence_order），"
            "该功能仅适用于已迁移的 KL8 项目"
        )

    url = _build_history_url(cfg, start, end)
    logger.info("下载【{}】历史数据: {}", cfg.name, url)
    try:
        html = client.get_text(url)
    except requests.exceptions.HTTPError as e:
        if cfg.code == "kl8":
            raise ValueError(
                "快乐8 趋势图页面无法访问，请检查网络或手动将 CSV 数据文件"
                "放置到 data/kl8/data.csv，格式：期数,红球_1,...,红球_20,开奖日期"
            ) from e
        raise
    # 快乐8使用趋势图专用解析器
    if cfg.code == "kl8":
        fresh_df = _parse_kl8_trend_chart(html)
    else:
        fresh_df = _parse_issue_list(cfg, html)

    save_dir = PATHS["data"] / cfg.code
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / DATA_FILE_NAME

    # 增量合并模式：读取已有数据，合并去重后保存
    existing_count = 0
    new_count = 0
    if merge and output_path.exists():
        try:
            existing_df = pd.read_csv(output_path, encoding="utf-8-sig")
        except Exception:
            try:
                existing_df = pd.read_csv(output_path, encoding="utf-8")
            except Exception as e:
                logger.warning("读取已有数据失败，将覆盖保存: {}", e)
                existing_df = None

        if existing_df is not None and "期数" in existing_df.columns:
            # 确保 期数 列类型一致（str），避免合并后排序报错
            existing_df["期数"] = existing_df["期数"].astype(str)
            fresh_df["期数"] = fresh_df["期数"].astype(str)
            existing_issues = set(existing_df["期数"].tolist())
            fresh_issues = set(fresh_df["期数"].tolist())
            new_count = len(fresh_issues - existing_issues)
            existing_count = len(existing_issues)

            if new_count > 0:
                combined = pd.concat([existing_df, fresh_df], ignore_index=True)
                combined.drop_duplicates(subset=["期数"], keep="last", inplace=True)
                combined.sort_values("期数", ascending=False, inplace=True)
                fresh_df = combined.reset_index(drop=True)
                logger.info(
                    "增量合并: 已有 {} 期, 新增 {} 期, 合并后 {} 期",
                    existing_count, new_count, len(fresh_df),
                )
            else:
                # 无新数据，保留已有数据，不覆盖文件
                logger.info("增量合并: 无新数据（已有 {} 期），保留原有数据", existing_count)
                fresh_df = existing_df.reset_index(drop=True)
        elif existing_df is not None:
            # 已有数据无期数列，直接覆盖
            logger.warning("已有数据格式异常（无期数列），将覆盖保存")

    fresh_df.to_csv(output_path, index=False, encoding="utf-8")

    # sd（福彩3D）下载后自动从 3d/ 原始数据补充试机号
    if cfg.code == "sd":
        try:
            _repair_sd_data(merge=True)
        except Exception as e:
            logger.warning("sd 试机号合并失败（不影响主数据）: {}", e)

    meta = DownloadResult(
        code=cfg.code,
        total_issues=len(fresh_df),
        saved_path=str(output_path),
        timestamp=datetime.utcnow().isoformat(),
    )
    logger.success("数据下载完成，共 {} 期，保存至 {}", meta.total_issues, output_path)
    (output_path.parent / "download_meta.json").write_text(
        json.dumps(meta.__dict__, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta


def _repair_sd_data(merge: bool = False) -> pd.DataFrame:
    """从 3d/ 原始数据重建或合并 sd/ 数据文件（含试机号、开奖号码）。

    Args:
        merge: True 时合并模式 — 读取已有 sd/data.csv，用 3d/ 数据补充试机号；
               False 时重建模式 — 完全从 3d/ 数据重建 sd/data.csv。
    """
    src = PATHS["data"] / "3d" / DATA_FILE_NAME
    dst = PATHS["data"] / "sd" / DATA_FILE_NAME
    if not src.exists():
        raise FileNotFoundError(f"原始 3D 数据不存在，无法重建: {src}")

    # 读取 3d/ 原始数据，提取期号→试机号映射（排除 -1 无数据标记）
    df_3d = pd.read_csv(src, encoding="utf-8")
    trycode_map = (
        df_3d[["issue", "tryCode"]]
        .dropna(subset=["tryCode"])
        .assign(issue=lambda x: x["issue"].astype(str))
        .query("tryCode != -1")
        .set_index("issue")["tryCode"]
        .to_dict()
    )

    if merge and dst.exists():
        # 合并模式：保留已有 sd 数据，补充缺失的试机号
        df_sd = pd.read_csv(dst, encoding="utf-8-sig")
        if "期数" not in df_sd.columns:
            logger.warning("sd 数据无期数列，切换为重建模式")
            merge = False
        else:
            df_sd["期数"] = df_sd["期数"].astype(str)
            # 检测试机号缺失：NaN、-1 或空字符串均视为缺失
            if "试机号" in df_sd.columns:
                missing = df_sd["试机号"].isna() | (df_sd["试机号"] == -1) | (df_sd["试机号"].astype(str) == "")
            else:
                missing = pd.Series(True, index=df_sd.index)
            missing_count = missing.sum()
            if missing_count == 0:
                logger.info("sd 数据试机号已完整，无需合并")
                return df_sd
            filled = 0
            cleared = 0
            # 将试机号列转为 object 类型，避免 dtype 不兼容警告
            df_sd["试机号"] = df_sd["试机号"].astype(object)
            for idx, row in df_sd[missing].iterrows():
                issue = row["期数"]
                if issue in trycode_map:
                    df_sd.at[idx, "试机号"] = int(trycode_map[issue])
                    filled += 1
                else:
                    # 3d/ 数据中也无试机号，置空
                    df_sd.at[idx, "试机号"] = ""
                    cleared += 1
            df_sd.to_csv(dst, index=False, encoding="utf-8-sig")
            logger.info("sd 数据试机号合并完成: 补充 {} 条, 置空 {} 条（3d/ 无数据）/ 共 {} 条缺失", filled, cleared, missing_count)
            return df_sd

    # 重建模式：完全从 3d/ 数据重建
    out = pd.DataFrame()
    out["期数"] = df_3d["issue"].astype(str)
    out["红球_1"] = df_3d["红球1"]
    out["红球_2"] = df_3d["红球2"]
    out["红球_3"] = df_3d["红球3"]
    out["试机号"] = out["期数"].map(trycode_map).apply(lambda x: int(x) if pd.notna(x) else "")
    out["开奖号码"] = df_3d["frontWinningNum"]
    out["开奖日期"] = df_3d["openTime"]
    out.to_csv(dst, index=False, encoding="utf-8-sig")
    logger.info("sd 数据已从 3d/ 原始数据重建: {} 期", len(out))
    return out


def load_history(code: str, data_path: Optional[str] = None) -> pd.DataFrame:
    """加载本地已下载的历史数据。自动检测损坏并尝试重建。

    使用 LRU 缓存避免同一请求内重复读取 CSV 文件。

    Args:
        code: 彩种代码
        data_path: 可选的自定义数据文件路径
    """
    from functools import lru_cache

    # 缓存键：基于文件路径 + 修改时间，确保数据更新后缓存失效
    cfg = LOTTERY_CONFIGS[code]
    path = Path(data_path) if data_path else PATHS["data"] / cfg.code / DATA_FILE_NAME
    if not path.exists():
        raise FileNotFoundError(f"未找到 {cfg.name} 历史数据，请先执行下载: {path}")

    cache_key = (str(path), path.stat().st_mtime)

    if cache_key in _history_cache:
        return _history_cache[cache_key].copy()

    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        logger.debug("utf-8-sig 解码失败，尝试 utf-8: {}", path)
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

    # sd（福彩3D）额外检查：开奖日期/试机号缺失时从 3d/ 原始数据补充
    if code == "sd":
        need_repair = False
        if "开奖日期" not in df.columns or "试机号" not in df.columns:
            logger.warning("sd 数据缺少 开奖日期/试机号 列，从 3d/ 原始数据重建...")
            need_repair = True
        elif "试机号" in df.columns and (df["试机号"].isna() | (df["试机号"] == -1) | (df["试机号"] == "")).any():
            missing_count = (df["试机号"].isna() | (df["试机号"] == -1) | (df["试机号"] == "")).sum()
            logger.info("sd 数据有 {} 条试机号缺失，从 3d/ 原始数据补充...", missing_count)
            repaired = _repair_sd_data(merge=True)
            _history_cache[cache_key] = repaired
            return repaired
        if need_repair:
            return _repair_sd_data(merge=False)

    # 缓存结果（限制缓存大小避免内存泄漏）
    if len(_history_cache) > _HISTORY_CACHE_MAX:
        _history_cache.popitem(last=False)
    _history_cache[cache_key] = df

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
