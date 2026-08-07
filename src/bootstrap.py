# -*- coding: utf-8 -*-
"""
应用启动引导模块（P3-02 新增）。

负责：
1. 读取 config.yaml 日志配置并应用到 loguru
2. 确保运行时目录存在
3. 初始化会话数据库
4. 检查数据充足性并同步增量数据

在 api.py lifespan() 中调用。
"""

from __future__ import annotations

__all__ = ["configure_logging", "bootstrap"]

import sys
from loguru import logger

# 各彩种最低数据量阈值（期数），低于此值视为数据不足需补充
MIN_DATA_THRESHOLD = {
    "ssq": 30,   # 双色球：至少 30 期
    "sd": 30,    # 福彩3D：至少 30 期
    "qlc": 30,   # 七乐彩：至少 30 期
    "dlt": 30,   # 大乐透：至少 30 期
    "pls": 30,   # 排列三：至少 30 期
    "qxc": 30,   # 七星彩：至少 30 期
}

# 启动时自动同步的彩种列表（排除已迁移的 kl8）
ACTIVE_LOTTERY_CODES = ["ssq", "sd", "qlc"]


def configure_logging() -> None:
    """
    根据 config.yaml 配置 loguru。

    解决原问题：config.yaml 定义了 logging 配置但代码未读取，
    导致日志级别、格式、输出文件等配置不生效。
    """
    try:
        from src.config import YAML_CONFIG
        cfg = YAML_CONFIG.get("logging", {})
    except Exception:
        cfg = {}

    level = cfg.get("level", "INFO")
    fmt = cfg.get(
        "format",
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level:<8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>",
    )
    log_file = cfg.get("file", "lottery_predict.log")

    # 防护：如果格式是 Python 标准 logging 格式（含 %），则使用 loguru 默认格式
    if "%" in fmt:
        logger.warning(
            "config.yaml 中的 logging.format 使用了 Python logging 语法（%s），"
            "loguru 不支持该语法，已自动切换为 loguru 默认格式。"
            "请将 format 改为 loguru 语法，如: {time} | {level} | {message}"
        )
        fmt = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level:<8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )

    # 移除默认 handler，避免重复输出
    logger.remove()

    # 控制台输出
    logger.add(sys.stderr, level=level, format=fmt, colorize=True)

    # 文件输出（按天轮转，保留 7 天）
    if log_file:
        logger.add(
            log_file,
            rotation="1 day",
            retention="7 days",
            encoding="utf-8",
            level=level,
            format=fmt,
        )

    # 拦截标准 logging 模块（如 uvicorn 的日志），统一走 loguru 格式
    import logging

    class InterceptHandler(logging.Handler):
        def emit(self, record):
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno
            logger.opt(depth=6, exception=record.exc_info).log(level, record.getMessage())

    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    logger.info("日志系统初始化完成: 级别={}, 输出文件={}", level, log_file)


def _check_data_sufficiency(code: str) -> tuple[int, bool]:
    """检查指定彩种的数据是否充足。

    Returns:
        (current_count, is_sufficient): 当前期数和是否充足
    """
    from src.config import PATHS, DATA_FILE_NAME
    from pathlib import Path

    data_path = Path(PATHS["data"]) / code / DATA_FILE_NAME
    if not data_path.exists():
        return 0, False

    try:
        import pandas as pd
        df = pd.read_csv(data_path, encoding="utf-8-sig", nrows=0)
        # 尝试读取实际行数
        df_full = pd.read_csv(data_path, encoding="utf-8-sig")
        count = len(df_full)
    except Exception:
        try:
            import pandas as pd
            df_full = pd.read_csv(data_path, encoding="utf-8")
            count = len(df_full)
        except Exception as e:
            logger.warning("读取 {} 数据失败: {}", code, e)
            return 0, False

    threshold = MIN_DATA_THRESHOLD.get(code, 30)
    return count, count >= threshold


def sync_startup_data() -> dict[str, dict]:
    """启动时检查数据充足性并同步增量数据。

    对每个活跃彩种：
    1. 检查 data.csv 是否存在且数据量充足
    2. 数据不足时调用 download_history(merge=True) 增量同步
    3. 网络异常不阻止服务启动，仅记录警告

    Returns:
        各彩种的同步结果摘要，如 {"ssq": {"count": 100, "synced": True, "new": 5}}
    """
    from src.config import LOTTERY_CONFIGS
    from src.data_fetcher import download_history

    results: dict[str, dict] = {}

    for code in ACTIVE_LOTTERY_CODES:
        cfg = LOTTERY_CONFIGS.get(code)
        if cfg is None:
            logger.warning("跳过未知彩种: {}", code)
            continue

        current_count, is_sufficient = _check_data_sufficiency(code)

        if is_sufficient:
            logger.info(
                "【{}】数据充足（{} 期，阈值 {}），尝试增量同步...",
                cfg.name, current_count, MIN_DATA_THRESHOLD.get(code, 30),
            )
        else:
            logger.warning(
                "【{}】数据不足（{} 期，阈值 {} 期），开始下载...",
                cfg.name, current_count, MIN_DATA_THRESHOLD.get(code, 30),
            )

        try:
            meta = download_history(code, merge=True)
            results[code] = {
                "count": meta.total_issues,
                "synced": True,
                "path": meta.saved_path,
            }
            logger.success(
                "【{}】数据同步完成，共 {} 期",
                cfg.name, meta.total_issues,
            )
        except Exception as e:
            logger.error(
                "【{}】数据同步失败: {}，服务仍将正常启动",
                cfg.name, e,
            )
            results[code] = {
                "count": current_count,
                "synced": False,
                "error": str(e),
            }

    return results


def bootstrap() -> None:
    """
    完整的应用启动引导流程。

    调用顺序：
    1. configure_logging() — 初始化日志
    2. ensure_runtime_directories() — 创建必要目录
    3. _init_session_db() — 初始化会话数据库
    4. sync_startup_data() — 检查数据充足性并同步增量数据
    """
    configure_logging()

    from src.config import ensure_runtime_directories
    ensure_runtime_directories()

    from src.session import _get_conn
    _get_conn()

    # 启动时检查并同步数据
    sync_results = sync_startup_data()
    synced_count = sum(1 for v in sync_results.values() if v["synced"])
    failed_count = sum(1 for v in sync_results.values() if not v["synced"])
    logger.info(
        "数据同步摘要: 成功 {} / 失败 {} / 总计 {}",
        synced_count, failed_count, len(sync_results),
    )

    logger.success("应用启动引导完成")
