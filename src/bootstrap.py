# -*- coding: utf-8 -*-
"""
应用启动引导模块（P3-02 新增）。

负责：
1. 读取 config.yaml 日志配置并应用到 loguru
2. 确保运行时目录存在
3. 初始化会话数据库

在 api.py startup() 中调用。
"""

from __future__ import annotations

import sys
from loguru import logger


def configure_logging():
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

    # 移除默认 handler，避免重复输出
    logger.remove()

    # 控制台输出
    logger.add(sys.stderr, level=level, format=fmt)

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

    logger.info("日志系统初始化完成: 级别={}, 输出文件={}", level, log_file)


def bootstrap():
    """
    完整的应用启动引导流程。

    调用顺序：
    1. configure_logging() — 初始化日志
    2. ensure_runtime_directories() — 创建必要目录
    3. _init_session_db() — 初始化会话数据库
    """
    configure_logging()

    from src.config import ensure_runtime_directories
    ensure_runtime_directories()

    from src.session import _get_conn
    _get_conn()

    logger.success("应用启动引导完成")
