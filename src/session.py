"""
会话管理模块 — 基于 SQLite 的持久化会话存储

替代原 api.py 中的单用户内存 session，支持：
- 多用户同时登录（不互相踢出）
- 服务重启后保持登录状态
- 自动过期清理
"""

import sqlite3
import time
import uuid
from pathlib import Path
from typing import Optional

# 会话数据库路径（与 users 目录同级）
SESSION_DB_PATH = Path("data/sessions.db")


def _get_conn() -> sqlite3.Connection:
    """获取数据库连接并确保表存在。"""
    SESSION_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(SESSION_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            token TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            created_at REAL NOT NULL,
            expires_at REAL NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_at)")
    return conn


def create_session(user_id: str, ttl_seconds: int = 86400 * 7) -> str:
    """创建新会话，返回 token。默认有效期 7 天。

    Args:
        user_id: 用户标识
        ttl_seconds: 过期时间（秒），默认 7 天

    Returns:
        会话 token（UUID 字符串）
    """
    token = uuid.uuid4().hex
    now = time.time()
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT INTO sessions (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (token, user_id, now, now + ttl_seconds),
        )
        conn.commit()
    finally:
        conn.close()
    return token


def validate_session(token: str) -> Optional[str]:
    """验证 token 是否有效，返回 user_id 或 None。

    Args:
        token: 会话 token

    Returns:
        user_id 如果有效，None 如果无效或已过期
    """
    if not token:
        return None
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT user_id, expires_at FROM sessions WHERE token = ?", (token,)
        ).fetchone()
        if row is None:
            return None
        if time.time() > row["expires_at"]:
            # 已过期，删除
            conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
            conn.commit()
            return None
        return row["user_id"]
    finally:
        conn.close()


def delete_session(token: str) -> None:
    """删除指定会话（登出）。"""
    conn = _get_conn()
    try:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
        conn.commit()
    finally:
        conn.close()


def cleanup_expired() -> int:
    """清理所有过期会话，返回删除数量。"""
    conn = _get_conn()
    try:
        cursor = conn.execute(
            "DELETE FROM sessions WHERE expires_at < ?", (time.time(),)
        )
        conn.commit()
        return cursor.rowcount
    finally:
        conn.close()


def get_active_count() -> int:
    """获取当前有效会话数（用于监控）。"""
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM sessions WHERE expires_at > ?", (time.time(),)
        ).fetchone()
        return row["cnt"] if row else 0
    finally:
        conn.close()
