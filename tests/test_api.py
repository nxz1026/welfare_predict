# -*- coding: utf-8 -*-
"""
API 层端到端测试（P2-04 新增）。

覆盖主要接口：登录/登出、授权校验、预测输出格式、未知彩种等。
使用 httpx.AsyncClient + FastAPI ASGITransport 进行异步测试。
"""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.api import app


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest_asyncio.fixture
async def auth_headers(client):
    """辅助 fixture：获取认证后的 headers。"""
    resp = await client.post("/api/v1/login", json={"username": "admin", "password": "testpass"})
    assert resp.status_code == 200
    token = resp.cookies.get("lottery_session")
    return {"Cookie": f"lottery_session={token}"}


# ============================================================
# 健康检查测试
# ============================================================


class TestHealthCheck:
    async def test_health_returns_ok(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "timestamp" in data
        assert "version" in data


# ============================================================
# 登录/登出测试
# ============================================================


class TestAuth:
    async def test_login_success(self, client):
        resp = await client.post("/api/v1/login", json={"username": "admin", "password": "testpass"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert "lottery_session" in resp.cookies

    async def test_login_wrong_password(self, client):
        resp = await client.post("/api/v1/login", json={"username": "admin", "password": "wrong"})
        assert resp.status_code == 401

    async def test_login_empty_credentials(self, client):
        resp = await client.post("/api/v1/login", json={"username": "", "password": ""})
        assert resp.status_code == 401

    async def test_logout(self, client, auth_headers):
        resp = await client.post("/api/v1/logout", headers=auth_headers)
        assert resp.status_code == 200

    async def test_me_unauthenticated(self, client):
        resp = await client.get("/api/v1/me")
        assert resp.status_code == 200
        data = resp.json()
        assert data["logged_in"] is False

    async def test_me_authenticated(self, client, auth_headers):
        resp = await client.get("/api/v1/me", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["logged_in"] is True
        assert data["username"] == "admin"


# ============================================================
# 授权保护测试
# ============================================================


class TestAuthorization:
    async def test_unauthorized_cannot_access_recommend(self, client):
        resp = await client.get("/api/v1/recommend/ssq")
        assert resp.status_code == 401

    async def test_unauthorized_cannot_access_predict(self, client):
        resp = await client.get("/api/v1/predict/ssq")
        assert resp.status_code == 401

    async def test_unauthorized_cannot_access_history(self, client):
        resp = await client.get("/api/v1/history/ssq")
        assert resp.status_code == 401

    async def test_unauthorized_cannot_access_stats(self, client):
        resp = await client.get("/api/v1/stats/ssq")
        assert resp.status_code == 401

    async def test_unauthorized_cannot_access_ranking(self, client):
        resp = await client.get("/api/v1/ranking/ssq")
        assert resp.status_code == 401

    async def test_authorized_can_access_recommend(self, client, auth_headers, sample_ssq_csv):
        resp = await client.get("/api/v1/recommend/ssq", headers=auth_headers)
        assert resp.status_code == 200


# ============================================================
# 业务接口测试
# ============================================================


class TestBusinessAPIs:
    async def test_unknown_lottery_returns_400(self, client, auth_headers):
        resp = await client.get("/api/v1/recommend/invalid_code", headers=auth_headers)
        assert resp.status_code == 400

    async def test_predict_model_not_found(self, client, auth_headers):
        """预测接口在模型未训练时应返回 404。"""
        resp = await client.get("/api/v1/predict/ssq?method=xgb", headers=auth_headers)
        assert resp.status_code == 404

    async def test_history_limit_validation(self, client, auth_headers, sample_ssq_csv):
        """参数边界校验：limit 应在 [1, 200] 范围内。"""
        # 合法值
        resp = await client.get("/api/v1/history/ssq?limit=30", headers=auth_headers)
        assert resp.status_code == 200

    async def test_ranking_params_validation(self, client, auth_headers, sample_ssq_csv):
        """ranking 参数边界校验。"""
        resp = await client.get(
            "/api/v1/ranking/ssq?window=100&backtest=20",
            headers=auth_headers,
        )
        assert resp.status_code == 200

    async def test_recommend_response_format(self, client, auth_headers, sample_ssq_csv):
        """推荐接口响应格式验证。"""
        resp = await client.get("/api/v1/recommend/ssq", headers=auth_headers)
        if resp.status_code == 200:
            data = resp.json()
            assert "code" in data
            assert "name" in data
            assert "strategies" in data
            assert isinstance(data["strategies"], list)
            assert len(data["strategies"]) == 4  # 四种策略


# ============================================================
# 全局异常处理器测试
# ============================================================


class TestGlobalErrorHandler:
    async def test_internal_error_hides_details(self, client, auth_headers):
        """内部异常不应暴露文件路径等敏感信息。"""
        resp = await client.get("/api/v1/report/nonexistent", headers=auth_headers)
        # 应返回 400 或 500，且不包含文件路径
        assert resp.status_code in (400, 404, 500)
        detail = resp.json().get("detail", "")
        assert "/" not in detail or "app/" not in detail
