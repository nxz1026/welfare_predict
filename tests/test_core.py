# -*- coding: utf-8 -*-
"""
单元测试 — 核心模块。

覆盖：
- feature_engineering: 特征计算正确性
- modeling: XGBoost 训练/预测
- recommendation: 策略生成
- user_history: 用户记录管理
- analysis: 报告生成

ponytail: 只写关键路径测试，全覆盖留给 CI。
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_lottery_config, LotteryModelConfig, SequenceModelSpec
from src.feature_engineering import (
    compute_hot_cold_features,
    compute_skip_features,
    build_feature_matrix,
    build_binary_labels,
)
from src.modeling import XGBoostPredictor
from src.recommendation import RecommendationEngine
from src.user_history import UserHistory
from src.analysis import generate_frequency_report, generate_missing_report


@pytest.fixture
def ssq_config():
    return get_lottery_config("ssq")


@pytest.fixture
def sample_df(ssq_config):
    """生成模拟双色球数据"""
    np.random.seed(42)
    n = 100
    data = {
        "期数": [f"2024{i:03d}" for i in range(1, n + 1)],
    }
    for i in range(1, 7):
        data[f"红球_{i}"] = np.random.randint(1, 34, size=n)
    data["蓝球_1"] = np.random.randint(1, 17, size=n)
    return pd.DataFrame(data)


@pytest.fixture
def xgb_model(ssq_config, sample_df):
    """预训练的 XGBoost 模型"""
    features = build_feature_matrix(sample_df, ssq_config)
    labels = build_binary_labels(sample_df, ssq_config)
    model = XGBoostPredictor(ssq_config, n_estimators=10, max_depth=3)
    model.train(features.values, labels)
    return model


# ============================================================
# feature_engineering 测试
# ============================================================

class TestFeatureEngineering:
    def test_hot_cold_shape(self, sample_df, ssq_config):
        hot_cold = compute_hot_cold_features(sample_df, ssq_config)
        assert hot_cold.shape == (len(sample_df), 33)

    def test_skip_shape(self, sample_df, ssq_config):
        skip = compute_skip_features(sample_df, ssq_config)
        assert skip.shape == (len(sample_df), 33)

    def test_skip_values_non_negative(self, sample_df, ssq_config):
        skip = compute_skip_features(sample_df, ssq_config)
        assert (skip.values >= 0).all()

    def test_feature_matrix_shape(self, sample_df, ssq_config):
        features = build_feature_matrix(sample_df, ssq_config)
        # 33 hot + 33 skip + 33 interval + 2 (sum/span) + 3 (odd/even) + 3 (prime) + 1 (ac) + 2 (stat) = 110
        assert features.shape[0] == len(sample_df)
        assert features.shape[1] == 110

    def test_binary_labels_shape(self, sample_df, ssq_config):
        labels = build_binary_labels(sample_df, ssq_config)
        assert labels.shape == (len(sample_df), 33)

    def test_binary_labels_values(self, sample_df, ssq_config):
        labels = build_binary_labels(sample_df, ssq_config)
        assert set(np.unique(labels)).issubset({0, 1})


# ============================================================
# modeling 测试
# ============================================================

class TestXGBoostModel:
    def test_train(self, xgb_model):
        assert xgb_model.is_trained
        assert len(xgb_model.models) > 0

    def test_predict_proba_shape(self, xgb_model, sample_df, ssq_config):
        features = build_feature_matrix(sample_df, ssq_config)
        proba = xgb_model.predict_proba(features.values)
        assert proba.shape == (len(sample_df), 33)

    def test_predict_proba_range(self, xgb_model, sample_df, ssq_config):
        features = build_feature_matrix(sample_df, ssq_config)
        proba = xgb_model.predict_proba(features.values)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_predict_shape(self, xgb_model, sample_df, ssq_config):
        features = build_feature_matrix(sample_df, ssq_config)
        pred = xgb_model.predict(features.values, top_k=6)
        assert pred.shape == (len(sample_df), 6)


# ============================================================
# recommendation 测试
# ============================================================

class TestRecommendation:
    def test_generate(self, sample_df, ssq_config):
        engine = RecommendationEngine("ssq")
        rec = engine.generate(sample_df)
        assert len(rec.strategies) == 4

    def test_strategy_names(self, sample_df, ssq_config):
        engine = RecommendationEngine("ssq")
        rec = engine.generate(sample_df)
        names = [s.strategy_name for s in rec.strategies]
        assert "conservative" in names
        assert "aggressive" in names
        assert "balanced" in names
        assert "mystic" in names

    def test_red_balls_count(self, sample_df, ssq_config):
        engine = RecommendationEngine("ssq")
        rec = engine.generate(sample_df)
        for s in rec.strategies:
            assert len(s.red_balls) == 6

    def test_red_balls_in_range(self, sample_df, ssq_config):
        engine = RecommendationEngine("ssq")
        rec = engine.generate(sample_df)
        for s in rec.strategies:
            assert all(1 <= b <= 33 for b in s.red_balls)


# ============================================================
# user_history 测试
# ============================================================

class TestUserHistory:
    def test_create_user(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            user = UserHistory("test_user", storage_dir=tmpdir)
            assert user.user_id == "test_user"

    def test_record_purchase(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            user = UserHistory("test_user", storage_dir=tmpdir)
            user.record_purchase({"reds": [1, 2, 3, 4, 5, 6], "blue": 7})
            freq = user.get_frequent_numbers()
            assert 1 in freq["reds"]

    def test_lucky_numbers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            user = UserHistory("test_user", storage_dir=tmpdir)
            user.record_purchase({"reds": [6, 8, 16], "blue": 8})
            lucky = user.get_lucky_numbers()
            assert 6 in lucky

    def test_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            user = UserHistory("test_user", storage_dir=tmpdir)
            user.record_purchase({"reds": [1, 2, 3, 4, 5, 6]}, cost=2.0)
            stats = user.get_stats()
            assert stats["total_spent"] == 2.0


# ============================================================
# analysis 测试
# ============================================================

class TestAnalysis:
    def test_frequency_report(self, sample_df, ssq_config):
        report = generate_frequency_report(sample_df, ssq_config)
        assert "号码频率统计报告" in report
        assert "热号" in report or "冷号" in report

    def test_missing_report(self, sample_df, ssq_config):
        report = generate_missing_report(sample_df, ssq_config)
        assert "遗漏值分析报告" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
