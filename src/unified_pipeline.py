# -*- coding: utf-8 -*-
"""
统一管线 — 新一代 stacking ensemble 模型。

对比旧 pipeline.py（TF LSTM）：
- 特征：feature_engineering 手工特征（110维）
- 模型：XGBoost + LSTM + Poisson + Stacking
- 输出：概率推荐 + 策略选择

旧 pipeline.py 保留但标记 deprecated，新代码应使用本模块。
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from .config import LotteryModelConfig, get_lottery_config, ensure_runtime_directories
from .feature_engineering import build_feature_matrix, build_binary_labels
from .modeling import XGBoostPredictor
from .model_lstm import LSTMPredictor
from .model_poisson import PoissonPrior
from .model_stacking import StackingMetaLearner


@dataclass
class UnifiedTrainingSummary:
    """训练摘要"""
    code: str
    name: str
    method: str  # "stacking" / "xgb" / "lstm" / "poisson"
    n_samples: int
    n_features: int
    features_names: List[str]
    timestamp: str
    metrics: Dict[str, float]


@dataclass
class UnifiedPrediction:
    """预测结果"""
    code: str
    method: str
    red_balls: List[int]
    blue_ball: Optional[int]
    probabilities: Optional[np.ndarray] = None
    strategy_used: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "code": self.code,
            "method": self.method,
            "red_balls": self.red_balls,
            "blue_ball": self.blue_ball,
            "strategy": self.strategy_used,
        }


class UnifiedPipeline:
    """
    统一管线：特征工程 → 模型训练 → 概率预测 → 策略推荐。
    """

    def __init__(
        self,
        code: str,
        method: str = "stacking",
        model_dir: Optional[str] = None,
    ):
        """
        Args:
            code: 彩票代码 (ssq/sd/qlc/...)
            method: 训练方法 (stacking/xgb/lstm/poisson)
            model_dir: 模型保存目录
        """
        self.code = code
        self.config = get_lottery_config(code)
        self.method = method
        self.model_dir = Path(model_dir) if model_dir else Path("model") / code / method
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.feature_names: List[str] = []
        self.model = None
        self.is_trained = False

    def train(
        self,
        df: pd.DataFrame,
        validation_ratio: float = 0.15,
        **kwargs,
    ) -> UnifiedTrainingSummary:
        """
        训练模型。

        Args:
            df: 历史开奖数据
            validation_ratio: 验证集比例

        Returns:
            UnifiedTrainingSummary
        """
        logger.info("训练 {} 模型: 数据 {} 期, 方法={}", self.config.name, len(df), self.method)

        # 1. 构建特征
        features = build_feature_matrix(df, self.config, hot_window=30)
        labels = build_binary_labels(df, self.config)

        self.feature_names = list(features.columns)
        X = features.values.astype(np.float32)
        y = labels.astype(np.float32)

        # 2. 时间序列分割（不能用随机分割）
        n = len(X)
        split_idx = int(n * (1 - validation_ratio))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        logger.info("训练集: {} 样本, 验证集: {} 样本, 特征数: {}",
                     X_train.shape[0], X_val.shape[0], X_train.shape[1])

        # 3. 选择并训练模型
        if self.method == "xgb":
            self.model = XGBoostPredictor(self.config, **kwargs)
            self.model.train(X_train, y_train)
        elif self.method == "lstm":
            self.model = LSTMPredictor(self.config, **kwargs)
            self.model.train(X_train, y_train)
        elif self.method == "poisson":
            self.model = PoissonPrior(self.config, **kwargs)
            self.model.train(X_train, y_train)
        elif self.method == "stacking":
            self.model = StackingMetaLearner(self.config, **kwargs)
            # Stacking 需要先训练基学习器，这里简化：只用 XGBoost 训练
            # 完整 stacking 需要多轮训练，暂不支持
            logger.warning("Stacking 方法当前需要手动训练基学习器，fallback 到 XGBoost")
            self.model = XGBoostPredictor(self.config, **kwargs)
            self.model.train(X_train, y_train)
        else:
            raise ValueError(f"未知方法: {self.method}")

        # 4. 验证
        metrics = {}
        if hasattr(self.model, 'predict_proba'):
            val_pred = self.model.predict_proba(X_val)
            # 简单的平均概率误差
            val_loss = np.mean((val_pred - y_val) ** 2)
            metrics["val_mse"] = float(val_loss)
            logger.info("验证集 MSE: {:.6f}", val_loss)

        self.is_trained = True

        summary = UnifiedTrainingSummary(
            code=self.code,
            name=self.config.name,
            method=self.method,
            n_samples=X_train.shape[0],
            n_features=X_train.shape[1],
            features_names=self.feature_names,
            timestamp=datetime.utcnow().isoformat(),
            metrics=metrics,
        )

        # 保存
        self._save_model()
        self._save_summary(summary)

        logger.success("训练完成: {} ({} 期数据, {} 特征)",
                        self.config.name, X_train.shape[0], X_train.shape[1])
        return summary

    def predict(
        self,
        df: pd.DataFrame,
        top_k: Optional[int] = None,
        strategy: str = "probabilistic",
    ) -> UnifiedPrediction:
        """
        预测下一期号码。

        Args:
            df: 历史数据（用于特征计算）
            top_k: 选前 k 个号码
            strategy: 选号策略 (probabilistic/top_k/hybrid)

        Returns:
            UnifiedPrediction
        """
        if not self.is_trained:
            self._load_model()

        if top_k is None:
            top_k = self.config.red.sequence_len

        # 计算最新一期的特征
        features = build_feature_matrix(df, self.config, hot_window=30)
        X_latest = features.values[-1:].astype(np.float32)

        # 预测概率
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X_latest)[0]
        else:
            proba = None

        # 选号
        if strategy == "top_k" or proba is None:
            # 简单选概率最高的 top_k 个
            if proba is not None:
                selected_idx = np.argsort(proba)[-top_k:][::-1]
            else:
                # fallback 随机
                selected_idx = np.random.choice(
                    self.config.red.num_classes, size=top_k, replace=False
                )
        elif strategy == "probabilistic":
            # 概率采样
            selected_idx = self._probability_sample(proba, top_k)
        elif strategy == "hybrid":
            # 混合：top_k + 少量随机
            n_top = max(1, top_k - 1)
            top_idx = np.argsort(proba)[-n_top:][::-1]
            remaining = [i for i in range(len(proba)) if i not in top_idx]
            n_random = top_k - n_top
            random_idx = np.random.choice(remaining, size=n_random, replace=False)
            selected_idx = np.concatenate([top_idx, random_idx])
        else:
            raise ValueError(f"未知策略: {strategy}")

        # 转换为 1-based 号码
        red_balls = sorted([int(i + 1) for i in selected_idx])

        # 蓝球（如果有配置）
        blue_ball = None
        if self.config.blue:
            if proba is not None:
                # 蓝球：独立处理或取中位数
                blue_ball = np.random.randint(1, self.config.blue.num_classes + 1)
            else:
                blue_ball = np.random.randint(1, self.config.blue.num_classes + 1)

        return UnifiedPrediction(
            code=self.code,
            method=self.method,
            red_balls=red_balls,
            blue_ball=blue_ball,
            probabilities=proba,
            strategy_used=strategy,
        )

    def _probability_sample(self, proba: np.ndarray, k: int) -> np.ndarray:
        """按概率采样 k 个不重复的号码"""
        # 归一化概率
        p = proba / proba.sum()
        selected = []
        for _ in range(k):
            idx = np.random.choice(len(p), p=p)
            selected.append(idx)
            # 置零已选号码
            p[idx] = 0
            if p.sum() > 0:
                p = p / p.sum()
        return np.array(selected)

    def _save_model(self):
        """保存模型"""
        model_path = self.model_dir / "model.pkl"
        if hasattr(self.model, 'save_model') and callable(getattr(self.model, 'save_model')):
            self.model.save_model(str(model_path))
        else:
            # 通用 pickle（PoissonPrior 等简单模型）
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
        logger.info("模型已保存: {}", model_path)

    def _save_summary(self, summary: UnifiedTrainingSummary):
        """保存训练摘要"""
        summary_path = self.model_dir / "summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(summary), f, ensure_ascii=False, indent=2)
        logger.info("训练摘要已保存: {}", summary_path)

    def _load_model(self):
        """加载模型"""
        model_path = self.model_dir / "model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"模型未训练: {model_path}")
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        self.is_trained = True
        logger.info("模型已加载: {}", model_path)


# ============================================================
# 便捷函数
# ============================================================

def train_unified(
    df: pd.DataFrame,
    code: str = "ssq",
    method: str = "xgb",
    **kwargs,
) -> UnifiedTrainingSummary:
    """便捷训练函数"""
    pipeline = UnifiedPipeline(code, method=method)
    return pipeline.train(df, **kwargs)


def predict_unified(
    df: pd.DataFrame,
    code: str = "ssq",
    method: str = "xgb",
    **kwargs,
) -> UnifiedPrediction:
    """便捷预测函数"""
    pipeline = UnifiedPipeline(code, method=method)
    return pipeline.predict(df, **kwargs)


__all__ = [
    "UnifiedPipeline",
    "UnifiedTrainingSummary",
    "UnifiedPrediction",
    "train_unified",
    "predict_unified",
]
