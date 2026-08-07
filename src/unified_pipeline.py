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
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from .config import LotteryModelConfig, get_lottery_config, ensure_runtime_directories
from .feature_engineering import build_feature_matrix, build_binary_labels
from .modeling import XGBoostPredictor
from .model_lstm import LSTMPredictor
from .model_poisson import PoissonPrior
from .model_io import ModelIO


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
    trained_issues: List[str] = field(default_factory=list)


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
        method: str = "xgb",  # 默认改为 xgb，stacking 需要完整实现
        model_dir: Optional[str] = None,
    ):
        """
        Args:
            code: 彩票代码 (ssq/sd/qlc/...)
            method: 训练方法 (xgb/lstm/poisson/stacking)
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
            # P1-01 修复：完整实现 Stacking 流程
            self._train_stacking(X_train, y_train, X_val, y_val, **kwargs)
        else:
            raise ValueError(f"未知方法: {self.method}")

        # 4. 验证
        metrics = {}
        if hasattr(self.model, 'predict_proba'):
            val_pred = self.model.predict_proba(X_val)
            val_loss = np.mean((val_pred - y_val) ** 2)
            metrics["val_mse"] = float(val_loss)
            logger.info("验证集 MSE: {:.6f}", val_loss)

        self.is_trained = True

        # 记录训练用的期号
        trained_issues = df["期数"].astype(str).iloc[:split_idx].tolist() if "期数" in df.columns else []

        summary = UnifiedTrainingSummary(
            code=self.code,
            name=self.config.name,
            method=self.method,
            n_samples=X_train.shape[0],
            n_features=X_train.shape[1],
            features_names=self.feature_names,
            timestamp=datetime.utcnow().isoformat(),
            metrics=metrics,
            trained_issues=trained_issues,
        )

        # 保存
        self._save_model()
        self._save_summary(summary)

        logger.success("训练完成: {} ({} 期数据, {} 特征)",
                        self.config.name, X_train.shape[0], X_train.shape[1])
        return summary

    def _train_stacking(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray, **kwargs) -> StackingEnsemble:
        """P1-01 修复：完整的 Stacking 训练流程。

        流程：
        1. 训练三个基学习器（XGB、LSTM、Poisson）
        2. 用基学习器生成 meta-features（验证集上的概率输出）
        3. 用 meta-features 训练元学习器（逻辑回归）
        4. 组合为 StackingEnsemble 对象
        """
        logger.info("=== 开始 Stacking 训练 ===")

        # 1. 训练基学习器
        xgb_model = XGBoostPredictor(self.config, **kwargs)
        xgb_model.train(X_train, y_train)

        lstm_model = LSTMPredictor(self.config, **kwargs)
        lstm_model.train(X_train, y_train)

        poisson_model = PoissonPrior(self.config, **kwargs)
        poisson_model.train(X_train, y_train)

        # 2. 生成 meta-features（在验证集上预测概率）
        logger.info("生成 meta-features...")
        xgb_proba = xgb_model.predict_proba(X_val)
        lstm_proba = lstm_model.predict_proba(X_val)
        poisson_proba = poisson_model.predict_proba(X_val)

        # 拼接 meta-features: (n_val, num_classes * 3)
        meta_X = np.concatenate([xgb_proba, lstm_proba, poisson_proba], axis=1)

        # 3. 训练元学习器（MultiOutputClassifier 适配多标签二值矩阵）
        from sklearn.multioutput import MultiOutputClassifier
        from sklearn.linear_model import LogisticRegression as LR
        base_lr = LR(max_iter=1000, C=1.0)
        meta_learner = MultiOutputClassifier(base_lr)
        meta_learner.fit(meta_X, y_val)

        # 4. 封装为 StackingEnsemble
        self.model = StackingEnsemble(
            base_models={
                "xgb": xgb_model,
                "lstm": lstm_model,
                "poisson": poisson_model,
            },
            meta_learner=meta_learner,
            config=self.config,
        )

        logger.success("=== Stacking 训练完成 ===")

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
            if proba is not None:
                selected_idx = np.argsort(proba)[-top_k:][::-1]
            else:
                selected_idx = np.random.choice(
                    self.config.red.num_classes, size=top_k, replace=False
                )
        elif strategy == "probabilistic":
            selected_idx = self._probability_sample(proba, top_k)
        elif strategy == "hybrid":
            n_top = max(1, top_k - 1)
            top_idx = np.argsort(proba)[-n_top:][::-1]
            remaining = [i for i in range(len(proba)) if i not in top_idx]
            n_random = top_k - n_top
            random_idx = np.random.choice(remaining, size=n_random, replace=False)
            selected_idx = np.concatenate([top_idx, random_idx])
        else:
            raise ValueError(f"未知策略: {strategy}")

        # 转换为 base 号码
        offset = self.config.red.min_val
        red_balls = sorted([int(i + offset) for i in selected_idx])

        # 蓝球（如果有配置）
        blue_ball = None
        if self.config.blue:
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
        p = proba / proba.sum()
        selected = []
        for _ in range(k):
            idx = np.random.choice(len(p), p=p)
            selected.append(idx)
            p[idx] = 0
            if p.sum() > 0:
                p = p / p.sum()
        return np.array(selected)

    def _save_model(self) -> None:
        """使用 ModelIO 统一保存模型（P1-04 修复）。"""
        model_path = self.model_dir / "model.pkl"
        metadata = {
            "code": self.code,
            "method": self.method,
            "feature_names": self.feature_names,
            "timestamp": datetime.utcnow().isoformat(),
        }
        ModelIO.save(self.model, model_path, metadata=metadata)
        logger.info("模型已保存: {}", model_path)

    def _save_summary(self, summary: UnifiedTrainingSummary) -> None:
        """保存训练摘要"""
        summary_path = self.model_dir / "summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(summary), f, ensure_ascii=False, indent=2)
        logger.info("训练摘要已保存: {}", summary_path)

    def _load_model(self) -> None:
        """使用 ModelIO 统一加载模型（P1-04 修复）。"""
        model_path = self.model_dir / "model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"模型未训练: {model_path}")
        self.model = ModelIO.load(model_path)
        self.is_trained = True
        logger.info("模型已加载: {}", model_path)


# ============================================================
# Stacking Ensemble 类（P1-01 新增）
# ============================================================


class StackingEnsemble:
    """Stacking 集成模型：组合多个基学习器的预测结果。"""

    def __init__(self, base_models: Dict[str, Any], meta_learner: Any, config: LotteryModelConfig) -> None:
        self.base_models = base_models
        self.meta_learner = meta_learner
        self.config = config

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测每个号码被选中的概率。

        MultiOutputClassifier.predict_proba 返回 List[(n, 2)]，
        需拼接为 (n, num_classes) 矩阵，每列取正类概率。
        """
        # 收集所有基学习器的概率预测
        probas = []
        for name in ["xgb", "lstm", "poisson"]:
            if name in self.base_models:
                probas.append(self.base_models[name].predict_proba(X))

        # 拼接作为 meta-features
        meta_X = np.concatenate(probas, axis=1)

        # 元学习器预测 — 兼容 MultiOutputClassifier 和 OneVsRestClassifier
        raw = self.meta_learner.predict_proba(meta_X)
        if isinstance(raw, list):
            # MultiOutputClassifier 返回 List[(n, 2)]
            n_samples = raw[0].shape[0]
            result = np.zeros((n_samples, len(raw)), dtype=np.float32)
            for i, proba_2col in enumerate(raw):
                result[:, i] = proba_2col[:, 1]  # 取正类概率
            return result
        # OneVsRestClassifier 或其他返回 (n, num_classes)
        return raw

    def save_model(self, path: str) -> None:
        """保存 Stacking 模型（包含所有子模型）。

        Args:
            path: 目录路径，各子模型保存到该目录下
        """
        import joblib
        import os
        import json
        os.makedirs(path, exist_ok=True)

        # 分别保存各基学习器
        for name, model in self.base_models.items():
            sub_path = os.path.join(path, f"{name}.pkl")
            if hasattr(model, 'save_model') and callable(model.save_model):
                model.save_model(sub_path)
            else:
                joblib.dump(model, sub_path)

        # 保存元学习器
        joblib.dump(self.meta_learner, os.path.join(path, "meta_learner.pkl"))

        # 保存配置与结构信息
        info = {
            "base_models": list(self.base_models.keys()),
            "code": self.config.code,
        }
        with open(os.path.join(path, "ensemble_info.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False)

    @classmethod
    def load_model(cls, path: str) -> "StackingEnsemble":
        """从目录加载 Stacking 模型。

        Args:
            path: 保存目录路径

        Returns:
            加载后的 StackingEnsemble 实例
        """
        import joblib
        import json
        from .config import get_lottery_config

        # 读取结构信息
        info_path = os.path.join(path, "ensemble_info.json")
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)

        config = get_lottery_config(info["code"])

        # 加载基学习器
        base_models = {}
        for name in info["base_models"]:
            sub_path = os.path.join(path, f"{name}.pkl")
            if os.path.exists(sub_path):
                base_models[name] = joblib.load(sub_path)

        # 加载元学习器
        meta_learner = joblib.load(os.path.join(path, "meta_learner.pkl"))

        return cls(
            base_models=base_models,
            meta_learner=meta_learner,
            config=config,
        )


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
    "StackingEnsemble",
    "train_unified",
    "predict_unified",
]
