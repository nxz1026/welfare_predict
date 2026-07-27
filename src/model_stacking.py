# -*- coding: utf-8 -*-
"""
Stacking 元学习器（已弃用，由 unified_pipeline.py 内的 StackingEnsemble 替代）。

此模块保留以兼容旧测试/导入，新代码请使用 UnifiedPipeline(method='stacking')。
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional

from sklearn.linear_model import LogisticRegression
from loguru import logger

from .config import LotteryModelConfig


class StackingMetaLearner:
    """
    Stacking 元学习器。
    """

    def __init__(
        self,
        config: LotteryModelConfig,
        C: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42,
    ):
        self.config = config
        self.num_classes = config.red.num_classes
        # 每个号码一个 LR 模型
        self.models: Dict[int, LogisticRegression] = {}
        self.is_trained = False
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state

    def _build_meta_features(
        self,
        probas: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        拼接基学习器的概率输出作为元特征。

        Args:
            probas: {"xgboost": (n, 33), "lstm": (n, 33), "poisson": (n, 33)}

        Returns:
            (n_samples, n_models * num_classes) 元特征矩阵
        """
        meta_features = []
        for model_name, proba in probas.items():
            meta_features.append(proba)
        return np.hstack(meta_features)

    def train(
        self,
        probas: Dict[str, np.ndarray],
        y: np.ndarray,
    ) -> None:
        """
        训练元学习器。

        Args:
            probas: 基学习器的输出概率字典
            y: (n_samples, num_classes) 二值标签
        """
        meta_X = self._build_meta_features(probas)
        logger.info("训练 Stacking 元学习器: 元特征维度 {}", meta_X.shape[1])

        for num_idx in range(self.num_classes):
            y_num = y[:, num_idx]
            pos_rate = y_num.mean()

            if pos_rate < 0.01 or pos_rate > 0.99:
                continue

            model = LogisticRegression(
                C=self.C,
                max_iter=self.max_iter,
                random_state=self.random_state,
                solver="lbfgs",
            )
            model.fit(meta_X, y_num)
            self.models[num_idx] = model

        self.is_trained = True
        logger.success("Stacking 训练完成: {} 个元学习器".format(len(self.models)))

    def predict_proba(
        self,
        probas: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        预测最终概率。

        Args:
            probas: 基学习器的输出概率字典

        Returns:
            (n_samples, num_classes) 概率矩阵
        """
        if not self.is_trained:
            raise RuntimeError("模型未训练")

        meta_X = self._build_meta_features(probas)
        n_samples = meta_X.shape[0]
        final_proba = np.zeros((n_samples, self.num_classes), dtype=np.float32)

        for num_idx, model in self.models.items():
            final_proba[:, num_idx] = model.predict_proba(meta_X)[:, 1]

        return final_proba

    def predict(self, probas: Dict[str, np.ndarray], top_k: Optional[int] = None) -> np.ndarray:
        """
        预测最终号码。

        Args:
            probas: 基学习器的输出概率字典
            top_k: 选前 k 个号码

        Returns:
            (n_samples, top_k) 预测号码（0-based）
        """
        if top_k is None:
            top_k = self.config.red.sequence_len

        proba = self.predict_proba(probas)
        pred = np.zeros((proba.shape[0], top_k), dtype=np.int32)
        for i in range(proba.shape[0]):
            pred[i] = np.argsort(proba[i])[-top_k:][::-1]
        return pred


__all__ = ["StackingMetaLearner"]
