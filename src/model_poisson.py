# -*- coding: utf-8 -*-
"""
泊松先验基学习器。

最简单的基线模型：基于历史频率统计每个号码被选中的概率。
不做任何学习，直接用历史出现频率作为先验概率。

ponytail: 这是一个 baseline 模型，不能单独使用。
作用：为 stacking 提供一个"无知"参考，防止其他模型过拟合。
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from loguru import logger

from .config import LotteryModelConfig


class PoissonPrior:
    """
    泊松先验（频率统计）。

    对于每个号码，概率 = 该号码历史出现次数 / 总期数。
    等价于用历史频率估计伯努利分布的参数 p。
    """

    def __init__(self, config: LotteryModelConfig):
        self.config = config
        self.num_classes = config.red.num_classes
        self.prior_probs: Optional[np.ndarray] = None  # (num_classes,)
        self.is_trained = False

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        计算先验概率。

        Args:
            X: (n_samples, n_features) 特征矩阵（不使用，仅保持接口一致）
            y: (n_samples, num_classes) 二值标签矩阵
        """
        # 每个号码的历史出现频率
        self.prior_probs = y.mean(axis=0)  # (num_classes,)
        self.is_trained = True

        logger.info("泊松先验训练完成")
        logger.debug("先验概率范围: {:.4f} - {:.4f}",
                     self.prior_probs.min(), self.prior_probs.max())

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        输出先验概率。

        Args:
            X: (n_samples, n_features) 特征矩阵

        Returns:
            (n_samples, num_classes) 概率矩阵（每行相同）
        """
        if not self.is_trained:
            raise RuntimeError("模型未训练")

        n_samples = X.shape[0]
        return np.tile(self.prior_probs, (n_samples, 1))

    def predict(self, X: np.ndarray, top_k: Optional[int] = None) -> np.ndarray:
        """
        基于先验概率选择号码。

        Args:
            X: (n_samples, n_features) 特征矩阵
            top_k: 选前 k 个号码

        Returns:
            (n_samples, top_k) 预测号码（0-based）
        """
        if top_k is None:
            top_k = self.config.red.sequence_len

        proba = self.predict_proba(X)
        pred = np.zeros((X.shape[0], top_k), dtype=np.int32)
        for i in range(X.shape[0]):
            pred[i] = np.argsort(proba[i])[-top_k:][::-1]
        return pred


__all__ = ["PoissonPrior"]
