# -*- coding: utf-8 -*-
"""
XGBoost 基学习器（二分类方案）。

把"33 选 6"建模为 33 个独立的二分类问题：
每个号码一个模型，预测该号码在下一期被选中的概率。
最终取概率最高的 6 个号码作为预测结果。

这种方案比"6 个多分类"更合理：
1. 不需要标签连续（二分类天然支持）
2. 每个号码独立建模，可解释性好
3. 自然处理"无放回抽样"约束
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional

import xgboost as xgb
from loguru import logger

from .config import LotteryModelConfig


class XGBoostPredictor:
    """
    XGBoost 基学习器（33 个二分类模型）。
    """

    def __init__(
        self,
        config: LotteryModelConfig,
        n_estimators: int = 200,
        max_depth: int = 4,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 3,
        gamma: float = 0.1,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        scale_pos_weight: float = 3.0,  # 正样本（选中）权重，应对不平衡
        random_state: int = 42,
    ):
        self.config = config
        self.num_classes = config.red.num_classes  # 33
        self.sequence_len = config.red.sequence_len  # 6
        self.models: Dict[int, xgb.XGBClassifier] = {}
        self.is_trained = False

        self.xgb_params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "min_child_weight": min_child_weight,
            "gamma": gamma,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "scale_pos_weight": scale_pos_weight,
            "random_state": random_state,
            "eval_metric": "logloss",
            "use_label_encoder": False,
            "verbosity": 0,
        }

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        训练 XGBoost 模型。

        Args:
            X: (n_samples, n_features) 特征矩阵
            y: (n_samples, num_classes) 二值标签矩阵，y[i][j]=1 表示第 j 个号码被选中
        """
        logger.info("训练 XGBoost 模型: {} 个号码的二分类器", self.num_classes)

        for num_idx in range(self.num_classes):
            y_num = y[:, num_idx]
            pos_rate = y_num.mean()

            # 如果某个号码从未出现或每次都出现，跳过
            if pos_rate < 0.01 or pos_rate > 0.99:
                logger.debug("号码 {}: 正样本比例 {:.2%}，跳过", num_idx + 1, pos_rate)
                continue

            model = xgb.XGBClassifier(**self.xgb_params)
            model.fit(X, y_num)
            self.models[num_idx] = model

            logger.debug("号码 {}: 正样本比例 {:.2%}, 训练完成", num_idx + 1, pos_rate)

        self.is_trained = True
        logger.success("XGBoost 训练完成: {} 个模型".format(len(self.models)))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测每个号码被选中的概率。

        Args:
            X: (n_samples, n_features) 特征矩阵

        Returns:
            (n_samples, num_classes) 概率矩阵
        """
        if not self.is_trained:
            raise RuntimeError("模型未训练")

        n_samples = X.shape[0]
        proba = np.zeros((n_samples, self.num_classes), dtype=np.float32)

        for num_idx, model in self.models.items():
            # predict_proba 返回 (n_samples, 2)，取正类概率
            proba[:, num_idx] = model.predict_proba(X)[:, 1]

        return proba

    def predict(self, X: np.ndarray, top_k: Optional[int] = None) -> np.ndarray:
        """
        预测选中的号码。

        Args:
            X: (n_samples, n_features) 特征矩阵
            top_k: 选前 k 个号码，默认使用 config 的 sequence_len

        Returns:
            (n_samples, top_k) 预测号码（0-based），按概率降序排列
        """
        if top_k is None:
            top_k = self.sequence_len

        proba = self.predict_proba(X)

        # 对每个样本，取概率最高的 top_k 个号码
        pred = np.zeros((X.shape[0], top_k), dtype=np.int32)
        for i in range(X.shape[0]):
            pred[i] = np.argsort(proba[i])[-top_k:][::-1]

        return pred

    def get_feature_importance(self) -> Dict[int, np.ndarray]:
        """获取各号码的特征重要性"""
        importance = {}
        for num_idx, model in self.models.items():
            importance[num_idx] = model.feature_importances_
        return importance

    def save_model(self, path: str) -> None:
        """保存模型"""
        import joblib
        joblib.dump(self, path)
        logger.info("XGBoost 模型已保存: {}".format(path))

    @classmethod
    def load_model(cls, path: str) -> "XGBoostPredictor":
        """加载模型"""
        import joblib
        model = joblib.load(path)
        logger.info("XGBoost 模型已加载: {}".format(path))
        return model


__all__ = ["XGBoostPredictor"]
