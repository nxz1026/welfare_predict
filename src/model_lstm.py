# -*- coding: utf-8 -*-
"""
MLP/DNN 基学习器（原 LSTM 重构）。

P1-02 修复说明：
原始实现将高维统计特征 reshape 为 (N, 1, D)，使 LSTM 退化为 Dense 层，
因为时间步=1 时 LSTM 无法学到任何时序模式。

重构方案：
- 输入已经是 feature_engineering 的高维统计特征（非时序原始序列）
- 此类场景下 MLP/DNN 比 LSTM 更合适
- 保留类名兼容性（LSTMPredictor），内部改为 MLP 架构

如果未来需要真正的时序建模，应在 preprocessing 阶段构建窗口序列特征，
而非在此处用 reshape 强行适配。

P3-04 修复说明：
TensorFlow 改为懒加载，避免在未安装 TF 的环境（如 Docker min 镜像）
中因顶层 import 导致 ImportError 崩溃。TF 仅在实际训练/预测时加载。
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from loguru import logger

from .config import LotteryModelConfig

# TensorFlow 懒加载：仅在需要时导入，避免未安装 TF 时整个模块无法导入
_tf = None


def _get_tf():
    """延迟导入 TensorFlow，未安装时抛出友好错误。"""
    global _tf
    if _tf is None:
        try:
            import tensorflow
            _tf = tensorflow
        except ImportError:
            raise ImportError(
                "TensorFlow 未安装。MLP/LSTM 模型需要 TensorFlow，"
                "请运行: pip install tensorflow==2.15.1 keras==2.15.0"
                "或使用 requirements.txt 安装完整依赖。"
            )
    return _tf


class LSTMPredictor:
    """
    MLP/DNN 基学习器（单模型多输出）。

    注意：类名因 API 兼容保留为 LSTMPredictor，但内部架构已从 LSTM 重构为 MLP。
    建议新代码使用 LSTMPredictor 时理解其为 MLP 实现。如需 LSTM，请使用 modeling.build_sequence_model()。
    """

    def __init__(
        self,
        config: LotteryModelConfig,
        hidden_units: list = [256, 128, 64],
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 50,
    ):
        self.config = config
        self.num_classes = config.red.num_classes  # 33
        self.sequence_len = config.red.sequence_len  # 6
        self.model = None  # 延迟创建，避免模块加载时依赖 TF
        self.is_trained = False

        self.hidden_units = hidden_units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

    def _build_model(self, input_shape: tuple):
        """构建 MLP 模型（替代原来的 LSTM）。"""
        tf = _get_tf()
        inputs = tf.keras.layers.Input(shape=input_shape, name="mlp_input")

        x = inputs
        for i, units in enumerate(self.hidden_units):
            x = tf.keras.layers.Dense(
                units,
                activation="relu",
                name=f"dense_{i}",
            )(x)
            x = tf.keras.layers.BatchNormalization(name=f"bn_{i}")(x)
            x = tf.keras.layers.Dropout(self.dropout, name=f"dropout_{i}")(x)

        x = tf.keras.layers.Dense(self.num_classes, activation="sigmoid", name="output")(x)

        model = tf.keras.Model(inputs=inputs, outputs=x, name="lottery_mlp")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        训练 MLP 模型。

        Args:
            X: (n_samples, n_features) 特征矩阵
            y: (n_samples, num_classes) 二值标签矩阵
        """
        tf = _get_tf()
        logger.info("训练 MLP 模型 (输入维度: {}, 输出维度: {})", X.shape[1], y.shape[1])

        # 直接使用特征矩阵，无需 reshape（P1-02 修复）
        self.model = self._build_model((X.shape[1],))
        self.model.summary(print_fn=logger.debug)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="loss",
                patience=8,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="loss",
                factor=0.5,
                patience=4,
                min_lr=1e-6,
            ),
        ]

        self.model.fit(
            X,
            y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=0,
        )

        self.is_trained = True
        logger.success("MLP 训练完成")

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

        # 直接输入，无需 reshape（P1-02 修复）
        return self.model.predict(X, verbose=0)

    def predict(self, X: np.ndarray, top_k: Optional[int] = None) -> np.ndarray:
        """
        预测选中的号码。

        Args:
            X: (n_samples, n_features) 特征矩阵
            top_k: 选前 k 个号码

        Returns:
            (n_samples, top_k) 预测号码（0-based）
        """
        if top_k is None:
            top_k = self.sequence_len

        proba = self.predict_proba(X)
        pred = np.zeros((X.shape[0], top_k), dtype=np.int32)
        for i in range(X.shape[0]):
            pred[i] = np.argsort(proba[i])[-top_k:][::-1]
        return pred

    def save_model(self, path: str) -> None:
        """保存模型"""
        self.model.save(path)
        logger.info("MLP 模型已保存: {}", path)

    def load_model(self, path: str) -> None:
        """加载模型"""
        tf = _get_tf()
        self.model = tf.keras.models.load_model(path)
        self.is_trained = True
        logger.info("MLP 模型已加载: {}", path)


__all__ = ["LSTMPredictor"]