# -*- coding: utf-8 -*-
"""
LSTM 基学习器。

使用 TensorFlow/Keras 构建多层 LSTM 模型。
采用二分类方案：33 个号码每个预测一个概率，最后取 top-6。

ponytail: 单模型多输出（非 33 个独立模型），共享 LSTM 特征提取层。
升级路径：如果 LSTM 表现不佳，可尝试 GRU 或减少层数。
"""

from __future__ import annotations

import numpy as np
from typing import Optional

import tensorflow as tf
from loguru import logger

from .config import LotteryModelConfig


class LSTMPredictor:
    """
    LSTM 基学习器（单模型多输出）。
    """

    def __init__(
        self,
        config: LotteryModelConfig,
        lstm_units: list = [64, 32],
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 50,
    ):
        self.config = config
        self.num_classes = config.red.num_classes  # 33
        self.sequence_len = config.red.sequence_len  # 6
        self.model: Optional[tf.keras.Model] = None
        self.is_trained = False

        self.lstm_units = lstm_units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

    def _build_model(self, input_shape: tuple) -> tf.keras.Model:
        """构建 LSTM 模型"""
        inputs = tf.keras.layers.Input(shape=input_shape, name="lstm_input")

        x = inputs
        for i, units in enumerate(self.lstm_units):
            return_sequences = i < len(self.lstm_units) - 1
            x = tf.keras.layers.LSTM(
                units,
                return_sequences=return_sequences,
                dropout=self.dropout,
                name=f"lstm_{i}",
            )(x)

        x = tf.keras.layers.Dropout(self.dropout, name="dropout")(x)
        x = tf.keras.layers.Dense(self.num_classes, activation="sigmoid", name="output")(x)

        model = tf.keras.Model(inputs=inputs, outputs=x, name="lottery_lstm")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        训练 LSTM 模型。

        Args:
            X: (n_samples, n_features) 特征矩阵
            y: (n_samples, num_classes) 二值标签矩阵
        """
        logger.info("训练 LSTM 模型")

        # 为 LSTM 增加时间维度：reshape 为 (n_samples, 1, n_features)
        # 这样 LSTM 可以学习特征间的时序模式
        X_lstm = X.reshape(X.shape[0], 1, X.shape[1])

        self.model = self._build_model((1, X.shape[1]))
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
            X_lstm,
            y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=0,
        )

        self.is_trained = True
        logger.success("LSTM 训练完成")

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

        X_lstm = X.reshape(X.shape[0], 1, X.shape[1])
        return self.model.predict(X_lstm, verbose=0)

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
        logger.info("LSTM 模型已保存: {}", path)

    def load_model(self, path: str) -> None:
        """加载模型"""
        self.model = tf.keras.models.load_model(path)
        self.is_trained = True
        logger.info("LSTM 模型已加载: {}", path)


__all__ = ["LSTMPredictor"]
