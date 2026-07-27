# -*- coding: utf-8 -*-
"""
彩票特征工程模块。

从原始开奖数据中提取机器学习特征，用于预测下一期号码。

核心特征族：
1. 热冷号特征：历史出现频次（已优化为滑动窗口 Counter，O(n)）
2. 间隔特征：距离上次出现的期数
3. 和值/跨度特征：号码总和的极差
4. 奇偶/质合特征：号码组成结构（质合已向量化）
5. AC值：算术复杂度

Author: Hermes Agent
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from collections import Counter

from .config import LotteryModelConfig, get_lottery_config


# 质数表（1-33 范围内的质数）
PRIMES_33 = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31}
PRIMES_35 = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31}

# 常量提取（P4-03）
DEFAULT_MAX_SKIP = 100
DEFAULT_MAX_INTERVAL = 50
DEFAULT_HOT_WINDOW = 30


def _get_primes(max_num: int) -> set:
    """获取 1-max_num 范围内的质数集合"""
    if max_num <= 33:
        return {p for p in PRIMES_33 if p <= max_num}
    else:
        return {p for p in PRIMES_35 if p <= max_num}


def compute_hot_cold_features(
    df: pd.DataFrame,
    config: LotteryModelConfig,
    window: int = DEFAULT_HOT_WINDOW,
) -> pd.DataFrame:
    """
    计算热冷号特征。

    对于每期，统计最近 window 期内各号码出现的次数。
    输出特征：hot_count_1, hot_count_2, ..., hot_count_N (N=num_classes)

    P1-03 优化：使用滑动窗口 Counter 替代每期全量重算，
    复杂度从 O(n × window × num_classes) 降低到 O(n × sequence_len)。
    """
    red_balls = _extract_red_balls(df, config)
    num_classes = config.red.num_classes

    features = np.zeros((len(df), num_classes), dtype=np.float32)
    counter = Counter()

    for i in range(len(df)):
        # 移除滑出窗口的号码
        if i > window:
            old = red_balls[i - window - 1]
            for b in old:
                counter[b] -= 1
                if counter[b] <= 0:
                    del counter[b]
        # 加入当前期新号码
        for b in red_balls[i]:
            counter[b] += 1
        # 输出当前计数
        for num in range(1, num_classes + 1):
            features[i, num - 1] = counter.get(num, 0)

    cols = [f"hot_count_{i}" for i in range(1, num_classes + 1)]
    return pd.DataFrame(features, columns=cols)


def compute_skip_features(
    df: pd.DataFrame,
    config: LotteryModelConfig,
) -> pd.DataFrame:
    """
    计算 skip（遗漏）特征。

    对于每个号码，计算它距离上次出现的期数。
    如果从未出现过，设为 max_skip（默认 100）。
    """
    red_balls = _extract_red_balls(df, config)
    num_classes = config.red.num_classes
    max_skip = DEFAULT_MAX_SKIP

    features = np.full((len(df), num_classes), max_skip, dtype=np.float32)
    last_seen = {num: max_skip for num in range(1, num_classes + 1)}

    for i in range(len(df)):
        current_numbers = set(red_balls[i])
        for num in range(1, num_classes + 1):
            if num in current_numbers:
                features[i, num - 1] = 0
                last_seen[num] = 0
            else:
                features[i, num - 1] = last_seen[num]
        # 更新所有号码的 skip 计数
        for num in last_seen:
            if num not in current_numbers:
                last_seen[num] += 1

    cols = [f"skip_{i}" for i in range(1, num_classes + 1)]
    return pd.DataFrame(features, columns=cols)


def compute_interval_features(
    df: pd.DataFrame,
    config: LotteryModelConfig,
) -> pd.DataFrame:
    """
    计算间隔特征（连续两次出现的期数差）。

    对于每期，计算每个号码上一次出现与再上一次出现的期间隔。
    如果只出现一次或从未出现，设为 max_interval（默认 50）。
    """
    red_balls = _extract_red_balls(df, config)
    num_classes = config.red.num_classes
    max_interval = DEFAULT_MAX_INTERVAL

    features = np.full((len(df), num_classes), max_interval, dtype=np.float32)
    positions = {num: [] for num in range(1, num_classes + 1)}

    for i in range(len(df)):
        current_numbers = set(red_balls[i])
        for num in current_numbers:
            positions[num].append(i)
        for num in range(1, num_classes + 1):
            pos_list = positions[num]
            if len(pos_list) >= 2:
                features[i, num - 1] = pos_list[-1] - pos_list[-2]
            else:
                features[i, num - 1] = max_interval

    cols = [f"interval_{i}" for i in range(1, num_classes + 1)]
    return pd.DataFrame(features, columns=cols)


def compute_sum_features(df: pd.DataFrame, config: LotteryModelConfig) -> pd.DataFrame:
    """计算和值特征：6 个红球之和"""
    red_balls = _extract_red_balls(df, config)
    sums = red_balls.sum(axis=1).astype(np.float32)
    return pd.DataFrame({"red_sum": sums})


def compute_span_features(df: pd.DataFrame, config: LotteryModelConfig) -> pd.DataFrame:
    """计算跨度特征：最大红球 - 最小红球"""
    red_balls = _extract_red_balls(df, config)
    spans = (red_balls.max(axis=1) - red_balls.min(axis=1)).astype(np.float32)
    return pd.DataFrame({"red_span": spans})


def compute_odd_even_features(df: pd.DataFrame, config: LotteryModelConfig) -> pd.DataFrame:
    """
    计算奇偶特征。
    - odd_count: 奇数个数
    - even_count: 偶数个数
    - odd_ratio: 奇数占比
    """
    red_balls = _extract_red_balls(df, config)
    odd_count = (red_balls % 2 == 1).sum(axis=1).astype(np.float32)
    even_count = (red_balls % 2 == 0).sum(axis=1).astype(np.float32)
    odd_ratio = odd_count / (odd_count + even_count)

    return pd.DataFrame({
        "odd_count": odd_count,
        "even_count": even_count,
        "odd_ratio": odd_ratio,
    })


def compute_prime_composite_features(df: pd.DataFrame, config: LotteryModelConfig) -> pd.DataFrame:
    """
    计算质合特征。
    - prime_count: 质数个数
    - composite_count: 合数个数
    - prime_ratio: 质数占比

    注意：1 既不是质数也不是合数，这里算作合数。

    P3-01 优化：使用 NumPy 向量化替代 Python 循环。
    """
    red_balls = _extract_red_balls(df, config)
    primes = _get_primes(config.red.num_classes)
    prime_mask = np.isin(red_balls, np.array(list(primes)))
    prime_count = prime_mask.sum(axis=1).astype(np.float32)
    composite_count = config.red.sequence_len - prime_count
    prime_ratio = prime_count / config.red.sequence_len

    return pd.DataFrame({
        "prime_count": prime_count,
        "composite_count": composite_count,
        "prime_ratio": prime_ratio,
    })


def compute_ac_value(df: pd.DataFrame, config: LotteryModelConfig) -> pd.DataFrame:
    """
    计算 AC 值（Arithmetic Complexity）。

    AC 值 = 所有两两差值的个数 - (号码个数 - 1)
    反映号码的分散程度。AC 值越大，号码越分散。
    """
    red_balls = _extract_red_balls(df, config)
    ac_values = np.zeros(len(df), dtype=np.float32)

    for i in range(len(df)):
        balls = sorted(red_balls[i])
        diffs = set()
        for j in range(len(balls)):
            for k in range(j + 1, len(balls)):
                diffs.add(balls[k] - balls[j])
        ac_values[i] = len(diffs) - (len(balls) - 1)

    return pd.DataFrame({"ac_value": ac_values})


def compute_statistical_features(df: pd.DataFrame, config: LotteryModelConfig) -> pd.DataFrame:
    """
    计算历史统计特征（基于滑动窗口的均值/标准差）。
    """
    red_balls = _extract_red_balls(df, config)
    window = 10

    means = np.zeros(len(df), dtype=np.float32)
    stds = np.zeros(len(df), dtype=np.float32)

    for i in range(len(df)):
        start = max(0, i - window)
        history = red_balls[start:i].flatten()
        if len(history) > 0:
            means[i] = np.mean(history)
            stds[i] = np.std(history)
        else:
            means[i] = config.red.num_classes / 2
            stds[i] = 0

    return pd.DataFrame({
        "hist_mean": means,
        "hist_std": stds,
    })


def _extract_red_balls(df: pd.DataFrame, config: LotteryModelConfig) -> np.ndarray:
    """提取红球号码矩阵"""
    cols = [f"红球_{i+1}" for i in range(config.red.sequence_len)]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"数据缺少列: {missing}")
    return df[cols].values.astype(np.int32)


def build_feature_matrix(
    df: pd.DataFrame,
    config: LotteryModelConfig,
    hot_window: int = DEFAULT_HOT_WINDOW,
) -> pd.DataFrame:
    """
    构建完整的特征矩阵。

    合并所有特征，返回 DataFrame 用于训练。
    """
    features = pd.DataFrame(index=df.index)

    # 热冷号
    hot = compute_hot_cold_features(df, config, window=hot_window)
    features = pd.concat([features, hot], axis=1)

    # 间隔（skip）
    skip = compute_skip_features(df, config)
    features = pd.concat([features, skip], axis=1)

    # 间隔（interval）
    interval = compute_interval_features(df, config)
    features = pd.concat([features, interval], axis=1)

    # 和值/跨度
    sum_feat = compute_sum_features(df, config)
    span_feat = compute_span_features(df, config)
    features = pd.concat([features, sum_feat, span_feat], axis=1)

    # 奇偶/质合
    odd_even = compute_odd_even_features(df, config)
    prime_comp = compute_prime_composite_features(df, config)
    features = pd.concat([features, odd_even, prime_comp], axis=1)

    # AC 值
    ac = compute_ac_value(df, config)
    features = pd.concat([features, ac], axis=1)

    # 历史统计
    stat = compute_statistical_features(df, config)
    features = pd.concat([features, stat], axis=1)

    return features


def build_labels(df: pd.DataFrame, config: LotteryModelConfig) -> np.ndarray:
    """
    构建标签矩阵。

    对于多球位预测，返回 (n_samples, sequence_len) 的矩阵。
    号码从 1-based 转为 0-based（XGBoost 需要）。
    """
    cols = [f"红球_{i+1}" for i in range(config.red.sequence_len)]
    balls = df[cols].values.astype(np.int32)
    return balls - 1  # 1-based -> 0-based


def build_binary_labels(df: pd.DataFrame, config: LotteryModelConfig) -> np.ndarray:
    """
    构建二分类标签矩阵。

    返回 (n_samples, num_classes) 的二值矩阵。
    y[i][j] = 1 表示第 j+1 个号码在第 i 期被选中。
    """
    cols = [f"红球_{i+1}" for i in range(config.red.sequence_len)]
    balls = df[cols].values.astype(np.int32) - 1  # 1-based -> 0-based

    n_samples = len(df)
    num_classes = config.red.num_classes
    binary_labels = np.zeros((n_samples, num_classes), dtype=np.int32)

    for i in range(n_samples):
        for ball in balls[i]:
            if 0 <= ball < num_classes:
                binary_labels[i, ball] = 1

    return binary_labels


__all__ = [
    "compute_hot_cold_features",
    "compute_skip_features",
    "compute_interval_features",
    "compute_sum_features",
    "compute_span_features",
    "compute_odd_even_features",
    "compute_prime_composite_features",
    "compute_ac_value",
    "compute_statistical_features",
    "build_feature_matrix",
    "build_labels",
    "build_binary_labels",
]
