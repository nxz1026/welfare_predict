import os
import random
from typing import Iterator

import numpy as np
import pytest

from src import config as project_config


@pytest.fixture(autouse=True)
def isolate_paths(tmp_path) -> Iterator[None]:
    """將 data/model 隔離到臨時目錄，確保測試不互相影響。"""

    original_paths = project_config.PATHS.copy()

    for key in list(project_config.PATHS.keys()):
        new_dir = tmp_path / key
        new_dir.mkdir(parents=True, exist_ok=True)
        project_config.PATHS[key] = new_dir

    yield

    # 恢復原始路徑
    for key, val in original_paths.items():
        project_config.PATHS[key] = val


@pytest.fixture(autouse=True)
def set_random_seed() -> None:
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf  # type: ignore

        tf.random.set_seed(seed)
    except Exception:
        pass
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


@pytest.fixture
def sample_ssq_csv(isolate_paths) -> str:
    """在隔离的 data 目录下创建 ssq 测试用 CSV 文件。"""
    import pandas as pd
    data_dir = project_config.PATHS["data"] / "ssq"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "data.csv"

    records = []
    for idx in range(30):
        issue = f"2024{idx:03d}"
        base = np.arange(1, 7) + idx % 5
        reds = sorted((base % 33) + 1)
        blue = (idx % 16) + 1
        row = {"期数": issue}
        for i, v in enumerate(reds, 1):
            row[f"红球_{i}"] = v
        row["蓝球_1"] = blue
        records.append(row)

    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    return str(csv_path)