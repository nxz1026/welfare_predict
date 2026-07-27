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