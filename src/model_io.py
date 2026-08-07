# -*- coding: utf-8 -*-
"""
统一模型序列化模块（P1-04 新增）。

解决原项目中 XGBoost 用 joblib、LSTM 用 TF native、
unified_pipeline 用 pickle 的混乱状态。

提供统一的 save/load 接口，根据模型类型自动选择最佳序列化方式：
- Keras/TF 模型: model.save() / tf.keras.models.load_model()
- sklearn/xgboost/custom: joblib.dump() / joblib.load()

同时保存元数据 (.meta.json)，记录模型类型、类名等信息。
"""

from __future__ import annotations

import json
import joblib
from pathlib import Path
from typing import Any, Optional, Dict


class ModelIO:
    """统一模型序列化接口。"""

    @staticmethod
    def save(model: Any, path: Path, metadata: Optional[Dict] = None) -> None:
        """
        保存模型到指定路径。

        Args:
            model: 要保存的模型对象
            path: 保存路径
            metadata: 额外元数据（如 code, method, feature_names 等）
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # 根据模型类型选择保存方式
        model_type = type(model).__name__

        # StackingEnsemble 特殊处理：保存为目录
        if model_type == "StackingEnsemble":
            ensemble_dir = path.with_suffix("") if path.suffix else Path(str(path) + "_ensemble")
            ensemble_dir.mkdir(parents=True, exist_ok=True)
            model.save_model(str(ensemble_dir))
            meta = {**(metadata or {}), 'type': 'stacking_ensemble', 'model_class': model_type}
            meta_path = path.with_suffix('.meta.json')
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
            return

        has_save_method = hasattr(model, 'save') and callable(getattr(model, 'save'))

        if has_save_method:
            # Keras/TF 模型
            model.save(str(path))
            meta = {**(metadata or {}), 'type': 'keras', 'model_class': model_type}
        else:
            # sklearn/xgboost/custom 对象
            joblib.dump(model, path)
            meta = {**(metadata or {}), 'type': 'joblib', 'model_class': model_type}

        # 保存元数据
        if metadata is not None:
            meta_path = path.with_suffix('.meta.json')
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    @staticmethod
    def load(path: Path) -> Any:
        """
        从指定路径加载模型。

        Args:
            path: 模型文件路径

        Returns:
            加载后的模型对象
        """
        path = Path(path)

        # 尝试读取元数据判断类型
        meta_path = path.with_suffix('.meta.json')
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                model_type = meta.get('type')
            except (json.JSONDecodeError, OSError):
                model_type = None
        else:
            model_type = None

        if model_type == 'stacking_ensemble':
            # 加载 StackingEnsemble
            ensemble_dir = path.with_suffix("") if path.suffix else Path(str(path) + "_ensemble")
            from .unified_pipeline import StackingEnsemble
            return StackingEnsemble.load_model(str(ensemble_dir))
        elif model_type == 'keras':
            import tensorflow as keras
            return keras.models.load_model(str(path))
        else:
            return joblib.load(path)

    @staticmethod
    def get_metadata(path: Path) -> Optional[Dict]:
        """
        读取模型的元数据。

        Args:
            path: 模型文件路径

        Returns:
            元数据字典，如果不存在则返回 None
        """
        meta_path = Path(path).with_suffix('.meta.json')
        if meta_path.exists():
            try:
                return json.loads(meta_path.read_text())
            except (json.JSONDecodeError, OSError):
                return None
        return None


__all__ = ["ModelIO"]
