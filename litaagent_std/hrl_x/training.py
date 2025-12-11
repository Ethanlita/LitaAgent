"""训练脚本骨架：L2/L3/L4 占位实现与接口。

说明：
- 仅提供流程与函数签名，实际模型可替换为 torch/tf 等实现。
- 默认使用简单的回归/监督占位，确保可运行与扩展。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from .data_pipeline import MacroSample, MicroSample


@dataclass
class TrainConfig:
    """训练配置占位."""

    lr: float = 1e-3
    epochs: int = 5
    batch_size: int = 32


class SimpleRegressor:
    """极简线性回归占位（作为 L2/L3 基线）。"""

    def __init__(self, input_dim: int, output_dim: int):
        rng = np.random.default_rng()
        self.W = rng.standard_normal((input_dim, output_dim)) * 0.01
        self.b = np.zeros(output_dim)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return x @ self.W + self.b

    def fit(self, x: np.ndarray, y: np.ndarray, lr: float, epochs: int):
        for _ in range(epochs):
            pred = self.predict(x)
            grad = (pred - y) / len(x)
            self.W -= lr * x.T @ grad
            self.b -= lr * grad.mean(axis=0)


def train_l2_bc(samples: List[MacroSample], cfg: TrainConfig) -> Any:
    """基于重构的宏观标签做简单回归（示例）。"""
    if not samples:
        raise ValueError("空的宏观样本，无法训练 L2")
    # 简化：仅使用 day 作为输入
    x = np.array([[s.day] for s in samples], dtype=np.float32)
    y = np.array(
        [
            [s.goal["buy_qty"], s.goal["buy_price"], s.goal["sell_qty"], s.goal["sell_price"]]
            for s in samples
        ],
        dtype=np.float32,
    )
    model = SimpleRegressor(input_dim=1, output_dim=4)
    model.fit(x, y, lr=cfg.lr, epochs=cfg.epochs)
    return model


def train_l3_bc(samples: List[MicroSample], cfg: TrainConfig) -> Any:
    """基于微观序列的残差监督占位（此处仅用长度特征）。"""
    if not samples:
        raise ValueError("空的微观样本，无法训练 L3")
    lengths = np.array([[len(s.history)] for s in samples], dtype=np.float32)
    # 标签：简单取 action price/qty 作为输出
    y = []
    for s in samples:
        price = s.action.get("price", 0.0) or 0.0
        qty = s.action.get("quantity", 0) or 0.0
        y.append([qty, price])
    y_arr = np.array(y, dtype=np.float32)
    model = SimpleRegressor(input_dim=1, output_dim=2)
    model.fit(lengths, y_arr, lr=cfg.lr, epochs=cfg.epochs)
    return model


def save_model(model: Any, path: str) -> None:
    """存储占位模型（numpy 保存参数）。"""
    np.savez(path, W=model.W, b=model.b)


def load_model(path: str) -> SimpleRegressor:
    """加载占位模型。"""
    data = np.load(path)
    m = SimpleRegressor(input_dim=data["W"].shape[0], output_dim=data["W"].shape[1])
    m.W = data["W"]
    m.b = data["b"]
    return m


__all__ = [
    "TrainConfig",
    "train_l2_bc",
    "train_l3_bc",
    "save_model",
    "load_model",
    "SimpleRegressor",
]
