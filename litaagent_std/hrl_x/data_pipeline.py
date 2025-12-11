"""数据流水线：解析 SCML 日志，构建 macro/micro 数据占位集合。

说明：
- 为避免依赖真实日志格式变更，此处只实现最小骨架与接口，便于后续替换。
- 读取路径下的 negotiations.csv（递归），并可选读取 world_stats 等信息。
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional

import pandas as pd


@dataclass
class MacroSample:
    """日级样本：用于 L2 目标生成模型."""

    day: int
    state_macro: dict
    goal: dict  # {"buy_qty":..., "buy_price":..., "sell_qty":..., "sell_price":...}


@dataclass
class MicroSample:
    """轮级样本：用于 L3 残差模型."""

    negotiation_id: str
    history: pd.DataFrame  # 有序报价序列
    action: dict  # {"quantity":..., "time":..., "price":..., "is_accept": bool}
    baseline: dict  # L1 基准，可后续计算


def load_negotiation_csv(log_dir: str) -> pd.DataFrame:
    """加载所有 negotiations.csv 并合并."""
    csv_files = glob.glob(os.path.join(log_dir, "**", "negotiations.csv"), recursive=True)
    frames = []
    for f in csv_files:
        try:
            frames.append(pd.read_csv(f))
        except Exception as exc:  # pragma: no cover - 容错
            print(f"[WARN] 读取 {f} 失败: {exc}")
    if not frames:
        raise FileNotFoundError(f"未找到 negotiations.csv 于 {log_dir}")
    return pd.concat(frames, ignore_index=True)


def build_macro_dataset(df: pd.DataFrame) -> List[MacroSample]:
    """按天聚合成交记录，反推日级目标标签（简单示例）。"""
    samples: List[MacroSample] = []
    if "time" not in df.columns or "offer" not in df.columns:
        return samples
    # 只看达成的记录
    grouped = df[df["response"] == "accept"].groupby("time")
    for day, g in grouped:
        buy_deals = g[g["is_buying"] == True] if "is_buying" in g else g  # noqa: E712
        sell_deals = g[g["is_buying"] == False] if "is_buying" in g else g
        buy_qty = int(buy_deals["quantity"].sum()) if "quantity" in g else 0
        sell_qty = int(sell_deals["quantity"].sum()) if "quantity" in g else 0
        buy_price = float(buy_deals["price"].max()) if "price" in g and not buy_deals.empty else 0.0
        sell_price = float(sell_deals["price"].min()) if "price" in g and not sell_deals.empty else 0.0

        state_macro = {"day": int(day)}
        goal = {
            "buy_qty": buy_qty,
            "buy_price": buy_price,
            "sell_qty": sell_qty,
            "sell_price": sell_price,
        }
        samples.append(MacroSample(day=int(day), state_macro=state_macro, goal=goal))
    return samples


def build_micro_dataset(df: pd.DataFrame, max_rounds: Optional[int] = None) -> List[MicroSample]:
    """按 negotiation_id 组装轮级序列，占位实现."""
    samples: List[MicroSample] = []
    if "id" not in df.columns or "round" not in df.columns:
        return samples
    for nid, g in df.groupby("id"):
        g_sorted = g.sort_values("round")
        if max_rounds is not None:
            g_sorted = g_sorted.head(max_rounds)
        action = {}
        if "offer" in g_sorted.columns:
            # 假设最后一轮代理动作作为标签
            last = g_sorted.iloc[-1]
            action = {
                "quantity": last.get("quantity"),
                "time": last.get("offer_time") or last.get("time"),
                "price": last.get("price"),
                "is_accept": last.get("response") == "accept",
            }
        samples.append(MicroSample(negotiation_id=str(nid), history=g_sorted, action=action, baseline={}))
    return samples


__all__ = [
    "MacroSample",
    "MicroSample",
    "load_negotiation_csv",
    "build_macro_dataset",
    "build_micro_dataset",
]
