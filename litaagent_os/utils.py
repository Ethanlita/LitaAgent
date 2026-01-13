from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional


@dataclass(frozen=True)
class Offer:
    q: int
    t: int
    p: float


def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def safe_round_rel(relative_time: Optional[float], step: int, n_steps: int) -> float:
    if relative_time is None:
        denom = max(1, n_steps - 1)
        rel = step / denom
    else:
        rel = relative_time
    if rel < 0.0 or rel > 1.0:
        rel = clamp(rel, 0.0, 1.0)
    return rel


def round_bucket(round_rel: float, T: int) -> int:
    if T <= 1:
        return 0
    return min(T - 1, int(math.floor(round_rel * T)))


def p_bin(p: float, p_min: float, p_max: float) -> int:
    if p_max <= p_min:
        return 0
    mid = (p_min + p_max) / 2.0
    return 0 if p <= mid else 1


def p_norm(p: float, p_min: float, p_max: float) -> float:
    if p_max <= p_min:
        return 0.0
    return (p - p_min) / (p_max - p_min)


def q_bucket(q: int, q_max: int) -> int:
    if q_max <= 0:
        return 0
    return max(0, min(int(q), q_max))


def q_norm(q: int, q_max: int) -> float:
    if q_max <= 0:
        return 0.0
    return q / float(q_max)


def q_bucket_coarse(q: int) -> Optional[int]:
    if q <= 0:
        return None
    if q == 1:
        return 1
    if q == 2:
        return 2
    if 3 <= q <= 4:
        return 3
    if 5 <= q <= 7:
        return 4
    return 5


def price_marginal_gain(price: float, trading_price: Optional[float], is_seller: bool) -> float:
    """计算价格的边际收益（已弃用，保留向后兼容）。"""
    if trading_price is None or trading_price <= 0:
        return price if is_seller else -price
    denom = max(1e-6, trading_price)
    if is_seller:
        return (price - trading_price) / denom
    return (trading_price - price) / denom


def compute_true_profit(price: float, exo_price: float, is_seller: bool) -> float:
    """
    计算真实利润（相对于外生合同价格）。
    
    对于 OneShot:
    - 卖家 (Level 0): 外生采购原材料，谈判销售产品
      利润 = 销售价格 - 外生采购价格
    - 买家 (Level 1): 谈判采购原材料，外生销售产品
      利润 = 外生销售价格 - 采购价格
    """
    if is_seller:
        # 卖家: 利润 = 卖价 - 外生采购成本
        return price - exo_price
    else:
        # 买家: 利润 = 外生销售收入 - 买价
        return exo_price - price


def match_price(p1: float, p2: float, eps: float = 1e-6) -> bool:
    return abs(p1 - p2) <= eps
