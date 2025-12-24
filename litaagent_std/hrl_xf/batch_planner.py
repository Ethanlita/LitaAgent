"""批次级规划器（方案 B）。

目标：在 StdAgent 的逐线程回调条件下，尽可能用"批次统一规划 + 动态预留"近似同步决策，
从而减少顺序依赖，并避免把全局预算粗暴切块给每个线程。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from negmas.outcomes import Outcome

from .l1_safety import L1SafetyLayer, recompute_q_safe_after_reservation


def plan_buy_offers_by_alpha(
    *,
    l1: L1SafetyLayer,
    buy_ids: List[str],
    actions: Dict[str, Tuple[float, float, int]],
    alphas: Dict[str, float],
    Q_safe: np.ndarray,
    B_free: float,
    current_step: int,
    raw_free: Optional[np.ndarray] = None,
) -> Dict[str, Optional[Outcome]]:
    """按 α 从高到低依次裁剪每个买方线程的动作，并动态扣减剩余资源.

    说明：
    - 不做"预算切块"，而是按线程动作的实际需求逐个"预留/扣账"；
    - 用 thread_id 作为稳定的次序打破平局，保证确定性。
    - 修复：正确维护 Q_safe 的逆向累积最小值不变式。
    
    Args:
        l1: L1SafetyLayer 实例
        buy_ids: 买方线程 ID 列表
        actions: 各线程的原始动作 {nid: (q, p, delta_t)}
        alphas: 各线程的优先级权重 {nid: alpha}
        Q_safe: 初始安全买入量向量，shape (H+1,)
        B_free: 初始可用资金
        current_step: 当前仿真步数
        raw_free: 原始可用空间向量 (C_total - L)，用于正确重算 Q_safe
    """

    offers: Dict[str, Optional[Outcome]] = {nid: None for nid in buy_ids}

    B_remain = float(max(0.0, B_free))
    Q_remain = Q_safe.astype(np.float32).copy()
    
    # 如果提供了 raw_free，则使用正确的动态预留方式
    if raw_free is not None:
        raw_free_remain = raw_free.astype(np.float32).copy()
    else:
        raw_free_remain = None

    for nid in sorted(buy_ids, key=lambda x: (-float(alphas.get(x, 0.0)), x)):
        if nid not in actions:
            continue
        q_final, p_final, t_final = actions[nid]
        qty, price, delta_t = l1.clip_action(
            action=(q_final, p_final, t_final),
            Q_safe=Q_remain,
            B_free=B_remain,
            is_buying=True,
        )
        if qty <= 0:
            offers[nid] = None
            continue

        B_remain -= float(qty) * float(price)
        
        # 正确的动态预留：对所有 k >= delta_t 扣减，然后重算 Q_safe
        if raw_free_remain is not None and qty > 0:
            Q_remain = recompute_q_safe_after_reservation(
                raw_free_remain, delta_t, float(qty)
            )
            # 同步更新 raw_free_remain
            for k in range(delta_t, len(raw_free_remain)):
                raw_free_remain[k] -= float(qty)
        else:
            # 回退到旧逻辑（向后兼容，但不够准确）
            if 0 <= delta_t < len(Q_remain):
                Q_remain[delta_t] = float(max(0.0, float(Q_remain[delta_t]) - float(qty)))

        offers[nid] = (int(qty), int(current_step + delta_t), float(price))

    return offers


def plan_sell_offers_by_alpha(
    *,
    l1: L1SafetyLayer,
    sell_ids: List[str],
    actions: Dict[str, Tuple[float, float, int]],
    alphas: Dict[str, float],
    Q_safe_sell: np.ndarray,
    current_step: int,
) -> Dict[str, Optional[Outcome]]:
    """按 α 从高到低依次裁剪每个卖方线程的动作，并动态扣减剩余可交付量.

    卖侧动态预留逻辑：
    - Q_safe_sell[δ] = 当前成品 + 累计产出 - 累计已承诺出库
    - 当在 δ 交货 q 单位时，需要对所有 k >= δ 扣减 q（因为这批货从 δ 开始被占用）
    
    Args:
        l1: L1SafetyLayer 实例
        sell_ids: 卖方线程 ID 列表
        actions: 各线程的原始动作 {nid: (q, p, delta_t)}
        alphas: 各线程的优先级权重 {nid: alpha}
        Q_safe_sell: 初始安全卖出量向量，shape (H+1,)
        current_step: 当前仿真步数
    
    Returns:
        offers: {nid: (qty, delivery_time, price) or None}
    """
    offers: Dict[str, Optional[Outcome]] = {nid: None for nid in sell_ids}
    
    Q_sell_remain = Q_safe_sell.astype(np.float32).copy()
    
    for nid in sorted(sell_ids, key=lambda x: (-float(alphas.get(x, 0.0)), x)):
        if nid not in actions:
            continue
        q_final, p_final, t_final = actions[nid]
        
        # 使用 clip_action 裁剪（卖侧）
        qty, price, delta_t = l1.clip_action(
            action=(q_final, p_final, t_final),
            Q_safe=np.zeros_like(Q_sell_remain),  # 买侧安全量对卖侧无关
            B_free=float('inf'),  # 卖出不消耗资金
            is_buying=False,
            Q_safe_sell=Q_sell_remain,
        )
        
        if qty <= 0:
            offers[nid] = None
            continue
        
        # 动态预留：对所有 k >= delta_t 扣减（因为交付的货物从 delta_t 开始被占用）
        for k in range(delta_t, len(Q_sell_remain)):
            Q_sell_remain[k] -= float(qty)
        Q_sell_remain = np.maximum(Q_sell_remain, 0)
        
        offers[nid] = (int(qty), int(current_step + delta_t), float(price))
    
    return offers


__all__ = [
    "plan_buy_offers_by_alpha",
    "plan_sell_offers_by_alpha",
]

