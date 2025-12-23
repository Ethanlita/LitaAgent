"""批次级规划器（方案 B）。

目标：在 StdAgent 的逐线程回调条件下，尽可能用“批次统一规划 + 动态预留”近似同步决策，
从而减少顺序依赖，并避免把全局预算粗暴切块给每个线程。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from negmas.outcomes import Outcome

from .l1_safety import L1SafetyLayer


def plan_buy_offers_by_alpha(
    *,
    l1: L1SafetyLayer,
    buy_ids: List[str],
    actions: Dict[str, Tuple[float, float, int]],
    alphas: Dict[str, float],
    Q_safe: np.ndarray,
    B_free: float,
    current_step: int,
) -> Dict[str, Optional[Outcome]]:
    """按 α 从高到低依次裁剪每个买方线程的动作，并动态扣减剩余资源.

    说明：
    - 不做“预算切块”，而是按线程动作的实际需求逐个“预留/扣账”；
    - 用 thread_id 作为稳定的次序打破平局，保证确定性。
    """

    offers: Dict[str, Optional[Outcome]] = {nid: None for nid in buy_ids}

    B_remain = float(max(0.0, B_free))
    Q_remain = Q_safe.astype(np.float32).copy()

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
        if 0 <= delta_t < len(Q_remain):
            Q_remain[delta_t] = float(max(0.0, float(Q_remain[delta_t]) - float(qty)))

        offers[nid] = (int(qty), int(current_step + delta_t), float(price))

    return offers


__all__ = [
    "plan_buy_offers_by_alpha",
]

