"""L1 层：提炼自 Penguin 的安全与微观基准逻辑（模式 B）。

该层只负责：
1) 基于当前目标生成保守的基准报价。
2) 对报价做安全裁剪，避免超量、超价、超资金。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from negmas.outcomes import Outcome


@dataclass
class DailyTarget:
    """日级目标（由 L2 产生），L1/L3 共用。"""

    target_quantity: int
    price_limit: float
    is_buying: bool
    executed_quantity: int = 0

    @property
    def remaining(self) -> int:
        return max(0, self.target_quantity - self.executed_quantity)

    def register_deal(self, quantity: int) -> None:
        self.executed_quantity += max(0, quantity)


class PenguinMicroBaseline:
    """微观基准与安全裁剪."""

    def __init__(self, safety_margin: float = 0.05):
        # 小幅安全边际，避免掏空现金或暴露过多库存
        self.safety_margin = safety_margin

    def baseline_offer(
        self,
        target: DailyTarget,
        delivery_time: int,
    ) -> Outcome:
        """给出保守基准报价."""
        qty = max(0, target.remaining)
        price = target.price_limit
        return qty, delivery_time, price

    def clip_offer(
        self,
        offer: Outcome,
        wallet: float,
        target: DailyTarget,
        is_buying: bool,
        inventory_capacity: Optional[int] = None,
    ) -> Outcome:
        """对报价做安全裁剪."""
        quantity, delivery_time, unit_price = offer

        # 不得超过目标剩余
        quantity = min(quantity, target.remaining)

        # 避免资金不足：缩减数量
        if is_buying and unit_price > 0:
            affordable = int(wallet / ((1.0 + self.safety_margin) * unit_price))
            if affordable >= 0:
                quantity = min(quantity, affordable)

        # 避免超库存（粗略按生产线容量限制）
        if inventory_capacity is not None:
            quantity = min(quantity, max(0, inventory_capacity))

        # 价格不越过目标底线
        if is_buying:
            unit_price = min(unit_price, target.price_limit)
        else:
            unit_price = max(unit_price, target.price_limit)

        # 保证非负
        quantity = max(0, int(quantity))
        return quantity, delivery_time, unit_price

