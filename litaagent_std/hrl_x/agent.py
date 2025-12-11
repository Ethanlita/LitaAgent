"""LitaAgent-HRL（模式 B 骨架版）。

L1：Penguin 微观安全/基准（不含宏观决策）。
L2：简易启发式日级目标（可后续替换为 RL）。
L3：残差位预留，当前以 0 残差运行。
"""

from __future__ import annotations

from typing import Dict, Optional

from negmas.sao import SAOResponse, SAOState, ResponseType
from negmas.outcomes import Outcome
from scml.std import StdAgent

from .l1_safety import DailyTarget, PenguinMicroBaseline


try:
    from scml_analyzer.auto_tracker import TrackerManager

    _TRACKER_AVAILABLE = True
except ImportError:  # pragma: no cover - 可选依赖
    _TRACKER_AVAILABLE = False
    TrackerManager = None


class LitaAgentHRL(StdAgent):
    """模式 B：L1 微观基准 + L2 日级目标 + L3 残差位."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.l1 = PenguinMicroBaseline()
        self.buy_target: Optional[DailyTarget] = None
        self.sell_target: Optional[DailyTarget] = None
        self.debug = False

    # ----------------- 生命周期 -----------------
    def init(self):
        super().init()
        if self.debug:
            self.log_info("LitaAgent-HRL init.")

    def before_step(self):
        super().before_step()
        self._ensure_targets()
        state_h = self._macro_obs()
        goals = self._heuristic_manager(state_h)
        self.buy_target = DailyTarget(
            target_quantity=int(goals["buy_qty"]),
            price_limit=goals["buy_price"],
            is_buying=True,
        )
        self.sell_target = DailyTarget(
            target_quantity=int(goals["sell_qty"]),
            price_limit=goals["sell_price"],
            is_buying=False,
        )
        if self.debug:
            self.log_info(
                f"Step {self.awi.current_step} targets "
                f"buy {self.buy_target.target_quantity} @{self.buy_target.price_limit:.2f}, "
                f"sell {self.sell_target.target_quantity} @{self.sell_target.price_limit:.2f}"
            )

    # ----------------- 低层谈判 -----------------
    def respond(self, negotiator_id: str, state: SAOState) -> SAOResponse:
        offer = state.current_offer
        if offer is None:
            return SAOResponse(ResponseType.REJECT_OFFER, None)

        is_buying = self._is_buying(negotiator_id)
        target = self.buy_target if is_buying else self.sell_target
        self._ensure_targets()
        target = target if target else DailyTarget(0, 0.0, is_buying=is_buying)

        qty, delivery, price = offer

        # 数量完成则结束谈判
        if target.remaining <= 0:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        # 价格检查
        price_ok = (price <= target.price_limit) if is_buying else (price >= target.price_limit)
        if price_ok and qty <= target.remaining:
            target.register_deal(qty)
            return SAOResponse(ResponseType.ACCEPT_OFFER, offer)

        # 生成反报价（L1 基准 + 残差=0）
        baseline = self.l1.baseline_offer(target, delivery_time=self.awi.current_step + 1)
        clipped = self.l1.clip_offer(
            baseline,
            wallet=self.awi.wallet,
            target=target,
            is_buying=is_buying,
            inventory_capacity=getattr(self.awi.profile, "n_lines", None),
        )
        # 如果剩余为 0，则结束
        if clipped[0] <= 0:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)
        return SAOResponse(ResponseType.REJECT_OFFER, clipped)

    def propose(self, negotiator_id: str, state: SAOState) -> Optional[Outcome]:
        is_buying = self._is_buying(negotiator_id)
        target = self.buy_target if is_buying else self.sell_target
        self._ensure_targets()
        target = target if target else DailyTarget(0, 0.0, is_buying=is_buying)

        if target.remaining <= 0:
            return None

        baseline = self.l1.baseline_offer(target, delivery_time=self.awi.current_step + 1)
        clipped = self.l1.clip_offer(
            baseline,
            wallet=self.awi.wallet,
            target=target,
            is_buying=is_buying,
            inventory_capacity=getattr(self.awi.profile, "n_lines", None),
        )
        if clipped[0] <= 0:
            return None
        return clipped

    # ----------------- 辅助逻辑 -----------------
    def _macro_obs(self) -> Dict[str, float]:
        in_p = self.awi.my_input_products
        out_p = self.awi.my_output_products

        # 处理列表/单值
        in_pid = in_p[0] if hasattr(in_p, "__len__") else in_p
        out_pid = out_p[0] if hasattr(out_p, "__len__") else out_p

        inventory = self.awi.current_inventory
        market_prices = getattr(self.awi, "trading_prices", {}) or {}

        return {
            "step_progress": self.awi.current_step / max(1, self.awi.n_steps),
            "balance": self.awi.wallet,
            "inventory_in": inventory.get(in_pid, 0),
            "inventory_out": inventory.get(out_pid, 0),
            "market_price_in": market_prices.get(in_pid, 10.0),
            "market_price_out": market_prices.get(out_pid, 20.0),
            "capacity": getattr(self.awi.profile, "n_lines", 1),
        }

    def _heuristic_manager(self, obs: Dict[str, float]) -> Dict[str, float]:
        """简易日级目标（可后续替换为 RL）。"""
        capacity = max(1, int(obs["capacity"]))
        need_buy = max(0, capacity - int(obs["inventory_in"]))
        sellable = max(0, int(obs["inventory_out"]))

        buy_price = obs["market_price_in"] * 1.05
        sell_price = obs["market_price_out"] * 0.95

        return {
            "buy_qty": need_buy,
            "buy_price": buy_price,
            "sell_qty": sellable,
            "sell_price": sell_price,
        }

    def _is_buying(self, negotiator_id: str) -> bool:
        negotiator = self._negotiators.get(negotiator_id)
        if negotiator and negotiator.ami:
            # seller 注解指向卖方 id
            return negotiator.ami.annotation.get("seller") != self.id
        # 默认买入
        return True

    def _ensure_targets(self) -> None:
        if self.buy_target is None:
            self.buy_target = DailyTarget(0, 0.0, is_buying=True)
        if self.sell_target is None:
            self.sell_target = DailyTarget(0, 0.0, is_buying=False)


class LitaAgentHRLTracked(LitaAgentHRL):
    """带 Tracker 的版本（并行模式可用）。"""

    _tracker_logger = None

    @property
    def tracker(self):
        if not _TRACKER_AVAILABLE:
            return None
        if self._tracker_logger is None and self.id:
            self._tracker_logger = TrackerManager.get_logger(self.id, "LitaAgentHRL")
        return self._tracker_logger

    def init(self):
        super().init()
        if self.tracker:
            self.tracker.custom("agent_initialized", n_steps=self.awi.n_steps)

    def before_step(self):
        super().before_step()
        if self.tracker:
            self.tracker.set_day(self.awi.current_step)
            self.tracker.inventory_state(
                raw=self.awi.current_inventory,
                product=self.awi.my_output_products,
                balance=self.awi.wallet,
            )

    def on_negotiation_success(self, contract, mechanism):
        super().on_negotiation_success(contract, mechanism)
        if self.tracker:
            self.tracker.contract_signed(
                id=str(contract.id),
                partner=contract.annotation.get("buyer") or contract.annotation.get("seller"),
                qty=contract.quantity,
                price=contract.unit_price,
                day=contract.time,
                is_seller=contract.annotation.get("seller") == self.id,
            )

