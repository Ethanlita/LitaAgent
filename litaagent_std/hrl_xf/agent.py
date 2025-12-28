"""LitaAgent-HRL (HRL-XF) 主代理类.

层级职责：
- L1: 安全护盾（Q_safe/time_mask/B_free）
- L2: 日级目标（16 维分桶目标）
- L3: 轮级决策（AOP 动作：ACCEPT/REJECT/END）
- L4: 全局监控 + α 优先级协调
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from negmas.sao import SAOResponse, SAOState, ResponseType
from negmas.outcomes import Outcome
from scml.std import StdAgent

from .l1_safety import L1SafetyLayer, L1Output
from .state_builder import StateBuilder, StateDict
from .l2_manager import L2StrategicManager, L2Output, BUCKET_RANGES
from .l3_executor import L3Actor, L3Input, L3Output, NegotiationRound, HeuristicL3Actor
from .l4_coordinator import L4Layer, GlobalBroadcast, ThreadState, delta_to_bucket

try:
    from scml_analyzer.auto_tracker import TrackerManager
    _TRACKER_AVAILABLE = True
except ImportError:
    _TRACKER_AVAILABLE = False
TrackerManager = None


@dataclass
class NegotiationContext:
    """单个谈判的上下文信息."""
    negotiation_id: str
    is_buying: bool
    partner_id: str
    history: List[NegotiationRound] = field(default_factory=list)
    l1_output: Optional[L1Output] = None
    l2_output: Optional[L2Output] = None
    l3_output: Optional[L3Output] = None
    last_relative_time: float = 0.0
    last_delta_t: Optional[int] = None
    last_step: int = 0
    last_offer: Optional[Tuple[int, float, int]] = None  # (q, p, delta_t)
    alpha: float = 0.0


class LitaAgentHRL(StdAgent):
    """HRL-XF 分层强化学习代理."""

    def __init__(
        self,
        *args,
        mode: str = "heuristic",
        horizon: int = 40,
        debug: bool = False,
        l2_model_path: Optional[str] = None,
        l3_model_path: Optional[str] = None,
        l4_model_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.mode = mode
        self.horizon = horizon
        self.debug = debug

        self.l1 = L1SafetyLayer(horizon=horizon)
        self.state_builder = StateBuilder(horizon=horizon)
        self.l2 = L2StrategicManager(mode=mode, horizon=horizon, model_path=l2_model_path)
        self.l3 = L3Actor(mode=mode, horizon=horizon, model_path=l3_model_path)
        self.l4 = L4Layer(
            horizon=horizon,
            use_neural_alpha=(mode == "neural"),
            model_path=l4_model_path,
        )
        self._fallback_l3 = HeuristicL3Actor(horizon=horizon)

        self._contexts: Dict[str, NegotiationContext] = {}
        self._current_state: Optional[StateDict] = None
        self._current_l2_output: Optional[L2Output] = None
        self._step_l1_buy: Optional[L1Output] = None
        self._step_l1_sell: Optional[L1Output] = None

    # ==================== 生命周期 ====================

    def init(self):
        super().init()
        if self.debug:
            self._log("LitaAgent-HRL (HRL-XF) initialized")

    def before_step(self):
        super().before_step()

        self._contexts.clear()

        self._current_state = self.state_builder.build(self.awi, is_buying=True)
        self._step_l1_buy = self.l1.compute(self.awi, is_buying=True)
        self._step_l1_sell = self.l1.compute(self.awi, is_buying=False)

        self._current_l2_output = self.l2.compute(
            self._current_state.x_static,
            self._current_state.X_temporal,
            is_buying=True,
            awi=self.awi,
        )

        self.l4.monitor.on_step_begin(
            self.awi,
            self._step_l1_buy,
            self._step_l1_sell,
            self._current_l2_output,
            self._current_state,
        )

        negotiators = getattr(self, "negotiators", {}) or {}
        for negotiator_id in negotiators.keys():
            ctx = self._get_or_create_context(negotiator_id)
            ctx.l1_output = self._step_l1_buy if ctx.is_buying else self._step_l1_sell
            ctx.l2_output = self._current_l2_output

        if self.debug and self._current_l2_output and self._step_l1_buy:
            self._log(
                f"Step {self.awi.current_step}: "
                f"L2 goal = {self._current_l2_output.goal_vector[:4]}... "
                f"B_free = {self._step_l1_buy.B_free:.2f}"
            )

    def step(self):
        super().step()
        if self.debug:
            n_buy = sum(1 for c in self._contexts.values() if c.is_buying)
            n_sell = len(self._contexts) - n_buy
            self._log(f"Step {self.awi.current_step} done: {n_buy} buy, {n_sell} sell")

    def on_negotiation_success(self, contract, mechanism) -> None:
        super().on_negotiation_success(contract, mechanism)
        try:
            agreement = getattr(contract, "agreement", None)
            if agreement is None:
                return

            if isinstance(agreement, dict):
                qty = float(agreement.get("quantity", 0) or 0)
                price = float(agreement.get("unit_price", agreement.get("price", 0)) or 0)
                day = int(agreement.get("time", agreement.get("day", 0)) or 0)
            else:
                qty = float(getattr(agreement, "quantity", 0) or 0)
                price = float(getattr(agreement, "unit_price", getattr(agreement, "price", 0)) or 0)
                day = int(getattr(agreement, "time", getattr(agreement, "day", 0)) or 0)

            annotation = getattr(contract, "annotation", {}) or {}
            product = annotation.get("product", None)
            if hasattr(product, "__len__") and not isinstance(product, (str, int)):
                product = product[0] if len(product) > 0 else None

            is_buy = product == self.awi.my_input_product
            if qty <= 0 or price <= 0:
                return

            self.l4.monitor.on_contract_signed(
                quantity=int(qty),
                unit_price=float(price),
                delivery_time=int(day),
                is_buying=bool(is_buy),
            )
        except Exception:
            return

    # ==================== 谈判接口 ====================

    def respond(self, negotiator_id: str, state: SAOState) -> SAOResponse:
        offer = state.current_offer
        if offer is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        ctx = self._get_or_create_context(negotiator_id)
        ctx.last_relative_time = float(getattr(state, "relative_time", 0.0) or 0.0)
        ctx.last_step = int(getattr(state, "step", 0) or 0)

        qty, delivery, price = offer
        delta_t = int(delivery - self.awi.current_step)
        delta_t = int(max(0, min(delta_t, self.horizon)))
        ctx.last_delta_t = delta_t
        ctx.last_offer = (int(qty), float(price), int(delta_t))
        ctx.history.append(NegotiationRound(
            quantity=qty,
            price=price,
            delta_t=delta_t,
            is_my_turn=False,
        ))

        broadcast, alpha_map = self._compute_global_control()
        ctx.alpha = float(alpha_map.get(negotiator_id, 0.0))
        l3_input = self._build_l3_input(ctx, broadcast, current_offer=ctx.last_offer)
        l3_output = self.l3.compute(l3_input)
        ctx.l3_output = l3_output

        if l3_output.action.action_type == "accept":
            if self._is_accept_feasible(offer, ctx, broadcast):
                return SAOResponse(ResponseType.ACCEPT_OFFER, offer)
            counter = self._fallback_counter_offer(ctx, broadcast)
            if counter is None:
                return SAOResponse(ResponseType.END_NEGOTIATION, None)
            return SAOResponse(ResponseType.REJECT_OFFER, counter)

        if l3_output.action.action_type == "end":
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        counter = self._resolve_counter_offer(ctx, l3_output.action.offer, negotiator_id, broadcast)
        if counter is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        self._record_my_offer(ctx, counter)
        return SAOResponse(ResponseType.REJECT_OFFER, counter)

    def propose(self, negotiator_id: str, state: SAOState) -> Optional[Outcome]:
        ctx = self._get_or_create_context(negotiator_id)
        ctx.last_relative_time = float(getattr(state, "relative_time", 0.0) or 0.0)
        ctx.last_step = int(getattr(state, "step", 0) or 0)

        current_offer = None
        if state.current_offer is not None:
            qty, delivery, price = state.current_offer
            delta_t = int(delivery - self.awi.current_step)
            delta_t = int(max(0, min(delta_t, self.horizon)))
            current_offer = (int(qty), float(price), int(delta_t))

        broadcast, alpha_map = self._compute_global_control()
        ctx.alpha = float(alpha_map.get(negotiator_id, 0.0))
        l3_input = self._build_l3_input(ctx, broadcast, current_offer=current_offer)
        l3_output = self.l3.compute(l3_input)
        ctx.l3_output = l3_output

        if l3_output.action.action_type == "end":
            return None

        offer = self._resolve_counter_offer(ctx, l3_output.action.offer, negotiator_id, broadcast)
        if offer is None:
            return None

        self._record_my_offer(ctx, offer)
        return offer

    # ==================== 核心逻辑 ====================

    def _compute_global_control(self) -> Tuple[GlobalBroadcast, Dict[str, float]]:
        active_ids = self._get_active_negotiator_ids()
        for nid in active_ids:
            ctx = self._get_or_create_context(nid)
            ctx.l1_output = self._step_l1_buy if ctx.is_buying else self._step_l1_sell
            ctx.l2_output = self._current_l2_output
            self._refresh_context_from_mechanism(ctx)

        n_buy = sum(1 for nid in active_ids if self._get_or_create_context(nid).is_buying)
        n_sell = max(0, len(active_ids) - n_buy)

        broadcast = self.l4.monitor.compute_broadcast(n_buy, n_sell)
        thread_states: List[ThreadState] = []
        for nid in active_ids:
            ctx = self._get_or_create_context(nid)
            ts = self._build_l4_thread_state(ctx, broadcast)
            if ts is not None:
                thread_states.append(ts)

        priorities = self.l4.compute_alphas(thread_states, broadcast)
        alpha_map = {p.thread_id: p.alpha for p in priorities}
        return broadcast, alpha_map

    def _build_l3_input(
        self,
        ctx: NegotiationContext,
        broadcast: GlobalBroadcast,
        current_offer: Optional[Tuple[int, float, int]] = None,
    ) -> L3Input:
        l1_output = ctx.l1_output
        if l1_output is None:
            time_mask = np.zeros((self.horizon + 1,), dtype=np.float32)
        else:
            time_mask = l1_output.time_mask

        if ctx.is_buying:
            Q_safe = broadcast.Q_safe_remaining
            B_free = float(broadcast.B_remaining)
        else:
            Q_safe = broadcast.Q_safe_sell_remaining
            B_free = 0.0

        min_price, max_price = self._get_price_bounds(ctx.negotiation_id)
        return L3Input(
            is_buying=ctx.is_buying,
            history=list(ctx.history),
            current_offer=current_offer,
            negotiation_step=ctx.last_step,
            relative_time=ctx.last_relative_time,
            partner_id=ctx.partner_id,
            global_broadcast=broadcast,
            alpha=float(ctx.alpha),
            time_mask=time_mask,
            Q_safe=Q_safe,
            B_free=B_free,
            min_price=float(min_price),
            max_price=float(max_price),
        )

    def _resolve_counter_offer(
        self,
        ctx: NegotiationContext,
        offer: Optional[Tuple[int, int, float]],
        negotiator_id: str,
        broadcast: GlobalBroadcast,
    ) -> Optional[Outcome]:
        if offer is None:
            return None

        qty, delivery_abs, price = offer
        delta_t = int(delivery_abs - int(self.awi.current_step))

        l1_output = ctx.l1_output
        if l1_output is None:
            return None

        min_price, max_price = self._get_price_bounds(negotiator_id)
        time_range = self._get_time_bounds(negotiator_id)
        delta_range = None
        if time_range is not None:
            min_abs, max_abs = time_range
            current_step = int(self.awi.current_step)
            min_delta = max(0, int(min_abs) - current_step)
            max_delta = max(0, int(max_abs) - current_step)
            delta_range = (min_delta, max_delta)
        if ctx.is_buying:
            q_final, p_final, dt_final = self.l1.clip_action(
                action=(float(qty), float(price), delta_t),
                Q_safe=broadcast.Q_safe_remaining,
                B_free=float(broadcast.B_remaining),
                is_buying=True,
                min_price=min_price,
                max_price=max_price,
                time_range=delta_range,
            )
        else:
            q_final, p_final, dt_final = self.l1.clip_action(
                action=(float(qty), float(price), delta_t),
                Q_safe=np.zeros_like(l1_output.Q_safe),
                B_free=float("inf"),
                is_buying=False,
                min_price=min_price,
                max_price=max_price,
                Q_safe_sell=broadcast.Q_safe_sell_remaining,
                time_range=delta_range,
            )

        if q_final <= 0:
            return None

        delivery_time = int(self.awi.current_step + dt_final)
        return (int(q_final), delivery_time, float(p_final))

    def _fallback_counter_offer(
        self,
        ctx: NegotiationContext,
        broadcast: GlobalBroadcast,
    ) -> Optional[Outcome]:
        l3_input = self._build_l3_input(ctx, broadcast, current_offer=ctx.last_offer)
        fallback = self._fallback_l3.compute(l3_input)
        if fallback.action.action_type != "reject":
            return None
        return self._resolve_counter_offer(ctx, fallback.action.offer, ctx.negotiation_id, broadcast)

    def _is_accept_feasible(
        self,
        offer: Outcome,
        ctx: NegotiationContext,
        broadcast: GlobalBroadcast,
    ) -> bool:
        qty, delivery, price = offer
        delta_t = int(delivery - int(self.awi.current_step))
        if delta_t < 0 or delta_t > self.horizon:
            return False
        time_range = self._get_time_bounds(ctx.negotiation_id)
        if time_range is not None:
            min_abs, max_abs = time_range
            if int(delivery) < int(min_abs) or int(delivery) > int(max_abs):
                return False

        l1_output = ctx.l1_output
        if l1_output is None:
            return False

        if ctx.is_buying:
            q_cap = float(broadcast.Q_safe_remaining[delta_t]) if delta_t < len(broadcast.Q_safe_remaining) else 0.0
            if qty > q_cap:
                return False
            if qty * price > float(broadcast.B_remaining):
                return False
        else:
            q_cap = float(broadcast.Q_safe_sell_remaining[delta_t]) if delta_t < len(broadcast.Q_safe_sell_remaining) else 0.0
            if qty > q_cap:
                return False

        return True

    def _build_l4_thread_state(self, ctx: NegotiationContext, broadcast: GlobalBroadcast) -> Optional[ThreadState]:
        l1_output = ctx.l1_output
        l2_output = ctx.l2_output
        if l1_output is None or l2_output is None:
            return None

        target_delta = self._select_target_delta(ctx, l1_output, l2_output)
        bucket = delta_to_bucket(target_delta)
        goal_gap = float(broadcast.goal_gap_buy[bucket] if ctx.is_buying else broadcast.goal_gap_sell[bucket])

        current_offer = ctx.last_offer
        if current_offer is None:
            last = self._get_last_opponent_offer(ctx)
            if last is not None:
                current_offer = (int(last.quantity), int(last.delta_t), float(last.price))
        else:
            try:
                q_val, p_val, d_val = current_offer
                current_offer = (int(q_val), int(d_val), float(p_val))
            except Exception:
                current_offer = None

        if ctx.is_buying:
            q_safe_at_t = float(
                broadcast.Q_safe_remaining[target_delta] if target_delta < len(broadcast.Q_safe_remaining) else 0.0
            )
            b_remaining = float(broadcast.B_remaining)
        else:
            q_safe_at_t = float(
                broadcast.Q_safe_sell_remaining[target_delta] if target_delta < len(broadcast.Q_safe_sell_remaining) else 0.0
            )
            b_remaining = 0.0

        return ThreadState(
            thread_id=ctx.negotiation_id,
            is_buying=ctx.is_buying,
            negotiation_step=int(ctx.last_step),
            relative_time=float(ctx.last_relative_time),
            current_offer=current_offer,
            history_len=len(ctx.history),
            target_bucket=int(bucket),
            goal_gap=goal_gap,
            Q_safe_at_t=q_safe_at_t,
            B_remaining=b_remaining,
        )

    def _select_target_delta(self, ctx: NegotiationContext, l1_output: L1Output, l2_output: L2Output) -> int:
        if ctx.last_delta_t is not None:
            return int(max(0, min(ctx.last_delta_t, self.horizon)))

        goals = l2_output.goal_vector.reshape(4, 4)
        q_idx = 0 if ctx.is_buying else 2
        bucket = int(np.argmax(goals[:, q_idx]))

        dmin, dmax = BUCKET_RANGES[bucket]
        dmin = max(0, dmin)
        dmax = min(self.horizon, dmax)

        if ctx.is_buying:
            q_safe = l1_output.Q_safe
        else:
            q_safe = l1_output.Q_safe_sell

        for d in range(dmin, dmax + 1):
            if l1_output.time_mask[d] != -np.inf and q_safe[d] > 0:
                return d
        for d in range(0, self.horizon + 1):
            if l1_output.time_mask[d] != -np.inf and q_safe[d] > 0:
                return d
        return 0

    def _record_my_offer(self, ctx: NegotiationContext, offer: Outcome) -> None:
        qty, delivery, price = offer
        delta_t = int(delivery - self.awi.current_step)
        delta_t = int(max(0, min(delta_t, self.horizon)))
        ctx.last_delta_t = delta_t
        ctx.last_offer = (int(qty), float(price), int(delta_t))
        ctx.history.append(NegotiationRound(
            quantity=qty,
            price=price,
            delta_t=delta_t,
            is_my_turn=True,
        ))

    def _get_last_opponent_offer(self, ctx: NegotiationContext) -> Optional[NegotiationRound]:
        for r in reversed(ctx.history):
            if not r.is_my_turn:
                return r
        if ctx.history:
            return ctx.history[-1]
        return None

    def _get_active_negotiator_ids(self) -> List[str]:
        active = getattr(self, "active_negotiators", None) or {}
        if len(active) > 0:
            return list(active.keys())
        negotiators = getattr(self, "negotiators", None) or {}
        if len(negotiators) > 0:
            return list(negotiators.keys())
        return list(self._contexts.keys())

    def _get_mechanism_state(self, negotiator_id: str):
        nmi = self.get_nmi(negotiator_id)
        if nmi is None:
            return None
        return getattr(nmi, "state", None)

    def _refresh_context_from_mechanism(self, ctx: NegotiationContext) -> None:
        st = self._get_mechanism_state(ctx.negotiation_id)
        if st is None:
            return

        rel_time = getattr(st, "relative_time", None)
        if rel_time is not None:
            ctx.last_relative_time = float(rel_time or 0.0)

        step_val = getattr(st, "step", None)
        if step_val is not None:
            ctx.last_step = int(step_val)

        offer = getattr(st, "current_offer", None)
        if offer is not None:
            try:
                qty, delivery, price = offer
                delta_t = int(delivery - self.awi.current_step)
                delta_t = int(max(0, min(delta_t, self.horizon)))
                ctx.last_delta_t = delta_t
                ctx.last_offer = (int(qty), float(price), int(delta_t))
            except Exception:
                pass

    def _get_or_create_context(self, negotiator_id: str) -> NegotiationContext:
        if negotiator_id not in self._contexts:
            is_buying = self._is_buying(negotiator_id)
            partner_id = self._get_partner_id(negotiator_id)
            self._contexts[negotiator_id] = NegotiationContext(
                negotiation_id=negotiator_id,
                is_buying=is_buying,
                partner_id=partner_id,
            )
        return self._contexts[negotiator_id]

    def _is_buying(self, negotiator_id: str) -> bool:
        nmi = self.get_nmi(negotiator_id)
        if nmi is None:
            return True
        annotation = getattr(nmi, "annotation", {}) or {}
        product = annotation.get("product")
        return product == self.awi.my_input_product

    def _get_partner_id(self, negotiator_id: str) -> str:
        nmi = self.get_nmi(negotiator_id)
        if nmi is None:
            return "unknown"
        annotation = getattr(nmi, "annotation", {}) or {}
        buyer_id = annotation.get("buyer")
        seller_id = annotation.get("seller")
        return seller_id if self._is_buying(negotiator_id) else buyer_id

    def _get_time_bounds(self, negotiator_id: str) -> Optional[Tuple[int, int]]:
        nmi = self.get_nmi(negotiator_id)
        if nmi is None:
            return None
        issues = getattr(nmi, "issues", None)
        if not issues or len(issues) < 2:
            return None
        try:
            issue = issues[1]
            return int(issue.min_value), int(issue.max_value)
        except Exception:
            return None

    def _get_price_bounds(self, negotiator_id: str) -> Tuple[float, float]:
        nmi = self.get_nmi(negotiator_id)
        if nmi is None:
            return 0.0, float("inf")
        issues = getattr(nmi, "issues", None)
        if issues is None:
            return 0.0, float("inf")
        try:
            issue = issues[2]
            return float(issue.min_value), float(issue.max_value)
        except Exception:
            return 0.0, float("inf")

    def _log(self, msg: str) -> None:
        agent_name = getattr(self, "name", "HRL")
        print(f"[{agent_name}] {msg}")


class LitaAgentHRLTracked(LitaAgentHRL):
    """带 Tracker 的 HRL-XF 代理."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tracker = None

    def init(self):
        super().init()
        if _TRACKER_AVAILABLE:
            try:
                self._tracker = TrackerManager.get_tracker(self)
                if self.debug:
                    self._log("Tracker initialized")
            except Exception as e:
                if self.debug:
                    self._log(f"Tracker init failed: {e}")

    def before_step(self):
        super().before_step()
        if self._tracker:
            self._tracker.record_state({
                "step": self.awi.current_step,
                "balance": self.awi.wallet,
                "inventory": dict(self.awi.current_inventory),
                "l2_goal": self._current_l2_output.goal_vector.tolist() if self._current_l2_output else None,
            })

    def respond(self, negotiator_id: str, state: SAOState) -> SAOResponse:
        response = super().respond(negotiator_id, state)
        if self._tracker:
            ctx = self._contexts.get(negotiator_id)
            action_op, offer = self._parse_response_action(response)
            fallback = state.current_offer if action_op == "ACCEPT" else None
            self._tracker.custom(
                "aop_action",
                negotiator_id=negotiator_id,
                mechanism_id=negotiator_id,
                partner=ctx.partner_id if ctx else "unknown",
                role="buyer" if ctx and ctx.is_buying else "seller",
                round=int(getattr(state, "step", 0) or 0),
                action_op=action_op,
                offer=self._pack_offer(offer, fallback),
            )
        return response

    def propose(self, negotiator_id: str, state: SAOState) -> Optional[Outcome]:
        offer = super().propose(negotiator_id, state)
        if self._tracker:
            if offer is not None:
                ctx = self._contexts.get(negotiator_id)
                self._tracker.custom(
                    "aop_action",
                    negotiator_id=negotiator_id,
                    mechanism_id=negotiator_id,
                    partner=ctx.partner_id if ctx else "unknown",
                    role="buyer" if ctx and ctx.is_buying else "seller",
                    round=int(getattr(state, "step", 0) or 0),
                    action_op="REJECT",
                    offer=self._pack_offer(offer, state.current_offer),
                )
        return offer

    def _parse_response_action(self, response: SAOResponse) -> Tuple[str, Optional[Outcome]]:
        if response.response == ResponseType.ACCEPT_OFFER:
            return "ACCEPT", response.outcome if hasattr(response, "outcome") else None
        if response.response == ResponseType.REJECT_OFFER:
            return "REJECT", response.outcome if hasattr(response, "outcome") else None
        return "END", None

    def _pack_offer(
        self,
        offer: Optional[Outcome],
        fallback: Optional[Outcome],
    ) -> Optional[Dict[str, Any]]:
        if offer is None:
            offer = fallback
        if offer is None:
            return None
        try:
            qty, delivery, price = offer
            delta_t = int(delivery - self.awi.current_step)
            return {
                "quantity": int(qty),
                "price": float(price),
                "delta_t": int(max(0, delta_t)),
            }
        except Exception:
            return None


__all__ = [
    "LitaAgentHRL",
    "LitaAgentHRLTracked",
    "NegotiationContext",
]
