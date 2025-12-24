"""LitaAgent-HRL (HRL-XF) 主代理类.

整合 L1-L4 层，实现 SCML StdAgent 接口。

层级职责：
- L1 (Safety): 时序 ATP 安全约束，输出 Q_safe, time_mask, baseline
- L2 (Manager): 日级战略规划，输出 16 维分桶目标
- L3 (Executor): 轮级残差执行，输出 (Δq, Δp, δt)
- L4 (Coordinator): 并发协调，输出线程权重

信息流：
AWI → StateBuilder → L1/L2 → L3 → L4 → 最终动作
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

from negmas.sao import SAOResponse, SAOState, ResponseType
from negmas.outcomes import Outcome
from scml.std import StdAgent

from .l1_safety import L1SafetyLayer, L1Output
from .state_builder import StateBuilder, StateDict
from .l2_manager import L2StrategicManager, L2Output
from .l3_executor import L3ResidualExecutor, L3Output, NegotiationRound
from .l4_coordinator import L4ThreadCoordinator, L4Output, ThreadState
from .batch_planner import plan_buy_offers_by_alpha, plan_sell_offers_by_alpha

# 尝试导入 Tracker（可选依赖）
try:
    from scml_analyzer.auto_tracker import TrackerManager
    _TRACKER_AVAILABLE = True
except ImportError:
    _TRACKER_AVAILABLE = False
TrackerManager = None


L4_THREAD_FEAT_DIM = 24
L4_GLOBAL_FEAT_DIM = 30


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
    l4_weight: float = 1.0
    l4_alpha: float = 1.0


@dataclass
class _BatchPlan:
    """同一谈判轮次下的批次规划缓存（用于消除顺序依赖）."""

    signature: Tuple[int, Tuple[str, ...], Tuple[int, ...]]
    offers: Dict[str, Optional[Outcome]]
    l4_output: L4Output


class LitaAgentHRL(StdAgent):
    """HRL-XF 分层强化学习代理.
    
    支持模式：
    - heuristic: 所有层使用启发式规则（无需训练）
    - neural: 使用预训练神经网络（需要 PyTorch）
    
    Args:
        mode: "heuristic" 或 "neural"
        horizon: 规划视界 H
        debug: 是否输出调试信息
        l2_model_path: L2 权重路径（neural 模式）
        l3_model_path: L3 权重路径（neural 模式）
        l4_model_path: L4 权重路径（neural 模式）
    """
    
    def __init__(
        self,
        *args,
        mode: str = "heuristic",
        horizon: int = 40,
        debug: bool = False,
        l2_model_path: Optional[str] = None,
        l3_model_path: Optional[str] = None,
        l4_model_path: Optional[str] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.mode = mode
        self.horizon = horizon
        self.debug = debug
        
        # 初始化各层
        self.l1 = L1SafetyLayer(horizon=horizon)
        self.state_builder = StateBuilder(horizon=horizon)
        self.l2 = L2StrategicManager(mode=mode, horizon=horizon, model_path=l2_model_path)
        self.l3 = L3ResidualExecutor(mode=mode, horizon=horizon, model_path=l3_model_path)
        self.l4 = L4ThreadCoordinator(
            mode=mode,
            horizon=horizon,
            model_path=l4_model_path,
            thread_feat_dim=L4_THREAD_FEAT_DIM,
            global_feat_dim=L4_GLOBAL_FEAT_DIM,
        )
        
        # 运行时状态
        self._contexts: Dict[str, NegotiationContext] = {}
        self._current_state: Optional[StateDict] = None
        self._current_l2_output: Optional[L2Output] = None
        self._step_l1_buy: Optional[L1Output] = None
        self._step_l1_sell: Optional[L1Output] = None
        self._batch_plan: Optional[_BatchPlan] = None
        self._buy_budget_committed: float = 0.0
        self._buy_q_committed: np.ndarray = np.zeros((horizon + 1,), dtype=np.float32)
    
    # ==================== 生命周期 ====================
    
    def init(self):
        """代理初始化."""
        super().init()
        if self.debug:
            self._log("LitaAgent-HRL (HRL-XF) initialized")
    
    def before_step(self):
        """每个仿真步开始前调用.
        
        职责：
        1. 构建当前状态
        2. 计算 L1 安全约束（买/卖两套）
        3. 计算 L2 日级目标
        """
        super().before_step()
        
        # 清理上一步的上下文
        self._contexts.clear()
        self._batch_plan = None
        self._buy_budget_committed = 0.0
        self._buy_q_committed = np.zeros((self.horizon + 1,), dtype=np.float32)
        
        # 构建状态（先用买方视角，后续按需切换）
        self._current_state = self.state_builder.build(self.awi, is_buying=True)
        
        # 计算 L1 安全约束
        self._step_l1_buy = self.l1.compute(self.awi, is_buying=True)
        self._step_l1_sell = self.l1.compute(self.awi, is_buying=False)
        
        # 计算 L2 日级目标
        self._current_l2_output = self.l2.compute(
            self._current_state.x_static,
            self._current_state.X_temporal,
            is_buying=True,  # L2 输出是对称的，包含买卖两方向
            awi=self.awi,  # neural 模式需要 awi 来计算 x_role（Multi-Hot 谈判能力）
        )

        # 预创建所有谈判上下文，减少后续 L4 计算的顺序依赖
        negotiators = getattr(self, "negotiators", {}) or {}
        for negotiator_id in negotiators.keys():
            ctx = self._get_or_create_context(negotiator_id)
            ctx.l1_output = self._step_l1_buy if ctx.is_buying else self._step_l1_sell
            ctx.l2_output = self._current_l2_output
        
        if self.debug:
            self._log(
                f"Step {self.awi.current_step}: "
                f"L2 goal = {self._current_l2_output.goal_vector[:4]}... "
                f"B_free = {self._step_l1_buy.B_free:.2f}"
            )
    
    def step(self):
        """每个仿真步结束时调用."""
        super().step()
        
        # 可以在这里收集训练数据或更新统计
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
            if not is_buy or qty <= 0 or price <= 0:
                return

            delta_t = int(max(0, min(day - int(self.awi.current_step), self.horizon)))
            self._buy_budget_committed += qty * price
            if 0 <= delta_t < len(self._buy_q_committed):
                self._buy_q_committed[delta_t] += qty

            # 资源已变化，下一次决策需要重新规划
            self._batch_plan = None
        except Exception:
            return
    
    # ==================== 谈判接口 ====================
    
    def respond(self, negotiator_id: str, state: SAOState) -> SAOResponse:
        """响应对手报价.
        
        Args:
            negotiator_id: 谈判者 ID
            state: 当前谈判状态
            
        Returns:
            SAOResponse: ACCEPT/REJECT/END
        """
        offer = state.current_offer
        if offer is None:
            return SAOResponse(ResponseType.REJECT_OFFER, None)
        
        # 获取或创建谈判上下文
        ctx = self._get_or_create_context(negotiator_id)
        ctx.last_relative_time = float(getattr(state, "relative_time", 0.0) or 0.0)
        
        # 记录对手报价到历史
        qty, delivery, price = offer
        t_current = self.awi.current_step
        delta_t = int(delivery - t_current)
        delta_t = int(max(0, min(delta_t, self.horizon)))
        ctx.last_delta_t = delta_t
        ctx.history.append(NegotiationRound(
            quantity=qty,
            price=price,
            delta_t=delta_t,
            is_my_turn=False
        ))
        
        # 获取 L1 约束
        l1_output = self._step_l1_buy if ctx.is_buying else self._step_l1_sell
        ctx.l1_output = l1_output
        ctx.l2_output = self._current_l2_output
        
        # 检查是否应该接受
        if self._should_accept(offer, ctx):
            return SAOResponse(ResponseType.ACCEPT_OFFER, offer)
        
        # 生成反报价
        counter_offer = self._generate_offer(ctx, state)
        
        if counter_offer is None or counter_offer[0] <= 0:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        # 记录自己的还价到历史（用于后续 L3/L4 特征构建）
        qty, delivery, price = counter_offer
        t_current = self.awi.current_step
        delta_t = int(delivery - t_current)
        delta_t = int(max(0, min(delta_t, self.horizon)))
        ctx.last_delta_t = delta_t
        ctx.history.append(
            NegotiationRound(
                quantity=qty,
                price=price,
                delta_t=delta_t,
                is_my_turn=True,
            )
        )
        
        return SAOResponse(ResponseType.REJECT_OFFER, counter_offer)
    
    def propose(self, negotiator_id: str, state: SAOState) -> Optional[Outcome]:
        """主动提出报价.
        
        Args:
            negotiator_id: 谈判者 ID
            state: 当前谈判状态
            
        Returns:
            Outcome: (quantity, delivery_time, price) 或 None
        """
        # 获取或创建谈判上下文
        ctx = self._get_or_create_context(negotiator_id)
        ctx.last_relative_time = float(getattr(state, "relative_time", 0.0) or 0.0)
        
        # 获取 L1 约束
        l1_output = self._step_l1_buy if ctx.is_buying else self._step_l1_sell
        ctx.l1_output = l1_output
        ctx.l2_output = self._current_l2_output
        
        # 生成报价
        offer = self._generate_offer(ctx, state)
        
        if offer is not None and offer[0] > 0:
            # 记录自己的报价到历史
            qty, delivery, price = offer
            t_current = self.awi.current_step
            delta_t = int(delivery - t_current)
            delta_t = int(max(0, min(delta_t, self.horizon)))
            ctx.last_delta_t = delta_t
            ctx.history.append(NegotiationRound(
                quantity=qty,
                price=price,
                delta_t=delta_t,
                is_my_turn=True
            ))
        
        return offer
    
    # ==================== 核心逻辑 ====================

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

        offer = getattr(st, "current_offer", None)
        if offer is not None:
            try:
                _, delivery, _ = offer
                delta_t = int(delivery - self.awi.current_step)
                ctx.last_delta_t = int(max(0, min(delta_t, self.horizon)))
            except Exception:
                pass

    def _make_batch_signature(
        self, negotiator_ids: List[str], focus_id: Optional[str] = None
    ) -> Tuple[int, Tuple[str, ...], Tuple[int, ...]]:
        ids = tuple(sorted(set(negotiator_ids)))
        steps: List[int] = []
        focus_step = 0
        for nid in ids:
            st = self._get_mechanism_state(nid)
            step = int(getattr(st, "step", 0) or 0) if st is not None else 0
            steps.append(step)
            if focus_id is not None and nid == focus_id:
                focus_step = step
        # 使用全局最小 step 作为“同步轮次”近似：单线程推进不会立即使缓存失效，从而减少顺序依赖
        global_round = min(steps) if len(steps) > 0 else 0
        if focus_id is None:
            focus_step = global_round
        return (int(self.awi.current_step), ids, (int(global_round), int(focus_step)))

    def _ensure_batch_plan(self, include_id: Optional[str] = None) -> _BatchPlan:
        ids = self._get_active_negotiator_ids()
        if include_id is not None and include_id not in ids:
            ids.append(include_id)

        signature = self._make_batch_signature(ids, focus_id=include_id)
        if self._batch_plan is not None and self._batch_plan.signature == signature:
            return self._batch_plan

        plan = self._plan_batch(list(signature[1]), signature)
        self._batch_plan = plan
        return plan

    def _compute_unclipped_action(self, ctx: NegotiationContext, l4_weight: float) -> Tuple[float, float, int]:
        l1_output = ctx.l1_output
        l2_output = ctx.l2_output
        if l1_output is None or l2_output is None:
            return (0.0, 0.0, 0)

        baseline = l1_output.baseline_action
        l3_output = self.l3.compute(
            history=ctx.history,
            goal=l2_output.goal_vector,
            is_buying=ctx.is_buying,
            time_mask=l1_output.time_mask,
            baseline=baseline,
        )
        ctx.l3_output = l3_output

        delta_q, delta_p = self.l4.modulate_action(l3_output.delta_q, l3_output.delta_p, l4_weight)
        q_final = baseline[0] + delta_q
        p_final = baseline[1] + delta_p
        t_final = int(l3_output.t_final)
        return (float(q_final), float(p_final), t_final)

    def _plan_batch(self, negotiator_ids: List[str], signature: Tuple[int, Tuple[str, ...], Tuple[int, ...]]) -> _BatchPlan:
        if self._current_state is None or self._current_l2_output is None:
            empty = {nid: None for nid in negotiator_ids}
            return _BatchPlan(
                signature=signature,
                offers=empty,
                l4_output=L4Output(thread_ids=[], weights=np.array([]), modulation_factors=np.array([]), conflict_scores=None),
            )

        # 1) 刷新所有线程的可观测状态（relative_time/current_offer 等），构建 L4 输入
        threads: List[ThreadState] = []
        for negotiator_id in negotiator_ids:
            ctx = self._get_or_create_context(negotiator_id)
            ctx.l1_output = self._step_l1_buy if ctx.is_buying else self._step_l1_sell
            ctx.l2_output = self._current_l2_output
            self._refresh_context_from_mechanism(ctx)
            ts = self._build_l4_thread_state(ctx)
            if ts is not None:
                threads.append(ts)

        if len(threads) == 0:
            threads.append(
                ThreadState(
                    thread_id="current",
                    thread_feat=np.zeros(L4_THREAD_FEAT_DIM, dtype=np.float32),
                    target_time=1,
                    role=0,
                    priority=1.0,
                )
            )

        global_feat = self._build_l4_global_feat(threads)
        l4_output = self.l4.compute(threads, global_feat)

        # 2) 为每个线程计算未裁剪动作（baseline + L3 residual，经 L4 modulation 调制）
        l4_weight_by_id: Dict[str, float] = {}
        l4_alpha_by_id: Dict[str, float] = {}
        if l4_output.thread_ids:
            for idx, tid in enumerate(l4_output.thread_ids):
                if tid is None:
                    continue
                if idx < len(l4_output.modulation_factors):
                    l4_weight_by_id[tid] = float(l4_output.modulation_factors[idx])
                if idx < len(l4_output.weights):
                    l4_alpha_by_id[tid] = float(l4_output.weights[idx])

        actions: Dict[str, Tuple[float, float, int]] = {}
        for negotiator_id in negotiator_ids:
            ctx = self._get_or_create_context(negotiator_id)
            ctx.l4_weight = float(l4_weight_by_id.get(negotiator_id, 1.0))
            ctx.l4_alpha = float(l4_alpha_by_id.get(negotiator_id, 1.0))
            actions[negotiator_id] = self._compute_unclipped_action(ctx, ctx.l4_weight)

        # 3) 全局动态预留：按 α 对买方线程进行预算/安全量裁剪（顺序无关：排序只依赖 α 与 thread_id）
        offers: Dict[str, Optional[Outcome]] = {nid: None for nid in negotiator_ids}

        buy_ids = [nid for nid in negotiator_ids if self._get_or_create_context(nid).is_buying]
        sell_ids = [nid for nid in negotiator_ids if nid not in buy_ids]

        if self._step_l1_buy is not None:
            B_available = max(0.0, float(self._step_l1_buy.B_free) - float(self._buy_budget_committed))
            Q_available = self._step_l1_buy.Q_safe.astype(np.float32).copy()
            n = min(len(Q_available), len(self._buy_q_committed))
            if n > 0:
                Q_available[:n] = np.maximum(0.0, Q_available[:n] - self._buy_q_committed[:n])
            # 获取 raw_free 用于正确的动态预留，并扣减已承诺量
            raw_free_orig = getattr(self._step_l1_buy, 'raw_free', None)
            if raw_free_orig is not None:
                raw_free = raw_free_orig.astype(np.float32).copy()
                # 扣减 _buy_q_committed：对每个 delta，扣减该 delta 及之后的所有位置
                # 因为在 delta 买入的货物从 delta 开始占用空间
                for delta in range(min(len(raw_free), len(self._buy_q_committed))):
                    committed = self._buy_q_committed[delta]
                    if committed > 0:
                        for k in range(delta, len(raw_free)):
                            raw_free[k] -= committed
                raw_free = np.maximum(raw_free, 0)
            else:
                raw_free = None
        else:
            B_available = 0.0
            Q_available = np.zeros((self.horizon + 1,), dtype=np.float32)
            raw_free = None

        offers.update(
            plan_buy_offers_by_alpha(
                l1=self.l1,
                buy_ids=buy_ids,
                actions=actions,
                alphas=l4_alpha_by_id,
                Q_safe=Q_available,
                B_free=B_available,
                current_step=int(self.awi.current_step),
                raw_free=raw_free,
            )
        )

        # 卖侧也需要动态预留，避免多个线程重复使用同一份 Q_safe_sell
        if self._step_l1_sell is not None:
            Q_sell_available = self._step_l1_sell.Q_safe_sell.astype(np.float32).copy()
        else:
            Q_sell_available = np.zeros((self.horizon + 1,), dtype=np.float32)
        
        offers.update(
            plan_sell_offers_by_alpha(
                l1=self.l1,
                sell_ids=sell_ids,
                actions=actions,
                alphas=l4_alpha_by_id,
                Q_safe_sell=Q_sell_available,
                current_step=int(self.awi.current_step),
            )
        )

        return _BatchPlan(signature=signature, offers=offers, l4_output=l4_output)
    
    def _should_accept(self, offer: Outcome, ctx: NegotiationContext) -> bool:
        """判断是否应该接受报价.
        
        判断标准：
        1. 价格在可接受范围内
        2. 数量在安全范围内
        3. 交货时间可行
        """
        qty, delivery, price = offer
        t_current = self.awi.current_step
        delta_t = min(delivery - t_current, self.horizon)
        
        l1_output = ctx.l1_output
        l2_output = ctx.l2_output
        
        if l1_output is None or l2_output is None:
            return False
        
        # 检查时间可行性
        if delta_t < 0 or delta_t > self.horizon:
            return False
        
        # 检查安全约束
        if ctx.is_buying:
            # 买方：检查库容和资金
            q_cap = float(l1_output.Q_safe[delta_t]) if delta_t < len(l1_output.Q_safe) else 0.0
            if 0 <= delta_t < len(self._buy_q_committed):
                q_cap = max(0.0, q_cap - float(self._buy_q_committed[delta_t]))
            if qty > q_cap:
                return False
            b_cap = max(0.0, float(l1_output.B_free) - float(self._buy_budget_committed))
            if qty * price > b_cap:
                return False
            
            # 检查价格（应低于限价）
            # 从 L2 目标中获取对应桶的价格限制
            bucket_goal = l2_output.get_bucket_goal(self._delta_to_bucket(delta_t))
            if price > bucket_goal["P_buy"] * 1.1:  # 允许 10% 容差
                return False
        else:
            # 卖方：检查可交付性和价格
            # 检查数量是否在安全卖出范围内
            q_sell_cap = float(l1_output.Q_safe_sell[delta_t]) if delta_t < len(l1_output.Q_safe_sell) else 0.0
            if qty > q_sell_cap:
                return False
            
            # 检查价格（应高于底价）
            bucket_goal = l2_output.get_bucket_goal(self._delta_to_bucket(delta_t))
            if price < bucket_goal["P_sell"] * 0.9:  # 允许 10% 容差
                return False
        
        return True
    
    def _generate_offer(
        self,
        ctx: NegotiationContext,
        state: SAOState
    ) -> Optional[Outcome]:
        """生成报价（方案 B：批次统一规划 + 动态预留 + 缓存）."""
        plan = self._ensure_batch_plan(include_id=ctx.negotiation_id)
        offer = plan.offers.get(ctx.negotiation_id)
        if offer is None:
            return None
        if offer[0] <= 0:
            return None
        return offer
    
    def _compute_l4_weights(self) -> L4Output:
        """返回当前批次（方案B）的 L4 输出（复用缓存以保持一致性）."""
        plan = self._ensure_batch_plan()
        return plan.l4_output

    def _build_l4_global_feat(self, threads: List[ThreadState]) -> np.ndarray:
        """构建 L4 的全局特征（对所有线程相同）."""
        goal = self._current_l2_output.goal_vector.astype(np.float32)  # (16,)
        x_static = self._current_state.x_static.astype(np.float32)  # (12,)
        n_buy = sum(1 for t in threads if t.role == 0)
        n_sell = len(threads) - n_buy
        counts = np.array([n_buy / 10.0, n_sell / 10.0], dtype=np.float32)
        global_feat = np.concatenate([goal, x_static, counts], axis=0)
        if global_feat.shape[0] != L4_GLOBAL_FEAT_DIM:
            padded = np.zeros(L4_GLOBAL_FEAT_DIM, dtype=np.float32)
            n = min(L4_GLOBAL_FEAT_DIM, global_feat.shape[0])
            padded[:n] = global_feat[:n]
            return padded
        return global_feat

    def _build_l4_thread_state(self, ctx: NegotiationContext) -> Optional[ThreadState]:
        """构建单线程的 L4 输入特征."""
        if ctx.l1_output is None or ctx.l2_output is None or self._current_state is None:
            return None

        baseline_q, baseline_p, baseline_t = ctx.l1_output.baseline_action

        # 线程关注的交期：优先用该谈判最近一次出现的 delta_t，否则用 baseline 的 t_base
        target_delta = int(ctx.last_delta_t) if ctx.last_delta_t is not None else int(baseline_t)
        target_delta = int(max(0, min(target_delta, self.horizon)))

        bucket = self._delta_to_bucket(target_delta)
        bucket_goal = ctx.l2_output.get_bucket_goal(bucket)

        max_inv = max(float(self.state_builder.max_inventory), 1.0)
        max_price = max(float(self.state_builder.max_price), 1.0)

        # 取该交期的时序切片（已归一化）
        X = self._current_state.X_temporal
        x_t = X[target_delta] if target_delta < X.shape[0] else np.zeros((10,), dtype=np.float32)

        # 谈判进度与最近报价特征
        n_rounds = len(ctx.history)
        n_rounds_norm = float(min(n_rounds, 20) / 20.0)
        rel_time = float(np.clip(ctx.last_relative_time, 0.0, 1.0))

        last_price_gap = 0.0
        last_qty_gap = 0.0
        if n_rounds > 0:
            last = ctx.history[-1]
            q_goal = float(bucket_goal["Q_buy"] if ctx.is_buying else bucket_goal["Q_sell"])
            p_goal = float(bucket_goal["P_buy"] if ctx.is_buying else bucket_goal["P_sell"])
            last_price_gap = (float(last.price) - p_goal) / max_price
            last_qty_gap = (float(last.quantity) - q_goal) / max_inv

        # 线程优先级：紧迫度 + 市场压力 + 谈判进度 + 可成交性（报价接近目标）
        urgency = 1.0 / (target_delta + 1.0)
        market_pressure = float(x_t[9] if ctx.is_buying else x_t[8])  # buy 用 sell_pressure，sell 用 buy_pressure
        closeness = 1.0 / (1.0 + abs(last_price_gap) * 5.0)
        priority = 1.0 + 0.5 * urgency + 0.5 * market_pressure + 0.5 * rel_time + 0.5 * closeness
        priority = float(np.clip(priority, 0.1, 3.0))

        # 线程特征向量（固定维度，可离线重建）
        feat = np.zeros(L4_THREAD_FEAT_DIM, dtype=np.float32)
        feat[0] = priority
        feat[1] = float(target_delta / max(self.horizon, 1))
        feat[2] = 1.0 if ctx.l1_output.time_mask[target_delta] > -np.inf else 0.0
        # 根据角色选择对应的 Q_safe
        if ctx.is_buying:
            feat[3] = float((ctx.l1_output.Q_safe[target_delta] / max_inv) if target_delta < len(ctx.l1_output.Q_safe) else 0.0)
        else:
            feat[3] = float((ctx.l1_output.Q_safe_sell[target_delta] / max_inv) if target_delta < len(ctx.l1_output.Q_safe_sell) else 0.0)
        feat[4] = float(baseline_q / max_inv)
        feat[5] = float(baseline_p / max_price)
        feat[6] = float(float(bucket_goal["Q_buy"]) / max_inv)
        feat[7] = float(float(bucket_goal["Q_sell"]) / max_inv)
        feat[8] = float(float(bucket_goal["P_buy"]) / max_price)
        feat[9] = float(float(bucket_goal["P_sell"]) / max_price)
        feat[10] = n_rounds_norm
        feat[11] = rel_time
        feat[12] = float(np.clip(last_price_gap, -2.0, 2.0))
        feat[13] = float(np.clip(last_qty_gap, -2.0, 2.0))
        feat[14:24] = x_t[:10].astype(np.float32)

        return ThreadState(
            thread_id=ctx.negotiation_id,
            thread_feat=feat,
            target_time=target_delta,
            role=0 if ctx.is_buying else 1,
            priority=priority,
        )
    
    # ==================== 辅助方法 ====================
    
    def _get_or_create_context(self, negotiator_id: str) -> NegotiationContext:
        """获取或创建谈判上下文."""
        if negotiator_id not in self._contexts:
            is_buying = self._is_buying(negotiator_id)
            partner_id = self._get_partner_id(negotiator_id)
            
            self._contexts[negotiator_id] = NegotiationContext(
                negotiation_id=negotiator_id,
                is_buying=is_buying,
                partner_id=partner_id
            )
        
        return self._contexts[negotiator_id]
    
    def _is_buying(self, negotiator_id: str) -> bool:
        """判断是否为买方角色.
        
        通过 nmi.annotation["product"] 判断：
        - 如果 product == my_input_product，则是买入
        - 如果 product == my_output_product，则是卖出
        """
        nmi = self.get_nmi(negotiator_id)
        if nmi is None:
            # 回退：假设买入
            return True
        
        # 检查产品类型判断买卖方向
        annotation = getattr(nmi, 'annotation', {}) or {}
        product = annotation.get('product')
        
        # 如果产品是我的输入产品，则是买入；否则是卖出
        return product == self.awi.my_input_product
    
    def _get_partner_id(self, negotiator_id: str) -> str:
        """获取谈判对手 ID."""
        nmi = self.get_nmi(negotiator_id)
        if nmi is None:
            return "unknown"
        
        annotation = getattr(nmi, 'annotation', {}) or {}
        buyer_id = annotation.get('buyer')
        seller_id = annotation.get('seller')
        
        # 如果是买入，对手是卖方；否则对手是买方
        if self._is_buying(negotiator_id):
            return seller_id or "unknown"
        else:
            return buyer_id or "unknown"
    
    def _delta_to_bucket(self, delta: int) -> int:
        """将相对交货时间映射到桶索引."""
        if delta <= 2:
            return 0
        elif delta <= 7:
            return 1
        elif delta <= 14:
            return 2
        else:
            return 3
    
    def _log(self, msg: str) -> None:
        """输出日志."""
        agent_name = getattr(self, 'name', 'HRL')
        print(f"[{agent_name}] {msg}")


class LitaAgentHRLTracked(LitaAgentHRL):
    """带 Tracker 的 HRL-XF 代理.
    
    在 LitaAgentHRL 基础上添加数据追踪功能，
    用于收集训练数据和分析代理行为。
    """
    
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
            # 记录状态
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
            if ctx:
                self._tracker.record_negotiation({
                    "negotiation_id": negotiator_id,
                    "is_buying": ctx.is_buying,
                    "response_type": response.response.name,
                    "l3_output": {
                        "delta_q": ctx.l3_output.delta_q if ctx.l3_output else None,
                        "delta_p": ctx.l3_output.delta_p if ctx.l3_output else None,
                        "delta_t": ctx.l3_output.delta_t if ctx.l3_output else None,
                    },
                    "l4_weight": ctx.l4_weight,
                })
        
        return response


__all__ = [
    "LitaAgentHRL",
    "LitaAgentHRLTracked",
    "NegotiationContext",
]
