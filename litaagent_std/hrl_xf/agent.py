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

# 尝试导入 Tracker（可选依赖）
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
    l4_weight: float = 1.0


class LitaAgentHRL(StdAgent):
    """HRL-XF 分层强化学习代理.
    
    支持模式：
    - heuristic: 所有层使用启发式规则（无需训练）
    - neural: 使用预训练神经网络（需要 PyTorch）
    
    Args:
        mode: "heuristic" 或 "neural"
        horizon: 规划视界 H
        debug: 是否输出调试信息
    """
    
    def __init__(
        self,
        *args,
        mode: str = "heuristic",
        horizon: int = 40,
        debug: bool = False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.mode = mode
        self.horizon = horizon
        self.debug = debug
        
        # 初始化各层
        self.l1 = L1SafetyLayer(horizon=horizon)
        self.state_builder = StateBuilder(horizon=horizon)
        self.l2 = L2StrategicManager(mode=mode, horizon=horizon)
        self.l3 = L3ResidualExecutor(mode=mode, horizon=horizon)
        self.l4 = L4ThreadCoordinator(mode=mode, horizon=horizon)
        
        # 运行时状态
        self._contexts: Dict[str, NegotiationContext] = {}
        self._current_state: Optional[StateDict] = None
        self._current_l2_output: Optional[L2Output] = None
        self._step_l1_buy: Optional[L1Output] = None
        self._step_l1_sell: Optional[L1Output] = None
    
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
        
        # 构建状态（先用买方视角，后续按需切换）
        self._current_state = self.state_builder.build(self.awi, is_buying=True)
        
        # 计算 L1 安全约束
        self._step_l1_buy = self.l1.compute(self.awi, is_buying=True)
        self._step_l1_sell = self.l1.compute(self.awi, is_buying=False)
        
        # 计算 L2 日级目标
        self._current_l2_output = self.l2.compute(
            self._current_state.x_static,
            self._current_state.X_temporal,
            is_buying=True  # L2 输出是对称的，包含买卖两方向
        )
        
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
        
        # 记录对手报价到历史
        qty, delivery, price = offer
        t_current = self.awi.current_step
        delta_t = delivery - t_current
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
            delta_t = delivery - t_current
            ctx.history.append(NegotiationRound(
                quantity=qty,
                price=price,
                delta_t=delta_t,
                is_my_turn=True
            ))
        
        return offer
    
    # ==================== 核心逻辑 ====================
    
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
            if qty > l1_output.Q_safe[delta_t]:
                return False
            if qty * price > l1_output.B_free:
                return False
            
            # 检查价格（应低于限价）
            # 从 L2 目标中获取对应桶的价格限制
            bucket_goal = l2_output.get_bucket_goal(self._delta_to_bucket(delta_t))
            if price > bucket_goal["P_buy"] * 1.1:  # 允许 10% 容差
                return False
        else:
            # 卖方：检查价格（应高于底价）
            bucket_goal = l2_output.get_bucket_goal(self._delta_to_bucket(delta_t))
            if price < bucket_goal["P_sell"] * 0.9:  # 允许 10% 容差
                return False
        
        return True
    
    def _generate_offer(
        self,
        ctx: NegotiationContext,
        state: SAOState
    ) -> Optional[Outcome]:
        """生成报价.
        
        流程：
        1. L1 提供基准动作和约束
        2. L3 生成残差
        3. L4 调制（如果有多个活跃线程）
        4. L1 裁剪最终动作
        """
        l1_output = ctx.l1_output
        l2_output = ctx.l2_output
        
        if l1_output is None or l2_output is None:
            return None
        
        # L1 基准动作
        baseline = l1_output.baseline_action
        
        # L3 残差
        l3_output = self.l3.compute(
            history=ctx.history,
            goal=l2_output.goal_vector,
            is_buying=ctx.is_buying,
            time_mask=l1_output.time_mask,
            baseline=baseline
        )
        ctx.l3_output = l3_output
        
        # L4 协调（获取权重）
        l4_output = self._compute_l4_weights()
        ctx.l4_weight = l4_output.modulation_factors[0] if len(l4_output.modulation_factors) > 0 else 1.0
        
        # 应用调制
        delta_q, delta_p = self.l4.modulate_action(
            l3_output.delta_q,
            l3_output.delta_p,
            ctx.l4_weight
        )
        
        # 计算最终动作
        q_final = baseline[0] + delta_q
        p_final = baseline[1] + delta_p
        t_final = l3_output.t_final
        
        # L1 安全裁剪
        clipped = self.l1.clip_action(
            action=(q_final, p_final, t_final),
            Q_safe=l1_output.Q_safe,
            B_free=l1_output.B_free,
            is_buying=ctx.is_buying
        )
        
        qty, price, delta_t = clipped
        
        if qty <= 0:
            return None
        
        # 转换为绝对交货时间
        delivery_time = self.awi.current_step + delta_t
        
        return (int(qty), int(delivery_time), float(price))
    
    def _compute_l4_weights(self) -> L4Output:
        """计算所有活跃线程的 L4 权重."""
        threads = []
        
        for ctx in self._contexts.values():
            if ctx.l3_output is not None:
                # 使用简化的隐状态（实际应使用 L3 的真实隐状态）
                hidden_state = np.zeros(128, dtype=np.float32)
                hidden_state[0] = ctx.l3_output.q_final
                hidden_state[1] = ctx.l3_output.p_final
                hidden_state[2] = ctx.l3_output.t_final
                
                threads.append(ThreadState(
                    thread_id=ctx.negotiation_id,
                    hidden_state=hidden_state,
                    target_time=ctx.l3_output.t_final,
                    role=0 if ctx.is_buying else 1
                ))
        
        # 如果没有线程，返回默认权重
        if len(threads) == 0:
            # 为当前线程创建一个虚拟条目
            threads.append(ThreadState(
                thread_id="current",
                hidden_state=np.zeros(128, dtype=np.float32),
                target_time=1,
                role=0
            ))
        
        goal = self._current_l2_output.goal_vector if self._current_l2_output else np.zeros(16)
        
        return self.l4.compute(threads, goal)
    
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
