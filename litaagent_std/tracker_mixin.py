"""
LitaAgent Tracker Mixin

为所有 LitaAgent 提供统一的数据记录功能。
不需要修改原有 Agent 的代码，只需在运行时动态注入即可。

Usage:
    from litaagent_std.tracker_mixin import inject_tracker_to_agents
    
    # 在比赛前注入 Tracker
    inject_tracker_to_agents([LitaAgentY, LitaAgentYR, LitaAgentN])
"""

import os
import sys
from typing import List, Type, Dict, Any, Optional
from functools import wraps

# 尝试导入 Tracker
try:
    from scml_analyzer.auto_tracker import (
        TrackerManager, 
        TrackerConfig, 
        AgentLogger,
        configure_tracker,
        save_all_logs,
    )
    TRACKER_AVAILABLE = True
except ImportError:
    TRACKER_AVAILABLE = False


class LitaTrackerMixin:
    """
    为 LitaAgent 提供 Tracker 功能的 Mixin
    
    自动记录:
    - 初始化参数
    - 每日库存状态
    - 协商开始/成功/失败
    - 合同签署
    - 关键决策点
    """
    
    _tracker_logger: Optional['AgentLogger'] = None
    _tracker_enabled: bool = True
    
    @property
    def tracker(self) -> Optional['AgentLogger']:
        """获取 Tracker Logger"""
        if not TRACKER_AVAILABLE or not self._tracker_enabled:
            return None
        
        if self._tracker_logger is None:
            agent_id = getattr(self, 'id', 'unknown')
            agent_type = type(self).__name__
            self._tracker_logger = TrackerManager.get_logger(agent_id, agent_type)
        
        return self._tracker_logger
    
    def track_event(self, event: str, **data):
        """记录自定义事件"""
        if self.tracker:
            self.tracker.custom(event, **data)
    
    def track_decision(self, name: str, result: Any, reason: str = "", **context):
        """记录决策"""
        if self.tracker:
            self.tracker.decision(name, result, reason, **context)
    
    def track_negotiation(self, event: str, partner: str, **data):
        """记录协商事件"""
        if self.tracker:
            if event == "started":
                self.tracker.negotiation_started(partner, data.get("issues", {}), data.get("is_seller", False))
            elif event == "offer_made":
                self.tracker.negotiation_offer_made(partner, data.get("offer", {}), data.get("reason", ""))
            elif event == "offer_received":
                self.tracker.negotiation_offer_received(partner, data.get("offer", {}))
            elif event == "accept":
                self.tracker.negotiation_accept(partner, data.get("offer", {}), data.get("reason", ""))
            elif event == "reject":
                self.tracker.negotiation_reject(partner, data.get("offer", {}), data.get("reason", ""))
            elif event == "success":
                self.tracker.negotiation_success(partner, data.get("agreement", {}))
            elif event == "failure":
                self.tracker.negotiation_failure(partner, data.get("reason", ""))
    
    def track_contract(self, contract_id: str, partner: str, quantity: int, 
                       price: float, delivery_day: int, is_seller: bool):
        """记录合同签署"""
        if self.tracker:
            self.tracker.contract_signed(contract_id, partner, quantity, price, delivery_day, is_seller)
    
    def track_inventory(self, raw_material: int, product: int, balance: float, **extra):
        """记录库存状态"""
        if self.tracker:
            self.tracker.inventory_state(raw_material, product, balance, **extra)
    
    def track_production(self, event: str, quantity: int, **data):
        """记录生产事件"""
        if self.tracker:
            if event == "scheduled":
                self.tracker.production_scheduled(quantity, data.get("day", 0))
            elif event == "executed":
                self.tracker.production_executed(quantity)


def patch_agent_class(agent_class: Type) -> Type:
    """
    动态修补 Agent 类以添加 Tracker 功能
    
    这种方法不需要修改原有代码，通过方法包装实现自动记录。
    """
    if not TRACKER_AVAILABLE:
        return agent_class
    
    # 保存原始方法
    original_init = agent_class.init if hasattr(agent_class, 'init') else None
    original_before_step = agent_class.before_step if hasattr(agent_class, 'before_step') else None
    original_on_negotiation_success = agent_class.on_negotiation_success if hasattr(agent_class, 'on_negotiation_success') else None
    original_on_negotiation_failure = agent_class.on_negotiation_failure if hasattr(agent_class, 'on_negotiation_failure') else None
    original_counter_all = agent_class.counter_all if hasattr(agent_class, 'counter_all') else None
    original_first_proposals = agent_class.first_proposals if hasattr(agent_class, 'first_proposals') else None
    
    # 添加 tracker 属性
    agent_class._tracker_logger = None
    agent_class._tracker_enabled = True
    
    def get_tracker(self) -> Optional[AgentLogger]:
        if not self._tracker_enabled:
            return None
        if self._tracker_logger is None:
            agent_id = getattr(self, 'id', 'unknown')
            agent_type = type(self).__name__
            self._tracker_logger = TrackerManager.get_logger(agent_id, agent_type)
        return self._tracker_logger
    
    agent_class.tracker = property(get_tracker)
    
    # 包装 init 方法
    if original_init:
        def patched_init(self):
            original_init(self)
            tracker = self.tracker
            if tracker:
                tracker.custom("agent_initialized", 
                    n_steps=getattr(self.awi, 'n_steps', None),
                    n_lines=getattr(self.awi, 'n_lines', None),
                    level=getattr(self.awi, 'level', None),
                )
        agent_class.init = patched_init
    
    # 包装 before_step 方法
    if original_before_step:
        def patched_before_step(self):
            original_before_step(self)
            tracker = self.tracker
            if tracker:
                tracker.set_day(self.awi.current_step)
                # 记录库存状态和每日数据
                try:
                    # OneShot/Std 环境 (2024+) 的 API:
                    # 注意：在 OneShot 环境中，产品是易腐的（perishable），库存总是为 0
                    # 所以我们需要记录其他有价值的数据：
                    # - current_balance: 当前余额
                    # - current_score: 当前分数 (balance / initial_balance)
                    # - current_exogenous_*: 外生合同（系统分配的合同）
                    # - needed_supplies/sales: 今日需要通过协商获得的数量
                    # - total_supplies/sales: 今日已签约的数量
                    
                    awi = self.awi
                    
                    # 基础财务数据
                    balance = getattr(awi, 'current_balance', 0.0) or 0.0
                    score = getattr(awi, 'current_score', 1.0) or 1.0
                    
                    # 外生合同 (exogenous contracts) - 由系统分配的必须履行的合同
                    exo_input_qty = getattr(awi, 'current_exogenous_input_quantity', 0) or 0
                    exo_output_qty = getattr(awi, 'current_exogenous_output_quantity', 0) or 0
                    exo_input_price = getattr(awi, 'current_exogenous_input_price', 0) or 0
                    exo_output_price = getattr(awi, 'current_exogenous_output_price', 0) or 0
                    
                    # 今日协商需求
                    needed_supplies = getattr(awi, 'needed_supplies', 0) or 0
                    needed_sales = getattr(awi, 'needed_sales', 0) or 0
                    
                    # 已签约数量
                    total_supplies = getattr(awi, 'total_supplies', 0) or 0
                    total_sales = getattr(awi, 'total_sales', 0) or 0
                    
                    # 成本和惩罚参数
                    disposal_cost = getattr(awi, 'current_disposal_cost', 0) or 0
                    shortfall_penalty = getattr(awi, 'current_shortfall_penalty', 0) or 0
                    storage_cost = getattr(awi, 'current_storage_cost', 0) or 0
                    
                    # 生产能力
                    n_lines = getattr(awi, 'n_lines', 0) or 0
                    
                    # 记录基本库存状态（使用外生合同数量作为"虚拟库存"概念）
                    # 在 OneShot 中: raw_material = 外生输入, product = 外生输出
                    tracker.inventory_state(
                        raw_material=exo_input_qty,  # 今日获得的原材料数量
                        product=exo_output_qty,       # 今日需要交付的产品数量
                        balance=balance,
                    )
                    
                    # 记录完整的每日状态
                    tracker.custom("daily_status",
                        # 财务状态
                        score=score,
                        balance=balance,
                        # 外生合同（系统分配）
                        exo_input_qty=exo_input_qty,
                        exo_input_price=exo_input_price,
                        exo_output_qty=exo_output_qty,
                        exo_output_price=exo_output_price,
                        # 协商需求
                        needed_supplies=needed_supplies,
                        needed_sales=needed_sales,
                        # 已签约
                        total_supplies=total_supplies,
                        total_sales=total_sales,
                        # 成本参数
                        disposal_cost=disposal_cost,
                        shortfall_penalty=shortfall_penalty,
                        storage_cost=storage_cost,
                        # 生产能力
                        n_lines=n_lines,
                    )
                        
                except Exception as e:
                    # 记录错误以便调试
                    tracker.custom("inventory_state_error", error=str(e))
        agent_class.before_step = patched_before_step
    
    # 包装 on_negotiation_success 方法
    if original_on_negotiation_success:
        def patched_on_negotiation_success(self, contract, mechanism):
            original_on_negotiation_success(self, contract, mechanism)
            tracker = self.tracker
            if tracker:
                try:
                    partner = [p for p in contract.partners if p != self.id][0]
                    agreement = contract.agreement
                    is_seller = not getattr(self.awi, 'is_first_level', True)
                    
                    tracker.contract_signed(
                        contract_id=str(contract.id),
                        partner=partner,
                        quantity=agreement.get("quantity", 0) if isinstance(agreement, dict) else getattr(agreement, 'quantity', 0),
                        price=agreement.get("unit_price", 0) if isinstance(agreement, dict) else getattr(agreement, 'unit_price', 0),
                        delivery_day=agreement.get("time", 0) if isinstance(agreement, dict) else getattr(agreement, 'time', 0),
                        is_seller=is_seller,
                    )
                    tracker.negotiation_success(partner, {
                        "quantity": agreement.get("quantity", 0) if isinstance(agreement, dict) else getattr(agreement, 'quantity', 0),
                        "price": agreement.get("unit_price", 0) if isinstance(agreement, dict) else getattr(agreement, 'unit_price', 0),
                    })
                except Exception:
                    pass
        agent_class.on_negotiation_success = patched_on_negotiation_success
    
    # 包装 on_negotiation_failure 方法
    if original_on_negotiation_failure:
        def patched_on_negotiation_failure(self, partners, annotation, mechanism, state):
            original_on_negotiation_failure(self, partners, annotation, mechanism, state)
            tracker = self.tracker
            if tracker:
                try:
                    partner = partners[0] if partners else "unknown"
                    tracker.negotiation_failure(partner, "negotiation_ended_without_agreement")
                except Exception:
                    pass
        agent_class.on_negotiation_failure = patched_on_negotiation_failure
    
    # 包装 counter_all 方法 - 记录每一轮协商的详细信息
    if original_counter_all:
        def patched_counter_all(self, offers, states):
            tracker = self.tracker
            responses = original_counter_all(self, offers, states)
            
            # 记录每个报价和响应
            if tracker:
                try:
                    for partner_id, offer in offers.items():
                        state = states.get(partner_id)
                        response = responses.get(partner_id)
                        
                        # 解析报价 (quantity, time, unit_price)
                        offer_qty = offer[0] if offer else 0
                        offer_time = offer[1] if offer and len(offer) > 1 else 0
                        offer_price = offer[2] if offer and len(offer) > 2 else 0
                        
                        # 记录收到的报价
                        tracker.negotiation_offer_received(partner_id, {
                            "quantity": offer_qty,
                            "delivery_day": offer_time,
                            "unit_price": offer_price,
                            "round": state.step if state else 0,
                        })
                        
                        # 记录我们的响应
                        if response:
                            response_type = str(response.response) if response.response else "unknown"
                            counter_offer = response.outcome
                            
                            if response_type == "ResponseType.ACCEPT_OFFER":
                                tracker.negotiation_accept(partner_id, {
                                    "quantity": offer_qty,
                                    "delivery_day": offer_time,
                                    "unit_price": offer_price,
                                }, reason="accepted_in_counter_all")
                            elif response_type == "ResponseType.REJECT_OFFER" and counter_offer:
                                counter_qty = counter_offer[0] if counter_offer else 0
                                counter_time = counter_offer[1] if counter_offer and len(counter_offer) > 1 else 0
                                counter_price = counter_offer[2] if counter_offer and len(counter_offer) > 2 else 0
                                
                                tracker.negotiation_offer_made(partner_id, {
                                    "quantity": counter_qty,
                                    "delivery_day": counter_time,
                                    "unit_price": counter_price,
                                    "round": state.step if state else 0,
                                }, reason="counter_offer")
                            elif response_type == "ResponseType.END_NEGOTIATION":
                                tracker.negotiation_reject(partner_id, {
                                    "quantity": offer_qty,
                                    "delivery_day": offer_time,
                                    "unit_price": offer_price,
                                }, reason="end_negotiation")
                except Exception:
                    pass
            
            return responses
        agent_class.counter_all = patched_counter_all
    
    # 包装 first_proposals 方法 - 记录协商开始和初始报价
    if original_first_proposals:
        def patched_first_proposals(self):
            tracker = self.tracker
            proposals = original_first_proposals(self)
            
            if tracker and proposals:
                try:
                    for partner_id, proposal in proposals.items():
                        if proposal:
                            prop_qty = proposal[0] if proposal else 0
                            prop_time = proposal[1] if proposal and len(proposal) > 1 else 0
                            prop_price = proposal[2] if proposal and len(proposal) > 2 else 0
                            
                            # 判断是买方还是卖方
                            is_seller = hasattr(self.awi, 'is_first_level') and not self.awi.is_first_level
                            
                            # 记录协商开始
                            tracker.negotiation_started(partner_id, {
                                "quantity_range": str(prop_qty),
                                "time_range": str(prop_time),
                                "price_range": str(prop_price),
                            }, is_seller=is_seller)
                            
                            # 记录初始报价
                            tracker.negotiation_offer_made(partner_id, {
                                "quantity": prop_qty,
                                "delivery_day": prop_time,
                                "unit_price": prop_price,
                                "round": 0,
                            }, reason="first_proposal")
                except Exception:
                    pass
            
            return proposals
        agent_class.first_proposals = patched_first_proposals
    
    return agent_class


def inject_tracker_to_agents(agent_classes: List[Type]) -> List[Type]:
    """
    为多个 Agent 类注入 Tracker 功能
    
    Args:
        agent_classes: Agent 类列表
        
    Returns:
        修补后的 Agent 类列表
    """
    if not TRACKER_AVAILABLE:
        print("[Tracker] Warning: scml_analyzer not available, tracking disabled")
        return agent_classes
    
    patched = []
    for cls in agent_classes:
        patched_cls = patch_agent_class(cls)
        patched.append(patched_cls)
    
    return patched


def setup_tracker_for_tournament(log_dir: str, enabled: bool = True):
    """
    为比赛配置 Tracker
    
    Args:
        log_dir: 日志目录
        enabled: 是否启用 Tracker
    """
    if not TRACKER_AVAILABLE:
        return
    
    tracker_dir = os.path.join(log_dir, "tracker_logs")
    os.makedirs(tracker_dir, exist_ok=True)
    
    TrackerConfig.configure(
        log_dir=tracker_dir,
        enabled=enabled,
        console_echo=False,
    )


def save_tracker_data(output_dir: Optional[str] = None):
    """保存所有 Tracker 数据"""
    if not TRACKER_AVAILABLE:
        return
    
    TrackerManager.save_all(output_dir)
