"""
LitaAgent Tracker Mixin

为所有 LitaAgent 提供统一的数据记录功能。

两种使用方式：
1. 继承方式（推荐，支持并行模式）：
   from litaagent_std.tracker_mixin import create_tracked_agent
   TrackedLitaAgentY = create_tracked_agent(LitaAgentY, log_dir="./tracker_logs")
   
2. 动态注入方式（仅支持单进程模式）：
   from litaagent_std.tracker_mixin import inject_tracker_to_agents
   inject_tracker_to_agents([LitaAgentY, LitaAgentYR])

Usage:
    from litaagent_std.tracker_mixin import create_tracked_agent
    
    # 创建带 Tracker 的 Agent 类（支持并行模式）
    TrackedLitaAgentY = create_tracked_agent(LitaAgentY, log_dir="./tracker_logs")
"""

import os
import sys
import hashlib
from typing import List, Type, Dict, Any, Optional, Tuple
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


def _patch_oneshot_adapter_execution_forward() -> None:
    try:
        from scml.oneshot.sysagents import DefaultOneShotAdapter
    except Exception:
        return
    patched = getattr(DefaultOneShotAdapter.on_contract_executed, "_lita_patched", False)
    if patched:
        return
    original = DefaultOneShotAdapter.on_contract_executed

    def _forward_on_contract_executed(self, contract):
        try:
            obj = getattr(self, "_obj", None)
            if obj is not None and hasattr(obj, "on_contract_executed"):
                obj.on_contract_executed(contract)
        except Exception:
            pass
        return original(self, contract)

    _forward_on_contract_executed._lita_patched = True
    DefaultOneShotAdapter.on_contract_executed = _forward_on_contract_executed


def _get_world_id_from_awi(awi: Any) -> str:
    if awi is None:
        return "unknown"
    world = getattr(awi, "_world", None)
    if world is not None:
        world_id = getattr(world, "id", None)
        if world_id is not None:
            return str(world_id)
    return "unknown"


def _world_id_hash(world_id: str, length: int = 8) -> str:
    try:
        return hashlib.sha1(world_id.encode("utf-8")).hexdigest()[:length]
    except Exception:
        return "unknown"


def _compose_tracker_agent_id(agent_id: str, world_id: str) -> str:
    if not world_id or world_id == "unknown":
        return agent_id
    return f"{agent_id}#W{_world_id_hash(world_id)}"


def _coerce_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _extract_issue_bounds(issue: Any) -> Tuple[Optional[float], Optional[float]]:
    if issue is None:
        return None, None
    if isinstance(issue, dict):
        min_val = _coerce_float(
            issue.get("min_price", issue.get("min_value", issue.get("min", issue.get("low"))))
        )
        max_val = _coerce_float(
            issue.get("max_price", issue.get("max_value", issue.get("max", issue.get("high"))))
        )
        if min_val is not None or max_val is not None:
            return min_val, max_val
    for name in ("min_value", "min", "low"):
        if hasattr(issue, name):
            min_val = _coerce_float(getattr(issue, name))
            if min_val is not None:
                break
        min_val = None
    for name in ("max_value", "max", "high"):
        if hasattr(issue, name):
            max_val = _coerce_float(getattr(issue, name))
            if max_val is not None:
                break
        max_val = None
    if min_val is not None or max_val is not None:
        return min_val, max_val
    if isinstance(issue, (list, tuple)) and len(issue) == 2:
        min_val = _coerce_float(issue[0])
        max_val = _coerce_float(issue[1])
        if min_val is not None or max_val is not None:
            return min_val, max_val
    return None, None


def _extract_price_bounds(issues: Any) -> Tuple[Optional[float], Optional[float]]:
    if issues is None:
        return None, None
    if isinstance(issues, dict):
        min_val = _coerce_float(issues.get("min_price"))
        max_val = _coerce_float(issues.get("max_price"))
        if min_val is not None or max_val is not None:
            return min_val, max_val
        if "price_range" in issues:
            min_val, max_val = _extract_issue_bounds(issues.get("price_range"))
            if min_val is not None or max_val is not None:
                return min_val, max_val
        if "issues" in issues:
            min_val, max_val = _extract_price_bounds(issues.get("issues"))
            if min_val is not None or max_val is not None:
                return min_val, max_val
        for key, value in issues.items():
            if "price" in str(key).lower():
                min_val, max_val = _extract_issue_bounds(value)
                if min_val is not None or max_val is not None:
                    return min_val, max_val
    if isinstance(issues, (list, tuple)):
        for item in issues:
            name = getattr(item, "name", None) or getattr(item, "title", None)
            if name and "price" in str(name).lower():
                min_val, max_val = _extract_issue_bounds(item)
                if min_val is not None or max_val is not None:
                    return min_val, max_val
        if len(issues) >= 3:
            return _extract_issue_bounds(issues[2])
    return None, None


def _ensure_tracker_logger(
    agent: Any,
    agent_id: str,
    agent_type: str,
    current_logger: Optional["AgentLogger"],
) -> Optional["AgentLogger"]:
    if not TRACKER_AVAILABLE:
        return None
    world_id = _get_world_id_from_awi(getattr(agent, "awi", None))
    logger_key = (agent_id, agent_type, world_id)
    if current_logger is None or getattr(agent, "_tracker_logger_key", None) != logger_key:
        logger_agent_id = _compose_tracker_agent_id(agent_id, world_id)
        current_logger = TrackerManager.get_logger(logger_agent_id, agent_type)
        setattr(agent, "_tracker_logger_key", logger_key)
    if current_logger is not None:
        current_logger.world_id = world_id
    return current_logger


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
        
        agent_id = getattr(self, 'id', 'unknown')
        agent_type = type(self).__name__
        self._tracker_logger = _ensure_tracker_logger(
            self,
            agent_id=str(agent_id),
            agent_type=str(agent_type),
            current_logger=self._tracker_logger,
        )
        
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
            mechanism_id = data.get("mechanism_id")
            negotiator_id = data.get("negotiator_id")
            buyer = data.get("buyer")
            seller = data.get("seller")
            is_seller = data.get("is_seller")
            role = data.get("role")
            if buyer is None and seller is None and is_seller is not None:
                buyer, seller = _infer_buyer_seller_from_role(getattr(self, "id", None), partner, bool(is_seller))
            if role is None and is_seller is not None:
                role = "seller" if is_seller else "buyer"

            if event == "started":
                self.tracker.negotiation_started(
                    partner,
                    data.get("issues", {}),
                    data.get("is_seller", False),
                    mechanism_id=mechanism_id,
                    negotiator_id=negotiator_id,
                    buyer=buyer,
                    seller=seller,
                    role=role,
                )
            elif event == "offer_made":
                self.tracker.negotiation_offer_made(
                    partner,
                    data.get("offer", {}),
                    data.get("reason", ""),
                    mechanism_id=mechanism_id,
                    negotiator_id=negotiator_id,
                    buyer=buyer,
                    seller=seller,
                    role=role,
                )
            elif event == "offer_received":
                self.tracker.negotiation_offer_received(
                    partner,
                    data.get("offer", {}),
                    mechanism_id=mechanism_id,
                    negotiator_id=negotiator_id,
                    buyer=buyer,
                    seller=seller,
                    role=role,
                )
            elif event == "accept":
                self.tracker.negotiation_accept(
                    partner,
                    data.get("offer", {}),
                    data.get("reason", ""),
                    mechanism_id=mechanism_id,
                    negotiator_id=negotiator_id,
                    buyer=buyer,
                    seller=seller,
                    role=role,
                )
            elif event == "reject":
                self.tracker.negotiation_reject(
                    partner,
                    data.get("offer", {}),
                    data.get("reason", ""),
                    mechanism_id=mechanism_id,
                    negotiator_id=negotiator_id,
                    buyer=buyer,
                    seller=seller,
                    role=role,
                )
            elif event == "success":
                self.tracker.negotiation_success(
                    partner,
                    data.get("agreement", {}),
                    mechanism_id=mechanism_id,
                    negotiator_id=negotiator_id,
                    buyer=buyer,
                    seller=seller,
                    role=role,
                )
            elif event == "failure":
                self.tracker.negotiation_failure(
                    partner,
                    data.get("reason", ""),
                    mechanism_id=mechanism_id,
                    negotiator_id=negotiator_id,
                    buyer=buyer,
                    seller=seller,
                    role=role,
                )
    
    def track_contract(
        self,
        contract_id: str,
        partner: str,
        quantity: int,
        price: float,
        delivery_day: int,
        is_seller: bool,
        buyer: Optional[str] = None,
        seller: Optional[str] = None,
    ):
        """记录合同签署"""
        if self.tracker:
            self.tracker.contract_signed(
                contract_id,
                partner,
                quantity,
                price,
                delivery_day,
                is_seller,
                buyer=buyer,
                seller=seller,
            )
    
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


def _build_agent_identity(agent: Any, base_class: Optional[Type] = None) -> Dict[str, Any]:
    """构建可用于匹配的代理身份信息."""
    cls = type(agent)
    if base_class is None:
        bases = getattr(cls, "__bases__", None) or ()
        base_class = bases[0] if bases else cls
    
    identity = {
        "agent_type_raw": cls.__name__,
        "agent_type_raw_full": f"{cls.__module__}.{cls.__name__}",
        "agent_type_raw_qualname": getattr(cls, "__qualname__", cls.__name__),
        "agent_type_base": base_class.__name__,
        "agent_type_base_full": f"{base_class.__module__}.{base_class.__name__}",
        "agent_type_base_qualname": getattr(base_class, "__qualname__", base_class.__name__),
    }
    
    name_fields = [
        ("agent_display_name", "name"),
        ("agent_registry_name", "registry_name"),
        ("agent_short_name", "short_name"),
        ("agent_name", "agent_name"),
    ]
    for out_key, attr in name_fields:
        value = getattr(agent, attr, None)
        if value:
            identity[out_key] = str(value)
    
    return identity


def _extract_contract_terms(contract) -> Tuple[int, float, int, Optional[str], Optional[str], Optional[int]]:
    """提取合约字段用于 tracking/commitments."""
    agreement = getattr(contract, 'agreement', contract)
    if isinstance(agreement, dict):
        qty = int(agreement.get('quantity', 0))
        price = float(agreement.get('unit_price', agreement.get('price', 0)))
        delivery = int(agreement.get('time', agreement.get('delivery_day', 0)))
        buyer = agreement.get('buyer', None)
        seller = agreement.get('seller', None)
    else:
        qty = int(getattr(agreement, 'quantity', 0))
        price = float(getattr(agreement, 'unit_price', getattr(agreement, 'price', 0)))
        delivery = int(getattr(agreement, 'time', getattr(agreement, 'delivery_day', 0)))
        buyer = getattr(contract, 'buyer', None)
        seller = getattr(contract, 'seller', None)

    annotation = getattr(contract, 'annotation', {}) or {}
    if isinstance(annotation, dict):
        if buyer is None:
            buyer = annotation.get('buyer', None)
        if seller is None:
            seller = annotation.get('seller', None)
    if buyer is None:
        buyer = getattr(contract, 'buyer', None)
    if seller is None:
        seller = getattr(contract, 'seller', None)

    product_index = annotation.get('product', None) if isinstance(annotation, dict) else None
    return qty, price, delivery, buyer, seller, product_index


def _infer_buyer_seller_from_role(
    agent_id: Optional[str],
    partner_id: Optional[str],
    is_seller: bool,
) -> Tuple[Optional[str], Optional[str]]:
    if not agent_id or not partner_id:
        return None, None
    if is_seller:
        return partner_id, agent_id
    return agent_id, partner_id


def _infer_contract_parties(
    agent: Any,
    contract: Any,
) -> Tuple[int, float, int, Optional[str], Optional[str], Optional[str], Optional[int]]:
    qty, price, delivery, buyer, seller, product_index = _extract_contract_terms(contract)
    agent_id = getattr(agent, "id", None)

    partner = None
    partners = getattr(contract, "partners", None)
    if partners:
        try:
            for p in list(partners):
                if agent_id and p != agent_id:
                    partner = p
                    break
        except Exception:
            pass

    annotation = getattr(contract, "annotation", {}) or {}
    if partner is None and isinstance(annotation, dict):
        ann_partners = annotation.get("partners", None)
        if ann_partners:
            try:
                for p in list(ann_partners):
                    if agent_id and p != agent_id:
                        partner = p
                        break
            except Exception:
                pass

    if agent_id and partner and (buyer is None or seller is None):
        my_input_product = getattr(agent.awi, "my_input_product", None)
        awi_is_first_level = getattr(agent.awi, "is_first_level", False)
        is_buying = _infer_is_buying_for_contract(
            agent_id,
            my_input_product,
            awi_is_first_level,
            buyer,
            product_index,
        )
        if is_buying:
            buyer = buyer or agent_id
            seller = seller or partner
        else:
            seller = seller or agent_id
            buyer = buyer or partner

    return qty, price, delivery, buyer, seller, partner, product_index


def _infer_is_buying_for_contract(
    agent_id: Optional[str],
    my_input_product: int,
    awi_is_first_level: bool,
    buyer: Optional[str],
    product_index: Optional[int],
) -> bool:
    """判断该合约对我方是买入还是卖出."""
    is_buying = None
    if buyer is not None and agent_id:
        is_buying = (agent_id == buyer)
    if is_buying is None and product_index is not None:
        is_buying = (product_index == my_input_product)
    if is_buying is None:
        is_buying = bool(awi_is_first_level)
    return is_buying


def _sum_mapping_values(mapping: Any) -> float:
    if not isinstance(mapping, dict):
        return 0.0
    total = 0.0
    for v in mapping.values():
        if isinstance(v, (int, float)):
            total += float(v)
    return total


def _fill_future_values(
    future_map: Any,
    current_step: int,
    target: List[float],
) -> None:
    if not isinstance(future_map, dict):
        return
    for step, per_partner in future_map.items():
        try:
            delta = int(step) - current_step
        except Exception:
            continue
        if 0 <= delta < len(target):
            target[delta] += _sum_mapping_values(per_partner)


def _build_commitments_from_future(
    awi: Any,
    current_step: int,
    max_len: int,
) -> Tuple[List[float], List[float], List[float], List[float], bool]:
    """基于 AWI 的 future_* 视图重建 commitments（含当前日）."""
    Q_in = [0.0] * max_len
    Q_out = [0.0] * max_len
    Payables = [0.0] * max_len
    Receivables = [0.0] * max_len

    supplies = getattr(awi, 'supplies', None)
    sales = getattr(awi, 'sales', None)
    future_supplies = getattr(awi, 'future_supplies', None)
    future_sales = getattr(awi, 'future_sales', None)
    supplies_cost = getattr(awi, 'supplies_cost', None)
    sales_cost = getattr(awi, 'sales_cost', None)
    future_supplies_cost = getattr(awi, 'future_supplies_cost', None)
    future_sales_cost = getattr(awi, 'future_sales_cost', None)

    has_any = any(x is not None for x in (
        supplies, sales, future_supplies, future_sales, supplies_cost, sales_cost
    ))
    if not has_any:
        return Q_in, Q_out, Payables, Receivables, False

    # 当前日 (delta=0)
    Q_in[0] = _sum_mapping_values(supplies)
    Q_out[0] = _sum_mapping_values(sales)
    Payables[0] = _sum_mapping_values(supplies_cost)
    Receivables[0] = _sum_mapping_values(sales_cost)

    # 未来日 (delta>0)
    _fill_future_values(future_supplies, current_step, Q_in)
    _fill_future_values(future_sales, current_step, Q_out)
    _fill_future_values(future_supplies_cost, current_step, Payables)
    _fill_future_values(future_sales_cost, current_step, Receivables)

    has_nonzero = any(Q_in) or any(Q_out) or any(Payables) or any(Receivables)
    return Q_in, Q_out, Payables, Receivables, has_nonzero


def _get_initial_balance(awi: Any, agent_id: Optional[str]) -> float:
    world = getattr(awi, "_world", None)
    if world is not None:
        initial_balances = getattr(world, "initial_balances", None)
        if isinstance(initial_balances, dict) and agent_id in initial_balances:
            try:
                return float(initial_balances[agent_id])
            except Exception:
                pass
    value = getattr(awi, "initial_balance", None)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _extract_offers_snapshot(awi, current_step: int) -> Dict[str, Any]:
    """提取当前活跃谈判的报价快照，用于离线重建 6-8 通道.
    
    数据结构:
        {
            'buy': [(delivery_time, quantity, unit_price), ...],  # 买入谈判中的活跃报价
            'sell': [(delivery_time, quantity, unit_price), ...], # 卖出谈判中的活跃报价
        }
    
    Args:
        awi: Agent World Interface
        current_step: 当前仿真步骤
        
    Returns:
        活跃报价快照字典
    """
    snapshot = {'buy': [], 'sell': []}
    
    try:
        # 买入谈判（我方作为买方，收到卖方报价）
        current_buy_offers = getattr(awi, 'current_buy_offers', None)
        if current_buy_offers and isinstance(current_buy_offers, dict):
            for partner_id, offer in current_buy_offers.items():
                if offer is None:
                    continue
                try:
                    # Outcome 格式: (quantity, time, unit_price)
                    quantity = float(offer[0])
                    delivery_time = int(offer[1])
                    unit_price = float(offer[2])
                    snapshot['buy'].append([delivery_time, quantity, unit_price])
                except (IndexError, TypeError, ValueError):
                    continue
        
        # 卖出谈判（我方作为卖方，收到买方报价）
        current_sell_offers = getattr(awi, 'current_sell_offers', None)
        if current_sell_offers and isinstance(current_sell_offers, dict):
            for partner_id, offer in current_sell_offers.items():
                if offer is None:
                    continue
                try:
                    quantity = float(offer[0])
                    delivery_time = int(offer[1])
                    unit_price = float(offer[2])
                    snapshot['sell'].append([delivery_time, quantity, unit_price])
                except (IndexError, TypeError, ValueError):
                    continue
    except Exception:
        pass
    
    return snapshot


def patch_agent_class(agent_class: Type, captured_log_dir: Optional[str] = None) -> Type:
    """
    动态修补 Agent 类以添加 Tracker 功能
    
    这种方法不需要修改原有代码，通过方法包装实现自动记录。
    
    Args:
        agent_class: 要修补的 Agent 类
        captured_log_dir: 捕获的日志目录路径，用于并行模式
    """
    if not TRACKER_AVAILABLE:
        return agent_class
    
    # 在类上保存 log_dir，这样可以在 pickle/unpickle 后仍然可用
    agent_class._tracker_log_dir = captured_log_dir
    
    # 保存原始方法
    original_init = agent_class.init if hasattr(agent_class, 'init') else None
    original_before_step = agent_class.before_step if hasattr(agent_class, 'before_step') else None
    original_step = agent_class.step if hasattr(agent_class, 'step') else None
    original_on_negotiation_success = agent_class.on_negotiation_success if hasattr(agent_class, 'on_negotiation_success') else None
    original_on_negotiation_failure = agent_class.on_negotiation_failure if hasattr(agent_class, 'on_negotiation_failure') else None
    original_on_contract_executed = agent_class.on_contract_executed if hasattr(agent_class, 'on_contract_executed') else None
    original_counter_all = agent_class.counter_all if hasattr(agent_class, 'counter_all') else None
    original_first_proposals = agent_class.first_proposals if hasattr(agent_class, 'first_proposals') else None
    
    # 添加 tracker 属性
    agent_class._tracker_logger = None
    agent_class._tracker_enabled = True
    
    def get_tracker(self) -> Optional[AgentLogger]:
        if not self._tracker_enabled:
            return None
        agent_id = getattr(self, 'id', 'unknown')
        agent_type = type(self).__name__
        self._tracker_logger = _ensure_tracker_logger(
            self,
            agent_id=str(agent_id),
            agent_type=str(agent_type),
            current_logger=self._tracker_logger,
        )
        return self._tracker_logger
    
    agent_class.tracker = property(get_tracker)

    def _get_nmi(self, partner_id: str):
        try:
            return self.get_nmi(partner_id)
        except Exception:
            return None

    def _get_mechanism_id(self, partner_id: str, state=None) -> Optional[str]:
        nmi = _get_nmi(self, partner_id)
        mech_id = getattr(nmi, "mechanism_id", None) if nmi is not None else None
        if not mech_id and state is not None:
            mech_id = getattr(state, "mechanism_id", None) or getattr(state, "id", None)
        return str(mech_id) if mech_id else None

    def _infer_is_seller(self, partner_id: str) -> bool:
        """尽量用 nmi.annotation 判断买卖方向，避免硬猜污染数据。"""
        nmi = _get_nmi(self, partner_id)
        annotation = getattr(nmi, "annotation", {}) or {}

        product_index = annotation.get("product", None)
        if hasattr(product_index, "__len__") and not isinstance(product_index, (str, int)):
            product_index = product_index[0] if len(product_index) > 0 else None

        my_input_product = getattr(self.awi, "my_input_product", None)
        my_output_product = getattr(self.awi, "my_output_product", None)

        if product_index is not None:
            if my_output_product is not None and product_index == my_output_product:
                return True
            if my_input_product is not None and product_index == my_input_product:
                return False

        seller = annotation.get("seller", None)
        buyer = annotation.get("buyer", None)
        if seller is not None and seller == getattr(self, "id", None):
            return True
        if buyer is not None and buyer == getattr(self, "id", None):
            return False

        if getattr(self.awi, "is_first_level", False):
            return True
        if getattr(self.awi, "is_last_level", False):
            return False
        return False

    def _ensure_negotiation_started(self, partner_id: str, state=None, issues: Optional[Dict[str, Any]] = None) -> Optional[str]:
        tracker = self.tracker
        if not tracker:
            return None
        mechanism_id = _get_mechanism_id(self, partner_id, state)
        sim_step = int(getattr(self.awi, "current_step", 0) or 0)
        if mechanism_id:
            seen_key = (mechanism_id, partner_id)
        else:
            seen_key = (partner_id, sim_step, "no_mechanism")
        seen = getattr(self, "_tracker_seen_negotiations", None)
        if seen is None:
            seen = set()
            setattr(self, "_tracker_seen_negotiations", seen)
        if seen_key in seen:
            return mechanism_id
        is_seller = _infer_is_seller(self, partner_id)
        issues_payload = issues
        min_price = None
        max_price = None
        if issues_payload is None:
            issues_payload = {}
            nmi = _get_nmi(self, partner_id)
            if nmi is not None:
                try:
                    issues_payload = {"issues": getattr(nmi, "issues", None)}
                except Exception:
                    issues_payload = {}
        else:
            if not isinstance(issues_payload, dict):
                issues_payload = {"issues": issues_payload}
            min_price = _coerce_float(issues_payload.get("min_price")) if isinstance(issues_payload, dict) else None
            max_price = _coerce_float(issues_payload.get("max_price")) if isinstance(issues_payload, dict) else None

        if min_price is None or max_price is None:
            nmi = _get_nmi(self, partner_id)
            if nmi is not None:
                nmi_issues = getattr(nmi, "issues", None)
                nmi_min, nmi_max = _extract_price_bounds(nmi_issues)
                if min_price is None:
                    min_price = nmi_min
                if max_price is None:
                    max_price = nmi_max

        if isinstance(issues_payload, dict):
            if min_price is not None:
                issues_payload["min_price"] = float(min_price)
            if max_price is not None:
                issues_payload["max_price"] = float(max_price)
        buyer, seller = _infer_buyer_seller_from_role(getattr(self, "id", None), partner_id, is_seller)
        tracker.negotiation_started(
            partner_id,
            issues_payload,
            is_seller=is_seller,
            mechanism_id=mechanism_id,
            negotiator_id=partner_id,
            buyer=buyer,
            seller=seller,
            role="seller" if is_seller else "buyer",
        )
        seen.add(seen_key)
        return mechanism_id

    def _pack_aop_offer(offer):
        if offer is None:
            return None
        try:
            qty, delivery_day, price = offer
        except Exception:
            return None
        return {
            "quantity": int(qty),
            "unit_price": float(price),
            "delivery_day": int(delivery_day),
        }

    def _normalize_response_type(resp):
        if resp is None:
            return "unknown"
        name = getattr(resp, "name", None)
        if name:
            return name
        if isinstance(resp, (int, float)):
            value = int(resp)
        else:
            text = str(resp)
            if text.startswith("ResponseType."):
                return text.split("ResponseType.", 1)[1]
            try:
                value = int(text)
            except Exception:
                value = None
        if value is not None:
            if value == 0:
                return "ACCEPT_OFFER"
            if value == 1:
                return "REJECT_OFFER"
            if value == 2:
                return "END_NEGOTIATION"
            if value == 3:
                return "NO_RESPONSE"
            if value == 4:
                return "WAIT"
        text = str(resp)
        if "ACCEPT" in text:
            return "ACCEPT_OFFER"
        if "REJECT" in text:
            return "REJECT_OFFER"
        if "END" in text:
            return "END_NEGOTIATION"
        return "unknown"
    
    # 包装 init 方法
    if original_init:
        def patched_init(self):
            original_init(self)
            tracker = self.tracker
            if tracker:
                identity = _build_agent_identity(self)
                tracker.custom("agent_initialized", 
                    n_steps=getattr(self.awi, 'n_steps', None),
                    n_lines=getattr(self.awi, 'n_lines', None),
                    level=getattr(self.awi, 'level', None),
                    **identity,
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
                    needed_supplies_raw = getattr(awi, 'needed_supplies', 0) or 0
                    needed_sales_raw = getattr(awi, 'needed_sales', 0) or 0
                    needed_supplies = max(0, int(needed_supplies_raw))
                    needed_sales = max(0, int(needed_sales_raw))
                    
                    # 已签约数量
                    total_supplies = getattr(awi, 'total_supplies', 0) or 0
                    total_sales = getattr(awi, 'total_sales', 0) or 0
                    
                    # 成本和惩罚参数
                    disposal_cost = getattr(awi, 'current_disposal_cost', 0) or 0
                    shortfall_penalty = getattr(awi, 'current_shortfall_penalty', 0) or 0
                    storage_cost = getattr(awi, 'current_storage_cost', 0) or 0
                    
                    # 生产能力
                    n_lines = getattr(awi, 'n_lines', 0) or 0
                    
                    # === HRL-XF 训练所需字段 ===
                    # 市场价格（用于计算 L1 baseline）
                    trading_prices = getattr(awi, 'trading_prices', None)
                    if trading_prices is not None:
                        trading_prices_list = list(trading_prices) if hasattr(trading_prices, '__iter__') else []
                    else:
                        trading_prices_list = []
                    
                    # 输入/输出产品索引
                    my_input_product = getattr(awi, 'my_input_product', 0) or 0
                    my_output_product = getattr(awi, 'my_output_product', 1) or 1
                    
                    # Spot 价格（从 trading_prices 提取）
                    spot_price_in = trading_prices_list[my_input_product] if my_input_product < len(trading_prices_list) else 10.0
                    spot_price_out = trading_prices_list[my_output_product] if my_output_product < len(trading_prices_list) else 20.0
                    
                    # 真实库存（Standard Track 有持久库存）
                    current_inventory = getattr(awi, 'current_inventory', None)
                    if current_inventory is not None:
                        if hasattr(current_inventory, '__iter__') and not isinstance(current_inventory, (str, dict)):
                            inventory_input = int(current_inventory[0]) if len(current_inventory) > 0 else 0
                            inventory_output = int(current_inventory[-1]) if len(current_inventory) > 0 else 0
                        elif isinstance(current_inventory, dict):
                            inventory_input = int(current_inventory.get('input', 0))
                            inventory_output = int(current_inventory.get('output', 0))
                        else:
                            inventory_input = int(current_inventory)
                            inventory_output = 0
                    else:
                        inventory_input = exo_input_qty
                        inventory_output = exo_output_qty
                    
                    # 合同承诺（用于计算 Q_safe 和 B_free）
                    # 尝试从 awi 获取未来承诺
                    n_steps = getattr(awi, 'n_steps', 100) or 100
                    current_step = getattr(awi, 'current_step', 0) or 0
                    remaining_steps = n_steps - current_step
                    
                    # 尝试基于 AWI future_* 重建 commitments
                    max_len = min(remaining_steps + 1, 50)
                    Q_in, Q_out, Payables, Receivables, has_future = _build_commitments_from_future(
                        awi, current_step, max_len
                    )
                    
                    # 记录基本库存状态
                    tracker.inventory_state(
                        raw_material=inventory_input,
                        product=inventory_output,
                        balance=balance,
                    )
                    
                    # === HRL-XF 6-8通道：获取活跃谈判快照 ===
                    offers_snapshot = _extract_offers_snapshot(awi, current_step)
                    
                    # 记录完整的每日状态（包含 HRL-XF 训练所需字段）
                    tracker.custom("daily_status",
                        # 财务状态
                        score=score,
                        balance=balance,
                        initial_balance=_get_initial_balance(awi, getattr(self, "id", None)),
                        # 外生合同（系统分配）
                        exo_input_qty=exo_input_qty,
                        exo_input_price=exo_input_price,
                        exo_output_qty=exo_output_qty,
                        exo_output_price=exo_output_price,
                        # 协商需求
                        needed_supplies=needed_supplies,
                        needed_sales=needed_sales,
                        needed_supplies_raw=needed_supplies_raw,
                        needed_sales_raw=needed_sales_raw,
                        # 已签约
                        total_supplies=total_supplies,
                        total_sales=total_sales,
                        # 成本参数
                        disposal_cost=disposal_cost,
                        shortfall_penalty=shortfall_penalty,
                        storage_cost=storage_cost,
                        # 生产能力
                        n_lines=n_lines,
                        n_steps=n_steps,
                        current_step=current_step,
                        # === HRL-XF 新增字段 ===
                        # 市场价格
                        spot_price_in=float(spot_price_in),
                        spot_price_out=float(spot_price_out),
                        trading_prices=trading_prices_list,
                        # 真实库存
                        inventory_input=inventory_input,
                        inventory_output=inventory_output,
                        # 合同承诺（用于离线计算 L1 baseline）
                        # H=40 天规划视界，记录 H+1=41 个时间点
                        commitments={
                            'Q_in': Q_in[:41],
                            'Q_out': Q_out[:41],
                            'Payables': Payables[:41],
                            'Receivables': Receivables[:41],
                        },
                        # 活跃谈判快照（用于离线重建 6-8 通道）
                        offers_snapshot=offers_snapshot,
                        # 产品索引
                        my_input_product=my_input_product,
                        my_output_product=my_output_product,
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
                    qty, price, time, buyer, seller, partner, product_index = _infer_contract_parties(self, contract)
                    if not partner:
                        partner = "unknown"

                    # 尽量使用 annotation/product/buyer 正确判断买卖方向（中间层不能用 is_first_level 硬猜）
                    my_output_product = getattr(self.awi, 'my_output_product', 1)
                    is_seller = None
                    if seller is not None and self.id == seller:
                        is_seller = True
                    elif buyer is not None and self.id == buyer:
                        is_seller = False
                    elif product_index is not None:
                        is_seller = (product_index == my_output_product)
                    else:
                        is_seller = not getattr(self.awi, 'is_first_level', True)

                    tracker.contract_signed(
                        contract_id=str(contract.id),
                        partner=partner,
                        quantity=qty,
                        price=price,
                        delivery_day=time,
                        is_seller=is_seller,
                        buyer=buyer,
                        seller=seller,
                    )
                    tracker.negotiation_success(
                        partner,
                        {
                            "quantity": qty,
                            "price": price,
                            "time": time,
                            "buyer": buyer,
                            "seller": seller,
                        },
                        buyer=buyer,
                        seller=seller,
                        role="seller" if is_seller else "buyer",
                    )
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
                    is_seller = _infer_is_seller(self, partner) if partner != "unknown" else False
                    buyer, seller = _infer_buyer_seller_from_role(getattr(self, "id", None), partner, is_seller)
                    tracker.negotiation_failure(
                        partner,
                        "negotiation_ended_without_agreement",
                        buyer=buyer,
                        seller=seller,
                        role="seller" if is_seller else "buyer",
                    )
                except Exception:
                    pass
        agent_class.on_negotiation_failure = patched_on_negotiation_failure

    # 包装 on_contract_executed 方法
    if original_on_contract_executed:
        def patched_on_contract_executed(self, contract):
            original_on_contract_executed(self, contract)
            tracker = self.tracker
            if tracker:
                try:
                    qty, price, delivery, buyer, seller, partner, product_index = _infer_contract_parties(self, contract)
                    current_step = int(getattr(self.awi, "current_step", 0) or 0)
                    my_output_product = getattr(self.awi, "my_output_product", 1)
                    is_seller = None
                    if seller is not None and self.id == seller:
                        is_seller = True
                    elif buyer is not None and self.id == buyer:
                        is_seller = False
                    elif product_index is not None:
                        is_seller = (product_index == my_output_product)
                    else:
                        is_seller = not getattr(self.awi, "is_first_level", True)

                    if is_seller:
                        sales = getattr(self, "_tracker_exec_sales_by_day", {})
                        sales[current_step] = int(sales.get(current_step, 0)) + int(qty)
                        self._tracker_exec_sales_by_day = sales
                    else:
                        supplies = getattr(self, "_tracker_exec_supplies_by_day", {})
                        supplies[current_step] = int(supplies.get(current_step, 0)) + int(qty)
                        self._tracker_exec_supplies_by_day = supplies

                    tracker.contract_executed(
                        contract_id=str(getattr(contract, "id", "")),
                        quantity=qty,
                        partner=partner,
                        price=price,
                        delivery_day=delivery,
                        buyer=buyer,
                        seller=seller,
                        role="seller" if is_seller else "buyer",
                    )
                except Exception:
                    pass
        agent_class.on_contract_executed = patched_on_contract_executed
    
    # 包装 step 方法 - 在最后一步自动保存 Tracker 数据（支持并行模式）
    # 注意：使用环境变量传递 log_dir，因为子进程会继承父进程的环境变量
    def patched_step(self):
        if original_step:
            original_step(self)
        
        # 检查是否是最后一步，如果是则保存 tracker 数据
        tracker = self.tracker
        if tracker:
            try:
                current_step = getattr(self.awi, 'current_step', 0)
                n_steps = getattr(self.awi, 'n_steps', 0)
                
                # 在最后一步保存数据
                if current_step >= n_steps - 1:
                    # 从 awi._world 获取 world_id
                    world_id = 'unknown'
                    if hasattr(self.awi, '_world') and self.awi._world:
                        world_id = getattr(self.awi._world, 'id', 'unknown') or 'unknown'
                    
                    # 更新 tracker 的 world_id
                    tracker.world_id = world_id
                    
                    # 使用环境变量获取 log_dir（子进程会继承父进程的环境变量）
                    log_dir = os.environ.get('SCML_TRACKER_LOG_DIR', None)
                    if log_dir is None:
                        # 回退到类属性
                        log_dir = getattr(type(self), '_tracker_log_dir', None)
                    if log_dir is None:
                        config = TrackerConfig.get()
                        log_dir = config.log_dir
                    
                    if log_dir:
                        # 确保目录存在
                        os.makedirs(log_dir, exist_ok=True)
                        
                        # 使用 world_id hash 避免并行模式下的文件名冲突
                        safe_agent_id = tracker.agent_id.replace("@", "_at_").replace("/", "_")
                        raw_world_id = str(world_id)
                        safe_world_id = raw_world_id.replace("/", "_").replace("\\", "_")
                        short_world_id = safe_world_id[:40]
                        world_hash = hashlib.sha1(raw_world_id.encode("utf-8")).hexdigest()[:12]
                        filename = f"agent_{safe_agent_id}_world_{short_world_id}_{world_hash}.json"
                        filepath = os.path.join(log_dir, filename)
                        tracker.save(filepath)
            except Exception as e:
                # 写到文件以便调试
                import traceback
                with open('./tracker_debug_error.txt', 'a') as f:
                    f.write(f"Error in patched_step: {e}\n")
                    f.write(traceback.format_exc())
                    f.write("\n---\n")
    
    agent_class.step = patched_step
    
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
                        mechanism_id = _ensure_negotiation_started(self, partner_id, state=state)
                        is_seller = _infer_is_seller(self, partner_id)
                        buyer, seller = _infer_buyer_seller_from_role(getattr(self, "id", None), partner_id, is_seller)
                        role = "seller" if is_seller else "buyer"
                        
                        # 解析报价 (quantity, time, unit_price)
                        offer_qty = offer[0] if offer else 0
                        offer_time = offer[1] if offer and len(offer) > 1 else 0
                        offer_price = offer[2] if offer and len(offer) > 2 else 0
                        
                        # 记录收到的报价
                        tracker.negotiation_offer_received(
                            partner_id,
                            {
                                "quantity": offer_qty,
                                "delivery_day": offer_time,
                                "unit_price": offer_price,
                                "round": state.step if state else 0,
                            },
                            mechanism_id=mechanism_id,
                            negotiator_id=partner_id,
                            buyer=buyer,
                            seller=seller,
                            role=role,
                        )
                        
                        # 记录我们的响应
                        if response:
                            response_type = _normalize_response_type(response.response)
                            counter_offer = response.outcome
                            
                            if response_type == "ACCEPT_OFFER":
                                tracker.negotiation_accept(
                                    partner_id,
                                    {
                                        "quantity": offer_qty,
                                        "delivery_day": offer_time,
                                        "unit_price": offer_price,
                                    },
                                    reason="accepted_in_counter_all",
                                    mechanism_id=mechanism_id,
                                    negotiator_id=partner_id,
                                    buyer=buyer,
                                    seller=seller,
                                    role=role,
                                )
                            elif response_type == "REJECT_OFFER" and counter_offer:
                                counter_qty = counter_offer[0] if counter_offer else 0
                                counter_time = counter_offer[1] if counter_offer and len(counter_offer) > 1 else 0
                                counter_price = counter_offer[2] if counter_offer and len(counter_offer) > 2 else 0
                                
                                tracker.negotiation_offer_made(
                                    partner_id,
                                    {
                                        "quantity": counter_qty,
                                        "delivery_day": counter_time,
                                        "unit_price": counter_price,
                                        "round": state.step if state else 0,
                                    },
                                    reason="counter_offer",
                                    mechanism_id=mechanism_id,
                                    negotiator_id=partner_id,
                                    buyer=buyer,
                                    seller=seller,
                                    role=role,
                                )
                            elif response_type == "END_NEGOTIATION":
                                tracker.negotiation_reject(
                                    partner_id,
                                    {
                                        "quantity": offer_qty,
                                        "delivery_day": offer_time,
                                        "unit_price": offer_price,
                                    },
                                    reason="end_negotiation",
                                    mechanism_id=mechanism_id,
                                    negotiator_id=partner_id,
                                    buyer=buyer,
                                    seller=seller,
                                    role=role,
                                )
                    
                    # AOP 动作记录（覆盖 states.keys()，补齐隐式 END）
                    for partner_id, state in states.items():
                        response = responses.get(partner_id)
                        current_offer = getattr(state, "current_offer", None) if state else None
                        is_seller = _infer_is_seller(self, partner_id)
                        round_idx = int(getattr(state, "step", 0) or 0)
                        sim_step = int(getattr(self.awi, "current_step", 0) or 0)
                        mechanism_id = _get_mechanism_id(self, partner_id, state)

                        action_op = "END"
                        response_type = "END_NEGOTIATION"
                        offer_payload = None
                        reason = "omit_in_counter_all"

                        if response is not None:
                            response_type = _normalize_response_type(response.response)
                            reason = "counter_all"
                            if response_type == "ACCEPT_OFFER":
                                action_op = "ACCEPT"
                                offer_payload = _pack_aop_offer(current_offer)
                            elif response_type == "REJECT_OFFER":
                                counter_offer = response.outcome
                                if counter_offer is not None:
                                    action_op = "REJECT"
                                    offer_payload = _pack_aop_offer(counter_offer)
                                else:
                                    action_op = "END"
                            elif response_type == "END_NEGOTIATION":
                                action_op = "END"

                        tracker.custom(
                            "aop_action",
                            sim_step=sim_step,
                            negotiator_id=partner_id,
                            mechanism_id=mechanism_id,
                            partner=partner_id,
                            role="seller" if is_seller else "buyer",
                            round=round_idx,
                            action_op=action_op,
                            response_type=response_type,
                            offer=offer_payload,
                            reason=reason,
                        )
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
                    def _infer_is_seller(negotiator_id: str) -> bool:
                        """尽量用 nmi.annotation 判断买卖方向，避免用层级硬猜污染数据。"""
                        nmi = None
                        try:
                            nmi = self.get_nmi(negotiator_id)
                        except Exception:
                            nmi = None

                        annotation = getattr(nmi, "annotation", {}) or {}

                        product_index = annotation.get("product", None)
                        if hasattr(product_index, "__len__") and not isinstance(product_index, (str, int)):
                            product_index = product_index[0] if len(product_index) > 0 else None

                        my_input_product = getattr(self.awi, "my_input_product", None)
                        my_output_product = getattr(self.awi, "my_output_product", None)

                        if product_index is not None:
                            if my_output_product is not None and product_index == my_output_product:
                                return True
                            if my_input_product is not None and product_index == my_input_product:
                                return False

                        seller = annotation.get("seller", None)
                        buyer = annotation.get("buyer", None)
                        if seller is not None and seller == getattr(self, "id", None):
                            return True
                        if buyer is not None and buyer == getattr(self, "id", None):
                            return False

                        # 仅在无法判断时才回退到供应链端点（中间层不适用）
                        if getattr(self.awi, "is_first_level", False):
                            return True
                        if getattr(self.awi, "is_last_level", False):
                            return False
                        return False

                    for partner_id, proposal in proposals.items():
                        if proposal:
                            prop_qty = proposal[0] if proposal else 0
                            prop_time = proposal[1] if proposal and len(proposal) > 1 else 0
                            prop_price = proposal[2] if proposal and len(proposal) > 2 else 0
                            is_seller = _infer_is_seller(partner_id)
                            buyer, seller = _infer_buyer_seller_from_role(getattr(self, "id", None), partner_id, is_seller)
                            role = "seller" if is_seller else "buyer"

                            issues_payload = {
                                "quantity_range": str(prop_qty),
                                "time_range": str(prop_time),
                                "price_range": str(prop_price),
                            }
                            mechanism_id = _ensure_negotiation_started(
                                self,
                                partner_id,
                                issues=issues_payload,
                            )

                            # 记录初始报价
                            tracker.negotiation_offer_made(
                                partner_id,
                                {
                                    "quantity": prop_qty,
                                    "delivery_day": prop_time,
                                    "unit_price": prop_price,
                                    "round": 0,
                                },
                                reason="first_proposal",
                                mechanism_id=mechanism_id,
                                negotiator_id=partner_id,
                                buyer=buyer,
                                seller=seller,
                                role=role,
                            )

                            tracker.custom(
                                "aop_action",
                                sim_step=int(getattr(self.awi, "current_step", 0) or 0),
                                negotiator_id=partner_id,
                                mechanism_id=mechanism_id,
                                partner=partner_id,
                                role=role,
                                round=0,
                                action_op="REJECT",
                                response_type="REJECT_OFFER",
                                offer=_pack_aop_offer(proposal),
                                reason="first_proposal",
                            )
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
    _patch_oneshot_adapter_execution_forward()
    if not TRACKER_AVAILABLE:
        print("[Tracker] Warning: scml_analyzer not available, tracking disabled")
        return agent_classes
    
    # 获取当前配置的 log_dir，用于并行模式
    config = TrackerConfig.get()
    captured_log_dir = config.log_dir
    
    # 设置环境变量，子进程会继承
    if captured_log_dir:
        os.environ['SCML_TRACKER_LOG_DIR'] = captured_log_dir
    
    patched = []
    for cls in agent_classes:
        patched_cls = patch_agent_class(cls, captured_log_dir=captured_log_dir)
        patched.append(patched_cls)
    
    return patched


def setup_tracker_for_tournament(log_dir: str, enabled: bool = True):
    """
    为比赛配置 Tracker

    Args:
        log_dir: 日志目录
        enabled: 是否启用 Tracker
    """
    _patch_oneshot_adapter_execution_forward()
    if not TRACKER_AVAILABLE:
        return
    
    tracker_dir = os.path.join(log_dir, "tracker_logs")
    os.makedirs(tracker_dir, exist_ok=True)
    
    # 设置环境变量，子进程会继承这个值
    os.environ['SCML_TRACKER_LOG_DIR'] = tracker_dir
    
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


DEFAULT_TRACKED_MODULE = "litaagent_std.tracked_registry"


def create_tracked_agent(
    base_class: Type,
    log_dir: str,
    agent_name_suffix: str = "Tracked",
    register: bool = True,
    register_module: Optional[str] = None,
    class_name: Optional[str] = None,
) -> Type:
    """
    创建一个带有 Tracker 功能的 Agent 子类（支持并行模式）
    
    这个函数创建一个新的类，继承自原始 Agent 类，并添加 Tracker 功能。
    由于创建的是真正的子类，所以在并行模式下也能正常工作。
    
    Args:
        base_class: 原始 Agent 类（如 LitaAgentY）
        log_dir: Tracker 日志目录（绝对路径）
        agent_name_suffix: 新类名后缀
        register: 是否写入跨进程注册表
        register_module: 注册模块（默认 tracked_registry）
        class_name: 显式指定类名（用于注册重建）
        
    Returns:
        带有 Tracker 功能的新 Agent 类
        
    Example:
        TrackedLitaAgentY = create_tracked_agent(
            LitaAgentY, 
            log_dir=os.path.abspath("./tracker_logs")
        )
        # 然后可以在比赛中使用 TrackedLitaAgentY
    """
    _patch_oneshot_adapter_execution_forward()
    if not TRACKER_AVAILABLE:
        print(f"[Tracker] Warning: scml_analyzer not available, returning original class")
        return base_class
    
    # 确保 log_dir 是绝对路径
    abs_log_dir = os.path.abspath(log_dir)
    target_module = register_module or DEFAULT_TRACKED_MODULE
    new_class_name = class_name or f"{base_class.__name__}{agent_name_suffix}"

    if register:
        try:
            from litaagent_std import tracked_registry

            base_path = f"{base_class.__module__}.{base_class.__name__}"
            tracked_registry.register(new_class_name, base_path)
        except Exception:
            pass
    
    # 动态创建新类
    class TrackedAgent(base_class):
        """带有 Tracker 功能的 Agent"""
        
        # 类级别变量，存储 log_dir
        _tracker_log_dir: str = abs_log_dir
        _tracker_logger: Optional[AgentLogger] = None
        _tracker_enabled: bool = True
        
        @property
        def tracker(self) -> Optional[AgentLogger]:
            """获取 Tracker Logger"""
            if not self._tracker_enabled:
                return None
            agent_id = getattr(self, 'id', 'unknown')
            agent_type = type(self).__bases__[0].__name__  # 使用父类名称
            self._tracker_logger = _ensure_tracker_logger(
                self,
                agent_id=str(agent_id),
                agent_type=str(agent_type),
                current_logger=self._tracker_logger,
            )
            return self._tracker_logger
        
        def init(self):
            """初始化时记录"""
            super().init()
            self._tracker_exec_sales_by_day = {}
            self._tracker_exec_supplies_by_day = {}
            tracker = self.tracker
            if tracker:
                identity = _build_agent_identity(self, base_class=base_class)
                tracker.custom("agent_initialized", 
                    n_steps=getattr(self.awi, 'n_steps', None),
                    n_lines=getattr(self.awi, 'n_lines', None),
                    level=getattr(self.awi, 'level', None),
                    **identity,
                )

        def _log_daily_status(self):
            tracker = self.tracker
            if not tracker:
                return
            try:
                awi = self.awi
                tracker.set_day(awi.current_step)
                balance = getattr(awi, 'current_balance', 0.0) or 0.0
                score = getattr(awi, 'current_score', 1.0) or 1.0
                exo_input_qty = getattr(awi, 'current_exogenous_input_quantity', 0) or 0
                exo_output_qty = getattr(awi, 'current_exogenous_output_quantity', 0) or 0
                exo_input_price = getattr(awi, 'current_exogenous_input_price', 0) or 0
                exo_output_price = getattr(awi, 'current_exogenous_output_price', 0) or 0
                needed_supplies_raw = getattr(awi, 'needed_supplies', 0) or 0
                needed_sales_raw = getattr(awi, 'needed_sales', 0) or 0
                needed_supplies = max(0, int(needed_supplies_raw))
                needed_sales = max(0, int(needed_sales_raw))
                total_supplies = getattr(awi, 'total_supplies', 0) or 0
                total_sales = getattr(awi, 'total_sales', 0) or 0
                disposal_cost = getattr(awi, 'current_disposal_cost', 0) or 0
                shortfall_penalty = getattr(awi, 'current_shortfall_penalty', 0) or 0
                storage_cost = getattr(awi, 'current_storage_cost', 0) or 0
                n_lines = getattr(awi, 'n_lines', 0) or 0
                n_steps = getattr(awi, 'n_steps', 100) or 100
                current_step = getattr(awi, 'current_step', 0) or 0

                exec_sales = int(getattr(self, "_tracker_exec_sales_by_day", {}).get(current_step, 0))
                exec_supplies = int(getattr(self, "_tracker_exec_supplies_by_day", {}).get(current_step, 0))

                # === HRL-XF 训练所需字段 ===
                trading_prices = getattr(awi, 'trading_prices', None)
                if trading_prices is not None:
                    trading_prices_list = list(trading_prices) if hasattr(trading_prices, '__iter__') else []
                else:
                    trading_prices_list = []

                my_input_product = getattr(awi, 'my_input_product', 0) or 0
                my_output_product = getattr(awi, 'my_output_product', 1) or 1
                spot_price_in = trading_prices_list[my_input_product] if my_input_product < len(trading_prices_list) else 10.0
                spot_price_out = trading_prices_list[my_output_product] if my_output_product < len(trading_prices_list) else 20.0

                # 真实库存
                current_inventory = getattr(awi, 'current_inventory', None)
                if current_inventory is not None:
                    if hasattr(current_inventory, '__iter__') and not isinstance(current_inventory, (str, dict)):
                        inventory_input = int(current_inventory[0]) if len(current_inventory) > 0 else 0
                        inventory_output = int(current_inventory[-1]) if len(current_inventory) > 0 else 0
                    elif isinstance(current_inventory, dict):
                        inventory_input = int(current_inventory.get('input', 0))
                        inventory_output = int(current_inventory.get('output', 0))
                    else:
                        inventory_input = int(current_inventory)
                        inventory_output = 0
                else:
                    inventory_input = exo_input_qty
                    inventory_output = exo_output_qty

                remaining_steps = n_steps - current_step

                # === 从 AWI future_* 重建 commitments ===
                max_len = min(remaining_steps + 1, 50)
                Q_in, Q_out, Payables, Receivables, has_future = _build_commitments_from_future(
                    awi, current_step, max_len
                )

                offers_snapshot = _extract_offers_snapshot(awi, current_step)

                tracker.inventory_state(
                    raw_material=inventory_input,
                    product=inventory_output,
                    balance=balance,
                )
                # P0 修复: 获取 n_products 用于正确计算 x_role
                n_products = getattr(awi, 'n_products', my_output_product + 1) or (my_output_product + 1)

                tracker.custom("daily_status",
                    score=score,
                    balance=balance,
                    initial_balance=_get_initial_balance(awi, getattr(self, "id", None)),
                    exo_input_qty=exo_input_qty,
                    exo_input_price=exo_input_price,
                    exo_output_qty=exo_output_qty,
                    exo_output_price=exo_output_price,
                    needed_supplies=needed_supplies,
                    needed_sales=needed_sales,
                    needed_supplies_raw=needed_supplies_raw,
                    needed_sales_raw=needed_sales_raw,
                    total_supplies=total_supplies,
                    total_sales=total_sales,
                    executed_supplies=exec_supplies,
                    executed_sales=exec_sales,
                    disposal_cost=disposal_cost,
                    shortfall_penalty=shortfall_penalty,
                    storage_cost=storage_cost,
                    n_lines=n_lines,
                    n_steps=n_steps,
                    current_step=current_step,
                    # HRL-XF 新增字段
                    spot_price_in=float(spot_price_in),
                    spot_price_out=float(spot_price_out),
                    trading_prices=trading_prices_list,
                    inventory_input=inventory_input,
                    inventory_output=inventory_output,
                    my_input_product=my_input_product,
                    my_output_product=my_output_product,
                    n_products=n_products,  # P0 修复: 用于正确计算 is_last_level 和 x_role
                    # 合同承诺（用于离线计算 L1 baseline）
                    # H=40 天规划视界，记录 H+1=41 个时间点
                    commitments={
                        'Q_in': Q_in[:41],
                        'Q_out': Q_out[:41],
                        'Payables': Payables[:41],
                        'Receivables': Receivables[:41],
                    },
                    # 活跃谈判快照（用于离线重建6-8 通道）
                    offers_snapshot=offers_snapshot,
                )
            except Exception:
                pass
        
        def before_step(self):
            """???????????? HRL-XF ???????"""
            super().before_step()
            tracker = self.tracker
            if tracker:
                tracker.set_day(self.awi.current_step)

        def step(self):
            """执行步骤，在最后一步保存数据"""
            super().step()
            self._log_daily_status()
            tracker = self.tracker
            if tracker:
                try:
                    current_step = getattr(self.awi, 'current_step', 0)
                    n_steps = getattr(self.awi, 'n_steps', 0)
                    
                    if current_step >= n_steps - 1:
                        # 获取 world_id
                        world_id = 'unknown'
                        if hasattr(self.awi, '_world') and self.awi._world:
                            world_id = getattr(self.awi._world, 'id', 'unknown') or 'unknown'
                        tracker.world_id = world_id
                        
                        # 保存文件
                        log_dir = self._tracker_log_dir
                        if log_dir:
                            os.makedirs(log_dir, exist_ok=True)
                            safe_agent_id = tracker.agent_id.replace("@", "_at_").replace("/", "_")
                            raw_world_id = str(world_id)
                            safe_world_id = raw_world_id.replace("/", "_").replace("\\", "_")
                            short_world_id = safe_world_id[:40]
                            world_hash = hashlib.sha1(raw_world_id.encode("utf-8")).hexdigest()[:12]
                            filename = f"agent_{safe_agent_id}_world_{short_world_id}_{world_hash}.json"
                            filepath = os.path.join(log_dir, filename)
                            tracker.save(filepath)
                except Exception:
                    pass
        
        def on_negotiation_success(self, contract, mechanism):
            """协商成功时记录"""
            super().on_negotiation_success(contract, mechanism)
            tracker = self.tracker
            if tracker:
                try:
                    qty, price, time, buyer, seller, partner, product_index = _infer_contract_parties(self, contract)
                    if not partner:
                        partner = "unknown"

                    # P1 修复: 使用 product_index 正确判断买卖方向
                    my_output_product = getattr(self.awi, 'my_output_product', 1)
                    is_seller = None
                    if seller is not None and self.id == seller:
                        is_seller = True
                    elif buyer is not None and self.id == buyer:
                        is_seller = False
                    elif product_index is not None:
                        is_seller = (product_index == my_output_product)
                    else:
                        # 最后回退: 基于层级（不推荐，可能不准确）
                        is_seller = not getattr(self.awi, 'is_first_level', True)

                    tracker.contract_signed(
                        contract_id=str(contract.id),
                        partner=partner,
                        quantity=qty,
                        price=price,
                        delivery_day=time,
                        is_seller=is_seller,
                        buyer=buyer,
                        seller=seller,
                    )
                    tracker.negotiation_success(
                        partner,
                        {"quantity": qty, "price": price, "time": time, "buyer": buyer, "seller": seller},
                        buyer=buyer,
                        seller=seller,
                        role="seller" if is_seller else "buyer",
                    )
                except Exception:
                    pass
        
        def on_negotiation_failure(self, partners, annotation, mechanism, state):
            """协商失败时记录"""
            super().on_negotiation_failure(partners, annotation, mechanism, state)
            tracker = self.tracker
            if tracker:
                try:
                    partner = partners[0] if partners else "unknown"
                    is_seller = self._infer_is_seller(partner) if partner != "unknown" else False
                    buyer, seller = _infer_buyer_seller_from_role(getattr(self, "id", None), partner, is_seller)
                    tracker.negotiation_failure(
                        partner,
                        "negotiation_ended_without_agreement",
                        buyer=buyer,
                        seller=seller,
                        role="seller" if is_seller else "buyer",
                    )
                except Exception:
                    pass

        def _infer_contract_is_seller(self, contract) -> bool:
            my_input_product = getattr(self.awi, "my_input_product", 0)
            my_output_product = getattr(self.awi, "my_output_product", 1)
            annotation = getattr(contract, "annotation", {}) or {}
            product_index = annotation.get("product", None) if isinstance(annotation, dict) else None
            if hasattr(product_index, "__len__") and not isinstance(product_index, (str, int)):
                product_index = product_index[0] if len(product_index) > 0 else None
            if product_index is not None:
                return product_index == my_output_product

            seller = annotation.get("seller", None)
            buyer = annotation.get("buyer", None)
            if seller is not None:
                return seller == getattr(self, "id", None)
            if buyer is not None:
                return buyer != getattr(self, "id", None)

            buyer_id = getattr(contract, "buyer", None)
            if buyer_id is not None:
                return buyer_id != getattr(self, "id", None)
            if getattr(self.awi, "is_first_level", False):
                return True
            if getattr(self.awi, "is_last_level", False):
                return False
            return False

        def on_contract_executed(self, contract):
            super().on_contract_executed(contract)
            tracker = self.tracker
            if tracker:
                try:
                    qty, price, delivery, buyer, seller, partner, product_index = _infer_contract_parties(self, contract)
                    current_step = int(getattr(self.awi, "current_step", 0) or 0)
                    is_seller = None
                    if seller is not None and self.id == seller:
                        is_seller = True
                    elif buyer is not None and self.id == buyer:
                        is_seller = False
                    else:
                        is_seller = self._infer_contract_is_seller(contract)
                    if is_seller:
                        sales = getattr(self, "_tracker_exec_sales_by_day", {})
                        sales[current_step] = int(sales.get(current_step, 0)) + int(qty)
                        self._tracker_exec_sales_by_day = sales
                    else:
                        supplies = getattr(self, "_tracker_exec_supplies_by_day", {})
                        supplies[current_step] = int(supplies.get(current_step, 0)) + int(qty)
                        self._tracker_exec_supplies_by_day = supplies
                    tracker.contract_executed(
                        contract_id=str(getattr(contract, "id", "")),
                        quantity=qty,
                        partner=partner,
                        price=price,
                        delivery_day=delivery,
                        buyer=buyer,
                        seller=seller,
                        role="seller" if is_seller else "buyer",
                    )
                except Exception:
                    pass

        def _infer_is_seller(self, partner_id: str) -> bool:
            nmi = None
            try:
                nmi = self.get_nmi(partner_id)
            except Exception:
                nmi = None

            annotation = getattr(nmi, "annotation", {}) or {}
            product_index = annotation.get("product", None)
            if hasattr(product_index, "__len__") and not isinstance(product_index, (str, int)):
                product_index = product_index[0] if len(product_index) > 0 else None

            my_input_product = getattr(self.awi, "my_input_product", None)
            my_output_product = getattr(self.awi, "my_output_product", None)

            if product_index is not None:
                return my_output_product is not None and product_index == my_output_product

            seller = annotation.get("seller", None)
            buyer = annotation.get("buyer", None)
            if seller is not None and seller == getattr(self, "id", None):
                return True
            if buyer is not None and buyer == getattr(self, "id", None):
                return False
            if getattr(self.awi, "is_first_level", False):
                return True
            if getattr(self.awi, "is_last_level", False):
                return False
            return False

        def _get_mechanism_id(self, partner_id: str, state=None) -> Optional[str]:
            nmi = None
            try:
                nmi = self.get_nmi(partner_id)
            except Exception:
                nmi = None
            mech_id = getattr(nmi, "mechanism_id", None) if nmi is not None else None
            if not mech_id and state is not None:
                mech_id = getattr(state, "mechanism_id", None) or getattr(state, "id", None)
            return str(mech_id) if mech_id else None

        def _ensure_negotiation_started(self, partner_id: str, state=None, issues: Optional[Dict[str, Any]] = None) -> Optional[str]:
            tracker = self.tracker
            if not tracker:
                return None
            mechanism_id = self._get_mechanism_id(partner_id, state)
            sim_step = int(getattr(self.awi, "current_step", 0) or 0)
            if mechanism_id:
                seen_key = (mechanism_id, partner_id)
            else:
                seen_key = (partner_id, sim_step, "no_mechanism")
            seen = getattr(self, "_tracker_seen_negotiations", None)
            if seen is None:
                seen = set()
                setattr(self, "_tracker_seen_negotiations", seen)
            if seen_key in seen:
                return mechanism_id
            is_seller = self._infer_is_seller(partner_id)
            issues_payload = issues
            min_price = None
            max_price = None
            if issues_payload is None:
                issues_payload = {}
                nmi = None
                try:
                    nmi = self.get_nmi(partner_id)
                except Exception:
                    nmi = None
                if nmi is not None:
                    try:
                        issues_payload = {"issues": getattr(nmi, "issues", None)}
                    except Exception:
                        issues_payload = {}
            else:
                if not isinstance(issues_payload, dict):
                    issues_payload = {"issues": issues_payload}
                if isinstance(issues_payload, dict):
                    min_price = _coerce_float(issues_payload.get("min_price"))
                    max_price = _coerce_float(issues_payload.get("max_price"))

            if min_price is None or max_price is None:
                nmi = None
                try:
                    nmi = self.get_nmi(partner_id)
                except Exception:
                    nmi = None
                if nmi is not None:
                    nmi_issues = getattr(nmi, "issues", None)
                    nmi_min, nmi_max = _extract_price_bounds(nmi_issues)
                    if min_price is None:
                        min_price = nmi_min
                    if max_price is None:
                        max_price = nmi_max

            if isinstance(issues_payload, dict):
                if min_price is not None:
                    issues_payload["min_price"] = float(min_price)
                if max_price is not None:
                    issues_payload["max_price"] = float(max_price)
            buyer, seller = _infer_buyer_seller_from_role(getattr(self, "id", None), partner_id, is_seller)
            tracker.negotiation_started(
                partner_id,
                issues_payload,
                is_seller=is_seller,
                mechanism_id=mechanism_id,
                negotiator_id=partner_id,
                buyer=buyer,
                seller=seller,
                role="seller" if is_seller else "buyer",
            )
            seen.add(seen_key)
            return mechanism_id

        def _pack_aop_offer(self, offer):
            if offer is None:
                return None
            try:
                qty, delivery_day, price = offer
            except Exception:
                return None
            return {
                "quantity": int(qty),
                "unit_price": float(price),
                "delivery_day": int(delivery_day),
            }

        def _normalize_response_type(self, resp):
            if resp is None:
                return "unknown"
            name = getattr(resp, "name", None)
            if name:
                return name
            if isinstance(resp, (int, float)):
                value = int(resp)
            else:
                text = str(resp)
                if text.startswith("ResponseType."):
                    return text.split("ResponseType.", 1)[1]
                try:
                    value = int(text)
                except Exception:
                    value = None
            if value is not None:
                if value == 0:
                    return "ACCEPT_OFFER"
                if value == 1:
                    return "REJECT_OFFER"
                if value == 2:
                    return "END_NEGOTIATION"
                if value == 3:
                    return "NO_RESPONSE"
                if value == 4:
                    return "WAIT"
            text = str(resp)
            if "ACCEPT" in text:
                return "ACCEPT_OFFER"
            if "REJECT" in text:
                return "REJECT_OFFER"
            if "END" in text:
                return "END_NEGOTIATION"
            return "unknown"
        
        def counter_all(self, offers, states):
            """记录协商报价"""
            responses = super().counter_all(offers, states)
            tracker = self.tracker
            if tracker:
                try:
                    for partner_id, offer in offers.items():
                        state = states.get(partner_id)
                        response = responses.get(partner_id)
                        mechanism_id = self._ensure_negotiation_started(partner_id, state=state)
                        is_seller = self._infer_is_seller(partner_id)
                        buyer, seller = _infer_buyer_seller_from_role(getattr(self, "id", None), partner_id, is_seller)
                        role = "seller" if is_seller else "buyer"
                        
                        offer_qty = offer[0] if offer else 0
                        offer_time = offer[1] if offer and len(offer) > 1 else 0
                        offer_price = offer[2] if offer and len(offer) > 2 else 0
                        
                        tracker.negotiation_offer_received(
                            partner_id,
                            {
                                "quantity": offer_qty,
                                "delivery_day": offer_time,
                                "unit_price": offer_price,
                                "round": state.step if state else 0,
                            },
                            mechanism_id=mechanism_id,
                            negotiator_id=partner_id,
                            buyer=buyer,
                            seller=seller,
                            role=role,
                        )
                        
                        if response:
                            response_type = self._normalize_response_type(response.response)
                            counter_offer = response.outcome
                            
                            if response_type == "ACCEPT_OFFER":
                                tracker.negotiation_accept(
                                    partner_id,
                                    {
                                        "quantity": offer_qty,
                                        "delivery_day": offer_time,
                                        "unit_price": offer_price,
                                    },
                                    reason="accepted_in_counter_all",
                                    mechanism_id=mechanism_id,
                                    negotiator_id=partner_id,
                                    buyer=buyer,
                                    seller=seller,
                                    role=role,
                                )
                            elif response_type == "REJECT_OFFER" and counter_offer:
                                counter_qty = counter_offer[0] if counter_offer else 0
                                counter_time = counter_offer[1] if counter_offer and len(counter_offer) > 1 else 0
                                counter_price = counter_offer[2] if counter_offer and len(counter_offer) > 2 else 0
                                
                                tracker.negotiation_offer_made(
                                    partner_id,
                                    {
                                        "quantity": counter_qty,
                                        "delivery_day": counter_time,
                                        "unit_price": counter_price,
                                        "round": state.step if state else 0,
                                    },
                                    reason="counter_offer",
                                    mechanism_id=mechanism_id,
                                    negotiator_id=partner_id,
                                    buyer=buyer,
                                    seller=seller,
                                    role=role,
                                )
                            elif response_type == "END_NEGOTIATION":
                                tracker.negotiation_reject(
                                    partner_id,
                                    {
                                        "quantity": offer_qty,
                                        "delivery_day": offer_time,
                                        "unit_price": offer_price,
                                    },
                                    reason="end_negotiation",
                                    mechanism_id=mechanism_id,
                                    negotiator_id=partner_id,
                                    buyer=buyer,
                                    seller=seller,
                                    role=role,
                                )

                    # AOP 动作记录（覆盖 states.keys()，补齐隐式 END）
                    for partner_id, state in states.items():
                        response = responses.get(partner_id)
                        current_offer = getattr(state, "current_offer", None) if state else None
                        is_seller = self._infer_is_seller(partner_id)
                        round_idx = int(getattr(state, "step", 0) or 0)
                        sim_step = int(getattr(self.awi, "current_step", 0) or 0)
                        mechanism_id = self._get_mechanism_id(partner_id, state)

                        action_op = "END"
                        response_type = "END_NEGOTIATION"
                        offer_payload = None
                        reason = "omit_in_counter_all"

                        if response is not None:
                            response_type = self._normalize_response_type(response.response)
                            reason = "counter_all"
                            if response_type == "ACCEPT_OFFER":
                                action_op = "ACCEPT"
                                offer_payload = self._pack_aop_offer(current_offer)
                            elif response_type == "REJECT_OFFER":
                                counter_offer = response.outcome
                                if counter_offer is not None:
                                    action_op = "REJECT"
                                    offer_payload = self._pack_aop_offer(counter_offer)
                                else:
                                    action_op = "END"
                            elif response_type == "END_NEGOTIATION":
                                action_op = "END"

                        tracker.custom(
                            "aop_action",
                            sim_step=sim_step,
                            negotiator_id=partner_id,
                            mechanism_id=mechanism_id,
                            partner=partner_id,
                            role="seller" if is_seller else "buyer",
                            round=round_idx,
                            action_op=action_op,
                            response_type=response_type,
                            offer=offer_payload,
                            reason=reason,
                        )
                except Exception:
                    pass
            return responses
        
        def first_proposals(self):
            """记录初始报价"""
            proposals = super().first_proposals()
            tracker = self.tracker
            if tracker and proposals:
                try:
                    for partner_id, proposal in proposals.items():
                        if proposal:
                            prop_qty = proposal[0] if proposal else 0
                            prop_time = proposal[1] if proposal and len(proposal) > 1 else 0
                            prop_price = proposal[2] if proposal and len(proposal) > 2 else 0
                            is_seller = self._infer_is_seller(partner_id)
                            buyer, seller = _infer_buyer_seller_from_role(getattr(self, "id", None), partner_id, is_seller)
                            role = "seller" if is_seller else "buyer"
                            issues_payload = {
                                "quantity_range": str(prop_qty),
                                "time_range": str(prop_time),
                                "price_range": str(prop_price),
                            }
                            mechanism_id = self._ensure_negotiation_started(
                                partner_id,
                                issues=issues_payload,
                            )
                            
                            tracker.negotiation_offer_made(
                                partner_id,
                                {
                                    "quantity": prop_qty,
                                    "delivery_day": prop_time,
                                    "unit_price": prop_price,
                                    "round": 0,
                                },
                                reason="first_proposal",
                                mechanism_id=mechanism_id,
                                negotiator_id=partner_id,
                                buyer=buyer,
                                seller=seller,
                                role=role,
                            )

                            tracker.custom(
                                "aop_action",
                                sim_step=int(getattr(self.awi, "current_step", 0) or 0),
                                negotiator_id=partner_id,
                                mechanism_id=mechanism_id,
                                partner=partner_id,
                                role=role,
                                round=0,
                                action_op="REJECT",
                                response_type="REJECT_OFFER",
                                offer=self._pack_aop_offer(proposal),
                                reason="first_proposal",
                            )
                except Exception:
                    pass
            return proposals
    
    # 设置类名和模块（重要：使模块指向注册模块，以便子进程可重建）
    TrackedAgent.__name__ = new_class_name
    TrackedAgent.__qualname__ = new_class_name
    TrackedAgent.__module__ = target_module

    # 将新类注册到目标模块中，这样子进程能找到它
    import importlib
    module = sys.modules.get(target_module)
    if module is None:
        module = importlib.import_module(target_module)
    setattr(module, new_class_name, TrackedAgent)
    
    return TrackedAgent
