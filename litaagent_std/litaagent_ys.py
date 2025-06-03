from __future__ import annotations

# ------------------ 基础依赖 ------------------
from typing import Any, Dict, List, Tuple, Iterable
from dataclasses import dataclass
import random
import os
import math
from collections import Counter, defaultdict # Added defaultdict
from uuid import uuid4

from numpy.random import choice as np_choice  # type: ignore

from scml.std import (
    StdSyncAgent,
    StdAWI,
    TIME,
    QUANTITY,
    UNIT_PRICE,
    )
from negmas import SAOState, SAOResponse, Outcome, Contract, ResponseType

# 内部工具 & manager
from .inventory_manager_ns import (
    InventoryManagerNS,
    IMContract,
    IMContractType,
    MaterialType,
)

# ------------------ 辅助函数 ------------------
# Helper functions

def _split_partners(partners: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """按 50 % / 30 % / 20 % 三段切分伙伴列表。"""
    # Split partners into 50%, 30% and 20% groups
    n = len(partners)
    return (
        partners[: int(n * 0.5)],
        partners[int(n * 0.5): int(n * 0.8)],
        partners[int(n * 0.8):],
    )

def _distribute(q: int, n: int) -> List[int]:
    """随机将 ``q`` 单位分配到 ``n`` 个桶，保证每桶至少 1（若可行）。"""
    # Randomly distribute ``q`` units into ``n`` buckets, each getting at least one if possible
    if n <= 0:
        return []

    if q < n:
        lst = [0] * (n - q) + [1] * q
        random.shuffle(lst)
        return lst
    if q == n:
        return [1] * n

    r = Counter(np_choice(n, q - n))
    return [r.get(i, 0) + 1 for i in range(n)]

# ------------------ 主代理实现 ------------------
# Main agent implementation

class LitaAgentYS(StdSyncAgent):
    """重构后的 LitaAgent N。支持三类采购策略与产能约束销售。"""

    # ------------------------------------------------------------------
    # 🌟 1. 初始化
    # 1. Initialization
    # ------------------------------------------------------------------

    def __init__(
        self,
        *args,
        min_profit_margin: float = 0.10,
        cheap_price_discount: float = 0.70,
        ptoday: float = 1.00,
        concession_curve_power: float = 1.5, 
        capacity_tight_margin_increase: float = 0.07, 
        procurement_cash_flow_limit_percent: float = 0.75, # Added from Step 6
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        # —— 参数 ——
        self.total_insufficient = None
        self.today_insufficient = None
        self.min_profit_margin = min_profit_margin         
        self.initial_min_profit_margin = min_profit_margin # Added from Step 7
        self.cheap_price_discount = cheap_price_discount   
        self.procurement_cash_flow_limit_percent = procurement_cash_flow_limit_percent # Added from Step 6
        self.concession_curve_power = concession_curve_power # Added from Step 9.b
        self.capacity_tight_margin_increase = capacity_tight_margin_increase # Added from Step 9.d
        
        # —— 运行时变量 ——
        self.im: InventoryManagerNS | None = None            
        self._market_price_avg: float = 0.0                
        self._market_material_price_avg: float = 0.0       
        self._market_product_price_avg: float = 0.0        
        self._recent_material_prices: List[float] = []     
        self._recent_product_prices: List[float] = []
        self._avg_window: int = 30                         
        self._ptoday: float = ptoday                       
        self.model = None                                  
        self.concession_model = None                       
        self._last_offer_price: Dict[str, float] = {}
        self.sales_completed: Dict[int, int] = {}
        self.purchase_completed: Dict[int, int] = {}  

        self.partner_stats: Dict[str, Dict[str, float]] = {}
        self.partner_models: Dict[str, Dict[str, float]] = {}
        self._last_partner_offer: Dict[str, float] = {}
        
        # Counters for dynamic profit margin adjustment (Added from Step 7)
        self._sales_successes_since_margin_update: int = 0
        self._sales_failures_since_margin_update: int = 0


    # ------------------------------------------------------------------
    # 🌟 2. World / 日常回调
    # ------------------------------------------------------------------

    def init(self) -> None:
        """在 World 初始化后调用；此处创建库存管理器。"""
        self.im = InventoryManagerNS(
            raw_storage_cost=self.awi.current_storage_cost,
            product_storage_cost=self.awi.current_storage_cost,
            processing_cost=self.awi.current_processing_cost,
            daily_production_capacity=self.awi.current_production_capacity,
            max_day=self.awi.n_steps,
        )


    def before_step(self) -> None:
        """每个 step 开始前调用；此处更新库存管理器状态。"""
        if not self.im:
            return

        current_day = self.awi.current_step
        
        # 更新库存管理器的当前天数
        # Update the current day of the inventory manager
        while self.im.current_day < current_day:
            self.im.update_day()

        # 获取当前不足的原材料量
        # Get the current insufficient raw materials
        self.today_insufficient = self.im.get_today_insufficient(current_day)
        self.total_insufficient = self.im.get_total_insufficient(current_day)

        # 处理外生合同（如果有）
        # Process exogenous contracts (if any)
        exogenous_contracts = self.awi.current_exogenous_contracts
        for contract in exogenous_contracts:
            # 创建合同 ID
            # Create contract ID
            exogenous_contract_id = f"exo_{contract.id}"
            
            # 确定合同类型和材料类型
            # Determine contract type and material type
            if contract.annotation["is_buy"]:
                # 采购合同（获取原材料）
                # Procurement contract (get raw materials)
                contract_type = IMContractType.SUPPLY
                material_type = MaterialType.RAW
            else:
                # 销售合同（交付产品）
                # Sales contract (deliver products)
                contract_type = IMContractType.DEMAND
                material_type = MaterialType.PRODUCT
                
            # 创建库存管理器合同对象
            # Create inventory manager contract object
            im_contract = IMContract(
                contract_id=exogenous_contract_id,
                partner_id=contract.annotation.get("seller", "MARKET") if contract.annotation["is_buy"] else contract.annotation.get("buyer", "MARKET"),
                type=contract_type,
                quantity=contract.agreement["quantity"],
                price=contract.agreement["unit_price"],
                delivery_time=contract.agreement["time"],
                bankruptcy_risk=0.0,  # 外生合同假设无破产风险 Assume no bankruptcy risk for exogenous contracts
                material_type=material_type
            )
            
            # 添加到库存管理器
            # Add to inventory manager
            self.im.add_transaction(im_contract)
            
            # 记录价格以更新市场价格估计
            # Record price to update market price estimate
            if contract_type == IMContractType.SUPPLY:
                self._recent_material_prices.append(contract.agreement["unit_price"])
            else:
                self._recent_product_prices.append(contract.agreement["unit_price"])

        # 更新市场价格估计
        # Update market price estimate
        self._update_dynamic_stockpiling_parameters()


    def step(self) -> None:
        """每个 step 结束时调用；此处更新市场价格估计。"""
        # 更新市场价格估计
        # Update market price estimate
        if len(self._recent_material_prices) > 0:
            self._market_material_price_avg = sum(self._recent_material_prices) / len(self._recent_material_prices)
        if len(self._recent_product_prices) > 0:
            self._market_product_price_avg = sum(self._recent_product_prices) / len(self._recent_product_prices)
            
        # 保持价格窗口大小
        # Keep price window size
        if len(self._recent_material_prices) > self._avg_window:
            self._recent_material_prices = self._recent_material_prices[-self._avg_window:]
        if len(self._recent_product_prices) > self._avg_window:
            self._recent_product_prices = self._recent_product_prices[-self._avg_window:]
            
        # 更新动态利润率参数
        # Update dynamic profit margin parameters
        self._update_dynamic_profit_margin_parameters()
        
        # 更新结果
        # Update result
        result = {}


    # ------------------------------------------------------------------
    # 🌟 3. 动态参数更新
    # ------------------------------------------------------------------

    def _update_dynamic_stockpiling_parameters(self) -> None:
        """更新动态库存参数。"""
        # 获取当前日期
        # Get current date
        current_day = self.awi.current_step
        
        # 如果没有库存管理器，直接返回
        # If there is no inventory manager, return directly
        if not self.im:
            return
            
        # 获取当前库存摘要
        # Get current inventory summary
        raw_summary = self.im.get_inventory_summary(current_day, MaterialType.RAW)
        product_summary = self.im.get_inventory_summary(current_day, MaterialType.PRODUCT)
        
        # 获取当前库存量
        # Get current inventory
        current_raw_stock = raw_summary["current_stock"]
        current_product_stock = product_summary["current_stock"]
        
        # 获取预期库存量
        # Get expected inventory
        expected_raw_stock = raw_summary["estimated_stock"]
        expected_product_stock = product_summary["estimated_stock"]
        
        # 获取当前生产计划
        # Get current production plan
        production_plan = self.im.get_production_plan_all()
        total_planned_production = sum(production_plan.values())
        
        # 获取未来交付计划
        # Get future delivery plan
        pending_demand = self.im.get_pending_contracts(is_supply=False)
        total_pending_demand = sum(c.quantity for c in pending_demand)
        
        # 获取未来采购计划
        # Get future procurement plan
        pending_supply = self.im.get_pending_contracts(is_supply=True)
        total_pending_supply = sum(c.quantity for c in pending_supply)
        
        # 计算库存覆盖率
        # Calculate inventory coverage
        raw_coverage = current_raw_stock / max(1, total_planned_production)
        product_coverage = current_product_stock / max(1, total_pending_demand)
        
        # 计算预期库存覆盖率
        # Calculate expected inventory coverage
        expected_raw_coverage = expected_raw_stock / max(1, total_planned_production)
        expected_product_coverage = expected_product_stock / max(1, total_pending_demand)
        
        # 根据库存覆盖率调整价格折扣
        # Adjust price discount based on inventory coverage
        # 如果原材料库存覆盖率低，提高折扣（更积极采购）
        # If raw material inventory coverage is low, increase discount (more aggressive procurement)
        # 如果产品库存覆盖率高，提高折扣（更积极销售）
        # If product inventory coverage is high, increase discount (more aggressive sales)
        old_discount = self.cheap_price_discount
        
        # 计算新折扣
        # Calculate new discount
        new_discount = old_discount
        
        # 如果原材料库存覆盖率低于 0.5，提高折扣
        # If raw material inventory coverage is less than 0.5, increase discount
        if raw_coverage < 0.5:
            new_discount = min(0.9, old_discount + 0.05)
        # 如果产品库存覆盖率高于 2.0，提高折扣
        # If product inventory coverage is greater than 2.0, increase discount
        elif product_coverage > 2.0:
            new_discount = min(0.9, old_discount + 0.05)
        # 如果原材料库存覆盖率高于 2.0，降低折扣
        # If raw material inventory coverage is greater than 2.0, decrease discount
        elif raw_coverage > 2.0:
            new_discount = max(0.5, old_discount - 0.05)
        # 如果产品库存覆盖率低于 0.5，降低折扣
        # If product inventory coverage is less than 0.5, decrease discount
        elif product_coverage < 0.5:
            new_discount = max(0.5, old_discount - 0.05)
            
        # 更新折扣
        # Update discount
        if abs(new_discount - old_discount) > 0.01:
            self.cheap_price_discount = new_discount


    def get_avg_raw_cost_fallback(self, current_day_for_im_summary: int, best_price_pid_for_fallback: str | None = None) -> float:
        """获取平均原材料成本（如果没有库存，则使用市场价格或最佳价格）。"""
        # 如果没有库存管理器，使用市场价格
        # If there is no inventory manager, use market price
        if not self.im:
            return self._market_material_price_avg
            
        # 获取当前库存摘要
        # Get current inventory summary
        raw_summary = self.im.get_inventory_summary(current_day_for_im_summary, MaterialType.RAW)
        
        # 如果有库存，使用库存平均成本
        # If there is inventory, use inventory average cost
        if raw_summary["current_stock"] > 0:
            return raw_summary["current_avg_cost"]
            
        # 如果有预期库存，使用预期库存平均成本
        # If there is expected inventory, use expected inventory average cost
        if raw_summary["estimated_stock"] > 0:
            return raw_summary["estimated_avg_cost"]
            
        # 如果有最佳价格，使用最佳价格
        # If there is best price, use best price
        if best_price_pid_for_fallback and best_price_pid_for_fallback in self._last_partner_offer:
            return self._last_partner_offer[best_price_pid_for_fallback]
            
        # 否则使用市场价格
        # Otherwise use market price
        return self._market_material_price_avg


    def _is_production_capacity_tight(self, day: int, quantity_being_considered: int = 0) -> bool:
        """检查指定日期的产能是否紧张。"""
        # 如果没有库存管理器，直接返回 False
        # If there is no inventory manager, return False directly
        if not self.im:
            return False
            
        # 获取当前可用产能
        # Get current available capacity
        available_capacity = self.im.get_available_production_capacity(day)
        
        # 如果可用产能小于考虑的数量，则产能紧张
        # If available capacity is less than the quantity being considered, capacity is tight
        return available_capacity < quantity_being_considered


    # ------------------------------------------------------------------
    # 🌟 4. 伙伴与价格管理
    # ------------------------------------------------------------------

    def _is_supplier(self, pid: str) -> bool:
        """检查指定伙伴是否为供应商。"""
        return pid in self.awi.my_suppliers

    def _is_consumer(self, pid: str) -> bool:
        """检查指定伙伴是否为消费者。"""
        return pid in self.awi.my_consumers

    def _best_price(self, pid: str) -> float:
        """获取与指定伙伴的最佳价格。"""
        return self.awi.current_output_issues[UNIT_PRICE].max_value if self._is_consumer(pid) else self.awi.current_input_issues[UNIT_PRICE].min_value

    def _is_price_too_high(self, pid: str, price: float) -> bool:
        """检查价格是否过高（对于采购）。"""
        if not self._is_supplier(pid):
            return False
        return price > self._market_material_price_avg * 1.2

    def _clamp_price(self, pid: str, price: float) -> float:
        """将价格限制在合理范围内。"""
        return min(self.awi.current_input_issues[UNIT_PRICE].max_value, max(self.awi.current_input_issues[UNIT_PRICE].min_value, price)) if self._is_supplier(pid) else min(self.awi.current_output_issues[UNIT_PRICE].max_value, max(self.awi.current_output_issues[UNIT_PRICE].min_value, price))

    def _expected_price(self, pid: str, default: float) -> float:
        """获取预期价格。"""
        # 如果是供应商，使用材料价格；否则使用产品价格
        # If it's a supplier, use material price; otherwise use product price
        market_price = self._market_material_price_avg if self._is_supplier(pid) else self._market_product_price_avg
        
        # 如果市场价格为 0，使用默认价格
        # If market price is 0, use default price
        if market_price <= 0:
            return default
            
        # 如果是供应商，使用折扣价格；否则使用加价价格
        # If it's a supplier, use discounted price; otherwise use marked-up price
        if self._is_supplier(pid):
            # 采购：期望价格 = 市场价格 * 折扣
            # Procurement: expected price = market price * discount
            return market_price * self.cheap_price_discount
        else:
            # 销售：期望价格 = 原材料成本 * (1 + 利润率)
            # Sales: expected price = raw material cost * (1 + profit margin)
            current_day = self.awi.current_step
            raw_cost = self.get_avg_raw_cost_fallback(current_day, None)
            
            # 如果产能紧张，提高利润率
            # If capacity is tight, increase profit margin
            profit_margin = self.min_profit_margin
            if self._is_production_capacity_tight(current_day + 1, 1):
                profit_margin += self.capacity_tight_margin_increase
                
            # 计算期望价格
            # Calculate expected price
            return raw_cost * (1.0 + profit_margin) + self.awi.current_processing_cost


    # ------------------------------------------------------------------
    # 🌟 5. 让步策略
    # ------------------------------------------------------------------

    def _calc_opponent_concession(self, pid: str, price: float) -> float:
        """计算对手的让步率。"""
        # 如果没有上一次出价，返回 0
        # If there is no previous offer, return 0
        if pid not in self._last_partner_offer:
            return 0.0
            
        # 获取上一次出价
        # Get previous offer
        last_price = self._last_partner_offer[pid]
        
        # 如果是供应商，计算降价比例；否则计算涨价比例
        # If it's a supplier, calculate price reduction ratio; otherwise calculate price increase ratio
        if self._is_supplier(pid):
            # 供应商降价 = 让步
            # Supplier price reduction = concession
            return max(0.0, (last_price - price) / max(0.01, last_price))
        else:
            # 消费者涨价 = 让步
            # Consumer price increase = concession
            return max(0.0, (price - last_price) / max(0.01, last_price))

    def _concession_multiplier(self, rel_time: float, opp_rate: float = 0.0) -> float:
        """根据相对时间和对手让步率计算让步乘数。"""
        # 基础让步率 = 相对时间 ^ 让步曲线幂
        # Base concession rate = relative time ^ concession curve power
        base_rate = rel_time ** self.concession_curve_power
        
        # 考虑对手让步率
        # Consider opponent concession rate
        return base_rate * (1.0 - opp_rate * 0.5)  # 对手让步越多，我们让步越少 The more the opponent concedes, the less we concede

    def _apply_concession(self, pid: str, target_price: float, state: SAOState | None, current_price: float) -> float:
        """应用让步策略，返回新价格。"""
        # 如果没有状态，返回目标价格
        # If there is no state, return target price
        if not state:
            return target_price
            
        # 计算相对时间
        # Calculate relative time
        n_rounds = self.awi.current_input_issues[TIME].max_value if self._is_supplier(pid) else self.awi.current_output_issues[TIME].max_value
        rel_time = state.step / n_rounds if n_rounds > 0 else 1.0
        
        # 计算对手让步率
        # Calculate opponent concession rate
        opp_rate = self._calc_opponent_concession(pid, current_price)
        
        # 计算让步乘数
        # Calculate concession multiplier
        concession_mult = self._concession_multiplier(rel_time, opp_rate)
        
        # 更新对手最后出价
        # Update opponent's last offer
        self._last_partner_offer[pid] = current_price
        
        # 如果是供应商，我们希望价格越低越好；否则我们希望价格越高越好
        # If it's a supplier, we want the price to be as low as possible; otherwise we want the price to be as high as possible
        if self._is_supplier(pid):
            # 采购：目标价格 <= 当前价格，让步 = 提高我们愿意接受的价格
            # Procurement: target price <= current price, concession = increase the price we are willing to accept
            # 新价格 = 目标价格 + (当前价格 - 目标价格) * 让步乘数
            # New price = target price + (current price - target price) * concession multiplier
            return target_price + (current_price - target_price) * concession_mult
        else:
            # 销售：目标价格 >= 当前价格，让步 = 降低我们要求的价格
            # Sales: target price >= current price, concession = reduce the price we demand
            # 新价格 = 目标价格 - (目标价格 - 当前价格) * 让步乘数
            # New price = target price - (target price - current price) * concession multiplier
            return target_price - (target_price - current_price) * concession_mult


    # ------------------------------------------------------------------
    # 🌟 6. 接受模型
    # ------------------------------------------------------------------

    def _update_acceptance_model(self, pid: str, price: float, accepted: bool) -> None:
        """更新接受模型。"""
        # 如果伙伴不在统计中，初始化
        # If partner is not in statistics, initialize
        if pid not in self.partner_stats:
            self.partner_stats[pid] = {
                "n_offers": 0,
                "n_accepted": 0,
                "last_price": 0.0,
                "min_price": float('inf') if self._is_supplier(pid) else 0.0,
                "max_price": 0.0 if self._is_supplier(pid) else float('inf')
            }
            
        # 更新统计
        # Update statistics
        stats = self.partner_stats[pid]
        stats["n_offers"] += 1
        stats["last_price"] = price
        
        if accepted:
            stats["n_accepted"] += 1
            
        # 更新价格范围
        # Update price range
        if self._is_supplier(pid):
            stats["min_price"] = min(stats["min_price"], price)
            stats["max_price"] = max(stats["max_price"], price)
        else:
            stats["min_price"] = min(stats["min_price"], price)
            stats["max_price"] = max(stats["max_price"], price)

    def _estimate_reservation_price(self, pid: str, default: float) -> float:
        """估计伙伴的保留价格。"""
        # 如果伙伴不在统计中，返回默认值
        # If partner is not in statistics, return default value
        if pid not in self.partner_stats:
            return default
            
        # 获取统计
        # Get statistics
        stats = self.partner_stats[pid]
        
        # 如果是供应商，返回最低价格；否则返回最高价格
        # If it's a supplier, return the lowest price; otherwise return the highest price
        return stats["min_price"] if self._is_supplier(pid) else stats["max_price"]

    def _pareto_counter_offer(self, pid: str, qty: int, t: int, price: float, state: SAOState | None) -> Tuple[float, List[str]]:
        """生成帕累托改进的还价。"""
        # 初始化日志
        # Initialize log
        reason_log = []
        
        # 获取当前日期
        # Get current date
        current_day = self.awi.current_step
        
        # 计算目标价格
        # Calculate target price
        target_price = self._expected_price(pid, price)
        reason_log.append(f"TargetP={target_price:.2f}")
        
        # 应用让步策略
        # Apply concession strategy
        new_price = self._apply_concession(pid, target_price, state, price)
        reason_log.append(f"AfterConcession={new_price:.2f}")
        
        # 限制价格在合理范围内
        # Limit price to reasonable range
        new_price = self._clamp_price(pid, new_price)
        reason_log.append(f"AfterClamp={new_price:.2f}")
        
        # 如果是供应商，检查价格是否过高
        # If it's a supplier, check if the price is too high
        if self._is_supplier(pid):
            # 采购：如果价格过高，拒绝
            # Procurement: if the price is too high, reject
            if self._is_price_too_high(pid, new_price):
                reason_log.append("PriceTooHigh")
                # 如果有库存管理器，检查是否紧急需要
                # If there is an inventory manager, check if it's urgently needed
                if self.im and self.today_insufficient > 0:
                    # 紧急需要，接受高价
                    # Urgently needed, accept high price
                    reason_log.append(f"ButUrgent(Need={self.today_insufficient})")
                else:
                    # 不紧急，拒绝高价
                    # Not urgent, reject high price
                    new_price = target_price
                    reason_log.append(f"Rejected->Target={target_price:.2f}")
        else:
            # 销售：检查是否有足够的产能
            # Sales: check if there is enough capacity
            if self.im:
                # 检查产能
                # Check capacity
                if self._is_production_capacity_tight(t, qty):
                    # 产能紧张，提高价格
                    # Capacity is tight, increase price
                    capacity_premium = self.awi.current_processing_cost * self.capacity_tight_margin_increase
                    new_price += capacity_premium
                    reason_log.append(f"CapacityTight+={capacity_premium:.2f}")
                    
                # 检查原材料成本
                # Check raw material cost
                raw_cost = self.get_avg_raw_cost_fallback(current_day, None)
                min_acceptable = raw_cost * (1.0 + self.min_profit_margin) + self.awi.current_processing_cost
                
                # 如果新价格低于最低可接受价格，拒绝
                # If the new price is lower than the minimum acceptable price, reject
                if new_price < min_acceptable:
                    new_price = min_acceptable
                    reason_log.append(f"BelowMinAcceptable->Min={min_acceptable:.2f}")
        
        # 更新最后出价
        # Update last offer
        self._last_offer_price[pid] = new_price
        
        return new_price, reason_log


    # ------------------------------------------------------------------
    # 🌟 7. 需求计算
    # ------------------------------------------------------------------

    def _get_sales_demand_first_layer(self) -> int:
        """获取第一层销售需求（当前库存）。"""
        if not self.im:
            return 0
        product_summary = self.im.get_inventory_summary(self.awi.current_step, MaterialType.PRODUCT)
        return int(product_summary["current_stock"])

    def _get_sales_demand_last_layer(self) -> int:
        """获取最后一层销售需求（产能）。"""
        return self.awi.current_production_capacity

    def _get_sales_demand_middle_layer_today(self) -> int:
        """获取今天的中间层销售需求（可用原材料）。"""
        if not self.im:
            return 0
        raw_summary = self.im.get_inventory_summary(self.awi.current_step, MaterialType.RAW)
        return int(raw_summary["current_stock"])

    def _get_sales_demand_middle_layer(self, day: int) -> int:
        """获取指定日期的中间层销售需求（预期可用原材料）。"""
        if not self.im:
            return 0
        raw_summary = self.im.get_inventory_summary(day, MaterialType.RAW)
        return int(raw_summary["estimated_stock"])

    def _get_supply_demand_middle_last_layer_today(self) -> int:
        """获取今天的中间和最后层供应需求（不足原材料）。"""
        if not self.im:
            return self.awi.current_production_capacity
        # 获取今天不足的原材料量
        # Get the amount of insufficient raw materials today
        today_insufficient = self.im.get_today_insufficient(self.awi.current_step)
        return int(today_insufficient)

    def _get_supply_demand_middle_last_layer(self, day: int) -> int:
        """获取指定日期的中间和最后层供应需求（总不足原材料）。"""
        if not self.im:
            return self.awi.current_production_capacity
        # 获取从当前到指定日期总共还需要的原材料量
        # Get the total amount of insufficient raw materials from now to the specified date
        total_insufficient = self.im.get_total_insufficient(day)
        return int(total_insufficient)

    def _get_supply_demand_first_layer(self) -> int:
        """获取第一层供应需求（紧急采购）。"""
        return self._get_supply_demand_middle_last_layer_today()

    def _distribute_todays_needs(self, partners: Iterable[str] | None = None) -> Dict[str, int]:
        """分配今天的需求到伙伴。"""
        # 如果没有指定伙伴，使用所有供应商
        # If no partners are specified, use all suppliers
        if partners is None:
            partners = self.awi.my_suppliers
            
        # 转换为列表
        # Convert to list
        partners_list = list(partners)
        
        # 如果没有伙伴，返回空字典
        # If there are no partners, return an empty dictionary
        if not partners_list:
            return {}
            
        # 获取今天的需求
        # Get today's needs
        needs = self._get_supply_demand_middle_last_layer_today()
        
        # 如果没有需求，返回空字典
        # If there are no needs, return an empty dictionary
        if needs <= 0:
            return {}
            
        # 分配需求到伙伴
        # Distribute needs to partners
        return self._distribute_to_partners(partners_list, needs)

    def _distribute_to_partners(self, partners: List[str], needs: int) -> Dict[str, int]:
        """将需求分配到伙伴。"""
        # 如果没有伙伴或没有需求，返回空字典
        # If there are no partners or no needs, return an empty dictionary
        if not partners or needs <= 0:
            return {}
            
        # 按照 50% / 30% / 20% 三段切分伙伴列表
        # Split partners into 50%, 30% and 20% groups
        p1, p2, p3 = _split_partners(partners)
        
        # 按照比例分配需求
        # Distribute needs according to proportion
        q1 = int(needs * 0.5) if p1 else 0
        q2 = int(needs * 0.3) if p2 else 0
        q3 = needs - q1 - q2 if p3 else 0
        
        # 随机分配到每个伙伴
        # Randomly distribute to each partner
        d1 = _distribute(q1, len(p1))
        d2 = _distribute(q2, len(p2))
        d3 = _distribute(q3, len(p3))
        
        # 合并结果
        # Merge results
        return dict(zip(p1 + p2 + p3, d1 + d2 + d3))


    # ------------------------------------------------------------------
    # 🌟 8. 谈判策略
    # ------------------------------------------------------------------

    def first_proposals(self) -> Dict[str, Outcome]:
        """生成初始提案。"""
        # 初始化结果
        # Initialize result
        proposals = {}
        
        # 获取当前日期
        # Get current date
        today = self.awi.current_step
        
        # 处理供应商（采购原材料）
        # Process suppliers (purchase raw materials)
        # 获取今天的采购需求
        # Get today's procurement needs
        supply_needs = self._distribute_todays_needs()
        
        # 为每个供应商生成提案
        # Generate proposals for each supplier
        for supplier_id, quantity in supply_needs.items():
            # 如果数量为 0，跳过
            # If quantity is 0, skip
            if quantity <= 0:
                continue
                
            # 计算价格
            # Calculate price
            price = self._expected_price(supplier_id, self.awi.current_input_issues[UNIT_PRICE].min_value)
            
            # 创建提案
            # Create proposal
            proposals[supplier_id] = {
                QUANTITY: quantity,
                TIME: today + 1,  # 明天交付 Deliver tomorrow
                UNIT_PRICE: price
            }
            
            # 更新最后出价
            # Update last offer
            self._last_offer_price[supplier_id] = price
        
        # 处理消费者（销售产品）
        # Process consumers (sell products)
        # 获取可销售的产品数量
        # Get the quantity of products that can be sold
        sellable_quantity = self._get_sales_demand_first_layer()  # 当前库存 Current inventory
        sellable_quantity += self._get_sales_demand_middle_layer_today()  # 今天可用原材料 Raw materials available today
        
        # 如果有可销售的产品，为每个消费者生成提案
        # If there are products that can be sold, generate proposals for each consumer
        if sellable_quantity > 0:
            # 获取所有消费者
            # Get all consumers
            consumers = list(self.awi.my_consumers)
            
            # 如果没有消费者，跳过
            # If there are no consumers, skip
            if not consumers:
                return proposals
                
            # 分配产品到消费者
            # Distribute products to consumers
            consumer_quantities = self._distribute_to_partners(consumers, sellable_quantity)
            
            # 为每个消费者生成提案
            # Generate proposals for each consumer
            for consumer_id, quantity in consumer_quantities.items():
                # 如果数量为 0，跳过
                # If quantity is 0, skip
                if quantity <= 0:
                    continue
                    
                # 计算价格
                # Calculate price
                price = self._expected_price(consumer_id, self.awi.current_output_issues[UNIT_PRICE].max_value)
                
                # 创建提案
                # Create proposal
                proposals[consumer_id] = {
                    QUANTITY: quantity,
                    TIME: today + 1,  # 明天交付 Deliver tomorrow
                    UNIT_PRICE: price
                }
                
                # 更新最后出价
                # Update last offer
                self._last_offer_price[consumer_id] = price
        
        return proposals


    def counter_all(self, offers: Dict[str, Outcome], states: Dict[str, SAOState]) -> Dict[str, SAOResponse]:
        """对所有报价进行还价。"""
        # 初始化结果
        # Initialize result
        responses = {}
        
        # 分离供应和销售报价
        # Separate supply and sales offers
        supply_offers = {pid: offer for pid, offer in offers.items() if self._is_supplier(pid)}
        sales_offers = {pid: offer for pid, offer in offers.items() if self._is_consumer(pid)}
        
        # 处理供应报价
        # Process supply offers
        supply_responses = self._process_supply_offers(supply_offers, states)
        responses.update(supply_responses)
        
        # 计算今天接受的供应数量
        # Calculate the quantity of supplies accepted today
        sum_qty_supply_offer_today = sum(
            offer[QUANTITY]
            for pid, offer in supply_offers.items()
            if responses.get(pid, SAOResponse(ResponseType.REJECT_OFFER, None)).response == ResponseType.ACCEPT_OFFER
            and offer[TIME] == self.awi.current_step + 1
        )
        
        # 处理销售报价
        # Process sales offers
        sales_responses = self._process_sales_offers(sales_offers, states, sum_qty_supply_offer_today)
        responses.update(sales_responses)
        
        return responses


    def _process_supply_offers(self, offers: Dict[str, Outcome], states: Dict[str, SAOState]) -> Dict[str, SAOResponse]:
        """处理供应报价。"""
        # 初始化结果
        # Initialize result
        responses = {}
        
        # 如果没有报价，返回空字典
        # If there are no offers, return an empty dictionary
        if not offers:
            return responses
            
        # 获取当前日期
        # Get current date
        today = self.awi.current_step
        
        # 分离紧急和计划采购报价
        # Separate emergency and planned procurement offers
        emergency_offers = {}
        planned_offers = {}
        
        # 获取今天紧急需要的原材料量
        # Get the amount of raw materials urgently needed today
        emergency_need = self._get_supply_demand_first_layer()
        
        # 如果有紧急需要，处理紧急采购报价
        # If there is an urgent need, process emergency procurement offers
        if emergency_need > 0:
            # 按价格排序，优先处理低价报价
            # Sort by price, prioritize low-price offers
            sorted_offers = sorted(
                offers.items(),
                key=lambda x: x[1][UNIT_PRICE]
            )
            
            # 分配紧急需要到报价
            # Distribute urgent needs to offers
            remaining_need = emergency_need
            for pid, offer in sorted_offers:
                # 如果没有剩余需要，将剩余报价添加到计划采购
                # If there is no remaining need, add remaining offers to planned procurement
                if remaining_need <= 0:
                    planned_offers[pid] = offer
                    continue
                    
                # 计算可接受的数量
                # Calculate acceptable quantity
                acceptable_qty = min(remaining_need, offer[QUANTITY])
                
                # 如果可接受的数量大于 0，添加到紧急采购
                # If acceptable quantity is greater than 0, add to emergency procurement
                if acceptable_qty > 0:
                    # 复制报价，修改数量
                    # Copy offer, modify quantity
                    emergency_offers[pid] = dict(offer)
                    emergency_offers[pid][QUANTITY] = acceptable_qty
                    
                    # 更新剩余需要
                    # Update remaining need
                    remaining_need -= acceptable_qty
                    
                    # 如果报价数量大于可接受的数量，将剩余部分添加到计划采购
                    # If offer quantity is greater than acceptable quantity, add remaining part to planned procurement
                    if offer[QUANTITY] > acceptable_qty:
                        planned_offers[pid] = dict(offer)
                        planned_offers[pid][QUANTITY] = offer[QUANTITY] - acceptable_qty
                else:
                    # 如果可接受的数量为 0，添加到计划采购
                    # If acceptable quantity is 0, add to planned procurement
                    planned_offers[pid] = offer
        else:
            # 如果没有紧急需要，所有报价都是计划采购
            # If there is no urgent need, all offers are planned procurement
            planned_offers = offers
            
        # 处理紧急采购报价
        # Process emergency procurement offers
        emergency_responses = self._process_emergency_supply_offers(emergency_offers, states)
        responses.update(emergency_responses)
        
        # 处理计划采购报价
        # Process planned procurement offers
        planned_responses = self._process_planned_supply_offers(planned_offers, states)
        responses.update(planned_responses)
        
        return responses


    def _process_emergency_supply_offers(self, offers: Dict[str, Outcome], states: Dict[str, SAOState]) -> Dict[str, SAOResponse]:
        """处理紧急供应报价。"""
        # 初始化结果
        # Initialize result
        responses = {}
        
        # 如果没有报价，返回空字典
        # If there are no offers, return an empty dictionary
        if not offers:
            return responses
            
        # 获取当前日期
        # Get current date
        today = self.awi.current_step
        
        # 处理每个报价
        # Process each offer
        for pid, offer in offers.items():
            # 获取价格
            # Get price
            price = offer[UNIT_PRICE]
            
            # 获取数量
            # Get quantity
            quantity = offer[QUANTITY]
            
            # 获取交付日期
            # Get delivery date
            delivery_time = offer[TIME]
            
            # 获取状态
            # Get state
            state = states.get(pid)
            
            # 计算目标价格
            # Calculate target price
            target_price = self._expected_price(pid, self.awi.current_input_issues[UNIT_PRICE].min_value)
            
            # 如果价格小于等于目标价格，接受
            # If price is less than or equal to target price, accept
            if price <= target_price:
                responses[pid] = SAOResponse(ResponseType.ACCEPT_OFFER, None)
                self._update_acceptance_model(pid, price, True)
                continue
                
            # 如果价格过高，但是紧急需要，也接受
            # If price is too high, but urgently needed, also accept
            if self._is_price_too_high(pid, price):
                # 紧急需要，接受高价
                # Urgently needed, accept high price
                responses[pid] = SAOResponse(ResponseType.ACCEPT_OFFER, None)
                self._update_acceptance_model(pid, price, True)
                continue
                
            # 否则，生成还价
            # Otherwise, generate counter offer
            new_price, _ = self._pareto_counter_offer(pid, quantity, delivery_time, price, state)
            
            # 创建还价
            # Create counter offer
            counter_offer = {
                QUANTITY: quantity,
                TIME: delivery_time,
                UNIT_PRICE: new_price
            }
            
            # 添加到结果
            # Add to result
            responses[pid] = SAOResponse(ResponseType.REJECT_OFFER, counter_offer)
            self._update_acceptance_model(pid, price, False)
            
        return responses


    def _process_planned_supply_offers(self, offers: Dict[str, Outcome], states: Dict[str, SAOState]) -> Dict[str, SAOResponse]:
        """处理计划供应报价。"""
        # 初始化结果
        # Initialize result
        responses = {}
        
        # 如果没有报价，返回空字典
        # If there are no offers, return an empty dictionary
        if not offers:
            return responses
            
        # 获取当前日期
        # Get current date
        today = self.awi.current_step
        
        # 处理每个报价
        # Process each offer
        for pid, offer in offers.items():
            # 获取价格
            # Get price
            price = offer[UNIT_PRICE]
            
            # 获取数量
            # Get quantity
            quantity = offer[QUANTITY]
            
            # 获取交付日期
            # Get delivery date
            delivery_time = offer[TIME]
            
            # 获取状态
            # Get state
            state = states.get(pid)
            
            # 计算目标价格
            # Calculate target price
            target_price = self._expected_price(pid, self.awi.current_input_issues[UNIT_PRICE].min_value)
            
            # 如果价格小于等于目标价格，接受
            # If price is less than or equal to target price, accept
            if price <= target_price:
                responses[pid] = SAOResponse(ResponseType.ACCEPT_OFFER, None)
                self._update_acceptance_model(pid, price, True)
                continue
                
            # 如果价格过高，拒绝
            # If price is too high, reject
            if self._is_price_too_high(pid, price):
                # 生成还价
                # Generate counter offer
                new_price, _ = self._pareto_counter_offer(pid, quantity, delivery_time, price, state)
                
                # 创建还价
                # Create counter offer
                counter_offer = {
                    QUANTITY: quantity,
                    TIME: delivery_time,
                    UNIT_PRICE: new_price
                }
                
                # 添加到结果
                # Add to result
                responses[pid] = SAOResponse(ResponseType.REJECT_OFFER, counter_offer)
                self._update_acceptance_model(pid, price, False)
                continue
                
            # 否则，生成还价
            # Otherwise, generate counter offer
            new_price, _ = self._pareto_counter_offer(pid, quantity, delivery_time, price, state)
            
            # 创建还价
            # Create counter offer
            counter_offer = {
                QUANTITY: quantity,
                TIME: delivery_time,
                UNIT_PRICE: new_price
            }
            
            # 添加到结果
            # Add to result
            responses[pid] = SAOResponse(ResponseType.REJECT_OFFER, counter_offer)
            self._update_acceptance_model(pid, price, False)
            
        return responses


    def _process_optional_supply_offers(self, offers: Dict[str, Outcome], states: Dict[str, SAOState]) -> Dict[str, SAOResponse]:
        """处理可选供应报价。"""
        # 初始化结果
        # Initialize result
        responses = {}
        
        # 如果没有报价，返回空字典
        # If there are no offers, return an empty dictionary
        if not offers:
            return responses
            
        # 获取当前日期
        # Get current date
        today = self.awi.current_step
        
        # 处理每个报价
        # Process each offer
        for pid, offer in offers.items():
            # 获取价格
            # Get price
            price = offer[UNIT_PRICE]
            
            # 获取数量
            # Get quantity
            quantity = offer[QUANTITY]
            
            # 获取交付日期
            # Get delivery date
            delivery_time = offer[TIME]
            
            # 获取状态
            # Get state
            state = states.get(pid)
            
            # 计算目标价格
            # Calculate target price
            target_price = self._expected_price(pid, self.awi.current_input_issues[UNIT_PRICE].min_value)
            
            # 如果价格小于等于目标价格，接受
            # If price is less than or equal to target price, accept
            if price <= target_price:
                responses[pid] = SAOResponse(ResponseType.ACCEPT_OFFER, None)
                self._update_acceptance_model(pid, price, True)
                continue
                
            # 如果价格过高，拒绝
            # If price is too high, reject
            if self._is_price_too_high(pid, price):
                # 拒绝，不还价
                # Reject, no counter offer
                responses[pid] = SAOResponse(ResponseType.REJECT_OFFER, None)
                self._update_acceptance_model(pid, price, False)
                continue
                
            # 否则，生成还价
            # Otherwise, generate counter offer
            new_price, _ = self._pareto_counter_offer(pid, quantity, delivery_time, price, state)
            
            # 创建还价
            # Create counter offer
            counter_offer = {
                QUANTITY: quantity,
                TIME: delivery_time,
                UNIT_PRICE: new_price
            }
            
            # 添加到结果
            # Add to result
            responses[pid] = SAOResponse(ResponseType.REJECT_OFFER, counter_offer)
            self._update_acceptance_model(pid, price, False)
            
        return responses


    def _process_sales_offers(self, offers: Dict[str, Outcome], states: Dict[str, SAOState], sum_qty_supply_offer_today: int) -> Dict[str, SAOResponse]:
        """处理销售报价。"""
        # 初始化结果
        # Initialize result
        responses = {}
        
        # 如果没有报价，返回空字典
        # If there are no offers, return an empty dictionary
        if not offers:
            return responses
            
        # 获取当前日期
        # Get current date
        today = self.awi.current_step
        
        # 获取可销售的产品数量
        # Get the quantity of products that can be sold
        sellable_quantity = self._get_sales_demand_first_layer()  # 当前库存 Current inventory
        sellable_quantity += min(self._get_sales_demand_middle_layer_today(), sum_qty_supply_offer_today)  # 今天可用原材料 Raw materials available today
        
        # 按价格排序，优先处理高价报价
        # Sort by price, prioritize high-price offers
        sorted_offers = sorted(
            offers.items(),
            key=lambda x: x[1][UNIT_PRICE],
            reverse=True
        )
        
        # 处理每个报价
        # Process each offer
        for pid, offer in sorted_offers:
            # 获取价格
            # Get price
            price = offer[UNIT_PRICE]
            
            # 获取数量
            # Get quantity
            quantity = offer[QUANTITY]
            
            # 获取交付日期
            # Get delivery date
            delivery_time = offer[TIME]
            
            # 获取状态
            # Get state
            state = states.get(pid)
            
            # 计算目标价格
            # Calculate target price
            target_price = self._expected_price(pid, self.awi.current_output_issues[UNIT_PRICE].max_value)
            
            # 计算最低可接受价格
            # Calculate minimum acceptable price
            raw_cost = self.get_avg_raw_cost_fallback(today, pid)
            min_acceptable = raw_cost * (1.0 + self.min_profit_margin) + self.awi.current_processing_cost
            
            # 如果价格大于等于目标价格，接受
            # If price is greater than or equal to target price, accept
            if price >= target_price:
                # 检查是否有足够的产品
                # Check if there are enough products
                if quantity <= sellable_quantity:
                    responses[pid] = SAOResponse(ResponseType.ACCEPT_OFFER, None)
                    self._update_acceptance_model(pid, price, True)
                    sellable_quantity -= quantity
                    continue
                else:
                    # 如果没有足够的产品，还价（减少数量）
                    # If there are not enough products, counter offer (reduce quantity)
                    counter_offer = {
                        QUANTITY: sellable_quantity,
                        TIME: delivery_time,
                        UNIT_PRICE: price
                    }
                    responses[pid] = SAOResponse(ResponseType.REJECT_OFFER, counter_offer)
                    self._update_acceptance_model(pid, price, False)
                    sellable_quantity = 0
                    continue
                    
            # 如果价格小于最低可接受价格，拒绝
            # If price is less than minimum acceptable price, reject
            if price < min_acceptable:
                # 生成还价
                # Generate counter offer
                new_price, _ = self._pareto_counter_offer(pid, quantity, delivery_time, price, state)
                
                # 创建还价
                # Create counter offer
                counter_offer = {
                    QUANTITY: min(quantity, sellable_quantity),
                    TIME: delivery_time,
                    UNIT_PRICE: new_price
                }
                
                # 添加到结果
                # Add to result
                responses[pid] = SAOResponse(ResponseType.REJECT_OFFER, counter_offer)
                self._update_acceptance_model(pid, price, False)
                continue
                
            # 否则，检查是否有足够的产品
            # Otherwise, check if there are enough products
            if quantity <= sellable_quantity:
                # 接受
                # Accept
                responses[pid] = SAOResponse(ResponseType.ACCEPT_OFFER, None)
                self._update_acceptance_model(pid, price, True)
                sellable_quantity -= quantity
            else:
                # 如果没有足够的产品，还价（减少数量）
                # If there are not enough products, counter offer (reduce quantity)
                counter_offer = {
                    QUANTITY: sellable_quantity,
                    TIME: delivery_time,
                    UNIT_PRICE: price
                }
                responses[pid] = SAOResponse(ResponseType.REJECT_OFFER, counter_offer)
                self._update_acceptance_model(pid, price, False)
                sellable_quantity = 0
            
        return responses


    # ------------------------------------------------------------------
    # 🌟 9. 合同管理
    # ------------------------------------------------------------------

    def get_partner_id(self, contract: Contract) -> str | None:
        """获取合同伙伴 ID。"""
        # 如果没有协议，返回 None
        # If there is no agreement, return None
        if not contract.agreement:
            return None
            
        # 获取买家和卖家
        # Get buyer and seller
        buyer = contract.annotation.get("buyer")
        seller = contract.annotation.get("seller")
        
        # 如果我是买家，返回卖家；否则返回买家
        # If I am the buyer, return the seller; otherwise return the buyer
        if buyer == self.id:
            return seller
        else:
            return buyer


    def on_negotiation_failure(self, partners: List[str], annotation: Dict[str, Any], mechanism: StdAWI, state: SAOState) -> None:
        """谈判失败回调。"""
        # 获取当前日期
        # Get current date
        current_day = self.awi.current_step
        
        # 如果是销售谈判，增加销售失败计数
        # If it's a sales negotiation, increase sales failure count
        if any(self._is_consumer(pid) for pid in partners):
            self._sales_failures_since_margin_update += 1
            
        # 如果是采购谈判，检查是否紧急需要
        # If it's a procurement negotiation, check if it's urgently needed
        if any(self._is_supplier(pid) for pid in partners):
            # 如果有库存管理器，检查是否紧急需要
            # If there is an inventory manager, check if it's urgently needed
            if self.im and self.today_insufficient > 0:
                # 紧急需要，但谈判失败，可能需要调整策略
                # Urgently needed, but negotiation failed, may need to adjust strategy
                pass


    def on_negotiation_success(self, contract: Contract, mechanism: StdAWI) -> None:
        """谈判成功回调。"""
        # 如果没有库存管理器，直接返回
        # If there is no inventory manager, return directly
        if not self.im:
            return
            
        # 获取伙伴 ID
        # Get partner ID
        partner_id = self.get_partner_id(contract)
        
        # 如果没有伙伴 ID，直接返回
        # If there is no partner ID, return directly
        if not partner_id:
            return
            
        # 如果没有协议，直接返回
        # If there is no agreement, return directly
        if not contract.agreement:
            return
            
        # 获取协议内容
        # Get agreement content
        agreement = contract.agreement
        
        # 获取数量、价格和交付日期
        # Get quantity, price and delivery date
        quantity = agreement[QUANTITY]
        price = agreement[UNIT_PRICE]
        delivery_time = agreement[TIME]
        
        # 确定合同类型和材料类型
        # Determine contract type and material type
        if self._is_supplier(partner_id):
            # 采购合同（获取原材料）
            # Procurement contract (get raw materials)
            contract_type = IMContractType.SUPPLY
            material_type = MaterialType.RAW
        else:
            # 销售合同（交付产品）
            # Sales contract (deliver products)
            contract_type = IMContractType.DEMAND
            material_type = MaterialType.PRODUCT
            # 增加销售成功计数
            # Increase sales success count
            self._sales_successes_since_margin_update += 1
            
        # 创建库存管理器合同对象
        # Create inventory manager contract object
        new_c = IMContract(
            contract_id=contract.id,
            partner_id=partner_id,
            type=contract_type,
            quantity=quantity,
            price=price,
            delivery_time=delivery_time,
            bankruptcy_risk=0.0,  # 假设无破产风险 Assume no bankruptcy risk
            material_type=material_type
        )
        
        # 添加到库存管理器
        # Add to inventory manager
        self.im.add_transaction(new_c)


    def on_contracts_finalized(self, signed: List[Contract], cancelled: List[Contract], rejected: List[Contract]) -> None:
        """合同最终确定回调。"""
        # 获取当前日期
        # Get current date
        current_day = self.awi.current_step
        
        # 如果没有库存管理器，直接返回
        # If there is no inventory manager, return directly
        if not self.im:
            return
            
        # 处理取消的合同
        # Process cancelled contracts
        for contract in cancelled:
            # 从库存管理器中移除合同
            # Remove contract from inventory manager
            removed = self.im.void_negotiated_contract(contract.id)


    # ------------------------------------------------------------------
    # 🌟 10. 合同签署
    # ------------------------------------------------------------------

    def sign_all_contracts(self, contracts: List[Contract]) -> List[bool]:
        """签署所有合同。"""
        # 如果没有库存管理器，盲目签署所有合同
        # If there is no inventory manager, blindly sign all contracts
        if not self.im:
            return [True] * len(contracts)
            
        # 初始化结果
        # Initialize result
        results = []
        
        # 解析合同
        # Parse contracts
        pending_sales_contracts = []
        pending_supply_contracts = []
        
        # 处理每个合同
        # Process each contract
        for contract_obj_iter in contracts:
            # 如果没有协议，跳过
            # If there is no agreement, skip
            if not contract_obj_iter.agreement:
                results.append(False)
                continue
                
            # 获取伙伴 ID
            # Get partner ID
            partner_id = self.get_partner_id(contract_obj_iter)
            
            # 如果没有伙伴 ID，跳过
            # If there is no partner ID, skip
            if not partner_id:
                results.append(False)
                continue
                
            # 获取协议内容
            # Get agreement content
            agreement = contract_obj_iter.agreement
            
            # 获取数量、价格和交付日期
            # Get quantity, price and delivery date
            quantity = agreement[QUANTITY]
            price = agreement[UNIT_PRICE]
            delivery_time = agreement[TIME]
            
            # 确定合同类型
            # Determine contract type
            if self._is_supplier(partner_id):
                # 采购合同
                # Procurement contract
                pending_supply_contracts.append((contract_obj_iter, partner_id, quantity, price, delivery_time))
            else:
                # 销售合同
                # Sales contract
                pending_sales_contracts.append((contract_obj_iter, partner_id, quantity, price, delivery_time))
                
        # 第一阶段：初步贪心筛选
        # Phase 1: Initial greedy filtering
        # 按照价格排序，优先处理高价销售合同和低价采购合同
        # Sort by price, prioritize high-price sales contracts and low-price procurement contracts
        pending_sales_contracts.sort(key=lambda x: x[3], reverse=True)  # 按价格降序 Sort by price in descending order
        pending_supply_contracts.sort(key=lambda x: x[3])  # 按价格升序 Sort by price in ascending order
        
        # 初始化签署决策
        # Initialize signing decisions
        initial_sign_decisions = {}
        
        # 处理销售合同
        # Process sales contracts
        for s_data in pending_sales_contracts:
            contract_obj, partner_id, quantity, price, delivery_time = s_data
            
            # 计算最低可接受价格
            # Calculate minimum acceptable price
            current_day = self.awi.current_step
            raw_cost = self.get_avg_raw_cost_fallback(current_day, partner_id)
            min_acceptable = raw_cost * (1.0 + self.min_profit_margin) + self.awi.current_processing_cost
            
            # 如果价格小于最低可接受价格，拒绝
            # If price is less than minimum acceptable price, reject
            if price < min_acceptable:
                initial_sign_decisions[contract_obj.id] = False
                continue
                
            # 否则，暂时接受
            # Otherwise, temporarily accept
            initial_sign_decisions[contract_obj.id] = True
            
        # 第二阶段：现金流检查
        # Phase 2: Cash flow check
        # 计算总采购成本和总销售收入
        # Calculate total procurement cost and total sales revenue
        total_procurement_cost = sum(price * quantity for _, _, quantity, price, _ in pending_supply_contracts)
        total_sales_revenue = sum(price * quantity for contract_obj, _, quantity, price, _ in pending_sales_contracts if initial_sign_decisions.get(contract_obj.id, False))
        
        # 如果总采购成本大于总销售收入的限制比例，需要降低采购
        # If total procurement cost is greater than the limit ratio of total sales revenue, need to reduce procurement
        if total_sales_revenue > 0 and total_procurement_cost > total_sales_revenue * self.procurement_cash_flow_limit_percent:
            # 计算需要降低的采购成本
            # Calculate the procurement cost that needs to be reduced
            target_procurement_cost = total_sales_revenue * self.procurement_cash_flow_limit_percent
            
            # 按照价格从高到低排序，优先降低高价采购
            # Sort by price from high to low, prioritize reducing high-price procurement
            pending_supply_contracts.sort(key=lambda x: x[3], reverse=True)
            
            # 降低采购，直到达到目标
            # Reduce procurement until reaching the target
            current_procurement_cost = total_procurement_cost
            for s_data in pending_supply_contracts:
                contract_obj, _, quantity, price, _ = s_data
                
                # 如果已经达到目标，保留剩余采购
                # If the target has been reached, keep the remaining procurement
                if current_procurement_cost <= target_procurement_cost:
                    break
                    
                # 否则，拒绝这个采购
                # Otherwise, reject this procurement
                initial_sign_decisions[contract_obj.id] = False
                current_procurement_cost -= price * quantity
        else:
            # 如果现金流正常，接受所有采购
            # If cash flow is normal, accept all procurement
            for s_data in pending_supply_contracts:
                contract_obj, _, _, _, _ = s_data
                initial_sign_decisions[contract_obj.id] = True
                
        # 最终决策
        # Final decision
        final_signed_contracts_ids = [k for k, v in initial_sign_decisions.items() if v]
        final_rejected_contracts_ids = [k for k, v in initial_sign_decisions.items() if not v]
        
        # 生成结果
        # Generate result
        for contract_obj in contracts:
            if contract_obj.id in final_signed_contracts_ids:
                results.append(True)
            else:
                results.append(False)
                
        return results


    # ------------------------------------------------------------------
    # 🌟 11. 动态利润率调整
    # ------------------------------------------------------------------

    def _update_dynamic_profit_margin_parameters(self) -> None:
        """更新动态利润率参数。"""
        # 获取当前日期
        # Get current date
        current_day = self.awi.current_step
        
        # 如果是第一天，直接返回
        # If it's the first day, return directly
        if current_day == 0:
            return
            
        # 如果是最后一天，直接返回
        # If it's the last day, return directly
        if current_day >= self.awi.n_steps - 1:
            return
            
        # 计算成功率
        # Calculate success rate
        total_attempts = self._sales_successes_since_margin_update + self._sales_failures_since_margin_update
        
        # 如果尝试次数太少，直接返回
        # If the number of attempts is too small, return directly
        if total_attempts < 5:
            return
            
        # 计算成功率
        # Calculate success rate
        success_rate = self._sales_successes_since_margin_update / total_attempts if total_attempts > 0 else 0
        
        # 根据成功率调整利润率
        # Adjust profit margin based on success rate
        old_margin = self.min_profit_margin
        
        # 如果成功率高，提高利润率
        # If success rate is high, increase profit margin
        if success_rate > 0.8:
            # 成功率高，提高利润率
            # High success rate, increase profit margin
            new_margin = min(0.5, old_margin + 0.02)
        # 如果成功率低，降低利润率
        # If success rate is low, decrease profit margin
        elif success_rate < 0.2:
            # 成功率低，降低利润率
            # Low success rate, decrease profit margin
            new_margin = max(0.05, old_margin - 0.02)
        else:
            # 成功率适中，保持利润率
            # Moderate success rate, maintain profit margin
            new_margin = old_margin
            
        # 更新利润率
        # Update profit margin
        if abs(new_margin - old_margin) > 0.001:
            self.min_profit_margin = new_margin
            
        # 重置计数器
        # Reset counters
        self._sales_successes_since_margin_update = 0
        self._sales_failures_since_margin_update = 0


    def update_profit_strategy(self, *, min_profit_margin: float | None = None, cheap_price_discount: float | None = None) -> None:
        """更新利润策略。"""
        # 更新最小利润率
        # Update minimum profit margin
        if min_profit_margin is not None:
            self.min_profit_margin = min_profit_margin
            
        # 更新折扣
        # Update discount
        if cheap_price_discount is not None:
            self.cheap_price_discount = cheap_price_discount

    def decide_with_model(self, obs: Any) -> Any:
        """使用模型决策。"""
        pass

if __name__ == "__main__":
    pass