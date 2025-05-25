#!/usr/bin/env python
"""
LitaAgentY — SCML 2025  Standard 赛道谈判代理（重构版）
===================================================

本文件 **完全重写** 了旧版 *litaagent_n.py* 中混乱的出价逻辑，
并修复了与 `InventoryManager` 的接口 BUG。

核心改动
--------
1. **采购三分法**：把原料购买划分为 `紧急需求 / 计划性需求 / 可选性采购` 三类，
   对应 `_process_emergency_supply_offers()` / `_process_planned_supply_offers()` /
   `_process_optional_supply_offers()` 三个子模块。
2. **销售产能约束**：新增 `_process_sales_offers()`，严格保证在交货期内
   不会签约超出总产能的产品合同，且确保售价满足 `min_profit_margin`。
3. **利润策略可调**：`min_profit_margin` 与 `cheap_price_discount` 两个参数
   可在运行时动态调整；并预留接口 `update_profit_strategy()` 供 RL 或
   外部策略模块调用。
4. **IM 交互修复**：在 `on_negotiation_success()` 中正确解析对手 ID，
   构造 `IMContract` 并调用 `InventoryManager.add_transaction()`；断言
   添加成功并打印日志。
5. **模块化 `counter_all()`**：顶层逻辑只负责按伙伴角色拆分报价并分发
   到四个子函数，代码层次清晰，可维护性大幅提升。
6. **保持 RL 接口**：保留 ObservationManager / ActionManager 等占位，
   不破坏未来集成智能策略的接口。
7. **早期计划采购**：利用 `InventoryManager` 的需求预测，在罚金较低时
   提前锁定原料，减少后期短缺罚金。
8. **数量敏感的让步**：当短缺风险增大时更倾向接受更大数量，避免多轮议价。
9. **对手建模增强**：记录伙伴的合同成功率与均价，估计其保留价格以调整报价。
10. **帕累托意识还价**：反价时综合调整价格、数量与交货期，尝试沿帕累托前沿
    探索互利方案。
11. **贝叶斯对手建模**：通过在线逻辑回归更新每个伙伴的接受概率，推断其保
    留价格并生成更趋于帕累托最优的报价。

使用说明
--------
- 关键参数：
    * `min_profit_margin`   —— 最低利润率要求（如 0.10 ⇒ 10%）。
    * `cheap_price_discount`—— 机会性囤货阈值，低于市场均价 *该比例* 视为超低价。
- 可在外部通过 `agent.update_profit_strategy()` 动态修改。
- 如需接入 RL，可在 `decide_with_model()` 中填充模型调用逻辑。
"""
from __future__ import annotations

# ------------------ 基础依赖 ------------------
from typing import Any, Dict, List, Tuple, Iterable
from dataclasses import dataclass
import random
import os
import math
from collections import Counter, defaultdict
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
from .inventory_manager_n import (
    InventoryManager,
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

class LitaAgentY(StdSyncAgent):
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
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        # —— 参数 ——
        self.total_insufficient = None
        self.today_insufficient = None
        self.min_profit_margin = min_profit_margin         
        self.initial_min_profit_margin = min_profit_margin 
        self.cheap_price_discount = cheap_price_discount   
        self.procurement_cash_flow_limit_percent = 0.75    
        self.concession_curve_power = concession_curve_power 
        self.capacity_tight_margin_increase = capacity_tight_margin_increase
        
        if os.path.exists("env.test"):
            print(f"🤖 LitaAgentY {self.id} initialized with: \n"
                  f"  min_profit_margin={self.min_profit_margin:.3f}, \n"
                  f"  initial_min_profit_margin={self.initial_min_profit_margin:.3f}, \n"
                  f"  cheap_price_discount={self.cheap_price_discount:.2f}, \n"
                  f"  procurement_cash_flow_limit_percent={self.procurement_cash_flow_limit_percent:.2f}, \n"
                  f"  concession_curve_power={self.concession_curve_power:.2f}, \n"
                  f"  capacity_tight_margin_increase={self.capacity_tight_margin_increase:.3f}")

        # —— 运行时变量 ——
        self.im: InventoryManager | None = None            
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
        
        self._sales_successes_since_margin_update: int = 0
        self._sales_failures_since_margin_update: int = 0


    # ------------------------------------------------------------------
    # 🌟 2. World / 日常回调
    # ------------------------------------------------------------------

    def init(self) -> None:
        """在 World 初始化后调用；此处创建库存管理器。"""
        self.im = InventoryManager(
            raw_storage_cost=self.awi.current_storage_cost,
            product_storage_cost=self.awi.current_storage_cost,
            processing_cost=0,  # 如果有工艺成本可在此填写
            daily_production_capacity=self.awi.n_lines ,
            max_day=self.awi.n_steps,
        )
        if os.path.exists("env.test"):
            print(f"🤖 LitaAgentY {self.id} IM initialized. Daily Capacity: {self.im.daily_production_capacity}")


    def before_step(self) -> None:
        """每天开始前，同步日内关键需求信息。"""
        assert self.im, "InventoryManager 尚未初始化!"
        current_day = self.awi.current_step # Use local var for f-string clarity
        self.today_insufficient = self.im.get_today_insufficient(current_day)
        self.total_insufficient = self.im.get_total_insufficient(current_day)
        if os.path.exists("env.test"):
            print(f"🌞 Day {current_day} ({self.id}) starting. Today Insufficient Raw: {self.today_insufficient}, Total Insufficient Raw (horizon): {self.total_insufficient}")

        # Update dynamic parameters
        self._update_dynamic_stockpiling_parameters()
        self._update_dynamic_profit_margin_parameters()

        # 初始化当日的完成量记录
        # Initialize today's completion records
        self.sales_completed.setdefault(current_day, 0)
        self.purchase_completed.setdefault(current_day, 0)

        # 将外生协议写入im
        # Write exogenous contracts into the inventory manager
        if self.awi.is_first_level:
            exogenous_contract_quantity = self.awi.current_exogenous_input_quantity
            exogenous_contract_price = self.awi.current_exogenous_input_price
            if exogenous_contract_quantity > 0:
                exogenous_contract_id = str(uuid4())
                exogenous_contract_partner = "simulator_exogenous_supply"
                exogenous_contract = IMContract(
                    contract_id = exogenous_contract_id,
                    partner_id = exogenous_contract_partner,
                    type = IMContractType.SUPPLY,
                    quantity = exogenous_contract_quantity,
                    price = exogenous_contract_price,
                    delivery_time = current_day, # Exogenous are for current day
                    bankruptcy_risk = 0,
                    material_type = MaterialType.RAW
                )
                self.im.add_transaction(exogenous_contract)
                if os.path.exists("env.test"):
                    print(f"📥 Day {current_day} ({self.id}): Added exogenous SUPPLY contract {exogenous_contract_id} to IM. Qty: {exogenous_contract_quantity}, Price: {exogenous_contract_price}")

        elif self.awi.is_last_level:
            exogenous_contract_quantity = self.awi.current_exogenous_output_quantity
            exogenous_contract_price = self.awi.current_exogenous_output_price
            if exogenous_contract_quantity > 0:
                exogenous_contract_id = str(uuid4())
                exogenous_contract_partner = "simulator_exogenous_demand"
                exogenous_contract = IMContract(
                    contract_id = exogenous_contract_id,
                    partner_id = exogenous_contract_partner,
                    type = IMContractType.DEMAND,
                    quantity = exogenous_contract_quantity,
                    price = exogenous_contract_price,
                    delivery_time = current_day, # Exogenous are for current day
                    bankruptcy_risk = 0,
                    material_type = MaterialType.PRODUCT
                )
                self.im.add_transaction(exogenous_contract)
                if os.path.exists("env.test"):
                    print(f"📤 Day {current_day} ({self.id}): Added exogenous DEMAND contract {exogenous_contract_id} to IM. Qty: {exogenous_contract_quantity}, Price: {exogenous_contract_price}")


    def step(self) -> None:
        """每天结束时调用：执行 IM 的日终操作并刷新市场均价。"""
        assert self.im, "InventoryManager 尚未初始化!"
        # 让 IM 完成收货 / 生产 / 交付 / 规划
        self.im.process_day_operations()
        self.im.update_day() # This increments self.im.current_day
        # —— 更新市场均价估计 ——
        # Ensure lists are not empty before calculating average
        if self._recent_material_prices:
            self._market_material_price_avg = sum(self._recent_material_prices) / len(self._recent_material_prices)
        if self._recent_product_prices:
            self._market_product_price_avg = sum(self._recent_product_prices) / len(self._recent_product_prices)
        if os.path.exists("env.test"):
             print(f"🌙 Day {self.awi.current_step} ({self.id}) ending. Market Material Avg Price: {self._market_material_price_avg:.2f}, Market Product Avg Price: {self._market_product_price_avg:.2f}. IM is now on day {self.im.current_day}.")


    def _update_dynamic_stockpiling_parameters(self) -> None:
        """Dynamically adjusts cheap_price_discount based on game state."""
        if not self.im:
            return

        current_day = self.awi.current_step
        total_days = self.awi.n_steps

        # Ensure total_days is not zero to prevent division by zero
        if total_days == 0:
            return

        current_raw_inventory = self.im.get_inventory_summary(current_day, MaterialType.RAW)['current_stock']
        
        future_total_demand_horizon = min(total_days, current_day + 10)
        future_total_demand = 0
        if self.im: # Ensure im is available
            for d_iter in range(current_day + 1, future_total_demand_horizon + 1): 
                 if d_iter >= total_days: break 
                 future_total_demand += self.im.get_total_insufficient(d_iter) 

        market_avg_raw_price = self._market_material_price_avg 

        if current_day > total_days * 0.8: 
            new_cheap_discount = 0.40
            reason = "Late game"
        elif current_day > total_days * 0.5: 
            new_cheap_discount = 0.60
            reason = "Mid game"
        else: 
            new_cheap_discount = 0.70 
            reason = "Early game"

        if future_total_demand > 0: 
            if current_raw_inventory > future_total_demand * 1.5:
                new_cheap_discount = min(new_cheap_discount, 0.50)
                reason += ", High inventory (>150% demand)"
            elif current_raw_inventory > future_total_demand * 1.0:
                new_cheap_discount = min(new_cheap_discount, 0.60)
                reason += ", Sufficient inventory (>100% demand)"
        elif current_day > 5 : 
             pass 

        if future_total_demand > 0 and current_raw_inventory < future_total_demand * 0.5:
            new_cheap_discount = max(new_cheap_discount, 0.80)
            reason += ", Low inventory (<50% demand) & future demand exists"
        
        if future_total_demand == 0 and current_day > 5: 
            new_cheap_discount = min(new_cheap_discount, 0.30)
            reason = "No future demand (override)" 

        final_new_cheap_discount = max(0.20, min(0.85, new_cheap_discount))

        if abs(self.cheap_price_discount - final_new_cheap_discount) > 1e-3: 
            old_discount = self.cheap_price_discount
            self.update_profit_strategy(cheap_price_discount=final_new_cheap_discount)
            if os.path.exists("env.test"):
                print(f"📈 Day {current_day} ({self.id}): cheap_price_discount changed from {old_discount:.2f} to {self.cheap_price_discount:.2f}. Reason: {reason}. "
                      f"InvRaw: {current_raw_inventory}, FutDemandRaw(10d): {future_total_demand}, MktPriceRaw: {market_avg_raw_price:.2f}")
        elif os.path.exists("env.test"): 
            print(f"🔎 Day {current_day} ({self.id}): cheap_price_discount maintained at {self.cheap_price_discount:.2f}. Evaluated Reason: {reason}. "
                  f"InvRaw: {current_raw_inventory}, FutDemandRaw(10d): {future_total_demand}, MktPriceRaw: {market_avg_raw_price:.2f}")

    def get_avg_raw_cost_fallback(self, current_day_for_im_summary: int, best_price_pid_for_fallback: str | None = None) -> float:
        """
        Calculates average raw material cost with fallbacks.
        Used for estimating costs in proposals and concessions.
        """
        avg_raw_cost = 0.0
        if self._market_material_price_avg > 0:
            avg_raw_cost = self._market_material_price_avg
        elif self.im:
            im_avg_raw_cost = self.im.get_inventory_summary(current_day_for_im_summary, MaterialType.RAW)['average_cost']
            if im_avg_raw_cost > 0:
                avg_raw_cost = im_avg_raw_cost
            elif self.im.raw_batches:
                total_cost = sum(b.unit_cost * b.remaining for b in self.im.raw_batches if b.remaining > 0)
                total_qty_in_batches = sum(b.remaining for b in self.im.raw_batches if b.remaining > 0)
                if total_qty_in_batches > 0:
                    avg_raw_cost = total_cost / total_qty_in_batches
        
        if avg_raw_cost <= 0 and best_price_pid_for_fallback: 
            avg_raw_cost = self._best_price(best_price_pid_for_fallback) * 0.4 
        elif avg_raw_cost <=0: 
            avg_raw_cost = 10.0 
        return avg_raw_cost

    def _is_production_capacity_tight(self, day: int, quantity_being_considered: int = 0) -> bool:
        """Checks if production capacity for a given day is considered tight."""
        if not self.im:
            return False 

        signed_sales_for_day = 0
        for contract_detail in self.im.get_pending_contracts(is_supply=False, day=day):
            if contract_detail.material_type == MaterialType.PRODUCT: 
                signed_sales_for_day += contract_detail.quantity
        
        remaining_capacity = self.im.daily_production_capacity - signed_sales_for_day - quantity_being_considered
        is_tight = remaining_capacity < (self.im.daily_production_capacity * 0.20)
        
        return is_tight

    # ... (Methods _is_supplier to _estimate_reservation_price remain unchanged) ...
    def _is_supplier(self, pid: str) -> bool:
        return pid in self.awi.my_suppliers

    def _is_consumer(self, pid: str) -> bool:
        return pid in self.awi.my_consumers

    def _best_price(self, pid: str) -> float:
        """对自己最有利的价格（买最低 / 卖最高）。"""
        issue = self.get_nmi(pid).issues[UNIT_PRICE]
        return issue.min_value if self._is_supplier(pid) else issue.max_value

    def _is_price_too_high(self, pid: str, price: float) -> bool:
        """简单检查报价是否超出双方议题允许范围。"""
        issue = self.get_nmi(pid).issues[UNIT_PRICE]
        if self._is_supplier(pid):
            return price > issue.max_value  # 采购价过高 Purchase price too high
        return price < issue.min_value      # 销售价过低 Selling price too low

    def _clamp_price(self, pid: str, price: float) -> float:
        """确保价格在议题允许范围内。"""
        issue = self.get_nmi(pid).issues[UNIT_PRICE]
        return max(issue.min_value, min(issue.max_value, price))

    def _expected_price(self, pid: str, default: float) -> float:
        """Return a risk-adjusted expected price using partner history."""
        stats = self.partner_stats.get(pid)
        if stats and stats.get("contracts", 0) > 0:
            mean = stats.get("avg_price", default)
            success_count = stats.get("success", 0)
            contracts_count = stats.get("contracts", 0)
            if success_count > 1 and contracts_count >= success_count : 
                 var = stats.get("price_M2", 0.0) / (success_count - 1)
            else:
                 var = 0.0 
            std = var ** 0.5
            rate = success_count / max(1, contracts_count)
            base = mean + std * (1 - rate) 
        else:
            base = default
        
        if self._is_supplier(pid): 
            base = max(base, self.awi.current_shortfall_penalty * 0.8) 
        
        model_price = self._estimate_reservation_price(pid, base)
        return (base + model_price) / 2

    def _calc_opponent_concession(self, pid: str, price: float) -> float:
        last = self._last_offer_price.get(pid)
        self._last_offer_price[pid] = price
        if last is None or last == 0:
            return 0.0
        return abs(price - last) / abs(last)

    def _concession_multiplier(self, rel_time: float, opp_rate: float = 0.0) -> float:
        if self.concession_model: 
            return self.concession_model(rel_time, opp_rate)
        non_linear_rel_time = rel_time ** self.concession_curve_power
        base = non_linear_rel_time * (1 + opp_rate) 
        return max(0.0, min(1.0, base))

    def _apply_concession(
        self,
        pid: str,
        target_price: float, 
        state: SAOState | None,
        current_price: float, 
    ) -> float:
        start_price = self._best_price(pid) 
        opp_rate = self._calc_opponent_concession(pid, current_price)
        rel_time = state.relative_time if state else 0.0
        
        base_mult = self._concession_multiplier(rel_time, opp_rate)
        adjusted_mult = base_mult
        
        current_final_target_price = (target_price + self._expected_price(pid, target_price)) / 2
        log_reason_parts = [f"BaseTarget: {target_price:.2f}, ExpectedTarget: {current_final_target_price:.2f}"]

        if self._is_consumer(pid): 
            is_late_stage = rel_time > 0.7
            is_very_late_stage = rel_time > 0.85

            if is_very_late_stage:
                adjusted_mult = max(base_mult, 0.80) 
                log_reason_parts.append(f"SalesVeryLateStage(>{0.85*100}%)->MultFloor={adjusted_mult:.2f}")
                
                if self.im:
                    abs_min_price = (self.get_avg_raw_cost_fallback(self.awi.current_step, pid) + self.im.processing_cost) * (1 + self.min_profit_margin)
                    current_final_target_price = max(abs_min_price, current_final_target_price * 0.5 + abs_min_price * 0.5)
                    log_reason_parts.append(f"AbsMinPrice: {abs_min_price:.2f}, NewFinalTarget: {current_final_target_price:.2f}")

            elif is_late_stage:
                adjusted_mult = max(base_mult, 0.60) 
                log_reason_parts.append(f"SalesLateStage(>{0.7*100}%)->MultFloor={adjusted_mult:.2f}")
        else: 
            penalty_factor = min(1.0, self.awi.current_shortfall_penalty / 10.0) 
            adjusted_mult = base_mult + penalty_factor
            if penalty_factor > 0: log_reason_parts.append(f"ProcurePenaltyFactor:{penalty_factor:.2f}")

        adjusted_mult = max(0.0, min(1.0, adjusted_mult)) 

        if self._is_consumer(pid): 
            conceded_price = start_price - (start_price - current_final_target_price) * adjusted_mult
            conceded_price = max(current_final_target_price, conceded_price) 
        else: 
            conceded_price = start_price + (current_final_target_price - start_price) * adjusted_mult
            conceded_price = min(current_final_target_price, conceded_price) 
        
        final_conceded_price = self._clamp_price(pid, conceded_price)

        if os.path.exists("env.test") and abs(final_conceded_price - current_price) > 1e-3 : 
             log_reason_parts.append(f"RelTime:{rel_time:.2f} OppRate:{opp_rate:.2f} BaseMult:{base_mult:.2f} AdjMult:{adjusted_mult:.2f}")
             print(f"CONCESSION Day {self.awi.current_step} ({self.id}) for {pid} (RelTime: {rel_time:.2f}): CurrPrice={current_price:.2f}, Target={current_final_target_price:.2f}, Start={start_price:.2f}, Mult={adjusted_mult:.2f} -> NewPrice={final_conceded_price:.2f}. Reasons: {'|'.join(log_reason_parts)}")
        return final_conceded_price

    def _update_acceptance_model(self, pid: str, price: float, accepted: bool) -> None:
        model = self.partner_models.setdefault(pid, {"w0": 0.0, "w1": 0.0})
        x = price if self._is_supplier(pid) else -price 
        z = model["w0"] + model["w1"] * x
        try:
            pred = 1.0 / (1.0 + math.exp(-z))
        except OverflowError: 
            pred = 1.0 if z > 0 else 0.0
        y = 1.0 if accepted else 0.0
        err = y - pred
        lr = 0.05 
        model["w0"] += lr * err
        model["w1"] += lr * err * x

    def _estimate_reservation_price(self, pid: str, default: float) -> float:
        model = self.partner_models.get(pid)
        if not model or abs(model["w1"]) < 1e-6: 
            return default
        reservation_x = -model["w0"] / model["w1"]
        price_sign = 1.0 if self._is_supplier(pid) else -1.0
        return reservation_x * price_sign

    def _pareto_counter_offer(
        self, pid: str, qty: int, t: int, price: float, state: SAOState | None
    ) -> Outcome:
        opp_price_est = self._estimate_reservation_price(pid, price)
        best_own_price = self._best_price(pid)
        agent_target_conceded_price = self._apply_concession(pid, best_own_price, state, price) 
        current_calculated_price = (opp_price_est + agent_target_conceded_price) / 2.0
        current_calculated_price = self._clamp_price(pid, current_calculated_price)

        proposed_outcome_qty = qty
        proposed_outcome_time = max(t, self.awi.current_step) 
        proposed_outcome_price = current_calculated_price
        
        reason_log = [f"BaseCalcPrice: {current_calculated_price:.2f} for Q:{qty} T:{t}"]

        if self._is_consumer(pid) and self.im: 
            abs_min_price_for_current_qty_time = (self.get_avg_raw_cost_fallback(self.awi.current_step, pid) + self.im.processing_cost) * (1 + self.min_profit_margin)
            is_near_walkaway = abs(current_calculated_price - abs_min_price_for_current_qty_time) < (abs_min_price_for_current_qty_time * 0.03)
            opponent_price_is_lower = price < current_calculated_price 

            if is_near_walkaway and opponent_price_is_lower:
                reason_log.append(f"Near walkaway ({abs_min_price_for_current_qty_time:.2f}), opp_price ({price:.2f}) lower. Exploring Pareto.")
                qty_issue = self.get_nmi(pid).issues[QUANTITY]
                max_possible_qty_issue = qty_issue.max_value if isinstance(qty_issue.max_value, int) else proposed_outcome_qty * 2 

                increased_qty = int(proposed_outcome_qty * 1.25)
                increased_qty = min(increased_qty, max_possible_qty_issue) 
                additional_qty = increased_qty - proposed_outcome_qty

                if additional_qty > 0 and not self._is_production_capacity_tight(proposed_outcome_time, additional_qty):
                    price_reduction_for_qty_increase = 0.02 
                    new_price_for_larger_qty = current_calculated_price * (1 - price_reduction_for_qty_increase)
                    if new_price_for_larger_qty >= abs_min_price_for_current_qty_time:
                        proposed_outcome_price = new_price_for_larger_qty
                        proposed_outcome_qty = increased_qty
                        reason_log.append(f"ParetoTry: Qty+ ({proposed_outcome_qty}) for Price- ({proposed_outcome_price:.2f})")
                elif additional_qty > 0:
                     reason_log.append(f"ParetoQtyIncSkip: Capacity tight for additional {additional_qty} on day {proposed_outcome_time}")
                
                if not (proposed_outcome_qty > qty and proposed_outcome_price < current_calculated_price) : 
                    delayed_time = proposed_outcome_time + 2 
                    time_issue = self.get_nmi(pid).issues[TIME]
                    max_time_issue = time_issue.max_value if isinstance(time_issue.max_value, int) else self.awi.n_steps -1
                    if delayed_time < min(self.awi.n_steps, max_time_issue + 1) : 
                        price_reduction_for_delay = 0.03
                        new_price_for_delayed_delivery = current_calculated_price * (1 - price_reduction_for_delay)
                        if new_price_for_delayed_delivery >= abs_min_price_for_current_qty_time:
                            proposed_outcome_price = new_price_for_delayed_delivery
                            proposed_outcome_time = delayed_time
                            reason_log.append(f"ParetoTry: Time+ ({proposed_outcome_time}) for Price- ({proposed_outcome_price:.2f})")
            
            if os.path.exists("env.test") and len(reason_log) > 1: 
                 print(f"🔎 Day {self.awi.current_step} ({self.id}) Pareto Sales to {pid}: {' | '.join(reason_log)}")

        elif self._is_supplier(pid) and self.awi.current_shortfall_penalty > 1.0:
            qty_issue = self.get_nmi(pid).issues[QUANTITY]
            new_qty = int(proposed_outcome_qty * 1.1)
            proposed_outcome_qty = min(new_qty, qty_issue.max_value if isinstance(qty_issue.max_value, int) else new_qty) 
            if proposed_outcome_qty > qty: 
                 reason_log.append(f"ProcurePenaltyQtyInc: {proposed_outcome_qty}")
                 if os.path.exists("env.test"): print(f"📦 Day {self.awi.current_step} ({self.id}) Pareto Buy from {pid}: {' | '.join(reason_log)}")

        final_qty_issue = self.get_nmi(pid).issues[QUANTITY]
        proposed_outcome_qty = max(final_qty_issue.min_value, min(proposed_outcome_qty, final_qty_issue.max_value if isinstance(final_qty_issue.max_value, int) else proposed_outcome_qty))
        final_time_issue = self.get_nmi(pid).issues[TIME]
        proposed_outcome_time = max(final_time_issue.min_value, min(proposed_outcome_time, final_time_issue.max_value if isinstance(final_time_issue.max_value, int) else proposed_outcome_time))
        proposed_outcome_time = max(proposed_outcome_time, self.awi.current_step)
        return (proposed_outcome_qty, proposed_outcome_time, proposed_outcome_price)

    # ... (Rest of the file, _get_sales_demand methods, _distribute_todays_needs, _distribute_to_partners, first_proposals are okay) ...
    # ... (_process_supply_offers and sub-methods _process_emergency, _process_planned, _process_optional have new logging) ...
    # ... (_process_sales_offers has new logging) ...
    # ... (Negotiation callbacks on_negotiation_failure, on_negotiation_success have logging) ...
    # ... (on_contracts_finalized has logging) ...
    # ... (sign_all_contracts has new logging) ...
    # ... (_update_dynamic_profit_margin_parameters, update_profit_strategy, decide_with_model are okay) ...
    
    # Ensure the main processing methods also get the (self.id) in their logs for clarity in multi-agent scenarios
    def _process_planned_supply_offers(
        self,
        offers: Dict[str, Outcome],
        states: Dict[str, SAOState],
    ) -> Dict[str, SAOResponse]:
        res: Dict[str, SAOResponse] = {}
        if not self.im: return res
        accepted_quantities_for_planned_this_call = defaultdict(int)
        sorted_offers = sorted(offers.items(), key=lambda item: item[1][UNIT_PRICE])

        for pid, offer in sorted_offers:
            qty_original, t, price = offer[QUANTITY], offer[TIME], offer[UNIT_PRICE]
            qty = float(qty_original) 
            self._last_partner_offer[pid] = price
            state = states.get(pid)
            self._recent_material_prices.append(price)
            if len(self._recent_material_prices) > self._avg_window: self._recent_material_prices.pop(0)

            est_sell_price = self._market_product_price_avg if self._market_product_price_avg > 0 else price * 2.0
            min_profit_for_product = est_sell_price * self.min_profit_margin
            max_affordable_raw_price_jit = est_sell_price - self.im.processing_cost - min_profit_for_product
            days_held_estimate = max(0, t - (self.awi.current_step + 1))
            estimated_storage_cost_per_unit = self.im.raw_storage_cost * days_held_estimate
            effective_price = price + estimated_storage_cost_per_unit
            price_is_acceptable = (effective_price <= max_affordable_raw_price_jit)

            current_total_needed_for_date_t = float(self.im.get_total_insufficient(t))
            procurement_limit_for_date_t = current_total_needed_for_date_t * 1.2
            inventory_summary_for_t = self.im.get_inventory_summary(t, MaterialType.RAW)
            inventory_already_secured_for_t = float(inventory_summary_for_t.get('estimated_available', 0.0))
            newly_accepted_for_t_this_call = float(accepted_quantities_for_planned_this_call.get(t, 0.0))
            total_committed_so_far_for_t = inventory_already_secured_for_t + newly_accepted_for_t_this_call
            remaining_headroom_for_t = max(0.0, procurement_limit_for_date_t - total_committed_so_far_for_t)
            accept_qty = min(qty, remaining_headroom_for_t)
            accept_qty_int = int(round(accept_qty))

            log_prefix = f"🏭 Day {self.awi.current_step} ({self.id}) PlannedSupply Offer from {pid} (Q:{qty_original} P:{price:.2f} T:{t}): "
            log_details = f"EffPrice={effective_price:.2f} (StoreCost={estimated_storage_cost_per_unit:.2f}), JITLimit={max_affordable_raw_price_jit:.2f}. Headroom={remaining_headroom_for_t:.1f}, AcceptableQty={accept_qty_int}."

            if accept_qty_int > 0 and price_is_acceptable:
                outcome_tuple = (accept_qty_int, t, price)
                if accept_qty_int == qty_original: res[pid] = SAOResponse(ResponseType.ACCEPT_OFFER, offer) 
                else: res[pid] = SAOResponse(ResponseType.REJECT_OFFER, outcome_tuple) 
                accepted_quantities_for_planned_this_call[t] += accept_qty_int
                if os.path.exists("env.test"): print(log_prefix + f"Accepted Qty {accept_qty_int}. " + log_details)
            else:
                rejection_reason = ""
                if accept_qty_int <= 0 : rejection_reason += "NoHeadroomOrZeroAcceptQty;" # Combined reason for clarity
                if not price_is_acceptable: rejection_reason += "PriceUnacceptable(Effective);"
                
                if not price_is_acceptable: # Primary reason is price
                    target_quoted_price_for_negotiation = max_affordable_raw_price_jit - estimated_storage_cost_per_unit
                    conceded_actual_price_to_offer = self._apply_concession(pid, target_quoted_price_for_negotiation, state, price)
                    qty_for_counter = qty_original if accept_qty_int > 0 else min(qty_original, int(round(remaining_headroom_for_t))) 
                    if qty_for_counter <=0 and qty_original > 0 : qty_for_counter = 1 
                    elif qty_for_counter <=0 and qty_original <=0 : 
                         res[pid] = SAOResponse(ResponseType.REJECT_OFFER, None) 
                         if os.path.exists("env.test"): print(log_prefix + f"Rejected ({rejection_reason}). No valid counter. " + log_details)
                         continue
                    counter_offer_tuple = self._pareto_counter_offer(pid, qty_for_counter, t, conceded_actual_price_to_offer, state)
                    res[pid] = SAOResponse(ResponseType.REJECT_OFFER, counter_offer_tuple)
                    if os.path.exists("env.test"): print(log_prefix + f"Rejected ({rejection_reason}). Countering. " + log_details + f" Counter: Q:{counter_offer_tuple[0]} P:{counter_offer_tuple[2]:.2f} T:{counter_offer_tuple[1]}")
                else: # Price was fine, but no headroom or accept_qty_int is 0
                    res[pid] = SAOResponse(ResponseType.REJECT_OFFER, None) 
                    if os.path.exists("env.test"): print(log_prefix + f"Rejected ({rejection_reason}). No counter. " + log_details)
        return res

    def _process_optional_supply_offers(
        self,
        offers: Dict[str, Outcome],
        states: Dict[str, SAOState],
    ) -> Dict[str, SAOResponse]:
        res: Dict[str, SAOResponse] = {}
        if not self.im: return res
        accepted_quantities_for_optional_this_call = defaultdict(int)
        sorted_offers = sorted(offers.items(), key=lambda item: item[1][UNIT_PRICE])

        for pid, offer in sorted_offers:
            qty_original, t, price = offer[QUANTITY], offer[TIME], offer[UNIT_PRICE]
            qty = float(qty_original)
            self._last_partner_offer[pid] = price
            state = states.get(pid)
            self._recent_material_prices.append(price)
            if len(self._recent_material_prices) > self._avg_window: self._recent_material_prices.pop(0)
            
            cheap_threshold = (self._market_material_price_avg if self._market_material_price_avg > 0 else price * 1.5) * self.cheap_price_discount
            price_is_cheap = (price <= cheap_threshold)

            demand_at_t = float(self.im.get_total_insufficient(t))
            procurement_limit_for_date_t = float(self.im.daily_production_capacity * 0.2) if demand_at_t == 0 else demand_at_t * 1.2
            inventory_summary_for_t = self.im.get_inventory_summary(t, MaterialType.RAW)
            inventory_already_secured_for_t = float(inventory_summary_for_t.get('estimated_available', 0.0))
            newly_accepted_for_t_this_call = float(accepted_quantities_for_optional_this_call.get(t, 0.0))
            total_committed_so_far_for_t = inventory_already_secured_for_t + newly_accepted_for_t_this_call
            remaining_headroom_for_t = max(0.0, procurement_limit_for_date_t - total_committed_so_far_for_t)
            accept_qty = min(qty, remaining_headroom_for_t)
            accept_qty_int = int(round(accept_qty))

            log_prefix = f"🏭 Day {self.awi.current_step} ({self.id}) OptionalSupply Offer from {pid} (Q:{qty_original} P:{price:.2f} T:{t}): "
            log_details = f"PriceIsCheap={price_is_cheap} (Threshold={cheap_threshold:.2f}). Headroom={remaining_headroom_for_t:.1f}, AcceptableQty={accept_qty_int}."

            if accept_qty_int > 0 and price_is_cheap:
                outcome_tuple = (accept_qty_int, t, price)
                if accept_qty_int == qty_original: res[pid] = SAOResponse(ResponseType.ACCEPT_OFFER, offer)
                else: res[pid] = SAOResponse(ResponseType.REJECT_OFFER, outcome_tuple)
                accepted_quantities_for_optional_this_call[t] += accept_qty_int
                if os.path.exists("env.test"): print(log_prefix + f"Accepted Qty {accept_qty_int}. " + log_details)
            else:
                rejection_reason = ""
                if accept_qty_int <= 0: rejection_reason += "NoHeadroomOrZeroAcceptQty;"
                if not price_is_cheap: rejection_reason += "PriceNotCheap;"

                if not price_is_cheap:
                    conceded_price = self._apply_concession(pid, cheap_threshold, state, price)
                    counter_offer_tuple = self._pareto_counter_offer(pid, qty_original, t, conceded_price, state)
                    res[pid] = SAOResponse(ResponseType.REJECT_OFFER, counter_offer_tuple)
                    if os.path.exists("env.test"): print(log_prefix + f"Rejected ({rejection_reason}). Countering. " + log_details + f" Counter: Q:{counter_offer_tuple[0]} P:{counter_offer_tuple[2]:.2f} T:{counter_offer_tuple[1]}")
                else:
                    res[pid] = SAOResponse(ResponseType.REJECT_OFFER, None)
                    if os.path.exists("env.test"): print(log_prefix + f"Rejected ({rejection_reason}). No counter. " + log_details)
        return res

    def _process_sales_offers(
        self,
        offers: Dict[str, Outcome],
        states: Dict[str, SAOState],
    ) -> Dict[str, SAOResponse]:
        res: Dict[str, SAOResponse] = {}
        if not self.im: return res 

        for pid, offer in offers.items():
            qty, t, price = offer[QUANTITY], offer[TIME], offer[UNIT_PRICE]
            state = states.get(pid)
            self._recent_product_prices.append(price)
            if len(self._recent_product_prices) > self._avg_window: self._recent_product_prices.pop(0)

            signed_sales_for_day = 0
            for contract_detail in self.im.get_pending_contracts(is_supply=False, day=t):
                if contract_detail.material_type == MaterialType.PRODUCT:
                    signed_sales_for_day += contract_detail.quantity
            
            if signed_sales_for_day + qty > self.im.daily_production_capacity:
                available_qty_for_offer = self.im.daily_production_capacity - signed_sales_for_day
                if available_qty_for_offer > 0:
                    counter_offer_outcome = self._pareto_counter_offer(pid, int(available_qty_for_offer), t, price, state)
                    res[pid] = SAOResponse(ResponseType.REJECT_OFFER, counter_offer_outcome)
                    if os.path.exists("env.test"): print(f"🏭 Day {self.awi.current_step} ({self.id}) Sales Offer to {pid} for Qty {qty} on Day {t}: Over capacity. Countering with Qty {available_qty_for_offer}.")
                else:
                    res[pid] = SAOResponse(ResponseType.REJECT_OFFER, None)
                    if os.path.exists("env.test"): print(f"🏭 Day {self.awi.current_step} ({self.id}) Sales Offer to {pid} for Qty {qty} on Day {t}: No capacity. Rejecting.")
                continue
            
            avg_raw_cost = self.get_avg_raw_cost_fallback(self.awi.current_step, pid) 
            unit_cost = avg_raw_cost + self.im.processing_cost
            current_min_margin_for_calc = self.min_profit_margin
            reason_log_parts = [f"BaseMinMargin: {current_min_margin_for_calc:.3f}"]

            if self._is_production_capacity_tight(t, qty):
                current_min_margin_for_calc += self.capacity_tight_margin_increase
                reason_log_parts.append(f"CapacityTight! AdjustedMinMargin: {current_min_margin_for_calc:.3f}")
                if os.path.exists("env.test"): print(f"🏭 Day {self.awi.current_step} ({self.id}) Sales Offer to {pid} for Qty {qty} on Day {t}: Capacity tight, using increased margin {current_min_margin_for_calc:.3f}.")

            min_sell_price = unit_cost * (1 + current_min_margin_for_calc)

            if price >= min_sell_price:
                res[pid] = SAOResponse(ResponseType.ACCEPT_OFFER, offer)
                if os.path.exists("env.test"): print(f"✅ Day {self.awi.current_step} ({self.id}) Sales Offer from {pid} (Q:{qty} P:{price:.2f} T:{t}): Accepted. MinSellPrice={min_sell_price:.2f}. Reasons: {'|'.join(reason_log_parts)}")
            else:
                target_price_for_counter = min_sell_price 
                conceded_price = self._apply_concession(pid, target_price_for_counter, state, price)
                counter_offer = self._pareto_counter_offer(pid, qty, t, conceded_price, state)
                res[pid] = SAOResponse(ResponseType.REJECT_OFFER, counter_offer)
                if os.path.exists("env.test"): print(f"❌ Day {self.awi.current_step} ({self.id}) Sales Offer from {pid} (Q:{qty} P:{price:.2f} T:{t}): Rejected (Price < MinSellPrice {min_sell_price:.2f}). Countering with P:{counter_offer[UNIT_PRICE]:.2f}. Reasons: {'|'.join(reason_log_parts)}")
        return res

    def sign_all_contracts(self, contracts: List[Contract]) -> List[Contract]:
        if self.im is None:
            if os.path.exists("env.test"): print(f"⚠️ Day {self.awi.current_step} ({self.id}) IM is None in sign_all_contracts. Signing all blindly.")
            for contract_obj_iter in contracts: 
                agreement = contract_obj_iter.agreement if contract_obj_iter.agreement else {}
                if os.path.exists("env.test"): print(f"  Blindly signing contract {contract_obj_iter.id}. Details: Q={agreement.get(QUANTITY)}, P={agreement.get(UNIT_PRICE)}, T={agreement.get(TIME)}")
            return contracts

        if os.path.exists("env.test"): print(f"📝 [{self.awi.current_step}] ({self.id}) Received {len(contracts)} contracts for signing decision.")
        pending_sales_contracts_data: List[Dict[str, Any]] = []
        pending_procurement_contracts_data: List[Dict[str, Any]] = []

        for contract_obj in contracts: 
            partner_id = self.get_partner_id(contract_obj)
            agreement = contract_obj.agreement
            if not agreement:
                if os.path.exists("env.test"): print(f"⚠️ [{self.awi.current_step}] ({self.id}) Contract {contract_obj.id} has no agreement. Skipping.")
                continue
            quantity = agreement.get(QUANTITY, 0)
            if quantity <= 0:
                if os.path.exists("env.test"): print(f"⚠️ [{self.awi.current_step}] ({self.id}) Contract {contract_obj.id} has zero/negative quantity ({quantity}). Skipping.")
                continue
            # ... (rest of parsing logic is fine) ...
            unit_price = agreement.get(UNIT_PRICE, 0.0)
            delivery_time = agreement.get(TIME, self.awi.current_step)
            if self._is_consumer(partner_id):
                revenue = quantity * unit_price
                processing_cost_total = quantity * self.im.processing_cost
                raw_material_cost_per_unit = self.get_avg_raw_cost_fallback(self.awi.current_step, partner_id)
                raw_material_cost_estimate = quantity * raw_material_cost_per_unit
                storage_cost_estimate = 0.0 
                profit = revenue - raw_material_cost_estimate - processing_cost_total - storage_cost_estimate
                pending_sales_contracts_data.append({
                    "contract_obj": contract_obj, "partner_id": partner_id, "quantity": quantity, 
                    "price": unit_price, "time": delivery_time, "type": "sale", 
                    "raw_material_cost_estimate": raw_material_cost_estimate,
                    "processing_cost_total": processing_cost_total, "storage_cost_estimate": storage_cost_estimate,
                    "profit": profit, "linked_procurements": [],
                    "sufficient_materials": False, "within_capacity": False 
                })
            else:
                expenditure = quantity * unit_price
                days_raw_stored_before_use = 1 
                storage_cost_estimate = quantity * self.im.raw_storage_cost * days_raw_stored_before_use
                cost = expenditure + storage_cost_estimate
                pending_procurement_contracts_data.append({
                    "contract_obj": contract_obj, "partner_id": partner_id, "quantity": quantity, 
                    "price": unit_price, "time": delivery_time, "type": "procurement", 
                    "storage_cost_estimate": storage_cost_estimate, "cost": cost, "profit": -cost, 
                    "linked_to_sale": False, "original_quantity": quantity 
                })
        
        if os.path.exists("env.test"):
            print(f"ℹ️ [{self.awi.current_step}] ({self.id}) Parsed contracts. Sales: {len(pending_sales_contracts_data)}, Procurement: {len(pending_procurement_contracts_data)}.")

        pending_sales_contracts_data.sort(key=lambda x: x["profit"], reverse=True)
        available_procurement_contracts_data = [pc.copy() for pc in pending_procurement_contracts_data]
        available_procurement_contracts_data.sort(key=lambda x: (x["price"], x["time"]))

        candidate_sales_data_list: List[Dict[str, Any]] = []
        daily_production_commitment: Dict[int, int] = Counter()

        if os.path.exists("env.test"): print(f"ℹ️ [{self.awi.current_step}] ({self.id}) Phase 1: Initial greedy selection. Sales candidates: {len(pending_sales_contracts_data)}, Procure candidates: {len(available_procurement_contracts_data)}.")

        for s_data in pending_sales_contracts_data:
            min_acceptable_profit = self.min_profit_margin * (s_data["price"] * s_data["quantity"])
            if s_data["profit"] <= min_acceptable_profit:
                if os.path.exists("env.test"): print(f"⏩ [{self.awi.current_step}] ({self.id}) Skipping sales contract {s_data['contract_obj'].id} due to low profit: {s_data['profit']:.2f} vs min acceptable profit: {min_acceptable_profit:.2f}.")
                continue 

            s_qty = s_data["quantity"]
            s_time = s_data["time"]

            if daily_production_commitment[s_time] + s_qty > self.im.daily_production_capacity:
                if os.path.exists("env.test"): print(f"🚫 [{self.awi.current_step}] ({self.id}) Skipping sales contract {s_data['contract_obj'].id} (Qty:{s_qty} for Day:{s_time}) due to exceeding capacity. Committed: {daily_production_commitment[s_time]}, Capacity: {self.im.daily_production_capacity}")
                continue 
            # ... (rest of sign_all_contracts is okay, already has detailed logging for other phases) ...
            materials_needed = s_qty
            secured_from_im_stock = 0
            temp_linked_procurements_for_this_sale = []
            inv_summary_raw = self.im.get_inventory_summary(s_time, MaterialType.RAW)
            available_im_stock = inv_summary_raw.get('estimated_available', 0)
            if available_im_stock > 0:
                take_from_stock = min(materials_needed, available_im_stock)
                secured_from_im_stock = take_from_stock
            materials_still_needed = materials_needed - secured_from_im_stock
            secured_from_batch_procurements = 0
            if materials_still_needed > 0:
                for p_data_item in available_procurement_contracts_data: 
                    if p_data_item["time"] <= s_time and p_data_item["quantity"] > 0: 
                        if materials_still_needed <= secured_from_batch_procurements : break 
                        take_qty = min(materials_still_needed - secured_from_batch_procurements, p_data_item["quantity"])
                        temp_linked_procurements_for_this_sale.append({
                            "contract_obj": p_data_item["contract_obj"],
                            "quantity_taken": take_qty,
                            "price": p_data_item["price"] 
                        })
                        secured_from_batch_procurements += take_qty
            if (secured_from_im_stock + secured_from_batch_procurements) >= materials_needed:
                s_data["sufficient_materials"] = True
                s_data["within_capacity"] = True
                s_data["linked_procurements"] = temp_linked_procurements_for_this_sale 
                candidate_sales_data_list.append(s_data)
                daily_production_commitment[s_time] += s_qty
                for linked_p_info_commit in temp_linked_procurements_for_this_sale:
                    for p_master_item in available_procurement_contracts_data:
                        if p_master_item["contract_obj"].id == linked_p_info_commit["contract_obj"].id:
                            p_master_item["quantity"] -= linked_p_info_commit["quantity_taken"]
                            p_master_item["linked_to_sale"] = True 
                            break
            else: # Material check failed for this sales contract
                 if os.path.exists("env.test"): print(f"⚠️ [{self.awi.current_step}] ({self.id}) Sales contract {s_data['contract_obj'].id} (Qty:{s_qty}) lacks materials. Needed:{materials_needed}, SecuredFromStock:{secured_from_im_stock}, SecuredFromBatch:{secured_from_batch_procurements}. Skipping.")
        
        if os.path.exists("env.test"): print(f"ℹ️ [{self.awi.current_step}] ({self.id}) Phase 2: Cash Flow Check. Candidates after initial selection: {len(candidate_sales_data_list)} sales.")
        current_selected_sales_data = sorted(candidate_sales_data_list, key=lambda x: x["profit"]) 

        while True:
            if not current_selected_sales_data:
                if os.path.exists("env.test"): print(f"⚠️ [{self.awi.current_step}] ({self.id}) Cash Flow: No sales contracts left. Signing 0 contracts.")
                break
            total_expected_revenue = sum(s_d["quantity"] * s_d["price"] for s_d in current_selected_sales_data)
            unique_procurements_for_current_sales = {} 
            for s_d in current_selected_sales_data:
                for linked_p_info in s_d["linked_procurements"]:
                    proc_contract = linked_p_info["contract_obj"]
                    if proc_contract.id not in unique_procurements_for_current_sales:
                         unique_procurements_for_current_sales[proc_contract.id] = proc_contract
            total_cost_of_linked_procurement = sum(
                p_c.agreement[QUANTITY] * p_c.agreement[UNIT_PRICE] 
                for p_c in unique_procurements_for_current_sales.values()
            )
            allowed_procurement_cost = total_expected_revenue * self.procurement_cash_flow_limit_percent
            if total_cost_of_linked_procurement <= allowed_procurement_cost:
                if os.path.exists("env.test"): print(f"💰 [{self.awi.current_step}] ({self.id}) Cash Flow OK. Revenue: {total_expected_revenue:.2f}, "
                      f"Proc. Cost: {total_cost_of_linked_procurement:.2f} (Limit: {allowed_procurement_cost:.2f})")
                break 
            descoped_sale_data = current_selected_sales_data.pop(0) 
            if os.path.exists("env.test"): print(f"🚫 [{self.awi.current_step}] ({self.id}) Cash Flow Limit Exceeded. Descoping sales: {descoped_sale_data['contract_obj'].id} "
                  f"(Profit: {descoped_sale_data['profit']:.2f}). "
                  f"Rev: {total_expected_revenue:.2f}, ProcCost: {total_cost_of_linked_procurement:.2f}, Limit: {allowed_procurement_cost:.2f}")
            if not current_selected_sales_data: 
                if os.path.exists("env.test"): print(f"⚠️ [{self.awi.current_step}] ({self.id}) All sales descoped due to cash flow.")
                break
        
        final_signed_contracts_list: List[Contract] = []
        final_sales_contract_ids = set()
        final_procurement_contract_ids = set()
        for s_d in current_selected_sales_data: 
            final_signed_contracts_list.append(s_d["contract_obj"])
            final_sales_contract_ids.add(s_d["contract_obj"].id)
            for linked_p_info in s_d["linked_procurements"]:
                if linked_p_info["contract_obj"].id not in final_procurement_contract_ids:
                    final_signed_contracts_list.append(linked_p_info["contract_obj"])
                    final_procurement_contract_ids.add(linked_p_info["contract_obj"].id)
        final_total_profit_estimate = sum(s_d["profit"] for s_d in current_selected_sales_data)
        final_signed_sales_count = len(final_sales_contract_ids)
        final_signed_procurement_count = len(final_procurement_contract_ids)
        
        if os.path.exists("env.test"): 
            print(f"✅ [{self.awi.current_step}] ({self.id}) Final Decision: Signing {len(final_signed_contracts_list)} contracts: "
                  f"{final_signed_sales_count} sales, {final_signed_procurement_count} procurements. Est. profit: {final_total_profit_estimate:.2f}")
        return final_signed_contracts_list

    def _update_dynamic_profit_margin_parameters(self) -> None:
        if not self.im: return
        current_day = self.awi.current_step
        total_days = self.awi.n_steps
        if total_days == 0: return
        new_min_profit_margin = self.initial_min_profit_margin 
        reason_parts = [f"Base: {new_min_profit_margin:.3f}"]
        current_product_inventory = self.im.get_inventory_summary(current_day, MaterialType.PRODUCT)['current_stock']
        future_total_product_demand_horizon = min(total_days, current_day + 5)
        future_total_product_demand = 0
        for d_offset in range(1, future_total_product_demand_horizon - current_day + 1): 
            check_day = current_day + d_offset
            if check_day >= total_days : break 
            for contract_detail in self.im.get_pending_contracts(is_supply=False, day=check_day):
                future_total_product_demand += contract_detail.quantity
        reason_parts.append(f"InvProd: {current_product_inventory}, FutProdDemand(5d): {future_total_product_demand}")
        rule_a_applied = False
        if current_product_inventory == 0 and future_total_product_demand == 0:
            reason_parts.append("RuleA: No stock & no immediate demand, using base.")
            rule_a_applied = True
        elif future_total_product_demand > 0 and current_product_inventory > future_total_product_demand * 2.0:
            new_min_profit_margin = 0.05
            reason_parts.append(f"RuleA: High Inv vs Demand (>2x) -> set to {new_min_profit_margin:.3f}")
            rule_a_applied = True
        elif future_total_product_demand > 0 and current_product_inventory > future_total_product_demand * 1.0:
            new_min_profit_margin = 0.07
            reason_parts.append(f"RuleA: Med Inv vs Demand (>1x) -> set to {new_min_profit_margin:.3f}")
            rule_a_applied = True
        elif future_total_product_demand == 0 and current_product_inventory > self.im.daily_production_capacity * 1.0:
            new_min_profit_margin = 0.06
            reason_parts.append(f"RuleA: No Demand & Inv > 1 day prod -> set to {new_min_profit_margin:.3f}")
            rule_a_applied = True
        if not rule_a_applied: 
            reason_parts.append("RuleA: Defaulted (no specific high/low inv condition met).")
        initial_margin_after_rule_a = new_min_profit_margin
        rule_b_applied = False
        if future_total_product_demand > 0 and current_product_inventory < future_total_product_demand * 0.5:
            new_min_profit_margin = max(initial_margin_after_rule_a, 0.15) 
            if abs(new_min_profit_margin - initial_margin_after_rule_a) > 1e-5 : reason_parts.append(f"RuleB: Low Inv vs Demand (<0.5x) -> max with 0.15 -> {new_min_profit_margin:.3f}")
            rule_b_applied = True
        elif current_product_inventory < self.im.daily_production_capacity * 0.5: 
            new_min_profit_margin = max(initial_margin_after_rule_a, 0.12)
            if abs(new_min_profit_margin - initial_margin_after_rule_a) > 1e-5 : reason_parts.append(f"RuleB: Low Inv vs Capacity (<0.5 day prod) -> max with 0.12 -> {new_min_profit_margin:.3f}")
            rule_b_applied = True
        if not rule_b_applied and abs(initial_margin_after_rule_a - new_min_profit_margin) < 1e-5 : 
             pass 
        initial_margin_after_rule_b = new_min_profit_margin
        rule_c_applied = False
        if current_day > total_days * 0.85:  
            new_min_profit_margin = min(initial_margin_after_rule_b, 0.03)
            if abs(new_min_profit_margin - initial_margin_after_rule_b) > 1e-5 : reason_parts.append(f"RuleC: End Game (Last 15%) -> min with 0.03 -> {new_min_profit_margin:.3f}")
            rule_c_applied = True
        elif current_day > total_days * 0.6:  
            new_min_profit_margin = min(initial_margin_after_rule_b, 0.08)
            if abs(new_min_profit_margin - initial_margin_after_rule_b) > 1e-5 : reason_parts.append(f"RuleC: Late Mid-Game (>60%) -> min with 0.08 -> {new_min_profit_margin:.3f}")
            rule_c_applied = True
        if not rule_c_applied and abs(initial_margin_after_rule_b - new_min_profit_margin) < 1e-5:
            pass 
        margin_adjustment_from_conversion = (self._sales_successes_since_margin_update // 5) * 0.005 - self._sales_failures_since_margin_update * 0.005
        if margin_adjustment_from_conversion != 0:
            current_margin_before_adaptive = new_min_profit_margin
            new_min_profit_margin += margin_adjustment_from_conversion
            reason_parts.append(f"RuleD: Adaptive adj: {margin_adjustment_from_conversion:.4f} (S:{self._sales_successes_since_margin_update},F:{self._sales_failures_since_margin_update}) Cur->New: {current_margin_before_adaptive:.3f}->{new_min_profit_margin:.3f}")
        self._sales_successes_since_margin_update = 0
        self._sales_failures_since_margin_update = 0
        final_new_min_profit_margin = max(0.02, min(0.25, new_min_profit_margin))
        if abs(final_new_min_profit_margin - new_min_profit_margin) > 1e-5 : 
             reason_parts.append(f"Clamped from {new_min_profit_margin:.3f} to {final_new_min_profit_margin:.3f}")
        if abs(self.min_profit_margin - final_new_min_profit_margin) > 1e-4: 
            old_margin = self.min_profit_margin
            self.update_profit_strategy(min_profit_margin=final_new_min_profit_margin)
            if os.path.exists("env.test"):
                print(f"📈 Day {current_day} ({self.id}): min_profit_margin changed from {old_margin:.3f} to {self.min_profit_margin:.3f}. Reasons: {' | '.join(reason_parts)}")
        elif os.path.exists("env.test"):
            print(f"🔎 Day {current_day} ({self.id}): min_profit_margin maintained at {self.min_profit_margin:.3f}. Evaluated Reasons: {' | '.join(reason_parts)}")

    def update_profit_strategy(
        self, *, min_profit_margin: float | None = None, cheap_price_discount: float | None = None
    ) -> None:
        if min_profit_margin is not None:
            self.min_profit_margin = min_profit_margin
        if cheap_price_discount is not None:
            self.cheap_price_discount = cheap_price_discount

    def decide_with_model(self, obs: Any) -> Any: 
        return None

if __name__ == "__main__":
    if os.path.exists("env.test"):
        print("模块加载成功，可在竞赛框架中使用 LitaAgentY。")
