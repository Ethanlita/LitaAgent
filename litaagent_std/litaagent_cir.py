from __future__ import annotations
from __future__ import annotations

"""
LitaAgentCIR — 库存敏感型统一策略（SDK 对接版）
=================================================

* 依托 `inventory_manager_n` **现有接口**（无 `has_capacity` 等自带方法），通过辅助函数 `_has_capacity/_has_budget/_calc_target_price` 实现原期望能力。
* 价格惩罚与市场波动使用简化逻辑（暂无全局市场统计时返回 0）。
* 兼容 `scml>=0.3.0`，无 `sign_all_contracts()`。
"""

from dataclasses import dataclass
from copy import deepcopy
from typing import Dict, List, Sequence, Tuple, Optional # Added Optional
import numpy as np

# from inventory_manager_n import InventoryManager, MaterialType  # type: ignore
from .custom_inventory_manager import CustomInventoryManager, IMContract, IMContractType, MaterialType

# ----------------------------------------------------
# Bayesian 罚金 / 机会成本模型
# ----------------------------------------------------
@dataclass
class BetaDist:
    a: float = 2.0
    b: float = 2.0
    def mean(self):
        return self.a / (self.a + self.b)
    def update(self, flag: bool):
        if flag:
            self.a += 1
        else:
            self.b += 1

@dataclass
class Offer(tuple):
    """简易占位: (quantity, time, price)"""
    def __new__(cls, data):
        return tuple.__new__(cls, data)
    @property
    def quantity(self):
        return self[0]
    @property
    def time(self):
        return self[1]
    @property
    def price(self):
        return self[2]

class BayesPenaltyModel:
    def __init__(self):
        self.penalty = BetaDist(2, 2)
        self.storage = BetaDist(2, 6)
    def observe_shortage(self, happened: bool):
        self.penalty.update(happened)
    def observe_overstock(self, happened: bool):
        self.storage.update(happened)
    def current_weights(self) -> Tuple[float, float]:
        return 10 * self.penalty.mean(), 4 * self.storage.mean()


# ------------------ 基础依赖 ------------------
from typing import Any, Dict, List, Tuple, Iterable, Optional # Added Optional
from dataclasses import dataclass
from itertools import combinations as iter_combinations # Added for combinations
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

class StockScorer:
    def __init__(self, im: InventoryManager, bayes: BayesPenaltyModel, today: int, window: int = 14):
        self.im, self.bayes, self.today, self.window = im, bayes, today, window

    def _calc_raw_over(self, im, day):
        raw_avail = im.get_inventory_summary(day, MaterialType.RAW)["estimated_available"]
        raw_need = im.get_total_insufficient(day)  # 需在 IM 中已有或补充方法
        return max(0, raw_avail - raw_need)

# ----------------------------------------------------
# λ_price (无市场波动)
# ----------------------------------------------------

def lambda_price(rel_t: float, deal_rate: float, λ0=0.1, α=0.5):
    return λ0 * (1 - α * deal_rate)

# ------------------ 主代理实现 ------------------
# Main agent implementation

class LitaAgentCIR(StdSyncAgent):
    """重构后的 LitaAgent CIR。支持三类采购策略与产能约束销售。"""

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
        p_threshold: float = 0.7, # Threshold for combined score
        q_threshold: float = 0.0, # Threshold for individual norm_profit (unused in current logic directly, but for future)
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        # —— 参数 ——
        self.bayes = None
        self.λ0 = None
        self.total_insufficient = None
        self.today_insufficient = None
        self.min_profit_margin = min_profit_margin         
        self.initial_min_profit_margin = min_profit_margin # Added from Step 7
        self.cheap_price_discount = cheap_price_discount   
        self.procurement_cash_flow_limit_percent = procurement_cash_flow_limit_percent # Added from Step 6
        self.concession_curve_power = concession_curve_power # Added from Step 9.b
        self.capacity_tight_margin_increase = capacity_tight_margin_increase # Added from Step 9.d
        self.p_threshold = p_threshold
        self.q_threshold = q_threshold
        
        if os.path.exists("env.test"): # Added from Step 11
            print(f"🤖 LitaAgentY {self.id} initialized with: \n"
                  f"  min_profit_margin={self.min_profit_margin:.3f}, \n"
                  f"  initial_min_profit_margin={self.initial_min_profit_margin:.3f}, \n"
                  f"  cheap_price_discount={self.cheap_price_discount:.2f}, \n"
                  f"  procurement_cash_flow_limit_percent={self.procurement_cash_flow_limit_percent:.2f}, \n"
                  f"  concession_curve_power={self.concession_curve_power:.2f}, \n"
                  f"  capacity_tight_margin_increase={self.capacity_tight_margin_increase:.3f}\n"
                  f"  p_threshold={self.p_threshold:.2f}, q_threshold={self.q_threshold:.2f}")

        # —— 运行时变量 ——
        self.im: Optional[CustomInventoryManager] = None # Updated type hint
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
        # Determine processing_cost
        processing_cost = 0.0
        # SCML structure: awi.profile.costs is a list of costs for each process.
        # If agent produces output product at index `my_output_product_idx`
        # and its input is at `my_input_product_idx`, the cost is usually related to these.
        # Assuming a single-step production for simplicity or that profile.costs[0] is relevant.
        # A more robust way: self.awi.profile.costs[self.awi.my_output_product_idx -1] if output is not the first product
        # or self.awi.profile.processes[self.awi.my_output_process_idx].cost
        # Using self.awi.profile.manufacturing_cost if available, else a default.
        if hasattr(self.awi.profile, 'manufacturing_cost'): # SCML >= 0.7.x
            processing_cost = self.awi.profile.manufacturing_cost
        elif hasattr(self.awi.profile, 'process_cost_per_unit'): # Older SCML
             processing_cost = self.awi.profile.process_cost_per_unit
        elif self.awi.profile.costs and len(self.awi.profile.costs) > 0: # Fallback to first cost
            processing_cost = self.awi.profile.costs[0]
        else: # Ultimate fallback
            processing_cost = 0 # Or some other default like 1.0
            if os.path.exists("env.test"):
                print(f"Warning ({self.id}): Could not determine processing_cost from AWI profile. Using {processing_cost}.")

        daily_capacity = self.awi.n_lines * self.awi.profile.line_capacity if hasattr(self.awi.profile, 'line_capacity') else self.awi.n_lines
        if hasattr(self.awi.profile, 'max_production_per_day'): # SCML >=0.7.x
            daily_capacity = self.awi.profile.max_production_per_day


        self.im = CustomInventoryManager(
            raw_storage_cost=self.awi.current_storage_cost, # Assuming same cost for raw and product
            product_storage_cost=self.awi.current_storage_cost,
            processing_cost=processing_cost,
            daily_production_capacity=daily_capacity,
            max_simulation_day=self.awi.n_steps,
            current_day=self.awi.current_step 
        )
        self.bayes = BayesPenaltyModel()
        self.λ0 = 0.1
        if os.path.exists("env.test"): 
            print(f"🤖 {self.id} CustomIM initialized. Daily Capacity: {self.im.daily_production_capacity}, Processing Cost: {self.im.processing_cost_per_unit}, Current Day (IM): {self.im.current_day}")


    def before_step(self) -> None:
        """每天开始前，同步日内关键需求信息。"""
        assert self.im, "CustomInventoryManager 尚未初始化!"
        current_day = self.awi.current_step 
        self.today_insufficient = self.im.get_today_insufficient_raw(current_day)
        self.total_insufficient = self.im.get_total_insufficient_raw(current_day, horizon=14) # Default horizon 14 days
        if os.path.exists("env.test"): 
            print(f"🌞 Day {current_day} ({self.id}) starting. CIM Day: {self.im.current_day}. Today Insufficient Raw: {self.today_insufficient}, Total Insufficient Raw (14d): {self.total_insufficient}")


        # 初始化当日的完成量记录
        # Initialize today's completion records
        self.sales_completed.setdefault(current_day, 0)
        self.purchase_completed.setdefault(current_day, 0)

        # 将外生协议写入im
        # Write exogenous contracts into the inventory manager
        if self.awi.is_first_level:
            exogenous_contract_quantity = self.awi.current_exogenous_input_quantity
            exogenous_contract_price = self.awi.current_exogenous_input_price
            if exogenous_contract_quantity > 0: # Added from Step 11
                exogenous_contract_id = str(uuid4())
                exogenous_contract_partner = "simulator_exogenous_supply" # More specific name
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
                if os.path.exists("env.test"): # Added from Step 11
                    print(f"📥 Day {current_day} ({self.id}): Added exogenous SUPPLY contract {exogenous_contract_id} to IM. Qty: {exogenous_contract_quantity}, Price: {exogenous_contract_price}")

        elif self.awi.is_last_level:
            exogenous_contract_quantity = self.awi.current_exogenous_output_quantity
            exogenous_contract_price = self.awi.current_exogenous_output_price
            if exogenous_contract_quantity > 0: # Added from Step 11
                exogenous_contract_id = str(uuid4())
                exogenous_contract_partner = "simulator_exogenous_demand" # More specific name
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
                if os.path.exists("env.test"): # Added from Step 11
                    print(f"📤 Day {current_day} ({self.id}): Added exogenous DEMAND contract {exogenous_contract_id} to IM. Qty: {exogenous_contract_quantity}, Price: {exogenous_contract_price}")


    def step(self) -> None:
        """每天结束时调用：执行 IM 的日终操作并刷新市场均价。"""
        assert self.im, "CustomInventoryManager 尚未初始化!"
        # 让 IM 完成收货 / 生产 / 交付 / 规划
        # CustomInventoryManager.process_day_end_operations advances its own current_day
        result = self.im.process_day_end_operations(self.awi.current_step)
        # self.im.update_day() # This is no longer needed.
        # —— 更新市场均价估计 ——
        # Ensure lists are not empty before calculating average
        if self._recent_material_prices:
            self._market_material_price_avg = sum(self._recent_material_prices) / len(self._recent_material_prices)
        if self._recent_product_prices:
            self._market_product_price_avg = sum(self._recent_product_prices) / len(self._recent_product_prices)
        if os.path.exists("env.test"): # Added from Step 11
             print(f"🌙 Day {self.awi.current_step} ({self.id}) ending. Market Material Avg Price: {self._market_material_price_avg:.2f}, Market Product Avg Price: {self._market_product_price_avg:.2f}. IM is now on day {self.im.current_day}.")
        
        # 输出每日状态报告
        self._print_daily_status_report(result)


    # ------------------------------------------------------------------
    # 🌟 3. 价格工具
    # Pricing utilities
    # ------------------------------------------------------------------

    def _is_supplier(self, pid: str) -> bool:
        return pid in self.awi.my_suppliers

    def _is_consumer(self, pid: str) -> bool:
        return pid in self.awi.my_consumers

    # ------------------------------------------------------------------
    # 🌟 3-a. 需求计算和需求分配
    # ------------------------------------------------------------------

    def _distribute_todays_needs(self, partners: Iterable[str] | None = None) -> Dict[str, int]:
        if partners is None:
            partners = self.negotiators.keys()
        partners = list(partners)
        response: Dict[str, int] = {p: 0 for p in partners}
        if not self.im : return response

        suppliers = [p for p in partners if self._is_supplier(p)]
        consumers = [p for p in partners if self._is_consumer(p)]

        total_buy_need = self.im.get_total_insufficient(self.awi.current_step)
        # Simplified sell_need: using available products
        total_sell_need = self.im.get_inventory_summary(self.awi.current_step, MaterialType.PRODUCT)["estimated_available"]
        
        if suppliers and total_buy_need > 0:
            response.update(self._distribute_to_partners(suppliers, total_buy_need))
        if consumers and total_sell_need > 0:
            response.update(self._distribute_to_partners(consumers, int(total_sell_need))) # Ensure int
        return response

    def _distribute_to_partners(self, partners: List[str], needs: int) -> Dict[str, int]:
        if needs <= 0 or not partners:
            return {p: 0 for p in partners}
        needs = int(needs)
        partners.sort(
            key=lambda p: self.partner_stats.get(p, {}).get("success", 0)
            / max(1, self.partner_stats.get(p, {}).get("contracts", 0)),
            reverse=True,
        )
        k = max(1, int(len(partners) * self._ptoday))
        k = min(k, len(partners))
        if needs < k:
             chosen_for_small_needs = random.sample(partners[:k], needs) 
             quantities = [1] * needs
             distribution = dict(zip(chosen_for_small_needs, quantities))
        else:
            chosen = partners[:k] 
            quantities = _distribute(needs, len(chosen))
            distribution = dict(zip(chosen, quantities))
        final_distribution = {p:0 for p in partners}
        final_distribution.update(distribution)
        return final_distribution

    # ------------------------------------------------------------------
    # 🌟 4. first_proposals — 首轮报价（可简化）
    # ------------------------------------------------------------------
    # Modified in Step 9.a (Turn 28) & 9.d (Turn 35)
    def first_proposals(self) -> Dict[str, Outcome]:
        partners = list(self.negotiators.keys())
        if not partners: 
            return {}
        
        # Filter partners based on layer (no buying for first layer, no selling for last layer)
        # This logic can be kept as it's independent of the deleted demand functions
        filtered_partners: List[str] = []
        for pid in partners:
            if self._is_supplier(pid) and self.awi.is_first_level:
                continue
            if self._is_consumer(pid) and self.awi.is_last_level:
                continue
            filtered_partners.append(pid)

        if not filtered_partners:
            return {}

        # Use the simplified _distribute_todays_needs
        distribution = self._distribute_todays_needs(filtered_partners)
        today = self.awi.current_step
        proposals: Dict[str, Outcome] = {}

        for pid, qty in distribution.items():
            if qty <= 0:
                continue
            
            nmi = self.get_nmi(pid) # Assuming get_nmi is an existing method
            # Determine price: min for buying (from supplier), max for selling (to consumer)
            # This logic is standard and can be kept
            price = nmi.issues[UNIT_PRICE].min_value if self._is_supplier(pid) else nmi.issues[UNIT_PRICE].max_value
            
            proposals[pid] = (qty, today, price) # Create proposal (quantity, time, price)

        return proposals

    # ------------------------------------------------------------------
    # 🌟 5. counter_all — 谈判核心（分派到子模块）
    # ------------------------------------------------------------------
    def score_offers(
        self,
        offers_to_evaluate: List[Tuple[str, Outcome]], # list of (negotiator_id, offer_outcome)
        current_im: CustomInventoryManager,
        awi: StdAWI, # For current_day, storage_costs etc.
        bayes_model: BayesPenaltyModel
    ) -> List[float]:
        today = awi.current_step
        horizon_days = 14  # Scoring window
        w_short, w_over = bayes_model.current_weights()

        offer_scores = []

        # Baseline Inventory Assessment
        # get_total_insufficient_raw(target_day, horizon) sums shortfalls from target_day up to target_day + horizon -1
        shortage_before_total = current_im.get_total_insufficient_raw(today, horizon=horizon_days) 
        urgent_shortage_today = current_im.get_today_insufficient_raw(today)
        
        product_surplus_before = 0.0
        for d_offset in range(horizon_days):
            day_in_horizon = today + d_offset
            product_summary = current_im.get_inventory_summary(day_in_horizon, MaterialType.PRODUCT)
            product_surplus_before += product_summary.get('estimated_available', 0.0)
        
        # Heuristic for normalization scale: sum of max possible shortage and surplus values over horizon
        # This is a rough guide. If daily_production_capacity is inf, this might be problematic.
        # Consider a large constant or a sum of all potential demands if available.
        # For now, let's use daily_production_capacity * horizon_days as a proxy for "significant quantity"
        max_impact_qty_heuristic = current_im.daily_production_capacity * horizon_days if current_im.daily_production_capacity != float('inf') else 10000 # Large fallback
        if max_impact_qty_heuristic == 0: max_impact_qty_heuristic = 1000 # Ensure not zero

        max_raw_score_scale = (w_short * max_impact_qty_heuristic) + (w_over * max_impact_qty_heuristic)
        if max_raw_score_scale == 0: # Avoid division by zero if weights are zero
            max_raw_score_scale = 1.0


        for negotiator_id, offer_outcome in offers_to_evaluate:
            quantity, time, unit_price = offer_outcome
            # is_seller_outcome: True if the offer is from a supplier (agent is BUYING raw material)
            #                    False if the offer is from a consumer (agent is SELLING product, this means partner is buying)
            # This was confusing. Let's rename: is_supply_contract_for_agent
            is_supply_contract_for_agent = self._is_supplier(negotiator_id) 

            sim_im = current_im.deepcopy()
            
            contract_type = IMContractType.SUPPLY if is_supply_contract_for_agent else IMContractType.DEMAND
            material_type = MaterialType.RAW if is_supply_contract_for_agent else MaterialType.PRODUCT
            
            contract = IMContract(
                contract_id=str(uuid4()), 
                partner_id=negotiator_id,
                type=contract_type,
                quantity=quantity,
                price=unit_price,
                delivery_time=time,
                material_type=material_type,
                bankruptcy_risk=0 # Assuming no risk for simulation
            )
            sim_im.add_transaction(contract) # This also triggers sim_im.plan_production()

            # Post-Offer Inventory Assessment
            shortage_after_total = sim_im.get_total_insufficient_raw(today, horizon=horizon_days)
            urgent_shortage_after_today = sim_im.get_today_insufficient_raw(today)
            
            product_surplus_after = 0.0
            for d_offset in range(horizon_days):
                day_in_horizon = today + d_offset
                product_summary_after = sim_im.get_inventory_summary(day_in_horizon, MaterialType.PRODUCT)
                product_surplus_after += product_summary_after.get('estimated_available', 0.0)

            # Calculate Score Components
            # a. Inventory Balance Score (IBS)
            delta_shortage = shortage_before_total - shortage_after_total # Positive is good
            delta_product_surplus = product_surplus_before - product_surplus_after # Positive is good (reduced surplus)
            
            # raw_score_ibs: Higher is better.
            # Penalize increase in product surplus (delta_product_surplus is negative) with 0.1 factor.
            # Benefit from decrease in product surplus (delta_product_surplus is positive) fully.
            penalty_for_surplus_increase = 0.0
            benefit_from_surplus_decrease = 0.0

            if delta_product_surplus < 0: # Product surplus increased
                penalty_for_surplus_increase = -delta_product_surplus * 0.1 # -delta is positive, then apply penalty factor
            else: # Product surplus decreased or stayed same
                benefit_from_surplus_decrease = delta_product_surplus # Full benefit for reduction

            raw_score_ibs = (w_short * delta_shortage) + (w_over * benefit_from_surplus_decrease) - (w_over * penalty_for_surplus_increase)
            
            # Normalization of IBS to [0,1] using sigmoid-like scaling centered at 0.5
            norm_ibs = 0.5 + 0.5 * (raw_score_ibs / max_raw_score_scale)
            norm_ibs = max(0.0, min(1.0, norm_ibs)) # Clamp to [0,1]

            # b. Time-based Bonus (TBB) - Placeholder
            tbb = 0.0 # Not implemented as per requirements

            # c. Urgency Bonus (UB)
            ub = 0.0
            if urgent_shortage_today > 0:
                urgent_need_addressed = urgent_shortage_today - urgent_shortage_after_today
                if urgent_need_addressed > 0:
                    ub = (urgent_need_addressed / urgent_shortage_today) * 0.20 # Max 20% contribution
            ub = max(0.0, min(ub, 0.20)) # Clamp UB
            
            # Combine Scores
            # Final score is primarily IBS, with UB providing a bonus on top of IBS's normalized value.
            # Let IBS range from 0 to (1-max_UB_potential=0.8), then add UB.
            # final_score = (norm_ibs * (1.0 - 0.20)) + ub 
            # OR as specified: norm_ibs * (1-ub) + ub, which gives UB more weight if norm_ibs is low.
            final_score = norm_ibs * (1.0 - ub) + ub # Let's stick to the prompt's formula
            final_score = max(0.0, min(1.0, final_score)) # Clamp to [0,1]

            offer_scores.append(final_score)
            
            if os.path.exists("env.test"):
                print(f"Offer Eval ({self.id} @ {today}): NegID={negotiator_id}, Qty={quantity}, Prc={unit_price}, Time={time}, IsSupplyContract={is_supply_contract_for_agent}\n"
                      f"  InvBefore: ShrtTot={shortage_before_total:.2f}, UrgentShrt={urgent_shortage_today:.2f}, ProdSurp={product_surplus_before:.2f}\n"
                      f"  InvAfter : ShrtTot={shortage_after_total:.2f}, UrgentShrt={urgent_shortage_after_today:.2f}, ProdSurp={product_surplus_after:.2f}\n"
                      f"  Deltas   : Shrt={delta_shortage:.2f}, ProdSurp={delta_product_surplus:.2f}\n"
                      f"  Scores   : IBS_raw={raw_score_ibs:.2f}, IBS_norm={norm_ibs:.3f}, UB={ub:.3f}, Final={final_score:.3f}")


        return offer_scores

    def counter_all(
        self,
        offers: Dict[str, Outcome],
        states: Dict[str, SAOState],
    ) -> Dict[str, SAOResponse]:
        responses: Dict[str, SAOResponse] = {}

        if not offers:
            return responses

        # Prepare offers for scoring
        offers_to_evaluate = []
        for nid, outcome in offers.items():
            if outcome: # Ensure outcome is not None
                 offers_to_evaluate.append((nid, outcome))
            else: # Handle cases where an offer might be None (e.g. initial empty offers)
                if os.path.exists("env.test"):
                    print(f"Warning: Received None outcome for negotiator {nid} in counter_all.")


        if not offers_to_evaluate: # If all offers were None or empty
            for nid in offers.keys(): # Still need to respond, typically reject
                 responses[nid] = SAOResponse(ResponseType.REJECT_OFFER, None)
            return responses
            
        # Score offers
        # Ensure self.im, self.awi, self.bayes are available and correctly passed
        if not self.im or not self.awi or not self.bayes:
            if os.path.exists("env.test"):
                print("Error: IM, AWI, or Bayes model not initialized. Cannot score offers.")
            # Fallback: reject all if critical components are missing
            for nid, offer_outcome_tuple in offers_to_evaluate:
                responses[nid] = SAOResponse(ResponseType.REJECT_OFFER, None)
            return responses

        scores = self.score_offers(offers_to_evaluate, self.im, self.awi, self.bayes)

        if os.path.exists("env.test"):
            print(f"Scores for offers at step {self.awi.current_step}: {scores}")

        # TODO: Use scores for selection and decision logic (Accept, Reject, Counter)
        # For now, reject all offers (as per previous placeholder)
        # This part will be implemented in subsequent steps.
        # For now, this method is used by _evaluate_offer_combinations.
        # The direct call from counter_all to score_offers will be replaced.
        pass # score_offers is now primarily a helper for _evaluate_offer_combinations

    def _calculate_combination_profit(self, combination: List[Tuple[str, Outcome]]) -> float:
        """Calculates the simple profit for a combination of offers."""
        profit = 0.0
        for negotiator_id, offer_outcome in combination:
            quantity, _, unit_price = offer_outcome
            if self._is_supplier(negotiator_id): # Agent is buying raw materials
                profit -= quantity * unit_price
            else: # Agent is selling products
                profit += quantity * unit_price
        return profit

    def _evaluate_offer_combinations(self,
                                    offers: Dict[str, Outcome],
                                    current_im: CustomInventoryManager,
                                    awi: StdAWI,
                                    bayes_model: BayesPenaltyModel) \
                                    -> Tuple[Optional[List[Tuple[str, Outcome]]], float, float]: # (best_combo, best_score, best_profit)
        if not offers:
            return None, -1.0, 0.0

        today = awi.current_step
        horizon_days = 14
        w_short, w_over = bayes_model.current_weights()

        offers_list = list(offers.items()) # List of (nid, outcome)

        # Generate combinations: singles and pairs. Optional: triplets if few offers.
        all_eval_combinations: List[List[Tuple[str, Outcome]]] = []
        all_eval_combinations.extend([[item] for item in offers_list]) # Singles
        if len(offers_list) >= 2:
            for pair in iter_combinations(offers_list, 2):
                all_eval_combinations.append(list(pair))
        
        # Optional: Add triplets if the number of offers is small to manage complexity
        # For example, if len(offers_list) < 7, then max 7C3 = 35 triplet combinations.
        # Let's set a threshold for total combinations to avoid excessive computation.
        # Max combinations: N (singles) + N*(N-1)/2 (pairs) + N*(N-1)*(N-2)/6 (triplets)
        if len(offers_list) >=3 and len(offers_list) < 7 : # Limit for triplets
             for triplet in iter_combinations(offers_list, 3):
                all_eval_combinations.append(list(triplet))


        if not all_eval_combinations:
            return None, -1.0, 0.0

        # Baseline Inventory Assessment (once)
        shortage_before_total = current_im.get_total_insufficient_raw(today, horizon=horizon_days)
        urgent_shortage_today = current_im.get_today_insufficient_raw(today)
        product_surplus_before = sum(
            current_im.get_inventory_summary(today + d_offset, MaterialType.PRODUCT).get('estimated_available', 0.0)
            for d_offset in range(horizon_days)
        )
        max_impact_qty_heuristic = current_im.daily_production_capacity * horizon_days \
            if current_im.daily_production_capacity != float('inf') else 10000
        if max_impact_qty_heuristic == 0: max_impact_qty_heuristic = 1000
        max_raw_score_scale = (w_short * max_impact_qty_heuristic) + (w_over * max_impact_qty_heuristic)
        if max_raw_score_scale == 0: max_raw_score_scale = 1.0

        scored_combinations = [] # List of (combo_final_score, combo_profit, combo_object)

        for combo in all_eval_combinations:
            sim_im = current_im.deepcopy()
            combo_valid = True
            for negotiator_id, offer_outcome in combo:
                quantity, time, unit_price = offer_outcome
                is_supply_contract_for_agent = self._is_supplier(negotiator_id)
                contract_type = IMContractType.SUPPLY if is_supply_contract_for_agent else IMContractType.DEMAND
                material_type = MaterialType.RAW if is_supply_contract_for_agent else MaterialType.PRODUCT
                
                contract = IMContract(
                    contract_id=str(uuid4()), partner_id=negotiator_id, type=contract_type,
                    quantity=quantity, price=unit_price, delivery_time=time,
                    material_type=material_type, bankruptcy_risk=0
                )
                if not sim_im.add_transaction(contract): # If any transaction fails (e.g., past date)
                    combo_valid = False
                    break 
            
            if not combo_valid:
                if os.path.exists("env.test"):
                    print(f"Debug ({self.id}): Combo invalid due to transaction error: {combo}")
                continue # Skip this combination

            # Post-Combination Assessment
            shortage_after_total = sim_im.get_total_insufficient_raw(today, horizon=horizon_days)
            urgent_shortage_after_today = sim_im.get_today_insufficient_raw(today)
            product_surplus_after = sum(
                sim_im.get_inventory_summary(today + d_offset, MaterialType.PRODUCT).get('estimated_available', 0.0)
                for d_offset in range(horizon_days)
            )

            # Calculate Score Components (IBS, UB)
            delta_shortage = shortage_before_total - shortage_after_total
            delta_product_surplus = product_surplus_before - product_surplus_after
            
            penalty_for_surplus_increase = -delta_product_surplus * 0.1 if delta_product_surplus < 0 else 0.0
            benefit_from_surplus_decrease = delta_product_surplus if delta_product_surplus > 0 else 0.0
            raw_score_ibs = (w_short * delta_shortage) + (w_over * benefit_from_surplus_decrease) - (w_over * penalty_for_surplus_increase)
            
            norm_ibs = max(0.0, min(1.0, 0.5 + 0.5 * (raw_score_ibs / max_raw_score_scale)))

            ub = 0.0
            if urgent_shortage_today > 0:
                urgent_need_addressed = urgent_shortage_today - urgent_shortage_after_today
                if urgent_need_addressed > 0:
                    ub = min(0.20, (urgent_need_addressed / urgent_shortage_today) * 0.20)
            
            combo_final_score = max(0.0, min(1.0, norm_ibs * (1.0 - ub) + ub))
            combo_profit = self._calculate_combination_profit(combo)
            
            scored_combinations.append({'score': combo_final_score, 'profit': combo_profit, 'combo': combo})

            if os.path.exists("env.test"):
                 print(f"Combo Eval ({self.id} @ {today}): ComboNIDs={[c[0] for c in combo]}\n"
                       f"  InvBefore: ShrtTot={shortage_before_total:.2f}, UrgShrt={urgent_shortage_today:.2f}, PrdSurp={product_surplus_before:.2f}\n"
                       f"  InvAfter : ShrtTot={shortage_after_total:.2f}, UrgShrt={urgent_shortage_after_today:.2f}, PrdSurp={product_surplus_after:.2f}\n"
                       f"  Scores   : IBS_raw={raw_score_ibs:.2f}, IBS_norm={norm_ibs:.3f}, UB={ub:.3f}, Profit={combo_profit:.2f} ==> FinalScore={combo_final_score:.3f}")

        if not scored_combinations:
            return None, -1.0, 0.0

        # Sort by score (desc), then by profit (desc) for tie-breaking
        scored_combinations.sort(key=lambda x: (x['score'], x['profit']), reverse=True)
        
        best_combo_data = scored_combinations[0]
        return best_combo_data['combo'], best_combo_data['score'], best_combo_data['profit']


    def counter_all(
        self,
        offers: Dict[str, Outcome],
        states: Dict[str, SAOState],
    ) -> Dict[str, SAOResponse]:
        responses: Dict[str, SAOResponse] = {}

        if not offers:
            return responses
            
        if not self.im or not self.awi or not self.bayes:
            if os.path.exists("env.test"):
                print(f"Error ({self.id}): IM, AWI, or Bayes model not initialized. Cannot evaluate combinations.")
            for nid_key in offers.keys(): # Corrected iteration over keys
                responses[nid_key] = SAOResponse(ResponseType.REJECT_OFFER, None)
            return responses

        best_combination, norm_score, raw_profit_of_best_combination = self._evaluate_offer_combinations(
            offers, self.im, self.awi, self.bayes
        )

        norm_profit = 0.0
        ufun_is_usable = False # Default to Option B (heuristic)
        
        # ... (ufun investigation logic remains, sets ufun_is_usable if Option A is viable) ...
        # For brevity, I'll skip re-pasting the ufun investigation block. It's assumed to be here.
        if os.path.exists("env.test") and self.ufun:
            print(f"Info ({self.id}): self.ufun type: {type(self.ufun)}")
            if hasattr(self.ufun, 'outcome_space') and self.ufun.outcome_space and hasattr(self.ufun.outcome_space, 'cartesian_issues'):
                 print(f"Info ({self.id}): self.ufun.outcome_space.cartesian_issues: {self.ufun.outcome_space.cartesian_issues}")
            if best_combination : 
                first_nid, first_outcome_tuple = best_combination[0]
                outcome_as_dict = {QUANTITY: first_outcome_tuple[QUANTITY], TIME: first_outcome_tuple[TIME], UNIT_PRICE: first_outcome_tuple[UNIT_PRICE]}
                try:
                    utility_val = self.ufun(outcome_as_dict)
                    if os.path.exists("env.test"): print(f"Info ({self.id}): self.ufun(sample_outcome_dict={outcome_as_dict}) = {utility_val}")
                    if 0.0 < utility_val < 1.0: ufun_is_usable = True
                    elif utility_val == 0.0 and hasattr(self, 'get_nmi') and self.get_nmi(first_nid) and first_outcome_tuple[UNIT_PRICE] == self.get_nmi(first_nid).issues[UNIT_PRICE].max_value: ufun_is_usable = True
                    elif utility_val == 1.0 and hasattr(self, 'get_nmi') and self.get_nmi(first_nid) and first_outcome_tuple[UNIT_PRICE] == self.get_nmi(first_nid).issues[UNIT_PRICE].min_value: ufun_is_usable = True
                except Exception: pass # Silently try tuple if dict fails
                if not ufun_is_usable: # Try tuple if dict failed or conditions not met
                    try:
                        utility_val_tuple = self.ufun(first_outcome_tuple)
                        if os.path.exists("env.test"): print(f"Info ({self.id}): self.ufun(sample_outcome_tuple={first_outcome_tuple}) = {utility_val_tuple}")
                        if 0.0 < utility_val_tuple < 1.0: ufun_is_usable = True
                    except Exception: pass

        if best_combination:
            if ufun_is_usable and self.ufun: # Option A
                # ... (Option A logic as previously implemented) ...
                total_normalized_utility = 0.0
                num_offers_in_combo = len(best_combination)
                for nid_combo, outcome_tuple_combo in best_combination: # Corrected iteration
                    outcome_as_dict_combo = {QUANTITY: outcome_tuple_combo[QUANTITY], TIME: outcome_tuple_combo[TIME], UNIT_PRICE: outcome_tuple_combo[UNIT_PRICE]}
                    try:
                        offer_utility = self.ufun(outcome_as_dict_combo)
                        total_normalized_utility += offer_utility
                    except Exception as e:
                        if os.path.exists("env.test"): print(f"Warning ({self.id}): Error in ufun call for {nid_combo}: {e}.")
                        total_normalized_utility += 0 
                if num_offers_in_combo > 0: norm_profit = total_normalized_utility / num_offers_in_combo
                else: norm_profit = 0.0
                norm_profit = max(0.0, min(1.0, norm_profit))
                if os.path.exists("env.test"): print(f"Info ({self.id}): Profit Norm Option A (ufun avg). norm_profit = {norm_profit:.3f}")
            else: # Option B
                # ... (Option B logic as previously implemented) ...
                if os.path.exists("env.test"): print(f"Info ({self.id}): Profit Norm Option B (heuristic). ufun_is_usable={ufun_is_usable}")
                if raw_profit_of_best_combination <= 0: norm_profit = 0.0
                else:
                    expected_max_profit_for_combo = 0.0
                    cat_sale_price = self.awi.profile.catalog_prices[self.awi.my_output_product_idx] if hasattr(self.awi.profile, 'catalog_prices') and self.awi.my_output_product_idx < len(self.awi.profile.catalog_prices) else (self.im.processing_cost_per_unit + 5) * 1.5
                    cat_buy_price = self.awi.profile.catalog_prices[self.awi.my_input_product_idx] if hasattr(self.awi.profile, 'catalog_prices') and self.awi.my_input_product_idx < len(self.awi.profile.catalog_prices) else self.im.processing_cost_per_unit * 0.5
                    for nid_b, outcome_tuple_b in best_combination: # Corrected iteration
                        qty_b, _, _ = outcome_tuple_b
                        if self._is_supplier(nid_b): expected_max_profit_for_combo += (cat_buy_price - 1) * qty_b
                        else: 
                            cost_of_goods = self.im.processing_cost_per_unit + cat_buy_price
                            expected_max_profit_for_combo += (cat_sale_price - cost_of_goods) * qty_b
                    if expected_max_profit_for_combo > 0: norm_profit = raw_profit_of_best_combination / expected_max_profit_for_combo
                    elif raw_profit_of_best_combination > 0: norm_profit = 0.75
                    else: norm_profit = 0.5
                norm_profit = max(0.0, min(1.0, norm_profit))

        if os.path.exists("env.test"):
            nids_in_best_str = [item[0] for item in best_combination] if best_combination else "None"
            print(f"CounterAll ({self.id}): Best combo NIDs: {nids_in_best_str}, norm_score: {norm_score:.3f}, raw_profit: {raw_profit_of_best_combination:.2f}, norm_profit: {norm_profit:.3f}")

        # Decision Logic
        if best_combination is None:
            for nid in offers.keys():
                responses[nid] = SAOResponse(ResponseType.REJECT_OFFER, None)
            return responses

        best_combo_nids = {nid for nid, _ in best_combination}

        if norm_score > self.p_threshold and norm_profit > self.q_threshold: # Case 1
            if os.path.exists("env.test"): print(f"Info ({self.id}): Case 1 Triggered (Accept Best Combo)")
            for nid, outcome in best_combination:
                responses[nid] = SAOResponse(ResponseType.ACCEPT_OFFER, outcome)
            for nid in offers.keys():
                if nid not in best_combo_nids:
                    responses[nid] = SAOResponse(ResponseType.REJECT_OFFER, None) # Placeholder for future counter for unmet
        
        elif norm_score <= self.p_threshold and norm_profit > self.q_threshold: # Case 2
            if os.path.exists("env.test"): print(f"Info ({self.id}): Case 2 Triggered (Counter for Inventory Opt.)")
            for nid, outcome in offers.items():
                current_qty, current_time, current_price = outcome
                new_qty, new_price = current_qty, current_price
                if self._is_supplier(nid): # Buying
                    new_price = current_price * 0.98
                    new_qty = int(max(1, current_qty * 1.05))
                else: # Selling
                    new_price = current_price * 1.02
                    new_qty = int(max(1, current_qty * 0.95))
                responses[nid] = SAOResponse(ResponseType.REJECT_OFFER, (new_qty, current_time, new_price))

        elif norm_score > self.p_threshold and norm_profit <= self.q_threshold: # Case 3
            if os.path.exists("env.test"): print(f"Info ({self.id}): Case 3 Triggered (Counter for Price Opt.)")
            for nid, outcome in offers.items():
                current_qty, current_time, current_price = outcome
                new_price = current_price
                if self._is_supplier(nid): # Buying
                    new_price = current_price * 0.95
                else: # Selling
                    new_price = current_price * 1.05
                responses[nid] = SAOResponse(ResponseType.REJECT_OFFER, (current_qty, current_time, new_price))
        
        else: # Case 4: norm_score <= self.p_threshold AND norm_profit <= self.q_threshold
            if os.path.exists("env.test"): print(f"Info ({self.id}): Case 4 Triggered (Counter for Both or Reject)")
            for nid, outcome in offers.items():
                current_qty, current_time, current_price = outcome
                new_qty, new_price = current_qty, current_price
                if self._is_supplier(nid): # Buying
                    new_price = current_price * 0.95
                    new_qty = int(max(1, current_qty * 1.05))
                else: # Selling
                    new_price = current_price * 1.05
                    new_qty = int(max(1, current_qty * 0.95))
                responses[nid] = SAOResponse(ResponseType.REJECT_OFFER, (new_qty, current_time, new_price))
                # Fallback to simple reject if more complex counter is not desired for case 4
                # responses[nid] = SAOResponse(ResponseType.REJECT_OFFER, None)
        
        return responses

    # ------------------------------------------------------------------
    # 🌟 6. 谈判回调
    # ------------------------------------------------------------------

    def get_partner_id(self, contract: Contract) -> str:
        for p in contract.partners:
            if p != self.id:
                return p
        if os.path.exists("env.test"): print(f"⚠️ ({self.id}) Could not determine partner ID for contract {contract.id}, partners: {contract.partners}, my ID: {self.id}")
        return "unknown_partner" # Should ideally not happen

    # Modified in Step 7 (Turn 20)
    def on_negotiation_success(self, contract: Contract, mechanism: StdAWI) -> None:
        assert self.im, "CustomInventoryManager 尚未初始化"
        partner = self.get_partner_id(contract)
        is_supply = partner in self.awi.my_suppliers
        if not is_supply: 
            self._sales_successes_since_margin_update += 1
        
        im_type = IMContractType.SUPPLY if is_supply else IMContractType.DEMAND
        mat_type = MaterialType.RAW if is_supply else MaterialType.PRODUCT
        agreement = contract.agreement
        if not agreement : 
            if os.path.exists("env.test"): print(f"Error ({self.id}): Contract {contract.id} has no agreement. Skipping IM update.")
            return

        new_c = IMContract(
            contract_id=contract.id, partner_id=partner, type=im_type,
            quantity=agreement["quantity"], price=agreement["unit_price"],
            delivery_time=agreement["time"], bankruptcy_risk=0.0, 
            material_type=mat_type,
        )
        added = self.im.add_transaction(new_c)
        assert added, f"❌ ({self.id}) CustomIM.add_transaction 失败! contract={contract.id}"

        # Re-fetch insufficient amounts after transaction, as plan_production is called in add_transaction
        self.today_insufficient = self.im.get_today_insufficient_raw(self.awi.current_step)
        self.total_insufficient = self.im.get_total_insufficient_raw(self.awi.current_step, horizon=14)

        if is_supply and agreement["time"] == self.awi.current_step:
            self.purchase_completed[self.awi.current_step] += agreement["quantity"]
        elif not is_supply and agreement["time"] == self.awi.current_step:
            self.sales_completed[self.awi.current_step] += agreement["quantity"]

        if os.path.exists("env.test"):
            print(f"✅ [{self.awi.current_step}] ({self.id}) Contract {contract.id} added to IM: {new_c}")


    def _print_daily_status_report(self, result) -> None:
        """输出每日库存、生产和销售状态报告，包括未来预测"""
        if not self.im:
            return
        
        current_day = self.awi.current_step
        horizon_days = min(10, self.awi.n_steps - current_day)  # 只预测未来10天或剩余天数
        
        # 表头
        header = "|   日期    |  原料真库存  |  原料预计库存   | 计划生产  |  剩余产能  |  产品真库存  |  产品预计库存  |  已签署销售量  |  实际产品交付  |"
        separator = "|" + "-" * (len(header) + 24) + "|"
        
        print("\n📊 每日状态报告")
        print(separator)
        print(header)
        print(separator)
        
        # 当前日期及未来预测
        for day_offset in range(horizon_days):
            forecast_day = current_day + day_offset
            
            # 从IM获取数据
            raw_summary = self.im.get_inventory_summary(forecast_day, MaterialType.RAW)
            product_summary = self.im.get_inventory_summary(forecast_day, MaterialType.PRODUCT)
            
            raw_current_stock = int(raw_summary.get('current_stock', 0.0))
            raw_estimated = int(raw_summary.get('estimated_available', 0.0))
            
            product_current_stock = int(product_summary.get('current_stock', 0.0))
            product_estimated = int(product_summary.get('estimated_available', 0.0))
            
            # 计划生产量 - CustomIM stores production_plan as Dict[day, qty]
            planned_production = int(self.im.production_plan.get(forecast_day, 0.0))
            
            # 剩余产能
            remaining_capacity = int(self.im.get_available_production_capacity(forecast_day))
            
            # 已签署的销售合同数量 - CustomIM stores these in self.pending_demand_contracts
            signed_sales = 0
            # Iterate through pending_demand_contracts that are for the forecast_day
            for dem_contract in self.im.pending_demand_contracts:
                if dem_contract.delivery_time == forecast_day:
                    signed_sales += dem_contract.quantity
            
            # Delivered products might not be directly in result dict from CustomIM.
            # This was from the old IM. Let's assume 0 for now or get from CustomIM if it provides this.
            # For simplicity, let's show 0 if not available in result.
            delivered_today = result.get("delivered_products", 0) if isinstance(result, dict) and day_offset == 0 else 0

            # 格式化并输出
            day_str = f"{forecast_day} (T+{day_offset})" if day_offset == 0 else f"{forecast_day} (T+{day_offset})"
            print(f"| {day_str:^6} | {raw_current_stock:^10} | {raw_estimated:^12} | {planned_production:^8} | {remaining_capacity:^8} | {product_current_stock:^10} | {product_estimated:^12} | {signed_sales:^12} | {delivered_today:^12} |")
        
        print(separator)
        print()

    # ---------------- capacity / budget ----------------
    def _has_capacity(self, offer_outcome: Outcome) -> bool: # Changed to Outcome type
        # Assuming offer_outcome is (quantity, time, price)
        qty, time, _ = offer_outcome
        # If agent is selling (i.e. partner is a consumer, offer is for PRODUCT)
        # This needs context of who the partner is to determine if agent is seller.
        # For now, let's assume if we check capacity, it's for selling.
        # A better way might be to pass `is_seller_perspective` boolean.
        # Let's assume this is called when agent considers *making* a product to sell.
        # Or, if it is checking capacity for a received buy offer (agent would be seller).
        
        # For selling products:
        inv_summary = self.im.get_inventory_summary(time, MaterialType.PRODUCT)
        estimated_product_available = inv_summary.get("estimated_available", 0.0)
        
        # Available production capacity on that day for new production
        # Note: get_available_production_capacity is for a specific day, not cumulative.
        # If the offer's delivery time `time` is far, production could happen on any day up to `time`.
        # This is a simplification: assumes production for this offer happens on `time`.
        # A more detailed check would see if `qty` can be produced *by* `time`.
        # current_plan_for_day_time = self.im.production_plan.get(time, 0.0)
        # capacity_on_delivery_day = self.im.get_available_production_capacity(time)
        
        # JIT planning means production is scheduled to meet demands.
        # So, if we accept a new demand (sell offer), plan_production will try to fit it.
        # The check should be: can plan_production accommodate this *additional* qty by 'time'?
        # This is complex. A simpler check: is current estimated_available + potential future prod. enough?
        # For now, let's use a simplified check based on estimated_available which already considers current plan.
        # If estimated_available (which includes planned production) is enough, then yes.
        # This doesn't check if *new* production for *this specific offer* can be added if current plan uses all capacity.
        
        # A simpler check for selling:
        # Can we satisfy quantity `qty` by day `time` given current inventory and production plan?
        # `estimated_available` from `get_inventory_summary` for products already factors in planned production.
        return qty <= estimated_product_available

    def _has_budget(self, off: Offer, budget: float) -> bool:
        return True if self._is_seller(off) else off.price*off.quantity <= budget

    # ---------------- price concede ----------------
    def _concede_price(self, off: Offer, λ: float):
        tgt = self._target_price(off)
        delta = abs(off.price - tgt)
        return max(tgt, off.price-λ*delta) if self._is_seller(off) else min(tgt, off.price+λ*delta)

    def _target_price(self, offer_outcome: Outcome, is_seller_perspective: bool): # Changed to Outcome
        # Assuming offer_outcome is (quantity, time, price)
        qty, time, _ = offer_outcome
        if is_seller_perspective: # Agent is selling a PRODUCT
            # Cost of producing: avg cost of raw materials + processing cost
            raw_summary = self.im.get_inventory_summary(time, MaterialType.RAW)
            # Use estimated_average_cost for future raw material cost projection
            avg_raw_cost = raw_summary.get("estimated_average_cost", 0.0) 
            # If avg_raw_cost is 0 (e.g. no raw stock/pending), use a fallback or market price if available
            if avg_raw_cost == 0 and self._market_material_price_avg > 0 : # Fallback to market average
                 avg_raw_cost = self._market_material_price_avg

            total_unit_cost_to_produce = avg_raw_cost + self.im.processing_cost_per_unit
            return total_unit_cost_to_produce * (1 + self.min_profit_margin)
        else: # Agent is buying RAW material
            # What's the value of this raw material to us?
            # Could be based on expected sale price of product minus processing cost.
            # Or, if there's an urgent need, it might be higher.
            # For now, let's use a reference based on average product prices if available.
            # This is a simplification, as "value" of raw material is context-dependent.
            product_summary = self.im.get_inventory_summary(time, MaterialType.PRODUCT)
            avg_product_sell_price_est = product_summary.get("estimated_average_cost", 0.0) # This is cost, not price.
                                                                                        # Need a better proxy for expected sell price.
            # If we have a market product price average, use that.
            if self._market_product_price_avg > 0:
                avg_product_sell_price_est = self._market_product_price_avg
            elif avg_product_sell_price_est == 0: # If product cost is also zero (no stock/plan)
                # Fallback: use a high arbitrary value for raw material if no other info
                # This implies we are willing to buy unless it's extremely expensive.
                # A better approach: use a default expected profit margin on a default product price.
                # For now, let's say target buy price is related to our cost of making product with it.
                 return (self.im.processing_cost_per_unit * (1+self.min_profit_margin)) * 0.8 # e.g. 80% of some baseline product value
            
            # Target buy price for raw: (Expected product sale price - processing cost) * (1 - some margin for ourselves)
            # Simplified: value_of_raw = avg_product_sell_price_est - self.im.processing_cost_per_unit
            # We want to buy it for less than this value.
            value_of_raw_to_agent = avg_product_sell_price_est - self.im.processing_cost_per_unit
            return max(0.01, value_of_raw_to_agent * (1 - self.min_profit_margin)) # Buy for cheaper than its value to us


    # ---------------- util ----------------
    # _is_seller and _is_consumer are fine.
    # The _is_seller(self, off: Offer) was specific to the Offer class.
    # We need a version for negotiator_id or a general way to know context.
    # The existing _is_supplier(pid) and _is_consumer(pid) are better.

# ----------------------------------------------------
# Inventory Cost Score Calculation Helper
# ----------------------------------------------------
def calculate_inventory_cost_score(
    im_state: CustomInventoryManager,
    current_day: int,
    last_simulation_day: int, # Typically awi.n_steps
    unit_shortfall_penalty: float,
    unit_storage_cost: float # Assuming a single storage cost for simplicity, or it can be passed as a dict/tuple
) -> float:
    total_cost_score = 0.0

    # Ensure the production plan within the im_state is up-to-date for the relevant horizon
    im_state.plan_production(up_to_day=last_simulation_day)

    # A. Calculate Product Shortfall Penalty
    # This needs to simulate day-by-day product availability vs. demand.
    # We'll make a temporary copy to simulate forward without altering the original im_state's current_day.
    sim_eval_im = im_state.deepcopy() # Make a copy to simulate operations without affecting the original
    
    # Ensure the simulation starts from the correct day for evaluation
    sim_eval_im.current_day = current_day 

    for d in range(current_day, last_simulation_day + 1):
        # 1. Demands due on day 'd'
        total_demand_qty_on_d = 0.0
        for contract in sim_eval_im.pending_demand_contracts:
            if contract.delivery_time == d:
                total_demand_qty_on_d += contract.quantity
        
        if total_demand_qty_on_d == 0: # No demand, no shortfall for this day based on contracts
            # Still need to account for storage for this day if we continue the loop here.
            # The storage calculation below will handle it.
            pass

        # 2. Product stock at the start of day 'd' (before day 'd' production)
        # This should be stock after day d-1 operations.
        # get_inventory_summary(d, ...) gives stock at start of day d.
        product_stock_at_start_of_d = sim_eval_im.get_inventory_summary(d, MaterialType.PRODUCT)['current_stock']
        
        # 3. Raw materials available for production on day 'd'
        # This should be raw stock at start of day d, plus any deliveries on day d.
        # For simplicity, let's assume get_inventory_summary(d, MaterialType.RAW)['current_stock']
        # correctly reflects raw materials that *can* be used for production on day d.
        # This means it includes materials that arrived on day d *before* production starts.
        # In CustomInventoryManager, _receive_materials happens before _execute_production.
        # So, raw_stock_info for day 'd' after _receive_materials would be needed.
        # A simpler proxy: raw stock at start of day 'd'. If deliveries on 'd' are crucial,
        # this might underestimate producible amount.
        # Let's use current_stock at start of day d for raw materials.
        raw_stock_for_prod_on_d = sim_eval_im.get_inventory_summary(d, MaterialType.RAW)['current_stock']
        
        # 4. Actual production on day 'd'
        planned_production_on_d = sim_eval_im.production_plan.get(d, 0.0)
        producible_on_d = min(planned_production_on_d, raw_stock_for_prod_on_d, sim_eval_im.daily_production_capacity)
        
        # 5. Total products available to deliver on day 'd'
        total_available_to_deliver_on_d = product_stock_at_start_of_d + producible_on_d
        
        # 6. Calculate shortfall for day 'd'
        if total_demand_qty_on_d > total_available_to_deliver_on_d:
            shortfall_on_d = total_demand_qty_on_d - total_available_to_deliver_on_d
            total_cost_score += shortfall_on_d * unit_shortfall_penalty
            if os.path.exists("env.test"):
                print(f"Debug (calc_inv_cost @ day {d}): Demand={total_demand_qty_on_d}, Avail={total_available_to_deliver_on_d}, Shortfall={shortfall_on_d}, Penalty={shortfall_on_d * unit_shortfall_penalty}")

        # For storage cost calculation, we need EOD stock.
        # Simulate day processing to update batches for next day's SOD stock.
        # This is a simplified simulation of what process_day_end_operations does,
        # focusing only on inventory changes relevant to future stock levels.
        
        # Temp IM to simulate this day's operations for accurate next-day SOD stock
        temp_day_sim_im = sim_eval_im.deepcopy() # Use a copy of the current state of sim_eval_im for this day's simulation
        temp_day_sim_im.current_day = d # Set to current processing day
        
        # Simulate this day's operations (simplified, focusing on inventory changes)
        # 1. Receive materials for day d (updates raw_material_batches)
        temp_day_sim_im._receive_materials(d)
        # 2. Execute production for day d (updates raw and product_batches)
        temp_day_sim_im._execute_production(d) # Uses its own production_plan.get(d,0)
        # 3. Deliver products for day d (updates product_batches)
        temp_day_sim_im._deliver_products(d)
        
        # Update sim_eval_im's batches to reflect EOD state of day 'd'
        # This makes sim_eval_im.get_inventory_summary(d+1,...) give correct SOD for d+1
        sim_eval_im.raw_material_batches = temp_day_sim_im.raw_material_batches
        sim_eval_im.product_batches = temp_day_sim_im.product_batches
        sim_eval_im.pending_supply_contracts = temp_day_sim_im.pending_supply_contracts
        sim_eval_im.pending_demand_contracts = temp_day_sim_im.pending_demand_contracts
        # No need to advance sim_eval_im.current_day here, as the loop variable 'd' controls the day being processed.


    # B. Calculate Total Storage Cost
    # Iterate again, this time using the original im_state, assuming its current_day is the actual start
    # or use a fresh copy if the above loop modified im_state in ways not intended for storage calculation.
    # For storage, we need SOD stock which is then stored for the whole day.
    # The loop for shortfall above *did* modify sim_eval_im's batches.
    # So, we need a fresh start for storage, or use the final state of sim_eval_im if that's desired.
    # The prompt says: "current_stock from get_inventory_summary(d, ...) refers to stock at the beginning of day d"
    # This means we can iterate using the *original* im_state for storage calculation if we interpret it as
    # calculating storage cost for *future* days based on the *initial* state + plan.
    # However, if decisions (like accepting an offer) are made based on this score, the storage cost
    # should reflect the state *after* those decisions.
    # Given this is a helper to score a *potential* state (im_state), we calculate storage on that state.
    
    # Re-initialize a sim for storage cost calculation based on the *final state* of inventory after all demands met/shortfalled
    # This uses the sim_eval_im which has processed deliveries/productions up to last_simulation_day
    
    # Let's recalculate storage costs based on the state of 'im_state' as passed, assuming it's the state to evaluate.
    # The shortfall loop *simulated* operations on `sim_eval_im`.
    # For calculating storage cost of `im_state`, we should use `im_state` directly as it represents
    # the state *after* a hypothetical decision (e.g. accepting an offer).
    # The production plan of im_state is already updated.

    for d in range(current_day, last_simulation_day + 1):
        raw_stock_info = im_state.get_inventory_summary(d, MaterialType.RAW)
        product_stock_info = im_state.get_inventory_summary(d, MaterialType.PRODUCT)
        
        # As per prompt clarification: 'current_stock' is SOD, stored for the entirety of day d.
        daily_storage_cost = (raw_stock_info.get('current_stock', 0.0) * im_state.raw_storage_cost_per_unit_per_day + \
                              product_stock_info.get('current_stock', 0.0) * im_state.product_storage_cost_per_unit_per_day)
        total_cost_score += daily_storage_cost
        if os.path.exists("env.test"):
            print(f"Debug (calc_inv_cost @ day {d}): RawStock={raw_stock_info.get('current_stock',0):.0f}, ProdStock={product_stock_info.get('current_stock',0):.0f}, StorageCost={daily_storage_cost:.2f}")

    return total_cost_score


# SDK respond wrappers
    def accept(self, nid: int):
        return super().respond(nid, self.actions.ACCEPT_OFFER)
    def reject(self, nid: int):
        return super().respond(nid, self.actions.REJECT_OFFER)
    def counter(self, nid: int, offer: Offer):
        return super().respond(nid, self.actions.COUNTER_OFFER, offer)

if __name__ == "__main__":
    if os.path.exists("env.test"):
        print("模块加载成功，可在竞赛框架中使用 LitaAgentY。")
