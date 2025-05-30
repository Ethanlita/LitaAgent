from __future__ import annotations

from scml import OneShotAWI

"""
LitaAgentCIR — 库存敏感型统一策略（SDK 对接版）
=================================================
"""

from dataclasses import dataclass
from copy import deepcopy
from typing import Dict, List, Sequence, Tuple, Optional # Added Optional
import numpy as np

from .inventory_manager_cir import InventoryManagerCIR, IMContract, IMContractType, MaterialType

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
from .inventory_manager_cir import (
    IMContract,
    IMContractType,
    MaterialType,
)

# ------------------ 主代理实现 ------------------
# Main agent implementation

class LitaAgentCIR(StdSyncAgent):
    """重构后的 LitaAgent CIR。"""

    # ------------------------------------------------------------------
    # 🌟 1. 初始化
    # 1. Initialization
    # ------------------------------------------------------------------

    def __init__(
        self,
        *args,
        concession_curve_power: float = 1.5, 
        capacity_tight_margin_increase: float = 0.07, 
        procurement_cash_flow_limit_percent: float = 0.75, # Added from Step 6
        p_threshold: float = 0.25, # Threshold for combined score
        q_threshold: float = 0.0, # Threshold for individual norm_profit (unused in current logic directly, but for future)
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        # —— 参数 ——
        self.total_insufficient = None
        self.today_insufficient = None
        self.procurement_cash_flow_limit_percent = procurement_cash_flow_limit_percent # Added from Step 6
        self.concession_curve_power = concession_curve_power # Added from Step 9.b
        self.capacity_tight_margin_increase = capacity_tight_margin_increase # Added from Step 9.d
        self.p_threshold = p_threshold
        self.q_threshold = q_threshold
        
        if os.path.exists("env.test"): # Added from Step 11
            print(f"🤖 LitaAgentY {self.id} initialized with: \n"
                  f"  procurement_cash_flow_limit_percent={self.procurement_cash_flow_limit_percent:.2f}, \n"
                  f"  concession_curve_power={self.concession_curve_power:.2f}, \n"
                  f"  capacity_tight_margin_increase={self.capacity_tight_margin_increase:.3f}\n"
                  f"  p_threshold={self.p_threshold:.2f}, q_threshold={self.q_threshold:.2f}")

        # —— 运行时变量 ——
        self.im: Optional[InventoryManagerCIR] = None # Updated type hint
        self._market_price_avg: float = 0.0                
        self._market_material_price_avg: float = 0.0       
        self._market_product_price_avg: float = 0.0        
        self._recent_material_prices: List[float] = []     
        self._recent_product_prices: List[float] = []
        self._avg_window: int = 30
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
        # 反正加工成本都是固定的，scml好像会自动优化这个，就当做0了
        processing_cost = 0.0
        daily_capacity = self.awi.n_lines

        self.im = InventoryManagerCIR(
            raw_storage_cost=self.awi.current_storage_cost, # same cost for raw and product
            product_storage_cost=self.awi.current_storage_cost,
            processing_cost=processing_cost,
            daily_production_capacity=daily_capacity,
            max_simulation_day=self.awi.n_steps,
            current_day=self.awi.current_step 
        )
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
                    quantity = int(exogenous_contract_quantity),
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
    # 🌟 4. first_proposals — 首轮报价（可简化）
    # ------------------------------------------------------------------
    def first_proposals(self) -> Dict[str, Outcome]:
        """
        Generates initial proposals to partners.
        Prices are set to the agent's optimal based on NMI.
        Needs/opportunities are distributed among available partners.

        生成向伙伴的初始报价。
        价格根据NMI设置为代理的最优价格。
        需求/机会被分配给可用的伙伴。
        """
        proposals: Dict[str, Outcome] = {}
        current_day = self.awi.current_step
        n_steps = self.awi.n_steps

        if not self.im:
            if os.path.exists("env.test"):
                print(f"Error ({self.id} @ {current_day}): InventoryManager not initialized. Cannot generate first proposals.")
            return proposals

        # --- 1. 采购原材料以满足短缺 (Procure raw materials to meet shortfall) ---
        target_procurement_quantity = self.total_insufficient if self.total_insufficient is not None else 0

        supplier_negotiators = [
            nid for nid in self.negotiators.keys()
            if self._is_supplier(nid) and not (self.awi.is_first_level and nid in self.awi.my_suppliers)
        ]

        if target_procurement_quantity > 0 and supplier_negotiators:
            # Distribute procurement needs among available suppliers
            # 将采购需求分配给可用的供应商
            # We aim to distribute, but also respect NMI min quantities.
            # If total need is small, we might not be able to propose to all.

            # First, get NMI min quantities for all potential suppliers to make a more informed distribution
            supplier_min_q_map: Dict[str, int] = {}
            for nid in supplier_negotiators:
                nmi_s = self.get_nmi(nid)
                min_q_s = 1 # Default min quantity
                if nmi_s and nmi_s.issues[QUANTITY] is not None:
                    min_q_s = int(round(nmi_s.issues[QUANTITY].min_value))
                supplier_min_q_map[nid] = max(1, min_q_s) # Ensure min_q is at least 1

            # Sort suppliers by their min_q_nmi to prioritize those with smaller minimums if total need is low
            # This is a simple heuristic. More complex would be to solve a knapsack-like problem.
            sorted_supplier_nids = sorted(supplier_negotiators, key=lambda nid: supplier_min_q_map[nid])

            remaining_procurement_need = target_procurement_quantity

            if os.path.exists("env.test"):
                print(f"Debug ({self.id} @ {current_day}): FirstProposals (Supply) - Total raw material need: {target_procurement_quantity}. "
                      f"Available suppliers: {len(sorted_supplier_nids)}.")

            for nid in sorted_supplier_nids:
                if remaining_procurement_need <= 0:
                    break # All needs met

                nmi = self.get_nmi(nid)
                min_q_nmi = supplier_min_q_map[nid]
                max_q_nmi = float('inf')
                min_p_nmi = 0.01
                # max_p_nmi is not strictly needed for proposing our best price (min_p_nmi for buying)
                # but good to have for clamping if we were to use a different pricing strategy.
                max_p_nmi_from_nmi = float('inf')
                min_t_nmi, max_t_nmi = current_day + 1, n_steps - 1

                if nmi:
                    if nmi.issues[QUANTITY] is not None:
                        # min_q_nmi already fetched
                        max_q_nmi = nmi.issues[QUANTITY].max_value
                    if nmi.issues[UNIT_PRICE] is not None:
                        min_p_nmi = nmi.issues[UNIT_PRICE].min_value
                        max_p_nmi_from_nmi = nmi.issues[UNIT_PRICE].max_value
                    if nmi.issues[TIME] is not None:
                        min_t_nmi = max(min_t_nmi, nmi.issues[TIME].min_value)
                        max_t_nmi = min(max_t_nmi, nmi.issues[TIME].max_value)

                # Determine proposal quantity for this supplier
                # Propose up to remaining need, but not less than NMI min, and not more than NMI max.
                propose_q_for_this_supplier = min(remaining_procurement_need, max_q_nmi)
                propose_q_for_this_supplier = max(propose_q_for_this_supplier, min_q_nmi)

                if propose_q_for_this_supplier <= 0 or propose_q_for_this_supplier > remaining_procurement_need :
                    # If min_q_nmi is greater than remaining need, we can't propose to this supplier for this need.
                    # Or if calculated quantity is invalid.
                    if os.path.exists("env.test") and propose_q_for_this_supplier > 0 :
                         print(f"Debug ({self.id} @ {current_day}): FirstProposals (Supply) - Skipping supplier {nid}. "
                               f"Min Q ({min_q_nmi}) > remaining need ({remaining_procurement_need}) or invalid propose_q ({propose_q_for_this_supplier}).")
                    continue

                propose_q = int(round(propose_q_for_this_supplier))

                # Determine proposal time
                propose_t = max(current_day + 1, min_t_nmi)
                propose_t = min(propose_t, max_t_nmi)
                propose_t = max(propose_t, current_day + 1) # Final check
                propose_t = min(propose_t, n_steps - 1)   # Final check

                # Determine proposal price (agent's best price from NMI)
                propose_p = min_p_nmi # Buying at the lowest possible NMI price
                # Ensure price is within NMI bounds if they were fetched (max_p_nmi_from_nmi)
                # This is more for safety, as we are picking min_p_nmi.
                propose_p = min(propose_p, max_p_nmi_from_nmi)


                if propose_q > 0:
                    proposals[nid] = (propose_q, propose_t, propose_p)
                    remaining_procurement_need -= propose_q
                    if os.path.exists("env.test"):
                        print(f"Debug ({self.id} @ {current_day}): FirstProposals (Supply) - To {nid}: Q={propose_q}, T={propose_t}, P={propose_p:.2f}. "
                              f"Remaining need: {remaining_procurement_need}")

            if remaining_procurement_need > 0 and os.path.exists("env.test"):
                print(f"Warning ({self.id} @ {current_day}): FirstProposals (Supply) - Still have {remaining_procurement_need} unmet raw material need after proposing to all suppliers.")

        # --- 2. 销售产成品 (Sell finished products) ---
        if not self.awi.is_last_level:
            sellable_horizon = min(3, n_steps - (current_day + 1))
            estimated_sellable_quantity = 0
            if sellable_horizon > 0:
                current_product_stock = self.im.get_inventory_summary(current_day, MaterialType.PRODUCT).get('current_stock', 0)
                planned_production_in_horizon = 0
                for d_offset in range(sellable_horizon):
                    planned_production_in_horizon += self.im.production_plan.get(current_day + d_offset, 0)
                committed_sales_in_horizon = 0
                for contract_s in self.im.pending_demand_contracts: # Renamed to avoid conflict
                    if current_day <= contract_s.delivery_time < current_day + sellable_horizon:
                        committed_sales_in_horizon += contract_s.quantity
                estimated_sellable_quantity = current_product_stock + planned_production_in_horizon - committed_sales_in_horizon
                estimated_sellable_quantity = max(0, estimated_sellable_quantity)

            consumer_negotiators = [
                nid for nid in self.negotiators.keys()
                if self._is_consumer(nid)
            ]

            if estimated_sellable_quantity > 0 and consumer_negotiators:
                # Similar distribution logic for selling
                consumer_min_q_map: Dict[str, int] = {}
                for nid_c in consumer_negotiators: # Renamed to avoid conflict
                    nmi_c = self.get_nmi(nid_c)
                    min_q_c = 1
                    if nmi_c and nmi_c.issues[QUANTITY] is not None:
                        min_q_c = int(round(nmi_c.issues[QUANTITY].min_value))
                    consumer_min_q_map[nid_c] = max(1, min_q_c)

                sorted_consumer_nids = sorted(consumer_negotiators, key=lambda nid: consumer_min_q_map[nid])
                remaining_sellable_quantity = estimated_sellable_quantity

                if os.path.exists("env.test"):
                    print(f"Debug ({self.id} @ {current_day}): FirstProposals (Demand) - Estimated sellable products: {estimated_sellable_quantity}. "
                          f"Available consumers: {len(sorted_consumer_nids)}.")

                for nid in sorted_consumer_nids:
                    if remaining_sellable_quantity <= 0:
                        break

                    nmi = self.get_nmi(nid)
                    min_q_nmi = consumer_min_q_map[nid]
                    max_q_nmi = float('inf')
                    # min_p_nmi is not strictly needed for proposing our best price (max_p_nmi for selling)
                    min_p_nmi_from_nmi = 0.01
                    max_p_nmi = float('inf')
                    min_t_nmi, max_t_nmi = current_day + 1, n_steps - 1

                    if nmi:
                        if nmi.issues[QUANTITY] is not None:
                                max_q_nmi = nmi.issues[QUANTITY].max_value
                        if nmi.issues[UNIT_PRICE] is not None:
                                min_p_nmi_from_nmi = nmi.issues[UNIT_PRICE].min_value
                                max_p_nmi = nmi.issues[UNIT_PRICE].max_value
                        if nmi.issues[TIME] is not None:
                                min_t_nmi = max(min_t_nmi, nmi.issues[TIME].min_value)
                                max_t_nmi = min(max_t_nmi, nmi.issues[TIME].max_value)

                    propose_q_for_this_consumer = min(remaining_sellable_quantity, max_q_nmi)
                    propose_q_for_this_consumer = max(propose_q_for_this_consumer, min_q_nmi)

                    if propose_q_for_this_consumer <= 0 or propose_q_for_this_consumer > remaining_sellable_quantity:
                        if os.path.exists("env.test") and propose_q_for_this_consumer > 0:
                            print(f"Debug ({self.id} @ {current_day}): FirstProposals (Demand) - Skipping consumer {nid}. "
                                  f"Min Q ({min_q_nmi}) > remaining sellable ({remaining_sellable_quantity}) or invalid propose_q ({propose_q_for_this_consumer}).")
                        continue

                    propose_q = int(round(propose_q_for_this_consumer))

                    propose_t = current_day + 2 # e.g., T+2 for selling
                    propose_t = max(propose_t, min_t_nmi)
                    propose_t = min(propose_t, max_t_nmi)
                    propose_t = max(propose_t, current_day + 1) # Final check
                    propose_t = min(propose_t, n_steps - 1)   # Final check

                    # Determine proposal price (agent's best price from NMI)
                    propose_p = max_p_nmi # Selling at the highest possible NMI price
                    # Ensure price is within NMI bounds if they were fetched (min_p_nmi_from_nmi)
                    propose_p = max(propose_p, min_p_nmi_from_nmi)


                    if propose_q > 0:
                        proposals[nid] = (propose_q, propose_t, propose_p)
                        remaining_sellable_quantity -= propose_q
                        if os.path.exists("env.test"):
                             print(f"Debug ({self.id} @ {current_day}): FirstProposals (Demand) - To {nid}: Q={propose_q}, T={propose_t}, P={propose_p:.2f}. "
                                   f"Remaining sellable: {remaining_sellable_quantity}")

                if remaining_sellable_quantity > 0 and os.path.exists("env.test"):
                    print(f"Warning ({self.id} @ {current_day}): FirstProposals (Demand) - Still have {remaining_sellable_quantity} sellable products after proposing to all consumers.")
            elif os.path.exists("env.test") and consumer_negotiators:
                 print(f"Debug ({self.id} @ {current_day}): FirstProposals (Demand) - No estimated sellable quantity or no consumers to propose to for sales.")
        return proposals

    # ------------------------------------------------------------------
    # 🌟 5. counter_all — 谈判核心（分派到子模块）
    # ------------------------------------------------------------------
    def score_offers(
            self,
            offer_combination: Dict[str, Outcome],  # 一个报价组合
            current_im: InventoryManagerCIR,  # 当前的库存管理器状态
            awi: OneShotAWI,  # AWI 实例，用于获取当前日期、总天数等
            # unit_shortfall_penalty: float,      # 可以作为参数传入，或在内部根据awi动态计算
            # unit_storage_cost: float            # 这个参数在calculate_inventory_cost_score中实际未使用，成本从im_state获取
    ) -> Tuple[float, float]:
        """
        评估一个报价组合的分数。
        分数 = (接受组合前的库存成本) - (接受组合后的库存成本)。
        成本由 calculate_inventory_cost_score 计算，越低越好。
        因此，本方法返回的分数越高，代表该报价组合带来的成本降低越多，越有利。
        Parameter:
            Offer list
            Inventory Manager(im)
            AWI
        Return:
            Tuple[raw_score, norm_score]
        """
        today = awi.current_step
        # last_simulation_day 对于 calculate_inventory_cost_score 是包含的
        # 如果 awi.n_steps 是总模拟步数 (例如 50, 代表天数 0 到 49),
        # 那么最后一天是 awi.n_steps - 1.
        last_day = awi.n_steps - 1

        # 1. 定义单位缺货惩罚 (unit_shortfall_penalty)
        unit_shortfall_penalty = self.awi.current_shortfall_penalty  # 默认回退值

        # unit_storage_cost
        current_unit_storage_cost = self.awi.current_storage_cost

        # 2. 计算 score_a: 接受报价组合前的总库存成本
        im_before = current_im.deepcopy()
        # 假设 calculate_inventory_cost_score 是在模块级别定义的函数
        score_a = self.calculate_inventory_cost_score(
            im_state=im_before,
            current_day=today,
            last_simulation_day=self.awi.n_steps,
            unit_shortfall_penalty=unit_shortfall_penalty,
            unit_storage_cost=current_unit_storage_cost  # 传递虚拟值
        )

        # 3. 计算 score_b: 接受报价组合后的总库存成本
        im_after = current_im.deepcopy()
        for negotiator_id, offer_outcome in offer_combination.items():
            if not offer_outcome:  # 防御性检查，确保 offer_outcome 不是 None
                if os.path.exists("env.test"):
                    print(
                        f"Warning ({self.id} @ {today}): Null offer_outcome for negotiator {negotiator_id} in combination. Skipping.")
                continue

            quantity, time, unit_price = offer_outcome
            is_supply_contract_for_agent = self._is_supplier(negotiator_id)

            contract_type = IMContractType.SUPPLY if is_supply_contract_for_agent else IMContractType.DEMAND
            material_type = MaterialType.RAW if is_supply_contract_for_agent else MaterialType.PRODUCT

            # 创建临时合约用于模拟
            sim_contract = IMContract(
                contract_id=str(uuid4()),  # 模拟用ID
                partner_id=negotiator_id,
                type=contract_type,
                quantity=int(quantity),
                price=unit_price,
                delivery_time=time,
                material_type=material_type,
                bankruptcy_risk=0.0  # 模拟中假设无破产风险
            )
            im_after.add_transaction(sim_contract)  # add_transaction 内部会调用 plan_production

        score_b = self.calculate_inventory_cost_score(
            im_state=im_after,
            current_day=today,
            last_simulation_day=self.awi.n_steps,
            unit_shortfall_penalty=unit_shortfall_penalty,
            unit_storage_cost=current_unit_storage_cost  # 传递虚拟值
        )

        # 4. 确保成本分数 a 和 b 不为负 (成本理论上应 >= 0)
        if score_a < 0:
            if os.path.exists("env.test"):
                print(
                    f"Warning ({self.id} @ {today}): score_a (cost_before) is negative: {score_a:.2f}. Clamping to 0.")
            score_a = 0.0
        if score_b < 0:
            if os.path.exists("env.test"):
                print(f"Warning ({self.id} @ {today}): score_b (cost_after) is negative: {score_b:.2f}. Clamping to 0.")
            score_b = 0.0

        # 5. 计算最终分数: score_a - score_b
        #    如果 score_b < score_a (接受组合后成本降低), 则 final_score 为正 (好)
        #    如果 score_b > score_a (接受组合后成本增加), 则 final_score 为负 (差)
        raw_final_score = score_a - score_b
        normalized_final_score = self.normalize_final_score(raw_final_score, score_a)

        if os.path.exists("env.test"):
            offer_details_str_list = []
            for nid, outcm in offer_combination.items():
                if outcm:
                    offer_details_str_list.append(
                        f"NID({nid}):Q({outcm[QUANTITY]})P({outcm[UNIT_PRICE]})T({outcm[TIME]})")
                else:
                    offer_details_str_list.append(f"NID({nid}):NullOutcome")
            offers_str = ", ".join(offer_details_str_list) if offer_details_str_list else "No offers in combo"

            print(f"ScoreOffers ({self.id} @ {today}): Combo Eval: [{offers_str}]\n"
                  f"  Cost Before (score_a)   : {score_a:.2f}\n"
                  f"  Cost After (score_b)    : {score_b:.2f}\n"
                  f"  Raw Score (a-b)         : {raw_final_score:.2f}\n"
                  f"  Normalized Score        : {normalized_final_score:.3f}")

        return (raw_final_score, normalized_final_score)

    def normalize_final_score(self, final_score: float, score_a: float) -> float:
        """
        将 final_score (score_a - score_b) 归一化到 [0, 1] 区间。
        score_a 是接受组合前的成本。
        """
        if score_a < 0:  # 理论上 score_a (成本) 不应为负，做个保护
            score_a = 0.0

        if score_a == 0:
            # 如果初始成本为0:
            # final_score = -score_b
            if final_score == 0:  # score_b 也为0, 成本保持为0
                return 0.5
            else:  # final_score < 0 (因为 score_b > 0), 成本增加了
                # 使用一个快速下降的函数，例如 0.5 * exp(final_score / C) C是一个缩放常数
                # 或者更简单地，如果成本增加，就给一个较低的分数
                return 0.25  # 或者更低，表示成本从0增加是不好的

        # 当 score_a > 0:
        # relative_improvement_ratio = final_score / score_a
        # final_score = score_a - score_b
        # relative_improvement_ratio = (score_a - score_b) / score_a = 1 - (score_b / score_a)
        #
        # 这个比率:
        # 如果 score_b = 0 (成本降为0) => ratio = 1
        # 如果 score_b = score_a (成本不变) => ratio = 0
        # 如果 score_b = 2 * score_a (成本翻倍) => ratio = -1
        #
        # 我们希望:
        # ratio = 1 (final_score = score_a) -> normalized = 1.0
        # ratio = 0 (final_score = 0)     -> normalized = 0.5
        # ratio = -1 (final_score = -score_a) -> normalized = 0.0 (或接近0)

        # 使用一个修改的 logistic 函数或者简单的映射
        # x 是 final_score。我们希望 x=0 时为 0.5，x=score_a 时接近1，x=-score_a 时接近0。
        # 可以将 final_score 先映射到 [-1, 1] 左右的范围（如果以 score_a 为尺度）

        scaled_score = final_score / score_a  # 这个值理论上可以是 (-inf, 1]

        # 使用 logistic 函数: 1 / (1 + exp(-k * x))
        # 我们希望 x=0 (scaled_score=0) 时为 0.5, 这是 logistic 函数在输入为0时的自然行为。
        # 我们需要选择 k。
        # 当 scaled_score = 1 (成本降为0), final_score = score_a.  exp(-k)
        # 当 scaled_score = -1 (成本翻倍), final_score = -score_a. exp(k)

        # k 值决定了曲线的陡峭程度。k越大，曲线在0附近越陡。
        # 例如 k=2:
        # scaled_score = 1  => 1 / (1 + exp(-2)) = 1 / (1 + 0.135) = 0.88
        # scaled_score = 0  => 0.5
        # scaled_score = -1 => 1 / (1 + exp(2))  = 1 / (1 + 7.389) = 0.119

        # 如果希望 scaled_score=1 时更接近1，scaled_score=-1 时更接近0，可以增大k
        # 例如 k=4:
        # scaled_score = 1  => 1 / (1 + exp(-4)) = 1 / (1 + 0.018) = 0.982
        # scaled_score = -1 => 1 / (1 + exp(4))  = 1 / (1 + 54.6)  = 0.018

        k = 2.5  # 可调参数

        # 为了防止 scaled_score 过大或过小导致 exp 溢出或精度问题，可以先裁剪一下
        # 虽然 final_score / score_a 的上限是1，但下限可以是负无穷。
        # 但实际中，成本增加几倍已经很差了。
        # 例如，如果成本增加了10倍 (score_b = 11 * score_a), final_score = -10 * score_a, scaled_score = -10
        # exp(-k * -10) = exp(25) 会非常大。

        # 我们可以对 scaled_score 进行裁剪，例如到 [-3, 1]
        # 如果 final_score > score_a (理论上不可能，因为 score_b >= 0), 则 final_score/score_a > 1
        # 但由于 score_b >= 0, final_score = score_a - score_b <= score_a. 所以 final_score / score_a <= 1.

        # 如果 final_score < -2 * score_a (即 score_b > 3 * score_a, 成本变成原来的3倍以上)
        # 此时 scaled_score < -2。
        # 我们可以认为成本增加超过一定倍数后，分数都应该非常接近0。

        # 调整一下，让 final_score = 0 对应 0.5
        # final_score = score_a (最大收益) 对应 接近 1
        # final_score = -score_a (成本增加一倍) 对应 接近 0
        # final_score = -2*score_a (成本增加两倍) 对应 更接近 0

        # 考虑使用 final_score 作为 logistic 函数的直接输入，但需要一个缩放因子。
        # x0 设为0。 k 需要根据 score_a 来调整，或者 final_score 除以 score_a。

        # 使用之前推导的 scaled_score = final_score / score_a
        # 这个 scaled_score 的理想范围是 [-1, 1]，对应成本翻倍到成本降为0。
        # 0 对应成本不变。

        # normalized = 0.5 + 0.5 * scaled_score  (如果 scaled_score 在 [-1, 1])
        # scaled_score = 1  => 0.5 + 0.5 = 1
        # scaled_score = 0  => 0.5
        # scaled_score = -1 => 0.5 - 0.5 = 0
        # 这个是最简单的线性映射。

        # 如果 final_score / score_a 可能超出 [-1, 1]：
        # 例如 final_score = -1.5 * score_a => scaled_score = -1.5 => 0.5 - 0.75 = -0.25 (需要裁剪)
        # 例如 final_score = 0.5 * score_a => scaled_score = 0.5 => 0.5 + 0.25 = 0.75 (在范围内)

        # 线性映射并裁剪:
        normalized_value = 0.5 + 0.5 * (final_score / score_a)

        # 裁剪到 [0, 1]
        normalized_value = max(0.0, min(1.0, normalized_value))

        return normalized_value

    def calculate_inventory_cost_score(
            self,
            im_state: InventoryManagerCIR,
            current_day: int,
            last_simulation_day: int,
            unit_shortfall_penalty: float,
            unit_storage_cost: float
            # Assuming a single storage cost for simplicity, or it can be passed as a dict/tuple
    ) -> float:
        total_cost_score = 0.0

        # Ensure the production plan within the im_state is up-to-date for the relevant horizon
        im_state.plan_production(up_to_day=last_simulation_day)

        # A. Calculate Product Shortfall Penalty
        # This needs to simulate day-by-day product availability vs. demand.
        # We'll make a temporary copy to simulate forward without altering the original im_state's current_day.
        sim_eval_im = im_state.deepcopy()  # Make a copy to simulate operations without affecting the original

        # Ensure the simulation starts from the correct day for evaluation
        sim_eval_im.current_day = current_day

        for d in range(current_day + 1, last_simulation_day + 1):
            # 1. Demands due on day 'd'
            total_demand_qty_on_d = 0.0
            for contract in sim_eval_im.pending_demand_contracts:
                if contract.delivery_time == d:
                    total_demand_qty_on_d += contract.quantity

            if total_demand_qty_on_d == 0:  # No demand, no shortfall for this day based on contracts
                # Still need to account for storage for this day if we continue the loop here.
                # The storage calculation be+low will handle it.
                pass

            # 2. Total products available to deliver on day 'd'
            total_available_to_deliver_on_d = sim_eval_im.get_inventory_summary(d, MaterialType.PRODUCT)['estimated_available']

            # 3. Calculate shortfall for day 'd'
            if total_demand_qty_on_d > total_available_to_deliver_on_d:
                shortfall_on_d = total_demand_qty_on_d - total_available_to_deliver_on_d
                total_cost_score += shortfall_on_d * unit_shortfall_penalty
                if os.path.exists("env.test"):
                    print(
                        f"Debug (calc_inv_cost @ day {d}): Demand={total_demand_qty_on_d}, Avail={total_available_to_deliver_on_d}, Shortfall={shortfall_on_d}, Penalty={shortfall_on_d * unit_shortfall_penalty}")

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
        """
        for d in range(current_day, last_simulation_day + 1):
            if(os.path.exists("env.test")):
                print(f"Debug (calc_inv_cost @ day {d}): Current day in im: {im_state.current_day} (Should be equal)")
            raw_stock_info = im_state.get_inventory_summary(d, MaterialType.RAW)
            product_stock_info = im_state.get_inventory_summary(d, MaterialType.PRODUCT)

            # As per prompt clarification: 'current_stock' is SOD, stored for the entirety of day d.
            daily_storage_cost = (
                        raw_stock_info.get('current_stock', 0.0) * unit_storage_cost +
                        product_stock_info.get('current_stock', 0.0) * unit_storage_cost)
            total_cost_score += daily_storage_cost
            if os.path.exists("env.test"):
                print(
                    f"Debug (calc_inv_cost @ day {d}): RawStock={raw_stock_info.get('current_stock', 0):.0f}, ProdStock={product_stock_info.get('current_stock', 0):.0f}, StorageCost={daily_storage_cost:.2f}")
            im_state.process_day_end_operations(d)
        """

        # C. Calculate excess inventory penalty
        # Get Real Inventory the day after last day, and add a penalty
        """
        im_curday = im_state.current_day
        if im_curday == last_simulation_day - 1:
            im_state.process_day_end_operations(im_curday)
        # else im_curday = last_simulation_day
        print(f"Debug (calc inventory penalty): Day in im: {im_state.current_day} Last simulation day: {last_simulation_day}")
        remain_raw = im_state.get_inventory_summary(im_state.current_day, MaterialType.RAW)['current_stock']
        remain_product = im_state.get_inventory_summary(im_state.current_day, MaterialType.PRODUCT)['current_stock']
        inventory_penalty = (remain_raw + remain_product) *  self.awi.current_disposal_cost
        total_cost_score += inventory_penalty
        """
        excess_inventory = 0
        excess_raw = im_state.get_inventory_summary(last_simulation_day + 1, MaterialType.RAW)
        excess_product = im_state.get_inventory_summary(last_simulation_day + 1, MaterialType.PRODUCT)
        excess_inventory = (excess_raw['current_stock'] + excess_product['current_stock']) * self.awi.current_disposal_cost
        total_cost_score += excess_inventory

        return total_cost_score

    def _evaluate_offer_combinations(
            self,
            offers: Dict[str, Outcome],
            im: InventoryManagerCIR,
            awi: OneShotAWI,
    ) -> Tuple[Optional[List[Tuple[str, Outcome]]], float, float]:
        """
            评估所有可能的报价组合，并返回得分最高的组合及其分数和盈利。

            一个组合至少包含一个报价，最多包含所有传入的报价。
            “分数”是指由 score_offers 方法计算得到的归一化分数。
            “盈利”是指由 score_offers 方法计算得到的原始成本降低量 (score_a - score_b)。

            返回:
                Tuple[Optional[List[Tuple[str, Outcome]]], float, float]:
                - 最佳报价组合 (以 (negotiator_id, Outcome) 元组列表的形式表示)，如果没有有效组合则为 None。
                - 最佳组合的归一化分数 (如果在 [0,1] 区间，否则为 -1.0 表示无有效分数)。
                - 最佳组合的原始盈利 (成本降低量)(这玩意没什么意义，不要在意他)
        """
        if not offers:
            return None, -1.0, 0.0  # 没有报价，无法形成组合

        # 将字典形式的 offers 转换为 (negotiator_id, Outcome) 元组的列表，方便组合
        offer_items_list: List[Tuple[str, Outcome]] = list(offers.items())

        best_combination_items: Optional[List[Tuple[str, Outcome]]] = None
        # 归一化分数通常在 [0, 1] 区间，初始化为区间外的值
        highest_normalized_score: float = -1.0
        # 盈利
        profit_of_best_combination: float = 0.0

        # 遍历所有可能的组合大小，从1到len(offer_items_list)
        for i in range(1, len(offer_items_list) + 1):
            # 生成当前大小的所有组合
            # iter_combinations 返回的是元组的元组，例如 ((nid1, out1), (nid2, out2))
            for combo_as_tuple_of_tuples in iter_combinations(offer_items_list, i):
                current_combination_list_of_tuples = list(combo_as_tuple_of_tuples)
                current_combination_dict = dict(current_combination_list_of_tuples)

                # 1. 计算成本降低量和归一化分数
                # 调用 score_offers 获取原始成本降低和归一化分数
                # 假设 score_offers 返回 (raw_cost_reduction, normalized_score)
                raw_cost_reduction, normalized_score = self.score_offers(
                    offer_combination=current_combination_dict,
                    current_im=im,
                    awi=awi
                )

                # 2. 计算该组合的直接盈利
                raw_current_profit, normalized_current_profit = self._calculate_combination_profit_and_normalize(
                    offer_combination=current_combination_dict,
                    awi=awi
                )

                if os.path.exists("env.test"):
                    combo_nids_str = [item[0] for item in current_combination_list_of_tuples]
                    print(f"Debug ({self.id} @ {awi.current_step}): Evaluating Combo NIDs: {combo_nids_str}, "
                          f"RawCostReduction(Deprecated): {raw_cost_reduction:.2f}, NormScore: {normalized_score:.3f}, "
                          f"CalculatedProfit: {normalized_current_profit:.2f}")

                # 更新最佳组合
                if normalized_score > highest_normalized_score:
                    highest_normalized_score = normalized_score
                    best_combination_items = current_combination_list_of_tuples
                    profit_of_best_combination = normalized_current_profit
                elif normalized_score == highest_normalized_score:
                    # 如果归一化分数相同，选择原始盈利（成本降低量）更大的那个
                    if normalized_current_profit > profit_of_best_combination:
                        best_combination_items = current_combination_list_of_tuples
                        profit_of_best_combination = normalized_current_profit

        if os.path.exists("env.test"):
            if best_combination_items:
                best_combo_nids_str = [item[0] for item in best_combination_items]
                print(f"Debug ({self.id} @ {awi.current_step}): Best Combo Found: NIDs: {best_combo_nids_str}, "
                      f"HighestNormScore: {highest_normalized_score:.3f}, "
                      f"ProfitOfBest (CostReduction): {profit_of_best_combination:.2f}")
            else:
                print(f"Debug ({self.id} @ {awi.current_step}): No suitable offer combination found "
                      f"(highest_normalized_score: {highest_normalized_score:.3f}).")

        return best_combination_items, highest_normalized_score, profit_of_best_combination

    def _calculate_combination_profit_and_normalize(
            self,
            offer_combination: Dict[str, Outcome],
            awi: OneShotAWI,
            # production_cost_per_unit: float = 0.0 # 生产成本明确为0
    ) -> Tuple[float, float]:
        """
        计算报价组合的直接盈利，并将其归一化到 [-1, 1] 区间。
        盈利 = (销售收入) - (采购支出)。生产成本在此版本中设为0。
        归一化基于从 NMI 获取的估算最大潜在盈利和最大潜在亏损。
        1.0 表示非常好的盈利。
        0.0 表示盈亏平衡。
        -1.0 表示较大的亏损。

        返回:
            Tuple[float, float]: (原始盈利, 归一化后的盈利)
        """
        actual_profit = 0.0
        # Represents the profit (revenue - cost) in the best-case price scenario for the agent
        max_potential_profit_scenario = 0.0
        # Represents the profit (revenue - cost) in the worst-case price scenario for the agent
        min_potential_profit_scenario = 0.0  # This will likely be negative, representing max loss

        for negotiator_id, outcome in offer_combination.items():
            if not outcome:
                continue
            quantity, _, unit_price = outcome

            nmi = self.get_nmi(negotiator_id)
            is_selling_to_consumer = not self._is_supplier(negotiator_id)

            min_est_price = nmi.issues[UNIT_PRICE].min_value
            max_est_price = nmi.issues[UNIT_PRICE].max_value
            if nmi is None and os.path.exists("env.test"):  # Log if NMI was missing and fallback was used
                print(
                    f"Warning ({self.id} @ {awi.current_step}): NMI missing for {negotiator_id}. Using heuristic price bounds: min={min_est_price:.2f}, max={max_est_price:.2f}")

            if is_selling_to_consumer:  # We are selling products
                # Actual profit from this offer
                actual_profit += quantity * unit_price
                # Contribution to max potential profit scenario (sell at highest price)
                max_potential_profit_scenario += quantity * max_est_price
                # Contribution to min potential profit scenario (sell at lowest price)
                min_potential_profit_scenario += quantity * min_est_price
            else:  # We are buying raw materials
                # Actual profit from this offer (it's a cost)
                actual_profit -= quantity * unit_price
                # Contribution to max potential profit scenario (buy at lowest price)
                # Cost is minimized, so profit contribution is - (quantity * min_est_price)
                max_potential_profit_scenario -= quantity * min_est_price
                # Contribution to min potential profit scenario (buy at highest price)
                # Cost is maximized, so profit contribution is - (quantity * max_est_price)
                min_potential_profit_scenario -= quantity * max_est_price

        # Normalize the actual_profit
        normalized_profit = 0.0

        # The range of potential profit is [min_potential_profit_scenario, max_potential_profit_scenario]
        # We want to map this range to [-1, 1]

        profit_range = max_potential_profit_scenario - min_potential_profit_scenario

        if profit_range <= 1e-6:  # Effectively zero or invalid range (e.g. max < min)
            if actual_profit > 1e-6:  # If there's actual profit despite no discernible range
                normalized_profit = 1.0
            elif actual_profit < -1e-6:  # If there's actual loss
                normalized_profit = -1.0
            else:  # actual_profit is also near zero
                normalized_profit = 0.0
        else:
            # Linear mapping: y = (y_max - y_min) * (x - x_min) / (x_max - x_min) + y_min
            # Here, x is actual_profit, [x_min, x_max] is [min_potential_profit_scenario, max_potential_profit_scenario]
            # And [y_min, y_max] is [-1, 1]
            normalized_profit = -1.0 + 2.0 * (actual_profit - min_potential_profit_scenario) / profit_range

        # Clamp the result to [-1, 1] in case actual_profit falls outside the estimated scenario range
        normalized_profit = max(-1.0, min(1.0, normalized_profit))

        if os.path.exists("env.test"):
            print(f"Debug ({self.id} @ {awi.current_step}): ProfitCalcNorm[-1,1] (NMI-based) for Combo: "
                  f"ActualProfit={actual_profit:.2f}, "
                  f"MaxPotentialProfitScen={max_potential_profit_scenario:.2f} (Best Case Profit), "
                  f"MinPotentialProfitScen={min_potential_profit_scenario:.2f} (Worst Case Profit), "
                  f"NormalizedProfit={normalized_profit:.3f}")

        return actual_profit, normalized_profit

    def _generate_counter_offer(
            self,
            negotiator_id: str,
            original_offer: Outcome,
            optimize_for_inventory: bool,
            optimize_for_profit: bool,
            inventory_target_quantity: Optional[int] = None
            # For Case 1.2, specific need from this partner / 针对情况1.2，来自此伙伴的特定需求
    ) -> Optional[Outcome]:
        """
        Generates a counter-offer based on optimization goals using heuristics.
        It adjusts quantity, time, and price of the original_offer.
        For time adjustments, it simulates the impact on inventory score.
        使用启发式方法，根据优化目标生成还价。
        它会调整原始报价的数量、时间和价格。
        对于时间调整，它会模拟其对库存分数的影响。
        """
        orig_q, orig_t, orig_p = original_offer

        nmi = self.get_nmi(negotiator_id)

        min_q_nmi, max_q_nmi = nmi.issues[QUANTITY].min_value, nmi.issues[QUANTITY].max_value
        min_p_nmi, max_p_nmi = nmi.issues[UNIT_PRICE].min_value, nmi.issues[UNIT_PRICE].max_value
        min_t_nmi, max_t_nmi = nmi.issues[TIME].min_value, nmi.issues[TIME].max_value

        # Initialize new_q, new_t, new_p with original values
        # 用原始值初始化 new_q, new_t, new_p
        new_q, new_t, new_p = orig_q, orig_t, orig_p
        is_buying = self._is_supplier(negotiator_id)  # True if we are buying from this supplier / 如果我们从此供应商处购买，则为 True

        # Heuristic parameters
        # 启发式参数
        epsilon_qty_change = 0.10
        price_concession_inventory_time_change = 0.01 # Smaller concession specifically for time change if it improves score / 如果能提高分数，为时间变化提供较小的让步
        price_concession_inventory_qty_change = 0.02
        price_target_profit_opt = 0.05

        # --- Store initial proposed quantity and price before time evaluation ---
        # --- 在时间评估前存储初始提议的数量和价格 ---
        temp_q_for_time_eval = orig_q
        temp_p_for_time_eval = orig_p

        if optimize_for_inventory:
            # Quantity adjustment logic (applied before time evaluation for simplicity in this version)
            # 数量调整逻辑 (在此版本中为简单起见，在时间评估前应用)
            if is_buying:
                current_agent_shortfall = self.total_insufficient if self.total_insufficient is not None else 0
                effective_need_delta = inventory_target_quantity if inventory_target_quantity is not None else current_agent_shortfall
                if effective_need_delta > 0:
                    qty_after_epsilon_increase = int(round(orig_q * (1 + epsilon_qty_change)))
                    temp_q_for_time_eval = min(qty_after_epsilon_increase, effective_need_delta)
                    temp_q_for_time_eval = max(temp_q_for_time_eval, int(round(min_q_nmi)))
                    # Make a price concession for quantity increase
                    # 为数量增加做价格让步
                    temp_p_for_time_eval = orig_p * (1 + price_concession_inventory_qty_change)
                elif inventory_target_quantity is None : # No specific target, no general shortfall / 没有特定目标，也没有一般性缺口
                    temp_q_for_time_eval = int(round(orig_q * (1 - epsilon_qty_change / 2)))
            else: # Selling products / 销售产品
                temp_q_for_time_eval = int(round(orig_q * (1 + epsilon_qty_change)))
                # Make a price concession for quantity increase (seller charges less)
                # 为数量增加做价格让步 (卖家收费更少)
                temp_p_for_time_eval = orig_p * (1 - price_concession_inventory_qty_change)

            new_q = temp_q_for_time_eval # Tentatively set new_q / 暂定 new_q
            new_p = temp_p_for_time_eval # Tentatively set new_p / 暂定 new_p

            # Time adjustment with simulation-based scoring (Scheme C)
            # 基于模拟评分的时间调整 (方案C)

            # Candidate times: original time, one step earlier (if buying), one step later (if selling)
            # 候选时间: 原始时间, 提早一天 (如果购买), 推迟一天 (如果销售)
            candidate_times = {orig_t} # Start with original time / 从原始时间开始
            if is_buying and orig_t > min_t_nmi : # min_t_nmi is at least current_step + 1 / min_t_nmi 至少是 current_step + 1
                candidate_times.add(max(min_t_nmi, orig_t - 1))
            elif not is_buying and orig_t < max_t_nmi:
                candidate_times.add(min(max_t_nmi, orig_t + 1))

            best_t_for_inventory = orig_t
            highest_simulated_score_for_time = -float('inf')

            # Evaluate score for original time (with potentially adjusted q and p from above)
            # 评估原始时间的得分 (使用上面可能已调整的 q 和 p)
            # We need a mechanism to score a single hypothetical offer.
            # For now, we'll use score_offers with a dict containing only this one offer.
            # This is computationally more expensive than a dedicated single-offer scorer.
            # 我们需要一种机制来对单个假设报价进行评分。
            # 目前，我们将使用 score_offers，其中包含一个仅包含此报价的字典。
            # 这比专门的单个报价评分器计算成本更高。

            # Score for the offer with original time but potentially modified Q and P
            # 对具有原始时间但可能修改了数量和价格的报价进行评分
            initial_offer_to_score = {negotiator_id: (new_q, orig_t, new_p)}
            # Assuming score_offers returns (raw_score, normalized_score)
            # 假设 score_offers 返回 (原始分数, 归一化分数)
            _, score_with_orig_t = self.score_offers(initial_offer_to_score, self.im, self.awi)
            highest_simulated_score_for_time = score_with_orig_t

            if os.path.exists("env.test"):
                print(f"Debug ({self.id} @ {self.awi.current_step}): TimeEval for NID {negotiator_id}: OrigT={orig_t}, Q={new_q}, P={new_p:.2f}, Score={score_with_orig_t:.3f}")

            for t_candidate in candidate_times:
                if t_candidate == orig_t: # Already scored / 已评分
                    continue

                # Simulate score for this t_candidate
                # 为这个 t_candidate 模拟得分
                # Assume quantity (new_q) is fixed for this time evaluation stage.
                # 假设数量 (new_q) 在此时间评估阶段是固定的。
                # Price might be slightly adjusted for making time more attractive.
                # 价格可能会略微调整以使时间更具吸引力。
                p_for_t_candidate = new_p # Start with the price adjusted for quantity / 从为数量调整后的价格开始
                if t_candidate != orig_t: # If time is different, make a small concession / 如果时间不同，则做小幅让步
                    if is_buying and t_candidate < orig_t: # Buying and earlier / 购买且更早
                        p_for_t_candidate = new_p * (1 + price_concession_inventory_time_change)
                    elif not is_buying and t_candidate > orig_t: # Selling and later / 销售且更晚
                        p_for_t_candidate = new_p * (1 - price_concession_inventory_time_change)

                offer_to_score = {negotiator_id: (new_q, t_candidate, p_for_t_candidate)}
                _, current_sim_score = self.score_offers(offer_to_score, self.im, self.awi)

                if os.path.exists("env.test"):
                    print(f"Debug ({self.id} @ {self.awi.current_step}): TimeEval for NID {negotiator_id}: CandT={t_candidate}, Q={new_q}, P={p_for_t_candidate:.2f}, Score={current_sim_score:.3f}")

                if current_sim_score > highest_simulated_score_for_time:
                    highest_simulated_score_for_time = current_sim_score
                    best_t_for_inventory = t_candidate
                    new_p = p_for_t_candidate # Update price if this time is chosen / 如果选择了这个时间，则更新价格

            new_t = best_t_for_inventory
            # new_q is already set from quantity optimization phase / new_q 已在数量优化阶段设置
            # new_p is set to the price that yielded the best time score (or from qty opt if time didn't change)
            # new_p 设置为产生最佳时间分数的那个价格（如果时间没有改变，则来自数量优化阶段）

        # --- Profit Optimization (as per 3.1, or part of 4.1) ---
        # This will override the price if optimize_for_profit is True
        # 如果 optimize_for_profit 为 True，这将覆盖价格
        if optimize_for_profit:
            if is_buying:
                new_p = orig_p * (1 - price_target_profit_opt) # Target a better price than original / 目标是比原始价格更好的价格
            else:
                new_p = orig_p * (1 + price_target_profit_opt) # Target a better price than original / 目标是比原始价格更好的价格

        # --- Final clamping and validation ---
        # --- 最终限制和验证 ---
        new_q = int(round(new_q))
        new_q = max(int(round(min_q_nmi)), min(new_q, int(round(max_q_nmi))))
        if new_q <= 0:
            if min_q_nmi > 0: new_q = int(round(min_q_nmi))
            else: return None

        new_t = int(round(new_t)) # Time should be an integer / 时间应为整数
        new_t = max(min_t_nmi, min(new_t, max_t_nmi))

        new_p = max(min_p_nmi, min(new_p, max_p_nmi))
        if new_p <= 0:
            if min_p_nmi > 0.001: new_p = min_p_nmi
            else: new_p = 0.01

        # Avoid countering with an offer identical to the original
        # 避免提出与原始报价相同的还价
        if new_q == orig_q and new_t == orig_t and abs(new_p - orig_p) < 1e-5:
            if os.path.exists("env.test"):
                print(
                    f"Debug ({self.id} @ {self.awi.current_step}): Counter for {negotiator_id} resulted in same as original. No counter generated.")
                # 调试 ({self.id} @ {self.awi.current_step}): 对 {negotiator_id} 的还价与原始报价相同。未生成还价。
            return None

        return new_q, new_t, new_p

    def counter_all(
        self,
        offers: Dict[str, Outcome], # partner_id -> (q, t, p) / 伙伴ID -> (数量, 时间, 价格)
        states: Dict[str, SAOState],
    ) -> Dict[str, SAOResponse]:
        responses: Dict[str, SAOResponse] = {}
        if not offers:
            return responses
            
        if not self.im or not self.awi:
            if os.path.exists("env.test"):
                print(f"Error ({self.id} @ {self.awi.current_step}): IM or AWI not initialized. Rejecting all offers.")
                # 错误 ({self.id} @ {self.awi.current_step}): IM 或 AWI 未初始化。拒绝所有报价。
            for nid_key in offers.keys():
                responses[nid_key] = SAOResponse(ResponseType.REJECT_OFFER, None)
            return responses

        # Default all responses to REJECT. We will override for ACCEPT or COUNTER.
        # 默认所有响应为拒绝。我们将针对接受或还价进行覆盖。
        for nid in offers.keys():
            responses[nid] = SAOResponse(ResponseType.REJECT_OFFER, None)

        # Evaluate combinations to find the best one and its scores
        # 评估组合以找到最佳组合及其分数
        best_combination_items, norm_score, norm_profit = self._evaluate_offer_combinations(
            offers, self.im, self.awi
        )

        if os.path.exists("env.test"):
            nids_in_best_str = [item[0] for item in best_combination_items] if best_combination_items else "None"
            print(f"CounterAll ({self.id} @ {self.awi.current_step}): Best combo NIDs: {nids_in_best_str}, norm_score: {norm_score:.3f}, norm_profit: {norm_profit:.3f}")
            # CounterAll ({self.id} @ {self.awi.current_step}): 最佳组合 NID: {nids_in_best_str}, norm_score: {norm_score:.3f}, norm_profit: {norm_profit:.3f}

        if best_combination_items is None: # No valid combination found / 未找到有效组合
            if os.path.exists("env.test"): print(f"Info ({self.id} @ {self.awi.current_step}): No best combination found by _evaluate_offer_combinations. All offers rejected.")
            # 信息 ({self.id} @ {self.awi.current_step}): _evaluate_offer_combinations 未找到最佳组合。所有报价均被拒绝。
            return responses # All already set to REJECT / 所有均已设置为拒绝

        best_combo_outcomes_dict = dict(best_combination_items)
        best_combo_nids_set = set(best_combo_outcomes_dict.keys())

        # --- Main Decision Logic ---
        # --- 主要决策逻辑 ---

        # Case 1: norm_score > p_threshold AND norm_profit > q_threshold
        # Action: Accept the best combination. If unmet needs remain, counter others.
        # 情况1: norm_score > p_threshold 且 norm_profit > q_threshold
        # 操作: 接受最佳组合。如果仍有未满足的需求，则对其他方还价。
        if norm_score > self.p_threshold and norm_profit > self.q_threshold:
            if os.path.exists("env.test"): print(f"Info ({self.id} @ {self.awi.current_step}): Case 1: Accept Best Combo (Score OK, Profit OK). Counter others if need.")
            # 信息 ({self.id} @ {self.awi.current_step}): 情况1: 接受最佳组合 (分数OK, 利润OK)。如果需要则对其他方还价。

            # 1.1 Accept the offers in the best combination
            # 1.1 接受最佳组合中的报价
            for nid, outcome in best_combo_outcomes_dict.items():
                responses[nid] = SAOResponse(ResponseType.ACCEPT_OFFER, outcome)

            # 1.2 Counter offers to OTHERS if unmet needs exist (primarily for procurement of raw materials)
            # 1.2 如果存在未满足的需求，则向其他方提出还价 (主要针对原材料采购)
            # Simulate accepted offers in a temporary IM to get a more accurate remaining need.
            # 在临时IM中模拟已接受的报价，以获得更准确的剩余需求。
            temp_im_for_case1_counters = self.im.deepcopy()
            for nid_accepted, outcome_accepted in best_combo_outcomes_dict.items():
                is_supply_contract = self._is_supplier(nid_accepted)
                contract_type = IMContractType.SUPPLY if is_supply_contract else IMContractType.DEMAND
                material_type = MaterialType.RAW if is_supply_contract else MaterialType.PRODUCT
                # Create a unique ID for the temporary contract for simulation
                # 为模拟创建临时合约的唯一ID
                temp_contract_id = f"temp_accept_{nid_accepted}_{self.id}_{self.awi.current_step}_{uuid4()}"

                sim_contract = IMContract(
                    contract_id=temp_contract_id, partner_id=nid_accepted, type=contract_type,
                    quantity=int(outcome_accepted[QUANTITY]), price=outcome_accepted[UNIT_PRICE],
                    delivery_time=outcome_accepted[TIME], material_type=material_type, bankruptcy_risk=0.0
                )
                temp_im_for_case1_counters.add_transaction(sim_contract) # This updates plan in temp_im / 这会更新 temp_im 中的计划

            # Get remaining raw material insufficiency after hypothetically accepting the best combo
            # 在假设接受最佳组合后，获取剩余的原材料不足量
            remaining_need_after_accepts = temp_im_for_case1_counters.get_total_insufficient_raw(
                self.awi.current_step, horizon=14
            )

            if remaining_need_after_accepts > 0:
                # Identify negotiators not in the best combo, who are suppliers (for raw material needs)
                # 识别不在最佳组合中且为供应商的谈判者 (针对原材料需求)
                negotiators_to_counter_case1 = [
                    nid for nid in offers.keys()
                    if nid not in best_combo_nids_set and self._is_supplier(nid)
                ]
                if negotiators_to_counter_case1:
                    # Distribute the remaining need among these negotiators
                    # 将剩余需求分配给这些谈判者
                    qty_per_negotiator_case1 = math.ceil(remaining_need_after_accepts / len(negotiators_to_counter_case1))
                    qty_per_negotiator_case1 = max(1, qty_per_negotiator_case1) # Ensure at least 1 / 确保至少为1

                    for nid_to_counter in negotiators_to_counter_case1:
                        original_offer = offers[nid_to_counter]
                        # Generate counter-offer focusing on inventory (filling the need)
                        # 生成以库存为重点的还价 (填补需求)
                        counter_outcome = self._generate_counter_offer(
                            nid_to_counter, original_offer,
                            optimize_for_inventory=True,
                            optimize_for_profit=False, # Primary focus is filling the need / 主要重点是填补需求
                            inventory_target_quantity=qty_per_negotiator_case1
                        )
                        if counter_outcome:
                            responses[nid_to_counter] = SAOResponse(ResponseType.REJECT_OFFER, counter_outcome)

        # Cases 2 & 4 Merged: norm_score <= p_threshold
        # Action: Reject best combination. Counter all offers.
        # Inventory optimization is primary. If norm_profit also <= q_threshold, optimize profit too.
        # 情况2和4合并: norm_score <= p_threshold
        # 操作: 拒绝最佳组合。对所有报价进行还价。
        # 库存优化是首要的。如果 norm_profit 也 <= q_threshold，则同时优化利润。
        elif norm_score <= self.p_threshold:
            also_optimize_for_profit = (norm_profit <= self.q_threshold) # True for original Case 4 / 对于原始情况4为 True

            if also_optimize_for_profit:
                if os.path.exists("env.test"): print(f"Info ({self.id} @ {self.awi.current_step}): Case 2/4 (Merged - Case 4 type): Counter ALL for Inventory then Profit (Score BAD, Profit BAD).")
                # 信息 ({self.id} @ {self.awi.current_step}): 情况2/4 (合并 - 情况4类型): 对所有报价进行库存优化然后利润优化 (分数差, 利润差)。
            else: # norm_profit > self.q_threshold (original Case 2) / norm_profit > self.q_threshold (原始情况2)
                if os.path.exists("env.test"): print(f"Info ({self.id} @ {self.awi.current_step}): Case 2/4 (Merged - Case 2 type): Counter ALL for Inventory Opt (Score BAD, Profit OK).")
                # 信息 ({self.id} @ {self.awi.current_step}): 情况2/4 (合并 - 情况2类型): 对所有报价进行库存优化 (分数差, 利润OK)。

            # Do NOT accept any offers from `best_combination` or any other.
            # Counter all offers based on the determined optimization strategy.
            # 不接受来自 `best_combination` 或任何其他组合的任何报价。
            # 根据确定的优化策略对所有报价进行还价。
            for nid, original_offer in offers.items():
                counter_outcome = self._generate_counter_offer(
                    nid, original_offer,
                    optimize_for_inventory=True,
                    optimize_for_profit=also_optimize_for_profit
                )
                if counter_outcome:
                    responses[nid] = SAOResponse(ResponseType.REJECT_OFFER, counter_outcome)

        # Case 3: norm_score > p_threshold AND norm_profit <= q_threshold
        # Action: Reject best combination. Counter all offers focusing on profit optimization.
        # 情况3: norm_score > p_threshold 且 norm_profit <= q_threshold
        # 操作: 拒绝最佳组合。对所有报价进行以利润优化为重点的还价。
        elif norm_profit <= self.q_threshold: # This implies norm_score > p_threshold due to the sequence of checks / 由于检查顺序，这意味着 norm_score > p_threshold
            if os.path.exists("env.test"): print(f"Info ({self.id} @ {self.awi.current_step}): Case 3: Counter ALL for Price Opt (Score OK, Profit BAD).")
            # 信息 ({self.id} @ {self.awi.current_step}): 情况3: 对所有报价进行价格优化 (分数OK, 利润差)。

            # Do NOT accept any offers.
            # Counter all offers to improve profit; inventory score was deemed acceptable.
            # 不接受任何报价。
            # 对所有报价进行还价以提高利润；库存分数被认为是可接受的。
            for nid, original_offer in offers.items():
                counter_outcome = self._generate_counter_offer(
                    nid, original_offer,
                    optimize_for_inventory=False, # Inventory score from best_combo was good / best_combo 的库存分数良好
                    optimize_for_profit=True
                )
                if counter_outcome:
                    responses[nid] = SAOResponse(ResponseType.REJECT_OFFER, counter_outcome)

        else:
            # This path should ideally not be reached if all conditions are covered.
            # All offers will remain REJECTED by default.
            # 如果所有条件都已覆盖，则理想情况下不应到达此路径。
            # 默认情况下，所有报价都将保持被拒绝状态。
            if os.path.exists("env.test"):
                print(f"Warning ({self.id} @ {self.awi.current_step}): counter_all logic fell through. All offers rejected by default.")
                # 警告 ({self.id} @ {self.awi.current_step}): counter_all 逻辑未覆盖所有情况。默认拒绝所有报价。

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
            quantity=int(agreement["quantity"]), price=agreement["unit_price"],
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
        if not self.im or not os.path.exists("env.test"):
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
            
            raw_current_stock = raw_summary.get('current_stock', 0)
            raw_estimated = raw_summary.get('estimated_available', 0)
            
            product_current_stock = product_summary.get('current_stock', 0)
            product_estimated = product_summary.get('estimated_available', 0)
            
            # 计划生产量 - CustomIM stores production_plan as Dict[day, qty]
            planned_production = self.im.production_plan.get(forecast_day, 0)
            
            # 剩余产能
            remaining_capacity = self.im.get_available_production_capacity(forecast_day)
            
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

# ----------------------------------------------------
# Inventory Cost Score Calculation Helper
# ----------------------------------------------------


if __name__ == "__main__":
    if os.path.exists("env.test"):
        print("模块加载成功，可在竞赛框架中使用 LitaAgentY。")
