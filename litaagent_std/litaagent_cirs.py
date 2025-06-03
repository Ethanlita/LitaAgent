from __future__ import annotations

from scml import OneShotAWI

"""
LitaAgentCIR — 库存敏感型统一策略（SDK 对接版）
=================================================
"""

from dataclasses import dataclass
from copy import deepcopy
from typing import Dict, List, Sequence, Tuple, Optional  # Added Optional
import numpy as np

from .inventory_manager_cirs import InventoryManagerCIRS, IMContract, IMContractType, MaterialType

# ------------------ 基础依赖 ------------------
from typing import Any, Dict, List, Tuple, Iterable, Optional  # Added Optional
from dataclasses import dataclass
from itertools import combinations as iter_combinations  # Added for combinations
import random
import os
import math
from collections import Counter, defaultdict  # Added defaultdict
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
from .inventory_manager_cirs import (
    IMContract,
    IMContractType,
    MaterialType,
)


# ------------------ 主代理实现 ------------------
# Main agent implementation

class LitaAgentCIRS(StdSyncAgent):
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
            procurement_cash_flow_limit_percent: float = 0.75,
            p_threshold: float = 0.5,
            q_threshold: float = -0.2,
            # 新增参数用于控制组合评估策略
            # ---
            # New parameters to control combination evaluation strategy
            combo_evaluation_strategy: str = "k_max",
            # 可选 "k_max", "beam_search", "simulated_annealing", "exhaustive_search" / Options: "k_max", "beam_search", "simulated_annealing", "exhaustive_search" # MODIFIED
            max_combo_size_for_k_max: int = 6,  # 当 strategy == "k_max" 时使用 / Used when strategy == "k_max"
            beam_width_for_beam_search: int = 3,
            # 当 strategy == "beam_search" 时使用 / Used when strategy == "beam_search"
            iterations_for_sa: int = 200,
            # 当 strategy == "simulated_annealing" 时使用 / Used when strategy == "simulated_annealing"
            sa_initial_temp: float = 1.0,  # SA 初始温度 / SA initial temperature
            sa_cooling_rate: float = 0.95,  # SA 冷却速率 / SA cooling rate
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        # —— 参数 ——
        self.total_insufficient = None
        self.today_insufficient = None
        self.procurement_cash_flow_limit_percent = procurement_cash_flow_limit_percent
        self.concession_curve_power = concession_curve_power
        self.capacity_tight_margin_increase = capacity_tight_margin_increase
        self.p_threshold = p_threshold
        self.q_threshold = q_threshold

        # 存储组合评估策略相关的参数
        # ---
        # Store parameters related to combination evaluation strategy
        self.combo_evaluation_strategy = combo_evaluation_strategy
        self.max_combo_size_for_k_max = max_combo_size_for_k_max
        self.beam_width = beam_width_for_beam_search
        self.sa_iterations = iterations_for_sa
        self.sa_initial_temp = sa_initial_temp
        self.sa_cooling_rate = sa_cooling_rate

        # ... (其他方法) ...

        # —— 运行时变量 ——
        self.im: Optional[InventoryManagerCIRS] = None  # Updated type hint
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

        self.im = InventoryManagerCIRS(
            raw_storage_cost=self.awi.current_storage_cost,  # same cost for raw and product
            product_storage_cost=self.awi.current_storage_cost,
            processing_cost=processing_cost,
            daily_production_capacity=daily_capacity,
            max_simulation_day=self.awi.n_steps,
            current_day=self.awi.current_step
        )

    def before_step(self) -> None:
        """每天开始前，同步日内关键需求信息。"""
        assert self.im, "CustomInventoryManager 尚未初始化!"
        current_day = self.awi.current_step
        self.today_insufficient = self.im.get_today_insufficient_raw(current_day)
        self.total_insufficient = self.im.get_total_insufficient_raw(current_day, horizon=14)  # Default horizon 14 days

        # 初始化当日的完成量记录
        # Initialize today's completion records
        self.sales_completed.setdefault(current_day, 0)
        self.purchase_completed.setdefault(current_day, 0)

        # 将外生协议写入im
        # Write exogenous contracts into the inventory manager
        if self.awi.is_first_level:
            exogenous_contract_quantity = self.awi.current_exogenous_input_quantity
            exogenous_contract_price = self.awi.current_exogenous_input_price
            if exogenous_contract_quantity > 0:  # Added from Step 11
                exogenous_contract_id = str(uuid4())
                exogenous_contract_partner = "simulator_exogenous_supply"  # More specific name
                exogenous_contract = IMContract(
                    contract_id=exogenous_contract_id,
                    partner_id=exogenous_contract_partner,
                    type=IMContractType.SUPPLY,
                    quantity=int(exogenous_contract_quantity),
                    price=exogenous_contract_price,
                    delivery_time=current_day,  # Exogenous are for current day
                    bankruptcy_risk=0,
                    material_type=MaterialType.RAW
                )
                self.im.add_transaction(exogenous_contract)

        elif self.awi.is_last_level:
            exogenous_contract_quantity = self.awi.current_exogenous_output_quantity
            exogenous_contract_price = self.awi.current_exogenous_output_price
            if exogenous_contract_quantity > 0:  # Added from Step 11
                exogenous_contract_id = str(uuid4())
                exogenous_contract_partner = "simulator_exogenous_demand"  # More specific name
                exogenous_contract = IMContract(
                    contract_id=exogenous_contract_id,
                    partner_id=exogenous_contract_partner,
                    type=IMContractType.DEMAND,
                    quantity=exogenous_contract_quantity,
                    price=exogenous_contract_price,
                    delivery_time=current_day,  # Exogenous are for current day
                    bankruptcy_risk=0,
                    material_type=MaterialType.PRODUCT
                )
                self.im.add_transaction(exogenous_contract)

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
        # return {}
        proposals: Dict[str, Outcome] = {}
        current_day = self.awi.current_step
        n_steps = self.awi.n_steps

        if not self.im:
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
                min_q_s = 1  # Default min quantity
                if nmi_s and nmi_s.issues[QUANTITY] is not None:
                    min_q_s = int(round(nmi_s.issues[QUANTITY].min_value))
                supplier_min_q_map[nid] = max(1, min_q_s)  # Ensure min_q is at least 1

            # Sort suppliers by their min_q_nmi to prioritize those with smaller minimums if total need is low
            # This is a simple heuristic. More complex would be to solve a knapsack-like problem.
            sorted_supplier_nids = sorted(supplier_negotiators, key=lambda nid: supplier_min_q_map[nid])

            remaining_procurement_need = target_procurement_quantity

            for nid in sorted_supplier_nids:
                if remaining_procurement_need <= 0:
                    break  # All needs met

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

                if propose_q_for_this_supplier <= 0 or propose_q_for_this_supplier > remaining_procurement_need:
                    # If min_q_nmi is greater than remaining need, we can't propose to this supplier for this need.
                    # Or if calculated quantity is invalid.
                    continue

                propose_q = int(round(propose_q_for_this_supplier))

                # Determine proposal time
                propose_t = max(current_day + 1, min_t_nmi)
                propose_t = min(propose_t, max_t_nmi)
                propose_t = max(propose_t, current_day + 1)  # Final check
                propose_t = min(propose_t, n_steps - 1)  # Final check

                # Determine proposal price (agent's best price from NMI)
                propose_p = min_p_nmi  # Buying at the lowest possible NMI price
                # Ensure price is within NMI bounds if they were fetched (max_p_nmi_from_nmi)
                # This is more for safety, as we are picking min_p_nmi.
                propose_p = min(propose_p, max_p_nmi_from_nmi)

                if propose_q > 0:
                    proposals[nid] = (propose_q, propose_t, propose_p)
                    remaining_procurement_need -= propose_q

        # --- 2. 销售产成品 (Sell finished products) ---
        if not self.awi.is_last_level:
            sellable_horizon = min(3, n_steps - (current_day + 1))
            estimated_sellable_quantity = 0
            if sellable_horizon > 0:
                current_product_stock = self.im.get_inventory_summary(current_day, MaterialType.PRODUCT).get(
                    'current_stock', 0)
                planned_production_in_horizon = 0
                for d_offset in range(sellable_horizon):
                    planned_production_in_horizon += self.im.production_plan.get(current_day + d_offset, 0)
                committed_sales_in_horizon = 0
                for contract_s in self.im.pending_demand_contracts:  # Renamed to avoid conflict
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
                for nid_c in consumer_negotiators:  # Renamed to avoid conflict
                    nmi_c = self.get_nmi(nid_c)
                    min_q_c = 1
                    if nmi_c and nmi_c.issues[QUANTITY] is not None:
                        min_q_c = int(round(nmi_c.issues[QUANTITY].min_value))
                    consumer_min_q_map[nid_c] = max(1, min_q_c)

                sorted_consumer_nids = sorted(consumer_negotiators, key=lambda nid: consumer_min_q_map[nid])
                remaining_sellable_quantity = estimated_sellable_quantity

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
                        continue

                    propose_q = int(round(propose_q_for_this_consumer))

                    propose_t = current_day + 2  # e.g., T+2 for selling
                    propose_t = max(propose_t, min_t_nmi)
                    propose_t = min(propose_t, max_t_nmi)
                    propose_t = max(propose_t, current_day + 1)  # Final check
                    propose_t = min(propose_t, n_steps - 1)  # Final check

                    # Determine proposal price (agent's best price from NMI)
                    propose_p = max_p_nmi  # Selling at the highest possible NMI price
                    # Ensure price is within NMI bounds if they were fetched (min_p_nmi_from_nmi)
                    propose_p = max(propose_p, min_p_nmi_from_nmi)

                    if propose_q > 0:
                        proposals[nid] = (propose_q, propose_t, propose_p)
                        remaining_sellable_quantity -= propose_q

        return proposals

    # ------------------------------------------------------------------
    # 🌟 5. counter_all — 谈判核心（分派到子模块）
    # ------------------------------------------------------------------
    def score_offers(
            self,
            offer_combination: Dict[str, Outcome],  # 一个报价组合
            current_im: InventoryManagerCIRS,  # 当前的库存管理器状态
            awi: OneShotAWI,  # AWI 实例，用于获取当前日期、总天数等
            # unit_shortfall_penalty: float,      # 可以作为参数传入，或在内部根据awi动态计算
            # unit_storage_cost: float            # 这个参数在calculate_inventory_cost_score中实际未使用，成本从im_state获取
    ) -> Tuple[float, float]:
        """
        评估一个报价组合的分数。
        分数 = (接受组合前的库存成本) - (接受组合后的库存成本)。
        成本由 calculate_inventory_cost_score 计算，越低越好。
        因此，本方法返回的分数越高，代表该报价组合带来的成本降低越多，越有利。
        ---
        Evaluates the score of an offer combination.
        Score = (inventory cost before accepting combo) - (inventory cost after accepting combo).
        Cost is calculated by calculate_inventory_cost_score; lower is better.
        Thus, a higher score returned by this method means the offer combo leads to a greater cost reduction, which is more favorable.
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
        # ---
        # last_simulation_day for calculate_inventory_cost_score is inclusive.
        # If awi.n_steps is the total number of simulation steps (e.g., 50, representing days 0 to 49),
        # then the last day is awi.n_steps - 1.
        actual_last_simulation_day = awi.n_steps - 1  # 修改点：明确定义实际的最后模拟日索引 / Modified: Explicitly define the actual last simulation day index

        # 1. 定义单位缺货惩罚 (unit_shortfall_penalty)
        # ---
        # 1. Define unit shortfall penalty
        unit_shortfall_penalty = self.awi.current_shortfall_penalty

        # unit_storage_cost
        current_unit_storage_cost = self.awi.current_storage_cost

        # 2. 计算 score_a: 接受报价组合前的总库存成本
        # ---
        # 2. Calculate score_a: total inventory cost before accepting the offer combination
        im_before = deepcopy(current_im)
        im_before.is_deepcopy = True
        # 假设 calculate_inventory_cost_score 是在模块级别定义的函数
        score_a = self.calculate_inventory_cost_score(
            im_state=im_before,
            current_day=today,
            last_simulation_day=actual_last_simulation_day,
            # 修改点：传递正确的最后模拟日 / Modified: Pass the correct last simulation day
            unit_shortfall_penalty=unit_shortfall_penalty,
            unit_storage_cost=current_unit_storage_cost  # 传递虚拟值
        )

        # 3. 计算 score_b: 接受报价组合后的总库存成本
        # ---
        # 3. Calculate score_b: total inventory cost after accepting the offer combination
        im_after = deepcopy(current_im)
        im_after.is_deepcopy = True
        for negotiator_id, offer_outcome in offer_combination.items():
            if not offer_outcome:  # 防御性检查，确保 offer_outcome 不是 None
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
            last_simulation_day=actual_last_simulation_day,
            # 修改点：传递正确的最后模拟日 / Modified: Pass the correct last simulation day
            unit_shortfall_penalty=unit_shortfall_penalty,
            unit_storage_cost=current_unit_storage_cost  # 传递虚拟值
        )

        # 4. 确保成本分数 a 和 b 不为负 (成本理论上应 >= 0)
        if score_a < 0:
            score_a = 0.0
        if score_b < 0:
            score_b = 0.0

        # 5. 计算最终分数: score_a - score_b
        #    如果 score_b < score_a (接受组合后成本降低), 则 final_score 为正 (好)
        #    如果 score_b > score_a (接受组合后成本增加), 则 final_score 为负 (差)
        raw_final_score = score_a - score_b
        normalized_final_score = self.normalize_final_score(raw_final_score, score_a)

        return raw_final_score, normalized_final_score

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
            im_state: InventoryManagerCIRS,
            current_day: int,
            last_simulation_day: int,
            # 这个参数现在代表实际的最后一天索引 (e.g., 49 if n_steps=50) / This parameter now represents the actual last day index (e.g., 49 if n_steps=50)
            unit_shortfall_penalty: float,
            unit_storage_cost: float
            # Assuming a single storage cost for simplicity, or it can be passed as a dict/tuple
    ) -> float:
        total_cost_score = 0.0

        # 确保传入的 im_state 的生产计划更新到正确的最后模拟日
        # ---
        # Ensure the production plan within the passed im_state is updated to the correct last simulation day
        im_state.plan_production(
            up_to_day=last_simulation_day)  # 修改点：明确指定 up_to_day / Modified: Explicitly specify up_to_day

        # A. 计算产品缺货惩罚
        # ---
        # A. Calculate Product Shortfall Penalty
        sim_eval_im_for_shortfall = deepcopy(im_state)
        sim_eval_im_for_shortfall.is_deepcopy = True
        sim_eval_im_for_shortfall.current_day = current_day

        for d in range(current_day + 1,
                       last_simulation_day + 1):  # 循环到 last_simulation_day (包含) / Loop up to last_simulation_day (inclusive)
            total_demand_qty_on_d = 0.0
            for contract in sim_eval_im_for_shortfall.pending_demand_contracts:
                if contract.delivery_time == d:
                    total_demand_qty_on_d += contract.quantity

            if total_demand_qty_on_d == 0:
                continue  # 当天无需求，继续下一天 / No demand for this day, continue to the next

            # 获取在 d 天开始时可用于交付的总产品量
            # ---
            # Get total products available for delivery at the start of day 'd'
            # 注意：get_inventory_summary(d, ...) 返回的是 d 天开始时的库存和基于当前计划的预估可用量
            # ---
            # Note: get_inventory_summary(d, ...) returns stock at the start of day d and estimated availability based on current plans
            total_available_to_deliver_on_d = sim_eval_im_for_shortfall.get_inventory_summary(d, MaterialType.PRODUCT)[
                'estimated_available']

            # 3. Calculate shortfall for day 'd'
            if total_demand_qty_on_d > total_available_to_deliver_on_d:
                shortfall_on_d = total_demand_qty_on_d - total_available_to_deliver_on_d
                total_cost_score += shortfall_on_d * unit_shortfall_penalty
            # 为了准确模拟后续天的缺货，需要模拟当天的交付（即使只是估算）
            # 这部分在原代码中缺失，但对于多日缺货计算是重要的。
            # 为简化，我们假设 get_inventory_summary 已经考虑了这一点，或者缺货计算是独立的。
            # 如果要更精确，这里应该更新 sim_eval_im_for_shortfall 的产品批次。
            # ---
            # To accurately simulate shortfall for subsequent days, today's delivery (even if estimated) needs to be simulated.
            # This part was missing in the original code but is important for multi-day shortfall calculation.
            # For simplicity, we assume get_inventory_summary already considers this, or shortfall calculation is independent.
            # For more precision, product batches in sim_eval_im_for_shortfall should be updated here.

        # B. 计算总存储成本
        # ---
        # B. Calculate Total Storage Cost
        # 使用传入的 im_state 进行存储成本计算，因为它代表了假设决策后的状态。
        # 它的 current_day 应该仍然是 current_day (即评估开始的日期)。
        # 我们将在这个副本上模拟每一天的结束操作。
        # ---
        # Use the passed im_state for storage cost calculation as it represents the state after a hypothetical decision.
        # Its current_day should still be current_day (i.e., the start day of the evaluation).
        # We will simulate end-of-day operations on this copy.
        sim_eval_im_for_storage = deepcopy(
            im_state)  # 使用一个新的副本来模拟存储成本计算过程 / Use a new copy to simulate the storage cost calculation process
        sim_eval_im_for_storage.is_deepcopy = True
        sim_eval_im_for_storage.current_day = current_day

        # Re-initialize a sim for storage cost calculation based on the *final state* of inventory after all demands met/shortfalled
        # This uses the sim_eval_im which has processed deliveries/productions up to last_simulation_day

        for d in range(current_day,
                       last_simulation_day + 1):  # 循环到 last_simulation_day (包含) / Loop up to last_simulation_day (inclusive)
            # 获取 d 天开始时的库存用于计算当天的存储成本
            # ---
            # Get stock at the start of day d to calculate storage cost for that day
            raw_stock_info = sim_eval_im_for_storage.get_inventory_summary(d, MaterialType.RAW)
            product_stock_info = sim_eval_im_for_storage.get_inventory_summary(d, MaterialType.PRODUCT)

        for d in range(current_day, last_simulation_day + 1):
            raw_stock_info = im_state.get_inventory_summary(d, MaterialType.RAW)
            product_stock_info = im_state.get_inventory_summary(d, MaterialType.PRODUCT)

            # As per prompt clarification: 'current_stock' is SOD, stored for the entirety of day d.
            daily_storage_cost = (
                    raw_stock_info.get('current_stock',
                                       0.0) * unit_storage_cost +  # unit_storage_cost 是外部传入的，或者从 self.awi 获取 / unit_storage_cost is passed externally or obtained from self.awi
                    product_stock_info.get('current_stock', 0.0) * unit_storage_cost)
            total_cost_score += daily_storage_cost

            # 推进模拟副本的天数以进行下一天的存储成本计算
            # ---
            # Advance the day of the simulation copy for the next day's storage cost calculation
            sim_eval_im_for_storage.process_day_end_operations(
                d)  # 这会将 sim_eval_im_for_storage.current_day 推进到 d + 1 / This will advance sim_eval_im_for_storage.current_day to d + 1

        # C. 计算期末库存处置成本
        # ---
        # C. Calculate excess inventory penalty (disposal cost at the end)
        # 此时，sim_eval_im_for_storage.current_day 应该是 last_simulation_day + 1
        # ---
        # At this point, sim_eval_im_for_storage.current_day should be last_simulation_day + 1
        day_for_disposal_check = last_simulation_day + 1

        # 我们需要的是在 last_simulation_day 结束后，即第 day_for_disposal_check 天开始时的库存
        # ---
        # We need the inventory at the start of day_for_disposal_check, which is after last_simulation_day ends.
        remain_raw = sim_eval_im_for_storage.get_inventory_summary(day_for_disposal_check, MaterialType.RAW)[
            'current_stock']
        remain_product = sim_eval_im_for_storage.get_inventory_summary(day_for_disposal_check, MaterialType.PRODUCT)[
            'current_stock']

        inventory_penalty = (remain_raw + remain_product) * self.awi.current_disposal_cost
        total_cost_score += inventory_penalty
        return total_cost_score

    def _evaluate_offer_combinations_exhaustive(  # NEW METHOD
            self,
            offers: Dict[str, Outcome],
            im: InventoryManagerCIRS,
            awi: OneShotAWI,
    ) -> Tuple[Optional[List[Tuple[str, Outcome]]], float]:
        """
        使用“全局搜索”（枚举所有非空子集）策略评估组合，主要基于库存得分。
        确保评估的组合至少包含一个offer。
        ---
        Evaluates combinations using the "exhaustive search" (all non-empty subsets) strategy,
        primarily based on inventory score.
        Ensures that evaluated combinations contain at least one offer.
        """
        if not offers:
            return None, -1.0

        offer_items_list: List[Tuple[str, Outcome]] = list(offers.items())
        num_offers_available = len(offer_items_list)

        best_combination_items: Optional[List[Tuple[str, Outcome]]] = None
        highest_norm_score: float = -1.0

        for i in range(1, num_offers_available + 1):  # Loop for all possible combination sizes
            for combo_as_tuple_of_tuples in iter_combinations(offer_items_list, i):
                current_combination_list_of_tuples = list(combo_as_tuple_of_tuples)
                current_combination_dict = dict(current_combination_list_of_tuples)

                _raw_cost_reduction, current_norm_score = self.score_offers(
                    offer_combination=current_combination_dict,
                    current_im=im,
                    awi=awi
                )

                if current_norm_score > highest_norm_score:
                    highest_norm_score = current_norm_score
                    best_combination_items = current_combination_list_of_tuples
                elif current_norm_score == highest_norm_score and best_combination_items:
                    # 如果分数相同，优先选择包含 offer 数量较少的组合
                    # ---
                    # If scores are the same, prefer combinations with fewer offers
                    if len(current_combination_list_of_tuples) < len(best_combination_items):
                        best_combination_items = current_combination_list_of_tuples

        return best_combination_items, highest_norm_score

    def _evaluate_offer_combinations_k_max(
            self,
            offers: Dict[str, Outcome],
            im: InventoryManagerCIRS,
            awi: OneShotAWI,
    ) -> Tuple[Optional[List[Tuple[str, Outcome]]], float]:
        """
        使用“限制K大小” 策略评估组合，主要基于库存得分。
        确保评估的组合至少包含一个offer。
        ---
        Evaluates combinations using the "limit K size" strategy, primarily based on inventory score.
        Ensures that evaluated combinations contain at least one offer.
        """
        if not offers:
            return None, -1.0

        # 将字典形式的 offers 转换为 (negotiator_id, Outcome) 元组的列表，方便组合
        offer_items_list: List[Tuple[str, Outcome]] = list(offers.items())

        best_combination_items: Optional[List[Tuple[str, Outcome]]] = None
        highest_norm_score: float = -1.0

        for i in range(1, min(len(offer_items_list), self.max_combo_size_for_k_max) + 1):
            for combo_as_tuple_of_tuples in iter_combinations(offer_items_list, i):
                # combo_as_tuple_of_tuples 保证了组合非空，因为 i 从 1 开始
                # ---
                # combo_as_tuple_of_tuples ensures the combination is non-empty as i starts from 1
                current_combination_list_of_tuples = list(combo_as_tuple_of_tuples)
                current_combination_dict = dict(current_combination_list_of_tuples)

                # 直接调用 score_offers 获取 norm_score
                # ---
                # Directly call score_offers to get norm_score
                _raw_cost_reduction, current_norm_score = self.score_offers(
                    offer_combination=current_combination_dict,
                    current_im=im,
                    awi=awi
                )

                if current_norm_score > highest_norm_score:
                    highest_norm_score = current_norm_score
                    best_combination_items = current_combination_list_of_tuples
                elif current_norm_score == highest_norm_score and best_combination_items:
                    if len(current_combination_list_of_tuples) < len(best_combination_items):
                        best_combination_items = current_combination_list_of_tuples

        return best_combination_items, highest_norm_score

    def _evaluate_offer_combinations_beam_search(
            self,
            offers: Dict[str, Outcome],
            im: InventoryManagerCIRS,
            awi: OneShotAWI,
    ) -> Tuple[Optional[List[Tuple[str, Outcome]]], float]:
        """
        使用 Beam Search 策略评估组合，主要基于库存得分。
        确保评估的组合至少包含一个offer。
        ---
        Evaluates combinations using the Beam Search strategy, primarily based on inventory score.
        Ensures that evaluated combinations contain at least one offer.
        """
        if not offers:
            return None, -1.0

        offer_items_list = list(offers.items())

        # beam 存储 (组合字典, norm_score) 元组
        # 初始束可以包含一个“哨兵”空组合，其分数为极低，以启动流程，
        # 但在选择和扩展时，我们只关心非空组合。
        # ---
        # beam stores (combo_dict, norm_score) tuples
        # Initial beam can contain a "sentinel" empty combo with a very low score to start the process,
        # but we only care about non-empty combinations during selection and expansion.
        beam: List[Tuple[Dict[str, Outcome], float]] = [({}, -float('inf'))]

        # 迭代构建组合
        # ---
        # Iteratively build combinations
        for k_round in range(len(offer_items_list)):  # 最多 M 轮 / At most M rounds
            candidates: List[Tuple[Dict[str, Outcome], float]] = []
            # processed_in_this_round 用于避免在同一轮次对完全相同的组合（基于NID集合）进行多次评估
            # ---
            # processed_in_this_round is used to avoid evaluating the exact same combination (based on NID set) multiple times in the same round
            processed_combo_keys_in_this_round = set()

            for current_combo_dict, _current_norm_score in beam:
                for offer_idx, (nid, outcome) in enumerate(offer_items_list):
                    if nid not in current_combo_dict:  # 确保不重复添加同一个伙伴的报价到当前路径
                        # ---
                        # Ensure not adding the same partner's offer repeatedly to the current path
                        new_combo_dict_list = list(current_combo_dict.items())
                        new_combo_dict_list.append((nid, outcome))
                        new_combo_dict_list.sort(key=lambda x: x[0])  # 排序以确保组合键的唯一性
                        # ---
                        # Sort to ensure uniqueness of the combination key

                        # new_combo_dict_list 现在至少包含一个元素
                        # ---
                        # new_combo_dict_list now contains at least one element
                        new_combo_tuple_key = tuple(item[0] for item in new_combo_dict_list)

                        if new_combo_tuple_key in processed_combo_keys_in_this_round:
                            continue
                        processed_combo_keys_in_this_round.add(new_combo_tuple_key)

                        new_combo_dict_final = dict(new_combo_dict_list)

                        # 只有非空组合才进行评估
                        # ---
                        # Only evaluate non-empty combinations
                        if new_combo_dict_final:
                            _raw, norm_score = self.score_offers(
                                offer_combination=new_combo_dict_final,
                                current_im=im,
                                awi=awi
                            )
                            candidates.append((new_combo_dict_final, norm_score))

            if not candidates:
                break  # 没有新的有效候选组合可以生成 / No new valid candidates can be generated

            # 将上一轮束中的有效（非空）组合也加入候选，因为它们可能是最终解
            # ---
            # Add valid (non-empty) combinations from the previous beam to candidates, as they might be the final solution
            for prev_combo_dict, prev_norm_score in beam:
                if prev_combo_dict:  # 只添加非空组合 / Only add non-empty combinations
                    # 避免重复添加已在candidates中的组合
                    # ---
                    # Avoid re-adding combinations already in candidates (based on object identity or a proper key)
                    # 为简单起见，这里假设如果它在beam中，并且是有效的，就值得再次考虑
                    # ---
                    # For simplicity, assume if it was in the beam and valid, it's worth considering again
                    # 更健壮的做法是检查是否已在candidates中（基于内容）
                    # ---
                    # A more robust approach would be to check if already in candidates (based on content)
                    candidates.append((prev_combo_dict, prev_norm_score))

            # 去重，因为上一轮的beam可能与新生成的candidates有重合
            # ---
            # Deduplicate, as the previous beam might overlap with newly generated candidates
            unique_candidates_dict: Dict[Tuple[str, ...], Tuple[Dict[str, Outcome], float]] = {}
            for cand_dict, cand_score in candidates:
                if not cand_dict: continue  # 忽略空的候选 / Ignore empty candidates
                cand_key = tuple(sorted(cand_dict.keys()))
                if cand_key not in unique_candidates_dict or cand_score > unique_candidates_dict[cand_key][1]:
                    unique_candidates_dict[cand_key] = (cand_dict, cand_score)

            sorted_candidates = sorted(list(unique_candidates_dict.values()), key=lambda x: x[1], reverse=True)
            beam = sorted_candidates[:self.beam_width]

            if not beam or not beam[0][0]:  # 如果束为空，或者束中最好的也是空组合（不应发生）
                # ---
                # If beam is empty, or the best in beam is an empty combo (should not happen)
                break
            if beam[0][1] < -0.99:  # 如果最好的候选 norm_score 仍然极差
                # ---
                # If the best candidate's norm_score is still extremely poor
                break

        # 从最终的束中选择适应度最高的非空组合
        # ---
        # Select the non-empty combination with the highest fitness from the final beam
        final_best_combo_dict: Optional[Dict[str, Outcome]] = None
        final_best_norm_score: float = -1.0

        for combo_d, n_score in beam:
            if combo_d:  # 确保组合非空 / Ensure combination is non-empty
                if n_score > final_best_norm_score:
                    final_best_norm_score = n_score
                    final_best_combo_dict = combo_d

        if final_best_combo_dict:
            return list(final_best_combo_dict.items()), final_best_norm_score
        else:
            return None, -1.0

    def _evaluate_offer_combinations_simulated_annealing(
            self,
            offers: Dict[str, Outcome],
            im: InventoryManagerCIRS,
            awi: OneShotAWI,
    ) -> Tuple[Optional[List[Tuple[str, Outcome]]], float]:
        """
        使用模拟退火策略评估组合，主要基于库存得分。
        确保最终选择的组合至少包含一个offer（如果可能）。
        ---
        Evaluates combinations using the Simulated Annealing strategy, primarily based on inventory score.
        Ensures the finally selected combination contains at least one offer (if possible).
        """
        if not offers:
            return None, -1.0

        offer_items_list = list(offers.items())
        num_offers = len(offer_items_list)

        # 初始解：可以从随机选择一个报价开始，以确保初始解非空
        # ---
        # Initial solution: can start by randomly selecting one offer to ensure the initial solution is non-empty
        if num_offers > 0:
            initial_nid, initial_outcome = random.choice(offer_items_list)
            current_solution_dict: Dict[str, Outcome] = {initial_nid: initial_outcome}
        else:  # 理论上不会到这里，因为上面有 if not offers 判断
            # ---
            # Theoretically won't reach here due to the 'if not offers' check above
            return None, -1.0

        _raw_init, current_norm_score = self.score_offers(current_solution_dict, im, awi)

        best_solution_dict = deepcopy(current_solution_dict)
        best_norm_score = current_norm_score

        temp = self.sa_initial_temp
        iterations_done = 0

        for i in range(self.sa_iterations):
            iterations_done = i + 1
            if temp < 1e-3:
                break

            neighbor_solution_dict = deepcopy(current_solution_dict)
            if num_offers == 0: break  # Should not happen due to initial check / 由于初始检查，不应发生

            action_type = random.choice(["add", "remove", "swap"])
            action_successful = False  # 标记邻域操作是否成功生成了一个与当前不同的解
            # ---
            # Flag if neighborhood operation successfully generated a different solution

            if action_type == "add" and len(neighbor_solution_dict) < num_offers:
                available_to_add = [item for item in offer_items_list if item[0] not in neighbor_solution_dict]
                if available_to_add:
                    nid_to_add, outcome_to_add = random.choice(available_to_add)
                    neighbor_solution_dict[nid_to_add] = outcome_to_add
                    action_successful = True
            elif action_type == "remove" and len(neighbor_solution_dict) > 1:  # 确保移除后至少还可能有一个（如果目标是保持非空）
                # 或者允许移除到空，但后续评估要处理
                # ---
                # Ensure at least one might remain after removal (if goal is to keep non-empty)
                # Or allow removal to empty, but subsequent evaluation must handle it
                nid_to_remove = random.choice(list(neighbor_solution_dict.keys()))
                del neighbor_solution_dict[nid_to_remove]
                action_successful = True
            elif action_type == "swap" and neighbor_solution_dict:  # 确保当前解非空才能交换
                # ---
                # Ensure current solution is non-empty to swap
                available_to_add = [item for item in offer_items_list if item[0] not in neighbor_solution_dict]
                if available_to_add:  # 必须有东西可以换入
                    # ---
                    # Must have something to swap in
                    nid_to_remove = random.choice(list(neighbor_solution_dict.keys()))
                    removed_outcome = neighbor_solution_dict.pop(nid_to_remove)

                    possible_to_add_for_swap = [item for item in available_to_add if item[0] != nid_to_remove]
                    if possible_to_add_for_swap:
                        nid_to_add, outcome_to_add = random.choice(possible_to_add_for_swap)
                        neighbor_solution_dict[nid_to_add] = outcome_to_add
                        action_successful = True
                    else:  # 没有其他可换入的，把移除的加回去
                        # ---
                        # No other to swap in, add the removed one back
                        neighbor_solution_dict[nid_to_remove] = removed_outcome

            if not action_successful or not neighbor_solution_dict:  # 如果邻域操作未改变解，或导致空解，则跳过此次迭代
                # （除非我们允许评估空解，但这里我们要求非空）
                # ---
                # If neighborhood op didn't change solution, or resulted in empty solution, skip iteration
                # (unless we allow evaluating empty solutions, but here we require non-empty)
                if not neighbor_solution_dict and current_solution_dict:  # 如果邻居变空了，但当前非空，则重新生成邻居
                    continue  # If neighbor became empty but current is not, regenerate neighbor

            # 只有当邻域解非空时才评估
            # ---
            # Only evaluate if the neighbor solution is non-empty
            if not neighbor_solution_dict:
                neighbor_norm_score = -float('inf')  # 给空解一个极差的分数
                # ---
                # Give empty solution a very poor score
            else:
                _raw_neighbor, neighbor_norm_score = self.score_offers(
                    neighbor_solution_dict, im, awi
                )

            if neighbor_norm_score > current_norm_score:
                current_solution_dict = deepcopy(neighbor_solution_dict)
                current_norm_score = neighbor_norm_score
                if current_norm_score > best_norm_score and current_solution_dict:  # 确保最佳解也非空
                    # ---
                    # Ensure best solution is also non-empty
                    best_solution_dict = deepcopy(current_solution_dict)
                    best_norm_score = current_norm_score
            elif temp > 1e-9:  # 仅当温度足够高时才考虑接受差解
                # ---
                # Only consider accepting worse solutions if temperature is high enough
                delta_fitness = current_norm_score - neighbor_norm_score
                acceptance_probability = math.exp(-delta_fitness / temp)
                if random.random() < acceptance_probability and neighbor_solution_dict:  # 确保接受的也是非空解
                    # ---
                    # Ensure accepted is also non-empty
                    current_solution_dict = deepcopy(neighbor_solution_dict)
                    current_norm_score = neighbor_norm_score

            temp *= self.sa_cooling_rate

        if not best_solution_dict:  # 如果最终最佳解是空（理论上不应发生，因为初始解非空）
            # ---
            # If the final best solution is empty (theoretically shouldn't happen as initial is non-empty)
            return None, -1.0

        return list(best_solution_dict.items()), best_norm_score

    def _evaluate_offer_combinations(
            self,
            offers: Dict[str, Outcome],
            im: InventoryManagerCIRS,
            awi: OneShotAWI,
    ) -> Tuple[Optional[List[Tuple[str, Outcome]]], float, float]:
        """
            评估报价组合，主要基于库存得分 (norm_score)。
            在确定最佳组合后，再为其计算一次利润得分 (norm_profit)。
            确保返回的最佳组合至少包含一个offer（如果输入offers非空）。
            ---
            Evaluates offer combinations, primarily based on inventory score (norm_score).
            Profit score (norm_profit) is calculated once for the determined best combination.
            Ensures the returned best combination contains at least one offer (if input offers is non-empty).
        """
        if not offers:
            return None, -1.0, 0.0

        best_combination_items: Optional[List[Tuple[str, Outcome]]] = None
        best_norm_score: float = -1.0  # 初始化为无效分数 / Initialize to an invalid score

        if self.combo_evaluation_strategy == "k_max":
            best_combination_items, best_norm_score = self._evaluate_offer_combinations_k_max(offers, im, awi)
        elif self.combo_evaluation_strategy == "exhaustive_search":  # MODIFIED: Added exhaustive_search
            best_combination_items, best_norm_score = self._evaluate_offer_combinations_exhaustive(offers, im, awi)
        elif self.combo_evaluation_strategy == "beam_search":
            best_combination_items, best_norm_score = self._evaluate_offer_combinations_beam_search(offers, im, awi)
        elif self.combo_evaluation_strategy == "simulated_annealing":
            best_combination_items, best_norm_score = self._evaluate_offer_combinations_simulated_annealing(offers, im,
                                                                                                            awi)
        else:
            best_combination_items, best_norm_score = self._evaluate_offer_combinations_k_max(offers, im, awi)

        if best_combination_items:  # 确保找到了一个非空的最佳组合
            # ---
            # Ensure a non-empty best combination was found
            best_combo_dict = dict(best_combination_items)
            _actual_profit, norm_profit_of_best = self._calculate_combination_profit_and_normalize(
                offer_combination=best_combo_dict,
                awi=awi
            )
            return best_combination_items, best_norm_score, norm_profit_of_best
        else:
            return None, -1.0, 0.0

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
        price_concession_inventory_time_change = 0.01  # Smaller concession specifically for time change if it improves score / 如果能提高分数，为时间变化提供较小的让步
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
                elif inventory_target_quantity is None:  # No specific target, no general shortfall / 没有特定目标，也没有一般性缺口
                    temp_q_for_time_eval = int(round(orig_q * (1 - epsilon_qty_change / 2)))
            else:  # Selling products / 销售产品
                temp_q_for_time_eval = int(round(orig_q * (1 + epsilon_qty_change)))
                # Make a price concession for quantity increase (seller charges less)
                # 为数量增加做价格让步 (卖家收费更少)
                temp_p_for_time_eval = orig_p * (1 - price_concession_inventory_qty_change)

            new_q = temp_q_for_time_eval  # Tentatively set new_q / 暂定 new_q
            new_p = temp_p_for_time_eval  # Tentatively set new_p / 暂定 new_p

            # Time adjustment with simulation-based scoring (Scheme C)
            # 基于模拟评分的时间调整 (方案C)

            # Candidate times: original time, one step earlier (if buying), one step later (if selling)
            # 候选时间: 原始时间, 提早一天 (如果购买), 推迟一天 (如果销售)
            candidate_times = {orig_t}  # Start with original time / 从原始时间开始
            if is_buying and orig_t > min_t_nmi:  # min_t_nmi is at least current_step + 1 / min_t_nmi 至少是 current_step + 1
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

            for t_candidate in candidate_times:
                if t_candidate == orig_t:  # Already scored / 已评分
                    continue

                # Simulate score for this t_candidate
                # 为这个 t_candidate 模拟得分
                # Assume quantity (new_q) is fixed for this time evaluation stage.
                # 假设数量 (new_q) 在此时间评估阶段是固定的。
                # Price might be slightly adjusted for making time more attractive.
                # 价格可能会略微调整以使时间更具吸引力。
                p_for_t_candidate = new_p  # Start with the price adjusted for quantity / 从为数量调整后的价格开始
                if t_candidate != orig_t:  # If time is different, make a small concession / 如果时间不同，则做小幅让步
                    if is_buying and t_candidate < orig_t:  # Buying and earlier / 购买且更早
                        p_for_t_candidate = new_p * (1 + price_concession_inventory_time_change)
                    elif not is_buying and t_candidate > orig_t:  # Selling and later / 销售且更晚
                        p_for_t_candidate = new_p * (1 - price_concession_inventory_time_change)

                offer_to_score = {negotiator_id: (new_q, t_candidate, p_for_t_candidate)}
                _, current_sim_score = self.score_offers(offer_to_score, self.im, self.awi)

                if current_sim_score > highest_simulated_score_for_time:
                    highest_simulated_score_for_time = current_sim_score
                    best_t_for_inventory = t_candidate
                    new_p = p_for_t_candidate  # Update price if this time is chosen / 如果选择了这个时间，则更新价格

            new_t = best_t_for_inventory
            # new_q is already set from quantity optimization phase / new_q 已在数量优化阶段设置
            # new_p is set to the price that yielded the best time score (or from qty opt if time didn't change)
            # new_p 设置为产生最佳时间分数的那个价格（如果时间没有改变，则来自数量优化阶段）

        # --- Profit Optimization (as per 3.1, or part of 4.1) ---
        # This will override the price if optimize_for_profit is True
        # 如果 optimize_for_profit 为 True，这将覆盖价格
        if optimize_for_profit:
            if is_buying:
                new_p = orig_p * (1 - price_target_profit_opt)  # Target a better price than original / 目标是比原始价格更好的价格
            else:
                new_p = orig_p * (1 + price_target_profit_opt)  # Target a better price than original / 目标是比原始价格更好的价格

        # --- Final clamping and validation ---
        # --- 最终限制和验证 ---
        new_q = int(round(new_q))
        new_q = max(int(round(min_q_nmi)), min(new_q, int(round(max_q_nmi))))
        if new_q <= 0:
            if min_q_nmi > 0:
                new_q = int(round(min_q_nmi))
            else:
                return None

        new_t = int(round(new_t))  # Time should be an integer / 时间应为整数
        new_t = max(min_t_nmi, min(new_t, max_t_nmi))

        new_p = max(min_p_nmi, min(new_p, max_p_nmi))
        if new_p <= 0:
            if min_p_nmi > 0.001:
                new_p = min_p_nmi
            else:
                new_p = 0.01

        # Avoid countering with an offer identical to the original
        # 避免提出与原始报价相同的还价
        if new_q == orig_q and new_t == orig_t and abs(new_p - orig_p) < 1e-5:
            # 调试 ({self.id} @ {self.awi.current_step}): 对 {negotiator_id} 的还价与原始报价相同。未生成还价。
            return None

        return new_q, new_t, new_p

    def counter_all(
            self,
            offers: Dict[str, Outcome],  # partner_id -> (q, t, p) / 伙伴ID -> (数量, 时间, 价格)
            states: Dict[str, SAOState],
    ) -> Dict[str, SAOResponse]:
        responses: Dict[str, SAOResponse] = {}
        if not offers:
            return responses

        if not self.im or not self.awi:
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

        if best_combination_items is None:  # No valid combination found / 未找到有效组合
            return responses  # All already set to REJECT / 所有均已设置为拒绝

        best_combo_outcomes_dict = dict(best_combination_items)
        best_combo_nids_set = set(best_combo_outcomes_dict.keys())

        # --- Main Decision Logic ---
        # --- 主要决策逻辑 ---

        # Case 1: norm_score > p_threshold AND norm_profit > q_threshold
        # Action: Accept the best combination. If unmet needs remain, counter others.
        # 情况1: norm_score > p_threshold 且 norm_profit > q_threshold
        # 操作: 接受最佳组合。如果仍有未满足的需求，则对其他方还价。
        if norm_score > self.p_threshold and norm_profit > self.q_threshold:
            # 1.1 Accept the offers in the best combination
            # 1.1 接受最佳组合中的报价
            for nid, outcome in best_combo_outcomes_dict.items():
                responses[nid] = SAOResponse(ResponseType.ACCEPT_OFFER, outcome)

            # 1.2 Counter offers to OTHERS if unmet needs exist (primarily for procurement of raw materials)
            # 1.2 如果存在未满足的需求，则向其他方提出还价 (主要针对原材料采购)
            # Simulate accepted offers in a temporary IM to get a more accurate remaining need.
            # 在临时IM中模拟已接受的报价，以获得更准确的剩余需求。
            temp_im_for_case1_counters = deepcopy(self.im)
            temp_im_for_case1_counters.is_deepcopy = True
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
                temp_im_for_case1_counters.add_transaction(
                    sim_contract)  # This updates plan in temp_im / 这会更新 temp_im 中的计划

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
                    qty_per_negotiator_case1 = math.ceil(
                        remaining_need_after_accepts / len(negotiators_to_counter_case1))
                    qty_per_negotiator_case1 = max(1, qty_per_negotiator_case1)  # Ensure at least 1 / 确保至少为1

                    for nid_to_counter in negotiators_to_counter_case1:
                        original_offer = offers[nid_to_counter]
                        # Generate counter-offer focusing on inventory (filling the need)
                        # 生成以库存为重点的还价 (填补需求)
                        counter_outcome = self._generate_counter_offer(
                            nid_to_counter, original_offer,
                            optimize_for_inventory=True,
                            optimize_for_profit=False,  # Primary focus is filling the need / 主要重点是填补需求
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
            also_optimize_for_profit = (norm_profit <= self.q_threshold)  # True for original Case 4 / 对于原始情况4为 True

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
        elif norm_profit <= self.q_threshold:  # This implies norm_score > p_threshold due to the sequence of checks / 由于检查顺序，这意味着 norm_score > p_threshold
            # 信息 ({self.id} @ {self.awi.current_step}): 情况3: 对所有报价进行价格优化 (分数OK, 利润差)。

            # Do NOT accept any offers.
            # Counter all offers to improve profit; inventory score was deemed acceptable.
            # 不接受任何报价。
            # 对所有报价进行还价以提高利润；库存分数被认为是可接受的。
            for nid, original_offer in offers.items():
                counter_outcome = self._generate_counter_offer(
                    nid, original_offer,
                    optimize_for_inventory=False,  # Inventory score from best_combo was good / best_combo 的库存分数良好
                    optimize_for_profit=True
                )
                if counter_outcome:
                    responses[nid] = SAOResponse(ResponseType.REJECT_OFFER, counter_outcome)
        return responses

    # ------------------------------------------------------------------
    # 🌟 6. 谈判回调
    # ------------------------------------------------------------------

    def get_partner_id(self, contract: Contract) -> str:
        for p in contract.partners:
            if p != self.id:
                return p
        return "unknown_partner"  # Should ideally not happen

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
        if not agreement:
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

# ----------------------------------------------------
# Inventory Cost Score Calculation Helper
# ----------------------------------------------------


if __name__ == "__main__":
    if os.path.exists("env.test"):
        print("模块加载成功，可在竞赛框架中使用 LitaAgentY。")
