from __future__ import annotations

from scml import OneShotAWI

"""
LitaAgentCIRS — 库存敏感型统一策略（SDK 对接版）
=================================================
"""

from dataclasses import dataclass
from copy import deepcopy
from typing import Dict, List, Sequence, Tuple, Optional # Added Optional
import numpy as np

from .inventory_manager_cirs import InventoryManagerCIRS, IMContract, IMContractType, MaterialType

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
        combo_evaluation_strategy: str = "k_max",  # 可选 "k_max", "beam_search", "simulated_annealing", "exhaustive_search" / Options: "k_max", "beam_search", "simulated_annealing", "exhaustive_search" # MODIFIED
        max_combo_size_for_k_max: int = 6, # 当 strategy == "k_max" 时使用 / Used when strategy == "k_max"
        beam_width_for_beam_search: int = 3, # 当 strategy == "beam_search" 时使用 / Used when strategy == "beam_search"
        iterations_for_sa: int = 200, # 当 strategy == "simulated_annealing" 时使用 / Used when strategy == "simulated_annealing"
        sa_initial_temp: float = 1.0, # SA 初始温度 / SA initial temperature
        sa_cooling_rate: float = 0.95, # SA 冷却速率 / SA cooling rate
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

        # —— 运行时变量 ——
        self.im: Optional[InventoryManagerCIRS] = None # Updated type hint
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
            raw_storage_cost=self.awi.current_storage_cost, # same cost for raw and product
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
        self.total_insufficient = self.im.get_total_insufficient_raw(current_day, horizon=14) # Default horizon 14 days

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
        价格设置为基于 NMI 的代理最优价格。
        需求/机会分配给可用的伙伴。
        """
        # 初始化结果
        # Initialize result
        proposals = {}
        
        # 获取当前日期
        # Get current date
        current_day = self.awi.current_step
        
        # 如果没有库存管理器，返回空字典
        # If there is no inventory manager, return an empty dictionary
        if not self.im:
            return {}
            
        # 处理供应商（采购原材料）
        # Process suppliers (purchase raw materials)
        # 获取今天紧急需要的原材料量
        # Get the amount of raw materials urgently needed today
        target_raw_procurement = self.today_insufficient
        
        # 如果有紧急需要，处理紧急采购
        # If there is an urgent need, process emergency procurement
        if target_raw_procurement > 0:
            # 获取所有供应商
            # Get all suppliers
            suppliers = list(self.awi.my_suppliers)
            
            # 如果没有供应商，跳过
            # If there are no suppliers, skip
            if not suppliers:
                pass
            else:
                # 分配需求到供应商
                # Distribute needs to suppliers
                # 简单平均分配
                # Simple average distribution
                qty_per_supplier = max(1, target_raw_procurement // len(suppliers))
                remainder = target_raw_procurement % len(suppliers)
                
                # 为每个供应商生成提案
                # Generate proposals for each supplier
                for i, supplier_id in enumerate(suppliers):
                    # 计算数量
                    # Calculate quantity
                    propose_q = qty_per_supplier + (1 if i < remainder else 0)
                    
                    # 如果数量为 0，跳过
                    # If quantity is 0, skip
                    if propose_q <= 0:
                        continue
                        
                    # 计算交付日期
                    # Calculate delivery date
                    propose_t = current_day + 1  # 明天交付 Deliver tomorrow
                    
                    # 计算价格
                    # Calculate price
                    # 使用 NMI 的最优价格
                    # Use NMI's optimal price
                    propose_p = self.awi.current_input_issues[UNIT_PRICE].min_value
                    
                    # 创建提案
                    # Create proposal
                    proposals[supplier_id] = {
                        QUANTITY: propose_q,
                        TIME: propose_t,
                        UNIT_PRICE: propose_p
                    }
                    
                    # 更新最后出价
                    # Update last offer
                    self._last_offer_price[supplier_id] = propose_p
        
        # 处理消费者（销售产品）
        # Process consumers (sell products)
        # 获取可销售的产品数量
        # Get the quantity of products that can be sold
        # 当前库存 + 今天可用原材料
        # Current inventory + raw materials available today
        raw_summary = self.im.get_inventory_summary(current_day, MaterialType.RAW)
        product_summary = self.im.get_inventory_summary(current_day, MaterialType.PRODUCT)
        
        # 估计可销售数量
        # Estimate sellable quantity
        estimated_sellable = int(product_summary["current_stock"]) + int(raw_summary["current_stock"])
        
        # 如果有可销售的产品，为每个消费者生成提案
        # If there are products that can be sold, generate proposals for each consumer
        if estimated_sellable > 0:
            # 获取所有消费者
            # Get all consumers
            consumers = list(self.awi.my_consumers)
            
            # 如果没有消费者，跳过
            # If there are no consumers, skip
            if not consumers:
                pass
            else:
                # 分配产品到消费者
                # Distribute products to consumers
                # 简单平均分配
                # Simple average distribution
                qty_per_consumer = max(1, estimated_sellable // len(consumers))
                remainder = estimated_sellable % len(consumers)
                
                # 为每个消费者生成提案
                # Generate proposals for each consumer
                for i, consumer_id in enumerate(consumers):
                    # 计算数量
                    # Calculate quantity
                    propose_q = qty_per_consumer + (1 if i < remainder else 0)
                    
                    # 如果数量为 0，跳过
                    # If quantity is 0, skip
                    if propose_q <= 0:
                        continue
                        
                    # 计算交付日期
                    # Calculate delivery date
                    propose_t = current_day + 1  # 明天交付 Deliver tomorrow
                    
                    # 计算价格
                    # Calculate price
                    # 使用 NMI 的最优价格
                    # Use NMI's optimal price
                    propose_p = self.awi.current_output_issues[UNIT_PRICE].max_value
                    
                    # 创建提案
                    # Create proposal
                    proposals[consumer_id] = {
                        QUANTITY: propose_q,
                        TIME: propose_t,
                        UNIT_PRICE: propose_p
                    }
                    
                    # 更新最后出价
                    # Update last offer
                    self._last_offer_price[consumer_id] = propose_p
        
        return proposals

    # ------------------------------------------------------------------
    # 🌟 5. score_offers — 评分函数
    # ------------------------------------------------------------------
    def score_offers(self, offer_combination: Dict[str, Outcome], current_im: InventoryManagerCIRS, awi: OneShotAWI) -> float:
        """
        Scores a combination of offers based on their impact on inventory and profit.
        
        评分一组报价，基于它们对库存和利润的影响。
        
        Args:
            offer_combination: Dictionary mapping negotiator IDs to their offers
            current_im: Current state of the inventory manager
            awi: Agent world interface
            
        Returns:
            float: Score for this combination of offers
        """
        # 如果组合为空，返回 0
        # If combination is empty, return 0
        if not offer_combination:
            return 0.0
            
        # 复制当前库存管理器状态
        # Copy current inventory manager state
        im_copy = deepcopy(current_im)
        im_copy.is_deepcopy = True # 标记为深拷贝，避免打印 / Mark as deepcopy to avoid printing
        
        # 获取当前日期
        # Get current date
        today = awi.current_step
        
        # 处理每个报价
        # Process each offer
        for negotiator_id, offer_outcome in offer_combination.items():
            # 如果报价为空，跳过
            # If offer is empty, skip
            if not offer_outcome:
                continue
                
            # 获取数量、价格和交付日期
            # Get quantity, price and delivery date
            quantity = offer_outcome.get(QUANTITY, 0)
            price = offer_outcome.get(UNIT_PRICE, 0.0)
            delivery_time = offer_outcome.get(TIME, today)
            
            # 确定合同类型和材料类型
            # Determine contract type and material type
            if self._is_supplier(negotiator_id):
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
            new_contract = IMContract(
                contract_id=f"sim_{negotiator_id}_{delivery_time}",
                partner_id=negotiator_id,
                type=contract_type,
                quantity=quantity,
                price=price,
                delivery_time=delivery_time,
                bankruptcy_risk=0.0,  # 假设无破产风险 Assume no bankruptcy risk
                material_type=material_type
            )
            
            # 添加到库存管理器
            # Add to inventory manager
            im_copy.add_transaction(new_contract)
            
        # 计算库存成本得分
        # Calculate inventory cost score
        inventory_score = self.calculate_inventory_cost_score(
            im_state=im_copy,
            current_day=today,
            last_simulation_day=min(today + 14, awi.n_steps - 1),  # 模拟未来 14 天 Simulate next 14 days
            unit_shortfall_penalty=10.0,  # 缺货惩罚 Shortfall penalty
            unit_storage_cost=awi.current_storage_cost  # 存储成本 Storage cost
        )
        
        # 计算利润得分
        # Calculate profit score
        profit_score = self._calculate_combination_profit_and_normalize(offer_combination, awi)
        
        # 组合得分
        # Combined score
        # 库存得分 [-1, 0]，利润得分 [-1, 1]
        # Inventory score [-1, 0], profit score [-1, 1]
        # 总分 = 库存得分 * p + 利润得分 * (1-p)
        # Total score = inventory_score * p + profit_score * (1-p)
        p = self.p_threshold  # 库存权重 Inventory weight
        combined_score = inventory_score * p + profit_score * (1 - p)
        
        return combined_score

    def normalize_final_score(self, final_score: float, score_a: float) -> float:
        """
        Normalizes the final score to be in the range [-1, 1].
        
        将最终得分归一化到 [-1, 1] 范围内。
        
        Args:
            final_score: The final score to normalize
            score_a: The score before adding the offer combination
            
        Returns:
            float: Normalized score
        """
        # 如果 score_a 为负，将其限制为一个小的正数
        # If score_a is negative, clamp it to a small positive number
        if score_a < 0:
            score_a = 0.01
            
        # 如果 final_score 为负，将其限制为一个小的正数
        # If final_score is negative, clamp it to a small positive number
        if final_score < 0:
            final_score = 0.01
            
        # 计算改进比例
        # Calculate improvement ratio
        improvement_ratio = final_score / score_a
        
        # 如果改进比例 > 1，表示有改进
        # If improvement ratio > 1, there is improvement
        if improvement_ratio > 1:
            # 归一化到 [0, 1]
            # Normalize to [0, 1]
            # 使用对数函数，避免过大的改进比例
            # Use log function to avoid too large improvement ratios
            normalized_score = min(1.0, math.log(improvement_ratio, 10))
        else:
            # 归一化到 [-1, 0]
            # Normalize to [-1, 0]
            # 使用线性函数
            # Use linear function
            normalized_score = max(-1.0, improvement_ratio - 1)
            
        return normalized_score

    def calculate_inventory_cost_score(self, im_state: InventoryManagerCIRS, current_day: int, last_simulation_day: int, unit_shortfall_penalty: float, unit_storage_cost: float) -> float:
        """
        Calculates a score based on inventory costs and shortfalls.
        
        计算基于库存成本和缺货的得分。
        
        Args:
            im_state: Inventory manager state
            current_day: Current day
            last_simulation_day: Last day to simulate
            unit_shortfall_penalty: Penalty per unit of shortfall
            unit_storage_cost: Cost per unit of storage
            
        Returns:
            float: Score in range [-1, 0] where 0 is best (no costs/shortfalls)
        """
        # 初始化成本
        # Initialize costs
        total_storage_cost = 0.0
        total_shortfall_penalty = 0.0
        
        # 模拟每一天
        # Simulate each day
        for d in range(current_day, last_simulation_day + 1):
            # 获取当天库存摘要
            # Get inventory summary for the day
            raw_stock_info = im_state.get_inventory_summary(d, MaterialType.RAW)
            product_stock_info = im_state.get_inventory_summary(d, MaterialType.PRODUCT)
            
            # 计算存储成本
            # Calculate storage costs
            raw_storage_cost = raw_stock_info.get("current_storage_cost", 0)
            product_storage_cost = product_stock_info.get("current_storage_cost", 0)
            total_storage_cost += raw_storage_cost + product_storage_cost
            
            # 计算缺货惩罚
            # Calculate shortfall penalties
            # 检查当天是否有需要交付的合同
            # Check if there are contracts that need to be delivered today
            day_for_disposal_check = d
            
            # 获取当天需要交付的产品合同
            # Get product contracts that need to be delivered today
            product_contracts_due = [c for c in im_state.pending_demand_contracts if c.delivery_time == day_for_disposal_check]
            
            # 计算当天需要交付的总量
            # Calculate total quantity that needs to be delivered today
            total_product_due = sum(c.quantity for c in product_contracts_due)
            
            # 获取当前产品库存
            # Get current product inventory
            current_product_stock = product_stock_info.get("current_stock", 0)
            
            # 计算缺货量
            # Calculate shortfall
            product_shortfall = max(0, total_product_due - current_product_stock)
            
            # 计算缺货惩罚
            # Calculate shortfall penalty
            shortfall_penalty = product_shortfall * unit_shortfall_penalty
            total_shortfall_penalty += shortfall_penalty
            
            # 模拟处理当天
            # Simulate processing the day
            im_state.process_day_end_operations(d)
            
        # 计算总成本
        # Calculate total cost
        total_cost = total_storage_cost + total_shortfall_penalty
        
        # 归一化到 [-1, 0] 范围
        # Normalize to [-1, 0] range
        # 使用指数函数，避免过大的成本
        # Use exponential function to avoid too large costs
        normalized_score = -min(1.0, total_cost / 1000.0)
        
        return normalized_score

    # ------------------------------------------------------------------
    # 🌟 6. 组合评估策略
    # Combination evaluation strategies
    # ------------------------------------------------------------------
    def _evaluate_offer_combinations_exhaustive(self, offers: Dict[str, Outcome], im: InventoryManagerCIRS, awi: OneShotAWI) -> Tuple[Dict[str, Outcome], float]:
        """
        Evaluates all possible combinations of offers using exhaustive search.
        
        使用穷举搜索评估所有可能的报价组合。
        
        Args:
            offers: Dictionary mapping negotiator IDs to their offers
            im: Inventory manager
            awi: Agent world interface
            
        Returns:
            Tuple[Dict[str, Outcome], float]: Best combination and its score
        """
        # 如果没有报价，返回空组合
        # If there are no offers, return empty combination
        if not offers:
            return {}, 0.0
            
        # 获取所有报价的 ID
        # Get IDs of all offers
        negotiator_ids = list(offers.keys())
        num_offers_available = len(negotiator_ids)
        
        # 如果报价数量太多，发出警告
        # If there are too many offers, issue a warning
        if num_offers_available > 10:
            # 报价太多，可能导致组合爆炸
            # Too many offers, may cause combination explosion
            # 限制为最多 10 个报价
            # Limit to at most 10 offers
            negotiator_ids = negotiator_ids[:10]
            num_offers_available = 10
            
        # 计算所有可能的组合
        # Calculate all possible combinations
        all_combinations = []
        for r in range(num_offers_available + 1):
            all_combinations.extend(iter_combinations(negotiator_ids, r))
            
        # 初始化最佳组合和得分
        # Initialize best combination and score
        best_combo = {}
        best_score = self.q_threshold  # 阈值 Threshold
        
        # 计算基准得分（空组合）
        # Calculate baseline score (empty combination)
        baseline_score = self.score_offers({}, im, awi)
        
        # 评估每个组合
        # Evaluate each combination
        for combo in all_combinations:
            # 跳过空组合（已经计算过基准得分）
            # Skip empty combination (already calculated baseline score)
            if not combo:
                continue
                
            # 构建组合
            # Build combination
            combo_dict = {nid: offers[nid] for nid in combo}
            
            # 评估组合
            # Evaluate combination
            combo_score = self.score_offers(combo_dict, im, awi)
            
            # 归一化得分
            # Normalize score
            normalized_score = self.normalize_final_score(combo_score, baseline_score)
            
            # 如果得分更好，更新最佳组合
            # If score is better, update best combination
            if normalized_score > best_score:
                best_combo = combo_dict
                best_score = normalized_score
                
        # 返回最佳组合和得分
        # Return best combination and score
        return best_combo, best_score

    def _evaluate_offer_combinations_k_max(self, offers: Dict[str, Outcome], im: InventoryManagerCIRS, awi: OneShotAWI) -> Tuple[Dict[str, Outcome], float]:
        """
        Evaluates combinations of offers using a greedy k-max approach.
        
        使用贪心 k-max 方法评估报价组合。
        
        Args:
            offers: Dictionary mapping negotiator IDs to their offers
            im: Inventory manager
            awi: Agent world interface
            
        Returns:
            Tuple[Dict[str, Outcome], float]: Best combination and its score
        """
        # 如果没有报价，返回空组合
        # If there are no offers, return empty combination
        if not offers:
            return {}, 0.0
            
        # 获取所有报价的 ID
        # Get IDs of all offers
        negotiator_ids = list(offers.keys())
        
        # 计算基准得分（空组合）
        # Calculate baseline score (empty combination)
        baseline_score = self.score_offers({}, im, awi)
        
        # 初始化最佳组合和得分
        # Initialize best combination and score
        best_combo = {}
        best_score = self.q_threshold  # 阈值 Threshold
        
        # 限制组合大小
        # Limit combination size
        max_combo_size = min(self.max_combo_size_for_k_max, len(negotiator_ids))
        
        # 计算所有指定大小的组合
        # Calculate all combinations of specified size
        for r in range(1, max_combo_size + 1):
            # 计算所有 r 大小的组合
            # Calculate all combinations of size r
            for combo in iter_combinations(negotiator_ids, r):
                # 构建组合
                # Build combination
                combo_dict = {nid: offers[nid] for nid in combo}
                
                # 评估组合
                # Evaluate combination
                combo_score = self.score_offers(combo_dict, im, awi)
                
                # 归一化得分
                # Normalize score
                normalized_score = self.normalize_final_score(combo_score, baseline_score)
                
                # 如果得分更好，更新最佳组合
                # If score is better, update best combination
                if normalized_score > best_score:
                    best_combo = combo_dict
                    best_score = normalized_score
                    
        # 返回最佳组合和得分
        # Return best combination and score
        return best_combo, best_score

    def _evaluate_offer_combinations_beam_search(self, offers: Dict[str, Outcome], im: InventoryManagerCIRS, awi: OneShotAWI) -> Tuple[Dict[str, Outcome], float]:
        """
        Evaluates combinations of offers using beam search.
        
        使用束搜索评估报价组合。
        
        Args:
            offers: Dictionary mapping negotiator IDs to their offers
            im: Inventory manager
            awi: Agent world interface
            
        Returns:
            Tuple[Dict[str, Outcome], float]: Best combination and its score
        """
        # 如果没有报价，返回空组合
        # If there are no offers, return empty combination
        if not offers:
            return {}, 0.0
            
        # 获取所有报价的 ID
        # Get IDs of all offers
        negotiator_ids = list(offers.keys())
        
        # 计算基准得分（空组合）
        # Calculate baseline score (empty combination)
        baseline_score = self.score_offers({}, im, awi)
        
        # 初始化最佳组合和得分
        # Initialize best combination and score
        best_combo_overall = {}
        best_score_overall = self.q_threshold  # 阈值 Threshold
        
        # 初始化束
        # Initialize beam
        beam = [{}]  # 从空组合开始 Start with empty combination
        beam_scores = [baseline_score]
        
        # 束搜索
        # Beam search
        for _ in range(len(negotiator_ids)):
            # 生成候选组合
            # Generate candidate combinations
            candidates = []
            candidate_scores = []
            
            # 对于束中的每个组合
            # For each combination in the beam
            for i, combo in enumerate(beam):
                combo_score = beam_scores[i]
                
                # 尝试添加每个未使用的报价
                # Try adding each unused offer
                for nid in negotiator_ids:
                    # 如果报价已经在组合中，跳过
                    # If offer is already in combination, skip
                    if nid in combo:
                        continue
                        
                    # 创建新组合
                    # Create new combination
                    new_combo = combo.copy()
                    new_combo[nid] = offers[nid]
                    
                    # 评估新组合
                    # Evaluate new combination
                    new_combo_score = self.score_offers(new_combo, im, awi)
                    
                    # 归一化得分
                    # Normalize score
                    normalized_score = self.normalize_final_score(new_combo_score, baseline_score)
                    
                    # 添加到候选列表
                    # Add to candidates list
                    candidates.append(new_combo)
                    candidate_scores.append(normalized_score)
                    
                    # 更新全局最佳
                    # Update global best
                    if normalized_score > best_score_overall:
                        best_combo_overall = new_combo
                        best_score_overall = normalized_score
                        
            # 如果没有候选，跳出
            # If there are no candidates, break
            if not candidates:
                break
                
            # 选择前 beam_width 个候选
            # Select top beam_width candidates
            if candidates:
                # 按得分排序
                # Sort by score
                sorted_indices = sorted(range(len(candidate_scores)), key=lambda i: candidate_scores[i], reverse=True)
                
                # 选择前 beam_width 个
                # Select top beam_width
                beam = [candidates[i] for i in sorted_indices[:self.beam_width]]
                beam_scores = [candidate_scores[i] for i in sorted_indices[:self.beam_width]]
            else:
                # 如果没有候选，清空束
                # If there are no candidates, clear beam
                beam = []
                beam_scores = []
                
        # 返回最佳组合和得分
        # Return best combination and score
        return best_combo_overall, best_score_overall

    def _evaluate_offer_combinations_simulated_annealing(self, offers: Dict[str, Outcome], im: InventoryManagerCIRS, awi: OneShotAWI) -> Tuple[Dict[str, Outcome], float]:
        """
        Evaluates combinations of offers using simulated annealing.
        
        使用模拟退火评估报价组合。
        
        Args:
            offers: Dictionary mapping negotiator IDs to their offers
            im: Inventory manager
            awi: Agent world interface
            
        Returns:
            Tuple[Dict[str, Outcome], float]: Best combination and its score
        """
        # 如果没有报价，返回空组合
        # If there are no offers, return empty combination
        if not offers:
            return {}, 0.0
            
        # 获取所有报价的 ID
        # Get IDs of all offers
        negotiator_ids = list(offers.keys())
        
        # 计算基准得分（空组合）
        # Calculate baseline score (empty combination)
        baseline_score = self.score_offers({}, im, awi)
        
        # 初始化最佳组合和得分
        # Initialize best combination and score
        best_combo = {}
        best_score = self.q_threshold  # 阈值 Threshold
        
        # 初始化当前组合和得分
        # Initialize current combination and score
        current_combo = {}
        current_score = baseline_score
        
        # 初始化温度
        # Initialize temperature
        temperature = self.sa_initial_temp
        
        # 模拟退火
        # Simulated annealing
        for _ in range(self.sa_iterations):
            # 生成邻居组合
            # Generate neighbor combination
            # 随机选择一个报价
            # Randomly select an offer
            nid = random.choice(negotiator_ids)
            
            # 如果报价在当前组合中，移除；否则添加
            # If offer is in current combination, remove it; otherwise add it
            neighbor_combo = current_combo.copy()
            if nid in neighbor_combo:
                del neighbor_combo[nid]
            else:
                neighbor_combo[nid] = offers[nid]
                
            # 评估邻居组合
            # Evaluate neighbor combination
            neighbor_score = self.score_offers(neighbor_combo, im, awi)
            
            # 归一化得分
            # Normalize score
            normalized_neighbor_score = self.normalize_final_score(neighbor_score, baseline_score)
            
            # 计算得分差异
            # Calculate score difference
            score_diff = normalized_neighbor_score - self.normalize_final_score(current_score, baseline_score)
            
            # 决定是否接受邻居
            # Decide whether to accept neighbor
            if score_diff > 0 or random.random() < math.exp(score_diff / temperature):
                # 接受邻居
                # Accept neighbor
                current_combo = neighbor_combo
                current_score = neighbor_score
                
                # 更新最佳组合
                # Update best combination
                if normalized_neighbor_score > best_score:
                    best_combo = neighbor_combo
                    best_score = normalized_neighbor_score
                    
            # 降低温度
            # Decrease temperature
            temperature *= self.sa_cooling_rate
            
        # 返回最佳组合和得分
        # Return best combination and score
        return best_combo, best_score

    def _evaluate_offer_combinations(self, offers: Dict[str, Outcome], im: InventoryManagerCIRS, awi: OneShotAWI) -> Tuple[Dict[str, Outcome], float]:
        """
        Evaluates combinations of offers using the selected strategy.
        
        使用选定的策略评估报价组合。
        
        Args:
            offers: Dictionary mapping negotiator IDs to their offers
            im: Inventory manager
            awi: Agent world interface
            
        Returns:
            Tuple[Dict[str, Outcome], float]: Best combination and its score
        """
        # 根据策略选择评估方法
        # Select evaluation method based on strategy
        if self.combo_evaluation_strategy == "exhaustive_search":
            # 穷举搜索
            # Exhaustive search
            return self._evaluate_offer_combinations_exhaustive(offers, im, awi)
        elif self.combo_evaluation_strategy == "k_max":
            # k-max 方法
            # k-max method
            return self._evaluate_offer_combinations_k_max(offers, im, awi)
        elif self.combo_evaluation_strategy == "beam_search":
            # 束搜索
            # Beam search
            return self._evaluate_offer_combinations_beam_search(offers, im, awi)
        elif self.combo_evaluation_strategy == "simulated_annealing":
            # 模拟退火
            # Simulated annealing
            return self._evaluate_offer_combinations_simulated_annealing(offers, im, awi)
        else:
            # 未知策略，使用 k-max 方法
            # Unknown strategy, use k-max method
            return self._evaluate_offer_combinations_k_max(offers, im, awi)

    def _calculate_combination_profit_and_normalize(self, offer_combination: Dict[str, Outcome], awi: OneShotAWI) -> float:
        """
        Calculates the profit of a combination of offers and normalizes it.
        
        计算报价组合的利润并归一化。
        
        Args:
            offer_combination: Dictionary mapping negotiator IDs to their offers
            awi: Agent world interface
            
        Returns:
            float: Normalized profit score in range [-1, 1]
        """
        # 如果组合为空，返回 0
        # If combination is empty, return 0
        if not offer_combination:
            return 0.0
            
        # 初始化收入和成本
        # Initialize revenue and cost
        total_revenue = 0.0
        total_cost = 0.0
        
        # 处理每个报价
        # Process each offer
        for negotiator_id, offer_outcome in offer_combination.items():
            # 如果报价为空，跳过
            # If offer is empty, skip
            if not offer_outcome:
                continue
                
            # 获取数量和价格
            # Get quantity and price
            quantity = offer_outcome.get(QUANTITY, 0)
            price = offer_outcome.get(UNIT_PRICE, 0.0)
            
            # 计算收入或成本
            # Calculate revenue or cost
            if self._is_supplier(negotiator_id):
                # 采购合同，增加成本
                # Procurement contract, increase cost
                total_cost += quantity * price
            else:
                # 销售合同，增加收入
                # Sales contract, increase revenue
                total_revenue += quantity * price
                
        # 计算利润
        # Calculate profit
        profit = total_revenue - total_cost
        
        # 归一化到 [-1, 1] 范围
        # Normalize to [-1, 1] range
        # 使用 tanh 函数
        # Use tanh function
        normalized_profit = math.tanh(profit / 1000.0)
        
        return normalized_profit

    # ------------------------------------------------------------------
    # 🌟 7. 还价生成
    # Counter offer generation
    # ------------------------------------------------------------------
    def _generate_counter_offer(self, negotiator_id: str, original_offer: Outcome, optimize_for_inventory: bool, optimize_for_profit: bool, inventory_target_quantity: Optional[int] = None) -> Outcome:
        """
        Generates a counter offer for a specific negotiator.
        
        为特定谈判者生成还价。
        
        Args:
            negotiator_id: ID of the negotiator
            original_offer: Original offer from the negotiator
            optimize_for_inventory: Whether to optimize for inventory
            optimize_for_profit: Whether to optimize for profit
            inventory_target_quantity: Target quantity for inventory optimization
            
        Returns:
            Outcome: Counter offer
        """
        # 获取原始报价的数量、价格和交付日期
        # Get quantity, price and delivery date from original offer
        orig_q = original_offer.get(QUANTITY, 0)
        orig_p = original_offer.get(UNIT_PRICE, 0.0)
        orig_t = original_offer.get(TIME, self.awi.current_step + 1)
        
        # 初始化还价
        # Initialize counter offer
        counter_offer = original_offer.copy()
        
        # 如果是供应商，优化采购
        # If it's a supplier, optimize procurement
        if self._is_supplier(negotiator_id):
            # 采购合同
            # Procurement contract
            
            # 优化数量
            # Optimize quantity
            q_candidate = orig_q
            if optimize_for_inventory and inventory_target_quantity is not None:
                # 如果需要优化库存，使用目标数量
                # If inventory optimization is needed, use target quantity
                q_candidate = inventory_target_quantity
                
            # 优化价格
            # Optimize price
            p_candidate = orig_p
            if optimize_for_profit:
                # 如果需要优化利润，降低价格
                # If profit optimization is needed, decrease price
                # 使用 NMI 的最优价格
                # Use NMI's optimal price
                p_candidate = self.awi.current_input_issues[UNIT_PRICE].min_value
                
            # 优化交付日期
            # Optimize delivery date
            t_candidate = orig_t
            # 如果交付日期太远，尝试提前
            # If delivery date is too far, try to advance it
            if t_candidate > self.awi.current_step + 3:
                t_candidate = self.awi.current_step + 1
                
            # 更新还价
            # Update counter offer
            counter_offer[QUANTITY] = q_candidate
            counter_offer[UNIT_PRICE] = p_candidate
            counter_offer[TIME] = t_candidate
            
        else:
            # 销售合同
            # Sales contract
            
            # 优化数量
            # Optimize quantity
            q_candidate = orig_q
            if optimize_for_inventory and inventory_target_quantity is not None:
                # 如果需要优化库存，使用目标数量
                # If inventory optimization is needed, use target quantity
                q_candidate = min(orig_q, inventory_target_quantity)
                
            # 优化价格
            # Optimize price
            p_candidate = orig_p
            if optimize_for_profit:
                # 如果需要优化利润，提高价格
                # If profit optimization is needed, increase price
                # 使用 NMI 的最优价格
                # Use NMI's optimal price
                p_candidate = self.awi.current_output_issues[UNIT_PRICE].max_value
                
            # 优化交付日期
            # Optimize delivery date
            t_candidate = orig_t
            # 如果交付日期太近，尝试延后
            # If delivery date is too close, try to delay it
            if t_candidate == self.awi.current_step + 1 and self.im and self.im.get_today_insufficient_raw(self.awi.current_step) > 0:
                t_candidate = self.awi.current_step + 2
                
            # 更新还价
            # Update counter offer
            counter_offer[QUANTITY] = q_candidate
            counter_offer[UNIT_PRICE] = p_candidate
            counter_offer[TIME] = t_candidate
            
        # 返回还价
        # Return counter offer
        return counter_offer

    # ------------------------------------------------------------------
    # 🌟 8. counter_all — 对所有报价进行还价
    # ------------------------------------------------------------------
    def counter_all(self, offers: Dict[str, Outcome], states: Dict[str, SAOState]) -> Dict[str, SAOResponse]:
        """
        Generates counter offers for all offers.
        
        为所有报价生成还价。
        
        Args:
            offers: Dictionary mapping negotiator IDs to their offers
            states: Dictionary mapping negotiator IDs to their states
            
        Returns:
            Dict[str, SAOResponse]: Dictionary mapping negotiator IDs to their responses
        """
        # 如果没有库存管理器或 AWI，拒绝所有报价
        # If there is no inventory manager or AWI, reject all offers
        if not self.im or not self.awi:
            return {pid: SAOResponse(ResponseType.REJECT_OFFER, None) for pid in offers}
            
        # 获取当前日期
        # Get current date
        today = self.awi.current_step
        
        # 评估最佳组合
        # Evaluate best combination
        best_combo, best_combo_score = self._evaluate_offer_combinations(offers, self.im, self.awi)
        
        # 初始化结果
        # Initialize result
        responses = {}
        
        # 如果最佳组合为空，拒绝所有报价
        # If best combination is empty, reject all offers
        if not best_combo:
            # 情况 1：没有最佳组合，拒绝所有报价
            # Case 1: No best combination, reject all offers
            
            # 分离供应和销售报价
            # Separate supply and sales offers
            supply_offers = {pid: offer for pid, offer in offers.items() if self._is_supplier(pid)}
            sales_offers = {pid: offer for pid, offer in offers.items() if self._is_consumer(pid)}
            
            # 处理供应报价
            # Process supply offers
            for pid, offer in supply_offers.items():
                # 获取数量、价格和交付日期
                # Get quantity, price and delivery date
                quantity = offer.get(QUANTITY, 0)
                price = offer.get(UNIT_PRICE, 0.0)
                delivery_time = offer.get(TIME, today + 1)
                
                # 如果价格过高，拒绝
                # If price is too high, reject
                if price > self.awi.current_input_issues[UNIT_PRICE].max_value * 0.8:
                    # 生成还价
                    # Generate counter offer
                    counter_offer = self._generate_counter_offer(
                        negotiator_id=pid,
                        original_offer=offer,
                        optimize_for_inventory=True,
                        optimize_for_profit=True,
                        inventory_target_quantity=None
                    )
                    
                    # 添加到结果
                    # Add to result
                    responses[pid] = SAOResponse(ResponseType.REJECT_OFFER, counter_offer)
                else:
                    # 否则，接受
                    # Otherwise, accept
                    responses[pid] = SAOResponse(ResponseType.ACCEPT_OFFER, None)
                    
            # 处理销售报价
            # Process sales offers
            for pid, offer in sales_offers.items():
                # 获取数量、价格和交付日期
                # Get quantity, price and delivery date
                quantity = offer.get(QUANTITY, 0)
                price = offer.get(UNIT_PRICE, 0.0)
                delivery_time = offer.get(TIME, today + 1)
                
                # 如果价格过低，拒绝
                # If price is too low, reject
                if price < self.awi.current_output_issues[UNIT_PRICE].min_value * 1.2:
                    # 生成还价
                    # Generate counter offer
                    counter_offer = self._generate_counter_offer(
                        negotiator_id=pid,
                        original_offer=offer,
                        optimize_for_inventory=True,
                        optimize_for_profit=True,
                        inventory_target_quantity=None
                    )
                    
                    # 添加到结果
                    # Add to result
                    responses[pid] = SAOResponse(ResponseType.REJECT_OFFER, counter_offer)
                else:
                    # 否则，接受
                    # Otherwise, accept
                    responses[pid] = SAOResponse(ResponseType.ACCEPT_OFFER, None)
                    
            return responses
            
        # 情况 2：有最佳组合，接受最佳组合中的报价，拒绝其他报价
        # Case 2: Has best combination, accept offers in best combination, reject others
        
        # 获取最佳组合中的谈判者 ID
        # Get negotiator IDs in best combination
        nids_in_best = set(best_combo.keys())
        
        # 处理每个报价
        # Process each offer
        for pid, offer in offers.items():
            # 如果报价在最佳组合中，接受
            # If offer is in best combination, accept
            if pid in nids_in_best:
                responses[pid] = SAOResponse(ResponseType.ACCEPT_OFFER, None)
            else:
                # 否则，拒绝并生成还价
                # Otherwise, reject and generate counter offer
                
                # 生成还价
                # Generate counter offer
                counter_offer = self._generate_counter_offer(
                    negotiator_id=pid,
                    original_offer=offer,
                    optimize_for_inventory=True,
                    optimize_for_profit=True,
                    inventory_target_quantity=None
                )
                
                # 添加到结果
                # Add to result
                responses[pid] = SAOResponse(ResponseType.REJECT_OFFER, counter_offer)
                
        return responses

    # ------------------------------------------------------------------
    # 🌟 9. 合同管理
    # Contract management
    # ------------------------------------------------------------------
    def get_partner_id(self, contract: Contract) -> str | None:
        """
        Gets the partner ID from a contract.
        
        从合同中获取伙伴 ID。
        
        Args:
            contract: Contract
            
        Returns:
            str | None: Partner ID or None if not found
        """
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

    def on_negotiation_success(self, contract: Contract, mechanism: StdAWI) -> None:
        """
        Called when a negotiation succeeds.
        
        谈判成功时调用。
        
        Args:
            contract: Contract
            mechanism: Agent world interface
        """
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

if __name__ == "__main__":
    pass