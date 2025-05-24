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
from collections import Counter
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
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        # —— 参数 ——
        self.total_insufficient = None
        self.today_insufficient = None
        self.min_profit_margin = min_profit_margin          # 最低利润率⬆ Minimum profit margin
        self.cheap_price_discount = cheap_price_discount    # 超低价折扣阈值⬇ Threshold for extremely low prices

        # —— 运行时变量 ——
        self.im: InventoryManager | None = None             # 库存管理器实例 Inventory manager instance
        self._market_price_avg: float = 0.0                 # 最近报价平均价 (估算市场均价) Recent market price average
        self._market_material_price_avg: float = 0.0        # 原料均价 Rolling window average for raw materials
        self._market_product_price_avg: float = 0.0         # 产品均价 Rolling window average for products
        self._recent_material_prices: List[float] = []      # 用滚动窗口估计市场价 Rolling window for market price
        self._recent_product_prices: List[float] = []
        self._avg_window: int = 30                          # 均价窗口大小 Average window size
        self._ptoday: float = ptoday                        # 当期挑选伙伴比例 Proportion of partners selected today
        self.model = None                                   # 预留的决策模型 Placeholder for decision model
        self.concession_model = None                        # 让步幅度模型接口
        self._last_offer_price: Dict[str, float] = {}
        # 记录每天的采购/销售完成量 {day: quantity}
        # Track daily completed purchase/sales quantity
        self.sales_completed: Dict[int, int] = {}
        self.purchase_completed: Dict[int, int] = {}  # 销售完成量 Purchase completion count

        # Opponent modeling statistics
        # {pid: {"avg_price": float, "price_M2": float, "contracts": int, "success": int}}
        self.partner_stats: Dict[str, Dict[str, float]] = {}
        # Logistic acceptance models per partner {pid: {"w0": float, "w1": float}}
        self.partner_models: Dict[str, Dict[str, float]] = {}
        # Track last price offered by partners
        self._last_partner_offer: Dict[str, float] = {}

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

    def before_step(self) -> None:
        """每天开始前，同步日内关键需求信息。"""
        assert self.im, "InventoryManager 尚未初始化!"
        self.today_insufficient = self.im.get_today_insufficient(self.awi.current_step)
        self.total_insufficient = self.im.get_total_insufficient(self.awi.current_step)

        # 根据库存与进度动态调整囤货折扣阈值
        raw_summary = self.im.get_inventory_summary(self.awi.current_step, MaterialType.RAW)
        available = raw_summary["estimated_available"]
        needed = max(1, self.total_insufficient)
        ratio = available / needed
        progress = self.awi.current_step / max(1, self.awi.n_steps)
        
        # Step 1: Determine base discount based on progress
        if progress >= 0.66:
            progress_discount = 0.5
        elif progress >= 0.33:
            progress_discount = 0.6
        else:
            progress_discount = 0.7
        
        # Step 2: Adjust discount based on inventory ratio
        if ratio >= 1.0:
            ratio_discount = 0.5
        elif ratio < 0.8:
            ratio_discount = 0.8
        else:
            ratio_discount = progress_discount
        
        # Step 3: Combine discounts, prioritizing the stricter condition
        self.cheap_price_discount = max(progress_discount, ratio_discount)
        
        # Inline comments explaining thresholds:
        # - Progress thresholds: Encourage higher discounts earlier in the timeline.
        # - Ratio thresholds: Ensure sufficient inventory by prioritizing higher discounts when inventory is low.

        # 初始化当日的完成量记录
        # Initialize today's completion records
        self.sales_completed.setdefault(self.awi.current_step, 0)
        self.purchase_completed.setdefault(self.awi.current_step, 0)

        # 将外生协议写入im
        # Write exogenous contracts into the inventory manager
        if self.awi.is_first_level:
            exogenous_contract_quantity = self.awi.current_exogenous_input_quantity
            exogenous_contract_price = self.awi.current_exogenous_input_price
            exogenous_contract_day = self.awi.current_step
            exogenous_contract_id = str(uuid4())
            exogenous_contract_partner = "simulator"

            exogenous_contract = IMContract(
                contract_id = exogenous_contract_id,
                partner_id = exogenous_contract_partner,
                type = IMContractType.SUPPLY,
                quantity = exogenous_contract_quantity,
                price = exogenous_contract_price,
                delivery_time = exogenous_contract_day,
                bankruptcy_risk = 0,
                material_type = MaterialType.RAW
            )
            self.im.add_transaction(exogenous_contract)
        elif self.awi.is_last_level:
            exogenous_contract_quantity = self.awi.current_exogenous_output_quantity
            exogenous_contract_price = self.awi.current_exogenous_output_price
            exogenous_contract_day = self.awi.current_step
            exogenous_contract_id = str(uuid4())
            exogenous_contract_partner = "simulator"
            exogenous_contract = IMContract(
                contract_id = exogenous_contract_id,
                partner_id = exogenous_contract_partner,
                type = IMContractType.DEMAND,
                quantity = exogenous_contract_quantity,
                price = exogenous_contract_price,
                delivery_time = exogenous_contract_day,
                bankruptcy_risk = 0,
                material_type = MaterialType.PRODUCT
            )
            self.im.add_transaction(exogenous_contract)

    def step(self) -> None:
        """每天结束时调用：执行 IM 的日终操作并刷新市场均价。"""
        assert self.im, "InventoryManager 尚未初始化!"
        # 让 IM 完成收货 / 生产 / 交付 / 规划
        self.im.process_day_operations()
        self.im.update_day()
        # —— 更新市场均价估计 ——
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
            var = stats.get("price_M2", 0.0) / max(1, stats.get("success", 0) - 1)
            std = var ** 0.5
            rate = stats.get("success", 0) / max(1, stats.get("contracts", 0))
            base = mean + std * (1 - rate)
        else:
            base = default
        # if we are buying and penalty is high, be willing to pay up to penalty
        if self._is_supplier(pid):
            base = max(base, self.awi.current_shortfall_penalty * 0.8)
        # Blend with reservation price estimated by logistic model
        model_price = self._estimate_reservation_price(pid, base)
        return (base + model_price) / 2

    # ------------------------------------------------------------------
    # 🌟 3-b. 动态让步策略
    # ------------------------------------------------------------------

    def _calc_opponent_concession(self, pid: str, price: float) -> float:
        """估算对手的让步速度（相邻报价差异比例）。"""
        last = self._last_offer_price.get(pid)
        self._last_offer_price[pid] = price
        if last is None or last == 0:
            return 0.0
        return abs(price - last) / abs(last)

    def _concession_multiplier(self, rel_time: float, opp_rate: float = 0.0) -> float:
        """根据时间和对手让步速度计算让步幅度。"""
        if self.concession_model:
            return self.concession_model(rel_time, opp_rate)
        base = rel_time * (1 + opp_rate)
        return max(0.0, min(1.0, base))

    def _apply_concession(
        self,
        pid: str,
        target_price: float,
        state: SAOState | None,
        current_price: float,
    ) -> float:
        """根据协商进度计算让步后的价格。"""
        start = self._best_price(pid)
        opp_rate = self._calc_opponent_concession(pid, current_price)
        rel = state.relative_time if state else 0.0
        # 加入短缺罚金影响，罚金越高让步越快
        penalty_factor = min(1.0, self.awi.current_shortfall_penalty / 10.0)
        mult = self._concession_multiplier(rel, opp_rate) + penalty_factor
        mult = max(0.0, min(1.0, mult))

        # 结合对手历史平均价作为期望目标价
        target_price = (target_price + self._expected_price(pid, target_price)) / 2
        if self._is_consumer(pid):
            # 我是卖家，价格从高到低
            price = start - (start - target_price) * mult
            price = max(target_price, price)
        else:
            # 我是买家，价格从低到高
            price = start + (target_price - start) * mult
            price = min(target_price, price)
        return self._clamp_price(pid, price)

    # ------------------------------------------------------------------
    # 🌟 Opponent utility estimation using logistic regression
    # ------------------------------------------------------------------

    def _update_acceptance_model(self, pid: str, price: float, accepted: bool) -> None:
        """Online update of a logistic acceptance model for a partner."""
        model = self.partner_models.setdefault(pid, {"w0": 0.0, "w1": 0.0})
        x = price if self._is_supplier(pid) else -price
        z = model["w0"] + model["w1"] * x
        pred = 1.0 / (1.0 + math.exp(-z))
        y = 1.0 if accepted else 0.0
        err = y - pred
        lr = 0.05
        model["w0"] += lr * err
        model["w1"] += lr * err * x

    def _estimate_reservation_price(self, pid: str, default: float) -> float:
        """Predict the partner's reservation price from the learned model."""
        model = self.partner_models.get(pid)
        if not model or abs(model["w1"]) < 1e-6:
            return default
        price_sign = 1.0 if self._is_supplier(pid) else -1.0
        return (-model["w0"] / model["w1"]) * price_sign

    def _pareto_counter_offer(
        self, pid: str, qty: int, t: int, price: float, state: SAOState | None
    ) -> Outcome:
        """Generate a counter offer balancing our target and estimated reservation price."""
        opp_price = self._estimate_reservation_price(pid, price)
        best_price = self._best_price(pid)
        target = self._apply_concession(pid, best_price, state, price)
        new_price = (opp_price + target) / 2
        new_price = self._clamp_price(pid, new_price)
        if self._is_supplier(pid) and self.awi.current_shortfall_penalty > 1.0:
            qty = int(qty * 1.1)
        return (qty, max(t, self.awi.current_step), new_price)

    # ------------------------------------------------------------------
    # 🌟 3-a. 需求计算和需求分配
    # ------------------------------------------------------------------

    def _get_sales_demand_first_layer(self) -> int:
        """
        第一层的销售需求，因为外生协议保证不超过产能，因此不考虑产能不足的问题
        由于协议会添加到库存中，直接调用库存管理器的预期可用库存（真实库存+即将入库） 说白了就是：只要能生产，剩多少就卖多少
        没卖完的im会自己留作库存
        *** 注意：由于签署未来协议不会扣减当前可用库存预期，为了防止罚款，当为首层时，只签署当日协议！ ***
        签署销售订单后正常排单，库存管理器会自己扣除库存
        """
        # For the first layer, sales demand equals available production since exogenous contracts ensure capacity is not exceeded.
        # Unsold products remain in inventory. We only sign contracts for the current day to avoid penalties.
        today_inventory_material = int(min(self.im.get_inventory_summary(self.awi.current_step, MaterialType.RAW)["estimated_available"], self.im.get_max_possible_production(self.awi.current_step)))
        return today_inventory_material

    def _get_sales_demand_last_layer(self) -> int:
        # 最后一层的销售需求为0，销售的外生协议由库存管理器管理，并将数据用于计算购买需求
        # Last layer has no sales demand; exogenous contracts are handled by the inventory manager and used to compute purchase needs
        return 0

    def _get_sales_demand_middle_layer_today(self) -> int:
        # 这个方法计算的是中间层 * 今天 * 的销售需求
        # Today's sales demand for middle layers = today's capacity - today's production plan + today's expected inventory
        # The expected inventory is real stock + scheduled production (including future) - signed contracts (query via IM)
        today_inventory_product = int(self.im.get_inventory_summary(self.awi.current_step, MaterialType.PRODUCT)["estimated_available"])
        return today_inventory_product

    def _get_sales_demand_middle_layer(self, day: int) -> int:
        # 这个方法计算的是中间层 * 在day的 * 销售需求
        # Sales demand for middle layers on a specific day
        # 在day的销售需求 = 到day为止的产能 + 今天的库存 - 到day为止的销售
        future_inventory_product = int(self.im.get_inventory_summary(day, MaterialType.PRODUCT)["estimated_available"])
        return future_inventory_product

    def _get_supply_demand_middle_last_layer_today(self) -> tuple[int, int, float]:
        # 这个方法计算的是中间层和最后层 * 今天 * 的购买需求
        # Calculate today's purchase demand for middle and last layers
        # return 紧急需求 计划需求 超额需求(超额需求是计划需求的20%)
        # returns emergency, planned and optional (20% extra) needs
        return (self.im.get_today_insufficient(self.awi.current_step),
                self.im.get_total_insufficient(self.awi.current_step),
                self.im.get_total_insufficient(self.awi.current_step) * 0.2)

    def _get_supply_demand_middle_last_layer(self, day: int) -> tuple[int, int, float]:
        # 这个方法计算的是中间层和最后层 * 在day的 * 购买需求
        # For the last layer this is mostly meaningless as exogenous contracts are same-day,
        # but it is future-proof if such contracts extend to futures
        # return 紧急需求 计划需求 超额需求(超额需求是计划需求的20%)
        # returns emergency, planned and optional (20% extra) needs
        return (
            self.im.get_total_insufficient(day),
            self.im.get_total_insufficient(day),
            self.im.get_total_insufficient(day) * 0.2,
        )

    def _get_supply_demand_first_layer(self) -> Tuple[int, int, int]:
        # 第一层没有采购需求，统统外生
        # 但是协议还是要录入im的
        return 0,0,0

    def _distribute_todays_needs(self, partners: Iterable[str] | None = None) -> Dict[str, int]:
        """随机将今日需求分配给一部分伙伴（按 _ptoday 比例）。"""
        # 暂且先这样
        # For now we keep this simple
        if partners is None:
            partners = self.negotiators.keys()
        partners = list(partners)

        # 初始化：默认所有伙伴分配量 0
        # Initialize all partner allocations to zero
        response: Dict[str, int] = {p: 0 for p in partners}

        # 分类伙伴
        # Classify partners
        suppliers = [p for p in partners if self._is_supplier(p)]
        consumers = [p for p in partners if self._is_consumer(p)]

        # buy_need, sell_need = self._needs_today()
        # 如果是第一层
        # If this is the first layer
        if self.awi.is_first_level:
            buy_need : int = sum(self._get_supply_demand_first_layer())
            sell_need : int = self._get_sales_demand_first_layer()
        # 如果是最后一层
        # If this is the last layer
        elif self.awi.is_last_level:
            buy_need : int = sum(self._get_supply_demand_middle_last_layer_today())
            sell_need : int = self._get_sales_demand_last_layer()
        # 如果在中间
        # Otherwise we are in the middle layer
        else:
            buy_need : int = sum(self._get_supply_demand_middle_last_layer_today())
            sell_need : int = self._get_sales_demand_middle_layer_today()

        # --- 1) 分配采购需求给供应商 ---
        # Allocate purchase needs to suppliers
        if suppliers and isinstance(buy_need, tuple):
            response.update(self._distribute_to_partners(suppliers, buy_need))

        # --- 2) 分配销售需求给顾客 ---
        # Allocate sales needs to consumers
        if consumers and sell_need > 0:
            # 由于计算需求时已经做过了限制，所以这里不需要再判断了
            # No further checks needed as demand calculations already apply limits
            response.update(self._distribute_to_partners(consumers, sell_need))

        return response

    def _distribute_to_partners(self, partners: List[str], needs: int) -> Dict[str, int]:
        """核心分配：随机挑选 ``_ptoday`` 比例伙伴分配 ``needs``。"""
        if needs <= 0 or not partners:
            return {p: 0 for p in partners}

        # 确保needs是整数
        needs = int(needs)  # 将needs转换为整数

        # 根据过往成功率为伙伴排序, 成功率高的优先分配
        partners.sort(
            key=lambda p: self.partner_stats.get(p, {}).get("success", 0)
            / max(1, self.partner_stats.get(p, {}).get("contracts", 0)),
            reverse=True,
        )
        random.shuffle(partners)

        k = max(1, int(len(partners) * self._ptoday))
        chosen = partners[:k]

        if needs < len(chosen):
            chosen = random.sample(chosen, random.randint(1, needs))

        quantities = _distribute(needs, len(chosen))
        distribution = dict(zip(chosen, quantities))
        return {p: distribution.get(p, 0) for p in partners}
    # ------------------------------------------------------------------
    # 🌟 4. first_proposals — 首轮报价（可简化）
    # ------------------------------------------------------------------

        # ------------------------------------------------------------------
    # 🌟 首轮报价 first_proposals
    # ------------------------------------------------------------------
    def first_proposals(self) -> Dict[str, Outcome]:
        """根据所在层级与当日需求生成 *首轮* 报价。

        逻辑概览
        ----------
        * 若代理位于 **第一层** （`awi.is_first_level=True`） ⇒ 只需要对下游
          顾客(消费者)发盘；上游供货合同全部由环境外生生成。
        * 若代理位于 **最后一层** (`awi.is_last_level=True`) ⇒ 只需要对上游
          供应商发盘；下游销售合同由环境外生生成。
        * 其余情况：同时对上游采购与下游销售谈判，数量基于
          `_distribute_todays_needs()` 结果。

        报价策略
        ----------
        * **销售**：单价取 `_best_price(pid)` (议题合法最高价)。
        * **采购**：单价取 `_best_price(pid)` (议题合法最低价)。
        * 未分配数量的伙伴暂不发盘，留待后续轮次处理。
        """
        partners = list(self.negotiators.keys())
        if not partners:
            return {}

        # —— 1. 过滤出需要发盘的伙伴 ——
        filtered: List[str] = []
        for pid in partners:
            if self._is_supplier(pid) and self.awi.is_first_level:
                # 第一层 ➔ 不需与供应商谈判
                continue
            if self._is_consumer(pid) and self.awi.is_last_level:
                # 最后一层 ➔ 不需与顾客谈判
                continue
            filtered.append(pid)
        if not filtered:
            return {}

        # —— 2. 计算分配量 ——
        distribution = self._distribute_todays_needs(filtered)

        # —— 3. 构建 Outcome 报价 ——
        today = self.awi.current_step
        proposals: Dict[str, Outcome] = {}
        for pid, qty in distribution.items():
            if qty <= 0:
                continue  # 0 量则暂不报盘
            price = self._best_price(pid)
            proposals[pid] = (qty, today, price)

        return proposals

    # ------------------------------------------------------------------
    # 🌟 5. counter_all — 谈判核心（分派到子模块）
    # ------------------------------------------------------------------

    def counter_all(
        self,
        offers: Dict[str, Outcome],
        states: Dict[str, SAOState],
    ) -> Dict[str, SAOResponse]:
        """根据报价来源（供应 / 顾客）拆分并调用子处理器。"""
        responses: Dict[str, SAOResponse] = {}
        # -------- 5‑A 供应商报价 --------
        supply_offers = {p: o for p, o in offers.items() if self._is_supplier(p)}
        supply_states = {p: states[p] for p in supply_offers}
        responses.update(self._process_supply_offers(supply_offers, supply_states))
        # -------- 5‑B 顾客报价 --------
        demand_offers = {p: o for p, o in offers.items() if self._is_consumer(p)}
        demand_states = {p: states[p] for p in demand_offers}
        responses.update(self._process_sales_offers(demand_offers, demand_states))
        return responses

    # ------------------------------------------------------------------
    # 🌟 5‑1. 供应报价拆分三类
    # Split supply offers into three categories
    # ------------------------------------------------------------------

    def _process_supply_offers(
        self,
        offers: Dict[str, Outcome],
        states: Dict[str, SAOState],
    ) -> Dict[str, SAOResponse]:
        """将供应报价拆分三类并整合结果。"""
        res: Dict[str, SAOResponse] = {}
        if not offers:
            return res

        today = self.awi.current_step
        today_handled_emergency_demand = 0
        today_handled_planned_demand = 0
        today_handled_optional_demand = 0

        offer_deliver_today = {}
        offer_deliver_later_planned = {}
        offer_deliver_optional_demand = {}

        # 将offer从低价到高价排序
        ordered = sorted(offers.items(), key=lambda x: x[1][UNIT_PRICE])

        for pid, offer in offers.items():
            self._last_partner_offer[pid] = offer[UNIT_PRICE]
            # 如果今天的紧急需求还没有满足
            if offer[TIME] == today and today_handled_emergency_demand < self.im.get_today_insufficient(self.awi.current_step):
                offer_deliver_today[pid] = offer
                today_handled_emergency_demand += offer[QUANTITY]
            # 如果今天的未来需求还没满足
            elif offer[TIME] > today and today_handled_planned_demand < self.im.get_total_insufficient(offer[TIME]):
                offer_deliver_later_planned[pid] = offer
                today_handled_planned_demand += offer[QUANTITY]
            # 如果今天的计划需求和紧急需求都满足了
            else:
                offer_deliver_optional_demand[pid] = offer
                today_handled_optional_demand += offer[QUANTITY]


        # —— 紧急需求：仅当今日仍有不足量时处理 ——
        em_res = self._process_emergency_supply_offers(
            offer_deliver_today, {p: states[p] for p in offer_deliver_today}
        )
        res.update(em_res)
        # 如果这样还满足不了今天的紧急需求，就拿一些未来报价来改日期
        # If emergency demand is still unmet, shift some future offers to today
        # 若仍有紧急需求未满足, 尝试从未来的报价中提前交付

        today_need = self.im.get_today_insufficient(self.awi.current_step)
        today_supplied = sum(
            resp.outcome[QUANTITY]
            for resp in em_res.values()
            if resp.outcome is not None and resp.outcome[TIME] == today
        )
        if today_need > today_supplied:
            shortage = today_need - today_supplied
            future_offers = sorted(
                offer_deliver_later_planned.items(), key=lambda x: x[1][UNIT_PRICE]
            )
            for pid, offer in future_offers:
                self._last_partner_offer[pid] = offer[UNIT_PRICE]
                if shortage <= 0:
                    break
                qty, price = offer[QUANTITY], offer[UNIT_PRICE]
                take = min(qty, shortage)
                counter_offer = self._pareto_counter_offer(
                    pid, take, today, price, states.get(pid)
                )
                res[pid] = SAOResponse(ResponseType.REJECT_OFFER, counter_offer)
                shortage -= take
                remaining = qty - take
                if remaining > 0:
                    offer_deliver_later_planned[pid] = (remaining, offer[TIME], price)
                else:
                    offer_deliver_later_planned.pop(pid, None)


        # —— 计划性需求 ——
        plan_res = self._process_planned_supply_offers(
            offer_deliver_later_planned, {p: states[p] for p in offer_deliver_later_planned}
        )
        res.update(plan_res)
        # —— 机会性采购 ——
        optional_res = self._process_optional_supply_offers(
            offer_deliver_optional_demand,
            {p: states[p] for p in offer_deliver_optional_demand},
        )
        res.update(optional_res)

        return res

    # ------------------------------------------------------------------
    # 🌟 5‑1‑a 紧急需求处理
    # ------------------------------------------------------------------

    def _process_emergency_supply_offers(
        self,
        offers: Dict[str, Outcome],
        states: Dict[str, SAOState],
    ) -> Dict[str, SAOResponse]:
        """处理今日必须到货的原料报价。"""
        res: Dict[str, SAOResponse] = {}
        if not offers or self.today_insufficient <= 0:
            return res
        # 按单价升序排序，优先选便宜的
        ordered = sorted(offers.items(), key=lambda x: x[1][UNIT_PRICE])
        remain_needed = self.today_insufficient
        penalty = self.awi.current_shortfall_penalty
        for pid, offer in ordered:
            self._last_partner_offer[pid] = offer[UNIT_PRICE]
            qty, price = offer[QUANTITY], offer[UNIT_PRICE]
            state = states.get(pid)
            best_price = self.get_nmi(pid).issues[UNIT_PRICE].min_value
            expected = self._expected_price(pid, best_price)
            # 更新均价窗口
            self._recent_material_prices.append(price)
            if len(self._recent_material_prices) > self._avg_window:
                self._recent_material_prices.pop(0)

            # Accept if price is not much higher than penalty
            if price <= penalty * 1.1:
                if penalty > price or qty <= remain_needed * 1.5:
                    accept_qty = min(qty, remain_needed)
                else:
                    accept_qty = remain_needed
                accept_offer = (accept_qty, offer[TIME], price)
                res[pid] = SAOResponse(ResponseType.ACCEPT_OFFER, accept_offer)
                remain_needed -= accept_qty
                if qty > accept_qty:
                    counter_qty = qty - accept_qty
                    counter_offer = self._pareto_counter_offer(
                        pid,
                        counter_qty,
                        max(offer[TIME], self.awi.current_step),
                        price,
                        state,
                    )
                    res[pid] = SAOResponse(ResponseType.REJECT_OFFER, counter_offer)
                if remain_needed <= 0:
                    break
                continue
            if price > penalty:
                counter = self._pareto_counter_offer(
                    pid,
                    qty,
                    max(offer[TIME], self.awi.current_step),
                    price,
                    state,
                )
                res[pid] = SAOResponse(ResponseType.REJECT_OFFER, counter)
                continue
            accept_qty = min(qty, remain_needed)
            if offer[TIME] == self.awi.current_step and accept_qty == qty:
                accept_offer = (accept_qty, offer[TIME], price)
                res[pid] = SAOResponse(ResponseType.ACCEPT_OFFER, accept_offer)
                remain_needed -= accept_qty
            else:
                counter_offer = self._pareto_counter_offer(
                    pid,
                    accept_qty,
                    self.awi.current_step,
                    price,
                    state,
                )
                res[pid] = SAOResponse(ResponseType.REJECT_OFFER, counter_offer)
            # 若还有余量未用，可压价重新还价
            if qty > accept_qty and remain_needed <= 0:
                counter_qty = qty - accept_qty
                counter_offer = self._pareto_counter_offer(
                    pid,
                    counter_qty,
                    max(offer[TIME], self.awi.current_step),
                    price,
                    state,
                )
                res[pid] = SAOResponse(ResponseType.REJECT_OFFER, counter_offer)
            if remain_needed <= 0:
                break
        return res

    # ------------------------------------------------------------------
    # 🌟 5‑1‑b 计划性需求处理
    # ------------------------------------------------------------------

    def _process_planned_supply_offers(
        self,
        offers: Dict[str, Outcome],
        states: Dict[str, SAOState],
    ) -> Dict[str, SAOResponse]:
        """为未来生产需求采购原料：全局控制每天的采购量。"""
        res: Dict[str, SAOResponse] = {}
        if not offers:
            return res

        # 计算各交货日仍可采购的上限 (需求量 120% 减去已签/预计库存)
        days = {o[TIME] for o in offers.values()}
        remain_need = {}
        for d in days:
            limit = self.im.get_total_insufficient(d) * 1.2
            est_inv = self.im.get_inventory_summary(d, MaterialType.RAW)["estimated_available"]
            remain_need[d] = max(0, int(limit - est_inv))

        # 按单价从低到高依次处理
        ordered = sorted(offers.items(), key=lambda x: x[1][UNIT_PRICE])
        for pid, offer in ordered:
            qty, t, price = offer[QUANTITY], offer[TIME], offer[UNIT_PRICE]
            self._last_partner_offer[pid] = price
            state = states.get(pid)

            self._recent_material_prices.append(price)
            if len(self._recent_material_prices) > self._avg_window:
                self._recent_material_prices.pop(0)

            est_sell_price = self._market_price_avg if self._market_price_avg > 0 else price * 2
            max_price_allowed = est_sell_price / (1 + self.min_profit_margin)

            need = remain_need.get(t, 0)
            if need <= 0:
                counter = self._pareto_counter_offer(pid, 0, t, price, state)
                res[pid] = SAOResponse(ResponseType.REJECT_OFFER, counter)
                continue

            accept_qty = min(qty, need)
            if price <= max_price_allowed:
                res[pid] = SAOResponse(ResponseType.ACCEPT_OFFER, (accept_qty, t, price))
            else:
                counter = self._pareto_counter_offer(pid, accept_qty, t, price, state)
                res[pid] = SAOResponse(ResponseType.REJECT_OFFER, counter)

            remain_need[t] = max(0, remain_need[t] - accept_qty)

            if qty > accept_qty:
                counter_qty = qty - accept_qty
                counter_offer = self._pareto_counter_offer(pid, counter_qty, t, price, state)
                if pid in res and res[pid].response_type == ResponseType.ACCEPT_OFFER:
                    # Combine ACCEPT_OFFER with counter-offer for remaining quantity
                    res[pid].outcome = (accept_qty, t, price, counter_offer)
                else:
                    res[pid] = SAOResponse(ResponseType.ACCEPT_OFFER, (accept_qty, t, price, counter_offer))
        return res

    # ------------------------------------------------------------------
    # 🌟 5‑1‑c 机会性采购处理
    # ------------------------------------------------------------------

    def _process_optional_supply_offers(
        self,
        offers: Dict[str, Outcome],
        states: Dict[str, SAOState],
    ) -> Dict[str, SAOResponse]:
        """仅在超低价时囤货。"""
        res: Dict[str, SAOResponse] = {}
        if not offers:
            return res

        days = {o[TIME] for o in offers.values()}
        remain_need: Dict[int, int] = {}
        for d in days:
            limit = self.im.get_total_insufficient(d) * 1.2
            est_inv = self.im.get_inventory_summary(d, MaterialType.RAW)["estimated_available"]
            remain_need[d] = max(0, int(limit - est_inv))

        ordered = sorted(offers.items(), key=lambda x: x[1][UNIT_PRICE])
        for pid, offer in ordered:
            qty, t, price = offer[QUANTITY], offer[TIME], offer[UNIT_PRICE]
            self._last_partner_offer[pid] = price
            state = states.get(pid)
            self._recent_material_prices.append(price)
            if len(self._recent_material_prices) > self._avg_window:
                self._recent_material_prices.pop(0)
            threshold = self._market_price_avg * self.cheap_price_discount if self._market_price_avg else price * 2
            need = remain_need.get(t, 0)
            if price <= threshold and need > 0:
                accept_qty = min(qty, need)
                res[pid] = SAOResponse(ResponseType.ACCEPT_OFFER, (accept_qty, t, price))
                remain_need[t] = max(0, remain_need[t] - accept_qty)
                if qty > accept_qty:
                    counter_qty = qty - accept_qty
                    counter_offer = self._pareto_counter_offer(pid, counter_qty, t, price, state)
                    res[pid] = SAOResponse(ResponseType.REJECT_OFFER, counter_offer)
            else:
                counter = self._pareto_counter_offer(pid, min(qty, need), t, price, state)
                res[pid] = SAOResponse(ResponseType.REJECT_OFFER, counter)
        return res

    # ------------------------------------------------------------------
    # 🌟 5‑2. 销售报价处理
    # Processing of sales offers
    # ------------------------------------------------------------------

    def _process_sales_offers(
        self,
        offers: Dict[str, Outcome],
        states: Dict[str, SAOState],
    ) -> Dict[str, SAOResponse]:
        """确保不超产能且满足利润率。"""
        # TODO: 有一个问题：如果达成了一笔当天的协议。那么这笔协议的不足量就会瞬间变成当天必须实现的采购量，需要根据这个重新规划生产、重新计算所需库存量来提出购买要求
        res: Dict[str, SAOResponse] = {}
        if not offers:
            return res
        assert self.im, "InventoryManager 未初始化"
        for pid, offer in offers.items():
            qty, t, price = offer[QUANTITY], offer[TIME], offer[UNIT_PRICE]
            state = states.get(pid)

            # 更新均价窗口
            self._recent_product_prices.append(price)
            if len(self._recent_product_prices) > self._avg_window:
                self._recent_product_prices.pop(0)

            # 1) 产能检查
            signed_qty = sum(
                c.quantity for c in self.im.get_pending_contracts(is_supply=False, day=t)
            )
            max_prod = self.im.get_max_possible_production(t)
            if signed_qty + qty > max_prod:
                # 超产能：部分接受或拒绝（简化：拒绝并还价减量）
                accept_qty = max_prod - signed_qty
                if accept_qty > 0:
                    counter_offer = self._pareto_counter_offer(
                        pid, accept_qty, t, price, state
                    )
                    res[pid] = SAOResponse(ResponseType.REJECT_OFFER, counter_offer)
                else:
                    res[pid] = SAOResponse(ResponseType.REJECT_OFFER, None)
                continue
            # 2) 利润检查
            # Profit check
            # 估算单位成本：用最近平均原料价 + 加工
            # Estimate unit cost: recent average raw price plus processing
            avg_raw_cost = self._market_price_avg or price * 0.5
            unit_cost = avg_raw_cost + self.im.processing_cost
            min_sell_price = unit_cost * (1 + self.min_profit_margin)
            if price >= min_sell_price:
                res[pid] = SAOResponse(ResponseType.ACCEPT_OFFER, offer)
            else:
                high_price = self.get_nmi(pid).issues[UNIT_PRICE].max_value
                counter_offer = self._pareto_counter_offer(
                    pid, qty, t, price, state
                )
                res[pid] = SAOResponse(ResponseType.REJECT_OFFER, counter_offer)
        return res

    # ------------------------------------------------------------------
    # 🌟 6. 谈判回调
    # ------------------------------------------------------------------

    def on_negotiation_failure(
        self,
        partners: List[str],
        annotation: Dict[str, Any],
        mechanism: StdAWI,
        state: SAOState,
    ) -> None:
        """谈判失败时更新伙伴统计信息"""
        for pid in partners:
            if pid == self.id:
                continue
            stats = self.partner_stats.setdefault(
                pid,
                {"avg_price": 0.0, "price_M2": 0.0, "contracts": 0, "success": 0},
            )
            stats["contracts"] += 1
            last_price = self._last_partner_offer.get(pid)
            if last_price is not None:
                self._update_acceptance_model(pid, last_price, False)

    def on_negotiation_success(self, contract: Contract, mechanism: StdAWI) -> None:
        """合同达成时，将其录入 InventoryManager。"""
        assert self.im, "InventoryManager 未初始化"
        # ---- 解析对手 ID ----
        partner = next(pid for pid in contract.partners if pid != self.id)
        is_supply = partner in self.awi.my_suppliers
        im_type = IMContractType.SUPPLY if is_supply else IMContractType.DEMAND
        mat_type = MaterialType.RAW if is_supply else MaterialType.PRODUCT
        new_c = IMContract(
            contract_id=contract.id,
            partner_id=partner,
            type=im_type,
            quantity=contract.agreement["quantity"],
            price=contract.agreement["unit_price"],
            delivery_time=contract.agreement["time"],
            bankruptcy_risk=0.0,
            material_type=mat_type,
        )
        added = self.im.add_transaction(new_c)
        assert added, f"❌ IM.add_transaction 失败! contract={contract.id}"

        # ---- update partner statistics ----
        stats = self.partner_stats.setdefault(
            partner,
            {"avg_price": 0.0, "price_M2": 0.0, "contracts": 0, "success": 0},
        )
        stats["contracts"] += 1
        stats["success"] += 1
        price = contract.agreement["unit_price"]
        n = stats["contracts"]
        delta = price - stats["avg_price"]
        stats["avg_price"] += delta / n
        stats["price_M2"] += delta * (price - stats["avg_price"])
        self._update_acceptance_model(partner, price, True)

        # 更新不足原材料数据
        self.today_insufficient = self.im.get_today_insufficient(self.awi.current_step)
        self.total_insufficient = self.im.get_total_insufficient(self.awi.current_step)

        # 这里是今天达成的，今天交付的协议，主要用于保障紧急需求
        if is_supply:
            if contract.agreement["time"] == self.awi.current_step:
                self.purchase_completed[self.awi.current_step] += contract.agreement["quantity"]
        elif not is_supply:
            if contract.agreement["time"] == self.awi.current_step:
                self.sales_completed[self.awi.current_step] += contract.agreement["quantity"]

        # 日志
        if os.path.exists("env.test"):
            print(f"✅ 合同已加入 IM: {new_c}")

    # ------------------------------------------------------------------
    # 🌟 7. 动态策略调节接口
    # Dynamic strategy adjustment API
    # ------------------------------------------------------------------

    def update_profit_strategy(
        self, *, min_profit_margin: float | None = None, cheap_price_discount: float | None = None
    ) -> None:
        """允许外部模块（RL/脚本）动态调节参数。"""
        if min_profit_margin is not None:
            self.min_profit_margin = min_profit_margin
        if cheap_price_discount is not None:
            self.cheap_price_discount = cheap_price_discount

    # ------------------------------------------------------------------
    # 🌟 8. 预留模型决策钩子（示例）
    # Reserved model decision hook (example)
    # ------------------------------------------------------------------

    def decide_with_model(self, obs: Any) -> Any:  # noqa: ANN401
        """如需集成 RL，可在此实现模型推断并返回动作。"""
        # TODO: 调用 self.model(obs) 等
        return None

# ----------------- (可选) CLI 调试入口 -----------------
# 用于本地 quick‑run，仅在教学 / 测试阶段开启。
if __name__ == "__main__":
    if os.path.exists("env.test"):
        print("模块加载成功，可在竞赛框架中使用 LitaAgentY。")
