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
from litaagent_std.inventory_manager_n import (
    InventoryManager,
    IMContract,
    IMContractType,
    MaterialType,
)

# ------------------ 辅助函数 ------------------

def _split_partners(partners: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """按 50 % / 30 % / 20 % 三段切分伙伴列表。"""
    n = len(partners)
    return (
        partners[: int(n * 0.5)],
        partners[int(n * 0.5): int(n * 0.8)],
        partners[int(n * 0.8):],
    )

def _distribute(q: int, n: int) -> List[int]:
    """随机将 ``q`` 单位分配到 ``n`` 个桶，保证每桶至少 1（若可行）。"""
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

class LitaAgentY(StdSyncAgent):
    """重构后的 LitaAgent N。支持三类采购策略与产能约束销售。"""

    # ------------------------------------------------------------------
    # 🌟 1. 初始化
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
        self.min_profit_margin = min_profit_margin          # 最低利润率⬆
        self.cheap_price_discount = cheap_price_discount    # 超低价折扣阈值⬇

        # —— 运行时变量 ——
        self.im: InventoryManager | None = None             # 库存管理器实例
        self._market_price_avg: float = 0.0                 # 最近报价平均价 (估算市场均价)
        self._recent_material_prices: List[float] = []      # 用滚动窗口估计市场价
        self._recent_product_prices: List[float] = []
        self._avg_window: int = 30                          # 均价窗口大小
        self._ptoday: float = ptoday                        # 当期挑选伙伴比例
        self.model = None                                   # 预留的决策模型
        # 记录每天的采购/销售完成量 {day: quantity}
        self.sales_completed: Dict[int, int] = {}           # 销售完成量

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

        # 初始化当日的完成量记录
        self.sales_completed.setdefault(self.awi.current_step, 0)
        self.purchase_completed.setdefault(self.awi.current_step, 0)

        # 将外生协议写入im
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
            return price > issue.max_value  # 采购价过高
        return price < issue.min_value      # 销售价过低

    def _clamp_price(self, pid: str, price: float) -> float:
        """确保价格在议题允许范围内。"""
        issue = self.get_nmi(pid).issues[UNIT_PRICE]
        return max(issue.min_value, min(issue.max_value, price))

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
        today_inventory_material = min(self.im.get_inventory_summary(self.awi.current_step, MaterialType.RAW)["estimated_available"], self.im.get_max_possible_production(self.awi.current_step))
        return today_inventory_material

    def _get_sales_demand_last_layer(self) -> int:
        # 最后一层的销售需求为0，销售的外生协议由库存管理器管理，并将数据用于计算购买需求
        return 0

    def _get_sales_demand_middle_layer(self) -> int:
        # 这个方法计算的是中间层 * 今天 * 的销售需求
        # 今天的销售需求 = 今天的产能 - 今天的生产计划 + 今天的产品（预期）库存
        # 今天的产品（预期）库存 = 真库存 + 已排产（包括未来） - 已签署的销售合同（这个可以调用im）
        today_inventory_product = self.im.get_inventory_summary(self.awi.current_step, MaterialType.PRODUCT)["estimated_available"]
        return today_inventory_product

    def _get_sales_demand_middle_layer(self, day: int) -> int:
        # 这个方法计算的是中间层 * 在day的 * 销售需求
        # 在day的销售需求 = 到day为止的产能 + 今天的库存 - 到day为止的销售
        future_inventory_product = self.im.get_inventory_summary(day, MaterialType.PRODUCT)["estimated_available"]
        return future_inventory_product

    def _get_supply_demand_middle_last_layer(self) -> tuple[int, int, float]:
        # 这个方法计算的是中间层和最后层 * 今天 * 的购买需求
        # return 紧急需求 计划需求 超额需求(超额需求是计划需求的20%)
        return (self.im.get_today_insufficient(self.awi.current_step),
                self.im.get_total_insufficient(self.awi.current_step),
                self.im.get_total_insufficient(self.awi.current_step) * 0.2)

    def _get_supply_demand_middle_last_layer(self, day: int) -> tuple[int, int, float]:
        # 这个方法计算的是中间层和最后层 * 在day的 * 购买需求
        # 对于最后一层，这方法其实，没什么意义，因为外生协议都是当天的（不过我觉得如果未来外生协议扩展到包括期货的话，这样也能兼容）
        # return 紧急需求 计划需求 超额需求(超额需求是计划需求的20%)
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
        if partners is None:
            partners = self.negotiators.keys()
        partners = list(partners)

        # 初始化：默认所有伙伴分配量 0
        response: Dict[str, int] = {p: 0 for p in partners}

        # 分类伙伴
        suppliers = [p for p in partners if self._is_supplier(p)]
        consumers = [p for p in partners if self._is_consumer(p)]

        # buy_need, sell_need = self._needs_today()
        # 如果是第一层
        if self.awi.is_first_level:
            buy_need = self._get_supply_demand_first_layer()
            sell_need = self._get_sales_demand_first_layer()
        # 如果是最后一层
        elif self.awi.is_last_level:
            buy_need = self._get_supply_demand_middle_last_layer()
            sell_need = self._get_sales_demand_last_layer()
        # 如果在中间
        else:
            buy_need = self._get_supply_demand_middle_last_layer()
            sell_need = self._get_sales_demand_middle_layer()

        # --- 1) 分配采购需求给供应商 ---
        if suppliers and isinstance(buy_need, tuple):
            response.update(self._distribute_to_partners(suppliers, buy_need))

        # --- 2) 分配销售需求给顾客 ---
        if consumers and sell_need > 0:
            # 由于计算需求时已经做过了限制，所以这里不需要再判断了
            response.update(self._distribute_to_partners(consumers, sell_need))

        return response

    def _distribute_to_partners(self, partners: List[str], needs: int) -> Dict[str, int]:
        """核心分配：随机挑选 ``_ptoday`` 比例伙伴分配 ``needs``。"""
        if needs <= 0 or not partners:
            return {p: 0 for p in partners}

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
        responses.update(self._process_supply_offers(supply_offers))
        # -------- 5‑B 顾客报价 --------
        demand_offers = {p: o for p, o in offers.items() if self._is_consumer(p)}
        responses.update(self._process_sales_offers(demand_offers))
        return responses

    # ------------------------------------------------------------------
    # 🌟 5‑1. 供应报价拆分三类
    # ------------------------------------------------------------------

    def _process_supply_offers(self, offers: Dict[str, Outcome]) -> Dict[str, SAOResponse]:
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
        em_res = self._process_emergency_supply_offers(offer_deliver_today)
        res.update(em_res)
        # TODO 如果这样还满足不了今天的紧急需求，就拿一些未来报价来改日期
        # —— 计划性需求 ——
        plan_res = self._process_planned_supply_offers(offer_deliver_later_planned)
        res.update(plan_res)
        # —— 机会性采购 ——
        optional_res = self._process_optional_supply_offers(offer_deliver_optional_demand)
        res.update(optional_res)

        return res

    # ------------------------------------------------------------------
    # 🌟 5‑1‑a 紧急需求处理
    # ------------------------------------------------------------------

    def _process_emergency_supply_offers(self, offers: Dict[str, Outcome]) -> Dict[str, SAOResponse]:
        """处理今日必须到货的原料报价。"""
        res: Dict[str, SAOResponse] = {}
        if not offers or self.today_insufficient <= 0:
            return res
        # 按单价升序排序，优先选便宜的
        ordered = sorted(offers.items(), key=lambda x: x[1][UNIT_PRICE])
        remain_needed = self.today_insufficient
        penalty = self.awi.current_shortfall_penalty
        for pid, offer in ordered:
            qty, price = offer[QUANTITY], offer[UNIT_PRICE]

            # 更新均价窗口
            self._recent_material_prices.append(price)
            if len(self._recent_material_prices) > self._avg_window:
                self._recent_material_prices.pop(0)

            if price > penalty:  # 比罚金贵，先拒绝并压价
                new_price = min(price * 0.9, penalty)  # 小幅压价（10%）
                counter = (qty, offer[TIME], new_price)
                res[pid] = SAOResponse(ResponseType.REJECT_OFFER, counter)
                continue
            accept_qty = min(qty, remain_needed)
            accept_offer = (accept_qty, offer[TIME], price)
            res[pid] = SAOResponse(ResponseType.ACCEPT_OFFER, accept_offer)
            remain_needed -= accept_qty
            # 若还有余量未用，可压价重新还价
            if qty > accept_qty and remain_needed <= 0:
                counter_qty = qty - accept_qty
                counter_price = min(price * 0.9, penalty)
                counter_offer = (counter_qty, offer[TIME], counter_price)
                res[pid] = SAOResponse(ResponseType.REJECT_OFFER, counter_offer)
            if remain_needed <= 0:
                break
        return res

    # ------------------------------------------------------------------
    # 🌟 5‑1‑b 计划性需求处理
    # ------------------------------------------------------------------

    def _process_planned_supply_offers(self, offers: Dict[str, Outcome]) -> Dict[str, SAOResponse]:
        """为未来生产需求采购原料：保证利润并智能调整采购量。"""
        res: Dict[str, SAOResponse] = {}
        for pid, offer in offers.items():
            qty, t, price = offer[QUANTITY], offer[TIME], offer[UNIT_PRICE]

            # 更新均价窗口
            self._recent_material_prices.append(price)
            if len(self._recent_material_prices) > self._avg_window:
                self._recent_material_prices.pop(0)

            # 1. 估算该交货日的产品预计售价（简化：用市场均价占位）
            est_sell_price = self._market_price_avg if self._market_price_avg > 0 else price * 2

            # 2. 获取产品单位成本（从IM获取预计产品平均成本）
            avg_product_cost = self.im.get_inventory_summary(t, MaterialType.PRODUCT)["estimated_average_cost"]
            # 如果没有产品成本记录，则使用当前报价 + 加工费估算
            unit_cost = avg_product_cost if avg_product_cost > 0 else price + self.im.processing_cost

            # 3. 计算最低可接受售价（满足利润率要求）
            max_price_allowed = est_sell_price / (1 + self.min_profit_margin)

            # 4. 检查需求量
            request_qty = self.im.get_total_insufficient(t)

            # 5. 决策逻辑
            if price <= max_price_allowed and qty <= request_qty:
                # 价格满足利润要求且数量不超出需求 - 直接接受
                res[pid] = SAOResponse(ResponseType.ACCEPT_OFFER, offer)

            elif price <= max_price_allowed and qty > request_qty:
                # 价格满足利润要求但数量超出需求 - 部分接受（简化：拒绝并减量，或者提早交付）
                if price < max_price_allowed * 0.9:
                    # 实在是太便宜了，推迟交货，但是买(因为越靠前需求量越大，因此可以签署最早到今天的协议，早到哪天根据价格和均价决定)
                    n_days_earlier = (self._market_material_price_avg - price) / self.im.raw_storage_cost
                    if n_days_earlier > 0:
                        # 如果有提前买多一点的必要，那就提前买多一点吧
                        offer_qty = self.im.get_total_insufficient(t - n_days_earlier)
                        offer_day = t - n_days_earlier
                        offer_price = price
                        res[pid] = SAOResponse(ResponseType.REJECT_OFFER, (offer_qty, offer_day, offer_price))
                    else:
                        # 如果没有提前买的必要,那就减量吧
                        offer_qty = request_qty
                        offer_day = t
                        offer_price = price
                        res[pid] = SAOResponse(ResponseType.REJECT_OFFER, (offer_qty, offer_day, offer_price))
                else:
                    # 如果也不是那么便宜，那就减量吧
                    offer_qty = request_qty
                    offer_day = t
                    offer_price = price
                    res[pid] = SAOResponse(ResponseType.REJECT_OFFER, (offer_qty, offer_day, offer_price))
            elif price >= max_price_allowed and qty <= request_qty:
                # 如果太贵了，但是数量还可以的话，那就降价
                offer_qty = qty
                offer_day = t
                offer_price = max_price_allowed
                res[pid] = SAOResponse(ResponseType.REJECT_OFFER, (offer_qty, offer_day, offer_price))
            else:
                # 如果又贵又超出需求量，就要求降价
                offer_qty = request_qty
                offer_day = t
                offer_price = max_price_allowed
                res[pid] = SAOResponse(ResponseType.REJECT_OFFER, (offer_qty, offer_day, offer_price))
        return res

    # ------------------------------------------------------------------
    # 🌟 5‑1‑c 机会性采购处理
    # ------------------------------------------------------------------

    def _process_optional_supply_offers(self, offers: Dict[str, Outcome]) -> Dict[str, SAOResponse]:
        """仅在超低价时囤货。"""
        res: Dict[str, SAOResponse] = {}
        if not offers:
            return res
        # 当前市场平均价（若为0则先记录报价再处理）
        for pid, offer in offers.items():
            qty, price = offer[QUANTITY], offer[UNIT_PRICE]
            # 更新均价窗口
            self._recent_material_prices.append(price)
            if len(self._recent_material_prices) > self._avg_window:
                self._recent_material_prices.pop(0)
            threshold = self._market_price_avg * self.cheap_price_discount if self._market_price_avg else price * 2
            if price <= threshold:
                # TODO 这个地方的实现还是有一些混乱，设想是以往签署的可选需求之和不超过对应日的计划外需求的20%， 但是现在好像只是计算这一单不超过20%。我怀疑会买很多很多
                # TODO 姑且先做成当日总预期库存不能超过计划需求的120%的形式吧
                estimated_material_inventory= self.im.get_inventory_summary(offer[TIME], MaterialType.MATERIAL)["estimated_available"]
                inventory_limit = self.im.get_total_insufficient(offer[TIME]) * 1.2
                accept_qty = inventory_limit - estimated_material_inventory if inventory_limit > 0 else 0
                if accept_qty > 0:
                    # 如果还满足需求条件，并且价格也够低 - 接受offer
                    res[pid] = SAOResponse(ResponseType.ACCEPT_OFFER, (qty, offer[TIME], price))
                else:
                    # 如果价格够低，但是数量太大 - 减少数量
                    counter = (accept_qty, offer[TIME], price)
                    res[pid] = SAOResponse(ResponseType.REJECT_OFFER, counter)
            else:
                # 如果太贵了 - 要求降价
                counter_price = threshold
                res[pid] = SAOResponse(ResponseType.REJECT_OFFER, (qty, offer[TIME], counter_price))
        return res

    # ------------------------------------------------------------------
    # 🌟 5‑2. 销售报价处理
    # ------------------------------------------------------------------

    def _process_sales_offers(self, offers: Dict[str, Outcome]) -> Dict[str, SAOResponse]:
        """确保不超产能且满足利润率。"""
        # TODO: 有一个问题：如果达成了一笔当天的协议。那么这笔协议的不足量就会瞬间变成当天必须实现的采购量，需要根据这个重新规划生产、重新计算所需库存量来提出购买要求
        res: Dict[str, SAOResponse] = {}
        if not offers:
            return res
        assert self.im, "InventoryManager 未初始化"
        for pid, offer in offers.items():
            qty, t, price = offer[QUANTITY], offer[TIME], offer[UNIT_PRICE]

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
                    counter_offer = (accept_qty, t, price)
                    res[pid] = SAOResponse(ResponseType.REJECT_OFFER, counter_offer)
                else:
                    res[pid] = SAOResponse(ResponseType.REJECT_OFFER, None)
                continue
            # 2) 利润检查
            # 估算单位成本：用最近平均原料价 + 加工
            avg_raw_cost = self._market_price_avg or price * 0.5
            unit_cost = avg_raw_cost + self.im.processing_cost
            min_sell_price = unit_cost * (1 + self.min_profit_margin)
            if price >= min_sell_price:
                res[pid] = SAOResponse(ResponseType.ACCEPT_OFFER, offer)
            else:
                counter_price = max(min_sell_price, price * 1.05)  # 小幅抬价
                counter = (qty, t, counter_price)
                res[pid] = SAOResponse(ResponseType.REJECT_OFFER, counter)
        return res

    # ------------------------------------------------------------------
    # 🌟 6. 合同成功回调
    # ------------------------------------------------------------------

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
            quantity=contract.issues[QUANTITY],
            price=contract.issues[UNIT_PRICE],
            delivery_time=contract.issues[TIME],
            bankruptcy_risk=0.0,
            material_type=mat_type,
        )
        added = self.im.add_transaction(new_c)
        assert added, f"❌ IM.add_transaction 失败! contract={contract.id}"

        # 更新不足原材料数据
        self.today_insufficient = self.im.get_today_insufficient(self.awi.current_step)
        self.total_insufficient = self.im.get_total_insufficient(self.awi.current_step)

        # 日志
        print(f"✅ 合同已加入 IM: {new_c}")

    # ------------------------------------------------------------------
    # 🌟 7. 动态策略调节接口
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
    # ------------------------------------------------------------------

    def decide_with_model(self, obs: Any) -> Any:  # noqa: ANN401
        """如需集成 RL，可在此实现模型推断并返回动作。"""
        # TODO: 调用 self.model(obs) 等
        return None

# ----------------- (可选) CLI 调试入口 -----------------
# 用于本地 quick‑run，仅在教学 / 测试阶段开启。
if __name__ == "__main__":
    print("模块加载成功，可在竞赛框架中使用 LitaAgentY。")
