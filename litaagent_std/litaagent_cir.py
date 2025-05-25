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
from typing import Dict, List, Sequence, Tuple
import numpy as np

from inventory_manager_n import InventoryManager, MaterialType  # type: ignore

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

    def score_batch(self, info: Sequence[Tuple[str, Offer, bool]], offers: Sequence[Offer]) -> np.ndarray:
        horizon = range(self.today, self.today + self.window)
        short_before = np.array([self.im.get_total_insufficient(d) for d in horizon])
        over_before = np.array([self._calc_raw_over(self.im, d) for d in horizon])
        w_short, w_over = self.bayes.current_weights()
        urgent = self.im.get_today_insufficient(self.today) > 0
        scores = np.zeros(len(offers))
        for i, off in enumerate(offers):
            pid = info[i][0]
            is_seller = info[1][2]
            clone = deepcopy(self.im)
            contract = IMContract(
                contract_id=str(uuid4()),
                partner_id=pid,
                type=IMContractType.SUPPLY if is_seller else IMContractType.DEMAND,
                quantity=off.quantity,
                price=off.price,
                delivery_time=off.time,
                bankruptcy_risk=0,
                material_type=MaterialType.RAW if is_seller else MaterialType.PRODUCT,
            )
            clone.add_transaction(contract)
            short_after = np.array([clone.get_total_insufficient(d) for d in horizon])
            over_after = np.array([max(0, self._calc_raw_over(self.im, d)) for d in horizon])
            base = w_short * (short_before - short_after).sum() - w_over * (over_after - over_before).sum()
            if urgent and (short_before[0] - short_after[0]) > 0:
                base *= 1.5
            scores[i] = base
        return scores


# ----------------------------------------------------
# 0‑1 背包动态规划选择器
# ----------------------------------------------------
@dataclass()
class dfsItem:
    pid: str
    score: float
    cost:float
    offer: Dict[str, Outcome]

def select_offers_dp(infos, scores: np.ndarray, budget, im: InventoryManager):
    """返回需要 Accept 的报价索引（二维背包按预算和库存限制最大化得分）。"""
    dfsitems = []
    # infos = [(nid, off, self._is_seller(off)) ] # off = (qty, time, price)
    for i, info in enumerate(infos):
        dfsitem = dfsItem(
            pid=info[0],
            score=scores[i],
            cost=info[1][0] * info[1][2] if info[2] else 0,
            offer=info, # (nid, off, self._is_seller(off))
        )

    n = len(dfsitems)
    prefix_best = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        prefix_best[i] = prefix_best[i + 1] + dfsitems[i].score
    best_score = -1
    best_pick = ()

    def feasible(chosen: List, im: InventoryManager) -> bool:
        """判断当前选中的物品是否满足库存要求。"""
        im_clone = deepcopy(im)
        # chosen 是 dfsitem的tuple
        chosen_clone = deepcopy(chosen)
        chosen_clone.sort(key=lambda x: x["offer"][1][1], reverse=True)
        for item in chosen_clone:
            pid = item["pid"]
            offer = item["offer"][1]
            qty = offer[0]
            time = offer[1]
            price = offer[2]
            is_seller = item["offer"][2]
            if im_clone.get_max_possible_production(time) < qty:
                return False
            contract = IMContract(
                contract_id=str(uuid4()),
                partner_id=pid,
                type=IMContractType.SUPPLY if is_seller else IMContractType.DEMAND,
                quantity=qty,
                price=price,
                delivery_time=time,
                bankruptcy_risk=0,
                material_type=MaterialType.RAW if is_seller else MaterialType.PRODUCT,
            )
            im_clone.add_transaction(contract)
        return True

    def dfs(idx: int, cost: int, score: int, chosen: list, im: InventoryManager):
        nonlocal best_score, best_pick

        # 剪枝 1：花超预算
        if cost > budget:
            return

        # 剪枝 2：即便全拿剩余物品也不可能超过当前最佳
        if score + prefix_best[idx] <= best_score:
            return

        # 如果走到叶节点（或提前验证可行的条件）
        if idx == n:
            if score > best_score and feasible(chosen, im):
                best_score = score
                best_pick = chosen
            return

        item = dfsitems[idx]

        # 分支 A：选择当前物品
        chosen.append(item)
        # 可行性单调 => 若当前子集不可行，则包含它的所有超集都不可行，可立即退回
        if feasible(chosen, im):
            dfs(idx + 1, cost + item.cost, score + item.score, chosen, im)
        chosen.pop()

        # 分支 B：不选择当前物品
        dfs(idx + 1, cost, score, chosen)

    dfs(0, 0, 0, [], im)
    return best_pick


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
        
        if os.path.exists("env.test"): # Added from Step 11
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
        
        # Counters for dynamic profit margin adjustment (Added from Step 7)
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
        self.bayes = BayesPenaltyModel()
        self.λ0 = 0.1
        if os.path.exists("env.test"): # Added from Step 11
            print(f"🤖 LitaAgentY {self.id} IM initialized. Daily Capacity: {self.im.daily_production_capacity}")


    def before_step(self) -> None:
        """每天开始前，同步日内关键需求信息。"""
        assert self.im, "InventoryManager 尚未初始化!"
        current_day = self.awi.current_step # Use local var for f-string clarity
        self.today_insufficient = self.im.get_today_insufficient(current_day)
        self.total_insufficient = self.im.get_total_insufficient(current_day)
        if os.path.exists("env.test"): # Added from Step 11
            print(f"🌞 Day {current_day} ({self.id}) starting. Today Insufficient Raw: {self.today_insufficient}, Total Insufficient Raw (horizon): {self.total_insufficient}")


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
        assert self.im, "InventoryManager 尚未初始化!"
        # 让 IM 完成收货 / 生产 / 交付 / 规划
        result = self.im.process_day_operations()
        self.im.update_day() # This increments self.im.current_day
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

    def _get_sales_demand_first_layer(self) -> int:
        if not self.im: return 0
        today_inventory_material = int(min(self.im.get_inventory_summary(self.awi.current_step, MaterialType.RAW)["estimated_available"], self.im.get_max_possible_production(self.awi.current_step)))
        return today_inventory_material

    def _get_sales_demand_last_layer(self) -> int:
        return 0

    def _get_sales_demand_middle_layer_today(self) -> int:
        if not self.im: return 0
        today_inventory_product = int(self.im.get_inventory_summary(self.awi.current_step, MaterialType.PRODUCT)["estimated_available"])
        return today_inventory_product

    def _get_sales_demand_middle_layer(self, day: int) -> int:
        if not self.im: return 0
        future_inventory_product = int(self.im.get_inventory_summary(day, MaterialType.PRODUCT)["estimated_available"])
        return future_inventory_product

    def _get_supply_demand_middle_last_layer_today(self) -> tuple[int, int, float]:
        if not self.im: return 0,0,0.0
        return (self.im.get_today_insufficient(self.awi.current_step),
                self.im.get_total_insufficient(self.awi.current_step),
                self.im.get_total_insufficient(self.awi.current_step) * 0.2)

    def _get_supply_demand_middle_last_layer(self, day: int) -> tuple[int, int, float]:
        if not self.im: return 0,0,0.0
        return (
            self.im.get_today_insufficient(day),
            self.im.get_total_insufficient(day), 
            self.im.get_total_insufficient(day) * 0.2, 
        )

    def _get_supply_demand_first_layer(self) -> Tuple[int, int, int]:
        return 0,0,0

    def _distribute_todays_needs(self, partners: Iterable[str] | None = None) -> Dict[str, int]:
        if partners is None:
            partners = self.negotiators.keys()
        partners = list(partners)
        response: Dict[str, int] = {p: 0 for p in partners}
        if not self.im : return response

        suppliers = [p for p in partners if self._is_supplier(p)]
        consumers = [p for p in partners if self._is_consumer(p)]
        sell_need = 0

        if self.awi.is_first_level:
            _ , _ , buy_need_optional_float = self._get_supply_demand_first_layer()
            sell_need = self._get_sales_demand_first_layer()
        elif self.awi.is_last_level:
            sell_need = self._get_sales_demand_last_layer() 
        else:
            sell_need = self._get_sales_demand_middle_layer_today()
        
        total_buy_need = self.im.get_total_insufficient(self.awi.current_step)
        if suppliers and total_buy_need > 0:
            response.update(self._distribute_to_partners(suppliers, total_buy_need))
        if consumers and sell_need > 0:
            response.update(self._distribute_to_partners(consumers, sell_need))
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
        if not partners: return {}
        filtered: List[str] = []
        for pid in partners:
            if self._is_supplier(pid) and self.awi.is_first_level: continue
            if self._is_consumer(pid) and self.awi.is_last_level: continue
            filtered.append(pid)
        if not filtered: return {}

        distribution = self._distribute_todays_needs(filtered)
        today = self.awi.current_step
        proposals: Dict[str, Outcome] = {}

        for pid, qty in distribution.items():
            if qty <= 0: continue
            nmi = self.get_nmi(pid)
            proposals[pid] = (qty, today, nmi.issues[UNIT_PRICE].min_value if self._is_supplier(pid) else nmi.issues[UNIT_PRICE].max_value)

        return proposals

    # ------------------------------------------------------------------
    # 🌟 5. counter_all — 谈判核心（分派到子模块）
    # ------------------------------------------------------------------

    def counter_all(
        self,
        offers: Dict[str, Outcome],
        states: Dict[str, SAOState],
    ) -> Dict[str, SAOResponse]:
        responses: Dict[str, SAOResponse] = {}

        if not offers:
            return {}
        ids, off_list = list(offers.keys()), list(offers.values())
        infos = [(nid, off, self._is_seller(nid)) for nid, off in offers.items()]
        scorer = StockScorer(self.im, self.bayes, today=self.awi.current_step)
        scores = scorer.score_batch(infos, off_list)

        budget = self.awi.current_balance
        headroom = lambda o: pass # 这个函数用来判断装入背包的offer组合是否满足库存和产能限制
        chosen_idx = select_offers_dp(infos, scores, budget, self.im)

        rel_t = self.awi.current_step / max(1, self.awi.n_steps)
        deal_rate = len(chosen_idx) / len(offers)
        λ_dyn = lambda_price(rel_t, deal_rate, self.λ0)

        responses = {}
        for i, nid in enumerate(ids):
            off = off_list[i] # (qty, time, price)
            if i in chosen_idx:
                responses[nid] = SAOResponse(ResponseType.ACCEPT_OFFER, off)
                continue
            if scores[i] <= 0:
                responses[nid] = SAOResponse(ResponseType.REJECT_OFFER, )
            else:
                # TODO：修复这个问题
                new_p = self._concede_price(off, λ_dyn)
                responses[nid] = SAOResponse(ResponseType.REJECT_OFFER, (off[0], off[1], new_p))

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
        assert self.im, "InventoryManager 尚未初始化"
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
        assert added, f"❌ ({self.id}) IM.add_transaction 失败! contract={contract.id}"

        self.today_insufficient = self.im.get_today_insufficient(self.awi.current_step)
        self.total_insufficient = self.im.get_total_insufficient(self.awi.current_step)

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
            
            raw_current_stock = int(raw_summary['current_stock'])
            raw_estimated = int(raw_summary['estimated_available'])
            
            product_current_stock = int(product_summary['current_stock'])
            product_estimated = int(product_summary['estimated_available'])
            
            # 计划生产量
            planned_production = int(self.im.get_production_plan(forecast_day))
            
            # 剩余产能
            remaining_capacity = int(self.im.get_available_production_capacity(forecast_day))
            
            # 已签署的销售合同数量
            signed_sales = 0
            for contract in self.im.get_pending_contracts(is_supply=False, day=forecast_day):
                if contract.material_type == MaterialType.PRODUCT:
                    signed_sales += contract.quantity
            
            # 格式化并输出
            day_str = f"{forecast_day} (T+{day_offset})" if day_offset == 0 else f"{forecast_day} (T+{day_offset})"
            print(f"| {day_str:^6} | {raw_current_stock:^10} | {raw_estimated:^12} | {planned_production:^8} | {remaining_capacity:^8} | {product_current_stock:^10} | {product_estimated:^12} | {signed_sales:^12} | {result["delivered_products"] if day_offset == 0 else 0:^12} |")
        
        print(separator)
        print()

    # ---------------- capacity / budget ----------------
    def _has_capacity(self, off: Offer) -> bool:
        if self._is_seller(off):
            inv = self.im.get_inventory_summary(off.time, MaterialType.PRODUCT)["estimated_available"]
            cap = self.im.get_available_production_capacity(off.time)
            return off.quantity <= inv + cap
        return True
    def _has_budget(self, off: Offer, budget: float) -> bool:
        return True if self._is_seller(off) else off.price*off.quantity <= budget

    # ---------------- price concede ----------------
    def _concede_price(self, off: Offer, λ: float):
        tgt = self._target_price(off)
        delta = abs(off.price - tgt)
        return max(tgt, off.price-λ*delta) if self._is_seller(off) else min(tgt, off.price+λ*delta)

    def _target_price(self, off: Offer):
        if self._is_seller(off):
            raw_avg = self.im.get_inventory_summary(off.time, MaterialType.RAW)["estimated_average_cost"]
            return (raw_avg + self.im.processing_cost)*(1+self.min_profit_margin)
        sell_avg = self.im.get_inventory_summary(off.time, MaterialType.PRODUCT)["estimated_average_cost"]
        return max(0.01, sell_avg - self.im.processing_cost - self.min_profit_margin)

    # ---------------- util ----------------
    def _is_seller(self, off: Offer):
        try:
            return self.awi.is_seller(self.id, off)
        except AttributeError:
            return True

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
