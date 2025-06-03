# inventory_manager_ns.py
# 这是给Agent使用的库存管理器，用来管理代理的协议、生产、和库存。
# This is the Inventory Manager for Agents to manage the agent's protocols, production, and inventory.

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import os


class IMContractType(Enum):
    SUPPLY = auto()  # 采购／上游供给 Upstream Supply
    DEMAND = auto()  # 销售／下游需求 Downstream Demand


class MaterialType(Enum):
    RAW = auto()  # 原材料 Raw material
    PRODUCT = auto()  # 产品 product


@dataclass
class IMContract:
    contract_id: str
    partner_id: str
    type: IMContractType
    quantity: float
    price: float
    delivery_time: int  # 交割日（天数）Day for delivery
    bankruptcy_risk: float
    material_type: MaterialType


@dataclass
class Batch:
    batch_id: str
    remaining: float  # 当前剩余量（FIFO 扣减时会减少） Current remaining quantity
    unit_cost: float  # 单位成本（入库价，不含存储） Unit cost (storage not included)
    production_time: int  # 入库／生产日期 Production date for products, delivery date for raw materials


class InventoryManagerNS:
    def __init__(
            self,
            raw_storage_cost: float,
            product_storage_cost: float,
            processing_cost: float,
            # 可选：日生产能力，若不指定则视为无限 Unlimited if not specified
            daily_production_capacity: Optional[float] = None,
            # 可选：最大仿真天数，默认100天 Should be initialized while the instance is created
            max_day: int = 100,
    ):
        # 当前仿真天
        # Current simulation day
        self.max_day = max_day  # 默认100天，可以在初始化时传入 Default is 100 days and can be provided at init
        self.current_day: int = 0

        # 成本参数 Cose parameters
        self.raw_storage_cost = raw_storage_cost
        self.product_storage_cost = product_storage_cost
        self.processing_cost = processing_cost
        self.daily_production_capacity = (
            daily_production_capacity if daily_production_capacity is not None else float("inf")
        )

        # 库存批次 Inventory batches
        self.raw_batches: List[Batch] = []
        self.product_batches: List[Batch] = []

        # 未交割的合同 Pending contracts
        self._pending_supply: List[IMContract] = []
        self._pending_demand: List[IMContract] = []

        # 生产计划  Production plan
        self.production_plan: Dict[int, float] = {}

        # 不足原料记录，结构为 {day: {"daily": 当日需要的原材料, "total": 总共需要的原材料}}
        # daily 表示必须在当天获取的原材料量，否则会违约
        # total 表示从当前到该日期总共还需要的原材料量
        # Insufficient raw material record, structure is {day: {"daily": raw material needed on that day, "total": total necessary raw material}}
        # 'daily' indicates the amount of raw material that must be obtained on that day, otherwise deliver will be fail
        # 'total' indicates the total amount of raw material needed from now to that date
        self.insufficient_raw: Dict[int, Dict[str, float]] = {}

        # 可扩展：最大可能生产
        # Maximum possible production, this is not related to the inventory
        self._max_possible_prod_cache: Dict[int, float] = {}

    def add_transaction(self, contract: IMContract) -> bool:
        """签订一份采购（上游）或销售（下游）合同。
        Sign a supply (upstream) or demand (downstream) contract."""
        # Sign a supply (upstream) or demand (downstream) contract.
        if contract.delivery_time < self.current_day:
            return False
        if contract.type == IMContractType.SUPPLY:
            self._pending_supply.append(contract)
        else:
            self._pending_demand.append(contract)

        # 每次添加新合同后，重新规划生产计划
        # Re-plan the production schedule whenever a new contract is added
        # 并计算不足库存量，确保代理可以及时应对变化
        # Calculate inventory shortfalls to ensure the agent can react in time
        self.plan_production(self.max_day)  # Use self.max_day for full horizon planning

        return True

    def void_negotiated_contract(self, contract_id: str) -> bool:
        """
        Removes a contract from pending lists if it was cancelled at signing.
        This is called when a contract previously added via on_negotiation_success
        is ultimately not signed (e.g., due to sign_all_contracts logic or opponent cancellation).
        """
        found_and_removed = False
        # Search in pending supply contracts
        for contract in self._pending_supply:
            if contract.contract_id == contract_id:
                self._pending_supply.remove(contract)
                found_and_removed = True
                break

        if not found_and_removed:
            # Search in pending demand contracts
            for contract in self._pending_demand:
                if contract.contract_id == contract_id:
                    self._pending_demand.remove(contract)
                    found_and_removed = True
                    break

        if found_and_removed:
            # Re-plan production as voiding a contract can change needs
            self.plan_production(self.max_day)  # Use self.max_day for full horizon planning

        return found_and_removed

    def receive_materials(self) -> List[str]:
        """
        将当天到货的上游合同转成原材料入库批次，FIFO 管理；
        返回本日完成交割的合同 ID 列表。
        Convert upstream contracts that arrive on the same day into raw material inbound batches, FIFO management;
        Returns a list of contract IDs that have completed delivery on the current day.
        """
        arrived = [c for c in self._pending_supply if c.delivery_time == self.current_day]
        for c in arrived:
            batch = Batch(
                batch_id=c.contract_id,
                remaining=c.quantity,
                unit_cost=c.price,
                production_time=self.current_day
            )
            self.raw_batches.append(batch)
            self._pending_supply.remove(c)
        return [c.contract_id for c in arrived]

    def _compute_summary(
            self,
            day: int,
            batches: List[Batch],
            storage_cost_per_unit: float,
            pending: List[IMContract],
            mtype: MaterialType
    ):
        """通用：计算指定 day 的真实库存 & 预期库存摘要。
        General utility to compute the real and expected inventory summary for the given day."""
        # 真实库存 Real inventory
        total_qty = 0.0
        total_cost = 0.0  # 含存储累积 Accumulated including storage costs
        total_store_cost = 0.0
        for b in batches:
            if b.remaining <= 0:
                continue
            days_stored = max(0, day - b.production_time)
            c0 = b.unit_cost * b.remaining
            cs = storage_cost_per_unit * days_stored * b.remaining
            total_qty += b.remaining
            total_cost += c0 + cs
            total_store_cost += cs

        avg_cost = total_cost / total_qty if total_qty > 0 else 0.0
        total_est = 0.0
        # 预期可用：指的是从今天开始，按照计划生产和交付的情况下，到指定日期的可用数量（指定日期之后的不计算）
        # [可用于生产的原材料]当日真实库存 + 当日至指定日期到货 - 当日至指定日期前一天的生产计划Estimated available inventory, = real + future(contracted)until day
        if mtype == MaterialType.RAW:
            future_in_inv = sum(c.quantity for c in pending if self.current_day <= c.delivery_time <= day)
            future_prod_plan = sum(self.get_production_plan(prod_day) for prod_day in range(self.current_day, day))

            total_est = total_qty + future_in_inv - future_prod_plan
        # [可用于交割的产品]当日真实库存 - 当日至指定日期前一天的交割 + 当日至指定日期的生产计划 Estimated available inventory, = real - future(contracted) until day + production_plan until day
        elif mtype == MaterialType.PRODUCT:
            future_out_inv = sum(c.quantity for c in pending if self.current_day <= c.delivery_time < day)
            future_prod_plan = sum(self.get_production_plan(prod_day) for prod_day in range(self.current_day, day+1))
            total_est = total_qty - future_out_inv + future_prod_plan
        # 预期成本：真实（含存储）+ 未来（仅入库价，不含后续存储） Estimated cost = real (including storage) + future (only inbound price, excluding subsequent storage)
        future_cost = sum(c.price * c.quantity for c in pending if c.delivery_time >= day)
        est_avg_cost = (total_cost + future_cost) / total_est if total_est > 0 else 0.0

        return {
            "current_stock": total_qty,
            "current_cost": total_cost,
            "current_storage_cost": total_store_cost,
            "current_avg_cost": avg_cost,
            "estimated_stock": total_est,
            "estimated_cost": total_cost + future_cost,
            "estimated_avg_cost": est_avg_cost
        }

    def get_inventory_summary(self, day: int, mtype: MaterialType):
        """获取指定 day 的库存摘要。
        Get inventory summary for the specified day."""
        if mtype == MaterialType.RAW:
            return self._compute_summary(
                day, self.raw_batches, self.raw_storage_cost, self._pending_supply, mtype
            )
        elif mtype == MaterialType.PRODUCT:
            return self._compute_summary(
                day, self.product_batches, self.product_storage_cost, self._pending_demand, mtype
            )
        else:
            raise ValueError(f"Unknown material type: {mtype}")

    def get_today_insufficient(self, day: int) -> float:
        """获取当日不足的原材料量。
        Get the amount of insufficient raw materials for the current day."""
        if day in self.insufficient_raw:
            return self.insufficient_raw[day]["daily"]
        return 0.0

    def get_total_insufficient(self, day: int) -> float:
        """获取从当前到指定日期总共还需要的原材料量。
        Get the total amount of insufficient raw materials from now to the specified date."""
        if day in self.insufficient_raw:
            return self.insufficient_raw[day]["total"]
        return 0.0

    def update_day(self):
        """更新当前天数，并执行当日的生产和交割。
        Update the current day and execute production and delivery for the current day."""
        self.current_day += 1
        self.process_day_operations()

    # 生产计划 Production planning
    def jit_production_plan_abs(
            self,
            start_day: int,
            horizon: int,
            capacity: int,
            inv_raw: int,
            inv_prod: int,
            future_raw_deliver: Dict[int, int],
            future_prod_deliver: Dict[int, int]
    ) -> Dict[int, int]:
        """
        JIT 生产计划算法（绝对时间版）
        JIT production planning algorithm (absolute time version)

        Args:
            start_day: 开始日期 Start day
            horizon: 规划天数 Planning horizon
            capacity: 日产能 Daily production capacity
            inv_raw: 当前原材料库存 Current raw material inventory
            inv_prod: 当前产品库存 Current product inventory
            future_raw_deliver: 未来原材料到货 {day: quantity} Future raw material delivery
            future_prod_deliver: 未来产品交付 {day: quantity} Future product delivery

        Returns:
            Dict[int, int]: 生产计划 {day: quantity} Production plan
        """
        # 初始化生产计划 Initialize production plan
        plan = {}
        # 初始化库存 Initialize inventory
        raw = inv_raw
        prod = inv_prod

        # 按照交付日期倒序排列 Sort by delivery date in descending order
        deliver_days = sorted(future_prod_deliver.keys(), reverse=True)

        # 对于每个交付日 For each delivery day
        for d_day in deliver_days:
            # 如果交付日在规划范围外，跳过 Skip if delivery day is outside planning horizon
            if d_day < start_day or d_day >= start_day + horizon:
                continue

            # 当前交付日需求量 Current delivery day demand
            demand = future_prod_deliver[d_day]

            # 如果库存足够，直接从库存扣除 If inventory is sufficient, deduct from inventory
            if prod >= demand:
                prod -= demand
                continue

            # 否则，需要生产 Otherwise, need to produce
            to_produce = demand - prod
            prod = 0  # 库存已用完 Inventory is depleted

            # 从交付日前一天开始，尽可能安排生产 Start from the day before delivery, schedule production as much as possible
            for p_day in range(d_day - 1, start_day - 1, -1):
                # 当天可用产能 Available capacity for the day
                avail_cap = capacity - plan.get(p_day, 0)
                if avail_cap <= 0:
                    continue

                # 当天可用原材料 Available raw materials for the day
                avail_raw = raw
                for r_day in range(start_day, p_day + 1):
                    avail_raw += future_raw_deliver.get(r_day, 0)

                # 当天已安排生产 Production already scheduled for the day
                for pp_day in range(start_day, p_day):
                    avail_raw -= plan.get(pp_day, 0)

                # 当天最多可生产 Maximum production for the day
                max_prod = min(avail_cap, avail_raw, to_produce)
                if max_prod <= 0:
                    continue

                # 安排生产 Schedule production
                plan[p_day] = plan.get(p_day, 0) + max_prod
                to_produce -= max_prod

                # 如果已经安排完，跳出 If all production is scheduled, break
                if to_produce <= 0:
                    break

        return plan

    def plan_production(self, up_to_day: int):
        """
        规划生产计划，并计算不足库存量。
        Plan production and calculate insufficient inventory.

        Args:
            up_to_day: 规划截止日期 Planning end date
        """
        # 清空生产计划 Clear production plan
        self.production_plan = {}
        # 清空不足原料记录 Clear insufficient raw material record
        self.insufficient_raw = {}

        # 获取当前库存 Get current inventory
        raw_summary = self.get_inventory_summary(self.current_day, MaterialType.RAW)
        prod_summary = self.get_inventory_summary(self.current_day, MaterialType.PRODUCT)
        inv_raw = raw_summary["current_stock"]
        inv_prod = prod_summary["current_stock"]

        # 获取未来原材料到货 Get future raw material delivery
        future_raw_deliver = {}
        for c in self._pending_supply:
            if c.delivery_time < self.current_day:
                continue
            future_raw_deliver[c.delivery_time] = future_raw_deliver.get(c.delivery_time, 0) + c.quantity

        # 获取未来产品交付 Get future product delivery
        future_prod_deliver = {}
        for c in self._pending_demand:
            if c.delivery_time < self.current_day:
                continue
            future_prod_deliver[c.delivery_time] = future_prod_deliver.get(c.delivery_time, 0) + c.quantity

        # 规划生产 Plan production
        plan = self.jit_production_plan_abs(
            self.current_day,
            up_to_day - self.current_day + 1,
            self.daily_production_capacity,
            inv_raw,
            inv_prod,
            future_raw_deliver,
            future_prod_deliver
        )

        # 更新生产计划 Update production plan
        self.production_plan = plan

        # 计算不足原料 Calculate insufficient raw material
        raw_used = 0
        for day in range(self.current_day, up_to_day + 1):
            # 当天生产计划 Production plan for the day
            prod_plan = plan.get(day, 0)
            # 当天原材料到货 Raw material delivery for the day
            raw_in = future_raw_deliver.get(day, 0)
            # 当天原材料使用 Raw material usage for the day
            raw_used += prod_plan
            # 当天原材料库存 Raw material inventory for the day
            raw_stock = inv_raw + sum(future_raw_deliver.get(d, 0) for d in range(self.current_day, day + 1)) - raw_used

            # 如果库存不足，记录不足量 If inventory is insufficient, record the insufficient amount
            if raw_stock < 0:
                self.insufficient_raw[day] = {
                    "daily": -raw_stock if prod_plan > 0 else 0,
                    "total": -raw_stock
                }

    def deliver_products(self) -> List[str]:
        """
        交付当天的产品，并返回成功交付的合同 ID 列表。
        Deliver products for the current day and return a list of successfully delivered contract IDs.
        """
        # 获取当天需要交付的合同 Get contracts that need to be delivered today
        to_deliver = [c for c in self._pending_demand if c.delivery_time == self.current_day]
        delivered = []

        # 按照价格从高到低排序，优先交付高价合同 Sort by price from high to low, prioritize high-price contracts
        to_deliver.sort(key=lambda c: c.price, reverse=True)

        # 交付产品 Deliver products
        for c in to_deliver:
            # 如果成功交付，添加到已交付列表 If successfully delivered, add to delivered list
            if self._reduce_product_inventory(c.quantity):
                delivered.append(c.contract_id)
                self._pending_demand.remove(c)
            else:
                # 如果交付失败，记录违约 If delivery fails, record breach
                pass

        return delivered

    def _reduce_product_inventory(self, quantity: float) -> bool:
        """
        从产品库存中扣减指定数量，返回是否成功。
        Reduce the specified quantity from product inventory, return whether successful.

        Args:
            quantity: 需要扣减的数量 Quantity to reduce

        Returns:
            bool: 是否成功扣减 Whether successfully reduced
        """
        # 如果库存不足，返回失败 If inventory is insufficient, return failure
        if sum(b.remaining for b in self.product_batches) < quantity:
            return False

        # 按照生产日期排序，优先使用早期生产的产品 Sort by production date, prioritize early produced products
        self.product_batches.sort(key=lambda b: b.production_time)

        # 扣减库存 Reduce inventory
        remaining = quantity
        for batch in self.product_batches:
            if batch.remaining <= 0:
                continue
            if batch.remaining >= remaining:
                batch.remaining -= remaining
                remaining = 0
                break
            else:
                remaining -= batch.remaining
                batch.remaining = 0

        # 清理空批次 Clean up empty batches
        self.product_batches = [b for b in self.product_batches if b.remaining > 0]

        return True

    def get_production_plan_all(self) -> Dict[int, float]:
        """获取所有生产计划。
        Get all production plans."""
        return self.production_plan

    def get_production_plan(self, day: int | None = None) -> float:
        """获取指定日期的生产计划，如果不指定日期则返回当天的。
        Get the production plan for the specified date, or the current day if not specified."""
        if day is None:
            day = self.current_day
        return self.production_plan.get(day, 0.0)

    def get_total_future_production_plan(self) -> float:
        """获取未来所有的生产计划总量。
        Get the total amount of all future production plans."""
        return sum(qty for day, qty in self.production_plan.items() if day >= self.current_day)

    def get_max_possible_production(self, day: int) -> float:
        """获取指定日期的最大可能生产量。
        Get the maximum possible production for the specified date."""
        # 如果已经缓存，直接返回 If already cached, return directly
        if day in self._max_possible_prod_cache:
            return self._max_possible_prod_cache[day]

        # 获取当前到指定日期的原材料库存 Get raw material inventory from now to the specified date
        raw_summary = self.get_inventory_summary(day, MaterialType.RAW)
        raw_stock = raw_summary["current_stock"]

        # 获取当前到指定日期的产能 Get production capacity from now to the specified date
        capacity = self.daily_production_capacity

        # 最大可能生产量为原材料库存和产能的较小值 Maximum possible production is the smaller of raw material inventory and capacity
        max_prod = min(raw_stock, capacity)

        # 缓存结果 Cache result
        self._max_possible_prod_cache[day] = max_prod

        return max_prod

    def get_available_production_capacity(self, day: int) -> float:
        """获取指定日期的可用产能。
        Get the available production capacity for the specified date."""
        # 如果是过去的日期，返回0 If it's a past date, return 0
        if day < self.current_day:
            return 0.0

        # 获取当天的生产计划 Get production plan for the day
        planned = self.production_plan.get(day, 0.0)

        # 可用产能为总产能减去已规划的生产量 Available capacity is total capacity minus planned production
        return max(0.0, self.daily_production_capacity - planned)

    def get_insufficient_raw(self) -> Dict[int, Dict[str, float]]:
        """获取不足原料记录。
        Get insufficient raw material record."""
        return self.insufficient_raw

    def simulate_future_inventory(self, up_to_day: int) -> Dict[int, Dict[str, Dict]]:
        """
        模拟未来库存变化，返回每天的库存摘要。
        Simulate future inventory changes and return inventory summary for each day.

        Args:
            up_to_day: 模拟截止日期 Simulation end date

        Returns:
            Dict[int, Dict[str, Dict]]: 每天的库存摘要 {day: {"raw": raw_summary, "product": product_summary}}
        """
        # 初始化结果 Initialize result
        result = {}

        # 复制当前状态 Copy current state
        raw_batches = [Batch(b.batch_id, b.remaining, b.unit_cost, b.production_time) for b in self.raw_batches]
        product_batches = [Batch(b.batch_id, b.remaining, b.unit_cost, b.production_time) for b in self.product_batches]
        pending_supply = [IMContract(
            c.contract_id, c.partner_id, c.type, c.quantity, c.price, c.delivery_time, c.bankruptcy_risk, c.material_type
        ) for c in self._pending_supply]
        pending_demand = [IMContract(
            c.contract_id, c.partner_id, c.type, c.quantity, c.price, c.delivery_time, c.bankruptcy_risk, c.material_type
        ) for c in self._pending_demand]

        # 模拟每一天 Simulate each day
        for day in range(self.current_day, up_to_day + 1):
            # 计算当天库存摘要 Calculate inventory summary for the day
            raw_summary = self._compute_summary(
                day, raw_batches, self.raw_storage_cost, pending_supply, MaterialType.RAW
            )
            product_summary = self._compute_summary(
                day, product_batches, self.product_storage_cost, pending_demand, MaterialType.PRODUCT
            )

            # 添加到结果 Add to result
            result[day] = {
                "raw": raw_summary,
                "product": product_summary
            }

            # 模拟当天操作 Simulate operations for the day
            # 1. 接收原材料 Receive raw materials
            arrived = [c for c in pending_supply if c.delivery_time == day]
            for c in arrived:
                batch = Batch(
                    batch_id=c.contract_id,
                    remaining=c.quantity,
                    unit_cost=c.price,
                    production_time=day
                )
                raw_batches.append(batch)
                pending_supply.remove(c)

            # 2. 执行生产 Execute production
            prod_qty = self.production_plan.get(day, 0.0)
            if prod_qty > 0:
                # 按照生产日期排序，优先使用早期到货的原材料 Sort by production date, prioritize early delivered raw materials
                raw_batches.sort(key=lambda b: b.production_time)

                # 扣减原材料库存 Reduce raw material inventory
                remaining = prod_qty
                for batch in raw_batches:
                    if batch.remaining <= 0:
                        continue
                    if batch.remaining >= remaining:
                        batch.remaining -= remaining
                        remaining = 0
                        break
                    else:
                        remaining -= batch.remaining
                        batch.remaining = 0

                # 清理空批次 Clean up empty batches
                raw_batches = [b for b in raw_batches if b.remaining > 0]

                # 添加产品批次 Add product batch
                product_batches.append(Batch(
                    batch_id=f"P{day}",
                    remaining=prod_qty,
                    unit_cost=self.processing_cost,
                    production_time=day
                ))

            # 3. 交付产品 Deliver products
            to_deliver = [c for c in pending_demand if c.delivery_time == day]
            to_deliver.sort(key=lambda c: c.price, reverse=True)

            for c in to_deliver:
                # 计算当前产品库存 Calculate current product inventory
                current_prod_stock = sum(b.remaining for b in product_batches)

                # 如果库存足够，交付产品 If inventory is sufficient, deliver products
                if current_prod_stock >= c.quantity:
                    # 按照生产日期排序，优先使用早期生产的产品 Sort by production date, prioritize early produced products
                    product_batches.sort(key=lambda b: b.production_time)

                    # 扣减产品库存 Reduce product inventory
                    remaining = c.quantity
                    for batch in product_batches:
                        if batch.remaining <= 0:
                            continue
                        if batch.remaining >= remaining:
                            batch.remaining -= remaining
                            remaining = 0
                            break
                        else:
                            remaining -= batch.remaining
                            batch.remaining = 0

                    # 清理空批次 Clean up empty batches
                    product_batches = [b for b in product_batches if b.remaining > 0]

                    # 移除已交付的合同 Remove delivered contract
                    pending_demand.remove(c)

        return result

    def process_day_operations(self):
        """
        执行当天的生产和交割。
        Execute production and delivery for the current day.
        """
        # 1. 接收原材料 Receive raw materials
        self.receive_materials()

        # 2. 执行生产 Execute production
        self.execute_production(self, self.current_day)

        # 3. 交付产品 Deliver products
        self.deliver_products()

    def get_pending_contracts(self, is_supply: bool = None, day: int = None) -> List[IMContract]:
        """
        获取未交割的合同。
        Get pending contracts.

        Args:
            is_supply: 是否是供应合同，None表示所有合同 Whether it's a supply contract, None means all contracts
            day: 指定日期，None表示所有日期 Specified date, None means all dates

        Returns:
            List[IMContract]: 未交割的合同列表 List of pending contracts
        """
        # 初始化结果 Initialize result
        result = []

        # 根据类型筛选 Filter by type
        if is_supply is None:
            contracts = self._pending_supply + self._pending_demand
        elif is_supply:
            contracts = self._pending_supply
        else:
            contracts = self._pending_demand

        # 根据日期筛选 Filter by date
        if day is None:
            result = contracts
        else:
            result = [c for c in contracts if c.delivery_time == day]

        return result

    def get_batch_details(self, day: int, mtype: MaterialType) -> List[Dict]:
        """
        获取指定日期的批次详情。
        Get batch details for the specified date.

        Args:
            day: 指定日期 Specified date
            mtype: 材料类型 Material type

        Returns:
            List[Dict]: 批次详情列表 List of batch details
        """
        # 初始化结果 Initialize result
        result = []

        # 根据类型选择批次 Select batches by type
        if mtype == MaterialType.RAW:
            batches = self.raw_batches
        elif mtype == MaterialType.PRODUCT:
            batches = self.product_batches
        else:
            raise ValueError(f"Unknown material type: {mtype}")

        # 计算每个批次的详情 Calculate details for each batch
        for b in batches:
            if b.remaining <= 0:
                continue
            days_stored = max(0, day - b.production_time)
            storage_cost = self.raw_storage_cost if mtype == MaterialType.RAW else self.product_storage_cost
            c0 = b.unit_cost * b.remaining
            cs = storage_cost * days_stored * b.remaining
            result.append({
                "batch_id": b.batch_id,
                "remaining": b.remaining,
                "unit_cost": b.unit_cost,
                "production_time": b.production_time,
                "days_stored": days_stored,
                "storage_cost": cs,
                "total_cost": c0 + cs
            })

        return result

def execute_production(im, day):
    """
    执行生产，将原材料转换为产品。
    Execute production, convert raw materials to products.

    Args:
        im: 库存管理器 Inventory manager
        day: 执行生产的日期 Date to execute production
    """
    # 获取当天的生产计划 Get production plan for the day
    prod_qty = im.get_production_plan(day)
    if prod_qty <= 0:
        return

    # 检查原材料库存是否足够 Check if raw material inventory is sufficient
    raw_stock = sum(b.remaining for b in im.raw_batches)
    if raw_stock < prod_qty:
        # 如果库存不足，调整生产计划 If inventory is insufficient, adjust production plan
        prod_qty = raw_stock

    # 按照生产日期排序，优先使用早期到货的原材料 Sort by production date, prioritize early delivered raw materials
    im.raw_batches.sort(key=lambda b: b.production_time)

    # 扣减原材料库存 Reduce raw material inventory
    remaining = prod_qty
    for batch in im.raw_batches:
        if batch.remaining <= 0:
            continue
        if batch.remaining >= remaining:
            batch.remaining -= remaining
            remaining = 0
            break
        else:
            remaining -= batch.remaining
            batch.remaining = 0

    # 清理空批次 Clean up empty batches
    im.raw_batches = [b for b in im.raw_batches if b.remaining > 0]

    # 添加产品批次 Add product batch
    im.product_batches.append(Batch(
        batch_id=f"P{day}",
        remaining=prod_qty,
        unit_cost=im.processing_cost,
        production_time=day
    ))

# Example usage (optional, for testing during development)
if __name__ == '__main__':
    # Initialize InventoryManagerNS
    im = InventoryManagerNS(
        raw_storage_cost=0.01, 
        product_storage_cost=0.02,
        processing_cost=2.0,
        daily_production_capacity=30.0,
        max_day=10
    )