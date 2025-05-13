# deprecated， DO NOT USE!!!!!
# There are unlimited bugs here!

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List, Set
from enum import Enum
from datetime import datetime

class ContractType(Enum):
    """合同类型"""
    SUPPLY = "supply"  # 原料采购合同
    SALES = "sales"    # 产品销售合同

class MaterialType(Enum):
    """材料类型"""
    RAW = "raw_material"    # 原料
    PRODUCT = "product"     # 产品

class ContractStatus(Enum):
    """合同状态"""
    PENDING = "pending"       # 待交割
    COMPLETED = "completed"   # 已完成
    FAILED = "failed"        # 交割失败

@dataclass
class Contract:
    """合同基本信息"""
    contract_id: str
    partner_id: str
    type: ContractType
    quantity: int
    price: float
    delivery_time: int
    bankruptcy_risk: float
    material_type: MaterialType

@dataclass
class InventoryStatus:
    """库存状态"""
    quantity: int          # 当前库存量
    avg_cost: float       # 加权平均成本
    storage_cost: float   # 单位存储成本
    consumed: int = 0     # 已消耗量 仅供RAW用
    ordered: int = 0      # 订单量 仅供Product用

@dataclass
class OrderStatus:
    """订单状态"""
    contract: Contract
    status: ContractStatus = ContractStatus.PENDING
    actual_quantity: int = 0
    completion_time: Optional[datetime] = None

@dataclass
class BatchInfo:
    """批次信息"""
    batch_id: str          # 批次编号
    quantity: int         # 数量
    unit_cost: float     # 单位成本
    production_time: int  # 生产/入库时间
    source_id: str       # 来源(合同ID或生产批次号)
    remaining: int       # 剩余数量

class InventoryManager:
    """库存管理系统"""
    
    def __init__(
        self, 
        raw_storage_cost: float = 1.0,      # 原料单位存储成本
        product_storage_cost: float = 2.0,   # 产品单位存储成本
        production_rate: int = 1,            # 生产转化率
        production_capacity: int = 100,      # 日生产能力
        processing_cost: float = 10.0        # 单位加工成本
    ):
        self.raw_storage_cost = raw_storage_cost
        self.product_storage_cost = product_storage_cost
        self.production_rate = production_rate
        self.production_capacity = production_capacity
        self.processing_cost = processing_cost
        self.current_day = 0

        # 库存记录
        self.raw_inventory: Dict[int, InventoryStatus] = {}
        self.product_inventory: Dict[int, InventoryStatus] = {}
        
        # 订单管理
        self.orders: Dict[str, OrderStatus] = {}
        self.failed_orders: Set[str] = set()
        
        # 生产计划
        self.production_schedule: Dict[int, int] = {}
        
        # 初始化第0天的库存
        initial_status = InventoryStatus(
            quantity=0,
            avg_cost=0.0,
            storage_cost=0.0,
            consumed=0,
            ordered=0
        )
        self.raw_inventory[0] = initial_status
        self.product_inventory[0] = initial_status

        # 批次管理
        self.raw_batches: List[BatchInfo] = []
        self.product_batches: List[BatchInfo] = []
        self.batch_counter = 0
        
    def _generate_batch_id(self, prefix: str) -> str:
        """生成批次编号"""
        self.batch_counter += 1
        return f"{prefix}_{self.current_day}_{self.batch_counter}"
        
    def receive_materials(self) -> List[str]:
        """
        处理当日交割的原料
        
        Returns:
            List[str]: 成功交割的合同ID列表
        """
        completed_contracts = []
        
        for order in self.orders.values():
            contract = order.contract
            if (contract.delivery_time == self.current_day and
                contract.type == ContractType.SUPPLY and
                contract.material_type == MaterialType.RAW and
                order.status == ContractStatus.PENDING):
                
                # 创建新批次
                batch = BatchInfo(
                    batch_id=self._generate_batch_id("RAW"),
                    quantity=contract.quantity,
                    unit_cost=contract.price,  # 初始单位成本为合同价格
                    production_time=self.current_day,
                    source_id=contract.contract_id,
                    remaining=contract.quantity
                )
                self.raw_batches.append(batch)
                
                # 更新订单状态
                order.status = ContractStatus.COMPLETED
                order.completion_time = datetime.now()
                order.actual_quantity = contract.quantity
                completed_contracts.append(contract.contract_id)
                
                # 更新库存状态
                curr_status = self.raw_inventory[self.current_day]
                self.raw_inventory[self.current_day] = InventoryStatus(
                    quantity=curr_status.quantity + contract.quantity,
                    avg_cost=self._calculate_new_avg_cost(
                        curr_status.quantity,
                        curr_status.avg_cost,
                        contract.quantity,
                        contract.price
                    ),
                    storage_cost=self._calculate_storage_cost(curr_status.quantity + contract.quantity, MaterialType.RAW),
                    consumed=curr_status.consumed,
                    ordered=curr_status.ordered
                )
                
        return completed_contracts
        
    def produce(self) -> bool:
        """
        执行当日生产计划，使用FIFO原则消耗原料
        
        Returns:
            bool: 生产是否成功
        """
        production_qty = self.production_schedule.get(self.current_day, 0)
        if production_qty == 0:
            return True
            
        raw_needed = production_qty * self.production_rate
        if self.raw_inventory[self.current_day].quantity < raw_needed:
            return False
            
        # 按FIFO原则消耗原料
        total_raw_cost = 0.0
        consumed_raw = 0
        used_batches = []  # 记录使用的批次及数量
        
        for batch in self.raw_batches:
            if consumed_raw >= raw_needed:
                break
                
            consume_qty = min(batch.remaining, raw_needed - consumed_raw)
            total_raw_cost += consume_qty * batch.unit_cost
            consumed_raw += consume_qty
            used_batches.append((batch, consume_qty))
            
        if consumed_raw < raw_needed:
            return False
            
        # 更新原料批次
        for batch, used_qty in used_batches:
            batch.remaining -= used_qty
            if batch.remaining == 0:
                self.raw_batches.remove(batch)
                
        # 计算生产成本和创建产品批次
        processing_cost = total_raw_cost * 0.2  # 加工成本假设为原料成本的20%
        total_cost = total_raw_cost + processing_cost
        unit_cost = total_cost / production_qty
        
        # 创建产品批次
        product_batch = BatchInfo(
            batch_id=self._generate_batch_id("PROD"),
            quantity=production_qty,
            unit_cost=unit_cost,
            production_time=self.current_day,
            source_id=f"PROD_{self.current_day}",
            remaining=production_qty
        )
        self.product_batches.append(product_batch)
        
        # 更新库存状态
        self._update_inventory_after_production(raw_needed, production_qty, unit_cost)
        return True
        
    def deliver_products(self) -> List[Tuple[str, bool]]:
        """
        交付当日的产品订单
        
        Returns:
            List[Tuple[str, bool]]: 合同ID和交付是否成功的列表
        """
        results = []
        
        for order in self.orders.values():
            contract = order.contract
            if (contract.delivery_time == self.current_day and
                contract.type == ContractType.SALES and
                contract.material_type == MaterialType.PRODUCT and
                order.status == ContractStatus.PENDING):
                
                # 检查库存是否足够
                if self.product_inventory[self.current_day].quantity < contract.quantity:
                    order.status = ContractStatus.FAILED
                    self.failed_orders.add(contract.contract_id)
                    results.append((contract.contract_id, False))
                    continue
                    
                # 按FIFO原则交付产品
                delivered = 0
                used_batches = []
                
                for batch in self.product_batches:
                    if delivered >= contract.quantity:
                        break
                        
                    deliver_qty = min(batch.remaining, contract.quantity - delivered)
                    delivered += deliver_qty
                    used_batches.append((batch, deliver_qty))
                    
                # 更新产品批次
                for batch, used_qty in used_batches:
                    batch.remaining -= used_qty
                    if batch.remaining == 0:
                        self.product_batches.remove(batch)
                        
                # 更新订单和库存状态
                order.status = ContractStatus.COMPLETED
                order.completion_time = datetime.now()
                order.actual_quantity = contract.quantity
                
                self._update_inventory_after_delivery(contract)
                results.append((contract.contract_id, True))
                
        return results
                        
    def update_day(self) -> None:
        """
        执行日期更新，将当前日期增加1天，并同步更新库存状态
        此方法应在每天结束时调用，用于准备下一天的库存状态
        """
        print(f"正在执行update_day， 当前日期: {self.current_day}")
        print(f"当前原料库存: {self.raw_inventory[self.current_day].quantity}")
        # 先更新批次的存储成本
        self.update_batches_storage_cost()
                            
        # 日期递增
        self.current_day += 1

        # 确保新的一天有库存状态记录
        print(f"{self.raw_inventory}")
        if self.current_day not in self.raw_inventory:
            # 复制前一天的状态作为新一天的起始状态
            prev_raw_status = self.raw_inventory[self.current_day - 1]
            print(f"复制前一天的原料库存状态: {prev_raw_status}")
            self.raw_inventory[self.current_day] = InventoryStatus(
                quantity=prev_raw_status.quantity,
                avg_cost=prev_raw_status.avg_cost,
                storage_cost=self._calculate_storage_cost(
                    prev_raw_status.quantity,
                    MaterialType.RAW
                ),
                consumed=0,  # 新的一天消耗量从0开始
                ordered=0    # 新的一天订单量从0开始
            )

        if self.current_day not in self.product_inventory:
            # 复制前一天的状态作为新一天的起始状态
            prev_prod_status = self.product_inventory[self.current_day - 1]
            self.product_inventory[self.current_day] = InventoryStatus(
                quantity=prev_prod_status.quantity,
                avg_cost=prev_prod_status.avg_cost,
                storage_cost=self._calculate_storage_cost(
                    prev_prod_status.quantity,
                    MaterialType.PRODUCT
                ),
                consumed=0,  # 新的一天消耗量从0开始
                ordered=0    # 新的一天订单量从0开始
            )
        print(f"结束执行update_day， 当前日期: {self.current_day}")
        print(f"当前原料库存: {self.raw_inventory[self.current_day].quantity}")
                            


    def get_inventory_summary(self, day: int, material_type: MaterialType) -> dict:
        """
        获取指定日期的库存汇总信息，包括预估可用量和预估平均成本
        
        Args:
            day: 查询日期
            material_type: 材料类型
            
        Returns:
            包含当前和预估库存信息的字典
        """
        inventory = (self.raw_inventory if material_type == MaterialType.RAW 
                    else self.product_inventory)
        
        # 获取基础库存信息
        status = inventory.get(day, InventoryStatus(0, 0.0, 0.0))
        
        # 获取当日待处理的订单
        pending_supply = [
            (order.contract.quantity, order.contract.price)
            for order in self.orders.values()
            if (order.contract.delivery_time == day and
                order.contract.material_type == material_type and
                order.contract.type == ContractType.SUPPLY and
                order.status == ContractStatus.PENDING)
        ]
        
        pending_sales = [
            order.contract.quantity
            for order in self.orders.values()
            if (order.contract.delivery_time == day and
                order.contract.material_type == material_type and
                order.contract.type == ContractType.SALES and
                order.status == ContractStatus.PENDING)
        ]
        
        failed = sum(
            order.contract.quantity
            for order in self.orders.values()
            if (order.contract.delivery_time == day and
                order.contract.material_type == material_type and
                order.contract.contract_id in self.failed_orders)
        )
        
        # 计算预估库存和成本
        if material_type == MaterialType.RAW:
            # 原料的预估：当前库存 + 待收货 - 生产计划消耗
            total_pending_in = sum(qty for qty, _ in pending_supply)
            # 从今天到查询日的生产计划之和
            total_pending_out = 0
            for d in range(self.current_day+1, day):
                total_pending_out += self.production_schedule.get(d, 0)
            production_consumption = self.production_schedule.get(day, 0) 
            
            # 预估可用量计算不变
            estimated_available = (status.quantity + 
                                 total_pending_in - 
                                 production_consumption)
            
            # 计算预估平均成本
            if estimated_available > 0:
                current_value = status.quantity * status.avg_cost
                pending_value = sum(qty * price for qty, price in pending_supply)
                estimated_avg_cost = ((current_value + pending_value) / 
                                    (status.quantity + total_pending_in))
            else:
                estimated_avg_cost = status.avg_cost
            
        else:  # MaterialType.PRODUCT
            # 产品的预估：当前库存 + 计划生产 - 待发货
            production_qty = self.production_schedule.get(day, 0)
            total_pending_in = 0
            for d in range(self.current_day, day):
                total_pending_in += self.production_schedule.get(d, 0)
            total_pending_out = sum(pending_sales)
            
            estimated_available = (status.quantity + 
                                 production_qty - 
                                 total_pending_out)
            
            # 计算预估平均成本
            if production_qty > 0:
                # 预估生产成本
                raw_cost = self._estimate_raw_material_cost(day, production_qty)
                processing_cost = self.processing_cost
                production_unit_cost = (raw_cost + processing_cost) / production_qty
                
                if estimated_available > 0:
                    current_value = status.quantity * status.avg_cost
                    production_value = production_qty * production_unit_cost
                    estimated_avg_cost = ((current_value + production_value) / 
                                        (status.quantity + production_qty))
                else:
                    estimated_avg_cost = status.avg_cost
            else:
                estimated_avg_cost = status.avg_cost
    
        return {
            "current_stock": status.quantity,
            "total_stock": status.quantity + status.consumed,
            "consumed": status.consumed,
            "pending_in": total_pending_in,
            "pending_out": total_pending_out,
            "failed_quantity": failed,
            "storage_cost": status.storage_cost,
            "average_cost": status.avg_cost,
            "estimated_available": max(0, estimated_available),
            "estimated_average_cost": estimated_avg_cost
        }

    def _estimate_raw_material_cost(self, day: int, production_qty: int) -> float:
        """
        估算生产所需原料的成本
        
        Args:
            day: 生产日期
            production_qty: 计划生产数量
            
        Returns:
            float: 预计的原料总成本
        """
        raw_needed = production_qty * self.production_rate
        raw_status = self.raw_inventory.get(day, InventoryStatus(0, 0.0, 0.0))
        
        if raw_status.quantity >= raw_needed:
            # 如果当前库存足够，使用当前平均成本
            return raw_needed * raw_status.avg_cost
        
        # 需要考虑待收货的原料成本
        pending_supply = [
            (order.contract.quantity, order.contract.price)
            for order in self.orders.values()
            if (order.contract.delivery_time == day and
                order.contract.material_type == MaterialType.RAW and
                order.contract.type == ContractType.SUPPLY and
                order.status == ContractStatus.PENDING)
        ]
        
        total_cost = raw_status.quantity * raw_status.avg_cost
        total_qty = raw_status.quantity
        
        for qty, price in pending_supply:
            if total_qty >= raw_needed:
                break
            usable_qty = min(qty, raw_needed - total_qty)
            total_cost += usable_qty * price
            total_qty += usable_qty
        
        return total_cost

    def _restore_inventory_status(self, 
                                day: int, 
                                status: InventoryStatus,
                                material_type: MaterialType) -> None:
        """还原库存状态"""
        inventory = (self.raw_inventory if material_type == MaterialType.RAW 
                    else self.product_inventory)
        inventory[day] = status
    def _calculate_storage_cost(self, quantity: int, material_type: MaterialType) -> float:
        """计算存储成本"""
        return (quantity * self.raw_storage_cost 
                if material_type == MaterialType.RAW 
                else quantity * self.product_storage_cost)

    def _update_inventory_after_production(self, 
                                        raw_used: int, 
                                        prod_qty: int, 
                                        unit_cost: float) -> None:
        """更新生产后的库存状态"""
        # 更新原料库存
        raw_status = self.raw_inventory[self.current_day]
        new_raw_qty = raw_status.quantity - raw_used
        self.raw_inventory[self.current_day] = InventoryStatus(
            quantity=new_raw_qty,
            avg_cost=raw_status.avg_cost,
            storage_cost=self._calculate_storage_cost(new_raw_qty, MaterialType.RAW),
            consumed=raw_status.consumed + raw_used,
            ordered=raw_status.ordered
        )
        
        # 更新产品库存
        prod_status = self.product_inventory[self.current_day]
        new_prod_qty = prod_status.quantity + prod_qty
        self.product_inventory[self.current_day] = InventoryStatus(
            quantity=new_prod_qty,
            avg_cost=self._calculate_new_avg_cost(
                prod_status.quantity,
                prod_status.avg_cost,
                prod_qty,
                unit_cost
            ),
            storage_cost=self._calculate_storage_cost(new_prod_qty, MaterialType.PRODUCT),
            consumed=prod_status.consumed,
            ordered=prod_status.ordered
        )
    
    def add_transaction(self, contract: Contract) -> bool:
        """添加新交易并更新库存预期"""
        if contract.delivery_time < self.current_day:
            return False
            
        self.orders[contract.contract_id] = OrderStatus(contract=contract)
        
        # 确保库存记录存在
        inventory = (self.raw_inventory if contract.material_type == MaterialType.RAW 
                    else self.product_inventory)
        
        for day in range(self.current_day, contract.delivery_time + 1):
            if day not in inventory:
                prev_status = inventory[day - 1]
                inventory[day] = InventoryStatus(
                    quantity=prev_status.quantity,
                    avg_cost=prev_status.avg_cost,
                    storage_cost=self._calculate_storage_cost(
                        prev_status.quantity,
                        contract.material_type
                    ),
                    consumed=0,
                    ordered=0
                )
        
        curr_status = inventory[contract.delivery_time]
        
        if contract.type == ContractType.SUPPLY:
            if contract.material_type == MaterialType.PRODUCT:
                self.orders.pop(contract.contract_id)
                return False
                
            # 不提前更新实际库存量，只记录合同
            # 实际库存量将在receive_materials方法执行时更新
            # 注意：这里不需要修改库存状态
            
        else:  # ContractType.SALES
            summary = self.get_inventory_summary(
                contract.delivery_time, 
                contract.material_type
            )
            if summary["estimated_available"] < contract.quantity:
                self.orders.pop(contract.contract_id)
                return False
                
            inventory[contract.delivery_time] = InventoryStatus(
                quantity=curr_status.quantity,
                avg_cost=curr_status.avg_cost,
                storage_cost=curr_status.storage_cost,
                consumed=curr_status.consumed,
                ordered=curr_status.ordered + contract.quantity
            )
            
            if (contract.material_type == MaterialType.RAW and 
                summary["estimated_available"] - contract.quantity < 
                self.production_schedule.get(contract.delivery_time, 0) * self.production_rate):
                self.orders.pop(contract.contract_id)
                self._restore_inventory_status(contract.delivery_time, curr_status, 
                                            contract.material_type)
                return False
        
        return True
    def _calculate_new_avg_cost(self, 
                              old_quantity: int, 
                              old_avg_cost: float, 
                              new_quantity: int, 
                              new_cost: float) -> float:
        """
        计算新的加权平均成本
        
        Args:
            old_quantity: 原有库存数量
            old_avg_cost: 原有平均成本
            new_quantity: 新增数量
            new_cost: 新增单位成本
            
        Returns:
            float: 新的加权平均成本
        """
        if old_quantity + new_quantity == 0:
            return 0.0
            
        total_value = old_quantity * old_avg_cost + new_quantity * new_cost
        return total_value / (old_quantity + new_quantity)
    def _update_inventory_after_delivery(self, contract: Contract) -> None:
        """
        更新交付产品后的库存状态
        
        Args:
            contract: 交付的合同信息
        """
        if contract.material_type != MaterialType.PRODUCT:
            return
            
        # 获取当前库存状态
        curr_status = self.product_inventory[self.current_day]
        
        # 计算交付后的新库存量
        new_quantity = curr_status.quantity - contract.quantity
        
        # 对于产品交付，我们不需要更新平均成本，因为FIFO出库不影响单位成本
        # 但需要更新存储成本
        self.product_inventory[self.current_day] = InventoryStatus(
            quantity=new_quantity,
            avg_cost=curr_status.avg_cost,  # 平均成本不变
            storage_cost=self._calculate_storage_cost(new_quantity, MaterialType.PRODUCT),
            consumed=curr_status.consumed + contract.quantity,
            ordered=curr_status.ordered
        )
        
    def update_batches_storage_cost(self) -> None:
        """
        更新所有批次的存储成本，并重新计算库存的平均成本
        此方法应在每天结束时调用，用于累积存储成本
        """
        # 更新原料批次
        total_raw_value = 0
        total_raw_quantity = 0
        
        for batch in self.raw_batches:
            # 计算存储天数
            storage_days = self.current_day - batch.production_time
            if storage_days > 0:
                # 计算当日存储成本
                daily_storage_cost = self.raw_storage_cost * batch.remaining
                # 累加到批次单位成本中
                additional_cost_per_unit = daily_storage_cost / batch.remaining
                batch.unit_cost += additional_cost_per_unit
            
            # 累加总价值和数量
            total_raw_value += batch.unit_cost * batch.remaining
            total_raw_quantity += batch.remaining
        
        # 更新产品批次
        total_product_value = 0
        total_product_quantity = 0
        
        for batch in self.product_batches:
            # 计算存储天数
            storage_days = self.current_day - batch.production_time
            if storage_days > 0:
                # 计算当日存储成本
                daily_storage_cost = self.product_storage_cost * batch.remaining
                # 累加到批次单位成本中
                additional_cost_per_unit = daily_storage_cost / batch.remaining
                batch.unit_cost += additional_cost_per_unit
            
            # 累加总价值和数量
            total_product_value += batch.unit_cost * batch.remaining
            total_product_quantity += batch.remaining
        
        # 更新库存状态中的平均成本
        if self.current_day in self.raw_inventory:
            curr_raw_status = self.raw_inventory[self.current_day]
            if total_raw_quantity > 0:
                new_raw_avg_cost = total_raw_value / total_raw_quantity
                self.raw_inventory[self.current_day] = InventoryStatus(
                    quantity=curr_raw_status.quantity,
                    avg_cost=new_raw_avg_cost,
                    storage_cost=self._calculate_storage_cost(curr_raw_status.quantity, MaterialType.RAW),
                    consumed=curr_raw_status.consumed,
                    ordered=curr_raw_status.ordered
                )
        
        if self.current_day in self.product_inventory:
            curr_prod_status = self.product_inventory[self.current_day]
            if total_product_quantity > 0:
                new_prod_avg_cost = total_product_value / total_product_quantity
                self.product_inventory[self.current_day] = InventoryStatus(
                    quantity=curr_prod_status.quantity,
                    avg_cost=new_prod_avg_cost,
                    storage_cost=self._calculate_storage_cost(curr_prod_status.quantity, MaterialType.PRODUCT),
                    consumed=curr_prod_status.consumed,
                    ordered=curr_prod_status.ordered
                )