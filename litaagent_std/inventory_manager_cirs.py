import os
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, TYPE_CHECKING
import uuid # For contract IDs
import copy # For deepcopy

if TYPE_CHECKING:
    # This is to avoid circular dependencies if CustomInventoryManager methods type hint themselves
    # or if other modules need to type hint CustomInventoryManager without full import.
    pass

class MaterialType(Enum):
    RAW = auto()
    PRODUCT = auto()

class IMContractType(Enum):
    SUPPLY = auto()  # Buying from supplier (supplies raw to agent)
    DEMAND = auto()  # Selling to consumer (agent demands product from its stock)

@dataclass
class IMContract:
    partner_id: str
    contract_id: str
    type: IMContractType
    quantity: int
    price: float
    delivery_time: int
    material_type: MaterialType
    bankruptcy_risk: float = 0.0

@dataclass
class Batch: # For tracking inventory
    batch_id: str # Added default factory for batch_id
    original_quantity: int
    remaining_quantity: int
    unit_cost: float # Cost at which it was acquired (raw) or produced (product)
    arrival_or_production_time: int
    material_type: MaterialType

class InventoryManagerCIRS:
    def __init__(self,
                 raw_storage_cost: float,
                 product_storage_cost: float,
                 processing_cost: float,
                 daily_production_capacity: int, # Was float, now int (except for float('inf'))
                 max_simulation_day: int,
                 current_day: int = 0):

        self.current_day: int = current_day
        self.max_simulation_day: int = max_simulation_day

        # Costs
        self.raw_storage_cost_per_unit_per_day: float = raw_storage_cost
        self.product_storage_cost_per_unit_per_day: float = product_storage_cost
        self.processing_cost_per_unit: float = processing_cost # Cost to convert 1 raw to 1 product

        self.daily_production_capacity: float = daily_production_capacity

        # Core Data Structures
        self.raw_material_batches: List[Batch] = []
        self.product_batches: List[Batch] = []

        # Pending contracts (future commitments)
        # These store IMContract objects
        self.pending_supply_contracts: List[IMContract] = [] # Raw materials agent will receive
        self.pending_demand_contracts: List[IMContract] = [] # Products agent must deliver

        # Production plans: Dict[day, quantity_to_produce]
        self.production_plan: Dict[int, float] = {}

        # Statistics / Metrics (optional, can be expanded later)
        self.total_raw_material_acquired: float = 0.0
        self.total_products_produced: float = 0.0
        self.total_products_sold: float = 0.0
        # ... any other metrics useful for the agent's strategy

        self.is_deepcopy = False
    # --- Core Logic Methods ---

    def add_transaction(self, contract: IMContract) -> bool:
        if contract.delivery_time < self.current_day:
            return False

        if contract.type == IMContractType.SUPPLY:
            if contract.material_type != MaterialType.RAW:
                return False
            self.pending_supply_contracts.append(contract)
            self.pending_supply_contracts.sort(key=lambda c: c.delivery_time)
        elif contract.type == IMContractType.DEMAND:
            if contract.material_type != MaterialType.PRODUCT:
                return False
            self.pending_demand_contracts.append(contract)
            self.pending_demand_contracts.sort(key=lambda c: c.delivery_time)
        else:
            return False

        # Re-plan production whenever a new contract is added
        self.plan_production()
        return True

    def get_inventory_summary(self, day: int, mtype: MaterialType) -> Dict[str, Any]:
        """
        Get a summary of inventory for a specific day and material type.
        
        Args:
            day: The day for which to get the summary
            mtype: The material type (RAW or PRODUCT)
            
        Returns:
            Dict with inventory summary information
        """
        # Initialize summary
        summary = {
            "current_stock": 0,
            "current_cost": 0.0,
            "current_storage_cost": 0.0,
            "current_avg_cost": 0.0,
            "estimated_stock": 0.0,
            "estimated_cost": 0.0,
            "estimated_avg_cost": 0.0
        }
        
        # Get the appropriate batches and storage cost based on material type
        if mtype == MaterialType.RAW:
            batches = self.raw_material_batches
            storage_cost = self.raw_storage_cost_per_unit_per_day
            pending_contracts = self.pending_supply_contracts
        else:  # PRODUCT
            batches = self.product_batches
            storage_cost = self.product_storage_cost_per_unit_per_day
            pending_contracts = self.pending_demand_contracts
        
        # Calculate current stock and costs
        total_quantity = 0
        total_base_cost = 0.0
        total_storage_cost = 0.0
        
        for batch in batches:
            if batch.remaining_quantity <= 0:
                continue
                
            # Calculate days stored
            days_stored = max(0, day - batch.arrival_or_production_time)
            
            # Calculate costs
            base_cost = batch.unit_cost * batch.remaining_quantity
            storage_cost_for_batch = storage_cost * days_stored * batch.remaining_quantity
            
            # Update totals
            total_quantity += batch.remaining_quantity
            total_base_cost += base_cost
            total_storage_cost += storage_cost_for_batch
        
        # Update summary with current values
        summary["current_stock"] = total_quantity
        summary["current_cost"] = total_base_cost + total_storage_cost
        summary["current_storage_cost"] = total_storage_cost
        
        # Calculate average cost if there's stock
        if total_quantity > 0:
            summary["current_avg_cost"] = (total_base_cost + total_storage_cost) / total_quantity
        
        # Calculate estimated future values
        if mtype == MaterialType.RAW:
            # For raw materials: current + future deliveries - planned production
            future_deliveries = sum(c.quantity for c in pending_contracts 
                                  if self.current_day <= c.delivery_time <= day)
            
            # Sum production plan from current day to the target day
            planned_production = sum(self.production_plan.get(d, 0) 
                                   for d in range(self.current_day, day))
            
            estimated_stock = total_quantity + future_deliveries - planned_production
            
        else:  # PRODUCT
            # For products: current - future deliveries + planned production
            future_deliveries = sum(c.quantity for c in pending_contracts 
                                  if self.current_day <= c.delivery_time <= day)
            
            # Sum production plan from current day to the target day
            planned_production = sum(self.production_plan.get(d, 0) 
                                   for d in range(self.current_day, day + 1))
            
            estimated_stock = total_quantity - future_deliveries + planned_production
        
        # Update summary with estimated values
        summary["estimated_stock"] = max(0, estimated_stock)  # Can't be negative
        
        # Calculate future costs (simplified - doesn't account for storage of future deliveries)
        future_cost = sum(c.price * c.quantity for c in pending_contracts 
                        if self.current_day <= c.delivery_time <= day)
        
        summary["estimated_cost"] = total_base_cost + total_storage_cost + future_cost
        
        # Calculate estimated average cost if there's estimated stock
        if estimated_stock > 0:
            summary["estimated_avg_cost"] = summary["estimated_cost"] / estimated_stock
        
        return summary

    def _receive_materials(self, day_being_processed: int) -> None:
        """
        Process all supply contracts scheduled for delivery on the specified day.
        Creates inventory batches for the received materials.
        
        Args:
            day_being_processed: The day for which to process deliveries
        """
        # Find all supply contracts for the current day
        contracts_to_receive = [c for c in self.pending_supply_contracts 
                              if c.delivery_time == day_being_processed]
        
        if not contracts_to_receive:
            return
            
        # Process each contract
        for contract in contracts_to_receive:
            # Verify contract is for raw materials
            if contract.material_type != MaterialType.RAW:
                continue
                
            # Create a new batch for the received materials
            new_batch = Batch(
                batch_id=contract.contract_id,
                original_quantity=contract.quantity,
                remaining_quantity=contract.quantity,
                unit_cost=contract.price,
                arrival_or_production_time=day_being_processed,
                material_type=MaterialType.RAW
            )
            
            # Add to inventory
            self.raw_material_batches.append(new_batch)
            
            # Update statistics
            self.total_raw_material_acquired += contract.quantity
            
            # Remove from pending contracts
            self.pending_supply_contracts.remove(contract)

    def _execute_production(self, day_being_processed: int) -> None:
        """
        Execute production for the specified day according to the production plan.
        Consumes raw materials and creates product batches.
        
        Args:
            day_being_processed: The day for which to execute production
        """
        # Get planned production for this day
        planned_production = self.production_plan.get(day_being_processed, 0)
        
        if planned_production <= 0:
            return
            
        # Check if we have enough raw materials
        total_raw_available = sum(batch.remaining_quantity for batch in self.raw_material_batches)
        
        if total_raw_available < planned_production:
            # Not enough raw materials, adjust production
            actual_production = total_raw_available
        else:
            actual_production = planned_production
            
        if actual_production <= 0:
            return
            
        # Sort raw material batches by arrival time (FIFO)
        self.raw_material_batches.sort(key=lambda b: b.arrival_or_production_time)
        
        # Consume raw materials
        remaining_to_consume = actual_production
        consumed_cost = 0.0  # Track the cost of consumed materials for product cost calculation
        
        for batch in self.raw_material_batches:
            if remaining_to_consume <= 0:
                break
                
            if batch.remaining_quantity <= 0:
                continue
                
            # Determine how much to consume from this batch
            consume_from_batch = min(batch.remaining_quantity, remaining_to_consume)
            
            # Update batch
            batch.remaining_quantity -= consume_from_batch
            
            # Update tracking variables
            consumed_cost += batch.unit_cost * consume_from_batch
            remaining_to_consume -= consume_from_batch
            
        # Clean up empty batches
        self.raw_material_batches = [b for b in self.raw_material_batches if b.remaining_quantity > 0]
        
        # Calculate average cost of consumed materials
        avg_raw_cost = consumed_cost / actual_production if actual_production > 0 else 0
        
        # Create product batch
        # Total cost includes raw material cost plus processing cost
        product_unit_cost = avg_raw_cost + self.processing_cost_per_unit
        
        new_product_batch = Batch(
            batch_id=f"PROD_{day_being_processed}_{uuid.uuid4().hex[:8]}",
            original_quantity=actual_production,
            remaining_quantity=actual_production,
            unit_cost=product_unit_cost,
            arrival_or_production_time=day_being_processed,
            material_type=MaterialType.PRODUCT
        )
        
        # Add to inventory
        self.product_batches.append(new_product_batch)
        
        # Update statistics
        self.total_products_produced += actual_production

    def _deliver_products(self, day_being_processed: int) -> None:
        """
        Process all demand contracts scheduled for delivery on the specified day.
        Removes products from inventory to fulfill contracts.
        
        Args:
            day_being_processed: The day for which to process deliveries
        """
        # Find all demand contracts for the current day
        contracts_to_deliver = [c for c in self.pending_demand_contracts 
                              if c.delivery_time == day_being_processed]
        
        if not contracts_to_deliver:
            return
            
        # Sort contracts by price (deliver highest price first if we have limited inventory)
        contracts_to_deliver.sort(key=lambda c: c.price, reverse=True)
        
        # Process each contract
        for contract in contracts_to_deliver:
            # Verify contract is for products
            if contract.material_type != MaterialType.PRODUCT:
                continue
                
            # Check if we have enough products
            total_products_available = sum(batch.remaining_quantity for batch in self.product_batches)
            
            if total_products_available < contract.quantity:
                # Not enough products to fulfill contract
                # In a real system, this would trigger a breach of contract
                # For now, we'll just skip it and leave it in pending
                continue
                
            # Sort product batches by production time (FIFO)
            self.product_batches.sort(key=lambda b: b.arrival_or_production_time)
            
            # Consume products
            remaining_to_deliver = contract.quantity
            
            for batch in self.product_batches:
                if remaining_to_deliver <= 0:
                    break
                    
                if batch.remaining_quantity <= 0:
                    continue
                    
                # Determine how much to take from this batch
                take_from_batch = min(batch.remaining_quantity, remaining_to_deliver)
                
                # Update batch
                batch.remaining_quantity -= take_from_batch
                
                # Update tracking variable
                remaining_to_deliver -= take_from_batch
                
            # Clean up empty batches
            self.product_batches = [b for b in self.product_batches if b.remaining_quantity > 0]
            
            # Update statistics
            self.total_products_sold += contract.quantity
            
            # Remove from pending contracts
            self.pending_demand_contracts.remove(contract)

    def process_day_end_operations(self, day_being_processed: int) -> None:
        """
        Process all operations for the end of a day: receive materials, execute production,
        and deliver products.
        
        Args:
            day_being_processed: The day to process
        """
        if day_being_processed < self.current_day:
            return
            
        # Process in order: receive, produce, deliver
        self._receive_materials(day_being_processed)
        self._execute_production(day_being_processed)
        self._deliver_products(day_being_processed)
        
        # Update current day
        self.current_day = day_being_processed + 1
        
        # Re-plan production after processing the day
        self.plan_production()

    def get_today_insufficient_raw(self, day: int) -> int:
        """
        Calculate how much raw material is needed for today's production but not available.
        
        Args:
            day: The day to check
            
        Returns:
            Amount of raw material shortage for the day
        """
        # Get planned production for the day
        planned_production = self.production_plan.get(day, 0)
        
        if planned_production <= 0:
            return 0
            
        # Get available raw materials
        available_raw = sum(batch.remaining_quantity for batch in self.raw_material_batches)
        
        # Calculate shortage
        shortage = max(0, planned_production - available_raw)
        
        return shortage

    def get_total_insufficient_raw(self, target_day: int, horizon: int) -> int:
        """
        Calculate the total raw material shortage over a time horizon.
        
        Args:
            target_day: The starting day
            horizon: Number of days to look ahead
            
        Returns:
            Total raw material shortage over the horizon
        """
        # Initialize variables
        total_shortage = 0
        cumulative_raw = sum(batch.remaining_quantity for batch in self.raw_material_batches)
        
        # Look at each day in the horizon
        for day in range(target_day, target_day + horizon):
            # Add any raw materials scheduled for delivery
            incoming_raw = sum(c.quantity for c in self.pending_supply_contracts 
                             if c.delivery_time == day)
            cumulative_raw += incoming_raw
            
            # Subtract production needs
            production_needs = self.production_plan.get(day, 0)
            
            if production_needs > cumulative_raw:
                # Record shortage
                day_shortage = production_needs - cumulative_raw
                total_shortage += day_shortage
                cumulative_raw = 0
            else:
                # Enough materials, update remaining
                cumulative_raw -= production_needs
                
        return total_shortage

    def plan_production(self, up_to_day: Optional[int] = None) -> None:
        """
        Plan production using a Just-In-Time (JIT) approach.
        Attempts to schedule production as late as possible while still meeting demand.
        
        Args:
            up_to_day: Optional maximum day to plan for (defaults to max_simulation_day)
        """
        if up_to_day is None:
            up_to_day = self.max_simulation_day
            
        # Clear existing production plan
        self.production_plan = {}
        
        # Get current inventory levels
        current_raw = sum(batch.remaining_quantity for batch in self.raw_material_batches)
        current_product = sum(batch.remaining_quantity for batch in self.product_batches)
        
        # Organize future raw deliveries by day
        raw_deliveries_by_day = {}
        for contract in self.pending_supply_contracts:
            if contract.delivery_time < self.current_day:
                continue
            raw_deliveries_by_day[contract.delivery_time] = raw_deliveries_by_day.get(contract.delivery_time, 0) + contract.quantity
            
        # Organize demand by day
        demands_by_day = {}
        for contract in self.pending_demand_contracts:
            if contract.delivery_time < self.current_day:
                continue
            demands_by_day[contract.delivery_time] = demands_by_day.get(contract.delivery_time, 0) + contract.quantity
            
        # Sort demand days in descending order (plan for latest demands first)
        demand_days = sorted(demands_by_day.keys(), reverse=True)
        
        # Track available inventory through the planning process
        available_raw = current_raw
        available_product = current_product
        
        # For each demand day, plan production
        for demand_delivery_day in demand_days:
            if demand_delivery_day < self.current_day:
                continue
                
            # How much do we need to deliver on this day?
            demand_quantity = demands_by_day[demand_delivery_day]
            
            # If we already have enough product, no need to produce more
            if available_product >= demand_quantity:
                available_product -= demand_quantity
                continue
                
            # Otherwise, we need to produce the difference
            remaining_to_plan_for_this_demand = demand_quantity - available_product
            available_product = 0  # We've allocated all available product
            
            # Plan production in reverse, starting from the day before delivery
            planning_day = demand_delivery_day - 1
            
            # Track reasons for planning failures
            planning_bottlenecks = []
            
            while remaining_to_plan_for_this_demand > 0 and planning_day >= self.current_day:
                # Check if we have capacity on this day
                day_capacity = self.daily_production_capacity
                already_planned = self.production_plan.get(planning_day, 0)
                available_capacity = max(0, day_capacity - already_planned)
                
                if available_capacity <= 0:
                    planning_bottlenecks.append(f"No capacity on day {planning_day}")
                    planning_day -= 1
                    continue
                    
                # Calculate raw materials available by this day
                raw_available_by_day = available_raw
                for day in range(self.current_day, planning_day + 1):
                    raw_available_by_day += raw_deliveries_by_day.get(day, 0)
                    
                # Subtract raw used in production before this day
                for day in range(self.current_day, planning_day):
                    raw_available_by_day -= self.production_plan.get(day, 0)
                    
                if raw_available_by_day <= 0:
                    planning_bottlenecks.append(f"No raw materials on day {planning_day}")
                    planning_day -= 1
                    continue
                    
                # Determine how much we can produce on this day
                can_produce = min(available_capacity, raw_available_by_day, remaining_to_plan_for_this_demand)
                
                if can_produce <= 0:
                    planning_bottlenecks.append(f"Cannot produce on day {planning_day}")
                    planning_day -= 1
                    continue
                    
                # Update production plan
                self.production_plan[planning_day] = self.production_plan.get(planning_day, 0) + can_produce
                
                # Update tracking variables
                remaining_to_plan_for_this_demand -= can_produce
                
                # Move to previous day
                planning_day -= 1
                
            # If we couldn't plan for all demand, log a warning
            if remaining_to_plan_for_this_demand > 0:
                # Create a summary of bottlenecks
                reason_details = set(planning_bottlenecks)
                reason_details_str = ", ".join(reason_details)
                
                if reason_details_str:
                    reason_str = f"Primary bottlenecks encountered: [{reason_details_str}]"
                else:
                    # This case should be rare if the demand was positive and planning was attempted.
                    # It might indicate no valid production days or an unexpected state.
                    reason_str = "Unable to schedule due to JIT window constraints (e.g., no available capacity or raw materials on any considered day)."

    def get_available_production_capacity(self, day: int) -> int:
        if day < self.current_day: # Cannot produce in the past
            return 0
        # Considers self.daily_production_capacity and self.production_plan for that day
        planned_production_for_day = self.production_plan.get(day, 0)
        return max(0, self.daily_production_capacity - planned_production_for_day)

# Example usage (optional, for testing during development)
if __name__ == '__main__':
    # Initialize CustomInventoryManager
    cim = InventoryManagerCIRS(
        raw_storage_cost=0.01, 
        product_storage_cost=0.02,
        processing_cost=2.0,
        daily_production_capacity=30.0,
        max_simulation_day=10,
        current_day=0
    )