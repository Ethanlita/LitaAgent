"""
Error Detectors Module

Provides a framework for detecting various failure patterns in SCML agent behavior.

Each detector identifies a specific type of failure:
- OverpricingDetector: Detects when agents price too high, causing negotiation failures
- UnderpricingDetector: Detects when agents price too low, missing opportunities
- InventoryStarvationDetector: Detects violations due to insufficient inventory
- ProductionIdleDetector: Detects underutilization of production capacity
- LossContractDetector: Detects contracts that result in losses
- NegotiationStallDetector: Detects agents with rigid negotiation strategies
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .log_parser import AgentData, ContractRecord, NegotiationRecord, SimulationData


@dataclass
class Issue:
    """Represents a detected failure instance."""
    error_type: str
    agent_name: str
    round: int  # Simulation step when the issue occurred
    severity: str = "medium"  # 'low', 'medium', 'high', 'critical'
    details: Dict[str, Any] = field(default_factory=dict)
    message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "error_type": self.error_type,
            "agent_name": self.agent_name,
            "round": self.round,
            "severity": self.severity,
            "message": self.message,
            **self.details
        }


class BaseErrorDetector(ABC):
    """
    Abstract base class for error detectors.
    
    Each detector identifies a specific type of failure pattern.
    Subclasses must implement the detect() method.
    """
    
    name: str = "BaseError"
    description: str = "Base error detector"
    
    def __init__(self, **params):
        """
        Initialize detector with configurable parameters.
        
        Args:
            **params: Configuration parameters for the detector
        """
        self.params = params
    
    @abstractmethod
    def detect(self, agent_data: AgentData, simulation_data: SimulationData) -> List[Issue]:
        """
        Detect failures for a specific agent.
        
        Args:
            agent_data: Data for the agent being analyzed
            simulation_data: Full simulation data for context
            
        Returns:
            List of Issue objects representing detected failures
        """
        pass
    
    def get_param(self, key: str, default: Any = None) -> Any:
        """Get a parameter value with fallback to default."""
        return self.params.get(key, default)


class OverpricingDetector(BaseErrorDetector):
    """
    Detects when an agent's prices are too high, causing negotiation failures.
    
    Trigger conditions:
    - Agent is seller and final offer price significantly exceeds opponent's last offer
    - Negotiation fails due to price disagreement
    
    Parameters:
        excess_price_ratio (float): Threshold for excess pricing (default: 0.2 = 20%)
        min_price_diff (float): Minimum absolute price difference to consider (default: 1.0)
    """
    
    name = "Overpricing"
    description = "报价过高导致交易失败"
    
    def __init__(self, excess_price_ratio: float = 0.2, min_price_diff: float = 1.0, **params):
        super().__init__(**params)
        self.params['excess_price_ratio'] = excess_price_ratio
        self.params['min_price_diff'] = min_price_diff
    
    def detect(self, agent_data: AgentData, simulation_data: SimulationData) -> List[Issue]:
        issues = []
        
        for neg in agent_data.failed_negotiations:
            # Get agent's role in negotiation
            is_seller = neg.seller == agent_data.name
            
            # Only check overpricing for sellers
            if not is_seller:
                continue
            
            # Get offers from both parties
            my_offers = neg.offers.get(agent_data.name, [])
            opponent_name = neg.buyer if is_seller else neg.seller
            opp_offers = neg.offers.get(opponent_name, [])
            
            if not my_offers or not opp_offers:
                continue
            
            # Get final offers (price is typically the last element in offer tuple)
            my_final_price = my_offers[-1][-1] if my_offers[-1] else None
            opp_final_price = opp_offers[-1][-1] if opp_offers[-1] else None
            
            if my_final_price is None or opp_final_price is None:
                continue
            
            # Check if seller's price is significantly higher
            price_diff = my_final_price - opp_final_price
            price_ratio = (my_final_price / opp_final_price - 1) if opp_final_price > 0 else 0
            
            excess_ratio = self.get_param('excess_price_ratio', 0.2)
            min_diff = self.get_param('min_price_diff', 1.0)
            
            if price_ratio > excess_ratio and price_diff > min_diff:
                # Determine severity based on price difference
                if price_ratio > 0.5:
                    severity = "high"
                elif price_ratio > 0.3:
                    severity = "medium"
                else:
                    severity = "low"
                
                issues.append(Issue(
                    error_type=self.name,
                    agent_name=agent_data.name,
                    round=neg.ended_at,
                    severity=severity,
                    details={
                        "negotiation_id": neg.id,
                        "my_final_price": my_final_price,
                        "opp_final_price": opp_final_price,
                        "difference_pct": round(price_ratio * 100, 2),
                        "price_diff": round(price_diff, 2),
                        "opponent": opponent_name,
                        "product": neg.product,
                    },
                    message=f"报价高出对方{round(price_ratio * 100, 1)}%，交易未达成"
                ))
        
        return issues


class UnderpricingDetector(BaseErrorDetector):
    """
    Detects when an agent's prices are too low as a buyer, missing opportunities.
    
    Trigger conditions:
    - Agent is buyer and final offer price significantly below opponent's last offer
    - Negotiation fails due to price disagreement
    
    Parameters:
        undercut_price_ratio (float): Threshold for undercutting (default: 0.2 = 20%)
    """
    
    name = "Underpricing"
    description = "出价过低导致交易失败"
    
    def __init__(self, undercut_price_ratio: float = 0.2, **params):
        super().__init__(**params)
        self.params['undercut_price_ratio'] = undercut_price_ratio
    
    def detect(self, agent_data: AgentData, simulation_data: SimulationData) -> List[Issue]:
        issues = []
        
        for neg in agent_data.failed_negotiations:
            # Get agent's role - only check underpricing for buyers
            is_buyer = neg.buyer == agent_data.name
            if not is_buyer:
                continue
            
            # Get offers
            my_offers = neg.offers.get(agent_data.name, [])
            opponent_name = neg.seller
            opp_offers = neg.offers.get(opponent_name, [])
            
            if not my_offers or not opp_offers:
                continue
            
            # Get final prices
            my_final_price = my_offers[-1][-1] if my_offers[-1] else None
            opp_final_price = opp_offers[-1][-1] if opp_offers[-1] else None
            
            if my_final_price is None or opp_final_price is None:
                continue
            
            # Check if buyer's price is significantly lower
            price_ratio = (opp_final_price / my_final_price - 1) if my_final_price > 0 else 0
            
            undercut_ratio = self.get_param('undercut_price_ratio', 0.2)
            
            if price_ratio > undercut_ratio:
                severity = "high" if price_ratio > 0.4 else "medium" if price_ratio > 0.25 else "low"
                
                issues.append(Issue(
                    error_type=self.name,
                    agent_name=agent_data.name,
                    round=neg.ended_at,
                    severity=severity,
                    details={
                        "negotiation_id": neg.id,
                        "my_final_price": my_final_price,
                        "opp_final_price": opp_final_price,
                        "difference_pct": round(price_ratio * 100, 2),
                        "opponent": opponent_name,
                        "product": neg.product,
                    },
                    message=f"出价低于对方{round(price_ratio * 100, 1)}%，未能达成交易"
                ))
        
        return issues


class InventoryStarvationDetector(BaseErrorDetector):
    """
    Detects when an agent fails to fulfill contracts due to insufficient inventory.
    
    Trigger conditions:
    - Contract breach occurred
    - At delivery time, agent's inventory was insufficient
    
    Parameters:
        check_pre_delivery_steps (int): Steps before delivery to check inventory (default: 3)
    """
    
    name = "InventoryStarvation"
    description = "库存不足导致违约"
    
    def __init__(self, check_pre_delivery_steps: int = 3, **params):
        super().__init__(**params)
        self.params['check_pre_delivery_steps'] = check_pre_delivery_steps
    
    def detect(self, agent_data: AgentData, simulation_data: SimulationData) -> List[Issue]:
        issues = []
        
        # Check breached contracts where agent is seller
        for contract in agent_data.contracts_as_seller:
            if not contract.is_breached:
                continue
            
            # Get inventory at delivery time
            delivery_step = contract.delivery_time
            inventory_at_delivery = self._get_inventory_at_step(
                agent_data, delivery_step, 'output'
            )
            
            # Calculate shortfall
            shortfall = contract.quantity - inventory_at_delivery
            
            if shortfall > 0:
                # Check penalty from stats
                penalty = self._get_shortfall_penalty_at_step(agent_data, delivery_step)
                
                severity = "critical" if shortfall >= contract.quantity * 0.5 else "high" if shortfall >= contract.quantity * 0.25 else "medium"
                
                issues.append(Issue(
                    error_type=self.name,
                    agent_name=agent_data.name,
                    round=delivery_step,
                    severity=severity,
                    details={
                        "contract_id": contract.id,
                        "promised_qty": contract.quantity,
                        "delivered_qty": max(0, contract.quantity - shortfall),
                        "shortfall": shortfall,
                        "inventory_at_delivery": inventory_at_delivery,
                        "penalty": penalty,
                        "buyer": contract.buyer_name,
                        "product": contract.product_name,
                    },
                    message=f"库存不足导致违约，缺口{shortfall}单位"
                ))
        
        return issues
    
    def _get_inventory_at_step(self, agent_data: AgentData, step: int, inv_type: str = 'output') -> int:
        """Get inventory level at a specific step."""
        if agent_data.stats is None:
            return 0
        
        inventory_list = agent_data.stats.inventory_output if inv_type == 'output' else agent_data.stats.inventory_input
        
        if step < len(inventory_list):
            return inventory_list[step]
        return 0
    
    def _get_shortfall_penalty_at_step(self, agent_data: AgentData, step: int) -> float:
        """Get shortfall penalty at a specific step."""
        if agent_data.stats is None:
            return 0.0
        
        if step < len(agent_data.stats.shortfall_penalty):
            return agent_data.stats.shortfall_penalty[step]
        return 0.0


class ProductionIdleDetector(BaseErrorDetector):
    """
    Detects when an agent's production line is idle despite having resources.
    
    Trigger conditions:
    - Production utilization is below threshold for consecutive steps
    - Agent has raw materials but isn't producing
    
    Parameters:
        idle_steps_threshold (int): Consecutive idle steps to trigger (default: 3)
        utilization_threshold (float): Below this utilization is considered idle (default: 0.2)
    """
    
    name = "ProductionIdle"
    description = "生产线闲置，产能未充分利用"
    
    def __init__(self, idle_steps_threshold: int = 3, utilization_threshold: float = 0.2, **params):
        super().__init__(**params)
        self.params['idle_steps_threshold'] = idle_steps_threshold
        self.params['utilization_threshold'] = utilization_threshold
    
    def detect(self, agent_data: AgentData, simulation_data: SimulationData) -> List[Issue]:
        issues = []
        
        if agent_data.stats is None:
            return issues
        
        productivity = agent_data.stats.productivity
        inventory_input = agent_data.stats.inventory_input
        shortfall_qty = agent_data.stats.shortfall_quantity
        
        idle_threshold = self.get_param('utilization_threshold', 0.2)
        min_idle_steps = self.get_param('idle_steps_threshold', 3)
        
        # Find consecutive idle periods
        idle_start = None
        idle_count = 0
        
        for step in range(len(productivity)):
            is_idle = productivity[step] < idle_threshold
            has_input = inventory_input[step] > 0 if step < len(inventory_input) else False
            has_demand = shortfall_qty[step] > 0 if step < len(shortfall_qty) else False
            
            if is_idle and (has_input or has_demand):
                if idle_start is None:
                    idle_start = step
                idle_count += 1
            else:
                # Check if we have a significant idle period
                if idle_count >= min_idle_steps:
                    avg_input = np.mean(inventory_input[idle_start:idle_start + idle_count]) if idle_start is not None else 0
                    total_shortfall = sum(shortfall_qty[idle_start:idle_start + idle_count]) if idle_start is not None else 0
                    
                    severity = "high" if idle_count >= 5 else "medium"
                    
                    issues.append(Issue(
                        error_type=self.name,
                        agent_name=agent_data.name,
                        round=idle_start,
                        severity=severity,
                        details={
                            "idle_start_round": idle_start,
                            "idle_end_round": idle_start + idle_count - 1,
                            "idle_duration": idle_count,
                            "avg_input_inventory": round(avg_input, 2),
                            "total_unmet_demand": total_shortfall,
                            "avg_productivity": round(np.mean(productivity[idle_start:idle_start + idle_count]), 3),
                        },
                        message=f"连续{idle_count}回合生产闲置，有原料但未充分生产"
                    ))
                
                idle_start = None
                idle_count = 0
        
        # Check final period
        if idle_count >= min_idle_steps and idle_start is not None:
            avg_input = np.mean(inventory_input[idle_start:idle_start + idle_count])
            total_shortfall = sum(shortfall_qty[idle_start:idle_start + idle_count])
            
            issues.append(Issue(
                error_type=self.name,
                agent_name=agent_data.name,
                round=idle_start,
                severity="medium",
                details={
                    "idle_start_round": idle_start,
                    "idle_end_round": idle_start + idle_count - 1,
                    "idle_duration": idle_count,
                    "avg_input_inventory": round(avg_input, 2),
                    "total_unmet_demand": total_shortfall,
                },
                message=f"连续{idle_count}回合生产闲置"
            ))
        
        return issues


class LossContractDetector(BaseErrorDetector):
    """
    Detects contracts that result in losses for the agent.
    
    Trigger conditions:
    - Contract profit margin is negative or below threshold
    
    Parameters:
        min_profit_margin (float): Minimum acceptable profit margin (default: 0.0)
        use_market_price (bool): Compare against market price (default: True)
    """
    
    name = "LossContract"
    description = "签订了亏本合约"
    
    def __init__(self, min_profit_margin: float = 0.0, use_market_price: bool = True, **params):
        super().__init__(**params)
        self.params['min_profit_margin'] = min_profit_margin
        self.params['use_market_price'] = use_market_price
    
    def detect(self, agent_data: AgentData, simulation_data: SimulationData) -> List[Issue]:
        issues = []
        
        market_prices = simulation_data.market_stats.average_prices if simulation_data.market_stats else {}
        
        # Check contracts where agent is seller
        for contract in agent_data.contracts_as_seller:
            product = contract.product_name
            market_price = market_prices.get(product.replace('p', ''), contract.unit_price)
            
            # Estimate if selling below market
            if market_price > 0:
                price_ratio = contract.unit_price / market_price
                min_margin = self.get_param('min_profit_margin', 0.0)
                
                if price_ratio < (1 - min_margin):
                    loss_pct = (1 - price_ratio) * 100
                    severity = "high" if loss_pct > 20 else "medium" if loss_pct > 10 else "low"
                    
                    issues.append(Issue(
                        error_type=self.name,
                        agent_name=agent_data.name,
                        round=contract.signed_at,
                        severity=severity,
                        details={
                            "contract_id": contract.id,
                            "unit_price": contract.unit_price,
                            "market_price": round(market_price, 2),
                            "quantity": contract.quantity,
                            "total_value": contract.total_value,
                            "estimated_loss": round((market_price - contract.unit_price) * contract.quantity, 2),
                            "loss_pct": round(loss_pct, 2),
                            "buyer": contract.buyer_name,
                            "product": product,
                        },
                        message=f"卖价低于市场均价{round(loss_pct, 1)}%"
                    ))
        
        # Check contracts where agent is buyer
        for contract in agent_data.contracts_as_buyer:
            product = contract.product_name
            market_price = market_prices.get(product.replace('p', ''), contract.unit_price)
            
            if market_price > 0:
                price_ratio = contract.unit_price / market_price
                min_margin = self.get_param('min_profit_margin', 0.0)
                
                if price_ratio > (1 + min_margin):
                    overpay_pct = (price_ratio - 1) * 100
                    severity = "high" if overpay_pct > 20 else "medium" if overpay_pct > 10 else "low"
                    
                    issues.append(Issue(
                        error_type=self.name,
                        agent_name=agent_data.name,
                        round=contract.signed_at,
                        severity=severity,
                        details={
                            "contract_id": contract.id,
                            "unit_price": contract.unit_price,
                            "market_price": round(market_price, 2),
                            "quantity": contract.quantity,
                            "total_value": contract.total_value,
                            "overpay_amount": round((contract.unit_price - market_price) * contract.quantity, 2),
                            "overpay_pct": round(overpay_pct, 2),
                            "seller": contract.seller_name,
                            "product": product,
                        },
                        message=f"买价高于市场均价{round(overpay_pct, 1)}%"
                    ))
        
        return issues


class NegotiationStallDetector(BaseErrorDetector):
    """
    Detects agents with rigid negotiation strategies leading to high failure rates.
    
    Trigger conditions:
    - Negotiation failure rate exceeds threshold
    - Agent shows low concession behavior
    
    Parameters:
        failure_rate_threshold (float): Max acceptable failure rate (default: 0.7)
        min_negotiations (int): Minimum negotiations to analyze (default: 5)
    """
    
    name = "NegotiationStall"
    description = "谈判策略僵化导致高失败率"
    
    def __init__(self, failure_rate_threshold: float = 0.7, min_negotiations: int = 5, **params):
        super().__init__(**params)
        self.params['failure_rate_threshold'] = failure_rate_threshold
        self.params['min_negotiations'] = min_negotiations
    
    def detect(self, agent_data: AgentData, simulation_data: SimulationData) -> List[Issue]:
        issues = []
        
        min_negs = self.get_param('min_negotiations', 5)
        if len(agent_data.negotiations) < min_negs:
            return issues
        
        failure_rate = 1 - agent_data.negotiation_success_rate
        threshold = self.get_param('failure_rate_threshold', 0.7)
        
        if failure_rate > threshold:
            # Analyze concession patterns
            avg_concession = self._calculate_avg_concession(agent_data)
            failed_neg_ids = [n.id for n in agent_data.failed_negotiations[:10]]  # Sample
            
            severity = "high" if failure_rate > 0.85 else "medium"
            
            issues.append(Issue(
                error_type=self.name,
                agent_name=agent_data.name,
                round=0,  # Overall issue, not specific to a round
                severity=severity,
                details={
                    "total_negotiations": len(agent_data.negotiations),
                    "failed_negotiations": len(agent_data.failed_negotiations),
                    "failure_rate": round(failure_rate, 3),
                    "avg_concession_rate": round(avg_concession, 3),
                    "sample_failed_neg_ids": failed_neg_ids,
                },
                message=f"谈判失败率{round(failure_rate * 100, 1)}%，策略可能过于僵化"
            ))
        
        return issues
    
    def _calculate_avg_concession(self, agent_data: AgentData) -> float:
        """Calculate average concession rate in negotiations."""
        concessions = []
        
        for neg in agent_data.negotiations:
            offers = neg.offers.get(agent_data.name, [])
            if len(offers) < 2:
                continue
            
            # Check price changes (assuming price is last element)
            prices = [o[-1] for o in offers if o]
            if len(prices) < 2:
                continue
            
            # Calculate total concession
            initial_price = prices[0]
            final_price = prices[-1]
            if initial_price > 0:
                concession = abs(final_price - initial_price) / initial_price
                concessions.append(concession)
        
        return np.mean(concessions) if concessions else 0.0


# Registry of all available detectors
DETECTOR_REGISTRY = {
    "overpricing": OverpricingDetector,
    "underpricing": UnderpricingDetector,
    "inventory_starvation": InventoryStarvationDetector,
    "production_idle": ProductionIdleDetector,
    "loss_contract": LossContractDetector,
    "negotiation_stall": NegotiationStallDetector,
}


def get_all_detectors(**params) -> List[BaseErrorDetector]:
    """
    Get instances of all available detectors.
    
    Args:
        **params: Parameters passed to all detectors
        
    Returns:
        List of detector instances
    """
    return [cls(**params) for cls in DETECTOR_REGISTRY.values()]


def get_detector(name: str, **params) -> BaseErrorDetector:
    """
    Get a specific detector by name.
    
    Args:
        name: Detector name (e.g., 'overpricing')
        **params: Parameters for the detector
        
    Returns:
        Detector instance
    """
    if name not in DETECTOR_REGISTRY:
        raise ValueError(f"Unknown detector: {name}. Available: {list(DETECTOR_REGISTRY.keys())}")
    return DETECTOR_REGISTRY[name](**params)
