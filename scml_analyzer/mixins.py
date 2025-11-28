"""
Agent Logging Enhancement Mixins

Provides mixins and decorators for agents to record detailed decision logs.
These logs complement the simulation-level logs with agent-internal information.
"""

import functools
import json
import logging
import os
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

# Type for decorated functions
F = TypeVar('F', bound=Callable[..., Any])


class AgentLogger:
    """
    Centralized logger for agent decisions.
    
    Handles writing decision logs to files and provides structured logging.
    """
    
    def __init__(
        self,
        agent_name: str,
        log_dir: Optional[str] = None,
        log_to_file: bool = True,
        log_to_console: bool = False,
    ):
        """
        Initialize agent logger.
        
        Args:
            agent_name: Name of the agent
            log_dir: Directory for log files
            log_to_file: Whether to write to file
            log_to_console: Whether to print to console
        """
        self.agent_name = agent_name
        self.log_dir = log_dir or "./agent_logs"
        self.log_to_file = log_to_file
        self.log_to_console = log_to_console
        
        self.logs: List[Dict] = []
        self._file_handle = None
        
        if self.log_to_file:
            os.makedirs(self.log_dir, exist_ok=True)
            log_path = os.path.join(self.log_dir, f"{agent_name}_decisions.jsonl")
            self._file_handle = open(log_path, 'w', encoding='utf-8')
    
    def log(
        self,
        event_type: str,
        step: int,
        data: Dict[str, Any],
        negotiation_id: Optional[str] = None,
        contract_id: Optional[str] = None,
    ):
        """
        Log a decision event.
        
        Args:
            event_type: Type of event (e.g., 'PROPOSE_OFFER', 'RESPOND', 'SIGN_CONTRACT')
            step: Current simulation step
            data: Event-specific data
            negotiation_id: Related negotiation ID if any
            contract_id: Related contract ID if any
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            "agent": self.agent_name,
            "step": step,
            "event_type": event_type,
            "negotiation_id": negotiation_id,
            "contract_id": contract_id,
            **data
        }
        
        self.logs.append(record)
        
        if self.log_to_file and self._file_handle:
            self._file_handle.write(json.dumps(record, ensure_ascii=False) + '\n')
            self._file_handle.flush()
        
        if self.log_to_console:
            print(f"[{self.agent_name}] {event_type}: {data}")
    
    def log_offer(
        self,
        step: int,
        negotiation_id: str,
        action: str,  # 'propose', 'accept', 'reject', 'end'
        offer: Optional[tuple] = None,
        opponent: Optional[str] = None,
        is_buy: bool = False,
        inventory: Optional[Dict] = None,
        reasoning: Optional[str] = None,
        **extra
    ):
        """
        Log a negotiation offer action.
        
        Args:
            step: Simulation step
            negotiation_id: Negotiation ID
            action: Action taken
            offer: Offer tuple (quantity, time, price)
            opponent: Opponent name
            is_buy: Whether this is a buy negotiation
            inventory: Current inventory state
            reasoning: Decision reasoning
            **extra: Additional fields
        """
        data = {
            "action": action,
            "offer": offer,
            "opponent": opponent,
            "is_buy": is_buy,
            "inventory": inventory,
            "reasoning": reasoning,
            **extra
        }
        self.log("NEGOTIATION_OFFER", step, data, negotiation_id=negotiation_id)
    
    def log_contract_decision(
        self,
        step: int,
        contract_id: str,
        decision: str,  # 'sign', 'reject'
        contract_details: Dict,
        reasoning: Optional[str] = None,
        **extra
    ):
        """
        Log a contract signing decision.
        
        Args:
            step: Simulation step
            contract_id: Contract ID
            decision: Decision made
            contract_details: Contract information
            reasoning: Decision reasoning
            **extra: Additional fields
        """
        data = {
            "decision": decision,
            "contract": contract_details,
            "reasoning": reasoning,
            **extra
        }
        self.log("CONTRACT_DECISION", step, data, contract_id=contract_id)
    
    def log_production_decision(
        self,
        step: int,
        production_quantity: int,
        available_capacity: int,
        raw_materials: int,
        pending_orders: int,
        reasoning: Optional[str] = None,
        **extra
    ):
        """
        Log a production decision.
        
        Args:
            step: Simulation step
            production_quantity: Quantity to produce
            available_capacity: Available production capacity
            raw_materials: Available raw materials
            pending_orders: Pending delivery orders
            reasoning: Decision reasoning
            **extra: Additional fields
        """
        data = {
            "production_quantity": production_quantity,
            "available_capacity": available_capacity,
            "raw_materials": raw_materials,
            "pending_orders": pending_orders,
            "reasoning": reasoning,
            **extra
        }
        self.log("PRODUCTION_DECISION", step, data)
    
    def log_state(
        self,
        step: int,
        balance: float,
        inventory_input: int,
        inventory_output: int,
        pending_buy_contracts: int,
        pending_sell_contracts: int,
        **extra
    ):
        """
        Log agent state at a given step.
        
        Args:
            step: Simulation step
            balance: Current balance
            inventory_input: Input inventory level
            inventory_output: Output inventory level
            pending_buy_contracts: Number of pending buy contracts
            pending_sell_contracts: Number of pending sell contracts
            **extra: Additional fields
        """
        data = {
            "balance": balance,
            "inventory_input": inventory_input,
            "inventory_output": inventory_output,
            "pending_buy_contracts": pending_buy_contracts,
            "pending_sell_contracts": pending_sell_contracts,
            **extra
        }
        self.log("AGENT_STATE", step, data)
    
    def close(self):
        """Close log file."""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None
    
    def __del__(self):
        self.close()


class LoggingNegotiatorMixin:
    """
    Mixin class to add logging capabilities to SCML agents.
    
    Add this mixin to your agent class to automatically log decisions.
    
    Usage:
        class MyAgent(OneShotAgent, LoggingNegotiatorMixin):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.init_logging()  # Initialize logging
            
            def propose(self, negotiator_id, state):
                offer = self._calculate_offer(state)
                self.log_propose(negotiator_id, state, offer)
                return offer
    """
    
    _agent_logger: Optional[AgentLogger] = None
    _log_enabled: bool = True
    _log_dir: str = "./agent_logs"
    
    def init_logging(
        self,
        log_dir: str = "./agent_logs",
        log_to_file: bool = True,
        log_to_console: bool = False,
        enabled: bool = True,
    ):
        """
        Initialize logging for this agent.
        
        Call this in your agent's __init__ after calling super().__init__().
        
        Args:
            log_dir: Directory for log files
            log_to_file: Whether to write to file
            log_to_console: Whether to print to console
            enabled: Whether logging is enabled
        """
        self._log_enabled = enabled
        self._log_dir = log_dir
        
        if enabled:
            agent_name = getattr(self, 'name', getattr(self, 'id', 'unknown_agent'))
            self._agent_logger = AgentLogger(
                agent_name=agent_name,
                log_dir=log_dir,
                log_to_file=log_to_file,
                log_to_console=log_to_console,
            )
    
    @property
    def logger(self) -> Optional[AgentLogger]:
        """Get the agent logger."""
        return self._agent_logger
    
    def log_propose(
        self,
        negotiator_id: str,
        state: Any,
        offer: Optional[tuple],
        reasoning: Optional[str] = None,
        **extra
    ):
        """Log a propose action."""
        if not self._log_enabled or not self._agent_logger:
            return
        
        # Get current step (try different attribute names)
        step = getattr(self, 'current_step', getattr(self, 'awi', {}).get('current_step', 0))
        if hasattr(step, '__call__'):
            step = step()
        
        # Get opponent info from state if available
        opponent = None
        if hasattr(state, 'partners'):
            partners = list(state.partners) if hasattr(state.partners, '__iter__') else []
            agent_name = getattr(self, 'name', '')
            opponent = next((p for p in partners if p != agent_name), None)
        
        # Get inventory if available
        inventory = None
        if hasattr(self, 'awi'):
            awi = self.awi
            if hasattr(awi, 'current_inventory_input') and hasattr(awi, 'current_inventory_output'):
                inventory = {
                    'input': awi.current_inventory_input,
                    'output': awi.current_inventory_output,
                }
        
        self._agent_logger.log_offer(
            step=step,
            negotiation_id=negotiator_id,
            action='propose',
            offer=offer,
            opponent=opponent,
            inventory=inventory,
            reasoning=reasoning,
            **extra
        )
    
    def log_respond(
        self,
        negotiator_id: str,
        state: Any,
        response: str,  # 'accept', 'reject', 'end'
        offer: Optional[tuple] = None,
        counter_offer: Optional[tuple] = None,
        reasoning: Optional[str] = None,
        **extra
    ):
        """Log a respond action."""
        if not self._log_enabled or not self._agent_logger:
            return
        
        step = getattr(self, 'current_step', 0)
        if hasattr(step, '__call__'):
            step = step()
        
        self._agent_logger.log_offer(
            step=step,
            negotiation_id=negotiator_id,
            action=response,
            offer=offer if response == 'accept' else counter_offer,
            reasoning=reasoning,
            received_offer=offer,
            **extra
        )
    
    def log_contract_signed(
        self,
        contract: Any,
        reasoning: Optional[str] = None,
        **extra
    ):
        """Log contract signing."""
        if not self._log_enabled or not self._agent_logger:
            return
        
        step = getattr(self, 'current_step', 0)
        if hasattr(step, '__call__'):
            step = step()
        
        contract_id = getattr(contract, 'id', str(contract))
        
        # Extract contract details
        contract_details = {}
        for attr in ['quantity', 'unit_price', 'delivery_time', 'product']:
            if hasattr(contract, attr):
                contract_details[attr] = getattr(contract, attr)
        
        self._agent_logger.log_contract_decision(
            step=step,
            contract_id=contract_id,
            decision='sign',
            contract_details=contract_details,
            reasoning=reasoning,
            **extra
        )
    
    def log_step_state(self, **extra):
        """Log agent state at current step."""
        if not self._log_enabled or not self._agent_logger:
            return
        
        step = getattr(self, 'current_step', 0)
        if hasattr(step, '__call__'):
            step = step()
        
        # Try to get state from AWI
        balance = 0
        inv_input = 0
        inv_output = 0
        pending_buy = 0
        pending_sell = 0
        
        if hasattr(self, 'awi'):
            awi = self.awi
            balance = getattr(awi, 'current_balance', 0)
            inv_input = getattr(awi, 'current_inventory_input', 0)
            inv_output = getattr(awi, 'current_inventory_output', 0)
            
            # Count pending contracts if available
            if hasattr(awi, 'current_buy_contracts'):
                pending_buy = len(awi.current_buy_contracts)
            if hasattr(awi, 'current_sell_contracts'):
                pending_sell = len(awi.current_sell_contracts)
        
        self._agent_logger.log_state(
            step=step,
            balance=balance,
            inventory_input=inv_input,
            inventory_output=inv_output,
            pending_buy_contracts=pending_buy,
            pending_sell_contracts=pending_sell,
            **extra
        )
    
    def finalize_logging(self):
        """Close logging. Call this when agent is done."""
        if self._agent_logger:
            self._agent_logger.close()


def log_decision(
    event_type: str = "DECISION",
    include_args: bool = True,
    include_result: bool = True,
) -> Callable[[F], F]:
    """
    Decorator to automatically log function calls as decisions.
    
    Usage:
        class MyAgent(OneShotAgent, LoggingNegotiatorMixin):
            @log_decision(event_type="PROPOSE")
            def propose(self, negotiator_id, state):
                return calculated_offer
    
    Args:
        event_type: Type of event to log
        include_args: Whether to include function arguments in log
        include_result: Whether to include return value in log
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Execute the function
            result = func(self, *args, **kwargs)
            
            # Log if logging is enabled
            if hasattr(self, '_agent_logger') and self._agent_logger and getattr(self, '_log_enabled', True):
                step = getattr(self, 'current_step', 0)
                if hasattr(step, '__call__'):
                    step = step()
                
                log_data = {
                    "function": func.__name__,
                }
                
                if include_args:
                    # Convert args to serializable format
                    log_data["args"] = [str(a)[:100] for a in args]
                    log_data["kwargs"] = {k: str(v)[:100] for k, v in kwargs.items()}
                
                if include_result:
                    log_data["result"] = str(result)[:200] if result is not None else None
                
                self._agent_logger.log(event_type, step, log_data)
            
            return result
        return wrapper  # type: ignore
    return decorator


class NegotiationLogger:
    """
    Context manager for logging a negotiation session.
    
    Usage:
        with NegotiationLogger(self, negotiation_id, partner) as nl:
            # ... negotiation logic ...
            nl.log_offer(offer)
            nl.log_response(response)
    """
    
    def __init__(
        self,
        agent: LoggingNegotiatorMixin,
        negotiation_id: str,
        partner: str,
        is_buy: bool = False,
    ):
        self.agent = agent
        self.negotiation_id = negotiation_id
        self.partner = partner
        self.is_buy = is_buy
        self.start_step = None
        self.offers_made = []
        self.responses = []
    
    def __enter__(self):
        if hasattr(self.agent, 'current_step'):
            self.start_step = self.agent.current_step
        
        if hasattr(self.agent, '_agent_logger') and self.agent._agent_logger:
            self.agent._agent_logger.log(
                "NEGOTIATION_START",
                self.start_step or 0,
                {
                    "partner": self.partner,
                    "is_buy": self.is_buy,
                },
                negotiation_id=self.negotiation_id,
            )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_step = getattr(self.agent, 'current_step', self.start_step or 0)
        
        if hasattr(self.agent, '_agent_logger') and self.agent._agent_logger:
            self.agent._agent_logger.log(
                "NEGOTIATION_END",
                end_step,
                {
                    "partner": self.partner,
                    "is_buy": self.is_buy,
                    "n_offers": len(self.offers_made),
                    "error": str(exc_val) if exc_val else None,
                },
                negotiation_id=self.negotiation_id,
            )
        return False
    
    def log_offer(self, offer: tuple, reasoning: Optional[str] = None):
        """Log an offer made during this negotiation."""
        self.offers_made.append(offer)
        if hasattr(self.agent, 'log_propose'):
            self.agent.log_propose(
                self.negotiation_id,
                None,  # state
                offer,
                reasoning=reasoning,
            )
    
    def log_response(self, response: str, offer: Optional[tuple] = None):
        """Log a response made during this negotiation."""
        self.responses.append((response, offer))
        if hasattr(self.agent, 'log_respond'):
            self.agent.log_respond(
                self.negotiation_id,
                None,  # state
                response,
                offer=offer,
            )


def load_agent_logs(log_dir: str, agent_name: Optional[str] = None) -> Dict[str, List[Dict]]:
    """
    Load agent decision logs from directory.
    
    Args:
        log_dir: Directory containing log files
        agent_name: Specific agent name to load (None = all agents)
        
    Returns:
        Dictionary mapping agent names to their log entries
    """
    logs = {}
    log_path = Path(log_dir)
    
    if not log_path.exists():
        return logs
    
    pattern = f"{agent_name}_decisions.jsonl" if agent_name else "*_decisions.jsonl"
    
    for log_file in log_path.glob(pattern):
        agent = log_file.stem.replace('_decisions', '')
        logs[agent] = []
        
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        logs[agent].append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    
    return logs


# Import Path for type hints
from pathlib import Path
