"""
Log Parser Module

Parses SCML/NegMAS simulation log files and provides structured data access.

Supported log files:
- contracts.csv: Signed contracts with execution status
- negotiations.csv: Detailed negotiation history
- stats.csv.csv: Per-step agent statistics
- actions.csv: Negotiation actions (offers, accepts, rejects)
- agents.csv: Agent metadata
- breaches.csv: Contract breach records
- info.json: Simulation configuration
- params.json: World parameters
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np


@dataclass
class ContractRecord:
    """Represents a single contract."""
    id: str
    seller_name: str
    buyer_name: str
    seller_type: str
    buyer_type: str
    delivery_time: int
    quantity: int
    unit_price: float
    signed_at: int
    concluded_at: int
    executed_at: Optional[int]
    negotiation_id: Optional[str]
    product_name: str
    breaches: Optional[str]
    is_executed: bool
    is_breached: bool
    
    @property
    def total_value(self) -> float:
        return self.quantity * self.unit_price


@dataclass
class NegotiationRecord:
    """Represents a negotiation session."""
    id: str
    partners: List[str]
    partner_types: List[str]
    buyer: str
    seller: str
    product: str
    is_buy: bool
    sim_step: int
    requested_at: int
    ended_at: int
    final_status: str  # 'succeeded', 'failed', 'timedout', etc.
    failed: bool
    agreement: Optional[Tuple]  # (quantity, time, price)
    n_neg_steps: int
    history: List[Dict]  # List of negotiation states
    offers: Dict[str, List[Tuple]]  # agent -> list of offers


@dataclass
class AgentStatsTimeSeries:
    """Time series statistics for an agent."""
    agent_name: str
    steps: List[int]
    scores: List[float]
    balances: List[float]
    bankrupt: List[bool]
    productivity: List[float]
    shortfall_quantity: List[int]
    shortfall_penalty: List[float]
    storage_cost: List[float]
    disposal_cost: List[float]
    inventory_input: List[int]
    inventory_output: List[int]
    
    def get_stat(self, stat_name: str) -> List:
        """Get a specific statistic by name."""
        return getattr(self, stat_name, [])
    
    def average(self, stat_name: str) -> float:
        """Get average value of a statistic."""
        values = self.get_stat(stat_name)
        return np.mean(values) if values else 0.0


@dataclass
class ActionRecord:
    """Represents a negotiation action (offer/response)."""
    id: int
    neg_id: int
    step: int
    relative_time: float
    sender: str
    receiver: str
    state: str  # 'continuing', 'agreement', 'rejection', etc.
    quantity: int
    delivery_step: int
    unit_price: float


@dataclass
class AgentData:
    """Aggregated data for a single agent."""
    name: str
    agent_type: str
    
    # Contracts
    contracts_as_seller: List[ContractRecord] = field(default_factory=list)
    contracts_as_buyer: List[ContractRecord] = field(default_factory=list)
    
    # Negotiations
    negotiations: List[NegotiationRecord] = field(default_factory=list)
    
    # Actions
    actions: List[ActionRecord] = field(default_factory=list)
    
    # Time series stats
    stats: Optional[AgentStatsTimeSeries] = None
    
    # Custom agent logs (if available)
    custom_logs: List[Dict] = field(default_factory=list)
    
    @property
    def all_contracts(self) -> List[ContractRecord]:
        return self.contracts_as_seller + self.contracts_as_buyer
    
    @property
    def breached_contracts(self) -> List[ContractRecord]:
        return [c for c in self.all_contracts if c.is_breached]
    
    @property
    def successful_negotiations(self) -> List[NegotiationRecord]:
        return [n for n in self.negotiations if not n.failed]
    
    @property
    def failed_negotiations(self) -> List[NegotiationRecord]:
        return [n for n in self.negotiations if n.failed]
    
    @property
    def negotiation_success_rate(self) -> float:
        if not self.negotiations:
            return 0.0
        return len(self.successful_negotiations) / len(self.negotiations)
    
    @property
    def breach_rate(self) -> float:
        if not self.all_contracts:
            return 0.0
        return len(self.breached_contracts) / len(self.all_contracts)


@dataclass
class MarketStats:
    """Market-wide statistics."""
    trading_prices: Dict[str, List[float]]  # product -> prices over time
    average_prices: Dict[str, float]  # product -> average price
    total_contracts: int
    total_breaches: int
    total_negotiations: int
    successful_negotiations: int


@dataclass
class SimulationData:
    """Complete simulation data container."""
    log_dir: str
    
    # Raw dataframes
    contracts_df: Optional[pd.DataFrame] = None
    negotiations_df: Optional[pd.DataFrame] = None
    stats_df: Optional[pd.DataFrame] = None
    actions_df: Optional[pd.DataFrame] = None
    agents_df: Optional[pd.DataFrame] = None
    breaches_df: Optional[pd.DataFrame] = None
    
    # Simulation info
    info: Dict = field(default_factory=dict)
    params: Dict = field(default_factory=dict)
    
    # Processed data
    agents: Dict[str, AgentData] = field(default_factory=dict)
    contracts: List[ContractRecord] = field(default_factory=list)
    negotiations: List[NegotiationRecord] = field(default_factory=list)
    market_stats: Optional[MarketStats] = None
    
    # Simulation metadata
    n_steps: int = 0
    n_agents: int = 0
    agent_names: List[str] = field(default_factory=list)
    
    def get_agent_data(self, agent_name: str) -> Optional[AgentData]:
        """Get data for a specific agent."""
        return self.agents.get(agent_name)
    
    def get_agent_names_by_type(self, agent_type: str) -> List[str]:
        """Get all agent names of a specific type."""
        return [name for name, data in self.agents.items() 
                if agent_type.lower() in data.agent_type.lower()]


class LogParser:
    """
    Parser for SCML/NegMAS simulation log files.
    
    Usage:
        parser = LogParser()
        data = parser.parse_directory("./logs")
        
        # Access agent data
        agent_data = data.get_agent_data("MyAgent@0")
        
        # Access contracts
        for contract in data.contracts:
            print(f"{contract.seller_name} -> {contract.buyer_name}: {contract.quantity}")
    """
    
    def __init__(self, exclude_system_agents: bool = True):
        """
        Initialize the log parser.
        
        Args:
            exclude_system_agents: If True, exclude SELLER/BUYER system agents from analysis
        """
        self.exclude_system_agents = exclude_system_agents
    
    def parse_directory(self, log_dir: str) -> SimulationData:
        """
        Parse all log files in a directory.
        
        Args:
            log_dir: Path to the directory containing log files
            
        Returns:
            SimulationData object containing all parsed data
        """
        log_dir = Path(log_dir)
        if not log_dir.exists():
            raise FileNotFoundError(f"Log directory not found: {log_dir}")
        
        data = SimulationData(log_dir=str(log_dir))
        
        # Load raw dataframes
        data.contracts_df = self._load_csv(log_dir / "contracts.csv")
        data.negotiations_df = self._load_csv(log_dir / "negotiations.csv")
        data.stats_df = self._load_csv(log_dir / "stats.csv.csv")  # Note: double .csv
        data.actions_df = self._load_csv(log_dir / "actions.csv")
        data.agents_df = self._load_csv(log_dir / "agents.csv")
        data.breaches_df = self._load_csv(log_dir / "breaches.csv")
        
        # Load JSON configs
        data.info = self._load_json(log_dir / "info.json")
        data.params = self._load_json(log_dir / "params.json")
        
        # Extract simulation metadata
        if data.stats_df is not None:
            data.n_steps = len(data.stats_df)
        
        # Parse agents
        data.agents, data.agent_names = self._parse_agents(data)
        data.n_agents = len(data.agent_names)
        
        # Parse contracts
        data.contracts = self._parse_contracts(data)
        
        # Parse negotiations
        data.negotiations = self._parse_negotiations(data)
        
        # Assign data to agents
        self._assign_data_to_agents(data)
        
        # Calculate market stats
        data.market_stats = self._calculate_market_stats(data)
        
        return data
    
    def _load_csv(self, path: Path) -> Optional[pd.DataFrame]:
        """Load a CSV file if it exists and is not empty."""
        if not path.exists():
            return None
        try:
            df = pd.read_csv(path)
            return df if not df.empty else None
        except pd.errors.EmptyDataError:
            return None
        except Exception as e:
            print(f"Warning: Could not load {path}: {e}")
            return None
    
    def _load_json(self, path: Path) -> Dict:
        """Load a JSON file if it exists."""
        if not path.exists():
            return {}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load {path}: {e}")
            return {}
    
    def _parse_agents(self, data: SimulationData) -> Tuple[Dict[str, AgentData], List[str]]:
        """Parse agent information."""
        agents = {}
        agent_names = []
        
        if data.agents_df is not None:
            for _, row in data.agents_df.iterrows():
                name = row['name']
                agent_type = row['type']
                
                # Skip system agents if configured
                if self.exclude_system_agents:
                    if name.startswith('SELLER') or name.startswith('BUYER') or name == 'NoAgent':
                        continue
                
                agents[name] = AgentData(name=name, agent_type=agent_type)
                agent_names.append(name)
        
        return agents, agent_names
    
    def _parse_contracts(self, data: SimulationData) -> List[ContractRecord]:
        """Parse contract records."""
        contracts = []
        
        if data.contracts_df is None:
            return contracts
        
        for _, row in data.contracts_df.iterrows():
            # Skip if seller/buyer is system agent
            if self.exclude_system_agents:
                if (row['seller_name'].startswith('SELLER') or 
                    row['seller_name'].startswith('BUYER') or
                    row['buyer_name'].startswith('SELLER') or 
                    row['buyer_name'].startswith('BUYER')):
                    continue
            
            is_executed = pd.notna(row.get('executed_at')) and row.get('executed_at', -1) >= 0
            is_breached = pd.notna(row.get('breaches')) and str(row.get('breaches', '')).strip() != ''
            
            contract = ContractRecord(
                id=str(row['id']),
                seller_name=row['seller_name'],
                buyer_name=row['buyer_name'],
                seller_type=str(row.get('seller_type', '')),
                buyer_type=str(row.get('buyer_type', '')),
                delivery_time=int(row['delivery_time']),
                quantity=int(row['quantity']),
                unit_price=float(row['unit_price']),
                signed_at=int(row.get('signed_at', -1)),
                concluded_at=int(row.get('concluded_at', -1)),
                executed_at=int(row['executed_at']) if pd.notna(row.get('executed_at')) and row.get('executed_at', -1) >= 0 else None,
                negotiation_id=str(row.get('negotiation_id', '')) if pd.notna(row.get('negotiation_id')) else None,
                product_name=str(row.get('product_name', '')),
                breaches=str(row.get('breaches', '')) if pd.notna(row.get('breaches')) else None,
                is_executed=is_executed,
                is_breached=is_breached,
            )
            contracts.append(contract)
        
        return contracts
    
    def _parse_negotiations(self, data: SimulationData) -> List[NegotiationRecord]:
        """Parse negotiation records."""
        negotiations = []
        
        if data.negotiations_df is None:
            return negotiations
        
        for _, row in data.negotiations_df.iterrows():
            # Parse partners
            partners = self._parse_list_field(row.get('partners', '[]'))
            partner_types = self._parse_list_field(row.get('partner_types', '[]'))
            
            # Skip if involves system agents
            if self.exclude_system_agents:
                if any(p.startswith('SELLER') or p.startswith('BUYER') for p in partners):
                    continue
            
            # Parse agreement
            agreement = None
            if pd.notna(row.get('agreement')) and row.get('agreement') not in ['None', '']:
                try:
                    agreement = eval(str(row['agreement']))
                except:
                    pass
            
            # Parse history
            history = []
            if pd.notna(row.get('history')):
                try:
                    history = eval(str(row['history']))
                except:
                    pass
            
            # Parse offers
            offers = {}
            if pd.notna(row.get('offers')):
                try:
                    offers = eval(str(row['offers']))
                except:
                    pass
            
            negotiation = NegotiationRecord(
                id=str(row['id']),
                partners=partners,
                partner_types=partner_types,
                buyer=str(row.get('buyer', '')),
                seller=str(row.get('seller', '')),
                product=str(row.get('product', '')),
                is_buy=bool(row.get('is_buy', False)),
                sim_step=int(row.get('sim_step', 0)),
                requested_at=int(row.get('requested_at', 0)),
                ended_at=int(row.get('ended_at', 0)),
                final_status=str(row.get('final_status', '')),
                failed=bool(row.get('failed', False)),
                agreement=agreement,
                n_neg_steps=int(row.get('step', 0)),
                history=history,
                offers=offers,
            )
            negotiations.append(negotiation)
        
        return negotiations
    
    def _parse_list_field(self, field_value) -> List[str]:
        """Parse a list field from CSV."""
        if pd.isna(field_value):
            return []
        try:
            return eval(str(field_value))
        except:
            return []
    
    def _assign_data_to_agents(self, data: SimulationData):
        """Assign contracts, negotiations, and stats to individual agents."""
        # Assign contracts
        for contract in data.contracts:
            if contract.seller_name in data.agents:
                data.agents[contract.seller_name].contracts_as_seller.append(contract)
            if contract.buyer_name in data.agents:
                data.agents[contract.buyer_name].contracts_as_buyer.append(contract)
        
        # Assign negotiations
        for neg in data.negotiations:
            for partner in neg.partners:
                if partner in data.agents:
                    data.agents[partner].negotiations.append(neg)
        
        # Assign stats time series
        if data.stats_df is not None:
            for agent_name, agent_data in data.agents.items():
                stats = self._extract_agent_stats(data.stats_df, agent_name)
                if stats:
                    agent_data.stats = stats
        
        # Assign actions
        if data.actions_df is not None:
            for _, row in data.actions_df.iterrows():
                sender = row['sender']
                if sender in data.agents:
                    action = ActionRecord(
                        id=int(row['id']),
                        neg_id=int(row['neg_id']),
                        step=int(row['step']),
                        relative_time=float(row['relative_time']),
                        sender=sender,
                        receiver=row['receiver'],
                        state=str(row['state']),
                        quantity=int(row['quantity']),
                        delivery_step=int(row['delivery_step']),
                        unit_price=float(row['unit_price']),
                    )
                    data.agents[sender].actions.append(action)
    
    def _extract_agent_stats(self, stats_df: pd.DataFrame, agent_name: str) -> Optional[AgentStatsTimeSeries]:
        """Extract time series statistics for an agent."""
        # Column naming pattern: stat_agentname
        score_col = f"score_{agent_name}"
        balance_col = f"balance_{agent_name}"
        
        if score_col not in stats_df.columns:
            return None
        
        steps = list(range(len(stats_df)))
        
        def get_col(col_name: str) -> List:
            full_col = f"{col_name}_{agent_name}"
            if full_col in stats_df.columns:
                return stats_df[full_col].fillna(0).tolist()
            return [0] * len(stats_df)
        
        return AgentStatsTimeSeries(
            agent_name=agent_name,
            steps=steps,
            scores=get_col("score"),
            balances=get_col("balance"),
            bankrupt=[bool(x) for x in get_col("bankrupt")],
            productivity=get_col("productivity"),
            shortfall_quantity=[int(x) for x in get_col("shortfall_quantity")],
            shortfall_penalty=get_col("shortfall_penalty"),
            storage_cost=get_col("storage_cost"),
            disposal_cost=get_col("disposal_cost"),
            inventory_input=[int(x) for x in get_col("inventory_input")],
            inventory_output=[int(x) for x in get_col("inventory_output")],
        )
    
    def _calculate_market_stats(self, data: SimulationData) -> MarketStats:
        """Calculate market-wide statistics."""
        trading_prices = {}
        
        if data.stats_df is not None:
            # Extract trading prices for each product
            for col in data.stats_df.columns:
                if col.startswith('trading_price_'):
                    product = col.replace('trading_price_', '')
                    trading_prices[product] = data.stats_df[col].tolist()
        
        average_prices = {p: np.mean(prices) for p, prices in trading_prices.items()}
        
        total_breaches = sum(1 for c in data.contracts if c.is_breached)
        successful_negs = sum(1 for n in data.negotiations if not n.failed)
        
        return MarketStats(
            trading_prices=trading_prices,
            average_prices=average_prices,
            total_contracts=len(data.contracts),
            total_breaches=total_breaches,
            total_negotiations=len(data.negotiations),
            successful_negotiations=successful_negs,
        )


# Convenience function
def parse_logs(log_dir: str, exclude_system_agents: bool = True) -> SimulationData:
    """
    Convenience function to parse log files.
    
    Args:
        log_dir: Path to log directory
        exclude_system_agents: Whether to exclude system agents
        
    Returns:
        SimulationData object
    """
    parser = LogParser(exclude_system_agents=exclude_system_agents)
    return parser.parse_directory(log_dir)
