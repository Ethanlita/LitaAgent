"""
Tracker - Agent Instrumentation for Custom Logging ("打点")

Provides a simple API for agents to record custom events, checkpoints,
and decisions during simulation. All data is written to JSON files
for later analysis.

Usage in your agent:
    from scml_analyzer import Tracker
    
    class MyAgent(StdSyncAgent):
        def init(self):
            # Get tracker for this agent (auto-created per agent ID)
            self.tracker = Tracker.get(self.id)
            self.tracker.event("initialized", {"my_param": 123})
        
        def before_step(self):
            self.tracker.checkpoint("step_start", 
                day=self.awi.current_step,
                balance=self.awi.current_balance
            )
        
        def respond_to_negotiation_request(self, ...):
            self.tracker.negotiation("request_received", {
                "partner": partner,
                "issues": issues
            })
        
        def on_negotiation_success(self, contract, mechanism):
            self.tracker.decision("accept_contract", {
                "contract_id": contract.id,
                "quantity": contract.agreement["quantity"],
                "price": contract.agreement["unit_price"],
                "reason": "price within acceptable range"
            })
"""

import os
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from enum import Enum


class EventType(Enum):
    """Types of trackable events."""
    CHECKPOINT = "checkpoint"      # Periodic state snapshots
    EVENT = "event"                # General events
    DECISION = "decision"          # Agent decisions with reasoning
    NEGOTIATION = "negotiation"    # Negotiation-related events
    CONTRACT = "contract"          # Contract events
    INVENTORY = "inventory"        # Inventory state changes
    ERROR = "error"                # Errors/warnings
    METRIC = "metric"              # Numeric metrics


@dataclass
class TrackedEvent:
    """A single tracked event."""
    timestamp: str
    event_type: str
    name: str
    day: Optional[int]
    data: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass 
class TrackerConfig:
    """Configuration for the Tracker."""
    enabled: bool = True
    log_dir: Optional[str] = None
    buffer_size: int = 100  # Flush to disk after this many events
    include_timestamp: bool = True
    console_echo: bool = False  # Also print to console


class Tracker:
    """
    Agent instrumentation tracker for custom logging.
    
    Each agent gets its own Tracker instance. Events are buffered
    and periodically flushed to JSON files.
    """
    
    # Class-level storage
    _instances: Dict[str, 'Tracker'] = {}
    _config: TrackerConfig = TrackerConfig()
    _lock = threading.Lock()
    
    @classmethod
    def configure(cls, 
                  log_dir: Optional[str] = None,
                  enabled: bool = True,
                  buffer_size: int = 100,
                  console_echo: bool = False):
        """Configure global tracker settings."""
        cls._config = TrackerConfig(
            enabled=enabled,
            log_dir=log_dir,
            buffer_size=buffer_size,
            console_echo=console_echo,
        )
        
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get(cls, agent_id: str) -> 'Tracker':
        """Get or create a tracker for an agent."""
        with cls._lock:
            if agent_id not in cls._instances:
                cls._instances[agent_id] = Tracker(agent_id)
            return cls._instances[agent_id]
    
    @classmethod
    def get_all_trackers(cls) -> Dict[str, 'Tracker']:
        """Get all tracker instances."""
        return cls._instances.copy()
    
    @classmethod
    def flush_all(cls):
        """Flush all trackers to disk."""
        for tracker in cls._instances.values():
            tracker.flush()
    
    @classmethod
    def cleanup(cls):
        """Cleanup all trackers."""
        cls.flush_all()
        cls._instances.clear()
    
    @classmethod
    def get_all_events(cls) -> Dict[str, List[Dict]]:
        """Get all events from all trackers."""
        return {
            agent_id: tracker.get_events()
            for agent_id, tracker in cls._instances.items()
        }
    
    def __init__(self, agent_id: str):
        """Initialize tracker for an agent."""
        self.agent_id = agent_id
        self._events: List[TrackedEvent] = []
        self._current_day: Optional[int] = None
        self._metrics: Dict[str, List[tuple]] = {}  # metric_name -> [(day, value), ...]
        
    def _record(self, event_type: EventType, name: str, 
                data: Dict[str, Any], day: Optional[int] = None):
        """Record an event."""
        if not Tracker._config.enabled:
            return
        
        event = TrackedEvent(
            timestamp=datetime.now().isoformat() if Tracker._config.include_timestamp else "",
            event_type=event_type.value,
            name=name,
            day=day if day is not None else self._current_day,
            data=data,
        )
        
        self._events.append(event)
        
        # Console echo
        if Tracker._config.console_echo:
            print(f"[TRACK:{self.agent_id}] {event_type.value}:{name} day={event.day} {data}")
        
        # Auto-flush if buffer full
        if len(self._events) >= Tracker._config.buffer_size:
            self.flush()
    
    def set_day(self, day: int):
        """Set the current simulation day (for auto-tagging events)."""
        self._current_day = day
    
    # ========== Main API Methods ==========
    
    def checkpoint(self, name: str, day: Optional[int] = None, **data):
        """
        Record a checkpoint (periodic state snapshot).
        
        Example:
            tracker.checkpoint("step_start", 
                day=5, 
                balance=1000, 
                inventory_raw=50,
                inventory_product=30
            )
        """
        self._record(EventType.CHECKPOINT, name, data, day)
    
    def event(self, name: str, data: Optional[Dict[str, Any]] = None, day: Optional[int] = None):
        """
        Record a general event.
        
        Example:
            tracker.event("exogenous_contract_received", {
                "contract_id": "abc123",
                "quantity": 10,
                "price": 150
            })
        """
        self._record(EventType.EVENT, name, data or {}, day)
    
    def decision(self, name: str, data: Optional[Dict[str, Any]] = None, day: Optional[int] = None):
        """
        Record a decision with reasoning.
        
        Example:
            tracker.decision("reject_offer", {
                "partner": "Agent@1",
                "offered_price": 20,
                "my_min_price": 25,
                "reason": "price too low"
            })
        """
        self._record(EventType.DECISION, name, data or {}, day)
    
    def negotiation(self, name: str, data: Optional[Dict[str, Any]] = None, day: Optional[int] = None):
        """
        Record a negotiation event.
        
        Example:
            tracker.negotiation("counter_offer", {
                "partner": "Agent@1",
                "their_offer": {"q": 5, "p": 20},
                "my_counter": {"q": 5, "p": 25}
            })
        """
        self._record(EventType.NEGOTIATION, name, data or {}, day)
    
    def contract(self, name: str, contract_id: str, 
                 quantity: int, price: float, partner: str,
                 contract_type: str = "unknown",
                 day: Optional[int] = None,
                 **extra):
        """
        Record a contract event.
        
        Example:
            tracker.contract("signed", 
                contract_id="abc123",
                quantity=10,
                price=150,
                partner="Agent@1",
                contract_type="supply"
            )
        """
        data = {
            "contract_id": contract_id,
            "quantity": quantity,
            "price": price,
            "partner": partner,
            "contract_type": contract_type,
            **extra
        }
        self._record(EventType.CONTRACT, name, data, day)
    
    def inventory(self, raw: int, product: int, balance: float,
                  day: Optional[int] = None, **extra):
        """
        Record inventory state.
        
        Example:
            tracker.inventory(raw=100, product=50, balance=5000)
        """
        data = {
            "raw_material": raw,
            "product": product,
            "balance": balance,
            **extra
        }
        self._record(EventType.INVENTORY, name="state", data=data, day=day)
    
    def metric(self, name: str, value: float, day: Optional[int] = None):
        """
        Record a numeric metric (for time-series analysis).
        
        Example:
            tracker.metric("profit_margin", 0.15)
            tracker.metric("success_rate", 0.8)
        """
        d = day if day is not None else self._current_day
        if name not in self._metrics:
            self._metrics[name] = []
        self._metrics[name].append((d, value))
        
        self._record(EventType.METRIC, name, {"value": value}, d)
    
    def error(self, name: str, message: str, day: Optional[int] = None, **extra):
        """
        Record an error or warning.
        
        Example:
            tracker.error("production_failed", 
                message="Insufficient raw materials",
                needed=50,
                available=30
            )
        """
        data = {"message": message, **extra}
        self._record(EventType.ERROR, name, data, day)
    
    # ========== Data Access ==========
    
    def get_events(self) -> List[Dict]:
        """Get all recorded events as dictionaries."""
        return [e.to_dict() for e in self._events]
    
    def get_events_by_type(self, event_type: EventType) -> List[Dict]:
        """Get events filtered by type."""
        return [e.to_dict() for e in self._events if e.event_type == event_type.value]
    
    def get_metrics(self) -> Dict[str, List[tuple]]:
        """Get all recorded metrics."""
        return self._metrics.copy()
    
    def get_decisions(self) -> List[Dict]:
        """Get all decision events."""
        return self.get_events_by_type(EventType.DECISION)
    
    # ========== Persistence ==========
    
    def flush(self):
        """Flush events to disk."""
        if not Tracker._config.log_dir or not self._events:
            return
        
        log_dir = Path(Tracker._config.log_dir)
        safe_id = self.agent_id.replace("@", "_at_").replace("/", "_")
        log_file = log_dir / f"tracker_{safe_id}.jsonl"
        
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                for event in self._events:
                    f.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
            self._events.clear()
        except Exception as e:
            print(f"Warning: Failed to flush tracker for {self.agent_id}: {e}")
    
    def save_summary(self) -> Optional[Path]:
        """Save a summary JSON file."""
        if not Tracker._config.log_dir:
            return None
        
        log_dir = Path(Tracker._config.log_dir)
        safe_id = self.agent_id.replace("@", "_at_").replace("/", "_")
        summary_file = log_dir / f"tracker_{safe_id}_summary.json"
        
        summary = {
            "agent_id": self.agent_id,
            "total_events": len(self._events),
            "metrics": self._metrics,
            "event_counts": {},
        }
        
        # Count events by type
        for event in self._events:
            etype = event.event_type
            summary["event_counts"][etype] = summary["event_counts"].get(etype, 0) + 1
        
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            return summary_file
        except Exception as e:
            print(f"Warning: Failed to save summary for {self.agent_id}: {e}")
            return None
