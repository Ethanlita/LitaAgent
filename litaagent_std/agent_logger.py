"""
Agent Logger Module

Provides file-based logging for LitaAgent variants.
Logs are written to files instead of console to avoid cluttering output during tournaments.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any


class AgentLogger:
    """
    A logger class for SCML agents that writes to files instead of console.
    
    Each agent instance gets its own log file based on agent ID.
    Supports structured logging with different levels.
    """
    
    # Class-level storage for all logger instances
    _loggers: Dict[str, 'AgentLogger'] = {}
    _log_dir: Optional[Path] = None
    _enabled: bool = True
    _console_echo: bool = False  # If True, also print to console
    
    @classmethod
    def set_log_directory(cls, log_dir: str):
        """Set the base directory for all agent logs."""
        cls._log_dir = Path(log_dir)
        cls._log_dir.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def enable(cls, enabled: bool = True):
        """Enable or disable logging globally."""
        cls._enabled = enabled
    
    @classmethod
    def set_console_echo(cls, echo: bool = True):
        """Enable or disable console echo (for debugging)."""
        cls._console_echo = echo
    
    @classmethod
    def get_logger(cls, agent_id: str, agent_type: str = "Agent") -> 'AgentLogger':
        """Get or create a logger for a specific agent."""
        if agent_id not in cls._loggers:
            cls._loggers[agent_id] = AgentLogger(agent_id, agent_type)
        return cls._loggers[agent_id]
    
    @classmethod
    def cleanup(cls):
        """Close all loggers and clean up resources."""
        for logger in cls._loggers.values():
            logger.close()
        cls._loggers.clear()
    
    def __init__(self, agent_id: str, agent_type: str = "Agent"):
        """Initialize logger for a specific agent."""
        self.agent_id = agent_id
        self.agent_type = agent_type
        self._file_handle = None
        self._log_file_path = None
        self._buffer = []  # Buffer logs until log dir is set
        
        # Initialize file if log dir is set
        if AgentLogger._log_dir is not None:
            self._init_file()
    
    def _init_file(self):
        """Initialize the log file."""
        if self._file_handle is not None:
            return
        
        if AgentLogger._log_dir is None:
            return
        
        # Create agent-specific log file
        safe_agent_id = self.agent_id.replace("@", "_at_").replace("/", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_file_path = AgentLogger._log_dir / f"{safe_agent_id}_{timestamp}.log"
        
        try:
            self._file_handle = open(self._log_file_path, 'w', encoding='utf-8')
            # Write header
            self._file_handle.write(f"# Agent Log: {self.agent_id} ({self.agent_type})\n")
            self._file_handle.write(f"# Started: {datetime.now().isoformat()}\n")
            self._file_handle.write("=" * 60 + "\n\n")
            
            # Flush buffer
            for buffered_msg in self._buffer:
                self._file_handle.write(buffered_msg + "\n")
            self._buffer.clear()
            
        except Exception as e:
            print(f"Warning: Could not create log file for {self.agent_id}: {e}")
    
    def _write(self, level: str, message: str):
        """Write a message to the log."""
        if not AgentLogger._enabled:
            return
        
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        formatted = f"[{timestamp}] [{level}] {message}"
        
        # Console echo if enabled
        if AgentLogger._console_echo:
            print(formatted)
        
        # Write to file
        if self._file_handle is not None:
            self._file_handle.write(formatted + "\n")
            self._file_handle.flush()
        elif AgentLogger._log_dir is not None:
            # Try to init file
            self._init_file()
            if self._file_handle is not None:
                self._file_handle.write(formatted + "\n")
                self._file_handle.flush()
            else:
                self._buffer.append(formatted)
        else:
            # Buffer until log dir is set
            self._buffer.append(formatted)
    
    def debug(self, message: str):
        """Log a debug message."""
        self._write("DEBUG", message)
    
    def info(self, message: str):
        """Log an info message."""
        self._write("INFO", message)
    
    def warning(self, message: str):
        """Log a warning message."""
        self._write("WARN", message)
    
    def error(self, message: str):
        """Log an error message."""
        self._write("ERROR", message)
    
    def step(self, day: int, message: str):
        """Log a step/day-related message."""
        self._write(f"DAY-{day:03d}", message)
    
    def negotiation(self, day: int, partner: str, action: str, details: str = ""):
        """Log a negotiation event."""
        msg = f"Negotiation with {partner}: {action}"
        if details:
            msg += f" | {details}"
        self._write(f"NEG-{day:03d}", msg)
    
    def contract(self, day: int, contract_type: str, contract_id: str, 
                 quantity: int, price: float, partner: str):
        """Log a contract event."""
        msg = f"{contract_type} contract {contract_id}: Q={quantity}, P={price:.2f}, Partner={partner}"
        self._write(f"CONTRACT-{day:03d}", msg)
    
    def inventory(self, day: int, raw: int, product: int, balance: float):
        """Log inventory state."""
        msg = f"Inventory: Raw={raw}, Product={product}, Balance={balance:.2f}"
        self._write(f"INV-{day:03d}", msg)
    
    def decision(self, day: int, decision_type: str, details: Dict[str, Any]):
        """Log a decision with structured data."""
        details_str = json.dumps(details, ensure_ascii=False)
        self._write(f"DECISION-{day:03d}", f"{decision_type}: {details_str}")
    
    def close(self):
        """Close the log file."""
        if self._file_handle is not None:
            self._file_handle.write("\n" + "=" * 60 + "\n")
            self._file_handle.write(f"# Ended: {datetime.now().isoformat()}\n")
            self._file_handle.close()
            self._file_handle = None


# Convenience functions for quick logging without agent context
def setup_agent_logging(log_dir: str, console_echo: bool = False):
    """Setup agent logging with the specified directory."""
    AgentLogger.set_log_directory(log_dir)
    AgentLogger.set_console_echo(console_echo)
    AgentLogger.enable(True)


def disable_agent_logging():
    """Disable all agent logging."""
    AgentLogger.enable(False)


def cleanup_agent_logging():
    """Cleanup all agent loggers."""
    AgentLogger.cleanup()


# A simple function to redirect print statements
class PrintToLogger:
    """
    Context manager to redirect print statements to a logger.
    
    Usage:
        with PrintToLogger(agent_logger):
            print("This goes to the log file")
    """
    
    def __init__(self, logger: AgentLogger):
        self.logger = logger
        self._original_print = None
    
    def __enter__(self):
        import builtins
        self._original_print = builtins.print
        
        def logged_print(*args, **kwargs):
            message = " ".join(str(a) for a in args)
            self.logger.info(message)
        
        builtins.print = logged_print
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import builtins
        builtins.print = self._original_print
        return False


if __name__ == "__main__":
    # Test the logger
    setup_agent_logging("./test_logs", console_echo=True)
    
    logger = AgentLogger.get_logger("TestAgent@0", "LitaAgentY")
    logger.info("Agent initialized")
    logger.step(1, "Starting day 1")
    logger.negotiation(1, "Partner@1", "PROPOSE", "Q=5, P=10")
    logger.contract(1, "SUPPLY", "contract-123", 5, 10.0, "Partner@1")
    logger.inventory(1, 100, 50, 1000.0)
    logger.decision(1, "ACCEPT_OFFER", {"partner": "Partner@1", "reason": "Good price"})
    logger.warning("Low inventory detected")
    logger.error("Contract execution failed")
    
    cleanup_agent_logging()
    print("Test complete! Check ./test_logs directory.")
