"""
SCML Agent Performance Analyzer

A comprehensive framework for running SCML tournaments and analyzing agent performance.

This package provides:
- Tournament runner with configurable presets
- Agent instrumentation ("打点") for custom logging
- NegMAS log parsing utilities
- Error detection framework for identifying agent failures
- Combined analysis of NegMAS logs + custom tracker data
- Report generation tools (JSON, console, charts)
- Post-tournament data import and visualization

Usage:
    # Run a tournament and analyze results
    from scml_analyzer import Tournament, TournamentConfig
    
    # Configure and run
    tournament = Tournament(
        agents=[MyAgent, TopAgent],
        config=TournamentConfig.quick_test()
    )
    results = tournament.run()
    
    # For agent instrumentation, use tracker in your agent:
    from scml_analyzer.tracker import Tracker
    
    tracker = Tracker.get("my_agent_id")
    tracker.checkpoint("step_10", day=10, inventory=100)
    tracker.decision("accept_offer", {"price": 50, "reason": "good_price"})
    
    # Post-tournament processing (auto data import + visualization)
    from scml_analyzer.postprocess import postprocess_tournament
    postprocess_tournament(results, tracker_log_dir, config)
"""

from .log_parser import LogParser, SimulationData, AgentData
from .detectors import (
    BaseErrorDetector,
    OverpricingDetector,
    UnderpricingDetector,
    InventoryStarvationDetector,
    ProductionIdleDetector,
    LossContractDetector,
    NegotiationStallDetector,
)
from .analyzer import (
    AnalysisResult,
    # Legacy exports
    FailureAnalyzer,
    analyze_simulation,
)
from .report import ReportGenerator
from .mixins import LoggingNegotiatorMixin, log_decision
from .tracker import Tracker, TrackerConfig, EventType, TrackedEvent
from .tournament import Tournament, TournamentConfig, TournamentResults
from .postprocess import postprocess_tournament

__version__ = "0.3.0"
__all__ = [
    # Tournament runner
    "Tournament",
    "TournamentConfig",
    "TournamentResults",
    # Agent instrumentation (打点)
    "Tracker",
    "TrackerConfig",
    "EventType",
    "TrackedEvent",
    # Analysis
    "AnalysisResult",
    # Post-processing
    "postprocess_tournament",
    # Core classes (legacy)
    "LogParser",
    "SimulationData",
    "AgentData",
    "FailureAnalyzer",
    "ReportGenerator",
    # Detectors
    "BaseErrorDetector",
    "OverpricingDetector",
    "UnderpricingDetector",
    "InventoryStarvationDetector",
    "ProductionIdleDetector",
    "LossContractDetector",
    "NegotiationStallDetector",
    # Mixins
    "LoggingNegotiatorMixin",
    "log_decision",
    # Functions (legacy)
    "analyze_simulation",
]
