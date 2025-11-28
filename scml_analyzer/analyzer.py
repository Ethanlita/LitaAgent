"""
Analyzer Module

Main analysis engine that combines:
1. NegMAS native log analysis (contracts, negotiations, stats)
2. Tracker data from agent instrumentation
3. Failure detection

Provides comprehensive post-tournament analysis.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type
from pathlib import Path
import json

from .log_parser import LogParser, SimulationData, AgentData
from .detectors import (
    BaseErrorDetector,
    Issue,
    get_all_detectors,
    get_detector,
    DETECTOR_REGISTRY,
)


@dataclass
class AgentAnalysisResult:
    """Analysis results for a single agent."""
    agent_name: str
    agent_type: str
    issues_by_type: Dict[str, List[Issue]] = field(default_factory=dict)
    tracker_events: List[Dict] = field(default_factory=list)
    tracker_metrics: Dict[str, List[tuple]] = field(default_factory=dict)
    
    @property
    def total_issues(self) -> int:
        return sum(len(issues) for issues in self.issues_by_type.values())
    
    @property
    def issue_counts(self) -> Dict[str, int]:
        return {name: len(issues) for name, issues in self.issues_by_type.items()}
    
    @property
    def severity_counts(self) -> Dict[str, int]:
        counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        for issues in self.issues_by_type.values():
            for issue in issues:
                counts[issue.severity] = counts.get(issue.severity, 0) + 1
        return counts
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "agent_name": self.agent_name,
            "agent_type": self.agent_type,
            "total_issues": self.total_issues,
            "issue_counts": self.issue_counts,
            "severity_counts": self.severity_counts,
            "issues": {
                name: [issue.to_dict() for issue in issues]
                for name, issues in self.issues_by_type.items()
            },
            "tracker_event_count": len(self.tracker_events),
            "tracker_metrics": self.tracker_metrics,
        }


@dataclass
class AnalysisReport:
    """Complete analysis report for a tournament."""
    tournament_dir: str
    n_worlds: int
    world_results: List['AnalysisResult'] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    tracker_summary: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_issues(self) -> int:
        return sum(r.total_issues for r in self.world_results)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tournament_dir": self.tournament_dir,
            "n_worlds": self.n_worlds,
            "total_issues": self.total_issues,
            "summary": self.summary,
            "tracker_summary": self.tracker_summary,
            "worlds": [r.to_dict() for r in self.world_results],
        }
    
    def save(self, path: Optional[str] = None) -> str:
        """Save report to JSON file."""
        if path is None:
            save_path = str(Path(self.tournament_dir) / "analysis_report.json")
        else:
            save_path = path
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        
        return save_path


@dataclass
class AnalysisResult:
    """Complete analysis results for a simulation."""
    simulation_dir: str
    n_agents: int
    n_steps: int
    agent_results: Dict[str, AgentAnalysisResult] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_issues(self) -> int:
        return sum(r.total_issues for r in self.agent_results.values())
    
    @property
    def issues_by_type(self) -> Dict[str, int]:
        """Aggregate issue counts by type across all agents."""
        counts = {}
        for result in self.agent_results.values():
            for issue_type, issues in result.issues_by_type.items():
                counts[issue_type] = counts.get(issue_type, 0) + len(issues)
        return counts
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "simulation_dir": self.simulation_dir,
            "n_agents": self.n_agents,
            "n_steps": self.n_steps,
            "total_issues": self.total_issues,
            "issues_by_type": self.issues_by_type,
            "summary": self.summary,
            "agents": {
                name: result.to_dict()
                for name, result in self.agent_results.items()
            }
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    def save_json(self, path: str):
        """Save results to JSON file."""
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.to_json())


class Analyzer:
    """
    Main analysis engine for tournament results.
    
    Combines NegMAS logs with Tracker instrumentation data.
    
    Usage:
        # After running a tournament
        analyzer = Analyzer(tournament_results.log_dir)
        report = analyzer.generate_report()
        report.save()
        
        # Or analyze specific aspects
        analyzer.analyze_agent("LitaAgentY")
        analyzer.get_tracker_decisions("LitaAgentYR")
    """
    
    def __init__(
        self,
        log_dir: str,
        detectors: Optional[List[BaseErrorDetector]] = None,
        detector_names: Optional[List[str]] = None,
        exclude_system_agents: bool = True,
    ):
        """
        Initialize analyzer.
        
        Args:
            log_dir: Tournament log directory
            detectors: Custom detector instances
            detector_names: Names of detectors to use
            exclude_system_agents: Exclude SELLER/BUYER system agents
        """
        self.log_dir = Path(log_dir)
        self.exclude_system_agents = exclude_system_agents
        
        # Set up detectors
        if detectors is not None:
            self.detectors = detectors
        elif detector_names is not None:
            self.detectors = [get_detector(name) for name in detector_names]
        else:
            self.detectors = get_all_detectors()
        
        # Load tracker data
        self._tracker_data = self._load_tracker_data()
    
    def _load_tracker_data(self) -> Dict[str, List[Dict]]:
        """Load tracker logs from disk."""
        tracker_dir = self.log_dir / "tracker_logs"
        tracker_data = {}
        
        if not tracker_dir.exists():
            return tracker_data
        
        # Load .json files (new format from auto_tracker.py)
        for log_file in tracker_dir.glob("agent_*.json"):
            agent_id = log_file.stem.replace("agent_", "").replace("_at_", "@")
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # entries contains the event log
                tracker_data[agent_id] = data.get("entries", [])
            except Exception as e:
                print(f"Warning: Failed to load tracker log {log_file}: {e}")
        
        # Also load legacy .jsonl files if present
        for log_file in tracker_dir.glob("tracker_*.jsonl"):
            agent_id = log_file.stem.replace("tracker_", "").replace("_at_", "@")
            events = []
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            events.append(json.loads(line))
                tracker_data[agent_id] = events
            except Exception as e:
                print(f"Warning: Failed to load tracker log {log_file}: {e}")
        
        return tracker_data
    
    def _find_world_dirs(self) -> List[Path]:
        """Find all world directories in the tournament."""
        return sorted([
            d for d in self.log_dir.iterdir()
            if d.is_dir() and d.name.startswith("world_")
        ])
    
    def analyze_world(self, world_dir: Path) -> AnalysisResult:
        """Analyze a single world's logs."""
        # Parse NegMAS logs
        parser = LogParser(exclude_system_agents=self.exclude_system_agents)
        sim_data = parser.parse_directory(str(world_dir))
        
        # Initialize result
        result = AnalysisResult(
            simulation_dir=str(world_dir),
            n_agents=sim_data.n_agents,
            n_steps=sim_data.n_steps,
        )
        
        # Analyze each agent
        for agent_name, agent_data in sim_data.agents.items():
            agent_result = self._analyze_agent(agent_data, sim_data)
            result.agent_results[agent_name] = agent_result
        
        # Generate summary
        result.summary = self._generate_world_summary(result, sim_data)
        
        return result
    
    def _analyze_agent(
        self, 
        agent_data: AgentData, 
        sim_data: SimulationData
    ) -> AgentAnalysisResult:
        """Analyze a single agent."""
        result = AgentAnalysisResult(
            agent_name=agent_data.name,
            agent_type=agent_data.agent_type,
        )
        
        # Run failure detectors
        for detector in self.detectors:
            issues = detector.detect(agent_data, sim_data)
            if issues:
                result.issues_by_type[detector.name] = issues
        
        # Add tracker data if available
        if agent_data.name in self._tracker_data:
            result.tracker_events = self._tracker_data[agent_data.name]
            # Extract metrics
            for event in result.tracker_events:
                if event.get("event_type") == "metric":
                    name = event.get("name", "unknown")
                    value = event.get("data", {}).get("value")
                    day = event.get("day")
                    if name not in result.tracker_metrics:
                        result.tracker_metrics[name] = []
                    result.tracker_metrics[name].append((day, value))
        
        return result
    
    def _generate_world_summary(
        self, 
        result: AnalysisResult, 
        sim_data: SimulationData
    ) -> Dict[str, Any]:
        """Generate summary for a world."""
        summary = {
            "total_agents": result.n_agents,
            "total_steps": result.n_steps,
            "total_issues": result.total_issues,
            "issues_by_type": result.issues_by_type,
        }
        
        if sim_data.market_stats:
            summary["market"] = {
                "total_contracts": sim_data.market_stats.total_contracts,
                "total_breaches": sim_data.market_stats.total_breaches,
            }
        
        return summary
    
    def generate_report(self, max_worlds: Optional[int] = None) -> AnalysisReport:
        """
        Generate a complete analysis report for the tournament.
        
        Args:
            max_worlds: Maximum number of worlds to analyze (None = all)
            
        Returns:
            AnalysisReport with all analysis results
        """
        world_dirs = self._find_world_dirs()
        
        if max_worlds is not None:
            world_dirs = world_dirs[:max_worlds]
        
        report = AnalysisReport(
            tournament_dir=str(self.log_dir),
            n_worlds=len(world_dirs),
        )
        
        # Analyze each world
        for world_dir in world_dirs:
            result = self.analyze_world(world_dir)
            report.world_results.append(result)
        
        # Generate overall summary
        report.summary = self._generate_tournament_summary(report)
        report.tracker_summary = self._generate_tracker_summary()
        
        return report
    
    def _generate_tournament_summary(self, report: AnalysisReport) -> Dict[str, Any]:
        """Generate tournament-level summary."""
        # Aggregate issues by agent type
        issues_by_agent_type = {}
        for world_result in report.world_results:
            for agent_name, agent_result in world_result.agent_results.items():
                agent_type = agent_result.agent_type
                if agent_type not in issues_by_agent_type:
                    issues_by_agent_type[agent_type] = {
                        "total_issues": 0,
                        "by_type": {},
                    }
                issues_by_agent_type[agent_type]["total_issues"] += agent_result.total_issues
                for issue_type, count in agent_result.issue_counts.items():
                    if issue_type not in issues_by_agent_type[agent_type]["by_type"]:
                        issues_by_agent_type[agent_type]["by_type"][issue_type] = 0
                    issues_by_agent_type[agent_type]["by_type"][issue_type] += count
        
        return {
            "n_worlds": report.n_worlds,
            "total_issues": report.total_issues,
            "issues_by_agent_type": issues_by_agent_type,
        }
    
    def _generate_tracker_summary(self) -> Dict[str, Any]:
        """Summarize tracker data across all agents."""
        summary = {
            "agents_tracked": len(self._tracker_data),
            "by_agent": {},
        }
        
        for agent_id, events in self._tracker_data.items():
            event_counts = {}
            decision_types = {}
            
            for event in events:
                etype = event.get("event_type", "unknown")
                event_counts[etype] = event_counts.get(etype, 0) + 1
                
                if etype == "decision":
                    dname = event.get("name", "unknown")
                    decision_types[dname] = decision_types.get(dname, 0) + 1
            
            summary["by_agent"][agent_id] = {
                "total_events": len(events),
                "event_counts": event_counts,
                "decision_types": decision_types,
            }
        
        return summary
    
    def get_agent_decisions(self, agent_type: str) -> List[Dict]:
        """Get all decision events for agents of a given type."""
        decisions = []
        for agent_id, events in self._tracker_data.items():
            if agent_type in agent_id:
                for event in events:
                    if event.get("event_type") == "decision":
                        decisions.append({
                            "agent_id": agent_id,
                            **event
                        })
        return decisions
    
    def get_agent_metrics(self, agent_type: str) -> Dict[str, List[tuple]]:
        """Get all metrics for agents of a given type."""
        metrics = {}
        for agent_id, events in self._tracker_data.items():
            if agent_type in agent_id:
                for event in events:
                    if event.get("event_type") == "metric":
                        name = event.get("name", "unknown")
                        value = event.get("data", {}).get("value")
                        day = event.get("day")
                        if name not in metrics:
                            metrics[name] = []
                        metrics[name].append((agent_id, day, value))
        return metrics


# Legacy class for compatibility
class FailureAnalyzer(Analyzer):
    """Legacy alias for Analyzer class."""
    pass


def analyze_simulation(
    log_dir: str,
    detectors: Optional[List[str]] = None,
    detector_params: Optional[Dict[str, Dict]] = None,
    exclude_system_agents: bool = True,
) -> AnalysisResult:
    """
    Convenience function to analyze a simulation.
    
    Args:
        log_dir: Path to log directory
        detectors: List of detector names to use (None = all)
        detector_params: Parameters for detectors
        exclude_system_agents: Whether to exclude system agents
        
    Returns:
        AnalysisResult
    """
    analyzer = Analyzer(
        log_dir,
        detector_names=detectors,
        exclude_system_agents=exclude_system_agents,
    )
    
    # Find first world dir
    world_dirs = analyzer._find_world_dirs()
    if world_dirs:
        return analyzer.analyze_world(world_dirs[0])
    else:
        # Try analyzing the log_dir directly
        parser = LogParser(exclude_system_agents=exclude_system_agents)
        sim_data = parser.parse_directory(log_dir)
        return AnalysisResult(
            simulation_dir=log_dir,
            n_agents=sim_data.n_agents,
            n_steps=sim_data.n_steps,
        )
