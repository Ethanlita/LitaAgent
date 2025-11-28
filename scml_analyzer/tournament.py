"""
Tournament Runner - Run SCML competitions with comprehensive logging.

Provides a unified interface for running tournaments with:
- Configurable scale (quick test to full competition)
- Automatic NegMAS log collection
- Integration with Tracker for agent instrumentation
- Post-run analysis hooks

Usage:
    from scml_analyzer import Tournament
    
    # Quick test
    tournament = Tournament(
        agents=[MyAgent, TopAgent],
        n_configs=2,
        n_steps=30,
        name="quick_test"
    )
    results = tournament.run()
    
    # Full competition
    tournament = Tournament(
        agents=[MyAgent, TopAgent1, TopAgent2],
        n_configs=10,
        n_runs_per_world=2,
        n_steps=100,
        name="full_competition"
    )
    results = tournament.run()
    
    # Access results
    print(results.rankings)
    print(results.log_dir)
"""

import os
import json
import warnings
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Type, Callable
from collections import defaultdict

# Suppress noisy warnings
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import pandas as pd


@dataclass
class TournamentConfig:
    """Configuration for a tournament."""
    # Scale settings
    n_configs: int = 3              # Number of world configurations
    n_runs_per_world: int = 1       # Runs per configuration
    n_steps: int = 50               # Simulation steps per run
    n_processes: int = 2            # Number of production levels
    
    # Logging settings
    log_negotiations: bool = True
    save_contracts: bool = True
    save_stats: bool = True
    log_stats_every: int = 1
    
    # Tournament settings
    name: Optional[str] = None
    base_log_dir: str = "./tournament_logs"
    
    # Presets
    @classmethod
    def quick_test(cls) -> 'TournamentConfig':
        """Quick test configuration."""
        return cls(n_configs=2, n_runs_per_world=1, n_steps=30, name="quick_test")
    
    @classmethod
    def standard(cls) -> 'TournamentConfig':
        """Standard tournament configuration."""
        return cls(n_configs=5, n_runs_per_world=2, n_steps=50, name="standard")
    
    @classmethod
    def full(cls) -> 'TournamentConfig':
        """Full competition configuration (SCML 2025 style)."""
        return cls(n_configs=10, n_runs_per_world=2, n_steps=100, name="full")


@dataclass
class TournamentResults:
    """Results from a tournament run."""
    # Basic info
    name: str
    timestamp: str
    log_dir: str
    config: TournamentConfig
    
    # Agents
    agent_types: List[str]
    
    # Scores
    scores: pd.DataFrame           # All individual scores
    rankings: pd.DataFrame         # Aggregated rankings
    winner: str
    
    # Metadata
    n_worlds_completed: int
    n_worlds_failed: int
    duration_seconds: float
    
    def save(self):
        """Save results to the log directory."""
        results_file = Path(self.log_dir) / "tournament_results.json"
        
        data = {
            "name": self.name,
            "timestamp": self.timestamp,
            "config": {
                "n_configs": self.config.n_configs,
                "n_runs_per_world": self.config.n_runs_per_world,
                "n_steps": self.config.n_steps,
            },
            "agent_types": self.agent_types,
            "winner": self.winner,
            "rankings": self.rankings.to_dict(orient='records'),
            "n_worlds_completed": self.n_worlds_completed,
            "n_worlds_failed": self.n_worlds_failed,
            "duration_seconds": self.duration_seconds,
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Also save detailed scores
        scores_file = Path(self.log_dir) / "scores.csv"
        self.scores.to_csv(scores_file, index=False)
        
        return results_file


class Tournament:
    """
    SCML Tournament Runner.
    
    Runs competitions between agents with comprehensive logging.
    """
    
    def __init__(
        self,
        agents: List[Type],
        config: Optional[TournamentConfig] = None,
        # Shorthand config options
        n_configs: Optional[int] = None,
        n_runs_per_world: Optional[int] = None,
        n_steps: Optional[int] = None,
        name: Optional[str] = None,
        # Callbacks
        on_world_start: Optional[Callable] = None,
        on_world_end: Optional[Callable] = None,
    ):
        """
        Initialize tournament.
        
        Args:
            agents: List of agent classes to compete
            config: Tournament configuration (or use shorthand options)
            n_configs: Number of world configurations
            n_runs_per_world: Runs per configuration  
            n_steps: Steps per simulation
            name: Tournament name
            on_world_start: Callback before each world runs
            on_world_end: Callback after each world completes
        """
        self.agents = agents
        
        # Build config
        if config is not None:
            self.config = config
        else:
            self.config = TournamentConfig()
        
        # Override with shorthand options
        if n_configs is not None:
            self.config.n_configs = n_configs
        if n_runs_per_world is not None:
            self.config.n_runs_per_world = n_runs_per_world
        if n_steps is not None:
            self.config.n_steps = n_steps
        if name is not None:
            self.config.name = name
        
        # Callbacks
        self.on_world_start = on_world_start
        self.on_world_end = on_world_end
        
        # State
        self._log_dir: Optional[str] = None
        self._start_time: Optional[datetime] = None
    
    def _create_log_dir(self) -> str:
        """Create the log directory for this tournament."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = self.config.name or "tournament"
        
        log_dir = os.path.join(
            self.config.base_log_dir,
            f"{name}_{timestamp}"
        )
        os.makedirs(log_dir, exist_ok=True)
        
        return log_dir
    
    def _setup_tracker(self, log_dir: str):
        """Setup the Tracker for agent instrumentation."""
        # 使用 auto_tracker 而不是旧的 tracker
        try:
            from .auto_tracker import TrackerConfig, TrackerManager
            
            tracker_dir = os.path.join(log_dir, "tracker_logs")
            os.makedirs(tracker_dir, exist_ok=True)
            
            TrackerConfig.configure(
                log_dir=tracker_dir,
                enabled=True,
                console_echo=False,
            )
        except ImportError:
            # 如果 auto_tracker 不可用，使用旧的 tracker
            from .tracker import Tracker
            
            tracker_dir = os.path.join(log_dir, "tracker_logs")
            os.makedirs(tracker_dir, exist_ok=True)
            
            Tracker.configure(
                log_dir=tracker_dir,
                enabled=True,
                console_echo=False,
            )
    
    def _save_tracker_data(self, log_dir: str):
        """Save all Tracker data after tournament."""
        try:
            from .auto_tracker import TrackerManager
            tracker_dir = os.path.join(log_dir, "tracker_logs")
            TrackerManager.save_all(tracker_dir)
        except Exception:
            try:
                from .tracker import Tracker
                Tracker.flush_all()
            except Exception:
                pass
    
    def _run_single_world(
        self,
        world_idx: int,
        log_dir: str,
    ) -> Optional[Dict[str, float]]:
        """Run a single world simulation."""
        from scml.std import SCML2024StdWorld
        
        world_log_dir = os.path.join(log_dir, f"world_{world_idx:03d}")
        os.makedirs(world_log_dir, exist_ok=True)
        
        try:
            # Callback
            if self.on_world_start:
                self.on_world_start(world_idx, world_log_dir)
            
            # Generate and run world
            world = SCML2024StdWorld(
                **SCML2024StdWorld.generate(
                    agent_types=self.agents,
                    n_steps=self.config.n_steps,
                    n_processes=self.config.n_processes,
                ),
                # Logging options
                log_folder=world_log_dir,
                log_to_file=True,
                log_negotiations=self.config.log_negotiations,
                save_signed_contracts=self.config.save_contracts,
                save_cancelled_contracts=self.config.save_contracts,
                save_negotiations=self.config.log_negotiations,
                save_resolved_breaches=True,
                save_unresolved_breaches=True,
                log_stats_every=self.config.log_stats_every,
            )
            
            world.run()
            
            # Collect scores
            scores = world.scores()
            
            # Map agent IDs to types
            agent_scores = {}
            for agent_id, score in scores.items():
                agent = world.agents.get(agent_id)
                if agent:
                    agent_type = type(agent).__name__
                    agent_scores[agent_type] = agent_scores.get(agent_type, [])
                    agent_scores[agent_type].append(score)
            
            # Callback
            if self.on_world_end:
                self.on_world_end(world_idx, world_log_dir, agent_scores)
            
            return agent_scores
            
        except Exception as e:
            print(f"  [ERROR] World {world_idx} failed: {e}")
            return None
    
    def run(self, verbose: bool = True) -> TournamentResults:
        """
        Run the tournament.
        
        Args:
            verbose: Print progress to console
            
        Returns:
            TournamentResults with scores and rankings
        """
        from .tracker import Tracker
        
        self._start_time = datetime.now()
        
        # Create log directory
        self._log_dir = self._create_log_dir()
        
        # Setup tracker
        self._setup_tracker(self._log_dir)
        
        if verbose:
            print("=" * 60)
            print("SCML Tournament")
            print("=" * 60)
            print(f"Name: {self.config.name or 'unnamed'}")
            print(f"Agents: {[a.__name__ for a in self.agents]}")
            print(f"Config: {self.config.n_configs} configs × {self.config.n_runs_per_world} runs × {self.config.n_steps} steps")
            print(f"Log dir: {self._log_dir}")
            print("-" * 60)
        
        # Run worlds
        all_scores = defaultdict(list)
        n_completed = 0
        n_failed = 0
        
        total_worlds = self.config.n_configs * self.config.n_runs_per_world
        
        for config_idx in range(self.config.n_configs):
            for run_idx in range(self.config.n_runs_per_world):
                world_idx = config_idx * self.config.n_runs_per_world + run_idx
                
                if verbose:
                    print(f"  Running world {world_idx + 1}/{total_worlds}...", end=" ")
                
                result = self._run_single_world(world_idx, self._log_dir)
                
                if result is not None:
                    for agent_type, scores in result.items():
                        all_scores[agent_type].extend(scores)
                    n_completed += 1
                    if verbose:
                        print("✓")
                else:
                    n_failed += 1
                    if verbose:
                        print("✗")
        
        # Flush tracker - 使用新方法
        self._save_tracker_data(self._log_dir)
        
        # Calculate duration
        duration = (datetime.now() - self._start_time).total_seconds()
        
        # Build scores DataFrame
        scores_data = []
        for agent_type, scores in all_scores.items():
            for score in scores:
                scores_data.append({
                    "agent_type": agent_type,
                    "score": score,
                })
        
        scores_df = pd.DataFrame(scores_data)
        
        # Build rankings
        if not scores_df.empty:
            rankings_df = scores_df.groupby("agent_type").agg({
                "score": ["mean", "std", "min", "max", "count"]
            }).round(4)
            rankings_df.columns = ["mean", "std", "min", "max", "count"]
            rankings_df = rankings_df.sort_values("mean", ascending=False).reset_index()
            winner = rankings_df.iloc[0]["agent_type"]
        else:
            rankings_df = pd.DataFrame()
            winner = "N/A"
        
        # Build results
        results = TournamentResults(
            name=self.config.name or "unnamed",
            timestamp=self._start_time.isoformat(),
            log_dir=self._log_dir,
            config=self.config,
            agent_types=[a.__name__ for a in self.agents],
            scores=scores_df,
            rankings=rankings_df,
            winner=winner,
            n_worlds_completed=n_completed,
            n_worlds_failed=n_failed,
            duration_seconds=duration,
        )
        
        # Save results
        results.save()
        
        if verbose:
            print("-" * 60)
            print(f"Tournament complete!")
            print(f"  Completed: {n_completed}, Failed: {n_failed}")
            print(f"  Duration: {duration:.1f}s")
            print(f"  Winner: {winner}")
            print("-" * 60)
            if not rankings_df.empty:
                print("\nRankings:")
                print(rankings_df.to_string(index=False))
        
        return results
    
    @classmethod
    def quick_test(cls, agents: List[Type], **kwargs) -> 'Tournament':
        """Create a quick test tournament."""
        return cls(agents, config=TournamentConfig.quick_test(), **kwargs)
    
    @classmethod
    def standard(cls, agents: List[Type], **kwargs) -> 'Tournament':
        """Create a standard tournament."""
        return cls(agents, config=TournamentConfig.standard(), **kwargs)
    
    @classmethod
    def full(cls, agents: List[Type], **kwargs) -> 'Tournament':
        """Create a full competition tournament."""
        return cls(agents, config=TournamentConfig.full(), **kwargs)
