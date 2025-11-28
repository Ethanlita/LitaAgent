"""
SCML 2025 Standard Track Tournament Runner

This script runs a tournament with the top agents from SCML 2025 Standard Track
according to the official SCML 2025 competition settings from the game description document.

SCML 2025 Standard Track Final Results:
1. AS0 (1.008782) - Winner 2025
2. PenguinAgent (0.992153) - Winner 2024
3. XenoSotaAgent (0.96013) - 2nd Place 2025
4. UltraSuperMiracleSoraFinalAgentZ (0.936938) - 3rd Place 2025
5. PriceTrendStdAgent - Finalist 2025

Settings from SCML 2025 Game Description (scml2025.pdf Table 1):
- Number of simulation days: 50 < S < 200
- Reporting period: 5
- Negotiation rounds limit: 20
- Negotiation time limit: 120 seconds
- Offer time limit: 10 seconds
- Negotiation speed multiplier: 21
- Trading price parameters: γ = 0.9, Q₋₁(p) = 50
- Quantity Multiplier σ: 3
- Negotiation Horizon H: 10
- Price Range κ: 0.1
- Number of lines per factory: 10
"""

import os
import json
import warnings
from datetime import datetime
from pathlib import Path

# Suppress TensorFlow warnings and disable GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logging
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import pandas as pd
from rich import print as rprint

# Import SCML modules
from scml.std import SCML2024StdWorld
from scml.utils import anac2024_std

# Import agents from scml_agents package
import scml_agents

# Default log directory for analysis
DEFAULT_LOG_DIR = "./tournament_logs"

def get_top_agents():
    """
    Get the top agents from SCML 2025 Standard Track.
    
    Rankings:
    1. AS0 - Winner 2025
    2. PenguinAgent - Winner 2024 (also competed in 2025)
    3. XenoSotaAgent - 2nd Place 2025
    4. UltraSuperMiracleSoraFinalAgentZ - 3rd Place 2025
    5. PriceTrendStdAgent - Finalist 2025
    """
    # Get 2025 finalists
    agents_2025 = scml_agents.get_agents(2025, track="std", finalists_only=True, as_class=True)
    
    # Get 2024 winner (PenguinAgent)
    agents_2024_winners = scml_agents.get_agents(2024, track="std", winners_only=True, as_class=True)
    
    # Combine agents (avoid duplicates)
    all_agents = list(agents_2025)
    for agent in agents_2024_winners:
        if agent not in all_agents:
            all_agents.append(agent)
    
    # Filter out agents that may cause issues (e.g., TensorFlow-based agents)
    # Known problematic agents can be excluded here
    filtered_agents = []
    problematic_keywords = []  # Add keywords if specific agents cause issues
    
    for agent in all_agents:
        agent_name = agent.__name__.lower()
        if not any(kw in agent_name for kw in problematic_keywords):
            filtered_agents.append(agent)
        else:
            rprint(f"[yellow]Skipping potentially problematic agent: {agent.__name__}[/yellow]")
    
    return filtered_agents


def run_single_world_simulation(agent_types, n_steps=100, log_dir=None):
    """
    Run a single world simulation with the given agent types.
    
    Args:
        agent_types: List of agent classes to participate
        n_steps: Number of simulation days (SCML 2025: 50-200)
        log_dir: Directory to save simulation logs (for analysis)
    """
    rprint(f"[bold green]Running single world simulation with {len(agent_types)} agent types[/bold green]")
    rprint(f"Agent types: {[a.__name__ for a in agent_types]}")
    
    # Set up log directory
    if log_dir is None:
        log_dir = os.path.join(DEFAULT_LOG_DIR, f"single_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate world with SCML 2025 settings and full logging enabled
    world = SCML2024StdWorld(
        **SCML2024StdWorld.generate(
            agent_types=agent_types,
            n_steps=n_steps,
            n_processes=2,  # Standard SCML 2025 uses 2 processes (L=2)
        ),
        construct_graphs=True,
        neg_n_steps=20,  # Negotiation rounds limit
        neg_time_limit=120,  # Negotiation time limit in seconds
        neg_step_time_limit=10,  # Offer time limit in seconds
        # Enable comprehensive logging for analysis
        log_folder=log_dir,
        log_to_file=True,
        log_negotiations=True,
        save_signed_contracts=True,
        save_cancelled_contracts=True,
        save_negotiations=True,
        save_resolved_breaches=True,
        save_unresolved_breaches=True,
        saved_details_level=4,  # Maximum detail level
        log_stats_every=1,  # Log stats every step
    )
    
    rprint(f"[cyan]Logs will be saved to: {log_dir}[/cyan]")
    
    # Draw initial world state
    rprint("[bold]Initial World Configuration:[/bold]")
    _, _ = world.draw()
    plt.show()
    
    # Run simulation
    rprint("[bold yellow]Running simulation...[/bold yellow]")
    world.run_with_progress()
    
    # Display results
    rprint("\n[bold green]===== Simulation Results =====[/bold green]")
    
    # Winners
    winner_profits = [100 * world.scores()[w.id] - 100 for w in world.winners]
    winner_types = [w.short_type_name for w in world.winners]
    rprint(f"[bold]Winners: {[w.name for w in world.winners]}[/bold]")
    rprint(f"Winner types: {winner_types}")
    rprint(f"Winner profits: {winner_profits}%")
    
    # Plot statistics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    plt.sca(axes[0, 0])
    world.plot_stats("n_negotiations")
    axes[0, 0].set_title("Number of Negotiations")
    
    plt.sca(axes[0, 1])
    world.plot_stats("bankrupt", ylegend=1.25)
    axes[0, 1].set_title("Bankruptcy Status")
    
    plt.sca(axes[1, 0])
    world.plot_stats("balance", ylegend=-0.05)
    axes[1, 0].set_title("Agent Balances")
    
    plt.sca(axes[1, 1])
    world.plot_stats("score", legend=False)
    axes[1, 1].set_title("Agent Scores")
    
    plt.tight_layout()
    plt.show()
    
    # Final scores
    rprint("\n[bold]Final Scores by Agent Type:[/bold]")
    scores_df = pd.DataFrame([
        {"Agent": name, "Score": world.scores()[agent.id], "Type": agent.short_type_name}
        for name, agent in world.agents.items()
        if not name.startswith("SELLER") and not name.startswith("BUYER")
    ])
    rprint(scores_df.sort_values("Score", ascending=False).to_string())
    
    rprint(f"\n[cyan]Simulation logs saved to: {log_dir}[/cyan]")
    rprint("[cyan]Run 'python -m scml_analyzer.analyze_failures <log_dir>' to analyze agent performance[/cyan]")
    
    return world, log_dir


def run_tournament(agent_types, n_configs=5, n_runs_per_world=1, n_steps=50, log_base_dir=None):
    """
    Run a full tournament with multiple configurations.
    
    Args:
        agent_types: List of agent classes to participate
        n_configs: Number of different world configurations
        n_runs_per_world: Number of times to repeat each simulation
        n_steps: Number of simulation days per world (SCML 2025: 50-200)
        log_base_dir: Base directory for tournament logs
    """
    rprint(f"\n[bold blue]{'='*60}[/bold blue]")
    rprint(f"[bold blue]Running SCML 2025 Standard Track Tournament[/bold blue]")
    rprint(f"[bold blue]{'='*60}[/bold blue]")
    rprint(f"Competitors: {[a.__name__ for a in agent_types]}")
    rprint(f"Configurations: {n_configs}")
    rprint(f"Runs per world: {n_runs_per_world}")
    rprint(f"Steps per simulation: {n_steps}")
    
    # Set up tournament log directory
    if log_base_dir is None:
        log_base_dir = os.path.join(DEFAULT_LOG_DIR, f"tournament_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(log_base_dir, exist_ok=True)
    
    rprint(f"[cyan]Tournament logs will be saved to: {log_base_dir}[/cyan]")
    
    # World generator function with logging enabled
    def world_generator(*args, **kwargs):
        """Generate world with full logging enabled."""
        # Create unique log folder for this world
        world_id = kwargs.get('__world_id', datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        world_log_dir = os.path.join(log_base_dir, f"world_{world_id}")
        os.makedirs(world_log_dir, exist_ok=True)
        
        return dict(
            log_folder=world_log_dir,
            log_to_file=True,
            log_negotiations=True,
            save_signed_contracts=True,
            save_cancelled_contracts=True,
            save_negotiations=True,
            save_resolved_breaches=True,
            save_unresolved_breaches=True,
            saved_details_level=4,
            log_stats_every=1,
        )
    
    # Run tournament using SCML's official tournament function
    results = anac2024_std(
        competitors=agent_types,
        n_configs=n_configs,
        n_runs_per_world=n_runs_per_world,
        n_steps=n_steps,
        print_exceptions=True,
        verbose=True,
        parallelism='parallel:0.5',  # 限制并行度为 CPU 核心数的 50%，避免卡死
        # Enable logging via world_params
        world_params=dict(
            log_folder=log_base_dir,
            log_to_file=True,
            log_negotiations=True,
            save_signed_contracts=True,
            save_cancelled_contracts=True,
            save_negotiations=True,
            save_resolved_breaches=True,
            save_unresolved_breaches=True,
            saved_details_level=4,
            log_stats_every=1,
        ),
    )
    
    # Shorten agent type names for readability
    def shorten_names(df, col):
        if col in df.columns:
            df[col] = df[col].str.split(".").str[-1]
        return df
    
    results.score_stats = shorten_names(results.score_stats, "agent_type")
    results.total_scores = shorten_names(results.total_scores, "agent_type")
    results.scores = shorten_names(results.scores, "agent_type")
    results.kstest = shorten_names(results.kstest, "a")
    results.kstest = shorten_names(results.kstest, "b")
    results.winners = [w.split(".")[-1] for w in results.winners]
    
    # Display results
    rprint("\n[bold green]===== Tournament Results =====[/bold green]")
    
    rprint(f"\n[bold]Winners: {results.winners}[/bold]")
    
    rprint("\n[bold]Score Statistics:[/bold]")
    rprint(results.score_stats.to_string())
    
    rprint("\n[bold]Total Scores (Final Ranking):[/bold]")
    rprint(results.total_scores.sort_values("score", ascending=False).to_string())
    
    rprint("\n[bold]Statistical Significance (KS Test):[/bold]")
    rprint(results.kstest.to_string())
    
    # Plot score distribution
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    results.total_scores.sort_values("score", ascending=True).plot(
        kind="barh", x="agent_type", y="score", legend=False
    )
    plt.xlabel("Average Score")
    plt.ylabel("Agent Type")
    plt.title("Tournament Final Rankings")
    
    plt.subplot(1, 2, 2)
    for agent_type in results.scores["agent_type"].unique():
        agent_scores = results.scores[results.scores["agent_type"] == agent_type]["score"]
        plt.hist(agent_scores, alpha=0.5, label=agent_type, bins=10)
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.title("Score Distribution by Agent Type")
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Save tournament results summary
    summary_path = os.path.join(log_base_dir, "tournament_summary.json")
    summary_data = {
        "n_configs": n_configs,
        "n_runs_per_world": n_runs_per_world,
        "n_steps": n_steps,
        "competitors": [a.__name__ for a in agent_types],
        "winners": results.winners,
        "total_scores": results.total_scores.to_dict(),
    }
    import json
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    rprint(f"\n[cyan]Tournament logs saved to: {log_base_dir}[/cyan]")
    rprint("[cyan]Run 'python -m scml_analyzer.analyze_failures <log_dir>' to analyze agent performance[/cyan]")
    
    return results, log_base_dir


def main(run_analysis: bool = True):
    """Main function to run the SCML 2025 tournament.
    
    Args:
        run_analysis: Whether to run failure analysis after tournament
    """
    rprint("[bold magenta]SCML 2025 Standard Track Tournament[/bold magenta]")
    rprint("[bold magenta]Top Agents Competition[/bold magenta]")
    rprint("="*60)
    
    # Get top agents from SCML 2025
    top_agents = get_top_agents()
    
    rprint(f"\n[bold]Loaded {len(top_agents)} agents:[/bold]")
    for i, agent in enumerate(top_agents, 1):
        rprint(f"  {i}. {agent.__name__}")
    
    # Run full tournament with SCML 2025 official settings
    rprint("\n[bold yellow]Running full tournament with SCML 2025 settings...[/bold yellow]")
    
    # Official SCML 2025 settings
    results, log_dir = run_tournament(
        agent_types=top_agents,
        n_configs=10,        # Number of world configurations
        n_runs_per_world=2,  # Repetitions per configuration
        n_steps=100,         # Days per simulation (SCML 2025 range: 50-200)
    )
    
    rprint("\n[bold green]Tournament completed![/bold green]")
    
    # Run failure analysis if requested
    if run_analysis:
        rprint("\n[bold yellow]Running failure analysis...[/bold yellow]")
        try:
            from scml_analyzer import LogParser, FailureAnalyzer, ReportGenerator
            
            # Find world log directories
            world_dirs = [d for d in Path(log_dir).iterdir() if d.is_dir() and d.name.startswith("world_")]
            
            if world_dirs:
                rprint(f"[cyan]Found {len(world_dirs)} world logs to analyze[/cyan]")
                
                all_analysis = {}
                for world_dir in world_dirs[:5]:  # Analyze first 5 worlds
                    parser = LogParser()
                    sim_data = parser.parse_directory(str(world_dir))
                    
                    if sim_data:
                        analyzer = FailureAnalyzer(sim_data)
                        results_analysis = analyzer.analyze_all_agents()
                        all_analysis[world_dir.name] = results_analysis
                
                # Generate summary report
                report_dir = Path(log_dir) / "analysis_reports"
                report_dir.mkdir(exist_ok=True)
                
                # Save combined analysis
                analysis_summary = {
                    "worlds_analyzed": len(all_analysis),
                    "worlds": {k: {agent: len(errors) for agent, errors in v.items()} 
                              for k, v in all_analysis.items()}
                }
                
                with open(report_dir / "analysis_summary.json", 'w') as f:
                    json.dump(analysis_summary, f, indent=2)
                
                rprint(f"\n[bold green]Analysis complete![/bold green]")
                rprint(f"[cyan]Reports saved to: {report_dir}[/cyan]")
            else:
                rprint("[yellow]No world logs found for analysis[/yellow]")
                
        except ImportError:
            rprint("[yellow]scml_analyzer not found. Install it to run failure analysis.[/yellow]")
        except Exception as e:
            rprint(f"[red]Analysis failed: {e}[/red]")
    
    # 启动可视化服务器
    rprint("\n[bold cyan]🌐 启动可视化服务器...[/bold cyan]")
    try:
        from scml_analyzer.visualizer import start_server
        start_server(log_dir, port=8080, open_browser=True)
    except ImportError:
        rprint("[yellow]无法导入 scml_analyzer.visualizer[/yellow]")
    except KeyboardInterrupt:
        rprint("\n[yellow]👋 服务器已停止[/yellow]")
    except Exception as e:
        rprint(f"[red]启动服务器失败: {e}[/red]")
    
    return results, log_dir


if __name__ == "__main__":
    main()
