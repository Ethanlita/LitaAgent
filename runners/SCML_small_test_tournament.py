"""
SCML Small Scale Test Tournament

A smaller tournament to test LitaAgent variants against top SCML agents.
Uses the scml_analyzer toolkit to track and analyze agent performance.

LitaAgent Variants:
- LitaAgentY: Base variant
- LitaAgentYR: Enhanced variant with dynamic profit margin
- LitaAgentN: N variant
- LitaAgentP: P variant
- LitaAgentCIR: CIR variant (with circular inventory)

Top SCML Agents (for comparison):
- AS0: SCML 2025 Winner
- PenguinAgent: SCML 2024 Winner
"""

import os
import json
import warnings
from datetime import datetime
from pathlib import Path

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import matplotlib.pyplot as plt
from rich import print as rprint
from rich.table import Table
from rich.console import Console

console = Console()


def get_lita_agents():
    """Load all working LitaAgent variants."""
    agents = []
    
    # LitaAgentY
    try:
        from litaagent_std.litaagent_y import LitaAgentY
        agents.append(LitaAgentY)
        rprint("[green]✓[/green] LitaAgentY loaded")
    except Exception as e:
        rprint(f"[red]✗[/red] LitaAgentY failed: {e}")
    
    # LitaAgentYR
    try:
        from litaagent_std.litaagent_yr import LitaAgentYR
        agents.append(LitaAgentYR)
        rprint("[green]✓[/green] LitaAgentYR loaded")
    except Exception as e:
        rprint(f"[red]✗[/red] LitaAgentYR failed: {e}")
    
    # LitaAgentN
    try:
        from litaagent_std.litaagent_n import LitaAgentN
        agents.append(LitaAgentN)
        rprint("[green]✓[/green] LitaAgentN loaded")
    except Exception as e:
        rprint(f"[red]✗[/red] LitaAgentN failed: {e}")
    
    # LitaAgentP
    try:
        from litaagent_std.litaagent_p import LitaAgentP
        agents.append(LitaAgentP)
        rprint("[green]✓[/green] LitaAgentP loaded")
    except Exception as e:
        rprint(f"[red]✗[/red] LitaAgentP failed: {e}")
    
    # LitaAgentCIR
    try:
        from litaagent_std.litaagent_cir import LitaAgentCIR
        agents.append(LitaAgentCIR)
        rprint("[green]✓[/green] LitaAgentCIR loaded")
    except Exception as e:
        rprint(f"[red]✗[/red] LitaAgentCIR failed: {e}")
    
    return agents


def get_top_agents():
    """Load top SCML agents for comparison."""
    agents = []
    
    # AS0 - SCML 2025 Winner
    try:
        from scml_agents.scml2025.standard.team_atsunaga import AS0
        agents.append(AS0)
        rprint("[green]✓[/green] AS0 (2025 Winner) loaded")
    except Exception as e:
        rprint(f"[red]✗[/red] AS0 failed: {e}")
    
    # PenguinAgent - SCML 2024 Winner
    try:
        from scml_agents.scml2024.standard.team_penguin import PenguinAgent
        agents.append(PenguinAgent)
        rprint("[green]✓[/green] PenguinAgent (2024 Winner) loaded")
    except Exception as e:
        rprint(f"[red]✗[/red] PenguinAgent failed: {e}")
    
    return agents


def run_small_tournament(
    lita_agents: list,
    top_agents: list,
    n_configs: int = 3,
    n_runs_per_world: int = 1,
    n_steps: int = 50,
):
    """
    Run a small-scale tournament with LitaAgents and top agents.
    
    Args:
        lita_agents: List of LitaAgent classes
        top_agents: List of top SCML agent classes
        n_configs: Number of world configurations (default: 3)
        n_runs_per_world: Number of runs per configuration (default: 1)
        n_steps: Number of simulation steps/days (default: 50)
    
    Returns:
        Tuple of (results dataframe, log_dir)
    """
    from scml.std import SCML2024StdWorld
    import pandas as pd
    from collections import defaultdict
    
    # Combine all agents
    all_agents = lita_agents + top_agents
    
    rprint(f"\n[bold cyan]Tournament Configuration:[/bold cyan]")
    rprint(f"  • Configurations: {n_configs}")
    rprint(f"  • Runs per config: {n_runs_per_world}")
    rprint(f"  • Steps per run: {n_steps}")
    rprint(f"  • Total agents: {len(all_agents)}")
    
    # Create log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_base_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "tournament_logs",
        f"small_test_{timestamp}"
    )
    os.makedirs(log_base_dir, exist_ok=True)
    
    rprint(f"  • Log directory: {log_base_dir}")
    
    # Run tournament manually
    rprint("\n[bold yellow]Starting tournament...[/bold yellow]")
    
    all_scores = defaultdict(list)
    total_runs = n_configs * n_runs_per_world
    run_count = 0
    
    for config_idx in range(n_configs):
        for run_idx in range(n_runs_per_world):
            run_count += 1
            world_log_dir = os.path.join(log_base_dir, f"world_{config_idx}_{run_idx}")
            os.makedirs(world_log_dir, exist_ok=True)
            
            rprint(f"  [dim]Running world {run_count}/{total_runs}...[/dim]")
            
            try:
                # Generate world with all agents
                world = SCML2024StdWorld(
                    **SCML2024StdWorld.generate(
                        agent_types=all_agents,
                        n_steps=n_steps,
                        n_processes=2,
                    ),
                    log_folder=world_log_dir,
                    log_to_file=True,
                    log_negotiations=True,
                    save_signed_contracts=True,
                    save_cancelled_contracts=True,
                    save_negotiations=True,
                    save_resolved_breaches=True,
                    save_unresolved_breaches=True,
                    log_stats_every=1,
                )
                
                # Run the world
                world.run()
                
                # Collect scores
                scores = world.scores()
                for agent_id, score in scores.items():
                    # Get agent type name
                    agent = world.agents.get(agent_id)
                    if agent:
                        agent_type = type(agent).__name__
                        all_scores[agent_type].append(score)
                        
            except Exception as e:
                rprint(f"    [red]World {run_count} failed: {e}[/red]")
                continue
    
    # Create results DataFrame
    results_data = []
    for agent_type, scores in all_scores.items():
        for score in scores:
            results_data.append({
                "agent_type": agent_type,
                "score": score
            })
    
    scores_df = pd.DataFrame(results_data)
    
    # Calculate total scores (average per agent type)
    total_scores = scores_df.groupby("agent_type").agg({
        "score": ["mean", "std", "count"]
    }).reset_index()
    total_scores.columns = ["agent_type", "score", "std", "count"]
    
    # Create a results object similar to tournament results
    class TournamentResults:
        def __init__(self, scores_df, total_scores):
            self.scores = scores_df
            self.total_scores = total_scores
            self.score_stats = total_scores.copy()
            self.winners = total_scores.nlargest(1, "score")["agent_type"].tolist()
            self.kstest = pd.DataFrame()  # Empty for now
    
    results = TournamentResults(scores_df, total_scores)
    
    rprint(f"  [green]Tournament completed! {run_count} worlds simulated.[/green]")
    
    return results, log_base_dir


def display_results(results, lita_agents, top_agents):
    """Display tournament results with focus on LitaAgent performance."""
    
    # Shorten agent type names
    def shorten_names(df, col):
        if col in df.columns:
            df[col] = df[col].str.split(".").str[-1]
        return df
    
    results.score_stats = shorten_names(results.score_stats, "agent_type")
    results.total_scores = shorten_names(results.total_scores, "agent_type")
    results.scores = shorten_names(results.scores, "agent_type")
    results.winners = [w.split(".")[-1] for w in results.winners]
    
    # Get agent names
    lita_names = [a.__name__ for a in lita_agents]
    top_names = [a.__name__ for a in top_agents]
    
    rprint("\n[bold green]" + "="*60 + "[/bold green]")
    rprint("[bold green]Tournament Results[/bold green]")
    rprint("[bold green]" + "="*60 + "[/bold green]")
    
    rprint(f"\n[bold]Winners: {results.winners}[/bold]")
    
    # Create results table
    table = Table(title="Final Rankings")
    table.add_column("Rank", style="cyan", justify="right")
    table.add_column("Agent", style="white")
    table.add_column("Type", style="dim")
    table.add_column("Score", style="green", justify="right")
    
    sorted_scores = results.total_scores.sort_values("score", ascending=False)
    for rank, (idx, row) in enumerate(sorted_scores.iterrows(), 1):
        agent_name = row["agent_type"]
        if agent_name in lita_names:
            agent_type = "[magenta]LitaAgent[/magenta]"
        elif agent_name in top_names:
            agent_type = "[yellow]Top Agent[/yellow]"
        else:
            agent_type = "Unknown"
        
        table.add_row(
            str(rank),
            agent_name,
            agent_type,
            f"{row['score']:.4f}"
        )
    
    console.print(table)
    
    # LitaAgent specific analysis
    rprint("\n[bold magenta]LitaAgent Performance Analysis:[/bold magenta]")
    
    lita_scores = results.total_scores[
        results.total_scores["agent_type"].isin(lita_names)
    ].sort_values("score", ascending=False)
    
    if not lita_scores.empty:
        best_lita = lita_scores.iloc[0]
        rprint(f"  • Best LitaAgent: {best_lita['agent_type']} (score: {best_lita['score']:.4f})")
        
        # Compare with top agents
        top_scores = results.total_scores[
            results.total_scores["agent_type"].isin(top_names)
        ]
        if not top_scores.empty:
            avg_top = top_scores["score"].mean()
            avg_lita = lita_scores["score"].mean()
            rprint(f"  • Average LitaAgent score: {avg_lita:.4f}")
            rprint(f"  • Average Top Agent score: {avg_top:.4f}")
            rprint(f"  • Performance gap: {(avg_top - avg_lita):.4f}")
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    # Bar chart
    plt.subplot(1, 2, 1)
    colors = ['magenta' if name in lita_names else 'steelblue' 
              for name in sorted_scores["agent_type"]]
    sorted_scores.sort_values("score", ascending=True).plot(
        kind="barh", x="agent_type", y="score", legend=False, color=colors[::-1]
    )
    plt.xlabel("Average Score")
    plt.ylabel("Agent")
    plt.title("Tournament Rankings\n(Magenta=LitaAgent, Blue=Top Agent)")
    
    # Score distribution
    plt.subplot(1, 2, 2)
    for agent_type in results.scores["agent_type"].unique():
        agent_scores = results.scores[results.scores["agent_type"] == agent_type]["score"]
        color = 'magenta' if agent_type in lita_names else 'steelblue'
        alpha = 0.7 if agent_type in lita_names else 0.4
        plt.hist(agent_scores, alpha=alpha, label=agent_type, bins=10, color=color)
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.title("Score Distribution")
    plt.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig("tournament_results.png", dpi=150)
    rprint("\n[cyan]Results plot saved to tournament_results.png[/cyan]")
    plt.show()
    
    return sorted_scores


def run_analysis(log_dir: str, focus_agents: list = None):
    """Run failure analysis on tournament logs."""
    
    rprint("\n[bold yellow]Running Failure Analysis...[/bold yellow]")
    
    try:
        from scml_analyzer import LogParser, FailureAnalyzer, ReportGenerator
        
        # Find world directories
        log_path = Path(log_dir)
        world_dirs = list(log_path.glob("**/contracts.csv"))
        
        if not world_dirs:
            # Try to find in subdirectories
            world_dirs = list(log_path.glob("contracts.csv"))
        
        rprint(f"[cyan]Found {len(world_dirs)} simulation logs[/cyan]")
        
        all_results = {}
        
        for contracts_file in world_dirs[:10]:  # Analyze up to 10 worlds
            world_dir = contracts_file.parent
            rprint(f"  Analyzing: {world_dir.name}")
            
            parser = LogParser()
            sim_data = parser.parse_directory(str(world_dir))
            
            if sim_data:
                analyzer = FailureAnalyzer(sim_data)
                
                # Focus on specified agents if provided
                if focus_agents:
                    for agent_name in focus_agents:
                        # Find matching agent in simulation
                        matching = [a for a in sim_data.get_agent_names() 
                                   if agent_name in a]
                        for match in matching:
                            errors = analyzer.analyze_agent(match)
                            if errors:
                                key = f"{world_dir.name}/{match}"
                                all_results[key] = errors
                else:
                    results = analyzer.analyze_all_agents()
                    for agent, errors in results.items():
                        if errors:
                            key = f"{world_dir.name}/{agent}"
                            all_results[key] = errors
        
        # Generate report
        if all_results:
            report_dir = log_path / "analysis_reports"
            report_dir.mkdir(exist_ok=True)
            
            # Summary
            rprint("\n[bold]Failure Analysis Summary:[/bold]")
            
            error_counts = {}
            for key, errors in all_results.items():
                for error in errors:
                    error_type = error.error_type
                    error_counts[error_type] = error_counts.get(error_type, 0) + 1
            
            for error_type, count in sorted(error_counts.items(), key=lambda x: -x[1]):
                rprint(f"  • {error_type}: {count} instances")
            
            # Save detailed report
            report_file = report_dir / "failure_analysis.json"
            report_data = {
                "summary": error_counts,
                "details": {k: [{"type": e.error_type, "severity": e.severity, 
                                "description": e.description} 
                               for e in v] 
                          for k, v in all_results.items()}
            }
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            rprint(f"\n[cyan]Detailed report saved to: {report_file}[/cyan]")
        else:
            rprint("[yellow]No failures detected in analyzed simulations[/yellow]")
            
    except ImportError as e:
        rprint(f"[red]Analysis module not available: {e}[/red]")
    except Exception as e:
        rprint(f"[red]Analysis failed: {e}[/red]")
        import traceback
        traceback.print_exc()


def main():
    """Main function to run the small test tournament."""
    
    rprint("[bold magenta]" + "="*60 + "[/bold magenta]")
    rprint("[bold magenta]SCML Small Scale Test Tournament[/bold magenta]")
    rprint("[bold magenta]LitaAgent vs Top Agents[/bold magenta]")
    rprint("[bold magenta]" + "="*60 + "[/bold magenta]")
    
    # Load agents
    rprint("\n[bold]Loading LitaAgent variants...[/bold]")
    lita_agents = get_lita_agents()
    
    rprint("\n[bold]Loading Top SCML agents...[/bold]")
    top_agents = get_top_agents()
    
    if not lita_agents:
        rprint("[red]No LitaAgents could be loaded! Exiting.[/red]")
        return None
    
    if not top_agents:
        rprint("[yellow]Warning: No top agents loaded, running LitaAgents only[/yellow]")
    
    # Show all agents
    rprint("\n[bold]Agents participating in tournament:[/bold]")
    all_agent_names = []
    for agent in lita_agents:
        rprint(f"  [magenta]• {agent.__name__}[/magenta] (LitaAgent)")
        all_agent_names.append(agent.__name__)
    for agent in top_agents:
        rprint(f"  [yellow]• {agent.__name__}[/yellow] (Top Agent)")
        all_agent_names.append(agent.__name__)
    
    # Run tournament
    rprint("\n[bold yellow]Starting Small Tournament...[/bold yellow]")
    rprint("[dim]Settings: 3 configs, 1 run each, 50 steps[/dim]")
    
    try:
        results, log_dir = run_small_tournament(
            lita_agents=lita_agents,
            top_agents=top_agents,
            n_configs=3,
            n_runs_per_world=1,
            n_steps=50,
        )
        
        rprint("\n[bold green]Tournament completed![/bold green]")
        
        # Display results
        sorted_scores = display_results(results, lita_agents, top_agents)
        
        # Save results to log directory
        results_file = os.path.join(log_dir, "tournament_results.json")
        results_data = {
            "timestamp": datetime.now().isoformat(),
            "lita_agents": [a.__name__ for a in lita_agents],
            "top_agents": [a.__name__ for a in top_agents],
            "winners": results.winners,
            "rankings": sorted_scores.to_dict(orient='records'),
        }
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Run failure analysis focused on LitaAgents
        lita_names = [a.__name__ for a in lita_agents]
        run_analysis(log_dir, focus_agents=lita_names)
        
        rprint(f"\n[bold cyan]All logs and reports saved to:[/bold cyan]")
        rprint(f"  {log_dir}")
        
        # 导入数据到 tournament_history 并启动可视化服务器
        rprint("\n[bold cyan]🌐 导入数据并启动可视化服务器...[/bold cyan]")
        try:
            from scml_analyzer.history import import_tournament
            from scml_analyzer.visualizer import start_server
            
            # 直接从 log_dir 导入（这些 runner 不使用 negmas tournament API）
            tournament_id = import_tournament(log_dir, copy_mode=False)
            if tournament_id:
                rprint(f"[green]✓ 数据已导入: {tournament_id}[/green]")
            
            # 启动无参数可视化服务器
            start_server(port=8080, open_browser=True)
        except ImportError as e:
            rprint(f"[yellow]无法导入模块: {e}[/yellow]")
        except KeyboardInterrupt:
            rprint("\n[yellow]👋 服务器已停止[/yellow]")
        except Exception as e:
            rprint(f"[red]启动服务器失败: {e}[/red]")
            import traceback
            traceback.print_exc()
        
        return results, log_dir
        
    except Exception as e:
        rprint(f"[red]Tournament failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
