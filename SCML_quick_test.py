"""
SCML Quick Test Tournament

A quick test to validate LitaAgent performance without verbose debug output.
Excludes LitaAgentCIR due to excessive debug logging.
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

import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from rich import print as rprint
from rich.table import Table
from rich.console import Console

console = Console()


def get_agents():
    """Load agents for testing - excluding verbose ones."""
    agents = []
    agent_names = []
    
    # LitaAgentY
    try:
        from litaagent_std.litaagent_y import LitaAgentY
        agents.append(LitaAgentY)
        agent_names.append("LitaAgentY")
        rprint("[green]✓[/green] LitaAgentY loaded")
    except Exception as e:
        rprint(f"[red]✗[/red] LitaAgentY failed: {e}")
    
    # LitaAgentYR
    try:
        from litaagent_std.litaagent_yr import LitaAgentYR
        agents.append(LitaAgentYR)
        agent_names.append("LitaAgentYR")
        rprint("[green]✓[/green] LitaAgentYR loaded")
    except Exception as e:
        rprint(f"[red]✗[/red] LitaAgentYR failed: {e}")
    
    # AS0 - SCML 2025 Winner
    try:
        from scml_agents.scml2025.standard.team_atsunaga import AS0
        agents.append(AS0)
        agent_names.append("AS0")
        rprint("[green]✓[/green] AS0 (2025 Winner) loaded")
    except Exception as e:
        rprint(f"[red]✗[/red] AS0 failed: {e}")
    
    # PenguinAgent - SCML 2024 Winner  
    try:
        from scml_agents.scml2024.standard.team_penguin import PenguinAgent
        agents.append(PenguinAgent)
        agent_names.append("PenguinAgent")
        rprint("[green]✓[/green] PenguinAgent (2024 Winner) loaded")
    except Exception as e:
        rprint(f"[red]✗[/red] PenguinAgent failed: {e}")
    
    return agents, agent_names


def run_quick_test(agents, n_worlds=2, n_steps=30):
    """
    Run a quick test with limited agents and steps.
    
    Args:
        agents: List of agent classes
        n_worlds: Number of worlds to simulate (default: 2)
        n_steps: Steps per simulation (default: 30)
    
    Returns:
        Tuple of (scores_df, log_dir)
    """
    from scml.std import SCML2024StdWorld
    
    rprint(f"\n[bold cyan]Quick Test Configuration:[/bold cyan]")
    rprint(f"  • Worlds: {n_worlds}")
    rprint(f"  • Steps per world: {n_steps}")
    rprint(f"  • Agents: {len(agents)}")
    
    # Create log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_base_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "tournament_logs",
        f"quick_test_{timestamp}"
    )
    os.makedirs(log_base_dir, exist_ok=True)
    rprint(f"  • Log directory: {log_base_dir}")
    
    # Run simulations
    rprint("\n[bold yellow]Running simulations...[/bold yellow]")
    
    all_scores = defaultdict(list)
    
    for world_idx in range(n_worlds):
        world_log_dir = os.path.join(log_base_dir, f"world_{world_idx}")
        os.makedirs(world_log_dir, exist_ok=True)
        
        rprint(f"  [dim]World {world_idx + 1}/{n_worlds}...[/dim]")
        
        try:
            world = SCML2024StdWorld(
                **SCML2024StdWorld.generate(
                    agent_types=agents,
                    n_steps=n_steps,
                    n_processes=2,
                ),
                log_folder=world_log_dir,
                log_to_file=True,
                save_signed_contracts=True,
                save_negotiations=True,
                log_stats_every=1,
            )
            
            world.run()
            
            # Collect scores
            scores = world.scores()
            for agent_id, score in scores.items():
                agent = world.agents.get(agent_id)
                if agent:
                    agent_type = type(agent).__name__
                    all_scores[agent_type].append(score)
                    
            rprint(f"    [green]✓ Complete[/green]")
            
        except Exception as e:
            rprint(f"    [red]✗ Failed: {e}[/red]")
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
    
    rprint(f"\n[green]✓ {n_worlds} worlds completed[/green]")
    
    return scores_df, log_base_dir


def display_results(scores_df, lita_names):
    """Display test results."""
    
    if scores_df.empty:
        rprint("[red]No results to display![/red]")
        return
    
    # Calculate statistics
    stats = scores_df.groupby("agent_type").agg({
        "score": ["mean", "std", "count", "min", "max"]
    }).round(4)
    stats.columns = ["Mean", "Std", "Count", "Min", "Max"]
    stats = stats.sort_values("Mean", ascending=False).reset_index()
    
    rprint("\n[bold green]" + "="*60 + "[/bold green]")
    rprint("[bold green]Quick Test Results[/bold green]")
    rprint("[bold green]" + "="*60 + "[/bold green]")
    
    # Results table
    table = Table(title="Agent Performance")
    table.add_column("Rank", style="cyan", justify="right")
    table.add_column("Agent", style="white")
    table.add_column("Type", style="dim")
    table.add_column("Mean Score", style="green", justify="right")
    table.add_column("Std", style="yellow", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")
    
    for rank, row in stats.iterrows():
        agent_name = row["agent_type"]
        agent_type = "[magenta]LitaAgent[/magenta]" if agent_name in lita_names else "[yellow]Top Agent[/yellow]"
        
        table.add_row(
            str(rank + 1),
            agent_name,
            agent_type,
            f"{row['Mean']:.4f}",
            f"{row['Std']:.4f}",
            f"{row['Min']:.4f}",
            f"{row['Max']:.4f}"
        )
    
    console.print(table)
    
    # Winner
    winner = stats.iloc[0]["agent_type"]
    rprint(f"\n[bold]Winner: {winner}[/bold]")
    
    # LitaAgent analysis
    lita_stats = stats[stats["agent_type"].isin(lita_names)]
    top_stats = stats[~stats["agent_type"].isin(lita_names)]
    
    if not lita_stats.empty and not top_stats.empty:
        avg_lita = lita_stats["Mean"].mean()
        avg_top = top_stats["Mean"].mean()
        
        rprint(f"\n[bold magenta]LitaAgent Analysis:[/bold magenta]")
        rprint(f"  • Best LitaAgent: {lita_stats.iloc[0]['agent_type']} ({lita_stats.iloc[0]['Mean']:.4f})")
        rprint(f"  • Average LitaAgent: {avg_lita:.4f}")
        rprint(f"  • Average Top Agent: {avg_top:.4f}")
        rprint(f"  • Gap: {(avg_top - avg_lita):.4f}")
    
    # Simple bar chart
    plt.figure(figsize=(10, 5))
    colors = ['magenta' if name in lita_names else 'steelblue' 
              for name in stats["agent_type"]]
    plt.barh(stats["agent_type"], stats["Mean"], color=colors)
    plt.xlabel("Mean Score")
    plt.ylabel("Agent")
    plt.title("Quick Test Results\n(Magenta=LitaAgent, Blue=Top Agent)")
    plt.tight_layout()
    plt.savefig("quick_test_results.png", dpi=150)
    rprint("\n[cyan]Results saved to quick_test_results.png[/cyan]")
    plt.show()
    
    return stats


def main():
    """Main function."""
    
    rprint("[bold magenta]" + "="*60 + "[/bold magenta]")
    rprint("[bold magenta]SCML Quick Test[/bold magenta]")
    rprint("[bold magenta]LitaAgent vs Top Agents[/bold magenta]")
    rprint("[bold magenta]" + "="*60 + "[/bold magenta]")
    
    # Load agents
    rprint("\n[bold]Loading agents (excluding verbose ones)...[/bold]")
    agents, agent_names = get_agents()
    
    lita_names = ["LitaAgentY", "LitaAgentYR"]
    
    if not agents:
        rprint("[red]No agents loaded! Exiting.[/red]")
        return None
    
    rprint(f"\n[bold]Agents in test: {agent_names}[/bold]")
    
    # Run quick test
    try:
        scores_df, log_dir = run_quick_test(
            agents=agents,
            n_worlds=2,
            n_steps=30,
        )
        
        # Display results
        stats = display_results(scores_df, lita_names)
        
        # Save results
        if stats is not None:
            results_file = os.path.join(log_dir, "results.json")
            stats.to_json(results_file, orient='records', indent=2)
            rprint(f"\n[cyan]Results saved to: {log_dir}[/cyan]")
        
        return scores_df, log_dir
        
    except Exception as e:
        rprint(f"[red]Test failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
