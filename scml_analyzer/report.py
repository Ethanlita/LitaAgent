"""
Report Generator Module

Generates various report formats from analysis results:
- JSON structured reports
- Console output with rich formatting
- Charts and visualizations (bar charts, heatmaps, timeline plots)
- HTML interactive reports (optional)
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import print as rprint
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from .analyzer import AnalysisResult, AgentAnalysisResult
from .log_parser import SimulationData, LogParser


class ReportGenerator:
    """
    Generates reports from analysis results.
    
    Usage:
        generator = ReportGenerator(results)
        
        # Console output
        generator.print_summary()
        
        # Save reports
        generator.save_json("report.json")
        generator.save_charts("./charts/")
        generator.save_html("report.html")  # Optional
    """
    
    def __init__(
        self,
        results: AnalysisResult,
        simulation_data: Optional[SimulationData] = None,
    ):
        """
        Initialize the report generator.
        
        Args:
            results: Analysis results to report on
            simulation_data: Optional simulation data for additional context
        """
        self.results = results
        self.sim_data = simulation_data
        
        # Load simulation data if not provided
        if self.sim_data is None and results.simulation_dir:
            try:
                parser = LogParser()
                self.sim_data = parser.parse_directory(results.simulation_dir)
            except:
                pass
        
        if HAS_RICH:
            self.console = Console()
    
    # ==================== Console Output ====================
    
    def print_summary(self):
        """Print a summary of the analysis to console."""
        if HAS_RICH:
            self._print_rich_summary()
        else:
            self._print_plain_summary()
    
    def _print_rich_summary(self):
        """Print summary using rich formatting."""
        console = self.console
        
        # Header
        console.print(Panel.fit(
            "[bold magenta]SCML Agent Performance Analysis Report[/bold magenta]",
            border_style="magenta"
        ))
        
        # Overview
        console.print("\n[bold cyan]📊 Overview[/bold cyan]")
        console.print(f"  Simulation Directory: {self.results.simulation_dir}")
        console.print(f"  Total Agents: {self.results.n_agents}")
        console.print(f"  Total Steps: {self.results.n_steps}")
        console.print(f"  Total Issues Detected: [bold red]{self.results.total_issues}[/bold red]")
        
        # Issues by type
        console.print("\n[bold cyan]🔍 Issues by Type[/bold cyan]")
        type_table = Table(show_header=True, header_style="bold")
        type_table.add_column("Error Type", style="cyan")
        type_table.add_column("Count", justify="right")
        type_table.add_column("Description")
        
        from .detectors import DETECTOR_REGISTRY
        for error_type, count in sorted(
            self.results.issues_by_type.items(), 
            key=lambda x: x[1], 
            reverse=True
        ):
            detector_cls = DETECTOR_REGISTRY.get(error_type.lower().replace(' ', '_'))
            desc = detector_cls.description if detector_cls else ""
            type_table.add_row(error_type, str(count), desc)
        
        console.print(type_table)
        
        # Agent summary table
        console.print("\n[bold cyan]👥 Agent Summary[/bold cyan]")
        agent_table = Table(show_header=True, header_style="bold")
        agent_table.add_column("Agent", style="cyan")
        agent_table.add_column("Type")
        agent_table.add_column("Total Issues", justify="right")
        agent_table.add_column("🔴 Critical", justify="right")
        agent_table.add_column("🟠 High", justify="right")
        agent_table.add_column("🟡 Medium", justify="right")
        agent_table.add_column("🟢 Low", justify="right")
        
        for name, agent_result in sorted(
            self.results.agent_results.items(),
            key=lambda x: x[1].total_issues,
            reverse=True
        ):
            sev = agent_result.severity_counts
            agent_table.add_row(
                name,
                agent_result.agent_type.split('.')[-1][:20],
                str(agent_result.total_issues),
                str(sev.get('critical', 0)),
                str(sev.get('high', 0)),
                str(sev.get('medium', 0)),
                str(sev.get('low', 0)),
            )
        
        console.print(agent_table)
        
        # Market context
        if 'market' in self.results.summary:
            market = self.results.summary['market']
            console.print("\n[bold cyan]📈 Market Context[/bold cyan]")
            console.print(f"  Total Contracts: {market['total_contracts']}")
            console.print(f"  Breach Rate: {market['breach_rate']:.1%}")
            console.print(f"  Negotiation Success Rate: {market['negotiation_success_rate']:.1%}")
    
    def _print_plain_summary(self):
        """Print summary without rich formatting."""
        print("=" * 60)
        print("SCML Agent Performance Analysis Report")
        print("=" * 60)
        
        print(f"\nSimulation Directory: {self.results.simulation_dir}")
        print(f"Total Agents: {self.results.n_agents}")
        print(f"Total Steps: {self.results.n_steps}")
        print(f"Total Issues Detected: {self.results.total_issues}")
        
        print("\nIssues by Type:")
        for error_type, count in sorted(
            self.results.issues_by_type.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            print(f"  {error_type}: {count}")
        
        print("\nAgent Summary:")
        print(f"{'Agent':<20} {'Issues':>8} {'Critical':>10} {'High':>8} {'Medium':>8} {'Low':>6}")
        print("-" * 60)
        for name, result in sorted(
            self.results.agent_results.items(),
            key=lambda x: x[1].total_issues,
            reverse=True
        ):
            sev = result.severity_counts
            print(f"{name[:20]:<20} {result.total_issues:>8} {sev.get('critical', 0):>10} "
                  f"{sev.get('high', 0):>8} {sev.get('medium', 0):>8} {sev.get('low', 0):>6}")
    
    def print_agent_details(self, agent_name: str):
        """Print detailed issues for a specific agent."""
        if agent_name not in self.results.agent_results:
            print(f"Agent not found: {agent_name}")
            return
        
        result = self.results.agent_results[agent_name]
        
        if HAS_RICH:
            self.console.print(f"\n[bold cyan]Agent: {agent_name}[/bold cyan]")
            self.console.print(f"Type: {result.agent_type}")
            self.console.print(f"Total Issues: {result.total_issues}")
            
            for error_type, issues in result.issues_by_type.items():
                self.console.print(f"\n[bold yellow]{error_type}[/bold yellow] ({len(issues)} issues)")
                for issue in issues[:5]:  # Show first 5
                    self.console.print(f"  • Round {issue.round}: {issue.message}")
                    for key, value in issue.details.items():
                        if key not in ['message']:
                            self.console.print(f"    - {key}: {value}")
                if len(issues) > 5:
                    self.console.print(f"  ... and {len(issues) - 5} more")
        else:
            print(f"\nAgent: {agent_name}")
            print(f"Type: {result.agent_type}")
            print(f"Total Issues: {result.total_issues}")
            
            for error_type, issues in result.issues_by_type.items():
                print(f"\n{error_type} ({len(issues)} issues)")
                for issue in issues[:5]:
                    print(f"  • Round {issue.round}: {issue.message}")
    
    # ==================== JSON Output ====================
    
    def save_json(self, path: str):
        """Save results to JSON file."""
        self.results.save_json(path)
        if HAS_RICH:
            self.console.print(f"[green]✓[/green] JSON report saved to: {path}")
        else:
            print(f"JSON report saved to: {path}")
    
    def get_json(self) -> str:
        """Get results as JSON string."""
        return self.results.to_json()
    
    # ==================== Chart Generation ====================
    
    def save_charts(self, output_dir: str, dpi: int = 150):
        """
        Generate and save all charts.
        
        Args:
            output_dir: Directory to save charts
            dpi: Resolution for saved images
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate charts
        self._save_error_type_bar_chart(output_dir, dpi)
        self._save_agent_error_heatmap(output_dir, dpi)
        self._save_severity_distribution(output_dir, dpi)
        
        if self.sim_data:
            self._save_timeline_charts(output_dir, dpi)
        
        if HAS_RICH:
            self.console.print(f"[green]✓[/green] Charts saved to: {output_dir}")
        else:
            print(f"Charts saved to: {output_dir}")
    
    def _save_error_type_bar_chart(self, output_dir: str, dpi: int):
        """Generate bar chart of error types."""
        if not self.results.issues_by_type:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        types = list(self.results.issues_by_type.keys())
        counts = list(self.results.issues_by_type.values())
        
        colors = plt.cm.Reds(np.linspace(0.4, 0.8, len(types)))
        bars = ax.barh(types, counts, color=colors)
        
        ax.set_xlabel('Issue Count')
        ax.set_title('Issues by Error Type')
        
        # Add value labels
        for bar, count in zip(bars, counts):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                   str(count), va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'error_types.png'), dpi=dpi)
        plt.close()
    
    def _save_agent_error_heatmap(self, output_dir: str, dpi: int):
        """Generate heatmap of agents vs error types."""
        if not self.results.agent_results:
            return
        
        # Build matrix
        agents = list(self.results.agent_results.keys())
        error_types = list(set(
            et for r in self.results.agent_results.values() 
            for et in r.issues_by_type.keys()
        ))
        
        if not error_types:
            return
        
        matrix = np.zeros((len(agents), len(error_types)))
        for i, agent in enumerate(agents):
            result = self.results.agent_results[agent]
            for j, et in enumerate(error_types):
                matrix[i, j] = len(result.issues_by_type.get(et, []))
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(max(10, len(error_types) * 1.5), max(8, len(agents) * 0.5)))
        
        cmap = LinearSegmentedColormap.from_list('custom', ['white', 'yellow', 'orange', 'red'])
        im = ax.imshow(matrix, cmap=cmap, aspect='auto')
        
        ax.set_xticks(range(len(error_types)))
        ax.set_yticks(range(len(agents)))
        ax.set_xticklabels(error_types, rotation=45, ha='right')
        ax.set_yticklabels([a.split('@')[0][:15] for a in agents])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Issue Count')
        
        # Add text annotations
        for i in range(len(agents)):
            for j in range(len(error_types)):
                value = int(matrix[i, j])
                if value > 0:
                    ax.text(j, i, str(value), ha='center', va='center',
                           color='black' if value < matrix.max() * 0.5 else 'white',
                           fontsize=8)
        
        ax.set_title('Agent-Error Type Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'agent_heatmap.png'), dpi=dpi)
        plt.close()
    
    def _save_severity_distribution(self, output_dir: str, dpi: int):
        """Generate pie chart of severity distribution."""
        # Aggregate severity counts
        total_severity = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for result in self.results.agent_results.values():
            for sev, count in result.severity_counts.items():
                total_severity[sev] = total_severity.get(sev, 0) + count
        
        # Filter non-zero
        labels = [k for k, v in total_severity.items() if v > 0]
        values = [v for v in total_severity.values() if v > 0]
        
        if not values:
            return
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        colors = {'critical': '#dc3545', 'high': '#fd7e14', 'medium': '#ffc107', 'low': '#28a745'}
        pie_colors = [colors.get(l, '#6c757d') for l in labels]
        
        ax.pie(values, labels=labels, colors=pie_colors, autopct='%1.1f%%',
               startangle=90, explode=[0.05] * len(values))
        ax.set_title('Issue Severity Distribution')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'severity_distribution.png'), dpi=dpi)
        plt.close()
    
    def _save_timeline_charts(self, output_dir: str, dpi: int):
        """Generate timeline charts for specific failure cases."""
        if not self.sim_data:
            return
        
        # Generate inventory timeline for agents with inventory issues
        for agent_name, result in self.results.agent_results.items():
            if 'InventoryStarvation' in result.issues_by_type:
                self._save_inventory_timeline(agent_name, output_dir, dpi)
                break  # Just one example
        
        # Generate negotiation timeline example
        for agent_name, result in self.results.agent_results.items():
            if 'Overpricing' in result.issues_by_type:
                issues = result.issues_by_type['Overpricing']
                if issues:
                    self._save_negotiation_example(agent_name, issues[0], output_dir, dpi)
                break
    
    def _save_inventory_timeline(self, agent_name: str, output_dir: str, dpi: int):
        """Generate inventory timeline chart."""
        agent_data = self.sim_data.get_agent_data(agent_name) if self.sim_data else None
        if not agent_data or not agent_data.stats:
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        steps = agent_data.stats.steps
        inv_input = agent_data.stats.inventory_input
        inv_output = agent_data.stats.inventory_output
        shortfall = agent_data.stats.shortfall_quantity
        
        ax.plot(steps, inv_input, label='Input Inventory', color='blue', linewidth=2)
        ax.plot(steps, inv_output, label='Output Inventory', color='green', linewidth=2)
        
        # Highlight shortfall events
        shortfall_steps = [s for s, q in zip(steps, shortfall) if q > 0]
        shortfall_values = [q for q in shortfall if q > 0]
        if shortfall_steps:
            ax2 = ax.twinx()
            ax2.bar(shortfall_steps, shortfall_values, alpha=0.3, color='red', 
                   label='Shortfall', width=0.8)
            ax2.set_ylabel('Shortfall Quantity', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
        
        ax.set_xlabel('Simulation Step')
        ax.set_ylabel('Inventory Level')
        ax.set_title(f'Inventory Timeline - {agent_name}')
        ax.legend(loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'inventory_timeline_{agent_name}.png'), dpi=dpi)
        plt.close()
    
    def _save_negotiation_example(
        self, 
        agent_name: str, 
        issue, 
        output_dir: str, 
        dpi: int
    ):
        """Generate negotiation price trajectory example."""
        neg_id = issue.details.get('negotiation_id')
        if not neg_id or not self.sim_data:
            return
        
        # Find negotiation
        agent_data = self.sim_data.get_agent_data(agent_name)
        if not agent_data:
            return
        
        negotiation = None
        for neg in agent_data.negotiations:
            if neg.id == neg_id:
                negotiation = neg
                break
        
        if not negotiation or not negotiation.offers:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot offer trajectories
        for partner, offers in negotiation.offers.items():
            if offers:
                prices = [o[-1] for o in offers if o]  # Price is last element
                rounds = list(range(len(prices)))
                label = f"{partner} ({'Seller' if partner == negotiation.seller else 'Buyer'})"
                color = 'red' if partner == agent_name else 'blue'
                ax.plot(rounds, prices, marker='o', label=label, color=color, linewidth=2)
        
        ax.set_xlabel('Negotiation Round')
        ax.set_ylabel('Offer Price')
        ax.set_title(f'Negotiation Price Trajectory\n(Failed - {issue.message})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        safe_neg_id = neg_id.replace('-', '_')[:20]
        plt.savefig(os.path.join(output_dir, f'negotiation_{safe_neg_id}.png'), dpi=dpi)
        plt.close()
    
    # ==================== HTML Report ====================
    
    def save_html(self, path: str):
        """
        Generate and save an interactive HTML report.
        
        Args:
            path: Output file path
        """
        html_content = self._generate_html()
        with open(path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        if HAS_RICH:
            self.console.print(f"[green]✓[/green] HTML report saved to: {path}")
        else:
            print(f"HTML report saved to: {path}")
    
    def _generate_html(self) -> str:
        """Generate HTML report content."""
        # Convert results to JSON for embedding
        results_json = self.results.to_json()
        
        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SCML Agent Analysis Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{ color: #333; }}
        h2 {{ color: #555; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
        .stat-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }}
        .stat-box {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-value {{ font-size: 2em; font-weight: bold; }}
        .stat-label {{ opacity: 0.9; }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{ background: #f8f9fa; font-weight: 600; }}
        tr:hover {{ background: #f8f9fa; }}
        .severity-critical {{ color: #dc3545; font-weight: bold; }}
        .severity-high {{ color: #fd7e14; }}
        .severity-medium {{ color: #ffc107; }}
        .severity-low {{ color: #28a745; }}
        .agent-section {{
            cursor: pointer;
            padding: 10px;
            border: 1px solid #eee;
            margin: 5px 0;
            border-radius: 4px;
        }}
        .agent-section:hover {{ background: #f8f9fa; }}
        .agent-details {{ display: none; padding: 15px; background: #f8f9fa; }}
        .chart {{ width: 100%; height: 400px; }}
    </style>
</head>
<body>
    <h1>🔍 SCML Agent Performance Analysis Report</h1>
    
    <div class="card">
        <h2>📊 Overview</h2>
        <div class="stat-grid">
            <div class="stat-box">
                <div class="stat-value">{self.results.n_agents}</div>
                <div class="stat-label">Total Agents</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{self.results.n_steps}</div>
                <div class="stat-label">Simulation Steps</div>
            </div>
            <div class="stat-box" style="background: linear-gradient(135deg, #f5576c 0%, #f093fb 100%);">
                <div class="stat-value">{self.results.total_issues}</div>
                <div class="stat-label">Issues Detected</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{len([r for r in self.results.agent_results.values() if r.total_issues > 0])}</div>
                <div class="stat-label">Agents with Issues</div>
            </div>
        </div>
    </div>
    
    <div class="card">
        <h2>📈 Issues by Type</h2>
        <div id="chart-error-types" class="chart"></div>
    </div>
    
    <div class="card">
        <h2>🗺️ Agent-Error Heatmap</h2>
        <div id="chart-heatmap" class="chart"></div>
    </div>
    
    <div class="card">
        <h2>👥 Agent Details</h2>
        {self._generate_agent_sections_html()}
    </div>
    
    <script>
        const data = {results_json};
        
        // Error types chart
        const errorTypes = Object.keys(data.issues_by_type);
        const errorCounts = Object.values(data.issues_by_type);
        
        Plotly.newPlot('chart-error-types', [{{
            x: errorCounts,
            y: errorTypes,
            type: 'bar',
            orientation: 'h',
            marker: {{ color: 'rgba(220, 53, 69, 0.8)' }}
        }}], {{
            margin: {{ l: 150, r: 50, t: 30, b: 50 }},
            xaxis: {{ title: 'Issue Count' }}
        }});
        
        // Heatmap
        const agents = Object.keys(data.agents);
        const allErrorTypes = [...new Set(
            agents.flatMap(a => Object.keys(data.agents[a].issues || {{}}))
        )];
        
        const heatmapData = agents.map(agent => 
            allErrorTypes.map(et => (data.agents[agent].issues?.[et] || []).length)
        );
        
        if (allErrorTypes.length > 0) {{
            Plotly.newPlot('chart-heatmap', [{{
                z: heatmapData,
                x: allErrorTypes,
                y: agents.map(a => a.split('@')[0]),
                type: 'heatmap',
                colorscale: 'Reds'
            }}], {{
                margin: {{ l: 100, r: 50, t: 30, b: 100 }}
            }});
        }}
        
        // Toggle agent details
        function toggleAgent(id) {{
            const details = document.getElementById('details-' + id);
            details.style.display = details.style.display === 'none' ? 'block' : 'none';
        }}
    </script>
</body>
</html>"""
        return html
    
    def _generate_agent_sections_html(self) -> str:
        """Generate HTML sections for each agent."""
        sections = []
        
        for i, (name, result) in enumerate(sorted(
            self.results.agent_results.items(),
            key=lambda x: x[1].total_issues,
            reverse=True
        )):
            sev = result.severity_counts
            
            # Issue details
            issue_rows = []
            for error_type, issues in result.issues_by_type.items():
                for issue in issues[:5]:  # Limit to 5 per type
                    severity_class = f"severity-{issue.severity}"
                    issue_rows.append(f"""
                        <tr>
                            <td>{issue.round}</td>
                            <td>{error_type}</td>
                            <td class="{severity_class}">{issue.severity}</td>
                            <td>{issue.message}</td>
                        </tr>
                    """)
            
            issues_html = f"""
                <table>
                    <tr><th>Round</th><th>Type</th><th>Severity</th><th>Message</th></tr>
                    {''.join(issue_rows) if issue_rows else '<tr><td colspan="4">No issues</td></tr>'}
                </table>
            """
            
            sections.append(f"""
                <div class="agent-section" onclick="toggleAgent({i})">
                    <strong>{name}</strong> - 
                    {result.total_issues} issues
                    (🔴 {sev.get('critical', 0)} / 🟠 {sev.get('high', 0)} / 
                     🟡 {sev.get('medium', 0)} / 🟢 {sev.get('low', 0)})
                </div>
                <div id="details-{i}" class="agent-details">
                    <p><strong>Type:</strong> {result.agent_type}</p>
                    {issues_html}
                </div>
            """)
        
        return '\n'.join(sections)


def generate_report(
    results: AnalysisResult,
    output_dir: str,
    formats: List[str] = ['json', 'console', 'charts'],
    simulation_data: Optional[SimulationData] = None,
):
    """
    Convenience function to generate reports in multiple formats.
    
    Args:
        results: Analysis results
        output_dir: Directory for output files
        formats: List of formats to generate ('json', 'console', 'charts', 'html')
        simulation_data: Optional simulation data for additional context
    """
    os.makedirs(output_dir, exist_ok=True)
    
    generator = ReportGenerator(results, simulation_data)
    
    if 'console' in formats:
        generator.print_summary()
    
    if 'json' in formats:
        generator.save_json(os.path.join(output_dir, 'analysis_report.json'))
    
    if 'charts' in formats:
        generator.save_charts(os.path.join(output_dir, 'charts'))
    
    if 'html' in formats:
        generator.save_html(os.path.join(output_dir, 'analysis_report.html'))
