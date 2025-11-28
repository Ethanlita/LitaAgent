#!/usr/bin/env python
"""
SCML Agent Failure Analysis Tool

Command-line tool for analyzing SCML simulation logs and detecting agent failures.

Usage:
    python analyze_failures.py <log_dir> [options]
    
Examples:
    # Basic analysis
    python analyze_failures.py ./logs
    
    # Full analysis with all outputs
    python analyze_failures.py ./logs -o ./reports --json --html --charts
    
    # Specific detectors only
    python analyze_failures.py ./logs --detectors overpricing,inventory_starvation
    
    # Custom thresholds
    python analyze_failures.py ./logs --config config.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scml_analyzer import (
    LogParser,
    FailureAnalyzer,
    ReportGenerator,
    analyze_simulation,
)
from scml_analyzer.detectors import DETECTOR_REGISTRY


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze SCML simulation logs for agent failures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Detectors:
  overpricing          - Detects when agents price too high
  underpricing         - Detects when agents price too low
  inventory_starvation - Detects violations due to insufficient inventory
  production_idle      - Detects underutilization of production capacity
  loss_contract        - Detects contracts that result in losses
  negotiation_stall    - Detects rigid negotiation strategies

Examples:
  %(prog)s ./logs                              # Basic analysis
  %(prog)s ./logs -o ./reports --all           # Full analysis
  %(prog)s ./logs --detectors overpricing      # Specific detector
  %(prog)s ./logs --config custom_config.json  # Custom settings
        """
    )
    
    parser.add_argument(
        "log_dir",
        type=str,
        help="Path to the directory containing simulation logs"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="./analysis_output",
        help="Output directory for reports (default: ./analysis_output)"
    )
    
    # Output format options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--json",
        action="store_true",
        help="Generate JSON report"
    )
    output_group.add_argument(
        "--html",
        action="store_true",
        help="Generate interactive HTML report"
    )
    output_group.add_argument(
        "--charts",
        action="store_true",
        help="Generate chart images"
    )
    output_group.add_argument(
        "--all",
        action="store_true",
        help="Generate all output formats (json, html, charts)"
    )
    output_group.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress console output"
    )
    
    # Detector options
    detector_group = parser.add_argument_group("Detector Options")
    detector_group.add_argument(
        "--detectors",
        type=str,
        default=None,
        help="Comma-separated list of detectors to use (default: all)"
    )
    detector_group.add_argument(
        "--list-detectors",
        action="store_true",
        help="List available detectors and exit"
    )
    
    # Configuration
    config_group = parser.add_argument_group("Configuration")
    config_group.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON configuration file with detector parameters"
    )
    config_group.add_argument(
        "--include-system-agents",
        action="store_true",
        help="Include system agents (SELLER/BUYER) in analysis"
    )
    
    # Threshold overrides
    threshold_group = parser.add_argument_group("Threshold Overrides")
    threshold_group.add_argument(
        "--excess-price-ratio",
        type=float,
        default=None,
        help="Threshold for overpricing detection (default: 0.2)"
    )
    threshold_group.add_argument(
        "--idle-steps",
        type=int,
        default=None,
        help="Consecutive idle steps threshold (default: 3)"
    )
    threshold_group.add_argument(
        "--failure-rate",
        type=float,
        default=None,
        help="Negotiation failure rate threshold (default: 0.7)"
    )
    
    # Agent filtering
    filter_group = parser.add_argument_group("Agent Filtering")
    filter_group.add_argument(
        "--agents",
        type=str,
        default=None,
        help="Comma-separated list of agent names to analyze"
    )
    filter_group.add_argument(
        "--agent-type",
        type=str,
        default=None,
        help="Filter agents by type (partial match)"
    )
    
    return parser.parse_args()


def list_detectors():
    """Print available detectors and their descriptions."""
    print("\nAvailable Error Detectors:")
    print("=" * 60)
    
    for name, cls in DETECTOR_REGISTRY.items():
        print(f"\n{name}")
        print(f"  Class: {cls.__name__}")
        print(f"  Description: {cls.description}")
        
        # Show configurable parameters
        import inspect
        sig = inspect.signature(cls.__init__)
        params = [p for p in sig.parameters.keys() if p not in ['self', 'params', 'kwargs']]
        if params:
            print(f"  Parameters: {', '.join(params)}")
    
    print("\n" + "=" * 60)


def load_config(config_path: str) -> Dict:
    """Load configuration from JSON file."""
    if not os.path.exists(config_path):
        print(f"Warning: Config file not found: {config_path}")
        return {}
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_detector_params(args) -> Dict[str, Dict]:
    """Build detector parameters from arguments."""
    params = {}
    
    # Load from config file if provided
    if args.config:
        config = load_config(args.config)
        params = config.get('detector_params', {})
    
    # Apply command-line overrides
    if args.excess_price_ratio is not None:
        params.setdefault('overpricing', {})['excess_price_ratio'] = args.excess_price_ratio
        params.setdefault('underpricing', {})['undercut_price_ratio'] = args.excess_price_ratio
    
    if args.idle_steps is not None:
        params.setdefault('production_idle', {})['idle_steps_threshold'] = args.idle_steps
    
    if args.failure_rate is not None:
        params.setdefault('negotiation_stall', {})['failure_rate_threshold'] = args.failure_rate
    
    return params


def main():
    """Main entry point."""
    args = parse_args()
    
    # Handle --list-detectors
    if args.list_detectors:
        list_detectors()
        return 0
    
    # Validate log directory
    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"Error: Log directory not found: {log_dir}")
        return 1
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which detectors to use
    detector_names = None
    if args.detectors:
        detector_names = [d.strip() for d in args.detectors.split(',')]
        # Validate detector names
        invalid = [d for d in detector_names if d not in DETECTOR_REGISTRY]
        if invalid:
            print(f"Error: Unknown detectors: {invalid}")
            print(f"Available: {list(DETECTOR_REGISTRY.keys())}")
            return 1
    
    # Build detector parameters
    detector_params = build_detector_params(args)
    
    # Determine output formats
    if args.all:
        formats = ['json', 'console', 'charts', 'html']
    else:
        formats = []
        if not args.quiet:
            formats.append('console')
        if args.json:
            formats.append('json')
        if args.html:
            formats.append('html')
        if args.charts:
            formats.append('charts')
        
        # Default to console + json if no format specified
        if not formats:
            formats = ['console', 'json']
    
    try:
        # Print header
        if not args.quiet:
            print("\n" + "=" * 60)
            print("SCML Agent Failure Analysis")
            print("=" * 60)
            print(f"Log Directory: {log_dir}")
            print(f"Output Directory: {output_dir}")
            if detector_names:
                print(f"Detectors: {', '.join(detector_names)}")
            else:
                print(f"Detectors: All ({len(DETECTOR_REGISTRY)})")
            print()
        
        # Run analysis
        analyzer = FailureAnalyzer(
            detector_names=detector_names,
            detector_params=detector_params,
            exclude_system_agents=not args.include_system_agents,
        )
        
        results = analyzer.analyze(str(log_dir))
        
        # Filter by agent name if specified
        if args.agents:
            agent_filter = [a.strip() for a in args.agents.split(',')]
            results.agent_results = {
                k: v for k, v in results.agent_results.items()
                if k in agent_filter
            }
        
        # Filter by agent type if specified
        if args.agent_type:
            results.agent_results = {
                k: v for k, v in results.agent_results.items()
                if args.agent_type.lower() in v.agent_type.lower()
            }
        
        # Load simulation data for reports
        parser = LogParser(exclude_system_agents=not args.include_system_agents)
        sim_data = parser.parse_directory(str(log_dir))
        
        # Generate reports
        generator = ReportGenerator(results, sim_data)
        
        if 'console' in formats:
            generator.print_summary()
        
        if 'json' in formats:
            json_path = output_dir / 'analysis_report.json'
            generator.save_json(str(json_path))
        
        if 'charts' in formats:
            charts_dir = output_dir / 'charts'
            generator.save_charts(str(charts_dir))
        
        if 'html' in formats:
            html_path = output_dir / 'analysis_report.html'
            generator.save_html(str(html_path))
        
        # Print summary
        if not args.quiet:
            print(f"\n{'=' * 60}")
            print(f"Analysis Complete!")
            print(f"Total Issues Found: {results.total_issues}")
            print(f"Agents Analyzed: {len(results.agent_results)}")
            print(f"Output saved to: {output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
