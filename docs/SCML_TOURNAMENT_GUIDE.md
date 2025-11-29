# SCML 2025 Tournament System Guide

This document describes the architecture and usage of the SCML 2025 tournament runner and analysis system.

## 1. System Architecture

The system consists of three main components:
1.  **Runners**: Scripts to execute SCML tournaments (`runners/`).
2.  **Tracker**: Automatic data collection system (`scml_analyzer/auto_tracker.py`).
3.  **Analyzer**: Data processing and visualization tools (`scml_analyzer/`).

### Directory Structure

```
LitaAgent/
├── runners/                  # Tournament execution scripts
│   ├── run_std_quick.py      # Quick test (Parallel, ~10 mins)
│   ├── run_std_full.py       # Full tournament (Official settings)
│   └── run_oneshot_full.py   # OneShot track runner
├── scml_analyzer/            # Analysis tools
│   ├── auto_tracker.py       # Data collection engine
│   ├── postprocess.py        # Data import & cleanup
│   ├── history.py            # Tournament history manager
│   └── visualizer.py         # Web-based dashboard
├── litaagent_std/            # Agent implementations
└── tournament_history/       # Archived tournament data
```

## 2. Running Tournaments

### Quick Test (Recommended for Development)
Runs a short tournament with LitaAgents and built-in agents.
- **Mode**: Parallel (Fast, with progress bar)
- **Steps**: 50
- **Configs**: 3
- **Output**: `results/std_quick_<timestamp>/`

```bash
python runners/run_std_quick.py
```

### Full Tournament (Official Settings)
Runs a full-scale tournament mimicking official SCML 2025 settings.
- **Mode**: Parallel
- **Steps**: 50-200 (Variable)
- **Configs**: 20
- **Output**: `results/std_full_<timestamp>/`

```bash
python runners/run_std_full.py
```

## 3. Tracker System

The `AutoTracker` system automatically collects detailed data from agents without manual logging calls.

### How it works
1.  **Injection**: Agents are decorated with `TrackedAgent` mixin.
2.  **Collection**: 
    - `negotiation_*`: Offers, acceptances, rejections.
    - `contract_*`: Signed contracts, execution, breaches.
    - `inventory_*`: Stock levels, balance (per step).
    - `production_*`: Scheduled and executed production.
3.  **Persistence**:
    - **Serial Mode**: Data is held in memory and saved at the end.
    - **Parallel Mode**: Each agent saves its own log to JSON at the last step of the simulation.
    - **Post-processing**: Individual JSON files are aggregated into `tracker_summary.json`.

### Usage in Agents
To track a new agent, simply inherit from `TrackedAgent` (or use `inject_tracker_to_agents` helper):

```python
from scml_analyzer.auto_tracker import TrackedAgent

class MyAgent(TrackedAgent, StdSyncAgent):
    def step(self):
        super().step()
        self.log("custom_event", value=123)
```

## 4. Analysis & Visualization

After a tournament finishes, the system automatically:
1.  **Saves Logs**: Collects all tracker data.
2.  **Imports History**: Moves data to `tournament_history/<id>/`.
3.  **Starts Visualizer**: Launches a local web server.

### Manual Visualization
If you want to view results later:

```bash
# Start the visualizer (views all tournaments in tournament_history)
python -m scml_analyzer.visualizer
```

### Data Export
Data is stored in standard formats in `tournament_history/<id>/`:
- `tracker_logs/*.json`: Detailed agent logs.
- `total_scores.csv`: Final rankings.
- `world_stats.csv`: Simulation statistics.

## 5. Troubleshooting

### Missing Progress Bar
- Ensure `parallelism='parallel'` is set in the runner.
- `serial` mode disables the `rich` progress bar provided by NegMas.

### Missing Data in Parallel Mode
- The system relies on agents saving their own data at the end of the simulation.
- Ensure `TrackedAgent.step()` is called and reaches the last step.
- `TrackerManager.rebuild_summary()` is used to reconstruct the index from individual files.

### Deadlocks
- If the simulation hangs, try setting `verbose=False` in `anac2024_std`.
- High-volume logging in parallel processes can cause pipe buffer deadlocks.
