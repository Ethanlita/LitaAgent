from __future__ import annotations

import argparse
import math
import os
from datetime import datetime
from pathlib import Path
from typing import List, Type

from runners.loky_patch import enable_loky_executor

enable_loky_executor()

from scml.utils import anac2024_std

from litaagent_std.litaagent_h import LitaAgentH
from litaagent_std.litaagent_hs import LitaAgentHS
from litaagent_std.tracker_mixin import create_tracked_agent

try:
    from scml_analyzer.auto_tracker import TrackerConfig

    _TRACKER_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TRACKER_AVAILABLE = False
    TrackerConfig = None


def _calc_max_worlds_per_config(
    target_worlds: int,
    n_configs: int,
    n_runs: int,
    n_competitors: int,
    n_per_world: int,
    round_robin: bool,
) -> int:
    if n_per_world >= n_competitors:
        n_sets = 1
    elif round_robin:
        n_sets = math.comb(n_competitors, n_per_world)
    else:
        n_sets = math.ceil(n_competitors / n_per_world)
    denom = max(1, n_configs * n_runs * n_sets)
    return max(1, math.ceil(target_worlds / denom))


def _build_competitors(tracker_log_dir: str) -> List[Type]:
    try:
        from scml_agents.scml2024.standard.team_penguin.penguinagent import PenguinAgent
    except Exception as exc:
        raise RuntimeError(f"Failed to import PenguinAgent: {exc}")

    tracked_penguin = create_tracked_agent(PenguinAgent, log_dir=tracker_log_dir)
    tracked_litah = create_tracked_agent(LitaAgentH, log_dir=tracker_log_dir)
    tracked_litahs = create_tracked_agent(LitaAgentHS, log_dir=tracker_log_dir)

    return [
        tracked_penguin,
        tracked_litah,
        tracked_litahs,
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Small round-robin runner for Penguin/LitaAgentH/LitaAgentHS"
    )
    parser.add_argument("--configs", type=int, default=1, help="Number of configs")
    parser.add_argument("--runs", type=int, default=1, help="Runs per world")
    parser.add_argument("--steps", type=int, default=None, help="Override n_steps")
    parser.add_argument(
        "--target-worlds", type=int, default=200, help="Target worlds (small scale)"
    )
    parser.add_argument(
        "--max-worlds-per-config",
        type=int,
        default=None,
        help="Override max worlds per config",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--parallelism", type=str, default="loky", help="parallel/serial/dask/loky")
    parser.add_argument(
        "--forced-logs-fraction", type=float, default=0.1, help="Force log fraction"
    )
    parser.add_argument("--no-csv", action="store_true", help="Disable CSV/negotiation logs")
    args = parser.parse_args()

    if not _TRACKER_AVAILABLE:
        raise RuntimeError("scml_analyzer is required for tracker logs")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or f"tournament_history/hrl_rr_small_phh_{timestamp}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tracker_dir = output_dir / "tracker_logs"
    tracker_dir.mkdir(parents=True, exist_ok=True)
    TrackerConfig.configure(log_dir=str(tracker_dir), enabled=True)
    os.environ["SCML_TRACKER_LOG_DIR"] = str(tracker_dir)

    competitors = _build_competitors(str(tracker_dir))
    n_competitors = len(competitors)
    n_per_world = n_competitors
    round_robin = True

    if args.max_worlds_per_config is None:
        max_worlds_per_config = _calc_max_worlds_per_config(
            args.target_worlds,
            args.configs,
            args.runs,
            n_competitors,
            n_per_world,
            round_robin,
        )
    else:
        max_worlds_per_config = args.max_worlds_per_config

    parallelism = args.parallelism
    if parallelism.startswith("loky"):
        os.environ["SCML_PARALLELISM"] = parallelism
        parallelism = "parallel"

    tournament_kwargs = {}
    if args.steps is not None:
        tournament_kwargs["n_steps"] = args.steps
    if args.no_csv:
        tournament_kwargs.update(
            {
                "log_ufuns": False,
                "log_negotiations": False,
                "save_signed_contracts": True,
                "save_cancelled_contracts": False,
                "save_negotiations": False,
                "save_resolved_breaches": False,
                "save_unresolved_breaches": False,
                "saved_details_level": 0,
                "log_stats_every": 0,
            }
        )

    name = f"HRLRRSmallPHH_{timestamp}"
    anac2024_std(
        competitors=competitors,
        n_configs=args.configs,
        n_runs_per_world=args.runs,
        n_competitors_per_world=n_per_world,
        max_worlds_per_config=max_worlds_per_config,
        tournament_path=str(output_dir),
        round_robin=round_robin,
        name=name,
        forced_logs_fraction=args.forced_logs_fraction,
        parallelism=parallelism,
        verbose=True,
        compact=False,
        print_exceptions=True,
        **tournament_kwargs,
    )


if __name__ == "__main__":
    main()
