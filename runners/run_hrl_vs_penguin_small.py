from __future__ import annotations

import argparse
import math
import os
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Type

from runners.loky_patch import enable_loky_executor

enable_loky_executor()

from scml.utils import anac2024_std

from litaagent_std.hrl_xf.agent import LitaAgentHRL
from litaagent_std.hrl_xf.agent_ippo import LitaAgentHRLIPPOTrain


class LitaAgentHRLHeuristicL4(LitaAgentHRL):
    """Force heuristic L4 even in neural mode."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if getattr(self, "l4", None) is not None:
            self.l4.use_neural_alpha = False
            self.l4.coordinator = None


class LitaAgentHRLAlphaZero(LitaAgentHRL):
    """Force L4 alpha to zero for all threads."""

    def _compute_global_control(self):
        broadcast, alpha_map = super()._compute_global_control()
        if not alpha_map:
            return broadcast, alpha_map
        return broadcast, {k: 0.0 for k in alpha_map}


DEFAULT_L2_MODEL = Path("training_runs/l2_qlog1p_w2_20251231_220338/l2_bc.pt")
DEFAULT_L3_MODEL = Path("training_runs/l3_bc_newl2_chunk_20260105_070123/l3_bc.pt")


def _calc_max_worlds_per_config(
    target_worlds: int,
    n_configs: int,
    n_runs: int,
    n_competitors: int,
    n_per_world: int,
) -> int:
    if n_per_world >= n_competitors:
        n_sets = 1
    else:
        n_sets = math.ceil(n_competitors / n_per_world)
    denom = max(1, n_configs * n_runs * n_sets)
    return max(1, math.ceil(target_worlds / denom))


def _resolve_model_path(path_str: str) -> str:
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return str(path)


def _resolve_optional_path(path_str: str | None) -> str | None:
    if not path_str:
        return None
    return _resolve_model_path(path_str)


def _build_competitors(
    *,
    l2_model_path: str,
    l3_model_path: str,
    l3_ippo_model_path: str | None,
    l4_model_path: str | None,
    use_neural_l4: bool,
    alpha_zero: bool,
    ippo_deterministic: bool,
) -> Tuple[List[Type], List[dict]]:
    try:
        from scml_agents.scml2024.standard.team_penguin.penguinagent import PenguinAgent
    except Exception as exc:
        raise RuntimeError(f"Failed to import PenguinAgent: {exc}")

    if l3_ippo_model_path:
        hrl_cls = LitaAgentHRLIPPOTrain
    elif alpha_zero:
        hrl_cls = LitaAgentHRLAlphaZero
    else:
        hrl_cls = LitaAgentHRL if use_neural_l4 else LitaAgentHRLHeuristicL4
    competitors = [hrl_cls, PenguinAgent]
    if l3_ippo_model_path:
        controller_params = {
            "mode": "neural",
            "l2_model_path": l2_model_path,
            "l3_model_path": l3_model_path,
            "l3_ippo_model_path": l3_ippo_model_path,
            "deterministic_policy": ippo_deterministic,
        }
    else:
        controller_params = {
            "mode": "neural",
            "l2_model_path": l2_model_path,
            "l3_model_path": l3_model_path,
            "l4_model_path": l4_model_path,
        }
    competitor_params = [
        {
            "controller_params": {
                **controller_params,
            }
        },
        {},
    ]
    return competitors, competitor_params


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Small HRL vs Penguin match (no tracker, loky parallel)"
    )
    parser.add_argument("--configs", type=int, default=1, help="Number of configs")
    parser.add_argument("--runs", type=int, default=1, help="Runs per world")
    parser.add_argument("--steps", type=int, default=None, help="Override n_steps")
    parser.add_argument("--target-worlds", type=int, default=2, help="Target worlds")
    parser.add_argument("--max-worlds-per-config", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--parallelism", type=str, default="loky", help="parallel/serial/dask/loky")
    parser.add_argument("--l2-model-path", type=str, default=str(DEFAULT_L2_MODEL))
    parser.add_argument("--l3-model-path", type=str, default=str(DEFAULT_L3_MODEL))
    parser.add_argument("--l3-ippo-model-path", type=str, default=None)
    parser.add_argument("--l4-model-path", type=str, default=None)
    parser.add_argument("--use-neural-l4", action="store_true", help="Enable neural L4 (requires L4 weights)")
    parser.add_argument("--alpha-zero", action="store_true", help="Force L4 alpha to 0 for all threads")
    parser.add_argument("--ippo-deterministic", action="store_true", help="Use deterministic IPPO actions")
    args = parser.parse_args()

    l2_model_path = _resolve_model_path(args.l2_model_path)
    l3_model_path = _resolve_model_path(args.l3_model_path)
    l3_ippo_model_path = _resolve_optional_path(args.l3_ippo_model_path)
    l4_model_path = _resolve_optional_path(args.l4_model_path)

    if args.use_neural_l4 and not l4_model_path and not l3_ippo_model_path:
        print("[WARN] --use-neural-l4 set but no --l4-model-path provided; L4 will be random.")

    competitors, competitor_params = _build_competitors(
        l2_model_path=l2_model_path,
        l3_model_path=l3_model_path,
        l3_ippo_model_path=l3_ippo_model_path,
        l4_model_path=l4_model_path,
        use_neural_l4=bool(args.use_neural_l4),
        alpha_zero=bool(args.alpha_zero),
        ippo_deterministic=bool(args.ippo_deterministic),
    )

    n_competitors = len(competitors)
    n_per_world = n_competitors
    if args.max_worlds_per_config is None:
        max_worlds_per_config = _calc_max_worlds_per_config(
            args.target_worlds,
            args.configs,
            args.runs,
            n_competitors,
            n_per_world,
        )
    else:
        max_worlds_per_config = args.max_worlds_per_config
    if max_worlds_per_config < n_competitors:
        raise ValueError(
            f"max_worlds_per_config={max_worlds_per_config} must be >= {n_competitors} for fair assignment"
        )

    parallelism = args.parallelism
    if parallelism.startswith("loky"):
        os.environ["SCML_PARALLELISM"] = parallelism
        parallelism = "parallel"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or f"tournament_history/hrl_vs_penguin_{timestamp}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tournament_kwargs = {}
    if args.steps is not None:
        tournament_kwargs["n_steps"] = args.steps

    name = f"HRLvsPenguin_{timestamp}"
    anac2024_std(
        competitors=competitors,
        competitor_params=competitor_params,
        n_configs=args.configs,
        n_runs_per_world=args.runs,
        n_competitors_per_world=n_per_world,
        max_worlds_per_config=max_worlds_per_config,
        tournament_path=str(output_dir),
        name=name,
        parallelism=parallelism,
        verbose=True,
        compact=False,
        print_exceptions=True,
        **tournament_kwargs,
    )


if __name__ == "__main__":
    main()
