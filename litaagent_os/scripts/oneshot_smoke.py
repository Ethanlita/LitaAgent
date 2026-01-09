from __future__ import annotations

import importlib.util
import os
import sys
import types
from datetime import datetime

from scml.utils import anac2024_oneshot
from scml.oneshot.agents import SyncRandomOneShotAgent

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from litaagent_os.agent import LitaAgentOS
from litaagent_std.tracker_mixin import (
    TRACKER_AVAILABLE,
    create_tracked_agent,
    save_tracker_data,
    setup_tracker_for_tournament,
)


def _ensure_package(pkg_name: str, pkg_path: str) -> None:
    if pkg_name in sys.modules:
        return
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [pkg_path]
    pkg.__package__ = pkg_name
    sys.modules[pkg_name] = pkg


def _prepare_packages(module_name: str, file_path: str) -> None:
    parts = module_name.split(".")
    pkg_parts = parts[:-1]
    path_parts = os.path.abspath(file_path).split(os.sep)
    if "scml_agents" not in path_parts or not pkg_parts:
        return
    base_idx = path_parts.index("scml_agents")
    base_dir = os.sep.join(path_parts[: base_idx + 1])
    _ensure_package("scml_agents", base_dir)
    current_path = base_dir
    for i in range(1, len(pkg_parts)):
        current_path = os.path.join(current_path, pkg_parts[i])
        _ensure_package(".".join(pkg_parts[: i + 1]), current_path)


def load_agent_class(file_path: str, module_name: str, class_name: str):
    abs_path = os.path.abspath(file_path)
    _prepare_packages(module_name, abs_path)
    spec = importlib.util.spec_from_file_location(module_name, abs_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块: {abs_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, class_name):
        raise ImportError(f"模块中未找到类: {class_name} ({abs_path})")
    return getattr(module, class_name)


def main() -> None:
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    tournament_path = os.path.join(ROOT, "run_logs", f"lita_os_smoke_{run_tag}")
    tracker_dir = os.path.join(tournament_path, "tracker_logs")
    cautious_path = os.path.join(
        ROOT,
        ".venv",
        "Lib",
        "site-packages",
        "scml_agents",
        "scml2024",
        "oneshot",
        "team_miyajima_oneshot",
        "cautious.py",
    )
    hori_path = os.path.join(
        ROOT,
        ".venv",
        "Lib",
        "site-packages",
        "scml_agents",
        "scml2025",
        "oneshot",
        "team_star_up",
        "agent.py",
    )
    cost_averse_path = os.path.join(
        ROOT,
        ".venv",
        "Lib",
        "site-packages",
        "scml_agents",
        "scml2025",
        "oneshot",
        "teamyuzuru",
        "agent.py",
    )
    CautiousOneShotAgent = load_agent_class(
        cautious_path,
        "scml_agents.scml2024.oneshot.team_miyajima_oneshot.cautious",
        "CautiousOneShotAgent",
    )
    HoriYamaAgent = load_agent_class(
        hori_path,
        "scml_agents.scml2025.oneshot.team_star_up.agent",
        "HoriYamaAgent",
    )
    CostAverseAgent = load_agent_class(
        cost_averse_path,
        "scml_agents.scml2025.oneshot.teamyuzuru.agent",
        "CostAverseAgent",
    )
    if TRACKER_AVAILABLE:
        setup_tracker_for_tournament(tournament_path, enabled=True)
        TrackedLitaAgentOS = create_tracked_agent(LitaAgentOS, log_dir=tracker_dir)
        print(f"Tracker 已启用: {tracker_dir}")
    else:
        print("Tracker 未启用：未检测到 scml_analyzer")
        TrackedLitaAgentOS = LitaAgentOS
    competitors = [
        TrackedLitaAgentOS,
        SyncRandomOneShotAgent,
        CautiousOneShotAgent,
        HoriYamaAgent,
        CostAverseAgent,
    ]
    results = anac2024_oneshot(
        competitors=competitors,
        competitor_params=[{} for _ in competitors],
        n_configs=1,
        n_runs_per_world=1,
        min_factories_per_level=2,
        max_worlds_per_config=len(competitors),
        n_competitors_per_world=len(competitors),
        parallelism="serial",
        compact=True,
        forced_logs_fraction=0.0,
        name=f"lita_os_smoke_{run_tag}",
        tournament_path=tournament_path,
        verbose=False,
        n_steps=4,
        neg_n_steps=4,
    )
    if TRACKER_AVAILABLE:
        save_tracker_data(tracker_dir)
    print("冒烟测试完成")
    print(results)


if __name__ == "__main__":
    main()
