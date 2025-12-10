#!/usr/bin/env python
"""
官方规模 SCML 2025 Standard 比赛（含 PenguinAgent，LitaAgents 启用 Tracker，带进度）。

目的：
- 验证 PenguinAgent 加入后的全量标准赛是否可跑通，并保留 tracker 日志。
- 使用 loky 并行、verbose 进度，配置同官方：n_configs=20, n_runs_per_world=2, n_steps=(50,200)。

可选参数：
- --configs, --runs 控制规模（默认 20、2），可用小规模先验证。
- --max-top 控制加载的 Top Agents 数量（默认 8）。
- --port 控制可视化端口，--no-server 关闭自动可视化。
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from runners.loky_patch import enable_loky_executor
enable_loky_executor()

import matplotlib
matplotlib.use('Agg')

from scml.utils import anac2024_std
from scml_analyzer.auto_tracker import TrackerConfig, TrackerManager
from litaagent_std.tracker_mixin import create_tracked_agent
from litaagent_std.litaagent_y import LitaAgentY
from litaagent_std.litaagent_yr import LitaAgentYR
from litaagent_std.litaagent_cir import LitaAgentCIR
from litaagent_std.litaagent_n import LitaAgentN
from litaagent_std.litaagent_p import LitaAgentP
from scml_agents.scml2024.standard.team_penguin.penguinagent import PenguinAgent

try:
    from scml_agents import get_agents
    TOP_AGENTS_2025 = get_agents(2025, as_class=True, top_only=8, track='std')
except Exception as e:
    print(f"⚠️ 无法加载 2025 Top Agents: {e}")
    TOP_AGENTS_2025 = []

def build_competitors(max_top: int | None = None):
    # LitaAgents with tracker subclasses
    log_dir = os.environ.get("SCML_TRACKER_LOG_DIR", ".")
    lita_bases = [LitaAgentY, LitaAgentYR, LitaAgentCIR, LitaAgentN, LitaAgentP]
    tracked_lita = [create_tracked_agent(cls, log_dir=log_dir) for cls in lita_bases]
    # PenguinAgent（无 tracker）
    penguin = [PenguinAgent]
    tops = TOP_AGENTS_2025 if max_top is None else TOP_AGENTS_2025[:max_top]
    competitors = tracked_lita + penguin + list(tops)
    lita_names = [c.__name__ for c in lita_bases]
    return competitors, lita_names


def save_results(output_dir, results, competitors, lita_names):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = {
        "tournament": {},
        "competitors": [c.__name__ for c in competitors],
        "winners": [w.split(".")[-1] for w in results.winners] if getattr(results, "winners", None) else [],
    }
    with open(out / "tournament_results.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def run_tournament(n_configs=20, n_runs=2, output_dir=None, port=8080, no_server=False, max_top=None):
    if output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/std_full_penguin_{ts}"
    output_dir = Path(output_dir)
    tracker_dir = output_dir / "tracker_logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    tracker_dir.mkdir(parents=True, exist_ok=True)
    os.environ["SCML_TRACKER_LOG_DIR"] = str(tracker_dir)

    print("=" * 70)
    print("🏆 SCML 2025 Standard 官方规模（Penguin + Lita tracker + Top Agents）")
    print("=" * 70)
    print(f"配置: n_configs={n_configs}, n_runs_per_world={n_runs}, n_steps=(50,200)")
    print(f"输出目录: {output_dir}")
    print(f"Tracker 日志: {tracker_dir}")

    TrackerManager._loggers.clear()
    TrackerConfig.configure(enabled=True, log_dir=str(tracker_dir), console_echo=False)

    competitors, lita_names = build_competitors(max_top=max_top)
    print(f"参赛者 ({len(competitors)}): {[c.__name__ for c in competitors]}")

    results = anac2024_std(
        competitors=competitors,
        n_configs=n_configs,
        n_runs_per_world=n_runs,
        n_steps=(50, 200),
        print_exceptions=True,
        verbose=True,
        parallelism="parallel",
    )

    if getattr(results, "total_scores", None) is not None:
        print("\n📈 排名:")
        sorted_scores = results.total_scores.sort_values("score", ascending=False)
        for rank, (_, row) in enumerate(sorted_scores.iterrows(), 1):
            agent_name = row["agent_type"].split(".")[-1]
            tag = "⭐" if agent_name in lita_names else ""
            print(f"  {rank}. {agent_name}: {row['score']:.4f} {tag}")

    save_results(output_dir, results, competitors, lita_names)

    from scml_analyzer.postprocess import postprocess_tournament
    postprocess_tournament(
        output_dir=output_dir,
        start_visualizer=not no_server,
        visualizer_port=port,
    )

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="官方规模 std 赛（Penguin + Lita tracker + Top Agents）")
    parser.add_argument("--configs", type=int, default=20, help="n_configs (默认20)")
    parser.add_argument("--runs", type=int, default=2, help="n_runs_per_world (默认2)")
    parser.add_argument("--max-top", type=int, default=8, help="加载的 Top Agents 数量上限")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录")
    parser.add_argument("--port", type=int, default=8080, help="可视化端口")
    parser.add_argument("--no-server", action="store_true", help="不自动启动可视化")
    args = parser.parse_args()

    run_tournament(
        n_configs=args.configs,
        n_runs=args.runs,
        max_top=args.max_top,
        output_dir=args.output_dir,
        port=args.port,
        no_server=args.no_server,
    )


if __name__ == "__main__":
    main()
