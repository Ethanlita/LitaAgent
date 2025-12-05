#!/usr/bin/env python
"""
官方规模的 SCML 2025 Standard 锦标赛（带进度显示，LitaAgents 启用 Tracker）。

特点：
- 规模与官方一致：n_configs=20，n_runs_per_world=2，n_steps=(50,200)
- 使用 loky 并行（通过 runners.loky_patch）
- 所有 LitaAgents 注入 tracker，日志写入输出目录下 tracker_logs
- 试图加载 2025 Top Agents（最多 8 个），与 LitaAgents 一起参赛
- verbose=True 显示进度
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
from litaagent_std.tracker_mixin import inject_tracker_to_agents
from litaagent_std.litaagent_y import LitaAgentY
from litaagent_std.litaagent_yr import LitaAgentYR
from litaagent_std.litaagent_cir import LitaAgentCIR
from litaagent_std.litaagent_n import LitaAgentN
from litaagent_std.litaagent_p import LitaAgentP

try:
    from scml_agents import get_agents
    TOP_AGENTS_2025 = get_agents(2025, as_class=True, top_only=8, track='std')
except Exception as e:
    print(f"⚠️ 无法加载 2025 Top Agents: {e}")
    TOP_AGENTS_2025 = []

TOURNAMENT_CONFIG = {
    "name": "SCML 2025 Standard 官方规模（Lita tracker + Top Agents）",
    "track": "std",
    "n_configs": 20,
    "n_runs_per_world": 2,
    "n_steps": (50, 200),
}


def build_competitors():
    lita_agents = [LitaAgentY, LitaAgentYR, LitaAgentCIR, LitaAgentN, LitaAgentP]
    tracked_lita = inject_tracker_to_agents(lita_agents)
    competitors = tracked_lita + list(TOP_AGENTS_2025)
    lita_names = [a.__name__ for a in lita_agents]
    return competitors, lita_names


def save_results(output_dir, results, competitors, lita_names):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = {
        "tournament": TOURNAMENT_CONFIG,
        "competitors": [c.__name__ for c in competitors],
        "winners": [w.split(".")[-1] for w in results.winners] if getattr(results, "winners", None) else [],
    }
    with open(out / "tournament_results.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def run_tournament(output_dir=None, port=8081, no_server=True):
    if output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/std_full_tracked_{ts}"
    output_dir = Path(output_dir)
    tracker_dir = output_dir / "tracker_logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    tracker_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"🏆 {TOURNAMENT_CONFIG['name']}")
    print("=" * 70)
    print(f"输出目录: {output_dir}")
    print(f"Tracker 日志: {tracker_dir}")
    print(f"Top Agents: {[a.__name__ for a in TOP_AGENTS_2025]}")

    TrackerManager._loggers.clear()
    TrackerConfig.configure(enabled=True, log_dir=str(tracker_dir), console_echo=False)

    competitors, lita_names = build_competitors()
    print(f"参赛者 ({len(competitors)}): {[c.__name__ for c in competitors]}")

    results = anac2024_std(
        competitors=competitors,
        n_configs=TOURNAMENT_CONFIG["n_configs"],
        n_runs_per_world=TOURNAMENT_CONFIG["n_runs_per_world"],
        n_steps=TOURNAMENT_CONFIG["n_steps"],
        print_exceptions=True,
        verbose=True,
        parallelism="parallel",
    )

    # 打印排名
    if getattr(results, "total_scores", None) is not None:
        print("\n📈 排名:")
        sorted_scores = results.total_scores.sort_values("score", ascending=False)
        for rank, (_, row) in enumerate(sorted_scores.iterrows(), 1):
            agent_name = row["agent_type"].split(".")[-1]
            tag = "⭐" if agent_name in lita_names else ""
            print(f"  {rank}. {agent_name}: {row['score']:.4f} {tag}")

    save_results(output_dir, results, competitors, lita_names)

    # 后处理（移动数据 + 可视化可选）
    from scml_analyzer.postprocess import postprocess_tournament
    postprocess_tournament(
        output_dir=output_dir,
        start_visualizer=not no_server,
        visualizer_port=port,
    )

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="官方规模 Standard 赛（Lita tracker + Top Agents）")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录")
    parser.add_argument("--port", type=int, default=8081, help="可视化端口（默认 8081）")
    parser.add_argument("--no-server", action="store_true", help="不自动启动可视化")
    args = parser.parse_args()
    run_tournament(output_dir=args.output_dir, port=args.port, no_server=args.no_server)


if __name__ == "__main__":
    main()
