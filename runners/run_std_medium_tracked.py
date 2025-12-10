#!/usr/bin/env python
"""
中等规模 SCML 2025 Standard 比赛（带进度显示，LitaAgents 启用 Tracker，Top Agents 适量）。

目标：
- 比官方规模更小，减少组合数量，加快诊断/验证（默认 n_configs=5，n_runs_per_world=1，n_steps=50-100）。
- 使用 loky 并行，显示进度。
- LitaAgents 注入 tracker，日志写到输出目录 tracker_logs。
- 尝试加载最多 6 个 2025 Top Agents（可通过参数限制）。
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
from litaagent_std.tracker_mixin import inject_tracker_to_agents, create_tracked_agent
from litaagent_std.litaagent_y import LitaAgentY
from litaagent_std.litaagent_yr import LitaAgentYR
from litaagent_std.litaagent_cir import LitaAgentCIR
from litaagent_std.litaagent_n import LitaAgentN
from litaagent_std.litaagent_p import LitaAgentP

try:
    from scml_agents import get_agents
    TOP_AGENTS_2025 = get_agents(2025, as_class=True, top_only=6, track='std')
except Exception as e:
    print(f"⚠️ 无法加载 2025 Top Agents: {e}")
    TOP_AGENTS_2025 = []

TOURNAMENT_CONFIG = {
    "name": "SCML 2025 Standard 中等规模（Lita tracker + Top Agents）",
    "track": "std",
    "n_configs": 5,
    "n_runs_per_world": 1,
    "n_steps": (50, 100),
}


def build_competitors(max_top: int | None = None):
    lita_agents = [LitaAgentY, LitaAgentYR, LitaAgentCIR, LitaAgentN, LitaAgentP]
    # 使用子类方式，支持并行
    log_dir = os.environ.get("SCML_TRACKER_LOG_DIR", "")
    tracked_lita = [
        create_tracked_agent(cls, log_dir=log_dir or ".")
        for cls in lita_agents
    ]
    tops = TOP_AGENTS_2025 if max_top is None else TOP_AGENTS_2025[:max_top]
    competitors = tracked_lita + list(tops)
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


def run_tournament(output_dir=None, port=8081, no_server=True, max_top: int | None = None):
    if output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/std_medium_tracked_{ts}"
    output_dir = Path(output_dir)
    tracker_dir = output_dir / "tracker_logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    tracker_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"🏆 {TOURNAMENT_CONFIG['name']}")
    print("=" * 70)
    print(f"输出目录: {output_dir}")
    print(f"Tracker 日志: {tracker_dir}")
    print(f"Top Agents (最多 {max_top or len(TOP_AGENTS_2025)}): {[a.__name__ for a in TOP_AGENTS_2025[:max_top] if TOP_AGENTS_2025]}")

    TrackerManager._loggers.clear()
    TrackerConfig.configure(enabled=True, log_dir=str(tracker_dir), console_echo=False)
    # 给子进程传递日志目录
    os.environ["SCML_TRACKER_LOG_DIR"] = str(tracker_dir)

    competitors, lita_names = build_competitors(max_top=max_top)
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
    parser = argparse.ArgumentParser(description="中等规模 Standard 赛（Lita tracker + Top Agents）")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录")
    parser.add_argument("--port", type=int, default=8081, help="可视化端口（默认 8081）")
    parser.add_argument("--no-server", action="store_true", help="不自动启动可视化")
    parser.add_argument("--max-top", type=int, default=None, help="限制 Top Agents 数量（默认 6）")
    args = parser.parse_args()
    run_tournament(output_dir=args.output_dir, port=args.port, no_server=args.no_server, max_top=args.max_top)


if __name__ == "__main__":
    main()
