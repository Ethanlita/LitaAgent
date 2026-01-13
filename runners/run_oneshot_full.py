#!/usr/bin/env python
"""
完整 SCML 2025 OneShot 比赛运行器

包含:
- 所有 LitaAgent 变体 (Y, YR, N, P, CIR)
- 2025 年排名前 5 的 OneShot Agents

设置:
- 配置数: 10
- 每配置运行次数: 2
- 每场步数: 50
- 总比赛数: 20 场

运行时间: 约 30-60 分钟
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 设置环境
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from runners.loky_patch import enable_loky_executor
enable_loky_executor()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scml.oneshot import SCML2024OneShotWorld
from scml.utils import anac2024_oneshot

# LitaAgent 系列
from litaagent_os.agent import LitaAgentOS

# Tracker 系统
from litaagent_std.tracker_mixin import inject_tracker_to_agents
from scml_analyzer.auto_tracker import TrackerConfig, TrackerManager

# Top Agents
try:
    from scml_agents import get_agents
    TOP_AGENTS_2025 = get_agents(2025, as_class=True, top_only=True, track='oneshot')
    print(f"✓ 加载 2025 OneShot Top Agents: {[a.__name__ for a in TOP_AGENTS_2025]}")
    
    # 找到 RChan 用于追踪
    RCHAN_AGENT = None
    for agent in TOP_AGENTS_2025:
        if 'RChan' in agent.__name__:
            RCHAN_AGENT = agent
            print(f"✓ 找到 RChan Agent: {agent.__name__}")
            break
except Exception as e:
    print(f"⚠️ 无法加载 2025 Top Agents: {e}")
    TOP_AGENTS_2025 = []
    RCHAN_AGENT = None


# 比赛配置
TOURNAMENT_CONFIG = {
    "name": "SCML 2025 OneShot 完整比赛",
    "track": "oneshot",
    "n_configs": 10,           # 配置数
    "n_runs_per_world": 2,     # 每配置运行次数
    "n_steps": 50,             # 固定 50 天 (便于分析)
}


def get_all_agents():
    """获取所有参赛 Agent"""
    # LitaAgents + RChan (都需要追踪)
    agents_to_track = [LitaAgentOS]
    if RCHAN_AGENT is not None:
        agents_to_track.append(RCHAN_AGENT)
    
    # 注入 Tracker
    tracked_agents = inject_tracker_to_agents(agents_to_track)
    
    # 其他 Top Agents (不追踪)
    other_top_agents = [a for a in TOP_AGENTS_2025 if a != RCHAN_AGENT]
    
    # 组合所有 Agents
    all_agents = tracked_agents + other_top_agents
    
    return all_agents, [a.__name__ for a in agents_to_track]


def save_tournament_results(output_dir: str, results, config: dict):
    """保存比赛结果为 visualizer 需要的格式"""
    # 创建 tournament_results.json
    rankings = []
    if hasattr(results, 'total_scores') and results.total_scores is not None:
        sorted_scores = results.total_scores.sort_values("score", ascending=False)
        for rank, (idx, row) in enumerate(sorted_scores.iterrows(), 1):
            agent_type = row["agent_type"].split(".")[-1]
            rankings.append({
                "rank": rank,
                "agent_type": agent_type,
                "score": float(row["score"]),
            })
    
    results_data = {
        "tournament": {
            "name": config["name"],
            "track": config["track"],
            "n_configs": config["n_configs"],
            "n_runs_per_world": config["n_runs_per_world"],
            "n_steps": config["n_steps"],
            "timestamp": datetime.now().isoformat(),
        },
        "rankings": rankings,
        "winners": [w.split(".")[-1] for w in results.winners] if results.winners else [],
    }
    
    results_file = os.path.join(output_dir, "tournament_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)
    
    return results_file


def run_tournament(output_dir: str = None, port: int = 8080, no_server: bool = False):
    """运行完整 OneShot 比赛"""
    
    # 设置输出目录
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/oneshot_full_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print(f"🏆 {TOURNAMENT_CONFIG['name']}")
    print("=" * 60)
    print(f"  • 配置数: {TOURNAMENT_CONFIG['n_configs']}")
    print(f"  • 每配置运行次数: {TOURNAMENT_CONFIG['n_runs_per_world']}")
    print(f"  • 每场步数: {TOURNAMENT_CONFIG['n_steps']}")
    print(f"  • 总比赛数: {TOURNAMENT_CONFIG['n_configs'] * TOURNAMENT_CONFIG['n_runs_per_world']}")
    print(f"  • 输出目录: {output_dir}")
    
    # 配置 Tracker
    print("\n📝 配置 Tracker 系统...")
    TrackerManager._loggers.clear()
    TrackerConfig.configure(
        enabled=True,
        log_dir=output_dir,
        console_echo=False
    )
    
    # 获取所有 Agent
    print("\n🤖 加载参赛 Agents...")
    all_agents, lita_names = get_all_agents()
    
    print(f"\n参赛 Agents ({len(all_agents)}):")
    for i, agent in enumerate(all_agents, 1):
        tag = "[LitaAgent]" if agent.__name__ in lita_names else "[Top Agent]"
        print(f"  {i}. {agent.__name__} {tag}")
    
    # 运行锦标赛
    print(f"\n🚀 开始比赛...")
    print("=" * 60)
    
    results = anac2024_oneshot(
        competitors=all_agents,
        n_configs=TOURNAMENT_CONFIG['n_configs'],
        n_runs_per_world=TOURNAMENT_CONFIG['n_runs_per_world'],
        n_steps=TOURNAMENT_CONFIG['n_steps'],
        print_exceptions=True,
        verbose=False,
        # 使用 parallel 模式以启用进度条，并利用 TrackedAgent 的自动保存功能
        parallelism='parallel',
    )
    
    # 显示结果
    print("\n" + "=" * 60)
    print("🏆 比赛结果")
    print("=" * 60)
    
    if hasattr(results, 'winners') and results.winners:
        print(f"\n🥇 冠军: {[w.split('.')[-1] for w in results.winners]}")
    
    if hasattr(results, 'total_scores') and results.total_scores is not None:
        print("\n📈 排名:")
        sorted_scores = results.total_scores.sort_values("score", ascending=False)
        for rank, (idx, row) in enumerate(sorted_scores.iterrows(), 1):
            agent_name = row["agent_type"].split(".")[-1]
            tag = "⭐" if agent_name in lita_names else ""
            print(f"  {rank}. {agent_name}: {row['score']:.4f} {tag}")
    
    print(f"\n✅ 比赛完成！")
    
    # 后处理：保存数据、导入到 tournament_history、启动 Visualizer
    from scml_analyzer.postprocess import postprocess_tournament
    postprocess_tournament(
        output_dir=output_dir,
        start_visualizer=not no_server,
        visualizer_port=port,
    )
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="运行完整 SCML 2025 OneShot 比赛"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="输出目录 (默认: results/oneshot_full_<timestamp>)"
    )
    parser.add_argument(
        "--port", type=int, default=8080,
        help="可视化服务器端口 (默认: 8080)"
    )
    parser.add_argument(
        "--no-server", action="store_true",
        help="不启动可视化服务器"
    )
    
    args = parser.parse_args()
    
    run_tournament(
        output_dir=args.output_dir,
        port=args.port,
        no_server=args.no_server,
    )


if __name__ == "__main__":
    main()
