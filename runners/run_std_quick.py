#!/usr/bin/env python
"""
快速 SCML 2025 Standard 比赛运行器

包含:
- 所有 LitaAgent 变体 (Y, YR, N, P, CIR)
- 2025 年排名前 5 的 Standard Agents

设置:
- 配置数: 3
- 每配置运行次数: 1
- 每场步数: 50
- 总比赛数: 3 场

运行时间: 约 10-20 分钟
"""

import os
import sys
import json
import argparse
import multiprocessing
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

# 使用 spawn 启动方法，避免 fork 导致的死锁问题
# 必须在导入其他模块之前设置
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # 已经设置过了

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scml.std import SCML2024StdWorld
from scml.utils import anac2024_std
from scml.std.agents import RandomStdAgent, GreedyStdAgent, SyncRandomStdAgent

# LitaAgent 系列
from litaagent_std.litaagent_y import LitaAgentY
from litaagent_std.litaagent_yr import LitaAgentYR
from litaagent_std.litaagent_cir import LitaAgentCIR
from litaagent_std.litaagent_n import LitaAgentN
from litaagent_std.litaagent_p import LitaAgentP

# Tracker 系统
from litaagent_std.tracker_mixin import inject_tracker_to_agents
from scml_analyzer.auto_tracker import TrackerConfig, TrackerManager

# Top Agents
try:
    from scml_agents import get_agents
    TOP_AGENTS_2025 = get_agents(2025, as_class=True, top_only=True, track='std')
    print(f"✓ 加载 2025 Standard Top Agents: {[a.__name__ for a in TOP_AGENTS_2025]}")
except Exception as e:
    print(f"⚠️ 无法加载 2025 Top Agents: {e}")
    TOP_AGENTS_2025 = []


# 比赛配置 - 快速版
TOURNAMENT_CONFIG = {
    "name": "SCML 2025 Standard 快速比赛",
    "track": "std",
    "n_configs": 3,                 # 配置数 (较少)
    "n_runs_per_world": 1,          # 每配置运行次数 (只运行1次)
    "n_steps": 50,                  # 每场步数 (较少)
    "max_worlds_per_config": 10,    # 每个配置最多 10 个 world (限制总数!)
}


def get_all_agents():
    """获取所有参赛 Agent"""
    # LitaAgents
    lita_agents = [LitaAgentY, LitaAgentYR, LitaAgentCIR, LitaAgentN, LitaAgentP]
    
    # 注入 Tracker
    tracked_agents = inject_tracker_to_agents(lita_agents)
    
    # 组合 LitaAgents + 2025 Top Agents
    competitors = tracked_agents + list(TOP_AGENTS_2025)
    
    # 如果参赛者太少，添加内置 Agent 填充
    if len(competitors) < 10:
        fillers = [
            RandomStdAgent, 
            GreedyStdAgent, 
            SyncRandomStdAgent
        ]
        # 注入 Tracker 到 fillers (可选，如果想分析它们)
        # tracked_fillers = inject_tracker_to_agents(fillers)
        competitors.extend(fillers)
    
    return competitors, [a.__name__ for a in lita_agents]


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
    """运行快速 Standard 比赛"""
    
    # 设置输出目录
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/std_quick_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print(f"⚡ {TOURNAMENT_CONFIG['name']}")
    print("=" * 60)
    print(f"  • 配置数: {TOURNAMENT_CONFIG['n_configs']}")
    print(f"  • 每配置运行次数: {TOURNAMENT_CONFIG['n_runs_per_world']}")
    print(f"  • 每场步数: {TOURNAMENT_CONFIG['n_steps']}")
    print(f"  • 每配置最大 World 数: {TOURNAMENT_CONFIG['max_worlds_per_config']}")
    max_total = TOURNAMENT_CONFIG['n_configs'] * TOURNAMENT_CONFIG['max_worlds_per_config']
    print(f"  • 最大总 World 数: {max_total}")
    print(f"  • 输出目录: {output_dir}")
    
    # 配置 Tracker
    print("\n📝 配置 Tracker 系统...")
    tracker_log_dir = os.path.join(output_dir, "tracker_logs")
    os.makedirs(tracker_log_dir, exist_ok=True)
    
    TrackerManager._loggers.clear()
    TrackerConfig.configure(
        enabled=True,
        log_dir=tracker_log_dir,
        console_echo=False
    )
    
    # 获取所有 Agent
    print("\n🤖 加载参赛 Agents...")
    all_agents, lita_names = get_all_agents()
    
    print(f"\n参赛 Agents ({len(all_agents)}):")
    for i, agent in enumerate(all_agents, 1):
        tag = "[LitaAgent]" if agent.__name__ in lita_names else ""
        print(f"  {i}. {agent.__name__} {tag}")
    
    # 运行锦标赛
    print(f"\n🚀 开始比赛...")
    print("=" * 60)
    
    results = anac2024_std(
        competitors=all_agents,
        n_configs=TOURNAMENT_CONFIG['n_configs'],
        n_runs_per_world=TOURNAMENT_CONFIG['n_runs_per_world'],
        n_steps=TOURNAMENT_CONFIG['n_steps'],
        max_worlds_per_config=TOURNAMENT_CONFIG['max_worlds_per_config'],
        print_exceptions=True,
        verbose=False,
        # 使用 parallel 模式以恢复进度条
        # Tracker 数据由 Agent 在 step() 中自动保存到文件
        parallelism='parallel',
        # 设置总超时时间为 30 分钟，防止无限等待
        total_timeout=1800,
    )
    
    # 重建 Tracker Summary (虽然 serial 模式下不需要，但保留无妨)
    print("\n📊 重建 Tracker Summary...")
    TrackerManager.rebuild_summary(tracker_log_dir)
    
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
        description="运行快速 SCML 2025 Standard 比赛"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="输出目录 (默认: results/std_quick_<timestamp>)"
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
