"""
完整的 SCML Tournament Runner
包含所有 LitaAgent + Top Agents + 内置 Agent
"""
import os
import sys

# 设置环境变量解决 Windows 编码问题
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 抑制 TensorFlow 警告

import matplotlib
matplotlib.use('Agg')  # 非交互式后端，避免显示问题
import matplotlib.pyplot as plt

from scml.oneshot import SCML2024OneShotWorld
from scml.std import SCML2024StdWorld
from scml.oneshot.agents import (
    GreedyOneShotAgent,
    RandDistOneShotAgent, 
    RandomOneShotAgent,
    SyncRandomOneShotAgent,
)
from scml.std.agents import (
    GreedyStdAgent,
    RandomStdAgent,
    SyncRandomStdAgent,
)

# LitaAgent 系列
from litaagent_std.litaagent_y import LitaAgentY
from litaagent_std.litaagent_yr import LitaAgentYR
from litaagent_std.litaagent_cir import LitaAgentCIR
from litaagent_std.litaagent_n import LitaAgentN
from litaagent_std.litaagent_p import LitaAgentP

# Tracker 系统
from litaagent_std.tracker_mixin import inject_tracker_to_agents
from scml_analyzer.auto_tracker import TrackerConfig, TrackerManager

# Top Agents (从 scml-agents 包)
try:
    from scml_agents import get_agents
    # 2025 Top Agents (最新!)
    top_agents_2025_oneshot = get_agents(2025, as_class=True, top_only=True, track='oneshot')
    top_agents_2025_std = get_agents(2025, as_class=True, top_only=True, track='std')
    # 2024 Top Agents
    top_agents_2024_oneshot = get_agents(2024, as_class=True, top_only=True, track='oneshot')
    top_agents_2024_std = get_agents(2024, as_class=True, top_only=True, track='std')
    # 2023 Top Agents
    top_agents_2023_oneshot = get_agents(2023, as_class=True, top_only=True, track='oneshot')
    top_agents_2023_std = get_agents(2023, as_class=True, top_only=True, track='std')
    print(f"Loaded Top Agents:")
    print(f"  2025 OneShot: {[a.__name__ for a in top_agents_2025_oneshot]}")
    print(f"  2025 Std: {[a.__name__ for a in top_agents_2025_std]}")
    print(f"  2024 OneShot: {[a.__name__ for a in top_agents_2024_oneshot]}")
    print(f"  2024 Std: {[a.__name__ for a in top_agents_2024_std]}")
    print(f"  2023 OneShot: {[a.__name__ for a in top_agents_2023_oneshot]}")
    print(f"  2023 Std: {[a.__name__ for a in top_agents_2023_std]}")
except Exception as e:
    print(f"Warning: Could not load top agents: {e}")
    top_agents_2025_oneshot = []
    top_agents_2025_std = []
    top_agents_2024_oneshot = []
    top_agents_2024_std = []
    top_agents_2023_oneshot = []
    top_agents_2023_std = []


def run_oneshot_tournament(n_steps=20, output_dir="tournament_results"):
    """运行 OneShot 赛道的比赛"""
    print("\n" + "="*60)
    print("Running OneShot Tournament")
    print("="*60)
    
    # 配置 Tracker
    log_dir = os.path.join(output_dir, "oneshot")
    os.makedirs(log_dir, exist_ok=True)
    TrackerConfig.configure(
        enabled=True,
        log_dir=log_dir,
        console_echo=False  # 减少输出噪音
    )
    
    # LitaAgents - 注入 Tracker
    lita_agents = [LitaAgentY, LitaAgentYR, LitaAgentCIR, LitaAgentN, LitaAgentP]
    print("  Injecting trackers to LitaAgents...")
    tracked_lita_agents = inject_tracker_to_agents(lita_agents)
    for agent_cls in tracked_lita_agents:
        print(f"    - {agent_cls.__name__}")
    
    # 组合所有 Agent 类型
    agent_types = (
        tracked_lita_agents +
        list(top_agents_2025_oneshot) +  # 2025 最新!
        list(top_agents_2024_oneshot) +
        list(top_agents_2023_oneshot) +
        [
            GreedyOneShotAgent,
            RandDistOneShotAgent,
            RandomOneShotAgent,
            SyncRandomOneShotAgent,
        ]
    )
    
    print(f"\nAgent types in tournament ({len(agent_types)}):")
    for i, a in enumerate(agent_types, 1):
        print(f"  {i}. {a.__name__}")
    
    # 创建世界
    print(f"\nGenerating world with n_steps={n_steps}...")
    world = SCML2024OneShotWorld(
        **SCML2024OneShotWorld.generate(
            agent_types=agent_types,
            n_steps=n_steps,
            n_processes=2,
        ),
        construct_graphs=True,
    )
    
    print(f"World created with {len(world.agents)} agents")
    
    # 运行比赛
    print("\nRunning tournament...")
    world.run_with_progress()
    
    # 保存 Tracker 数据
    print("\nSaving tracker data...")
    TrackerManager.save_all(os.path.join(log_dir, "tracker_logs"))
    
    # 保存结果
    scores = world.scores()
    print("\n" + "="*60)
    print("OneShot Tournament Results")
    print("="*60)
    
    # 按分数排序
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for rank, (agent_id, score) in enumerate(sorted_scores, 1):
        print(f"  {rank}. {agent_id}: {score:.2f}")
    
    # 保存图表
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        world.plot_stats("score", ax=axes[0, 0])
        axes[0, 0].set_title("Scores Over Time")
        
        world.plot_stats("balance", ax=axes[0, 1])
        axes[0, 1].set_title("Balance Over Time")
        
        world.plot_stats("n_contracts_signed", ax=axes[1, 0])
        axes[1, 0].set_title("Contracts Signed")
        
        world.plot_stats("n_negotiations", ax=axes[1, 1])
        axes[1, 1].set_title("Negotiations")
        
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, "oneshot_results.png"), dpi=150)
        print(f"\nResults saved to {log_dir}/oneshot_results.png")
    except Exception as e:
        print(f"Warning: Could not save plots: {e}")
    
    return world, scores


def run_std_tournament(n_steps=20, output_dir="tournament_results"):
    """运行 Std 赛道的比赛"""
    print("\n" + "="*60)
    print("Running Std Tournament")
    print("="*60)
    
    # 配置 Tracker
    log_dir = os.path.join(output_dir, "std")
    os.makedirs(log_dir, exist_ok=True)
    
    # 重置 TrackerManager
    TrackerManager._loggers.clear()
    TrackerConfig.configure(
        enabled=True,
        log_dir=log_dir,
        console_echo=False
    )
    
    # LitaAgents - 注入 Tracker
    lita_agents = [LitaAgentY, LitaAgentYR, LitaAgentCIR, LitaAgentN, LitaAgentP]
    print("  Injecting trackers to LitaAgents...")
    tracked_lita_agents = inject_tracker_to_agents(lita_agents)
    for agent_cls in tracked_lita_agents:
        print(f"    - {agent_cls.__name__}")
    
    # 组合所有 Agent 类型
    agent_types = (
        tracked_lita_agents +
        list(top_agents_2025_std) +  # 2025 最新!
        list(top_agents_2024_std) +
        list(top_agents_2023_std) +
        [
            GreedyStdAgent,
            RandomStdAgent,
            SyncRandomStdAgent,
        ]
    )
    
    print(f"\nAgent types in tournament ({len(agent_types)}):")
    for i, a in enumerate(agent_types, 1):
        print(f"  {i}. {a.__name__}")
    
    # 创建世界
    print(f"\nGenerating world with n_steps={n_steps}...")
    world = SCML2024StdWorld(
        **SCML2024StdWorld.generate(
            agent_types=agent_types,
            n_steps=n_steps,
            n_processes=2,
        ),
        construct_graphs=True,
    )
    
    print(f"World created with {len(world.agents)} agents")
    
    # 运行比赛
    print("\nRunning tournament...")
    world.run_with_progress()
    
    # 保存 Tracker 数据
    print("\nSaving tracker data...")
    TrackerManager.save_all(os.path.join(log_dir, "tracker_logs"))
    
    # 保存结果
    scores = world.scores()
    print("\n" + "="*60)
    print("Std Tournament Results")
    print("="*60)
    
    # 按分数排序
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for rank, (agent_id, score) in enumerate(sorted_scores, 1):
        print(f"  {rank}. {agent_id}: {score:.2f}")
    
    # 保存图表
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        world.plot_stats("score", ax=axes[0, 0])
        axes[0, 0].set_title("Scores Over Time")
        
        world.plot_stats("balance", ax=axes[0, 1])
        axes[0, 1].set_title("Balance Over Time")
        
        world.plot_stats("n_contracts_signed", ax=axes[1, 0])
        axes[1, 0].set_title("Contracts Signed")
        
        world.plot_stats("n_negotiations", ax=axes[1, 1])
        axes[1, 1].set_title("Negotiations")
        
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, "std_results.png"), dpi=150)
        print(f"\nResults saved to {log_dir}/std_results.png")
    except Exception as e:
        print(f"Warning: Could not save plots: {e}")
    
    return world, scores


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run SCML Tournament with LitaAgents and Top Agents")
    parser.add_argument("--track", choices=["oneshot", "std", "both"], default="oneshot",
                       help="Which track to run (default: oneshot)")
    parser.add_argument("--n-steps", type=int, default=20,
                       help="Number of simulation steps (default: 20)")
    parser.add_argument("--output-dir", type=str, default="tournament_results",
                       help="Output directory for results (default: tournament_results)")
    
    args = parser.parse_args()
    
    print("="*60)
    print("SCML Full Tournament Runner")
    print("="*60)
    print(f"Track: {args.track}")
    print(f"Steps: {args.n_steps}")
    print(f"Output: {args.output_dir}")
    
    if args.track in ["oneshot", "both"]:
        run_oneshot_tournament(n_steps=args.n_steps, output_dir=args.output_dir)
    
    if args.track in ["std", "both"]:
        run_std_tournament(n_steps=args.n_steps, output_dir=args.output_dir)
    
    print("\n" + "="*60)
    print("Tournament Complete!")
    print("="*60)
