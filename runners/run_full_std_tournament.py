"""
完整的 SCML Standard Track Tournament Runner
包含所有 LitaAgent + Top Agents + 内置 Agent
比赛完成后自动分析数据

使用方法:
    python run_full_std_tournament.py
    python run_full_std_tournament.py --n-steps 50
    python run_full_std_tournament.py --n-steps 100 --output-dir my_results
"""
import os
import sys
import json
from pathlib import Path
from datetime import datetime
from collections import Counter

# 设置环境变量解决 Windows 编码问题
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 抑制 TensorFlow 警告

import matplotlib
matplotlib.use('Agg')  # 非交互式后端，避免显示问题
import matplotlib.pyplot as plt

from scml.std import SCML2024StdWorld
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
    top_agents_2025_std = get_agents(2025, as_class=True, top_only=True, track='std')
    # 2024 Top Agents
    top_agents_2024_std = get_agents(2024, as_class=True, top_only=True, track='std')
    # 2023 Top Agents
    top_agents_2023_std = get_agents(2023, as_class=True, top_only=True, track='std')
    print(f"Loaded Top Agents:")
    print(f"  2025 Std: {[a.__name__ for a in top_agents_2025_std]}")
    print(f"  2024 Std: {[a.__name__ for a in top_agents_2024_std]}")
    print(f"  2023 Std: {[a.__name__ for a in top_agents_2023_std]}")
except Exception as e:
    print(f"Warning: Could not load top agents: {e}")
    top_agents_2025_std = []
    top_agents_2024_std = []
    top_agents_2023_std = []


def analyze_tracker_data(log_dir: str) -> dict:
    """
    分析追踪数据
    
    Args:
        log_dir: 追踪数据目录
        
    Returns:
        分析结果字典
    """
    tracker_dir = Path(log_dir)
    if not tracker_dir.exists():
        print(f"⚠️ 追踪目录不存在: {tracker_dir}")
        return {}
    
    results = {
        "agents": {},
        "summary": {
            "total_agents": 0,
            "total_negotiations": 0,
            "total_contracts": 0,
            "avg_success_rate": 0.0,
            "total_production_scheduled": 0,
        }
    }
    
    # 读取汇总文件
    summary_file = tracker_dir / "tracker_summary.json"
    if summary_file.exists():
        with open(summary_file, 'r', encoding='utf-8') as f:
            summary = json.load(f)
            results["world_id"] = summary.get("world_id", "unknown")
    
    # 分析每个 Agent 的数据
    agent_files = list(tracker_dir.glob("agent_*.json"))
    
    for agent_file in agent_files:
        with open(agent_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        agent_id = data.get("agent_id", "unknown")
        agent_type = data.get("agent_type", "unknown")
        entries = data.get("entries", [])
        
        # 统计事件
        categories = Counter(e.get('category') for e in entries)
        events = Counter(e.get('event') for e in entries)
        
        # 计算谈判成功率
        neg_success = events.get('success', 0)
        neg_failed = events.get('failure', 0)
        total_neg = neg_success + neg_failed
        success_rate = neg_success / total_neg if total_neg > 0 else 0
        
        # 合同统计
        contracts_signed = events.get('signed', 0)
        
        # 生产统计
        production_scheduled = events.get('scheduled', 0)
        
        results["agents"][agent_id] = {
            "type": agent_type,
            "total_entries": len(entries),
            "categories": dict(categories),
            "events": dict(events),
            "negotiations": {
                "total": total_neg,
                "success": neg_success,
                "failed": neg_failed,
                "success_rate": success_rate,
            },
            "contracts": {
                "signed": contracts_signed,
            },
            "production": {
                "scheduled": production_scheduled,
            }
        }
        
        # 更新汇总
        results["summary"]["total_negotiations"] += total_neg
        results["summary"]["total_contracts"] += contracts_signed
        results["summary"]["total_production_scheduled"] += production_scheduled
    
    results["summary"]["total_agents"] = len(agent_files)
    
    if results["summary"]["total_agents"] > 0:
        total_success = sum(
            a["negotiations"]["success"] 
            for a in results["agents"].values()
        )
        total_neg = sum(
            a["negotiations"]["total"]
            for a in results["agents"].values()
        )
        results["summary"]["avg_success_rate"] = total_success / total_neg if total_neg > 0 else 0
    
    return results


def print_analysis(results: dict, scores: dict):
    """打印分析结果"""
    print("\n" + "=" * 60)
    print("📊 Standard 赛道数据分析报告")
    print("=" * 60)
    
    # 比赛结果
    print("\n🏆 比赛排名:")
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for rank, (agent_id, score) in enumerate(sorted_scores, 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        print(f"  {medal} {rank}. {agent_id}: {score:.3f}")
    
    # 追踪数据分析
    if results.get("agents"):
        print("\n📈 LitaAgent 表现分析:")
        print("-" * 60)
        
        for agent_id, data in results["agents"].items():
            agent_score = scores.get(agent_id, 0)
            neg = data["negotiations"]
            
            print(f"\n  [{data['type']}] {agent_id}")
            print(f"    • 得分: {agent_score:.3f}")
            print(f"    • 谈判: {neg['total']} 次 (成功 {neg['success']}, 失败 {neg['failed']})")
            print(f"    • 成功率: {neg['success_rate']:.1%}")
            print(f"    • 签约数: {data['contracts']['signed']}")
            if data.get("production", {}).get("scheduled", 0) > 0:
                print(f"    • 生产计划: {data['production']['scheduled']} 次")
        
        # 汇总统计
        summary = results["summary"]
        print("\n📋 汇总统计:")
        print(f"  • 追踪 Agent 数: {summary['total_agents']}")
        print(f"  • 总谈判次数: {summary['total_negotiations']}")
        print(f"  • 总签约数: {summary['total_contracts']}")
        print(f"  • 平均成功率: {summary['avg_success_rate']:.1%}")


def run_std_tournament(n_steps=50, output_dir="tournament_results"):
    """运行 Standard 赛道的比赛"""
    print("\n" + "=" * 60)
    print("Running Standard Tournament")
    print("=" * 60)
    
    # 配置 Tracker
    log_dir = os.path.join(output_dir, "std")
    tracker_log_dir = os.path.join(log_dir, "tracker_logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # 清除之前的 Logger
    TrackerManager._loggers.clear()
    
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
    
    # 运行比赛（带进度条）
    print("\nRunning tournament...")
    world.run_with_progress()
    
    # 保存 Tracker 数据
    print("\nSaving tracker data...")
    TrackerManager.save_all(tracker_log_dir)
    
    # 保存结果
    scores = world.scores()
    print("\n" + "=" * 60)
    print("Standard Tournament Results")
    print("=" * 60)
    
    # 按分数排序
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for rank, (agent_id, score) in enumerate(sorted_scores, 1):
        print(f"  {rank}. {agent_id}: {score:.2f}")
    
    # 分析追踪数据
    print("\n" + "=" * 60)
    print("Analyzing Tracker Data...")
    print("=" * 60)
    
    analysis_results = analyze_tracker_data(tracker_log_dir)
    print_analysis(analysis_results, scores)
    
    # 保存分析报告
    report_file = os.path.join(log_dir, "analysis_report.json")
    report_data = {
        "tournament": {
            "track": "std",
            "n_steps": n_steps,
            "n_agents": len(world.agents),
            "n_agent_types": len(agent_types),
            "timestamp": datetime.now().isoformat(),
        },
        "scores": {k: float(v) for k, v in scores.items()},
        "rankings": [
            {"rank": i + 1, "agent": agent_id, "score": float(score)}
            for i, (agent_id, score) in enumerate(sorted_scores)
        ],
        "analysis": analysis_results,
    }
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n📁 Results saved to: {log_dir}")
    print(f"  • Tracker logs: {tracker_log_dir}")
    print(f"  • Analysis report: {report_file}")
    
    # 保存图表（尝试）
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 尝试绘制统计图
        plt.sca(axes[0, 0])
        world.plot_stats("score")
        axes[0, 0].set_title("Scores Over Time")
        
        plt.sca(axes[0, 1])
        world.plot_stats("balance")
        axes[0, 1].set_title("Balance Over Time")
        
        plt.sca(axes[1, 0])
        world.plot_stats("n_contracts_signed")
        axes[1, 0].set_title("Contracts Signed")
        
        plt.sca(axes[1, 1])
        world.plot_stats("n_negotiations")
        axes[1, 1].set_title("Negotiations")
        
        plt.tight_layout()
        plot_file = os.path.join(log_dir, "std_results.png")
        plt.savefig(plot_file, dpi=150)
        print(f"  • Plot: {plot_file}")
    except Exception as e:
        print(f"Warning: Could not save plots: {e}")
    
    return world, scores, analysis_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run SCML Standard Tournament with LitaAgents and Top Agents"
    )
    parser.add_argument(
        "--n-steps", type=int, default=50,
        help="Number of simulation steps (default: 50, recommended: 50-200)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="tournament_results",
        help="Output directory for results (default: tournament_results)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SCML Full Standard Tournament Runner")
    print("=" * 60)
    print(f"Steps: {args.n_steps}")
    print(f"Output: {args.output_dir}")
    
    world, scores, analysis = run_std_tournament(
        n_steps=args.n_steps,
        output_dir=args.output_dir
    )
    
    print("\n" + "=" * 60)
    print("Tournament Complete!")
    print("=" * 60)
    
    # 导入数据到 tournament_history
    log_dir = os.path.join(args.output_dir, "std")
    try:
        from scml_analyzer.history import import_tournament
        tournament_id = import_tournament(log_dir, copy_mode=False)
        if tournament_id:
            print(f"✓ 数据已导入: {tournament_id}")
    except Exception as e:
        print(f"⚠ 导入失败: {e}")
    
    # 启动无参数可视化服务器
    print("\n启动可视化服务器...")
    try:
        from scml_analyzer.visualizer import start_server
        start_server(port=8080, open_browser=True)
    except KeyboardInterrupt:
        print("\n👋 服务器已停止")
    except Exception as e:
        print(f"启动服务器失败: {e}")
