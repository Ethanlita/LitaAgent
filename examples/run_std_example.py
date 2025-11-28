#!/usr/bin/env python
"""
Standard 赛道示例

演示如何运行 Standard 比赛并自动分析数据。
Standard 赛道比 OneShot 更复杂，涉及多日库存管理和生产计划。

包含：
1. 配置 Tracker 系统
2. 注入追踪到 LitaAgents
3. 运行比赛（带进度条）
4. 保存追踪数据
5. 自动分析结果

使用方法：
    python examples/run_std_example.py
    python examples/run_std_example.py --n-steps 50
    python examples/run_std_example.py --verbose
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter

# 设置环境
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 添加项目根目录到 path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

# SCML Standard 导入
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

# Top Agents
try:
    from scml_agents import get_agents
    TOP_AGENTS_2025_STD = get_agents(2025, as_class=True, top_only=True, track='std')
    TOP_AGENTS_2024_STD = get_agents(2024, as_class=True, top_only=True, track='std')
    TOP_AGENTS_2023_STD = get_agents(2023, as_class=True, top_only=True, track='std')
except ImportError:
    print("Warning: scml-agents not installed, skipping top agents")
    TOP_AGENTS_2025_STD = []
    TOP_AGENTS_2024_STD = []
    TOP_AGENTS_2023_STD = []


def analyze_tracker_data(log_dir: str) -> dict:
    """
    分析追踪数据（Standard 版本，包含更多库存和生产分析）
    
    Args:
        log_dir: 追踪数据目录
        
    Returns:
        分析结果字典
    """
    tracker_dir = Path(log_dir) / "tracker_logs"
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
        production_executed = events.get('executed', 0)
        
        # 库存状态分析
        inventory_entries = [e for e in entries if e.get('category') == 'inventory']
        daily_status_entries = [e for e in entries if e.get('event') == 'daily_status']
        
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
                "executed": production_executed,
            },
            "inventory_snapshots": len(inventory_entries),
            "daily_reports": len(daily_status_entries),
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
        print("\n📈 Agent 表现分析:")
        print("-" * 60)
        
        for agent_id, data in results["agents"].items():
            agent_score = scores.get(agent_id, 0)
            neg = data["negotiations"]
            prod = data["production"]
            
            print(f"\n  [{data['type']}] {agent_id}")
            print(f"    • 得分: {agent_score:.3f}")
            print(f"    • 谈判: {neg['total']} 次 (成功 {neg['success']}, 失败 {neg['failed']})")
            print(f"    • 成功率: {neg['success_rate']:.1%}")
            print(f"    • 签约数: {data['contracts']['signed']}")
            print(f"    • 生产计划: {prod['scheduled']} 次")
            print(f"    • 库存快照: {data['inventory_snapshots']} 次")
        
        # 汇总统计
        summary = results["summary"]
        print("\n📋 汇总统计:")
        print(f"  • 追踪 Agent 数: {summary['total_agents']}")
        print(f"  • 总谈判次数: {summary['total_negotiations']}")
        print(f"  • 总签约数: {summary['total_contracts']}")
        print(f"  • 平均成功率: {summary['avg_success_rate']:.1%}")
        print(f"  • 总生产计划: {summary['total_production_scheduled']}")


def save_tournament_results(output_dir: str, scores: dict, track: str, n_steps: int):
    """
    保存比赛结果为 tournament_results.json，供 visualizer 使用
    """
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    rankings = []
    for agent_id, score in sorted_scores:
        rankings.append({
            "agent_type": agent_id,
            "mean": score,
            "std": 0.0,
            "min": score,
            "max": score,
            "count": 1,
        })
    
    tournament_data = {
        "track": track,
        "n_steps": n_steps,
        "n_agents": len(scores),
        "timestamp": datetime.now().isoformat(),
        "rankings": rankings,
        "scores": {k: float(v) for k, v in scores.items()},
    }
    
    results_file = os.path.join(output_dir, "tournament_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(tournament_data, f, ensure_ascii=False, indent=2)
    
    return results_file


def run_std_tournament(n_steps: int = 50, output_dir: str = None, verbose: bool = False, 
                       port: int = 8080, no_server: bool = False):
    """
    运行 Standard 比赛
    
    Args:
        n_steps: 模拟步数 (Standard 推荐 50-200)
        output_dir: 输出目录
        verbose: 是否显示详细信息
        port: 可视化服务器端口
        no_server: 是否不启动可视化服务器
    """
    # 设置输出目录
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = str(ROOT_DIR / "results" / f"std_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("🎮 SCML Standard 比赛示例")
    print("=" * 60)
    print(f"  • 步数: {n_steps}")
    print(f"  • 输出目录: {output_dir}")
    
    # 1. 配置 Tracker
    print("\n📝 配置 Tracker 系统...")
    TrackerManager._loggers.clear()  # 清除之前的 Logger
    TrackerConfig.configure(
        enabled=True,
        log_dir=output_dir,
        console_echo=verbose,
    )
    
    # 2. 准备 Agents
    print("\n🤖 准备 Agents...")
    
    # LitaAgents - 注入 Tracker
    lita_agents = [LitaAgentY, LitaAgentYR, LitaAgentCIR, LitaAgentN, LitaAgentP]
    tracked_lita_agents = inject_tracker_to_agents(lita_agents)
    
    print("  LitaAgents (已追踪):")
    for agent_cls in tracked_lita_agents:
        print(f"    - {agent_cls.__name__}")
    
    # 其他 Agents
    other_agents = [
        GreedyStdAgent,
        RandomStdAgent,
        SyncRandomStdAgent,
    ]
    
    print("  内置 Agents:")
    for agent_cls in other_agents:
        print(f"    - {agent_cls.__name__}")
    
    # Top Agents
    top_agents = list(TOP_AGENTS_2025_STD) + list(TOP_AGENTS_2024_STD)
    
    if top_agents:
        print("  Top Agents:")
        for agent_cls in top_agents[:5]:
            print(f"    - {agent_cls.__name__}")
        if len(top_agents) > 5:
            print(f"    ... 以及 {len(top_agents) - 5} 个其他 Top Agent")
    
    # 组合所有 Agents
    all_agents = tracked_lita_agents + other_agents + top_agents
    print(f"\n  总计: {len(all_agents)} 种 Agent 类型")
    
    # 3. 创建世界
    print(f"\n🌍 创建比赛世界 (n_steps={n_steps})...")
    world = SCML2024StdWorld(
        **SCML2024StdWorld.generate(
            agent_types=all_agents,
            n_steps=n_steps,
            n_processes=2,
        ),
        construct_graphs=True,
    )
    print(f"  创建了 {len(world.agents)} 个 Agent 实例")
    
    # 4. 运行比赛（带进度条）
    print("\n🏃 运行比赛...")
    world.run_with_progress()
    
    # 5. 保存追踪数据
    print("\n💾 保存追踪数据...")
    tracker_log_dir = os.path.join(output_dir, "tracker_logs")
    TrackerManager.save_all(tracker_log_dir)
    
    # 6. 获取比赛结果
    scores = world.scores()
    
    # 7. 分析数据
    print("\n🔍 分析追踪数据...")
    analysis_results = analyze_tracker_data(output_dir)
    
    # 8. 打印结果
    print_analysis(analysis_results, scores)
    
    # 9. 保存 tournament_results.json (用于可视化)
    save_tournament_results(output_dir, scores, "std", n_steps)
    
    print(f"\n✅ 完成！结果已保存到: {output_dir}")
    print(f"  • 追踪数据: {tracker_log_dir}")
    
    # 10. 启动可视化服务器
    if not no_server:
        print("\n🌐 启动可视化服务器...")
        try:
            from scml_analyzer.visualizer import start_server
            start_server(output_dir, port=port, open_browser=True)
        except ImportError:
            print("  ⚠️ 无法导入 scml_analyzer.visualizer")
        except KeyboardInterrupt:
            print("\n👋 服务器已停止")
        except Exception as e:
            print(f"  ⚠️ 启动服务器失败: {e}")
    else:
        print("\n📌 提示: 使用以下命令启动可视化服务器:")
        print(f"  python -m scml_analyzer.visualizer --data \"{output_dir}\"")
    
    return world, scores, analysis_results


def main():
    parser = argparse.ArgumentParser(
        description="运行 SCML Standard 比赛示例"
    )
    parser.add_argument(
        "--n-steps", type=int, default=50,
        help="模拟步数 (默认: 50, Standard 推荐 50-200)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="输出目录 (默认: results/std_<timestamp>)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="显示详细信息"
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
    
    world, scores, analysis = run_std_tournament(
        n_steps=args.n_steps,
        output_dir=args.output_dir,
        verbose=args.verbose,
        port=args.port,
        no_server=args.no_server,
    )


if __name__ == "__main__":
    main()
