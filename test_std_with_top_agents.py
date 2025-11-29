"""
Standard Track 验证测试
使用所有 LitaAgent Tracked 版本 + 2025 Top 5 Standard agents
使用 parallel 模式
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# 设置 tracker 日志目录
log_dir = Path(__file__).parent / "test_std_top_logs"
log_dir.mkdir(exist_ok=True)
os.environ["SCML_TRACKER_LOG_DIR"] = str(log_dir)

# 导入 SCML 模块
from scml.utils import anac2024_std

# 导入我们的 Tracked agents
from litaagent_std.litaagent_y import LitaAgentYTracked
from litaagent_std.litaagent_yr import LitaAgentYRTracked
from litaagent_std.litaagent_ys import LitaAgentYSTracked
from litaagent_std.litaagent_p import LitaAgentPTracked
from litaagent_std.litaagent_n import LitaAgentNTracked
from litaagent_std.litaagent_cir import LitaAgentCIRTracked
from litaagent_std.litaagent_cirs import LitaAgentCIRSTracked

# 导入 2025 Top Standard agents
from scml_agents.scml2025.standard import (
    AS0,
    XenoSotaAgent,
    UltraSuperMiracleSoraFinalAgentZ,
    PriceTrendStdAgent,
    PonponAgent,
)


def main():
    print("=" * 60)
    print("Standard Track 验证测试")
    print("=" * 60)
    
    # 我们的 Tracked agents
    our_agents = [
        LitaAgentYTracked,
        LitaAgentYRTracked,
        LitaAgentYSTracked,
        LitaAgentPTracked,
        LitaAgentNTracked,
        LitaAgentCIRTracked,
        LitaAgentCIRSTracked,
    ]
    
    # 2025 Top 5 Standard agents
    top_agents = [
        AS0,
        XenoSotaAgent,
        UltraSuperMiracleSoraFinalAgentZ,
        PriceTrendStdAgent,
        PonponAgent,
    ]
    
    # 合并所有 agents
    all_agents = our_agents + top_agents
    
    print("\n参与测试的 Agents:")
    print("-" * 40)
    print("LitaAgent Tracked 版本:")
    for agent in our_agents:
        print(f"  - {agent.__name__}")
    print("\n2025 Top Standard agents:")
    for agent in top_agents:
        print(f"  - {agent.__name__}")
    
    print(f"\n共 {len(all_agents)} 个 agents")
    print(f"\nTracker 日志目录: {log_dir}")
    print("-" * 40)
    
    # 运行锦标赛 - 使用 parallel 模式
    print("\n开始运行锦标赛 (parallel 模式)...")
    print("这可能需要几分钟时间...\n")
    
    start_time = datetime.now()
    
    results = anac2024_std(
        competitors=all_agents,
        n_configs=2,        # 2 个配置
        n_runs_per_world=1, # 每个 world 运行 1 次
        n_steps=10,         # 每个 world 10 步
        parallelism="parallel",  # 并行模式
        print_exceptions=True,
    )
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "=" * 60)
    print("锦标赛完成!")
    print(f"耗时: {duration:.2f} 秒")
    print("=" * 60)
    
    # 显示结果
    if results is not None:
        print("\n锦标赛结果:")
        print("-" * 40)
        
        # 获取得分
        scores = results.scores
        if scores is not None and len(scores) > 0:
            # 按得分排序
            sorted_scores = scores.sort_values(by='score', ascending=False)
            print(sorted_scores.to_string())
        else:
            print("无得分数据")
        
        # 保存结果
        results_dir = Path(__file__).parent / "test_std_top_results"
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存完整结果
        results_file = results_dir / f"results_{timestamp}.json"
        try:
            # 获取基本结果信息
            result_data = {
                "timestamp": timestamp,
                "duration_seconds": duration,
                "n_configs": 2,
                "n_runs_per_world": 1,
                "n_steps": 10,
                "agents": [a.__name__ for a in all_agents],
                "log_dir": str(log_dir),
            }
            
            if scores is not None and len(scores) > 0:
                result_data["scores"] = scores.to_dict('records')
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
            print(f"\n结果已保存到: {results_file}")
        except Exception as e:
            print(f"保存结果时出错: {e}")
    
    # 检查 tracker 日志
    print("\n" + "-" * 40)
    print("检查 Tracker 日志...")
    tracker_files = list(log_dir.glob("*.json"))
    print(f"生成了 {len(tracker_files)} 个 tracker 日志文件")
    
    if tracker_files:
        print("\n日志文件列表:")
        for f in sorted(tracker_files)[:10]:  # 只显示前10个
            print(f"  - {f.name}")
        if len(tracker_files) > 10:
            print(f"  ... 还有 {len(tracker_files) - 10} 个文件")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print(f"使用 visualizer 查看详细分析:")
    print(f"  python run_scml_analyzer.py {log_dir}")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    main()
