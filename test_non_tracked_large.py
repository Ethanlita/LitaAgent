"""
大规模测试：使用非 Tracked 版本的 agents

目标：确认问题是否出在 TrackerMixin 的线程锁序列化上
"""

import os
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore')

def main():
    print("=" * 70)
    print("大规模测试：非 Tracked Agents + Parallel 模式")
    print("=" * 70)
    
    from scml.utils import anac2024_std
    
    # 使用非 Tracked 版本
    from litaagent_std.litaagent_y import LitaAgentY
    from litaagent_std.litaagent_yr import LitaAgentYR
    from litaagent_std.litaagent_n import LitaAgentN
    from litaagent_std.litaagent_p import LitaAgentP
    from litaagent_std.litaagent_cir import LitaAgentCIR
    
    from scml_agents.scml2024.standard import AX, CautiousStdAgent, DogAgent, Group2, MatchingPennies
    
    # 12 个 agents（不使用 Tracked 版本）
    all_agents = [
        LitaAgentY,
        LitaAgentYR,
        LitaAgentN,
        LitaAgentP,
        LitaAgentCIR,
        AX,
        CautiousStdAgent,
        DogAgent,
        Group2,
        MatchingPennies,
    ]
    
    print(f"\nAgents ({len(all_agents)}):")
    for i, a in enumerate(all_agents):
        print(f"  {i+1}. {a.__name__}")
    
    print(f"\n配置: n_configs=2, n_runs_per_world=1, n_steps=10")
    print(f"并行模式: parallel:0.75 (12 workers)")
    print(f"\n如果这个测试成功完成，说明问题出在 TrackerMixin")
    print(f"如果这个测试也卡死，说明问题不在 TrackerMixin")
    
    print("\n" + "=" * 70)
    print("开始测试...")
    print("=" * 70)
    
    start = time.time()
    try:
        results = anac2024_std(
            competitors=all_agents,
            n_configs=2,
            n_runs_per_world=1,
            n_steps=10,
            parallelism="parallel:0.75",
            compact=True,
            print_exceptions=True,
        )
        
        elapsed = time.time() - start
        print(f"\n✓ 成功完成! 耗时: {elapsed:.1f}秒")
        
        if hasattr(results, 'scores') and results.scores is not None:
            print(f"\n结果摘要:")
            print(results.scores.head(10))
            
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n✗ 失败: {e}")
        print(f"耗时: {elapsed:.1f}秒")

if __name__ == "__main__":
    main()
