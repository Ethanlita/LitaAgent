"""
测试：只使用非 Tracked 版本的 agents，验证 PR #21 的假设

假设：如果问题出在 Tracker 的线程锁序列化，那么非 Tracked 版本应该正常工作
"""

import os
import sys
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore')

def main():
    print("=" * 70)
    print("测试：只使用非 Tracked 版本的 agents")
    print("验证 PR #21 假设：Tracker 线程锁是否是卡死原因")
    print("=" * 70)
    
    from scml.utils import anac2024_std
    
    # 导入非 Tracked 版本
    from litaagent_std.litaagent_y import LitaAgentY
    from litaagent_std.litaagent_yr import LitaAgentYR
    from litaagent_std.litaagent_p import LitaAgentP
    from litaagent_std.litaagent_n import LitaAgentN
    
    # 2024 Top agents（没有 Tracker）
    from scml_agents.scml2024.standard import AX, CautiousStdAgent
    
    # 只用非 Tracked 版本
    agents = [
        LitaAgentY,
        LitaAgentYR,
        AX,
        CautiousStdAgent,
    ]
    
    print(f"\nAgents (非 Tracked): {[a.__name__ for a in agents]}")
    print(f"配置: n_configs=1, n_steps=10, parallelism=parallel:0.25")
    
    start = time.time()
    try:
        results = anac2024_std(
            competitors=agents,
            n_configs=1,
            n_runs_per_world=1,
            n_steps=10,
            parallelism="parallel:0.25",
            compact=True,
            print_exceptions=True,
        )
        elapsed = time.time() - start
        print(f"\n✓ 成功! 耗时: {elapsed:.1f}秒")
        print("\n结论: 非 Tracked 版本正常工作，PR #21 的假设正确！")
        print("       问题出在 Tracker 的线程锁序列化")
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n✗ 失败: {e}")
        print(f"耗时: {elapsed:.1f}秒")
        print("\n结论: 非 Tracked 版本也卡死，问题不在 Tracker")

if __name__ == "__main__":
    main()
