"""
精确隔离测试 - 找出导致卡死的 agent

策略：
1. 先测试 7 个 LitaAgent（我们自己的）是否能正常运行
2. 再逐一添加 2024 Top agents，找出问题来源
"""

import time
import os
import sys
import multiprocessing
from pathlib import Path

# 设置 Tracker 日志目录
log_dir = Path(__file__).parent / "test_isolate_logs"
log_dir.mkdir(exist_ok=True)
os.environ["SCML_TRACKER_LOG_DIR"] = str(log_dir)

# 抑制 TensorFlow 警告
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

from scml.utils import anac2024_std

from litaagent_std.litaagent_y import LitaAgentYTracked
from litaagent_std.litaagent_ys import LitaAgentYSTracked
from litaagent_std.litaagent_yr import LitaAgentYRTracked
from litaagent_std.litaagent_n import LitaAgentNTracked
from litaagent_std.litaagent_p import LitaAgentPTracked
from litaagent_std.litaagent_cir import LitaAgentCIRTracked
from litaagent_std.litaagent_cirs import LitaAgentCIRSTracked

from scml_agents.scml2024.standard import AX, CautiousStdAgent, DogAgent, Group2, MatchingPennies

def run_test(name: str, agents: list, timeout: int = 120):
    """运行单个测试"""
    print(f"\n{'='*60}")
    print(f"测试: {name}")
    print(f"Agents ({len(agents)}): {[a.__name__ for a in agents]}")
    print(f"{'='*60}")
    
    start = time.time()
    try:
        results = anac2024_std(
            competitors=agents,
            n_configs=1,  # 减少到 1 个配置
            n_runs_per_world=1,
            n_steps=10,
            parallelism="parallel:0.25",  # 用很少的 workers
            total_timeout=timeout,
            compact=True,
            print_exceptions=True,
        )
        elapsed = time.time() - start
        print(f"✓ 成功! 耗时: {elapsed:.2f}秒")
        return True, elapsed
    except Exception as e:
        elapsed = time.time() - start
        print(f"✗ 失败: {e}")
        print(f"  耗时: {elapsed:.2f}秒")
        return False, elapsed

def main():
    print("=" * 70)
    print("精确隔离测试 - 找出导致卡死的 agent")
    print("=" * 70)
    print(f"CPU 核心数: {multiprocessing.cpu_count()}")
    print(f"使用 n_configs=1, n_steps=10, parallelism=parallel:0.25")
    
    # LitaAgents
    lita_agents = [
        LitaAgentYTracked,
        LitaAgentYSTracked,
        LitaAgentYRTracked,
        LitaAgentNTracked,
        LitaAgentPTracked,
        LitaAgentCIRTracked,
        LitaAgentCIRSTracked,
    ]
    
    # 2024 Top agents
    top_2024 = [AX, CautiousStdAgent, DogAgent, Group2, MatchingPennies]
    
    results = []
    
    # 测试 1: 只用 2 个 LitaAgent
    success, elapsed = run_test(
        "2 LitaAgents only",
        [LitaAgentYTracked, LitaAgentYRTracked],
        timeout=60
    )
    results.append(("2 LitaAgents", success))
    if not success:
        print("基本测试失败，停止")
        return
    
    # 测试 2: 只用 4 个 LitaAgent
    success, elapsed = run_test(
        "4 LitaAgents only",
        lita_agents[:4],
        timeout=90
    )
    results.append(("4 LitaAgents", success))
    if not success:
        print("4 LitaAgents 失败，停止")
        return
    
    # 测试 3: 全部 7 个 LitaAgent
    success, elapsed = run_test(
        "All 7 LitaAgents",
        lita_agents,
        timeout=120
    )
    results.append(("7 LitaAgents", success))
    if not success:
        print("7 LitaAgents 失败，问题在我们自己的 agents 中")
        return
    
    # 测试 4-8: 逐一添加 2024 Top agents
    current_agents = lita_agents.copy()
    for top_agent in top_2024:
        current_agents.append(top_agent)
        success, elapsed = run_test(
            f"LitaAgents + {top_agent.__name__}",
            current_agents,
            timeout=150
        )
        results.append((f"+{top_agent.__name__}", success))
        if not success:
            print(f"\n⚠️ 问题 agent 找到: {top_agent.__name__}")
            break
    
    # 汇总
    print("\n" + "=" * 70)
    print("测试结果汇总")
    print("=" * 70)
    for name, success in results:
        status = "✓" if success else "✗"
        print(f"  {status} {name}")

if __name__ == "__main__":
    main()
