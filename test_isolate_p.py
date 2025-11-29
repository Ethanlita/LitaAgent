"""
隔离测试 LitaAgentPTracked

测试策略:
1. 测试 8 agents（不含 LitaAgentPTracked）- 应该成功
2. 测试用 LitaAgentPTracked 替换 Group2（保持 8 个）
3. 找出是 LitaAgentP 本身的问题还是 9 个 agents 的组合问题
"""

import os
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from pathlib import Path
log_dir = Path(__file__).parent / "test_isolate_p_logs"
log_dir.mkdir(exist_ok=True)
os.environ["SCML_TRACKER_LOG_DIR"] = str(log_dir)

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

from scml.utils import anac2024_std
from dask.distributed import Client

from litaagent_std.litaagent_y import LitaAgentYTracked
from litaagent_std.litaagent_yr import LitaAgentYRTracked
from litaagent_std.litaagent_ys import LitaAgentYSTracked
from litaagent_std.litaagent_n import LitaAgentNTracked
from litaagent_std.litaagent_p import LitaAgentPTracked
from litaagent_std.litaagent_cir import LitaAgentCIRTracked
from litaagent_std.litaagent_cirs import LitaAgentCIRSTracked

from scml_agents.scml2024.standard import AX, CautiousStdAgent, DogAgent, Group2, MatchingPennies

def run_test(name, agents, client):
    """运行单个测试"""
    print(f"\n{'='*60}")
    print(f"测试: {name}")
    print(f"Agents ({len(agents)}): {[a.__name__ for a in agents]}")
    print(f"{'='*60}")
    
    start = time.time()
    try:
        results = anac2024_std(
            competitors=agents,
            n_configs=1,
            n_runs_per_world=1,
            n_steps=10,
            parallelism="distributed",
            compact=True,
            print_exceptions=True,
        )
        elapsed = time.time() - start
        print(f"✓ 成功! 耗时: {elapsed:.1f}秒")
        return True
    except Exception as e:
        elapsed = time.time() - start
        print(f"✗ 失败: {e}")
        print(f"耗时: {elapsed:.1f}秒")
        return False

def main():
    print("=" * 70)
    print("隔离测试 LitaAgentPTracked")
    print("=" * 70)
    
    client = Client(n_workers=4, threads_per_worker=1)
    print(f"Dask Client: {client}")
    
    # 测试 1: 8 agents 不含 LitaAgentPTracked（已知应该成功）
    agents_8_no_p = [
        LitaAgentYTracked,
        LitaAgentYRTracked,
        AX,
        CautiousStdAgent,
        LitaAgentYSTracked,
        DogAgent,
        LitaAgentNTracked,
        Group2,
    ]
    
    # 测试 2: 8 agents 用 LitaAgentPTracked 替换 Group2
    agents_8_with_p = [
        LitaAgentYTracked,
        LitaAgentYRTracked,
        AX,
        CautiousStdAgent,
        LitaAgentYSTracked,
        DogAgent,
        LitaAgentNTracked,
        LitaAgentPTracked,  # 替换 Group2
    ]
    
    # 测试 3: 只测试 LitaAgentPTracked 与 2 个简单 agent
    agents_p_simple = [
        LitaAgentYTracked,
        LitaAgentPTracked,
        AX,
    ]
    
    results = []
    
    # 先测试简单组合
    results.append(("P + 简单组合 (3)", run_test("P + 简单组合", agents_p_simple, client)))
    
    # 再测试 8 agents 不含 P
    results.append(("8 agents 不含 P", run_test("8 agents 不含 P", agents_8_no_p, client)))
    
    # 最后测试 8 agents 含 P
    results.append(("8 agents 含 P", run_test("8 agents 含 P", agents_8_with_p, client)))
    
    client.close()
    
    # 汇总
    print("\n" + "=" * 70)
    print("测试结果汇总")
    print("=" * 70)
    for name, success in results:
        status = "✓" if success else "✗"
        print(f"  {status} {name}")
    
    if results[0][1] and results[1][1] and not results[2][1]:
        print("\n⚠️ 问题: LitaAgentPTracked 与其他 7 个 agents 组合时出问题")
    elif not results[0][1]:
        print("\n⚠️ 问题: LitaAgentPTracked 本身有问题")

if __name__ == "__main__":
    main()
