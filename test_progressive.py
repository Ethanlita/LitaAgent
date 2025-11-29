"""
渐进式增加 agents 数量测试 - 使用 Dask

找出在什么数量时开始出问题
"""

import os
import time

# 设置环境
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from pathlib import Path
log_dir = Path(__file__).parent / "test_progressive_logs"
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

# 所有 agents 按顺序
ALL_AGENTS = [
    LitaAgentYTracked,
    LitaAgentYRTracked,
    AX,
    CautiousStdAgent,
    LitaAgentYSTracked,
    DogAgent,
    LitaAgentNTracked,
    Group2,
    LitaAgentPTracked,
    MatchingPennies,
    LitaAgentCIRTracked,
    LitaAgentCIRSTracked,
]

def run_test(agents, client, timeout_seconds=120):
    """运行测试，返回是否成功和耗时"""
    import signal
    import threading
    
    result = {"success": False, "elapsed": 0}
    exception_info = {"error": None}
    
    def target():
        try:
            start = time.time()
            anac2024_std(
                competitors=agents,
                n_configs=1,  # 减少配置数
                n_runs_per_world=1,
                n_steps=10,
                parallelism="distributed",
                compact=True,
                print_exceptions=True,
            )
            result["success"] = True
            result["elapsed"] = time.time() - start
        except Exception as e:
            exception_info["error"] = str(e)
            result["elapsed"] = time.time() - start
    
    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout=timeout_seconds)
    
    if thread.is_alive():
        # 超时了
        return False, timeout_seconds, "TIMEOUT"
    
    if exception_info["error"]:
        return False, result["elapsed"], exception_info["error"]
    
    return result["success"], result["elapsed"], None

def main():
    print("=" * 70)
    print("渐进式 Agents 数量测试 (Dask Distributed)")
    print("=" * 70)
    
    # 启动 Dask 集群（只启动一次）
    print("启动 Dask 集群...")
    client = Client(n_workers=4, threads_per_worker=1)
    print(f"Client: {client}\n")
    
    results = []
    
    # 从 2 个 agents 开始测试
    for n in range(2, len(ALL_AGENTS) + 1):
        agents = ALL_AGENTS[:n]
        print(f"\n测试 {n} 个 agents: {[a.__name__ for a in agents][-3:]}", end=" ... ")
        
        success, elapsed, error = run_test(agents, client, timeout_seconds=90)
        
        if success:
            print(f"✓ 成功 ({elapsed:.1f}s)")
            results.append((n, True, elapsed))
        else:
            if error == "TIMEOUT":
                print(f"✗ 超时 (>90s)")
            else:
                print(f"✗ 错误: {error}")
            results.append((n, False, elapsed))
            print(f"\n⚠️ 问题出现在 {n} 个 agents 时")
            break
    
    client.close()
    
    # 汇总
    print("\n" + "=" * 70)
    print("测试结果汇总")
    print("=" * 70)
    for n, success, elapsed in results:
        status = "✓" if success else "✗"
        print(f"  {status} {n} agents: {elapsed:.1f}s")

if __name__ == "__main__":
    main()
