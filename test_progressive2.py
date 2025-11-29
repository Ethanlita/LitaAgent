"""
继续渐进式测试 - 从 7 个 agents 开始
给更长的超时时间，让它有机会完成
"""

import os
import time

# 设置环境
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from pathlib import Path
log_dir = Path(__file__).parent / "test_progressive2_logs"
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

def main():
    print("=" * 70)
    print("渐进式 Agents 数量测试 - 从 7 个开始")
    print("=" * 70)
    
    # 启动 Dask 集群
    print("启动 Dask 集群...")
    client = Client(n_workers=4, threads_per_worker=1)
    print(f"Client: {client}\n")
    
    # 从 9 个 agents 开始（7,8 已成功）
    for n in range(9, len(ALL_AGENTS) + 1):
        agents = ALL_AGENTS[:n]
        agent_names = [a.__name__ for a in agents]
        
        print(f"\n{'='*60}")
        print(f"测试 {n} 个 agents")
        print(f"新增: {agent_names[-1]}")
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
            print(f"✓ {n} agents 成功! 耗时: {elapsed:.1f}秒")
        except Exception as e:
            elapsed = time.time() - start
            print(f"✗ {n} agents 失败: {e}")
            print(f"耗时: {elapsed:.1f}秒")
            break
    
    client.close()
    print("\nDask 集群已关闭")

if __name__ == "__main__":
    main()
