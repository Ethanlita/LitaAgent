"""
使用 Dask Distributed 模式测试完整的 12 agents 组合
"""

import os
import time
import sys

# 设置环境
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from pathlib import Path
log_dir = Path(__file__).parent / "test_dask_full_logs"
log_dir.mkdir(exist_ok=True)
os.environ["SCML_TRACKER_LOG_DIR"] = str(log_dir)

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

from scml.utils import anac2024_std
from dask.distributed import Client

from litaagent_std.litaagent_y import LitaAgentYTracked
from litaagent_std.litaagent_ys import LitaAgentYSTracked
from litaagent_std.litaagent_yr import LitaAgentYRTracked
from litaagent_std.litaagent_n import LitaAgentNTracked
from litaagent_std.litaagent_p import LitaAgentPTracked
from litaagent_std.litaagent_cir import LitaAgentCIRTracked
from litaagent_std.litaagent_cirs import LitaAgentCIRSTracked

from scml_agents.scml2024.standard import AX, CautiousStdAgent, DogAgent, Group2, MatchingPennies

def main():
    print("=" * 70)
    print("使用 Dask Distributed 测试 12 个 agents")
    print("=" * 70)
    
    # 所有 agents
    agents = [
        LitaAgentYTracked,
        LitaAgentYSTracked,
        LitaAgentYRTracked,
        LitaAgentNTracked,
        LitaAgentPTracked,
        LitaAgentCIRTracked,
        LitaAgentCIRSTracked,
        AX,
        CautiousStdAgent,
        DogAgent,
        Group2,
        MatchingPennies,
    ]
    
    print(f"\nAgents ({len(agents)}):")
    for i, a in enumerate(agents):
        print(f"  {i+1}. {a.__name__}")
    
    print(f"\n配置: n_configs=2, n_runs_per_world=1, n_steps=10")
    print(f"并行模式: distributed (Dask)")
    
    # 启动 Dask 本地集群
    print("\n启动 Dask 集群...")
    client = Client(n_workers=8, threads_per_worker=1)
    print(f"Dask client: {client}")
    
    print("\n开始模拟...")
    start = time.time()
    
    try:
        results = anac2024_std(
            competitors=agents,
            n_configs=2,
            n_runs_per_world=1,
            n_steps=10,
            parallelism="distributed",
            compact=True,
            print_exceptions=True,
        )
        
        elapsed = time.time() - start
        print(f"\n✓ 成功完成! 耗时: {elapsed:.2f}秒")
        
        # 显示结果摘要
        if hasattr(results, 'scores') and results.scores is not None:
            print(f"\n结果摘要:")
            scores = results.scores
            if len(scores) > 0:
                print(scores.head(15))
                
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n✗ 失败: {e}")
        print(f"耗时: {elapsed:.2f}秒")
    
    finally:
        client.close()
        print("\nDask 集群已关闭")

if __name__ == "__main__":
    main()
