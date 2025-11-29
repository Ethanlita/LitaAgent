#!/usr/bin/env python
"""
快速并行测试 - 用于诊断死锁问题
只运行 1 个 config，少量 agents
"""

import os
import sys
import multiprocessing

# 使用 spawn 启动方法
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import matplotlib
matplotlib.use('Agg')

from scml.utils import anac2024_std
from scml.std.agents import RandomStdAgent, GreedyStdAgent

# 只使用简单的内置 agents 来测试
from litaagent_std.litaagent_y import LitaAgentY
from litaagent_std.litaagent_yr import LitaAgentYR

def main():
    print("=" * 60)
    print("🧪 快速并行测试")
    print("=" * 60)
    
    # 最简配置
    competitors = [
        LitaAgentY,
        LitaAgentYR,
        RandomStdAgent,
        GreedyStdAgent,
    ]
    
    print(f"\n参赛者: {[c.__name__ for c in competitors]}")
    print("配置: n_configs=1, n_steps=20")
    print()
    
    results = anac2024_std(
        competitors=competitors,
        n_configs=1,
        n_runs_per_world=1,
        n_steps=20,
        print_exceptions=True,
        verbose=False,
        parallelism='parallel',
        total_timeout=300,  # 5 分钟超时
    )
    
    print("\n" + "=" * 60)
    print("✅ 测试完成！")
    
    if hasattr(results, 'winners') and results.winners:
        print(f"🏆 冠军: {[w.split('.')[-1] for w in results.winners]}")
    
    if hasattr(results, 'total_scores') and results.total_scores is not None:
        print("\n📊 排名:")
        sorted_scores = results.total_scores.sort_values("score", ascending=False)
        for rank, (idx, row) in enumerate(sorted_scores.iterrows(), 1):
            agent_name = row["agent_type"].split(".")[-1]
            print(f"  {rank}. {agent_name}: {row['score']:.4f}")


if __name__ == "__main__":
    main()
