#!/usr/bin/env python
"""
诊断并行执行卡死问题 - 渐进式测试

逐步增加规模，找到卡死的临界点
"""

import os
import sys
import time
import threading
import multiprocessing
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import matplotlib
matplotlib.use('Agg')

from scml.utils import anac2024_std
from scml.std.agents import RandomStdAgent, GreedyStdAgent, SyncRandomStdAgent

# LitaAgent 系列
from litaagent_std.litaagent_y import LitaAgentY
from litaagent_std.litaagent_yr import LitaAgentYR
from litaagent_std.litaagent_cir import LitaAgentCIR
from litaagent_std.litaagent_n import LitaAgentN
from litaagent_std.litaagent_p import LitaAgentP

# Top Agents
try:
    from scml_agents import get_agents
    TOP_AGENTS_2025 = get_agents(2025, as_class=True, top_only=True, track='std')
    print(f"✓ 加载 2025 Standard Top Agents: {[a.__name__ for a in TOP_AGENTS_2025]}")
except Exception as e:
    print(f"⚠️ 无法加载 2025 Top Agents: {e}")
    TOP_AGENTS_2025 = []

monitor_stop = threading.Event()


def get_process_info():
    """获取子进程信息"""
    import subprocess
    try:
        result = subprocess.run(
            ['ps', '-o', 'pid,ppid,state,%cpu,%mem,time,cmd', '--ppid', str(os.getpid())],
            capture_output=True, text=True, timeout=5
        )
        lines = result.stdout.strip().split('\n')
        # 过滤出 python 进程
        python_lines = [l for l in lines if 'python' in l.lower() or 'PID' in l]
        return '\n'.join(python_lines)
    except Exception as e:
        return f"Error: {e}"


def background_monitor(interval=20):
    """后台监控线程"""
    start = time.time()
    while not monitor_stop.is_set():
        time.sleep(interval)
        elapsed = time.time() - start
        print(f"\n{'='*50}")
        print(f"⏱️  [{elapsed:.0f}s] 进程状态监控")
        print(f"{'='*50}")
        print(get_process_info())
        print(f"{'='*50}\n")


def run_test(competitors, n_configs, n_steps, timeout_seconds, test_name):
    """运行单个测试"""
    print("\n" + "=" * 70)
    print(f"🧪 {test_name}")
    print("=" * 70)
    print(f"  - 参赛者数量: {len(competitors)}")
    print(f"  - 参赛者: {[c.__name__ for c in competitors]}")
    print(f"  - n_configs: {n_configs}")
    print(f"  - n_steps: {n_steps}")
    print(f"  - 超时: {timeout_seconds}s")
    print(f"  - 主进程 PID: {os.getpid()}")
    
    monitor_stop.clear()
    monitor = threading.Thread(target=background_monitor, args=(15,), daemon=True)
    monitor.start()
    
    start_time = time.time()
    success = False
    
    try:
        results = anac2024_std(
            competitors=competitors,
            n_configs=n_configs,
            n_runs_per_world=1,
            n_steps=n_steps,
            print_exceptions=True,
            verbose=False,
            parallelism='parallel',
            total_timeout=timeout_seconds,
        )
        
        elapsed = time.time() - start_time
        success = True
        
        print(f"\n✅ {test_name} 完成! (耗时 {elapsed:.1f}s)")
        
        if hasattr(results, 'winners') and results.winners:
            print(f"🏆 冠军: {[w.split('.')[-1] for w in results.winners]}")
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ {test_name} 失败! (耗时 {elapsed:.1f}s)")
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        monitor_stop.set()
        time.sleep(1)
    
    return success


def main():
    print("=" * 70)
    print("🔍 渐进式并行测试 - 寻找卡死临界点")
    print("=" * 70)
    
    # 测试序列：逐步增加复杂度
    tests = [
        # (competitors, n_configs, n_steps, timeout, name)
        (
            [LitaAgentY, RandomStdAgent, GreedyStdAgent],
            1, 20, 120,
            "测试 1: 3 agents, 1 config, 20 steps"
        ),
        (
            [LitaAgentY, LitaAgentYR, RandomStdAgent, GreedyStdAgent],
            1, 30, 180,
            "测试 2: 4 agents, 1 config, 30 steps"
        ),
        (
            [LitaAgentY, LitaAgentYR, LitaAgentCIR, RandomStdAgent, GreedyStdAgent],
            1, 50, 300,
            "测试 3: 5 agents, 1 config, 50 steps"
        ),
        (
            [LitaAgentY, LitaAgentYR, LitaAgentCIR, LitaAgentN, LitaAgentP, 
             RandomStdAgent, GreedyStdAgent],
            2, 50, 600,
            "测试 4: 7 agents, 2 configs, 50 steps"
        ),
    ]
    
    # 如果有 Top Agents，添加一个额外测试
    if TOP_AGENTS_2025:
        tests.append((
            [LitaAgentY, LitaAgentYR] + list(TOP_AGENTS_2025)[:2] + [RandomStdAgent],
            2, 50, 600,
            "测试 5: 含 Top Agent, 2 configs, 50 steps"
        ))
    
    results = []
    for competitors, n_configs, n_steps, timeout, name in tests:
        success = run_test(competitors, n_configs, n_steps, timeout, name)
        results.append((name, success))
        
        if not success:
            print(f"\n⚠️ 测试失败，停止后续测试")
            break
        
        # 短暂休息
        time.sleep(2)
    
    # 总结
    print("\n" + "=" * 70)
    print("📊 测试总结")
    print("=" * 70)
    for name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"  {status}: {name}")


if __name__ == "__main__":
    main()
