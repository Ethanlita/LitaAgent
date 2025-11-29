#!/usr/bin/env python
"""
精确诊断 negmas 并行执行卡死问题

直接监控 ProcessPoolExecutor 的 as_completed 行为
"""

import os
import sys
import time
import signal
import threading
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import matplotlib
matplotlib.use('Agg')


def get_child_processes():
    """获取所有子进程 PID 和状态"""
    import subprocess
    try:
        result = subprocess.run(
            ['ps', '-o', 'pid,state,time', '--ppid', str(os.getpid())],
            capture_output=True, text=True, timeout=5
        )
        lines = result.stdout.strip().split('\n')[1:]  # Skip header
        processes = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                pid = int(parts[0])
                state = parts[1]
                processes.append((pid, state))
        return processes
    except Exception as e:
        return []


def run_with_monitoring():
    """运行比赛并监控 as_completed 行为"""
    from scml.utils import anac2024_std
    from scml.std.agents import RandomStdAgent, GreedyStdAgent, SyncRandomStdAgent
    from litaagent_std.litaagent_y import LitaAgentY
    from litaagent_std.litaagent_yr import LitaAgentYR
    from litaagent_std.litaagent_cir import LitaAgentCIR
    from litaagent_std.litaagent_n import LitaAgentN
    from litaagent_std.litaagent_p import LitaAgentP
    
    try:
        from scml_agents import get_agents
        TOP_AGENTS = get_agents(2025, as_class=True, top_only=True, track='std')
    except:
        TOP_AGENTS = []
    
    competitors = [
        LitaAgentY, LitaAgentYR, LitaAgentCIR, LitaAgentN, LitaAgentP,
    ] + list(TOP_AGENTS) + [
        RandomStdAgent, GreedyStdAgent, SyncRandomStdAgent
    ]
    
    print("=" * 70)
    print("🔬 精确监控测试")
    print("=" * 70)
    print(f"参赛者数量: {len(competitors)}")
    print(f"参赛者: {[c.__name__ for c in competitors]}")
    print(f"配置: n_configs=3, n_steps=50")
    print(f"主进程 PID: {os.getpid()}")
    print()
    
    # 设置一个监控线程
    stop_monitor = threading.Event()
    last_progress = [0]
    stall_start = [None]
    
    def monitor():
        while not stop_monitor.is_set():
            time.sleep(10)
            procs = get_child_processes()
            n_running = len([p for p in procs if p[1] in ['R', 'S', 'D']])
            n_zombie = len([p for p in procs if p[1] == 'Z'])
            
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
                  f"子进程: {n_running} running, {n_zombie} zombie | "
                  f"States: {[p[1] for p in procs[:10]]}")
            
            # 如果没有活跃子进程但程序还在运行，可能是卡住了
            if n_running == 0 and not stop_monitor.is_set():
                print("⚠️  警告: 没有活跃的子进程!")
    
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    
    # 使用较小的 max_worlds_per_config 来限制 worlds 数量
    print("🚀 开始比赛 (max_worlds_per_config=20)...")
    
    try:
        results = anac2024_std(
            competitors=competitors,
            n_configs=3,
            n_runs_per_world=1,
            n_steps=50,
            max_worlds_per_config=20,  # 限制每个 config 的 world 数量！
            print_exceptions=True,
            verbose=False,
            parallelism='parallel',
            total_timeout=600,  # 10 分钟超时
        )
        
        stop_monitor.set()
        
        print("\n" + "=" * 70)
        print("✅ 比赛完成!")
        print("=" * 70)
        
        if hasattr(results, 'winners') and results.winners:
            print(f"🏆 冠军: {[w.split('.')[-1] for w in results.winners]}")
        
        if hasattr(results, 'total_scores') and results.total_scores is not None:
            print("\n📊 排名:")
            sorted_scores = results.total_scores.sort_values("score", ascending=False)
            for rank, (idx, row) in enumerate(sorted_scores.iterrows(), 1):
                agent_name = row["agent_type"].split(".")[-1]
                print(f"  {rank}. {agent_name}: {row['score']:.4f}")
                
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断")
        stop_monitor.set()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        stop_monitor.set()


if __name__ == "__main__":
    run_with_monitoring()
