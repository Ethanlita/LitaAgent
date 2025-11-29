#!/usr/bin/env python
"""
诊断并行执行卡死问题

这个脚本会:
1. 运行一个简单的并行任务
2. 监控所有子进程状态
3. 定期打印进度信息
4. 检测卡住的情况
"""

import os
import sys
import time
import signal
import threading
import multiprocessing
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 全局变量用于监控
monitor_stop = threading.Event()
futures_status = {}


def get_process_info():
    """获取当前所有 Python 子进程信息"""
    import subprocess
    try:
        result = subprocess.run(
            ['ps', '-o', 'pid,ppid,state,time,cmd', '--ppid', str(os.getpid())],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout
    except Exception as e:
        return f"Error getting process info: {e}"


def monitor_thread(executor, future_to_id, check_interval=10):
    """监控线程：定期打印状态"""
    start_time = time.time()
    last_completed = 0
    stall_count = 0
    
    while not monitor_stop.is_set():
        time.sleep(check_interval)
        
        elapsed = time.time() - start_time
        
        # 统计 future 状态
        done_count = sum(1 for f in future_to_id.keys() if f.done())
        running_count = sum(1 for f in future_to_id.keys() if f.running())
        pending_count = len(future_to_id) - done_count
        
        print(f"\n{'='*60}")
        print(f"⏱️  监控报告 [{datetime.now().strftime('%H:%M:%S')}] (已运行 {elapsed:.0f}s)")
        print(f"{'='*60}")
        print(f"📊 Future 状态:")
        print(f"   - 已完成: {done_count}/{len(future_to_id)}")
        print(f"   - 运行中: {running_count}")
        print(f"   - 等待中: {pending_count}")
        
        # 检测是否卡住
        if done_count == last_completed and done_count < len(future_to_id):
            stall_count += 1
            print(f"\n⚠️  警告: 进度停滞 (连续 {stall_count} 次检查无新完成)")
            
            if stall_count >= 3:
                print(f"\n🔍 子进程状态:")
                print(get_process_info())
                
                # 打印未完成的 futures
                print(f"\n📋 未完成的任务:")
                for f, task_id in future_to_id.items():
                    if not f.done():
                        state = "running" if f.running() else "pending"
                        print(f"   - Task {task_id}: {state}")
        else:
            stall_count = 0
        
        last_completed = done_count
        
        if done_count == len(future_to_id):
            print("\n✅ 所有任务已完成!")
            break


def simple_task(task_id, duration=2):
    """简单测试任务"""
    import random
    actual_duration = duration + random.random() * 2
    time.sleep(actual_duration)
    return f"Task {task_id} completed in {actual_duration:.2f}s (pid={os.getpid()})"


def run_simple_parallel_test():
    """运行简单的并行测试"""
    print("=" * 60)
    print("🧪 简单并行测试 (不使用 SCML)")
    print("=" * 60)
    
    n_tasks = 20
    max_workers = min(8, multiprocessing.cpu_count())
    
    print(f"\n配置:")
    print(f"  - 任务数: {n_tasks}")
    print(f"  - 最大工作进程: {max_workers}")
    print(f"  - 主进程 PID: {os.getpid()}")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交任务
        future_to_id = {}
        for i in range(n_tasks):
            future = executor.submit(simple_task, i)
            future_to_id[future] = i
        
        print(f"\n📤 已提交 {len(future_to_id)} 个任务")
        
        # 启动监控线程
        monitor = threading.Thread(
            target=monitor_thread, 
            args=(executor, future_to_id, 5),
            daemon=True
        )
        monitor.start()
        
        # 使用 as_completed 收集结果
        print("\n🔄 等待任务完成...\n")
        completed = 0
        for future in as_completed(future_to_id.keys(), timeout=120):
            try:
                result = future.result(timeout=10)
                completed += 1
                print(f"  [{completed}/{n_tasks}] {result}")
            except Exception as e:
                print(f"  ❌ Task {future_to_id[future]} failed: {e}")
        
        monitor_stop.set()
    
    print("\n" + "=" * 60)
    print("✅ 简单并行测试完成!")
    print("=" * 60)


def run_scml_parallel_test():
    """运行 SCML 并行测试（带监控）"""
    print("\n" + "=" * 60)
    print("🎮 SCML 并行测试")
    print("=" * 60)
    
    import matplotlib
    matplotlib.use('Agg')
    
    from scml.utils import anac2024_std
    from scml.std.agents import RandomStdAgent, GreedyStdAgent
    from litaagent_std.litaagent_y import LitaAgentY
    
    competitors = [
        LitaAgentY,
        RandomStdAgent,
        GreedyStdAgent,
    ]
    
    print(f"\n配置:")
    print(f"  - 参赛者: {[c.__name__ for c in competitors]}")
    print(f"  - n_configs: 1")
    print(f"  - n_steps: 20")
    print(f"  - 主进程 PID: {os.getpid()}")
    
    # 启动一个后台监控线程
    def background_monitor():
        start = time.time()
        while not monitor_stop.is_set():
            time.sleep(15)
            elapsed = time.time() - start
            print(f"\n⏱️  [{elapsed:.0f}s] 子进程状态:")
            print(get_process_info())
    
    monitor = threading.Thread(target=background_monitor, daemon=True)
    monitor.start()
    
    print("\n🚀 开始比赛...\n")
    
    try:
        results = anac2024_std(
            competitors=competitors,
            n_configs=1,
            n_runs_per_world=1,
            n_steps=20,
            print_exceptions=True,
            verbose=False,
            parallelism='parallel',
            total_timeout=180,  # 3 分钟超时
        )
        
        monitor_stop.set()
        
        print("\n✅ SCML 测试完成!")
        
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
        monitor_stop.set()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        monitor_stop.set()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="诊断并行执行问题")
    parser.add_argument("--simple", action="store_true", help="只运行简单并行测试")
    parser.add_argument("--scml", action="store_true", help="只运行 SCML 并行测试")
    args = parser.parse_args()
    
    if args.simple or (not args.simple and not args.scml):
        run_simple_parallel_test()
    
    if args.scml or (not args.simple and not args.scml):
        monitor_stop.clear()
        run_scml_parallel_test()


if __name__ == "__main__":
    main()
