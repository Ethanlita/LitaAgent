#!/usr/bin/env python3
"""
精确诊断脚本：验证以下假设
1. 真的有未完成的 future 吗？
2. 子进程都结束了，为什么 future 未完成？
3. 主进程真的卡在等待 Future 的循环吗？
4. timeout 为什么没有效果？

方法：Monkey-patch negmas 的关键函数，插入监控代码
"""

import sys
import os
import time
import threading
import signal
import traceback
from datetime import datetime
from concurrent import futures as cf
from concurrent.futures import ProcessPoolExecutor, as_completed

# 日志文件
LOG_FILE = "diagnose_logs/futures_trace.log"
os.makedirs("diagnose_logs", exist_ok=True)

def log(msg):
    """线程安全的日志记录"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    line = f"[{ts}] {msg}\n"
    with open(LOG_FILE, "a") as f:
        f.write(line)
    print(line, end="", flush=True)

# =====================================================================
# Monkey-patch as_completed 来监控 future 状态
# =====================================================================

original_as_completed = cf.as_completed

def monitored_as_completed(fs, timeout=None):
    """
    包装 as_completed，监控每个 future 的状态
    """
    fs_list = list(fs)
    total = len(fs_list)
    log(f"as_completed() called with {total} futures, timeout={timeout}")
    
    completed_count = 0
    start_time = time.time()
    
    # 启动后台线程，定期报告状态
    stop_monitor = threading.Event()
    
    def monitor_thread():
        while not stop_monitor.is_set():
            elapsed = time.time() - start_time
            pending = sum(1 for f in fs_list if not f.done())
            running = sum(1 for f in fs_list if f.running())
            cancelled = sum(1 for f in fs_list if f.cancelled())
            done = sum(1 for f in fs_list if f.done())
            
            log(f"[Monitor {elapsed:.0f}s] Total={total} Done={done} Running={running} Pending={pending} Cancelled={cancelled}")
            
            # 如果所有 future 都完成了，记录详情
            if pending == 0 and done == total:
                log(f"[Monitor] ALL FUTURES DONE! But we're still in as_completed...")
                # 检查每个 future 的状态
                for i, f in enumerate(fs_list[:5]):  # 只打印前5个
                    try:
                        exc = f.exception(timeout=0)
                        if exc:
                            log(f"  Future[{i}]: exception={type(exc).__name__}: {exc}")
                        else:
                            log(f"  Future[{i}]: completed successfully")
                    except Exception as e:
                        log(f"  Future[{i}]: check failed: {e}")
            
            stop_monitor.wait(10)  # 每10秒检查一次
    
    monitor = threading.Thread(target=monitor_thread, daemon=True)
    monitor.start()
    
    try:
        for future in original_as_completed(fs_list, timeout=timeout):
            completed_count += 1
            elapsed = time.time() - start_time
            
            # 检查这个 future 的状态
            try:
                exc = future.exception(timeout=0)
                status = f"exception: {type(exc).__name__}" if exc else "success"
            except:
                status = "unknown"
            
            log(f"as_completed yielded future {completed_count}/{total} after {elapsed:.1f}s, status={status}")
            yield future
    except cf.TimeoutError:
        elapsed = time.time() - start_time
        pending = sum(1 for f in fs_list if not f.done())
        log(f"as_completed TIMEOUT after {elapsed:.1f}s! Pending futures: {pending}")
        raise
    finally:
        stop_monitor.set()
        monitor.join(timeout=1)
        elapsed = time.time() - start_time
        log(f"as_completed() finished/exited after {elapsed:.1f}s, yielded {completed_count}/{total}")

# 替换 as_completed
cf.as_completed = monitored_as_completed

# =====================================================================
# 信号处理：按 Ctrl+C 时打印当前堆栈
# =====================================================================

def signal_handler(signum, frame):
    log("=" * 60)
    log("SIGNAL RECEIVED - Dumping stack traces of all threads")
    log("=" * 60)
    
    for thread_id, stack in sys._current_frames().items():
        thread_name = "unknown"
        for t in threading.enumerate():
            if t.ident == thread_id:
                thread_name = t.name
                break
        
        log(f"\nThread {thread_id} ({thread_name}):")
        for filename, lineno, name, line in traceback.extract_stack(stack):
            log(f"  File '{filename}', line {lineno}, in {name}")
            if line:
                log(f"    {line}")
    
    log("=" * 60)
    # 不退出，让程序继续运行，用户可以再次 Ctrl+C 退出

signal.signal(signal.SIGUSR1, signal_handler)

log(f"Diagnosis script started. PID={os.getpid()}")
log(f"Send SIGUSR1 to dump stack: kill -USR1 {os.getpid()}")
log("")

# =====================================================================
# 运行锦标赛
# =====================================================================

if __name__ == "__main__":
    # 现在导入 scml，此时 as_completed 已经被替换
    from scml.utils import anac2024_std
    from scml.std.agents import RandomStdAgent, GreedyStdAgent, SyncRandomStdAgent
    
    # 导入我们的 agents
    from litaagent_std.litaagent_y import LitaAgentY
    from litaagent_std.litaagent_yr import LitaAgentYR
    from litaagent_std.litaagent_cir import LitaAgentCIR
    from litaagent_std.litaagent_n import LitaAgentN
    from litaagent_std.litaagent_p import LitaAgentP
    
    # 尝试加载 Top Agents
    try:
        from scml_agents import get_agents
        TOP_AGENTS = get_agents(2025, as_class=True, top_only=True, track='std')
        log(f"Loaded Top Agents: {[a.__name__ for a in TOP_AGENTS]}")
    except Exception as e:
        log(f"Cannot load Top Agents: {e}")
        TOP_AGENTS = []
    
    # 使用和 diagnose_deep.py 相同的配置
    agents = [
        LitaAgentY, LitaAgentYR, LitaAgentCIR, LitaAgentN, LitaAgentP,
    ] + list(TOP_AGENTS) + [
        RandomStdAgent, GreedyStdAgent, SyncRandomStdAgent
    ]
    
    log(f"Starting tournament with {len(agents)} agents")
    log(f"Agents: {[a.__name__ for a in agents]}")
    log(f"Config: n_configs=3, n_steps=50, parallelism='parallel', verbose=False")
    
    try:
        results = anac2024_std(
            competitors=agents,
            n_configs=3,
            n_steps=50,
            parallelism='parallel',
            verbose=False,
            # 不设置 timeout，等待自然结束或 Hang
        )
        
        log("Tournament completed successfully!")
        log(f"Results type: {type(results)}")
        
    except cf.TimeoutError as e:
        log(f"Tournament timed out: {e}")
    except Exception as e:
        log(f"Tournament failed with exception: {type(e).__name__}: {e}")
        traceback.print_exc()
    
    log("Script ending.")
