#!/usr/bin/env python
"""
深度诊断并行执行卡死问题

与 run_std_quick 相同规模:
- 9 agents
- n_configs=3
- n_steps=50
- 不设置 max_worlds_per_config（让它生成所有组合）
- 不设置 total_timeout

监控日志输出到文件: diagnose_logs/monitor_*.log
"""

import os
import sys
import time
import atexit
import signal
import threading
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 创建日志目录
LOG_DIR = PROJECT_ROOT / "diagnose_logs"
LOG_DIR.mkdir(exist_ok=True)

# 日志文件
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
MONITOR_LOG = LOG_DIR / f"monitor_{TIMESTAMP}.log"
MAIN_LOG = LOG_DIR / f"main_{TIMESTAMP}.log"


def log_to_file(filepath, message):
    """写入日志文件"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] {message}\n")
        f.flush()


def get_child_processes_detailed():
    """获取所有子进程的详细信息"""
    import subprocess
    try:
        # 获取当前进程的所有子进程
        result = subprocess.run(
            ['ps', '-o', 'pid,ppid,state,%cpu,%mem,etime,cmd', '--ppid', str(os.getpid())],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout
    except Exception as e:
        return f"Error: {e}"


def get_all_python_processes():
    """获取系统中所有 Python 进程"""
    import subprocess
    try:
        result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True, text=True, timeout=5
        )
        lines = result.stdout.split('\n')
        python_lines = [l for l in lines if 'python' in l.lower()]
        return '\n'.join(python_lines)
    except Exception as e:
        return f"Error: {e}"


def get_system_load():
    """获取系统负载"""
    try:
        with open('/proc/loadavg', 'r') as f:
            return f.read().strip()
    except:
        return "N/A"


class ProcessMonitor:
    """进程监控器"""
    
    def __init__(self, log_file, interval=5):
        self.log_file = log_file
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread = None
        self.start_time = time.time()
        self.last_child_count = 0
        self.stall_count = 0
        self.progress_history = []
        
    def start(self):
        """启动监控"""
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        log_to_file(self.log_file, "=" * 80)
        log_to_file(self.log_file, "监控启动")
        log_to_file(self.log_file, f"主进程 PID: {os.getpid()}")
        log_to_file(self.log_file, "=" * 80)
        
    def stop(self):
        """停止监控"""
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2)
        log_to_file(self.log_file, "监控停止")
        
    def _monitor_loop(self):
        """监控循环"""
        check_count = 0
        while not self.stop_event.is_set():
            time.sleep(self.interval)
            check_count += 1
            elapsed = time.time() - self.start_time
            
            # 获取子进程信息
            child_info = get_child_processes_detailed()
            child_lines = [l for l in child_info.split('\n') if l.strip() and 'PID' not in l]
            child_count = len(child_lines)
            
            # 统计状态
            states = {}
            for line in child_lines:
                parts = line.split()
                if len(parts) >= 3:
                    state = parts[2]
                    states[state] = states.get(state, 0) + 1
            
            # 系统负载
            load = get_system_load()
            
            # 记录
            log_to_file(self.log_file, "-" * 60)
            log_to_file(self.log_file, f"检查 #{check_count} | 运行时间: {elapsed:.0f}s | 系统负载: {load}")
            log_to_file(self.log_file, f"子进程数: {child_count} | 状态分布: {states}")
            
            # 检测异常
            if child_count == 0 and elapsed > 30:
                self.stall_count += 1
                log_to_file(self.log_file, f"⚠️ 警告: 没有子进程! 连续 {self.stall_count} 次")
                
                if self.stall_count >= 3:
                    log_to_file(self.log_file, "❌ 可能已卡死! 记录所有 Python 进程:")
                    all_python = get_all_python_processes()
                    log_to_file(self.log_file, all_python)
            else:
                self.stall_count = 0
            
            # 记录子进程变化
            if child_count != self.last_child_count:
                log_to_file(self.log_file, f"子进程数变化: {self.last_child_count} -> {child_count}")
                self.last_child_count = child_count
            
            # 每分钟记录一次详细信息
            if check_count % 12 == 0:  # 每 60 秒
                log_to_file(self.log_file, "=== 详细子进程列表 ===")
                log_to_file(self.log_file, child_info)


def main():
    import matplotlib
    matplotlib.use('Agg')
    
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
        print(f"✓ 加载 Top Agents: {[a.__name__ for a in TOP_AGENTS]}")
    except Exception as e:
        print(f"⚠️ 无法加载 Top Agents: {e}")
        TOP_AGENTS = []
    
    # 与 run_std_quick 相同的 agents
    competitors = [
        LitaAgentY, LitaAgentYR, LitaAgentCIR, LitaAgentN, LitaAgentP,
    ] + list(TOP_AGENTS) + [
        RandomStdAgent, GreedyStdAgent, SyncRandomStdAgent
    ]
    
    print("=" * 70)
    print("🔬 深度诊断 - 与 run_std_quick 相同规模")
    print("=" * 70)
    print(f"参赛者数量: {len(competitors)}")
    print(f"参赛者: {[c.__name__ for c in competitors]}")
    print(f"配置: n_configs=3, n_steps=50, max_worlds_per_config=None")
    print(f"主进程 PID: {os.getpid()}")
    print(f"监控日志: {MONITOR_LOG}")
    print(f"主日志: {MAIN_LOG}")
    print()
    
    # 记录到主日志
    log_to_file(MAIN_LOG, "=" * 80)
    log_to_file(MAIN_LOG, "深度诊断启动")
    log_to_file(MAIN_LOG, f"参赛者: {[c.__name__ for c in competitors]}")
    log_to_file(MAIN_LOG, f"PID: {os.getpid()}")
    log_to_file(MAIN_LOG, "=" * 80)
    
    # 启动监控
    monitor = ProcessMonitor(MONITOR_LOG, interval=5)
    monitor.start()
    
    # 注册退出处理
    def cleanup():
        monitor.stop()
        log_to_file(MAIN_LOG, "程序退出")
    atexit.register(cleanup)
    
    # 信号处理
    def signal_handler(sig, frame):
        log_to_file(MAIN_LOG, f"收到信号: {sig}")
        monitor.stop()
        sys.exit(1)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("🚀 开始比赛 (无超时限制)...")
    log_to_file(MAIN_LOG, "比赛开始")
    
    start_time = time.time()
    
    try:
        results = anac2024_std(
            competitors=competitors,
            n_configs=3,
            n_runs_per_world=1,
            n_steps=50,
            # 不设置 max_worlds_per_config，生成所有组合
            print_exceptions=True,
            verbose=False,
            parallelism='parallel',
            # 不设置 total_timeout
        )
        
        elapsed = time.time() - start_time
        log_to_file(MAIN_LOG, f"比赛完成! 耗时: {elapsed:.1f}s")
        
        print("\n" + "=" * 70)
        print(f"✅ 比赛完成! 耗时: {elapsed:.1f}s")
        print("=" * 70)
        
        if hasattr(results, 'winners') and results.winners:
            winners = [w.split('.')[-1] for w in results.winners]
            print(f"🏆 冠军: {winners}")
            log_to_file(MAIN_LOG, f"冠军: {winners}")
        
        if hasattr(results, 'total_scores') and results.total_scores is not None:
            print("\n📊 排名:")
            sorted_scores = results.total_scores.sort_values("score", ascending=False)
            for rank, (idx, row) in enumerate(sorted_scores.iterrows(), 1):
                agent_name = row["agent_type"].split(".")[-1]
                score = row['score']
                print(f"  {rank}. {agent_name}: {score:.4f}")
                log_to_file(MAIN_LOG, f"排名 {rank}: {agent_name} = {score:.4f}")
                
    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        log_to_file(MAIN_LOG, f"用户中断! 运行时间: {elapsed:.1f}s")
        print(f"\n⚠️ 用户中断 (运行 {elapsed:.1f}s)")
    except Exception as e:
        elapsed = time.time() - start_time
        log_to_file(MAIN_LOG, f"错误: {e}")
        log_to_file(MAIN_LOG, f"运行时间: {elapsed:.1f}s")
        import traceback
        log_to_file(MAIN_LOG, traceback.format_exc())
        print(f"\n❌ 错误: {e}")
        traceback.print_exc()
    finally:
        monitor.stop()
        
    print(f"\n📁 监控日志已保存到: {MONITOR_LOG}")


if __name__ == "__main__":
    main()
