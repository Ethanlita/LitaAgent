import os
import sys
import time
import multiprocessing
# Force spawn method to simulate Mac/Windows behavior
# try:
#     multiprocessing.set_start_method('spawn', force=True)
# except RuntimeError:
#     pass

from pathlib import Path
from scml.utils import anac2024_std
from scml_agents.scml2025.standard import AS0
from litaagent_std.litaagent_y import LitaAgentY
from litaagent_std.tracker_mixin import inject_tracker_to_agents
from scml_analyzer.auto_tracker import TrackerConfig, TrackerManager

# 设置环境
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['SCML_TRACKER_LOG_DIR'] = "reproduce_logs"

def run_reproduction():
    print("开始复现死锁问题...")
    print(f"PID: {os.getpid()}")
    
    # 配置 Tracker
    TrackerManager._loggers.clear()
    TrackerConfig.configure(
        enabled=True,
        log_dir="reproduce_logs",
        console_echo=False
    )
    
    # 注入 Tracker
    lita_agents = [LitaAgentY]
    tracked_agents = inject_tracker_to_agents(lita_agents)
    
    competitors = tracked_agents + [AS0]
    
    print(f"参赛 Agents: {[c.__name__ for c in competitors]}")
    
    # 运行并行比赛
    # n_configs=5, n_runs_per_world=2 应该足够产生多个进程和负载
    start_time = time.time()
    try:
        results = anac2024_std(
            competitors=competitors,
            n_configs=2,
            n_runs_per_world=1,
            n_steps=50,
            parallelism='parallel', # 关键点：使用 parallel
            print_exceptions=True,
            verbose=False,
        )
        print("运行完成！")
    except Exception as e:
        print(f"运行出错: {e}")
    
    print(f"总耗时: {time.time() - start_time:.2f} 秒")

if __name__ == "__main__":
    # 确保目录存在
    os.makedirs("reproduce_logs", exist_ok=True)
    run_reproduction()
