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
import json
import threading
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# 使用 spawn 保持与默认 ProcessPoolExecutor 行为一致
import multiprocessing as mp
try:
    mp.set_start_method("spawn")
except RuntimeError:
    pass

os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 创建日志目录
LOG_DIR = PROJECT_ROOT / "diagnose_logs"
LOG_DIR.mkdir(exist_ok=True)

# 结果目录（确保在沙箱内写入）
RESULTS_ROOT = PROJECT_ROOT / "results"
RESULTS_ROOT.mkdir(exist_ok=True)

# 日志文件
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
MONITOR_LOG = LOG_DIR / f"monitor_{TIMESTAMP}.log"
MAIN_LOG = LOG_DIR / f"main_{TIMESTAMP}.log"
TOURNAMENT_DIR = RESULTS_ROOT / f"clean_run_{TIMESTAMP}"
TOURNAMENT_DIR.mkdir(parents=True, exist_ok=True)
WORKER_TRACE = LOG_DIR / f"worker_trace_{TIMESTAMP}.log"
FUTURE_TRACE = LOG_DIR / f"future_trace_{TIMESTAMP}.log"
MONITOR_TRACE = LOG_DIR / f"executor_monitor_{TIMESTAMP}.log"


def log_to_file(filepath, message):
    """写入日志文件"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] {message}\n")
        f.flush()


def log_trace(event: str, **data):
    """记录 worker 级别的关键事件"""
    payload = {"event": event, "pid": os.getpid(), "ts": time.time()}
    payload.update(data)
    with open(WORKER_TRACE, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        f.flush()


def log_future(event: str, **data):
    """记录 future 状态"""
    payload = {"event": event, "pid": os.getpid(), "ts": time.time()}
    payload.update(data)
    with open(FUTURE_TRACE, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        f.flush()


_ORIGINAL_RUN_WORLDS = None
_REQUESTED_PARALLELISM = None


def traced_run_worlds(
    worlds_params,
    world_generator,
    score_calculator,
    world_progress_callback,
    dry_run,
    save_world_stats,
    override_ran_worlds,
    save_progress_every,
    attempts_path,
    max_attempts,
    verbose,
):
    """顶层定义以支持 spawn pickling"""
    global _ORIGINAL_RUN_WORLDS
    try:
        import negmas.tournaments.tournaments as nt  # 延迟 import 便于 pickling
        run_id = nt._hash(worlds_params)
    except Exception:
        run_id = None
    names = []
    try:
        for wp in worlds_params:
            name = None
            if isinstance(wp, dict):
                wp_info = wp.get("world_params") or {}
                name = wp_info.get("name") or wp_info.get("config_id")
            names.append(name)
    except Exception:
        pass
    log_trace("worker_start", run_id=run_id, names=names)
    atexit.register(lambda: log_trace("worker_exit", run_id=run_id))
    try:
        if _ORIGINAL_RUN_WORLDS is None:
            import negmas.tournaments.tournaments as nt  # type: ignore
            _ORIGINAL_RUN_WORLDS = nt.__dict__.get("_run_worlds")
        result = _ORIGINAL_RUN_WORLDS(
            worlds_params,
            world_generator,
            score_calculator,
            world_progress_callback,
            dry_run,
            save_world_stats,
            override_ran_worlds,
            save_progress_every,
            attempts_path,
            max_attempts,
            verbose,
        )
        log_trace("worker_done", run_id=run_id)
        return result
    except Exception as e:
        log_trace("worker_error", run_id=run_id, error=str(e))
        raise


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
    
    # 给子进程传递环境变量，触发 sitecustomize.py 中的 worker 追踪补丁
    os.environ["SCML_PATCH_WORKER_TRACE"] = "1"
    os.environ["SCML_WORKER_TRACE_FILE"] = str(WORKER_TRACE)
    os.environ["PYTHONFAULTHANDLER"] = "1"
    os.environ["PYTHONPATH"] = f"{PROJECT_ROOT}{os.pathsep}" + os.environ.get("PYTHONPATH", "")

    # 直接在父进程中 monkeypatch negmas._run_worlds（fork 模式会继承，spawn 也可 pickling）
    import negmas.tournaments.tournaments as nt
    global _ORIGINAL_RUN_WORLDS
    _ORIGINAL_RUN_WORLDS = nt._run_worlds
    nt._run_worlds = traced_run_worlds
    # 监控 futures 状态：monkeypatch _submit_all 以在提交时挂回调
    _ORIGINAL_SUBMIT_ALL = nt._submit_all

    def traced_submit_all(
        executor,
        assigned,
        run_ids,
        world_generator,
        score_calculator,
        world_progress_callback,
        override_ran_worlds,
        attempts_path,
        verbose,
        max_attempts,
    ):
        # 先构建 run_id/name 列表，与 future 列表顺序对应
        mapped_run_ids = []
        mapped_names = []
        for worlds_params in assigned:
            rid = nt._hash(worlds_params)
            if rid in run_ids:
                continue
            mapped_run_ids.append(rid)
            names = []
            try:
                for wp in worlds_params:
                    if isinstance(wp, dict):
                        info = wp.get("world_params") or {}
                        names.append(info.get("name") or info.get("config_id"))
            except Exception:
                pass
            mapped_names.append(names)

        future_results, timeout = _ORIGINAL_SUBMIT_ALL(
            executor,
            assigned,
            run_ids,
            world_generator,
            score_calculator,
            world_progress_callback,
            override_ran_worlds,
            attempts_path,
            verbose,
            max_attempts,
        )

        for idx, fut in enumerate(future_results):
            rid = mapped_run_ids[idx] if idx < len(mapped_run_ids) else None
            nm = mapped_names[idx] if idx < len(mapped_names) else None
            log_future("future_submitted", run_id=rid, names=nm)
            # 便于监控时关联
            fut._run_id = rid
            fut._names = nm

            def _cb(f, rid=rid, nm=nm):
                info = {"run_id": rid, "names": nm}
                if f.cancelled():
                    info["state"] = "cancelled"
                else:
                    exc = f.exception()
                    if exc is None:
                        info["state"] = "done"
                    else:
                        info["state"] = "error"
                        info["error"] = repr(exc)
                log_future("future_done", **info)

            fut.add_done_callback(_cb)

        # 记录所有 future 引用，启动一个监控线程（每个 executor 只启动一次）
        monitored = getattr(executor, "_scml_monitored_futures", None)
        if monitored is None:
            monitored = []
            executor._scml_monitored_futures = monitored
        monitored.extend(future_results)

        if not getattr(executor, "_scml_monitor_started", False):
            try:
                t = threading.Thread(
                    target=monitor_executor, args=(executor, monitored), daemon=True
                )
                t.start()
                executor._scml_monitor_started = True
                log_future("monitor_started", max_workers=getattr(executor, "_max_workers", None))
            except Exception as e:
                log_future("monitor_start_error", error=str(e))

        return future_results, timeout

    nt._submit_all = traced_submit_all

    # 替换 executor：提供 loky 后端（parallelism='loky' 或 'loky:<fraction>'）
    import concurrent.futures as cf
    from joblib.externals.loky import ProcessPoolExecutor as LokyExecutor
    _ORIGINAL_GET_EXECUTOR = nt._get_executor

    def _parse_max_workers(parallelism):
        if not isinstance(parallelism, str):
            return None
        if ":" in parallelism:
            try:
                frac = float(parallelism.split(":")[1])
                if 0 < frac <= 1:
                    return max(1, int(os.cpu_count() * frac))
            except Exception:
                return None
        return None

    def traced_get_executor(parallelism, verbose, total_timeout=None, scheduler_ip=None, scheduler_port=None):
        effective = parallelism
        requested = _REQUESTED_PARALLELISM
        if isinstance(requested, str) and requested.startswith("loky"):
            effective = requested
        if isinstance(effective, str) and effective.startswith("loky"):
            max_workers = _parse_max_workers(parallelism)
            exec_kwargs = {}
            if max_workers:
                exec_kwargs["max_workers"] = max_workers
            executor = LokyExecutor(**exec_kwargs)
            return executor, cf.as_completed
        return _ORIGINAL_GET_EXECUTOR(parallelism, verbose, total_timeout, scheduler_ip, scheduler_port)

    nt._get_executor = traced_get_executor

    # 监控线程：定期检查进程存活与 pending futures
    def monitor_executor(executor, futures, interval=10):
        while True:
            time.sleep(interval)
            try:
                procs = getattr(executor, "_processes", {}) or {}
                alive = [pid for pid, p in procs.items() if p.is_alive()]
                pending = [f for f in futures if not f.done()]
                log_future(
                    "executor_monitor",
                    n_processes=len(alive),
                    pids=alive,
                    pending=len(pending),
                )
                # 如果无活跃进程但仍有 pending，记录详细 run_id/名称，必要时可取消（暂不取消，仅日志）
                if len(alive) == 0 and pending:
                    pending_info = []
                    for f in pending:
                        rid = getattr(f, "_run_id", None)
                        nm = getattr(f, "_names", None)
                        pending_info.append({"run_id": rid, "names": nm})
                    log_future("executor_stall", pending=len(pending), info=pending_info)
            except Exception as e:
                log_future("monitor_error", error=str(e))
                break

    from scml.utils import anac2024_std
    from scml.std.agents import RandomStdAgent, GreedyStdAgent, SyncRandomStdAgent
    from litaagent_std.litaagent_y import LitaAgentY
    from litaagent_std.litaagent_yr import LitaAgentYR
    from litaagent_std.litaagent_cir import LitaAgentCIR
    from litaagent_std.litaagent_n import LitaAgentN
    from litaagent_std.litaagent_p import LitaAgentP
    
    try:
        from scml_agents import get_agents
        TOP_AGENTS = get_agents(2025, as_class=True, top_only=5, track='std')
        print(f"✓ 加载 Top Agents ({len(TOP_AGENTS)}): {[a.__name__ for a in TOP_AGENTS]}")
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
    global _REQUESTED_PARALLELISM
    parallelism = os.environ.get("SCML_PARALLELISM", "loky")
    _REQUESTED_PARALLELISM = parallelism
    negmas_parallelism = (
        "parallel" if isinstance(parallelism, str) and parallelism.startswith("loky") else parallelism
    )
    print(f"配置: n_configs=3, n_steps=50, max_worlds_per_config=None")
    print(f"并行模式: {parallelism}（传给 negmas: {negmas_parallelism}）")
    print(f"主进程 PID: {os.getpid()}")
    print(f"监控日志: {MONITOR_LOG}")
    print(f"主日志: {MAIN_LOG}")
    print(f"Negmas 输出目录: {TOURNAMENT_DIR}")
    print(f"Future 追踪: {FUTURE_TRACE}")
    print()
    
    # 记录到主日志
    log_to_file(MAIN_LOG, "=" * 80)
    log_to_file(MAIN_LOG, "深度诊断启动")
    log_to_file(MAIN_LOG, f"参赛者: {[c.__name__ for c in competitors]}")
    log_to_file(MAIN_LOG, f"PID: {os.getpid()}")
    log_to_file(MAIN_LOG, f"并行模式: {parallelism}（negmas: {negmas_parallelism}）")
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
            parallelism=negmas_parallelism,
            tournament_path=TOURNAMENT_DIR,
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
        
        if (
            hasattr(results, 'total_scores')
            and results.total_scores is not None
            and not results.total_scores.empty
            and "score" in results.total_scores
        ):
            print("\n📊 排名:")
            sorted_scores = results.total_scores.sort_values("score", ascending=False)
            for rank, (idx, row) in enumerate(sorted_scores.iterrows(), 1):
                agent_name = row["agent_type"].split(".")[-1]
                score = row['score']
                print(f"  {rank}. {agent_name}: {score:.4f}")
                log_to_file(MAIN_LOG, f"排名 {rank}: {agent_name} = {score:.4f}")
        else:
            print("\n📊 排名信息不可用（total_scores 为空或缺少 score 列）")
            log_to_file(MAIN_LOG, "total_scores 为空或缺少 score 列，跳过排名输出")
                
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
