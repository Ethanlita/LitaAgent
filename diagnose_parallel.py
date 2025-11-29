"""
诊断 Parallel 模式卡死问题
"""
import os
import sys
import time
import multiprocessing
from pathlib import Path
from datetime import datetime

# 设置环境
os.environ["SCML_TRACKER_LOG_DIR"] = str(Path(__file__).parent / "test_diag_logs")


# 全局函数用于 pickle
def _simple_task(x):
    """简单任务"""
    import time
    time.sleep(0.1)
    return x * 2


def test_simple_parallel():
    """测试简单的并行任务"""
    print("测试 1: 简单并行任务...")
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_simple_task, i) for i in range(10)]
        for f in as_completed(futures):
            print(f"  结果: {f.result()}")
    print("  ✓ 简单并行任务正常\n")


def test_scml_import():
    """测试 SCML 导入"""
    print("测试 2: SCML 导入...")
    try:
        from scml.utils import anac2024_std
        from scml.std import SCML2024StdWorld
        print("  ✓ SCML 导入正常\n")
    except Exception as e:
        print(f"  ✗ SCML 导入失败: {e}\n")
        return False
    return True


def test_agent_import():
    """测试 Agent 导入"""
    print("测试 3: Agent 导入...")
    try:
        from litaagent_std.litaagent_y import LitaAgentY, LitaAgentYTracked
        from scml_agents.scml2025.standard import AS0
        print(f"  LitaAgentY: {LitaAgentY}")
        print(f"  LitaAgentYTracked: {LitaAgentYTracked}")
        print(f"  AS0: {AS0}")
        print("  ✓ Agent 导入正常\n")
    except Exception as e:
        print(f"  ✗ Agent 导入失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False
    return True


def test_agent_pickle():
    """测试 Agent 是否可以 pickle"""
    print("测试 4: Agent Pickle...")
    try:
        import pickle
        from litaagent_std.litaagent_y import LitaAgentY, LitaAgentYTracked
        from scml_agents.scml2025.standard import AS0
        
        # 测试类的 pickle
        for agent_cls in [LitaAgentY, LitaAgentYTracked, AS0]:
            pickled = pickle.dumps(agent_cls)
            unpickled = pickle.loads(pickled)
            print(f"  {agent_cls.__name__}: ✓ pickle 正常")
        
        print("  ✓ Agent Pickle 正常\n")
    except Exception as e:
        print(f"  ✗ Agent Pickle 失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False
    return True


def test_single_world_serial():
    """测试单个 World 串行运行"""
    print("测试 5: 单个 World 串行运行...")
    try:
        from scml.utils import anac2024_std
        from litaagent_std.litaagent_y import LitaAgentYTracked
        from scml_agents.scml2025.standard import AS0
        
        start = time.time()
        results = anac2024_std(
            competitors=[LitaAgentYTracked, AS0],
            n_configs=1,
            n_runs_per_world=1,
            n_steps=5,
            parallelism="serial",
            print_exceptions=True,
        )
        duration = time.time() - start
        print(f"  耗时: {duration:.2f} 秒")
        print("  ✓ 单个 World 串行运行正常\n")
    except Exception as e:
        print(f"  ✗ 单个 World 串行运行失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False
    return True


def test_single_world_parallel():
    """测试单个 World 并行运行"""
    print("测试 6: 单个 World 并行运行...")
    try:
        from scml.utils import anac2024_std
        from litaagent_std.litaagent_y import LitaAgentYTracked
        from scml_agents.scml2025.standard import AS0
        
        start = time.time()
        results = anac2024_std(
            competitors=[LitaAgentYTracked, AS0],
            n_configs=1,
            n_runs_per_world=1,
            n_steps=5,
            parallelism="parallel",
            print_exceptions=True,
        )
        duration = time.time() - start
        print(f"  耗时: {duration:.2f} 秒")
        print("  ✓ 单个 World 并行运行正常\n")
    except Exception as e:
        print(f"  ✗ 单个 World 并行运行失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False
    return True


def test_multiple_worlds_parallel():
    """测试多个 World 并行运行"""
    print("测试 7: 多个 World 并行运行 (2 configs)...")
    try:
        from scml.utils import anac2024_std
        from litaagent_std.litaagent_y import LitaAgentYTracked
        from scml_agents.scml2025.standard import AS0
        
        start = time.time()
        results = anac2024_std(
            competitors=[LitaAgentYTracked, AS0],
            n_configs=2,
            n_runs_per_world=1,
            n_steps=5,
            parallelism="parallel",
            print_exceptions=True,
        )
        duration = time.time() - start
        print(f"  耗时: {duration:.2f} 秒")
        print("  ✓ 多个 World 并行运行正常\n")
    except Exception as e:
        print(f"  ✗ 多个 World 并行运行失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False
    return True


def test_top5_serial():
    """测试 Top 5 Agents 串行运行"""
    print("测试 8: Top 5 Agents 串行运行...")
    try:
        from scml.utils import anac2024_std
        from scml_agents.scml2025.standard import (
            AS0, XenoSotaAgent, UltraSuperMiracleSoraFinalAgentZ,
            PriceTrendStdAgent, PonponAgent,
        )
        
        agents = [AS0, XenoSotaAgent, UltraSuperMiracleSoraFinalAgentZ, PriceTrendStdAgent, PonponAgent]
        
        start = time.time()
        results = anac2024_std(
            competitors=agents,
            n_configs=1,
            n_runs_per_world=1,
            n_steps=5,
            parallelism="serial",
            print_exceptions=True,
        )
        duration = time.time() - start
        print(f"  耗时: {duration:.2f} 秒")
        print("  ✓ Top 5 Agents 串行运行正常\n")
    except Exception as e:
        print(f"  ✗ Top 5 Agents 串行运行失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False
    return True


def test_top5_parallel_with_timeout():
    """测试 Top 5 Agents 并行运行（带超时检测）"""
    print("测试 9: Top 5 Agents 并行运行 (带超时检测，最多 120 秒)...")
    
    import threading
    import queue
    
    result_queue = queue.Queue()
    
    def run_tournament():
        try:
            from scml.utils import anac2024_std
            from scml_agents.scml2025.standard import (
                AS0, XenoSotaAgent, UltraSuperMiracleSoraFinalAgentZ,
                PriceTrendStdAgent, PonponAgent,
            )
            
            agents = [AS0, XenoSotaAgent, UltraSuperMiracleSoraFinalAgentZ, PriceTrendStdAgent, PonponAgent]
            
            start = time.time()
            results = anac2024_std(
                competitors=agents,
                n_configs=1,
                n_runs_per_world=1,
                n_steps=5,
                parallelism="parallel",
                print_exceptions=True,
            )
            duration = time.time() - start
            result_queue.put(("success", duration))
        except Exception as e:
            import traceback
            result_queue.put(("error", str(e) + "\n" + traceback.format_exc()))
    
    thread = threading.Thread(target=run_tournament)
    thread.start()
    
    # 等待最多 120 秒
    start_wait = time.time()
    while thread.is_alive() and (time.time() - start_wait) < 120:
        thread.join(timeout=5)
        if thread.is_alive():
            elapsed = time.time() - start_wait
            print(f"  等待中... ({elapsed:.0f}秒)")
    
    if thread.is_alive():
        print("  ⚠ 超时！任务可能卡死")
        print("  可能的原因:")
        print("    1. ProcessPoolExecutor 死锁（Windows 常见问题）")
        print("    2. 某个 Agent 的 counter_all() 或其他方法无限循环")
        print("    3. 内存不足导致 swap")
        print("\n  建议:")
        print("    - 使用 serial 模式进行调试")
        print("    - 检查 Top 5 agents 中是否有问题的 agent")
        print("    - 尝试减少 n_configs")
        return False
    
    try:
        result = result_queue.get_nowait()
        if result[0] == "success":
            print(f"  耗时: {result[1]:.2f} 秒")
            print("  ✓ Top 5 Agents 并行运行正常\n")
            return True
        else:
            print(f"  ✗ 运行出错: {result[1]}\n")
            return False
    except queue.Empty:
        print("  ✗ 未收到结果\n")
        return False


def main():
    print("=" * 60)
    print("SCML Parallel 模式诊断工具")
    print("=" * 60)
    print(f"Python 版本: {sys.version}")
    print(f"操作系统: {sys.platform}")
    print(f"多进程启动方式: {multiprocessing.get_start_method()}")
    print(f"CPU 数量: {multiprocessing.cpu_count()}")
    print("=" * 60 + "\n")
    
    # 创建日志目录
    log_dir = Path(__file__).parent / "test_diag_logs"
    log_dir.mkdir(exist_ok=True)
    
    # 运行测试
    test_simple_parallel()
    
    if not test_scml_import():
        return
    
    if not test_agent_import():
        return
    
    if not test_agent_pickle():
        return
    
    if not test_single_world_serial():
        return
    
    if not test_single_world_parallel():
        print("  ⚠ 单个 World 并行失败，跳过后续并行测试")
        return
    
    if not test_multiple_worlds_parallel():
        print("  ⚠ 多个 World 并行失败，这可能是卡死的原因")
    
    if not test_top5_serial():
        print("  ⚠ Top 5 串行失败，某个 Top agent 可能有问题")
        return
    
    test_top5_parallel_with_timeout()
    
    print("=" * 60)
    print("诊断完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
