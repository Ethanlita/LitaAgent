"""
深度诊断 Parallel 模式卡死问题

思路：
1. 监控子进程的状态
2. 检查是否是某个特定 agent 导致的问题
3. 添加超时机制
"""
import os
import sys
import time
import signal
import multiprocessing
from pathlib import Path
from datetime import datetime

# 设置环境
log_dir = Path(__file__).parent / "test_deep_diag_logs"
log_dir.mkdir(exist_ok=True)
os.environ["SCML_TRACKER_LOG_DIR"] = str(log_dir)

# 设置 multiprocessing 日志
import logging
logging.basicConfig(level=logging.DEBUG)
mp_logger = multiprocessing.log_to_stderr()
mp_logger.setLevel(logging.WARNING)


def test_serial_mode():
    """先用 serial 模式确认基本功能正常"""
    print("=" * 60)
    print("测试 1: Serial 模式基准测试")
    print("=" * 60)
    
    from scml.utils import anac2024_std
    from litaagent_std.litaagent_y import LitaAgentYTracked
    from litaagent_std.litaagent_yr import LitaAgentYRTracked
    from scml_agents.scml2025.standard import AS0, XenoSotaAgent
    
    agents = [LitaAgentYTracked, LitaAgentYRTracked, AS0, XenoSotaAgent]
    
    print(f"Agents: {[a.__name__ for a in agents]}")
    print(f"配置: n_configs=2, n_steps=10, parallelism=serial")
    
    start = time.time()
    results = anac2024_std(
        competitors=agents,
        n_configs=2,
        n_runs_per_world=1,
        n_steps=10,
        parallelism="serial",  # Serial 模式
        print_exceptions=True,
    )
    duration = time.time() - start
    
    print(f"✓ Serial 模式完成，耗时: {duration:.2f} 秒")
    return True


def test_parallel_with_fewer_workers():
    """测试减少 worker 数量是否能避免卡死"""
    print("\n" + "=" * 60)
    print("测试 2: Parallel 模式 - 限制 worker 数量")
    print("=" * 60)
    
    from scml.utils import anac2024_std
    from litaagent_std.litaagent_y import LitaAgentYTracked
    from litaagent_std.litaagent_yr import LitaAgentYRTracked
    from scml_agents.scml2025.standard import AS0, XenoSotaAgent
    
    agents = [LitaAgentYTracked, LitaAgentYRTracked, AS0, XenoSotaAgent]
    
    # 尝试使用 parallel:0.25 来限制 worker 数量
    print(f"Agents: {[a.__name__ for a in agents]}")
    print(f"配置: n_configs=2, n_steps=10, parallelism=parallel:0.25 (使用 25% CPU)")
    
    start = time.time()
    results = anac2024_std(
        competitors=agents,
        n_configs=2,
        n_runs_per_world=1,
        n_steps=10,
        parallelism="parallel:0.25",  # 只使用 25% 的 CPU
        print_exceptions=True,
    )
    duration = time.time() - start
    
    print(f"✓ Parallel:0.25 模式完成，耗时: {duration:.2f} 秒")
    return True


def test_parallel_full():
    """测试完整的 parallel 模式"""
    print("\n" + "=" * 60)
    print("测试 3: Parallel 模式 - 全部 CPU")
    print("=" * 60)
    
    from scml.utils import anac2024_std
    from litaagent_std.litaagent_y import LitaAgentYTracked
    from litaagent_std.litaagent_yr import LitaAgentYRTracked
    from scml_agents.scml2025.standard import AS0, XenoSotaAgent
    
    agents = [LitaAgentYTracked, LitaAgentYRTracked, AS0, XenoSotaAgent]
    
    print(f"Agents: {[a.__name__ for a in agents]}")
    print(f"配置: n_configs=2, n_steps=10, parallelism=parallel")
    
    start = time.time()
    results = anac2024_std(
        competitors=agents,
        n_configs=2,
        n_runs_per_world=1,
        n_steps=10,
        parallelism="parallel",
        print_exceptions=True,
    )
    duration = time.time() - start
    
    print(f"✓ Parallel 模式完成，耗时: {duration:.2f} 秒")
    return True


def test_identify_problematic_agent():
    """逐一测试每个 agent 找出问题"""
    print("\n" + "=" * 60)
    print("测试 4: 隔离测试 - 找出问题 Agent")
    print("=" * 60)
    
    from scml.utils import anac2024_std
    from litaagent_std.litaagent_y import LitaAgentYTracked
    from litaagent_std.litaagent_yr import LitaAgentYRTracked
    from litaagent_std.litaagent_ys import LitaAgentYSTracked
    from litaagent_std.litaagent_p import LitaAgentPTracked
    from litaagent_std.litaagent_n import LitaAgentNTracked
    from litaagent_std.litaagent_cir import LitaAgentCIRTracked
    from litaagent_std.litaagent_cirs import LitaAgentCIRSTracked
    
    from scml_agents.scml2025.standard import (
        AS0, XenoSotaAgent, UltraSuperMiracleSoraFinalAgentZ,
        PriceTrendStdAgent, PonponAgent,
    )
    
    # 基准 agent
    base = LitaAgentYTracked
    
    # 所有要测试的 agents
    test_agents = [
        LitaAgentYRTracked, LitaAgentYSTracked, LitaAgentPTracked,
        LitaAgentNTracked, LitaAgentCIRTracked, LitaAgentCIRSTracked,
        AS0, XenoSotaAgent, UltraSuperMiracleSoraFinalAgentZ,
        PriceTrendStdAgent, PonponAgent,
    ]
    
    for agent in test_agents:
        print(f"\n测试: {base.__name__} vs {agent.__name__}")
        try:
            start = time.time()
            results = anac2024_std(
                competitors=[base, agent],
                n_configs=2,
                n_runs_per_world=1,
                n_steps=10,
                parallelism="parallel",
                print_exceptions=True,
            )
            duration = time.time() - start
            print(f"  ✓ 成功，耗时: {duration:.2f} 秒")
        except Exception as e:
            print(f"  ✗ 失败: {e}")


def test_with_total_timeout():
    """测试使用 total_timeout 参数"""
    print("\n" + "=" * 60)
    print("测试 5: 使用 total_timeout 参数")
    print("=" * 60)
    
    from scml.utils import anac2024_std
    from litaagent_std.litaagent_y import LitaAgentYTracked
    from litaagent_std.litaagent_yr import LitaAgentYRTracked
    from litaagent_std.litaagent_ys import LitaAgentYSTracked
    from litaagent_std.litaagent_p import LitaAgentPTracked
    from litaagent_std.litaagent_n import LitaAgentNTracked
    from litaagent_std.litaagent_cir import LitaAgentCIRTracked
    from litaagent_std.litaagent_cirs import LitaAgentCIRSTracked
    
    from scml_agents.scml2025.standard import (
        AS0, XenoSotaAgent, UltraSuperMiracleSoraFinalAgentZ,
        PriceTrendStdAgent, PonponAgent,
    )
    
    all_agents = [
        LitaAgentYTracked, LitaAgentYRTracked, LitaAgentYSTracked,
        LitaAgentPTracked, LitaAgentNTracked, LitaAgentCIRTracked, LitaAgentCIRSTracked,
        AS0, XenoSotaAgent, UltraSuperMiracleSoraFinalAgentZ, PriceTrendStdAgent, PonponAgent,
    ]
    
    print(f"Agents: {len(all_agents)} 个")
    print(f"配置: n_configs=2, n_steps=10, parallelism=parallel, total_timeout=300")
    
    start = time.time()
    try:
        results = anac2024_std(
            competitors=all_agents,
            n_configs=2,
            n_runs_per_world=1,
            n_steps=10,
            parallelism="parallel",
            print_exceptions=True,
            total_timeout=300,  # 5 分钟超时
        )
        duration = time.time() - start
        print(f"✓ 完成，耗时: {duration:.2f} 秒")
    except Exception as e:
        duration = time.time() - start
        print(f"✗ 异常: {e}")
        print(f"耗时: {duration:.2f} 秒")


def main():
    print("=" * 60)
    print("深度诊断 Parallel 卡死问题")
    print("=" * 60)
    print(f"时间: {datetime.now()}")
    print(f"Python: {sys.version}")
    print(f"CPU 数量: {multiprocessing.cpu_count()}")
    print(f"默认启动方式: {multiprocessing.get_start_method()}")
    print("=" * 60)
    
    # 1. Serial 模式基准测试
    try:
        test_serial_mode()
    except Exception as e:
        print(f"Serial 测试失败: {e}")
        return
    
    # 2. 限制 worker 数量测试
    try:
        test_parallel_with_fewer_workers()
    except Exception as e:
        print(f"限制 worker 测试失败: {e}")
    
    # 3. 全部 CPU 测试
    try:
        test_parallel_full()
    except Exception as e:
        print(f"Full parallel 测试失败: {e}")
    
    # 4. 隔离测试
    try:
        test_identify_problematic_agent()
    except Exception as e:
        print(f"隔离测试失败: {e}")
    
    # 5. 使用 total_timeout
    try:
        test_with_total_timeout()
    except Exception as e:
        print(f"Timeout 测试失败: {e}")
    
    print("\n" + "=" * 60)
    print("诊断完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
