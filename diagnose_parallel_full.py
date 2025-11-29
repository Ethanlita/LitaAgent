"""
诊断大规模并行测试卡死问题
测试所有 LitaAgent + Top 5 agents
"""
import os
import sys
import time
import threading
import queue
from pathlib import Path

# 设置环境
log_dir = Path(__file__).parent / "test_diag_full_logs"
log_dir.mkdir(exist_ok=True)
os.environ["SCML_TRACKER_LOG_DIR"] = str(log_dir)


def run_tournament_in_thread(agents, n_configs, n_steps, parallelism, result_queue, timeout_seconds=300):
    """在线程中运行锦标赛，带超时检测"""
    def _run():
        try:
            from scml.utils import anac2024_std
            start = time.time()
            results = anac2024_std(
                competitors=agents,
                n_configs=n_configs,
                n_runs_per_world=1,
                n_steps=n_steps,
                parallelism=parallelism,
                print_exceptions=True,
            )
            duration = time.time() - start
            result_queue.put(("success", duration, results))
        except Exception as e:
            import traceback
            result_queue.put(("error", str(e), traceback.format_exc()))
    
    thread = threading.Thread(target=_run)
    thread.start()
    
    start_wait = time.time()
    while thread.is_alive() and (time.time() - start_wait) < timeout_seconds:
        thread.join(timeout=10)
        if thread.is_alive():
            elapsed = time.time() - start_wait
            print(f"  进行中... ({elapsed:.0f}秒 / {timeout_seconds}秒)")
    
    return thread.is_alive()  # True = 超时


def test_incremental_agents():
    """逐步增加 agent 数量来定位问题"""
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
    
    # 逐步测试
    test_cases = [
        ("2 LitaAgents", [LitaAgentYTracked, LitaAgentYRTracked]),
        ("3 LitaAgents", [LitaAgentYTracked, LitaAgentYRTracked, LitaAgentYSTracked]),
        ("4 LitaAgents", [LitaAgentYTracked, LitaAgentYRTracked, LitaAgentYSTracked, LitaAgentPTracked]),
        ("5 LitaAgents", [LitaAgentYTracked, LitaAgentYRTracked, LitaAgentYSTracked, LitaAgentPTracked, LitaAgentNTracked]),
        ("7 LitaAgents", [LitaAgentYTracked, LitaAgentYRTracked, LitaAgentYSTracked, LitaAgentPTracked, LitaAgentNTracked, LitaAgentCIRTracked, LitaAgentCIRSTracked]),
        ("2 Top Agents", [AS0, XenoSotaAgent]),
        ("5 Top Agents", [AS0, XenoSotaAgent, UltraSuperMiracleSoraFinalAgentZ, PriceTrendStdAgent, PonponAgent]),
        ("3 LitaAgents + 3 Top", [LitaAgentYTracked, LitaAgentYRTracked, LitaAgentYSTracked, AS0, XenoSotaAgent, UltraSuperMiracleSoraFinalAgentZ]),
        ("5 LitaAgents + 5 Top", [LitaAgentYTracked, LitaAgentYRTracked, LitaAgentYSTracked, LitaAgentPTracked, LitaAgentNTracked, AS0, XenoSotaAgent, UltraSuperMiracleSoraFinalAgentZ, PriceTrendStdAgent, PonponAgent]),
        ("7 LitaAgents + 5 Top (完整测试)", [
            LitaAgentYTracked, LitaAgentYRTracked, LitaAgentYSTracked, 
            LitaAgentPTracked, LitaAgentNTracked, LitaAgentCIRTracked, LitaAgentCIRSTracked,
            AS0, XenoSotaAgent, UltraSuperMiracleSoraFinalAgentZ, PriceTrendStdAgent, PonponAgent
        ]),
    ]
    
    for name, agents in test_cases:
        print(f"\n{'='*60}")
        print(f"测试: {name} ({len(agents)} agents)")
        print(f"{'='*60}")
        
        result_queue = queue.Queue()
        timeout = run_tournament_in_thread(
            agents=agents,
            n_configs=1,
            n_steps=5,
            parallelism="parallel",
            result_queue=result_queue,
            timeout_seconds=120
        )
        
        if timeout:
            print(f"  ⚠ 超时！此配置可能导致卡死")
            print(f"  Agents: {[a.__name__ for a in agents]}")
            return name, agents
        
        try:
            result = result_queue.get_nowait()
            if result[0] == "success":
                print(f"  ✓ 成功！耗时: {result[1]:.2f} 秒")
            else:
                print(f"  ✗ 错误: {result[1]}")
                print(result[2])
                return name, agents
        except queue.Empty:
            print(f"  ⚠ 未收到结果")
            return name, agents
    
    return None, None


def test_problem_agent_isolation():
    """隔离测试可能有问题的 agent"""
    from litaagent_std.litaagent_y import LitaAgentYTracked
    from scml_agents.scml2025.standard import (
        AS0, XenoSotaAgent, UltraSuperMiracleSoraFinalAgentZ,
        PriceTrendStdAgent, PonponAgent,
    )
    
    base_agent = LitaAgentYTracked
    test_agents = [AS0, XenoSotaAgent, UltraSuperMiracleSoraFinalAgentZ, PriceTrendStdAgent, PonponAgent]
    
    print("\n隔离测试: 逐一与 LitaAgentYTracked 配对测试 Top Agents")
    
    for agent in test_agents:
        print(f"\n测试: LitaAgentYTracked + {agent.__name__}")
        result_queue = queue.Queue()
        timeout = run_tournament_in_thread(
            agents=[base_agent, agent],
            n_configs=2,
            n_steps=10,
            parallelism="parallel",
            result_queue=result_queue,
            timeout_seconds=120
        )
        
        if timeout:
            print(f"  ⚠ {agent.__name__} 可能导致卡死！")
        else:
            try:
                result = result_queue.get_nowait()
                if result[0] == "success":
                    print(f"  ✓ 成功！耗时: {result[1]:.2f} 秒")
                else:
                    print(f"  ✗ 错误")
            except queue.Empty:
                print(f"  ⚠ 未收到结果")


def main():
    print("=" * 60)
    print("大规模并行测试诊断")
    print("=" * 60)
    print(f"Python 版本: {sys.version}")
    print(f"CPU 数量: {os.cpu_count()}")
    print("=" * 60)
    
    # 测试逐步增加 agent 数量
    problem_config, problem_agents = test_incremental_agents()
    
    if problem_config:
        print(f"\n{'='*60}")
        print(f"发现问题配置: {problem_config}")
        print(f"问题 Agents: {[a.__name__ for a in problem_agents]}")
        print(f"{'='*60}")
        
        # 进一步隔离测试
        test_problem_agent_isolation()
    else:
        print(f"\n{'='*60}")
        print("所有测试通过！Parallel 模式应该正常工作")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
