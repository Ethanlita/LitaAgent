"""
复现 n_configs=2 的卡死问题
"""
import os
import sys
import time
import threading
import queue
from pathlib import Path

# 设置环境
log_dir = Path(__file__).parent / "test_reproduce_logs"
log_dir.mkdir(exist_ok=True)
os.environ["SCML_TRACKER_LOG_DIR"] = str(log_dir)


def run_with_timeout(func, timeout_seconds, name):
    """运行函数并检测超时"""
    result_queue = queue.Queue()
    
    def wrapper():
        try:
            result = func()
            result_queue.put(("success", result))
        except Exception as e:
            import traceback
            result_queue.put(("error", str(e), traceback.format_exc()))
    
    thread = threading.Thread(target=wrapper)
    thread.start()
    
    start = time.time()
    while thread.is_alive() and (time.time() - start) < timeout_seconds:
        thread.join(timeout=10)
        if thread.is_alive():
            print(f"  {name}: {time.time()-start:.0f}秒...")
    
    if thread.is_alive():
        return None, "TIMEOUT"
    
    try:
        return result_queue.get_nowait()
    except queue.Empty:
        return None, "NO_RESULT"


def test_configs():
    """测试不同的 n_configs 设置"""
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
    
    print(f"测试 {len(all_agents)} 个 agents")
    
    # 测试不同配置
    configs = [
        (1, 5),   # n_configs=1, n_steps=5
        (1, 10),  # n_configs=1, n_steps=10
        (2, 5),   # n_configs=2, n_steps=5
        (2, 10),  # n_configs=2, n_steps=10 - 这是卡死的配置
    ]
    
    for n_configs, n_steps in configs:
        print(f"\n{'='*60}")
        print(f"测试 n_configs={n_configs}, n_steps={n_steps}")
        print(f"{'='*60}")
        
        def run_test():
            start = time.time()
            results = anac2024_std(
                competitors=all_agents,
                n_configs=n_configs,
                n_runs_per_world=1,
                n_steps=n_steps,
                parallelism="parallel",
                print_exceptions=True,
            )
            return time.time() - start
        
        result = run_with_timeout(run_test, timeout_seconds=300, name=f"n_configs={n_configs}")
        
        if result[1] == "TIMEOUT":
            print(f"  ⚠ 超时 (>300秒)! 这就是卡死的配置!")
            return n_configs, n_steps
        elif result[0] == "success":
            print(f"  ✓ 成功，耗时: {result[1]:.2f} 秒")
        else:
            print(f"  ✗ 错误: {result[1]}")
    
    return None, None


def main():
    print("=" * 60)
    print("复现卡死问题诊断")
    print("=" * 60)
    
    problem_config = test_configs()
    
    if problem_config[0]:
        print(f"\n{'='*60}")
        print(f"发现问题配置: n_configs={problem_config[0]}, n_steps={problem_config[1]}")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print("所有配置都正常完成")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
