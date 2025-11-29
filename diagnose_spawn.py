"""
验证 multiprocessing spawn 问题

测试：在子进程中 import agents 是否会卡住
"""

import os
import sys
import time
import multiprocessing
from multiprocessing import Process, Queue

# 抑制警告
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def worker_import_test(queue, agent_module, agent_name):
    """在子进程中测试 import"""
    try:
        start = time.time()
        # 动态 import
        module = __import__(agent_module, fromlist=[agent_name])
        agent_class = getattr(module, agent_name)
        elapsed = time.time() - start
        queue.put(("success", agent_name, elapsed))
    except Exception as e:
        queue.put(("error", agent_name, str(e)))

def test_agent_import(agent_module: str, agent_name: str, timeout: int = 30):
    """测试单个 agent 的 import"""
    print(f"测试 import: {agent_name} ...", end=" ", flush=True)
    
    queue = Queue()
    p = Process(target=worker_import_test, args=(queue, agent_module, agent_name))
    p.start()
    p.join(timeout=timeout)
    
    if p.is_alive():
        p.terminate()
        p.join()
        print(f"✗ 超时 ({timeout}秒)")
        return False
    
    if not queue.empty():
        status, name, info = queue.get()
        if status == "success":
            print(f"✓ 成功 ({info:.2f}秒)")
            return True
        else:
            print(f"✗ 错误: {info}")
            return False
    else:
        print(f"✗ 无响应")
        return False

def main():
    # 强制使用 spawn（Windows 默认）
    multiprocessing.set_start_method('spawn', force=True)
    
    print("=" * 60)
    print("测试子进程 import 是否会卡住")
    print(f"Start method: {multiprocessing.get_start_method()}")
    print("=" * 60)
    
    # 要测试的 agents
    agents_to_test = [
        # LitaAgents
        ("litaagent_std.litaagent_y", "LitaAgentYTracked"),
        ("litaagent_std.litaagent_yr", "LitaAgentYRTracked"),
        ("litaagent_std.litaagent_ys", "LitaAgentYSTracked"),
        ("litaagent_std.litaagent_n", "LitaAgentNTracked"),
        ("litaagent_std.litaagent_p", "LitaAgentPTracked"),
        ("litaagent_std.litaagent_cir", "LitaAgentCIRTracked"),
        ("litaagent_std.litaagent_cirs", "LitaAgentCIRSTracked"),
        # 2024 Top agents
        ("scml_agents.scml2024.standard", "AX"),
        ("scml_agents.scml2024.standard", "CautiousStdAgent"),
        ("scml_agents.scml2024.standard", "DogAgent"),
        ("scml_agents.scml2024.standard", "Group2"),
        ("scml_agents.scml2024.standard", "MatchingPennies"),
    ]
    
    results = []
    for module, name in agents_to_test:
        success = test_agent_import(module, name, timeout=30)
        results.append((name, success))
    
    print("\n" + "=" * 60)
    print("结果汇总")
    print("=" * 60)
    
    failed = [name for name, success in results if not success]
    if failed:
        print(f"⚠️ 以下 agents 在子进程中 import 有问题:")
        for name in failed:
            print(f"  - {name}")
    else:
        print("✓ 所有 agents 都能在子进程中正常 import")
        print("\n问题可能出在 NegMas 的 ProcessPoolExecutor 使用方式上")

if __name__ == "__main__":
    main()
