"""
最小化测试：直接用 multiprocessing 运行 SCML world

测试目标：确定是 NegMas 的 ProcessPoolExecutor 使用问题还是其他问题
"""

import os
import sys
import time
import multiprocessing
from multiprocessing import Pool

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore')

def run_single_world(config):
    """在子进程中运行单个 world"""
    # 需要在子进程中 import
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    from scml.std import SCML2024StdWorld
    
    try:
        world = SCML2024StdWorld(**config, construct_graphs=False)
        world.run()
        return ("success", world.current_step, world.name)
    except Exception as e:
        import traceback
        return ("error", str(e), traceback.format_exc())

def main():
    # 强制使用 spawn
    if sys.platform == 'win32':
        multiprocessing.set_start_method('spawn', force=True)
    
    print("=" * 70)
    print("最小化 Multiprocessing 测试")
    print(f"Start method: {multiprocessing.get_start_method()}")
    print(f"CPU count: {multiprocessing.cpu_count()}")
    print("=" * 70)
    
    from scml.std import SCML2024StdWorld
    from litaagent_std.litaagent_y import LitaAgentYTracked
    from litaagent_std.litaagent_yr import LitaAgentYRTracked
    from scml_agents.scml2024.standard import AX, CautiousStdAgent
    
    agents = [LitaAgentYTracked, LitaAgentYRTracked, AX, CautiousStdAgent]
    
    # 生成多个 world 配置
    print("\n生成 4 个 World 配置...")
    configs = []
    for i in range(4):
        config = SCML2024StdWorld.generate(
            agent_types=agents,
            n_steps=10,
            n_processes=2,
        )
        configs.append(config)
    print(f"生成了 {len(configs)} 个配置")
    
    # 测试 1: 用 Pool(1) 运行 1 个 world（相当于单线程）
    print("\n--- 测试 1: Pool(1) 运行 1 个 world ---")
    start = time.time()
    try:
        with Pool(1) as pool:
            results = pool.map(run_single_world, [configs[0]])
        print(f"结果: {results}")
        print(f"耗时: {time.time() - start:.1f}秒")
    except Exception as e:
        print(f"错误: {e}")
    
    # 测试 2: 用 Pool(2) 并行运行 2 个 worlds
    print("\n--- 测试 2: Pool(2) 运行 2 个 worlds ---")
    start = time.time()
    try:
        with Pool(2) as pool:
            results = pool.map(run_single_world, configs[:2])
        print(f"结果: {results}")
        print(f"耗时: {time.time() - start:.1f}秒")
    except Exception as e:
        print(f"错误: {e}")
    
    # 测试 3: 用 Pool(4) 并行运行 4 个 worlds
    print("\n--- 测试 3: Pool(4) 运行 4 个 worlds ---")
    start = time.time()
    try:
        with Pool(4) as pool:
            results = pool.map(run_single_world, configs)
        print(f"结果: {results}")
        print(f"耗时: {time.time() - start:.1f}秒")
    except Exception as e:
        print(f"错误: {e}")
    
    print("\n完成!")

if __name__ == "__main__":
    main()
