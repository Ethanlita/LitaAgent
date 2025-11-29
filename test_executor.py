"""
测试 ProcessPoolExecutor 是否有问题

NegMas 使用 concurrent.futures.ProcessPoolExecutor，我们来直接测试它
"""

import os
import sys
import time
from concurrent import futures

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore')

def run_single_world(config):
    """在子进程中运行单个 world"""
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
    print("=" * 70)
    print("测试 ProcessPoolExecutor（NegMas 使用的方式）")
    print("=" * 70)
    
    from scml.std import SCML2024StdWorld
    from litaagent_std.litaagent_y import LitaAgentYTracked
    from litaagent_std.litaagent_yr import LitaAgentYRTracked
    from scml_agents.scml2024.standard import AX, CautiousStdAgent
    
    agents = [LitaAgentYTracked, LitaAgentYRTracked, AX, CautiousStdAgent]
    
    # 生成 world 配置
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
    
    # 测试 1: 使用 ProcessPoolExecutor + submit + as_completed（NegMas 方式）
    print("\n--- 测试 1: ProcessPoolExecutor + as_completed (4 workers, 4 tasks) ---")
    start = time.time()
    try:
        # 模拟 NegMas 的用法
        kwargs = {"max_workers": 4}
        # Python 3.11+ 支持 max_tasks_per_child
        import sys
        if sys.version_info >= (3, 11):
            kwargs["max_tasks_per_child"] = 10
        
        with futures.ProcessPoolExecutor(**kwargs) as executor:
            # 提交所有任务
            future_results = [executor.submit(run_single_world, cfg) for cfg in configs]
            print(f"提交了 {len(future_results)} 个任务")
            
            # 用 as_completed 等待结果（NegMas 的方式）
            for i, future in enumerate(futures.as_completed(future_results)):
                try:
                    result = future.result(timeout=60)
                    print(f"  任务 {i+1}: {result[0]} - {result[2]}")
                except futures.TimeoutError:
                    print(f"  任务 {i+1}: 超时")
                except Exception as e:
                    print(f"  任务 {i+1}: 错误 - {e}")
        
        print(f"耗时: {time.time() - start:.1f}秒")
    except Exception as e:
        print(f"错误: {e}")
    
    # 测试 2: 更多任务
    print("\n--- 测试 2: ProcessPoolExecutor (4 workers, 8 tasks) ---")
    configs8 = configs * 2
    start = time.time()
    try:
        with futures.ProcessPoolExecutor(**kwargs) as executor:
            future_results = [executor.submit(run_single_world, cfg) for cfg in configs8]
            print(f"提交了 {len(future_results)} 个任务")
            
            completed = 0
            for i, future in enumerate(futures.as_completed(future_results)):
                try:
                    result = future.result(timeout=60)
                    completed += 1
                    print(f"  任务 {completed}/{len(configs8)}: {result[0]}")
                except Exception as e:
                    print(f"  任务 {i+1}: 错误 - {e}")
        
        print(f"耗时: {time.time() - start:.1f}秒")
    except Exception as e:
        print(f"错误: {e}")
    
    print("\n完成!")

if __name__ == "__main__":
    main()
