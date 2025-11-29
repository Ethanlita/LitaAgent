"""
测试 NegMas 传递给子进程的对象是否可以 pickle
"""

import os
import pickle
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore')

def test_pickle(name, obj):
    """测试对象是否可以正确 pickle"""
    print(f"测试 {name}...", end=" ")
    try:
        data = pickle.dumps(obj)
        size = len(data)
        obj2 = pickle.loads(data)
        print(f"✓ ({size:,} bytes)")
        return True
    except Exception as e:
        print(f"✗ {type(e).__name__}: {e}")
        return False

def main():
    print("=" * 70)
    print("测试 NegMas 锦标赛传递给子进程的对象")
    print("=" * 70)
    
    # 导入 SCML 相关
    from scml.std import SCML2024StdWorld
    
    from litaagent_std.litaagent_y import LitaAgentYTracked
    from litaagent_std.litaagent_yr import LitaAgentYRTracked
    from scml_agents.scml2024.standard import AX, CautiousStdAgent
    
    agents = [LitaAgentYTracked, LitaAgentYRTracked, AX, CautiousStdAgent]
    
    # 生成 world 配置
    print("\n生成 World 配置...")
    config = SCML2024StdWorld.generate(
        agent_types=agents,
        n_steps=10,
        n_processes=2,
    )
    
    print("\n--- 测试配置字典 ---")
    test_pickle("config (整个配置)", config)
    
    print("\n--- 测试配置中的各部分 ---")
    for key, value in config.items():
        test_pickle(f"config['{key}']", value)
    
    # 测试 world generator
    print("\n--- 测试 World Generator ---")
    test_pickle("SCML2024StdWorld (类)", SCML2024StdWorld)
    test_pickle("SCML2024StdWorld.generate (方法)", SCML2024StdWorld.generate)
    
    # 测试 negmas 内部函数
    print("\n--- 测试 NegMas 内部函数 ---")
    from negmas.tournaments.tournaments import _run_worlds, _get_executor
    test_pickle("_run_worlds", _run_worlds)
    
    print("\n完成!")

if __name__ == "__main__":
    main()
