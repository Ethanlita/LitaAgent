"""
测试 Agent 实例的 pickle 序列化

重点测试：在模拟环境中创建的 agent 实例
"""

import os
import pickle
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore')

def test_pickle_instance(name, instance):
    """测试实例是否可以正确 pickle"""
    print(f"测试 {name} 实例...", end=" ")
    try:
        data = pickle.dumps(instance)
        size = len(data)
        instance2 = pickle.loads(data)
        print(f"✓ ({size:,} bytes)")
        return True, size
    except Exception as e:
        print(f"✗ 错误: {type(e).__name__}: {e}")
        return False, 0

def main():
    print("=" * 70)
    print("测试 Agent 实例的 Pickle 序列化")
    print("=" * 70)
    
    from scml.std import SCML2024StdWorld
    
    from litaagent_std.litaagent_y import LitaAgentYTracked
    from litaagent_std.litaagent_yr import LitaAgentYRTracked
    from litaagent_std.litaagent_ys import LitaAgentYSTracked
    from litaagent_std.litaagent_n import LitaAgentNTracked
    from litaagent_std.litaagent_p import LitaAgentPTracked
    from litaagent_std.litaagent_cir import LitaAgentCIRTracked
    from litaagent_std.litaagent_cirs import LitaAgentCIRSTracked
    
    from scml_agents.scml2024.standard import AX, CautiousStdAgent, DogAgent, Group2, MatchingPennies
    
    all_agents = [
        LitaAgentYTracked,
        LitaAgentYRTracked,
        LitaAgentYSTracked,
        LitaAgentNTracked,
        LitaAgentPTracked,
        LitaAgentCIRTracked,
        LitaAgentCIRSTracked,
        AX,
        CautiousStdAgent,
        DogAgent,
        Group2,
        MatchingPennies,
    ]
    
    # 创建一个简单的 world 来初始化 agents
    print("\n创建测试 World...")
    world = SCML2024StdWorld(
        **SCML2024StdWorld.generate(
            agent_types=all_agents[:4],  # 只用 4 个 agents
            n_steps=5,
            n_processes=2,
        ),
        construct_graphs=False,
    )
    
    print(f"World 创建成功, 有 {len(world.agents)} 个 agents\n")
    
    # 测试 world 中的每个 agent 实例
    print("--- 测试 World 中的 Agent 实例 ---")
    for agent_id, agent in world.agents.items():
        test_pickle_instance(f"{agent.__class__.__name__} ({agent_id})", agent)
    
    # 测试整个 world
    print("\n--- 测试整个 World ---")
    test_pickle_instance("SCML2024StdWorld", world)
    
    print("\n完成!")

if __name__ == "__main__":
    main()
