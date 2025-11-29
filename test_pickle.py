"""
测试 Agent 的 pickle 序列化

如果 agent 无法正确序列化，会导致 multiprocessing 死锁
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
        # 序列化
        data = pickle.dumps(obj)
        size = len(data)
        # 反序列化
        obj2 = pickle.loads(data)
        print(f"✓ ({size:,} bytes)")
        return True, size
    except Exception as e:
        print(f"✗ 错误: {e}")
        return False, 0

def main():
    print("=" * 70)
    print("测试 Agent 类的 Pickle 序列化")
    print("=" * 70)
    
    # 导入所有 agents
    from litaagent_std.litaagent_y import LitaAgentY, LitaAgentYTracked
    from litaagent_std.litaagent_yr import LitaAgentYR, LitaAgentYRTracked
    from litaagent_std.litaagent_ys import LitaAgentYSTracked
    from litaagent_std.litaagent_n import LitaAgentN, LitaAgentNTracked
    from litaagent_std.litaagent_p import LitaAgentP, LitaAgentPTracked
    from litaagent_std.litaagent_cir import LitaAgentCIR, LitaAgentCIRTracked
    from litaagent_std.litaagent_cirs import LitaAgentCIRSTracked
    
    from scml_agents.scml2024.standard import AX, CautiousStdAgent, DogAgent, Group2, MatchingPennies
    
    print("\n--- LitaAgent 类（非 Tracked）---")
    agents_base = [
        ("LitaAgentY", LitaAgentY),
        ("LitaAgentYR", LitaAgentYR),
        ("LitaAgentN", LitaAgentN),
        ("LitaAgentP", LitaAgentP),
        ("LitaAgentCIR", LitaAgentCIR),
    ]
    
    for name, cls in agents_base:
        test_pickle(name, cls)
    
    print("\n--- LitaAgent 类（Tracked 版本）---")
    agents_tracked = [
        ("LitaAgentYTracked", LitaAgentYTracked),
        ("LitaAgentYRTracked", LitaAgentYRTracked),
        ("LitaAgentYSTracked", LitaAgentYSTracked),
        ("LitaAgentNTracked", LitaAgentNTracked),
        ("LitaAgentPTracked", LitaAgentPTracked),
        ("LitaAgentCIRTracked", LitaAgentCIRTracked),
        ("LitaAgentCIRSTracked", LitaAgentCIRSTracked),
    ]
    
    for name, cls in agents_tracked:
        test_pickle(name, cls)
    
    print("\n--- 2024 Top Agents ---")
    agents_2024 = [
        ("AX", AX),
        ("CautiousStdAgent", CautiousStdAgent),
        ("DogAgent", DogAgent),
        ("Group2", Group2),
        ("MatchingPennies", MatchingPennies),
    ]
    
    for name, cls in agents_2024:
        test_pickle(name, cls)
    
    print("\n--- 测试 TrackerMixin ---")
    from litaagent_std.tracker_mixin import TrackerMixin
    test_pickle("TrackerMixin", TrackerMixin)
    
    print("\n完成!")

if __name__ == "__main__":
    main()
