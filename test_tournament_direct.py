"""
更精确的测试：直接模拟 NegMas tournament 的执行方式
"""

import os
import sys
import time
from concurrent import futures

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore')

def main():
    print("=" * 70)
    print("测试：模拟 NegMas Tournament 执行流程")
    print("=" * 70)
    
    from scml.std import SCML2024StdWorld
    from scml.utils import (
        anac2024_std_world_generator,
        anac2024_config_generator_std,
        anac_assigner_std,
        balance_calculator_std,
    )
    from negmas.tournaments import tournament, create_tournament
    
    from litaagent_std.litaagent_y import LitaAgentYTracked
    from litaagent_std.litaagent_yr import LitaAgentYRTracked
    from litaagent_std.litaagent_ys import LitaAgentYSTracked
    from litaagent_std.litaagent_n import LitaAgentNTracked
    from litaagent_std.litaagent_p import LitaAgentPTracked
    from litaagent_std.litaagent_cir import LitaAgentCIRTracked
    from litaagent_std.litaagent_cirs import LitaAgentCIRSTracked
    from scml_agents.scml2024.standard import AX, CautiousStdAgent, DogAgent, Group2, MatchingPennies
    
    # 测试 pickle
    import pickle
    print("\n--- 测试 World Generator 函数的 pickle ---")
    funcs_to_test = [
        ("anac2024_std_world_generator", anac2024_std_world_generator),
        ("anac2024_config_generator_std", anac2024_config_generator_std),
        ("anac_assigner_std", anac_assigner_std),
        ("balance_calculator_std", balance_calculator_std),
    ]
    for name, func in funcs_to_test:
        try:
            data = pickle.dumps(func)
            pickle.loads(data)
            print(f"  ✓ {name}: OK ({len(data)} bytes)")
        except Exception as e:
            print(f"  ✗ {name}: {e}")
    
    # 从 4 个 agents 开始测试
    all_agents = [
        LitaAgentYTracked,
        LitaAgentYRTracked,
        AX,
        CautiousStdAgent,
    ]
    
    print(f"\n--- 测试直接调用 tournament (4 agents, parallel) ---")
    print(f"Agents: {[a.__name__ for a in all_agents]}")
    
    start = time.time()
    try:
        results = tournament(
            competitors=all_agents,
            n_configs=1,
            n_runs_per_world=1,
            parallelism="parallel:0.25",  # 只用 25% CPU
            world_generator=anac2024_std_world_generator,
            config_generator=anac2024_config_generator_std,
            config_assigner=anac_assigner_std,
            score_calculator=balance_calculator_std,
            compact=True,
            n_agents_per_competitor=1,
            n_competitors_per_world=4,
            round_robin=True,
            std_world=True,
            print_exceptions=True,
        )
        print(f"✓ 成功! 耗时: {time.time() - start:.1f}秒")
    except Exception as e:
        print(f"✗ 失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n完成!")

if __name__ == "__main__":
    main()
