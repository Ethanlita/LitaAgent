"""
诊断测试 - 测试不同 worker 数量下的并行表现

找出最佳的 parallelism 配置
"""

import time
import os
import sys
import multiprocessing
from pathlib import Path

# 设置 Tracker 日志目录
log_dir = Path(__file__).parent / "test_worker_diagnose_logs"
log_dir.mkdir(exist_ok=True)
os.environ["SCML_TRACKER_LOG_DIR"] = str(log_dir)

# 抑制 TensorFlow 警告
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

from scml.utils import anac2024_std

from litaagent_std.litaagent_y import LitaAgentYTracked
from litaagent_std.litaagent_ys import LitaAgentYSTracked
from litaagent_std.litaagent_yr import LitaAgentYRTracked
from litaagent_std.litaagent_n import LitaAgentNTracked
from litaagent_std.litaagent_p import LitaAgentPTracked
from litaagent_std.litaagent_cir import LitaAgentCIRTracked
from litaagent_std.litaagent_cirs import LitaAgentCIRSTracked

from scml_agents.scml2024.standard import AX, CautiousStdAgent, DogAgent, Group2, MatchingPennies

# 获取 CPU 数量
CPU_COUNT = multiprocessing.cpu_count()
print(f"系统 CPU 核心数: {CPU_COUNT}")

def get_all_agents():
    """获取所有要测试的 agents"""
    # 我们自己的 tracked agents
    lita_agents = [
        LitaAgentYTracked,
        LitaAgentYSTracked,
        LitaAgentYRTracked,
        LitaAgentNTracked,
        LitaAgentPTracked,
        LitaAgentCIRTracked,
        LitaAgentCIRSTracked,
    ]
    
    # 2024 Standard 赛道 Top 5 agents
    top_2024 = [AX, CautiousStdAgent, DogAgent, Group2, MatchingPennies]
    
    return lita_agents + top_2024

def test_parallelism(parallelism_value: str, agents: list, timeout: int = 300):
    """测试特定的并行度配置"""
    print(f"\n{'='*60}")
    print(f"测试并行度: {parallelism_value}")
    print(f"Agent 数量: {len(agents)}")
    print(f"超时时间: {timeout} 秒")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # 使用 anac2024_std 运行真正的锦标赛
        results = anac2024_std(
            competitors=agents,
            n_configs=2,
            n_runs_per_world=1,
            n_steps=10,
            parallelism=parallelism_value,
            total_timeout=timeout,
            compact=True,
            print_exceptions=True,
        )
        
        elapsed = time.time() - start_time
        
        # 检查结果
        if results is not None:
            print(f"✓ 成功完成!")
            print(f"  耗时: {elapsed:.2f} 秒")
            return True, elapsed
        else:
            print(f"✗ 结果为 None")
            return False, elapsed
            
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"✗ 失败: {e}")
        return False, elapsed

def main():
    """主测试流程"""
    print("=" * 70)
    print("并行 Worker 数量诊断测试")
    print("=" * 70)
    
    agents = get_all_agents()
    print(f"\n共有 {len(agents)} 个 agents:")
    for i, a in enumerate(agents):
        print(f"  {i+1}. {a.__name__}")
    
    # 测试配置
    # parallel:X 表示使用 X 比例的 CPU (0.25 = 25% = 4核, 0.5 = 50% = 8核)
    test_configs = [
        ("parallel:0.25", 300),   # 4 workers (25% of 16)
        ("parallel:0.5", 300),    # 8 workers (50% of 16)
        ("parallel:0.75", 300),   # 12 workers (75% of 16)
        # ("parallel", 300),       # 16 workers - 已知会卡住，跳过
    ]
    
    results = []
    
    for parallelism, timeout in test_configs:
        success, elapsed = test_parallelism(parallelism, agents, timeout)
        results.append((parallelism, success, elapsed))
        
        if not success:
            print(f"\n⚠️ {parallelism} 失败，停止进一步测试更高的并行度")
            break
        
        # 短暂休息让系统稳定
        time.sleep(3)
    
    # 汇总结果
    print("\n" + "=" * 70)
    print("测试结果汇总")
    print("=" * 70)
    
    max_working_parallelism = None
    for parallelism, success, elapsed in results:
        status = "✓ 成功" if success else "✗ 失败/超时"
        print(f"  {parallelism}: {status} ({elapsed:.2f}秒)")
        if success:
            max_working_parallelism = parallelism
    
    print("\n" + "-" * 70)
    if max_working_parallelism:
        print(f"📌 推荐使用的并行度: {max_working_parallelism}")
    else:
        print(f"⚠️ 建议使用 serial 模式")
    print("-" * 70)

if __name__ == "__main__":
    main()
