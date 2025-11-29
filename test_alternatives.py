"""
测试 NegMas 并行死锁问题的解决方案

方案：使用 loky 替代 ProcessPoolExecutor
loky 是一个更健壮的多进程库，专门为解决 Windows 上的各种问题设计
"""

import os
import time
import sys

# 设置环境
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from pathlib import Path
log_dir = Path(__file__).parent / "test_loky_logs"
log_dir.mkdir(exist_ok=True)
os.environ["SCML_TRACKER_LOG_DIR"] = str(log_dir)

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

def test_with_serial():
    """使用 serial 模式作为基准"""
    print("\n" + "=" * 60)
    print("测试 Serial 模式 (基准)")
    print("=" * 60)
    
    from scml.utils import anac2024_std
    from litaagent_std.litaagent_y import LitaAgentYTracked
    from litaagent_std.litaagent_yr import LitaAgentYRTracked
    from scml_agents.scml2024.standard import AX, CautiousStdAgent
    
    agents = [LitaAgentYTracked, LitaAgentYRTracked, AX, CautiousStdAgent]
    
    start = time.time()
    results = anac2024_std(
        competitors=agents,
        n_configs=1,
        n_runs_per_world=1,
        n_steps=10,
        parallelism="serial",
        compact=True,
        print_exceptions=True,
    )
    elapsed = time.time() - start
    print(f"✓ Serial 完成! 耗时: {elapsed:.2f}秒")
    return elapsed

def test_check_loky():
    """检查是否安装了 loky"""
    try:
        import loky
        print(f"\n✓ loky 已安装: {loky.__version__}")
        return True
    except ImportError:
        print("\n✗ loky 未安装")
        print("  建议运行: pip install loky")
        return False

def test_dask_distributed():
    """测试使用 dask distributed 模式"""
    print("\n" + "=" * 60)
    print("测试 Dask Distributed 模式")
    print("=" * 60)
    
    try:
        import dask
        from dask.distributed import Client
        print(f"dask 版本: {dask.__version__}")
    except ImportError:
        print("✗ dask 未安装")
        print("  建议运行: pip install dask[complete]")
        return None
    
    from scml.utils import anac2024_std
    from litaagent_std.litaagent_y import LitaAgentYTracked
    from litaagent_std.litaagent_yr import LitaAgentYRTracked
    from scml_agents.scml2024.standard import AX, CautiousStdAgent
    
    agents = [LitaAgentYTracked, LitaAgentYRTracked, AX, CautiousStdAgent]
    
    try:
        # 启动本地 dask 集群
        client = Client(n_workers=4, threads_per_worker=1)
        print(f"Dask client: {client}")
        
        start = time.time()
        results = anac2024_std(
            competitors=agents,
            n_configs=1,
            n_runs_per_world=1,
            n_steps=10,
            parallelism="distributed",
            compact=True,
            print_exceptions=True,
        )
        elapsed = time.time() - start
        client.close()
        print(f"✓ Dask Distributed 完成! 耗时: {elapsed:.2f}秒")
        return elapsed
    except Exception as e:
        print(f"✗ Dask 失败: {e}")
        return None

def main():
    print("=" * 70)
    print("测试 NegMas 并行死锁解决方案")
    print("=" * 70)
    
    # 1. Serial 基准
    serial_time = test_with_serial()
    
    # 2. 检查 loky
    test_check_loky()
    
    # 3. 测试 dask
    dask_time = test_dask_distributed()
    
    # 汇总
    print("\n" + "=" * 70)
    print("测试结果")
    print("=" * 70)
    print(f"Serial 模式: {serial_time:.2f}秒")
    if dask_time:
        print(f"Dask 模式: {dask_time:.2f}秒")
    
    print("\n建议:")
    print("1. 如果 dask 工作正常，使用 parallelism='distributed'")
    print("2. 否则使用 parallelism='serial'（慢但可靠）")

if __name__ == "__main__":
    main()
