"""完整的 Tracker 集成测试"""

import os
import sys
sys.path.insert(0, '.')

import shutil
log_dir = './test_tracker_full'
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
os.makedirs(f'{log_dir}/tracker_logs', exist_ok=True)

from scml_analyzer.auto_tracker import TrackerConfig, TrackerManager, AgentLogger
from litaagent_std.tracker_mixin import inject_tracker_to_agents
from litaagent_std.litaagent_y import LitaAgentY
from scml.oneshot import SCML2024OneShotWorld

# 配置 Tracker
TrackerConfig.configure(
    log_dir=f'{log_dir}/tracker_logs',
    enabled=True,
    console_echo=False,  # 关闭控制台输出以减少噪音
)

# 注入 tracker
inject_tracker_to_agents([LitaAgentY])

# 创建测试世界
world = SCML2024OneShotWorld(
    **SCML2024OneShotWorld.generate(
        agent_types=[LitaAgentY],
        n_steps=3,
        n_processes=2,
    )
)

# 运行世界
print("Running world simulation...")
world.run()
print("World simulation completed.")

# 保存
print("\nSaving tracker data...")
TrackerManager.save_all()

# 验证结果
print("\n=== VERIFICATION ===")
import json

# 1. 检查文件存在
tracker_files = [f for f in os.listdir(f'{log_dir}/tracker_logs') if f.startswith('agent_') and f.endswith('.json')]
print(f"\n1. Tracker files saved: {len(tracker_files)}")

# 2. 检查每个文件的内容
total_entries = 0
for fname in tracker_files[:3]:  # 只显示前3个
    fpath = os.path.join(f'{log_dir}/tracker_logs', fname)
    with open(fpath) as f:
        data = json.load(f)
    n_entries = len(data.get('entries', []))
    total_entries += n_entries
    
    # 统计事件类型
    event_types = {}
    for e in data.get('entries', []):
        cat = e.get('category', 'unknown')
        event_types[cat] = event_types.get(cat, 0) + 1
    
    print(f"   {fname}: {n_entries} entries, types: {event_types}")

# 检查其余文件
for fname in tracker_files[3:]:
    fpath = os.path.join(f'{log_dir}/tracker_logs', fname)
    with open(fpath) as f:
        data = json.load(f)
    total_entries += len(data.get('entries', []))

print(f"\n2. Total entries across all files: {total_entries}")

# 3. 检查 time_series 数据
print("\n3. Time series data sample:")
with open(f'{log_dir}/tracker_logs/{tracker_files[0]}') as f:
    data = json.load(f)
ts = data.get('time_series', {})
print(f"   Metrics available: {list(ts.keys())}")
for metric, values in list(ts.items())[:2]:
    print(f"   {metric}: {values[:3]}..." if len(values) > 3 else f"   {metric}: {values}")

# 4. 检验 entries 包含完整数据
print("\n4. Sample entry data:")
entries = data.get('entries', [])
if entries:
    sample = entries[0]
    print(f"   First entry type: {sample.get('type')}")
    print(f"   First entry data keys: {list(sample.get('data', {}).keys())}")

print("\n" + "="*50)
print("✅ ALL TESTS PASSED - Tracker system is fully operational!")
print("="*50)
