"""检查 LOS 是否超卖（卖出数量 > 外生输入数量）"""
import json
from pathlib import Path

# 找到最新的比赛
tracker_dir = Path(r"D:\SCML_initial\tournament_history\20260110_161604_oneshot\tracker_logs")
los_files = list(tracker_dir.glob("agent_*LOS*.json"))

if not los_files:
    print("No LOS tracker files found")
    exit(1)

for f in los_files[:3]:
    print(f"\n=== {f.name} ===")
    data = json.loads(f.read_text(encoding="utf-8"))
    
    # 获取 agent_initialized 信息
    init = next((e for e in data["entries"] if e["event"] == "agent_initialized"), None)
    if init:
        d = init["data"]
        print(f"  Level: {d.get('level')}, n_lines: {d.get('n_lines')}")
    
    # 获取 started 事件（每个 step 开始时）
    starts = [e for e in data["entries"] if e["event"] == "started"]
    print(f"  Total steps (started events): {len(starts)}")
    
    # 看 accept 了多少
    accepts = [e for e in data["entries"] if e["event"] == "accept"]
    print(f"  Total accepts: {len(accepts)}")
    
    # 查看 accept 的详情
    for e in accepts[:5]:
        d = e["data"]
        print(f"    Step {e['day']}: accept q={d.get('quantity')}, p={d.get('unit_price')}, partner={d.get('partner', '?')[:20]}")
    
    # 按 step 统计接受数量
    accept_by_step = {}
    for e in accepts:
        step = e["day"]
        q = e["data"].get("quantity", 0)
        accept_by_step[step] = accept_by_step.get(step, 0) + q
    
    print("  Accepted quantity by step:")
    for step, q in sorted(accept_by_step.items()):
        print(f"    Step {step}: accepted_q = {q}")
    
    # 看 stats 里的 balance 变化
    stats = data.get("stats", {})
    print(f"  Final stats: balance_start={stats.get('balance_start')}, balance_end={stats.get('balance_end')}")
